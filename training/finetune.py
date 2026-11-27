"""Stage 2: recurrent PPO fine-tuning of the imitation policy (training only).

    .venv\\Scripts\\python.exe -m training.finetune --init runs/il3/best.pt --run ppo --workers 10

The pipeline is "imitation -> reinforcement learning": stage 1 (training/dagger.py) distils the
privileged teacher into the sensor-only network; stage 2 (this file) improves that network with
PPO (Schulman et al. 2017) on the game's own outcomes, which the teacher's script never optimised.

* Policy: submission/doomnet/policy.py unchanged (the value head, unused by imitation, is trained
  here). Recurrent PPO: rollouts of T decisions from N engines; each env's GRU state at the rollout
  start is stored and the sequence is re-run with episode-start resets inside it.
* Reward: training/ppo_rewards.py -- scorer-aligned extrinsic terms + count-based curiosity on the
  agent's own spatial map.
* Value warm-up: the first --value-warmup updates train only the value head (the policy is frozen),
  so the first policy gradients use sensible advantages instead of an untrained critic's noise.
* Teacher anchor: the teacher's labels are still collected; a cross-entropy term (coefficient
  annealed --bc-coef -> --bc-final) keeps early updates from forgetting the imitation skills while
  PPO moves the policy where the reward says the teacher was wrong.
* GAE(gamma, lambda); clipped surrogate; entropy bonus; KL early stop per epoch; fp16 autocast.
Every episode end (exit, death, tick cap, stall cutoff) is terminal: no bootstrap.
"""
from __future__ import annotations

import argparse
import json
import random
import time
from collections import defaultdict, deque
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]


def to_tensors(arrs: dict, device):
    """Packed numpy arrays with leading dims (T, B) -> network input tensors."""
    import torch

    from submission.doomnet import obs_adapter as A
    from training.dagger import unpack_map

    T, B = arrs["dist"].shape[:2]
    rays = torch.zeros(T, B, A.RAY_CHANNELS, A.NUM_RAYS, device=device)
    rays[:, :, 0] = torch.from_numpy(arrs["dist"].astype(np.float32)).to(device)
    hit = torch.from_numpy(arrs["hit"].astype(np.int64)).to(device)
    rays[:, :, 1:].scatter_(2, hit.unsqueeze(2), 1.0)
    return {"rays": rays,
            "ent_type": torch.from_numpy(arrs["ent_type"].astype(np.int64)).to(device),
            "ent_feat": torch.from_numpy(arrs["ent_feat"].astype(np.float32)).to(device),
            "ent_mask": torch.from_numpy(arrs["ent_mask"]).to(device),
            "scalars": torch.from_numpy(arrs["scalars"]).to(device),
            "map": unpack_map(torch.from_numpy(arrs["map"]).to(device)),
            "rgb": torch.from_numpy(arrs["rgb"]).to(device)}


def pack_step(xs: list[dict]) -> dict:
    """Worker observations of one step -> compact arrays (B, ...)."""
    rays = np.stack([x["rays"] for x in xs])
    return {"dist": rays[:, 0].astype(np.float16), "hit": rays[:, 1:].argmax(1).astype(np.uint8),
            "ent_type": np.stack([x["ent_type"] for x in xs]).astype(np.uint8),
            "ent_feat": np.stack([x["ent_feat"] for x in xs]).astype(np.float16),
            "ent_mask": np.stack([x["ent_mask"] for x in xs]).astype(bool),
            "scalars": np.stack([x["scalars"] for x in xs]).astype(np.float32),
            "map": np.stack([x["map"] for x in xs]).astype(np.uint8),
            "rgb": np.stack([x["rgb"] for x in xs]).astype(np.uint8)}


def run_seq(policy, x: dict, h0, starts):
    """x (T, B, ...), h0 (1, B, H), starts (T, B) 1.0 where step t begins an episode
    -> logits (T, B, A), values (T, B). Same computation as acting, step by step through the GRU."""
    import torch

    e = policy.encode(x)
    h, zs = h0, []
    for t in range(e.shape[0]):
        h = h * (1.0 - starts[t]).view(1, -1, 1).to(h.dtype)
        z, h = policy.gru(e[t:t + 1], h)
        zs.append(z[0])
    z = torch.stack(zs)
    return policy.pi(z), policy.v(z).squeeze(-1)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--init", required=True, help="imitation checkpoint (runs/<run>/best.pt)")
    ap.add_argument("--run", default="ppo")
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--updates", type=int, default=600)
    ap.add_argument("--rollout", type=int, default=256, help="decisions per env per update")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--minibatches", type=int, default=2)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--gamma", type=float, default=0.995)
    ap.add_argument("--lam", type=float, default=0.95)
    ap.add_argument("--clip", type=float, default=0.1)
    ap.add_argument("--ent", type=float, default=0.005)
    ap.add_argument("--vf", type=float, default=0.5)
    ap.add_argument("--target-kl", type=float, default=0.02)
    ap.add_argument("--bc-coef", type=float, default=0.3, help="teacher cross-entropy weight at the start")
    ap.add_argument("--bc-final", type=float, default=0.02, help="... annealed linearly to this")
    ap.add_argument("--value-warmup", type=int, default=8, help="updates training only the value head")
    ap.add_argument("--kl-ref", type=float, default=0.0,
                    help="weight of KL(pi_init || pi): anchor to the starting (imitation) policy; its action "
                         "probabilities are computed during collection with its own GRU state, so they are exact")
    ap.add_argument("--snapshot-every", type=int, default=25)
    ap.add_argument("--families", default=None, help="json dict family -> weight")
    ap.add_argument("--pools", nargs="+", default=["train", "train2"],
                    help="map pools (train2: 1,600 fresh layouts, so RL replays each map only a few times)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()

    import torch
    import torch.nn.functional as F

    from submission.doomnet import obs_adapter as A
    from submission.doomnet.actions import NAMES, NUM_ACTIONS
    from submission.doomnet.policy import INPUT_KEYS, DoomPolicy, policy_from_state_dict
    from training.dagger import FAMILIES, Collector, unpack_map
    from training.ppo_rewards import RewardConfig

    torch.manual_seed(a.seed)
    rng = random.Random(a.seed)
    device = torch.device(a.device if torch.cuda.is_available() else "cpu")
    run = REPO / "runs" / a.run
    run.mkdir(parents=True, exist_ok=True)
    fams = json.loads(a.families) if a.families else FAMILIES
    rcfg = RewardConfig()
    (run / "config.json").write_text(json.dumps({**vars(a), "families": fams, "rewards": rcfg.__dict__}, indent=1))
    log = open(run / "log.jsonl", "a", buffering=1)

    ck = torch.load(a.init, map_location="cpu", weights_only=False)
    policy = policy_from_state_dict(ck["state_dict"]).to(device)
    ref = None
    if a.kl_ref > 0:
        import copy
        ref = copy.deepcopy(policy).eval()
        for p_ in ref.parameters():
            p_.requires_grad_(False)
    opt = torch.optim.Adam(policy.parameters(), lr=a.lr, eps=1e-5)
    opt_v = torch.optim.Adam(policy.v.parameters(), lr=3e-4, eps=1e-5)
    scaler = torch.amp.GradScaler(enabled=device.type == "cuda")
    print(f"[ppo] init {a.init} (iter {ck.get('meta', {}).get('iter')}), {a.workers} envs x {a.rollout} steps/update",
          flush=True)

    N, T = a.workers, a.rollout
    col = Collector(N, {"families": fams, "pool": a.pools, "seed": 1000 + a.seed, "rewards": rcfg.__dict__})
    h = policy.initial_state(N, device)
    h_ref = policy.initial_state(N, device) if ref is not None else None
    start_flag = np.ones(N, np.float32)
    recent: deque = deque(maxlen=200)
    best = -1e9

    def step_policy(xs, h, net=None):
        net = net or policy
        batch = A.stack(xs)
        xt = {k: (unpack_map(batch[k]) if k == "map" else torch.from_numpy(batch[k])).to(device) for k in INPUT_KEYS}
        e = net.encode(xt).unsqueeze(0)
        z, h2 = net.gru(e, h)
        return net.pi(z[0]), net.v(z[0]).squeeze(-1), h2

    def save(path, u, extra=None):
        torch.save({"state_dict": {k: v.detach().cpu() for k, v in policy.state_dict().items()},
                    "meta": {"arch": "doompolicy-v2", "use_map": policy.use_map, "num_actions": NUM_ACTIONS,
                             "actions": list(NAMES), "stage": "ppo", "update": u, "init": str(a.init),
                             "greedy": False, **(extra or {})}}, path)

    try:
        for u in range(a.updates):
            t0 = time.time()
            policy.eval()
            steps, acts, logps, vals, rews, dones, starts, labels, ref_lp = [], [], [], [], [], [], [], [], []
            h0 = h.detach().clone()
            with torch.no_grad():
                for t in range(T):
                    xs = [c[0] for c in col.cur]
                    labels.append([c[1] for c in col.cur])
                    steps.append(pack_step(xs))
                    starts.append(start_flag.copy())
                    logits, v, h = step_policy(xs, h)
                    if ref is not None:
                        rl, _, h_ref = step_policy(xs, h_ref, ref)
                        ref_lp.append(torch.log_softmax(rl.float(), -1).cpu().numpy())
                    dist = torch.distributions.Categorical(logits=logits.float())
                    act = dist.sample()
                    acts.append(act.cpu().numpy())
                    logps.append(dist.log_prob(act).cpu().numpy())
                    vals.append(v.float().cpu().numpy())
                    for i in range(N):
                        col.conns[i].send(("step", int(acts[-1][i])))
                    r_t, d_t = np.zeros(N, np.float32), np.zeros(N, np.float32)
                    start_flag[:] = 0.0
                    for i in range(N):
                        msg = col.conns[i].recv()
                        if msg[0] == "error":
                            raise RuntimeError(f"worker {i} failed:\n{msg[1]}")
                        col.cur[i] = msg
                        r_t[i] = msg[3]
                        if msg[2] is not None:
                            d_t[i] = 1.0
                            start_flag[i] = 1.0
                            h[:, i] = 0.0
                            if h_ref is not None:
                                h_ref[:, i] = 0.0
                            recent.append(msg[2])
                    rews.append(r_t)
                    dones.append(d_t)
                _, last_v, _ = step_policy([c[0] for c in col.cur], h)
                last_v = last_v.float().cpu().numpy()
            t_col = time.time() - t0

            # GAE
            R, D, V = np.array(rews), np.array(dones), np.array(vals)
            adv = np.zeros_like(R)
            nxt_v, nxt_a = last_v, np.zeros(N, np.float32)
            for t in reversed(range(T)):
                nonterm = 1.0 - D[t]
                delta = R[t] + a.gamma * nxt_v * nonterm - V[t]
                nxt_a = delta + a.gamma * a.lam * nonterm * nxt_a
                adv[t] = nxt_a
                nxt_v = V[t]
            ret = adv + V

            # update
            t1 = time.time()
            policy.train()
            arrs = {k: np.stack([s[k] for s in steps]) for k in steps[0]}
            X = to_tensors(arrs, device)
            ACT = torch.as_tensor(np.array(acts), device=device)
            LOGP = torch.as_tensor(np.array(logps), device=device)
            ADV = torch.as_tensor(adv, device=device)
            RET = torch.as_tensor(ret, device=device)
            ST = torch.as_tensor(np.array(starts), device=device)
            LAB = torch.as_tensor(np.array(labels), device=device)
            REF = torch.as_tensor(np.array(ref_lp), device=device) if ref is not None else None
            warm = u < a.value_warmup
            frac = min(1.0, u / max(1, a.updates - 1))
            bc_coef = a.bc_coef + (a.bc_final - a.bc_coef) * frac
            stats = defaultdict(list)
            stop = False
            for ep_i in range(1 if warm else a.epochs):
                perm = np.random.permutation(N)
                for mb in np.array_split(perm, a.minibatches):
                    idx = torch.as_tensor(mb, device=device)
                    xb = {k: v[:, idx] for k, v in X.items()}
                    with torch.autocast(device.type, dtype=torch.float16, enabled=device.type == "cuda"):
                        logits, values = run_seq(policy, xb, h0[:, idx], ST[:, idx])
                    logits, values = logits.float(), values.float()
                    v_loss = 0.5 * ((values - RET[:, idx]) ** 2).mean()
                    if warm:
                        opt_v.zero_grad(set_to_none=True)
                        scaler.scale(a.vf * v_loss).backward()
                        scaler.unscale_(opt_v)
                        torch.nn.utils.clip_grad_norm_(policy.v.parameters(), 1.0)
                        scaler.step(opt_v)
                        scaler.update()
                        policy.zero_grad(set_to_none=True)
                        stats["v_loss"].append(v_loss.item())
                        continue
                    dist = torch.distributions.Categorical(logits=logits)
                    logp = dist.log_prob(ACT[:, idx])
                    ratio = torch.exp(logp - LOGP[:, idx])
                    ab = ADV[:, idx]
                    ab = (ab - ab.mean()) / (ab.std() + 1e-8)
                    pg = -torch.min(ratio * ab, ratio.clamp(1 - a.clip, 1 + a.clip) * ab).mean()
                    ent = dist.entropy().mean()
                    bc = F.cross_entropy(logits.reshape(-1, NUM_ACTIONS), LAB[:, idx].reshape(-1))
                    loss = pg + a.vf * v_loss - a.ent * ent + bc_coef * bc
                    if REF is not None:
                        lr_ = REF[:, idx]
                        kl_ref = (lr_.exp() * (lr_ - torch.log_softmax(logits, -1))).sum(-1).mean()
                        loss = loss + a.kl_ref * kl_ref
                        stats["kl_ref"].append(kl_ref.item())
                    opt.zero_grad(set_to_none=True)
                    scaler.scale(loss).backward()
                    scaler.unscale_(opt)
                    torch.nn.utils.clip_grad_norm_(policy.parameters(), 0.5)
                    scaler.step(opt)
                    scaler.update()
                    with torch.no_grad():
                        kl = ((ratio - 1) - torch.log(ratio)).mean().item()
                    for k_, v_ in (("pg", pg), ("v_loss", v_loss), ("ent", ent), ("bc", bc)):
                        stats[k_].append(v_.item())
                    stats["kl"].append(kl)
                    stats["clipfrac"].append(((ratio - 1).abs() > a.clip).float().mean().item())
                    stats["teacher_agree"].append((logits.argmax(-1) == LAB[:, idx]).float().mean().item())
                    if kl > 1.5 * a.target_kl:
                        stop = True
                        break
                if stop:
                    break
            t_up = time.time() - t1

            # report
            eps = list(recent)
            fam = defaultdict(list)
            for e in eps:
                fam[e["family"]].append(e)
            fam_s = {}
            for f, ee in sorted(fam.items()):
                mon = [e for e in ee if e["monsters"]]
                fam_s[f] = {"n": len(ee), "exit": float(np.mean([e["exit"] for e in ee])),
                            "dead": float(np.mean([e["dead"] for e in ee])),
                            "kill_frac": float(np.mean([e["kills"] / e["monsters"] for e in mon])) if mon else None,
                            "ret": float(np.mean([e.get("ret", 0.0) for e in ee]))}
            score = (float(np.mean([v["kill_frac"] if f == "combat" and v["kill_frac"] is not None else v["exit"]
                                    for f, v in fam_s.items()])) if fam_s else None)
            row = {"update": u, "warmup": warm, "bc_coef": round(bc_coef, 4), "t_collect": round(t_col, 1),
                   "t_update": round(t_up, 1), "reward_mean": float(R.mean()), "score_recent": score,
                   "families": fam_s, **{k: float(np.mean(v)) for k, v in stats.items()}, "time": time.time()}
            log.write(json.dumps(row) + "\n")
            line = "  ".join(f"{f} x{v['exit']:.2f}" + (f"/k{v['kill_frac']:.2f}" if v["kill_frac"] is not None else "")
                             for f, v in fam_s.items())
            print(f"[ppo {u:4d}]{' WARMUP' if warm else ''} r/step {R.mean():+.3f}  v_loss {row.get('v_loss', 0):.3f}  "
                  f"kl {row.get('kl', 0):.4f}  ent {row.get('ent', 0):.2f}  teacher_agree {row.get('teacher_agree', 0):.2f}  "
                  f"| recent200: {line}  score {score if score is None else round(score, 3)}  "
                  f"({t_col:.0f}+{t_up:.0f}s)", flush=True)
            save(run / "latest.pt", u, {"score_recent": score})
            if u % a.snapshot_every == 0 or u == a.updates - 1:
                save(run / f"u{u:04d}.pt", u, {"score_recent": score})
            if not warm and score is not None and len(eps) >= 100 and score > best:
                best = score
                save(run / "best.pt", u, {"score_recent": score})
    finally:
        col.close()
        log.close()


if __name__ == "__main__":
    main()
