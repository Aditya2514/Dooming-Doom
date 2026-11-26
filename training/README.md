# Training pipeline: privileged teacher → sensor-only student (DAgger)

Everything in `training/` is training-time only; nothing in `submission/` imports it (a test checks).
The submitted agent is a neural network that sees **only the official observation**.

## Idea in one paragraph

Learning DOOM from reward alone needs far more experience than a laptop can give, and our first
attempt (PPO with 9 independent buttons and 12 shaped reward terms) ended up pressing buttons almost
at random. Instead we **distill a privileged teacher into a sensor-only student**. The teacher
(`expert.py`) can see the floor plan of the training map and its true position, which makes good
play easy to script. But we keep its *decisions* imitable: it explores the nearest area its rays
haven't swept yet, goes for keys and the exit only after they've been seen, fights what is visible,
dodges projectiles it can see, and presses USE once at a door. The student (`submission/doomnet/`)
learns to reproduce those decisions from rays, entities, vitals and its own **spatial map**, with
**DAgger** so it also learns to recover from its own mistakes.

## Pieces

| Piece | Code | Notes |
|---|---|---|
| Observation adapter (shared train/eval) | `submission/doomnet/obs_adapter.py` | rays one-hot, entities, vitals, previous buttons; dead-reckoned pose; **SpatialMap** |
| Spatial map | `SpatialMap` in the adapter | 320×320 grid of 64-unit cells in the episode's dead-reckoned frame: visited / swept by rays / door seen / locked door seen. Network input: 24×24 heading-up crop |
| Action space | `submission/doomnet/actions.py` | 16 macro-actions (FWD, FWD+TURN, TURN, ATK, ATK+TURN, USE, STRAFE, ...): no contradictory presses |
| Network | `submission/doomnet/policy.py` | ray Conv1d (circular) + entity attention (null token) + vitals + map CNN → GRU 256 → 16 logits |
| Teacher | `training/expert.py` | BFS on a 16-unit grid of the generated map's floor plan; ray-swept "seen" map; dodge, fight, door, unstick, go-to |
| Collectors | `training/collect.py` | one engine per process, torch-free; returns (network inputs, teacher label) per step |
| DAgger trainer | `training/dagger.py` | lockstep collection, student batched on GPU, FIFO dataset (bit-packed maps), full-episode GRU training with fp16 autocast |
| Maps | `training/maps/` | own UDMF generator: combat, rooms, key-door A/B/C, easy key-door, mixed gauntlets (train/validation/test pools with disjoint layouts) |
| Evaluation | `training/evaluate.py` | runs the **submitted** `DoomAgent` on held-out maps; proxy score modelled on the official categories; `expert` mode for the teacher's ceiling |
| Metrics | `training/metrics.py` | keys, doors / locked doors opened, 128-unit cells visited (from `privileged_state()`) |
| Videos | `training/record.py` | MP4 of the submitted agent on one map |

## DAgger loop (`dagger.py`)

1. **Iteration 0:** the teacher drives all episodes (200k labelled states); train 1,500 updates.
2. **Iterations 1+:** each episode is either *student-only* (40%: on-policy states and an honest
   running score) or *mixed* (each step the teacher's action with probability β, else the
   student's; β anneals 0.5 → 0.1). **Every** visited state gets the teacher's label.
3. Episodes go into a FIFO dataset (1.5M states); 500 updates of cross-entropy (label smoothing 0.05)
   per iteration on whole episodes through the GRU, exactly the computation used when acting.
4. `runs/<run>/best.pt` keeps the checkpoint with the best student-only score (mean of exit rate per
   family, kill fraction on combat maps).

Episodes end at exit, death, the map's tick cap, all enemies dead (combat maps), or 250 decisions
without moving 64 units.

## Stage 2: PPO fine-tuning (`finetune.py`)

Imitation can only be as good as the teacher's script, and it never optimises the actual outcome.
Stage 2 starts from the best imitation checkpoint and runs **recurrent PPO** on the game itself:

* **Reward** (`ppo_rewards.py`, training-only ground truth): exit +10 (+health/time bonus), key +3,
  door +1 / locked door +3, kill +1, new 128-unit area +0.05, damage −0.02/pt, early death −5×(time
  left), −0.002/decision; plus **count-based curiosity** +0.04/√n on entering a cell of the agent's
  *own* spatial map for the n-th time (so the bonus depends only on what the agent knows).
* **Value warm-up:** the value head (unused by imitation) trains alone for the first 8 updates.
* **Teacher anchor:** the teacher still labels every state; a cross-entropy term annealed 0.3 → 0.02
  stops the first updates from erasing the imitation skills.
* 10 envs × 256 decisions per update, GAE(0.995, 0.95), clip 0.1, entropy 0.005, KL early stop 0.02,
  3 epochs × 2 minibatches; the GRU state at each rollout start is stored and the sequence re-run with
  in-sequence episode resets (identical to acting).
* Snapshots every 25 updates; the shipped checkpoint is picked by `evaluate.py` on held-out maps, and
  PPO only replaces the imitation model if it scores at least as well.

## Teacher details (`expert.py`)

Priority per decision: **dodge** (a visible projectile will pass within 56 units in the next 40
tics → strafe off its line) → **fight** (nearest visible enemy within 1,100 units: turn, fire when
on target, back off from demons) → **door** (closed door ≤ 100 units ahead on our path → one USE,
then wait 14 decisions, since USE on a moving door reverses it) → **unstick** → **go to**: a seen key
still needed > the exit once seen > a visible weapon / health (when hurt) / armour > the nearest
cell its rays haven't swept (frontier) > the exit. Locked doors are walls until their key is held,
so "find key → come back to the locked door → exit" falls out of frontier exploration.

Teacher on the validation pools (`python -m training.evaluate expert`): exit 100% on rooms, easy
key-door, key-door A/B/C and mixed gauntlets; combat: 97% survival, 0.97 kill fraction.

## Commands

```powershell
uv sync --group train                                   # torch, scipy, psutil, imageio-ffmpeg
.venv\Scripts\python.exe -m training.maps.gen --all     # deterministic map pools
.venv\Scripts\python.exe -m training.evaluate expert    # teacher ceiling
.venv\Scripts\python.exe -m training.dagger --run il3 --workers 10 --iters 30          # stage 1
.venv\Scripts\python.exe -m training.finetune --init runs\il3\best.pt --run ppo          # stage 2
.venv\Scripts\python.exe -m training.watch --run ppo                                     # live viewer
.venv\Scripts\python.exe -m training.evaluate runs\ppo\best.pt --family rooms keydoor mixed combat
copy runs\ppo\best.pt submission\weights\actor.pt
.venv\Scripts\python.exe -m training.record submission\weights\actor.pt --family mixed --index 0
.venv\Scripts\python.exe -m pytest -q tests
python validate.py
```
