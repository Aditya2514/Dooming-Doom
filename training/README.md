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
| Maps | `training/maps/` | own UDMF generator: combat, rooms, key-door A/B/C, easy key-door, mixed gauntlets; alcove or **floor-pad** exits; pools train / train2 / validation / valpad / test with disjoint layouts |
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

Episodes end at exit, death, the map's tick cap, or 250 decisions without moving 64 units. As in
the real engine, killing every enemy does **not** end an episode; combat arenas have an exit pad.

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

**Measured outcome (27 Nov):** in two runs (lr 1e-4 with entropy 0.005; lr 3e-5, no entropy, teacher
term 0.5 → 0.1, KL target 0.01) PPO *lowered* held-out validation exits from 0.58 (its imitation start)
to 0.38 after ~34 policy updates. Most likely the freshly trained critic's advantages are noise that
erodes a finely imitated policy faster than the reward improves it. The DAgger policy ships.

## Stage 3: RGB exit detection (`pretrain_rgb.py`, then `dagger.py --init`)

**Why.** Rays and entities have no exit class. The official sample map's exit is a red `GATE3` floor
pad; our generated pads had the room's floor texture, so no sensor could see them, and the teacher
walked to exit coordinates the student could not perceive (`valpad` exits 0.44). Changes:

* **Maps:** pad exits are `GATE3` (half of `train` / `train2`, all of `valpad`); combat arenas get an
  exit pad (killing every enemy does not end a map); random wall/floor textures per map so "not the
  usual floor" cannot mean "exit".
* **Teacher:** the exit becomes a target only once it has been **on camera** (within 40° of the heading,
  line of sight along the official rays, ≤ 400 units), not after a 360° ray sweep. Teacher ceiling
  unchanged: 0.99 exits on `valpad` and `test`.
* **Student:** `RgbEncoder` (30 × 40 pooled frame → 64) fused before the GRU. Widening the rays-only
  checkpoint adds it with zero fusion weights, so the network starts as the old policy (test: 1e-5).
* **Auxiliary exit head** (`policy.rgb_aux`, training only, never used to act): "exit on screen, and in
  which quarter of the view?", labelled from the generator's exit + line of sight
  (`expert.exit_view_class`). Learned through imitation alone it stayed at 0 recall (exits are on
  screen in ~4% of states), so `pretrain_rgb.py` first trains the RGB branch alone on 90k teacher
  frames: validation recall 0.63, precision 0.66, false alarms 1.3% per frame; on the official map
  (never trained on) 37% of on-screen frames detected, 3 false alarms in 389.
* **DAgger** continues from that checkpoint (aux loss 0.5, class weight 10, lr 1e-4, 4,096 states ×
  80 updates per iteration, 30k states collected per iteration, `train` + `train2`), on CPU.

```
python -m training.pretrain_rgb --init runs/il4/iter06.pt --out runs/rgbpre2.pt --frames 90000 --epochs 12 --cache rgbframes.npz
python -m training.dagger --run rgb4 --init runs/rgbpre2.pt --workers 2 --iters 8 --steps 30000 --updates 80 \
    --batch-steps 4096 --cap 400000 --lr 1e-4 --aux 0.5 --aux-pos-weight 10 --beta 0.3 0.1 --pools train train2
```

Iterations were noisy (test-pool exits 0.65 / 0.75 / 0.64 at iterations 3 / 5 / 6). Iteration 5 was
picked on `test` / `valpad`, then scored **once** on two fresh pools (`confirm`, `confirmpad`: new
seeds, never used for training or selection). Note: on maps without monsters an episode is
deterministic (the agent reseeds its sampler at `reset()`), so extra `--seeds` repeat the same runs;
use new maps, not new seeds, to confirm a choice.

## Shipped model: RGB exit detector on the longer-trained student

`rgb_best` iteration 3: a longer-trained rays + map student (`best.pt`, DAgger iteration 17 on a second machine)
→ `pretrain_rgb.py` (90k teacher frames; detector validation recall 0.49, precision 0.57, 1.6% false alarms
per frame) → DAgger with the RGB branch (`--init`, 80k states and 300 updates per iteration, 1M-state dataset).
Scored once on the fresh `confirm` + `confirmpad` pools (200 maps, one episode each), paired:

| Fresh maps | Previous submission (rays, `4847f3a`) | RGB only (`rgb4` iter 5) | Longer-trained rays (`best.pt`) | **Shipped** |
|---|---|---|---|---|
| exits, all 200 maps | 0.61 | 0.66 | 0.645 | **0.77** |
| red floor-pad exits (90) | 0.48 | 0.66 | 0.51 | **0.78** |
| alcove exits + combat (110) | 0.72 | 0.66 | 0.75 | **0.76** |
| — rooms / easy key / key-door / mixed | 0.90 / 1.00 / 0.70 / 0.63 | 0.60 / 0.90 / 0.63 / 0.60 | 0.75 / 1.00 / 0.83 / 0.77 | 0.75 / 1.00 / 0.80 / 0.60 |
| — combat arena exits | 0.55 | 0.75 | 0.50 | **0.85** |
| kill fraction · deaths | 0.86 · 8 | 0.91 · 4 | 0.93 · 5 | **0.93 · 4** |
| official `two_rooms`, 20 runs | 20/20 (105) | 20/20 (162) | 20/20 (171) | **20/20 (71)** |

Paired exit differences of the shipped model: +16 ± 8 % (previous submission), +11 ± 8 % (RGB only),
+12.5 ± 8 % (`best.pt`), 95% CI. Validator: exit on `two_rooms` in 68 decisions, median 4.7 ms per decision.

## Earlier RGB-only model (fresh confirmation maps, paired against the submitted model)

This branch is a **post-submission experiment**. The submission evaluated at the deadline is the rays-only
model on `main` (`il5` iteration 4). Both models below played the same fresh maps (`confirm`,
`confirmpad`: never used for training or for picking either checkpoint), one episode per map, actions sampled.

| Fresh maps | Submitted, rays-only (`il5` iter 4, `main`) | **RGB (`rgb4` iter 5, this branch)** | Paired RGB − submitted (95% CI) |
|---|---|---|---|
| floor-pad exits (`confirmpad`, 90 maps) | 0.48 | **0.66** | **+17.8 ± 13.3 % exits, +18.2 ± 15.2 proxy** |
| — rooms / easy key / key-door / mixed | 0.45 / 0.70 / 0.40 / 0.50 | **0.60 / 0.90 / 0.50 / 0.77** | |
| alcove exits + combat (`confirm`, 110 maps) | **0.72** | 0.66 | −5.5 ± 11.6 % exits (not significant) |
| — rooms / easy key / key-door / mixed | **0.90 / 1.00 / 0.70 / 0.63** | 0.60 / 0.90 / 0.63 / 0.60 | |
| — combat arena with exit pad | 0.55 | **0.75** | |
| **all 200 maps** | 0.61 | **0.66** | +5.0 ± 8.9 % exits, +5.9 ± 10.2 proxy (not significant) |
| deaths (of 200) | 8 | **4** | |
| official `two_rooms.wad`, 20 runs | 20 / 20 (105 decisions) | 20 / 20 (162 decisions) | |

Trade-off: the RGB student finds red floor-pad exits (the official sample map's style) far more often and
leaves combat arenas more often, but lost plain multi-room skill (0.90 → 0.60); the RGB run's smaller
dataset (0.2M vs 1M states) is the likely cause. Latency on the official validator: median 3.9 ms per
decision (rays-only: 3.0 ms).

### Submitted model (`main`) on the earlier validation pools

| Map family | Teacher | DAgger iter 4 | DAgger iter 6 | **Shipped: iter 6 + 4 iters on train+train2** |
|---|---|---|---|---|
| rooms (exit) | 1.00 | 0.40 | 0.95 | **0.80** |
| easy key-door (exit) | 1.00 | 0.80 | 0.95 | **0.90** |
| key-door A/B/C (exit) | 1.00 | 0.27 | 0.67 | **0.77** |
| mixed gauntlet (exit) | 1.00 | 0.45 | 0.65 | **0.74** |
| combat (kill fraction) | 0.97 | 1.00 | 0.98 | **0.98** |
| floor-pad exits, all families (`valpad`) | 0.99 | 0.29 | 0.41 | **0.47** |
| official `two_rooms.wad` (10 runs) | – | 1/10 | 9/10 | **10/10** |

Last two columns: 390 validation episodes (3 seeds) + 210 `valpad` episodes; iter 4: 1 seed; actions sampled
as submitted. Paired over the same 600 episodes the shipped model leads iter 6 by +3.8 ± 5.0 proxy points
(+3.3% exits): better on key-door, gauntlets and pad exits (higher-weight tiers), worse on plain rooms.

Shipped weights `runs/il5/iter04.pt`: `python -m training.dagger --run il4 --workers 10 --steps0 200000
--steps 80000 --updates0 1000 --updates 500 --cap 1000000` (iterations 0–6), then
`python -m training.dagger --run il5 --resume runs/il4/iter06.pt --pools train train2 --steps 100000
--updates 500` (iterations 1–4).

## Teacher details (`expert.py`)

Priority per decision: **dodge** (a visible projectile will pass within 56 units in the next 40
tics → strafe off its line) → **fight** (nearest visible enemy within 1,100 units: turn, fire when
on target, back off from demons) → **door** (closed door ≤ 100 units ahead on our path → one USE,
then wait 14 decisions, since USE on a moving door reverses it) → **unstick** → **go to**: a seen key
still needed > the exit once it has been on camera > a visible weapon / health (when hurt) / armour > the nearest
cell its rays haven't swept (frontier) > the exit. Locked doors are walls until their key is held,
so "find key → come back to the locked door → exit" falls out of frontier exploration.

Teacher on the validation pools (`python -m training.evaluate expert`): exit 100% on rooms, easy
key-door, key-door A/B/C and mixed gauntlets; combat: 97% survival, 0.97 kill fraction.

## Commands

```powershell
uv sync --group train                                   # torch, scipy, psutil, imageio-ffmpeg
.venv\Scripts\python.exe -m training.maps.gen --all     # deterministic map pools
.venv\Scripts\python.exe -m training.evaluate expert    # teacher ceiling
.venv\Scripts\python.exe -m training.dagger --run il4 --workers 10 --iters 30          # stage 1 (DAgger)
.venv\Scripts\python.exe -m training.finetune --init runs\il4\best.pt --run ppo          # stage 2 (PPO, optional)
.venv\Scripts\python.exe -m training.watch --run il4                                     # live viewer (--teacher)
.venv\Scripts\python.exe -m training.evaluate runs\il4\iter06.pt --pool validation      # held-out maps
.venv\Scripts\python.exe -m training.evaluate runs\il4\iter06.pt --pool valpad --family rooms keyeasy keydoor mixed
.venv\Scripts\python.exe -m training.record submission\weights\actor.pt --wad maps\two_rooms.wad   # demo video
.venv\Scripts\python.exe -m training.record submission\weights\actor.pt --family mixed --index 0
.venv\Scripts\python.exe -m pytest -q tests
python validate.py
```
