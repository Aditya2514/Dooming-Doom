# Training pipeline

Everything here is training-only: nothing in `submission/` imports it (a test enforces that).
The agent only ever sees the official observation dictionary; ground truth from
`env.privileged_state()` is used for rewards, the critic and map validation, never as an input.

## Map from spec to code

| Piece | Code | Notes |
|---|---|---|
| Shared observation adapter | `submission/doomnet/obs_adapter.py` | One module for training workers **and** `DoomAgent`; a test checks bit-identical tensors on real games |
| Acting network (baseline / memory) | `submission/doomnet/model.py` | 691,401 / 885,929 acting parameters |
| Episodic memory bookkeeping | `submission/doomnet/memory.py` | 8 recent + 8 event slots; identical class at train and eval time |
| Critic (16 privileged inputs) + aux head | `training/model.py` | 726,731 / 921,259 parameters in total |
| Recurrent PPO, GAE, rollout buffer | `training/ppo.py`, `training/rollout.py` | 32-step chunks, KL guard, recompute test before the first update |
| Memory on the learner side | `training/memory_rollout.py` | per-rollout record tables, one encoding per record per minibatch |
| Reward shaping | `training/rewards.py` | scorer-aligned terms, all from `privileged_state()` |
| Vector env | `training/vec_env.py` | one process per env, shared memory, same-step autoreset, in-engine map switching |
| Maps | `training/maps/` | own UDMF generator (combat, rooms, key-door A/B/C, mixed) + Freedoom 2 levels |
| Curriculum | `training/curriculum.py` | 4 stages, validation gates through the submitted agent, deadlines, replay |
| Evaluation + proxy scorer | `training/evaluate.py` | matched map-seed episodes, Wilson intervals, ablations |
| Selection / freeze / test / latency | `training/select.py` | frozen rule, test pool opened once |
| Videos, memory demo | `training/record.py`, `demo/memory_demo.py` | |
| Windows ViZDoom latency fix | `training/win_ipc.py` | 3.2x faster engine IPC on Windows |

## Episode endings (spec F1)

The wrapper returns `truncated=True` at the map's tick cap, but the scored attempt is over and
the returned observation is stale, so the tick cap is treated as a **termination** (no
bootstrap). Only our own cutoffs bootstrap, from a genuine observation. Tested in
`tests/test_ppo_core.py` (worked GAE examples for both ending types).

## Rewards (training only)

| Term | Value | Source |
|---|---|---|
| Exit | +10, + 2 x health fraction + 2 x time-left fraction | `exit_reached`, `health`, `tic` |
| Keycard | +2 per new key flag | `keys` |
| Door, first opening (ordinary / locked sector) | +0.5 / +2.0, once per sector per episode | sector heights, `door_lock` |
| Exploration | +0.02 per new 128-unit cell, cap +3 / episode | `x`, `y` |
| Kill | +0.5 | `kills` |
| Damage taken | -0.01 per point | `damage_taken` |
| Early death | -5 x (1 - time fraction) | `dead`, `tic` |
| Time | -0.001 per decision | |
| Ammo spent (our addition) | -0.002 per unit | own vitals (videos showed the agent emptying the pistol at walls) |

Door rewards were enabled only after `tests/test_door_reward.py` passed on a real WAD (ordinary
door, door open at reset, raised-floor door, locked door with and without the key, reopening,
and all six card/skull locks).

## Curriculum

| Stage | Maps (train / val / test) | gamma | Gate (validation, submitted agent) | Deadline |
|---|---|---|---|---|
| 1 Combat | `combat` 100/20/20 | 0.99 | survival >= 0.70 and kill fraction >= 0.60 | T+6 h |
| 2 Multi-room | `rooms` 100/20/20 | 0.995 | exit >= 0.60 | T+10 h |
| 3 Key-door | `keydoor` A/B/C 120/30/30 | 0.995 | exit >= 0.50, every variant >= 0.30 | T+15 h |
| 4 Mixed | `mixed` 160/40/40 + Freedoom 13/2/2 | 0.995 | final | |

Earlier stages are replayed (20%; 50% in Stage 3, rooms-heavy, after Stage 3 at 20% replay
erased the exit skill within 360k decisions). Each worker stays in one WAD so maps switch inside
the running engine (0.07 s vs 0.54 s for a new engine). Gate evals sample actions (thresholding
at 0.5 loops in place while policy entropy is high).

## Commands

```powershell
python -m uv sync                                                  # torch 2.15 cu132, psutil, imageio-ffmpeg
.venv\Scripts\python.exe -m training.maps.gen --all                # deterministic map pools
.venv\Scripts\python.exe -m training.train --bench --envs 12 16 20 24
powershell -ExecutionPolicy Bypass -File training\train_loop.ps1 --run mem --arch memory `
    --warm-start runs\base\latest.pt --curriculum --t0 2026-11-26T14:20 --envs 24 --budget 100000000
.venv\Scripts\python.exe -m training.evaluate runs\mem\latest.pt --family rooms keydoor --sample
.venv\Scripts\python.exe -m training.select rank runs\mem\best_stage*.pt runs\mem\latest.pt
.venv\Scripts\python.exe -m training.select freeze ; .venv\Scripts\python.exe -m training.select latency
.venv\Scripts\python.exe -m demo.memory_demo runs\mem\latest.pt --variant B
.venv\Scripts\python.exe -m pytest -q tests --ignore=tests/test_remote.py --ignore=tests/test_validate.py
```
