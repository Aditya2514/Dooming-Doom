# Ablations and post-submission experiments

Every number below is from **held-out maps never trained on**, played by the **submitted inference path**
(`submission/agent.py`, one CPU thread, actions sampled), and **paired**: each model plays the same
600 episodes (130 validation maps × 3 seeds + 70 floor-pad-exit maps × 3 seeds). Differences are
reported as mean ± 95% confidence half-width of the per-episode paired difference.
Proxy score: our reimplementation of the published scoring categories (`training/evaluate.py`).

## 1. Does the spatial map help?

Same DAgger recipe, same seed, same teacher data (iteration 0 is byte-identical), only `--no-map` changed
(`runs/il4` vs `runs/abl_nomap`, both at iteration 6).

| Held-out | **With spatial map** | Without |
|---|---|---|
| Multi-room exits | **0.95** | 0.35 |
| Easy key-door exits | **0.95** | 0.85 |
| Key-door A/B/C exits | **0.67** | 0.50 |
| Mixed gauntlet exits | **0.65** | 0.38 |
| Floor-pad exits | **0.41** | 0.29 |
| Combat kill fraction | 0.98 | 0.95 |
| Episodes ending at the tick cap | **249** | 363 |
| **Paired advantage of the map** | **+23.9 ± 5.7 proxy, +19.2 ± 4.8 % exits** | |

On identical teacher data the map model also imitates better from the start (teacher agreement 93.1 %
vs 88.8 % after iteration 0; 84.3 % vs 78.9 % at iteration 6): the teacher's exploration decisions depend
on where it has already looked, and the map is the only input that carries that.

## 2. Is DAgger needed, or would behavioural cloning do?

Iteration 0 of `runs/il4` is plain behavioural cloning on 200k teacher-driven states; iteration 6 adds six
rounds of DAgger (student-driven states labelled by the teacher).

| Held-out | **DAgger (iter 6)** | Behavioural cloning (iter 0) |
|---|---|---|
| Multi-room exits | **0.95** | 0.45 |
| Easy key-door exits | **0.95** | 0.35 |
| Key-door A/B/C exits | **0.67** | 0.10 |
| Mixed gauntlet exits | **0.65** | 0.22 |
| Floor-pad exits | **0.41** | 0.25 |
| Deaths (of 600) | **12** | 23 |
| **Paired advantage of DAgger** | **+45.0 ± 5.9 proxy, +34.0 ± 4.7 % exits** | |

Cloning alone almost never finishes a key-door map: the long-horizon chain (find key → return → exit)
breaks at the first state the teacher never visited. DAgger's student-driven data teaches the recovery.

## 3. Action selection (submitted model)

| Mode | Paired vs sampling at T = 1.0 (submitted) |
|---|---|
| Sampling, T = 0.75 | −1.4 ± 4.6 proxy, −1.7 ± 4.0 % exits |
| Sampling, T = 0.5 | +4.0 ± 5.5 proxy, +3.3 ± 4.5 % exits (pad exits +10 pts, key-door −10 pts) |
| **Greedy (argmax)** | **−6.1 ± 5.0 proxy, −5.5 ± 4.2 % exits**; 268 vs 235 episodes end at the tick cap |

Greedy action selection gets stuck in loops; sampling escapes them. No temperature is significantly
better than 1.0.

## 4. Which DAgger snapshot shipped

| Model | Validation exits | Floor-pad exits | `two_rooms.wad` (10 runs) |
|---|---|---|---|
| `il4` iteration 6 | 0.646 | 0.414 | 9 / 10 |
| **`il5` iteration 4 (submitted)**: iteration 6 + 4 iterations on train + train2 (half floor-pad exits) | **0.667** | **0.471** | **10 / 10** |

Paired: +3.8 ± 5.0 proxy, +3.3 ± 4.2 % exits: better on key-door (0.77 vs 0.67), gauntlets (0.74 vs 0.65)
and pad exits, worse on plain multi-room maps (0.80 vs 0.95).

## 5. PPO fine-tuning

See `training/README.md`: from an imitation start, PPO with an untrained critic lowered held-out exits
(0.58 → 0.38 after ~34 policy updates, two runs). A third attempt anchors the policy with
KL(π_init ‖ π) instead of the teacher's labels and warms the critic up for 60 updates before any policy
update (`training/finetune.py --kl-ref`); results are added here when evaluated.

## Reproduce

```powershell
.venv\Scripts\python.exe -m training.dagger --run abl_nomap --no-map --workers 10 --iters 30 --steps0 200000 --steps 80000 --updates0 1000 --updates 500 --cap 1000000
.venv\Scripts\python.exe -m training.evaluate runs\abl_nomap\iter06.pt --pool validation --family rooms keyeasy keydoor mixed combat --seeds 3 --out nomap_val.json
.venv\Scripts\python.exe -m training.evaluate runs\abl_nomap\iter06.pt --pool valpad --family rooms keyeasy keydoor mixed --seeds 3 --out nomap_pad.json
```
(paired differences: match episodes on `(pool, map, seed)` across the JSON files.)
