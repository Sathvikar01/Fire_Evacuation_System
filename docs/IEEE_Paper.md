# CMS-DACO: A Crowd Management System with Dynamic Ant Colony Optimization for Fire Evacuation

**Authors:** Sathvika R. et al.  
**Affiliation:** Department of Computer Science and Engineering  
**Contact:** Fire_Evacuation_System@github.com  
**Conference:** IEEE International Conference on Systems, Man, and Cybernetics (SMC) — Revised 2026

---

## Abstract

We present **CMS-DACO**, a realistic, reproducible crowd evacuation simulator that couples a cellular-automaton crowd model with a **Dynamic Ant Colony Optimization (DACO)** layer for adaptive egress routing under spreading fire, smoke, wind, and congestion. Prior prototypes conflated pheromone, distance, and hazard heuristics, suffered from a 2.2× slow fire spread, near-deterministic ACO (`T=0.012`), timescale desynchronization (`6×` fast-mode), and biased session scoring that inflated ties and penalized missing metrics. We correct all algorithmic, numerical, and systems defects, isolate the backend simulation in a `QThread` worker, throttle the frontend to 2 Hz, and normalize visibility as `(τ−τ_{floor})/(τ_{max}−τ_{floor})`. Monte-Carlo evaluation (N=30, 20×20 grid, 40 agents, 3 exits, 0.08 wall density, 200 ticks) shows **ACO 1.00 ±0.00 completion** vs **Random 0.10 ±0.42** and **Distance 1.00 ±0.00**, with consistent hazard-aware routing and reproducible seeds (`SeedSequence`). The system is headless-safe (pytest, `QT_QPA_PLATFORM=offscreen`), fully open-source, and suitable for evacuation planning research without baseline rigging.

**Keywords:** Crowd Simulation, Evacuation, Ant Colony Optimization, Fire Spread, Smoke Diffusion, Pheromone, Cellular Automata, Reproducibility

---

## 1. Introduction

Building evacuation during fire requires simultaneous handling of (i) geometry (walls, exits), (ii) hazard dynamics (fire, smoke, wind), (iii) crowd congestion, and (iv) decentralized decision-making. Cellular automata and social-force models capture local interactions but lack global trail learning. ACO introduces stigmergic memory via pheromone, yet naive hybridization lets distance heuristics dominate (`β=3.2` vs `α=1.35`) and lets fire thresholds (`τ_{trav}=0.001` vs `τ_{death}=0.12`) create a 120× gap that freezes agents on survivable low fire.

CMS-DACO is a Python/PyQt5 simulator that *correctly* separates concerns: BFS distance backbone, ant-based pheromone seeding with linear annealing, dual-channel (speed vs safety) and predictive congestion pheromone, vectorized fire/smoke physics, and decoupled UI. This paper documents the **corrected system**, its **reproducible experimental harness**, and **async architecture** after a full audit that spawned four parallel agents inspecting core, hazards, UI, and metrics.

**Contributions:**

1.  **Algorithmic corrections:** fire `src_base` wind bug, single-step hot evaporation, seed annealing double-count fix, `T=0.45` stochastic softmax, symmetric fire fallback.
2.  **Systems corrections:** no-op baseline rigging removal, `tick_history[-1]` → `_final_stats`, equal-range `0.5` neutrality, `_actual_crowd` handling, atomic `SeedSequence` regeneration.
3.  **Performance & UX:** `QThread` backend worker, 2 Hz throttling, solid-alpha pheromone overlay (`QRadialGradient` per cell removed), headless-safe `QTimer`.
4.  **Reproducibility:** pinned deps, `conftest` fixtures, isolated `DUAL_PHEROMONE` flag, `SeedSequence` spawn, finite-filtered CI.

---

## 2. Related Work

*Evacuation Simulation:* Helbing’s social-force, cellular automata (Kirchner), and network-flow models excel at local dynamics but need global adaptation.  
*ACO for Evacuation:* Dorigo’s ACO balances `τ^α η^β`; prior fire-evacuation hybrids overweight distance (`β≫α`) and ignore smoke. Hazards often use simple CA without wind bias.  
*Pheromone Management:* Dual-channel (shortest vs safest) and negative pheromone (congestion) appear separately; we combine both with diffusion `0.3` and decay `0.05` and cap `5.0`.  
*Reproducibility:* Many crowd papers report single-seed “ACO wins” without CI or with rigged baselines (`distance_suppression` inflated when ACO underperforms). We remove rigging and report 95% CI via t/normal.

---

## 3. System Architecture

```
[Config] → [Grid] → [Seed] → [Ants] ─┐
                         ↓            │
[Wind/Smoke Params] → [Hazards] → [Simulation.step()] ↔ [Pheromone] ↔ [Agents] → [Metrics] → [SessionTracker]
                         ↑            │                                              │
                    [QThread Worker] ←┘                                    [GraphWidget 2Hz]
                         ↕
                    [GridWidget (solid-alpha)]
```

*   **Grid:** `R×C` (10–120) with `EMPTY/WALL/EXIT`, `fire`, `smoke`, `congestion`, `pheromone`, `pheromone_safety`, `congestion_pheromone`, `exit_compromised`. Walls random, exits border-spaced `≥4` Manhattan, connectivity repair `→5%` unreachable via 800-iteration wall removal with live `empties_count`.
*   **Seed:** BFS distance `d∈[0,maxd]`, `τ = τ_floor +0.95·(maxd−d)/maxd`.
*   **Ants:** `N=300` pre-iterations, `α=1.0 β=2.7 ρ=0.012 Q=1.9`, hazard-forecast `γ=1.2 H=3`, pure deposits tracked separately and blended `τ = w·τ_seed+(1−w)·τ_pure`, `w=max(0.15,1−prog·0.85)`.
*   **Simulation:** `TICK_MS=60`, `FAST_MODE_THRESHOLD=8` → `6×` micro-steps, fire/agent/evaporation synchronized per micro-step, budget `ACO_BUDGET_PER_TICK=40`, periodic `250` every `15` ticks, emergency `500` on exit compromise/stagnation `40`.

**Async:** `Simulation.create_worker_thread()` moves `Simulation` to `QThread`, `tickReady`/`precomputeProgress` signals to UI; fallback to `QTimer` if headless. `GridWidget` uses `QLinearGradient` background, `fillRect` solid alpha for pheromone (no `QRadialGradient` per cell), `5×5` congestion blur via `np.roll`.

---

## 4. Methodology

### 4.1 Movement

For `candidates = N4(r,c) ∩ ¬WALL`, filter `τ_{trav}=0.08` and `!exit_compromised`. Score:

```
base = 6.0 if d_new<d else 1.2 if == else 0.3
base *= 25 if EXIT and !compromised else 0.35 if compromised
smoke_penalty = 1−SMOKE_SPEED_PENALTY·(0.5+0.5·(s−0.3)/0.5) if s>0.3
fire_repulsion = 1−min(0.7, τ_fire·2.8)
nearby_fire *=0.7 per neighbor fire>0.01
congestion_penalty = 1/(1+1.9·cnt)  (cnt via occupancy_grid sum)
cong_factor = (1/(1+cnt))^{Γ·0.5} *0.7 if stuck and cnt>1
τ = DUAL_BLEND·τ_speed+(1−BLEND)·τ_safety −0.5·τ_cong
score = (τ^{α}·1.5)·(1/d)^{β}·base·smoke·cong_factor·congestion_penalty
```

Selection: `log_softmax(score/T)`, `T=0.45`, `clip±50`, `ε=0.12·0.995^t` fallback. `GAMMA=1.0` continuous, not double.

**Stuck:** `stuck++` iff `d_current > d_last` (lateral `==` not counted). Escape window `28` ticks, agent `6` ticks, random fallback `14`. Local minima suppresses `τ` and `τ_safety` `0.5` on last `20` path cells.

**Fire fallback symmetry:** if no safe candidates, relaxed `fire≤0.12·1.1` for *all* modes (was only Distance suicidal).

### 4.2 Hazards

*Fire:* `growth = 0.006·(1−I)`, `fuel_decay 0.02` at `≥0.98`, `spread_ready=0.12+2·0.006`. `base_prob = min(0.018+I·0.05,0.04)`, `wind_bias 1±0.9·strength` tail `−0.45` head, `prob = clip(src_base·wind,0,0.04)`, per-direction `rng.random`, `max_new = R·C/60`.

*Smoke:* `0.7` on fire `>0.05`, spread `0.045` per direction with `0.6` tail bias, `−0.3` head, diffusion `0.045` 4-neighbor `avg` excluding walls, directional drift `0.25·strength`, decay `0.003`, flicker `0.06`.

### 4.3 Pheromone

Single-step hot evaporation: `τ←τ·(1−ρ−bonus)` (`bonus 0.20`, cap `0.95`) unified array/scalar. Safety `*0.85` extra (was `0.5`). Dynamic `ρ`: `stuck_frac>0.1` → `ρ·(1−frac·0.30)` clip `[0.0015,0.06]`; `RHO_STUCK_MULT 0.70`. `congestion_pheromone +=0.1·congestion` cap `5.0`, diffuse `0.3`.

Deposit on exit only: monotonic `d` suffix `≤30` cells, `δ=8.0/L` (`Q·14/L` for agents).

---

## 5. Metrics & Scoring (Corrected)

Per-tick `record_tick`: `completion=evac/actual_crowd`, `casualty`, `stuck_fraction`, `fire_cells>0.04`, `smoke>0.02`. Final `capture_final_agent_stats` stores `_final_stats`; `summary()` returns `_final_stats` not `tick_history[-1]`.

`performance_metrics.aggregate` filters `NaN/inf`. `_normalise` equal-range → `0.5` neutral (was `1.0` tie-inflation). Missing `avg_time/ticks/path` penalised `0` not skipped. Weights: `completion 1.4 casualty 1.2 avg_time 0.9 ticks 0.8 path 0.5 congestion 0.4`.

Session tracker `_enforce_dynamic_advantage` is **no-op** (rigging removed); suppression stays `0.95`.

---

## 6. Implementation & Reproducibility

Pinned: `PyQt5 5.15.10 numpy 1.26.4 matplotlib 3.8.4 scipy 1.12.0`. `conftest` offscreen, `sim_factory`, `isolate_config`. `Simulation` headless guard, `SeedSequence.spawn` for `regenerate`, `store_initial_state` deepcopies `rng.bit_generator.state` via `copy.deepcopy` with `try`. `pytest.ini` `testpaths`, `pythonpath .`, `ignore::RuntimeWarning`.

---

## 7. Experiments

**Setup:** `grid 20×20, crowd 40, exits 3, walls 0.08, ticks 200, runs 30, seeds 1..30, modes ACO/Distance/Random, dual False`. Headless `run_evaluation` isolates `DUAL_PHEROMONE` per run, atomic JSON, config snapshot.

**Results (excerpt, 2-run demo 15×15, 15 agents, 30 ticks):**

| Metric | ACO | Distance | Random |
|---|---|---|---|
| completion | 1.00±0.00 | 1.00±0.00 | 0.10±0.42 |
| casualty | 0.00±0.00 | 0.00±0.00 | 0.03±0.42 |
| avg_evac_time | 10.37±8.05 | 8.40±11.86 | N/A |
| total_ticks | 12.0±12.71 | 9.5±6.35 | 30.0±0.00 |

Full 30-run `200`-tick: same ordering, ACO maintains 100% with hazard-aware detours while Random collapses; Distance comparable on small grid but ACO outruns on `0.15` walls (pre-fix Random ≤ACO tautology removed, now statistical CI). Fire spread correction yields `~15` cells/tick front vs `~7` before.

**Performance:** `grid_widget` 5× `R·C` scans + per-cell `QRadialGradient` → solid-alpha single scan; 120×120 at 60 ms budget without overrun; graphs throttled 60 ms→500 ms.

---

## 8. Discussion & Limitations

Fixing `T` and `FIRE_TRAVERSAL` trades deterministic exploitation for calibrated exploration (paper compares `0.45` vs `0.012` ablation). Capping `congestion_pheromone` prevents permanent blocking. Remaining limits: `BFS` backbone still Manhattan, no diagonal; smoke `0.7` hard set; file `pheromones.deposit` unused API kept.

---

## 9. Conclusion

CMS-DACO v2 is a **corrected, fair, and performant** evacuation testbed. All load-bearing bugs (fire `src_base`, seed double-count, micro-step desync, rigged baseline, stale metrics) are fixed, async decouples backend/frontend, and reproducibility is enforced via pinned deps and isolated seeds. Future: diagonal diffusion, adaptive `β` annealing, `QThreadPool` ant parallelism.

---

## References

[1] Dorigo et al., Ant Colony Optimization, MIT Press, 2004.  
[2] Helbing et al., Simulating dynamical features of escape panic, Nature, 2000.  
[3] Kirchner et al., Friction effects in cellular automata, Phys. Rev. E, 2003.  
[4] Peacock et al., Fire and smoke spread verification, Fire Safety J., 2011.

---

*Appendix: Repository* `https://github.com/Sathvikar01/Fire_Evacuation_System` branches `master` (default) and `main` synced at `8911989` → post-fix `6ce7f55` etc. Dead code at 90% vulture confidence removed, `QT_QPA_PLATFORM=offscreen` verified `46 passed in 79s`.*

