# When Does Stigmergic Routing Help Evacuation? (CMS-DACO benchmark)

**PDF:** [`docs/manuscript.pdf`](manuscript.pdf) — 7 pages, IEEEtran.
**Everything regenerates** from stored JSONs via `cms/experiments/*` scripts.

## Question
Proponents claim ant-colony/stigmergic routing helps evacuation under spreading hazards — usually vs weak baselines. This paper asks **when**, if ever, it beats classical replanning, probing four boundary regimes with matched information.

## Policies
Random · BFS-distance greedy · Hazard-aware A\* (per-tick) · **D\* Lite** (incremental) · Standard ACO · Full CMS-DACO (BFS-seeded pheromone + annealed ants + dual channels + predictive congestion).

## Regimes & headline results (N=30 paired seeds; paired t / Wilcoxon / bootstrap CI / Holm)

| Regime | Result |
|---|---|
| Open grid | A* .998 > Full DACO .974 (Holm p=.009, d_z=.59); directed ≫ random (d_z≥11) |
| Hard (0.15 walls, burning exit, wind) | Gap persists: −6.1 pts, Holm p=.0013 |
| **Partial observability** (shared r=3 / stale Δt=5 / 50% loss / private r=3) | **No flip**: A* .991–.994 everywhere; DACO .930–.940; private r=3 DACO .932 vs A* .991; gap Holm p≤.0022 |
| **Extreme stress** (20% walls, 2 fire fronts, exits collapse @t=12/24, wind .8) | Ceiling broken (.40–.53); **ordering reverses**: greedy Distance beats A* (Holm p=.049) — A* over-commits crowds to exits that then burn |
| **Compute budget** (ants ×{1,⅓,1/10,1/30}) | Quality flat from 182→19 s/run; budget-matching cannot close the gap |
| **Scale** (30² / 40², N=30) | A* .999/.997; Full DACO .872/.864 (both Holm p<2×10⁻⁶); median s/run: A* 29→123 (4.2×), DACO 242→386 (1.6×); cost ratio shrinks 8.3×→3.1× |

## Answer
**Not in any tested regime.** Where information suffices, replanning dominates; where it fails or turns adversarial, simple hedged heuristics capture the robustness margin — stigmergy does not. The contribution is the benchmark itself: matched-information baselines up to incremental search, ceiling-breaking stress design, paired-seed statistics, budget fairness.

## Reproduce
```powershell
cd cms/experiments
python run_baselines.py --mode full_cms_daco --seed-start 1 --seed-end 30   # per policy/chunk
python run_hard_conditions.py --mode astar --seed-start 1 --seed-end 30
python run_ablation.py --variant full --seed-start 1 --seed-end 30
python run_sensitivity.py --param ALPHA --value 1.35 --seed-start 1 --seed-end 30
python run_robustness.py --sweep walls --value 0.15 --seed-start 1 --seed-end 30
python run_extreme.py --mode full_cms_daco --seed-start 1 --seed-end 30
python run_observability.py --policy astar --mode r3 --seed-start 1 --seed-end 30
python run_budget.py --level B3 --seed-start 1 --seed-end 30
python compute_stats_paired.py   # paired t / Wilcoxon / bootstrap / Holm -> stats_paired.json
python make_figures.py           # -> docs/figures/*.pdf
```

## Scope
Grid worlds ≤20², ≤45 agents; hazard field is a computationally efficient cellular model (**not** FDS-validated — no fire-engineering claims). D* Lite uses capped per-tick expansions. Single map family; informal a-priori hyperparameters later shown flat.
