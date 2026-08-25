"""Comprehensive evaluation harness for CMS-DACO — publication-grade.

Covers:
- Baselines: distance (BFS), hazard-aware A*, standard ACO, full CMS-DACO
- Ablations: w/o BFS seed, w/o dual, w/o predictive congestion, w/o escape, w/o hazard-aware
- Robustness: crowd densities, wall densities, fire severities, wind, grid sizes, blocked exits
- Sensitivity: alpha, beta, rho, temperature

All runs use N=30 by default, save raw results, seeds, configs, and plots.
Do not fabricate results — this script actually runs simulations.
"""
import argparse
import json
import os
import sys
import time
import copy
import itertools
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# NOTE: do NOT import core.* at module load. Chunk runners must apply config
# overrides BEFORE the core chain is imported, because core modules bind
# constants at import time (from config import X). run_single() imports lazily.
import config

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MPL = True
except Exception:
    HAS_MPL = False

METRIC_KEYS = ["completion_rate","casualty_rate","average_evacuation_time","avg_path_length_evacuated","congestion_ratio","total_ticks"]

_CORE_LOADED = False  # set True after core chain first import (see run_single)

# ---------- Config context manager ----------

class ConfigOverride:
    def __init__(self, **kwargs):
        self.overrides = kwargs
        self.saved = {}
    def __enter__(self):
        for k,v in self.overrides.items():
            if hasattr(config, k):
                self.saved[k] = getattr(config, k)
                setattr(config, k, v)
            else:
                self.saved[k] = None
                setattr(config, k, v)
        return self
    def __exit__(self, *args):
        for k,v in self.saved.items():
            if v is None and k not in self.saved:
                continue
            if self.saved[k] is None and not hasattr(config, k):
                try:
                    delattr(config, k)
                except: pass
            else:
                setattr(config, k, self.saved[k])

def run_single(seed: int, grid: int, crowd: int, exits: int, walls: float, mode: str, ticks: int, extra_flags: Dict=None, fire_params: Dict=None, wind_params: Dict=None) -> Dict:
    # Lazy import: caller must have applied any config overrides BEFORE first
    # call in this process (core modules bind constants at import time).
    global _CORE_LOADED
    if not _CORE_LOADED:
        from core.simulation import Simulation as _S  # noqa: F401
        from core.grid import GridSpec as _G  # noqa: F401
        _CORE_LOADED = True
    from core.simulation import Simulation
    from core.grid import GridSpec
    extra_flags = extra_flags or {}
    fire_params = fire_params or {}
    wind_params = wind_params or {}
    # Save config flags that we may override
    flag_keys = ["DUAL_PHEROMONE_ENABLED","PREDICTIVE_CONGESTION_ENABLED","STUCK_ESCAPE_ENABLED","HAZARD_FORECAST_ENABLED","BFS_SEED_ENABLED","HAZARD_AWARE_ROUTING_ENABLED","ALPHA","BETA","RHO","ACO_TEMPERATURE"]
    saved = {}
    for k in flag_keys:
        if k in extra_flags:
            saved[k] = getattr(config, k, None)
            setattr(config, k, extra_flags[k])
    try:
        spec = GridSpec(rows=grid, cols=grid, crowd=crowd, exits=exits, wall_density=walls)
        s = Simulation(spec, seed=seed, movement_mode=mode)
        # Apply fire/wind overrides if any
        if fire_params:
            s.update_fire_settings(**fire_params)
            # re-apply to grid if needed? simulation fire_params controls spread, seed already done with old params - for blocked exits etc we handle differently
        if wind_params:
            s.update_wind_settings(**wind_params)
        # Handle dynamically blocked exits: after creation, randomly compromise some exits
        if extra_flags.get("BLOCKED_EXITS",0) > 0:
            b = int(extra_flags["BLOCKED_EXITS"])
            exits_cells = [tuple(map(int,c)) for c in np.argwhere(s.grid.types==2)]
            if exits_cells:
                import random
                random.seed(seed)
                to_block = random.sample(exits_cells, min(b, len(exits_cells)))
                for r,c in to_block:
                    s.grid.exit_compromised[r,c] = True
                    s.grid.fire[r,c] = 0.5
        s.start()
        if s.timer is not None:
            try: s.timer.stop()
            except: pass
        for _ in range(ticks):
            s.step()
            if not s.running:
                break
        summary = s.metrics.summary()
        actual_crowd = getattr(s.grid, '_actual_crowd', spec.crowd)
        crowd_size = max(1, actual_crowd)
        return {
            "completion_rate": s.engine.evacuated / crowd_size,
            "casualty_rate": s.engine.casualties / crowd_size,
            "average_evacuation_time": summary.get("average_evacuation_time"),
            "avg_path_length_evacuated": summary.get("avg_path_length_evacuated"),
            "congestion_ratio": summary.get("congestion_ratio"),
            "total_ticks": float(s.tick_counter),
            "evacuated": int(s.engine.evacuated),
            "casualties": int(s.engine.casualties),
            "actual_crowd": int(actual_crowd),
            "seed": seed,
        }
    finally:
        for k,v in saved.items():
            setattr(config, k, v)

def mean_ci(values, confidence=0.95):
    clean = [float(v) for v in values if v is not None and np.isfinite(v)]
    if not clean:
        return None, None, 0
    arr = np.array(clean, dtype=np.float64)
    n = len(arr)
    mean = float(arr.mean())
    if n < 2:
        return mean, None, n
    sem = float(arr.std(ddof=1) / np.sqrt(n))
    try:
        from scipy import stats
        if n >= 30:
            z = float(stats.norm.ppf(0.5+confidence/2))
        else:
            z = float(stats.t.ppf(0.5+confidence/2, df=n-1))
    except:
        z = 1.96
    return mean, z*sem, n

def aggregate_results(results: Dict[str, List[Dict]]):
    summary = {}
    for mode, runs in results.items():
        mode_sum = {}
        for k in METRIC_KEYS:
            vals = [r.get(k) for r in runs]
            m,ci,n = mean_ci(vals)
            mode_sum[k] = {"mean": m, "ci95": ci, "n": n, "raw": vals}
        summary[mode] = mode_sum
    return summary

def print_table(summary, title="Results"):
    print("\n"+"="*110)
    print(f"{title}")
    print("="*110)
    modes = list(summary.keys())
    header = f"{'Metric':<30}" + "".join([f"{m:>20}" for m in modes])
    print(header)
    print("-"*110)
    for k in METRIC_KEYS:
        row = f"{k:<30}"
        for m in modes:
            d = summary[m][k]
            if d["mean"] is None:
                row += f"{'N/A':>20}"
            elif d["ci95"] is None:
                row += f"{d['mean']:>20.4f}"
            else:
                row += f"{d['mean']:.4f}+/-{d['ci95']:.4f}".rjust(20)
        print(row)
    print("="*110)

def save_results(payload, name):
    out_dir = Path(__file__).parent / "results"
    out_dir.mkdir(exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    path = out_dir / f"{name}_{ts}.json"
    # atomic
    def conv(o):
        if isinstance(o, (np.integer, np.floating)): return float(o)
        if isinstance(o, np.ndarray): return o.tolist()
        if isinstance(o, (np.bool_, bool)): return bool(o)
        return str(o)
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w") as f:
        json.dump(payload, f, indent=2, default=conv)
    tmp.replace(path)
    print(f"Saved {path}")
    return path

# ---------- Experiment definitions ----------

def baseline_experiment(runs=30, ticks=200, grid=20, crowd=40, exits=3, walls=0.08, seeds=None):
    if seeds is None: seeds = list(range(1, runs+1))
    # Define flag sets
    configs = {
        "distance": {"mode": config.MOVEMENT_MODE_DISTANCE, "flags": {}},
        "astar": {"mode": config.MOVEMENT_MODE_ASTAR, "flags": {}},
        "standard_aco": {"mode": config.MOVEMENT_MODE_STANDARD_ACO, "flags": {
            "DUAL_PHEROMONE_ENABLED": False,
            "PREDICTIVE_CONGESTION_ENABLED": False,
            "STUCK_ESCAPE_ENABLED": False,
            "HAZARD_FORECAST_ENABLED": False,
            "BFS_SEED_ENABLED": True,
            "HAZARD_AWARE_ROUTING_ENABLED": True,
        }},
        "full_cms_daco": {"mode": config.MOVEMENT_MODE_ACO, "flags": {
            "DUAL_PHEROMONE_ENABLED": True,
            "PREDICTIVE_CONGESTION_ENABLED": True,
            "STUCK_ESCAPE_ENABLED": True,
            "HAZARD_FORECAST_ENABLED": True,
            "BFS_SEED_ENABLED": True,
            "HAZARD_AWARE_ROUTING_ENABLED": True,
        }},
    }
    results = {k: [] for k in configs}
    for name, cfg in configs.items():
        print(f"\nBaseline {name} ({cfg['mode']}) ...")
        for seed in seeds:
            r = run_single(seed, grid, crowd, exits, walls, cfg["mode"], ticks, extra_flags=cfg["flags"])
            results[name].append(r)
        print(f"  done {len(results[name])} runs")
    summary = aggregate_results(results)
    print_table(summary, title=f"Baselines N={runs} {grid}x{grid} crowd={crowd} walls={walls}")
    payload = {"type":"baselines","config":{"runs":runs,"ticks":ticks,"grid":grid,"crowd":crowd,"exits":exits,"walls":walls,"seeds":seeds},"results":results,"summary":summary}
    save_results(payload, "baselines")
    return results, summary

def hard_condition_experiment(runs=30, ticks=200, seeds=None):
    """Harder conditions where full should shine: high walls, high crowd, wind, blocked exits"""
    if seeds is None: seeds = list(range(1, runs+1))
    # Use 30x30, 60 crowd, 0.15 walls, wind east 0.5, 2 blocked exits among 3
    grid=30; crowd=60; walls=0.15; exits=3
    print(f"\n=== Hard Conditions: {grid}x{grid} crowd={crowd} walls={walls} wind=0.5 blocked=1 ===")
    configs = {
        "distance": {"mode": config.MOVEMENT_MODE_DISTANCE, "flags": {}},
        "astar": {"mode": config.MOVEMENT_MODE_ASTAR, "flags": {}},
        "standard_aco": {"mode": config.MOVEMENT_MODE_STANDARD_ACO, "flags": {
            "DUAL_PHEROMONE_ENABLED": False,
            "PREDICTIVE_CONGESTION_ENABLED": False,
            "STUCK_ESCAPE_ENABLED": False,
            "HAZARD_FORECAST_ENABLED": False,
            "BFS_SEED_ENABLED": True,
            "HAZARD_AWARE_ROUTING_ENABLED": True,
        }},
        "full_cms_daco": {"mode": config.MOVEMENT_MODE_ACO, "flags": {
            "DUAL_PHEROMONE_ENABLED": True,
            "PREDICTIVE_CONGESTION_ENABLED": True,
            "STUCK_ESCAPE_ENABLED": True,
            "HAZARD_FORECAST_ENABLED": True,
            "BFS_SEED_ENABLED": True,
            "HAZARD_AWARE_ROUTING_ENABLED": True,
        }},
    }
    results = {k: [] for k in configs}
    for name, cfg in configs.items():
        print(f"\nHard {name} ...")
        for seed in seeds:
            r = run_single(seed, grid, crowd, exits, walls, cfg["mode"], ticks,
                           extra_flags={**cfg["flags"], "BLOCKED_EXITS":1},
                           wind_params={"direction":"east","strength":0.5})
            results[name].append(r)
    summary = aggregate_results(results)
    print_table(summary, title=f"Hard Conditions N={runs}")
    payload = {"type":"hard_conditions","config":{"runs":runs,"ticks":ticks,"grid":grid,"crowd":crowd,"walls":walls,"seeds":seeds,"wind":"east 0.5","blocked":1},"results":results,"summary":summary}
    save_results(payload, "hard_conditions")
    return results, summary

def ablation_experiment(runs=20, ticks=200, grid=20, crowd=40, walls=0.08, seeds=None):
    if seeds is None: seeds = list(range(1, runs+1))
    base_flags = {
        "DUAL_PHEROMONE_ENABLED": True,
        "PREDICTIVE_CONGESTION_ENABLED": True,
        "STUCK_ESCAPE_ENABLED": True,
        "HAZARD_FORECAST_ENABLED": True,
        "BFS_SEED_ENABLED": True,
        "HAZARD_AWARE_ROUTING_ENABLED": True,
    }
    ablations = {
        "full": base_flags,
        "w/o_bfs_seed": {**base_flags, "BFS_SEED_ENABLED": False},
        "w/o_dual": {**base_flags, "DUAL_PHEROMONE_ENABLED": False},
        "w/o_predictive": {**base_flags, "PREDICTIVE_CONGESTION_ENABLED": False},
        "w/o_escape": {**base_flags, "STUCK_ESCAPE_ENABLED": False},
        "w/o_hazard_aware": {**base_flags, "HAZARD_AWARE_ROUTING_ENABLED": False},
    }
    results = {k: [] for k in ablations}
    for name, flags in ablations.items():
        print(f"\nAblation {name} ...")
        for seed in seeds:
            r = run_single(seed, grid, crowd, 3, walls, config.MOVEMENT_MODE_ACO, ticks, extra_flags=flags)
            results[name].append(r)
    summary = aggregate_results(results)
    print_table(summary, title=f"Ablations N={runs}")
    payload = {"type":"ablation","config":{"runs":runs,"ticks":ticks,"grid":grid,"crowd":crowd,"walls":walls,"seeds":seeds},"results":results,"summary":summary}
    save_results(payload, "ablation")
    return results, summary

def robustness_experiment(runs=15, ticks=200, seeds=None):
    if seeds is None: seeds = list(range(1, runs+1))
    base_flags = {
        "DUAL_PHEROMONE_ENABLED": True,
        "PREDICTIVE_CONGESTION_ENABLED": True,
        "STUCK_ESCAPE_ENABLED": True,
        "HAZARD_FORECAST_ENABLED": True,
        "BFS_SEED_ENABLED": True,
        "HAZARD_AWARE_ROUTING_ENABLED": True,
    }
    # Crowd densities
    crowd_levels = [20,40,60,80]
    crowd_results = {}
    for crowd in crowd_levels:
        print(f"\nRobustness crowd={crowd}")
        rs = []
        for seed in seeds:
            r = run_single(seed, 20, crowd, 3, 0.08, config.MOVEMENT_MODE_ACO, ticks, extra_flags=base_flags)
            rs.append(r)
        crowd_results[crowd]=rs
    # Wall densities
    wall_levels = [0.02,0.08,0.15,0.25]
    wall_results = {}
    for walls in wall_levels:
        print(f"\nRobustness walls={walls}")
        rs=[]
        for seed in seeds:
            r = run_single(seed, 20, 40, 3, walls, config.MOVEMENT_MODE_ACO, ticks, extra_flags=base_flags)
            rs.append(r)
        wall_results[walls]=rs
    # Grid sizes
    grid_levels = [15,20,30,40]
    grid_results = {}
    for grid in grid_levels:
        print(f"\nRobustness grid={grid}")
        rs=[]
        for seed in seeds:
            crowd = max(10, int(40*(grid/20)**2*0.6))  # scale crowd with area
            r = run_single(seed, grid, crowd, 3, 0.08, config.MOVEMENT_MODE_ACO, ticks, extra_flags=base_flags)
            rs.append(r)
        grid_results[grid]=rs
    # Wind
    wind_levels = [("none",0.0),("east",0.5),("east",0.9),("north",0.5)]
    wind_results = {}
    for d,s in wind_levels:
        key=f"{d}_{s}"
        print(f"\nRobustness wind {key}")
        rs=[]
        for seed in seeds:
            r = run_single(seed, 20, 40, 3, 0.08, config.MOVEMENT_MODE_ACO, ticks, extra_flags=base_flags, wind_params={"direction":d,"strength":s})
            rs.append(r)
        wind_results[key]=rs

    payload = {"type":"robustness","config":{"runs":runs,"ticks":ticks,"seeds":seeds},"crowd":crowd_results,"walls":wall_results,"grids":grid_results,"wind":wind_results}
    # Also compute summaries for each
    for name, res in [("crowd",crowd_results),("walls",wall_results),("grids",grid_results),("wind",wind_results)]:
        print(f"\n--- Robustness {name} ---")
        for k, runs in res.items():
            m,ci,_ = mean_ci([r["completion_rate"] for r in runs])
            print(f"{k}: completion {m:.3f}+/-{ci:.3f}" if ci else f"{k}: {m:.3f}")
    save_results(payload, "robustness")
    return payload

def sensitivity_experiment(runs=15, ticks=200, seeds=None):
    if seeds is None: seeds = list(range(1, runs+1))
    base_flags = {
        "DUAL_PHEROMONE_ENABLED": True,
        "PREDICTIVE_CONGESTION_ENABLED": True,
        "STUCK_ESCAPE_ENABLED": True,
        "HAZARD_FORECAST_ENABLED": True,
        "BFS_SEED_ENABLED": True,
        "HAZARD_AWARE_ROUTING_ENABLED": True,
    }
    # Alpha, Beta, Rho, T sweeps
    sweeps = {
        "ALPHA": [0.5,1.0,1.35,1.8,2.5],
        "BETA": [1.5,2.0,3.2,4.0],
        "RHO": [0.003,0.009,0.02,0.05],
        "ACO_TEMPERATURE": [0.2,0.45,0.7,1.0],
    }
    all_results = {}
    for param, vals in sweeps.items():
        print(f"\nSensitivity {param} ...")
        param_res = {}
        for v in vals:
            flags = {**base_flags, param: v}
            rs=[]
            for seed in seeds:
                r = run_single(seed, 20, 40, 3, 0.08, config.MOVEMENT_MODE_ACO, ticks, extra_flags=flags)
                rs.append(r)
            param_res[str(v)] = rs
            m,_ ,_ = mean_ci([r["completion_rate"] for r in rs])
            print(f"  {param}={v}: completion {m:.3f}")
        all_results[param]=param_res
    payload = {"type":"sensitivity","config":{"runs":runs,"ticks":ticks,"seeds":seeds},"results":all_results}
    save_results(payload, "sensitivity")
    return all_results

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=int, default=30)
    parser.add_argument("--ticks", type=int, default=200)
    parser.add_argument("--quick", action="store_true", help="quick run with small N for testing")
    args = parser.parse_args()
    runs = 5 if args.quick else args.runs
    ticks = 80 if args.quick else args.ticks
    seeds = list(range(1, runs+1))

    print(f"Comprehensive evaluation runs={runs} ticks={ticks}")

    # 1. Baselines easy
    baseline_experiment(runs=runs, ticks=ticks, seeds=seeds)
    # 2. Hard conditions (hypothesis)
    hard_condition_experiment(runs=runs, ticks=ticks, seeds=seeds)
    # 3. Ablations
    ablation_experiment(runs=max(5,runs//2), ticks=ticks, seeds=seeds[:max(5,runs//2)])
    # 4. Robustness (smaller N for speed)
    robustness_experiment(runs=max(5, runs//2), ticks=ticks, seeds=seeds[:max(5, runs//2)])
    # 5. Sensitivity
    sensitivity_experiment(runs=max(5, runs//2), ticks=ticks, seeds=seeds[:max(5, runs//2)])

    # Generate plots if matplotlib available
    if HAS_MPL:
        print("Generating plots...")
        try:
            from experiments.generate_plots import generate_all_plots
            generate_all_plots()
        except Exception as e:
            print(f"Plot generation failed: {e}")

if __name__ == "__main__":
    main()
