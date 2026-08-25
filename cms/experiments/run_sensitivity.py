"""Chunked sensitivity sweep for ALPHA, BETA, RHO, ACO_TEMPERATURE.

Usage:
  python run_sensitivity.py --param ALPHA --value 1.0 --seed-start 1 --seed-end 15
  python run_sensitivity.py --merge
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from experiments.run_comprehensive_evaluation import (
    run_single, aggregate_results, save_results,
)
import config

GRID = 15
CROWD = 30
EXITS = 3
WALLS = 0.12
TICKS = 40

BASE_FLAGS = {
    "DUAL_PHEROMONE_ENABLED": True,
    "PREDICTIVE_CONGESTION_ENABLED": True,
    "STUCK_ESCAPE_ENABLED": True,
    "HAZARD_FORECAST_ENABLED": True,
    "BFS_SEED_ENABLED": True,
    "HAZARD_AWARE_ROUTING_ENABLED": True,
}

SWEEPS = {
    "ALPHA": [0.5, 1.0, 1.35, 1.8, 2.5],
    "BETA": [1.5, 2.0, 3.2, 4.0],
    "RHO": [0.003, 0.009, 0.02, 0.05],
    "ACO_TEMPERATURE": [0.2, 0.45, 0.7, 1.0],
}


def fmt(v):
    return str(v).replace(".", "p")


def run_chunk(param, value, seed_start, seed_end):
    # CRITICAL: apply param BEFORE any core import (fresh process per chunk).
    for k, v in BASE_FLAGS.items():
        setattr(config, k, v)
    setattr(config, param, value)
    results = []
    t0 = time.time()
    for seed in range(seed_start, seed_end + 1):
        r = run_single(seed, GRID, CROWD, EXITS, WALLS,
                       config.MOVEMENT_MODE_ACO, TICKS)
        results.append(r)
        print(f"{param}={value} seed {seed}: comp={r['completion_rate']:.3f} "
              f"({time.time()-t0:.0f}s)", flush=True)
    out = Path(__file__).parent / "results" / f"sens_{param}_{fmt(value)}_{seed_start}_{seed_end}.json"
    def conv(o):
        if isinstance(o, (np.integer, np.floating)): return float(o)
        if isinstance(o, np.ndarray): return o.tolist()
        return str(o)
    tmp = out.with_suffix(".tmp")
    tmp.write_text(json.dumps({"param": param, "value": value,
                               "results": results}, default=conv))
    tmp.replace(out)
    print(f"Saved {out}", flush=True)


def merge():
    files = sorted(Path(__file__).parent.glob("results/sens_*_[0-9]*_[0-9]*.json"))
    grouped = {}
    for f in files:
        data = json.loads(f.read_text())
        grouped.setdefault(data["param"], {}).setdefault(str(data["value"]), []).extend(data["results"])
    # verify completeness
    for param, vals in SWEEPS.items():
        for v in vals:
            if str(v) not in grouped.get(param, {}):
                print(f"Missing {param}={v}")
                sys.exit(1)
    summary = {}
    for param, vals in grouped.items():
        summary[param] = {}
        for v, runs in vals.items():
            comps = [r["completion_rate"] for r in runs]
            times = [r["average_evacuation_time"] for r in runs if r["average_evacuation_time"] is not None]
            summary[param][v] = {
                "completion_mean": float(np.mean(comps)),
                "completion_ci": float(1.96*np.std(comps, ddof=1)/np.sqrt(len(comps))) if len(comps) > 1 else 0.0,
                "evac_time_mean": float(np.mean(times)) if times else None,
                "n": len(runs),
            }
            print(f"{param}={v}: completion {summary[param][v]['completion_mean']:.4f} "
                  f"±{summary[param][v]['completion_ci']:.4f} evac_t {summary[param][v]['evac_time_mean']}")
    payload = {"type": "sensitivity", "config": {"grid": GRID, "crowd": CROWD,
               "exits": EXITS, "walls": WALLS, "ticks": TICKS},
               "summary": summary}
    save_results(payload, "sensitivity_final")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--param", choices=list(SWEEPS.keys()))
    ap.add_argument("--value", type=float)
    ap.add_argument("--seed-start", type=int, default=1)
    ap.add_argument("--seed-end", type=int, default=30)
    ap.add_argument("--merge", action="store_true")
    args = ap.parse_args()
    if args.merge:
        merge()
    else:
        run_chunk(args.param, args.value, args.seed_start, args.seed_end)
