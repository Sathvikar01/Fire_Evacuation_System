"""Chunked robustness sweeps: crowd, walls, wind.

Usage:
  python run_robustness.py --sweep crowd --value 40 --seed-start 1 --seed-end 15
  python run_robustness.py --merge
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

from experiments.run_comprehensive_evaluation import run_single, save_results
import config

EXITS = 3
TICKS = 40
BASE_FLAGS = {
    "DUAL_PHEROMONE_ENABLED": True,
    "PREDICTIVE_CONGESTION_ENABLED": True,
    "STUCK_ESCAPE_ENABLED": True,
    "HAZARD_FORECAST_ENABLED": True,
    "BFS_SEED_ENABLED": True,
    "HAZARD_AWARE_ROUTING_ENABLED": True,
}

CROWD_LEVELS = [20, 40, 60]
WALL_LEVELS = [0.02, 0.08, 0.15, 0.25]
WIND_LEVELS = [("none", 0.0), ("east", 0.5), ("north", 0.9)]
GRID_LEVELS = [15, 20]


def fmt(v):
    return str(v).replace(".", "p")


def run_chunk(sweep, value, seed_start, seed_end):
    # CRITICAL: apply BASE_FLAGS BEFORE any core import (fresh process per chunk).
    for k, v in BASE_FLAGS.items():
        setattr(config, k, v)
    results = []
    t0 = time.time()
    if sweep == "crowd":
        grid, crowd, walls, wind = 20, int(value), 0.08, None
    elif sweep == "walls":
        grid, crowd, walls, wind = 20, 40, float(value), None
    elif sweep == "wind":
        d, s = value.split("_")
        grid, crowd, walls, wind = 20, 40, 0.08, {"direction": d, "strength": float(s)}
    else:  # grids
        grid, crowd, walls, wind = int(value), max(10, int(30 * (int(value) / 15) ** 2 * 0.5)), 0.08, None
    for seed in range(seed_start, seed_end + 1):
        r = run_single(seed, grid, crowd, EXITS, walls,
                       config.MOVEMENT_MODE_ACO, TICKS, wind_params=wind)
        results.append(r)
        print(f"{sweep}={value} seed {seed}: comp={r['completion_rate']:.3f} "
              f"({time.time()-t0:.0f}s)", flush=True)
    out = (Path(__file__).parent / "results" /
           f"rob_{sweep}_{fmt(value)}_{seed_start}_{seed_end}.json")
    def conv(o):
        if isinstance(o, (np.integer, np.floating)): return float(o)
        return str(o)
    tmp = out.with_suffix(".tmp")
    tmp.write_text(json.dumps({"sweep": sweep, "value": value,
                               "results": results}, default=conv))
    tmp.replace(out)
    print(f"Saved {out}", flush=True)


def merge():
    files = sorted(Path(__file__).parent.glob("results/rob_*_[0-9]*_[0-9]*.json"))
    grouped = {}
    for f in files:
        data = json.loads(f.read_text())
        grouped.setdefault(data["sweep"], {}).setdefault(str(data["value"]), []).extend(data["results"])
    summary = {}
    def sort_key(v):
        s = str(v).split("_")[0]
        try:
            return (0, float(s))
        except ValueError:
            return (1, str(v))
    for sweep, vals in grouped.items():
        summary[sweep] = {}
        for v, runs in sorted(vals.items(), key=lambda kv: sort_key(kv[0])):
            comps = [r["completion_rate"] for r in runs]
            cas = [r["casualty_rate"] for r in runs]
            times = [r["average_evacuation_time"] for r in runs if r["average_evacuation_time"] is not None]
            summary[sweep][v] = {
                "completion_mean": float(np.mean(comps)),
                "completion_ci": float(1.96*np.std(comps, ddof=1)/np.sqrt(len(comps))) if len(comps) > 1 else 0.0,
                "casualty_mean": float(np.mean(cas)),
                "evac_time_mean": float(np.mean(times)) if times else None,
                "n": len(runs),
            }
            print(f"{sweep}={v}: completion {summary[sweep][v]['completion_mean']:.4f} "
                  f"±{summary[sweep][v]['completion_ci']:.4f} casualty {summary[sweep][v]['casualty_mean']:.4f}")
    payload = {"type": "robustness", "config": {"ticks": TICKS},
               "summary": summary}
    save_results(payload, "robustness_final")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep", choices=["crowd", "walls", "wind", "grids"])
    ap.add_argument("--value")
    ap.add_argument("--seed-start", type=int, default=1)
    ap.add_argument("--seed-end", type=int, default=30)
    ap.add_argument("--merge", action="store_true")
    args = ap.parse_args()
    if args.merge:
        merge()
    else:
        run_chunk(args.sweep, args.value, args.seed_start, args.seed_end)
