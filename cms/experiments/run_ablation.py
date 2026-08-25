"""Chunked ablation + robustness + sensitivity runners.

Each invocation handles one variant across a seed range and saves partial JSON.
Merge at the end. This keeps every shell call under tool timeouts while using
parallel processes for throughput.

Usage:
  python run_ablation.py --variant full --seed-start 1 --seed-end 15
  python run_ablation.py --merge
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
WALLS = 0.12          # moderately hard so differences show
TICKS = 40

BASE_FLAGS = {
    "DUAL_PHEROMONE_ENABLED": True,
    "PREDICTIVE_CONGESTION_ENABLED": True,
    "STUCK_ESCAPE_ENABLED": True,
    "HAZARD_FORECAST_ENABLED": True,
    "BFS_SEED_ENABLED": True,
    "HAZARD_AWARE_ROUTING_ENABLED": True,
}

VARIANTS = {
    "full": BASE_FLAGS,
    "wo_bfs_seed": {**BASE_FLAGS, "BFS_SEED_ENABLED": False},
    "wo_dual": {**BASE_FLAGS, "DUAL_PHEROMONE_ENABLED": False},
    "wo_predictive": {**BASE_FLAGS, "PREDICTIVE_CONGESTION_ENABLED": False},
    "wo_escape": {**BASE_FLAGS, "STUCK_ESCAPE_ENABLED": False},
    "wo_hazard_aware": {**BASE_FLAGS, "HAZARD_AWARE_ROUTING_ENABLED": False},
}


def run_chunk(variant, seed_start, seed_end):
    flags = VARIANTS[variant]
    # CRITICAL: apply flags BEFORE any core import (fresh process per chunk).
    for k, v in flags.items():
        setattr(config, k, v)
    results = []
    t0 = time.time()
    for seed in range(seed_start, seed_end + 1):
        r = run_single(seed, GRID, CROWD, EXITS, WALLS,
                       config.MOVEMENT_MODE_ACO, TICKS)
        results.append(r)
        print(f"{variant} seed {seed}: comp={r['completion_rate']:.3f} "
              f"({time.time()-t0:.0f}s)", flush=True)
    out = Path(__file__).parent / "results" / f"abl_{variant}_{seed_start}_{seed_end}.json"
    def conv(o):
        if isinstance(o, (np.integer, np.floating)): return float(o)
        if isinstance(o, np.ndarray): return o.tolist()
        return str(o)
    tmp = out.with_suffix(".tmp")
    tmp.write_text(json.dumps({"variant": variant, "results": results}, default=conv))
    tmp.replace(out)
    print(f"Saved {out}", flush=True)


def merge():
    files = sorted(Path(__file__).parent.glob("results/abl_*_[0-9]*_[0-9]*.json"))
    grouped = {}
    for f in files:
        data = json.loads(f.read_text())
        grouped.setdefault(data["variant"], []).extend(data["results"])
    missing = [v for v in VARIANTS if v not in grouped]
    if missing:
        print(f"Missing variants: {missing}")
        sys.exit(1)
    summary = aggregate_results(grouped)
    print("\nABLATION SUMMARY")
    for k in ["completion_rate","casualty_rate","average_evacuation_time",
              "avg_path_length_evacuated","congestion_ratio","total_ticks"]:
        row = f"{k:<32}"
        for m in VARIANTS:
            d = summary[m][k]
            if d["mean"] is None:
                row += f"{'N/A':>20}"
            elif d["ci95"] is None:
                row += f"{d['mean']:>20.4f}"
            else:
                row += f"{d['mean']:.4f}±{d['ci95']:.4f}".rjust(22)
        print(row)
    payload = {"type": "ablation", "config": {"grid": GRID, "crowd": CROWD,
               "exits": EXITS, "walls": WALLS, "ticks": TICKS},
               "results": grouped, "summary": summary}
    save_results(payload, "ablation_final")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", choices=list(VARIANTS.keys()))
    ap.add_argument("--seed-start", type=int, default=1)
    ap.add_argument("--seed-end", type=int, default=30)
    ap.add_argument("--merge", action="store_true")
    args = ap.parse_args()
    if args.merge:
        merge()
    else:
        run_chunk(args.variant, args.seed_start, args.seed_end)
