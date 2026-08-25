"""Chunked baseline runner (distance / astar / standard_aco / full_cms_daco).

Flags are applied to config BEFORE core import (fresh process per chunk).
Usage:
  python run_baselines.py --mode full_cms_daco --seed-start 1 --seed-end 15
  python run_baselines.py --merge
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
WALLS = 0.08
TICKS = 40

CONFIGS = {
    "distance": {"mode": config.MOVEMENT_MODE_DISTANCE, "flags": {}},
    "astar": {"mode": config.MOVEMENT_MODE_ASTAR, "flags": {}},
    "standard_aco": {
        "mode": config.MOVEMENT_MODE_STANDARD_ACO,
        "flags": {
            "DUAL_PHEROMONE_ENABLED": False,
            "PREDICTIVE_CONGESTION_ENABLED": False,
            "STUCK_ESCAPE_ENABLED": False,
            "HAZARD_FORECAST_ENABLED": False,
            "BFS_SEED_ENABLED": True,
            "HAZARD_AWARE_ROUTING_ENABLED": True,
        },
    },
    "full_cms_daco": {
        "mode": config.MOVEMENT_MODE_ACO,
        "flags": {
            "DUAL_PHEROMONE_ENABLED": True,
            "PREDICTIVE_CONGESTION_ENABLED": True,
            "STUCK_ESCAPE_ENABLED": True,
            "HAZARD_FORECAST_ENABLED": True,
            "BFS_SEED_ENABLED": True,
            "HAZARD_AWARE_ROUTING_ENABLED": True,
        },
    },
}


def run_chunk(mode_name, seed_start, seed_end):
    cfg = CONFIGS[mode_name]
    for k, v in cfg["flags"].items():
        setattr(config, k, v)
    results = []
    t0 = time.time()
    for seed in range(seed_start, seed_end + 1):
        r = run_single(seed, GRID, CROWD, EXITS, WALLS, cfg["mode"], TICKS)
        results.append(r)
        print(f"{mode_name} seed {seed}: comp={r['completion_rate']:.3f} "
              f"({time.time()-t0:.0f}s)", flush=True)
    out = Path(__file__).parent / "results" / f"base_{mode_name}_{seed_start}_{seed_end}.json"
    def conv(o):
        if isinstance(o, (np.integer, np.floating)): return float(o)
        return str(o)
    tmp = out.with_suffix(".tmp")
    tmp.write_text(json.dumps({"mode": mode_name, "results": results}, default=conv))
    tmp.replace(out)
    print(f"Saved {out}", flush=True)


def merge():
    files = sorted(Path(__file__).parent.glob("results/base_*_[0-9]*_[0-9]*.json"))
    grouped = {}
    for f in files:
        data = json.loads(f.read_text())
        grouped.setdefault(data["mode"], []).extend(data["results"])
    missing = [m for m in CONFIGS if m not in grouped]
    if missing:
        print(f"Missing modes: {missing}"); sys.exit(1)
    summary = aggregate_results(grouped)
    print("\n" + "=" * 112)
    print(f"EASY BASELINES N={len(grouped['distance'])} {GRID}x{GRID} crowd={CROWD} walls={WALLS} ticks={TICKS}")
    print("=" * 112)
    header = f"{'Metric':<32}" + "".join([f"{m:>20}" for m in CONFIGS])
    print(header)
    for k in ["completion_rate","casualty_rate","average_evacuation_time",
              "avg_path_length_evacuated","congestion_ratio","total_ticks"]:
        row = f"{k:<32}"
        for m in CONFIGS:
            d = summary[m][k]
            if d["mean"] is None:
                row += f"{'N/A':>20}"
            elif d["ci95"] is None:
                row += f"{d['mean']:>20.4f}"
            else:
                row += f"{d['mean']:.4f}\u00b1{d['ci95']:.4f}".rjust(22)
        print(row)
    print("=" * 112)
    payload = {"type": "baselines",
               "config": {"grid": GRID, "crowd": CROWD, "exits": EXITS,
                          "walls": WALLS, "ticks": TICKS},
               "results": grouped, "summary": summary}
    save_results(payload, "baselines_final")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=list(CONFIGS.keys()))
    ap.add_argument("--seed-start", type=int, default=1)
    ap.add_argument("--seed-end", type=int, default=30)
    ap.add_argument("--merge", action="store_true")
    args = ap.parse_args()
    if args.merge:
        merge()
    else:
        run_chunk(args.mode, args.seed_start, args.seed_end)
