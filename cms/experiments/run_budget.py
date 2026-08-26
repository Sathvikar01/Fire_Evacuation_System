"""Compute-budget-matched ACO: scale ant iterations + reroute budgets together.

Levels (factor applied to ANT_PRE_ITERS, PERIODIC_REROUTE_ITERS,
EMERGENCY_REROUTE_ITERS, ACO_BUDGET_PER_TICK):
  B1 = 1x (default), B2 = 1/3, B3 = 1/10, B4 = 1/30

Reports completion and wall-clock seconds/run so a fair budget-matched
comparison with per-tick A* can be made.

Usage: python run_budget.py --level B2 --seed-start 1 --seed-end 30
       python run_budget.py --merge
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

GRID = 15
CROWD = 30
EXITS = 3
WALLS = 0.15          # hard environment to match headline comparison
TICKS = 40

BASE_FLAGS = {
    "DUAL_PHEROMONE_ENABLED": True,
    "PREDICTIVE_CONGESTION_ENABLED": True,
    "STUCK_ESCAPE_ENABLED": True,
    "HAZARD_FORECAST_ENABLED": True,
    "BFS_SEED_ENABLED": True,
    "HAZARD_AWARE_ROUTING_ENABLED": True,
}

LEVELS = {
    "B1": {"ANT_PRE_ITERS": 300, "PERIODIC_REROUTE_ITERS": 250,
           "EMERGENCY_REROUTE_ITERS": 250, "ACO_BUDGET_PER_TICK": 40},
    "B2": {"ANT_PRE_ITERS": 100, "PERIODIC_REROUTE_ITERS": 83,
           "EMERGENCY_REROUTE_ITERS": 83, "ACO_BUDGET_PER_TICK": 13},
    "B3": {"ANT_PRE_ITERS": 30,  "PERIODIC_REROUTE_ITERS": 25,
           "EMERGENCY_REROUTE_ITERS": 25, "ACO_BUDGET_PER_TICK": 4},
    "B4": {"ANT_PRE_ITERS": 10,  "PERIODIC_REROUTE_ITERS": 8,
           "EMERGENCY_REROUTE_ITERS": 8,  "ACO_BUDGET_PER_TICK": 1},
}


def run_chunk(level, seed_start, seed_end):
    vals = LEVELS[level]
    for k, v in BASE_FLAGS.items():
        setattr(config, k, v)
    for k, v in vals.items():
        setattr(config, k, int(v))

    results = []
    t0 = time.time()
    for seed in range(seed_start, seed_end + 1):
        t_run = time.time()
        r = run_single(seed, GRID, CROWD, EXITS, WALLS,
                       config.MOVEMENT_MODE_ACO, TICKS,
                       extra_flags={"BLOCKED_EXITS": 1},
                       wind_params={"direction": "east", "strength": 0.5})
        r["wall_seconds"] = time.time() - t_run
        results.append(r)
        print(f"{level} seed {seed}: comp={r['completion_rate']:.3f} "
              f"{r['wall_seconds']:.1f}s ({time.time()-t0:.0f}s)", flush=True)
    out = Path(__file__).parent / "results" / f"bud_{level}_{seed_start}_{seed_end}.json"
    def conv(o):
        if isinstance(o, (np.integer, np.floating)): return float(o)
        return str(o)
    tmp = out.with_suffix(".tmp")
    tmp.write_text(json.dumps({"level": level, "results": results}, default=conv))
    tmp.replace(out)
    print(f"Saved {out}", flush=True)


def merge():
    files = sorted(Path(__file__).parent.glob("results/bud_*_1_30.json"))
    grouped = {}
    for f in files:
        data = json.loads(f.read_text())
        grouped.setdefault(data["level"], []).extend(data["results"])
    missing = [l for l in LEVELS if l not in grouped]
    if missing:
        print(f"Missing levels: {missing}"); sys.exit(1)
    summary = {}
    for lvl in LEVELS:
        runs = grouped[lvl]
        cs = [r["completion_rate"] for r in runs]
        ws = [r["wall_seconds"] for r in runs]
        summary[lvl] = {
            "completion_mean": float(np.mean(cs)),
            "completion_ci": float(1.96*np.std(cs, ddof=1)/np.sqrt(len(cs))) if len(cs) > 1 else 0.0,
            "sec_per_run_mean": float(np.mean(ws)),
            "n": len(runs),
        }
        print(f"{lvl}: completion {summary[lvl]['completion_mean']:.4f}"
              f" ±{summary[lvl]['completion_ci']:.4f} "
              f"sec/run {summary[lvl]['sec_per_run_mean']:.2f}")
    payload = {"type": "budget",
               "config": {"grid": GRID, "crowd": CROWD, "walls": WALLS,
                          "ticks": TICKS}, "summary": summary}
    save_results(payload, "budget_final")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--level", choices=list(LEVELS.keys()))
    ap.add_argument("--seed-start", type=int, default=1)
    ap.add_argument("--seed-end", type=int, default=30)
    ap.add_argument("--merge", action="store_true")
    args = ap.parse_args()
    if args.merge:
        merge()
    else:
        run_chunk(args.level, args.seed_start, args.seed_end)
