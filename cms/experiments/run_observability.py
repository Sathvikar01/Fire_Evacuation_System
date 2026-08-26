"""Partial-observability study: when does stigmergy help?

Policies {astar, dstar, full_cms_daco} x observability
{full, r3, r2, stale5, stale10, loss50} on the HARD environment.
Belief gates planner costs/penalties; physics/deaths always use true fields.

Usage:
  python run_observability.py --policy astar --mode radius --value 3 --seed-start 1 --seed-end 30
  python run_observability.py --merge
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
WALLS = 0.15
TICKS = 40

POLICY_FLAGS = {
    "astar": {"mode": config.MOVEMENT_MODE_ASTAR, "flags": {}},
    "dstar": {"mode": config.MOVEMENT_MODE_DSTAR, "flags": {}},
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

# mode_key -> config overrides for OBSERVABILITY_*
OBS = {
    "full":    {"OBSERVABILITY_MODE": "full"},
    "r3":      {"OBSERVABILITY_MODE": "radius", "SENSOR_RADIUS": 3},
    "stale5":  {"OBSERVABILITY_MODE": "stale",  "STALE_TICKS": 5},
    "loss50":  {"OBSERVABILITY_MODE": "loss",   "SENSE_LOSS_PROB": 0.5},
}


def run_chunk(policy, obs_key, seed_start, seed_end):
    cfg = POLICY_FLAGS[policy]
    # Apply ALL flags BEFORE core import (fresh process per chunk)
    setattr(config, "OBSERVABILITY_MODE", "full")
    setattr(config, "SENSOR_RADIUS", 3)
    setattr(config, "STALE_TICKS", 5)
    setattr(config, "SENSE_LOSS_PROB", 1.0)
    for k, v in cfg["flags"].items():
        setattr(config, k, v)
    for k, v in OBS[obs_key].items():
        setattr(config, k, v)

    results = []
    t0 = time.time()
    for seed in range(seed_start, seed_end + 1):
        r = run_single(seed, GRID, CROWD, EXITS, WALLS, cfg["mode"], TICKS,
                       extra_flags={"BLOCKED_EXITS": 1},
                       wind_params={"direction": "east", "strength": 0.5})
        results.append(r)
        print(f"{policy}/{obs_key} seed {seed}: comp={r['completion_rate']:.3f} "
              f"({time.time()-t0:.0f}s)", flush=True)
    out = Path(__file__).parent / "results" / f"obs_{policy}_{obs_key}_{seed_start}_{seed_end}.json"
    def conv(o):
        if isinstance(o, (np.integer, np.floating)): return float(o)
        return str(o)
    tmp = out.with_suffix(".tmp")
    tmp.write_text(json.dumps({"policy": policy, "obs": obs_key,
                               "results": results}, default=conv))
    tmp.replace(out)
    print(f"Saved {out}", flush=True)


def merge():
    files = sorted(Path(__file__).parent.glob("results/obs_*_*_[0-9]*_[0-9]*.json"))
    grouped = {}
    for f in files:
        data = json.loads(f.read_text())
        grouped.setdefault(data["policy"], {}).setdefault(data["obs"], []).extend(
            data["results"])
    missing = [(p, o) for p in POLICY_FLAGS for o in OBS
               if o not in grouped.get(p, {})]
    if missing:
        print(f"Missing combos: {missing}")
        sys.exit(1)
    summary = {}
    for p, modes in grouped.items():
        summary[p] = {}
        for o, runs in modes.items():
            cs = [r["completion_rate"] for r in runs]
            summary[p][o] = {
                "completion_mean": float(np.mean(cs)),
                "completion_ci": float(1.96*np.std(cs, ddof=1)/np.sqrt(len(cs))) if len(cs) > 1 else 0.0,
                "n": len(runs),
            }
            print(f"{p:>14}/{o:<8} completion {summary[p][o]['completion_mean']:.4f}"
                  f" ±{summary[p][o]['completion_ci']:.4f}")
    payload = {"type": "observability",
               "config": {"grid": GRID, "crowd": CROWD, "exits": EXITS,
                          "walls": WALLS, "ticks": TICKS,
                          "wind": "east/0.5", "blocked": 1},
               "results": grouped, "summary": summary}
    save_results(payload, "observability_final")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", choices=list(POLICY_FLAGS.keys()))
    ap.add_argument("--mode", choices=list(OBS.keys()))
    ap.add_argument("--seed-start", type=int, default=1)
    ap.add_argument("--seed-end", type=int, default=30)
    ap.add_argument("--merge", action="store_true")
    args = ap.parse_args()
    if args.merge:
        merge()
    else:
        run_chunk(args.policy, args.mode, args.seed_start, args.seed_end)
