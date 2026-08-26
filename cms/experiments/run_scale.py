"""Scale sweep: does per-tick replanning cost actually bite at 30^2 / 40^2?

Envs mirror the hard condition (0.15 walls, one pre-blocked exit, wind E/0.5)
with crowd and ticks scaled by area. Policies: distance, astar, dstar, full.
Wall-clock seconds per run recorded for the runtime-growth claim.

Usage:
  python run_scale.py --env S30 --policy astar --seed-start 1 --seed-end 20
  python run_scale.py --merge
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

ENVS = {
    "S30": dict(grid=30, crowd=60, exits=4, walls=0.15, ticks=60),
    "S40": dict(grid=40, crowd=100, exits=4, walls=0.15, ticks=90),
}

POLICIES = {
    "distance": {"mode": config.MOVEMENT_MODE_DISTANCE, "flags": {}},
    "astar": {"mode": config.MOVEMENT_MODE_ASTAR, "flags": {}},
    "dstar": {"mode": config.MOVEMENT_MODE_DSTAR, "flags": {}},
    "full_cms_daco": {
        "mode": config.MOVEMENT_MODE_ACO,
        "flags": {"DUAL_PHEROMONE_ENABLED": True,
                  "PREDICTIVE_CONGESTION_ENABLED": True,
                  "STUCK_ESCAPE_ENABLED": True,
                  "HAZARD_FORECAST_ENABLED": True,
                  "BFS_SEED_ENABLED": True,
                  "HAZARD_AWARE_ROUTING_ENABLED": True},
    },
}


def run_chunk(env_key, policy, seed_start, seed_end):
    env = ENVS[env_key]
    cfg = POLICIES[policy]
    for k, v in cfg["flags"].items():
        setattr(config, k, v)

    results = []
    t0 = time.time()
    for seed in range(seed_start, seed_end + 1):
        t_run = time.time()
        r = run_single(seed, env["grid"], env["crowd"], env["exits"], env["walls"],
                       cfg["mode"], env["ticks"],
                       extra_flags={"BLOCKED_EXITS": 1},
                       wind_params={"direction": "east", "strength": 0.5})
        r["wall_seconds"] = time.time() - t_run
        results.append(r)
        print(f"{env_key}/{policy} seed {seed}: comp={r['completion_rate']:.3f} "
              f"{r['wall_seconds']:.1f}s ({time.time()-t0:.0f}s)", flush=True)
    out = (Path(__file__).parent / "results" /
           f"scl_{env_key}_{policy}_{seed_start}_{seed_end}.json")
    def conv(o):
        if isinstance(o, (np.integer, np.floating)): return float(o)
        return str(o)
    tmp = out.with_suffix(".tmp")
    tmp.write_text(json.dumps({"env": env_key, "policy": policy,
                               "results": results}, default=conv))
    tmp.replace(out)
    print(f"Saved {out}", flush=True)


def merge():
    files = sorted(Path(__file__).parent.glob("results/scl_*_*_[0-9]*_[0-9]*.json"))
    grouped = {}
    for f in files:
        data = json.loads(f.read_text())
        grouped.setdefault(data["env"], {}).setdefault(data["policy"], []).extend(
            data["results"])
    summary = {}
    for env, polys in grouped.items():
        summary[env] = {}
        for pol, runs in polys.items():
            cs = [r["completion_rate"] for r in runs]
            ws = [r["wall_seconds"] for r in runs]
            summary[env][pol] = {
                "completion_mean": float(np.mean(cs)),
                "completion_ci": float(1.96*np.std(cs, ddof=1)/np.sqrt(len(cs))) if len(cs) > 1 else 0.0,
                "sec_per_run_mean": float(np.mean(ws)),
                "sec_per_run_median": float(np.median(ws)),
                "sec_per_run_min": float(np.min(ws)),
                "sec_per_run_max": float(np.max(ws)),
                "n": len(runs)}
            print(f"{env}/{pol}: completion {summary[env][pol]['completion_mean']:.4f}"
                  f" ±{summary[env][pol]['completion_ci']:.4f} "
                  f"sec/run median {summary[env][pol]['sec_per_run_median']:.2f} "
                  f"[{summary[env][pol]['sec_per_run_min']:.1f}-{summary[env][pol]['sec_per_run_max']:.1f}]")
    payload = {"type": "scale", "config": ENVS, "results": grouped, "summary": summary}
    save_results(payload, "scale_final")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--env", choices=list(ENVS.keys()))
    ap.add_argument("--policy", choices=list(POLICIES.keys()))
    ap.add_argument("--seed-start", type=int, default=1)
    ap.add_argument("--seed-end", type=int, default=20)
    ap.add_argument("--merge", action="store_true")
    args = ap.parse_args()
    if args.merge:
        merge()
    else:
        run_chunk(args.env, args.policy, args.seed_start, args.seed_end)
