"""Resumable driver for the observability study (runs as background process).

Iterates (policy, obs-mode) combos sequentially, appending results atomically.
Skips combos whose result file already covers the requested seed range.

Usage: python obs_driver.py <policy> <seed_start> <seed_end> [modes_csv]
"""
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config
from experiments.run_comprehensive_evaluation import run_single

GRID, CROWD, EXITS, WALLS, TICKS = 15, 30, 3, 0.15, 40

POLICY_FLAGS = {
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

OBS = {
    "full":    {"OBSERVABILITY_MODE": "full"},
    "r3":      {"OBSERVABILITY_MODE": "radius", "SENSOR_RADIUS": 3},
    "stale5":  {"OBSERVABILITY_MODE": "stale",  "STALE_TICKS": 5},
    "loss50":  {"OBSERVABILITY_MODE": "loss",   "SENSE_LOSS_PROB": 0.5},
    "priv3":   {"OBSERVABILITY_MODE": "privater3", "SENSOR_RADIUS": 3},
}


def main():
    policy = sys.argv[1]
    seed_start, seed_end = int(sys.argv[2]), int(sys.argv[3])
    modes = (sys.argv[4].split(",") if len(sys.argv) > 4 else list(OBS.keys()))

    cfg = POLICY_FLAGS[policy]
    out = Path(__file__).parent / "results" / f"obs_{policy}_{{obs}}_{seed_start}_{seed_end}.json"

    for obs_key in modes:
        target = Path(str(out).replace("{obs}", obs_key))
        if target.exists():
            print(f"skip existing {target.name}", flush=True)
            continue
        # Fresh process state per combo is ideal; emulate by setting flags now
        # (driver process imports core lazily via run_single on first use).
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
            t_run = time.time()
            r = run_single(seed, GRID, CROWD, EXITS, WALLS, cfg["mode"], TICKS,
                           extra_flags={"BLOCKED_EXITS": 1},
                           wind_params={"direction": "east", "strength": 0.5})
            r["wall_seconds"] = time.time() - t_run
            results.append(r)
            print(f"{policy}/{obs_key} seed {seed}: comp={r['completion_rate']:.3f} "
                  f"{r['wall_seconds']:.1f}s ({time.time()-t0:.0f}s)", flush=True)
            # atomic incremental save after EVERY seed (resumable)
            def conv(o):
                if isinstance(o, (np.integer, np.floating)): return float(o)
                return str(o)
            tmp = target.with_suffix(".tmp")
            tmp.write_text(json.dumps({"policy": policy, "obs": obs_key,
                                       "results": results}, default=conv))
            tmp.replace(target)


if __name__ == "__main__":
    main()
