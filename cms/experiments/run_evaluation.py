"""Monte Carlo evaluation harness for the Fire Evacuation System.

Runs each movement mode multiple times with different seeds, collects
metrics, and reports mean ± 95% CI for each metric. Produces a summary
table suitable for inclusion in a research paper.

Usage:
    python experiments/run_evaluation.py --runs 30 --ticks 200
    python experiments/run_evaluation.py --runs 10 --grid 40 --crowd 100

Output:
    Prints a summary table and saves results to experiments/results/
"""
import argparse
import json
import os
import sys
import time
from typing import Dict, List, Optional, Tuple

import numpy as np

# Ensure package root is importable (deduplicated, absolute)
from pathlib import Path as _Path
_pkg_root = str(_Path(__file__).resolve().parent.parent)
if _pkg_root not in sys.path:
    sys.path.insert(0, _pkg_root)

from core.simulation import Simulation
from core.grid import GridSpec
import config
# Store original dual flag for restoration (research: avoid global contamination)
_ORIG_DUAL_FLAG = config.DUAL_PHEROMONE_ENABLED


METRIC_KEYS = [
    "completion_rate",
    "casualty_rate",
    "average_evacuation_time",
    "avg_path_length_all",
    "avg_path_length_evacuated",
    "congestion_ratio",
    "total_ticks",
]


def run_single(
    seed: int,
    grid: int,
    crowd: int,
    exits: int,
    walls: float,
    mode: str,
    ticks: int,
    dual_pheromone: bool = False,
) -> Dict[str, Optional[float]]:
    """Run a single simulation and return final metrics.

    Research: dual_pheromone flag is isolated per-run via context manager so
    sequential runs in same process don't contaminate global config.
    """
    orig_dual = config.DUAL_PHEROMONE_ENABLED
    config.DUAL_PHEROMONE_ENABLED = dual_pheromone
    # Keep deprecated alias in sync
    try:
        config.USE_DUAL_PHEROMONE = dual_pheromone
    except Exception:
        pass
    try:
        spec = GridSpec(rows=grid, cols=grid, crowd=crowd, exits=exits, wall_density=walls)
        s = Simulation(spec, seed=seed, movement_mode=mode)

        # Start the simulation so the metrics/run-state is properly initialized.
        s.start()
        # Headless: timer may be None (no QApplication). Guard.
        if s.timer is not None:
            try:
                s.timer.stop()
            except Exception:
                pass

        for _ in range(ticks):
            s.step()
            # The sim auto-pauses when all agents are evacuated or dead; stop looping.
            if not s.running:
                break

        summary = s.metrics.summary()
        # Use actual placed crowd for rate (handles truncated crowd)
        actual_crowd = getattr(s.grid, '_actual_crowd', spec.crowd)
        crowd_size = max(1, actual_crowd)
        return {
            "completion_rate": s.engine.evacuated / crowd_size,
            "casualty_rate": s.engine.casualties / crowd_size,
            "average_evacuation_time": summary.get("average_evacuation_time"),
            "avg_path_length_all": summary.get("avg_path_length_all"),
            "avg_path_length_evacuated": summary.get("avg_path_length_evacuated"),
            "congestion_ratio": summary.get("congestion_ratio"),
            "total_ticks": float(s.tick_counter),
        }
    finally:
        config.DUAL_PHEROMONE_ENABLED = orig_dual
        try:
            config.USE_DUAL_PHEROMONE = orig_dual
        except Exception:
            pass


def mean_ci(values: List[Optional[float]], confidence: float = 0.95) -> Tuple[Optional[float], Optional[float]]:
    """Compute mean and 95% confidence interval (t-distribution).

    Falls back to normal approximation (1.96) if scipy not available.
    """
    clean = [v for v in values if v is not None and np.isfinite(v)]
    if not clean:
        return None, None
    arr = np.array(clean, dtype=np.float64)
    n = len(arr)
    mean = float(arr.mean())
    if n < 2:
        return mean, None
    sem = float(arr.std(ddof=1) / np.sqrt(n)) if n > 1 else 0.0
    try:
        from scipy import stats
        if n >= 30:
            z = float(stats.norm.ppf(0.5 + confidence / 2))
        else:
            z = float(stats.t.ppf(0.5 + confidence / 2, df=n - 1))
    except Exception:
        # Fallback without scipy — use 1.96 for normal approx
        z = 1.96
    ci = z * sem
    return mean, ci


def run_evaluation(
    runs: int = 30,
    ticks: int = 200,
    grid: int = 20,
    crowd: int = 40,
    exits: int = 3,
    walls: float = 0.08,
    seeds: Optional[List[int]] = None,
    dual_pheromone: bool = False,
) -> Dict:
    """Run Monte Carlo evaluation across all movement modes.

    Reproducibility: seeds default to 1..runs sequential but caller can pass
    SeedSequence-derived shuffled seeds. KeyboardInterrupt saves partial results.
    """
    if seeds is None:
        seeds = list(range(1, runs + 1))

    modes = [config.MOVEMENT_MODE_ACO, config.MOVEMENT_MODE_DISTANCE, config.MOVEMENT_MODE_RANDOM]
    results: Dict[str, List[Dict]] = {mode: [] for mode in modes}

    print(f"Monte Carlo Evaluation: {runs} runs, {ticks} ticks, {grid}x{grid} grid, {crowd} agents")
    print(f"Modes: {', '.join(modes)}")
    print(f"Dual pheromone: {dual_pheromone}")
    print("-" * 80)

    # KeyboardInterrupt handling — save partial on Ctrl+C for long runs
    try:
        for mode in modes:
            t0 = time.perf_counter()
            for i, seed in enumerate(seeds):
                metrics = run_single(seed, grid, crowd, exits, walls, mode, ticks, dual_pheromone)
                results[mode].append(metrics)
                if (i + 1) % 10 == 0:
                    print(f"  {mode}: {i+1}/{runs} runs complete")
            elapsed = time.perf_counter() - t0
            print(f"  {mode}: {runs} runs in {elapsed:.1f}s")
    except KeyboardInterrupt:
        print("\nInterrupted — returning partial results")
        # Continue to aggregation with whatever completed

    # Aggregate — filter finite for n
    summary = {}
    for mode, runs_list in results.items():
        mode_summary = {}
        for key in METRIC_KEYS:
            values = [r.get(key) for r in runs_list]
            mean, ci = mean_ci(values)
            n_finite = len([v for v in values if v is not None and np.isfinite(v)])
            mode_summary[key] = {"mean": mean, "ci95": ci, "n": n_finite}
        summary[mode] = mode_summary

    # Store config hash for reproducibility (ALPHA etc)
    try:
        import hashlib as _hl
        cfg_snapshot = {k: getattr(config, k) for k in ["ALPHA","BETA","GAMMA","RHO","Q","ACO_TEMPERATURE","FIRE_TRAVERSAL_THRESHOLD","FIRE_SAFE_THRESHOLD"] if hasattr(config, k)}
    except Exception:
        cfg_snapshot = {}
    return {"summary": summary, "raw": results, "config": {
        "runs": runs, "ticks": ticks, "grid": grid,
        "crowd": crowd, "exits": exits, "walls": walls,
        "dual_pheromone": dual_pheromone,
        "cfg": cfg_snapshot,
        "seeds": seeds,
    }}


def print_summary_table(summary: Dict):
    """Print a formatted summary table."""
    print("\n" + "=" * 100)
    print(f"{'Metric':<30} {'ACO':>20} {'Distance':>20} {'Random':>20}")
    print("=" * 100)
    for key in METRIC_KEYS:
        row = f"{key:<30}"
        for mode in [config.MOVEMENT_MODE_ACO, config.MOVEMENT_MODE_DISTANCE, config.MOVEMENT_MODE_RANDOM]:
            stats = summary.get(mode, {}).get(key, {})
            mean = stats.get("mean")
            ci = stats.get("ci95")
            if mean is None:
                row += f"{'N/A':>20}"
            elif ci is None:
                row += f"{mean:>20.4f}"
            else:
                row += f"{mean:.4f}+/-{ci:.4f}".rjust(20)
        print(row)
    print("=" * 100)


def save_results(payload: Dict, output_dir: str = None):
    """Save results to JSON (atomic, handles numpy types, adds unique suffix on collision)."""
    if output_dir is None:
        output_dir = os.path.join(os.path.dirname(__file__), "results")
    os.makedirs(output_dir, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    base = os.path.join(output_dir, f"evaluation_{timestamp}")
    filename = base + ".json"
    # Avoid overwrite within same second
    counter = 0
    while os.path.exists(filename):
        counter += 1
        filename = f"{base}_{counter}.json"
    # Convert numpy types to python for proper JSON numbers (not strings)
    def _convert(obj):
        if isinstance(obj, (np.integer, np.floating)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.bool_, bool)):
            return bool(obj)
        return str(obj)
    # Atomic write via temp file
    tmp = filename + ".tmp"
    with open(tmp, "w") as f:
        json.dump(payload, f, indent=2, default=_convert)
    os.replace(tmp, filename)
    print(f"\nResults saved to: {filename}")
    return filename


def main():
    parser = argparse.ArgumentParser(description="Monte Carlo evaluation of Fire Evacuation System")
    parser.add_argument("--runs", type=int, default=30, help="Number of runs per mode")
    parser.add_argument("--ticks", type=int, default=200, help="Max ticks per run")
    parser.add_argument("--grid", type=int, default=20, help="Grid size (NxN)")
    parser.add_argument("--crowd", type=int, default=40, help="Number of agents")
    parser.add_argument("--exits", type=int, default=3, help="Number of exits")
    parser.add_argument("--walls", type=float, default=0.08, help="Wall density")
    parser.add_argument("--dual", action="store_true", help="Enable dual pheromone channel")
    args = parser.parse_args()

    payload = run_evaluation(
        runs=args.runs,
        ticks=args.ticks,
        grid=args.grid,
        crowd=args.crowd,
        exits=args.exits,
        walls=args.walls,
        dual_pheromone=args.dual,
    )
    print_summary_table(payload["summary"])
    save_results(payload)


if __name__ == "__main__":
    main()
