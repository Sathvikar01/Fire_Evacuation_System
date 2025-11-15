import time
from typing import Any, Dict, List, Optional

import numpy as np

# Thresholds used when creating per-tick summaries. Values were chosen to avoid
# counting tiny floating point leftovers that are visually insignificant.
_FIRE_THRESHOLD = 0.04
_SMOKE_THRESHOLD = 0.02


def _ensure_scalar(value: Optional[np.ndarray]) -> Dict[str, Optional[float]]:
    if value is None:
        return {"avg": None, "min": None, "max": None}
    arr = np.asarray(value, dtype=np.float64)
    if arr.size == 0:
        return {"avg": None, "min": None, "max": None}
    return {
        "avg": float(arr.mean()),
        "min": float(arr.min()),
        "max": float(arr.max()),
    }


class SimulationMetrics:
    """Collects per-tick runtime data plus aggregated summaries."""

    def __init__(self, spec):
        self.reset(spec)

    def reset(self, spec) -> None:
        self.spec = spec
        self.tick_history: List[Dict[str, Any]] = []
        self.tick_durations_ms: List[float] = []
        self.reroute_events: List[Dict[str, Any]] = []
        self._run_started_at: Optional[float] = None
        self._running_since: Optional[float] = None
        self._wall_runtime: float = 0.0
        self._final_stats: Dict[str, Optional[float]] = {
            "completion_rate": None,
            "casualty_rate": None,
            "average_evacuation_time": None,
            "avg_path_length_all": None,
            "avg_path_length_evacuated": None,
            "congestion_ratio": None,
            "total_ticks": None,
        }

    def mark_run_start(self) -> None:
        now = time.perf_counter()
        if self._run_started_at is None:
            self._run_started_at = now
        self._running_since = now

    def mark_run_pause(self) -> None:
        if self._running_since is None:
            return
        now = time.perf_counter()
        self._wall_runtime += (now - self._running_since)
        self._running_since = None

    def wall_clock_ms(self) -> float:
        total = self._wall_runtime
        if self._running_since is not None:
            total += (time.perf_counter() - self._running_since)
        return total * 1000.0

    def latest_snapshot(self) -> Optional[Dict[str, Any]]:
        return self.tick_history[-1] if self.tick_history else None

    def record_tick(
        self,
        *,
        sim,
        tick: int,
        runtime_ms: float,
        dynamic_rho,
    ) -> None:
        grid = sim.grid
        engine = sim.engine

        evacuated = int(engine.evacuated)
        casualties = int(engine.casualties)
        remaining = len(grid.agents)
        crowd_size = max(1, grid.spec.crowd)

        stuck_values = list(getattr(engine, "stuck_counter", {}).values())
        total_tracked = len(stuck_values)
        stuck_threshold = getattr(sim, "stuck_window", 0)
        stuck_count = sum(1 for v in stuck_values if v >= stuck_threshold) if stuck_threshold else 0
        stuck_fraction = (stuck_count / total_tracked) if total_tracked else 0.0
        avg_stuck = (sum(stuck_values) / total_tracked) if total_tracked else 0.0

        dist_values = [d for d in getattr(engine, "last_dist", {}).values() if d is not None]
        avg_distance = (sum(dist_values) / len(dist_values)) if dist_values else None

        rho_stats = _ensure_scalar(dynamic_rho)

        congestion_peak = float(grid.congestion.max()) if grid.congestion.size else 0.0
        fire_cells = int(np.count_nonzero(grid.fire > _FIRE_THRESHOLD))
        smoke_cells = int(np.count_nonzero(grid.smoke > _SMOKE_THRESHOLD))
        exit_mask = getattr(grid, "exit_compromised", None)
        compromised = int(exit_mask.sum()) if exit_mask is not None else 0

        snapshot = {
            "tick": tick,
            "runtime_ms": float(runtime_ms),
            "evacuated": evacuated,
            "casualties": casualties,
            "remaining": remaining,
            "completion_rate": evacuated / crowd_size,
            "casualty_rate": casualties / crowd_size,
            "stuck_agents": stuck_count,
            "stuck_fraction": stuck_fraction,
            "stuck_average_ticks": avg_stuck,
            "average_goal_distance": float(avg_distance) if avg_distance is not None else None,
            "dynamic_rho_avg": rho_stats["avg"],
            "dynamic_rho_min": rho_stats["min"],
            "dynamic_rho_max": rho_stats["max"],
            "congestion_peak": congestion_peak,
            "fire_cells": fire_cells,
            "smoke_cells": smoke_cells,
            "compromised_exits": compromised,
            "reroute_count": sim.reroute_count,
            "agents_active": remaining,
        }

        self.tick_history.append(snapshot)
        self.tick_durations_ms.append(float(runtime_ms))

    def record_reroute(self, tick: int, reason: str) -> None:
        self.reroute_events.append({"tick": tick, "reason": reason})

    def capture_final_agent_stats(
        self,
        *,
        evacuated: int,
        casualties: int,
        crowd: int,
        total_ticks: int,
        agent_metrics,
    ) -> None:
        crowd = max(1, crowd)
        completion_rate = evacuated / crowd
        casualty_rate = casualties / crowd

        avg_path_all: Optional[float] = None
        avg_path_evacuated: Optional[float] = None
        avg_evac_time: Optional[float] = None

        if agent_metrics is not None:
            lengths_all = list(getattr(agent_metrics, "path_length", {}).values())
            if lengths_all:
                avg_path_all = float(sum(lengths_all) / len(lengths_all))

            evac_lengths = [
                agent_metrics.path_length[aid]
                for aid, evacuated_flag in getattr(agent_metrics, "is_evacuated", {}).items()
                if evacuated_flag and aid in agent_metrics.path_length
            ]
            if evac_lengths:
                avg_path_evacuated = float(sum(evac_lengths) / len(evac_lengths))

            avg_evac_time = agent_metrics.get_average_evacuation_time()

        congested_ticks = sum(1 for entry in self.tick_history if entry.get("congestion_peak", 0) > 1)
        total_recorded_ticks = len(self.tick_history)
        congestion_ratio = (
            float(congested_ticks) / float(total_recorded_ticks)
            if total_recorded_ticks
            else None
        )

        self._final_stats.update(
            {
                "completion_rate": float(completion_rate),
                "casualty_rate": float(casualty_rate),
                "average_evacuation_time": float(avg_evac_time) if avg_evac_time is not None else None,
                "avg_path_length_all": avg_path_all,
                "avg_path_length_evacuated": avg_path_evacuated,
                "congestion_ratio": congestion_ratio,
                "total_ticks": float(total_ticks),
            }
        )

    def summary(self) -> Dict[str, Any]:
        if not self.tick_history:
            return {
                "tick_count": 0,
                "runtime_ms_avg": 0.0,
                "runtime_ms_max": 0.0,
                "runtime_ms_p95": 0.0,
                "stuck_agents_peak": 0,
                "stuck_fraction_peak": 0.0,
                "congestion_peak": 0.0,
                "fire_cells_peak": 0,
                "smoke_cells_peak": 0,
                "wall_clock_ms": 0.0,
                "dynamic_rho_avg": None,
                "reroute_events": list(self.reroute_events),
                "completion_rate_final": 0.0,
                "casualty_rate_final": 0.0,
                "average_evacuation_time": None,
                "avg_path_length_all": None,
                "avg_path_length_evacuated": None,
                "congestion_ratio": None,
                "total_ticks_final": None,
            }

        durations = np.array(self.tick_durations_ms, dtype=np.float64)
        stuck_peaks = [entry["stuck_agents"] for entry in self.tick_history]
        stuck_frac = [entry["stuck_fraction"] for entry in self.tick_history]
        congestion = [entry["congestion_peak"] for entry in self.tick_history]
        fire_cells = [entry["fire_cells"] for entry in self.tick_history]
        smoke_cells = [entry["smoke_cells"] for entry in self.tick_history]
        rho_values = [entry["dynamic_rho_avg"] for entry in self.tick_history if entry["dynamic_rho_avg"] is not None]

        return {
            "tick_count": len(self.tick_history),
            "runtime_ms_avg": float(durations.mean()) if durations.size else 0.0,
            "runtime_ms_max": float(durations.max()) if durations.size else 0.0,
            "runtime_ms_p95": float(np.percentile(durations, 95)) if durations.size else 0.0,
            "stuck_agents_peak": int(max(stuck_peaks) if stuck_peaks else 0),
            "stuck_fraction_peak": float(max(stuck_frac) if stuck_frac else 0.0),
            "congestion_peak": float(max(congestion) if congestion else 0.0),
            "fire_cells_peak": int(max(fire_cells) if fire_cells else 0),
            "smoke_cells_peak": int(max(smoke_cells) if smoke_cells else 0),
            "dynamic_rho_avg": float(np.mean(rho_values)) if rho_values else None,
            "wall_clock_ms": self.wall_clock_ms(),
            "reroute_events": list(self.reroute_events),
            "completion_rate_final": self.tick_history[-1]["completion_rate"],
            "casualty_rate_final": self.tick_history[-1]["casualty_rate"],
            "average_evacuation_time": self._final_stats.get("average_evacuation_time"),
            "avg_path_length_all": self._final_stats.get("avg_path_length_all"),
            "avg_path_length_evacuated": self._final_stats.get("avg_path_length_evacuated"),
            "congestion_ratio": self._final_stats.get("congestion_ratio"),
            "total_ticks_final": self._final_stats.get("total_ticks"),
        }

    def export(self, include_history: bool = True) -> Dict[str, Any]:
        payload = {
            "grid_spec": {
                "rows": getattr(self.spec, "rows", None),
                "cols": getattr(self.spec, "cols", None),
                "crowd": getattr(self.spec, "crowd", None),
                "exits": getattr(self.spec, "exits", None),
                "wall_density": getattr(self.spec, "wall_density", None),
            },
            "summary": self.summary(),
            "tick_count": len(self.tick_history),
        }
        if include_history:
            payload["tick_history"] = list(self.tick_history)
            payload["tick_durations_ms"] = list(self.tick_durations_ms)
        else:
            payload["tick_history"] = []
            payload["tick_durations_ms"] = []
        return payload