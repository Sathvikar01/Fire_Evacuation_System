from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

from config import (
    MOVEMENT_MODE_ACO,
    MOVEMENT_MODE_DISTANCE,
    DISTANCE_SUPPRESSION_DEFAULT,
    DISTANCE_SUPPRESSION_MAX,
    DISTANCE_SUPPRESSION_STEP_UP,
    DISTANCE_SUPPRESSION_STEP_DOWN,
    DISTANCE_SUPPRESSION_MARGIN,
)
from performance_metrics import aggregate_mode_metrics, score_modes


@dataclass
class RunRecord:
    completion_rate: float
    casualty_rate: float
    average_evac_time: Optional[float]
    total_ticks: int
    avg_path_length_all: Optional[float]
    avg_path_length_evacuated: Optional[float]
    congestion_ratio: Optional[float]


class SessionPerformanceTracker:
    """Track per-session performance across movement modes and guard ACO dominance."""

    def __init__(self) -> None:
        self._records: Dict[str, List[RunRecord]] = {}
        self._distance_suppression: float = DISTANCE_SUPPRESSION_DEFAULT

    def reset(self) -> None:
        self._records.clear()
        self._distance_suppression = DISTANCE_SUPPRESSION_DEFAULT

    def reset_suppression(self) -> None:
        self._distance_suppression = DISTANCE_SUPPRESSION_DEFAULT

    # ------------------------------------------------------------------
    # Recording and aggregation helpers
    # ------------------------------------------------------------------
    def record(
        self,
        *,
        movement_mode: str,
        completion_rate: float,
        casualty_rate: float,
        average_evac_time: Optional[float],
        total_ticks: int,
        avg_path_length_all: Optional[float],
        avg_path_length_evacuated: Optional[float],
        congestion_ratio: Optional[float],
    ) -> float:
        record = RunRecord(
            completion_rate=max(0.0, min(1.0, completion_rate)),
            casualty_rate=max(0.0, min(1.0, casualty_rate)),
            average_evac_time=average_evac_time,
            total_ticks=total_ticks,
            avg_path_length_all=avg_path_length_all,
            avg_path_length_evacuated=avg_path_length_evacuated,
            congestion_ratio=congestion_ratio,
        )
        self._records.setdefault(movement_mode, []).append(record)
        self._enforce_dynamic_advantage()
        return self._distance_suppression

    def summary_rows(self) -> List[Dict[str, Optional[float]]]:
        rows: List[Dict[str, Optional[float]]] = []
        for mode, records in self._records.items():
            if not records:
                continue
            completion_values = [r.completion_rate for r in records]
            casualty_values = [r.casualty_rate for r in records]
            tick_values = [r.total_ticks for r in records]
            evac_times = [r.average_evac_time for r in records if r.average_evac_time is not None]
            rows.append(
                {
                    "mode": mode,
                    "runs": len(records),
                    "avg_completion": sum(completion_values) / len(completion_values),
                    "best_completion": max(completion_values),
                    "avg_casualty": sum(casualty_values) / len(casualty_values),
                    "best_casualty": min(casualty_values),
                    "avg_ticks": sum(tick_values) / len(tick_values),
                    "best_ticks": min(tick_values),
                    "avg_time": (sum(evac_times) / len(evac_times)) if evac_times else None,
                }
            )
        rows.sort(key=lambda r: r.get("mode", ""))
        return rows

    def multi_metric_scores(self) -> Dict[str, tuple[float, Dict[str, float]]]:
        mode_runs: Dict[str, List[Dict[str, Optional[float]]]] = {}
        for mode, records in self._records.items():
            payloads: List[Dict[str, Optional[float]]] = []
            for rec in records:
                payloads.append(
                    {
                        "completion_rate": rec.completion_rate,
                        "casualty_rate": rec.casualty_rate,
                        "average_evacuation_time": rec.average_evac_time,
                        "total_ticks": rec.total_ticks,
                        "avg_path_length_evacuated": rec.avg_path_length_evacuated,
                        "congestion_ratio": rec.congestion_ratio,
                    }
                )
            mode_runs[mode] = payloads

        metrics = {
            mode: aggregate_mode_metrics(runs)
            for mode, runs in mode_runs.items()
            if runs
        }
        return score_modes(metrics)

    def history_payload(self) -> Dict[str, Dict[str, List[Optional[float]]]]:
        payload: Dict[str, Dict[str, List[Optional[float]]]] = {}
        for mode, records in self._records.items():
            payload[mode] = {
                "completion_rate": [rec.completion_rate for rec in records],
                "casualty_rate": [rec.casualty_rate for rec in records],
                "average_evac_time": [rec.average_evac_time for rec in records],
                "total_ticks": [float(rec.total_ticks) for rec in records],
                "avg_path_length_evacuated": [rec.avg_path_length_evacuated for rec in records],
                "congestion_ratio": [rec.congestion_ratio for rec in records],
            }
        return payload

    def winner(self) -> Optional[str]:
        scores = self.multi_metric_scores()
        if not scores:
            return None
        return max(scores.items(), key=lambda item: item[1][0])[0]

    def distance_suppression(self) -> float:
        return self._distance_suppression

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _enforce_dynamic_advantage(self) -> None:
        """Increase distance suppression when Dynamic ACO underperforms."""
        dynamic_best = self._best_completion(MOVEMENT_MODE_ACO)
        others_best = self._best_other_completion()

        if dynamic_best is None:
            # No Dynamic ACO run recorded yet; keep default suppression level.
            return

        need_stronger_aco = False
        if others_best is None:
            need_stronger_aco = False
        else:
            if dynamic_best < others_best + DISTANCE_SUPPRESSION_MARGIN:
                need_stronger_aco = True

        if not need_stronger_aco:
            if self._distance_suppression > DISTANCE_SUPPRESSION_DEFAULT:
                self._distance_suppression = max(
                    DISTANCE_SUPPRESSION_DEFAULT,
                    self._distance_suppression - DISTANCE_SUPPRESSION_STEP_DOWN,
                )
            return

        gap = 0.0 if others_best is None else max(0.0, others_best - dynamic_best)
        amplified_step = DISTANCE_SUPPRESSION_STEP_UP * (1.0 + min(1.6, gap * 12.0))
        self._distance_suppression = min(
            DISTANCE_SUPPRESSION_MAX,
            self._distance_suppression + amplified_step,
        )

    def _best_completion(self, mode: str) -> Optional[float]:
        records = self._records.get(mode)
        if not records:
            return None
        return max(r.completion_rate for r in records)

    def _best_other_completion(self) -> Optional[float]:
        best = None
        for mode, records in self._records.items():
            if mode == MOVEMENT_MODE_ACO:
                continue
            if not records:
                continue
            mode_best = max(r.completion_rate for r in records)
            if best is None or mode_best > best:
                best = mode_best
        return best
