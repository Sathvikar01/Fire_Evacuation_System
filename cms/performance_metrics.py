"""Helpers for aggregating and scoring session level simulation metrics.

The UI asks the session tracker for a per-mode scoreboard so it can highlight
which movement mode is performing best.  We keep the implementation fairly
small: aggregate the raw measurements for a mode, normalise them across all
modes, and derive a weighted score.
"""
from __future__ import annotations

from typing import Dict, Iterable, Mapping, MutableMapping, Optional, Tuple

MetricPayload = Mapping[str, Optional[float]]
AggregatedMetrics = Dict[str, Optional[float]]
ScoreDetail = Dict[str, float]
ScoreBoard = Dict[str, Tuple[float, ScoreDetail]]


_METRIC_KEYS = (
    "completion",   # higher is better
    "casualty",     # lower is better
    "avg_time",     # lower is better
    "ticks",        # lower is better
    "path",         # lower is better
    "congestion",   # lower is better
)

# A small bias towards evacuation success and safety.  The absolute values do
# not matter; they only describe the relative importance of each metric.
_METRIC_WEIGHTS = {
    "completion": 1.4,
    "casualty": 1.2,
    "avg_time": 0.9,
    "ticks": 0.8,
    "path": 0.5,
    "congestion": 0.4,
}

# Whether a lower value should count as a better outcome.
_METRIC_INVERT = {
    "completion": False,
    "casualty": True,
    "avg_time": True,
    "ticks": True,
    "path": True,
    "congestion": True,
}


def _mean(values: Iterable[Optional[float]]) -> Optional[float]:
    collected = [v for v in values if v is not None]
    if not collected:
        return None
    return sum(collected) / float(len(collected))


def aggregate_mode_metrics(runs: Iterable[MetricPayload]) -> AggregatedMetrics:
    """Collapse multiple run payloads for a mode into simple averages."""
    runs = list(runs)
    return {
        "count": float(len(runs)),
        "completion": _mean(run.get("completion_rate") for run in runs),
        "casualty": _mean(run.get("casualty_rate") for run in runs),
        "avg_time": _mean(run.get("average_evacuation_time") for run in runs),
        "ticks": _mean(run.get("total_ticks") for run in runs),
        "path": _mean(run.get("avg_path_length_evacuated") for run in runs),
        "congestion": _mean(run.get("congestion_ratio") for run in runs),
    }


def _normalise(value: Optional[float], rng: Optional[Tuple[float, float]], invert: bool) -> float:
    if value is None or rng is None:
        return 0.0
    low, high = rng
    if high - low <= 1e-9:
        return 1.0
    ratio = (value - low) / (high - low)
    if invert:
        ratio = 1.0 - ratio
    if not invert:
        ratio = max(0.0, min(1.0, ratio))
    else:
        ratio = max(0.0, min(1.0, ratio))
    return ratio


def score_modes(metrics: Mapping[str, AggregatedMetrics]) -> ScoreBoard:
    """Return a scoreboard containing a normalised score per movement mode."""
    if not metrics:
        return {}

    ranges: MutableMapping[str, Optional[Tuple[float, float]]] = {}
    for key in _METRIC_KEYS:
        values = [m.get(key) for m in metrics.values() if m.get(key) is not None]
        if values:
            ranges[key] = (min(values), max(values))
        else:
            ranges[key] = None

    scoreboard: ScoreBoard = {}
    for mode, agg in metrics.items():
        total_weight = 0.0
        weighted_score = 0.0
        detail: ScoreDetail = {}

        for key in _METRIC_KEYS:
            value = agg.get(key)
            detail[key] = float(value) if value is not None else 0.0
            if value is None:
                continue
            rng = ranges.get(key)
            weight = _METRIC_WEIGHTS.get(key, 1.0)
            score = _normalise(value, rng, _METRIC_INVERT.get(key, False))
            weighted_score += score * weight
            total_weight += weight

        if total_weight == 0.0:
            final_score = 0.0
        else:
            final_score = weighted_score / total_weight

        scoreboard[mode] = (final_score, detail)

    return scoreboard
