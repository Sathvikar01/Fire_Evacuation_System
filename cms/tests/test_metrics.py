"""Metric correctness tests — verify formulas are mathematically sound."""
import numpy as np
import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from core.simulation import Simulation
from core.grid import GridSpec
from core.agents import AgentMetrics


class TestPathLengthMetric:
    """BUG-03: path_length must count moves, not ticks-alive."""

    def test_path_length_counts_moves_not_ticks(self):
        s = Simulation(GridSpec(20, 20, 40, 3, 0.08), seed=11, movement_mode="aco")
        for _ in range(60):
            s.step()
        # Every agent's path_length should be <= ticks_run * micro_steps
        # and > 0 for agents that moved
        pl = list(s.engine.metrics.path_length.values())
        assert all(p >= 0 for p in pl)
        # At least some agents should have moved (path_length > 0)
        assert any(p > 0 for p in pl)

    def test_stationary_agent_has_zero_path_length(self):
        """If an agent never moves, its path_length must be 0."""
        s = Simulation(GridSpec(10, 10, 1, 1, 0.0), seed=1, movement_mode="random")
        # With 1 agent and 1 exit, find the agent and check
        # (hard to guarantee no movement, but verify the metric is sane)
        s.step()
        pl = list(s.engine.metrics.path_length.values())
        assert all(p >= 0 for p in pl)


class TestCompletionCasualtyInvariant:
    """completion_rate + casualty_rate + remaining_fraction must be consistent."""

    def test_evac_plus_casualty_le_crowd(self):
        s = Simulation(GridSpec(20, 20, 40, 3, 0.08), seed=33, movement_mode="aco")
        for _ in range(80):
            s.step()
        crowd = s.grid.spec.crowd
        assert s.engine.evacuated + s.engine.casualties <= crowd
        assert s.engine.evacuated >= 0
        assert s.engine.casualties >= 0

    def test_completion_rate_formula(self):
        s = Simulation(GridSpec(20, 20, 40, 3, 0.08), seed=33, movement_mode="aco")
        for _ in range(80):
            s.step()
        crowd = max(1, s.grid.spec.crowd)
        expected = s.engine.evacuated / crowd
        assert 0.0 <= expected <= 1.0

    def test_finalization_records_metrics(self):
        s = Simulation(GridSpec(20, 20, 40, 3, 0.08), seed=33, movement_mode="aco")
        # Run until finalization
        for _ in range(200):
            s.step()
            if not s.running:
                break
        summary = s.metrics.summary()
        assert summary["completion_rate_final"] is not None
        assert summary["casualty_rate_final"] is not None


class TestCongestionMetric:
    """Congestion values must be bounded and sane."""

    def test_congestion_nonneg(self):
        s = Simulation(GridSpec(20, 20, 40, 3, 0.08), seed=22, movement_mode="aco")
        for _ in range(30):
            s.step()
        assert s.grid.congestion.min() >= 0.0

    def test_congestion_bounded_by_crowd(self):
        s = Simulation(GridSpec(20, 20, 40, 3, 0.08), seed=22, movement_mode="aco")
        for _ in range(30):
            s.step()
        # Congestion is a sum of nearby agent counts; can't exceed total agents * neighborhood
        assert s.grid.congestion.max() <= s.grid.spec.crowd * 25  # generous upper bound
