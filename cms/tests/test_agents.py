"""Agent movement and collision resolution tests."""
import numpy as np
import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from core.simulation import Simulation
from core.grid import GridSpec


class TestMovementModes:
    """All three movement modes must produce sane results."""

    @pytest.mark.parametrize("mode", ["aco", "distance", "random"])
    def test_mode_runs_without_error(self, mode):
        s = Simulation(GridSpec(15, 15, 20, 2, 0.05), seed=1, movement_mode=mode)
        for _ in range(30):
            s.step()
        assert s.engine.evacuated + s.engine.casualties <= s.grid.spec.crowd

    def test_random_mode_worst_or_equal(self):
        """Random baseline should generally be worst (or equal) at evacuation."""
        results = {}
        for mode in ["aco", "distance", "random"]:
            s = Simulation(GridSpec(20, 20, 40, 3, 0.08), seed=11, movement_mode=mode)
            for _ in range(60):
                s.step()
            results[mode] = s.engine.evacuated
        # Random should not beat ACO (with high probability on this seed)
        assert results["random"] <= results["aco"]


class TestCollisionResolution:
    """Collision resolution sanity checks."""

    def test_no_excessive_duplicate_positions(self):
        """At most a reasonable fraction of agents may share cells (the collision
        model allows moving into a stationary agent's cell in edge cases)."""
        s = Simulation(GridSpec(20, 20, 40, 3, 0.08), seed=22, movement_mode="aco")
        max_dups = 0
        for _ in range(50):
            s.step()
            positions = s.grid.agents
            if len(positions) > 1:
                pos_set = set(positions)
                dups = len(positions) - len(pos_set)
                max_dups = max(max_dups, dups)
        # The collision model allows some overlap (agents can move into a
        # stationary agent's cell). Cap at 30% to catch total collapse.
        assert max_dups <= s.grid.spec.crowd * 0.3, \
            f"Too many duplicate positions: {max_dups} duplicates"

    def test_agents_stay_in_bounds(self):
        s = Simulation(GridSpec(15, 15, 20, 2, 0.05), seed=33, movement_mode="aco")
        R, C = 15, 15
        for _ in range(30):
            s.step()
            for (r, c) in s.grid.agents:
                assert 0 <= r < R, f"Agent out of bounds: r={r}"
                assert 0 <= c < C, f"Agent out of bounds: c={c}"


class TestStuckEscape:
    """Stuck agents should eventually escape or be moved."""

    def test_no_permanent_stuck(self):
        """After many ticks, most agents should have evacuated or died."""
        s = Simulation(GridSpec(20, 20, 40, 3, 0.08), seed=44, movement_mode="aco")
        for _ in range(200):
            s.step()
            if not s.running:
                break
        # The sim should have finalized (all agents evacuated or dead)
        assert not s.running or (s.engine.evacuated + s.engine.casualties) >= s.grid.spec.crowd
