"""Property-based tests for simulation invariants."""
import numpy as np
import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from core.simulation import Simulation
from core.grid import GridSpec


class TestSimulationInvariants:
    """Invariants that must hold at every tick."""

    @pytest.mark.parametrize("seed", [1, 42])
    def test_agent_count_conservation(self, seed):
        """evacuated + casualties + remaining == crowd (no agents lost)."""
        s = Simulation(GridSpec(20, 20, 40, 3, 0.08), seed=seed, movement_mode="aco")
        crowd = s.grid.spec.crowd
        for _ in range(50):
            s.step()
            total = s.engine.evacuated + s.engine.casualties + len(s.grid.agents)
            assert total == crowd, \
                f"Agent count violated: evac={s.engine.evacuated} " \
                f"cas={s.engine.casualties} remaining={len(s.grid.agents)} " \
                f"total={total} expected={crowd}"

    @pytest.mark.parametrize("seed", [1, 42])
    def test_pheromone_finite(self, seed):
        """Pheromone values must be finite (no NaN/Inf)."""
        s = Simulation(GridSpec(15, 15, 20, 2, 0.05), seed=seed, movement_mode="aco")
        for _ in range(20):
            s.step()
            assert np.all(np.isfinite(s.grid.pheromone)), "Pheromone contains NaN/Inf"
            assert np.all(np.isfinite(s.grid.fire)), "Fire contains NaN/Inf"
            assert np.all(np.isfinite(s.grid.smoke)), "Smoke contains NaN/Inf"

    @pytest.mark.parametrize("seed", [1, 42])
    def test_fire_smoke_bounded(self, seed):
        """Fire and smoke must stay in [0, 1]."""
        s = Simulation(GridSpec(15, 15, 20, 2, 0.05), seed=seed, movement_mode="aco")
        for _ in range(30):
            s.step()
            assert s.grid.fire.min() >= 0.0
            assert s.grid.fire.max() <= 1.0
            assert s.grid.smoke.min() >= 0.0
            assert s.grid.smoke.max() <= 1.0
