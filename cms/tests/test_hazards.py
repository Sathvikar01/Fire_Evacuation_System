"""Fire and smoke spread correctness tests."""
import numpy as np
import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from core.simulation import Simulation
from core.grid import GridSpec, WALL
from core.hazards import step_fire_and_smoke


class TestFireSpread:
    """Fire must stay within bounds and respect walls."""

    def test_fire_never_exceeds_1(self):
        s = Simulation(GridSpec(20, 20, 40, 3, 0.08), seed=44, movement_mode="aco")
        for _ in range(50):
            s.step()
        assert s.grid.fire.max() <= 1.0

    def test_fire_never_negative(self):
        s = Simulation(GridSpec(20, 20, 40, 3, 0.08), seed=44, movement_mode="aco")
        for _ in range(50):
            s.step()
        assert s.grid.fire.min() >= 0.0

    def test_fire_does_not_enter_walls(self):
        s = Simulation(GridSpec(20, 20, 40, 3, 0.15), seed=44, movement_mode="aco")
        for _ in range(50):
            s.step()
        wall_mask = s.grid.types == WALL
        assert not np.any(s.grid.fire[wall_mask] > 0.01)

    def test_fire_starts_from_single_source(self):
        s = Simulation(GridSpec(25, 25, 50, 3, 0.1), seed=88, movement_mode="aco")
        # At t=0, should have exactly 1 fire cell (single_source=True by default)
        assert np.count_nonzero(s.grid.fire > 0.01) == 1


class TestSmokeSpread:
    """Smoke must be bounded and non-negative."""

    def test_smoke_never_exceeds_1(self):
        s = Simulation(GridSpec(20, 20, 40, 3, 0.08), seed=44, movement_mode="aco")
        for _ in range(50):
            s.step()
        assert s.grid.smoke.max() <= 1.0

    def test_smoke_never_negative(self):
        s = Simulation(GridSpec(20, 20, 40, 3, 0.08), seed=44, movement_mode="aco")
        for _ in range(50):
            s.step()
        assert s.grid.smoke.min() >= 0.0

    def test_smoke_associated_with_fire(self):
        """Fire cells should generally have smoke."""
        s = Simulation(GridSpec(20, 20, 40, 3, 0.08), seed=44, movement_mode="aco")
        for _ in range(30):
            s.step()
        fire_cells = s.grid.fire > 0.05
        if fire_cells.any():
            smoke_at_fire = s.grid.smoke[fire_cells]
            assert smoke_at_fire.mean() > 0.1  # Fire cells should have smoke


class TestHazardStepDeterminism:
    """step_fire_and_smoke must be deterministic with the same RNG state."""

    def test_same_rng_same_output(self):
        from core.grid import Grid, GridSpec
        spec = GridSpec(20, 20, 40, 3, 0.1)
        g = Grid(spec, np.random.default_rng(99))
        types = g.types.copy()
        fire = g.fire.copy()
        smoke = g.smoke.copy()

        rng1 = np.random.default_rng(42)
        nf1, ns1 = step_fire_and_smoke(types, fire.copy(), smoke.copy(), rng1)

        rng2 = np.random.default_rng(42)
        nf2, ns2 = step_fire_and_smoke(types, fire.copy(), smoke.copy(), rng2)

        assert np.array_equal(nf1, nf2)
        assert np.array_equal(ns1, ns2)
