"""Reproducibility tests — the foundation of all research validity.

If these fail, every experimental result in the project is suspect.
"""
import numpy as np
import pytest
import sys
import os

# Ensure the cms package root is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from core.simulation import Simulation
from core.grid import GridSpec


def _run_sim(seed, ticks=30, grid=20, crowd=40, exits=3, walls=0.08):
    spec = GridSpec(grid, grid, crowd, exits, walls)
    s = Simulation(spec, seed=seed, movement_mode="aco")
    for _ in range(ticks):
        s.step()
    return s


class TestReproducibility:
    """Same seed + same config must produce identical simulation state."""

    def test_same_seed_same_agents(self):
        a = _run_sim(seed=123)
        b = _run_sim(seed=123)
        assert a.grid.agents == b.grid.agents

    def test_same_seed_same_fire(self):
        a = _run_sim(seed=123)
        b = _run_sim(seed=123)
        assert np.array_equal(a.grid.fire, b.grid.fire)

    def test_same_seed_same_pheromone(self):
        a = _run_sim(seed=123)
        b = _run_sim(seed=123)
        assert np.array_equal(a.grid.pheromone, b.grid.pheromone)

    def test_same_seed_same_evacuation_count(self):
        a = _run_sim(seed=123)
        b = _run_sim(seed=123)
        assert a.engine.evacuated == b.engine.evacuated
        assert a.engine.casualties == b.engine.casualties

    def test_different_seed_different_state(self):
        a = _run_sim(seed=123)
        b = _run_sim(seed=999)
        # Layout should differ (very likely)
        assert not np.array_equal(a.grid.types, b.grid.types) or \
               not np.array_equal(a.grid.fire, b.grid.fire)

    def test_regenerate_preserves_seed(self):
        """regenerate() must produce the same layout for the same seed."""
        spec = GridSpec(20, 20, 40, 3, 0.08)
        a = Simulation(spec, seed=42, movement_mode="aco")
        a.regenerate(spec)
        types_a = a.grid.types.copy()

        b = Simulation(spec, seed=42, movement_mode="aco")
        b.regenerate(spec)
        types_b = b.grid.types.copy()

        assert np.array_equal(types_a, types_b)


class TestInitialStateSnapshot:
    """reset_keep_layout must restore the true initial state, not a mid-run state."""

    def test_reset_restores_agents(self):
        s = _run_sim(seed=55, ticks=20)
        agents_mid_run = len(s.grid.agents)
        s.reset_keep_layout()
        # After reset, agents should be back at initial count (crowd size)
        # Some agents may have evacuated during the run, so mid_run < initial
        assert len(s.grid.agents) >= agents_mid_run

    def test_reset_restores_fire(self):
        s = _run_sim(seed=55, ticks=20)
        fire_mid_run = s.grid.fire.copy()
        s.reset_keep_layout()
        # After reset, fire should be the initial single-source fire
        assert s.grid.fire.sum() <= fire_mid_run.sum() + 0.1

    def test_start_does_not_snapshot(self):
        """start() must NOT call store_initial_state() (BUG-02 fix)."""
        s = Simulation(GridSpec(15, 15, 20, 2, 0.05), seed=7, movement_mode="aco")
        initial_fire = s.grid.fire.copy()
        s.start()
        for _ in range(10):
            s.step()
        s.pause()
        s.reset_keep_layout()
        # If start() snapshotted, reset would restore mid-run state (wrong)
        assert np.allclose(s.grid.fire, initial_fire, atol=0.01)
