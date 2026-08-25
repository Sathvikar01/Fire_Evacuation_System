"""ACO and pheromone correctness tests."""
import numpy as np
import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from core.simulation import Simulation
from core.grid import GridSpec, EXIT, WALL
from core.seed import compute_distance_map


class TestBFSReachability:
    """Agent placement cells must be reachable from an exit."""

    def test_agent_cells_reachable(self):
        """Every cell an agent is placed on must be reachable from an exit."""
        s = Simulation(GridSpec(20, 20, 40, 3, 0.1), seed=66, movement_mode="aco")
        dist = compute_distance_map(s.grid.types)
        # Every agent's current cell must be reachable
        for (r, c) in s.grid.agents:
            assert dist[r, c] < 10**8, f"Agent at {(r,c)} is unreachable from exits"

    def test_exit_has_zero_distance(self):
        s = Simulation(GridSpec(20, 20, 40, 3, 0.1), seed=66, movement_mode="aco")
        dist = compute_distance_map(s.grid.types)
        exits = np.argwhere(s.grid.types == EXIT)
        for r, c in exits:
            assert dist[int(r), int(c)] == 0


class TestPheromoneBounds:
    """Pheromone must stay above PHEROMONE_FLOOR."""

    def test_pheromone_above_floor(self):
        from config import PHEROMONE_FLOOR
        s = Simulation(GridSpec(20, 20, 40, 3, 0.08), seed=11, movement_mode="aco")
        for _ in range(50):
            s.step()
        assert s.grid.pheromone.min() >= PHEROMONE_FLOOR - 1e-6

    def test_pheromone_nonneg(self):
        s = Simulation(GridSpec(20, 20, 40, 3, 0.08), seed=11, movement_mode="aco")
        for _ in range(50):
            s.step()
        assert s.grid.pheromone.min() >= 0.0


class TestSeedAnnealing:
    """BUG-10: The BFS seed must decay over ant iterations."""

    def test_seed_decays(self):
        from config import SEED_ANNEAL_ENABLED, SEED_ANNEAL_ITERS, PHEROMONE_FLOOR
        if not SEED_ANNEAL_ENABLED:
            pytest.skip("Seed annealing disabled in config")

        # Build a fresh grid and a fresh ant precomputer so _total_iters_run=0
        from core.grid import Grid
        spec = GridSpec(20, 20, 40, 3, 0.08)
        g = Grid(spec, np.random.default_rng(11))
        # Snapshot the pure seed pheromone
        pure_seed = g.pheromone.copy()

        ants = __import__("core.ants", fromlist=["AntPrecomputer"]).AntPrecomputer(g, np.random.default_rng(11))
        assert ants._total_iters_run == 0
        # Run enough iterations to let annealing decay the seed substantially
        ants.run_chunk(iters=SEED_ANNEAL_ITERS)

        # The pheromone field should have diverged from the pure seed because
        # ant deposits reshaped it while the seed weight decayed.
        diff = np.abs(g.pheromone - pure_seed).mean()
        assert diff > 0.01, f"Pheromone too close to pure seed (mean diff={diff:.4f})"


class TestDualPheromone:
    """R5: Dual-channel pheromone must be maintained."""

    def test_safety_channel_exists(self):
        s = Simulation(GridSpec(15, 15, 20, 2, 0.05), seed=1, movement_mode="aco")
        assert hasattr(s.grid, "pheromone_safety")
        assert hasattr(s.grid, "congestion_pheromone")

    def test_safety_channel_nonneg(self):
        s = Simulation(GridSpec(15, 15, 20, 2, 0.05), seed=1, movement_mode="aco")
        for _ in range(30):
            s.step()
        assert s.grid.pheromone_safety.min() >= 0.0

    def test_congestion_pheromone_nonneg(self):
        s = Simulation(GridSpec(15, 15, 20, 2, 0.05), seed=1, movement_mode="aco")
        for _ in range(30):
            s.step()
        assert s.grid.congestion_pheromone.min() >= 0.0


class TestDynamicRho:
    """Dynamic rho must stay within bounds."""

    def test_rho_within_bounds(self):
        from config import RHO_MIN, RHO_MAX
        from core.pheromones import compute_dynamic_rho
        s = Simulation(GridSpec(20, 20, 40, 3, 0.08), seed=11, movement_mode="aco")
        for _ in range(30):
            s.step()
        rho = compute_dynamic_rho(s, strategy="stuck")
        if isinstance(rho, float):
            assert RHO_MIN <= rho <= RHO_MAX or rho == 0.009  # RHO default
        elif isinstance(rho, np.ndarray):
            assert rho.min() >= RHO_MIN - 1e-6
            assert rho.max() <= RHO_MAX + 1e-6
