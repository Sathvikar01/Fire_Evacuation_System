"""Shared pytest fixtures for CMS-DACO — research reproducibility."""
import os
import sys
import pytest
import numpy as np

# Ensure cms root on path (handles python -m pytest from repo root or cms/)
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

# Force headless Qt platform so tests don't need display
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from core.grid import GridSpec
from core.simulation import Simulation
import config


@pytest.fixture
def rng():
    return np.random.default_rng(42)


@pytest.fixture
def small_spec():
    return GridSpec(rows=20, cols=20, crowd=20, exits=2, wall_density=0.05)


@pytest.fixture
def sim_factory():
    """Factory to create Simulation with deterministic seed and no QTimer side effects."""
    def _make(spec=None, seed=42, mode=None):
        if spec is None:
            spec = GridSpec(rows=20, cols=20, crowd=20, exits=2, wall_density=0.05)
        s = Simulation(spec, seed=seed, movement_mode=mode or config.MOVEMENT_MODE_ACO)
        # Stop timer if it was created (headless should not have one)
        if getattr(s, 'timer', None) is not None:
            try:
                s.timer.stop()
            except Exception:
                pass
        return s
    return _make


@pytest.fixture(autouse=True)
def isolate_config():
    """Save/restore global config flags that tests may mutate."""
    saved = {
        'DUAL_PHEROMONE_ENABLED': config.DUAL_PHEROMONE_ENABLED,
        'USE_DUAL_PHEROMONE': getattr(config, 'USE_DUAL_PHEROMONE', False),
        'ACO_TEMPERATURE': config.ACO_TEMPERATURE,
        'FIRE_TRAVERSAL_THRESHOLD': config.FIRE_TRAVERSAL_THRESHOLD,
    }
    yield
    for k, v in saved.items():
        setattr(config, k, v)
