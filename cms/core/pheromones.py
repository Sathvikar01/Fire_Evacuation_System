import numpy as np
from typing import Sequence
from config import RHO, Q, FIRE_LOCAL_EVAP_BONUS, PHEROMONE_FLOOR, FIRE_LOW_THRESHOLD
from config import RHO_MIN, RHO_MAX, RHO_STUCK_MULT, RHO_AGENT_GAMMA, RHO_CONGESTION_MULT, STUCK_FRAC_TRIGGER, PER_CELL_RHO

def evaporate(pheromone: np.ndarray, fire: np.ndarray, rho=None):
    """Evaporate pheromone with optional dynamic rho (scalar or per-cell array)"""
    if rho is None:
        rho = RHO
    
    # Apply evaporation (supports both scalar and array rho)
    pheromone *= (1.0 - rho)
    
    # Extra evaporation near fire
    hot = fire > FIRE_LOW_THRESHOLD
    if hot.any():
        if isinstance(rho, np.ndarray):
            # Per-cell rho: apply extra evaporation to hot cells
            extra_rho = np.where(hot, np.minimum(rho + FIRE_LOCAL_EVAP_BONUS, 0.95), 0.0)
            pheromone[hot] *= (1.0 - extra_rho[hot])
        else:
            # Scalar rho: use original logic
            extra = max(0.8, 1.0 - (rho + FIRE_LOCAL_EVAP_BONUS))
            pheromone[hot] *= extra
    
    np.maximum(pheromone, PHEROMONE_FLOOR, out=pheromone)

def compute_dynamic_rho(sim, strategy='stuck'):
    """Compute dynamic evaporation rate based on simulation state
    
    Args:
        sim: Simulation instance with grid, engine, and metrics
        strategy: 'stuck', 'agents', or 'congestion'
    
    Returns:
        float or np.ndarray: Dynamic RHO value(s)
    """
    if strategy == 'stuck':
        # Stuck-based: reduce evaporation when many agents are stuck
        active_agents = len(sim.grid.agents)
        if active_agents == 0:
            return RHO
        
        stuck_count = sum(1 for aid in sim.grid.agent_ids 
                         if aid in sim.engine.stuck_counter and sim.engine.stuck_counter[aid] >= sim.stuck_window)
        stuck_frac = stuck_count / active_agents
        
        if stuck_frac <= STUCK_FRAC_TRIGGER:
            return RHO
        
        # Reduce evaporation proportionally to stuck fraction
        rho_dynamic = RHO * (1.0 - stuck_frac * (1.0 - RHO_STUCK_MULT))
        return np.clip(rho_dynamic, RHO_MIN, RHO_MAX)
    
    elif strategy == 'agents':
        # Agents-based: reduce evaporation as fewer agents remain
        initial_agents = sim.grid.spec.crowd
        agents_remaining = len(sim.grid.agents)
        if initial_agents == 0:
            return RHO
        
        agents_frac = agents_remaining / initial_agents
        rho_dynamic = RHO * max(agents_frac ** RHO_AGENT_GAMMA, 0.05)
        return np.clip(rho_dynamic, RHO_MIN, RHO_MAX)
    
    elif strategy == 'congestion':
        # Congestion-based per-cell: increase evaporation in congested areas
        if not PER_CELL_RHO:
            return RHO
        
        congestion_normalized = sim.grid.congestion / max(1.0, sim.grid.congestion.max())
        rho_grid = RHO * (1.0 + congestion_normalized * RHO_CONGESTION_MULT)
        return np.clip(rho_grid, RHO_MIN, RHO_MAX)
    
    else:
        return RHO

def deposit(pheromone: np.ndarray, path_cells, scale: float = 3.0):
    if not path_cells: return
    L = max(1, len(path_cells))
    delta = (Q * scale) / float(L)
    for r,c in path_cells:
        pheromone[r,c] += delta

def reinforce_success(pheromone: np.ndarray, path_cells, success_scale: float = 14.0):
    if not path_cells: return
    L = max(1, len(path_cells))
    delta = (Q * success_scale) / float(L)
    for r,c in path_cells:
        pheromone[r,c] += delta

def punish_area(pheromone: np.ndarray, center_r: int, center_c: int, radius: int = 2, factor: float = 0.75):
    R, C = pheromone.shape
    r0 = max(0, center_r - radius); r1 = min(R, center_r + radius + 1)
    c0 = max(0, center_c - radius); c1 = min(C, center_c + radius + 1)
    pheromone[r0:r1, c0:c1] *= factor
    np.maximum(pheromone, PHEROMONE_FLOOR, out=pheromone)

def evaporate_region(pheromone: np.ndarray, center_r: int, center_c: int, radius: int = 3):
    """Force pheromone around a critical cell back to the floor value."""
    R, C = pheromone.shape
    r0 = max(0, center_r - radius); r1 = min(R, center_r + radius + 1)
    c0 = max(0, center_c - radius); c1 = min(C, center_c + radius + 1)
    pheromone[r0:r1, c0:c1] = PHEROMONE_FLOOR

def suppress_path(pheromone: np.ndarray, path_cells: Sequence[tuple[int, int]], factor: float = 0.55):
    """Reduce pheromone along a path to help agents escape local minima."""
    if not path_cells:
        return
    unique = { (int(r), int(c)) for r, c in path_cells }
    for r, c in unique:
        pheromone[r, c] = max(PHEROMONE_FLOOR, float(pheromone[r, c]) * factor)
