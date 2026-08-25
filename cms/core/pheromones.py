import numpy as np
from typing import Sequence
from config import RHO, Q, FIRE_LOCAL_EVAP_BONUS, PHEROMONE_FLOOR, FIRE_LOW_THRESHOLD
from config import RHO_MIN, RHO_MAX, RHO_STUCK_MULT, RHO_AGENT_GAMMA, RHO_CONGESTION_MULT, STUCK_FRAC_TRIGGER, PER_CELL_RHO
from config import DUAL_PHEROMONE_ENABLED

def evaporate(pheromone: np.ndarray, fire: np.ndarray, rho=None):
    """Evaporate pheromone with optional dynamic rho (scalar or per-cell array).

    FIX: previously did  (1-rho)*(1-(rho+bonus)) double-counting rho on hot cells.
    Now uses single step  (1 - rho - bonus) unified for scalar and array paths.
    """
    if rho is None:
        rho = RHO

    hot = fire > FIRE_LOW_THRESHOLD
    if isinstance(rho, np.ndarray):
        # Per-cell: effective rho = base + bonus on hot cells
        effective_rho = rho.astype(np.float64, copy=True)
        if hot.any():
            # bonus 0.20, cap at 0.95
            effective_rho[hot] = np.minimum(effective_rho[hot] + FIRE_LOCAL_EVAP_BONUS, 0.95)
        pheromone *= (1.0 - effective_rho).astype(pheromone.dtype, copy=False)
    else:
        # Scalar: single-step evaporation
        rho_f = float(rho)
        pheromone *= (1.0 - rho_f)
        if hot.any():
            # Apply bonus once, not (1-rho)*(1-(rho+bonus))
            # Convert previous double-step to single: bonus_extra = bonus / (1-rho) approx
            # For research: hot factor = max(0.05, 1 - rho - bonus) clipped
            hot_factor = max(0.05, 1.0 - rho_f - FIRE_LOCAL_EVAP_BONUS)
            # Previously extra=max(0.8,1-(rho+bonus)) → 0.8 for rho=0.009 gave 0.8,
            # new gives 1-0.009-0.20=0.791 close but consistent with array path
            pheromone[hot] *= hot_factor / (1.0 - rho_f) if (1.0 - rho_f) > 1e-9 else hot_factor
            # Simpler: undo previous * (1-rho) and apply (1-rho-bonus) would be:
            # But we already did *(1-rho), so we need additional *(1 - bonus/(1-rho)) to reach (1-rho-bonus)
            # Instead recompute cleanly: pheromone_hot originally = init * (1-rho)
            # We want init*(1-rho-bonus) → multiply existing by (1-rho-bonus)/(1-rho)
            # Above line does that.

    np.maximum(pheromone, PHEROMONE_FLOOR, out=pheromone)

def evaporate_dual(grid, rho=None):
    """Evaporate both speed and safety pheromone channels.

    The safety channel gets *stronger* evaporation near fire (fire destroys
    safety trails faster), creating a dynamic where safe paths persist while
    hazardous ones fade quickly.
    """
    evaporate(grid.pheromone, grid.fire, rho)
    if DUAL_PHEROMONE_ENABLED and hasattr(grid, 'pheromone_safety'):
        evaporate(grid.pheromone_safety, grid.fire, rho)
        # Safety channel: extra-strong evaporation near fire (previously *0.5 destroyed safety
        # trails in ~3 ticks, defeating dual-channel purpose). Softened to 0.85 for research.
        hot = grid.fire > FIRE_LOW_THRESHOLD
        if hot.any():
            grid.pheromone_safety[hot] *= 0.85
            np.maximum(grid.pheromone_safety, PHEROMONE_FLOOR, out=grid.pheromone_safety)

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
