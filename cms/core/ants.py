import numpy as np
from typing import Tuple, List
from .grid import Grid, EMPTY, WALL, EXIT
from config import (
    ANT_PRE_ITERS, ANT_ALPHA, ANT_BETA, ANT_RHO, ANT_Q, ANT_MAX_STEPS,
    PHEROMONE_FLOOR, FIRE_LOW_THRESHOLD,
    SEED_ANNEAL_ENABLED, SEED_ANNEAL_ITERS, SEED_ANNEAL_FLOOR,
    HAZARD_FORECAST_ENABLED, HAZARD_FORECAST_HORIZON, HAZARD_FORECAST_GAMMA,
)

def manhattan(a: Tuple[int,int], b: Tuple[int,int]) -> int:
    return abs(a[0]-b[0]) + abs(a[1]-b[1])

class AntPrecomputer:
    def __init__(self, grid: Grid, rng: np.random.Generator):
        self.g = grid
        self.rng = rng
        self.exit_cells = [(int(r), int(c)) for (r,c) in np.argwhere(self.g.types == EXIT)]
        # Snapshot the BFS seed so we can anneal it over iterations.
        # Without this, the seed (amplitude ~1.0) dominates ant deposits
        # (~0.03/cell) by ~30x, making the "ACO" essentially a distance walk.
        self._seed_pheromone = self.g.pheromone.copy()
        # Track pure ant deposits separately to avoid double-counting seed after evaporation
        self._pure_deposits = np.zeros_like(self.g.pheromone)
        self._pure_deposits_safety = np.zeros_like(self.g.pheromone_safety) if hasattr(self.g, 'pheromone_safety') else None
        self._total_iters_run = 0

    def _neighbors(self, r, c):
        for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
            nr, nc = r+dr, c+dc
            if 0 <= nr < self.g.spec.rows and 0 <= nc < self.g.spec.cols:
                if self.g.types[nr, nc] != WALL:
                    yield nr, nc

    def _single_ant_walk(self) -> List[Tuple[int,int]]:
        # FIX: filter fire cells so ants don't seed pheromone through fire
        candidates = np.argwhere((self.g.types == EMPTY) & (self.g.fire <= FIRE_LOW_THRESHOLD))
        if len(candidates) == 0:
            candidates = np.argwhere(self.g.types == EMPTY)
        if len(candidates) == 0 or len(self.exit_cells) == 0:
            return []
        sr, sc = map(int, candidates[self.rng.integers(len(candidates))])
        r, c = sr, sc
        path = [(r,c)]
        visited = set(path)
        steps = 0
        while steps < ANT_MAX_STEPS:
            steps += 1
            if self.g.types[r,c] == EXIT:
                return path
            candidates = [(nr,nc) for (nr,nc) in self._neighbors(r,c) if (nr,nc) not in visited]
            if not candidates:
                candidates = list(self._neighbors(r,c))
                if not candidates: break
            scores = []
            for (nr,nc) in candidates:
                d = min(abs(nr-e[0]) + abs(nc-e[1]) for e in self.exit_cells)
                d = max(1, d)
                ph = self.g.pheromone[nr,nc]
                hazard = self.g.fire[nr, nc]
                # R4: Hazard forecasting. Instead of using current fire intensity,
                # predict future intensity by linear extrapolation of growth_step
                # over HAZARD_FORECAST_HORIZON ticks. This makes ants route around
                # where fire *will be*, not just where it currently is.
                if HAZARD_FORECAST_ENABLED:
                    from config import FIRE_GROWTH_STEP
                    predicted_hazard = min(1.0, hazard + HAZARD_FORECAST_HORIZON * FIRE_GROWTH_STEP * (1.0 - hazard))
                    fire_penalty = 1.0 if predicted_hazard <= FIRE_LOW_THRESHOLD else max(0.02, 1.0 - HAZARD_FORECAST_GAMMA * predicted_hazard)
                else:
                    fire_penalty = 1.0 if hazard <= FIRE_LOW_THRESHOLD else max(0.05, 1.0 - 3.5 * hazard)
                s = (ph ** ANT_ALPHA) * ((1.0 / d) ** ANT_BETA) * fire_penalty
                if self.g.types[nr, nc] == EXIT: s *= 12.0
                scores.append(max(s, 1e-12))
            tot = float(sum(scores))
            if tot <= 0.0 or not np.isfinite(tot):
                choice = int(self.rng.integers(len(candidates)))
            else:
                probs = np.array(scores, dtype=np.float64)
                # Single normalization with clip and finite check (previously double-norm)
                probs = probs / tot
                probs = np.clip(probs, 0.0, 1.0)
                s = probs.sum()
                if s > 0 and np.isfinite(s):
                    probs /= s
                else:
                    probs = np.full(len(candidates), 1.0 / len(candidates))
                choice = int(self.rng.choice(len(candidates), p=probs))
            r, c = candidates[choice]
            path.append((r,c))
            visited.add((r,c))
        return []

    def _apply_path_deposit(self, path, q_scale=ANT_Q):
        if not path: return
        L = max(1, len(path))
        delta = (q_scale * 3.5) / float(L)
        for r,c in path:
            self.g.pheromone[r,c] += delta
            self._pure_deposits[r,c] += delta
            # R5: Deposit on the safety channel weighted by inverse hazard.
            from config import DUAL_PHEROMONE_ENABLED
            if DUAL_PHEROMONE_ENABLED and hasattr(self.g, 'pheromone_safety') and self._pure_deposits_safety is not None:
                hazard = float(self.g.fire[r, c])
                safety_weight = max(0.1, 1.0 - hazard * 2.0)
                inc = delta * safety_weight
                self.g.pheromone_safety[r, c] += inc
                self._pure_deposits_safety[r,c] += inc

    def run_chunk(self, iters=20):
        # Snapshot pre-chunk iteration count for per-iteration annealing.
        start_iters = self._total_iters_run

        for _ in range(iters):
            # Evaporate both the field and the pure deposits (so deposits also decay)
            self.g.pheromone *= (1.0 - ANT_RHO)
            self._pure_deposits *= (1.0 - ANT_RHO)
            from config import DUAL_PHEROMONE_ENABLED
            if DUAL_PHEROMONE_ENABLED and hasattr(self.g, 'pheromone_safety') and self._pure_deposits_safety is not None:
                self.g.pheromone_safety *= (1.0 - ANT_RHO)
                self._pure_deposits_safety *= (1.0 - ANT_RHO)
            path = self._single_ant_walk()
            if path:
                self._apply_path_deposit(path, q_scale=ANT_Q)
        self._total_iters_run += iters

        # Correct annealing: blend seed with PURE deposits only (not already-seeded field)
        # Previously ant_field already contained evaporated seed → double-counted seed.
        if SEED_ANNEAL_ENABLED and SEED_ANNEAL_ITERS > 0:
            progress = min(1.0, self._total_iters_run / float(SEED_ANNEAL_ITERS))
            seed_weight = max(SEED_ANNEAL_FLOOR, 1.0 - progress * (1.0 - SEED_ANNEAL_FLOOR))
        else:
            seed_weight = 1.0

        # Re-blend in-place to avoid rebinding grid.pheromone (external views would go stale)
        if SEED_ANNEAL_ENABLED:
            # pheromone = w*seed + (1-w)*pure_deposits + floor blending handled via max
            blended = seed_weight * self._seed_pheromone + (1.0 - seed_weight) * self._pure_deposits
            # Preserve floor and evaporated seed contribution already in pure_deposits?
            # Use copyto to keep same ndarray object
            np.copyto(self.g.pheromone, blended)
            np.maximum(self.g.pheromone, PHEROMONE_FLOOR, out=self.g.pheromone)
            from config import DUAL_PHEROMONE_ENABLED
            if DUAL_PHEROMONE_ENABLED and hasattr(self.g, 'pheromone_safety') and self._pure_deposits_safety is not None:
                blended_safety = seed_weight * self._seed_pheromone + (1.0 - seed_weight) * self._pure_deposits_safety
                np.copyto(self.g.pheromone_safety, blended_safety)
                np.maximum(self.g.pheromone_safety, PHEROMONE_FLOOR, out=self.g.pheromone_safety)
        else:
            np.maximum(self.g.pheromone, PHEROMONE_FLOOR, out=self.g.pheromone)
            from config import DUAL_PHEROMONE_ENABLED
            if DUAL_PHEROMONE_ENABLED and hasattr(self.g, 'pheromone_safety'):
                np.maximum(self.g.pheromone_safety, PHEROMONE_FLOOR, out=self.g.pheromone_safety)

    def run(self, iters: int = ANT_PRE_ITERS):
        self.run_chunk(iters)

    def emergency(self, iters: int = None):
        its = iters if iters is not None else ANT_PRE_ITERS
        self.run_chunk(its)
