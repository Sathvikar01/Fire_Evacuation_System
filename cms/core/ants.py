import numpy as np
from typing import Tuple, List
from .grid import Grid, EMPTY, WALL, EXIT
from config import (
    ANT_PRE_ITERS, ANT_ALPHA, ANT_BETA, ANT_RHO, ANT_Q, ANT_MAX_STEPS,
    PHEROMONE_FLOOR, FIRE_LOW_THRESHOLD
)

def manhattan(a: Tuple[int,int], b: Tuple[int,int]) -> int:
    return abs(a[0]-b[0]) + abs(a[1]-b[1])

class AntPrecomputer:
    def __init__(self, grid: Grid, rng: np.random.Generator):
        self.g = grid
        self.rng = rng
        self.exit_cells = [(int(r), int(c)) for (r,c) in np.argwhere(self.g.types == EXIT)]

    def _neighbors(self, r, c):
        for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
            nr, nc = r+dr, c+dc
            if 0 <= nr < self.g.spec.rows and 0 <= nc < self.g.spec.cols:
                if self.g.types[nr, nc] != WALL:
                    yield nr, nc

    def _single_ant_walk(self) -> List[Tuple[int,int]]:
        empties = np.argwhere(self.g.types == EMPTY)
        if len(empties) == 0 or len(self.exit_cells) == 0:
            return []
        sr, sc = map(int, empties[self.rng.integers(len(empties))])
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
                fire_penalty = 1.0 if hazard <= FIRE_LOW_THRESHOLD else max(0.05, 1.0 - 3.5 * hazard)
                s = (ph ** ANT_ALPHA) * ((1.0 / d) ** ANT_BETA) * fire_penalty
                if self.g.types[nr, nc] == EXIT: s *= 12.0
                scores.append(max(s, 1e-12))
            tot = float(sum(scores))
            if tot <= 0.0:
                choice = int(self.rng.integers(len(candidates)))
            else:
                probs = np.array(scores, dtype=np.float64)
                probs /= probs.sum()
                probs = np.clip(probs, 0.0, 1.0)
                probs /= probs.sum()
                choice = int(self.rng.choice(len(candidates), p=probs))
            r, c = candidates[choice]
            path.append((r,c))
            visited.add((r,c))
        return []

    def _apply_path_deposit(self, path, q_scale=ANT_Q):
        if not path: return
        L = max(1, len(path))
        delta = (q_scale * 3.5) / float(L)  # Increased from 2.8 for stronger initial pheromone
        for r,c in path:
            self.g.pheromone[r,c] += delta

    def run_chunk(self, iters=20):
        for _ in range(iters):
            self.g.pheromone *= (1.0 - ANT_RHO)
            path = self._single_ant_walk()
            if path:
                self._apply_path_deposit(path, q_scale=ANT_Q)
        np.maximum(self.g.pheromone, PHEROMONE_FLOOR, out=self.g.pheromone)

    def run(self, iters: int = ANT_PRE_ITERS):
        self.run_chunk(iters)

    def emergency(self, iters: int = None):
        its = iters if iters is not None else ANT_PRE_ITERS
        self.run_chunk(its)
