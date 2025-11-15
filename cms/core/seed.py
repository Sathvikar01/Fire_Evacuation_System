import numpy as np
from .grid import EXIT, WALL
from config import PHEROMONE_FLOOR

def compute_distance_map(types):
    R, C = types.shape
    INF = 10**9
    dist = np.full((R, C), INF, dtype=np.int32)
    from collections import deque
    q = deque()
    exits = np.argwhere(types == EXIT)
    for er, ec in exits:
        dist[int(er), int(ec)] = 0
        q.append((int(er), int(ec)))
    while q:
        r, c = q.popleft()
        d = dist[r, c]
        for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
            nr, nc = r+dr, c+dc
            if 0 <= nr < R and 0 <= nc < C:
                if types[nr, nc] != WALL and dist[nr, nc] > d + 1:
                    dist[nr, nc] = d + 1
                    q.append((nr, nc))
    return dist

def seed_pheromone_from_dist(grid):
    dist = compute_distance_map(grid.types)
    valid = dist < 10**8
    if not valid.any():
        return
    maxd = float(dist[valid].max())
    if maxd <= 0: maxd = 1.0
    scaled = (maxd - dist.astype(float)) / maxd
    scaled = np.clip(scaled, 0.0, 1.0)
    # set base pheromone stronger closer to exits
    grid.pheromone[valid] = PHEROMONE_FLOOR + 0.95 * scaled[valid]
