import numpy as np
from dataclasses import dataclass
from typing import List, Tuple
from config import (
    PHEROMONE_FLOOR,
    FIRE_SPAWN_BAND,
    FIRE_SPAWN_BAND_WIDTH,
    FIRE_SAFE_THRESHOLD,
    FIRE_SINGLE_SOURCE,
    FIRE_SPAWN_COUNT,
    NO_SPAWN_IN_FIRE,
)

EMPTY = 0
WALL  = 1
EXIT  = 2

@dataclass
class GridSpec:
    rows: int
    cols: int
    crowd: int
    exits: int
    wall_density: float

class Grid:
    def __init__(self, spec: GridSpec, rng: np.random.Generator, fire_params: dict | None = None):
        self.spec = spec
        self.rng = rng

        R, C = spec.rows, spec.cols
        self.types = np.zeros((R, C), dtype=np.int8)
        self.pheromone = np.full((R, C), PHEROMONE_FLOOR, dtype=np.float32)
        self.fire = np.zeros((R, C), dtype=np.float32)
        self.smoke = np.zeros((R, C), dtype=np.float32)
        self.congestion = np.zeros((R, C), dtype=np.float32)
        self.exit_compromised = np.zeros((R, C), dtype=bool)
        self.agents: List[Tuple[int,int]] = []
        self.agent_ids: List[int] = []  # Persistent unique IDs for each agent
        self.next_agent_id: int = 0  # Counter for generating unique IDs

        self.fire_params = self._normalize_fire_params(fire_params)

        self._randomize_layout()
        # seed pheromone from distance map (fast backbone)
        try:
            from .seed import seed_pheromone_from_dist
            seed_pheromone_from_dist(self)
        except Exception:
            pass

        self._initial_snapshot_ready = False
        self.store_initial_state()

    def inside(self, r, c):
        return 0 <= r < self.spec.rows and 0 <= c < self.spec.cols

    def neighbors4(self, r, c):
        for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
            nr, nc = r+dr, c+dc
            if self.inside(nr, nc):
                yield nr, nc

    def _normalize_fire_params(self, params: dict | None) -> dict:
        base = {
            "band": FIRE_SPAWN_BAND,
            "band_width": FIRE_SPAWN_BAND_WIDTH,
            "single_source": FIRE_SINGLE_SOURCE,
            "spawn_count": FIRE_SPAWN_COUNT,
        }
        if params:
            for key in base:
                if key in params:
                    base[key] = params[key]
        return base

    def set_fire_params(self, params: dict):
        self.fire_params = self._normalize_fire_params(params)

    def store_initial_state(self):
        """Snapshot current layout and dynamic fields for later restoration."""
        self._initial_types = self.types.copy()
        self._initial_fire = self.fire.copy()
        self._initial_smoke = self.smoke.copy()
        self._initial_pheromone = self.pheromone.copy()
        self._initial_congestion = self.congestion.copy()
        self._initial_agents = list(self.agents)
        self._initial_agent_ids = list(self.agent_ids)
        self._initial_next_agent_id = self.next_agent_id
        self._initial_exit_compromised = self.exit_compromised.copy()
        self._initial_snapshot_ready = True

    def restore_initial_state(self):
        if not getattr(self, "_initial_snapshot_ready", False):
            return False
        self.types[:, :] = self._initial_types
        self.fire[:, :] = self._initial_fire
        self.smoke[:, :] = self._initial_smoke
        self.pheromone[:, :] = self._initial_pheromone
        self.congestion[:, :] = self._initial_congestion
        self.agents = list(self._initial_agents)
        self.agent_ids = list(self._initial_agent_ids)
        self.next_agent_id = self._initial_next_agent_id
        self.exit_compromised[:, :] = self._initial_exit_compromised
        return True

    # ---------- BFS reachability helper ----------
    def _bfs_from_exits(self):
        R, C = self.types.shape
        reachable = np.zeros((R, C), dtype=bool)
        from collections import deque
        q = deque()
        exits = np.argwhere(self.types == EXIT)
        for er, ec in exits:
            reachable[int(er), int(ec)] = True
            q.append((int(er), int(ec)))
        while q:
            r, c = q.popleft()
            for nr, nc in self.neighbors4(r, c):
                if not reachable[nr, nc] and self.types[nr, nc] != WALL:
                    reachable[nr, nc] = True
                    q.append((nr, nc))
        return reachable

    # ---------- layout generator with connectivity repair ----------
    def _randomize_layout(self):
        R, C = self.spec.rows, self.spec.cols
        total = R * C
        self.exit_compromised.fill(False)

        # 1) random walls
        n_walls = int(total * self.spec.wall_density)
        if n_walls > 0:
            positions = self.rng.choice(total, n_walls, replace=False)
            self.types.flat[positions] = WALL

        # 2) place exits (border preferentially), spaced apart
        border = [(0,c) for c in range(C)] + [(R-1,c) for c in range(C)]
        border += [(r,0) for r in range(R)] + [(r,C-1) for r in range(R)]
        self.rng.shuffle(border)
        placed_exits = []

        def far_enough(a, b):
            return abs(a[0]-b[0]) + abs(a[1]-b[1]) >= 4

        for r,c in border:
            if self.types[r,c] == WALL: continue
            if all(far_enough((r,c), e) for e in placed_exits):
                self.types[r,c] = EXIT
                placed_exits.append((r,c))
                if len(placed_exits) >= self.spec.exits: break

        # fallback to interior if not enough
        if len(placed_exits) < self.spec.exits:
            empties = np.argwhere(self.types != WALL)
            self.rng.shuffle(empties)
            for rr,cc in empties:
                rr,cc = int(rr), int(cc)
                if all(far_enough((rr,cc), e) for e in placed_exits):
                    self.types[rr,cc] = EXIT
                    placed_exits.append((rr,cc))
                    if len(placed_exits) >= self.spec.exits: break

        # 3) connectivity check and repair if too many unreachable
        reachable = self._bfs_from_exits()
        unreachable = np.argwhere(~reachable & (self.types != WALL))
        empties_count = np.sum(self.types != WALL)
        if empties_count > 0 and len(unreachable) / float(empties_count) > 0.05:
            attempts = 0
            while attempts < 800 and np.any(~reachable & (self.types != WALL)):
                attempts += 1
                unreachable = np.argwhere(~reachable & (self.types != WALL))
                ur, uc = map(int, unreachable[self.rng.integers(len(unreachable))])
                # remove a nearby wall
                candidates = []
                for rr in range(max(0, ur-1), min(R, ur+2)):
                    for cc in range(max(0, uc-1), min(C, uc+2)):
                        if self.types[rr, cc] == WALL:
                            candidates.append((rr, cc))
                if candidates:
                    rr, cc = candidates[self.rng.integers(len(candidates))]
                    self.types[rr, cc] = EMPTY
                reachable = self._bfs_from_exits()
                unreachable = np.argwhere(~reachable & (self.types != WALL))
                if len(unreachable) / float(empties_count) <= 0.05:
                    break

        # 4) seed initial fire band before placing agents
        self.seed_initial_fire()

        # 5) place agents in reachable empty cells and not near exits
        reachable = self._bfs_from_exits()
        empties = np.argwhere((self.types == EMPTY) & reachable)
        self.rng.shuffle(empties)

        def near_exit(r,c):
            for er,ec in placed_exits:
                if abs(er-r) + abs(ec-c) < 3:
                    return True
            return False

        cnt = 0
        for rr,cc in empties:
            rr,cc = int(rr), int(cc)
            if near_exit(rr,cc): continue
            if NO_SPAWN_IN_FIRE and self.fire[rr,cc] > 0.01:
                continue
            agent_id = self.next_agent_id
            self.next_agent_id += 1
            self.agents.append((rr,cc))
            self.agent_ids.append(agent_id)
            cnt += 1
            if cnt >= self.spec.crowd:
                break

    def clear_dynamic(self):
        self.fire.fill(0.0)
        self.smoke.fill(0.0)
        self.congestion.fill(0.0)

    def seed_initial_fire(self):
        """Create an initial fire band on one side of the map."""
        self.fire.fill(0.0)
        self.smoke.fill(0.0)
        params = self.fire_params
        band_width = params.get("band_width", FIRE_SPAWN_BAND_WIDTH)
        if band_width <= 0.0:
            return

        orientation = str(params.get("band", FIRE_SPAWN_BAND)).lower().strip()
        if orientation == "random":
            orientation = self.rng.choice(["west", "east", "north", "south"])
        if orientation not in {"west", "east", "north", "south"}:
            orientation = "west"

        R, C = self.spec.rows, self.spec.cols
        band_fraction = max(0.05, min(0.8, float(band_width)))
        rows = (0, R)
        cols = (0, C)
        if orientation == "west":
            cols = (0, max(1, int(C * band_fraction)))
        elif orientation == "east":
            cols = (max(0, C - int(C * band_fraction)), C)
        elif orientation == "north":
            rows = (0, max(1, int(R * band_fraction)))
        elif orientation == "south":
            rows = (max(0, R - int(R * band_fraction)), R)

        candidates = []
        for r in range(rows[0], rows[1]):
            for c in range(cols[0], cols[1]):
                if self.types[r, c] in (WALL, EXIT):
                    continue
                candidates.append((r, c))

        if not candidates:
            return

        single_source = bool(params.get("single_source", FIRE_SINGLE_SOURCE))
        spawn_count = int(params.get("spawn_count", FIRE_SPAWN_COUNT))
        if single_source:
            target = 1
        else:
            target = max(1, min(len(candidates), spawn_count if spawn_count > 0 else int(0.015 * R * C)))

        if target == 1:
            r, c = candidates[int(self.rng.integers(len(candidates)))]
            upper = min(0.35, FIRE_SAFE_THRESHOLD + 0.15)
            base = float(self.rng.uniform(FIRE_SAFE_THRESHOLD, upper))
            self.fire[r, c] = base
            self.smoke[r, c] = max(self.smoke[r, c], 0.5)
        else:
            picks = self.rng.choice(len(candidates), target, replace=False)
            for idx in np.atleast_1d(picks):
                r, c = candidates[int(idx)]
                upper = min(0.35, FIRE_SAFE_THRESHOLD + 0.15)
                base = float(self.rng.uniform(FIRE_SAFE_THRESHOLD, upper))
                self.fire[r, c] = base
                self.smoke[r, c] = max(self.smoke[r, c], 0.5)

    def reset_pheromone(self):
        self.pheromone.fill(PHEROMONE_FLOOR)
