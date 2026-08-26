import numpy as np
import logging
import heapq
from collections import deque
from typing import List, Tuple, Dict, Optional
from .grid import Grid, EMPTY, WALL, EXIT
from config import (
    ALPHA, BETA, GAMMA, SMOKE_SPEED_PENALTY, FAST_MODE_THRESHOLD,
    MOVEMENT_MODE_ACO, MOVEMENT_MODE_RANDOM, MOVEMENT_MODE_DISTANCE,
    MOVEMENT_MODE_ASTAR, MOVEMENT_MODE_STANDARD_ACO, MOVEMENT_MODE_DSTAR,
    ENABLE_METRICS_TRACKING, CONGESTION_PENALTY_FACTOR,
    FIRE_SAFE_THRESHOLD, FIRE_DEATH_THRESHOLD, FIRE_LOW_THRESHOLD,
    SMOKE_PENALTY_THRESHOLD, AVOID_COMPROMISED_EXITS, FIRE_EXIT_COMPROMISED_THRESHOLD,
    ACO_TEMPERATURE, FIRE_TRAVERSAL_THRESHOLD, DISTANCE_SUPPRESSION_DEFAULT,
    STUCK_ESCAPE_ENABLED, STUCK_ESCAPE_AGENT_TICKS, STUCK_ESCAPE_RANDOM_TICKS,
    STUCK_ESCAPE_GLOBAL_RATIO, STUCK_ESCAPE_DURATION,
    STUCK_ESCAPE_DISTANCE_WEIGHT, STUCK_ESCAPE_PHEROMONE_WEIGHT,
    STUCK_ESCAPE_HAZARD_WEIGHT, STUCK_ESCAPE_CONGESTION_WEIGHT,
    PHEROMONE_FLOOR, DUAL_PHEROMONE_ENABLED, DUAL_PHEROMONE_BLEND,
    BFS_SEED_ENABLED, HAZARD_AWARE_ROUTING_ENABLED,
)
from config import EXPLORATION_EPS, EXPLORATION_DECAY, EXPLORATION_MIN
from .pheromones import reinforce_success, suppress_path, evaporate_region

logger = logging.getLogger(__name__)

def manhattan(a: Tuple[int,int], b: Tuple[int,int]) -> int:
    return abs(a[0]-b[0]) + abs(a[1]-b[1])

class AgentMetrics:
    """Track per-agent evacuation metrics"""
    def __init__(self):
        self.start_tick: Dict[int, int] = {}
        self.end_tick: Dict[int, int] = {}
        self.path_length: Dict[int, int] = {}
        self.is_evacuated: Dict[int, bool] = {}
        self.is_casualty: Dict[int, bool] = {}
    
    def reset(self):
        self.start_tick.clear()
        self.end_tick.clear()
        self.path_length.clear()
        self.is_evacuated.clear()
        self.is_casualty.clear()
    
    def get_evacuation_times(self) -> List[int]:
        """Returns list of evacuation times (ticks) for successfully evacuated agents"""
        times = []
        for agent_id, evacuated in self.is_evacuated.items():
            if evacuated and agent_id in self.end_tick and agent_id in self.start_tick:
                times.append(self.end_tick[agent_id] - self.start_tick[agent_id])
        return times
    
    def get_average_evacuation_time(self) -> Optional[float]:
        times = self.get_evacuation_times()
        return sum(times) / len(times) if times else None

class AgentEngine:
    def __init__(
        self,
        grid: Grid,
        rng: np.random.Generator,
        movement_mode: str = MOVEMENT_MODE_ACO,
        enable_agent_deposits: bool = True,
        avoid_compromised_exits: bool = AVOID_COMPROMISED_EXITS,
    ):
        self.grid = grid
        self.rng = rng
        self.evacuated = 0
        self.casualties = 0
        self.movement_mode = movement_mode
        self.enable_agent_deposits = enable_agent_deposits
        self.avoid_compromised_exits = avoid_compromised_exits
        self.fire_avoid_threshold = FIRE_SAFE_THRESHOLD
        self.distance_suppression = DISTANCE_SUPPRESSION_DEFAULT

        # Partial observability: planner belief fields (None => full knowledge)
        self.belief_fire = None
        self.belief_smoke = None
        self.belief_changed = []

        # D* Lite instances per agent (lazy) for MOVEMENT_MODE_DSTAR
        from config import MOVEMENT_MODE_DSTAR as _MD
        self._dstar_planners: Dict[int, "object"] = {}
        if movement_mode == _MD:
            try:
                from .dstar import DStarLite
                self._DStarLite = DStarLite
            except Exception:
                self._DStarLite = None


        self.exit_cells = [(int(r), int(c)) for (r,c) in np.argwhere(self.grid.types == EXIT)]
        # Use agent IDs as keys instead of indices
        self.last_paths: Dict[int, List[Tuple[int,int]]] = {aid: [] for aid in grid.agent_ids}
        self.prev_pos: Dict[int, Optional[Tuple[int,int]]] = {aid: None for aid in grid.agent_ids}
        self.last_dist: Dict[int, Optional[int]] = {aid: None for aid in grid.agent_ids}
        self.stuck_counter: Dict[int, int] = {aid: 0 for aid in grid.agent_ids}  # Track stuck agents
        self.hybrid_escape_until = 0  # Tick timestamp until which hybrid escape mode stays active
        self.recent_positions: Dict[int, deque[Tuple[int, int]]] = {
            aid: deque(maxlen=32) for aid in grid.agent_ids
        }
        self.escape_cooldown: Dict[int, int] = {aid: -9999 for aid in grid.agent_ids}
        
        # Metrics tracking
        self.metrics = AgentMetrics() if ENABLE_METRICS_TRACKING else None
        self.current_tick = 0
        if self.metrics:
            for agent_id in grid.agent_ids:
                self.metrics.start_tick[agent_id] = 0
                self.metrics.path_length[agent_id] = 0

    def set_avoid_compromised_exits(self, enabled: bool):
        self.avoid_compromised_exits = enabled

    def set_distance_suppression(self, value: float):
        self.distance_suppression = float(max(0.0, min(0.95, value)))

    def _record_position(self, agent_id: int, pos: Tuple[int, int]):
        history = self.recent_positions.setdefault(agent_id, deque(maxlen=32))
        history.append(pos)
        return history

    def _handle_local_minima(self, agent_id: int, pos: Tuple[int, int]):
        if not STUCK_ESCAPE_ENABLED:
            return
        last_trigger = self.escape_cooldown.get(agent_id, -9999)
        if self.current_tick - last_trigger < max(4, STUCK_ESCAPE_DURATION // 3):
            return

        path = self.last_paths.get(agent_id, [])[-20:]
        if path:
            suppress_path(self.grid.pheromone, path, factor=0.5)
            # Also suppress safety and congestion channels so escape doesn't follow burning trail via alternate channel
            if hasattr(self.grid, 'pheromone_safety'):
                suppress_path(self.grid.pheromone_safety, path, factor=0.6)
            if hasattr(self.grid, 'congestion_pheromone'):
                # Congestion pheromone is negative; suppressing means reducing it (less avoidance)
                pass

        recent = list(self.recent_positions.get(agent_id, []))
        if recent:
            suppress_path(self.grid.pheromone, recent[-16:], factor=0.55)
            if hasattr(self.grid, 'pheromone_safety'):
                suppress_path(self.grid.pheromone_safety, recent[-16:], factor=0.65)

        evaporate_region(self.grid.pheromone, pos[0], pos[1], radius=1)
        if hasattr(self.grid, 'pheromone_safety'):
            evaporate_region(self.grid.pheromone_safety, pos[0], pos[1], radius=1)
        self.escape_cooldown[agent_id] = self.current_tick

    def _get_exit_targets(self) -> List[Tuple[int, int]]:
        if not self.exit_cells:
            return []
        if not self.avoid_compromised_exits:
            return self.exit_cells
        allowed = []
        exit_mask = getattr(self.grid, "exit_compromised", None)
        for er, ec in self.exit_cells:
            if self.grid.fire[er, ec] >= FIRE_EXIT_COMPROMISED_THRESHOLD:
                continue
            if exit_mask is not None and exit_mask[er, ec]:
                continue
            allowed.append((er, ec))
        return allowed or self.exit_cells

    def _distance_to_goal(self, pos: Tuple[int, int]) -> int:
        targets = self._get_exit_targets()
        if not targets:
            targets = self.exit_cells or [pos]
        return min(manhattan(pos, goal) for goal in targets)

    def _is_exit_allowed(self, cell: Tuple[int, int]) -> bool:
        if not self.avoid_compromised_exits:
            return True
        r, c = cell
        if self.grid.types[r, c] != EXIT:
            return True
        if self.grid.fire[r, c] >= FIRE_EXIT_COMPROMISED_THRESHOLD:
            return False
        exit_mask = getattr(self.grid, "exit_compromised", None)
        if exit_mask is not None and exit_mask[r, c]:
            return False
        return True

    def _is_fire_safe(self, cell: Tuple[int, int]) -> bool:
        r, c = cell
        return self.grid.fire[r, c] <= FIRE_TRAVERSAL_THRESHOLD

    def _filter_candidates(self, candidates: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
        safe = []
        for cell in candidates:
            if not self._is_fire_safe(cell):
                continue
            if not self._is_exit_allowed(cell):
                continue
            safe.append(cell)
        return safe

    def _update_escape_window(self):
        if not STUCK_ESCAPE_ENABLED or not self.stuck_counter:
            return
        stuck_agents = sum(1 for v in self.stuck_counter.values() if v >= STUCK_ESCAPE_AGENT_TICKS)
        ratio = stuck_agents / float(len(self.stuck_counter))
        if ratio >= STUCK_ESCAPE_GLOBAL_RATIO:
            self.hybrid_escape_until = max(self.hybrid_escape_until, self.current_tick + STUCK_ESCAPE_DURATION)

    def _escape_window_active(self) -> bool:
        if not STUCK_ESCAPE_ENABLED:
            return False
        return self.current_tick <= self.hybrid_escape_until

    def _should_use_escape(self, agent_id: int) -> bool:
        if not STUCK_ESCAPE_ENABLED:
            return False
        if self._escape_window_active():
            return True
        return self.stuck_counter.get(agent_id, 0) >= STUCK_ESCAPE_AGENT_TICKS

    def choose_move_random(self, candidates: List[Tuple[int,int]]) -> int:
        """Random movement baseline: pick random neighbor"""
        return int(self.rng.integers(len(candidates)))

    # ---------- belief accessors (partial observability) ----------
    def _pf(self, r: int, c: int) -> float:
        """Planner-side fire intensity (belief if available, else truth)."""
        bf = getattr(self, "belief_fire", None)
        return float(bf[r, c]) if bf is not None else float(self.grid.fire[r, c])

    def _ps(self, r: int, c: int) -> float:
        """Planner-side smoke density."""
        bs = getattr(self, "belief_smoke", None)
        return float(bs[r, c]) if bs is not None else float(self.grid.smoke[r, c])

    def _hazard_cost(self, r: int, c: int) -> float:
        """Unified planner cost for A*/D* edge entry (uses belief)."""
        if self.grid.types[r, c] == WALL:
            return float("inf")
        if not HAZARD_AWARE_ROUTING_ENABLED:
            return 1.0
        cost = 1.0 + 15.0*self._pf(r, c) + 4.0*self._ps(r, c)
        occ = float(self._occupancy_grid[r, c])
        cost += occ * 2.5
        if self.grid.types[r, c] == EXIT and bool(getattr(self.grid, "exit_compromised", np.zeros_like(self.grid.fire))[r, c]):
            cost += 8.0
        return cost
    
    def choose_move_distance(
        self,
        agent_id: int,
        r: int,
        c: int,
        candidates: List[Tuple[int, int]],
    ) -> int:
        """Greedy move biased toward near exits while respecting hazards and congestion."""

        targets = self._get_exit_targets()
        if not targets:
            return int(self.rng.integers(len(candidates)))

        current_dist = min(manhattan((r, c), e) for e in targets)
        suppression = max(0.0, min(0.98, self.distance_suppression))

        progress_weight = 1.25 - 0.55 * suppression
        detour_penalty = 0.35 + 0.25 * suppression
        hazard_reward = 0.45 + 0.3 * (1.0 - suppression)
        hazard_penalty = 0.65 + 0.25 * suppression
        congestion_weight = 0.12 + 0.08 * (1.0 - suppression)
        mistake_rate = 0.05 + 0.25 * suppression
        backtrack_penalty = 0.5 + 0.4 * suppression

        prev_step = self.prev_pos.get(agent_id)

        current_fire = float(self.grid.fire[r, c])
        current_smoke = float(self.grid.smoke[r, c])
        neighbor_fire = 0.0
        for fr, fc in self.grid.neighbors4(r, c):
            neighbor_fire = max(neighbor_fire, float(self.grid.fire[fr, fc]))
        current_hazard_score = current_fire * 3.0 + current_smoke * 0.8 + neighbor_fire * 1.2

        scored = []
        best_progress = None

        for idx, (nr, nc) in enumerate(candidates):
            dist = min(manhattan((nr, nc), e) for e in targets)
            progress = current_dist - dist

            smoke_val = float(self.grid.smoke[nr, nc])
            fire_val = float(self.grid.fire[nr, nc])
            neighbor_fire_val = 0.0
            for fr, fc in self.grid.neighbors4(nr, nc):
                neighbor_fire_val = max(neighbor_fire_val, float(self.grid.fire[fr, fc]))
            candidate_hazard = fire_val * 3.0 + neighbor_fire_val * 1.6 + smoke_val * 0.75

            r0 = max(0, nr - 1); r1 = min(self.grid.spec.rows, nr + 2)
            c0 = max(0, nc - 1); c1 = min(self.grid.spec.cols, nc + 2)
            congestion = int(self._occupancy_grid[r0:r1, c0:c1].sum())

            score = 0.0
            if progress > 0:
                score += progress * progress_weight
            elif progress == 0:
                score -= 0.05 * suppression
            else:
                score += progress * detour_penalty

            hazard_delta = current_hazard_score - candidate_hazard
            if hazard_delta > 0:
                score += hazard_delta * hazard_reward
            score -= candidate_hazard * hazard_penalty

            if congestion > 1:
                score -= congestion * congestion_weight

            if self.grid.types[nr, nc] == EXIT:
                score += 1.5 + 0.4 * (1.0 - suppression)

            if prev_step is not None and (nr, nc) == prev_step:
                score -= backtrack_penalty

            last_dist = self.last_dist.get(agent_id)
            if last_dist is not None and dist >= last_dist and progress <= 0:
                stuck_factor = min(1.0, self.stuck_counter.get(agent_id, 0) / 8.0)
                score -= stuck_factor * 0.35

            score += float(self.rng.normal(0.0, 0.015))

            scored.append((idx, score, progress, candidate_hazard))
            if best_progress is None or progress > best_progress:
                best_progress = progress

        if not scored:
            return int(self.rng.integers(len(candidates)))

        scored.sort(key=lambda x: x[1], reverse=True)
        top_score = scored[0][1]
        best_indices = [idx for idx, score, _, _ in scored if score >= top_score - 0.04]
        lateral_indices = [idx for idx, _, prog, _ in scored if prog == 0]

        if best_progress is not None and best_progress <= 0 and self.rng.random() < 0.2:
            safest = sorted(scored, key=lambda entry: entry[3])
            return safest[0][0]

        if best_indices and self.rng.random() > mistake_rate:
            return int(self.rng.choice(best_indices))

        if lateral_indices:
            return int(self.rng.choice(lateral_indices))

        return scored[0][0]

    def choose_move_astar(self, r: int, c: int, candidates: List[Tuple[int,int]]) -> int:
        """Hazard-aware A* baseline: Dijkstra over grid costs including fire, smoke, congestion."""
        targets = set(self._get_exit_targets() or self.exit_cells)
        if not targets:
            return int(self.rng.integers(len(candidates)))
        # Cost map via belief-aware unified cost (full mode => true fields)
        def cell_cost(nr, nc):
            return self._hazard_cost(nr, nc)

        # Use Dijkstra (A* with Manhattan heuristic) from current pos
        # Since grid small (<=120), Dijkstra is fine; limit visited to ~R*C
        import heapq as _hq
        dist = {(r,c): 0.0}
        prev = {}
        pq = [(0.0 + min(manhattan((r,c), t) for t in targets), 0.0, (r,c))]
        visited = set()
        found_target = None
        best_target_cost = float('inf')
        # Early exit when reaching any target
        while pq:
            f, g, node = _hq.heappop(pq)
            if node in visited:
                continue
            visited.add(node)
            if node in targets:
                if g < best_target_cost:
                    best_target_cost = g
                    found_target = node
                    break
            nr, nc = node
            for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                ar, ac = nr+dr, nc+dc
                if not (0 <= ar < self.grid.spec.rows and 0 <= ac < self.grid.spec.cols):
                    continue
                if self.grid.types[ar,ac] == WALL:
                    continue
                # Respect traversal threshold unless no alternative (allow if all blocked)
                if HAZARD_AWARE_ROUTING_ENABLED and self.grid.fire[ar,ac] > FIRE_TRAVERSAL_THRESHOLD:
                    # Still allow but with high cost; if all candidates fire-blocked, we already have fallback
                    pass
                ng = g + cell_cost(ar, ac)
                if ng < dist.get((ar,ac), float('inf')):
                    dist[(ar,ac)] = ng
                    prev[(ar,ac)] = node
                    h = min(manhattan((ar,ac), t) for t in targets)
                    _hq.heappush(pq, (ng + h, ng, (ar,ac)))
        # Reconstruct first step toward found_target
        if found_target is None:
            # No path found -> fall back to distance greedy
            return self.choose_move_distance(-1, r, c, candidates)
        # Walk back from target to start to find first step
        cur = found_target
        # If start is target (already at exit), stay?
        if cur == (r,c):
            # Choose among candidates the one closest to target
            best = min(candidates, key=lambda cell: min(manhattan(cell, t) for t in targets))
            return candidates.index(best)
        while prev.get(cur) != (r,c) and prev.get(cur) is not None:
            cur = prev[cur]
            if cur == (r,c):
                break
        # cur is now the first step after start, or found_target if adjacent
        # Find which candidate equals cur (or closest)
        if cur in candidates:
            return candidates.index(cur)
        # If cur not in immediate candidates (due to cost model), pick closest candidate to cur
        best = min(candidates, key=lambda cell: manhattan(cell, cur))
        return candidates.index(best)

    def choose_move_dstar(self, r: int, c: int, candidates: List[Tuple[int,int]], agent_id: int) -> int:
        """D* Lite incremental replanning baseline (belief-aware costs)."""
        if getattr(self, "_DStarLite", None) is None:
            return self.choose_move_astar(r, c, candidates)
        targets = set(self._get_exit_targets() or self.exit_cells)
        if not targets:
            return int(self.rng.integers(len(candidates)))

        planner = self._dstar_planners.get(agent_id)
        if planner is None or planner.goals != frozenset(targets):
            # (re)initialize; cost_of closes over CURRENT belief each call
            planner = self._DStarLite(
                self.grid.spec.rows, self.grid.spec.cols,
                cost_of=lambda nr, nc: self._hazard_cost(nr, nc),
                goals=targets,
            )
            self._dstar_planners[agent_id] = planner
            planner.compute_shortest_path((r, c))

        # Incremental updates only where belief actually changed this tick
        changed = list(getattr(self, "belief_changed", []) or [])
        if changed:
            planner.on_agent_moved((r, c))          # accrue km for agent motion
            planner.notify_costs_changed(changed)
        else:
            planner.on_agent_moved((r, c))
        planner.compute_shortest_path((r, c), max_expansions=600)

        nxt = planner.next_step((r, c))
        if nxt is None:
            return self.choose_move_distance(agent_id, r, c, candidates)
        if nxt in candidates:
            return candidates.index(nxt)
        best = min(candidates, key=lambda cell: manhattan(cell, nxt))
        return candidates.index(best)

    def choose_move_aco(self, r: int, c: int, candidates: List[Tuple[int,int]], 
                        occupied: set, agent_id: int) -> int:
        """ACO-based movement with pheromone and distance scoring"""
        # Epsilon-greedy exploration: small probability to pick a random move
        eps = max(EXPLORATION_MIN, EXPLORATION_EPS * (EXPLORATION_DECAY ** self.current_tick))
        if self.rng.random() < eps:
            return int(self.rng.integers(len(candidates)))
        targets = self._get_exit_targets()
        old_d = min(manhattan((r,c), e) for e in targets) if targets else 1
        old_d = max(1, old_d)

        consider = candidates
        if self.avoid_compromised_exits:
            filtered = []
            for (nr, nc) in consider:
                if self.grid.types[nr, nc] == EXIT:
                    if self.grid.fire[nr, nc] >= FIRE_EXIT_COMPROMISED_THRESHOLD:
                        continue
                    if getattr(self.grid, "exit_compromised", None) is not None and self.grid.exit_compromised[nr, nc]:
                        continue
                filtered.append((nr, nc))
            if filtered:
                consider = filtered
        if not consider:
            consider = candidates

        scores = []
        for (nr,nc) in consider:
            new_d = min(manhattan((nr,nc), e) for e in targets) if targets else 1
            new_d = max(1, new_d)
            # improved movement logic:
            # More aggressive movement toward exits
            if new_d < old_d:
                base = 6.0     # Strong push toward exits
            elif new_d == old_d:
                base = 1.2     # Allow some sideways movement
            else:
                base = 0.3     # Stronger penalty for backward steps

            if self.grid.types[nr,nc] == EXIT:
                compromised = False
                if self.avoid_compromised_exits:
                    compromised = (
                        self.grid.fire[nr, nc] >= FIRE_EXIT_COMPROMISED_THRESHOLD
                        or (getattr(self.grid, "exit_compromised", None) is not None and self.grid.exit_compromised[nr, nc])
                    )
                if compromised:
                    base *= 0.35
                else:
                    base *= 25.0   # Very strong exit attraction
            # Hazard-aware routing ablation: if disabled, ignore smoke/fire
            if not HAZARD_AWARE_ROUTING_ENABLED:
                smoke_penalty = 1.0
                fire_repulsion = 1.0
                nearby_fire_penalty = 1.0
            else:
                # Smoke penalty scales with intensity; reads BELIEF (partial obs)
                smoke_val = self._ps(nr, nc)
                if smoke_val > SMOKE_PENALTY_THRESHOLD:
                    smoke_penalty = max(0.05, 1.0 - SMOKE_SPEED_PENALTY * (0.5 + 0.5 * min(1.0, (smoke_val - SMOKE_PENALTY_THRESHOLD) / 0.5)))
                else:
                    smoke_penalty = 1.0
                fire_repulsion = 1.0 - min(0.7, self._pf(nr, nc) * 2.8)
                nearby_fire_penalty = 1.0
                for fr, fc in self.grid.neighbors4(nr, nc):
                    if self._pf(fr, fc) > 0.01:
                        nearby_fire_penalty *= 0.7
                base *= max(0.1, fire_repulsion * nearby_fire_penalty)
            if self.prev_pos.get(agent_id) == (nr,nc): base *= 0.2

            # Count nearby congestion using occupancy grid (O(1) numpy slice)
            r0 = max(0, nr-1); r1 = min(self.grid.spec.rows, nr+2)
            c0 = max(0, nc-1); c1 = min(self.grid.spec.cols, nc+2)
            cong_count = int(self._occupancy_grid[r0:r1, c0:c1].sum())
            nearby_walls = int(np.count_nonzero(self.grid.types[r0:r1, c0:c1] == WALL))

            # CONGESTION PENALTY (penalize crowded cells, don't reward them)
            # Agents should avoid congested areas, not flock to them
            # Extra penalty when congestion occurs near obstacles (bottlenecks)
            base_penalty = CONGESTION_PENALTY_FACTOR
            if nearby_walls >= 4:  # Narrow corridor or tight space
                base_penalty *= 1.8  # Much stronger penalty near obstacles
            congestion_penalty = 1.0 / (1.0 + base_penalty * cong_count)
            base *= congestion_penalty

            # FIX: GAMMA was stuck-only second congestion penalty (double with congestion_penalty).
            # For research, apply GAMMA continuously but with reduced weight to avoid 96% suppression.
            if cong_count > 0:
                cong_factor = ((1.0 / (1.0 + cong_count)) ** (GAMMA * 0.5))
            else:
                cong_factor = 1.0
            # If stuck, amplify slightly (was only trigger before)
            if (self.last_dist.get(agent_id) is not None) and (new_d > self.last_dist.get(agent_id, 1e9)) and cong_count > 1:
                cong_factor *= 0.7

            pher = self.grid.pheromone[nr,nc]
            # R5: Blend speed and safety pheromone channels (imports at top now).
            if DUAL_PHEROMONE_ENABLED and hasattr(self.grid, 'pheromone_safety'):
                pher_safety = self.grid.pheromone_safety[nr, nc]
                pher = DUAL_PHEROMONE_BLEND * pher + (1.0 - DUAL_PHEROMONE_BLEND) * pher_safety
            # R6: Subtract predictive congestion pheromone (negative signal).
            if hasattr(self.grid, 'congestion_pheromone'):
                cong_pher = float(self.grid.congestion_pheromone[nr, nc])
                pher = max(PHEROMONE_FLOOR, pher - cong_pher * 0.5)
            # Pheromone influence — keep 1.5 boost but document as Q scaling
            pheromone_factor = (pher ** ALPHA) * 1.5
            distance_factor = (1.0 / new_d) ** BETA
            # Single congestion term (cong_factor already includes GAMMA); combine with base penalty multiplicatively
            score = pheromone_factor * distance_factor * base * smoke_penalty * cong_factor * congestion_penalty
            score *= (1.0 + 0.03 * self.rng.random())
            scores.append(max(score, 1e-12))

        tot = float(sum(scores))
        if tot <= 0.0 or not np.isfinite(tot):
            choice = int(self.rng.integers(len(consider)))
        else:
            weights = np.array(scores, dtype=np.float64)
            temp = max(ACO_TEMPERATURE, 1e-3)
            # With temp=0.45, log scaling ~2.2x not 83x, so clip to 50 not 700 to avoid overflow
            log_scores = np.log(weights + 1e-12)
            scaled = log_scores / temp
            scaled -= scaled.max()
            weights = np.exp(np.clip(scaled, -50, 50))
            weights_sum = weights.sum()
            if weights_sum <= 0.0 or not np.isfinite(weights_sum):
                choice = int(self.rng.integers(len(consider)))
            else:
                probs = weights / weights_sum
                # Guard against NaN probs
                if not np.all(np.isfinite(probs)):
                    choice = int(self.rng.integers(len(consider)))
                else:
                    choice = int(self.rng.choice(len(consider), p=probs))
        
        # Map back to original candidates
        return candidates.index(consider[choice])

    def choose_move_escape(
        self,
        agent_id: int,
        r: int,
        c: int,
        candidates: List[Tuple[int, int]],
        occupied: set,
    ) -> int:
        """Hybrid fallback combining pheromone and distance pressure to break local minima."""
        if not candidates:
            return 0

        targets = self._get_exit_targets() or self.exit_cells or [(r, c)]
        current_dist = min(manhattan((r, c), goal) for goal in targets)
        max_pher = max(1e-6, max(float(self.grid.pheromone[nr, nc]) for nr, nc in candidates))

        scored = []
        for idx, (nr, nc) in enumerate(candidates):
            dist = min(manhattan((nr, nc), goal) for goal in targets)
            progress = current_dist - dist
            distance_component = progress if progress > 0 else progress * 0.35

            pher_ratio = float(self.grid.pheromone[nr, nc]) / max_pher
            pher_component = (pher_ratio + 1e-5) ** 1.1

            hazard = float(self.grid.fire[nr, nc]) * 1.6 + float(self.grid.smoke[nr, nc]) * 0.45
            for fr, fc in self.grid.neighbors4(nr, nc):
                hazard = max(hazard, float(self.grid.fire[fr, fc]) * 1.1)

            r0 = max(0, nr - 1); r1 = min(self.grid.spec.rows, nr + 2)
            c0 = max(0, nc - 1); c1 = min(self.grid.spec.cols, nc + 2)
            congestion = int(self._occupancy_grid[r0:r1, c0:c1].sum())

            revisit_penalty = 0.45 if self.prev_pos.get(agent_id) == (nr, nc) else 0.0
            exit_bonus = 4.5 if self.grid.types[nr, nc] == EXIT else 0.0

            score = (
                STUCK_ESCAPE_DISTANCE_WEIGHT * distance_component
                + STUCK_ESCAPE_PHEROMONE_WEIGHT * pher_component
                - STUCK_ESCAPE_HAZARD_WEIGHT * hazard
                - STUCK_ESCAPE_CONGESTION_WEIGHT * congestion
                - revisit_penalty
                + exit_bonus
            )
            score += float(self.rng.normal(0.0, 0.008))
            scored.append((idx, score))

        if not scored:
            return self.choose_move_random(candidates)

        scored.sort(key=lambda item: item[1], reverse=True)
        best_score = scored[0][1]
        wiggle = [idx for idx, score in scored if score >= best_score - 0.05]
        if wiggle:
            return int(self.rng.choice(wiggle))
        return scored[0][0]

    def step(self):
        self.current_tick += 1

        active_pairs = list(zip(self.grid.agent_ids, self.grid.agents))
        if not active_pairs:
            return

        occupied = set(pos for _, pos in active_pairs)
        # Build an occupancy-count grid once per step so per-candidate congestion
        # scans become O(1) numpy reads instead of O(9) set lookups per candidate.
        R, C = self.grid.spec.rows, self.grid.spec.cols
        occupancy_grid = np.zeros((R, C), dtype=np.int16)
        for (ar, ac) in self.grid.agents:
            occupancy_grid[ar, ac] += 1
        self._occupancy_grid = occupancy_grid

        if self.metrics:
            for agent_id, _ in active_pairs:
                if agent_id not in self.metrics.start_tick:
                    self.metrics.start_tick[agent_id] = self.current_tick
                    self.metrics.path_length[agent_id] = 0
                self.last_paths.setdefault(agent_id, [])
                self.prev_pos.setdefault(agent_id, None)
                self.last_dist.setdefault(agent_id, None)
                self.stuck_counter.setdefault(agent_id, 0)

        self._update_escape_window()

        planned_moves: Dict[int, Tuple[Tuple[int, int], Tuple[int, int]]] = {}
        evacuated_this_step = set()
        casualty_this_step = set()

        for agent_id, (r, c) in active_pairs:
            self._record_position(agent_id, (r, c))
            # BUG-07: Fire proximity warning. If fire ignites on the agent's
            # current cell, it dies below. But if fire is adjacent (a neighbor
            # just ignited), give the agent a pre-ignition escape chance by
            # boosting its stuck counter toward the escape threshold.
            if self.grid.fire[r, c] > FIRE_DEATH_THRESHOLD:
                self.casualties += 1
                casualty_this_step.add(agent_id)
                if self.metrics and agent_id not in self.metrics.is_casualty:
                    self.metrics.is_casualty[agent_id] = True
                    self.metrics.end_tick[agent_id] = self.current_tick
                continue

            # Fire proximity warning: if any neighbor has active fire, boost
            # the stuck counter so the agent escapes before fire reaches it.
            if self.movement_mode == MOVEMENT_MODE_ACO:
                fire_nearby = False
                for fr, fc in self.grid.neighbors4(r, c):
                    if self.grid.fire[fr, fc] > FIRE_LOW_THRESHOLD:
                        fire_nearby = True
                        break
                if fire_nearby:
                    cur_stuck = self.stuck_counter.get(agent_id, 0)
                    if cur_stuck < STUCK_ESCAPE_AGENT_TICKS:
                        self.stuck_counter[agent_id] = STUCK_ESCAPE_AGENT_TICKS

            if self.grid.types[r, c] == EXIT:
                self.evacuated += 1
                evacuated_this_step.add(agent_id)
                if self.metrics and agent_id not in self.metrics.is_evacuated:
                    self.metrics.is_evacuated[agent_id] = True
                    self.metrics.end_tick[agent_id] = self.current_tick
                if self.movement_mode == MOVEMENT_MODE_ACO and self.enable_agent_deposits:
                    prev_path = self.last_paths.get(agent_id, [])
                    if prev_path:
                        # BUG-11: Reinforce only the monotonic-progress suffix
                        # (steps that strictly decreased distance to exit),
                        # not backtracking/oscillation. This prevents reinforcing
                        # loops that the agent walked through before escaping.
                        targets = self._get_exit_targets() or self.exit_cells
                        monotonic = []
                        last_d = None
                        for (pr, pc) in prev_path:
                            d = min(manhattan((pr, pc), e) for e in targets)
                            if last_d is None or d < last_d:
                                monotonic.append((pr, pc))
                                last_d = d
                        if monotonic:
                            recent = monotonic[-30:] if len(monotonic) > 30 else monotonic
                            reinforce_success(self.grid.pheromone, recent, success_scale=8.0)
                continue

            candidates = []
            for nr, nc in self.grid.neighbors4(r, c):
                if self.grid.types[nr, nc] == WALL:
                    continue
                candidates.append((nr, nc))

            if not candidates:
                planned_moves[agent_id] = ((r, c), (r, c))
                continue

            safe_candidates = self._filter_candidates(candidates)
            if not safe_candidates:
                # Research fix: symmetric fallback for all modes. Previously DISTANCE could
                # walk through fire at 0.132 while ACO stalled → rigged comparison.
                # Now both allow relaxed up to 1.1*SAFE (0.132) if no safe cell exists.
                relaxed = [cell for cell in candidates if self.grid.fire[cell[0], cell[1]] <= FIRE_SAFE_THRESHOLD * 1.1]
                if relaxed:
                    safe_candidates = relaxed
                else:
                    # No relaxed candidate either — stall (stay) for all modes
                    planned_moves[agent_id] = ((r, c), (r, c))
                    continue

            candidate_pool = safe_candidates
            if self.movement_mode == MOVEMENT_MODE_RANDOM:
                choice = self.choose_move_random(candidate_pool)
            elif self.movement_mode == MOVEMENT_MODE_DISTANCE:
                choice = self.choose_move_distance(agent_id, r, c, candidate_pool)
            elif self.movement_mode == MOVEMENT_MODE_ASTAR:
                choice = self.choose_move_astar(r, c, candidate_pool)
            elif self.movement_mode == MOVEMENT_MODE_DSTAR:
                choice = self.choose_move_dstar(r, c, candidate_pool, agent_id)
            else:  # ACO and STANDARD_ACO share ACO chooser (flags differentiate via config)
                current_dist = self._distance_to_goal((r, c))
                last_known_dist = self.last_dist.get(agent_id)
                # FIX: >= counted lateral detours as stuck (valid maze detour). Only > is truly stuck.
                if last_known_dist is not None and current_dist > last_known_dist:
                    self.stuck_counter[agent_id] = min(255, self.stuck_counter.get(agent_id, 0) + 1)
                    if self.stuck_counter[agent_id] == STUCK_ESCAPE_AGENT_TICKS:
                        self._handle_local_minima(agent_id, (r, c))
                elif last_known_dist is not None and current_dist == last_known_dist:
                    # Lateral move: small increment, not full stuck
                    self.stuck_counter[agent_id] = min(255, self.stuck_counter.get(agent_id, 0) + 0)  # no increment for lateral
                else:
                    self.stuck_counter[agent_id] = 0

                if self._should_use_escape(agent_id):
                    choice = self.choose_move_escape(agent_id, r, c, candidate_pool, occupied)
                    self.stuck_counter[agent_id] = max(0, self.stuck_counter.get(agent_id, 0) - 2)
                elif self.stuck_counter.get(agent_id, 0) >= STUCK_ESCAPE_RANDOM_TICKS:
                    choice = self.choose_move_random(candidate_pool)
                    self.stuck_counter[agent_id] = 0
                else:
                    choice = self.choose_move_aco(r, c, candidate_pool, occupied, agent_id)

            nr, nc = candidate_pool[choice]
            planned_moves[agent_id] = ((r, c), (nr, nc))

        target_map: Dict[Tuple[int, int], List[int]] = {}
        for aid, (_, to_pos) in planned_moves.items():
            target_map.setdefault(to_pos, []).append(aid)

        final_moves: Dict[int, Tuple[Tuple[int, int], Tuple[int, int]]] = {}

        # BUG-12: Chain-movement resolution. In pedestrian cellular automata,
        # agent A can move to cell B if agent B is vacating B (moving elsewhere).
        # This enables chain flows (A->B, B->C, C->D) which are the primary
        # throughput mechanism at bottlenecks. We iteratively resolve moves
        # where the target cell's current occupant has a confirmed move away.
        # Build: current position of each agent that has planned a move.
        planned_from = {aid: planned_moves[aid][0] for aid in planned_moves}
        planned_to = {aid: planned_moves[aid][1] for aid in planned_moves}
        # Which agents have already been confirmed?
        confirmed = set()

        def cell_is_being_vacated(cell):
            """Return the agent_id currently at `cell` that will move away, or None."""
            for aid, from_pos in planned_from.items():
                if aid in confirmed:
                    continue
                if from_pos == cell:
                    return aid
            return None

        # Iterative chain resolution: up to len(agents) passes.
        changed = True
        passes = 0
        max_passes = len(planned_moves) + 1
        while changed and passes < max_passes:
            changed = False
            passes += 1
            for to_pos, aids in list(target_map.items()):
                unresolved = [a for a in aids if a not in confirmed]
                if not unresolved:
                    continue
                # Check if the cell is being vacated by its current occupant.
                occupant = cell_is_being_vacated(to_pos)
                if occupant is not None:
                    # The occupant is moving away, so one contender can claim it.
                    # Prefer the agent whose source is farthest from an exit
                    # (most urgent to move). Tie-break randomly.
                    if len(unresolved) == 1:
                        winner = unresolved[0]
                    else:
                        winner = int(self.rng.integers(len(unresolved)))
                        winner = unresolved[winner]
                    final_moves[winner] = (planned_from[winner], to_pos)
                    confirmed.add(winner)
                    target_map[to_pos] = [a for a in aids if a not in confirmed]
                    changed = True

        # Resolve remaining conflicts (same-target and 2-agent swaps).
        for to_pos, aids in target_map.items():
            unresolved = [a for a in aids if a not in confirmed]
            if not unresolved:
                continue

            if len(unresolved) == 1:
                aid = unresolved[0]
                final_moves[aid] = planned_moves[aid]
                continue

            if len(unresolved) == 2:
                aid1, aid2 = unresolved
                from1 = planned_from[aid1]
                from2 = planned_from[aid2]
                dest2 = planned_to[aid2]
                dest1 = planned_to[aid1]
                if from1 == dest2 and from2 == dest1:
                    final_moves[aid1] = (from1, dest1)
                    final_moves[aid2] = (from2, dest2)
                    continue

            winner_idx = int(self.rng.integers(len(unresolved)))
            for idx, aid in enumerate(unresolved):
                from_pos = planned_from[aid]
                planned_to_pos = planned_to[aid]
                if idx == winner_idx:
                    final_moves[aid] = (from_pos, planned_to_pos)
                else:
                    final_moves[aid] = (from_pos, from_pos)

        survivors = [aid for aid, _ in active_pairs if aid not in evacuated_this_step and aid not in casualty_this_step]
        missing_agents = [aid for aid in survivors if aid not in final_moves]
        if missing_agents:
            logger.warning("Missing agents detected: %s", missing_agents)
            lookup = {aid: pos for aid, pos in active_pairs}
            for missing_id in missing_agents:
                fallback_pos = lookup.get(missing_id, (0, 0))
                final_moves[missing_id] = (fallback_pos, fallback_pos)
                logger.debug("Fallback: keeping agent %s at %s", missing_id, fallback_pos)

        new_agents: List[Tuple[int, int]] = []
        new_agent_ids: List[int] = []
        new_paths: Dict[int, List[Tuple[int, int]]] = {}
        new_prev: Dict[int, Tuple[int, int]] = {}
        new_ldist: Dict[int, int] = {}
        new_stuck: Dict[int, int] = {}
        new_recent: Dict[int, deque[Tuple[int, int]]] = {}
        new_escape_cooldown: Dict[int, int] = {}

        for aid, start_pos in active_pairs:
            if aid in evacuated_this_step or aid in casualty_this_step:
                continue
            from_pos, to_pos = final_moves.get(aid, (start_pos, start_pos))
            new_agents.append(to_pos)
            new_agent_ids.append(aid)

            prev_path = self.last_paths.get(aid, [])
            # FIX: previously added both from_pos and to_pos each tick → duplicates, length 2x, deposit 0.5x
            if to_pos != from_pos:
                # prev_path already ends with from_pos from previous tick, so only add to_pos
                if prev_path and prev_path[-1] == from_pos:
                    updated_path = prev_path + [to_pos]
                else:
                    updated_path = prev_path + [from_pos, to_pos]
            else:
                # Stay: only add if not already ending there
                if prev_path and prev_path[-1] == from_pos:
                    updated_path = prev_path
                else:
                    updated_path = prev_path + [from_pos]
            if len(updated_path) > 80:
                updated_path = updated_path[-80:]
            new_paths[aid] = updated_path

            new_prev[aid] = from_pos
            new_ldist[aid] = self._distance_to_goal(to_pos)
            new_stuck[aid] = self.stuck_counter.get(aid, 0)
            new_recent[aid] = self.recent_positions.get(aid, deque(maxlen=32))
            new_escape_cooldown[aid] = self.escape_cooldown.get(aid, -9999)

            if self.metrics and to_pos != from_pos:
                self.metrics.path_length[aid] = self.metrics.path_length.get(aid, 0) + 1

        self.grid.agents = new_agents
        self.grid.agent_ids = new_agent_ids
        self.last_paths = new_paths
        self.prev_pos = new_prev
        self.last_dist = new_ldist
        self.stuck_counter = new_stuck
        self.recent_positions = new_recent
        self.escape_cooldown = new_escape_cooldown

        if self.metrics:
            for agent_id in list(self.last_paths.keys()):
                if agent_id in self.metrics.is_evacuated or agent_id in self.metrics.is_casualty:
                    self.last_paths.pop(agent_id, None)
                    self.prev_pos.pop(agent_id, None)
                    self.last_dist.pop(agent_id, None)
                    self.stuck_counter.pop(agent_id, None)
                    self.recent_positions.pop(agent_id, None)
                    self.escape_cooldown.pop(agent_id, None)
