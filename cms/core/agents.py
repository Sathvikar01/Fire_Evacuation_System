import numpy as np
from collections import deque
from typing import List, Tuple, Dict, Optional
from .grid import Grid, EMPTY, WALL, EXIT
from config import (
    ALPHA, BETA, GAMMA, SMOKE_SPEED_PENALTY, FAST_MODE_THRESHOLD,
    MOVEMENT_MODE_ACO, MOVEMENT_MODE_RANDOM, MOVEMENT_MODE_DISTANCE,
    ENABLE_METRICS_TRACKING, CONGESTION_PENALTY_FACTOR, MAX_OCCUPANCY_ALLOWED,
    FIRE_SAFE_THRESHOLD, FIRE_DEATH_THRESHOLD,
    SMOKE_PENALTY_THRESHOLD, AVOID_COMPROMISED_EXITS, FIRE_EXIT_COMPROMISED_THRESHOLD,
    ACO_TEMPERATURE, FIRE_TRAVERSAL_THRESHOLD, DISTANCE_SUPPRESSION_DEFAULT,
    STUCK_ESCAPE_ENABLED, STUCK_ESCAPE_AGENT_TICKS, STUCK_ESCAPE_RANDOM_TICKS,
    STUCK_ESCAPE_GLOBAL_RATIO, STUCK_ESCAPE_DURATION,
    STUCK_ESCAPE_DISTANCE_WEIGHT, STUCK_ESCAPE_PHEROMONE_WEIGHT,
    STUCK_ESCAPE_HAZARD_WEIGHT, STUCK_ESCAPE_CONGESTION_WEIGHT,
)
from config import EXPLORATION_EPS, EXPLORATION_DECAY, EXPLORATION_MIN
from .pheromones import reinforce_success, suppress_path, evaporate_region

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

        recent = list(self.recent_positions.get(agent_id, []))
        if recent:
            suppress_path(self.grid.pheromone, recent[-16:], factor=0.55)

        evaporate_region(self.grid.pheromone, pos[0], pos[1], radius=1)
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
        occupancy_counts: Dict[Tuple[int, int], int] = {}
        for pos in self.grid.agents:
            occupancy_counts[pos] = occupancy_counts.get(pos, 0) + 1

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

            congestion = 0
            for rr in range(max(0, nr - 1), min(self.grid.spec.rows, nr + 2)):
                for cc in range(max(0, nc - 1), min(self.grid.spec.cols, nc + 2)):
                    congestion += occupancy_counts.get((rr, cc), 0)

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
            smoke_penalty = max(0.05, 1.0 - SMOKE_SPEED_PENALTY) if self.grid.smoke[nr,nc] > SMOKE_PENALTY_THRESHOLD else 1.0
            fire_repulsion = 1.0 - min(0.7, self.grid.fire[nr, nc] * 2.8)
            nearby_fire_penalty = 1.0
            for fr, fc in self.grid.neighbors4(nr, nc):
                if self.grid.fire[fr, fc] > 0.01:
                    nearby_fire_penalty *= 0.4
            base *= max(0.1, fire_repulsion * nearby_fire_penalty)
            if self.prev_pos.get(agent_id) == (nr,nc): base *= 0.2

            # Count nearby congestion using actual occupied positions
            cong_count = 0
            nearby_walls = 0
            for rr in range(max(0, nr-1), min(self.grid.spec.rows, nr+2)):
                for cc in range(max(0, nc-1), min(self.grid.spec.cols, nc+2)):
                    if (rr,cc) in occupied:
                        cong_count += 1
                    if self.grid.types[rr,cc] == WALL:
                        nearby_walls += 1

            # CONGESTION PENALTY (penalize crowded cells, don't reward them)
            # Agents should avoid congested areas, not flock to them
            # Extra penalty when congestion occurs near obstacles (bottlenecks)
            base_penalty = CONGESTION_PENALTY_FACTOR
            if nearby_walls >= 4:  # Narrow corridor or tight space
                base_penalty *= 1.8  # Much stronger penalty near obstacles
            congestion_penalty = 1.0 / (1.0 + base_penalty * cong_count)
            base *= congestion_penalty

            stuck = (self.last_dist.get(agent_id) is not None) and (new_d >= self.last_dist.get(agent_id))
            cong_factor = ((1.0 / (1.0 + cong_count)) ** GAMMA) if (stuck and cong_count > 1) else 1.0

            pher = self.grid.pheromone[nr,nc]
            # Increase pheromone influence for better trail following
            pheromone_factor = (pher ** ALPHA) * 1.5  # Boost pheromone effect
            distance_factor = (1.0 / new_d) ** BETA
            score = pheromone_factor * distance_factor * base * smoke_penalty * cong_factor * congestion_penalty
            score *= (1.0 + 0.03 * self.rng.random())
            scores.append(max(score, 1e-12))

        tot = float(sum(scores))
        if tot <= 0.0:
            choice = int(self.rng.integers(len(consider)))
        else:
            weights = np.array(scores, dtype=np.float64)
            temp = max(ACO_TEMPERATURE, 1e-3)
            log_scores = np.log(weights + 1e-12)
            scaled = log_scores / temp
            scaled -= scaled.max()
            weights = np.exp(np.clip(scaled, -700, 700))
            weights_sum = weights.sum()
            if weights_sum <= 0.0 or not np.isfinite(weights_sum):
                choice = int(self.rng.integers(len(consider)))
            else:
                probs = weights / weights_sum
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

            congestion = 0
            for rr in range(max(0, nr - 1), min(self.grid.spec.rows, nr + 2)):
                for cc in range(max(0, nc - 1), min(self.grid.spec.cols, nc + 2)):
                    if (rr, cc) in occupied:
                        congestion += 1

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
            if self.grid.fire[r, c] > FIRE_DEATH_THRESHOLD:
                self.casualties += 1
                casualty_this_step.add(agent_id)
                if self.metrics and agent_id not in self.metrics.is_casualty:
                    self.metrics.is_casualty[agent_id] = True
                    self.metrics.end_tick[agent_id] = self.current_tick
                continue

            if self.grid.types[r, c] == EXIT:
                self.evacuated += 1
                evacuated_this_step.add(agent_id)
                if self.metrics and agent_id not in self.metrics.is_evacuated:
                    self.metrics.is_evacuated[agent_id] = True
                    self.metrics.end_tick[agent_id] = self.current_tick
                if self.movement_mode == MOVEMENT_MODE_ACO and self.enable_agent_deposits:
                    prev_path = self.last_paths.get(agent_id, [])
                    if prev_path:
                        recent = prev_path[-30:] if len(prev_path) > 30 else prev_path
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
                if self.movement_mode == MOVEMENT_MODE_DISTANCE:
                    relaxed = [cell for cell in candidates if self.grid.fire[cell[0], cell[1]] <= FIRE_SAFE_THRESHOLD * 1.1]
                    safe_candidates = relaxed or candidates
                else:
                    planned_moves[agent_id] = ((r, c), (r, c))
                    continue

            candidate_pool = safe_candidates
            if self.movement_mode == MOVEMENT_MODE_RANDOM:
                choice = self.choose_move_random(candidate_pool)
            elif self.movement_mode == MOVEMENT_MODE_DISTANCE:
                choice = self.choose_move_distance(agent_id, r, c, candidate_pool)
            else:
                current_dist = self._distance_to_goal((r, c))
                last_known_dist = self.last_dist.get(agent_id)
                if last_known_dist is not None and current_dist >= last_known_dist:
                    self.stuck_counter[agent_id] = min(255, self.stuck_counter.get(agent_id, 0) + 1)
                    if self.stuck_counter[agent_id] == STUCK_ESCAPE_AGENT_TICKS:
                        self._handle_local_minima(agent_id, (r, c))
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
        for to_pos, aids in target_map.items():
            if len(aids) == 1:
                aid = aids[0]
                final_moves[aid] = planned_moves[aid]
                continue

            if len(aids) == 2:
                aid1, aid2 = aids
                from1, _ = planned_moves[aid1]
                from2, dest2 = planned_moves[aid2]
                _, dest1 = planned_moves[aid1]
                if from1 == dest2 and from2 == dest1:
                    final_moves[aid1] = (from1, dest1)
                    final_moves[aid2] = (from2, dest2)
                    continue

            winner_idx = int(self.rng.integers(len(aids)))
            for idx, aid in enumerate(aids):
                from_pos, planned_to = planned_moves[aid]
                if idx == winner_idx:
                    final_moves[aid] = (from_pos, planned_to)
                else:
                    final_moves[aid] = (from_pos, from_pos)

        survivors = [aid for aid, _ in active_pairs if aid not in evacuated_this_step and aid not in casualty_this_step]
        missing_agents = [aid for aid in survivors if aid not in final_moves]
        if missing_agents:
            print(f"WARNING: Missing agents detected: {missing_agents}")
            lookup = {aid: pos for aid, pos in active_pairs}
            for missing_id in missing_agents:
                fallback_pos = lookup.get(missing_id, (0, 0))
                final_moves[missing_id] = (fallback_pos, fallback_pos)
                print(f"  Fallback: keeping agent {missing_id} at {fallback_pos}")

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
            if to_pos != from_pos:
                updated_path = prev_path + [from_pos, to_pos]
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

            if self.metrics:
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
