import time
import copy
import numpy as np
from PyQt5.QtCore import QTimer

from .grid import Grid, GridSpec, EXIT
from .agents import AgentEngine
from .pheromones import evaporate, deposit, punish_area, compute_dynamic_rho
from .hazards import step_fire_and_smoke
from .ants import AntPrecomputer
from .metrics import SimulationMetrics
from .session_tracker import SessionPerformanceTracker
from config import (
    TICK_MS,
    PERIODIC_REROUTE_TICKS, PERIODIC_REROUTE_ITERS,
    EMERGENCY_REROUTE_ITERS,
    STAGNANT_TICKS_TRIGGER, FAST_MODE_THRESHOLD, FAST_MODE_STEPS_PER_TICK,
    CONGESTION_UPDATE_TICKS, ACO_BUDGET_PER_TICK,
    MOVEMENT_MODE_DEFAULT, MOVEMENT_MODE_ACO, MOVEMENT_MODE_DISTANCE,
    ENABLE_ANT_PRECOMPUTE, ENABLE_AGENT_DEPOSITS,
    RANDOM_SEED, RHO_DYNAMIC_ENABLED, RHO_DYNAMIC_MODE, STUCK_WINDOW,
    DEPOSIT_ON_EXIT, FIRE_EXIT_COMPROMISED_THRESHOLD, NO_SPAWN_IN_FIRE,
    FIRE_SPAWN_BAND, FIRE_SPAWN_BAND_WIDTH, FIRE_SINGLE_SOURCE, FIRE_SPAWN_COUNT,
    FIRE_SPREAD_BASE, FIRE_SPREAD_RATE_MAX, FIRE_SPREAD_DELAY_TICKS, FIRE_GROWTH_STEP,
    FIRE_FUEL_PER_CELL, FIRE_FUEL_DECAY, FIRE_FLICKER_INTENSITY, FIRE_SAFE_THRESHOLD,
    FIRE_LOW_THRESHOLD, SMOKE_SPREAD_BASE, SMOKE_DIFFUSION_RATE, SMOKE_DECAY_RATE,
    SMOKE_DIRECTIONAL_BIAS, WIND_DIRECTION, WIND_STRENGTH, AVOID_COMPROMISED_EXITS,
    RHO, DISTANCE_SUPPRESSION_DEFAULT,
)

class Simulation:
    def __init__(self, spec: GridSpec, seed: int = None, movement_mode: str = None):
        if seed is None:
            seed = RANDOM_SEED
        self.rng = np.random.default_rng(seed)
        self._initial_rng_state = None

        self.fire_params = {
            "band": FIRE_SPAWN_BAND,
            "band_width": FIRE_SPAWN_BAND_WIDTH,
            "single_source": FIRE_SINGLE_SOURCE,
            "spawn_count": FIRE_SPAWN_COUNT,
            "spread_base": FIRE_SPREAD_BASE,
            "spread_rate_max": FIRE_SPREAD_RATE_MAX,
            "spread_delay": FIRE_SPREAD_DELAY_TICKS,
            "growth_step": FIRE_GROWTH_STEP,
            "fuel_per_cell": FIRE_FUEL_PER_CELL,
            "fuel_decay": FIRE_FUEL_DECAY,
            "flicker": FIRE_FLICKER_INTENSITY,
            "safe_threshold": FIRE_SAFE_THRESHOLD,
            "low_threshold": FIRE_LOW_THRESHOLD,
        }
        self.smoke_params = {
            "base_spread": SMOKE_SPREAD_BASE,
            "diffusion_rate": SMOKE_DIFFUSION_RATE,
            "decay_rate": SMOKE_DECAY_RATE,
            "directional_bias": SMOKE_DIRECTIONAL_BIAS,
        }
        self.wind_params = {
            "direction": WIND_DIRECTION,
            "strength": WIND_STRENGTH,
        }
        self.avoid_compromised_exits = AVOID_COMPROMISED_EXITS

        self.grid = Grid(spec, self.rng, fire_params=self.fire_params)
        self.seed = seed
        
        # Movement and pheromone control
        self.movement_mode = movement_mode if movement_mode else MOVEMENT_MODE_DEFAULT
        self.enable_ant_precompute = ENABLE_ANT_PRECOMPUTE
        self.enable_agent_deposits = ENABLE_AGENT_DEPOSITS
        if self.movement_mode != MOVEMENT_MODE_ACO:
            self.enable_ant_precompute = False
            self.enable_agent_deposits = False
        
        # Dynamic evaporation control
        self.rho_dynamic_enabled = RHO_DYNAMIC_ENABLED
        self.rho_dynamic_mode = RHO_DYNAMIC_MODE
        self.stuck_window = STUCK_WINDOW

        self.ants = AntPrecomputer(self.grid, self.rng)
        # quick refine after BFS seed (only if ants enabled and ACO mode)
        if self._use_pheromone():
            self.ants.run_chunk(iters=60)

        self.engine = AgentEngine(
            self.grid,
            self.rng,
            movement_mode=self.movement_mode,
            enable_agent_deposits=self.enable_agent_deposits,
            avoid_compromised_exits=self.avoid_compromised_exits,
        )
        self.timer = QTimer(); self.timer.setInterval(TICK_MS)
        self.timer.timeout.connect(self.step)

        self.running = False
        self.enable_auto_spread = True

        self.tick_counter = 0
        self.last_evac_total = 0
        self.stagnant_ticks = 0

        self._aco_budget = 0
        self.reroute_count = 0  # Track number of reroutes for metrics
        self.precomputing = False
        self.precompute_iterations_target = 0
        self.precompute_iterations_done = 0
        self.precompute_progress = 0.0
        self._precompute_chunksize = 1

        self.metrics = SimulationMetrics(spec)
        self.session_tracker = SessionPerformanceTracker()
        self.distance_suppression = DISTANCE_SUPPRESSION_DEFAULT
        self._run_result_recorded = False

        self.engine.set_distance_suppression(self.distance_suppression)

        self.store_initial_state()

    def _use_pheromone(self) -> bool:
        """Check if pheromone operations should be active (ACO mode only)"""
        return self.movement_mode == MOVEMENT_MODE_ACO and (self.enable_ant_precompute or self.enable_agent_deposits)

    def store_initial_state(self):
        if hasattr(self.grid, "store_initial_state"):
            self.grid.store_initial_state()
        self._snapshot_rng_state()

    def _snapshot_rng_state(self):
        try:
            self._initial_rng_state = copy.deepcopy(self.rng.bit_generator.state)
        except Exception:
            self._initial_rng_state = None

    def _restore_rng_state(self):
        if self._initial_rng_state is None:
            return
        try:
            self.rng.bit_generator.state = copy.deepcopy(self._initial_rng_state)
        except Exception:
            pass

    def update_fire_settings(self, **params):
        if "band" in params and isinstance(params["band"], str):
            params["band"] = params["band"].lower()
        if "band_width" in params:
            params["band_width"] = max(0.05, min(0.8, float(params["band_width"])))
        if "spawn_count" in params:
            params["spawn_count"] = max(1, int(params["spawn_count"]))
        if "single_source" in params:
            params["single_source"] = bool(params["single_source"])
        self.fire_params.update(params)
        if hasattr(self.grid, "set_fire_params"):
            self.grid.set_fire_params(self.fire_params)

    def update_smoke_settings(self, **params):
        if "diffusion_rate" in params:
            params["diffusion_rate"] = max(0.0, min(0.5, float(params["diffusion_rate"])))
        if "decay_rate" in params:
            params["decay_rate"] = max(0.0, min(0.2, float(params["decay_rate"])))
        self.smoke_params.update(params)

    def update_wind_settings(self, **params):
        if "direction" in params and isinstance(params["direction"], str):
            params["direction"] = params["direction"].lower()
        if "strength" in params:
            params["strength"] = max(0.0, min(1.0, float(params["strength"])))
        self.wind_params.update(params)

    def set_avoid_compromised_exits(self, enabled: bool):
        self.avoid_compromised_exits = bool(enabled)
        if self.engine:
            self.engine.set_avoid_compromised_exits(self.avoid_compromised_exits)

    def restore_distance_baseline(self):
        if self.movement_mode != MOVEMENT_MODE_DISTANCE:
            return
        if self.session_tracker:
            self.session_tracker.reset_suppression()
            baseline = self.session_tracker.distance_suppression()
        else:
            baseline = DISTANCE_SUPPRESSION_DEFAULT
        self.distance_suppression = baseline
        if self.engine:
            self.engine.set_distance_suppression(self.distance_suppression)

    def start_precompute(self, total_seconds: int = 8, iterations_total: int = 400):
        if not self._use_pheromone():
            return  # Skip precompute if not using pheromones
        ticks = max(1, int((total_seconds * 1000) / TICK_MS))
        self.precomputing = True
        self.precompute_iterations_target = iterations_total
        self.precompute_iterations_done = 0
        self._precompute_chunksize = max(1, iterations_total // ticks)
        self.precompute_progress = 0.0

    def _do_precompute_chunk(self):
        if not self._use_pheromone():
            self.precomputing = False
            return
        need = self.precompute_iterations_target - self.precompute_iterations_done
        if need <= 0:
            self.precomputing = False
            self.precompute_progress = 1.0
            return
        chunk = min(self._precompute_chunksize, need)
        self.ants.run_chunk(iters=chunk)
        self.precompute_iterations_done += chunk
        self.precompute_progress = float(self.precompute_iterations_done) / float(self.precompute_iterations_target)
        if self.precompute_iterations_done >= self.precompute_iterations_target:
            self.precomputing = False
            self.precompute_progress = 1.0

    def regenerate(self, spec: GridSpec):
        self.rng = np.random.default_rng()
        self.grid = Grid(spec, self.rng, fire_params=self.fire_params)
        self.ants = AntPrecomputer(self.grid, self.rng)
        if self._use_pheromone():
            self.ants.run_chunk(iters=60)
        self.engine = AgentEngine(
            self.grid,
            self.rng,
            movement_mode=self.movement_mode,
            enable_agent_deposits=self.enable_agent_deposits,
            avoid_compromised_exits=self.avoid_compromised_exits,
        )
        self.engine.set_distance_suppression(self.distance_suppression)
        self.tick_counter = 0
        self.last_evac_total = 0
        self.stagnant_ticks = 0
        self._aco_budget = 0
        self.reroute_count = 0
        self.metrics.reset(spec)
        self._run_result_recorded = False
        self.store_initial_state()
        self.restore_distance_baseline()
        self.restore_distance_baseline()

    def start(self):
        if not self.running:
            self.store_initial_state()
            self.running = True
            self.metrics.mark_run_start()
            self._run_result_recorded = False
            self.timer.start()

    def pause(self):
        if self.running:
            self.running = False
            self.metrics.mark_run_pause()
            self.timer.stop()

    def reset_keep_layout(self):
        spec = self.grid.spec
        restored = self.grid.restore_initial_state()
        if not restored:
            # Fallback to full reseed if snapshot missing
            self.grid.agents.clear()
            self.grid.agent_ids.clear()
            empties = np.argwhere(self.grid.types == 0)
            self.rng.shuffle(empties)
            placed = 0
            for idx in range(len(empties)):
                if placed >= spec.crowd:
                    break
                r,c = map(int, empties[idx])
                if NO_SPAWN_IN_FIRE and self.grid.fire[r, c] > 0.01:
                    continue
                agent_id = self.grid.next_agent_id
                self.grid.next_agent_id += 1
                self.grid.agents.append((r,c))
                self.grid.agent_ids.append(agent_id)
                placed += 1
            self.grid.clear_dynamic()
            self.grid.set_fire_params(self.fire_params)
            self.grid.seed_initial_fire()
            self.grid.exit_compromised.fill(False)
            self.grid.reset_pheromone()
        else:
            self.grid.exit_compromised.fill(False)
            self._restore_rng_state()
        self.ants = AntPrecomputer(self.grid, self.rng)
        if self._use_pheromone() and not restored:
            self.ants.run_chunk(iters=60)
        self.engine = AgentEngine(
            self.grid,
            self.rng,
            movement_mode=self.movement_mode,
            enable_agent_deposits=self.enable_agent_deposits,
            avoid_compromised_exits=self.avoid_compromised_exits,
        )
        self.engine.set_distance_suppression(self.distance_suppression)
        self.tick_counter = 0
        self.last_evac_total = 0
        self.stagnant_ticks = 0
        self._aco_budget = 0
        self.reroute_count = 0
        self.metrics.reset(spec)
        self._run_result_recorded = False
        self.store_initial_state()

    def _is_exit_compromised(self):
        exits = np.argwhere(self.grid.types == EXIT)
        self.grid.exit_compromised.fill(False)
        compromised = False
        for r, c in exits:
            blocked = False
            if self.grid.fire[r, c] > FIRE_EXIT_COMPROMISED_THRESHOLD:
                blocked = True
            else:
                walls = 0
                total = 0
                for nr, nc in [(r-1, c), (r+1, c), (r, c-1), (r, c+1)]:
                    if 0 <= nr < self.grid.spec.rows and 0 <= nc < self.grid.spec.cols:
                        total += 1
                        if self.grid.types[nr, nc] == 1:
                            walls += 1
                if total > 0 and walls >= 3:
                    blocked = True
            if blocked:
                self.grid.exit_compromised[r, c] = True
            compromised = compromised or blocked
        return compromised

    def _update_congestion_map(self):
        g = self.grid
        g.congestion.fill(0.0)
        if len(g.agents) <= 2:
            return
        pos2count = {}
        for (r,c) in g.agents:
            pos2count[(r,c)] = pos2count.get((r,c), 0) + 1
        for (r,c), count in pos2count.items():
            if count <= 1: continue
            for rr in range(max(0, r-2), min(g.spec.rows, r+3)):
                for cc in range(max(0, c-2), min(g.spec.cols, c+3)):
                    g.congestion[rr, cc] += count
            # Only punish pheromone if using pheromones
            # Stronger punishment with larger radius for congested areas
            if self._use_pheromone():
                punish_area(g.pheromone, r, c, radius=2, factor=0.65)

    def _instant_emergency_reroute(self, reason: str = "emergency"):
        if not self._use_pheromone():
            return  # Skip reroute if not using pheromones
        self._aco_budget += EMERGENCY_REROUTE_ITERS * 2
        self.reroute_count += 1
        for _ in range(3):
            if self._aco_budget <= 0: break
            chunk = min(ACO_BUDGET_PER_TICK, self._aco_budget)
            self.ants.run_chunk(iters=chunk)
            self._aco_budget -= chunk
        self.metrics.record_reroute(self.tick_counter, reason)

    def _maybe_emergency_reroute(self):
        need = False
        reason = None
        if self._is_exit_compromised():
            need = True
            reason = "exit_compromised"
        evac = self.engine.evacuated
        if evac <= self.last_evac_total:
            self.stagnant_ticks += 1
        else:
            self.stagnant_ticks = 0
        self.last_evac_total = evac
        if self.stagnant_ticks >= STAGNANT_TICKS_TRIGGER:
            need = True
            self.stagnant_ticks = 0
            reason = "stagnation" if reason is None else f"{reason}+stagnation"
        if need:
            self._instant_emergency_reroute(reason or "emergency")
            self.grid.congestion.fill(0.0)

    def _spend_aco_budget(self):
        if not self._use_pheromone():
            return  # Skip ACO budget if not using pheromones
        if self._aco_budget <= 0: return
        chunk = min(ACO_BUDGET_PER_TICK, self._aco_budget)
        self.ants.run_chunk(iters=chunk)
        self._aco_budget -= chunk

    def step(self):
        # precompute chunk (if active)
        if self.precomputing:
            self._do_precompute_chunk()
            return

        self.tick_counter += 1
        tick_start = time.perf_counter()
        dynamic_rho_snapshot = None

        if self.enable_auto_spread:
            self.grid.fire, self.grid.smoke = step_fire_and_smoke(
                self.grid.types,
                self.grid.fire,
                self.grid.smoke,
                self.rng,
                fire_params=self.fire_params,
                smoke_params=self.smoke_params,
                wind_params=self.wind_params,
            )

        micro_steps = FAST_MODE_STEPS_PER_TICK if len(self.grid.agents) <= FAST_MODE_THRESHOLD else 1
        for _ in range(micro_steps):
            self.engine.step()

        # Only evaporate pheromones if using pheromones (ACO mode)
        if self._use_pheromone():
            if self.rho_dynamic_enabled:
                dynamic_rho_snapshot = compute_dynamic_rho(self, strategy=self.rho_dynamic_mode)
                evaporate(self.grid.pheromone, self.grid.fire, rho=dynamic_rho_snapshot)
            else:
                dynamic_rho_snapshot = RHO
                evaporate(self.grid.pheromone, self.grid.fire)
        
        # Agent pheromone deposits (only if using pheromones)
        # If DEPOSIT_ON_EXIT is enabled, skip continual per-tick deposits
        if self._use_pheromone() and self.enable_agent_deposits and not DEPOSIT_ON_EXIT:
            for agent_id in self.engine.last_paths:
                path = self.engine.last_paths[agent_id]
                if len(path) >= 2:
                    deposit(self.grid.pheromone, path[-10:], scale=5.0)  # Increased from 3.0

        if self.tick_counter % CONGESTION_UPDATE_TICKS == 0 and len(self.grid.agents) > 3:
            self._update_congestion_map()

        # Periodic reroute (only if using pheromones)
        if self._use_pheromone() and self.tick_counter % PERIODIC_REROUTE_TICKS == 0:
            self._aco_budget += PERIODIC_REROUTE_ITERS
            self.reroute_count += 1

        self._maybe_emergency_reroute()
        self._spend_aco_budget()

        runtime_ms = (time.perf_counter() - tick_start) * 1000.0
        self.metrics.record_tick(sim=self, tick=self.tick_counter, runtime_ms=runtime_ms, dynamic_rho=dynamic_rho_snapshot)

        if (self.engine.evacuated + self.engine.casualties) >= self.grid.spec.crowd:
            if not self._run_result_recorded:
                self._finalize_run()
            self.pause()

    def _finalize_run(self):
        crowd = max(1, self.grid.spec.crowd)
        completion_rate = self.engine.evacuated / crowd
        casualty_rate = self.engine.casualties / crowd
        avg_time = None
        if self.engine.metrics:
            avg_time = self.engine.metrics.get_average_evacuation_time()
        self.metrics.capture_final_agent_stats(
            evacuated=self.engine.evacuated,
            casualties=self.engine.casualties,
            crowd=crowd,
            total_ticks=self.tick_counter,
            agent_metrics=self.engine.metrics,
        )
        metric_summary = self.metrics.summary()
        suppression = self.session_tracker.record(
            movement_mode=self.movement_mode,
            completion_rate=completion_rate,
            casualty_rate=casualty_rate,
            average_evac_time=avg_time,
            total_ticks=self.tick_counter,
            avg_path_length_all=metric_summary.get("avg_path_length_all"),
            avg_path_length_evacuated=metric_summary.get("avg_path_length_evacuated"),
            congestion_ratio=metric_summary.get("congestion_ratio"),
        )
        self.distance_suppression = suppression
        self.engine.set_distance_suppression(self.distance_suppression)
        self._run_result_recorded = True

    def reset_session_statistics(self):
        self.session_tracker.reset()
        self.distance_suppression = DISTANCE_SUPPRESSION_DEFAULT
        if self.engine:
            self.engine.set_distance_suppression(self.distance_suppression)
