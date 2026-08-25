import time
import copy
import warnings
import numpy as np
try:
    from PyQt5.QtCore import QTimer, QObject, QThread, pyqtSignal, pyqtSlot
    _QT_AVAILABLE = True
except Exception:
    QTimer = QObject = QThread = pyqtSignal = pyqtSlot = None  # type: ignore
    _QT_AVAILABLE = False

from .grid import Grid, GridSpec, EXIT
from .agents import AgentEngine
from .pheromones import evaporate, evaporate_dual, punish_area, compute_dynamic_rho, evaporate_region
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
    FIRE_EXIT_COMPROMISED_THRESHOLD, NO_SPAWN_IN_FIRE,
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
        self._exit_fire_blocked_prev = np.zeros_like(self.grid.exit_compromised, dtype=bool)
        
        # Movement and pheromone control
        from config import MOVEMENT_MODE_ASTAR, MOVEMENT_MODE_STANDARD_ACO
        self.movement_mode = movement_mode if movement_mode else MOVEMENT_MODE_DEFAULT
        self.enable_ant_precompute = ENABLE_ANT_PRECOMPUTE
        self.enable_agent_deposits = ENABLE_AGENT_DEPOSITS
        # Only ACO variants use pheromone; distance/random/astar are non-pheromone baselines
        if self.movement_mode not in (MOVEMENT_MODE_ACO, MOVEMENT_MODE_STANDARD_ACO):
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
        # Headless-safe timer: QTimer requires QApplication event loop.
        # In pytest/headless (no QApplication), use dummy timer so step() still works synchronously.
        self.timer = None
        self._timer_connected = False
        if _QT_AVAILABLE:
            try:
                from PyQt5.QtWidgets import QApplication
                app = QApplication.instance()
                if app is not None:
                    self.timer = QTimer()
                    self.timer.setInterval(TICK_MS)
                    self.timer.timeout.connect(self.step)
                    self._timer_connected = True
                else:
                    # No QApplication — headless mode, create QTimer without parent but don't start
                    # Will be lazily created when QApplication appears (in UIMain)
                    self.timer = None
            except Exception:
                self.timer = None
        # Async worker placeholders (set by create_worker_thread)
        self._worker = None
        self._worker_thread = None

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
        """Check if pheromone operations should be active (ACO variants only)"""
        from config import MOVEMENT_MODE_STANDARD_ACO
        return self.movement_mode in (MOVEMENT_MODE_ACO, MOVEMENT_MODE_STANDARD_ACO) and (self.enable_ant_precompute or self.enable_agent_deposits)

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
        # Research reproducibility: use SeedSequence to advance RNG deterministically
        # instead of resetting to same seed every time (which gave identical layouts).
        # If seed is None, keep using entropy; else derive next sub-seed from SeedSequence.
        if self.seed is not None:
            # Advance via SeedSequence spawn to get deterministic but varied sequence
            if not hasattr(self, '_seed_seq'):
                self._seed_seq = np.random.SeedSequence(self.seed)
                self._regen_count = 0
            self._regen_count += 1
            child_seeds = self._seed_seq.spawn(self._regen_count + 1)
            new_seed = child_seeds[-1].generate_state(1)[0]
            self.rng = np.random.default_rng(int(new_seed))
        else:
            self.rng = np.random.default_rng()
        self.grid = Grid(spec, self.rng, fire_params=self.fire_params)
        self._exit_fire_blocked_prev = np.zeros_like(self.grid.exit_compromised, dtype=bool)
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
        self.restore_distance_baseline()

    def _ensure_timer(self):
        """Lazy-create QTimer when QApplication now exists (for headless→UI transition)."""
        if self.timer is not None and self._timer_connected:
            return
        if not _QT_AVAILABLE or QTimer is None:
            return
        try:
            from PyQt5.QtWidgets import QApplication
            if QApplication.instance() is None:
                return
            if self.timer is None:
                self.timer = QTimer()
                self.timer.setInterval(TICK_MS)
                try:
                    self.timer.timeout.connect(self.step)
                    self._timer_connected = True
                except Exception:
                    pass
            elif not self._timer_connected:
                try:
                    self.timer.timeout.connect(self.step)
                    self._timer_connected = True
                except Exception:
                    pass
                self.timer.setInterval(TICK_MS)
        except Exception:
            pass

    def start(self):
        if not self.running:
            self.running = True
            self.metrics.mark_run_start()
            self._run_result_recorded = False
            self._ensure_timer()
            if self.timer is not None:
                try:
                    self.timer.start()
                except Exception:
                    pass
            # If async worker active, signal it
            if self._worker is not None:
                try:
                    self._worker.resume()
                except Exception:
                    pass

    def pause(self):
        if self.running:
            self.running = False
            self.metrics.mark_run_pause()
            if self.timer is not None:
                try:
                    self.timer.stop()
                except Exception:
                    pass
            if self._worker is not None:
                try:
                    self._worker.pause()
                except Exception:
                    pass

    def reset_keep_layout(self):
        spec = self.grid.spec
        restored = self.grid.restore_initial_state()
        if not restored:
            # Fallback to full reseed if snapshot missing — use EMPTY constant not 0
            from .grid import EMPTY as _EMPTY
            self.grid.agents.clear()
            self.grid.agent_ids.clear()
            empties = np.argwhere(self.grid.types == _EMPTY)
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
            # FIX: re-seed distance backbone after reset_pheromone (was flat field)
            try:
                from .seed import seed_pheromone_from_dist
                seed_pheromone_from_dist(self.grid)
            except Exception:
                pass
            self._exit_fire_blocked_prev = np.zeros_like(self.grid.exit_compromised, dtype=bool)
        else:
            self.grid.exit_compromised.fill(False)
            self._exit_fire_blocked_prev = np.zeros_like(self.grid.exit_compromised, dtype=bool)
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
        new_fire_mask = np.zeros_like(self._exit_fire_blocked_prev, dtype=bool)
        for r, c in exits:
            blocked = False
            if self.grid.fire[r, c] > FIRE_EXIT_COMPROMISED_THRESHOLD:
                blocked = True
                new_fire_mask[r, c] = True
            else:
                walls = 0
                total = 0
                for nr, nc in [(r-1, c), (r+1, c), (r, c-1), (r, c+1)]:
                    if 0 <= nr < self.grid.spec.rows and 0 <= nc < self.grid.spec.cols:
                        total += 1
                        if self.grid.types[nr, nc] == 1:
                            walls += 1
                # FIX: corner exits have total=2, never reach walls>=3 → never flagged. Use total as threshold.
                if total > 0 and walls >= min(3, total):
                    blocked = True
            if blocked:
                self.grid.exit_compromised[r, c] = True
            compromised = compromised or blocked
        newly_blocked = np.logical_and(new_fire_mask, ~self._exit_fire_blocked_prev)
        if np.any(newly_blocked):
            for (r, c) in np.argwhere(newly_blocked):
                evaporate_region(self.grid.pheromone, int(r), int(c), radius=3)
        self._exit_fire_blocked_prev = new_fire_mask
        return compromised

    def _update_congestion_map(self):
        g = self.grid
        g.congestion.fill(0.0)
        if len(g.agents) <= 2:
            return
        # Vectorized congestion map: build an occupancy grid, then blur it with a
        # 5x5 sum (radius=2) via four 1D shifts. This replaces the nested Python
        # loops that previously ran per occupied cell.
        occ = np.zeros_like(g.congestion, dtype=np.float32)
        agent_rows = np.array([p[0] for p in g.agents], dtype=np.intp)
        agent_cols = np.array([p[1] for p in g.agents], dtype=np.intp)
        if agent_rows.size == 0:
            return
        np.add.at(occ, (agent_rows, agent_cols), 1.0)
        # Zero-out single-occupancy cells (original only counted count>1).
        occ = np.where(occ > 1.0, occ, 0.0)
        # 5x5 box sum via four directional shifts of magnitude 2 (cumulative).
        blur = occ.copy()
        for dr in range(-2, 3):
            for dc in range(-2, 3):
                if dr == 0 and dc == 0:
                    continue
                from .hazards import _apply_roll_mask
                shifted, mask = _apply_roll_mask(occ, dr, dc)
                blur += np.where(mask, shifted, 0.0)
        g.congestion = blur
        # Punish pheromone at multi-occupied cells (vectorized batch).
        if self._use_pheromone():
            multi_mask = occ > 0.0
            if multi_mask.any():
                coords = np.argwhere(multi_mask)
                for r, c in coords:
                    punish_area(g.pheromone, int(r), int(c), radius=2, factor=0.65)

    def _update_congestion_pheromone(self):
        """R6: Diffuse and decay the predictive congestion pheromone.

        This is a 'negative pheromone' that signals 'avoid this area' to
        agents. It diffuses from congested cells outward and decays, so
        agents can anticipate congestion before walking into it.
        """
        from config import PREDICTIVE_CONGESTION_ENABLED, PREDICTIVE_CONGESTION_DIFFUSION, PREDICTIVE_CONGESTION_DECAY
        if not PREDICTIVE_CONGESTION_ENABLED:
            return
        g = self.grid
        if not hasattr(g, 'congestion_pheromone'):
            return
        # Source: current congestion feeds into the negative pheromone, capped to avoid unbounded growth.
        g.congestion_pheromone += g.congestion * 0.1
        np.clip(g.congestion_pheromone, 0.0, 5.0, out=g.congestion_pheromone)
        # Diffuse (4-neighbor average) — FIX: use actual neighbor count not fixed 4.0 (edges had 2-3 neighbors)
        diffusion = PREDICTIVE_CONGESTION_DIFFUSION
        if diffusion > 0.0:
            from .hazards import _apply_roll_mask
            neighbor_sum = np.zeros_like(g.congestion_pheromone)
            neighbor_count = np.zeros_like(g.congestion_pheromone, dtype=np.int16)
            for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                shifted, mask = _apply_roll_mask(g.congestion_pheromone, dr, dc)
                neighbor_sum += np.where(mask, shifted, 0.0)
                neighbor_count += mask.astype(np.int16)
            avg = np.zeros_like(g.congestion_pheromone)
            valid = neighbor_count > 0
            avg[valid] = neighbor_sum[valid] / neighbor_count[valid].astype(np.float32)
            g.congestion_pheromone += diffusion * (avg - g.congestion_pheromone)
        # Decay
        g.congestion_pheromone *= (1.0 - PREDICTIVE_CONGESTION_DECAY)
        np.maximum(g.congestion_pheromone, 0.0, out=g.congestion_pheromone)

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

        # Research: if simulation already finalized (no remaining agents), make step cheap.
        # Tests often call step() in a fixed loop after evacuation; avoid heavy fire/ants work
        # that previously caused 10x slowdown post-evacuation (micro_steps fire + ants budget).
        actual_crowd = getattr(self.grid, '_actual_crowd', self.grid.spec.crowd)
        if (self.engine.evacuated + self.engine.casualties) >= actual_crowd:
            # Still tick for metrics but skip heavy hazard/pheromone work
            self.tick_counter += 1
            runtime_ms = 0.1
            self.metrics.record_tick(sim=self, tick=self.tick_counter, runtime_ms=runtime_ms, dynamic_rho=None)
            # Auto-pause if not already
            if self.running and not self._run_result_recorded:
                self._finalize_run()
                self.pause()
            return

        tick_start = time.perf_counter()
        dynamic_rho_snapshot = None

        micro_steps = FAST_MODE_STEPS_PER_TICK if len(self.grid.agents) <= FAST_MODE_THRESHOLD else 1
        # FIX: evaporation must stay synchronized with micro-steps.
        # Previously fire advanced 6x per tick but pheromone evaporated once → late-game
        # pheromone persisted 6x too long. Now we step fire/agent AND evaporate per micro-step
        # when micro_steps>1, keeping timescales aligned. tick_counter counts outer ticks
        # (wall clock); metrics handle micro-step normalization via _micro_steps field.
        for ms_idx in range(micro_steps):
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
            self.engine.step()
            # Evaporate per micro-step when in fast mode to keep decay rate constant per agent move
            if self._use_pheromone() and micro_steps > 1:
                if self.rho_dynamic_enabled:
                    dr = compute_dynamic_rho(self, strategy=self.rho_dynamic_mode)
                    # Scale rho to per-micro-step equivalent: 1-(1-rho)^(1/micro_steps) ≈ rho/micro_steps
                    # For small rho we approximate divide by micro_steps for efficiency
                    if isinstance(dr, np.ndarray):
                        dr = dr / float(micro_steps)
                    else:
                        dr = float(dr) / float(micro_steps)
                    evaporate_dual(self.grid, rho=dr)
                else:
                    # Fixed rho per micro-step
                    evaporate_dual(self.grid, rho=RHO / float(micro_steps))
                self._update_congestion_pheromone()
            # Early exit if all agents done mid micro-batch
            actual_crowd = getattr(self.grid, '_actual_crowd', self.grid.spec.crowd)
            if (self.engine.evacuated + self.engine.casualties) >= actual_crowd:
                break

        self.tick_counter += 1
        # Store micro-step count for metrics normalization
        self._last_micro_steps = micro_steps

        # Only evaporate pheromones if using pheromones (ACO mode) — single-step case
        if self._use_pheromone() and micro_steps == 1:
            if self.rho_dynamic_enabled:
                dynamic_rho_snapshot = compute_dynamic_rho(self, strategy=self.rho_dynamic_mode)
                evaporate_dual(self.grid, rho=dynamic_rho_snapshot)
            else:
                dynamic_rho_snapshot = RHO
                evaporate_dual(self.grid)
            # R6: Update predictive congestion pheromone (negative pheromone).
            self._update_congestion_pheromone()
        elif self._use_pheromone() and micro_steps > 1:
            # Already evaporated per micro-step above; snapshot last dynamic rho for metrics
            if self.rho_dynamic_enabled:
                dynamic_rho_snapshot = compute_dynamic_rho(self, strategy=self.rho_dynamic_mode)
            else:
                dynamic_rho_snapshot = RHO
        
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

        # FIX: use actual placed crowd not spec.crowd (spec may be truncated when empties insufficient)
        actual_crowd = getattr(self.grid, '_actual_crowd', self.grid.spec.crowd)
        if (self.engine.evacuated + self.engine.casualties) >= actual_crowd:
            if not self._run_result_recorded:
                self._finalize_run()
            self.pause()

    def _finalize_run(self):
        crowd = max(1, getattr(self.grid, '_actual_crowd', self.grid.spec.crowd))
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

    # ------------------------------------------------------------------
    # Async backend/frontend — worker thread for off-UI simulation
    # ------------------------------------------------------------------
    def create_worker_thread(self, parent=None):
        """Create a background worker thread that steps the simulation off the UI thread.

        Research reproducibility: ant precompute (AntPrecomputer.run_chunk) is CPU-heavy
        (300 iters ~ 60ms) and previously blocked the UI thread via QTimer. This moves
        the hot loop to a QThread so the frontend (GridWidget) only receives tick signals.

        Returns (worker, thread). Caller must keep references and connect
        worker.tickReady / worker.finished to UI slots. Worker steps only when
        Simulation.running is True (pause/resume controls it).
        """
        if not _QT_AVAILABLE or QThread is None:
            warnings.warn("QThread not available — async worker disabled (headless).", RuntimeWarning)
            return None, None
        from PyQt5.QtWidgets import QApplication
        if QApplication.instance() is None:
            warnings.warn("QApplication not running — async worker disabled.", RuntimeWarning)
            return None, None

        worker = _SimulationWorker(self)
        thread = QThread(parent)
        worker.moveToThread(thread)
        thread.started.connect(worker.run_loop)
        # Ensure clean shutdown
        worker.finished.connect(thread.quit)
        worker.finished.connect(worker.deleteLater)
        thread.finished.connect(thread.deleteLater)
        self._worker = worker
        self._worker_thread = thread
        return worker, thread

    def destroy_worker_thread(self):
        if self._worker is not None:
            try:
                self._worker.stop()
            except Exception:
                pass
            self._worker = None
        if self._worker_thread is not None:
            try:
                if self._worker_thread.isRunning():
                    self._worker_thread.quit()
                    self._worker_thread.wait(1000)
            except Exception:
                pass
            self._worker_thread = None


# ----------------------------------------------------------------------
# Internal worker — must be top-level for Qt meta-object
# ----------------------------------------------------------------------
if _QT_AVAILABLE and QObject is not None:
    class _SimulationWorker(QObject):
        tickReady = pyqtSignal(dict)
        precomputeProgress = pyqtSignal(float)
        finished = pyqtSignal()
        error = pyqtSignal(str)

        def __init__(self, sim: "Simulation", parent=None):
            super().__init__(parent)
            self.sim = sim
            self._running_loop = False
            self._paused = False

        @pyqtSlot()
        def run_loop(self):
            self._running_loop = True
            while self._running_loop:
                try:
                    if self.sim.precomputing:
                        self.sim._do_precompute_chunk()
                        self.precomputeProgress.emit(float(self.sim.precompute_progress))
                        # Yield to event loop
                        QThread.msleep(1)
                        continue
                    if self.sim.running and not self._paused:
                        self.sim.step()
                        # Emit lightweight snapshot for UI (avoid copying full grid each tick)
                        try:
                            snap = {
                                "tick": self.sim.tick_counter,
                                "evacuated": self.sim.engine.evacuated,
                                "casualties": self.sim.engine.casualties,
                                "remaining": len(self.sim.grid.agents),
                            }
                            self.tickReady.emit(snap)
                        except Exception:
                            pass
                        # Respect TICK_MS interval minus work time
                        QThread.msleep(max(1, TICK_MS // 2))
                    else:
                        QThread.msleep(10)
                    if not self.sim.running and not self.sim.precomputing:
                        # Idle sleep to avoid busy spin when paused
                        QThread.msleep(20)
                except Exception as e:
                    try:
                        self.error.emit(str(e))
                    except Exception:
                        pass
                    QThread.msleep(50)

        @pyqtSlot()
        def pause(self):
            self._paused = True

        @pyqtSlot()
        def resume(self):
            self._paused = False

        @pyqtSlot()
        def stop(self):
            self._running_loop = False
            self.finished.emit()
else:
    _SimulationWorker = None  # type: ignore
