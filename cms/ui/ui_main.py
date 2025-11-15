from PyQt5.QtWidgets import QWidget, QVBoxLayout, QMessageBox, QTabWidget, QSplitter
from PyQt5.QtCore import Qt
from ui.grid_widget import GridWidget
from ui.controls_widget import ControlsWidget
from ui.graph_widget import GraphWidget
from core.grid import EXIT
from config import MOVEMENT_MODE_ACO, MOVEMENT_MODE_DISTANCE

class UIMain(QWidget):
    def __init__(self, sim, on_regen):
        super().__init__()
        self.sim = sim
        self.on_regen = on_regen

        self.grid = GridWidget(sim)
        self.controls = ControlsWidget()
        self.controls.setMinimumHeight(260)
        self.graph = GraphWidget()
        self.graph.setMinimumHeight(340)

        self.tabs = QTabWidget()
        self.tabs.setDocumentMode(True)

        sim_tab = QWidget()
        sim_layout = QVBoxLayout(sim_tab)
        sim_layout.setContentsMargins(0, 0, 0, 0)
        sim_split = QSplitter(Qt.Horizontal)
        sim_split.setChildrenCollapsible(False)
        sim_split.addWidget(self.grid)
        sim_split.addWidget(self.controls)
        sim_split.setStretchFactor(0, 3)
        sim_split.setStretchFactor(1, 1)
        sim_split.setSizes([900, 360])
        sim_layout.addWidget(sim_split)

        metrics_tab = QWidget()
        metrics_layout = QVBoxLayout(metrics_tab)
        metrics_layout.setContentsMargins(0, 0, 0, 0)
        metrics_layout.addWidget(self.graph)

        self.tabs.addTab(sim_tab, "Simulation")
        self.tabs.addTab(metrics_tab, "Performance")

        root = QVBoxLayout(self)
        root.setContentsMargins(6, 6, 6, 6)
        root.addWidget(self.tabs)

        # wiring
        self.controls.startClicked.connect(self.sim.start)
        self.controls.pauseClicked.connect(self.sim.pause)
        self.controls.resetClicked.connect(self._reset_keep_layout)
        self.controls.regenClicked.connect(self._regen)
        self.controls.toolChanged.connect(self.grid.setTool)
        self.controls.autoSpreadChanged.connect(self._auto_spread)
        self.controls.movementModeChanged.connect(self._on_movement_mode_changed)
        self.controls.highlightExitsChanged.connect(self.grid.toggleExitHighlight)
        self.controls.compareRequested.connect(self._show_comparison_dialog)
        self.controls.sessionResetRequested.connect(self._reset_session_stats)

        self.sim.timer.timeout.connect(self._on_tick)

    def _auto_spread(self, on):
        self.sim.enable_auto_spread = on
    
    def _on_movement_mode_changed(self, mode):
        self.sim.movement_mode = mode
        self.sim.engine.movement_mode = mode
        is_aco = (mode == MOVEMENT_MODE_ACO)
        self.sim.enable_ant_precompute = is_aco
        self.sim.enable_agent_deposits = is_aco
        if hasattr(self.sim, 'engine') and self.sim.engine:
            self.sim.engine.enable_agent_deposits = self.sim.enable_agent_deposits
            if mode == MOVEMENT_MODE_DISTANCE:
                self.sim.engine.set_distance_suppression(self.sim.distance_suppression)

        self.sim.restore_distance_baseline()

        self.grid.show_pheromone = is_aco
        self.grid.show_congestion = is_aco
        was_running = self.sim.running
        if was_running:
            self.sim.pause()
        self.sim.reset_keep_layout()
        if was_running:
            self.sim.start()
        self.grid.update()
    
    def _reset_keep_layout(self):
        self.sim.reset_keep_layout()
        self._update_title()
        self.grid.update()

    def _regen(self, spec):
        was_running = self.sim.running
        self.sim.pause()
        self.on_regen(spec)
        if was_running:
            self.sim.start()
        self.grid.sim = self.sim
        self._update_title()
        self.grid.update()

    def _show_comparison_dialog(self):
        rows = self.sim.session_tracker.summary_rows()
        winner = self.sim.session_tracker.winner()
        scores = self.sim.session_tracker.multi_metric_scores()
        if not rows:
            QMessageBox.information(self, "Comparison", "Run each movement mode to capture performance data before comparing.")
            return

        lines = []
        for row in rows:
            mode = row.get("mode", "?")
            completion = row.get("avg_completion") or 0.0
            casualties = row.get("avg_casualty") or 0.0
            runs = row.get("runs", 0)
            lines.append(f"{mode}: completion {completion*100:.1f}% (avg), casualties {casualties*100:.1f}% over {runs} runs")

        if winner:
            lines.append("")
            lines.append(f"Winner: {winner}")

        if scores:
            lines.append("")
            lines.append("Multi-metric scoreboard:")
            for mode, (score, detail) in sorted(scores.items(), key=lambda item: item[1][0], reverse=True):
                lines.append(
                    f"  {mode}: score {score:.3f} | completion {detail['completion']:.2f}, casualty {detail['casualty']:.2f}, "
                    f"time {detail['avg_time']:.2f}, ticks {detail['ticks']:.2f}, path {detail['path']:.2f}, congest {detail['congestion']:.2f}"
                )

        QMessageBox.information(self, "Mode Comparison", "\n".join(lines))

    def _reset_session_stats(self):
        self.sim.reset_session_statistics()
        self._update_title()

    def _on_tick(self):
        self._update_title()
        self.grid.update()

    def _update_title(self):
        evac = self.sim.engine.evacuated
        dead = self.sim.engine.casualties
        remain = len(self.sim.grid.agents)
        mode_str = self.sim.movement_mode.upper()
        
        # Calculate metrics
        avg_evac = self.sim.engine.metrics.get_average_evacuation_time() if self.sim.engine.metrics else None
        evac_times = self.sim.engine.metrics.get_evacuation_times() if self.sim.engine.metrics else []
        median_evac = sorted(evac_times)[len(evac_times)//2] if evac_times else None
        max_evac = max(evac_times) if evac_times else None

        latest_snapshot = self.sim.metrics.latest_snapshot()
        runtime_ms = latest_snapshot.get("runtime_ms") if latest_snapshot else None
        stuck_agents = latest_snapshot.get("stuck_agents") if latest_snapshot else None
        stuck_fraction = latest_snapshot.get("stuck_fraction") if latest_snapshot else None
        dynamic_rho = latest_snapshot.get("dynamic_rho_avg") if latest_snapshot else None
        fire_cells = latest_snapshot.get("fire_cells") if latest_snapshot else None
        smoke_cells = latest_snapshot.get("smoke_cells") if latest_snapshot else None
        congestion_peak = latest_snapshot.get("congestion_peak") if latest_snapshot else None
        
        # Update embedded metrics graph panel
        self.graph.updateMetrics(
            evacuated=evac,
            casualties=dead,
            remaining=remain,
            avg_evac=avg_evac,
            median_evac=median_evac,
            max_evac=max_evac,
            total_ticks=self.sim.tick_counter,
            crowd_size=self.sim.grid.spec.crowd,
            runtime_ms=runtime_ms,
            stuck_agents=stuck_agents,
            stuck_fraction=stuck_fraction,
            dynamic_rho=dynamic_rho,
            reroutes=self.sim.reroute_count,
            fire_cells=fire_cells,
            smoke_cells=smoke_cells,
            congestion=congestion_peak,
        )
        
        # Update graph title
        self.graph.setText(f"<b>Performance Metrics [{mode_str}] — Tick {self.sim.tick_counter}</b>")
        tick_label = f"Tick: {self.sim.tick_counter}"
        rho_label = f" | ρ: {dynamic_rho:.4f}" if dynamic_rho is not None else ""
        runtime_label = f" | Tick Runtime: {runtime_ms:.2f} ms" if runtime_ms is not None else ""
        self.setWindowTitle(
            f"CMS-DACO [{mode_str}] | Evacuated: {evac} | Casualties: {dead} | Remaining: {remain} | {tick_label}{rho_label}{runtime_label}"
        )

        exit_mask = getattr(self.sim.grid, "exit_compromised", None)
        compromised = int(exit_mask.sum()) if exit_mask is not None else 0
        total_exits = int((self.sim.grid.types == EXIT).sum())
        self.controls.update_exit_status(compromised, total_exits)

        session_rows = self.sim.session_tracker.summary_rows()
        winner = self.sim.session_tracker.winner()
        suppression = self.sim.session_tracker.distance_suppression()
        scoreboard = self.sim.session_tracker.multi_metric_scores()
        history_payload = self.sim.session_tracker.history_payload()
        self.graph.updateComparison(session_rows, winner, suppression, scoreboard, history_payload)
