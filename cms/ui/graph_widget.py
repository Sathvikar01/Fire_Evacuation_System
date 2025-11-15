import numpy as np
from PyQt5.QtWidgets import (
	QWidget,
	QVBoxLayout,
	QLabel,
	QGridLayout,
	QFrame,
	QComboBox,
	QHBoxLayout,
)
from PyQt5.QtCore import Qt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT
from matplotlib.figure import Figure


class GraphWidget(QWidget):
	def __init__(self, parent=None):
		super().__init__(parent)
		lay = QVBoxLayout(self)
		lay.setContentsMargins(10, 10, 10, 10)
		lay.setSpacing(10)

		self.titleLbl = QLabel("<b>Performance Metrics</b>")
		self.titleLbl.setAlignment(Qt.AlignCenter)
		lay.addWidget(self.titleLbl)

		self._build_metrics_block(lay)
		self._build_comparison_table(lay)
		self._build_graphs(lay)

		lay.addStretch()

		self._comparison_cells = []
		self._comparison_placeholder = None
		self._scoreboard = {}
		self._history = {}
		self._graph_signature = None

		self._refresh_comparison_plot()
		self._refresh_history_plot()

	def _build_metrics_block(self, layout: QVBoxLayout) -> None:
		self.metricsFrame = QFrame()
		self.metricsFrame.setFrameStyle(QFrame.StyledPanel | QFrame.Raised)
		metricsLayout = QGridLayout(self.metricsFrame)
		metricsLayout.setHorizontalSpacing(12)
		metricsLayout.setVerticalSpacing(6)

		self.labels = {}
		metrics = [
			("evacuated", "Evacuated:"),
			("casualties", "Casualties:"),
			("remaining", "Remaining:"),
			("completion", "Completion Rate:"),
			("avg_evac", "Avg Evac Time:"),
			("median_evac", "Median Evac Time:"),
			("max_evac", "Max Evac Time:"),
			("total_ticks", "Total Ticks:"),
			("runtime_ms", "Runtime / Tick:"),
			("stuck_agents", "Stuck Agents:"),
			("stuck_fraction", "Stuck Fraction:"),
			("dynamic_rho", "Dynamic Rho:"),
			("reroutes", "Reroutes:"),
			("fire_cells", "Burning Cells:"),
			("smoke_cells", "Smoky Cells:"),
			("congestion", "Peak Congestion:"),
		]

		for i, (key, label_text) in enumerate(metrics):
			label = QLabel(label_text)
			label.setStyleSheet("font-weight: bold;")
			value = QLabel("--")
			metricsLayout.addWidget(label, i, 0)
			metricsLayout.addWidget(value, i, 1)
			self.labels[key] = value

		metricsLayout.setColumnStretch(0, 0)
		metricsLayout.setColumnStretch(1, 1)
		layout.addWidget(self.metricsFrame)

	def _build_comparison_table(self, layout: QVBoxLayout) -> None:
		self.winnerLabel = QLabel("Winner: --")
		self.winnerLabel.setAlignment(Qt.AlignCenter)
		self.winnerLabel.setStyleSheet("font-weight: bold; font-size: 12pt;")
		layout.addWidget(self.winnerLabel)

		self.comparisonFrame = QFrame()
		self.comparisonFrame.setFrameStyle(QFrame.StyledPanel | QFrame.Raised)
		self.comparisonLayout = QGridLayout(self.comparisonFrame)
		self.comparisonLayout.setHorizontalSpacing(12)
		self.comparisonLayout.setVerticalSpacing(6)
		self.comparisonHeaders = [
			"Mode",
			"Runs",
			"Avg Completion",
			"Best Completion",
			"Avg Casualty",
			"Avg Ticks",
		]
		for col, header in enumerate(self.comparisonHeaders):
			header_label = QLabel(header)
			header_label.setStyleSheet("font-weight: bold;")
			self.comparisonLayout.addWidget(header_label, 0, col)
			self.comparisonLayout.setColumnStretch(col, 1 if col > 0 else 0)
		layout.addWidget(self.comparisonFrame)

		self.suppressionLabel = QLabel("Distance Suppression: --")
		self.suppressionLabel.setAlignment(Qt.AlignCenter)
		self.suppressionLabel.setStyleSheet("color: #555;")
		layout.addWidget(self.suppressionLabel)

	def _build_graphs(self, layout: QVBoxLayout) -> None:
		graphsRow = QHBoxLayout()
		graphsRow.setContentsMargins(0, 0, 0, 0)
		graphsRow.setSpacing(10)

		# Comparison (bar) graph
		self.metricSelector = QComboBox()
		self.metricSelector.addItem("Composite Score", ("score", 1.0, "Score"))
		self.metricSelector.addItem("Completion Rate (%)", ("completion", 100.0, "Completion %"))
		self.metricSelector.addItem("Casualty Rate (%)", ("casualty", 100.0, "Casualty %"))
		self.metricSelector.addItem("Average Evac Time", ("avg_time", None, "Ticks"))
		self.metricSelector.addItem("Average Ticks", ("ticks", None, "Ticks"))
		self.metricSelector.addItem("Average Path Length", ("path", None, "Tiles"))
		self.metricSelector.addItem("Congestion Ratio (%)", ("congestion", 100.0, "Congestion %"))
		self.metricSelector.currentIndexChanged.connect(self._refresh_comparison_plot)

		self.compareFigure = Figure(figsize=(4.6, 3.4))
		self.compareCanvas = FigureCanvas(self.compareFigure)
		self.compareAxes = self.compareFigure.add_subplot(111)
		self.compareToolbar = NavigationToolbar2QT(self.compareCanvas, self)

		compareFrame = QFrame()
		compareFrame.setFrameStyle(QFrame.StyledPanel | QFrame.Raised)
		compareLayout = QVBoxLayout(compareFrame)
		compareLayout.setContentsMargins(6, 6, 6, 6)
		compareLayout.setSpacing(6)
		selectorRow = QHBoxLayout()
		selectorRow.addWidget(QLabel("Mode Metric:"))
		selectorRow.addWidget(self.metricSelector, 1)
		compareLayout.addLayout(selectorRow)
		compareLayout.addWidget(self.compareCanvas, 1)
		compareLayout.addWidget(self.compareToolbar, 0)
		graphsRow.addWidget(compareFrame, 1)

		# History graph
		self.historyMetricSelector = QComboBox()
		self.historyMetricSelector.addItem("Completion Rate (%)", ("completion_rate", 100.0, "Completion %"))
		self.historyMetricSelector.addItem("Casualty Rate (%)", ("casualty_rate", 100.0, "Casualty %"))
		self.historyMetricSelector.addItem("Average Evac Time", ("average_evac_time", None, "Ticks"))
		self.historyMetricSelector.addItem("Total Ticks", ("total_ticks", None, "Ticks"))
		self.historyMetricSelector.addItem("Path Length", ("avg_path_length_evacuated", None, "Tiles"))
		self.historyMetricSelector.addItem("Congestion Ratio (%)", ("congestion_ratio", 100.0, "Congestion %"))
		self.historyMetricSelector.currentIndexChanged.connect(self._refresh_history_plot)

		self.historyFigure = Figure(figsize=(4.6, 3.4))
		self.historyCanvas = FigureCanvas(self.historyFigure)
		self.historyAxes = self.historyFigure.add_subplot(111)
		self.historyToolbar = NavigationToolbar2QT(self.historyCanvas, self)

		historyFrame = QFrame()
		historyFrame.setFrameStyle(QFrame.StyledPanel | QFrame.Raised)
		historyLayout = QVBoxLayout(historyFrame)
		historyLayout.setContentsMargins(6, 6, 6, 6)
		historyLayout.setSpacing(6)
		historySelectorRow = QHBoxLayout()
		historySelectorRow.addWidget(QLabel("History Metric:"))
		historySelectorRow.addWidget(self.historyMetricSelector, 1)
		historyLayout.addLayout(historySelectorRow)
		historyLayout.addWidget(self.historyCanvas, 1)
		historyLayout.addWidget(self.historyToolbar, 0)
		graphsRow.addWidget(historyFrame, 1)

		layout.addLayout(graphsRow)

	def setText(self, text: str):
		"""Legacy hook used by ui_main to update the title label."""
		self.titleLbl.setText(text)

	def updateMetrics(
		self,
		evacuated=None,
		casualties=None,
		remaining=None,
		avg_evac=None,
		median_evac=None,
		max_evac=None,
		total_ticks=None,
		crowd_size=None,
		runtime_ms=None,
		stuck_agents=None,
		stuck_fraction=None,
		dynamic_rho=None,
		reroutes=None,
		fire_cells=None,
		smoke_cells=None,
		congestion=None,
	):
		if evacuated is not None:
			self.labels["evacuated"].setText(str(evacuated))
		if casualties is not None:
			self.labels["casualties"].setText(str(casualties))
		if remaining is not None:
			self.labels["remaining"].setText(str(remaining))
		if crowd_size is not None and evacuated is not None:
			completion = (evacuated / crowd_size * 100) if crowd_size > 0 else 0
			self.labels["completion"].setText(f"{completion:.1f}%")
		if avg_evac is not None:
			self.labels["avg_evac"].setText(f"{avg_evac:.1f}")
		if median_evac is not None:
			self.labels["median_evac"].setText(f"{median_evac:.1f}")
		if max_evac is not None:
			self.labels["max_evac"].setText(f"{max_evac:.0f}")
		if total_ticks is not None:
			self.labels["total_ticks"].setText(str(total_ticks))
		if runtime_ms is not None:
			self.labels["runtime_ms"].setText(f"{runtime_ms:.2f} ms")
		if stuck_agents is not None:
			self.labels["stuck_agents"].setText(str(stuck_agents))
		if stuck_fraction is not None:
			self.labels["stuck_fraction"].setText(f"{stuck_fraction*100:.1f}%")
		if dynamic_rho is not None:
			self.labels["dynamic_rho"].setText(f"{dynamic_rho:.4f}")
		if reroutes is not None:
			self.labels["reroutes"].setText(str(reroutes))
		if fire_cells is not None:
			self.labels["fire_cells"].setText(str(fire_cells))
		if smoke_cells is not None:
			self.labels["smoke_cells"].setText(str(smoke_cells))
		if congestion is not None:
			self.labels["congestion"].setText(f"{congestion:.2f}")

	def updateComparison(self, rows, winner, suppression, scoreboard=None, history=None):
		scoreboard = scoreboard or {}
		history = history or {}

		for widget in self._comparison_cells:
			widget.deleteLater()
		self._comparison_cells = []

		if self._comparison_placeholder is not None:
			self._comparison_placeholder.deleteLater()
			self._comparison_placeholder = None

		if not rows:
			placeholder = QLabel("Run simulations to populate comparison metrics.")
			placeholder.setStyleSheet("color: #777;")
			placeholder.setAlignment(Qt.AlignCenter)
			self.comparisonLayout.addWidget(placeholder, 1, 0, 1, len(self.comparisonHeaders))
			self._comparison_placeholder = placeholder
		else:
			for row_idx, row in enumerate(rows, start=1):
				values = [
					row.get("mode", "--"),
					str(row.get("runs", 0)),
					f"{(row.get('avg_completion') or 0.0)*100:.1f}%",
					f"{(row.get('best_completion') or 0.0)*100:.1f}%",
					f"{(row.get('avg_casualty') or 0.0)*100:.1f}%",
					f"{row.get('avg_ticks') or 0:.1f}",
				]
				for col, text in enumerate(values):
					lbl = QLabel(text)
					lbl.setAlignment(Qt.AlignCenter)
					self.comparisonLayout.addWidget(lbl, row_idx, col)
					self._comparison_cells.append(lbl)

		self.winnerLabel.setText(f"Winner: {winner}" if winner else "Winner: --")
		if suppression is not None:
			self.suppressionLabel.setText(f"Distance Suppression: {suppression:.2f}")
		else:
			self.suppressionLabel.setText("Distance Suppression: --")

		self._scoreboard = scoreboard
		self._history = history
		signature = (
			tuple(sorted((mode, data[0]) for mode, data in scoreboard.items())),
			tuple(
				sorted(
					(mode, len(payload.get("completion_rate", [])))
					for mode, payload in history.items()
				)
			),
		)
		if signature != self._graph_signature:
			self._graph_signature = signature
			self._refresh_comparison_plot()
			self._refresh_history_plot()

	def _refresh_comparison_plot(self):
		axes = getattr(self, "compareAxes", None)
		if axes is None:
			return
		axes.clear()
		metric = self.metricSelector.currentData()
		if metric is None:
			self.compareCanvas.draw_idle()
			return
		metric_key, scale, ylabel = metric
		if not self._scoreboard:
			axes.text(0.5, 0.5, "Run each mode to view comparisons", ha="center", va="center", color="#777")
			axes.set_xticks([])
			axes.set_yticks([])
			self.compareCanvas.draw_idle()
			return

		modes = sorted(self._scoreboard.keys())
		values = []
		missing = False
		for mode in modes:
			score, detail = self._scoreboard.get(mode, (0.0, {}))
			if metric_key == "score":
				raw_val = score
			else:
				raw_val = detail.get(metric_key)
			if raw_val is None:
				missing = True
				raw_val = 0.0
			if scale is not None:
				raw_val = raw_val * scale
			values.append(raw_val)

		x = np.arange(len(values))
		colors = ["#4c72b0", "#dd8452", "#55a868", "#c44e52"]
		axes.bar(x, values, color=[colors[i % len(colors)] for i in range(len(values))], width=0.55)
		axes.set_xticks(x)
		axes.set_xticklabels([mode.upper() for mode in modes])
		axes.set_ylabel(ylabel)
		axes.set_title("Per-mode Performance")
		axes.grid(axis="y", alpha=0.2)
		if missing:
			axes.text(0.02, 0.92, "*Missing metrics shown as 0", transform=axes.transAxes, fontsize=8, color="#666")
		self.compareFigure.tight_layout(pad=1.1)
		self.compareCanvas.draw_idle()

	def _refresh_history_plot(self):
		axes = getattr(self, "historyAxes", None)
		if axes is None:
			return
		axes.clear()
		metric = self.historyMetricSelector.currentData()
		if metric is None:
			self.historyCanvas.draw_idle()
			return
		metric_key, scale, ylabel = metric

		drawn = False
		for mode in sorted(self._history.keys()):
			series = self._history[mode].get(metric_key, [])
			if not series:
				continue
			y_values = []
			for value in series:
				if value is None:
					y_values.append(np.nan)
				else:
					y_values.append(value * scale if scale is not None else value)
			if not y_values:
				continue
			drawn = True
			x = np.arange(1, len(y_values) + 1)
			axes.plot(x, y_values, marker="o", label=mode.upper())

		if not drawn:
			axes.text(0.5, 0.5, "Run simulations to build history", ha="center", va="center", color="#777")
			axes.set_xticks([])
			axes.set_yticks([])
			self.historyCanvas.draw_idle()
			return

		axes.set_xlabel("Run #")
		axes.set_ylabel(ylabel)
		axes.set_title("Run History (per mode)")
		axes.grid(True, alpha=0.2)
		axes.legend(loc="best")
		self.historyFigure.tight_layout(pad=1.1)
		self.historyCanvas.draw_idle()
