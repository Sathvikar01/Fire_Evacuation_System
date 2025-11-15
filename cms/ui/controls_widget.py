from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QPushButton,
    QLabel, QSpinBox, QDoubleSpinBox, QCheckBox, QComboBox,
    QGroupBox, QGridLayout
)
from PyQt5.QtCore import pyqtSignal

from core.grid import GridSpec
from config import (
    GRID_DEFAULT, CROWD_DEFAULT, EXITS_DEFAULT, WALL_DENSITY_DEFAULT,
    MOVEMENT_MODE_ACO, MOVEMENT_MODE_RANDOM, MOVEMENT_MODE_DISTANCE
)

class ControlsWidget(QWidget):
    startClicked = pyqtSignal()
    pauseClicked = pyqtSignal()
    resetClicked = pyqtSignal()
    regenClicked = pyqtSignal(object)   # GridSpec
    toolChanged = pyqtSignal(str)
    autoSpreadChanged = pyqtSignal(bool)
    movementModeChanged = pyqtSignal(str)
    compareRequested = pyqtSignal()
    sessionResetRequested = pyqtSignal()
    highlightExitsChanged = pyqtSignal(bool)

    def __init__(self, parent=None):
        super().__init__(parent)

        self.setMinimumWidth(360)

        self.gridSize = QSpinBox(); self.gridSize.setRange(10, 120); self.gridSize.setValue(GRID_DEFAULT)
        self.crowd = QSpinBox(); self.crowd.setRange(0, 2000); self.crowd.setValue(CROWD_DEFAULT)
        self.exits = QSpinBox(); self.exits.setRange(1, 40); self.exits.setValue(EXITS_DEFAULT)
        self.wallDensity = QDoubleSpinBox(); self.wallDensity.setRange(0.0, 0.6); self.wallDensity.setSingleStep(0.01); self.wallDensity.setValue(WALL_DENSITY_DEFAULT)

        self.btnStart = QPushButton("Start")
        self.btnPause = QPushButton("Pause")
        self.btnReset = QPushButton("Reset Run")
        self.btnRegen = QPushButton("New Layout")
        self.btnCompare = QPushButton("Compare Modes")
        self.btnResetStats = QPushButton("Reset Session Stats")

        for btn in (self.btnStart, self.btnPause, self.btnReset, self.btnCompare, self.btnResetStats, self.btnRegen):
            btn.setMinimumWidth(110)

        self.movementModeBox = QComboBox()
        self.mode_order = [
            ("Dynamic ACO", MOVEMENT_MODE_ACO),
            ("Distance-Greedy", MOVEMENT_MODE_DISTANCE),
            ("Random Walk", MOVEMENT_MODE_RANDOM),
        ]
        for label, _mode in self.mode_order:
            self.movementModeBox.addItem(label)
        self.movementModeBox.setCurrentIndex(0)

        self.modeDescriptions = {
            MOVEMENT_MODE_ACO: "Dynamic ACO uses adaptive pheromone trails and congestion balancing.",
            MOVEMENT_MODE_DISTANCE: "Distance-Greedy heads toward the nearest exit but may wander when suppressed.",
            MOVEMENT_MODE_RANDOM: "Random Walk provides a chaotic baseline for benchmarking.",
        }

        self.toolBox = QComboBox()
        self.toolBox.addItems([
            "Select Tool", "Add Wall", "Remove Wall", "Add Exit", "Remove Exit",
            "Add Fire", "Remove Fire"
        ])

        self.chkAutoSpread = QCheckBox("Auto Fire Spread"); self.chkAutoSpread.setChecked(True)
        self.chkHighlightExits = QCheckBox("Highlight Exits"); self.chkHighlightExits.setChecked(True)

        self.lblModeHint = QLabel(self.modeDescriptions[MOVEMENT_MODE_ACO])
        self.lblModeHint.setWordWrap(True)
        self.lblModeHint.setStyleSheet("color: #444;")

        self.lblExitStatus = QLabel("Exits clear")
        self.lblExitStatus.setStyleSheet("font-weight: bold;")

        runBox = QGroupBox("Simulation Control")
        runLayout = QGridLayout()
        runLayout.setHorizontalSpacing(10)
        runLayout.setVerticalSpacing(8)
        runLayout.addWidget(self.btnStart, 0, 0)
        runLayout.addWidget(self.btnPause, 0, 1)
        runLayout.addWidget(self.btnReset, 0, 2)
        runLayout.addWidget(self.btnCompare, 1, 0)
        runLayout.addWidget(self.btnResetStats, 1, 1)
        runLayout.setColumnStretch(3, 1)
        runBox.setLayout(runLayout)

        layoutBox = QGroupBox("Layout Settings")
        layoutGrid = QGridLayout()
        layoutGrid.setHorizontalSpacing(12)
        layoutGrid.setVerticalSpacing(6)
        layoutGrid.addWidget(QLabel("Grid:"), 0, 0)
        layoutGrid.addWidget(self.gridSize, 0, 1)
        layoutGrid.addWidget(QLabel("Crowd:"), 0, 2)
        layoutGrid.addWidget(self.crowd, 0, 3)
        layoutGrid.addWidget(QLabel("Exits:"), 1, 0)
        layoutGrid.addWidget(self.exits, 1, 1)
        layoutGrid.addWidget(QLabel("Wall%:"), 1, 2)
        layoutGrid.addWidget(self.wallDensity, 1, 3)
        layoutGrid.addWidget(self.btnRegen, 0, 4, 2, 1)
        layoutGrid.setColumnStretch(1, 1)
        layoutGrid.setColumnStretch(3, 1)
        layoutGrid.setColumnStretch(4, 1)
        layoutBox.setLayout(layoutGrid)

        modeBox = QGroupBox("Movement & Tools")
        modeGrid = QGridLayout()
        modeGrid.setHorizontalSpacing(12)
        modeGrid.setVerticalSpacing(8)
        modeGrid.addWidget(QLabel("Movement Mode:"), 0, 0)
        modeGrid.addWidget(self.movementModeBox, 0, 1, 1, 2)
        modeGrid.addWidget(QLabel("Edit Tool:"), 1, 0)
        modeGrid.addWidget(self.toolBox, 1, 1, 1, 2)
        modeGrid.addWidget(self.chkAutoSpread, 2, 0)
        modeGrid.addWidget(self.chkHighlightExits, 2, 1)
        modeGrid.setColumnStretch(1, 1)
        modeGrid.setColumnStretch(2, 1)
        modeBox.setLayout(modeGrid)

        infoBox = QGroupBox("Status & Guidance")
        infoLayout = QVBoxLayout()
        infoLayout.setSpacing(6)
        infoLayout.addWidget(self.lblModeHint)
        infoLayout.addWidget(self.lblExitStatus)
        infoLayout.addStretch(1)
        infoBox.setLayout(infoLayout)

        root = QVBoxLayout(self)
        root.setContentsMargins(10, 10, 10, 10)
        root.setSpacing(12)
        root.addWidget(runBox)
        root.addWidget(layoutBox)
        root.addWidget(modeBox)
        root.addWidget(infoBox)
        root.addStretch(1)

        self.btnStart.clicked.connect(self.startClicked.emit)
        self.btnPause.clicked.connect(self.pauseClicked.emit)
        self.btnReset.clicked.connect(self.resetClicked.emit)
        self.btnRegen.clicked.connect(self._emit_regen)
        self.btnCompare.clicked.connect(self.compareRequested.emit)
        self.btnResetStats.clicked.connect(self.sessionResetRequested.emit)
        self.toolBox.currentIndexChanged.connect(self._emit_tool)
        self.chkAutoSpread.toggled.connect(self.autoSpreadChanged.emit)
        self.chkHighlightExits.toggled.connect(self.highlightExitsChanged.emit)
        self.movementModeBox.currentIndexChanged.connect(self._emit_movement_mode)

    def _emit_regen(self):
        n = self.gridSize.value()
        spec = GridSpec(
            rows=n, cols=n,
            crowd=self.crowd.value(),
            exits=self.exits.value(),
            wall_density=float(self.wallDensity.value())
        )
        self.regenClicked.emit(spec)

    def _emit_tool(self, idx):
        mapping = {
            1: "wall_add", 2: "wall_del", 3: "exit_add", 4: "exit_del",
            5: "fire_add", 6: "fire_del"
        }
        self.toolChanged.emit(mapping.get(idx, ""))
    
    def _emit_movement_mode(self, idx):
        if idx < 0 or idx >= len(self.mode_order):
            return
        _, mode_key = self.mode_order[idx]
        self.lblModeHint.setText(self.modeDescriptions.get(mode_key, ""))
        self.movementModeChanged.emit(mode_key)
    
    def update_exit_status(self, compromised: int, total: int):
        if total <= 0:
            self.lblExitStatus.setText("No exits configured")
        elif compromised <= 0:
            self.lblExitStatus.setText("Exits clear")
        else:
            self.lblExitStatus.setText(f"Blocked exits: {compromised}/{total}")
