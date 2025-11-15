from PyQt5.QtWidgets import QWidget
from PyQt5.QtGui import QPainter, QColor, QPen, QFont, QLinearGradient, QRadialGradient
from PyQt5.QtCore import Qt, QRect

from config import (
    COLOR_FREE, COLOR_WALL, COLOR_EXIT, COLOR_EXIT_BLOCKED, COLOR_AGENT,
    COLOR_FIRE, COLOR_SMOKE, COLOR_CONGESTION, COLOR_PHEROMONE, COLOR_GRID_LINE,
    MOVEMENT_MODE_ACO,
    PHEROMONE_VISIBILITY_SCALE, PHEROMONE_VALUE_GAMMA, PHEROMONE_ALPHA_MAX,
)
from core.grid import EMPTY, WALL, EXIT

TOOL_WALL_ADD = "wall_add"
TOOL_WALL_DEL = "wall_del"
TOOL_EXIT_ADD = "exit_add"
TOOL_EXIT_DEL = "exit_del"
TOOL_FIRE_ADD = "fire_add"
TOOL_FIRE_DEL = "fire_del"

class GridWidget(QWidget):
    def __init__(self, sim, parent=None):
        super().__init__(parent)
        self.sim = sim
        self.tool = None
        self.show_pheromone = True
        self.show_congestion = True
        self.highlight_exits = True
        self.setMinimumSize(640, 640)
        self.setMouseTracking(True)

    def setTool(self, tool: str):
        self.tool = tool

    def togglePheromone(self, on: bool):
        self.show_pheromone = on; self.update()

    def toggleCongestion(self, on: bool):
        self.show_congestion = on; self.update()

    def toggleExitHighlight(self, on: bool):
        self.highlight_exits = on; self.update()

    def mousePressEvent(self, e):
        r,c = self._pos_to_cell(e.x(), e.y())
        if r is None: return
        self.apply_tool(r,c); self.update()

    def _pos_to_cell(self, x, y):
        g = self.sim.grid; R, C = g.spec.rows, g.spec.cols
        if R <= 0 or C <= 0: return None, None
        cw = self.width() / C; ch = self.height() / R
        c = int(x // cw); r = int(y // ch)
        if 0 <= r < R and 0 <= c < C: return r, c
        return None, None

    def apply_tool(self, r, c):
        g = self.sim.grid; t = self.tool
        if t == TOOL_WALL_ADD:
            if g.types[r,c] == EXIT: return
            if (r,c) in g.agents: return
            g.types[r,c] = WALL; g.fire[r,c] = 0.0; g.smoke[r,c] = 0.0
        elif t == TOOL_WALL_DEL:
            if g.types[r,c] == WALL: g.types[r,c] = EMPTY
        elif t == TOOL_EXIT_ADD:
            if g.types[r,c] != WALL and (r,c) not in g.agents: g.types[r,c] = EXIT; g.fire[r,c] = 0.0; g.smoke[r,c] = 0.0
        elif t == TOOL_EXIT_DEL:
            if g.types[r,c] == EXIT: g.types[r,c] = EMPTY
        elif t == TOOL_FIRE_ADD:
            if g.types[r,c] not in (WALL, EXIT): g.fire[r,c] = 1.0; g.smoke[r,c] = max(g.smoke[r,c], 0.8)
        elif t == TOOL_FIRE_DEL:
            g.fire[r,c] = 0.0; g.smoke[r,c] = 0.0

    def paintEvent(self, _):
        g = self.sim.grid
        R, C = g.spec.rows, g.spec.cols
        if R <= 0 or C <= 0:
            return

        cw = self.width() / C
        ch = self.height() / R

        qp = QPainter(self)
        qp.setRenderHint(QPainter.Antialiasing, True)

        background = QLinearGradient(0, 0, 0, self.height())
        background.setColorAt(0.0, QColor(248, 248, 252))
        background.setColorAt(1.0, QColor(215, 218, 228))
        qp.fillRect(self.rect(), background)

        highlight_strip = max(1, int(ch * 0.08))
        shadow_strip = max(1, int(ch * 0.10))

        def tile_rect(r, c):
            x = int(round(c * cw))
            y = int(round(r * ch))
            return QRect(x, y, int(cw) + 1, int(ch) + 1)

        def draw_floor(r, c, rgb):
            rect = tile_rect(r, c)
            base = QColor(*rgb)
            gradient = QLinearGradient(rect.topLeft(), rect.bottomRight())
            gradient.setColorAt(0.0, base.lighter(118))
            gradient.setColorAt(0.5, base)
            gradient.setColorAt(1.0, base.darker(120))
            qp.fillRect(rect, gradient)

            # subtle top highlight and bottom shadow to create depth
            qp.fillRect(QRect(rect.left(), rect.top(), rect.width(), highlight_strip), QColor(255, 255, 255, 35))
            qp.fillRect(QRect(rect.left(), rect.bottom() - shadow_strip, rect.width(), shadow_strip), QColor(0, 0, 0, 40))

        def draw_wall(r, c):
            rect = tile_rect(r, c)
            draw_floor(r, c, COLOR_FREE)
            top_h = int(ch * 0.55)
            top_rect = QRect(rect.left(), rect.top() - int(ch * 0.12), rect.width(), top_h)
            if top_rect.top() < 0:
                top_rect.translate(0, -top_rect.top())
            top_grad = QLinearGradient(top_rect.topLeft(), top_rect.bottomRight())
            wall_color = QColor(*COLOR_WALL)
            top_grad.setColorAt(0.0, wall_color.lighter(130))
            top_grad.setColorAt(0.5, wall_color)
            top_grad.setColorAt(1.0, wall_color.darker(150))
            qp.fillRect(top_rect, top_grad)

            front_rect = QRect(rect.left(), rect.top() + top_h - int(ch * 0.12), rect.width(), rect.height() - top_h + int(ch * 0.12))
            if front_rect.height() > 0:
                front_grad = QLinearGradient(front_rect.topLeft(), front_rect.bottomLeft())
                front_grad.setColorAt(0.0, wall_color.darker(120))
                front_grad.setColorAt(1.0, wall_color.darker(170))
                qp.fillRect(front_rect, front_grad)

        def draw_exit(r, c, compromised=False):
            rect = tile_rect(r, c)
            color = QColor(*COLOR_EXIT_BLOCKED) if compromised else QColor(*COLOR_EXIT)
            glow = QColor(color)
            glow.setAlpha(80 if not compromised else 140)
            qp.fillRect(rect.adjusted(-2, -2, 2, 2), glow)
            gradient = QLinearGradient(rect.topLeft(), rect.bottomRight())
            gradient.setColorAt(0.0, color.lighter(140))
            gradient.setColorAt(0.5, color)
            gradient.setColorAt(1.0, color.darker(120))
            qp.fillRect(rect, gradient)

            if compromised and self.highlight_exits:
                pen = QPen(QColor(200, 40, 40))
                pen.setWidth(max(2, int(min(cw, ch) * 0.08)))
                qp.setPen(pen)
                qp.drawRect(rect)
                qp.setPen(Qt.NoPen)

        # Base tiles (floor, walls, exits)
        for r in range(R):
            for c in range(C):
                cell_type = g.types[r, c]
                if cell_type == WALL:
                    draw_wall(r, c)
                elif cell_type == EXIT:
                    compromised = bool(getattr(g, "exit_compromised", None) is not None and g.exit_compromised[r, c])
                    draw_exit(r, c, compromised)
                else:
                    draw_floor(r, c, COLOR_FREE)

        wind_params = getattr(self.sim, "wind_params", {"direction": "none", "strength": 0.0})
        direction = str(wind_params.get("direction", "none")).lower()
        strength = float(wind_params.get("strength", 0.0))
        wind_vec = {
            "north": (-1, 0),
            "south": (1, 0),
            "west": (0, -1),
            "east": (0, 1),
        }.get(direction, (0, 0))
        smoke_offset_x = wind_vec[1] * strength * cw * 0.25
        smoke_offset_y = wind_vec[0] * strength * ch * 0.25

        # Pheromone overlay (soft radial glow)
        if self.show_pheromone and getattr(self.sim, 'movement_mode', None) == MOVEMENT_MODE_ACO:
            mxp = max(0.0001, float(g.pheromone.max()))
            for r in range(R):
                for c in range(C):
                    raw = float(g.pheromone[r, c]) / mxp
                    boosted = min(1.0, raw * PHEROMONE_VISIBILITY_SCALE)
                    if boosted <= 0.02:
                        continue
                    level = boosted ** PHEROMONE_VALUE_GAMMA
                    rect = tile_rect(r, c)
                    radius = max(4.0, min(cw, ch) * 0.6)
                    center = rect.center()
                    radial = QRadialGradient(center, radius)
                    base_color = QColor(*COLOR_PHEROMONE).darker(140)
                    base_color.setAlpha(int(PHEROMONE_ALPHA_MAX * level))
                    radial.setColorAt(0.0, base_color)
                    fade = QColor(base_color)
                    fade.setAlpha(0)
                    radial.setColorAt(1.0, fade)
                    qp.fillRect(rect, radial)

        # Congestion overlay (cyan heatmap under smoke)
        if self.show_congestion and getattr(self.sim, 'movement_mode', None) == MOVEMENT_MODE_ACO:
            mx_cong = max(1.0, float(g.congestion.max()))
            for r in range(R):
                for c in range(C):
                    v = g.congestion[r, c] / mx_cong
                    if v <= 0.05:
                        continue
                    rect = tile_rect(r, c)
                    overlay = QColor(*COLOR_CONGESTION)
                    overlay.setAlpha(int(140 * (v ** 0.7)))
                    qp.fillRect(rect, overlay)

        # Smoke (directional drift)
        for r in range(R):
            for c in range(C):
                s = g.smoke[r, c]
                if s <= 0.05:
                    continue
                rect = tile_rect(r, c)
                smoke = QColor(*COLOR_SMOKE)
                smoke.setAlpha(int(180 * s))
                offset_rect = rect.translated(int(smoke_offset_x), int(smoke_offset_y))
                qp.fillRect(offset_rect, smoke)

        # Fire (glowing columns)
        for r in range(R):
            for c in range(C):
                intensity = g.fire[r, c]
                if intensity <= 0.03:
                    continue
                rect = tile_rect(r, c)
                center = rect.center()
                radius = max(cw, ch) * 0.6
                hot = QRadialGradient(center, radius)
                core = QColor(255, 180, 40)
                core.setAlpha(min(255, int(240 * intensity)))
                rim = QColor(255, 70, 20)
                rim.setAlpha(min(220, int(200 * intensity)))
                hot.setColorAt(0.0, core)
                hot.setColorAt(0.5, rim)
                transparent = QColor(255, 120, 0, 0)
                hot.setColorAt(1.0, transparent)
                qp.fillRect(rect, hot)

        # Agents with drop shadows
        qp.setPen(Qt.NoPen)
        for (r, c) in g.agents:
            rect = tile_rect(r, c)
            shadow = QRect(rect.left() + int(cw * 0.25), rect.top() + int(ch * 0.4), int(cw * 0.5), int(ch * 0.35))
            qp.setBrush(QColor(0, 0, 0, 90))
            qp.drawEllipse(shadow)

            agent_rect = QRect(rect.left() + int(cw * 0.25), rect.top() + int(ch * 0.15), int(cw * 0.5), int(ch * 0.5))
            agent_grad = QLinearGradient(agent_rect.topLeft(), agent_rect.bottomRight())
            core = QColor(*COLOR_AGENT)
            agent_grad.setColorAt(0.0, core.lighter(150))
            agent_grad.setColorAt(1.0, core.darker(130))
            qp.setBrush(agent_grad)
            qp.drawEllipse(agent_rect)

        # Precompute progress overlay
        if getattr(self.sim, "precomputing", False):
            pct = getattr(self.sim, "precompute_progress", 0.0)
            wbar = int(self.width() * 0.6)
            hbar = 20
            x = (self.width() - wbar) // 2
            y = (self.height() - hbar) // 2
            qp.setPen(Qt.NoPen)
            qp.setBrush(QColor(20, 20, 20, 220))
            qp.drawRoundedRect(x, y, wbar, hbar, 8, 8)
            qp.setBrush(QColor(COLOR_PHEROMONE[0], COLOR_PHEROMONE[1], COLOR_PHEROMONE[2], 230))
            qp.drawRoundedRect(x + 3, y + 3, int((wbar - 6) * pct), hbar - 6, 6, 6)
            qp.setPen(QColor(240, 240, 240))
            qp.setFont(QFont("Arial", 10))
            qp.drawText(x, y, wbar, hbar, Qt.AlignCenter, f"Seeding paths… {int(pct * 100)}%")

        # Grid lines for clarity
        pen = QPen(QColor(*COLOR_GRID_LINE))
        pen.setWidth(1)
        qp.setPen(pen)
        for rr in range(R + 1):
            y = int(round(rr * ch))
            qp.drawLine(0, y, int(self.width()), y)
        for cc in range(C + 1):
            x = int(round(cc * cw))
            qp.drawLine(x, 0, x, int(self.height()))

        qp.end()
