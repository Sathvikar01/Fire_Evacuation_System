"""Compact, correct D* Lite (Koenig & Likhachev 2002) for grid worlds.

Optimized-first incremental search: repairs shortest paths to the goal as
edge costs change instead of replanning from scratch. Supports agent motion
via km and batch cost-change notifications.

Edge-cost convention: traversing INTO cell v costs ``cost_of(v)``; walls are
inf. The goal set holds virtual rhs=0 terminals.
"""
import heapq
import math
from typing import Callable, Dict, List, Set, Tuple

Cell = Tuple[int, int]
INF = float("inf")


class DStarLite:
    def __init__(self, rows: int, cols: int,
                 cost_of: Callable[[int, int], float],
                 goals: Set[Cell]):
        self.R, self.C = rows, cols
        self.cost_of = cost_of
        self.goals = frozenset(goals)
        self.km = 0.0
        self.last: Cell = next(iter(goals)) if goals else (0, 0)
        self.g: Dict[Cell, float] = {}
        self.rhs: Dict[Cell, float] = {}
        self._keys: Dict[Cell, Tuple[float, float]] = {}
        self._heap: List[Tuple[float, float, int, Cell]] = []
        self._tick = 0
        for gl in self.goals:
            self.rhs[gl] = 0.0
            self._update_queue(gl)

    # ---------- helpers ----------
    @staticmethod
    def _h(a: Cell, b: Cell) -> float:
        return abs(a[0]-b[0]) + abs(a[1]-b[1])

    def _pred(self, u: Cell) -> List[Cell]:
        r, c = u
        out = []
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nr, nc = r+dr, c+dc
            if 0 <= nr < self.R and 0 <= nc < self.C:
                out.append((nr, nc))
        return out

    def _succ(self, u: Cell) -> List[Cell]:
        return self._pred(u)

    def _calc_key(self, s: Cell, start: Cell) -> Tuple[float, float]:
        k2 = min(self.g.get(s, INF), self.rhs.get(s, INF))
        return (k2 + self._h(start, s) + self.km, k2)

    def _update_queue(self, s: Cell):
        """Insert/refresh/remove s in the queue to match (g,rhs) consistency."""
        g, rh = self.g.get(s, INF), self.rhs.get(s, INF)
        if g != rh:
            self._keys[s] = self._calc_key(s, self.last)
        else:
            self._keys.pop(s, None)

    def _rebuild_heap(self):
        self._tick += 1
        self._heap = [(k[0], k[1], self._tick, s)
                      for s, k in self._keys.items()]
        heapq.heapify(self._heap)

    def _top_key(self) -> Tuple[float, float]:
        self._rebuild_heap()
        return self._heap[0][:2] if self._heap else (INF, INF)

    # ---------- core ----------
    def _update_vertex(self, u: Cell):
        if u in self.goals:
            self._update_queue(u)
            return
        best = INF
        cu = self.cost_of(*u)
        if cu < INF:
            for s in self._succ(u):
                cand = self.cost_of(*s) + self.g.get(s, INF)
                if cand < best:
                    best = cand
        self.rhs[u] = best
        self._update_queue(u)

    def compute_shortest_path(self, start: Cell, max_expansions: int = 3000):
        expansions = 0
        while True:
            top = self._top_key()
            k_start = self._calc_key(start, start)
            if not (top < k_start or self.rhs.get(start, INF) > self.g.get(start, INF)):
                break
            if top == (INF, INF) or expansions >= max_expansions:
                break
            self._rebuild_heap()
            _, _, _, u = heapq.heappop(self._heap)
            self._keys.pop(u, None)
            expansions += 1
            k_old = top
            k_new = self._calc_key(u, start)
            g, rh = self.g.get(u, INF), self.rhs.get(u, INF)
            if k_old < k_new:
                self._update_queue(u)          # reinsert with new key
            elif g > rh:                        # overconsistent -> make consistent
                self.g[u] = rh
                for p in self._pred(u):
                    self._update_vertex(p)
            else:                               # underconsistent -> propagate
                self.g[u] = INF
                self._update_vertex(u)
                for p in self._pred(u):
                    self._update_vertex(p)

    def notify_costs_changed(self, cells):
        """Batch notification that cost_of() changed at these cells."""
        touched: Set[Cell] = set()
        for v in cells:
            if v in self.goals:
                continue
            touched.add(v)
            touched.update(self._pred(v))
        for u in touched:
            self._update_vertex(u)

    def on_agent_moved(self, new_pos: Cell):
        self.km += self._h(self.last, new_pos)
        self.last = new_pos

    def reset_goal(self, goals: Set[Cell]):
        """Full reset when the target set changes (e.g., exit compromised)."""
        self.goals = frozenset(goals)
        self.g.clear(); self.rhs.clear(); self._keys.clear()
        self._heap.clear()
        for gl in self.goals:
            self.rhs[gl] = 0.0
        self.last = next(iter(goals)) if goals else (0, 0)
        self.km = 0.0

    def next_step(self, start: Cell) -> Cell | None:
        if start in self.goals:
            return None
        best, best_v = None, INF
        for s in self._succ(start):
            cs = self.cost_of(*s)
            if cs >= INF:
                continue
            v = cs + self.g.get(s, INF)
            if v < best_v:
                best_v, best = v, s
        return best
