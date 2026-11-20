"""Individual traces on the mean PSTH plot: every trial (or animal) behind its average.

The mean +/- SEM hides how consistent the single responses are. This overlay
draws each row of the displayed PSTH matrix (a trial in Individual scope, an
animal in Group scope) on the same axes, with three choices:

* what is shown: the average only, the average with the individual traces, or
  the individual traces only;
* how traces are told apart: shades of the average color that fade with the
  row order (the first trial is the most opaque, the last the faintest), or
  one color per trace taken from an ordered color scale;
* the line style (full, dashed, dotted, dash-dot) and an optional legend.

Whatever the choice, moving the pointer over a trace highlights it and names
it, so a legend is not needed to identify one trial among many.

Display only: the matrix, the mean, the SEM, the metrics and the exports are
never changed. Dense traces are down-sampled for drawing by pyqtgraph (peak
method), never in the data.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pyqtgraph as pg
from PySide6 import QtCore, QtGui, QtWidgets

from psth_groups import relayout_legend

# =============================================================================
# Configuration
# =============================================================================

#: What the mean PSTH plot shows (code -> menu label).
SHOW_MODES: Dict[str, str] = {
    "average": "Average only",
    "both": "Average and individual traces",
    "traces": "Individual traces only",
}
#: How individual traces are told apart.
COLOR_MODES: Dict[str, str] = {
    "fade": "Shades of the average color (first = darkest)",
    "distinct": "One color per trace",
}
#: Line styles of the individual traces.
LINE_STYLES: Dict[str, str] = {
    "solid": "Full line",
    "dash": "Dashed",
    "dot": "Dotted",
    "dashdot": "Dash-dot",
}
_QT_STYLES = {
    "solid": QtCore.Qt.PenStyle.SolidLine, "dash": QtCore.Qt.PenStyle.DashLine,
    "dot": QtCore.Qt.PenStyle.DotLine, "dashdot": QtCore.Qt.PenStyle.DashDotLine,
}
#: Default view: exactly the previous behavior (average only).
DEFAULT_TRACE_VIEW: Dict[str, object] = {"show": "average", "color": "fade", "style": "solid", "legend": False}

ALPHA_FIRST = 235            # opacity of the first trace in "fade" mode (0-255)
ALPHA_LAST = 55              # opacity of the last trace in "fade" mode
DISTINCT_ALPHA = 215         # opacity of traces in "distinct" mode
TRACE_WIDTH = 1.0            # px
HIGHLIGHT_WIDTH = 2.6        # px, trace under the pointer
MAX_TRACES = 300             # more rows than this are thinned evenly for drawing
LEGEND_ENTRIES_FADE = 6      # legend budget: a gradient needs only a few key entries
LEGEND_ENTRIES_DISTINCT = 12  # legend budget when every trace has its own color
MEAN_WIDTH_WITH_TRACES = 2.8  # px: the average stays readable on top of its traces
HOVER_DISTANCE_PX = 9        # pointer-to-trace distance that still selects a trace
#: Up to this many traces get clearly different hues; more follow an ordered color scale.
QUALITATIVE = ("#0072B2", "#E69F00", "#009E73", "#D55E00", "#CC79A7", "#56B4E9",
               "#F0E442", "#7F3C8D", "#11A579", "#A5AA99")


# =============================================================================
# Pure helpers (testable without a window)
# =============================================================================

def normalize_trace_view(view: Optional[Dict[str, object]]) -> Dict[str, object]:
    """Fill missing keys and replace unknown codes by the defaults."""
    out = dict(DEFAULT_TRACE_VIEW)
    if isinstance(view, dict):
        for key, allowed in (("show", SHOW_MODES), ("color", COLOR_MODES), ("style", LINE_STYLES)):
            if view.get(key) in allowed:
                out[key] = view[key]
        out["legend"] = bool(view.get("legend", out["legend"]))
    return out


def fade_alphas(count: int) -> List[int]:
    """Opacity per row, from ``ALPHA_FIRST`` (first row) down to ``ALPHA_LAST`` (last row)."""
    if count <= 0:
        return []
    return [int(round(v)) for v in np.linspace(ALPHA_FIRST, ALPHA_LAST, count)] if count > 1 else [ALPHA_FIRST]


def distinct_colors(count: int) -> List[Tuple[int, int, int]]:
    """``count`` RGB colors: separate hues for a few traces, an ordered scale for many."""
    if count <= 0:
        return []
    if count <= len(QUALITATIVE):
        return [QtGui.QColor(code).getRgb()[:3] for code in QUALITATIVE[:count]]
    try:
        scale = pg.colormap.get("turbo")
    except Exception:
        scale = pg.colormap.get("viridis")
    # Stay away from the very dark / very bright ends so every trace is visible.
    return [tuple(int(v) for v in scale.map(float(p), mode="byte")[:3]) for p in np.linspace(0.06, 0.94, count)]


def trace_colors(count: int, base_color, mode: str) -> List[Tuple[int, int, int, int]]:
    """RGBA of each row of one segment (a recording's trials or a group's animals)."""
    if mode == "distinct":
        return [(r, g, b, DISTINCT_ALPHA) for r, g, b in distinct_colors(count)]
    red, green, blue = QtGui.QColor(base_color).getRgb()[:3]
    return [(red, green, blue, alpha) for alpha in fade_alphas(count)]


def thinned_indices(count: int, limit: int = MAX_TRACES) -> List[int]:
    """Rows drawn: all of them, or an even selection (first and last kept) above ``limit``."""
    if count <= limit:
        return list(range(count))
    return sorted({int(round(v)) for v in np.linspace(0, count - 1, limit)})


def legend_indices(count: int, limit: int) -> List[int]:
    """Rows named in the legend: all when few, otherwise evenly spread with both ends."""
    if count <= limit:
        return list(range(count))
    return sorted({int(round(v)) for v in np.linspace(0, count - 1, limit)})


def nearest_trace(rows: np.ndarray, tvec: np.ndarray, x: float, y: float, y_per_px: float,
                  max_px: float = HOVER_DISTANCE_PX, subset: Optional[Sequence[int]] = None) -> Optional[int]:
    """Row whose value at time ``x`` is closest to ``y``, if within ``max_px`` pixels.

    With ``subset`` (row indices), only those rows are compared and the result
    is an index into ``subset``. Only one column is read, so this stays fast
    for long windows.
    """
    rows, tvec = np.asarray(rows, float), np.asarray(tvec, float)
    if rows.ndim != 2 or rows.shape[0] == 0 or tvec.size != rows.shape[1] or tvec.size == 0:
        return None
    if subset is not None and len(subset) == 0:
        return None
    if not np.isfinite(x) or not np.isfinite(y) or x < tvec[0] or x > tvec[-1] or y_per_px <= 0:
        return None
    column = int(np.clip(np.searchsorted(tvec, x), 0, tvec.size - 1))
    if column and abs(tvec[column - 1] - x) < abs(tvec[column] - x):
        column -= 1
    values = rows[:, column] if subset is None else rows[np.asarray(subset, int), column]
    distance = np.abs(values - y) / y_per_px
    distance[~np.isfinite(distance)] = np.inf
    best = int(np.argmin(distance))
    return best if distance[best] <= max_px else None


# =============================================================================
# Overlay
# =============================================================================

class TraceOverlay(QtCore.QObject):
    """Draws and manages the individual traces of one mean-PSTH plot."""

    def __init__(self, plot: pg.PlotWidget, legend: pg.LegendItem, parent=None) -> None:
        super().__init__(parent or plot)
        self.plot = plot
        self.legend = legend
        self.curves: List[pg.PlotDataItem] = []      # one per drawn row
        self.row_of_curve: List[int] = []            # matrix row of each curve
        self.colors: List[Tuple[int, int, int, int]] = []
        self.labels: List[str] = []
        self.rows = np.zeros((0, 0))
        self.tvec = np.zeros(0)
        self.style = "solid"
        self._highlighted: Optional[int] = None      # index into self.curves
        self.tip = pg.TextItem(anchor=(0.0, 1.0), fill=pg.mkBrush(20, 24, 34, 220), color="#F2F5FA")
        self.tip.setFont(QtGui.QFont("Segoe UI", 8))
        self.tip.setZValue(50)
        self.tip.hide()
        plot.addItem(self.tip, ignoreBounds=True)

    # ------------------------------------------------------------------ state
    def has_traces(self) -> bool:
        return bool(self.curves)

    def clear(self) -> None:
        """Remove every trace and legend entry."""
        for curve in self.curves:
            self.plot.removeItem(curve)
        self.curves, self.row_of_curve, self.colors, self.labels = [], [], [], []
        self.rows, self.tvec = np.zeros((0, 0)), np.zeros(0)
        self._highlighted = None
        self.tip.hide()
        self.legend.clear()
        self.legend.hide()

    def curve_data(self) -> List[Tuple[np.ndarray, np.ndarray]]:
        """(x, y) of each drawn trace, for legend placement."""
        return [(self.tvec, self.rows[row]) for row in self.row_of_curve]

    # ------------------------------------------------------------------- draw
    def _pen(self, color: Tuple[int, int, int, int], width: float = TRACE_WIDTH) -> QtGui.QPen:
        return pg.mkPen(QtGui.QColor(*color), width=width, style=_QT_STYLES.get(self.style, _QT_STYLES["solid"]))

    def draw(self, tvec: np.ndarray, rows: np.ndarray, labels: Sequence[str],
             segments: Sequence[Tuple[int, int, object, str]], view: Dict[str, object]) -> None:
        """Draw the rows of ``rows`` (one trace each) in the chosen colors and style.

        ``segments``: (start, stop, base color, name) row ranges that share a
        base color: the whole matrix in Individual scope, one per group in the
        group view. The fade (first = darkest) restarts in every segment.
        """
        self.clear()
        view = normalize_trace_view(view)
        rows, tvec = np.asarray(rows, float), np.asarray(tvec, float)
        if rows.ndim != 2 or rows.shape[0] == 0 or tvec.size != rows.shape[1]:
            return
        self.rows, self.tvec, self.style = rows, tvec, str(view["style"])
        count = rows.shape[0]
        self.labels = [str(labels[i]) if i < len(labels) else f"Row {i + 1}" for i in range(count)]
        # Colors: per segment for the fade, across the whole matrix for distinct colors.
        colors: List[Tuple[int, int, int, int]] = [(128, 128, 128, 160)] * count
        if view["color"] == "distinct":
            colors = trace_colors(count, None, "distinct")
        else:
            for start, stop, base, _name in segments:
                for offset, color in enumerate(trace_colors(max(0, stop - start), base, "fade")):
                    colors[start + offset] = color
        self.colors = colors
        for row in thinned_indices(count):
            if not np.any(np.isfinite(rows[row])):
                continue
            curve = pg.PlotDataItem(tvec, rows[row], pen=self._pen(colors[row]), connect="finite")
            curve.setZValue(-3)          # behind the SEM band and the mean
            self.plot.addItem(curve)
            # Set after the item has a view box: pyqtgraph resolves the view
            # while these options are applied and fails on a parentless item.
            curve.setDownsampling(auto=True, method="peak")
            curve.setClipToView(True)
            self.curves.append(curve)
            self.row_of_curve.append(row)
        if view["legend"] and self.curves:
            ranges = list(segments) or [(0, count, None, "")]
            budget = LEGEND_ENTRIES_DISTINCT if view["color"] == "distinct" else LEGEND_ENTRIES_FADE
            # The budget is shared by the segments (groups); each names at
            # least its first and last trace, which is what a fade needs.
            limit = max(2, budget // max(1, len(ranges)))
            for start, stop, _base, name in ranges:
                drawn = [i for i, row in enumerate(self.row_of_curve) if start <= row < stop]
                for position in legend_indices(len(drawn), limit):
                    index = drawn[position]
                    label = self.labels[self.row_of_curve[index]]
                    self.legend.addItem(self.curves[index], f"{name}: {label}" if name else label)
            relayout_legend(self.legend, 1)
            self.legend.show()

    # ------------------------------------------------------------------ hover
    def hover(self, scene_position) -> Optional[str]:
        """Highlight and name the trace under the pointer; returns its label (or None)."""
        if not self.curves:
            return None
        view = self.plot.getPlotItem().getViewBox()
        if not view.sceneBoundingRect().contains(scene_position):
            self._set_highlight(None)
            return None
        point = view.mapSceneToView(scene_position)
        (_x0, _x1), (y0, y1) = view.viewRange()
        height = max(1.0, view.sceneBoundingRect().height())
        row = nearest_trace(self.rows, self.tvec, point.x(), point.y(), (y1 - y0) / height,
                            subset=self.row_of_curve)
        self._set_highlight(row)
        if row is None:
            return None
        label = self.labels[self.row_of_curve[row]]
        self.tip.setText(label)
        self.tip.setPos(point.x(), point.y())
        self.tip.show()
        return label

    def _set_highlight(self, index: Optional[int]) -> None:
        if index == self._highlighted:
            return
        if self._highlighted is not None and self._highlighted < len(self.curves):
            previous = self._highlighted
            self.curves[previous].setPen(self._pen(self.colors[self.row_of_curve[previous]]))
            self.curves[previous].setZValue(-3)
        self._highlighted = index
        if index is None:
            self.tip.hide()
            return
        red, green, blue, _alpha = self.colors[self.row_of_curve[index]]
        self.curves[index].setPen(pg.mkPen(QtGui.QColor(red, green, blue, 255), width=HIGHLIGHT_WIDTH))
        self.curves[index].setZValue(5)


# =============================================================================
# Menu
# =============================================================================

def populate_trace_menu(menu: QtWidgets.QMenu, view: Dict[str, object], change) -> None:
    """Fill ``menu`` with the trace options; ``change(key, value)`` applies one."""
    view = normalize_trace_view(view)
    for title, key, choices in ((None, "show", SHOW_MODES), ("Trace colors", "color", COLOR_MODES),
                                ("Line style", "style", LINE_STYLES)):
        target = menu if title is None else menu.addMenu(title)
        group = QtGui.QActionGroup(target)
        for code, label in choices.items():
            action = target.addAction(label)
            action.setCheckable(True)
            action.setChecked(view[key] == code)
            group.addAction(action)
            action.triggered.connect(lambda _checked=False, k=key, c=code: change(k, c))
        if title is None:
            menu.addSeparator()
    legend = menu.addAction("Trace legend")
    legend.setCheckable(True)
    legend.setChecked(bool(view["legend"]))
    legend.toggled.connect(lambda checked: change("legend", bool(checked)))
    hint = menu.addAction("Tip: point at a trace to highlight and name it")
    hint.setEnabled(False)
