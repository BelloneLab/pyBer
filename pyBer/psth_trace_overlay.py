"""Individual traces on the mean PSTH plot: every trial (or animal) behind its average.

The mean +/- SEM hides how consistent the single responses are. This overlay
draws each row of the displayed PSTH matrix (a trial in Individual scope, an
animal in Group scope) on the same axes, with three choices:

* what is shown: the average only, the average with the individual traces, or
  the individual traces only;
* how traces are told apart: rank by row order or response amplitude, using
  shades of the average color or a multicolor palette;
* opacity, line style (full, dashed, dotted, dash-dot) and an optional legend.

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
    "fade": "Shades of the average color",
    "distinct": "One color per trace",
}
RANK_MODES = {"order": "Trial / row order", "amplitude": "Response amplitude"}
OPACITY_PRESETS = {"auto": "Automatic (trace density)", 25: "25%", 50: "50%", 75: "75%", 100: "100%"}
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
DEFAULT_TRACE_VIEW: Dict[str, object] = {
    "show": "average", "color": "fade", "rank": "order", "opacity": "auto", "style": "solid", "legend": False,
}

# Perceived brightness in sRGB, from the least to the most prominent rank.
# Both ends retain contrast against the plot even for very pale or dark base colors.
DARK_PLOT_RAMP = (0.36, 0.90)
LIGHT_PLOT_RAMP = (0.65, 0.13)
LUMA_WEIGHTS = (0.2126, 0.7152, 0.0722)
TRACE_WIDTH = 1.0            # px
HIGHLIGHT_WIDTH = 2.6        # px, trace under the pointer
MAX_TRACES = 300             # more rows than this are thinned evenly for drawing
LEGEND_ENTRIES_FADE = 6      # legend budget: a gradient needs only a few key entries
LEGEND_ENTRIES_DISTINCT = 12  # legend budget when every trace has its own color
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
        for key, allowed in (("show", SHOW_MODES), ("color", COLOR_MODES), ("rank", RANK_MODES), ("style", LINE_STYLES)):
            if isinstance(view.get(key), str) and view[key] in allowed:
                out[key] = view[key]
        opacity = view.get("opacity")
        if type(opacity) is int and 1 <= opacity <= 100:
            out["opacity"] = opacity
        out["legend"] = bool(view.get("legend", out["legend"]))
    return out


def response_amplitudes(rows, tvec, pre_window, post_window) -> np.ndarray:
    """Largest absolute Post excursion from each row's finite Pre mean, in signal units."""
    rows, tvec = np.asarray(rows, float), np.asarray(tvec, float)
    missing = np.full(len(rows), np.nan)
    if pre_window is None or post_window is None:
        return missing
    pre = rows[:, (tvec >= pre_window[0]) & (tvec <= pre_window[1])]
    post = rows[:, (tvec >= post_window[0]) & (tvec <= post_window[1])]
    if not pre.shape[1] or not post.shape[1]:
        return missing
    # An unobserved window has no rank. Never turn a missing trial into a zero response.
    finite = np.isfinite(pre)
    counts = finite.sum(axis=1)
    baseline = np.divide(np.sum(pre, axis=1, where=finite), counts, out=missing, where=counts > 0)
    excursion = np.abs(post - baseline[:, None])
    return np.fmax.reduce(np.where(np.isfinite(excursion), excursion, np.nan), axis=1)


def _ranks(values: np.ndarray) -> np.ndarray:
    """Space observed amplitude levels evenly. Ties share a shade and NaNs stay unranked."""
    ranks = np.full(values.shape, np.nan)
    valid = np.isfinite(values)
    levels, inverse = np.unique(values[valid], return_inverse=True)
    ranks[valid] = inverse / (len(levels) - 1) if len(levels) > 1 else 0.5
    return ranks


def trace_opacity(view: dict, count: int) -> int:
    """Return a percent. In Auto, dense overlays and overlays behind a mean use less ink."""
    if view["opacity"] != "auto":
        return int(view["opacity"])
    percent = 108 - 8 * np.log2(max(1, count))
    if view["show"] == "traces":
        percent += 14
    return int(round(np.clip(percent, 48, 95)))


def distinct_colors(count: int) -> List[Tuple[int, int, int]]:
    """``count`` RGB colors: separate hues for a few traces, an ordered scale for many."""
    if count <= 0:
        return []
    if count <= len(QUALITATIVE):
        return [QtGui.QColor(code).getRgb()[:3] for code in QUALITATIVE[:count]]
    try:
        scale = pg.colormap.get("turbo")
    except FileNotFoundError:
        scale = pg.colormap.get("viridis")
    # Stay away from the very dark / very bright ends so every trace is visible.
    colors = scale.map(np.linspace(0.06, 0.94, count), mode="byte")
    return [tuple(int(v) for v in color[:3]) for color in colors]


def trace_colors(count: int, base_color, mode: str, *, ranks=None, background="#101520",
                 opacity: int = 100, rank_by: str = "order") -> List[Tuple[int, int, int, int]]:
    """Return colors in original row order. Only RGB encodes rank, and alpha controls opacity."""
    if count <= 0:
        return []
    if ranks is None:
        ranks = np.linspace(0, 1, count) if count > 1 else np.array([0.5])
    ranks = np.asarray(ranks, float)
    missing = ~np.isfinite(ranks)
    positions = np.where(missing, 0.5, ranks)
    alpha = round(255 * opacity / 100)
    surface = np.array(pg.mkColor(background).getRgbF()[:3])
    surface_luma = surface @ LUMA_WEIGHTS
    dark = surface_luma < 0.5
    if mode == "distinct":
        if rank_by == "order":
            rgb = np.array(distinct_colors(count))
        else:
            # Viridis encodes amplitude monotonically, including for just a few trials.
            scale = pg.colormap.get("viridis")
            if not dark:
                positions = 1 - positions
            rgb = scale.map(0.12 + 0.76 * positions, mode="byte")[:, :3].copy()
    else:
        # Interpolate through the chosen hue at prescribed brightness levels.
        # A blue, yellow, or grayscale selection spans the same visible range.
        base = np.array(pg.mkColor(base_color).getRgbF()[:3])
        luma = base @ LUMA_WEIGHTS
        low, high = DARK_PLOT_RAMP if dark else LIGHT_PLOT_RAMP
        low = max(low, surface_luma + 0.15) if dark else min(low, surface_luma - 0.15)
        target = (low + (high - low) * positions)[:, None]
        shade = base * target / max(luma, 1e-12)
        tint = base + (1 - base) * (target - luma) / max(1 - luma, 1e-12)
        rgb = np.rint(255 * np.where(target < luma, shade, tint)).astype(int)
    rgb[missing] = 152 if dark else 110
    return [(int(r), int(g), int(b), alpha) for r, g, b in rgb]


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
    is an index into ``subset``. Interpolate the two adjacent samples so the
    pointer follows sloping lines even in a coarsely binned PSTH.
    """
    rows, tvec = np.asarray(rows, float), np.asarray(tvec, float)
    if rows.ndim != 2 or rows.shape[0] == 0 or tvec.size != rows.shape[1] or tvec.size == 0:
        return None
    if subset is not None and len(subset) == 0:
        return None
    if not np.isfinite(x) or not np.isfinite(y) or x < tvec[0] or x > tvec[-1] or y_per_px <= 0:
        return None
    column = int(np.searchsorted(tvec, x))
    selected = slice(None) if subset is None else np.asarray(subset, int)
    values = rows[selected, column]
    if column and tvec[column] != x:
        previous = rows[selected, column - 1]
        fraction = (x - tvec[column - 1]) / (tvec[column] - tvec[column - 1])
        # A gap touching either endpoint remains unselectable between samples.
        with np.errstate(invalid="ignore"):
            values = previous + fraction * (values - previous)
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
        self.amplitudes = np.empty(0)               # finite-window excursions in signal units, if ranked
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
        self.amplitudes = np.empty(0)
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
             segments: Sequence[Tuple[int, int, object, str]], view: Dict[str, object], *,
             pre_window=None, post_window=None, restyle=False) -> None:
        """Draw the rows of ``rows`` (one trace each) in the chosen colors and style.

        ``segments``: (start, stop, base color, name) row ranges that share a
        base color: the whole matrix in Individual scope, one per group in the
        group view. A single-hue scale restarts in every segment. Window
        boundaries refer to seconds on the displayed alignment clock.
        """
        view = normalize_trace_view(view)
        rows, tvec = np.asarray(rows, float), np.asarray(tvec, float)
        # Menu and window edits reuse the live curves only for the exact cached
        # dataset. A data refresh always rebuilds paths and the finite-row selection.
        reuse = restyle and self.rows is rows and self.tvec is tvec
        if reuse:
            self._set_highlight(None)
            self.legend.clear()
            self.legend.hide()
        else:
            self.clear()
        if rows.ndim != 2 or rows.shape[0] == 0 or tvec.size != rows.shape[1]:
            return
        self.rows, self.tvec, self.style = rows, tvec, str(view["style"])
        count = rows.shape[0]
        self.labels = [str(labels[i]) if i < len(labels) else f"Row {i + 1}" for i in range(count)]
        # Rank the complete matrix, even when the rendering budget samples only some rows.
        if view["rank"] == "amplitude":
            self.amplitudes = response_amplitudes(rows, tvec, pre_window, post_window)
        else:
            self.amplitudes = np.empty(0)
        if reuse:
            drawn_rows = self.row_of_curve
        else:
            drawn_rows = [row for row in thinned_indices(count) if np.isfinite(rows[row]).any()]
        opacity = trace_opacity(view, len(drawn_rows))
        surface = self.plot.backgroundBrush().color()
        ranges = list(segments) or [(0, count, "#808080", "")]
        color_ranges = [(0, count, None, "")] if view["color"] == "distinct" else ranges
        self.colors = [(128, 128, 128, round(255 * opacity / 100))] * count
        for start, stop, base, _name in color_ranges:
            ranks = _ranks(self.amplitudes[start:stop]) if self.amplitudes.size else None
            self.colors[start:stop] = trace_colors(stop - start, base, view["color"], ranks=ranks,
                                                   background=surface, opacity=opacity, rank_by=view["rank"])
        for index, row in enumerate(drawn_rows):
            pen = self._pen(self.colors[row])
            if reuse:
                self.curves[index].setPen(pen)
                continue
            curve = pg.PlotDataItem(tvec, rows[row], pen=pen, connect="finite")
            curve.setZValue(-3)          # behind the SEM band and the mean
            self.plot.addItem(curve)
            # Set after the item has a view box: pyqtgraph resolves the view
            # while these options are applied and fails on a parentless item.
            curve.setDownsampling(auto=True, method="peak")
            curve.setClipToView(True)
            self.curves.append(curve)
            self.row_of_curve.append(row)
        if view["legend"] and self.curves:
            budget = LEGEND_ENTRIES_DISTINCT if view["color"] == "distinct" else LEGEND_ENTRIES_FADE
            # The budget is shared by the segments (groups); each names at
            # least its first and last trace, which is what a fade needs.
            limit = max(2, budget // max(1, len(ranges)))
            for start, stop, _base, name in ranges:
                drawn = [i for i, row in enumerate(self.row_of_curve) if start <= row < stop]
                if self.amplitudes.size:
                    drawn.sort(key=lambda i: np.nan_to_num(self.amplitudes[self.row_of_curve[i]], nan=np.inf))
                for position in legend_indices(len(drawn), limit):
                    index = drawn[position]
                    label = self._row_label(self.row_of_curve[index])
                    self.legend.addItem(self.curves[index], f"{name}: {label}" if name else label)
            relayout_legend(self.legend, 1)
            self.legend.show()

    # ------------------------------------------------------------------ hover
    def _row_label(self, row: int) -> str:
        """Keep the original trial ID and add the measured excursion when color represents amplitude."""
        if not self.amplitudes.size:
            return self.labels[row]
        value = self.amplitudes[row]
        amplitude = f"{value:.3g}" if np.isfinite(value) else "unavailable"
        return f"{self.labels[row]} · amplitude {amplitude}"

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
        label = self._row_label(self.row_of_curve[row])
        self.tip.setText(label)
        # Place the tip toward the middle so trial names are readable at the plot edges.
        rect = view.sceneBoundingRect()
        self.tip.setAnchor((float(scene_position.x() > rect.center().x()),
                            float(scene_position.y() > rect.center().y())))
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
        self.curves[index].setZValue(1)   # over the other trials, below the mean


# =============================================================================
# Menu
# =============================================================================

def populate_trace_menu(menu: QtWidgets.QMenu, view: Dict[str, object], change) -> None:
    """Fill ``menu`` with the trace options; ``change(key, value)`` applies one."""
    view = normalize_trace_view(view)
    for title, key, choices in ((None, "show", SHOW_MODES), ("Trace colors", "color", COLOR_MODES),
                                ("Color by", "rank", RANK_MODES), ("Trace opacity", "opacity", OPACITY_PRESETS),
                                ("Line style", "style", LINE_STYLES)):
        target = menu if title is None else menu.addMenu(title)
        group = QtGui.QActionGroup(target)
        for code, label in choices.items():
            action = target.addAction(label)
            action.setCheckable(True)
            action.setChecked(view[key] == code)
            group.addAction(action)
            action.triggered.connect(lambda _checked=False, k=key, c=code: change(k, c))
            if code == "amplitude":
                action.setToolTip("Largest absolute change in Post from that row's mean in Pre. "
                                  "Set both windows in PSTH metrics.")
        if key == "opacity":
            current = view["opacity"]
            custom = target.addAction(f"Custom ({current}%)…" if current not in choices else "Custom…")
            custom.setCheckable(True)
            group.addAction(custom)
            custom.setChecked(current not in choices)

            def choose_opacity():
                """Allow exact percentages as well as the one-click presets."""
                percent, ok = QtWidgets.QInputDialog.getInt(
                    menu, "Trace opacity", "Individual trace opacity (%)",
                    current if isinstance(current, int) else 70, 1, 100)
                if ok:
                    change("opacity", percent)

            custom.triggered.connect(choose_opacity)
        if key == "rank":
            target.setToolTipsVisible(True)
            target.addAction("Single hue: later or larger = stronger contrast").setEnabled(False)
        if title is None:
            menu.addSeparator()
    legend = menu.addAction("Trace legend")
    legend.setCheckable(True)
    legend.setChecked(bool(view["legend"]))
    legend.toggled.connect(lambda checked: change("legend", bool(checked)))
    hint = menu.addAction("Tip: point at a trace to highlight and name it")
    hint.setEnabled(False)
