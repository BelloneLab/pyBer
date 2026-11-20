"""Side-by-side columns for several selected behaviors: heatmap, mean PSTH and metric.

When two or more behaviors are checked in the Events panel, each one gets a
column with the same three panels, so they can be compared at a glance:

* one heatmap per behavior, all on the SAME color scale (one shared color bar);
* one mean +/- SEM PSTH per behavior, all on the SAME y range, in the color
  the behavior has everywhere else (comparison plot, trace preview);
* one pre/post metric panel per behavior.

Layout rules (nothing is ever cut):

* a column is never narrower than ``MIN_COLUMN_WIDTH``; when the columns do not
  fit side by side they wrap onto further rows ("bands") and the results area
  scrolls vertically;
* every panel keeps a minimum height;
* the long y-axis labels are written once per band (first column), tick labels
  are thinned by pyqtgraph, and the behavior name in the header is elided with
  its full text in the tooltip.

This module only draws. The matrices are computed by the PSTH panel with the
same function, filters and settings as the single-behavior view.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pyqtgraph as pg
from PySide6 import QtCore, QtGui, QtWidgets

import missing_data
from plot_empty_state import set_plot_has_data
from postprocessing_core import mean_sem
from postprocessing_style import style_plot
from psth_metric_panels import MetricPanel
from psth_mean_band import sem_band, style_mean_band
from psth_trace_overlay import TraceOverlay, normalize_trace_view

# =============================================================================
# Configuration
# =============================================================================

#: Behavior colors, shared with the comparison plot of the Events panel.
BEHAVIOR_COLORS: Tuple[Tuple[int, int, int], ...] = (
    (89, 174, 255), (245, 153, 92), (124, 207, 151), (202, 153, 239), (240, 203, 99), (104, 210, 213))
MAX_BEHAVIORS = 6            # columns drawn at most (more are named in the status bar)
MIN_COLUMN_WIDTH = 300       # px: narrower columns wrap onto another band
COLORBAR_WIDTH = 70          # px, shared color bar
HEADER_HEIGHT = 24           # px, behavior name row
HEAT_MIN_HEIGHT = 150        # px per panel
PSTH_MIN_HEIGHT = 170
METRIC_MIN_HEIGHT = 200
BAND_SPACING = 10            # px between bands and between columns
LEFT_AXIS_WIDTH = 58         # px: identical in every panel so the plots line up
MAX_HEAT_TICKS = 6           # row labels on a heatmap
INVISIBLE_PEN = (0, 0, 0, 0)  # RGBA of the SEM band edges (drawn, but fully transparent)


def behavior_color(index: int) -> Tuple[int, int, int]:
    """Color of the ``index``-th selected behavior (cycles after six)."""
    return BEHAVIOR_COLORS[index % len(BEHAVIOR_COLORS)]


def columns_per_band(available_width: int, count: int, min_width: int = MIN_COLUMN_WIDTH,
                     reserved: int = COLORBAR_WIDTH) -> int:
    """How many behavior columns fit side by side (at least one, at most ``count``)."""
    if count <= 0:
        return 1
    usable = max(0, int(available_width) - reserved)
    return int(max(1, min(count, (usable + BAND_SPACING) // (min_width + BAND_SPACING))))


def shared_range(entries: Sequence[Dict[str, object]], include_rows: bool) -> Optional[Tuple[float, float]]:
    """Common y range of the mean +/- SEM (and of the single rows when they are drawn)."""
    low, high = np.inf, -np.inf
    for entry in entries:
        matrix = entry.get("matrix")
        if matrix is None or not np.size(matrix):
            continue
        mean, sem, _counts = mean_sem(np.asarray(matrix, float))
        # The mean still sets the axis range in bins with only one observation.
        values = [mean, mean - sem, mean + sem] + ([np.asarray(matrix, float)] if include_rows else [])
        for block in values:
            finite = block[np.isfinite(block)]
            if finite.size:
                low, high = min(low, float(finite.min())), max(high, float(finite.max()))
    if not np.isfinite(low) or not np.isfinite(high):
        return None
    span = high - low or max(abs(high) * 0.1, 1.0)
    return low - 0.06 * span, high + 0.06 * span


class ElidedLabel(QtWidgets.QLabel):
    """One-line label that shortens its text with an ellipsis instead of being cut."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._full = ""
        self._dot: Optional[Tuple[int, int, int]] = None
        self.setTextFormat(QtCore.Qt.TextFormat.RichText)
        self.setMinimumWidth(40)
        self.setSizePolicy(QtWidgets.QSizePolicy.Policy.Ignored, QtWidgets.QSizePolicy.Policy.Fixed)

    def set_full_text(self, text: str, dot: Optional[Tuple[int, int, int]] = None) -> None:
        """Text to show (elided when needed), optionally after a colored dot."""
        self._full = str(text)
        self._dot = dot
        self.setToolTip(self._full)
        self._elide()

    def full_text(self) -> str:
        return self._full

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._elide()

    def _elide(self) -> None:
        from html import escape
        reserved = 24 if self._dot is not None else 4
        text = escape(self.fontMetrics().elidedText(self._full, QtCore.Qt.TextElideMode.ElideRight,
                                                    max(10, self.width() - reserved)))
        if self._dot is not None:
            red, green, blue = self._dot
            text = f"<span style='color: rgb({red},{green},{blue});'>\u25cf</span> {text}"
        self.setText(text)


# =============================================================================
# One behavior
# =============================================================================

class BehaviorColumn(QtWidgets.QWidget):
    """Header, heatmap, mean PSTH and metric panel of one behavior."""

    extraMetricCreated = QtCore.Signal(object)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.name = ""
        self.setObjectName("figureSurface")      # transparent, like the other figure containers
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)
        self.header = ElidedLabel()
        self.header.setFixedHeight(HEADER_HEIGHT)
        layout.addWidget(self.header)
        self.heat = pg.PlotWidget()
        self.psth = pg.PlotWidget()
        self.metric = MetricPanel()
        self.extra_metrics: Dict[str, MetricPanel] = {}
        self.metric.setMinimumSize(150, METRIC_MIN_HEIGHT)
        for plot, height, stretch in ((self.heat, HEAT_MIN_HEIGHT, 3), (self.psth, PSTH_MIN_HEIGHT, 3),
                                      (self.metric, METRIC_MIN_HEIGHT, 3)):
            plot.setMinimumHeight(height)
            plot.setMinimumWidth(150)
            layout.addWidget(plot, stretch)
        # Heatmap items.
        self.image = pg.ImageItem()
        self.heat.addItem(self.image)
        self.heat_zero = pg.InfiniteLine(pos=0.0, angle=90, movable=False,
                                         pen=pg.mkPen((245, 245, 245), width=1.0,
                                                      style=QtCore.Qt.PenStyle.DotLine))
        self.heat_zero.setZValue(20)
        self.heat.addItem(self.heat_zero)
        # PSTH items: SEM band between two invisible curves, mean on top. The
        # edges need a (transparent) pen: pyqtgraph builds no path for a curve
        # without pen, and the band between two such curves stays empty.
        self.sem_high = self.psth.plot(pen=INVISIBLE_PEN)
        self.sem_low = self.psth.plot(pen=INVISIBLE_PEN)
        self.sem_band = sem_band(self.sem_high, self.sem_low)
        self.psth.addItem(self.sem_band)
        self.mean = self.psth.plot(pen=pg.mkPen("#4D93BD", width=2.0))
        self.mean.setZValue(2)
        self.psth_zero = pg.InfiniteLine(pos=0.0, angle=90, movable=False,
                                         pen=pg.mkPen((150, 160, 175), width=1.0,
                                                      style=QtCore.Qt.PenStyle.DashLine))
        self.psth.addItem(self.psth_zero, ignoreBounds=True)
        self.trace_legend = pg.LegendItem(offset=(10, 8), verSpacing=-6)
        self.trace_legend.setParentItem(self.psth.getPlotItem().getViewBox())
        self.trace_legend.hide()
        self.traces = TraceOverlay(self.psth, self.trace_legend, parent=self)
        self.psth.scene().sigMouseMoved.connect(self.traces.hover)
        # Heatmap and PSTH of a column share their time axis (zoom together).
        self.heat.setXLink(self.psth)
        for plot in (self.heat, self.psth):
            plot.getAxis("left").setWidth(LEFT_AXIS_WIDTH)
            plot.setMenuEnabled(True)
        self.heat.getAxis("bottom").setStyle(showValues=False)
        self.heat.getAxis("bottom").setHeight(6)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        QtCore.QTimer.singleShot(0, self._recenter_axis_labels)

    def _recenter_axis_labels(self) -> None:
        """pyqtgraph places an axis label when the axis is resized; a label set
        on a plot that is laid out later would otherwise stay off-center."""
        try:
            for plot in (self.heat, self.psth, self.metric):
                for side in ("left", "bottom"):
                    plot.getAxis(side).resizeEvent()
        except RuntimeError:      # the column was deleted before the queued call ran
            pass

    # ------------------------------------------------------------------ theme
    def apply_theme(self, palette: Dict[str, object]) -> None:
        """Same axes, fonts and surface as the other result panels."""
        for plot in (self.heat, self.psth, self.metric, *self.extra_metrics.values()):
            style_plot(plot, palette)
            plot.getPlotItem().layout.setContentsMargins(4, 4, 8, 4)
        for plot in (self.heat, self.psth):
            plot.getAxis("left").setWidth(LEFT_AXIS_WIDTH)
        self.heat.showGrid(x=False, y=False)
        self.metric.set_colors(palette)
        for plot in self.extra_metrics.values():
            plot.set_colors(palette)
        self.header.setStyleSheet(f"color: {palette['text']}; font-weight: 600; font-size: 12px;")

    # ------------------------------------------------------------------- data
    def sync_extra_metrics(self, entry, keys, palette, visible, first_in_band) -> None:
        """Render selected metrics and named windows using this event's own results."""
        for key in list(self.extra_metrics):
            if key not in keys:
                plot = self.extra_metrics.pop(key)
                self.layout().removeWidget(plot)
                plot.hide()
                plot.deleteLater()
        first_extra = self.layout().count() - len(self.extra_metrics)
        for index, key in enumerate(keys):
            plot = self.extra_metrics.get(key)
            if plot is None:
                plot = MetricPanel()
                plot.setMinimumSize(150, METRIC_MIN_HEIGHT)
                self.extra_metrics[key] = plot
                if palette is not None:
                    style_plot(plot, palette)
                    plot.set_colors(palette)
                self.extraMetricCreated.emit(plot)
            # Reinsert in selection order after the primary metric.
            if self.layout().indexOf(plot) != first_extra + index:
                self.layout().insertWidget(first_extra + index, plot, 3)
            plot.setVisible(visible)
            result = entry.get("metrics", {}).get(key)
            plot.show_result(result, row_colors=entry.get("row_colors"))
            summary = result["summary"] if result else None
            set_plot_has_data(plot, bool(summary and (summary["pre_n"] or summary["post_n"])))
            if not first_in_band:
                plot.setLabel("left", "")

    def show_entry(self, entry: Dict[str, object], color: Tuple[int, int, int], levels: Tuple[float, float],
                   lookup_table, y_range: Optional[Tuple[float, float]], units: str, time_label: str,
                   trace_view: Dict[str, object], missing_mode: str, missing_trim: bool,
                   first_in_band: bool, show: Dict[str, bool]) -> None:
        """Draw one behavior. ``entry``: name, matrix, tvec, labels, metric (or matrix None)."""
        self.name = str(entry["name"])
        matrix, tvec = entry.get("matrix"), entry.get("tvec")
        ready = matrix is not None and np.size(matrix) > 0 and tvec is not None and np.any(np.isfinite(matrix))
        rows = int(np.shape(matrix)[0]) if ready else 0
        unit = str(entry.get("row_unit", "trials"))
        self.header.set_full_text(f"{self.name} (n={rows} {unit})" if ready else f"{self.name} (no events)",
                                  dot=tuple(int(v) for v in color))
        self.heat.setVisible(show.get("heat", True))
        self.psth.setVisible(show.get("psth", True))
        self.metric.setVisible(show.get("metric", True))
        for plot in (self.heat, self.psth):
            set_plot_has_data(plot, bool(ready))
        if not ready:
            self.traces.clear()
            self.metric.show_result(None)
            set_plot_has_data(self.metric, False)
            return
        matrix, tvec = np.asarray(matrix, float), np.asarray(tvec, float)
        # --- heatmap (display-only handling of missing samples, shared levels)
        drawn, drawn_t, _report = missing_data.prepare_heatmap(matrix, tvec, missing_mode, missing_trim)
        self.image.setLookupTable(lookup_table)
        self.image.setImage(drawn.T, autoLevels=False)
        self.image.setLevels(list(levels))
        half = float(drawn_t[1] - drawn_t[0]) / 2 if drawn_t.size > 1 else 0.5
        self.image.setRect(QtCore.QRectF(float(drawn_t[0]) - half, 0, float(drawn_t[-1] - drawn_t[0]) + 2 * half, rows))
        self.heat.setYRange(0, rows, padding=0)
        step = max(1, int(np.ceil(rows / MAX_HEAT_TICKS)))
        self.heat.getAxis("left").setTicks([[(index + 0.5, str(index + 1)) for index in range(0, rows, step)]])
        self.heat.setLabel("left", unit.capitalize() if first_in_band else "")
        self.heat_zero.setVisible(bool(tvec[0] <= 0.0 <= tvec[-1]))
        # --- mean PSTH
        mean, sem, _counts = mean_sem(matrix)
        self.mean.setData(tvec, mean, connect="finite")
        self.sem_high.setData(tvec, mean + sem, connect="finite")
        self.sem_low.setData(tvec, mean - sem, connect="finite")
        self.set_trace_view(entry, color, trace_view)
        self.psth.setXRange(float(tvec[0]), float(tvec[-1]), padding=0)
        if y_range is not None:
            self.psth.setYRange(y_range[0], y_range[1], padding=0)
        self.psth.setLabel("left", units if first_in_band else "")
        self.psth.setLabel("bottom", time_label)
        self.psth_zero.setVisible(bool(tvec[0] <= 0.0 <= tvec[-1]))
        # --- metric
        result = entry.get("metric")
        self.metric.show_result(result, row_colors=entry.get("row_colors"))
        summary = result["summary"] if result else None
        set_plot_has_data(self.metric, bool(summary and (summary["pre_n"] or summary["post_n"])))
        if not first_in_band:
            self.metric.setLabel("left", "")
        QtCore.QTimer.singleShot(0, self._recenter_axis_labels)

    def set_trace_view(self, entry, color, view, *, restyle=False) -> None:
        """Restyle one cached PSTH, keeping the heatmap, statistics and current zoom intact."""
        pen_color = QtGui.QColor(*color)
        style_mean_band(self.mean, self.sem_band, pen_color, self.psth.backgroundBrush().color(),
                        mean_width=2.0, combined=view["show"] == "both")
        for item in (self.mean, self.sem_band):
            item.setVisible(view["show"] != "traces")
        if view["show"] == "average":
            self.traces.clear()
        else:
            rows = entry["matrix"]
            self.traces.draw(entry["tvec"], rows, entry.get("labels") or [], [(0, len(rows), pen_color, "")], view,
                             pre_window=entry.get("pre_window"), post_window=entry.get("post_window"), restyle=restyle)


# =============================================================================
# All behaviors
# =============================================================================

class MultiBehaviorView(QtWidgets.QWidget):
    """Wrapping grid of behavior columns with one shared color bar."""

    #: Emitted when the number of bands (and so the needed height) changes.
    heightChanged = QtCore.Signal()
    columnCreated = QtCore.Signal(object)
    columnDrawn = QtCore.Signal(object, object, object)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.columns: List[BehaviorColumn] = []
        self.entries: List[Dict[str, object]] = []
        self._per_band = 0
        self._palette: Optional[Dict[str, object]] = None
        self._show = {"heat": True, "psth": True, "metric": True}
        self._show_extra_metrics = True
        self._extra_metric_keys = []
        self._render_args: Optional[Dict[str, object]] = None
        self.grid = QtWidgets.QGridLayout(self)
        self.grid.setContentsMargins(4, 4, 4, 4)
        self.grid.setHorizontalSpacing(BAND_SPACING)
        self.grid.setVerticalSpacing(BAND_SPACING)
        # Shared color bar, aligned with the heatmaps of the first band.
        self.colorbar_host = QtWidgets.QWidget()
        self.colorbar_host.setObjectName("figureSurface")
        host = QtWidgets.QVBoxLayout(self.colorbar_host)
        host.setContentsMargins(0, HEADER_HEIGHT + 2, 0, 0)
        host.setSpacing(0)
        self.colorbar_widget = pg.GraphicsLayoutWidget()
        self.colorbar_widget.setFixedWidth(COLORBAR_WIDTH)
        self.colorbar_widget.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
        self.colorbar_widget.setStyleSheet("border: none;")
        self.colorbar = pg.ColorBarItem(values=(0, 1), colorMap=pg.colormap.get("viridis"), width=12,
                                        interactive=False, rounding=.001, colorMapMenu=False)
        for key in ("left", "top", "bottom"):
            self.colorbar.getAxis(key).setPen(pg.mkPen(None))
        self.colorbar_widget.addItem(self.colorbar)
        host.addWidget(self.colorbar_widget)
        host.addStretch(1)
        self.colorbar_host.setFixedWidth(COLORBAR_WIDTH)

    # ----------------------------------------------------------------- layout
    def band_height(self) -> int:
        """Minimum height of one band of columns with the visible panels."""
        height = HEADER_HEIGHT + 2
        for key, minimum in (("heat", HEAT_MIN_HEIGHT), ("psth", PSTH_MIN_HEIGHT), ("metric", METRIC_MIN_HEIGHT)):
            if self._show.get(key, True):
                height += minimum + 2
        if self._show_extra_metrics:
            height += len(self._extra_metric_keys) * (METRIC_MIN_HEIGHT + 2)
        return height

    def bands(self) -> int:
        per_band = max(1, self._per_band or 1)
        return max(1, -(-len(self.entries) // per_band))

    def required_height(self) -> int:
        """Height needed to show every band without squeezing a panel."""
        bands = self.bands()
        return bands * self.band_height() + (bands - 1) * BAND_SPACING + 8

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._relayout()
        QtCore.QTimer.singleShot(0, self._sync_colorbar)

    def _sync_colorbar(self) -> None:
        """Keep the shared color bar as tall as the heatmaps of the first band."""
        if self.columns and self.entries and self.columns[0].heat.height() > 0:
            self.colorbar_widget.setFixedHeight(max(HEAT_MIN_HEIGHT, self.columns[0].heat.height()))

    def _relayout(self, force: bool = False) -> None:
        """Place the columns: as many per band as fit at ``MIN_COLUMN_WIDTH``."""
        count = len(self.entries)
        per_band = columns_per_band(self.width(), count)
        if per_band == self._per_band and not force:
            return
        previous_bands = self.bands()
        self._per_band = per_band
        while self.grid.count():
            self.grid.takeAt(0)
        for index, column in enumerate(self.columns[:count]):
            self.grid.addWidget(column, index // per_band, index % per_band)
            self.grid.setColumnStretch(index % per_band, 1)
            column.show()
        for column in self.columns[count:]:
            column.hide()
        for extra in range(per_band, self.grid.columnCount()):
            self.grid.setColumnStretch(extra, 0)
        self.grid.addWidget(self.colorbar_host, 0, per_band)
        self.colorbar_host.setVisible(bool(self._show.get("heat", True)) and count > 0)
        self.setMinimumHeight(self.required_height())
        # Axis labels are written once per band: redraw with the new positions.
        if self._render_args is not None:
            self._draw()
        if self.bands() != previous_bands or force:
            self.heightChanged.emit()

    # ------------------------------------------------------------------- data
    def apply_theme(self, palette: Dict[str, object]) -> None:
        self._palette = palette
        for column in self.columns:
            column.apply_theme(palette)
        self.colorbar_widget.setBackground(palette["surface"])
        self.colorbar.axis.setTextPen(pg.mkPen(palette["muted"]))
        self.colorbar.axis.setPen(pg.mkPen(palette["axis"]))

    def set_visible_panels(self, heat: bool, psth: bool, metric: bool, *, extras=None) -> None:
        show = {"heat": bool(heat), "psth": bool(psth), "metric": bool(metric)}
        show_extras = bool(metric if extras is None else extras)
        if self._show == show and self._show_extra_metrics == show_extras:
            return
        self._show, self._show_extra_metrics = show, show_extras
        if self.entries:
            self._relayout(force=True)

    def clear(self) -> None:
        self.entries = []
        self._render_args = None
        for column in self.columns:
            column.traces.clear()
            column.hide()

    def set_trace_view(self, view, pre_window, post_window) -> None:
        """Apply a menu or window change to the cached columns without rebuilding the PSTHs."""
        if self._render_args is None:
            return
        rows_were_visible = self._render_args["trace_view"].get("show", "average") != "average"
        self._render_args["trace_view"] = view
        rows_visible = view["show"] != "average"
        # Preserve the current zoom unless switching between mean and individual-trial extents.
        limits = shared_range(self.entries, rows_visible) if rows_visible != rows_were_visible else None
        for index, (entry, column) in enumerate(zip(self.entries, self.columns)):
            entry.update(pre_window=pre_window, post_window=post_window)
            if column.psth.property("hasPlotData"):
                column.set_trace_view(entry, entry.get("color", behavior_color(index)), view, restyle=True)
                if limits is not None:
                    column.psth.setYRange(*limits, padding=0)

    def show_entries(self, entries: Sequence[Dict[str, object]], *, color_map, levels: Tuple[float, float],
                     units: str, time_label: str, trace_view: Dict[str, object], missing_mode: str,
                     missing_trim: bool) -> None:
        """Draw one column per entry (name, matrix, tvec, labels, row_unit, metric, color)."""
        self.entries = list(entries)[:MAX_BEHAVIORS]
        self._extra_metric_keys = list(dict.fromkeys(key for entry in self.entries
                                                     for key in entry.get("extra_metric_keys", ())))
        while len(self.columns) < len(self.entries):
            column = BehaviorColumn(self)
            if self._palette is not None:
                column.apply_theme(self._palette)
            self.columns.append(column)
            self.columnCreated.emit(column)
        self._render_args = {"color_map": color_map, "levels": levels, "units": units, "time_label": time_label,
                             "trace_view": trace_view, "missing_mode": missing_mode, "missing_trim": missing_trim}
        self._relayout(force=True)

    def _draw(self) -> None:
        args = self._render_args
        view = normalize_trace_view(args["trace_view"])
        y_range = shared_range(self.entries, include_rows=view["show"] != "average")
        lookup = args["color_map"].getLookupTable(nPts=256, alpha=True)
        self.colorbar.setColorMap(args["color_map"])
        low, high = args["levels"]
        self.colorbar.setLevels((low, high))
        self.colorbar.rounding = max((high - low) / 1000., 1e-15)
        self._sync_colorbar()
        per_band = max(1, self._per_band)
        for index, (entry, column) in enumerate(zip(self.entries, self.columns)):
            column.show_entry(entry, entry.get("color", behavior_color(index)), (low, high), lookup, y_range,
                              args["units"], args["time_label"], view, args["missing_mode"],
                              args["missing_trim"], first_in_band=index % per_band == 0, show=self._show)
            column.sync_extra_metrics(entry, self._extra_metric_keys, self._palette,
                                      self._show_extra_metrics, index % per_band == 0)
            self.columnDrawn.emit(column, entry, dict(self._show))
