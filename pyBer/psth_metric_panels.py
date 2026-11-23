"""Compact paired-distribution panels with explicit median and mean markers."""
from __future__ import annotations

import numpy as np
import pyqtgraph as pg
from PySide6 import QtCore, QtGui, QtWidgets


# =============================================================================
# Configuration
# =============================================================================

PANEL_MIN_SIZE = (230, 230)       # smallest legible metric panel (pixels)
GRID_MAX_COLUMNS = 3              # extra metric panels per row at most
GRID_CELL_MIN_WIDTH = 262         # panel + card margins + grid spacing
TITLE_MIN_POINT_SIZE = 7.0        # long titles ("AUC: Pre vs Early vs Late") shrink down to this
TITLE_POINT_STEP = 0.5
TITLE_RIGHT_MARGIN = 8            # pixels kept free at the right of a fitted title


class MetricGrid(QtWidgets.QWidget):
    """As many columns (one to three) as fit without squeezing a panel below its minimum width."""
    columnsChanged = QtCore.Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.columns = 2

    def resizeEvent(self, event):
        super().resizeEvent(event)
        columns = max(1, min(GRID_MAX_COLUMNS, self.width() // GRID_CELL_MIN_WIDTH))
        if columns != self.columns:
            self.columns = columns
            self.columnsChanged.emit()


class MetricPanel(pg.PlotWidget):
    """One metric per independently scaled card, shared by screen and image export."""

    def __init__(self, parent=None):
        super().__init__(parent=parent, title="PSTH metrics")
        self.setMinimumSize(*PANEL_MIN_SIZE)
        self.bars = [pg.BarGraphItem(x=[x], height=[0], width=0.25) for x in (0, 1)]
        self.pairs = self.plot(connect="finite", pen=pg.mkPen("#8893A1", width=0.8))
        self.points = [self.plot(pen=None, symbol="o", symbolSize=4) for _ in (0, 1)]
        self.medians = [self.plot(pen=pg.mkPen("#4D93BD", width=2.5)) for _ in (0, 1)]
        self.means = [self.plot(pen=None, symbol="d", symbolSize=6) for _ in (0, 1)]
        self.errors = [pg.ErrorBarItem(beam=0.10) for _ in (0, 1)]
        for item in self.errors:
            self.addItem(item)
        # Retain legacy bar handles for older extensions without hiding data
        # behind filled bars. All summaries are now displayed as markers.
        for item in self.bars:
            item.hide()
        # Anchor the note's top inside the reserved headroom. A bottom anchor
        # places its first line above the view boundary in short metric cards.
        self.note = pg.TextItem(anchor=(0.5, 0.0))
        self.note.setFont(QtGui.QFont("Segoe UI", 8))
        self.addItem(self.note)
        self.setXRange(-0.4, 1.5, padding=0)
        self.getAxis("bottom").setTicks([[(0, "Pre"), (1, "Post")]])
        self.result = None
        self._data_bounds = None
        self._group_items = []
        # Whether a note belongs to the current result. ``note.isVisible()`` is
        # False while the whole plot is hidden (no data yet), which used to
        # skip the placement and leave the note clipped under the title.
        self._note_wanted = False
        # Summary style of the group views: points | bars | box (group_metric_views).
        self.summary_style = "points"
        # Titles of the group / window views are fitted to the panel width;
        # the base size is the one set by the plot style (read on first use).
        self._title_fit = False
        self._title_base = None
        self.set_colors({"text": "#CED8E6", "accent": "#4D93BD"})
        self.getViewBox().sigResized.connect(self._fit_note)
        self.getViewBox().sigResized.connect(self._fit_title)
        self.getViewBox().sigXRangeChanged.connect(self._fit_note)
        self.getPlotItem().visibleChanged.connect(self._fit_note)

    def _fit_note(self, *_args):
        """Keep the summary centered and legible when a metric card resizes."""
        if not self._note_wanted:
            return
        view = self.getViewBox()
        font = QtGui.QFont("Segoe UI", 8)
        self.note.setFont(font)
        width = self.note.boundingRect().width()
        available = max(1., view.sceneBoundingRect().width() - 12)
        if width > available:
            font.setPointSizeF(max(6., 8 * available / width))
            self.note.setFont(font)
        if self._data_bounds is not None:
            low, high = self._data_bounds
            span = high - low or max(abs(high) * 0.1, 1.0)
            height = max(1., view.sceneBoundingRect().height())
            # Text has a fixed pixel height; a fixed percentage of data range
            # cannot protect it from the highest point in short metric cards.
            headroom = self.note.boundingRect().height() + 18.
            data_fraction = max(.1, 1. - headroom / height)
            bottom = low - span * .08
            extent = span * 1.08 / data_fraction
            self.setYRange(bottom, bottom + extent, padding=0)
            self.note.setPos(float(np.mean(view.viewRange()[0])),
                             bottom + extent * (1. - 4. / height))

    def _fit_title(self, *_args):
        """Shrink a long view title to the panel width instead of letting it widen the plot.

        A pyqtgraph title reserves its full text width, so a title longer than
        the panel pushes the right side of the plot out of view. The size is
        chosen from font metrics (no trial layouts) and the reserved width is
        released.
        """
        if not self._title_fit:
            return
        label = self.getPlotItem().titleLabel
        if self._title_base is None:
            self._title_base = str(label.opts.get("size", "10pt"))
        available = self.width() - label.geometry().left() - TITLE_RIGHT_MARGIN
        if available <= 20 or not label.text:
            return
        try:
            size = float(self._title_base.replace("pt", ""))
        except ValueError:
            size = 10.0
        font = QtGui.QFont(label.item.font())
        font.setBold(bool(label.opts.get("bold", False)))
        while size > TITLE_MIN_POINT_SIZE:
            font.setPointSizeF(size)
            if QtGui.QFontMetricsF(font).horizontalAdvance(label.text) <= available:
                break
            size -= TITLE_POINT_STEP
        wanted = f"{size:g}pt"
        if label.opts.get("size") != wanted:
            label.setText(label.text, size=wanted)
        label.setMinimumWidth(0)

    def _restore_title(self):
        """Give the title its style size back before a plain (unfitted) title is shown."""
        self._title_fit = False
        label = self.getPlotItem().titleLabel
        if self._title_base is not None and label.opts.get("size") != self._title_base:
            label.setText(label.text, size=self._title_base)

    def set_colors(self, palette):
        """Respect the active plot palette without confusing summary encodings."""
        colors = [QtGui.QColor(palette["accent"]), QtGui.QColor("#D28A64")]
        text = palette["text"]
        self._point_colors = colors
        self._text_color = QtGui.QColor(text)
        for index, color in enumerate(colors):
            self.points[index].setSymbolBrush(pg.mkBrush(color))
            self.points[index].setSymbolPen(pg.mkPen(None))
            self.medians[index].setPen(pg.mkPen(color, width=2.5))
            self.means[index].setSymbolBrush(pg.mkBrush(text))
            self.means[index].setSymbolPen(pg.mkPen(text))
            self.errors[index].setData(pen=pg.mkPen(text, width=1))
        faint = QtGui.QColor(text)
        faint.setAlpha(65)
        self.pairs.setPen(pg.mkPen(faint, width=0.7))
        self.note.setColor(text)
        if getattr(self, "result", None) is not None and (
                getattr(self, "_row_colors", None) is not None or "group_comparison" in self.result
                or "window_view" in self.result):
            # A theme change must not drop the per-recording group tint.
            self.show_result(self.result, row_colors=self._row_colors)

    def show_result(self, result, row_colors=None):
        """Display paired native reductions and complete finite-row summaries.

        ``row_colors`` (one color per matrix row) tints the points by recording
        group; medians then use the neutral text color so they are not read as
        a group. Statistics are unchanged.
        """
        self.result = result
        self._row_colors = row_colors
        self._data_bounds = None
        self._restore_title()
        self.pairs.setData([], [])
        for item in self.points + self.medians + self.means:
            item.setData([], [])
        for item in self.errors:
            item.setData(x=np.array([]), y=np.array([]), top=np.array([]), bottom=np.array([]))
        self.note.hide()
        self._note_wanted = False
        self.getAxis("bottom").setHeight(None)   # group views reserve extra tick-label lines
        for item in self._group_items:
            self.removeItem(item)
        self._group_items = []
        self.getAxis("bottom").setTicks([[(0, "Pre"), (1, "Post")]])
        self.setXRange(-0.4, 1.5, padding=0)
        if result is None:
            self.setYRange(0, 1, padding=0)
            return
        if result.get("group_comparison"):
            self.summary_style = result.get("group_style", self.summary_style)
            self._show_groups(result)
            return
        if result.get("window_view"):
            self.summary_style = result.get("window_style", self.summary_style)
            self._show_windows(result, row_colors)
            return
        summary = result["summary"]
        # Window names chosen by the user (Pre / Post unless renamed or extra).
        self.getAxis("bottom").setTicks([[(0, str(summary.get("pre_label", "Pre"))),
                                          (1, str(summary.get("post_label", "Post")))]])
        pre, post = result["pre_values"], result["post_values"]
        paired = np.isfinite(pre) & np.isfinite(post)
        jitter = np.linspace(-0.12, 0.12, len(pre))
        xp, yp = [], []
        for index in np.flatnonzero(paired):
            xp.extend([jitter[index], 1 + jitter[index], np.nan])
            yp.extend([pre[index], post[index], np.nan])
        self.pairs.setData(xp, yp, connect="finite")
        bounds = []
        tinted = row_colors is not None and len(row_colors) == len(pre)
        for index, (name, values) in enumerate((("pre", pre), ("post", post))):
            valid = np.isfinite(values)
            if tinted:
                brushes = [pg.mkBrush(color) for color, keep in zip(row_colors, valid) if keep]
                self.points[index].setData(index + jitter[valid], values[valid], symbolBrush=brushes)
                self.medians[index].setPen(pg.mkPen(self._text_color, width=2.5))
            else:
                self.points[index].setData(index + jitter[valid], values[valid],
                                           symbolBrush=pg.mkBrush(self._point_colors[index]))
                self.medians[index].setPen(pg.mkPen(self._point_colors[index], width=2.5))
            bounds.extend(values[valid])
            median, mean, sem = summary[name + "_median"], summary[name], summary[name + "_sem"]
            if np.isfinite(median):
                self.medians[index].setData([index - 0.22, index + 0.22], [median, median])
            if np.isfinite(mean):
                error = float(sem) if np.isfinite(sem) else 0.0
                self.means[index].setData([index + 0.28], [mean])
                self.errors[index].setData(x=np.array([index + 0.28]), y=np.array([mean]),
                                           top=np.array([error]), bottom=np.array([error]))
                bounds.extend([mean - error, mean + error])
        self.setTitle(summary["metric"])
        self.setLabel("left", summary["units"])
        self.setToolTip("Colored lines: median. Diamonds: mean with SEM. " +
                        summary["assumption_note"] + " " + summary["reduction_level"])
        if bounds:
            low, high = min(bounds), max(bounds)
            self._data_bounds = (low, high)
            span = high - low or max(abs(high) * 0.1, 1.0)
            self.setYRange(low - span * 0.08, high + span * 0.48, padding=0)
            pvalue = summary["paired_p_holm"]
            if np.isfinite(pvalue):
                self.note.setText(f"Holm p={pvalue:.3g} · n={summary['paired_n']}\nMedian line | Mean ± SEM")
                self.note.setPos(0.55, high + span * 0.40)
                self.note.show()
                self._note_wanted = True
                self._fit_note()
            else:
                # No test was possible: no "descriptive only" text on the plot.
                self.setYRange(low - span * 0.08, high + span * 0.10, padding=0)

    def _show_groups(self, result):
        """Group view chosen by the user (change, Post only, Pre only or paired Pre/Post).

        The view description carries the per-animal values, the statistics
        that match the representation and the annotation text.
        """
        from group_metric_views import view_from_comparison
        summary = result["summary"]
        view = result.get("group_view")
        if view is None:   # results attached by older callers: Post - Pre comparison only
            view = view_from_comparison(result["group_comparison"], f"{summary['metric']}: Post \u2212 Pre")
            view["zero_line"] = True
        self._show_view(result, view)

    def _show_windows(self, result, row_colors=None):
        """All metric windows in one panel (Pre, Post, Late, ...), rows joined by lines.

        The view description comes from ``window_comparison.build_window_view``:
        one column per window, the Friedman test in the note and one bracket
        per tested pair of windows. ``row_colors`` tints the points by
        recording group; the summaries then use the neutral text color.
        """
        view = result["window_view"]
        columns = view["columns"]
        if row_colors is not None and columns and len(row_colors) == len(columns[0]["values"]):
            neutral = self._text_color.name()
            view = {**view, "columns": [{**column, "color": neutral, "point_colors": list(row_colors)}
                                        for column in columns]}
        self._show_view(result, view)

    def _show_view(self, result, view):
        """Draw one view description (groups or windows) with its title, note and tooltip."""
        from group_metric_views import draw_group_view, view_note
        summary = result["summary"]
        self._group_items = draw_group_view(self, view, self._text_color.name(), self.summary_style)
        self.setTitle(view["title"])
        self._title_fit = True
        self._fit_title()
        self.setLabel("left", summary["units"])
        self.setToolTip(view["tooltip"])
        bounds = view.get("_drawn_bounds")
        note = view_note(view, self.summary_style)
        legend = view.get("legend") or []
        if bounds is not None and (note or legend):
            self._data_bounds = bounds
            if legend:
                # Overlaid groups: a colored legend line, then the statistics.
                import html
                entries = "&nbsp;&nbsp;&nbsp;".join(
                    f'<span style="color:{entry["color"]}">●</span> {html.escape(str(entry["name"]))}'
                    for entry in legend)
                self.note.setHtml(entries + "".join("<br>" + html.escape(line) for line in note.split("\n") if line))
            else:
                self.note.setText(note)
            self.note.show()
            self._note_wanted = True
            self._fit_note()
        elif bounds is not None:
            # Nothing was tested: no annotation, so the data use the full height.
            low, high = bounds
            span = high - low or max(abs(high) * 0.1, 1.0)
            self.setYRange(low - span * 0.08, high + span * 0.10, padding=0)
