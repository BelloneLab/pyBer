"""Compact paired-distribution panels with explicit median and mean markers."""
from __future__ import annotations

import numpy as np
import pyqtgraph as pg
from PySide6 import QtCore, QtGui, QtWidgets


class MetricGrid(QtWidgets.QWidget):
    """Use one legible column in a narrow results drawer and two when space allows."""
    columnsChanged = QtCore.Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.columns = 2

    def resizeEvent(self, event):
        super().resizeEvent(event)
        columns = 1 if self.width() < 650 else 2
        if columns != self.columns:
            self.columns = columns
            self.columnsChanged.emit()


class MetricPanel(pg.PlotWidget):
    """One metric per independently scaled card, shared by screen and image export."""

    def __init__(self, parent=None):
        super().__init__(parent=parent, title="PSTH metrics")
        self.setMinimumSize(230, 230)
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
        self.set_colors({"text": "#CED8E6", "accent": "#4D93BD"})
        self.getViewBox().sigResized.connect(self._fit_note)
        self.getViewBox().sigXRangeChanged.connect(self._fit_note)

    def _fit_note(self, *_args):
        """Keep the summary centered and legible when a metric card resizes."""
        if not self.note.isVisible():
            return
        view = self.getViewBox()
        self.note.setPos(float(np.mean(view.viewRange()[0])), self.note.pos().y())
        font = QtGui.QFont("Segoe UI", 8)
        self.note.setFont(font)
        width = self.note.boundingRect().width()
        available = max(1., view.sceneBoundingRect().width() - 12)
        if width > available:
            font.setPointSizeF(max(6., 8 * available / width))
            self.note.setFont(font)

    def set_colors(self, palette):
        """Respect the active plot palette without confusing summary encodings."""
        colors = [QtGui.QColor(palette["accent"]), QtGui.QColor("#D28A64")]
        text = palette["text"]
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

    def show_result(self, result):
        """Display paired native reductions and complete finite-row summaries."""
        self.result = result
        self.pairs.setData([], [])
        for item in self.points + self.medians + self.means:
            item.setData([], [])
        for item in self.errors:
            item.setData(x=np.array([]), y=np.array([]), top=np.array([]), bottom=np.array([]))
        self.note.hide()
        if result is None:
            self.setYRange(0, 1, padding=0)
            return
        summary = result["summary"]
        pre, post = result["pre_values"], result["post_values"]
        paired = np.isfinite(pre) & np.isfinite(post)
        jitter = np.linspace(-0.12, 0.12, len(pre))
        xp, yp = [], []
        for index in np.flatnonzero(paired):
            xp.extend([jitter[index], 1 + jitter[index], np.nan])
            yp.extend([pre[index], post[index], np.nan])
        self.pairs.setData(xp, yp, connect="finite")
        bounds = []
        for index, (name, values) in enumerate((("pre", pre), ("post", post))):
            valid = np.isfinite(values)
            self.points[index].setData(index + jitter[valid], values[valid])
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
            span = high - low or max(abs(high) * 0.1, 1.0)
            self.setYRange(low - span * 0.08, high + span * 0.48, padding=0)
            pvalue = summary["paired_p_holm"]
            test = f"Holm p={pvalue:.3g}" if np.isfinite(pvalue) else "Descriptive only"
            self.note.setText(f"{test} · n={summary['paired_n']}\nMedian line | Mean ± SEM")
            self.note.setPos(0.55, high + span * 0.40)
            self.note.show()
            self._fit_note()
