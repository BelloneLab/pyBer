"""Theme-aware, exportable rendering for the selectable behavior summary panel."""
from __future__ import annotations

import numpy as np
import pyqtgraph as pg
from PySide6 import QtCore, QtGui


def render_behavior_summary(plot, summary, palette):
    """Draw bins faithfully, with quiet file traces and a shaded uncertainty band.

    The numerical summary owns all statistics. This function only renders those
    values, never interpolating across missing bins or changing the export data.
    """
    plot.clear()
    title = {"Cumulative duration": "Cumulative duration", "cumulative": "Cumulative duration",
             "Cumulative bout count": "Cumulative bouts", "Onset-to-onset interval": "Onset intervals",
             "Bout duration over time": "Median bout duration"}.get(summary["metric"], summary["title"])
    plot.setTitle(title, color=palette["text"], size="11pt")
    # The compact side card still names seconds explicitly; the tooltip and
    # exported summary carry the complete recording-relative time definition.
    x_label = "Time from start (s)" if summary["kind"] == "timeseries" else summary["x_label"]
    plot.setLabel("bottom", x_label)
    plot.setLabel("left", summary["y_label"])
    plot.setToolTip("\n".join(summary.get("notes", [])))
    if not summary.get("has_data"):
        return
    edges = np.asarray(summary["edges"], float)
    values = np.asarray(summary["values"], float)
    sem = np.asarray(summary.get("sem", np.zeros_like(values)), float)
    accent = QtGui.QColor(palette["accent"])

    def color(alpha):
        tint = QtGui.QColor(accent)
        tint.setAlpha(alpha)
        return tint

    x = (edges[:-1] + edges[1:]) / 2
    histogram = summary["kind"] == "histogram"
    cumulative = summary["metric"] in ("cumulative", "Cumulative duration", "Cumulative bout count")
    duration_time = summary["metric"] == "Bout duration over time"
    if histogram:
        # Center bars within their actual numerical bins; leave breathing room.
        plot.addItem(pg.BarGraphItem(x=x, height=values, width=np.diff(edges) * .72,
                                     brush=pg.mkBrush(color(200)), pen=pg.mkPen(None)))
    else:
        if cumulative:
            # Accumulation starts at zero at the recording origin. Include that
            # anchor so a single-bin recording still has a visible line/area.
            x = edges.copy()
            values = np.r_[0., values]
            sem = np.r_[0., sem]
        for file_values in np.asarray(summary.get("per_file_values", []), float):
            if len(summary.get("file_ids", [])) > 1:
                if cumulative:
                    file_values = np.r_[0., file_values]
                plot.plot(x, file_values, pen=pg.mkPen(color(58), width=1), connect="finite")
        if cumulative:
            # Accrued observed time carries through gaps without inventing bouts.
            plot.plot(x, values, pen=pg.mkPen(None), fillLevel=0,
                      brush=pg.mkBrush(color(30)), connect="finite")
        elif not duration_time:
            plot.addItem(pg.BarGraphItem(x=x, height=values, width=np.diff(edges) * .56,
                                         brush=pg.mkBrush(color(95)), pen=pg.mkPen(None)))
        if duration_time:
            low = plot.plot(x, summary["lower"], pen=pg.mkPen(None), connect="finite")
            high = plot.plot(x, summary["upper"], pen=pg.mkPen(None), connect="finite")
            plot.addItem(pg.FillBetweenItem(low, high, brush=pg.mkBrush(color(42))))
            if values.size == 1:
                # A one-point fill has no area, so explicitly draw its IQR.
                plot.addItem(pg.ErrorBarItem(x=x, y=values, top=summary["upper"] - values,
                                             bottom=values - summary["lower"], beam=float(np.diff(edges)[0]) * .15,
                                             pen=pg.mkPen(color(150), width=1.2)))
        elif sem.shape == values.shape and np.any(np.isfinite(sem) & (sem > 0)):
            low = plot.plot(x, np.maximum(0, values - sem), pen=pg.mkPen(None), connect="finite")
            high = plot.plot(x, values + sem, pen=pg.mkPen(None), connect="finite")
            plot.addItem(pg.FillBetweenItem(low, high, brush=pg.mkBrush(color(42))))
        plot.plot(x, values, pen=pg.mkPen(accent, width=2), connect="finite",
                  symbol="o" if x.size <= 24 else None, symbolSize=4,
                  symbolPen=pg.mkPen(None), symbolBrush=pg.mkBrush(accent))
    tops = values + np.where(np.isfinite(sem), sem, 0) if sem.shape == values.shape else values
    if not histogram:
        tops = np.r_[tops, np.asarray(summary.get("per_file_values", []), float).reshape(-1)]
    if duration_time:
        tops = np.r_[tops, summary["upper"]]
    finite = tops[np.isfinite(tops)]
    maximum = float(np.max(finite)) if finite.size else 0
    if maximum <= 0:
        maximum = 1.
    plot.setXRange(float(edges[0]), float(edges[-1]), padding=.04)
    statistics = summary.get("statistics", {})
    median = statistics.get("median")
    if median is not None and np.isfinite(median):
        # The annotation is anchored in reserved chart headroom. Histograms use
        # raw unbinned observations; time curves use a defined per-file overall
        # rate, occupancy, duration median or final total, independent of bins.
        neutral = QtGui.QColor(palette["text"])
        neutral.setAlpha(150)
        if histogram:
            reference = pg.InfiniteLine(pos=median, angle=90, movable=False,
                                       pen=pg.mkPen(neutral, width=1, style=QtCore.Qt.PenStyle.DashLine))
            reference.setSpan(0, .78)
            reference.setZValue(5)
            plot.addItem(reference)
        prefix = "Final median" if cumulative else "Median"
        unit = statistics.get("unit", "")
        message = f"{prefix} {median:.3g} {unit}\nIQR {statistics['q1']:.3g} to {statistics['q3']:.3g}"
        annotation = pg.TextItem(message, color=palette["muted"], anchor=(1, 0))
        annotation.setFont(QtGui.QFont("Segoe UI", 8))
        annotation.setToolTip(statistics.get("definition", ""))
        annotation.setPos(float(edges[-1]), maximum * 1.27)
        annotation.setZValue(10)
        plot.addItem(annotation)
    plot.setYRange(0, maximum * (1.30 if median is not None else 1.12), padding=0)
    plot.enableAutoRange(axis=pg.ViewBox.XAxis, enable=False)
    plot.enableAutoRange(axis=pg.ViewBox.YAxis, enable=False)
