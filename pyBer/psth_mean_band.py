"""Draw the PSTH mean and its uncertainty consistently across dashboard panels."""
from __future__ import annotations

import pyqtgraph as pg
from PySide6 import QtGui
from psth_groups import SEM_ALPHA

MEAN_WIDTH_WITH_TRACES = 3.0     # pixels along the center of the SEM ribbon
HALO_EXTRA_WIDTH = 2.4           # a narrow separating stroke on either side
MIN_FILL_ALPHA = 115             # 0-255, when individual traces lie underneath
MIN_EDGE_ALPHA = 175
MIN_LUMA_DISTANCE = 80           # minimum difference on Qt's 0-255 gray scale


def sem_band(high: pg.PlotDataItem, low: pg.PlotDataItem) -> pg.FillBetweenItem:
    """Follow the displayed curve paths as they are clipped or resampled on zoom."""
    # PlotCurveItem emits when pyqtgraph updates the *displayed* samples.
    # PlotDataItem only signals when the source dataset is replaced.
    band = pg.FillBetweenItem(high.curve, low.curve)
    band.setZValue(-1)
    return band


def _visible_ink(color: QtGui.QColor, surface: QtGui.QColor) -> QtGui.QColor:
    """Keep the hue, tinting or shading only colors too close to the background."""
    ink = QtGui.QColor(color)
    level, backdrop = QtGui.qGray(ink.rgb()), QtGui.qGray(surface.rgb())
    if abs(level - backdrop) < MIN_LUMA_DISTANCE:
        end = 255 if backdrop < 128 else 0
        target = backdrop + (MIN_LUMA_DISTANCE if end else -MIN_LUMA_DISTANCE)
        fraction = (target - level) / (end - level)
        channels = [round(value + fraction * (end - value)) for value in ink.getRgb()[:3]]
        ink.setRgb(*channels, ink.alpha())
    return ink


def style_mean_band(mean: pg.PlotDataItem, band: pg.FillBetweenItem, mean_color: QtGui.QColor,
                    surface: QtGui.QColor, *, mean_width: float, combined: bool,
                    fill_color: QtGui.QColor | None = None, edge_color: QtGui.QColor | None = None) -> None:
    """Show an outlined filled SEM and a separated mean when trial traces are visible."""
    width = MEAN_WIDTH_WITH_TRACES if combined else mean_width
    mean.setPen(pg.mkPen(mean_color, width=width))
    mean.setZValue(2)
    # Group and behavior summaries inherit their identifying color. The pooled
    # mean can have independent fill and edge colors from Plot Styling.
    fill = QtGui.QColor(fill_color if fill_color is not None else mean_color)
    if fill_color is None:
        fill.setAlpha(SEM_ALPHA)
    if combined:
        # An opaque hairline casing keeps the chosen mean hue legible even when
        # several traces coincide. Fall back to dark/light ink for low-contrast
        # choices such as a yellow mean on white or a navy mean on black.
        halo = _visible_ink(surface, mean_color)
        mean.setShadowPen(pg.mkPen(halo, width=width + HALO_EXTRA_WIDTH))
        fill = _visible_ink(fill, surface)
        fill.setAlpha(max(fill.alpha(), MIN_FILL_ALPHA))
        edge = _visible_ink(edge_color if edge_color is not None else mean_color, surface)
        edge.setAlpha(max(edge_color.alpha(), MIN_EDGE_ALPHA) if edge_color is not None else MIN_EDGE_ALPHA)
        band.setPen(pg.mkPen(edge, width=1.0))
    else:
        mean.setShadowPen(None)
        band.setPen(None)
    band.setBrush(pg.mkBrush(fill))
