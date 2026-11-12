"""Editable vector export of pyqtgraph dashboards (PDF and SVG).

Why this module exists
----------------------
Rendering a Qt widget tree into ``QPdfWriter`` or ``QSvgGenerator`` produces
files that look right but are hard to edit in Illustrator or Inkscape:

* the QGraphicsView background cache is embedded as one bitmap behind the plot;
* thick pyqtgraph curves are drawn as thousands of independent 2-point segments;
* cosmetic pens become ``vector-effect="non-scaling-stroke"`` inside huge
  non-uniform transforms, which Illustrator ignores (strokes explode or vanish).

This module walks the *displayed* pyqtgraph items instead of their pixels and
rebuilds the same figure with matplotlib, whose PDF/SVG writers emit one path
per curve, real text objects (``pdf.fonttype=42`` / ``svg.fonttype='none'``)
and plain RGB fills. Every position is taken from the live Qt geometry, so the
layout, tick labels and axis text match the screen. Heatmaps stay images (an
image is the honest representation of a matrix) but sit in their own object.

Only presentation is read. No plot data, view range or analysis state changes.
"""
from __future__ import annotations

import logging
import os
import re
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pyqtgraph as pg
from PySide6 import QtCore, QtGui, QtWidgets

_LOG = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration (presentation constants, easy to tune)
# ---------------------------------------------------------------------------
SCREEN_DPI = 96.0                 # Qt logical pixels per inch on Windows
PX_TO_PT = 72.0 / SCREEN_DPI      # one logical pixel expressed in points
QT_TEXT_MARGIN_PX = 4.0           # QGraphicsTextItem document margin
FALLBACK_FONTS = ["Segoe UI", "Arial", "Helvetica", "DejaVu Sans"]
MAX_SCATTER_POINTS = 20000        # beyond this, per-point styles use the defaults

# matplotlib settings applied only while writing (never globally).
VECTOR_RC = {
    "pdf.fonttype": 42,          # embed TrueType: text stays editable text
    "ps.fonttype": 42,
    "svg.fonttype": "none",      # SVG <text> elements instead of glyph paths
    "svg.hashsalt": "pyber",     # deterministic ids for reproducible files
    "path.simplify": False,      # keep every displayed vertex
    "image.interpolation": "none",
    "savefig.transparent": False,
}

_SYMBOLS = {"o": "o", "s": "s", "t": "v", "t1": "^", "t2": ">", "t3": "<",
            "d": "D", "+": "+", "x": "x", "p": "p", "h": "h", "star": "*",
            "|": "|", "_": "_", "arrow_up": "^", "arrow_down": "v",
            "arrow_left": "<", "arrow_right": ">", "crosshair": "+"}


# ---------------------------------------------------------------------------
# Small Qt -> matplotlib conversions
# ---------------------------------------------------------------------------
def _rgba(color: QtGui.QColor) -> Tuple[float, float, float, float]:
    """QColor -> matplotlib RGBA tuple in 0..1."""
    c = QtGui.QColor(color)
    return (c.redF(), c.greenF(), c.blueF(), c.alphaF())


def _brush_rgba(brush) -> Optional[Tuple[float, float, float, float]]:
    """Return a fill color, or None for an empty/transparent brush."""
    if brush is None:
        return None
    if not isinstance(brush, QtGui.QBrush):
        try:
            brush = pg.mkBrush(brush)
        except Exception:
            return None
    if brush.style() == QtCore.Qt.BrushStyle.NoBrush:
        return None
    rgba = _rgba(brush.color())
    return rgba if rgba[3] > 0 else None


def _pen_kwargs(pen) -> Optional[Dict[str, object]]:
    """Translate a QPen to Line2D keyword arguments (None = invisible pen)."""
    if pen is None:
        return None
    if not isinstance(pen, QtGui.QPen):
        try:
            pen = pg.mkPen(pen)
        except Exception:
            return None
    if pen.style() == QtCore.Qt.PenStyle.NoPen:
        return None
    rgba = _rgba(pen.color())
    if rgba[3] <= 0:
        return None
    width_px = pen.widthF() if pen.widthF() > 0 else 1.0  # cosmetic 0 = 1 px
    style = pen.style()
    linestyle: object = "-"
    if style == QtCore.Qt.PenStyle.DashLine:
        linestyle = (0, (4, 2))
    elif style == QtCore.Qt.PenStyle.DotLine:
        linestyle = (0, (1, 2))
    elif style == QtCore.Qt.PenStyle.DashDotLine:
        linestyle = (0, (4, 2, 1, 2))
    elif style == QtCore.Qt.PenStyle.DashDotDotLine:
        linestyle = (0, (4, 2, 1, 2, 1, 2))
    elif style == QtCore.Qt.PenStyle.CustomDashLine:
        pattern = [max(0.1, float(v)) for v in pen.dashPattern()]
        linestyle = (0, tuple(pattern)) if pattern else "-"
    cap = {QtCore.Qt.PenCapStyle.FlatCap: "butt", QtCore.Qt.PenCapStyle.RoundCap: "round"}.get(
        pen.capStyle(), "projecting")
    join = {QtCore.Qt.PenJoinStyle.MiterJoin: "miter", QtCore.Qt.PenJoinStyle.RoundJoin: "round"}.get(
        pen.joinStyle(), "bevel")
    return {"color": rgba, "linewidth": width_px * PX_TO_PT, "linestyle": linestyle,
            "solid_capstyle": cap, "solid_joinstyle": join}


def _font_kwargs(font: Optional[QtGui.QFont]) -> Dict[str, object]:
    """QFont -> matplotlib text keywords (size in points, family fallbacks)."""
    font = QtGui.QFont(font) if font is not None else QtWidgets.QApplication.font()
    size = font.pointSizeF()
    if size <= 0:
        size = max(1.0, font.pixelSize() * PX_TO_PT)
    family = [font.family()] + [name for name in FALLBACK_FONTS if name != font.family()]
    weight = "bold" if font.weight() >= QtGui.QFont.Weight.DemiBold else "normal"
    style = "italic" if font.italic() else "normal"
    return {"fontsize": size, "fontfamily": family, "fontweight": weight, "fontstyle": style}


def _parse_label_size(text: object) -> Optional[float]:
    """pyqtgraph LabelItem sizes are strings such as '10pt' or '12px'."""
    match = re.match(r"\s*([\d.]+)\s*(pt|px)?\s*$", str(text or ""))
    if not match:
        return None
    value = float(match.group(1))
    return value * PX_TO_PT if match.group(2) == "px" else value


def _qimage_to_rgba(image: QtGui.QImage) -> Optional[np.ndarray]:
    """Copy a QImage into an (h, w, 4) uint8 array without touching the source."""
    if image is None or image.isNull():
        return None
    converted = image.convertToFormat(QtGui.QImage.Format.Format_RGBA8888)
    height, width = converted.height(), converted.width()
    buffer = np.frombuffer(converted.constBits(), dtype=np.uint8,
                           count=converted.bytesPerLine() * height)
    array = buffer.reshape(height, converted.bytesPerLine())[:, :width * 4]
    return array.reshape(height, width, 4).copy()


def _split_finite(x: np.ndarray, y: np.ndarray) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Contiguous runs where both coordinates are finite (for fills)."""
    finite = np.isfinite(x) & np.isfinite(y)
    runs = []
    if not finite.any():
        return runs
    edges = np.flatnonzero(np.diff(np.r_[0, finite.astype(np.int8), 0]))
    for start, stop in zip(edges[::2], edges[1::2]):
        runs.append((x[start:stop], y[start:stop]))
    return runs


# ---------------------------------------------------------------------------
# Coordinate mapping from a GraphicsView scene to figure fractions
# ---------------------------------------------------------------------------
class _ViewMapper:
    """Map scene points of one QGraphicsView to figure-fraction coordinates."""

    def __init__(self, root: QtWidgets.QWidget, view: QtWidgets.QGraphicsView):
        self.root_w = max(1.0, float(root.width()))
        self.root_h = max(1.0, float(root.height()))
        origin = view.viewport().mapTo(root, QtCore.QPoint(0, 0))
        self.dx, self.dy = float(origin.x()), float(origin.y())
        self.transform = view.viewportTransform()

    def px(self, scene_point: QtCore.QPointF) -> Tuple[float, float]:
        point = self.transform.map(QtCore.QPointF(scene_point))
        return point.x() + self.dx, point.y() + self.dy

    def fig(self, scene_point: QtCore.QPointF) -> Tuple[float, float]:
        x, y = self.px(scene_point)
        return x / self.root_w, 1.0 - y / self.root_h

    def fig_rect(self, scene_rect: QtCore.QRectF) -> Tuple[float, float, float, float]:
        """Return (left, bottom, width, height) in figure fractions."""
        x0, y1 = self.fig(scene_rect.topLeft())
        x1, y0 = self.fig(scene_rect.bottomRight())
        return min(x0, x1), min(y0, y1), abs(x1 - x0), abs(y1 - y0)


def _widget_fig_rect(root: QtWidgets.QWidget, widget: QtWidgets.QWidget) -> Tuple[float, float, float, float]:
    origin = widget.mapTo(root, QtCore.QPoint(0, 0))
    w, h = max(1.0, float(root.width())), max(1.0, float(root.height()))
    return origin.x() / w, 1.0 - (origin.y() + widget.height()) / h, widget.width() / w, widget.height() / h


# ---------------------------------------------------------------------------
# The exporter
# ---------------------------------------------------------------------------
class VectorFigureBuilder:
    """Translate the visible pyqtgraph content of ``root`` into a matplotlib Figure."""

    def __init__(self, root: QtWidgets.QWidget):
        self.root = root
        self.figure = None
        self.skipped: List[str] = []   # unknown item types (rasterized or ignored)
        self._gid_counts: Dict[str, int] = {}

    # -- public -----------------------------------------------------------
    def build(self):
        from matplotlib.figure import Figure

        root = self.root
        # Pending layout requests (e.g. a panel just restored) must settle so
        # the exported geometry is the geometry the user is about to see.
        QtCore.QCoreApplication.sendPostedEvents(None, QtCore.QEvent.Type.LayoutRequest)
        width = max(1, root.width() if root.width() > 0 else root.sizeHint().width())
        height = max(1, root.height() if root.height() > 0 else root.sizeHint().height())
        fig = Figure(figsize=(width / SCREEN_DPI, height / SCREEN_DPI), dpi=SCREEN_DPI)
        self.figure = fig
        views = self._visible_views()
        backgrounds = set()
        for view in views:
            color = _brush_rgba(view.backgroundBrush())
            if color is not None:
                backgrounds.add(tuple(round(v, 3) for v in color))
        # One shared surface keeps the export reading as a single figure.
        fig.patch.set_facecolor(next(iter(backgrounds)) if len(backgrounds) == 1 else (1, 1, 1, 0))
        for view in views:
            self._export_view(view)
        self._export_labels(views)
        return fig

    # -- discovery --------------------------------------------------------
    def _visible(self, widget: QtWidgets.QWidget) -> bool:
        if widget is self.root:
            return widget.width() > 0 and widget.height() > 0
        return widget.isVisibleTo(self.root) and widget.width() > 0 and widget.height() > 0

    def _visible_views(self) -> List[pg.GraphicsView]:
        candidates: List[QtWidgets.QWidget] = []
        if isinstance(self.root, QtWidgets.QGraphicsView):
            candidates.append(self.root)
        candidates.extend(self.root.findChildren(QtWidgets.QGraphicsView))
        seen, views = set(), []
        for view in candidates:
            if id(view) in seen or not self._visible(view) or view.scene() is None:
                continue
            seen.add(id(view))
            views.append(view)
        return views

    def _gid(self, base: str) -> str:
        """Readable, unique object ids for the SVG layer panel."""
        clean = re.sub(r"[^A-Za-z0-9_-]+", "_", base).strip("_") or "item"
        count = self._gid_counts.get(clean, 0)
        self._gid_counts[clean] = count + 1
        return clean if count == 0 else f"{clean}_{count}"

    # -- one QGraphicsView ------------------------------------------------
    def _export_view(self, view: QtWidgets.QGraphicsView) -> None:
        from matplotlib.patches import Rectangle

        fig = self.figure
        mapper = _ViewMapper(self.root, view)
        rect = _widget_fig_rect(self.root, view.viewport())
        color = _brush_rgba(view.backgroundBrush())
        if color is not None:
            fig.patches.append(Rectangle(rect[:2], rect[2], rect[3], transform=fig.transFigure,
                                         facecolor=color, edgecolor="none", zorder=-100,
                                         gid=self._gid("background")))
        scene = view.scene()
        plot_items = [item for item in scene.items()
                      if isinstance(item, pg.PlotItem) and item.isVisible()]
        handled_other = False
        for item in scene.items():
            if isinstance(item, pg.HistogramLUTItem) and item.isVisible():
                handled_other = True
        if not plot_items or handled_other:
            # Unknown composite widgets keep their exact appearance as one image.
            self._raster_widget(view, rect)
            if not plot_items:
                return
        for plot_item in plot_items:
            try:
                self._export_plot_item(plot_item, mapper)
            except Exception:  # never lose the whole export to one exotic item
                _LOG.exception("Vector export skipped a plot item")

    def _raster_widget(self, widget: QtWidgets.QWidget, rect) -> None:
        scale = 3.0
        image = QtGui.QImage(max(1, int(widget.width() * scale)), max(1, int(widget.height() * scale)),
                             QtGui.QImage.Format.Format_ARGB32)
        image.fill(QtCore.Qt.GlobalColor.transparent)
        painter = QtGui.QPainter(image)
        painter.scale(scale, scale)
        QtWidgets.QWidget.render(widget, painter, QtCore.QPoint(), QtGui.QRegion(),
                                 QtWidgets.QWidget.RenderFlag.DrawChildren)
        painter.end()
        array = _qimage_to_rgba(image)
        if array is None:
            return
        ax = self.figure.add_axes(rect, gid=self._gid("raster_widget"))
        ax.set_axis_off()
        ax.imshow(array, extent=(0, 1, 0, 1), aspect="auto", interpolation="none", zorder=-50)
        self.skipped.append(type(widget).__name__)

    # -- one PlotItem -----------------------------------------------------
    def _export_plot_item(self, plot_item: pg.PlotItem, mapper: _ViewMapper) -> None:
        vb = plot_item.getViewBox()
        title = ""
        try:
            title = plot_item.titleLabel.item.toPlainText().strip()
        except Exception:
            pass
        name = title or type(plot_item).__name__
        local = vb.boundingRect()
        left, bottom, width, height = mapper.fig_rect(vb.mapRectToScene(local))
        ax = self.figure.add_axes((left, bottom, max(width, 1e-6), max(height, 1e-6)),
                                  gid=self._gid(f"panel_{name}"))
        ax.set_axis_off()
        ax.patch.set_visible(False)
        top_left = vb.mapToView(local.topLeft())
        bottom_right = vb.mapToView(local.bottomRight())
        x0, x1 = top_left.x(), bottom_right.x()
        y_top, y_bottom = top_left.y(), bottom_right.y()
        if x0 == x1:
            x1 = x0 + 1.0
        if y_top == y_bottom:
            y_top = y_bottom + 1.0
        ax.set_xlim(x0, x1)
        ax.set_ylim(y_bottom, y_top)
        ax.set_autoscale_on(False)

        skip = set()
        region = getattr(plot_item, "region", None)   # ColorBarItem handles
        if region is not None:
            skip.add(id(region))
        items = [item for item in vb.addedItems if id(item) not in skip]
        items.sort(key=lambda item: item.zValue())
        for order, item in enumerate(items):
            try:
                self._export_item(ax, vb, item, zorder=1 + order * 0.01 + item.zValue() * 1e-4, name=name)
            except Exception:
                _LOG.exception("Vector export skipped %s", type(item).__name__)
                self.skipped.append(type(item).__name__)
        legend = getattr(plot_item, "legend", None)
        if legend is not None and legend.isVisible():
            self._export_legend(legend, mapper, name)
        self._export_plot_decorations(plot_item, mapper, name)
        for side in ("left", "bottom", "right", "top"):
            entry = plot_item.axes.get(side) if isinstance(plot_item.axes, dict) else None
            axis = entry.get("item") if entry else None
            if axis is not None and axis.isVisible():
                self._export_axis(axis, mapper, name)
        self._export_title(plot_item, mapper, name)

    # -- data items -------------------------------------------------------
    @staticmethod
    def _to_view(vb: pg.ViewBox, item, x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Apply the item's (rarely non-identity) transform into view coordinates."""
        transform, ok = item.itemTransform(vb.childGroup)
        if not ok or transform.isIdentity():
            return x, y
        xv = transform.m11() * x + transform.m21() * y + transform.dx()
        yv = transform.m12() * x + transform.m22() * y + transform.dy()
        return xv, yv

    def _export_item(self, ax, vb, item, zorder: float, name: str) -> None:
        if not item.isVisible():
            return
        if isinstance(item, pg.PlotDataItem):
            if item.curve.isVisible():
                self._export_curve(ax, vb, item.curve, zorder, name)
            if item.scatter.isVisible():
                self._export_scatter(ax, vb, item.scatter, zorder + 0.001, name)
        elif isinstance(item, pg.PlotCurveItem):
            self._export_curve(ax, vb, item, zorder, name)
        elif isinstance(item, pg.ScatterPlotItem):
            self._export_scatter(ax, vb, item, zorder, name)
        elif isinstance(item, pg.FillBetweenItem):
            self._export_fill_between(ax, vb, item, zorder, name)
        elif isinstance(item, pg.BarGraphItem):
            self._export_bars(ax, vb, item, zorder, name)
        elif isinstance(item, pg.ErrorBarItem):
            self._export_error_bars(ax, vb, item, zorder, name)
        elif isinstance(item, pg.LinearRegionItem):
            self._export_region(ax, item, zorder, name)
        elif isinstance(item, pg.InfiniteLine):
            self._export_infinite_line(ax, item, zorder, name)
        elif isinstance(item, pg.ImageItem):
            self._export_image_item(ax, vb, item, zorder, name)
        elif isinstance(item, pg.TextItem):
            self._export_text_item(ax, vb, item, zorder, name)
        elif isinstance(item, QtWidgets.QGraphicsPixmapItem):
            self._export_pixmap_item(ax, vb, item, zorder, name)
        elif isinstance(item, pg.LegendItem):
            return
        else:
            self._raster_item(ax, vb, item, zorder, name)

    def _export_curve(self, ax, vb, curve: pg.PlotCurveItem, zorder: float, name: str) -> None:
        x, y = curve.getData()
        if x is None or y is None or len(x) == 0:
            return
        x = np.asarray(x, float)
        y = np.asarray(y, float)
        x, y = self._to_view(vb, curve, x, y)
        opts = curve.opts
        step = opts.get("stepMode")
        if step in ("center", True) and x.size == y.size + 1:
            x = np.repeat(x, 2)[1:-1]
            y = np.repeat(y, 2)
        connect = opts.get("connect", "all")
        if isinstance(connect, np.ndarray) and connect.size == x.size:
            breaks = np.flatnonzero(~connect.astype(bool)[:-1]) + 1
            x = np.insert(x, breaks, np.nan)
            y = np.insert(y, breaks, np.nan)
        elif isinstance(connect, str) and connect == "pairs":
            n = (x.size // 2) * 2
            x = np.insert(x[:n].reshape(-1, 2), 2, np.nan, axis=1).ravel()
            y = np.insert(y[:n].reshape(-1, 2), 2, np.nan, axis=1).ravel()
        fill_level = opts.get("fillLevel")
        fill = _brush_rgba(opts.get("brush"))
        if fill is not None and fill_level is not None and np.isfinite(fill_level):
            finite = np.isfinite(x) & np.isfinite(y)
            ax.fill_between(x, y, float(fill_level), where=finite, interpolate=False,
                            facecolor=fill, edgecolor="none", linewidth=0, zorder=zorder - 0.0005,
                            gid=self._gid(f"{name}_area"))
        pen = _pen_kwargs(opts.get("pen"))
        if pen is None:
            return
        drawstyle = {"left": "steps-post", "right": "steps-pre"}.get(step, "default")
        ax.plot(x, y, zorder=zorder, drawstyle=drawstyle, gid=self._gid(f"{name}_curve"), **pen)

    def _export_scatter(self, ax, vb, scatter: pg.ScatterPlotItem, zorder: float, name: str) -> None:
        data = getattr(scatter, "data", None)
        if data is None or len(data) == 0:
            return
        visible = data["visible"].astype(bool) if "visible" in data.dtype.names else np.ones(len(data), bool)
        data = data[visible]
        if len(data) == 0:
            return
        x, y = self._to_view(vb, scatter, np.asarray(data["x"], float), np.asarray(data["y"], float))
        opts = scatter.opts
        sizes = np.asarray(data["size"], float)
        sizes = np.where(sizes < 0, float(opts.get("size", 7)), sizes)
        default_symbol = opts.get("symbol", "o")
        per_point = len(data) <= MAX_SCATTER_POINTS

        def resolve(values, default, convert):
            default_value = convert(default)
            if not per_point:
                return [default_value] * len(values)
            return [default_value if value is None else convert(value) for value in values]

        faces = resolve(data["brush"], opts.get("brush"), _brush_rgba)
        edge_pens = resolve(data["pen"], opts.get("pen"), _pen_kwargs)
        symbols = [s if isinstance(s, str) else default_symbol for s in
                   (data["symbol"] if per_point else [default_symbol] * len(data))]
        symbols = [s if s is not None else default_symbol for s in symbols]
        for symbol in sorted(set(map(str, symbols))):
            mask = np.array([str(s) == symbol for s in symbols])
            face = [faces[i] if faces[i] is not None else (0, 0, 0, 0) for i in np.flatnonzero(mask)]
            edge = [edge_pens[i]["color"] if edge_pens[i] else (0, 0, 0, 0) for i in np.flatnonzero(mask)]
            widths = [edge_pens[i]["linewidth"] if edge_pens[i] else 0.0 for i in np.flatnonzero(mask)]
            ax.scatter(x[mask], y[mask], s=(sizes[mask] * PX_TO_PT) ** 2, marker=_SYMBOLS.get(symbol, "o"),
                       c=face, edgecolors=edge, linewidths=widths, zorder=zorder,
                       gid=self._gid(f"{name}_points"))

    @staticmethod
    def _curve_data(vb, curve) -> Tuple[np.ndarray, np.ndarray]:
        x, y = curve.getData()
        if x is None or y is None:
            return np.array([]), np.array([])
        x, y = np.asarray(x, float), np.asarray(y, float)
        target = curve.curve if isinstance(curve, pg.PlotDataItem) else curve
        return VectorFigureBuilder._to_view(vb, target, x, y)

    def _export_fill_between(self, ax, vb, item: pg.FillBetweenItem, zorder: float, name: str) -> None:
        curves = getattr(item, "curves", None)
        if not curves or len(curves) != 2:
            return
        face = _brush_rgba(item.brush())
        if face is None:
            return
        (x1, y1), (x2, y2) = (self._curve_data(vb, curve) for curve in curves)
        if x1.size == 0 or x2.size == 0:
            return
        if x1.shape == x2.shape and np.allclose(x1, x2, equal_nan=True):
            where = np.isfinite(x1) & np.isfinite(y1) & np.isfinite(y2)
            ax.fill_between(x1, y1, y2, where=where, interpolate=False, facecolor=face,
                            edgecolor="none", linewidth=0, zorder=zorder, gid=self._gid(f"{name}_band"))
            return
        for (ax1, ay1), (ax2, ay2) in zip(_split_finite(x1, y1), _split_finite(x2, y2)):
            ax.fill(np.r_[ax1, ax2[::-1]], np.r_[ay1, ay2[::-1]], facecolor=face, edgecolor="none",
                    linewidth=0, zorder=zorder, gid=self._gid(f"{name}_band"))

    def _export_bars(self, ax, vb, item: pg.BarGraphItem, zorder: float, name: str) -> None:
        from matplotlib.collections import PatchCollection
        from matplotlib.patches import Rectangle

        x0, y0, x1, y1 = (np.atleast_1d(np.asarray(v, float)) for v in item._getNormalizedCoords())
        count = int(max(x0.size, y0.size, x1.size, y1.size))
        if count == 0:
            return
        x0, y0, x1, y1 = (np.broadcast_to(v, (count,)) for v in (x0, y0, x1, y1))
        brushes = item._brushes if getattr(item, "_brushes", None) is not None else [item._sharedBrush] * count
        pens = item._pens if getattr(item, "_pens", None) is not None else [item._sharedPen] * count
        rects, faces, edges, widths = [], [], [], []
        for index in range(count):
            if not all(np.isfinite(v[index]) for v in (x0, y0, x1, y1)):
                continue
            rects.append(Rectangle((x0[index], y0[index]), x1[index] - x0[index], y1[index] - y0[index]))
            faces.append(_brush_rgba(brushes[index]) or (0, 0, 0, 0))
            pen = _pen_kwargs(pens[index])
            edges.append(pen["color"] if pen else (0, 0, 0, 0))
            widths.append(pen["linewidth"] if pen else 0.0)
        if rects:
            ax.add_collection(PatchCollection(rects, facecolors=faces, edgecolors=edges,
                                              linewidths=widths, zorder=zorder,
                                              gid=self._gid(f"{name}_bars")))

    def _export_error_bars(self, ax, vb, item: pg.ErrorBarItem, zorder: float, name: str) -> None:
        from matplotlib.collections import LineCollection

        opts = item.opts
        if opts.get("x") is None or opts.get("y") is None:
            return
        x = np.atleast_1d(np.asarray(opts["x"], float))
        y = np.atleast_1d(np.asarray(opts["y"], float))
        pen = _pen_kwargs(opts.get("pen") if opts.get("pen") is not None else pg.getConfigOption("foreground"))
        if pen is None or x.size == 0:
            return
        beam = opts.get("beam")
        segments = []

        def arr(key):
            value = opts.get(key)
            return None if value is None else np.broadcast_to(np.asarray(value, float), x.shape)

        height, top, bottom = arr("height"), arr("top"), arr("bottom")
        if height is not None or top is not None or bottom is not None:
            if height is not None:
                lo, hi = y - height / 2.0, y + height / 2.0
            else:
                lo = y - bottom if bottom is not None else y
                hi = y + top if top is not None else y
            for xi, a, b in zip(x, lo, hi):
                segments.append([(xi, a), (xi, b)])
                if beam:
                    if height is not None or top is not None:
                        segments.append([(xi - beam / 2.0, b), (xi + beam / 2.0, b)])
                    if height is not None or bottom is not None:
                        segments.append([(xi - beam / 2.0, a), (xi + beam / 2.0, a)])
        width, right, left = arr("width"), arr("right"), arr("left")
        if width is not None or right is not None or left is not None:
            if width is not None:
                lo, hi = x - width / 2.0, x + width / 2.0
            else:
                lo = x - left if left is not None else x
                hi = x + right if right is not None else x
            for yi, a, b in zip(y, lo, hi):
                segments.append([(a, yi), (b, yi)])
                if beam:
                    if width is not None or right is not None:
                        segments.append([(b, yi - beam / 2.0), (b, yi + beam / 2.0)])
                    if width is not None or left is not None:
                        segments.append([(a, yi - beam / 2.0), (a, yi + beam / 2.0)])
        segments = [seg for seg in segments if np.all(np.isfinite(seg))]
        if segments:
            ax.add_collection(LineCollection(segments, colors=[pen["color"]], linewidths=pen["linewidth"],
                                             zorder=zorder, gid=self._gid(f"{name}_errorbars")))

    def _export_region(self, ax, item: pg.LinearRegionItem, zorder: float, name: str) -> None:
        low, high = item.getRegion()
        vertical = item.orientation in ("vertical", getattr(pg.LinearRegionItem, "Vertical", "vertical"))
        face = _brush_rgba(getattr(item, "brush", None))
        if face is not None:
            span = ax.axvspan if vertical else ax.axhspan
            span(low, high, facecolor=face, edgecolor="none", linewidth=0, zorder=zorder,
                 gid=self._gid(f"{name}_region"))
        for line in getattr(item, "lines", []):
            if line.isVisible():
                self._export_infinite_line(ax, line, zorder + 0.0001, name)

    def _export_infinite_line(self, ax, item: pg.InfiniteLine, zorder: float, name: str) -> None:
        pen = _pen_kwargs(getattr(item, "pen", None))
        if pen is None:
            return
        angle = float(getattr(item, "angle", 90)) % 180
        value = item.value()
        if abs(angle - 90) < 1e-6:
            ax.axvline(float(value), zorder=zorder, gid=self._gid(f"{name}_vline"), **pen)
        elif abs(angle) < 1e-6:
            ax.axhline(float(value), zorder=zorder, gid=self._gid(f"{name}_hline"), **pen)
        else:  # oblique line through its position
            pos = item.pos()
            slope = np.tan(np.radians(angle))
            ax.axline((pos.x(), pos.y()), slope=slope, zorder=zorder, gid=self._gid(f"{name}_line"), **pen)

    def _place_image(self, ax, vb, item, array: np.ndarray, width: float, height: float,
                     zorder: float, name: str, offset: QtCore.QPointF = QtCore.QPointF(0, 0)) -> None:
        p00 = vb.mapFromItemToView(item, QtCore.QPointF(offset.x(), offset.y()))
        p11 = vb.mapFromItemToView(item, QtCore.QPointF(offset.x() + width, offset.y() + height))
        # Row 0 of the QImage is drawn at item y=0, so it sits at p00.y().
        ax.imshow(array, extent=(p00.x(), p11.x(), p11.y(), p00.y()), origin="upper",
                  aspect="auto", interpolation="none", zorder=zorder, alpha=float(item.opacity()),
                  gid=self._gid(f"{name}_image"))

    def _export_image_item(self, ax, vb, item: pg.ImageItem, zorder: float, name: str) -> None:
        if item.image is None:
            return
        if item.qimage is None or getattr(item, "_renderRequired", False):
            item.render()
        array = _qimage_to_rgba(item.qimage)
        if array is None:
            return
        # The QImage is the exact lookup-table output the user sees.
        self._place_image(ax, vb, item, array, array.shape[1], array.shape[0], zorder, name)

    def _export_pixmap_item(self, ax, vb, item: QtWidgets.QGraphicsPixmapItem, zorder: float, name: str) -> None:
        pixmap = item.pixmap()
        if pixmap.isNull():
            return
        array = _qimage_to_rgba(pixmap.toImage())
        if array is None:
            return
        self._place_image(ax, vb, item, array, pixmap.width(), pixmap.height(), zorder, name,
                          offset=item.offset())

    def _export_text_item(self, ax, vb, item: pg.TextItem, zorder: float, name: str) -> None:
        from matplotlib.transforms import offset_copy

        text = item.textItem.toPlainText()
        if not text.strip():
            return
        parent = item.parentItem()
        if parent is None or parent is vb.childGroup:
            pos = item.pos()   # addItem() parents text to the view's child group
        else:
            pos = vb.mapFromItemToView(parent, item.pos())
        anchor = item.anchor
        ax_frac, ay_frac = float(anchor[0]), float(anchor[1])
        ha = "left" if ax_frac < 0.25 else "right" if ax_frac > 0.75 else "center"
        va = "top" if ay_frac < 0.25 else "bottom" if ay_frac > 0.75 else "center"
        margin = QT_TEXT_MARGIN_PX * PX_TO_PT
        dx = margin if ha == "left" else -margin if ha == "right" else 0.0
        dy = -margin if va == "top" else margin if va == "bottom" else 0.0
        transform = offset_copy(ax.transData, fig=self.figure, x=dx, y=dy, units="points")
        kwargs = _font_kwargs(item.textItem.font())
        bbox = None
        fill = _brush_rgba(getattr(item, "fill", None))
        if fill is not None:
            bbox = {"facecolor": fill, "edgecolor": "none", "pad": 1.5}
        ax.text(pos.x(), pos.y(), text, transform=transform, ha=ha, va=va, multialignment="left",
                color=_rgba(item.color), rotation=float(getattr(item, "angle", 0) or 0),
                rotation_mode="anchor", zorder=zorder, clip_on=True, bbox=bbox,
                gid=self._gid(f"{name}_text"), **kwargs)

    def _raster_item(self, ax, vb, item, zorder: float, name: str) -> None:
        """Last resort for custom items: paint just this item into an image."""
        rect = item.boundingRect()
        if rect.isEmpty():
            return
        scale = 3.0
        view_rect = vb.mapFromItemToView(item, rect).boundingRect()
        image = QtGui.QImage(max(1, int(rect.width() * scale) if rect.width() < 4000 else 4000),
                             max(1, int(rect.height() * scale) if rect.height() < 4000 else 4000),
                             QtGui.QImage.Format.Format_ARGB32)
        image.fill(QtCore.Qt.GlobalColor.transparent)
        painter = QtGui.QPainter(image)
        try:
            painter.scale(image.width() / rect.width(), image.height() / rect.height())
            painter.translate(-rect.topLeft())
            item.paint(painter, QtWidgets.QStyleOptionGraphicsItem(), None)
        finally:
            painter.end()
        array = _qimage_to_rgba(image)
        if array is None:
            return
        ax.imshow(array, extent=(view_rect.left(), view_rect.right(), view_rect.bottom(), view_rect.top()),
                  origin="upper", aspect="auto", interpolation="none", zorder=zorder,
                  gid=self._gid(f"{name}_raster"))
        self.skipped.append(type(item).__name__)

    # -- decorations (positions taken from the live scene) ------------------
    def _fig_text(self, x: float, y: float, text: str, **kwargs) -> None:
        self.figure.text(x, y, text, **kwargs)

    def _export_plot_decorations(self, plot_item: pg.PlotItem, mapper: _ViewMapper, name: str) -> None:
        """Plain rectangles parented to the PlotItem (e.g. group strips beside a heatmap)."""
        from matplotlib.patches import Rectangle

        for child in plot_item.childItems():
            if type(child) is not QtWidgets.QGraphicsRectItem or not child.isVisible():
                continue
            rect = child.mapRectToScene(child.rect())
            if rect.isEmpty():
                continue
            face = _brush_rgba(child.brush())
            pen = _pen_kwargs(child.pen())
            if face is None and pen is None:
                continue
            left, bottom, width, height = mapper.fig_rect(rect)
            self.figure.patches.append(Rectangle(
                (left, bottom), width, height, transform=self.figure.transFigure,
                facecolor=face or "none", edgecolor=pen["color"] if pen else "none",
                linewidth=pen["linewidth"] if pen else 0, zorder=5, gid=self._gid(f"{name}_group_strip")))

    def _export_axis(self, axis: pg.AxisItem, mapper: _ViewMapper, name: str) -> None:
        from matplotlib.lines import Line2D

        image = QtGui.QImage(4, 4, QtGui.QImage.Format.Format_ARGB32)
        # Measure tick text at the screen's DPI so label thinning matches the view.
        dpi = QtWidgets.QApplication.primaryScreen().logicalDotsPerInch() if QtWidgets.QApplication.primaryScreen() else SCREEN_DPI
        image.setDotsPerMeterX(int(round(dpi / 0.0254)))
        image.setDotsPerMeterY(int(round(dpi / 0.0254)))
        painter = QtGui.QPainter(image)
        try:
            specs = axis.generateDrawSpecs(painter)
        finally:
            painter.end()
        if not specs:
            return
        axis_spec, tick_specs, text_specs = specs
        fig = self.figure
        side = axis.orientation
        gid = self._gid(f"{name}_axis_{side}")
        lines = [(axis_spec[0], axis_spec[1], axis_spec[2])] + list(tick_specs)
        for pen, p1, p2 in lines:
            kwargs = _pen_kwargs(pen)
            if kwargs is None:
                continue
            a = mapper.fig(axis.mapToScene(QtCore.QPointF(p1)))
            b = mapper.fig(axis.mapToScene(QtCore.QPointF(p2)))
            kwargs["solid_capstyle"] = "butt"
            fig.add_artist(Line2D([a[0], b[0]], [a[1], b[1]], transform=fig.transFigure,
                                  zorder=5, gid=gid, **kwargs))
        font = axis.style.get("tickFont") or QtWidgets.QApplication.font()
        font_kwargs = _font_kwargs(font)
        color = _rgba(axis.textPen().color())
        for rect, _flags, text in text_specs:
            scene_rect = axis.mapRectToScene(QtCore.QRectF(rect))
            if side == "left":
                x, y = mapper.fig(QtCore.QPointF(scene_rect.right(), scene_rect.center().y()))
                ha, va = "right", "center"
            elif side == "right":
                x, y = mapper.fig(QtCore.QPointF(scene_rect.left(), scene_rect.center().y()))
                ha, va = "left", "center"
            elif side == "top":
                x, y = mapper.fig(QtCore.QPointF(scene_rect.center().x(), scene_rect.bottom()))
                ha, va = "center", "bottom"
            else:
                x, y = mapper.fig(QtCore.QPointF(scene_rect.center().x(), scene_rect.top()))
                ha, va = "center", "top"
            self._fig_text(x, y, text, ha=ha, va=va, color=color, zorder=6, gid=gid, **font_kwargs)
        label = getattr(axis, "label", None)
        if label is not None and label.isVisible():
            text = label.toPlainText().strip()
            if text:
                scene_rect = label.mapRectToScene(label.boundingRect())
                x, y = mapper.fig(scene_rect.center())
                rotation = 90 if side in ("left", "right") else 0
                self._fig_text(x, y, text, ha="center", va="center", rotation=rotation,
                               rotation_mode="anchor", color=_rgba(label.defaultTextColor()), zorder=6,
                               gid=self._gid(f"{name}_axis_label_{side}"), **_font_kwargs(label.font()))

    def _export_title(self, plot_item: pg.PlotItem, mapper: _ViewMapper, name: str) -> None:
        label = getattr(plot_item, "titleLabel", None)
        if label is None or not label.isVisible():
            return
        item = label.item
        text = item.toPlainText().strip()
        if not text:
            return
        scene_rect = item.mapRectToScene(item.boundingRect())
        if scene_rect.height() <= 1 or label.geometry().height() <= 1:
            return
        x, y = mapper.fig(QtCore.QPointF(scene_rect.left() + QT_TEXT_MARGIN_PX, scene_rect.center().y()))
        font_kwargs = _font_kwargs(item.font())
        size = _parse_label_size(label.opts.get("size"))
        if size:
            font_kwargs["fontsize"] = size
        if label.opts.get("bold"):
            font_kwargs["fontweight"] = "bold"
        color = label.opts.get("color")
        rgba = _rgba(pg.mkColor(color)) if color is not None else _rgba(item.defaultTextColor())
        self._fig_text(x, y, text, ha="left", va="center", color=rgba, zorder=6,
                       gid=self._gid(f"title_{name}"), **font_kwargs)

    def _export_legend(self, legend: pg.LegendItem, mapper: _ViewMapper, name: str) -> None:
        from matplotlib.lines import Line2D
        from matplotlib.patches import Rectangle

        fig = self.figure
        gid = self._gid(f"{name}_legend")
        background = _brush_rgba(legend.opts.get("brush")) if hasattr(legend, "opts") else None
        if background is not None:
            left, bottom, width, height = mapper.fig_rect(legend.mapRectToScene(legend.boundingRect()))
            fig.patches.append(Rectangle((left, bottom), width, height, transform=fig.transFigure,
                                         facecolor=background, edgecolor="none", zorder=6.5, gid=gid))
        for sample, label in list(getattr(legend, "items", [])):
            if not (sample.isVisible() and label.isVisible()):
                continue
            source = getattr(sample, "item", None)
            rect = sample.mapRectToScene(sample.boundingRect())
            y_mid = rect.center().y()
            a = mapper.fig(QtCore.QPointF(rect.left() + 2, y_mid))
            b = mapper.fig(QtCore.QPointF(rect.right() - 2, y_mid))
            opts = getattr(source, "opts", {}) if source is not None else {}
            pen = _pen_kwargs(opts.get("pen")) if isinstance(opts, dict) else None
            if pen is None and isinstance(source, pg.BarGraphItem):
                fill = _brush_rgba(getattr(source, "_sharedBrush", None))
                if fill is not None:
                    pen = {"color": fill, "linewidth": 6 * PX_TO_PT}
            if pen is not None:
                fig.add_artist(Line2D([a[0], b[0]], [a[1], b[1]], transform=fig.transFigure,
                                      zorder=7, gid=gid, **pen))
            symbol = opts.get("symbol") if isinstance(opts, dict) else None
            if symbol is not None:
                face = _brush_rgba(opts.get("symbolBrush"))
                fig.add_artist(Line2D([(a[0] + b[0]) / 2], [a[1]], transform=fig.transFigure,
                                      marker=_SYMBOLS.get(str(symbol), "o"), linestyle="none",
                                      markersize=float(opts.get("symbolSize", 7)) * PX_TO_PT,
                                      markerfacecolor=face or "none", markeredgecolor="none",
                                      zorder=7, gid=gid))
            text_item = label.item
            text = text_item.toPlainText().strip()
            if text:
                text_rect = text_item.mapRectToScene(text_item.boundingRect())
                x, y = mapper.fig(QtCore.QPointF(text_rect.left() + QT_TEXT_MARGIN_PX, text_rect.center().y()))
                color = label.opts.get("color") if hasattr(label, "opts") else None
                rgba = _rgba(pg.mkColor(color)) if color is not None else _rgba(text_item.defaultTextColor())
                font_kwargs = _font_kwargs(text_item.font())
                size = _parse_label_size(label.opts.get("size")) if hasattr(label, "opts") else None
                if size:
                    font_kwargs["fontsize"] = size
                self._fig_text(x, y, text, ha="left", va="center", color=rgba, zorder=7, gid=gid, **font_kwargs)

    def _export_labels(self, views: Sequence[QtWidgets.QWidget]) -> None:
        """Visible QLabel captions outside plots (card titles, notes)."""
        for label in self.root.findChildren(QtWidgets.QLabel):
            if not self._visible(label) or not label.text().strip():
                continue
            if any(view.isAncestorOf(label) for view in views):
                continue
            text = label.text()
            if label.textFormat() != QtCore.Qt.TextFormat.PlainText:
                document = QtGui.QTextDocument()
                document.setHtml(text)
                text = document.toPlainText()
            left, bottom, width, height = _widget_fig_rect(self.root, label)
            align = label.alignment()
            if align & QtCore.Qt.AlignmentFlag.AlignRight:
                x, ha = left + width, "right"
            elif align & QtCore.Qt.AlignmentFlag.AlignHCenter:
                x, ha = left + width / 2, "center"
            else:
                x, ha = left, "left"
            color = _rgba(label.palette().color(label.foregroundRole()))
            self._fig_text(x, bottom + height / 2, text, ha=ha, va="center", color=color,
                           gid=self._gid("caption"), **_font_kwargs(label.font()))


# ---------------------------------------------------------------------------
# Convenience wrappers used by the application
# ---------------------------------------------------------------------------
def build_vector_figure(root: QtWidgets.QWidget):
    """Return a matplotlib Figure reproducing the visible plots of ``root``."""
    return VectorFigureBuilder(root).build()


def write_vector_figure(root: QtWidgets.QWidget, path: str, figure=None) -> bool:
    """Write ``root`` to an editable PDF or SVG (format chosen by extension)."""
    import matplotlib
    from matplotlib.backends.backend_agg import FigureCanvasAgg

    extension = os.path.splitext(path)[1].lower().lstrip(".")
    if extension not in ("pdf", "svg"):
        raise ValueError(f"Unsupported vector format: {extension}")
    logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)
    with matplotlib.rc_context(VECTOR_RC):
        fig = figure if figure is not None else build_vector_figure(root)
        FigureCanvasAgg(fig)
        fig.savefig(path, format=extension, dpi=SCREEN_DPI, facecolor=fig.get_facecolor(),
                    metadata={"Creator": "pyBer"} if extension == "pdf" else None)
    return os.path.isfile(path) and os.path.getsize(path) > 0


def export_vector_files(root: QtWidgets.QWidget, base_path: str,
                        formats: Iterable[str] = ("pdf", "svg")) -> Dict[str, str]:
    """Build the figure once and write each requested vector format."""
    import matplotlib

    written: Dict[str, str] = {}
    formats = [fmt.lower().lstrip(".") for fmt in formats]
    if not formats:
        return written
    with matplotlib.rc_context(VECTOR_RC):
        figure = build_vector_figure(root)
    for fmt in formats:
        path = f"{base_path}.{fmt}"
        if write_vector_figure(root, path, figure=figure):
            written[fmt] = path
    return written
