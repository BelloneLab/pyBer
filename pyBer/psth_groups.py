"""Group-aware presentation of PSTH matrices (pure functions + plot helpers).

Rows of a displayed PSTH matrix are either recordings (animal level, label =
file id) or trials (label = ``"<file id> | Trial j"``). Given recording-group
assignments these helpers select, reorder and summarize rows. Statistics use
the same ``mean_sem`` as the pooled view, so a one-group view of a group is
numerically identical to the pooled view of those recordings.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pyqtgraph as pg
from PySide6 import QtCore, QtGui, QtWidgets

from postprocessing_core import mean_sem
from recording_groups import UNGROUPED_COLOR, UNGROUPED_LABEL

# Presentation constants
BAND_WIDTH_PX = 6        # colored group strip drawn left of the heatmap
BAND_GAP_PX = 3          # gap between the strip and the plotting area
SEPARATOR_WIDTH = 1.5    # px line between group blocks in the heatmap
GROUP_LINE_WIDTH = 1.8   # px mean line per group
SEM_ALPHA = 60           # 0-255 alpha of the per-group SEM band


@dataclass
class GroupBlock:
    """Contiguous rows of one group inside a (reordered) display matrix."""
    group_id: Optional[str]
    name: str
    color: str
    start: int
    stop: int
    n_files: int = 0

    @property
    def n_rows(self) -> int:
        return self.stop - self.start


@dataclass
class GroupDisplay:
    """Result of applying a group view to a display matrix."""
    mode: str                               # "all" | "only"
    matrix: np.ndarray
    labels: List[str]
    blocks: List[GroupBlock] = field(default_factory=list)
    row_colors: List[str] = field(default_factory=list)
    level: str = "animals"

    def matches(self, matrix: np.ndarray) -> bool:
        """True when this presentation describes ``matrix`` (same rows)."""
        try:
            return int(np.asarray(matrix).shape[0]) == int(self.matrix.shape[0]) and (
                sum(block.n_rows for block in self.blocks) == int(self.matrix.shape[0]))
        except Exception:
            return False


def row_file_id(label: object) -> str:
    """File id behind a row label for both animal and trial levels."""
    text = str(label)
    return text.split(" | ", 1)[0] if " | " in text else text


def build_group_display(matrix: np.ndarray, labels: Sequence[str], mode: str,
                        groups: Sequence[Dict[str, str]], group_of: Callable[[str], Optional[str]],
                        only_group: Optional[str] = None, level: str = "animals",
                        include_ungrouped: bool = True,
                        hidden_groups: Sequence[str] = ()) -> Optional[GroupDisplay]:
    """Select/reorder rows by group.

    ``hidden_groups``: ids of groups left out of ``mode='all'`` ('' stands for
    the ungrouped rows). Their rows are not drawn; nothing is recomputed.

    ``mode='all'``: rows are stably reordered into group blocks (group order,
    then ungrouped). ``mode='only'``: rows of ``only_group`` are kept. Returns
    None when the labels do not describe the matrix rows (nothing is guessed).
    """
    matrix = np.asarray(matrix, float)
    labels = list(labels)
    if matrix.ndim != 2 or len(labels) != matrix.shape[0]:
        return None
    file_ids = [row_file_id(label) for label in labels]
    order = [group["id"] for group in groups]
    by_id = {group["id"]: group for group in groups}
    row_group = [group_of(fid) for fid in file_ids]
    row_group = [gid if gid in by_id else None for gid in row_group]
    if mode == "only":
        if only_group not in by_id:
            return None
        keep = [i for i, gid in enumerate(row_group) if gid == only_group]
        group = by_id[only_group]
        blocks = [GroupBlock(only_group, group["name"], group["color"], 0, len(keep),
                             len({file_ids[i] for i in keep}))] if keep else []
        return GroupDisplay("only", matrix[keep] if keep else matrix[:0], [labels[i] for i in keep],
                            blocks, [group["color"]] * len(keep), level)
    blocks: List[GroupBlock] = []
    index_order: List[int] = []
    hidden = {str(value or "") for value in hidden_groups}
    for gid in order + ([None] if include_ungrouped else []):
        rows = [i for i, value in enumerate(row_group) if value == gid]
        if not rows or str(gid or "") in hidden:
            continue
        start = len(index_order)
        index_order.extend(rows)
        group = by_id.get(gid)
        blocks.append(GroupBlock(gid, group["name"] if group else UNGROUPED_LABEL,
                                 group["color"] if group else UNGROUPED_COLOR,
                                 start, len(index_order), len({file_ids[i] for i in rows})))
    row_colors = []
    for block in blocks:
        row_colors.extend([block.color] * block.n_rows)
    return GroupDisplay("all", matrix[index_order] if index_order else matrix[:0],
                        [labels[i] for i in index_order], blocks, row_colors, level)


def block_summaries(display: GroupDisplay) -> List[Tuple[GroupBlock, np.ndarray, np.ndarray, np.ndarray]]:
    """Mean, SEM and contributing count per group block."""
    out = []
    for block in display.blocks:
        sub = display.matrix[block.start:block.stop]
        if sub.size == 0:
            continue
        mean, sem, counts = mean_sem(sub)
        out.append((block, mean, sem, counts))
    return out


def legend_label(block: GroupBlock, level: str) -> str:
    """'WT (n=12)' for recordings; trials name their unit explicitly."""
    if level == "trials":
        return f"{block.name} (n={block.n_rows} trials, {block.n_files} rec.)"
    return f"{block.name} (n={block.n_rows})"


# ---------------------------------------------------------------------------
# Heatmap decorations: a colored strip left of the plot + block separators
# ---------------------------------------------------------------------------
class HeatmapGroupBands(QtCore.QObject):
    """Keep colored group strips beside the heatmap, outside the data area.

    The strips are QGraphicsRectItems parented to the PlotItem (not the view),
    so they never cover data; they follow resize, zoom and pan.
    """

    def __init__(self, plot: pg.PlotWidget, separator_color: str = "#111d30", parent=None) -> None:
        super().__init__(parent)
        self.plot = plot
        self.plot_item = plot.getPlotItem()
        self.view = self.plot_item.getViewBox()
        self.separator_color = separator_color
        self.blocks: List[GroupBlock] = []
        self._rects: List[QtWidgets.QGraphicsRectItem] = []
        self._lines: List[pg.InfiniteLine] = []
        self.view.sigResized.connect(self._reposition)
        self.view.sigRangeChanged.connect(self._reposition)

    def ensure_axis_offset(self) -> None:
        """Keep tick labels clear of the strips (theme styling resets the offset)."""
        axis = self.plot_item.getAxis("left")
        offset = 6 + (BAND_WIDTH_PX + BAND_GAP_PX + 3 if self.blocks else 0)
        try:
            if axis.style["tickTextOffset"][0] != offset:
                axis.setStyle(tickTextOffset=offset)
        except Exception:
            pass

    def set_blocks(self, blocks: Sequence[GroupBlock]) -> None:
        self.clear()
        self.blocks = list(blocks)
        self.ensure_axis_offset()
        if not self.blocks:
            return
        for block in self.blocks:
            rect = QtWidgets.QGraphicsRectItem(self.plot_item)
            rect.setPen(QtGui.QPen(QtCore.Qt.PenStyle.NoPen))
            rect.setBrush(QtGui.QColor(block.color))
            rect.setZValue(20)
            rect.setToolTip(f"{block.name}: {block.n_rows} row(s)")
            self._rects.append(rect)
        for block in self.blocks[1:]:
            line = pg.InfiniteLine(pos=float(block.start), angle=0, movable=False,
                                   pen=pg.mkPen(self.separator_color, width=SEPARATOR_WIDTH))
            line.setZValue(25)
            self.plot.addItem(line)
            self._lines.append(line)
        self._reposition()

    def set_separator_color(self, color: str) -> None:
        self.separator_color = color
        for line in self._lines:
            line.setPen(pg.mkPen(color, width=SEPARATOR_WIDTH))

    def clear(self) -> None:
        for rect in self._rects:
            scene = rect.scene()
            if scene is not None:
                scene.removeItem(rect)
            rect.setParentItem(None)
        for line in self._lines:
            self.plot.removeItem(line)
        self._rects, self._lines, self.blocks = [], [], []
        try:
            self.plot_item.getAxis("left").setStyle(tickTextOffset=6)
        except Exception:
            pass

    def _reposition(self, *_args) -> None:
        if not self._rects:
            return
        self.ensure_axis_offset()
        view_rect = self.view.mapRectToItem(self.plot_item, self.view.boundingRect())
        x1 = view_rect.left() - BAND_GAP_PX
        x0 = x1 - BAND_WIDTH_PX
        for rect, block in zip(self._rects, self.blocks):
            top = self.view.mapViewToScene(QtCore.QPointF(0, float(block.stop)))
            bottom = self.view.mapViewToScene(QtCore.QPointF(0, float(block.start)))
            y_a = self.plot_item.mapFromScene(top).y()
            y_b = self.plot_item.mapFromScene(bottom).y()
            y0 = max(min(y_a, y_b), view_rect.top())
            y1 = min(max(y_a, y_b), view_rect.bottom())
            rect.setVisible(y1 > y0)
            rect.setRect(QtCore.QRectF(x0, y0, BAND_WIDTH_PX, max(0.0, y1 - y0 - 1.0)))


def block_ticks(blocks: Sequence[GroupBlock], level: str, max_chars: int = 9) -> List[Tuple[float, str]]:
    """One y tick per group block at its center. Names are shortened to fit
    the fixed heatmap axis (the legend and tooltips carry the full name and n)."""
    ticks = []
    for block in blocks:
        center = (block.start + block.stop) / 2.0
        name = block.name if len(block.name) <= max_chars else block.name[:max_chars - 1] + "."
        ticks.append((center, name))
    return ticks


def draw_group_comparison(plot: pg.PlotWidget, comparison: Dict[str, object], text_color: str,
                          style: str = "points") -> List[pg.GraphicsObject]:
    """One column per group of per-recording values, in the chosen summary style.

    Thin wrapper kept for callers that hold a plain between-group comparison
    (global signal metrics). Returns the created items so callers can remove
    them before the next draw.
    """
    from group_metric_views import draw_group_view, view_from_comparison
    return draw_group_view(plot, view_from_comparison(comparison), text_color, style)


# ---------------------------------------------------------------------------
# Legend placement: the corner that hides the least of the curves
# ---------------------------------------------------------------------------
#: Candidate positions, in order of preference when several are equally free:
#: (name, horizontal anchor 0..1, vertical anchor 0..1 from the top).
LEGEND_POSITIONS: Tuple[Tuple[str, float, float], ...] = (
    ("top right", 1.0, 0.0), ("top left", 0.0, 0.0),
    ("bottom right", 1.0, 1.0), ("bottom left", 0.0, 1.0),
    ("top center", 0.5, 0.0), ("bottom center", 0.5, 1.0),
)
LEGEND_MARGIN_PX = 8            # gap between the legend and the plot edge
LEGEND_SAMPLES_PER_CURVE = 600  # points tested per curve (dense traces are thinned)


def legend_overlap_counts(curves: Sequence[Tuple[np.ndarray, np.ndarray]], view_range,
                          view_size_px: Tuple[float, float], legend_size_px: Tuple[float, float],
                          margin_px: float = LEGEND_MARGIN_PX) -> Dict[str, int]:
    """Number of curve samples under the legend for every candidate position.

    ``curves``: (x, y) arrays in data coordinates. ``view_range``: ((x0, x1),
    (y0, y1)). Sizes are in pixels. Pure function, so the choice is testable.
    """
    (x0, x1), (y0, y1) = view_range
    width, height = float(view_size_px[0]), float(view_size_px[1])
    counts = {name: 0 for name, _ax, _ay in LEGEND_POSITIONS}
    if width <= 0 or height <= 0 or x1 == x0 or y1 == y0:
        return counts
    legend_w, legend_h = float(legend_size_px[0]), float(legend_size_px[1])
    for x, y in curves:
        x, y = np.asarray(x, float), np.asarray(y, float)
        if x.size == 0 or x.size != y.size:
            continue
        step = max(1, x.size // LEGEND_SAMPLES_PER_CURVE)
        x, y = x[::step], y[::step]
        keep = np.isfinite(x) & np.isfinite(y)
        # Pixel position inside the view; y grows downwards like the screen.
        px = (x[keep] - x0) / (x1 - x0) * width
        py = (y1 - y[keep]) / (y1 - y0) * height
        for name, ax, ay in LEGEND_POSITIONS:
            left = margin_px + ax * (width - legend_w - 2 * margin_px)
            top = margin_px + ay * (height - legend_h - 2 * margin_px)
            inside = ((px >= left - 4) & (px <= left + legend_w + 4) &
                      (py >= top - 4) & (py <= top + legend_h + 4))
            counts[name] += int(np.count_nonzero(inside))
    return counts


def best_legend_position(counts: Dict[str, int]) -> Tuple[str, float, float]:
    """Least-covered candidate; ties keep the preferred order of LEGEND_POSITIONS."""
    return min(LEGEND_POSITIONS, key=lambda item: counts.get(item[0], 0))


def place_legend_clear_of_curves(legend: pg.LegendItem, view_box: pg.ViewBox,
                                 curves: Sequence[Tuple[np.ndarray, np.ndarray]]) -> str:
    """Lay out and anchor ``legend`` where it covers the fewest curve samples.

    Besides the six positions, the entries are tried in one column, two
    columns and one row: a short, wide plot is usually clearer of curves along
    its top or bottom edge. Layouts wider than the plot are skipped. Ties keep
    fewer columns, then the preferred position order. Returns the position name.
    """
    size = view_box.size()
    entries = max(1, len(legend.items))
    best = None
    for columns in sorted({1, min(2, entries), entries}):
        legend.setColumnCount(columns)
        box = legend.boundingRect()
        if columns > 1 and box.width() > size.width() - 2 * LEGEND_MARGIN_PX:
            continue
        counts = legend_overlap_counts(curves, view_box.viewRange(), (size.width(), size.height()),
                                       (box.width(), box.height()))
        name, ax, ay = best_legend_position(counts)
        if best is None or counts[name] < best[0]:
            best = (counts[name], columns, name, ax, ay)
    _count, columns, name, ax, ay = best
    legend.setColumnCount(columns)
    offset = (LEGEND_MARGIN_PX * (1 - 2 * ax) if ax != 0.5 else 0.0, LEGEND_MARGIN_PX * (1 - 2 * ay))
    legend.anchor(itemPos=(ax, ay), parentPos=(ax, ay), offset=offset)
    return name
