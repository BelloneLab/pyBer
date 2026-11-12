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
                        include_ungrouped: bool = True) -> Optional[GroupDisplay]:
    """Select/reorder rows by group.

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
    for gid in order + ([None] if include_ungrouped else []):
        rows = [i for i, value in enumerate(row_group) if value == gid]
        if not rows:
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
