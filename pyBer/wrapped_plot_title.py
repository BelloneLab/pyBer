"""Plot titles that wrap within the canvas and remain part of image exports."""
import math

import pyqtgraph as pg
from PySide6 import QtCore, QtGui


class WrappedPlotTitle(pg.LabelItem):
    """Allocate enough title height without forcing the plot wider than its panel."""

    def __init__(self, plot_item):
        self._plot_item = plot_item
        self._laying_out = False
        super().__init__(justify="left")
        option = self.item.document().defaultTextOption()
        option.setWrapMode(QtGui.QTextOption.WrapMode.WrapAtWordBoundaryOrAnywhere)
        self.item.document().setDefaultTextOption(option)

    def resizeEvent(self, event):
        """Reflow the text whenever the available plot width changes."""
        if self._laying_out:
            return
        self._laying_out = True
        try:
            self.item.setTextWidth(max(1.0, self.rect().width()))
            self.item.setPos(0, 0)
            self.updateMin()
        finally:
            self._laying_out = False

    def updateMin(self):
        """Reserve the complete wrapped height, keeping horizontal size flexible."""
        height = math.ceil(self.item.boundingRect().height())
        self.setMinimumWidth(0)
        self.setMinimumHeight(height)
        self.setMaximumHeight(height)
        self._sizeHint = {
            QtCore.Qt.SizeHint.MinimumSize: (0, height),
            QtCore.Qt.SizeHint.PreferredSize: (0, height),
            QtCore.Qt.SizeHint.MaximumSize: (-1, height),
        }
        self._plot_item.layout.setRowFixedHeight(0, height)
        self.updateGeometry()


def install_wrapped_title(plot_item):
    """Replace only the title item, preserving the normal plotting/export API."""
    old = plot_item.titleLabel
    plot_item.layout.removeItem(old)
    old.setParentItem(None)
    if old.scene() is not None:
        old.scene().removeItem(old)
    plot_item.titleLabel = WrappedPlotTitle(plot_item)
    plot_item.layout.addItem(plot_item.titleLabel, 0, 1)
    plot_item.setTitle("Output")
