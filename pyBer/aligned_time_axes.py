"""Synchronize exact time ranges without screen-position compensation."""
from PySide6 import QtCore


class AlignedTimeAxes(QtCore.QObject):
    """Keep stacked, equal-width ViewBoxes on the same time scale in both directions.

    Pyqtgraph's screen-position-based links can retain a transient width mismatch
    while a color-scale widget opens. Explicit ranges avoid that compensation.
    The containing grid supplies matching widths and fixed left-axis margins.
    """

    def __init__(self, first, second, parent=None):
        super().__init__(parent)
        self.plots = (first, second)
        self.views = (first.getViewBox(), second.getViewBox())
        for plot in self.plots:
            plot.installEventFilter(self)
        self._updating = False
        for view in self.views:
            view.sigXRangeChanged.connect(self._sync)

    @QtCore.Slot(object, object)
    def _sync(self, source, limits):
        """Forward a zoom, pan or fit once, without a recursive range update."""
        if self._updating:
            return
        self._updating = True
        try:
            for view in self.views:
                if view is not source:
                    view.setXRange(float(limits[0]), float(limits[1]), padding=0)
        finally:
            self._updating = False

    def eventFilter(self, watched, event):
        """A long title must never force one scene wider than its plot widget."""
        if event.type() in (QtCore.QEvent.Type.Resize, QtCore.QEvent.Type.Show):
            watched.plotItem.titleLabel.setMinimumWidth(0)
        return False
