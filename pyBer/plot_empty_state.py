"""Quiet empty-canvas presentation shared by both scientific workspaces."""

from PySide6 import QtCore, QtWidgets


class PlotEmptyState(QtWidgets.QFrame):
    """A small typographic prompt, without invented chart marks or hero artwork."""

    def __init__(self, title="Load a recording to begin", hint="Your traces will appear here.", parent=None):
        super().__init__(parent)
        self.setObjectName("plotEmptyWorkspace")
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(32, 32, 32, 32)
        layout.setSpacing(8)
        layout.addStretch(1)
        self.title = QtWidgets.QLabel(title)
        self.title.setObjectName("plotEmptyTitle")
        self.title.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.title.setWordWrap(True)
        self.hint = QtWidgets.QLabel(hint)
        self.hint.setObjectName("plotEmptyHint")
        self.hint.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.hint.setWordWrap(True)
        layout.addWidget(self.title)
        layout.addWidget(self.hint)
        layout.addStretch(1)


def set_plot_has_data(plot, ready: bool) -> None:
    """Hide the entire graphics item, including its axes, title and overlays.

    Keep the PlotWidget and its parent card in the layout so existing splitter
    proportions, theme styling and linked plot axes survive an empty state.
    Visibility changes never clear or transform scientific data.
    """
    ready = bool(ready)
    plot.setProperty("hasPlotData", ready)
    plot.getPlotItem().setVisible(ready)
