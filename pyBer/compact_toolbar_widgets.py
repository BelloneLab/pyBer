"""Single-line context and preprocessing actions that retain their existing signals."""
from PySide6 import QtCore, QtGui, QtWidgets


class ContextLabel(QtWidgets.QLabel):
    """Elide visible context only, keeping complete text accessible on hover."""
    def __init__(self, text="", parent=None):
        super().__init__(text, parent)
        self.setMinimumWidth(0)
        self.setSizePolicy(QtWidgets.QSizePolicy.Policy.Ignored, QtWidgets.QSizePolicy.Policy.Preferred)
        self.setTextFormat(QtCore.Qt.TextFormat.PlainText)
        self.setToolTip(text)

    def setText(self, text):
        super().setText(text)
        self.setToolTip(text)

    def paintEvent(self, event):
        painter = QtGui.QPainter(self)
        painter.setPen(self.palette().color(QtGui.QPalette.ColorRole.WindowText))
        painter.setFont(self.font())
        text = self.fontMetrics().elidedText(self.text(), QtCore.Qt.TextElideMode.ElideRight,
                                           self.contentsRect().width())
        painter.drawText(self.contentsRect(), self.alignment() | QtCore.Qt.AlignmentFlag.AlignVCenter, text)


def compact_preprocessing_toolbar(owner, bar):
    """Combine workflow, selection and history in one controls-only row."""
    plots = owner.plots
    layout = bar.layout()
    while layout.count():
        widget = layout.takeAt(0).widget()
        if widget is not None:
            widget.hide()
    layout.setContentsMargins(8, 4, 8, 4)
    layout.setSpacing(5)
    plots._inline_toolbar = True
    plots._plot_header.hide()
    plots._plot_tools.hide()
    owner.btn_workflow_qc.hide()
    owner.btn_workflow_export.setText("Export")
    owner.btn_pre_selection = QtWidgets.QToolButton(bar)
    owner.btn_pre_selection.setText("Selection")
    owner.btn_pre_selection.setPopupMode(QtWidgets.QToolButton.ToolButtonPopupMode.InstantPopup)
    selection = QtWidgets.QMenu(owner.btn_pre_selection)
    owner.btn_pre_selection.setMenu(selection)
    add = selection.addAction("Add from selector", plots.btn_add_region.click)
    box = selection.addAction("Box select")
    box.setCheckable(True)
    box.triggered.connect(lambda checked: plots.btn_box_select.setChecked(checked))
    selection.addSeparator()
    clear = selection.addAction("Clear manual regions", plots.btn_clear_regions.click)

    owner.btn_pre_view = QtWidgets.QToolButton(bar)
    owner.btn_pre_view.setText("View")
    owner.btn_pre_view.setPopupMode(QtWidgets.QToolButton.ToolButtonPopupMode.InstantPopup)
    view = QtWidgets.QMenu(owner.btn_pre_view)
    owner.btn_pre_view.setMenu(view)
    thresholds = view.addAction("Show thresholds")
    thresholds.setCheckable(True)
    thresholds.triggered.connect(lambda checked: plots.btn_thresholds.setChecked(checked))
    owner.menu_plot_style.setTitle("Plot style")
    view.addMenu(owner.menu_plot_style)

    def refresh():
        """Menu states track loading, keyboard actions and programmatic changes."""
        for action, button in ((add, plots.btn_add_region), (box, plots.btn_box_select),
                               (clear, plots.btn_clear_regions), (thresholds, plots.btn_thresholds)):
            action.setEnabled(button.isEnabled())
        box.setChecked(plots.btn_box_select.isChecked())
        thresholds.setChecked(plots.btn_thresholds.isChecked())

    selection.aboutToShow.connect(refresh)
    view.aboutToShow.connect(refresh)
    # Recording details already appear elsewhere; keep legacy labels hidden.
    plots.lbl_title.hide()
    plots.lbl_status.hide()
    for widget in (owner.btn_workflow_load, owner.btn_sensor, owner.btn_workflow_export,
                   plots.btn_undo, plots.btn_redo, owner.btn_pre_selection, owner.btn_pre_view):
        layout.addWidget(widget)
        widget.setFixedHeight(32)
        widget.show()
        if isinstance(widget, (QtWidgets.QPushButton, QtWidgets.QToolButton)):
            widget.setStyleSheet("padding: 3px 8px; font-size: 12px;")
    plots.btn_undo.setFixedWidth(32)
    plots.btn_redo.setFixedWidth(32)
    layout.addStretch(1)
    bar.setFixedHeight(44)
