"""Compact view menu and portable, named plot-splitter preferences."""
from PySide6 import QtCore, QtGui, QtWidgets


def split_plot_layout(host, orientation):
    """Replace fixed spacing with a draggable divider, retaining plot ownership."""
    layout = host.layout()
    splitter = QtWidgets.QSplitter(orientation, host)
    splitter.setChildrenCollapsible(False)
    splitter.setHandleWidth(6)
    splitter.setToolTip("Drag the divider to resize plots. Sizes are saved automatically.")
    widgets = [layout.itemAt(i).widget() for i in range(layout.count())]
    for widget in widgets:
        if widget is not None:
            layout.removeWidget(widget)
            if orientation == QtCore.Qt.Orientation.Horizontal:
                widget.setMinimumWidth(0)
                policy = widget.sizePolicy()
                policy.setHorizontalPolicy(QtWidgets.QSizePolicy.Policy.Ignored)
                widget.setSizePolicy(policy)
            splitter.addWidget(widget)
    layout.addWidget(splitter)
    splitter.setSizes([1000] * splitter.count())
    return splitter


class PlotSplitterPreferences(QtCore.QObject):
    """Store named sizes, ignoring zero sizes caused by hidden views or startup."""
    changed = QtCore.Signal()

    def __init__(self, splitters, parent=None):
        super().__init__(parent)
        self.splitters = splitters
        self.defaults = {key: ([240, 500, 220, 360] if key == "rows" else [1000] * widget.count())
                         for key, widget in splitters.items()}
        self.saved = {key: list(value) for key, value in self.defaults.items()}
        for widget in splitters.values():
            widget.splitterMoved.connect(self._moved)
            widget.installEventFilter(self)

    def _moved(self, *_args):
        """Keep user drag sizes immediately, before a debounced preference write."""
        self.snapshot()
        self.changed.emit()

    def snapshot(self):
        """Return JSON-compatible sizes, preserving hidden panel proportions."""
        for key, widget in self.splitters.items():
            sizes = widget.sizes()
            if widget.isVisible() and sizes:
                self.saved[key] = [size if size > 0 else previous
                                   for size, previous in zip(sizes, self.saved[key])]
        return {key: list(sizes) for key, sizes in self.saved.items()}

    def restore(self, values):
        """Accept only finite positive integer sizes for known splitter names."""
        if not isinstance(values, dict):
            return
        for key, widget in self.splitters.items():
            sizes = values.get(key)
            if (isinstance(sizes, list) and len(sizes) == widget.count() and
                    all(type(size) is int and 0 < size <= 1000000 for size in sizes)):
                self.saved[key] = list(sizes)
                widget.setSizes(sizes)

    def reset(self):
        """Restore comfortable proportions without changing analysis settings."""
        self.restore(self.defaults)
        self.changed.emit()

    def eventFilter(self, watched, event):
        """Reapply stored proportions once a previously hidden view has geometry."""
        if event.type() == QtCore.QEvent.Type.Show:
            QtCore.QTimer.singleShot(0, self, self._apply_visible)
        return False

    def _apply_visible(self):
        for key, widget in self.splitters.items():
            if widget.isVisible():
                widget.setSizes(self.saved[key])


def create_view_menu(panel):
    """Expose existing controls as menu actions, preserving their normal signals."""
    button = QtWidgets.QToolButton(panel)
    button.setText("View")
    button.setPopupMode(QtWidgets.QToolButton.ToolButtonPopupMode.InstantPopup)
    menu = QtWidgets.QMenu(button)
    button.setMenu(menu)
    bindings = []
    for label, combo in (("Layout", panel.combo_view_layout), ("Theme", panel.combo_plot_preset),
                         ("Heatmap contrast", panel.combo_heat_scale)):
        submenu = menu.addMenu(label)
        group = QtGui.QActionGroup(submenu)
        group.setExclusive(True)
        for index in range(combo.count()):
            action = submenu.addAction(combo.itemText(index))
            action.setCheckable(True)
            group.addAction(action)
            action.triggered.connect(lambda _checked=False, c=combo, i=index: c.setCurrentIndex(i))
            bindings.append((action, combo, index))
    menu.addSeparator()
    auto = menu.addAction("Auto scale heatmap", panel.btn_auto_heat_scale.click)
    menu.addAction("Fit plots", panel.btn_fit_psth.click)
    edit = menu.addAction("Edit heatmap scale")
    edit.setCheckable(True)
    edit.triggered.connect(lambda checked: panel.btn_edit_scale.setChecked(checked))
    menu.addSeparator()
    menu.addAction("Reset panel sizes", panel.plot_splitter_preferences.reset)
    menu.addAction("Save current view", panel._save_settings)

    def refresh():
        """Reflect changes made by project restoration or other app controls."""
        for action, combo, index in bindings:
            action.setChecked(combo.currentIndex() == index)
            action.setEnabled(combo.isEnabled())
        auto.setEnabled(panel.btn_auto_heat_scale.isEnabled())
        edit.setEnabled(panel.btn_edit_scale.isEnabled())
        edit.setChecked(panel.btn_edit_scale.isChecked())

    menu.aboutToShow.connect(refresh)
    return button


def compact_results_toolbar(panel):
    """Put actions, scope, file and elided status on one non-wrapping row."""
    layout = panel._post_transport_bar.layout()
    while layout.count():
        item = layout.takeAt(0)
        widget = item.widget()
        if widget is not None:
            widget.hide()
    layout.setContentsMargins(8, 4, 8, 4)
    layout.setSpacing(5)
    panel.btn_action_compute.setText("PSTH")
    panel.btn_action_compute.setToolTip("Compute PSTH with the current analysis settings")
    panel.btn_action_export.setText("Export")
    panel.combo_toolbar_scope = QtWidgets.QComboBox(panel)
    panel.combo_toolbar_scope.addItems(["Individual", "Group"])
    panel.combo_toolbar_scope.setToolTip("Choose one recording or the group summary")
    panel.combo_toolbar_scope.setFixedWidth(112)
    panel.combo_toolbar_scope.currentIndexChanged.connect(panel.tab_visual_mode.setCurrentIndex)

    def reflect_scope(index):
        """Project restoration and other navigation keep the visible picker current."""
        with QtCore.QSignalBlocker(panel.combo_toolbar_scope):
            panel.combo_toolbar_scope.setCurrentIndex(index)

    panel.tab_visual_mode.currentChanged.connect(reflect_scope)
    reflect_scope(panel.tab_visual_mode.currentIndex())
    panel.combo_individual_file.setMinimumWidth(100)
    panel.combo_individual_file.setMaximumWidth(420)
    panel.combo_individual_file.setSizePolicy(QtWidgets.QSizePolicy.Policy.Ignored,
                                            QtWidgets.QSizePolicy.Policy.Fixed)
    panel.combo_individual_file.setAccessibleName("Displayed recording")
    for combo in (panel.combo_toolbar_scope, panel.combo_individual_file):
        combo.setStyleSheet("QComboBox { padding: 3px 22px 3px 8px; min-height: 20px; font-size: 12px; }")
    panel.lbl_status.setMinimumWidth(0)
    panel.lbl_status.setMaximumWidth(260)
    for widget in (panel.btn_action_load, panel.btn_action_compute, panel.btn_action_export,
                   panel.btn_action_undo, panel.btn_action_redo, panel.btn_view_menu,
                   panel.combo_toolbar_scope, panel.combo_individual_file, panel.lbl_status,
                   panel.btn_action_hide):
        layout.addWidget(widget, 2 if widget is panel.combo_individual_file else
                         1 if widget is panel.lbl_status else 0)
        widget.setFixedHeight(32)
        widget.show()
        if isinstance(widget, (QtWidgets.QPushButton, QtWidgets.QToolButton)):
            widget.setStyleSheet("padding: 3px 8px; font-size: 12px;")
    panel.btn_action_undo.setFixedWidth(32)
    panel.btn_action_redo.setFixedWidth(32)
    panel._post_transport_bar.setFixedHeight(44)
    # Keep secondary operations accessible without spending permanent width.
    menu = panel.btn_view_menu.menu()
    menu.addSeparator()
    menu.addAction("Plot style...", panel.btn_style.click)
    menu.addAction("Help", panel.btn_action_help.click)
    panel.menu_action_load.addSeparator()
    panel.menu_action_load.addAction("Reset analysis...", panel.btn_action_reset.click)
