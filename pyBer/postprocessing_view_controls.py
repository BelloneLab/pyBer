"""Compact view menu and portable, named plot-splitter preferences."""
from PySide6 import QtCore, QtGui, QtWidgets

from recording_groups import UNGROUPED_COLOR, UNGROUPED_LABEL, color_dot_icon

# Drawer toggle: chevron size, its distance from the button's right edge, and
# the room kept for it beside the caption (pixels).
DRAWER_CHEVRON_HALF_WIDTH = 4
DRAWER_CHEVRON_RIGHT_MARGIN = 14
DRAWER_CHEVRON_RESERVED = 26


class DrawerToggleButton(QtWidgets.QPushButton):
    """A toggle that opens a drawer below it, marked by a chevron.

    The chevron points down while the drawer is closed (it will open downward)
    and up while it is open (it will fold back). The caption stays centered in
    the remaining width.
    """

    def __init__(self, text: str = "", parent=None) -> None:
        super().__init__(text, parent)
        self.setCheckable(True)
        self.setStyleSheet(f"padding-right: {DRAWER_CHEVRON_RESERVED}px;")

    def paintEvent(self, event) -> None:
        super().paintEvent(event)
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
        color = self.palette().color(QtGui.QPalette.ColorRole.ButtonText)
        if not self.isEnabled():
            color.setAlphaF(0.4)
        pen = QtGui.QPen(color, 1.7)
        pen.setCapStyle(QtCore.Qt.PenCapStyle.RoundCap)
        pen.setJoinStyle(QtCore.Qt.PenJoinStyle.RoundJoin)
        painter.setPen(pen)
        half = DRAWER_CHEVRON_HALF_WIDTH
        cx = self.width() - DRAWER_CHEVRON_RIGHT_MARGIN
        cy = self.height() / 2.0
        tip = half / 2.0 if not self.isChecked() else -half / 2.0   # down when closed, up when open
        painter.drawPolyline([QtCore.QPointF(cx - half, cy - tip), QtCore.QPointF(cx, cy + tip),
                              QtCore.QPointF(cx + half, cy - tip)])
        painter.end()


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
    # Retain the Python wrappers as well as Qt ownership. Some PySide6
    # versions otherwise destroy a submenu when action.menu() is inspected.
    button._view_submenus = []
    bindings = []
    for label, combo in (("Layout", panel.combo_view_layout), ("Theme", panel.combo_plot_preset),
                         ("Heatmap contrast", panel.combo_heat_scale)):
        submenu = QtWidgets.QMenu(label, menu)
        menu.addMenu(submenu)
        button._view_submenus.append(submenu)
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
    # "Hide drawer" left the toolbar: clicking the active rail icon closes the
    # drawer. In Group scope its place is taken by a Groups menu: one check
    # mark per group shows or hides that group's rows, mean trace and metrics.
    panel.btn_action_hide.hide()
    panel.btn_group_view = QtWidgets.QToolButton(panel)
    panel.btn_group_view.setPopupMode(QtWidgets.QToolButton.ToolButtonPopupMode.InstantPopup)
    panel.btn_group_view.setAccessibleName("Show or hide groups")
    panel.btn_group_view.setToolTip(
        "Choose which groups are shown in the heatmap, the mean traces and the metrics. "
        "Hidden groups keep their data; they are only left out of the display.")
    panel.menu_group_view = QtWidgets.QMenu(panel.btn_group_view)
    panel.btn_group_view.setMenu(panel.menu_group_view)

    def group_entries():
        """(id, name, color) of each group, then the ungrouped recordings ('')."""
        entries = [(group["id"], group["name"], group["color"]) for group in panel.recording_groups.groups()]
        return entries + [("", UNGROUPED_LABEL, UNGROUPED_COLOR)]

    def rebuild_group_menu():
        menu = panel.menu_group_view
        menu.clear()
        hidden = panel._hidden_groups()
        for group_id, name, color in group_entries():
            action = menu.addAction(color_dot_icon(color), name)
            action.setCheckable(True)
            action.setChecked(group_id not in hidden)
            action.setData(group_id)
            action.toggled.connect(lambda shown, gid=group_id: panel.set_group_hidden(gid, not shown))
        menu.addSeparator()
        show_all = menu.addAction("Show all groups", panel.show_all_groups)
        show_all.setEnabled(bool(hidden))
        menu.addSeparator()
        blocks = menu.addAction("Group view (colored blocks, one mean per group)")
        blocks.setCheckable(True)
        blocks.setChecked(panel.cb_group_view.isChecked())
        blocks.toggled.connect(panel.cb_group_view.setChecked)

    def reflect_group_view(*_args):
        """Keep the caption and visibility in step with the groups and the scope."""
        total = len(group_entries())
        shown = total - len(panel._hidden_groups())
        panel.btn_group_view.setText(
            "Hide/show groups" if shown == total else f"Hide/show groups ({shown}/{total})")
        panel.group_chip_bar.refresh_eyes()
        panel.btn_group_view.setVisible(panel.tab_visual_mode.currentIndex() == 1)

    panel.menu_group_view.aboutToShow.connect(rebuild_group_menu)
    panel.menu_group_view.aboutToHide.connect(reflect_group_view)
    panel.cb_group_view.toggled.connect(reflect_group_view)
    panel.recording_groups.changed.connect(reflect_group_view)
    panel.tab_visual_mode.currentChanged.connect(reflect_group_view)
    panel._rebuild_group_menu = rebuild_group_menu
    panel._reflect_group_view = reflect_group_view
    for widget in (panel.btn_action_load, panel.btn_action_compute, panel.btn_action_export,
                   panel.btn_action_undo, panel.btn_action_redo, panel.btn_view_menu,
                   panel.combo_toolbar_scope, panel.combo_individual_file, panel.lbl_status,
                   panel.btn_group_view):
        layout.addWidget(widget, 2 if widget is panel.combo_individual_file else
                         1 if widget is panel.lbl_status else 0)
        widget.setFixedHeight(32)
        widget.show()
        if isinstance(widget, (QtWidgets.QPushButton, QtWidgets.QToolButton)):
            widget.setStyleSheet("padding: 3px 8px; font-size: 12px;")
    reflect_group_view()
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
