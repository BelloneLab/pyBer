"""Right-click color menu for the PSTH heatmap scale (display only).

The heatmap color settings used to occupy a row above the plot. They now live
in a context menu on the color bar (and on the histogram editor, so the menu
stays reachable in both scale views):

* the current limits ("Auto: -2.6 to 2.7 · units"), as a read-only header;
* **Contrast**: Full range, Robust (2-98%), Symmetric about zero;
* **Auto scale**: release fixed limits;
* **Set exact limits...**: a small dialog with validated Min / Max;
* **Palette**: PSTH heatmap palette only;
* **Histogram editor**: switch the color bar for the draggable histogram.

Every action reuses the panel's existing controls and methods, so saved
settings, projects, undo and exports keep working unchanged. Nothing here
modifies PSTH rows: limits saturate colors, never data.
"""
from contextlib import contextmanager
from typing import Callable, Optional, Tuple

from PySide6 import QtCore, QtGui, QtWidgets

from heatmap_display import color_name

# Palette choices offered in the menu: (label, colormap name).
PALETTES = (("Viridis", "viridis"), ("Cividis", "cividis"), ("Blue-white-red", "CET-D1"),
            ("Plasma", "plasma"), ("Inferno", "inferno"), ("Magma", "magma"), ("Turbo", "turbo"))
COLORBAR_TOOLTIP = ("Drag the handles to adjust contrast. Right-click for contrast mode, auto scale, "
                    "exact limits, palette and the histogram editor. Display only.")


@contextmanager
def plot_export_view(panel, widget):
    """Exports always show the compact color bar, never the histogram editor."""
    scales = getattr(panel, "psth_scale_container", None)
    if scales is None or widget is None or not (widget is scales or widget.isAncestorOf(scales)):
        yield
        return
    widgets = (panel.heat_lut, panel.heat_colorbar_widget)
    hidden = [item.isHidden() for item in widgets]
    try:
        if panel.plot_heat.property("hasPlotData"):
            panel.heat_lut.hide()
            panel.heat_colorbar_widget.show()
        scales.layout().activate()
        panel.heat_figure.layout().activate()
        yield
    finally:
        for item, was_hidden in zip(widgets, hidden):
            item.setVisible(not was_hidden)
        scales.layout().activate()
        panel.heat_figure.layout().activate()


class DisplayLimitSpinBox(QtWidgets.QDoubleSpinBox):
    """Shows limits with full precision instead of a fixed decimal count."""

    def textFromValue(self, value):
        return f"{value:.12g}"


class HeatmapLimitsDialog(QtWidgets.QDialog):
    """Exact display limits; stays open on invalid input (Min must be < Max)."""

    def __init__(self, panel, parent=None) -> None:
        super().__init__(parent)
        self.panel = panel
        self.setWindowTitle("Heatmap color limits")
        self.setModal(True)
        layout = QtWidgets.QVBoxLayout(self)
        form = QtWidgets.QFormLayout()
        self.minimum, self.maximum = DisplayLimitSpinBox(), DisplayLimitSpinBox()
        low, high = current_levels(panel)
        for name, spin, value in (("Min", self.minimum, low), ("Max", self.maximum, high)):
            spin.setRange(-1e12, 1e12)
            spin.setDecimals(15)
            spin.setValue(value)
            spin.setSingleStep(max((high - low) / 100.0, 1e-9))
            spin.setMinimumWidth(140)
            spin.setAccessibleName("Heatmap color " + name.lower())
            spin.setToolTip("Display limit in the current heatmap units; does not change signal values.")
            form.addRow(name, spin)
        layout.addLayout(form)
        note = QtWidgets.QLabel(f"Units: {panel._psth_units()}. Values beyond the limits saturate in color only.")
        note.setProperty("class", "hint")
        note.setWordWrap(True)
        layout.addWidget(note)
        self.error = QtWidgets.QLabel("Min must be lower than Max.")
        self.error.setStyleSheet("color: #d97706;")
        self.error.hide()
        layout.addWidget(self.error)
        buttons = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.StandardButton.Ok
                                             | QtWidgets.QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self.apply)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def apply(self) -> bool:
        accepted = bool(self.panel._set_heatmap_display_levels(self.minimum.value(), self.maximum.value()))
        self.error.setVisible(not accepted)
        if accepted:
            self.accept()
        return accepted


def current_levels(panel) -> Tuple[float, float]:
    levels = panel.img.getLevels()
    if levels is None or len(levels) != 2:
        return 0.0, 1.0
    return float(levels[0]), float(levels[1])


class HeatmapColorMenu(QtCore.QObject):
    """Context menu attached to the heatmap color bar and histogram editor."""

    def __init__(self, panel) -> None:
        super().__init__(panel)
        self.panel = panel
        # Replaceable in tests: show a menu at a global position.
        self.exec_menu: Callable[[QtWidgets.QMenu, QtCore.QPoint], object] = (
            lambda menu, pos: menu.exec(pos))
        self.last_menu: Optional[QtWidgets.QMenu] = None
        for view in (panel.heat_colorbar_widget, panel.heat_lut):
            view.viewport().installEventFilter(self)
            view.setToolTip(COLORBAR_TOOLTIP)

    # -- information ---------------------------------------------------------
    def summary_text(self) -> str:
        panel = self.panel
        if not panel.plot_heat.property("hasPlotData"):
            return "No heatmap"
        low, high = current_levels(panel)
        fixed = bool(panel._style.get("heatmap_levels_manual", False))
        return f"{'Fixed' if fixed else 'Auto'}: {low:.4g} to {high:.4g} · {panel._psth_units()}"

    def sync(self, *_args) -> None:
        """Keep the color bar tooltip describing the current limits."""
        text = self.summary_text()
        tooltip = f"{text}\n{COLORBAR_TOOLTIP}"
        for view in (self.panel.heat_colorbar_widget, self.panel.heat_lut):
            view.setToolTip(tooltip)

    # -- menu ----------------------------------------------------------------
    def build_menu(self, parent: Optional[QtWidgets.QWidget] = None) -> QtWidgets.QMenu:
        panel = self.panel
        ready = bool(panel.plot_heat.property("hasPlotData"))
        menu = QtWidgets.QMenu(parent or panel)
        header = menu.addAction(self.summary_text())
        header.setEnabled(False)
        menu.addSeparator()

        contrast = menu.addMenu("Contrast")
        contrast.setToolTipsVisible(True)
        group = QtGui.QActionGroup(contrast)
        group.setExclusive(True)
        combo = panel.combo_heat_scale
        for index in range(combo.count()):
            action = contrast.addAction(combo.itemText(index))
            action.setCheckable(True)
            action.setChecked(index == combo.currentIndex())
            action.setToolTip("Selecting a mode resets fixed limits; data are unchanged.")
            group.addAction(action)
            action.triggered.connect(lambda _=False, i=index: combo.setCurrentIndex(i))
        contrast.setEnabled(ready)

        auto = menu.addAction("Auto scale")
        auto.setToolTip("Release fixed limits and fit the selected contrast mode to the displayed heatmap.")
        auto.triggered.connect(panel.btn_auto_heat_scale.click)
        auto.setEnabled(ready)
        limits = menu.addAction("Set exact limits...")
        limits.triggered.connect(self.open_limits_dialog)
        limits.setEnabled(ready)

        palette_menu = menu.addMenu("Palette")
        palettes = QtGui.QActionGroup(palette_menu)
        palettes.setExclusive(True)
        current = color_name(panel._style)
        names = [name for _label, name in PALETTES]
        entries = list(PALETTES) + ([(current, current)] if current not in names else [])
        for label, name in entries:
            action = palette_menu.addAction(label)
            action.setCheckable(True)
            action.setChecked(name == current)
            action.setData(name)
            palettes.addAction(action)
            action.triggered.connect(lambda _=False, n=name: panel._set_heatmap_palette(n))

        menu.addSeparator()
        editor = menu.addAction("Histogram editor")
        editor.setCheckable(True)
        editor.setChecked(panel.btn_edit_scale.isChecked())
        editor.setToolTip("Replace the color bar by a histogram with draggable limits.")
        editor.toggled.connect(panel.btn_edit_scale.setChecked)
        editor.setEnabled(ready)
        self.last_menu = menu
        return menu

    def popup(self, global_pos: QtCore.QPoint) -> None:
        menu = self.build_menu()
        self.exec_menu(menu, global_pos)

    def open_limits_dialog(self) -> int:
        dialog = HeatmapLimitsDialog(self.panel, self.panel)
        return dialog.exec()

    def eventFilter(self, watched: QtCore.QObject, event: QtCore.QEvent) -> bool:
        if event.type() == QtCore.QEvent.Type.ContextMenu:
            self.popup(event.globalPos())
            return True
        return False


def find_action(menu: QtWidgets.QMenu, *path: str) -> Optional[QtGui.QAction]:
    """Return the action reached by following menu texts (helper for tests and scripts)."""
    current = menu
    for depth, text in enumerate(path):
        match = next((action for action in current.actions() if action.text() == text), None)
        if match is None:
            return None
        if depth == len(path) - 1:
            return match
        current = match.menu()
        if current is None:
            return None
    return None
