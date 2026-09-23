"""Compact controls for display-only PSTH color adjustments."""
from contextlib import contextmanager
from PySide6 import QtCore, QtWidgets
from heatmap_display import color_name


@contextmanager
def plot_export_view(panel, widget):
    """Keep interactive color controls out of plot images and restore the view."""
    controls = getattr(panel, "heatmap_color_controls", None)
    if controls is None or widget is None or not widget.isAncestorOf(controls):
        yield
        return
    widgets = (controls, panel.heat_lut, panel.heat_colorbar_widget)
    hidden = [item.isHidden() for item in widgets]
    try:
        controls.hide()
        if panel.plot_heat.property("hasPlotData"):
            panel.heat_lut.hide()
            panel.heat_colorbar_widget.show()
        panel.psth_scale_container.layout().activate()
        panel.psth_shared_card.layout().activate()
        panel.heat_figure.layout().activate()
        yield
    finally:
        for item, was_hidden in zip(widgets, hidden):
            item.setVisible(not was_hidden)
        panel.psth_scale_container.layout().activate()
        panel.psth_shared_card.layout().activate()
        panel.heat_figure.layout().activate()


class DisplayLimitSpinBox(QtWidgets.QDoubleSpinBox):
    def textFromValue(self, value):
        return f"{value:.12g}"


class HeatmapColorControls(QtWidgets.QWidget):
    def __init__(self, panel):
        super().__init__(panel)
        self.panel = panel
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(2, 0, 2, 2)
        layout.setSpacing(3)
        row = QtWidgets.QHBoxLayout()
        row.setSpacing(4)
        row.addWidget(QtWidgets.QLabel("Colors"))
        self.mode = QtWidgets.QComboBox()
        self.mode.addItems([panel.combo_heat_scale.itemText(i)
                            for i in range(panel.combo_heat_scale.count())])
        self.mode.setMinimumWidth(75)
        self.mode.setSizePolicy(QtWidgets.QSizePolicy.Policy.Ignored, QtWidgets.QSizePolicy.Policy.Fixed)
        self.mode.setToolTip(panel.combo_heat_scale.toolTip())
        self.mode.activated.connect(panel.combo_heat_scale.setCurrentIndex)
        panel.combo_heat_scale.currentIndexChanged.connect(self.sync)
        row.addWidget(self.mode, 1)
        self.auto = QtWidgets.QPushButton("Auto")
        self.auto.setToolTip("Release fixed limits and fit the selected contrast mode to the displayed heatmap.")
        self.auto.clicked.connect(panel.btn_auto_heat_scale.click)
        row.addWidget(self.auto)
        self.adjust = QtWidgets.QPushButton("Adjust")
        self.adjust.setCheckable(True)
        self.adjust.setToolTip("Show spatial-style color cursors, palette and exact display limits.")
        self.adjust.toggled.connect(panel.btn_edit_scale.setChecked)
        panel.btn_edit_scale.toggled.connect(self.sync)
        row.addWidget(self.adjust)
        for button in (self.auto, self.adjust):
            button.setStyleSheet("padding: 3px 7px; min-height: 20px;")
        layout.addLayout(row)
        self.summary = QtWidgets.QLabel()
        self.summary.setMinimumWidth(0)
        self.summary.setSizePolicy(QtWidgets.QSizePolicy.Policy.Ignored, QtWidgets.QSizePolicy.Policy.Fixed)
        layout.addWidget(self.summary)
        self.details = QtWidgets.QWidget()
        details = QtWidgets.QFormLayout(self.details)
        details.setContentsMargins(0, 0, 0, 0)
        details.setSpacing(3)
        self.palette = QtWidgets.QComboBox()
        for label, name in (("Viridis", "viridis"), ("Cividis", "cividis"),
                            ("Blue–white–red", "CET-D1"), ("Plasma", "plasma"),
                            ("Inferno", "inferno"), ("Magma", "magma"), ("Turbo", "turbo")):
            self.palette.addItem(label, name)
        self.palette.setToolTip("PSTH heatmap only. Blue–white–red with symmetric limits places white at zero.")
        self.palette.activated.connect(lambda _index: panel._set_heatmap_palette(self.palette.currentData()))
        details.addRow("Palette", self.palette)
        limits = QtWidgets.QHBoxLayout()
        limits.setSpacing(4)
        self.minimum, self.maximum = DisplayLimitSpinBox(), DisplayLimitSpinBox()
        for name, spin in (("Min", self.minimum), ("Max", self.maximum)):
            spin.setRange(-1e12, 1e12)
            spin.setDecimals(15)
            spin.setKeyboardTracking(False)
            spin.setMinimumWidth(65)
            spin.setSizePolicy(QtWidgets.QSizePolicy.Policy.Ignored, QtWidgets.QSizePolicy.Policy.Fixed)
            spin.setAccessibleName("Heatmap color " + name.lower())
            spin.setToolTip("Display limit in the current heatmap units; does not change signal values.")
            spin.editingFinished.connect(self._apply_limits)
            limits.addWidget(QtWidgets.QLabel(name))
            limits.addWidget(spin, 1)
        details.addRow(limits)
        self.error = QtWidgets.QLabel("Min must be lower than Max.")
        self.error.setStyleSheet("color: #d97706;")
        self.error.hide()
        details.addRow(self.error)
        layout.addWidget(self.details)
        self.details.hide()
        self.hide()

    def _apply_limits(self):
        accepted = self.panel._set_heatmap_display_levels(self.minimum.value(), self.maximum.value())
        self.error.setVisible(not accepted)

    def sync(self, *_args):
        panel = self.panel
        ready = bool(panel.plot_heat.property("hasPlotData"))
        self.setVisible(ready and not panel.plot_heat.isHidden())
        self.setEnabled(ready)
        with QtCore.QSignalBlocker(self.mode), QtCore.QSignalBlocker(self.adjust):
            self.mode.setCurrentIndex(panel.combo_heat_scale.currentIndex())
            self.adjust.setChecked(panel.btn_edit_scale.isChecked())
        self.details.setVisible(self.adjust.isChecked())
        panel._refresh_results_minimum_heights()
        name = color_name(panel._style)
        index = self.palette.findData(name)
        if index < 0:
            self.palette.addItem(name, name)
            index = self.palette.count() - 1
        with QtCore.QSignalBlocker(self.palette):
            self.palette.setCurrentIndex(index)
        if not ready:
            return
        levels = panel.img.getLevels()
        if levels is None or len(levels) != 2:
            return
        low, high = map(float, levels)
        for spin, value in ((self.minimum, low), (self.maximum, high)):
            with QtCore.QSignalBlocker(spin):
                spin.setValue(value)
                spin.setSingleStep(max((high - low) / 100., 1e-9))
        fixed = bool(panel._style.get("heatmap_levels_manual", False))
        text = f"{'Fixed' if fixed else 'Auto'}: {low:.4g} to {high:.4g} · {panel._psth_units()}"
        self.summary.setText(text)
        self.summary.setToolTip(text + "\nDisplay only: values beyond the limits saturate in color, not in the data. "
                                "Fixed limits persist across behavior and recording changes; changing normalization resets them.")
        self.error.hide()
