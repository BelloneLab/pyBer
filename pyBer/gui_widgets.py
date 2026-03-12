# gui_widgets.py
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np

from PySide6 import QtCore, QtWidgets, QtGui
import pyqtgraph as pg

from analysis_core import ProcessingParams, ProcessedTrial


# -------------------- helpers --------------------

def make_step_post(x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.size == 0 or y.size == 0:
        return np.array([], dtype=float), np.array([], dtype=float)
    n = min(x.size, y.size)
    x = x[:n]
    y = y[:n]
    X = np.repeat(x, 2)
    Y = np.repeat(y, 2)
    if n >= 2:
        X[1:-1:2] = x[1:]
        X[2::2] = x[1:]
    return X, Y


def attach_right_axis(plot_widget: pg.PlotWidget) -> pg.ViewBox:
    pi = plot_widget.getPlotItem()
    pi.showAxis("right")
    right_axis = pi.getAxis("right")
    right_axis.setStyle(tickTextOffset=8)

    vb2 = pg.ViewBox()
    pi.scene().addItem(vb2)
    right_axis.linkToView(vb2)
    vb2.setXLink(pi.vb)

    def _update() -> None:
        vb2.setGeometry(pi.vb.sceneBoundingRect())
        vb2.linkedViewChanged(pi.vb, vb2.XAxis)

    pi.vb.sigResized.connect(_update)
    _update()
    return vb2


def optimize_plot_widget(pw: pg.PlotWidget) -> None:
    pi = pw.getPlotItem()
    pi.setDownsampling(mode="peak")
    pi.setClipToView(True)
    pi.setMenuEnabled(False)
    pw.setAntialiasing(False)
    pw.showGrid(x=True, y=True, alpha=0.25)


# -------------------- Metadata --------------------

class MetadataForm(QtWidgets.QWidget):
    def __init__(self, parent: Optional[QtWidgets.QWidget] = None) -> None:
        super().__init__(parent)
        layout = QtWidgets.QVBoxLayout(self)

        form = QtWidgets.QFormLayout()
        self.animal_id = QtWidgets.QLineEdit()
        self.session = QtWidgets.QLineEdit()
        self.trial = QtWidgets.QLineEdit()
        self.treatment = QtWidgets.QLineEdit()
        form.addRow("Animal ID", self.animal_id)
        form.addRow("Session", self.session)
        form.addRow("Trial", self.trial)
        form.addRow("Treatment", self.treatment)
        layout.addLayout(form)

        box = QtWidgets.QGroupBox("Custom metadata (key/value)")
        v = QtWidgets.QVBoxLayout(box)
        self.table = QtWidgets.QTableWidget(0, 2)
        self.table.setHorizontalHeaderLabels(["Key", "Value"])
        self.table.horizontalHeader().setStretchLastSection(True)
        v.addWidget(self.table)

        row = QtWidgets.QHBoxLayout()
        btn_add = QtWidgets.QPushButton("Add metadata entry")
        btn_rm = QtWidgets.QPushButton("Remove selected entry")
        row.addWidget(btn_add)
        row.addWidget(btn_rm)
        row.addStretch(1)
        v.addLayout(row)
        layout.addWidget(box)

        btn_add.clicked.connect(self._add_row)
        btn_rm.clicked.connect(self._remove_selected)

    def _add_row(self) -> None:
        r = self.table.rowCount()
        self.table.insertRow(r)
        self.table.setItem(r, 0, QtWidgets.QTableWidgetItem(""))
        self.table.setItem(r, 1, QtWidgets.QTableWidgetItem(""))

    def _remove_selected(self) -> None:
        rows = sorted({i.row() for i in self.table.selectedIndexes()}, reverse=True)
        for r in rows:
            self.table.removeRow(r)

    def to_dict(self) -> Dict[str, str]:
        d = {
            "animal_id": self.animal_id.text().strip(),
            "session": self.session.text().strip(),
            "trial": self.trial.text().strip(),
            "treatment": self.treatment.text().strip(),
        }
        for r in range(self.table.rowCount()):
            k_item = self.table.item(r, 0)
            v_item = self.table.item(r, 1)
            k = (k_item.text().strip() if k_item else "")
            v = (v_item.text().strip() if v_item else "")
            if k:
                d[f"custom:{k}"] = v
        return d

    def from_dict(self, d: Dict[str, str]) -> None:
        self.animal_id.setText(d.get("animal_id", ""))
        self.session.setText(d.get("session", ""))
        self.trial.setText(d.get("trial", ""))
        self.treatment.setText(d.get("treatment", ""))

        self.table.setRowCount(0)
        for k, v in d.items():
            if k.startswith("custom:"):
                r = self.table.rowCount()
                self.table.insertRow(r)
                self.table.setItem(r, 0, QtWidgets.QTableWidgetItem(k.replace("custom:", "", 1)))
                self.table.setItem(r, 1, QtWidgets.QTableWidgetItem(v))


class MetadataDialog(QtWidgets.QDialog):
    def __init__(self, channels: List[str], existing: Optional[Dict[str, Dict[str, str]]] = None, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Metadata")
        self.resize(650, 420)

        self.forms: Dict[str, MetadataForm] = {}
        layout = QtWidgets.QVBoxLayout(self)

        self.tabs = QtWidgets.QTabWidget()
        for ch in channels:
            form = MetadataForm()
            self.forms[ch] = form
            self.tabs.addTab(form, ch)
        layout.addWidget(self.tabs)

        buttons = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

        if existing:
            for ch, d in existing.items():
                if ch in self.forms:
                    self.forms[ch].from_dict(d)

    def get_metadata(self) -> Dict[str, Dict[str, str]]:
        return {ch: f.to_dict() for ch, f in self.forms.items()}


# -------------------- Artifact panel --------------------

class ArtifactPanel(QtWidgets.QWidget):
    regionsChanged = QtCore.Signal(list)  # list[(t0,t1)]

    def __init__(self, parent: Optional[QtWidgets.QWidget] = None) -> None:
        super().__init__(parent)

        layout = QtWidgets.QVBoxLayout(self)
        layout.setSpacing(8)

        self.table = QtWidgets.QTableWidget(0, 2)
        self.table.setHorizontalHeaderLabels(["Start (s)", "End (s)"])
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        self.table.setEditTriggers(QtWidgets.QAbstractItemView.DoubleClicked | QtWidgets.QAbstractItemView.SelectedClicked)
        layout.addWidget(self.table)

        form = QtWidgets.QFormLayout()
        self.edit_start = QtWidgets.QLineEdit()
        self.edit_end = QtWidgets.QLineEdit()
        dv = QtGui.QDoubleValidator(bottom=-1e12, top=1e12, decimals=6)
        self.edit_start.setValidator(dv)
        self.edit_end.setValidator(dv)
        form.addRow("Start (s)", self.edit_start)
        form.addRow("End (s)", self.edit_end)
        layout.addLayout(form)

        row = QtWidgets.QHBoxLayout()
        self.btn_add = QtWidgets.QPushButton("Add")
        self.btn_update = QtWidgets.QPushButton("Update selected")
        self.btn_delete = QtWidgets.QPushButton("Delete selected")
        row.addWidget(self.btn_add)
        row.addWidget(self.btn_update)
        row.addWidget(self.btn_delete)
        layout.addLayout(row)

        self.btn_clear = QtWidgets.QPushButton("Clear all")
        layout.addWidget(self.btn_clear)

        self.btn_add.clicked.connect(self._add_from_fields)
        self.btn_update.clicked.connect(self._update_selected_from_fields)
        self.btn_delete.clicked.connect(self._delete_selected)
        self.btn_clear.clicked.connect(self._clear_all)
        self.table.itemSelectionChanged.connect(self._sync_fields_from_selection)
        self.table.cellChanged.connect(self._on_table_cell_changed)

        self._block_table_signals = False

    def set_regions(self, regions: List[Tuple[float, float]]) -> None:
        self._block_table_signals = True
        try:
            self.table.setRowCount(0)
            for t0, t1 in regions:
                r = self.table.rowCount()
                self.table.insertRow(r)
                self.table.setItem(r, 0, QtWidgets.QTableWidgetItem(f"{t0:.6f}"))
                self.table.setItem(r, 1, QtWidgets.QTableWidgetItem(f"{t1:.6f}"))
        finally:
            self._block_table_signals = False

    def regions(self) -> List[Tuple[float, float]]:
        out: List[Tuple[float, float]] = []
        for r in range(self.table.rowCount()):
            a = self.table.item(r, 0)
            b = self.table.item(r, 1)
            try:
                t0 = float(a.text()) if a else float("nan")
                t1 = float(b.text()) if b else float("nan")
            except Exception:
                continue
            if np.isfinite(t0) and np.isfinite(t1):
                out.append((min(t0, t1), max(t0, t1)))
        out.sort(key=lambda x: x[0])
        return out

    def _emit(self) -> None:
        self.regionsChanged.emit(self.regions())

    def _add_from_fields(self) -> None:
        try:
            t0 = float(self.edit_start.text())
            t1 = float(self.edit_end.text())
        except Exception:
            return
        r = self.table.rowCount()
        self.table.insertRow(r)
        self.table.setItem(r, 0, QtWidgets.QTableWidgetItem(f"{min(t0,t1):.6f}"))
        self.table.setItem(r, 1, QtWidgets.QTableWidgetItem(f"{max(t0,t1):.6f}"))
        self._emit()

    def _update_selected_from_fields(self) -> None:
        rows = sorted({i.row() for i in self.table.selectedIndexes()})
        if not rows:
            return
        try:
            t0 = float(self.edit_start.text())
            t1 = float(self.edit_end.text())
        except Exception:
            return
        for r in rows:
            self.table.item(r, 0).setText(f"{min(t0,t1):.6f}")
            self.table.item(r, 1).setText(f"{max(t0,t1):.6f}")
        self._emit()

    def _delete_selected(self) -> None:
        rows = sorted({i.row() for i in self.table.selectedIndexes()}, reverse=True)
        for r in rows:
            self.table.removeRow(r)
        if rows:
            self._emit()

    def _clear_all(self) -> None:
        self.table.setRowCount(0)
        self._emit()

    def _sync_fields_from_selection(self) -> None:
        rows = sorted({i.row() for i in self.table.selectedIndexes()})
        if len(rows) != 1:
            return
        r = rows[0]
        a = self.table.item(r, 0)
        b = self.table.item(r, 1)
        if a and b:
            self.edit_start.setText(a.text())
            self.edit_end.setText(b.text())

    def _on_table_cell_changed(self, _r: int, _c: int) -> None:
        if self._block_table_signals:
            return
        self._emit()


# -------------------- Parameter panel --------------------

class ParameterPanel(QtWidgets.QGroupBox):
    paramsChanged = QtCore.Signal()

    def __init__(self, parent=None) -> None:
        super().__init__("Processing Parameters", parent)
        self._build_ui()
        self._wire()

    def _minw(self, w: QtWidgets.QWidget, px: int = 170) -> None:
        w.setMinimumWidth(px)

    def _build_ui(self) -> None:
        layout = QtWidgets.QFormLayout(self)
        layout.setVerticalSpacing(10)

        self.artifact_mode = QtWidgets.QComboBox()
        self.artifact_mode.addItems(["Global MAD (dx)", "Adaptive MAD (windowed dx)"])
        self._minw(self.artifact_mode)

        self.mad_threshold = QtWidgets.QDoubleSpinBox()
        self.mad_threshold.setRange(1.0, 50.0)
        self.mad_threshold.setDecimals(2)
        self.mad_threshold.setValue(8.0)
        self._minw(self.mad_threshold)

        self.adaptive_window_sec = QtWidgets.QDoubleSpinBox()
        self.adaptive_window_sec.setRange(0.5, 120.0)
        self.adaptive_window_sec.setDecimals(2)
        self.adaptive_window_sec.setValue(5.0)
        self._minw(self.adaptive_window_sec)

        self.pad_sec = QtWidgets.QDoubleSpinBox()
        self.pad_sec.setRange(0.0, 10.0)
        self.pad_sec.setDecimals(3)
        self.pad_sec.setValue(0.25)
        self._minw(self.pad_sec)

        self.lowpass_hz = QtWidgets.QDoubleSpinBox()
        self.lowpass_hz.setRange(0.1, 200.0)
        self.lowpass_hz.setDecimals(2)
        self.lowpass_hz.setValue(12.0)
        self._minw(self.lowpass_hz)

        self.filter_order = QtWidgets.QSpinBox()
        self.filter_order.setRange(1, 8)
        self.filter_order.setValue(3)
        self._minw(self.filter_order)

        self.airpls_lambda = QtWidgets.QDoubleSpinBox()
        self.airpls_lambda.setRange(1.0, 1e12)
        self.airpls_lambda.setDecimals(2)
        self.airpls_lambda.setValue(1e6)
        self._minw(self.airpls_lambda)

        self.airpls_porder = QtWidgets.QSpinBox()
        self.airpls_porder.setRange(1, 3)
        self.airpls_porder.setValue(1)
        self._minw(self.airpls_porder)

        self.airpls_itermax = QtWidgets.QSpinBox()
        self.airpls_itermax.setRange(1, 100)
        self.airpls_itermax.setValue(15)
        self._minw(self.airpls_itermax)

        self.ref_fit_method = QtWidgets.QComboBox()
        self.ref_fit_method.addItems(["OLS (recommended)", "Lasso", "None"])
        self._minw(self.ref_fit_method)

        self.lasso_alpha = QtWidgets.QDoubleSpinBox()
        self.lasso_alpha.setRange(1e-6, 1.0)
        self.lasso_alpha.setDecimals(6)
        self.lasso_alpha.setValue(0.001)
        self._minw(self.lasso_alpha)

        self.dff_method = QtWidgets.QComboBox()
        self.dff_method.addItems(["Regression dF/F ((F-F0)/F0)", "Z-difference (z465 - z405)"])
        self._minw(self.dff_method)

        self.output_view = QtWidgets.QComboBox()
        self.output_view.addItems(["dF/F", "Z-score (signal)", "Z-score (motion-corrected)"])
        self._minw(self.output_view)

        self.preview_max_points = QtWidgets.QSpinBox()
        self.preview_max_points.setRange(1000, 200000)
        self.preview_max_points.setSingleStep(1000)
        self.preview_max_points.setValue(15000)
        self._minw(self.preview_max_points)

        layout.addRow("Artifact detection", self.artifact_mode)
        layout.addRow("MAD threshold (k)", self.mad_threshold)
        layout.addRow("Adaptive window (sec)", self.adaptive_window_sec)
        layout.addRow("Artifact pad (sec)", self.pad_sec)
        layout.addRow("Low-pass cutoff (Hz)", self.lowpass_hz)
        layout.addRow("Filter order", self.filter_order)
        layout.addRow("airPLS λ", self.airpls_lambda)
        layout.addRow("airPLS p-order", self.airpls_porder)
        layout.addRow("airPLS iter max", self.airpls_itermax)
        layout.addRow("Reference fit", self.ref_fit_method)
        layout.addRow("Lasso α", self.lasso_alpha)
        layout.addRow("dF/F definition", self.dff_method)
        layout.addRow("Preview/output", self.output_view)
        layout.addRow("Preview max points", self.preview_max_points)

        self._update_visibility()

    def _wire(self) -> None:
        for w in [self.artifact_mode, self.ref_fit_method, self.dff_method, self.output_view]:
            w.currentIndexChanged.connect(self._changed)
        for w in [
            self.mad_threshold, self.adaptive_window_sec, self.pad_sec,
            self.lowpass_hz, self.filter_order,
            self.airpls_lambda, self.airpls_porder, self.airpls_itermax,
            self.lasso_alpha, self.preview_max_points
        ]:
            w.valueChanged.connect(self._changed)

    def _changed(self) -> None:
        self._update_visibility()
        self.paramsChanged.emit()

    def _update_visibility(self) -> None:
        adaptive = self.artifact_mode.currentText().startswith("Adaptive")
        self.adaptive_window_sec.setEnabled(adaptive)
        lasso = self.ref_fit_method.currentText().startswith("Lasso")
        self.lasso_alpha.setEnabled(lasso)

    def get_params(self) -> ProcessingParams:
        artifact_mode = "adaptive_mad" if self.artifact_mode.currentText().startswith("Adaptive") else "global_mad"

        ref_txt = self.ref_fit_method.currentText()
        if ref_txt.startswith("OLS"):
            ref_fit_method = "ols"
        elif ref_txt.startswith("Lasso"):
            ref_fit_method = "lasso"
        else:
            ref_fit_method = "none"

        dff_method = "regression_dff" if self.dff_method.currentText().startswith("Regression") else "z_diff"

        out_txt = self.output_view.currentText()
        if out_txt.startswith("Z-score (signal)"):
            output_view = "z_signal"
        elif out_txt.startswith("Z-score (motion"):
            output_view = "z_corrected"
        else:
            output_view = "dff"

        return ProcessingParams(
            artifact_mode=artifact_mode,
            mad_threshold=float(self.mad_threshold.value()),
            adaptive_window_sec=float(self.adaptive_window_sec.value()),
            artifact_pad_sec=float(self.pad_sec.value()),
            lowpass_hz=float(self.lowpass_hz.value()),
            filter_order=int(self.filter_order.value()),
            airpls_lambda=float(self.airpls_lambda.value()),
            airpls_porder=int(self.airpls_porder.value()),
            airpls_itermax=int(self.airpls_itermax.value()),
            ref_fit_method=ref_fit_method,
            lasso_alpha=float(self.lasso_alpha.value()),
            dff_method=dff_method,
            output_view=output_view,
            preview_max_points=int(self.preview_max_points.value()),
        )


# -------------------- File panel --------------------

class FileQueuePanel(QtWidgets.QGroupBox):
    openFileRequested = QtCore.Signal()
    openFolderRequested = QtCore.Signal()
    exportRequested = QtCore.Signal()
    metadataRequested = QtCore.Signal()
    updatePreviewRequested = QtCore.Signal()
    toggleArtifactsRequested = QtCore.Signal()

    currentFileChanged = QtCore.Signal(int)
    channelChanged = QtCore.Signal(str)
    triggerChanged = QtCore.Signal(str)
    splitChanged = QtCore.Signal(bool)

    def __init__(self, parent=None) -> None:
        super().__init__("Data", parent)
        layout = QtWidgets.QVBoxLayout(self)
        layout.setSpacing(10)

        row = QtWidgets.QHBoxLayout()
        self.btn_open = QtWidgets.QPushButton("Open .doric File…")
        self.btn_folder = QtWidgets.QPushButton("Add Folder…")
        row.addWidget(self.btn_open)
        row.addWidget(self.btn_folder)
        layout.addLayout(row)

        self.list_files = QtWidgets.QListWidget()
        layout.addWidget(self.list_files)

        sel_box = QtWidgets.QGroupBox("Selection")
        form = QtWidgets.QFormLayout(sel_box)

        self.combo_channel = QtWidgets.QComboBox()
        self.combo_trigger = QtWidgets.QComboBox()
        self.combo_channel.setMinimumWidth(170)
        self.combo_trigger.setMinimumWidth(170)

        self.chk_split = QtWidgets.QCheckBox("Split multi-animal file (AIN01/AIN02) on export/batch")
        self.chk_split.setChecked(True)

        self.btn_metadata = QtWidgets.QPushButton("Metadata…")
        self.btn_update = QtWidgets.QPushButton("Update Preview")
        self.btn_artifacts = QtWidgets.QPushButton("Artifacts…")

        form.addRow("Channel (preview)", self.combo_channel)
        form.addRow("Analog/Digital channel (overlay)", self.combo_trigger)
        form.addRow(self.chk_split)
        form.addRow(self.btn_metadata)
        form.addRow(self.btn_update)
        form.addRow(self.btn_artifacts)
        layout.addWidget(sel_box)

        self.btn_export = QtWidgets.QPushButton("Export processed HDF5…")
        layout.addWidget(self.btn_export)

        self.progress = QtWidgets.QProgressBar()
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        layout.addWidget(self.progress)

        self.combo_channel.setEnabled(False)
        self.combo_trigger.setEnabled(False)

        self.btn_open.clicked.connect(self.openFileRequested.emit)
        self.btn_folder.clicked.connect(self.openFolderRequested.emit)
        self.btn_export.clicked.connect(self.exportRequested.emit)
        self.btn_metadata.clicked.connect(self.metadataRequested.emit)
        self.btn_update.clicked.connect(self.updatePreviewRequested.emit)
        self.btn_artifacts.clicked.connect(self.toggleArtifactsRequested.emit)

        self.list_files.currentRowChanged.connect(self.currentFileChanged.emit)
        self.combo_channel.currentTextChanged.connect(self.channelChanged.emit)
        self.combo_trigger.currentTextChanged.connect(self.triggerChanged.emit)
        self.chk_split.toggled.connect(self.splitChanged.emit)

    def add_file(self, path: str) -> None:
        self.list_files.addItem(path)
        if self.list_files.count() == 1:
            self.list_files.setCurrentRow(0)

    def add_files(self, paths: List[str]) -> None:
        for p in paths:
            self.add_file(p)

    def file_at(self, idx: int) -> Optional[str]:
        item = self.list_files.item(idx)
        return item.text() if item else None

    def selected_channel(self) -> str:
        return self.combo_channel.currentText().strip()

    def selected_trigger(self) -> str:
        return self.combo_trigger.currentText().strip()

    def set_available_channels(self, channels: List[str]) -> None:
        self.combo_channel.blockSignals(True)
        self.combo_channel.clear()
        self.combo_channel.addItems(channels)
        self.combo_channel.setEnabled(len(channels) > 0)
        self.combo_channel.blockSignals(False)

    def set_available_triggers(self, triggers: List[str]) -> None:
        self.combo_trigger.blockSignals(True)
        self.combo_trigger.clear()
        self.combo_trigger.addItem("")
        for t in triggers:
            self.combo_trigger.addItem(t)
        self.combo_trigger.setEnabled(True)
        self.combo_trigger.blockSignals(False)

    def set_progress(self, pct: int) -> None:
        self.progress.setValue(max(0, min(100, pct)))


# -------------------- Plot dashboard --------------------

class PlotDashboard(QtWidgets.QFrame):
    manualRegionFromSelectorRequested = QtCore.Signal()
    manualRegionCenteredAtRequested = QtCore.Signal(float)
    showArtifactsRequested = QtCore.Signal()

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._proc_need_autorange = True
        self._out_need_autorange = True
        self._build()

    def _build(self) -> None:
        layout = QtWidgets.QVBoxLayout(self)
        layout.setSpacing(8)

        header = QtWidgets.QHBoxLayout()
        self.lbl_title = QtWidgets.QLabel("No file loaded")
        self.lbl_title.setStyleSheet("font-weight: 700; font-size: 12pt;")
        header.addWidget(self.lbl_title)
        header.addStretch(1)

        self.btn_add_region = QtWidgets.QPushButton("Add manual artifact from selector")
        self.btn_clear_regions = QtWidgets.QPushButton("Clear manual regions")
        self.btn_show_artifacts = QtWidgets.QPushButton("Artifacts…")
        header.addWidget(self.btn_add_region)
        header.addWidget(self.btn_clear_regions)
        header.addWidget(self.btn_show_artifacts)
        layout.addLayout(header)

        self.plot_raw = pg.PlotWidget(title="Raw signals (465 / 405)")
        self.plot_proc = pg.PlotWidget(title="Processing preview (filtered + baseline)")
        self.plot_out = pg.PlotWidget(title="Output")
        for pw in (self.plot_raw, self.plot_proc, self.plot_out):
            optimize_plot_widget(pw)

        self.plot_proc.setXLink(self.plot_raw)
        self.plot_out.setXLink(self.plot_raw)

        layout.addWidget(self.plot_raw, stretch=2)
        layout.addWidget(self.plot_proc, stretch=2)
        layout.addWidget(self.plot_out, stretch=1)

        self.raw_465_curve = self.plot_raw.plot(pen=pg.mkPen((70, 220, 120), width=1.6))
        self.raw_405_curve = self.plot_raw.plot(pen=pg.mkPen((160, 120, 255), width=1.4, style=QtCore.Qt.PenStyle.DashLine))

        self.proc_465_curve = self.plot_proc.plot(pen=pg.mkPen((70, 220, 120), width=1.2))
        self.proc_405_curve = self.plot_proc.plot(pen=pg.mkPen((160, 120, 255), width=1.2, style=QtCore.Qt.PenStyle.DashLine))
        self.base_465_curve = self.plot_proc.plot(pen=pg.mkPen((190, 190, 190), width=1.0, style=QtCore.Qt.PenStyle.DotLine))
        self.base_405_curve = self.plot_proc.plot(pen=pg.mkPen((190, 190, 190), width=1.0, style=QtCore.Qt.PenStyle.DotLine))

        self.out_curve = self.plot_out.plot(pen=pg.mkPen((80, 170, 255), width=1.4))

        self.raw_vb_right = attach_right_axis(self.plot_raw)
        self.proc_vb_right = attach_right_axis(self.plot_proc)
        self.out_vb_right = attach_right_axis(self.plot_out)

        self.digital_raw = pg.PlotCurveItem(pen=pg.mkPen((220, 180, 80), width=1.1))
        self.digital_proc = pg.PlotCurveItem(pen=pg.mkPen((220, 180, 80), width=1.1))
        self.digital_out = pg.PlotCurveItem(pen=pg.mkPen((220, 180, 80), width=1.1))
        self.raw_vb_right.addItem(self.digital_raw)
        self.proc_vb_right.addItem(self.digital_proc)
        self.out_vb_right.addItem(self.digital_out)

        self.region_selector = pg.LinearRegionItem(values=(0, 1), brush=pg.mkBrush(60, 130, 246, 45))
        self.region_selector.setZValue(10)
        self.plot_raw.addItem(self.region_selector)

        self.log_label = QtWidgets.QLabel("")
        self.log_label.setStyleSheet("color: #AAB4C4; font-size: 8.5pt; padding: 2px 6px;")
        self.log_label.setFixedHeight(18)
        layout.addWidget(self.log_label)

        self.btn_add_region.clicked.connect(self.manualRegionFromSelectorRequested.emit)
        self.btn_show_artifacts.clicked.connect(self.showArtifactsRequested.emit)

    def reset_view_state(self) -> None:
        self._proc_need_autorange = True
        self._out_need_autorange = True

    def set_title(self, text: str) -> None:
        self.lbl_title.setText(text)

    def set_log(self, text: str) -> None:
        self.log_label.setText(text)

    def selector_region(self) -> Tuple[float, float]:
        t0, t1 = self.region_selector.getRegion()
        return float(min(t0, t1)), float(max(t0, t1))

    def _set_digital_overlay(self, tt: Optional[np.ndarray], trig: Optional[np.ndarray], label: str) -> None:
        if tt is None or trig is None or len(tt) < 2 or len(trig) != len(tt):
            self.digital_raw.setData([], [])
            self.digital_proc.setData([], [])
            self.digital_out.setData([], [])
            return

        y = np.asarray(trig, dtype=float)
        y = np.where(np.isfinite(y), y, 0.0)
        mx = float(np.max(y)) if y.size else 1.0
        if mx == 0:
            mx = 1.0
        y = y / mx

        Xs, Ys = make_step_post(np.asarray(tt, dtype=float), y)
        self.digital_raw.setData(Xs, Ys)
        self.digital_proc.setData(Xs, Ys)
        self.digital_out.setData(Xs, Ys)

        for vb in (self.raw_vb_right, self.proc_vb_right, self.out_vb_right):
            vb.setYRange(-0.05, 1.05, padding=0.0)

        self.plot_raw.getPlotItem().getAxis("right").setLabel(label)
        self.plot_proc.getPlotItem().getAxis("right").setLabel(label)
        self.plot_out.getPlotItem().getAxis("right").setLabel(label)

    def show_raw_only(self, time: np.ndarray, raw465: np.ndarray, raw405: np.ndarray,
                      trigger_time: Optional[np.ndarray], trigger: Optional[np.ndarray], trigger_label: str) -> None:
        self.raw_465_curve.setData(time, raw465, connect="finite", skipFiniteCheck=True)
        self.raw_405_curve.setData(time, raw405, connect="finite", skipFiniteCheck=True)
        self._set_digital_overlay(trigger_time, trigger, trigger_label)
        self.plot_raw.enableAutoRange()
        self.plot_raw.autoRange()

        if time.size > 2:
            self.region_selector.setRegion((
                float(time[0] + 0.05 * (time[-1] - time[0])),
                float(time[0] + 0.12 * (time[-1] - time[0]))
            ))

    def update_plots(self, processed: ProcessedTrial) -> None:
        t = processed.time

        self.proc_465_curve.setData(t, processed.filtered_465, connect="finite", skipFiniteCheck=True)
        self.proc_405_curve.setData(t, processed.filtered_405, connect="finite", skipFiniteCheck=True)
        self.base_465_curve.setData(t, processed.baseline_465, connect="finite", skipFiniteCheck=True)
        self.base_405_curve.setData(t, processed.baseline_405, connect="finite", skipFiniteCheck=True)

        self.out_curve.setData(t, processed.output, connect="finite", skipFiniteCheck=True)
        self.plot_out.setTitle(f"Output: {processed.output_label}")

        self._set_digital_overlay(processed.trigger_time, processed.trigger, "Digital")

        if self._proc_need_autorange:
            self.plot_proc.enableAutoRange()
            self.plot_proc.autoRange()
            self._proc_need_autorange = False

        if self._out_need_autorange:
            self.plot_out.enableAutoRange()
            self.plot_out.autoRange()
            self._out_need_autorange = False
