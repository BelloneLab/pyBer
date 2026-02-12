# gui_preprocessing.py
from __future__ import annotations

from typing import Callable, Dict, List, Optional, Tuple
import json
import os

import numpy as np
from PySide6 import QtCore, QtWidgets, QtGui
import pyqtgraph as pg

from analysis_core import (
    ProcessingParams,
    ProcessedTrial,
    OUTPUT_MODES,
    BASELINE_METHODS,
    REFERENCE_FIT_METHODS,
)


def _optimize_plot(w: pg.PlotWidget) -> None:
    w.setMenuEnabled(True)
    w.showGrid(x=True, y=True, alpha=0.25)
    w.setMouseEnabled(x=True, y=True)
    pi = w.getPlotItem()
    pi.setClipToView(True)
    pi.setDownsampling(auto=True, mode="peak")
    pi.setAutoVisible(y=True)
    try:
        pi.enableAutoRange(axis=pg.ViewBox.YAxis, enable=True)
    except TypeError:
        try:
            pi.enableAutoRange(pg.ViewBox.YAxis)
        except Exception:
            pass


def _first_not_none(d: dict, *keys, default=None):
    """
    Return the first present key in d whose value is not None.
    Important: does NOT use boolean evaluation, so numpy arrays are safe.
    """
    for k in keys:
        if k in d:
            v = d.get(k, None)
            if v is not None:
                return v
    return default


def _compact_combo(combo: QtWidgets.QComboBox, min_chars: int = 6) -> None:
    combo.setSizeAdjustPolicy(QtWidgets.QComboBox.AdjustToMinimumContentsLengthWithIcon)
    combo.setMinimumContentsLength(min_chars)
    combo.setSizePolicy(QtWidgets.QSizePolicy.Policy.Expanding, QtWidgets.QSizePolicy.Policy.Fixed)


_FITTED_REF_MODES = {
    "dFF (motion corrected with fitted ref)",
    "zscore (motion corrected with fitted ref)",
}

_OUTPUT_DEFINITIONS: Dict[str, str] = {
    "dFF (non motion corrected)": "dFF = (sig_f - b_sig) / b_sig",
    "zscore (non motion corrected)": "z = zscore(dFF)",
    "dFF (motion corrected via subtraction)": "dFF = dFF_sig - dFF_ref",
    "zscore (motion corrected via subtraction)": "z = zscore(dFF_sig - dFF_ref)",
    "zscore (subtractions)": "z = zscore(dFF_sig) - zscore(dFF_ref)",
    "dFF (motion corrected with fitted ref)": "dFF = (sig_f - fitted_ref) / fitted_ref",
    "zscore (motion corrected with fitted ref)": "z = zscore((sig_f - fitted_ref) / fitted_ref)",
    "Raw signal (465)": "output = filtered/resampled 465 signal",
}


def _system_locale() -> QtCore.QLocale:
    return QtCore.QLocale.system()


def _parse_float_text(text: str) -> Optional[float]:
    s = (text or "").strip()
    if not s:
        return None

    loc = _system_locale()
    val, ok = loc.toDouble(s)
    if ok:
        return float(val)

    # Accept both decimal dot and decimal comma regardless of current locale.
    for cand in (s.replace(",", "."), s.replace(".", ",")):
        val, ok = loc.toDouble(cand)
        if ok:
            return float(val)
        try:
            return float(cand)
        except Exception:
            pass
    return None


class PlaceholderListWidget(QtWidgets.QListWidget):
    def __init__(self, placeholder_text: str = "", parent=None) -> None:
        super().__init__(parent)
        self._placeholder_text = placeholder_text

    def setPlaceholderText(self, text: str) -> None:
        self._placeholder_text = str(text or "")
        self.viewport().update()

    def paintEvent(self, event) -> None:
        super().paintEvent(event)
        if self.count() > 0 or not self._placeholder_text:
            return
        p = QtGui.QPainter(self.viewport())
        p.setRenderHint(QtGui.QPainter.RenderHint.TextAntialiasing, True)
        color = self.palette().color(QtGui.QPalette.ColorRole.Text)
        color.setAlpha(110)
        p.setPen(color)
        rect = self.viewport().rect().adjusted(12, 12, -12, -12)
        p.drawText(
            rect,
            QtCore.Qt.AlignmentFlag.AlignTop | QtCore.Qt.AlignmentFlag.AlignHCenter | QtCore.Qt.TextFlag.TextWordWrap,
            self._placeholder_text,
        )
        p.end()


class CollapsibleSection(QtWidgets.QWidget):
    toggled = QtCore.Signal(bool)

    def __init__(self, title: str, parent=None) -> None:
        super().__init__(parent)
        self._title = title
        self._expanded = True
        self._build_ui()

    def _build_ui(self) -> None:
        root = QtWidgets.QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(4)

        self.btn_toggle = QtWidgets.QToolButton()
        self.btn_toggle.setToolButtonStyle(QtCore.Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        self.btn_toggle.setArrowType(QtCore.Qt.ArrowType.DownArrow)
        self.btn_toggle.setCheckable(True)
        self.btn_toggle.setChecked(True)
        self.btn_toggle.setText(self._title)
        self.btn_toggle.setSizePolicy(QtWidgets.QSizePolicy.Policy.Expanding, QtWidgets.QSizePolicy.Policy.Fixed)
        self.btn_toggle.clicked.connect(self._on_toggle)

        self.lbl_summary = QtWidgets.QLabel("")
        self.lbl_summary.setProperty("class", "hint")
        self.lbl_summary.setAlignment(QtCore.Qt.AlignmentFlag.AlignRight | QtCore.Qt.AlignmentFlag.AlignVCenter)
        self.lbl_summary.setSizePolicy(QtWidgets.QSizePolicy.Policy.Expanding, QtWidgets.QSizePolicy.Policy.Fixed)

        header = QtWidgets.QHBoxLayout()
        header.setContentsMargins(0, 0, 0, 0)
        header.setSpacing(8)
        header.addWidget(self.btn_toggle, stretch=1)
        header.addWidget(self.lbl_summary, stretch=1)

        self.content = QtWidgets.QWidget()
        self.content_layout = QtWidgets.QVBoxLayout(self.content)
        self.content_layout.setContentsMargins(8, 8, 8, 8)
        self.content_layout.setSpacing(8)

        frame = QtWidgets.QFrame()
        frame.setFrameShape(QtWidgets.QFrame.Shape.StyledPanel)
        frame_layout = QtWidgets.QVBoxLayout(frame)
        frame_layout.setContentsMargins(0, 0, 0, 0)
        frame_layout.setSpacing(0)
        frame_layout.addWidget(self.content)
        self.frame = frame

        root.addLayout(header)
        root.addWidget(frame)

    def set_summary(self, text: str) -> None:
        self.lbl_summary.setText(str(text or ""))

    def set_content_widget(self, widget: QtWidgets.QWidget) -> None:
        while self.content_layout.count():
            item = self.content_layout.takeAt(0)
            w = item.widget()
            if w is not None:
                w.setParent(None)
        self.content_layout.addWidget(widget)

    def is_expanded(self) -> bool:
        return self._expanded

    def set_expanded(self, expanded: bool) -> None:
        exp = bool(expanded)
        self._expanded = exp
        self.btn_toggle.setChecked(exp)
        self.btn_toggle.setArrowType(QtCore.Qt.ArrowType.DownArrow if exp else QtCore.Qt.ArrowType.RightArrow)
        self.frame.setVisible(exp)
        self.lbl_summary.setVisible(not exp)
        self.toggled.emit(exp)

    def _on_toggle(self, checked: bool) -> None:
        self._expanded = bool(checked)
        self.btn_toggle.setArrowType(QtCore.Qt.ArrowType.DownArrow if checked else QtCore.Qt.ArrowType.RightArrow)
        self.frame.setVisible(bool(checked))
        self.lbl_summary.setVisible(not bool(checked))
        self.toggled.emit(bool(checked))


# ----------------------------- Metadata dialog -----------------------------

class MetadataForm(QtWidgets.QWidget):
    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        layout = QtWidgets.QVBoxLayout(self)

        form = QtWidgets.QFormLayout()
        form.setFieldGrowthPolicy(QtWidgets.QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)

        self.ed_animal_1 = QtWidgets.QLineEdit()
        self.ed_session = QtWidgets.QLineEdit()
        self.ed_trial = QtWidgets.QLineEdit()
        self.ed_treatment = QtWidgets.QLineEdit()
        self.ed_sex = QtWidgets.QLineEdit()
        self.ed_age = QtWidgets.QLineEdit()
        self.ed_site = QtWidgets.QLineEdit()
        self.ed_sensor = QtWidgets.QLineEdit()
        self.ed_experiment = QtWidgets.QLineEdit()

        self.ed_animal_1.setPlaceholderText("Animal ID")
        self.ed_session.setPlaceholderText("Session")
        self.ed_trial.setPlaceholderText("Trial")
        self.ed_treatment.setPlaceholderText("Treatment")
        self.ed_sex.setPlaceholderText("Sex")
        self.ed_age.setPlaceholderText("Age")
        self.ed_site.setPlaceholderText("Recording site")
        self.ed_sensor.setPlaceholderText("Sensor")
        self.ed_experiment.setPlaceholderText("Experiment")

        form.addRow("Animal ID", self.ed_animal_1)
        form.addRow("Session", self.ed_session)
        form.addRow("Trial", self.ed_trial)
        form.addRow("Treatment", self.ed_treatment)
        form.addRow("Sex", self.ed_sex)
        form.addRow("Age", self.ed_age)
        form.addRow("Recording site", self.ed_site)
        form.addRow("Sensor", self.ed_sensor)
        form.addRow("Experiment", self.ed_experiment)
        layout.addLayout(form)

        grp = QtWidgets.QGroupBox("Additional metadata (key/value)")
        v = QtWidgets.QVBoxLayout(grp)

        self.table = QtWidgets.QTableWidget(0, 2)
        self.table.setHorizontalHeaderLabels(["Key", "Value"])
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.verticalHeader().setVisible(False)
        self.table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.SingleSelection)

        btnrow = QtWidgets.QHBoxLayout()
        self.btn_add = QtWidgets.QPushButton("Add metadata field")
        self.btn_del = QtWidgets.QPushButton("Remove selected field")
        btnrow.addWidget(self.btn_add)
        btnrow.addWidget(self.btn_del)
        btnrow.addStretch(1)

        v.addWidget(self.table)
        v.addLayout(btnrow)
        layout.addWidget(grp, stretch=1)

        self.btn_add.clicked.connect(self._add_row)
        self.btn_del.clicked.connect(self._del_row)

    def _add_row(self, key: str = "", value: str = "") -> None:
        r = self.table.rowCount()
        self.table.insertRow(r)
        self.table.setItem(r, 0, QtWidgets.QTableWidgetItem(str(key)))
        self.table.setItem(r, 1, QtWidgets.QTableWidgetItem(str(value)))

    def _del_row(self) -> None:
        sel = self.table.selectionModel().selectedRows()
        if not sel:
            return
        self.table.removeRow(sel[0].row())

    def from_dict(self, d: Dict[str, str]) -> None:
        self.ed_animal_1.setText(str(d.get("animal_id_1", d.get("animal_id", ""))))
        self.ed_session.setText(str(d.get("session", "")))
        self.ed_trial.setText(str(d.get("trial", "")))
        self.ed_treatment.setText(str(d.get("treatment", "")))
        self.ed_sex.setText(str(d.get("sex", "")))
        self.ed_age.setText(str(d.get("age", "")))
        self.ed_site.setText(str(d.get("recording_site", "")))
        self.ed_sensor.setText(str(d.get("sensor", "")))
        self.ed_experiment.setText(str(d.get("experiment", "")))

        reserved = {
            "animal_id",
            "animal_id_1",
            "session",
            "trial",
            "treatment",
            "sex",
            "age",
            "recording_site",
            "sensor",
            "experiment",
        }
        self.table.setRowCount(0)
        for k, v in (d or {}).items():
            if k in reserved:
                continue
            self._add_row(k, v)

    def to_dict(self) -> Dict[str, str]:
        out: Dict[str, str] = {}
        animal_1 = self.ed_animal_1.text().strip()
        if animal_1:
            out["animal_id_1"] = animal_1
            out["animal_id"] = animal_1
        session = self.ed_session.text().strip()
        trial = self.ed_trial.text().strip()
        treatment = self.ed_treatment.text().strip()
        sex = self.ed_sex.text().strip()
        age = self.ed_age.text().strip()
        site = self.ed_site.text().strip()
        sensor = self.ed_sensor.text().strip()
        experiment = self.ed_experiment.text().strip()
        if session:
            out["session"] = session
        if trial:
            out["trial"] = trial
        if treatment:
            out["treatment"] = treatment
        if sex:
            out["sex"] = sex
        if age:
            out["age"] = age
        if site:
            out["recording_site"] = site
        if sensor:
            out["sensor"] = sensor
        if experiment:
            out["experiment"] = experiment

        for r in range(self.table.rowCount()):
            k_item = self.table.item(r, 0)
            v_item = self.table.item(r, 1)
            k = (k_item.text().strip() if k_item else "")
            v = (v_item.text().strip() if v_item else "")
            if k:
                out[k] = v
        return out


class MetadataDialog(QtWidgets.QDialog):
    def __init__(
        self,
        channels: List[str],
        existing: Optional[Dict[str, Dict[str, str]]] = None,
        defaults: Optional[Dict[str, str]] = None,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Metadata")
        self.setModal(True)
        self._channels = list(channels or [])
        self._existing = dict(existing or {})
        self._defaults = dict(defaults or {})
        self._forms: Dict[str, MetadataForm] = {}
        self._build_ui()
        self._load_initial()

    def _build_ui(self) -> None:
        self.resize(700, 520)
        layout = QtWidgets.QVBoxLayout(self)

        self.tabs = QtWidgets.QTabWidget()
        for ch in self._channels:
            form = MetadataForm()
            self._forms[ch] = form
            self.tabs.addTab(form, ch)
        layout.addWidget(self.tabs)

        tmpl_row = QtWidgets.QHBoxLayout()
        self.btn_save_template = QtWidgets.QPushButton("Save template")
        self.btn_load_template = QtWidgets.QPushButton("Load template")
        tmpl_row.addWidget(self.btn_save_template)
        tmpl_row.addWidget(self.btn_load_template)
        tmpl_row.addStretch(1)
        layout.addLayout(tmpl_row)

        row = QtWidgets.QHBoxLayout()
        row.addStretch(1)
        self.btn_ok = QtWidgets.QPushButton("OK")
        self.btn_cancel = QtWidgets.QPushButton("Cancel")
        self.btn_ok.setDefault(True)
        row.addWidget(self.btn_ok)
        row.addWidget(self.btn_cancel)
        layout.addLayout(row)

        self.btn_ok.clicked.connect(self.accept)
        self.btn_cancel.clicked.connect(self.reject)
        self.btn_save_template.clicked.connect(self._save_template)
        self.btn_load_template.clicked.connect(self._load_template)

    def _merged_defaults_for(self, ch: str) -> Dict[str, str]:
        base = dict(self._defaults or {})
        base.update(self._existing.get(ch, {}))
        return base

    def _load_initial(self) -> None:
        if not self._forms:
            return
        for ch, form in self._forms.items():
            form.from_dict(self._merged_defaults_for(ch))

    def _current_form(self) -> Optional[MetadataForm]:
        idx = self.tabs.currentIndex()
        if idx < 0:
            return None
        ch = self.tabs.tabText(idx)
        return self._forms.get(ch)

    def _save_template(self) -> None:
        form = self._current_form()
        if not form:
            return
        start_dir = os.getcwd()
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self,
            "Save metadata template",
            os.path.join(start_dir, "metadata_template.json"),
            "JSON files (*.json)",
        )
        if not path:
            return
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(form.to_dict(), f, indent=2)
        except Exception:
            pass

    def _load_template(self) -> None:
        form = self._current_form()
        if not form:
            return
        start_dir = os.getcwd()
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Load metadata template",
            start_dir,
            "JSON files (*.json)",
        )
        if not path:
            return
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict):
                form.from_dict({str(k): str(v) for k, v in data.items()})
        except Exception:
            pass

    def get_metadata(self) -> Dict[str, Dict[str, str]]:
        return {ch: form.to_dict() for ch, form in self._forms.items()}


# ----------------------------- Artifact panel -----------------------------

class ArtifactPanel(QtWidgets.QDialog):
    regionsChanged = QtCore.Signal(list)
    selectionChanged = QtCore.Signal(list)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Artifact Regions")
        self.setModal(False)
        self.resize(560, 520)

        self._regions: List[Tuple[float, float]] = []
        self._auto_regions: List[Tuple[float, float]] = []
        self._auto_checked: List[bool] = []
        self._auto_updating = False
        self._build_ui()

    def _build_ui(self) -> None:
        layout = QtWidgets.QVBoxLayout(self)

        auto_group = QtWidgets.QGroupBox("Auto-detected (threshold)")
        auto_layout = QtWidgets.QVBoxLayout(auto_group)
        self.table_auto = QtWidgets.QTableWidget(0, 4)
        self.table_auto.setHorizontalHeaderLabels(["ID", "Remove", "Start (s)", "End (s)"])
        self.table_auto.horizontalHeader().setStretchLastSection(True)
        self.table_auto.verticalHeader().setVisible(False)
        self.table_auto.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows)
        self.table_auto.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.ExtendedSelection)
        auto_layout.addWidget(self.table_auto)
        layout.addWidget(auto_group)

        manual_group = QtWidgets.QGroupBox("Manual artifacts")
        manual_layout = QtWidgets.QVBoxLayout(manual_group)
        self.table = QtWidgets.QTableWidget(0, 3)
        self.table.setHorizontalHeaderLabels(["ID", "Start (s)", "End (s)"])
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.verticalHeader().setVisible(False)
        self.table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.ExtendedSelection)

        manual_layout.addWidget(self.table)

        addrow = QtWidgets.QHBoxLayout()
        self.ed_start = QtWidgets.QDoubleSpinBox()
        self.ed_end = QtWidgets.QDoubleSpinBox()
        for ed in (self.ed_start, self.ed_end):
            ed.setDecimals(3)
            ed.setRange(-1e9, 1e9)
            ed.setKeyboardTracking(False)
            ed.setMinimumWidth(140)

        self.btn_add = QtWidgets.QPushButton("Add")
        self.btn_update = QtWidgets.QPushButton("Update selected")
        self.btn_del = QtWidgets.QPushButton("Delete selected")
        self.btn_clear = QtWidgets.QPushButton("Clear manual")

        addrow.addWidget(QtWidgets.QLabel("Start:"))
        addrow.addWidget(self.ed_start)
        addrow.addWidget(QtWidgets.QLabel("End:"))
        addrow.addWidget(self.ed_end)
        addrow.addStretch(1)
        addrow.addWidget(self.btn_add)
        manual_layout.addLayout(addrow)

        btnrow = QtWidgets.QHBoxLayout()
        btnrow.addWidget(self.btn_update)
        btnrow.addWidget(self.btn_del)
        btnrow.addWidget(self.btn_clear)
        btnrow.addStretch(1)

        self.btn_close = QtWidgets.QPushButton("Close")
        btnrow.addWidget(self.btn_close)
        manual_layout.addLayout(btnrow)
        layout.addWidget(manual_group)

        self.btn_add.clicked.connect(self._on_add)
        self.btn_update.clicked.connect(self._on_update_selected)
        self.btn_del.clicked.connect(self._on_delete_selected)
        self.btn_clear.clicked.connect(self._on_clear)
        self.btn_close.clicked.connect(self._on_close)

        self.table.itemSelectionChanged.connect(self._sync_edits_from_selected)
        self.table.itemSelectionChanged.connect(self._emit_selection)
        self.table_auto.itemSelectionChanged.connect(self._emit_selection)
        self.table_auto.itemChanged.connect(self._on_auto_item_changed)

    def set_regions(self, regions: List[Tuple[float, float]]) -> None:
        self._regions = [(float(a), float(b)) for a, b in (regions or [])]
        self._regions = [(min(a, b), max(a, b)) for a, b in self._regions]
        self._rebuild_table()

    def set_auto_regions(
        self,
        regions: List[Tuple[float, float]],
        checked_regions: Optional[List[Tuple[float, float]]] = None,
    ) -> None:
        self._auto_regions = [(float(a), float(b)) for a, b in (regions or [])]
        self._auto_regions = [(min(a, b), max(a, b)) for a, b in self._auto_regions]
        self._auto_regions.sort(key=lambda x: x[0])

        if checked_regions is None:
            self._auto_checked = [True for _ in self._auto_regions]
        else:
            checked = [(min(a, b), max(a, b)) for a, b in (checked_regions or [])]
            self._auto_checked = [self._region_in_list(r, checked) for r in self._auto_regions]
        self._rebuild_auto_table()

    def regions(self) -> List[Tuple[float, float]]:
        return list(self._regions)

    def _region_in_list(self, target: Tuple[float, float], regions: List[Tuple[float, float]], tol: float = 1e-3) -> bool:
        return any((abs(target[0] - a) <= tol and abs(target[1] - b) <= tol) for a, b in regions)

    def _auto_checked_regions(self) -> List[Tuple[float, float]]:
        out = []
        for keep, reg in zip(self._auto_checked, self._auto_regions):
            if keep:
                out.append(reg)
        return out

    def _combined_regions(self) -> List[Tuple[float, float]]:
        regs = self._auto_checked_regions() + list(self._regions)
        regs.sort(key=lambda x: x[0])
        return regs

    def _rebuild_table(self) -> None:
        self.table.setRowCount(0)
        for idx, (a, b) in enumerate(self._regions, start=1):
            r = self.table.rowCount()
            self.table.insertRow(r)
            id_item = QtWidgets.QTableWidgetItem(str(idx))
            id_item.setFlags(id_item.flags() & ~QtCore.Qt.ItemFlag.ItemIsEditable)
            id_item.setTextAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
            self.table.setItem(r, 0, id_item)
            self.table.setItem(r, 1, QtWidgets.QTableWidgetItem(f"{a:.3f}"))
            self.table.setItem(r, 2, QtWidgets.QTableWidgetItem(f"{b:.3f}"))
        self._emit_selection()

    def _rebuild_auto_table(self) -> None:
        self._auto_updating = True
        try:
            self.table_auto.setRowCount(0)
            for i, (a, b) in enumerate(self._auto_regions, start=1):
                r = self.table_auto.rowCount()
                self.table_auto.insertRow(r)
                id_item = QtWidgets.QTableWidgetItem(str(i))
                id_item.setFlags(id_item.flags() & ~QtCore.Qt.ItemFlag.ItemIsEditable)
                id_item.setTextAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
                self.table_auto.setItem(r, 0, id_item)

                chk = QtWidgets.QTableWidgetItem("")
                chk.setFlags(QtCore.Qt.ItemFlag.ItemIsEnabled | QtCore.Qt.ItemFlag.ItemIsUserCheckable)
                chk.setCheckState(QtCore.Qt.CheckState.Checked if self._auto_checked[i - 1] else QtCore.Qt.CheckState.Unchecked)
                chk.setTextAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
                self.table_auto.setItem(r, 1, chk)

                start_item = QtWidgets.QTableWidgetItem(f"{a:.3f}")
                start_item.setFlags(start_item.flags() & ~QtCore.Qt.ItemFlag.ItemIsEditable)
                self.table_auto.setItem(r, 2, start_item)

                end_item = QtWidgets.QTableWidgetItem(f"{b:.3f}")
                end_item.setFlags(end_item.flags() & ~QtCore.Qt.ItemFlag.ItemIsEditable)
                self.table_auto.setItem(r, 3, end_item)
        finally:
            self._auto_updating = False
        self._emit_selection()

    def _emit(self) -> None:
        self.regionsChanged.emit(self._combined_regions())

    def _emit_selection(self) -> None:
        selected: List[Tuple[float, float]] = []
        rows = self.table.selectionModel().selectedRows()
        for r in rows:
            idx = r.row()
            if 0 <= idx < len(self._regions):
                selected.append(self._regions[idx])
        rows_auto = self.table_auto.selectionModel().selectedRows()
        for r in rows_auto:
            idx = r.row()
            if 0 <= idx < len(self._auto_regions):
                selected.append(self._auto_regions[idx])
        self.selectionChanged.emit(selected)

    def _sync_edits_from_selected(self) -> None:
        rows = self.table.selectionModel().selectedRows()
        if not rows:
            return
        r = rows[0].row()
        if 0 <= r < len(self._regions):
            a, b = self._regions[r]
            self.ed_start.setValue(float(a))
            self.ed_end.setValue(float(b))

    def _on_auto_item_changed(self, item: QtWidgets.QTableWidgetItem) -> None:
        if self._auto_updating:
            return
        if item.column() != 1:
            return
        row = item.row()
        if 0 <= row < len(self._auto_checked):
            self._auto_checked[row] = (item.checkState() == QtCore.Qt.CheckState.Checked)
            self._emit()

    def _on_add(self) -> None:
        a = float(self.ed_start.value())
        b = float(self.ed_end.value())
        if not np.isfinite(a) or not np.isfinite(b):
            return
        a, b = (min(a, b), max(a, b))
        self._regions.append((a, b))
        self._regions.sort(key=lambda x: x[0])
        self._rebuild_table()
        self._emit()

    def _on_update_selected(self) -> None:
        rows = self.table.selectionModel().selectedRows()
        if not rows:
            return
        r = rows[0].row()
        a = float(self.ed_start.value())
        b = float(self.ed_end.value())
        a, b = (min(a, b), max(a, b))
        if 0 <= r < len(self._regions):
            self._regions[r] = (a, b)
            self._regions.sort(key=lambda x: x[0])
            self._rebuild_table()
            self._emit()

    def _on_delete_selected(self) -> None:
        rows = self.table.selectionModel().selectedRows()
        if not rows:
            return
        idx = sorted({r.row() for r in rows}, reverse=True)
        changed = False
        for r in idx:
            if 0 <= r < len(self._regions):
                del self._regions[r]
                changed = True
        if changed:
            self._rebuild_table()
            self._emit()

    def _on_clear(self) -> None:
        self._regions = []
        self._rebuild_table()
        self._emit()

    def _on_close(self) -> None:
        parent = self.parentWidget()
        if isinstance(parent, QtWidgets.QDockWidget):
            parent.setVisible(False)
        else:
            self.hide()

    def closeEvent(self, event) -> None:
        parent = self.parentWidget()
        if isinstance(parent, QtWidgets.QDockWidget):
            event.ignore()
            parent.setVisible(False)
            return
        super().closeEvent(event)


# ----------------------------- File queue panel -----------------------------

class FileQueuePanel(QtWidgets.QGroupBox):
    openFileRequested = QtCore.Signal()
    openFolderRequested = QtCore.Signal()
    selectionChanged = QtCore.Signal()

    channelChanged = QtCore.Signal(str)
    triggerChanged = QtCore.Signal(str)
    timeWindowChanged = QtCore.Signal()

    updatePreviewRequested = QtCore.Signal()
    metadataRequested = QtCore.Signal()
    exportRequested = QtCore.Signal()
    toggleArtifactsRequested = QtCore.Signal()
    advancedOptionsRequested = QtCore.Signal()
    qcRequested = QtCore.Signal()
    batchQcRequested = QtCore.Signal()

    def __init__(self, parent=None) -> None:
        super().__init__("Data", parent)
        self._current_dir_hint: str = ""
        self._build_ui()

    def _build_ui(self) -> None:
        v = QtWidgets.QVBoxLayout(self)
        v.setSpacing(8)
        v.setContentsMargins(8, 8, 8, 8)

        # Top actions
        top_row = QtWidgets.QHBoxLayout()
        self.btn_open = QtWidgets.QPushButton("Open File")
        self.btn_folder = QtWidgets.QPushButton("Add Folder")
        self.btn_open.setProperty("class", "bluePrimarySmall")
        self.btn_folder.setProperty("class", "blueSecondarySmall")
        for b in (self.btn_open, self.btn_folder):
            b.setSizePolicy(QtWidgets.QSizePolicy.Policy.Expanding, QtWidgets.QSizePolicy.Policy.Fixed)
        top_row.addWidget(self.btn_open)
        top_row.addWidget(self.btn_folder)

        # File list fills available height
        self.list_files = PlaceholderListWidget("Drop files here or click Open File")
        self.list_files.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.ExtendedSelection)
        self.list_files.setMinimumHeight(180)

        self.btn_remove_file = QtWidgets.QPushButton("Remove selected")
        self.btn_remove_file.setProperty("class", "blueSecondarySmall")
        self.btn_remove_file.setSizePolicy(QtWidgets.QSizePolicy.Policy.Expanding, QtWidgets.QSizePolicy.Policy.Fixed)
        self.btn_remove_file.setEnabled(False)
        self.btn_remove_file.clicked.connect(self._remove_selected_files)

        # Selection block
        self.grp_sel = QtWidgets.QGroupBox("Selection")
        form = QtWidgets.QGridLayout(self.grp_sel)
        form.setContentsMargins(8, 8, 8, 8)
        form.setHorizontalSpacing(6)
        form.setVerticalSpacing(6)

        self.combo_channel = QtWidgets.QComboBox()
        self.combo_channel.setMinimumWidth(60)
        _compact_combo(self.combo_channel, min_chars=6)

        self.combo_trigger = QtWidgets.QComboBox()
        self.combo_trigger.setMinimumWidth(60)
        _compact_combo(self.combo_trigger, min_chars=6)
        self.combo_trigger.addItem("")

        self.edit_time_start = QtWidgets.QLineEdit()
        self.edit_time_end = QtWidgets.QLineEdit()
        for ed in (self.edit_time_start, self.edit_time_end):
            ed.setPlaceholderText("Start (s)" if ed is self.edit_time_start else "End (s)")
            val = QtGui.QDoubleValidator(0.0, 1e9, 3, ed)
            val.setLocale(_system_locale())
            ed.setValidator(val)
            ed.setSizePolicy(QtWidgets.QSizePolicy.Policy.Expanding, QtWidgets.QSizePolicy.Policy.Fixed)

        form.addWidget(QtWidgets.QLabel("Channel"), 0, 0)
        form.addWidget(self.combo_channel, 0, 1, 1, 3)
        form.addWidget(QtWidgets.QLabel("Analog/Digital channel"), 1, 0)
        form.addWidget(self.combo_trigger, 1, 1, 1, 3)
        form.addWidget(QtWidgets.QLabel("Time window"), 2, 0)
        form.addWidget(self.edit_time_start, 2, 1)
        form.addWidget(QtWidgets.QLabel("to"), 2, 2)
        form.addWidget(self.edit_time_end, 2, 3)
        self.btn_cutting = QtWidgets.QPushButton("Cutting / Sectioning")
        self.btn_cutting.setProperty("class", "blueSecondarySmall")
        self.btn_cutting.setSizePolicy(QtWidgets.QSizePolicy.Policy.Expanding, QtWidgets.QSizePolicy.Policy.Fixed)
        form.addWidget(self.btn_cutting, 3, 0, 1, 4)
        form.setColumnStretch(1, 1)
        form.setColumnStretch(3, 1)

        self.lbl_hint = QtWidgets.QLabel("")
        self.lbl_hint.setProperty("class", "hint")
        self.lbl_hint.setTextInteractionFlags(QtCore.Qt.TextInteractionFlag.TextSelectableByMouse)

        v.addLayout(top_row)
        v.addWidget(self.list_files, stretch=1)
        v.addWidget(self.btn_remove_file)
        v.addWidget(self.grp_sel)
        v.addWidget(self.lbl_hint)

        # Legacy action buttons removed from layout; keep hidden buttons for compatibility.
        self.btn_metadata = QtWidgets.QPushButton("Metadata")
        self.btn_update = QtWidgets.QPushButton("Update")
        self.btn_artifacts = QtWidgets.QPushButton("Artifacts")
        self.btn_export = QtWidgets.QPushButton("Export")
        self.btn_advanced = QtWidgets.QPushButton("Cutting / Sectioning")
        self.btn_qc = QtWidgets.QPushButton("Quality check")
        self.btn_qc_batch = QtWidgets.QPushButton("Batch quality metrics")
        for b in (
            self.btn_metadata,
            self.btn_update,
            self.btn_artifacts,
            self.btn_export,
            self.btn_advanced,
            self.btn_qc,
            self.btn_qc_batch,
        ):
            b.setVisible(False)

        self.btn_open.clicked.connect(self.openFileRequested.emit)
        self.btn_folder.clicked.connect(self.openFolderRequested.emit)
        self.list_files.itemSelectionChanged.connect(self.selectionChanged.emit)
        self.list_files.itemSelectionChanged.connect(self._update_remove_button)

        self.combo_channel.currentTextChanged.connect(self.channelChanged.emit)
        self.combo_trigger.currentTextChanged.connect(self.triggerChanged.emit)
        self.edit_time_start.textChanged.connect(lambda *_: self.timeWindowChanged.emit())
        self.edit_time_end.textChanged.connect(lambda *_: self.timeWindowChanged.emit())

        self.btn_update.clicked.connect(self.updatePreviewRequested.emit)
        self.btn_metadata.clicked.connect(self.metadataRequested.emit)
        self.btn_export.clicked.connect(self.exportRequested.emit)
        self.btn_artifacts.clicked.connect(self.toggleArtifactsRequested.emit)
        self.btn_advanced.clicked.connect(self.advancedOptionsRequested.emit)
        self.btn_cutting.clicked.connect(self.advancedOptionsRequested.emit)
        self.btn_qc.clicked.connect(self.qcRequested.emit)
        self.btn_qc_batch.clicked.connect(self.batchQcRequested.emit)

    def set_path_hint(self, text: str) -> None:
        self.lbl_hint.setText(text)
        if text and os.path.isdir(text):
            self._current_dir_hint = text

    def path_hint(self) -> str:
        return self.lbl_hint.text()

    def set_current_dir_hint(self, dir_path: str) -> None:
        self._current_dir_hint = dir_path or ""
        if dir_path:
            self.lbl_hint.setText(dir_path)

    def current_dir_hint(self) -> str:
        return self._current_dir_hint

    def add_file(self, path: str) -> None:
        item = QtWidgets.QListWidgetItem(os.path.basename(path))
        item.setToolTip(path)
        item.setData(QtCore.Qt.ItemDataRole.UserRole, path)
        self.list_files.addItem(item)
        if self.list_files.count() == 1:
            self.list_files.setCurrentRow(0)
            item0 = self.list_files.item(0)
            if item0 is not None:
                item0.setSelected(True)
        try:
            d = os.path.dirname(path)
            if d and os.path.isdir(d):
                self._current_dir_hint = d
        except Exception:
            pass

    def all_paths(self) -> List[str]:
        out: List[str] = []
        for i in range(self.list_files.count()):
            item = self.list_files.item(i)
            if item is None:
                continue
            path = item.data(QtCore.Qt.ItemDataRole.UserRole)
            out.append(str(path if path else item.text()))
        return out

    def selected_paths(self) -> List[str]:
        out: List[str] = []
        for it in self.list_files.selectedItems():
            path = it.data(QtCore.Qt.ItemDataRole.UserRole)
            out.append(str(path if path else it.text()))
        return out

    def set_available_channels(self, chans: List[str]) -> None:
        self.combo_channel.blockSignals(True)
        try:
            self.combo_channel.clear()
            for c in chans:
                self.combo_channel.addItem(c)
        finally:
            self.combo_channel.blockSignals(False)
        if chans:
            self.combo_channel.setCurrentIndex(0)

    def set_available_triggers(self, triggers: List[str]) -> None:
        self.combo_trigger.blockSignals(True)
        try:
            self.combo_trigger.clear()
            self.combo_trigger.addItem("")
            for t in triggers:
                self.combo_trigger.addItem(t)
        finally:
            self.combo_trigger.blockSignals(False)

    def set_channel(self, ch: str) -> None:
        idx = self.combo_channel.findText(ch)
        if idx >= 0:
            self.combo_channel.setCurrentIndex(idx)

    def set_trigger(self, trig: str) -> None:
        idx = self.combo_trigger.findText(trig)
        if idx >= 0:
            self.combo_trigger.setCurrentIndex(idx)

    def time_window(self) -> Tuple[Optional[float], Optional[float]]:
        def _parse(text: str) -> Optional[float]:
            return _parse_float_text(text)

        return _parse(self.edit_time_start.text()), _parse(self.edit_time_end.text())

    def _remove_selected_files(self) -> None:
        selected_items = self.list_files.selectedItems()
        for item in selected_items:
            row = self.list_files.row(item)
            self.list_files.takeItem(row)
        self.selectionChanged.emit()

    def _update_remove_button(self) -> None:
        self.btn_remove_file.setEnabled(len(self.list_files.selectedItems()) > 0)


class SectionParamsDialog(QtWidgets.QDialog):
    def __init__(self, params: ProcessingParams, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Section Parameters")
        self.setModal(True)
        layout = QtWidgets.QVBoxLayout(self)
        self.panel = ParameterPanel(self)
        self.panel.set_params(params)
        layout.addWidget(self.panel)

        row = QtWidgets.QHBoxLayout()
        row.addStretch(1)
        btn_ok = QtWidgets.QPushButton("OK")
        btn_cancel = QtWidgets.QPushButton("Cancel")
        btn_ok.setDefault(True)
        row.addWidget(btn_ok)
        row.addWidget(btn_cancel)
        layout.addLayout(row)

        btn_ok.clicked.connect(self.accept)
        btn_cancel.clicked.connect(self.reject)

    def get_params(self) -> ProcessingParams:
        return self.panel.get_params()


class AdvancedOptionsDialog(QtWidgets.QDialog):
    def __init__(
        self,
        cutouts: List[Tuple[float, float]],
        sections: List[Dict[str, object]],
        base_params: ProcessingParams,
        request_box_select: Optional[Callable[[Callable[[float, float], None]], None]] = None,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Advanced Options")
        self.setModal(False)
        self.resize(720, 520)
        self._base_params = base_params
        self._request_box_select = request_box_select

        layout = QtWidgets.QVBoxLayout(self)

        # Cutouts
        grp_cut = QtWidgets.QGroupBox("Cut out regions (set to NaN, excluded from output)")
        vcut = QtWidgets.QVBoxLayout(grp_cut)
        self.table_cut = QtWidgets.QTableWidget(0, 3)
        self.table_cut.setHorizontalHeaderLabels(["ID", "Start (s)", "End (s)"])
        self.table_cut.horizontalHeader().setStretchLastSection(True)
        self.table_cut.verticalHeader().setVisible(False)
        self.table_cut.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows)
        self.table_cut.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.ExtendedSelection)

        cut_btns = QtWidgets.QHBoxLayout()
        self.btn_cut_add = QtWidgets.QPushButton("Add cut")
        self.btn_cut_del = QtWidgets.QPushButton("Delete cut")
        self.btn_cut_box = QtWidgets.QPushButton("Box select")
        cut_btns.addWidget(self.btn_cut_add)
        cut_btns.addWidget(self.btn_cut_del)
        cut_btns.addWidget(self.btn_cut_box)
        cut_btns.addStretch(1)

        vcut.addWidget(self.table_cut)
        vcut.addLayout(cut_btns)

        # Sections
        grp_sec = QtWidgets.QGroupBox("Sections (processed separately)")
        vsec = QtWidgets.QVBoxLayout(grp_sec)
        self.table_sec = QtWidgets.QTableWidget(0, 4)
        self.table_sec.setHorizontalHeaderLabels(["ID", "Start (s)", "End (s)", "Params"])
        self.table_sec.horizontalHeader().setStretchLastSection(True)
        self.table_sec.verticalHeader().setVisible(False)
        self.table_sec.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows)
        self.table_sec.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.ExtendedSelection)

        sec_btns = QtWidgets.QHBoxLayout()
        self.btn_sec_add = QtWidgets.QPushButton("Add section")
        self.btn_sec_del = QtWidgets.QPushButton("Delete section")
        self.btn_sec_params = QtWidgets.QPushButton("Edit params")
        self.btn_sec_box = QtWidgets.QPushButton("Box select")
        sec_btns.addWidget(self.btn_sec_add)
        sec_btns.addWidget(self.btn_sec_del)
        sec_btns.addWidget(self.btn_sec_params)
        sec_btns.addWidget(self.btn_sec_box)
        sec_btns.addStretch(1)

        vsec.addWidget(self.table_sec)
        vsec.addLayout(sec_btns)

        layout.addWidget(grp_cut)
        layout.addWidget(grp_sec)

        row = QtWidgets.QHBoxLayout()
        row.addStretch(1)
        btn_ok = QtWidgets.QPushButton("OK")
        btn_cancel = QtWidgets.QPushButton("Cancel")
        btn_ok.setDefault(True)
        row.addWidget(btn_ok)
        row.addWidget(btn_cancel)
        layout.addLayout(row)

        self.btn_cut_add.clicked.connect(self._add_cut)
        self.btn_cut_del.clicked.connect(self._del_cut)
        self.btn_sec_add.clicked.connect(self._add_section)
        self.btn_sec_del.clicked.connect(self._del_section)
        self.btn_sec_params.clicked.connect(self._edit_section_params)
        btn_ok.clicked.connect(self.accept)
        btn_cancel.clicked.connect(self.reject)
        self.btn_cut_box.clicked.connect(self._box_select_cut)
        self.btn_sec_box.clicked.connect(self._box_select_section)

        if not self._request_box_select:
            self.btn_cut_box.setEnabled(False)
            self.btn_sec_box.setEnabled(False)

        self._load_cutouts(cutouts)
        self._load_sections(sections)

    def _load_cutouts(self, cutouts: List[Tuple[float, float]]) -> None:
        self.table_cut.setRowCount(0)
        for (a, b) in cutouts or []:
            r = self.table_cut.rowCount()
            self.table_cut.insertRow(r)
            self.table_cut.setItem(r, 1, QtWidgets.QTableWidgetItem(f"{float(a):.3f}"))
            self.table_cut.setItem(r, 2, QtWidgets.QTableWidgetItem(f"{float(b):.3f}"))
        self._refresh_cut_ids()

    def _load_sections(self, sections: List[Dict[str, object]]) -> None:
        self.table_sec.setRowCount(0)
        for sec in sections or []:
            r = self.table_sec.rowCount()
            self.table_sec.insertRow(r)
            start = float(sec.get("start", 0.0))
            end = float(sec.get("end", 0.0))
            params = sec.get("params")
            self._set_section_row(r, start, end, params)
        self._refresh_section_ids()

    def _set_id_item(self, table: QtWidgets.QTableWidget, row: int, idx: int) -> None:
        item = QtWidgets.QTableWidgetItem(str(idx))
        item.setFlags(item.flags() & ~QtCore.Qt.ItemFlag.ItemIsEditable)
        item.setTextAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        table.setItem(row, 0, item)

    def _refresh_cut_ids(self) -> None:
        for r in range(self.table_cut.rowCount()):
            self._set_id_item(self.table_cut, r, r + 1)

    def _refresh_section_ids(self) -> None:
        for r in range(self.table_sec.rowCount()):
            self._set_id_item(self.table_sec, r, r + 1)

    @staticmethod
    def _selected_single_row(table: QtWidgets.QTableWidget) -> Optional[int]:
        rows = table.selectionModel().selectedRows()
        if len(rows) == 1:
            return rows[0].row()
        return None

    def _box_select_cut(self) -> None:
        if not self._request_box_select:
            return
        self._request_box_select(self._apply_box_to_cut)

    def _box_select_section(self) -> None:
        if not self._request_box_select:
            return
        self._request_box_select(self._apply_box_to_section)

    def _apply_box_to_cut(self, t0: float, t1: float) -> None:
        if not np.isfinite(t0) or not np.isfinite(t1) or t0 == t1:
            return
        a, b = (float(min(t0, t1)), float(max(t0, t1)))
        row = self._selected_single_row(self.table_cut)
        if row is None:
            row = self.table_cut.rowCount()
            self.table_cut.insertRow(row)
        self.table_cut.setItem(row, 1, QtWidgets.QTableWidgetItem(f"{a:.3f}"))
        self.table_cut.setItem(row, 2, QtWidgets.QTableWidgetItem(f"{b:.3f}"))
        self._refresh_cut_ids()
        self.table_cut.setCurrentCell(row, 1)

    def _apply_box_to_section(self, t0: float, t1: float) -> None:
        if not np.isfinite(t0) or not np.isfinite(t1) or t0 == t1:
            return
        a, b = (float(min(t0, t1)), float(max(t0, t1)))
        row = self._selected_single_row(self.table_sec)
        params_dict: Optional[Dict[str, object]] = None
        if row is not None:
            item = self.table_sec.item(row, 3)
            if item:
                data = item.data(QtCore.Qt.ItemDataRole.UserRole)
                if isinstance(data, dict):
                    params_dict = data
        if row is None:
            row = self.table_sec.rowCount()
            self.table_sec.insertRow(row)
            params_dict = self._base_params.to_dict()
        self._set_section_row(row, a, b, params_dict)
        self._refresh_section_ids()
        self.table_sec.setCurrentCell(row, 1)

    def _set_section_row(self, row: int, start: float, end: float, params: Optional[Dict[str, object]]) -> None:
        self._set_id_item(self.table_sec, row, row + 1)
        self.table_sec.setItem(row, 1, QtWidgets.QTableWidgetItem(f"{float(start):.3f}"))
        self.table_sec.setItem(row, 2, QtWidgets.QTableWidgetItem(f"{float(end):.3f}"))
        p = ProcessingParams.from_dict(params) if isinstance(params, dict) else self._base_params
        summary = f"{p.output_mode} | {p.baseline_method}"
        item = QtWidgets.QTableWidgetItem(summary)
        item.setData(QtCore.Qt.ItemDataRole.UserRole, p.to_dict())
        self.table_sec.setItem(row, 3, item)

    def _add_cut(self) -> None:
        r = self.table_cut.rowCount()
        self.table_cut.insertRow(r)
        self.table_cut.setItem(r, 1, QtWidgets.QTableWidgetItem(""))
        self.table_cut.setItem(r, 2, QtWidgets.QTableWidgetItem(""))
        self._refresh_cut_ids()

    def _del_cut(self) -> None:
        rows = self.table_cut.selectionModel().selectedRows()
        for row in sorted((r.row() for r in rows), reverse=True):
            self.table_cut.removeRow(row)
        if rows:
            self._refresh_cut_ids()

    def _add_section(self) -> None:
        r = self.table_sec.rowCount()
        self.table_sec.insertRow(r)
        self._set_section_row(r, 0.0, 0.0, self._base_params.to_dict())
        self._refresh_section_ids()

    def _del_section(self) -> None:
        rows = self.table_sec.selectionModel().selectedRows()
        for row in sorted((r.row() for r in rows), reverse=True):
            self.table_sec.removeRow(row)
        if rows:
            self._refresh_section_ids()

    def _edit_section_params(self) -> None:
        rows = self.table_sec.selectionModel().selectedRows()
        if not rows:
            return
        r = rows[0].row()
        item = self.table_sec.item(r, 3)
        params_dict = item.data(QtCore.Qt.ItemDataRole.UserRole) if item else None
        params = ProcessingParams.from_dict(params_dict) if isinstance(params_dict, dict) else self._base_params
        dlg = SectionParamsDialog(params, self)
        if dlg.exec() != QtWidgets.QDialog.DialogCode.Accepted:
            return
        new_params = dlg.get_params()
        self._set_section_row(
            r,
            float(self._read_float(self.table_sec.item(r, 1))),
            float(self._read_float(self.table_sec.item(r, 2))),
            new_params.to_dict(),
        )
        self._refresh_section_ids()

    @staticmethod
    def _read_float(item: Optional[QtWidgets.QTableWidgetItem]) -> float:
        if item is None:
            return 0.0
        val = _parse_float_text(item.text())
        return float(val) if val is not None else 0.0

    def get_cutouts(self) -> List[Tuple[float, float]]:
        out: List[Tuple[float, float]] = []
        for r in range(self.table_cut.rowCount()):
            a = self._read_float(self.table_cut.item(r, 1))
            b = self._read_float(self.table_cut.item(r, 2))
            if b <= a:
                continue
            out.append((a, b))
        return out

    def get_sections(self) -> List[Dict[str, object]]:
        out: List[Dict[str, object]] = []
        for r in range(self.table_sec.rowCount()):
            a = self._read_float(self.table_sec.item(r, 1))
            b = self._read_float(self.table_sec.item(r, 2))
            if b <= a:
                continue
            item = self.table_sec.item(r, 3)
            params_dict = item.data(QtCore.Qt.ItemDataRole.UserRole) if item else None
            out.append({"start": a, "end": b, "params": params_dict})
        return out


# ----------------------------- Parameter panel -----------------------------

class ParameterPanel(QtWidgets.QGroupBox):
    paramsChanged = QtCore.Signal()
    metadataRequested = QtCore.Signal()
    previewRequested = QtCore.Signal()
    artifactsRequested = QtCore.Signal()
    artifactOverlayToggled = QtCore.Signal(bool)
    exportRequested = QtCore.Signal()
    advancedOptionsRequested = QtCore.Signal()
    qcRequested = QtCore.Signal()
    batchQcRequested = QtCore.Signal()

    def __init__(self, parent=None) -> None:
        super().__init__("Processing Parameters", parent)
        self._help_texts = self._build_help_texts()
        self._config_state_exporter: Optional[Callable[[], Dict[str, object]]] = None
        self._config_state_importer: Optional[Callable[[Dict[str, object]], None]] = None
        self._build_ui()
        self._wire()

    def _build_help_texts(self) -> Dict[str, str]:
        return {
            "artifact_mode": (
                "Artifact detection mode uses the 465 signal derivative (dx).\n"
                "- Global MAD: single threshold for the full trace (fast, stable).\n"
                "- Adaptive MAD: threshold computed in sliding windows (handles drift)."
            ),
            "mad_k": (
                "MAD threshold (k) scales the derivative threshold.\n"
                "Higher k = fewer artifacts flagged; lower k = more sensitive."
            ),
            "adaptive_window_s": (
                "Adaptive window length in seconds for windowed MAD.\n"
                "Shorter windows follow local noise; longer windows are smoother."
            ),
            "artifact_pad_s": (
                "Pad (seconds) added around detected artifacts.\n"
                "Larger pad masks more surrounding points."
            ),
            "lowpass_hz": (
                "Low-pass cutoff (Hz) applied before decimation.\n"
                "Lower cutoff removes more high-frequency noise but can blur fast events."
            ),
            "filter_order": (
                "Butterworth filter order. Higher order = sharper cutoff but more ringing risk."
            ),
            "target_fs_hz": (
                "Target sampling rate (Hz) for decimation.\n"
                "Lower values speed processing and plotting but reduce time resolution."
            ),
            "baseline_method": (
                "Baseline method (pybaselines):\n"
                "- asls: asymmetric least squares; uses p to favor baseline below peaks.\n"
                "- arpls: asymmetrically reweighted penalized least squares; robust to peaks.\n"
                "- airpls: adaptive iteratively reweighted penalized least squares; good for drift.\n"
                "Method choice affects how aggressively the baseline follows slow trends."
            ),
            "baseline_lambda": (
                "Baseline lambda (x e y) is the smoothness penalty.\n"
                "Larger values = smoother baseline; smaller values = more flexible baseline."
            ),
            "baseline_diff_order": (
                "Baseline diff_order sets the derivative order for smoothness (usually 2).\n"
                "Higher order enforces smoother curvature."
            ),
            "baseline_max_iter": (
                "Baseline max_iter limits the iterative solver iterations.\n"
                "Higher values may improve convergence but take longer."
            ),
            "baseline_tol": (
                "Baseline tol controls convergence tolerance.\n"
                "Smaller values are stricter but may require more iterations."
            ),
            "asls_p": (
                "AsLS p controls asymmetry (only used for asls).\n"
                "Smaller p keeps baseline below peaks more aggressively."
            ),
            "output_mode": (
                "Defines the exported trace. See formula below."
            ),
            "invert_polarity": (
                "Flips the sign of both raw channels (465 and 405) before any processing.\n"
                "Use this only if your acquisition polarity is inverted."
            ),
            "reference_fit": (
                "Used only for fitted-reference motion correction modes:\n"
                "- OLS: standard linear fit (recommended).\n"
                "- Lasso: sparse linear fit (uses alpha).\n"
                "- RLM (HuberT): robust fit with Huber weighting.\n"
                "Affects how the 405 reference is fit to the 465 signal."
            ),
            "lasso_alpha": (
                "Lasso alpha controls regularization strength (only for Lasso).\n"
                "Higher alpha = stronger shrinkage; lower alpha = closer to OLS.\n"
                "If scikit-learn is unavailable, Lasso falls back to OLS."
            ),
            "rlm_huber_t": (
                "HuberT threshold (t) for robust regression.\n"
                "Lower t is more outlier-resistant; higher t is closer to OLS."
            ),
            "rlm_max_iter": (
                "Maximum iterations for robust regression reweighting."
            ),
            "rlm_tol": (
                "Convergence tolerance for robust regression.\n"
                "Smaller values are stricter but may take more iterations."
            ),
        }

    def _show_help(self, key: str, title: str) -> None:
        text = self._help_texts.get(key, "No help available for this setting.")
        QtWidgets.QMessageBox.information(self, title, text)

    def _label_with_help(self, text: str, key: str) -> QtWidgets.QWidget:
        w = QtWidgets.QWidget()
        h = QtWidgets.QHBoxLayout(w)
        h.setContentsMargins(0, 0, 0, 0)
        h.setSpacing(6)
        lbl = QtWidgets.QLabel(text)
        btn = QtWidgets.QPushButton("?")
        btn.setProperty("class", "help")
        btn.setFixedSize(22, 22)
        btn.setToolTip("Explain this setting")
        btn.clicked.connect(lambda *_: self._show_help(key, text))
        h.addWidget(lbl)
        h.addStretch(1)
        h.addWidget(btn)
        return w

    def _build_ui(self) -> None:
        root = QtWidgets.QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(8)

        def mk_dspin(minw=60, decimals=3) -> QtWidgets.QDoubleSpinBox:
            s = QtWidgets.QDoubleSpinBox()
            s.setMinimumWidth(minw)
            s.setDecimals(decimals)
            s.setKeyboardTracking(False)
            s.setLocale(_system_locale())
            s.setGroupSeparatorShown(False)
            return s

        def mk_spin(minw=60) -> QtWidgets.QSpinBox:
            s = QtWidgets.QSpinBox()
            s.setMinimumWidth(minw)
            s.setKeyboardTracking(False)
            return s

        # Artifacts controls
        self.cb_artifact = QtWidgets.QCheckBox("Enable artifact detection")
        self.cb_artifact.setChecked(True)
        self.cb_show_artifact_overlay = QtWidgets.QCheckBox("Show artifact detection overlay")
        self.cb_show_artifact_overlay.setChecked(True)
        self.cb_show_artifact_overlay.setToolTip(
            "Toggle detected artifact interval overlays on the raw plot."
        )
        self.combo_artifact = QtWidgets.QComboBox()
        self.combo_artifact.addItems(["Global MAD (dx)", "Adaptive MAD (windowed)"])
        _compact_combo(self.combo_artifact, min_chars=6)
        self.spin_mad = mk_dspin()
        self.spin_mad.setRange(1.0, 50.0)
        self.spin_mad.setValue(8.0)
        self.spin_adapt_win = mk_dspin()
        self.spin_adapt_win.setRange(0.2, 60.0)
        self.spin_adapt_win.setValue(5.0)
        self.spin_pad = mk_dspin()
        self.spin_pad.setRange(0.0, 10.0)
        self.spin_pad.setValue(0.25)

        art_content = QtWidgets.QWidget()
        art_form = QtWidgets.QFormLayout(art_content)
        art_form.setFieldGrowthPolicy(QtWidgets.QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)
        art_form.addRow(self.cb_artifact)
        art_form.addRow(self.cb_show_artifact_overlay)
        art_form.addRow(self._label_with_help("Method", "artifact_mode"), self.combo_artifact)
        art_form.addRow(self._label_with_help("MAD threshold (k)", "mad_k"), self.spin_mad)
        art_form.addRow(self._label_with_help("Adaptive window (s)", "adaptive_window_s"), self.spin_adapt_win)
        art_form.addRow(self._label_with_help("Artifact pad (s)", "artifact_pad_s"), self.spin_pad)

        # Filtering controls
        self.cb_filtering = QtWidgets.QCheckBox("Enable filtering")
        self.cb_filtering.setChecked(True)
        self.spin_lowpass = mk_dspin()
        self.spin_lowpass.setRange(0.1, 200.0)
        self.spin_lowpass.setValue(12.0)
        self.spin_filt_order = mk_spin()
        self.spin_filt_order.setRange(1, 8)
        self.spin_filt_order.setValue(3)
        self.spin_target_fs = mk_dspin(decimals=1)
        self.spin_target_fs.setRange(1.0, 1000.0)
        self.spin_target_fs.setValue(100.0)
        self.cb_invert = QtWidgets.QCheckBox("Invert signal polarity (465/405)")
        self.cb_invert.setChecked(False)
        self.cb_invert.setToolTip(
            "Flips both raw channels before artifact detection, filtering, baseline, and output computation."
        )

        filt_content = QtWidgets.QWidget()
        filt_form = QtWidgets.QFormLayout(filt_content)
        filt_form.setFieldGrowthPolicy(QtWidgets.QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)
        filt_form.addRow(self.cb_filtering)
        filt_form.addRow(self._label_with_help("Low-pass cutoff (Hz)", "lowpass_hz"), self.spin_lowpass)
        filt_form.addRow(self._label_with_help("Filter order", "filter_order"), self.spin_filt_order)
        filt_form.addRow(self._label_with_help("Target FS (Hz)", "target_fs_hz"), self.spin_target_fs)
        filt_form.addRow(self._label_with_help("Invert signal polarity", "invert_polarity"), self.cb_invert)

        # Baseline controls
        self.combo_baseline = QtWidgets.QComboBox()
        self.combo_baseline.addItems([m for m in BASELINE_METHODS])
        _compact_combo(self.combo_baseline, min_chars=6)
        self.spin_lam_x = mk_dspin(minw=90, decimals=3)
        self.spin_lam_x.setRange(0.1, 9.999)
        self.spin_lam_x.setValue(1.0)
        self.spin_lam_y = mk_spin(minw=80)
        self.spin_lam_y.setRange(-3, 12)
        self.spin_lam_y.setValue(9)
        self.lbl_lam_preview = QtWidgets.QLabel("= 1e9")
        self.lbl_lam_preview.setProperty("class", "hint")
        self.spin_diff = mk_spin()
        self.spin_diff.setRange(1, 3)
        self.spin_diff.setValue(2)
        self.spin_iter = mk_spin()
        self.spin_iter.setRange(1, 200)
        self.spin_iter.setValue(50)
        self.spin_tol = mk_dspin(decimals=6)
        self.spin_tol.setRange(1e-8, 1e-1)
        self.spin_tol.setValue(1e-3)
        self.spin_asls_p = mk_dspin(decimals=4)
        self.spin_asls_p.setRange(0.001, 0.5)
        self.spin_asls_p.setValue(0.01)

        lam_row = QtWidgets.QHBoxLayout()
        lam_row.setSpacing(6)
        lam_row.addWidget(self.spin_lam_x)
        lam_row.addWidget(QtWidgets.QLabel("e"))
        lam_row.addWidget(self.spin_lam_y)
        lam_row.addWidget(self.lbl_lam_preview, stretch=1)
        lam_widget = QtWidgets.QWidget()
        lam_widget.setLayout(lam_row)

        self.baseline_advanced_group = QtWidgets.QGroupBox("Advanced parameters")
        self.baseline_advanced_group.setVisible(False)
        baseline_adv_form = QtWidgets.QFormLayout(self.baseline_advanced_group)
        baseline_adv_form.addRow(self._label_with_help("diff_order", "baseline_diff_order"), self.spin_diff)
        baseline_adv_form.addRow(self._label_with_help("max_iter", "baseline_max_iter"), self.spin_iter)
        baseline_adv_form.addRow(self._label_with_help("tol", "baseline_tol"), self.spin_tol)
        baseline_adv_form.addRow(self._label_with_help("AsLS p", "asls_p"), self.spin_asls_p)

        self.btn_toggle_advanced = QtWidgets.QPushButton("Show advanced baseline options")
        self.btn_toggle_advanced.setProperty("class", "compactSmall")
        self.btn_toggle_advanced.clicked.connect(self._toggle_advanced_baseline)

        base_content = QtWidgets.QWidget()
        base_form = QtWidgets.QFormLayout(base_content)
        base_form.setFieldGrowthPolicy(QtWidgets.QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)
        base_form.addRow(self._label_with_help("Method", "baseline_method"), self.combo_baseline)
        base_form.addRow(self._label_with_help("Lambda", "baseline_lambda"), lam_widget)
        base_form.addRow(self.btn_toggle_advanced)
        base_form.addRow(self.baseline_advanced_group)

        # Output controls
        self.combo_output = QtWidgets.QComboBox()
        self.combo_output.addItems(OUTPUT_MODES)
        _compact_combo(self.combo_output, min_chars=8)
        self.combo_output.setToolTip("Defines the exported trace. See formula below.")
        for i in range(self.combo_output.count()):
            mode = self.combo_output.itemText(i)
            tip = "Defines the exported trace. See formula below."
            if mode == "zscore (subtractions)":
                tip = "Difference of z-scored channels; not the same as zscore of the difference."
            self.combo_output.setItemData(i, tip, QtCore.Qt.ItemDataRole.ToolTipRole)
        self.ed_output_definition = QtWidgets.QLineEdit()
        self.ed_output_definition.setReadOnly(True)
        self.ed_output_definition.setFocusPolicy(QtCore.Qt.FocusPolicy.NoFocus)
        self.ed_output_definition.setPlaceholderText("Definition updates with output mode")
        self.combo_ref_fit = QtWidgets.QComboBox()
        self.combo_ref_fit.addItems(REFERENCE_FIT_METHODS)
        _compact_combo(self.combo_ref_fit, min_chars=6)
        self.combo_ref_fit.setToolTip("Used only for fitted-reference motion correction modes.")
        self.spin_lasso = mk_dspin(decimals=6)
        self.spin_lasso.setRange(1e-6, 1.0)
        self.spin_lasso.setValue(1e-3)
        self.spin_lasso.setToolTip(
            "Higher alpha means stronger shrinkage; if sklearn is missing, Lasso falls back to OLS."
        )
        self.spin_rlm_huber_t = mk_dspin(decimals=3)
        self.spin_rlm_huber_t.setRange(0.1, 10.0)
        self.spin_rlm_huber_t.setValue(1.345)
        self.spin_rlm_max_iter = mk_spin()
        self.spin_rlm_max_iter.setRange(1, 500)
        self.spin_rlm_max_iter.setValue(50)
        self.spin_rlm_tol = mk_dspin(decimals=8)
        self.spin_rlm_tol.setRange(1e-12, 1e-2)
        self.spin_rlm_tol.setValue(1e-6)

        self.output_params_stack = QtWidgets.QStackedWidget()
        page_none = QtWidgets.QWidget()
        self.output_params_stack.addWidget(page_none)
        page_lasso = QtWidgets.QWidget()
        page_lasso_form = QtWidgets.QFormLayout(page_lasso)
        page_lasso_form.setContentsMargins(0, 0, 0, 0)
        page_lasso_form.addRow(self._label_with_help("Lasso alpha", "lasso_alpha"), self.spin_lasso)
        self.output_params_stack.addWidget(page_lasso)
        page_rlm = QtWidgets.QWidget()
        page_rlm_v = QtWidgets.QVBoxLayout(page_rlm)
        page_rlm_v.setContentsMargins(0, 0, 0, 0)
        robust_group = QtWidgets.QGroupBox("Robust fit")
        robust_form = QtWidgets.QFormLayout(robust_group)
        robust_form.addRow(self._label_with_help("HuberT (t)", "rlm_huber_t"), self.spin_rlm_huber_t)
        robust_form.addRow(self._label_with_help("Max iter", "rlm_max_iter"), self.spin_rlm_max_iter)
        robust_form.addRow(self._label_with_help("Tol", "rlm_tol"), self.spin_rlm_tol)
        page_rlm_v.addWidget(robust_group)
        self.output_params_stack.addWidget(page_rlm)

        out_content = QtWidgets.QWidget()
        out_form = QtWidgets.QFormLayout(out_content)
        out_form.setFieldGrowthPolicy(QtWidgets.QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)
        self.lbl_output_mode = self._label_with_help("Output mode", "output_mode")
        self.lbl_reference_fit = self._label_with_help("Reference fit method", "reference_fit")
        out_form.addRow(self.lbl_output_mode, self.combo_output)
        out_form.addRow("Definition", self.ed_output_definition)
        out_form.addRow(self.lbl_reference_fit, self.combo_ref_fit)
        out_form.addRow(self.output_params_stack)

        # QC + Export card
        self.btn_artifacts_panel = QtWidgets.QPushButton("Artifacts")
        self.btn_qc = QtWidgets.QPushButton("Quality check")
        self.btn_qc_batch = QtWidgets.QPushButton("Batch quality metrics")
        self.btn_export = QtWidgets.QPushButton("Export CSV/H5")
        self.btn_metadata = QtWidgets.QPushButton("Metadata")
        self.btn_advanced = QtWidgets.QPushButton("Cutting / Sectioning")
        self.btn_save_config = QtWidgets.QPushButton("Save config")
        self.btn_load_config = QtWidgets.QPushButton("Load config")
        for b in (
            self.btn_artifacts_panel,
            self.btn_qc,
            self.btn_qc_batch,
            self.btn_export,
            self.btn_metadata,
            self.btn_advanced,
            self.btn_save_config,
            self.btn_load_config,
        ):
            b.setProperty("class", "compactSmall")
            b.setSizePolicy(QtWidgets.QSizePolicy.Policy.Expanding, QtWidgets.QSizePolicy.Policy.Fixed)
        self.btn_export.setProperty("class", "compactPrimarySmall")
        self.btn_save_config.clicked.connect(self._save_config)
        self.btn_load_config.clicked.connect(self._load_config)
        self.btn_artifacts_panel.clicked.connect(self.artifactsRequested.emit)
        self.btn_qc.clicked.connect(self.qcRequested.emit)
        self.btn_qc_batch.clicked.connect(self.batchQcRequested.emit)
        self.btn_export.clicked.connect(self.exportRequested.emit)
        self.btn_metadata.clicked.connect(self.metadataRequested.emit)
        self.btn_advanced.clicked.connect(self.advancedOptionsRequested.emit)

        qc_content = QtWidgets.QWidget()
        qc_grid = QtWidgets.QGridLayout(qc_content)
        qc_grid.setContentsMargins(0, 0, 0, 0)
        qc_grid.setHorizontalSpacing(6)
        qc_grid.setVerticalSpacing(6)
        qc_grid.addWidget(self.btn_export, 0, 0, 1, 2)
        qc_grid.addWidget(self.btn_artifacts_panel, 1, 0)
        qc_grid.addWidget(self.btn_advanced, 1, 1)
        qc_grid.addWidget(self.btn_qc, 2, 0)
        qc_grid.addWidget(self.btn_qc_batch, 2, 1)
        qc_grid.addWidget(self.btn_metadata, 3, 0)
        qc_grid.addWidget(self.btn_save_config, 3, 1)
        qc_grid.addWidget(self.btn_load_config, 4, 1)
        qc_grid.setColumnStretch(0, 1)
        qc_grid.setColumnStretch(1, 1)

        self.lbl_fs = QtWidgets.QLabel("FS: -")
        self.lbl_fs.setProperty("class", "hint")
        qc_grid.addWidget(self.lbl_fs, 5, 0, 1, 2)

        # Cards
        self.card_artifacts = CollapsibleSection("Artifacts")
        self.card_artifacts.set_content_widget(art_content)
        self.card_filtering = CollapsibleSection("Filtering")
        self.card_filtering.set_content_widget(filt_content)
        self.card_baseline = CollapsibleSection("Baseline")
        self.card_baseline.set_content_widget(base_content)
        self.card_output = CollapsibleSection("Output")
        self.card_output.set_content_widget(out_content)
        self.card_actions = CollapsibleSection("QC + Export")
        self.card_actions.set_content_widget(qc_content)
        self.card_actions.set_expanded(True)

        # Expand defaults
        self.card_artifacts.set_expanded(bool(self.cb_artifact.isChecked()))
        self.card_filtering.set_expanded(bool(self.cb_filtering.isChecked()))
        self.card_baseline.set_expanded(True)
        self.card_output.set_expanded(True)

        root.addWidget(self.card_artifacts)
        root.addWidget(self.card_filtering)
        root.addWidget(self.card_baseline)
        root.addWidget(self.card_output)
        root.addWidget(self.card_actions)
        root.addStretch(1)

        self._update_lambda_preview()
        self._update_output_definition()
        self._update_output_controls()
        self._update_section_summaries()

    def _update_artifact_enabled(self) -> None:
        enabled = self.cb_artifact.isChecked()
        self.combo_artifact.setEnabled(enabled)
        self.spin_mad.setEnabled(enabled)
        self.spin_adapt_win.setEnabled(enabled)
        self.spin_pad.setEnabled(enabled)
        if not enabled and self.card_artifacts.is_expanded():
            self.card_artifacts.set_expanded(False)
        self._update_section_summaries()
        self.paramsChanged.emit()

    def _update_filtering_enabled(self) -> None:
        enabled = self.cb_filtering.isChecked()
        self.spin_lowpass.setEnabled(enabled)
        self.spin_filt_order.setEnabled(enabled)
        if not enabled and self.card_filtering.is_expanded():
            self.card_filtering.set_expanded(False)
        self._update_section_summaries()
        self.paramsChanged.emit()

    def _update_lambda_preview(self) -> None:
        lam = self._lambda_value()
        self.lbl_lam_preview.setText(f"= {lam:.2e}")

    def _is_fitted_output_mode(self) -> bool:
        return self.combo_output.currentText().strip() in _FITTED_REF_MODES

    def _update_output_definition(self) -> None:
        mode = self.combo_output.currentText().strip()
        self.ed_output_definition.setText(_OUTPUT_DEFINITIONS.get(mode, ""))
        tip = self.combo_output.currentData(QtCore.Qt.ItemDataRole.ToolTipRole)
        if tip:
            self.combo_output.setToolTip(str(tip))

    def _update_output_controls(self) -> None:
        fitted_mode = self._is_fitted_output_mode()
        self.lbl_reference_fit.setVisible(fitted_mode)
        self.combo_ref_fit.setVisible(fitted_mode)
        self.combo_ref_fit.setEnabled(fitted_mode)

        if not fitted_mode:
            self.output_params_stack.setCurrentIndex(0)
            self.output_params_stack.setVisible(False)
            self._update_section_summaries()
            return

        method = self.combo_ref_fit.currentText()
        if method.startswith("Lasso"):
            idx = 1
        elif method.startswith("RLM"):
            idx = 2
        else:
            idx = 0

        self.output_params_stack.setCurrentIndex(idx)
        self.output_params_stack.setVisible(idx != 0)
        self._update_section_summaries()

    def _fmt_num(self, value: float, decimals: int = 3) -> str:
        text = f"{float(value):.{int(decimals)}f}"
        text = text.rstrip("0").rstrip(".")
        return text if text else "0"

    def _update_section_summaries(self) -> None:
        if self.cb_artifact.isChecked():
            mode = self.combo_artifact.currentText()
            method = "Adaptive MAD" if mode.startswith("Adaptive") else "Global MAD"
            if mode.startswith("Adaptive"):
                summary = (
                    f"{method}, k={self._fmt_num(self.spin_mad.value(), 2)}, "
                    f"window={self._fmt_num(self.spin_adapt_win.value(), 2)}s, "
                    f"pad={self._fmt_num(self.spin_pad.value(), 2)}s"
                )
            else:
                summary = (
                    f"{method}, k={self._fmt_num(self.spin_mad.value(), 2)}, "
                    f"pad={self._fmt_num(self.spin_pad.value(), 2)}s"
                )
        else:
            summary = "Off"
        self.card_artifacts.set_summary(summary)

        if self.cb_filtering.isChecked():
            summary = (
                f"LP {self._fmt_num(self.spin_lowpass.value(), 2)} Hz, "
                f"order {int(self.spin_filt_order.value())}, "
                f"target {self._fmt_num(self.spin_target_fs.value(), 2)} Hz"
            )
        else:
            summary = "Off"
        self.card_filtering.set_summary(summary)

        summary = f"{self.combo_baseline.currentText()}, lambda={self._lambda_value():.2e}"
        self.card_baseline.set_summary(summary)
        self.card_output.set_summary(self.combo_output.currentText())

    def _wire(self) -> None:
        def emit_noargs(*_args) -> None:
            self.paramsChanged.emit()

        widgets = (
            self.combo_artifact,
            self.spin_mad,
            self.spin_adapt_win,
            self.spin_pad,
            self.spin_lowpass,
            self.spin_filt_order,
            self.spin_target_fs,
            self.combo_baseline,
            self.spin_lam_x,
            self.spin_lam_y,
            self.spin_diff,
            self.spin_iter,
            self.spin_tol,
            self.spin_asls_p,
            self.combo_output,
            self.combo_ref_fit,
            self.spin_lasso,
            self.spin_rlm_huber_t,
            self.spin_rlm_max_iter,
            self.spin_rlm_tol,
        )
        for w in widgets:
            if isinstance(w, QtWidgets.QComboBox):
                w.currentIndexChanged.connect(emit_noargs)
            else:
                w.valueChanged.connect(emit_noargs)

        self.spin_lam_x.valueChanged.connect(lambda *_: self._update_lambda_preview())
        self.spin_lam_y.valueChanged.connect(lambda *_: self._update_lambda_preview())
        self.combo_output.currentIndexChanged.connect(lambda *_: self._update_output_definition())
        self.combo_output.currentIndexChanged.connect(lambda *_: self._update_output_controls())
        self.combo_ref_fit.currentIndexChanged.connect(lambda *_: self._update_output_controls())

        # Keep collapsed card summaries synchronized with current values.
        for w in widgets:
            if isinstance(w, QtWidgets.QComboBox):
                w.currentIndexChanged.connect(lambda *_: self._update_section_summaries())
            else:
                w.valueChanged.connect(lambda *_: self._update_section_summaries())

        self.cb_artifact.stateChanged.connect(self._update_artifact_enabled)
        self.cb_filtering.stateChanged.connect(self._update_filtering_enabled)
        self.cb_invert.stateChanged.connect(emit_noargs)
        self.cb_show_artifact_overlay.toggled.connect(lambda v: self.artifactOverlayToggled.emit(bool(v)))

    def _toggle_advanced_baseline(self) -> None:
        """Toggle visibility of baseline advanced parameters."""
        is_visible = self.baseline_advanced_group.isVisible()
        self.baseline_advanced_group.setVisible(not is_visible)
        self.btn_toggle_advanced.setText(
            "Hide advanced baseline options" if not is_visible else "Show advanced baseline options"
        )

    def set_config_state_hooks(
        self,
        exporter: Optional[Callable[[], Dict[str, object]]],
        importer: Optional[Callable[[Dict[str, object]], None]],
    ) -> None:
        self._config_state_exporter = exporter
        self._config_state_importer = importer

    def _save_config(self) -> None:
        """Save current preprocessing parameters to a JSON file."""
        params = self.get_params()
        start_dir = os.getcwd()
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self,
            "Save preprocessing config",
            os.path.join(start_dir, "preprocessing_config.json"),
            "JSON files (*.json)",
        )
        if not path:
            return

        config = {
            "artifact_detection_enabled": self.cb_artifact.isChecked(),
            "artifact_overlay_visible": self.cb_show_artifact_overlay.isChecked(),
            "filtering_enabled": self.cb_filtering.isChecked(),
            "parameters": params.to_dict(),
        }
        if callable(self._config_state_exporter):
            try:
                extra_state = self._config_state_exporter()
                if isinstance(extra_state, dict) and extra_state:
                    config["ui_state"] = extra_state
            except Exception:
                pass

        try:
            import json
            with open(path, "w") as f:
                json.dump(config, f, indent=2)
            QtWidgets.QMessageBox.information(self, "Success", "Configuration saved successfully.")
        except Exception as e:
            QtWidgets.QMessageBox.warning(self, "Error", f"Failed to save config: {e}")

    def _load_config(self) -> None:
        """Load preprocessing parameters from a JSON file."""
        start_dir = os.getcwd()
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Load preprocessing config",
            start_dir,
            "JSON files (*.json)",
        )
        if not path:
            return

        try:
            import json
            with open(path, "r") as f:
                config = json.load(f)

            # Load parameters
            if "parameters" in config:
                params = ProcessingParams.from_dict(config["parameters"])
                self.set_params(params)

            # Load toggles
            if "artifact_detection_enabled" in config:
                self.cb_artifact.setChecked(config["artifact_detection_enabled"])
            if "artifact_overlay_visible" in config:
                self.cb_show_artifact_overlay.setChecked(bool(config["artifact_overlay_visible"]))
            if "filtering_enabled" in config:
                self.cb_filtering.setChecked(config["filtering_enabled"])
            if callable(self._config_state_importer):
                try:
                    ui_state = config.get("ui_state")
                    if isinstance(ui_state, dict):
                        self._config_state_importer(ui_state)
                except Exception:
                    pass

            QtWidgets.QMessageBox.information(self, "Success", "Configuration loaded successfully.")
        except Exception as e:
            QtWidgets.QMessageBox.warning(self, "Error", f"Failed to load config: {e}")

    def _lambda_value(self) -> float:
        x = float(self.spin_lam_x.value())
        y = int(self.spin_lam_y.value())
        return float(x * (10.0 ** y))

    def get_params(self) -> ProcessingParams:
        return ProcessingParams(
            artifact_detection_enabled=self.cb_artifact.isChecked(),
            artifact_mode=self.combo_artifact.currentText(),
            mad_k=float(self.spin_mad.value()),
            adaptive_window_s=float(self.spin_adapt_win.value()),
            artifact_pad_s=float(self.spin_pad.value()),
            lowpass_hz=float(self.spin_lowpass.value()),
            filter_order=int(self.spin_filt_order.value()),
            target_fs_hz=float(self.spin_target_fs.value()),
            baseline_method=self.combo_baseline.currentText(),
            baseline_lambda=float(self._lambda_value()),
            baseline_diff_order=int(self.spin_diff.value()),
            baseline_max_iter=int(self.spin_iter.value()),
            baseline_tol=float(self.spin_tol.value()),
            asls_p=float(self.spin_asls_p.value()),
            output_mode=self.combo_output.currentText(),
            reference_fit=self.combo_ref_fit.currentText(),
            lasso_alpha=float(self.spin_lasso.value()),
            rlm_huber_t=float(self.spin_rlm_huber_t.value()),
            rlm_max_iter=int(self.spin_rlm_max_iter.value()),
            rlm_tol=float(self.spin_rlm_tol.value()),
            invert_polarity=self.cb_invert.isChecked(),
        )

    def set_params(self, params: ProcessingParams) -> None:
        if not params:
            return
        self.cb_artifact.setChecked(bool(getattr(params, "artifact_detection_enabled", True)))
        self.combo_artifact.setCurrentText(str(params.artifact_mode))
        self.spin_mad.setValue(float(params.mad_k))
        self.spin_adapt_win.setValue(float(params.adaptive_window_s))
        self.spin_pad.setValue(float(params.artifact_pad_s))
        self.spin_lowpass.setValue(float(params.lowpass_hz))
        self.spin_filt_order.setValue(int(params.filter_order))
        self.spin_target_fs.setValue(float(params.target_fs_hz))
        self.cb_invert.setChecked(bool(getattr(params, "invert_polarity", False)))

        self.combo_baseline.setCurrentText(str(params.baseline_method))
        lam = float(params.baseline_lambda)
        if lam > 0:
            import math
            y = int(math.floor(math.log10(lam)))
            x = lam / (10.0 ** y)
            self.spin_lam_x.setValue(float(x))
            self.spin_lam_y.setValue(int(y))
        self.spin_diff.setValue(int(params.baseline_diff_order))
        self.spin_iter.setValue(int(params.baseline_max_iter))
        self.spin_tol.setValue(float(params.baseline_tol))
        self.spin_asls_p.setValue(float(params.asls_p))

        self.combo_output.setCurrentText(str(params.output_mode))
        self.combo_ref_fit.setCurrentText(str(params.reference_fit))
        self.spin_lasso.setValue(float(params.lasso_alpha))
        self.spin_rlm_huber_t.setValue(float(getattr(params, "rlm_huber_t", 1.345)))
        self.spin_rlm_max_iter.setValue(int(getattr(params, "rlm_max_iter", 50)))
        self.spin_rlm_tol.setValue(float(getattr(params, "rlm_tol", 1e-6)))

        self._update_lambda_preview()
        self._update_output_definition()
        self._update_output_controls()
        self._update_section_summaries()

    def set_fs_info(self, fs_actual: float, fs_target: float, fs_used: float) -> None:
        self.lbl_fs.setText(f"FS: actual={fs_actual:.2f} Hz -> used={fs_used:.2f} Hz (target={fs_target:.2f})")

    def artifact_overlay_visible(self) -> bool:
        return bool(self.cb_show_artifact_overlay.isChecked())

    def set_artifact_overlay_visible(self, visible: bool) -> None:
        self.cb_show_artifact_overlay.setChecked(bool(visible))

# ----------------------------- Plot dashboard -----------------------------

class ArtifactSelectViewBox(pg.ViewBox):
    dragSelectionFinished = QtCore.Signal(float, float)
    dragSelectionCleared = QtCore.Signal()

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._drag_start = None
        self._drag_enabled = False
        self._rect_item = QtWidgets.QGraphicsRectItem()
        self._rect_item.setPen(pg.mkPen((90, 190, 255), width=1.0))
        self._rect_item.setBrush(pg.mkBrush(90, 190, 255, 40))
        self._rect_item.setZValue(1000)
        self._rect_item.setVisible(False)
        self.addItem(self._rect_item)

    def set_drag_enabled(self, enabled: bool) -> None:
        self._drag_enabled = bool(enabled)
        if not self._drag_enabled:
            self.clear_selection()

    def clear_selection(self) -> None:
        self._rect_item.setVisible(False)
        self._rect_item.setRect(QtCore.QRectF())
        self._drag_start = None

    def mouseDragEvent(self, ev, axis=None) -> None:
        if self._drag_enabled and ev.button() == QtCore.Qt.MouseButton.LeftButton:
            pos = self.mapSceneToView(ev.scenePos())
            if ev.isStart():
                self._drag_start = pos
                self._rect_item.setVisible(True)
            if self._drag_start is not None:
                x0 = self._drag_start.x()
                y0 = self._drag_start.y()
                x1 = pos.x()
                y1 = pos.y()
                rect = QtCore.QRectF(min(x0, x1), min(y0, y1), abs(x1 - x0), abs(y1 - y0))
                self._rect_item.setRect(rect)
            if ev.isFinish():
                if self._drag_start is not None:
                    x0 = self._drag_start.x()
                    x1 = pos.x()
                    if x0 != x1:
                        self.dragSelectionFinished.emit(float(min(x0, x1)), float(max(x0, x1)))
                self._drag_start = None
            ev.accept()
            return
        super().mouseDragEvent(ev, axis=axis)

    def mouseClickEvent(self, ev) -> None:
        if self._drag_enabled and ev.button() == QtCore.Qt.MouseButton.RightButton:
            self.clear_selection()
            self.dragSelectionCleared.emit()
            ev.accept()
            return
        super().mouseClickEvent(ev)


class PlotDashboard(QtWidgets.QWidget):
    manualRegionFromSelectorRequested = QtCore.Signal()
    manualRegionFromDragRequested = QtCore.Signal(float, float)
    clearManualRegionsRequested = QtCore.Signal()
    showArtifactsRequested = QtCore.Signal()
    boxSelectionCleared = QtCore.Signal()
    artifactThresholdsToggled = QtCore.Signal(bool)

    xRangeChanged = QtCore.Signal(float, float)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._sync_guard = False
        self._artifact_overlay_visible = True
        self._artifact_thresholds_visible = True
        self._artifact_regions: List[pg.LinearRegionItem] = []
        self._artifact_region_bounds: List[Tuple[float, float]] = []
        self._artifact_labels: List[pg.TextItem] = []
        self._last_overlay_time: Optional[np.ndarray] = None
        self._last_overlay_signal: Optional[np.ndarray] = None
        self._last_overlay_regions: List[Tuple[float, float]] = []
        self._artifact_pen_default = pg.mkPen((240, 130, 90), width=1.0)
        self._artifact_brush_default = pg.mkBrush(240, 130, 90, 40)
        self._artifact_pen_selected = pg.mkPen((255, 220, 120), width=2.0)
        self._artifact_brush_selected = pg.mkBrush(255, 220, 120, 90)
        self._build_ui()

    def _build_ui(self) -> None:
        v = QtWidgets.QVBoxLayout(self)
        v.setSpacing(8)

        top = QtWidgets.QHBoxLayout()
        self.lbl_title = QtWidgets.QLabel("No file loaded")
        self.lbl_title.setStyleSheet("font-weight: 900; font-size: 12pt;")
        self.lbl_status = QtWidgets.QLabel("Channel: - | A/D: None | Fs: - -> - Hz | Mode: -")
        self.lbl_status.setProperty("class", "hint")
        self.lbl_status.setTextInteractionFlags(QtCore.Qt.TextInteractionFlag.TextSelectableByMouse)
        top.addWidget(self.lbl_title)
        top.addStretch(1)
        top.addWidget(self.lbl_status)
        v.addLayout(top)

        tools = QtWidgets.QHBoxLayout()
        self.btn_add_region = QtWidgets.QPushButton("Add from selector")
        self.btn_clear_regions = QtWidgets.QPushButton("Clear manual")
        self.btn_artifacts = QtWidgets.QPushButton("Artifacts")
        self.btn_box_select = QtWidgets.QPushButton("Box select")
        self.btn_box_select.setCheckable(True)
        self.btn_thresholds = QtWidgets.QPushButton("Thresholds: ON")
        self.btn_thresholds.setCheckable(True)
        self.btn_thresholds.setChecked(True)
        for b in (
            self.btn_add_region,
            self.btn_clear_regions,
            self.btn_artifacts,
            self.btn_box_select,
            self.btn_thresholds,
        ):
            b.setProperty("class", "compactSmall")
        tools.addWidget(self.btn_add_region)
        tools.addWidget(self.btn_clear_regions)
        tools.addWidget(self.btn_artifacts)
        tools.addWidget(self.btn_box_select)
        tools.addWidget(self.btn_thresholds)
        tools.addStretch(1)
        v.addLayout(tools)

        self._raw_vb = ArtifactSelectViewBox()
        self.plot_raw = pg.PlotWidget(viewBox=self._raw_vb, title="Raw signals (465 /405)")
        self.plot_proc = pg.PlotWidget(title="Filtered + baselines")
        self.plot_out = pg.PlotWidget(title="Output")
        for w in (self.plot_raw, self.plot_proc, self.plot_out):
            _optimize_plot(w)

        # Primary axis for 465 signal
        self.curve_465 = self.plot_raw.plot(pen=pg.mkPen((80, 250, 160), width=1.3))

        # Twin axis for 405 (isobestic) signal - share Y axis with 465
        self.plot_raw_pi = self.plot_raw.getPlotItem()
        self.plot_raw_pi.showAxis("right")
        self.plot_raw_pi.getAxis("right").setLabel("405 (isobestic)", color=(160, 120, 255))

        # For true twin axis, we plot both curves on the same plot area
        # The 405 curve will use the same Y axis scaling as 465
        self.curve_405 = self.plot_raw.plot(pen=pg.mkPen((160, 120, 255, 128), width=1.2))  # Alpha for isobestic

        pen_env = pg.mkPen((240, 200, 90), width=1.0, style=QtCore.Qt.PenStyle.DashLine)
        self.curve_thr_hi = self.plot_raw.plot(pen=pen_env)
        self.curve_thr_lo = self.plot_raw.plot(pen=pen_env)

        self.curve_f465 = self.plot_proc.plot(pen=pg.mkPen((80, 250, 160), width=1.1))
        self.curve_f405 = self.plot_proc.plot(pen=pg.mkPen((160, 120, 255), width=1.0))
        self.curve_b465 = self.plot_proc.plot(
            pen=pg.mkPen((220, 220, 220), width=1.0, style=QtCore.Qt.PenStyle.DashLine)
        )
        self.curve_b405 = self.plot_proc.plot(
            pen=pg.mkPen((160, 160, 160), width=1.0, style=QtCore.Qt.PenStyle.DashLine)
        )

        # Add baseline curves to raw plot as well (fainter)
        self.curve_b465_raw = self.plot_raw.plot(
            pen=pg.mkPen((220, 220, 220, 100), width=1.0, style=QtCore.Qt.PenStyle.DashLine)
        )
        self.curve_b405_raw = self.plot_raw.plot(
            pen=pg.mkPen((160, 160, 160, 100), width=1.0, style=QtCore.Qt.PenStyle.DashLine)
        )

        self.curve_out = self.plot_out.plot(pen=pg.mkPen((90, 190, 255), width=1.2))
        self._raw_y_curves = [self.curve_465, self.curve_405, self.curve_b465_raw, self.curve_b405_raw]
        self._proc_y_curves = [self.curve_f465, self.curve_f405, self.curve_b465, self.curve_b405]
        self._out_y_curves = [self.curve_out]

        self.selector = pg.LinearRegionItem(values=(0, 1), brush=(80, 120, 200, 60))
        self.plot_raw.addItem(self.selector)

        self._dio_pen = pg.mkPen((230, 180, 80), width=1.2)
        self.vb_dio_raw, self.curve_dio_raw = self._add_dio_axis(self.plot_raw, "A/D")
        self.vb_dio_proc, self.curve_dio_proc = self._add_dio_axis(self.plot_proc, "A/D")
        self.vb_dio_out, self.curve_dio_out = self._add_dio_axis(self.plot_out, "A/D")

        self.lbl_log = QtWidgets.QLabel("")
        self.lbl_log.setProperty("class", "hint")

        self.plot_splitter = QtWidgets.QSplitter(QtCore.Qt.Orientation.Vertical)
        self.plot_splitter.addWidget(self.plot_raw)
        self.plot_splitter.addWidget(self.plot_proc)
        self.plot_splitter.addWidget(self.plot_out)
        self.plot_splitter.setSizes([320, 260, 260])

        v.addWidget(self.plot_splitter, stretch=1)
        v.addWidget(self.lbl_log)

        self.btn_add_region.clicked.connect(self.manualRegionFromSelectorRequested.emit)
        self.btn_clear_regions.clicked.connect(self.clearManualRegionsRequested.emit)
        self.btn_artifacts.clicked.connect(self.showArtifactsRequested.emit)
        self.btn_box_select.toggled.connect(self._toggle_box_select)
        self.btn_thresholds.toggled.connect(self._on_thresholds_toggled)

        self.plot_raw.getViewBox().sigXRangeChanged.connect(self._emit_xrange_from_any)
        self.plot_proc.getViewBox().sigXRangeChanged.connect(self._emit_xrange_from_any)
        self.plot_out.getViewBox().sigXRangeChanged.connect(self._emit_xrange_from_any)

        self._raw_vb.dragSelectionFinished.connect(self._on_drag_select_finished)
        self._raw_vb.dragSelectionCleared.connect(self._on_drag_select_cleared)

        self._sync_artifact_threshold_curves_visibility()
        self._toggle_box_select(False)

    def _add_dio_axis(self, plot: pg.PlotWidget, label: str):
        pi = plot.getPlotItem()
        vb = pg.ViewBox()
        vb.setMouseEnabled(x=False, y=False)
        vb.setYRange(-0.1, 1.1)
        pi.showAxis("right")
        pi.getAxis("right").setLabel(label)
        pi.scene().addItem(vb)
        pi.getAxis("right").linkToView(vb)
        vb.setXLink(pi.vb)

        curve = pg.PlotCurveItem(pen=self._dio_pen)
        vb.addItem(curve)

        def _update():
            vb.setGeometry(pi.vb.sceneBoundingRect())
            vb.linkedViewChanged(pi.vb, vb.XAxis)

        pi.vb.sigResized.connect(_update)
        _update()
        return vb, curve

    def _emit_xrange_from_any(self, _vb, x_range) -> None:
        if self._sync_guard:
            return
        try:
            x0, x1 = x_range
            self.xRangeChanged.emit(float(x0), float(x1))
        except Exception:
            pass

    def _on_drag_select_finished(self, t0: float, t1: float) -> None:
        if not np.isfinite(t0) or not np.isfinite(t1):
            return
        self.selector.setVisible(True)
        self.selector.setRegion((t0, t1))
        self.manualRegionFromDragRequested.emit(float(min(t0, t1)), float(max(t0, t1)))

    def _on_drag_select_cleared(self) -> None:
        self.selector.setVisible(False)
        self.boxSelectionCleared.emit()

    def _toggle_box_select(self, enabled: bool) -> None:
        self._raw_vb.set_drag_enabled(enabled)
        self.btn_box_select.setText("Box select: ON" if enabled else "Box select")

    def _sync_artifact_threshold_curves_visibility(self) -> None:
        visible = bool(self._artifact_thresholds_visible)
        self.curve_thr_hi.setVisible(visible)
        self.curve_thr_lo.setVisible(visible)
        self.btn_thresholds.blockSignals(True)
        self.btn_thresholds.setChecked(visible)
        self.btn_thresholds.setText("Thresholds: ON" if visible else "Thresholds: OFF")
        self.btn_thresholds.blockSignals(False)

    def _on_thresholds_toggled(self, checked: bool) -> None:
        self.set_artifact_thresholds_visible(bool(checked))
        self.artifactThresholdsToggled.emit(bool(checked))

    def set_xrange_all(self, x0: float, x1: float) -> None:
        self._sync_guard = True
        try:
            self.plot_raw.setXRange(x0, x1, padding=0)
            self.plot_proc.setXRange(x0, x1, padding=0)
            self.plot_out.setXRange(x0, x1, padding=0)
        finally:
            self._sync_guard = False

    def set_full_xrange(self, t: np.ndarray) -> None:
        if t is None or np.asarray(t).size < 2:
            return
        x0 = float(np.nanmin(t))
        x1 = float(np.nanmax(t))
        if not np.isfinite(x0) or not np.isfinite(x1) or x0 == x1:
            return
        self.set_xrange_all(x0, x1)
        self.auto_range_all(x0=x0, x1=x1)

    def _auto_range_plot(
        self,
        plot: pg.PlotWidget,
        curves: List[pg.PlotCurveItem],
        x0: Optional[float],
        x1: Optional[float],
    ) -> None:
        ymins: List[float] = []
        ymaxs: List[float] = []
        for curve in curves:
            try:
                xx, yy = curve.getData()
            except Exception:
                continue
            if xx is None or yy is None:
                continue
            x = np.asarray(xx, float)
            y = np.asarray(yy, float)
            if x.size == 0 or y.size == 0:
                continue
            m = np.isfinite(x) & np.isfinite(y)
            if x0 is not None and x1 is not None:
                m &= (x >= float(x0)) & (x <= float(x1))
            if not np.any(m):
                continue
            yv = y[m]
            ymins.append(float(np.nanmin(yv)))
            ymaxs.append(float(np.nanmax(yv)))

        if ymins and ymaxs:
            ymin = min(ymins)
            ymax = max(ymaxs)
            if not np.isfinite(ymin) or not np.isfinite(ymax):
                return
            if ymin == ymax:
                span = abs(ymin) if ymin != 0 else 1.0
                ymin -= 0.5 * span
                ymax += 0.5 * span
            plot.setYRange(float(ymin), float(ymax), padding=0.08)
        else:
            try:
                plot.getPlotItem().enableAutoRange(axis=pg.ViewBox.YAxis, enable=True)
                plot.getPlotItem().autoRange()
            except Exception:
                pass

    def auto_range_all(self, x0: Optional[float] = None, x1: Optional[float] = None) -> None:
        if x0 is not None and x1 is not None and np.isfinite(x0) and np.isfinite(x1) and x1 > x0:
            self.set_xrange_all(float(x0), float(x1))
        self._auto_range_plot(self.plot_raw, self._raw_y_curves, x0, x1)
        self._auto_range_plot(self.plot_proc, self._proc_y_curves, x0, x1)
        self._auto_range_plot(self.plot_out, self._out_y_curves, x0, x1)

    def set_title(self, text: str) -> None:
        self.lbl_title.setText(text)

    def set_status(self, text: str) -> None:
        self.lbl_status.setText(text or "")

    def set_log(self, msg: str) -> None:
        self.lbl_log.setText(msg)

    def selector_region(self) -> Tuple[float, float]:
        r = self.selector.getRegion()
        return float(min(r)), float(max(r))

    def _scale_reference_to_signal(self, sig: np.ndarray, ref: np.ndarray) -> np.ndarray:
        s = np.asarray(sig, float)
        r = np.asarray(ref, float)
        m = np.isfinite(s) & np.isfinite(r)
        if np.sum(m) < 2:
            return r
        s_mu = float(np.nanmean(s[m]))
        r_mu = float(np.nanmean(r[m]))
        s_std = float(np.nanstd(s[m]))
        r_std = float(np.nanstd(r[m]))
        if not np.isfinite(s_std) or not np.isfinite(r_std) or r_std == 0:
            return r
        return (r - r_mu) * (s_std / r_std) + s_mu

    def _clear_artifact_overlays(self) -> None:
        for item in self._artifact_regions:
            self.plot_raw.removeItem(item)
        for item in self._artifact_labels:
            self.plot_raw.removeItem(item)
        self._artifact_regions = []
        self._artifact_region_bounds = []
        self._artifact_labels = []

    def _update_artifact_overlays(
        self,
        t: np.ndarray,
        raw_sig: np.ndarray,
        regions: Optional[List[Tuple[float, float]]],
    ) -> None:
        self._last_overlay_time = None if t is None else np.asarray(t, float)
        self._last_overlay_signal = None if raw_sig is None else np.asarray(raw_sig, float)
        self._last_overlay_regions = list(regions or [])
        self._clear_artifact_overlays()
        if not self._artifact_overlay_visible:
            return
        if t is None or raw_sig is None or not regions:
            return
        tt = np.asarray(t, float)
        yy = np.asarray(raw_sig, float)
        for idx, (a, b) in enumerate(regions, start=1):
            region = pg.LinearRegionItem(
                values=(float(a), float(b)),
                movable=False,
                brush=self._artifact_brush_default,
                pen=self._artifact_pen_default,
            )
            region.setZValue(8)
            self.plot_raw.addItem(region)
            self._artifact_regions.append(region)
            self._artifact_region_bounds.append((float(a), float(b)))

            mask = (tt >= float(a)) & (tt <= float(b))
            if np.any(mask) and np.any(np.isfinite(yy[mask])):
                y_pos = float(np.nanmax(yy[mask]))
            else:
                (y0, y1) = self.plot_raw.getViewBox().viewRange()[1]
                y_pos = float(y1 - 0.05 * (y1 - y0))

            label = pg.TextItem(str(idx), color=(240, 240, 240), anchor=(0.5, 0.5))
            label.setPos(float((a + b) * 0.5), y_pos)
            label.setZValue(9)
            self.plot_raw.addItem(label)
            self._artifact_labels.append(label)

    def set_artifact_overlay_visible(self, visible: bool) -> None:
        self._artifact_overlay_visible = bool(visible)
        if not self._artifact_overlay_visible:
            self._clear_artifact_overlays()
            return
        if self._last_overlay_time is not None and self._last_overlay_signal is not None:
            self._update_artifact_overlays(
                self._last_overlay_time,
                self._last_overlay_signal,
                self._last_overlay_regions,
            )

    def artifact_overlay_visible(self) -> bool:
        return bool(self._artifact_overlay_visible)

    def set_artifact_thresholds_visible(self, visible: bool) -> None:
        self._artifact_thresholds_visible = bool(visible)
        self._sync_artifact_threshold_curves_visibility()

    def artifact_thresholds_visible(self) -> bool:
        return bool(self._artifact_thresholds_visible)

    def highlight_artifact_regions(self, regions: List[Tuple[float, float]]) -> None:
        if not self._artifact_regions or not self._artifact_region_bounds:
            return
        selected = [(min(a, b), max(a, b)) for a, b in (regions or [])]

        def _overlaps(a: Tuple[float, float], b: Tuple[float, float], tol: float = 1e-3) -> bool:
            return (a[0] <= b[1] + tol) and (b[0] <= a[1] + tol)

        for item, bounds in zip(self._artifact_regions, self._artifact_region_bounds):
            is_sel = any(_overlaps(bounds, s) for s in selected)
            if is_sel:
                item.setBrush(self._artifact_brush_selected)
                item.setPen(self._artifact_pen_selected)
            else:
                item.setBrush(self._artifact_brush_default)
                item.setPen(self._artifact_pen_default)

    def _set_dio(self, t: np.ndarray, dio: Optional[np.ndarray], name: str = "") -> None:
        if dio is None or np.asarray(dio).size == 0:
            self.curve_dio_raw.setData([], [])
            self.curve_dio_proc.setData([], [])
            self.curve_dio_out.setData([], [])
            return

        tt = np.asarray(t, float)
        yy = np.asarray(dio, float)
        n = min(tt.size, yy.size)
        tt, yy = tt[:n], yy[:n]

        self.curve_dio_raw.setData(tt, yy, connect="finite", skipFiniteCheck=True)
        self.curve_dio_proc.setData(tt, yy, connect="finite", skipFiniteCheck=True)
        self.curve_dio_out.setData(tt, yy, connect="finite", skipFiniteCheck=True)

        if name:
            self.plot_raw.getPlotItem().getAxis("right").setLabel(f"A/D ({name})")
            self.plot_proc.getPlotItem().getAxis("right").setLabel(f"A/D ({name})")
            self.plot_out.getPlotItem().getAxis("right").setLabel(f"A/D ({name})")
        else:
            self.plot_raw.getPlotItem().getAxis("right").setLabel("A/D")
            self.plot_proc.getPlotItem().getAxis("right").setLabel("A/D")
            self.plot_out.getPlotItem().getAxis("right").setLabel("A/D")

    # -------------------- Compatibility API expected by main.py --------------------

    def show_raw(self, *args, **kwargs) -> None:
        """
        Backward/forward compatible raw display.

        Supports any of:
          - show_raw(time, raw465, raw405, ...)
          - show_raw(time=..., raw465=..., raw405=...)
          - show_raw(time=..., signal_465=..., reference_405=...)
          - show_raw(time=..., raw_signal=..., raw_reference=...)
        """
        # Positional support: (time, sig, ref)
        t = s = r = None
        if len(args) >= 3:
            t, s, r = args[0], args[1], args[2]
        else:
            # keyword aliases (main.py uses raw465/raw405)
            t = _first_not_none(kwargs, "time", "t", "Time")
            s = _first_not_none(kwargs, "raw465", "signal_465", "raw_signal", "sig", "signal")
            r = _first_not_none(kwargs, "raw405", "reference_405", "raw_reference", "ref", "reference")

        if t is None or s is None or r is None:
            # fail silently but clear plot (prevents hard crashes)
            self.curve_465.setData([], [])
            self.curve_405.setData([], [])
            self.curve_thr_hi.setData([], [])
            self.curve_thr_lo.setData([], [])
            self._sync_artifact_threshold_curves_visibility()
            self._set_dio(np.asarray([]), None, "")
            return

        t = np.asarray(t, float)
        s = np.asarray(s, float)
        r = np.asarray(r, float)
        n = min(t.size, s.size, r.size)
        t, s, r = t[:n], s[:n], r[:n]

        self.curve_465.setData(t, s, connect="finite", skipFiniteCheck=True)
        r_scaled = self._scale_reference_to_signal(s, r)
        self.curve_405.setData(t, r_scaled, connect="finite", skipFiniteCheck=True)
        self.set_full_xrange(t)
        self._clear_artifact_overlays()

        # Thresholds (either scalars or arrays): do not use "or" on arrays.
        thr_hi = _first_not_none(kwargs, "thr_hi", "raw_thr_hi", "mad_hi", "hi_thr")
        thr_lo = _first_not_none(kwargs, "thr_lo", "raw_thr_lo", "mad_lo", "lo_thr")

        if thr_hi is None or thr_lo is None:
            self.curve_thr_hi.setData([], [])
            self.curve_thr_lo.setData([], [])
        else:
            th = np.asarray(thr_hi, float)
            tl = np.asarray(thr_lo, float)
            if th.size == 1:
                th = np.full_like(t, float(th))
            if tl.size == 1:
                tl = np.full_like(t, float(tl))
            nn = min(t.size, th.size, tl.size)
            self.curve_thr_hi.setData(t[:nn], th[:nn], connect="finite", skipFiniteCheck=True)
            self.curve_thr_lo.setData(t[:nn], tl[:nn], connect="finite", skipFiniteCheck=True)
        self._sync_artifact_threshold_curves_visibility()

        # Baselines for raw plot (if available)
        baseline_sig = _first_not_none(kwargs, "baseline_sig", "b_sig", "sig_base")
        baseline_ref = _first_not_none(kwargs, "baseline_ref", "b_ref", "ref_base")

        if baseline_sig is not None:
            y = np.asarray(baseline_sig, float)
            n = min(t.size, y.size)
            self.curve_b465_raw.setData(t[:n], y[:n], connect="finite", skipFiniteCheck=True)
        else:
            self.curve_b465_raw.setData([], [])

        if baseline_ref is not None:
            y = np.asarray(baseline_ref, float)
            n = min(t.size, y.size)
            self.curve_b405_raw.setData(t[:n], y[:n], connect="finite", skipFiniteCheck=True)
        else:
            self.curve_b405_raw.setData([], [])

        dio = _first_not_none(kwargs, "dio", "digital", "dio_y", "trig", "trigger")
        dio_time = _first_not_none(kwargs, "trig_time", "trigger_time", "dio_time", default=t)
        dio_name = _first_not_none(kwargs, "dio_name", "digital_name", "trigger_name", default="") or ""
        self._set_dio(np.asarray(dio_time, float), dio, str(dio_name))

        title = _first_not_none(kwargs, "title", "file_label")
        if title is not None:
            self.set_title(str(title))

    def show_processing(self, *args, **kwargs) -> None:
        """
        Compatible processing display.

        Supports:
          - show_processing(time, sig_f=..., ref_f=..., baseline_sig=..., baseline_ref=...)
          - show_processing(time=..., sig_f=..., ref_f=..., b_sig=..., b_ref=...)
        """
        if len(args) >= 1:
            t = args[0]
        else:
            t = _first_not_none(kwargs, "time", "t", "Time")

        if t is None:
            self.curve_f465.setData([], [])
            self.curve_f405.setData([], [])
            self.curve_b465.setData([], [])
            self.curve_b405.setData([], [])
            self._set_dio(np.asarray([]), None, "")
            return

        t = np.asarray(t, float)

        sig_f = _first_not_none(kwargs, "sig_f", "signal_f", "f465")
        ref_f = _first_not_none(kwargs, "ref_f", "reference_f", "f405")

        baseline_sig = _first_not_none(kwargs, "baseline_sig", "b_sig", "sig_base")
        baseline_ref = _first_not_none(kwargs, "baseline_ref", "b_ref", "ref_base")

        def _set_curve(curve, y):
            if y is None:
                curve.setData([], [])
                return
            y = np.asarray(y, float)
            n = min(t.size, y.size)
            curve.setData(t[:n], y[:n], connect="finite", skipFiniteCheck=True)

        _set_curve(self.curve_f465, sig_f)
        _set_curve(self.curve_f405, ref_f)
        _set_curve(self.curve_b465, baseline_sig)
        _set_curve(self.curve_b405, baseline_ref)

        dio = _first_not_none(kwargs, "dio", "digital", "dio_y")
        dio_name = _first_not_none(kwargs, "dio_name", "digital_name", "trigger_name", default="") or ""
        self._set_dio(t, dio, str(dio_name))
        self.set_full_xrange(t)

    def show_output(self, *args, **kwargs) -> None:
        """
        Compatible output display.

        Supports:
          - show_output(time, output, label="...", dio=..., dio_name=...)
          - show_output(time=..., y=..., label=...)
        """
        if len(args) >= 2:
            t, y = args[0], args[1]
        else:
            t = _first_not_none(kwargs, "time", "t", "Time")
            y = _first_not_none(kwargs, "output", "y", "dff", "zscore")

        if t is None or y is None:
            self.curve_out.setData([], [])
            self._set_dio(np.asarray([]), None, "")
            return

        t = np.asarray(t, float)
        y = np.asarray(y, float)
        n = min(t.size, y.size)
        t, y = t[:n], y[:n]

        self.curve_out.setData(t, y, connect="finite", skipFiniteCheck=True)
        self.set_full_xrange(t)

        label = _first_not_none(kwargs, "label", "output_label", default="Output")
        context = _first_not_none(kwargs, "output_context", "label_context", default="")
        title = f"Output: {label}" if not context else f"Output: {label} | {context}"
        self.plot_out.setTitle(title)

        dio = _first_not_none(kwargs, "dio", "digital", "dio_y")
        dio_name = _first_not_none(kwargs, "dio_name", "digital_name", "trigger_name", default="") or ""
        self._set_dio(t, dio, str(dio_name))

    # -------------------- Modern API (kept) --------------------

    def update_plots(self, processed: ProcessedTrial) -> None:
        t = np.asarray(processed.time, float)
        self.show_raw(
            t, processed.raw_signal, processed.raw_reference,
            dio=processed.dio, dio_name=processed.dio_name,
            thr_hi=processed.raw_thr_hi, thr_lo=processed.raw_thr_lo
        )
        self.show_processing(
            t,
            sig_f=processed.sig_f, ref_f=processed.ref_f,
            baseline_sig=processed.baseline_sig, baseline_ref=processed.baseline_ref,
            dio=processed.dio, dio_name=processed.dio_name
        )
        self.show_output(
            t, processed.output,
            label=processed.output_label,
            output_context=getattr(processed, "output_context", ""),
            dio=processed.dio, dio_name=processed.dio_name
        )
        self._update_artifact_overlays(t, processed.raw_signal, processed.artifact_regions_sec)

