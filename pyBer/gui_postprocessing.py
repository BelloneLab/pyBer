# gui_postprocessing.py
from __future__ import annotations

import os
import re
import json
import logging
from pathlib import Path
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PySide6 import QtCore, QtWidgets, QtGui
import pyqtgraph as pg
import h5py

from analysis_core import ProcessedTrial
from ethovision_process_gui import clean_sheet

_DOCK_STATE_VERSION = 3
_POST_DOCK_STATE_KEY = "post_main_dock_state_v4"
_POST_DOCK_PREFIX = "post."
_PRE_DOCK_PREFIX = "pre."
_BEHAVIOR_PARSE_BINARY = "binary_columns"
_BEHAVIOR_PARSE_TIMESTAMPS = "timestamp_columns"
_LOG = logging.getLogger(__name__)


def _opt_plot(w: pg.PlotWidget) -> None:
    w.setMenuEnabled(True)
    w.showGrid(x=True, y=True, alpha=0.25)
    w.setMouseEnabled(x=True, y=True)
    pi = w.getPlotItem()
    pi.setClipToView(True)
    pi.setDownsampling(auto=True, mode="peak")
    pi.setAutoVisible(y=True)


def _compact_combo(combo: QtWidgets.QComboBox, min_chars: int = 6) -> None:
    combo.setSizeAdjustPolicy(QtWidgets.QComboBox.AdjustToMinimumContentsLengthWithIcon)
    combo.setMinimumContentsLength(min_chars)
    combo.setSizePolicy(QtWidgets.QSizePolicy.Policy.Expanding, QtWidgets.QSizePolicy.Policy.Fixed)


_AIN_SUFFIX_RE = re.compile(r"_AIN0*([0-9]+)$", re.IGNORECASE)


def _strip_ain_suffix(name: str) -> str:
    return _AIN_SUFFIX_RE.sub("", str(name))


def _is_doric_channel_align(text: str) -> bool:
    return "doric" in (text or "").strip().lower()


class FileDropList(QtWidgets.QListWidget):
    filesDropped = QtCore.Signal(list)
    orderChanged = QtCore.Signal()

    def __init__(self, parent: Optional[QtWidgets.QWidget] = None) -> None:
        super().__init__(parent)
        self.setAcceptDrops(True)
        self.setDragEnabled(True)
        self.setDropIndicatorShown(True)
        self.setDefaultDropAction(QtCore.Qt.DropAction.MoveAction)
        self.setDragDropMode(QtWidgets.QAbstractItemView.DragDropMode.DragDrop)

    def dragEnterEvent(self, event: QtGui.QDragEnterEvent) -> None:
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
            return
        super().dragEnterEvent(event)

    def dragMoveEvent(self, event: QtGui.QDragMoveEvent) -> None:
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
            return
        super().dragMoveEvent(event)

    def dropEvent(self, event: QtGui.QDropEvent) -> None:
        if event.mimeData().hasUrls():
            paths = []
            for url in event.mimeData().urls():
                if url.isLocalFile():
                    path = url.toLocalFile()
                    if path:
                        paths.append(path)
            if paths:
                self.filesDropped.emit(paths)
                event.acceptProposedAction()
                return
        super().dropEvent(event)
        self.orderChanged.emit()


def _extract_rising_edges(time: np.ndarray, dio: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    t = np.asarray(time, float)
    x = np.asarray(dio, float)
    if t.size < 2 or x.size != t.size:
        return np.array([], float)
    b = x > threshold
    on = np.where((~b[:-1]) & (b[1:]))[0] + 1
    return t[on]


def _dock_area_to_int(value: object, fallback: int = 2) -> int:
    """
    Convert Qt DockWidgetArea enum/flag objects (or stored values) to int safely.
    Some PySide6 builds do not allow int(Qt enum) directly.
    """
    try:
        enum_value = getattr(value, "value", None)
        if enum_value is not None:
            return int(enum_value)
    except Exception:
        pass
    try:
        if isinstance(value, str):
            v = value.strip().lower()
            if "left" in v:
                return 1
            if "right" in v:
                return 2
            if "top" in v:
                return 4
            if "bottom" in v:
                return 8
    except Exception:
        pass
    try:
        return int(value)
    except Exception:
        return int(fallback)


def _to_bool(value: object, default: bool = False) -> bool:
    """
    Convert mixed QSettings bool payloads (bool/int/str) safely.
    """
    if isinstance(value, bool):
        return value
    if value is None:
        return bool(default)
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        v = value.strip().lower()
        if v in {"1", "true", "yes", "y", "on"}:
            return True
        if v in {"0", "false", "no", "n", "off", ""}:
            return False
    return bool(default)

def _extract_events_with_durations(
    time: np.ndarray,
    dio: np.ndarray,
    threshold: float = 0.5,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns event onset times and durations (seconds) based on threshold crossings.
    """
    t = np.asarray(time, float)
    x = np.asarray(dio, float)
    if t.size < 2 or x.size != t.size:
        return np.array([], float), np.array([], float)

    b = x > threshold
    rising = np.where((~b[:-1]) & (b[1:]))[0] + 1
    falling = np.where((b[:-1]) & (~b[1:]))[0] + 1
    if rising.size == 0 or falling.size == 0:
        return np.array([], float), np.array([], float)

    times = []
    durations = []
    fi = 0
    for ri in rising:
        while fi < falling.size and falling[fi] <= ri:
            fi += 1
        if fi >= falling.size:
            break
        t0 = float(t[ri])
        t1 = float(t[falling[fi]])
        if t1 > t0:
            times.append(t0)
            durations.append(t1 - t0)
        fi += 1

    return np.asarray(times, float), np.asarray(durations, float)


def _extract_onsets_offsets(
    time: np.ndarray,
    dio: np.ndarray,
    threshold: float = 0.5,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    t = np.asarray(time, float)
    x = np.asarray(dio, float)
    if t.size < 2 or x.size != t.size:
        return np.array([], float), np.array([], float), np.array([], float)
    b = x > threshold
    rising = np.where((~b[:-1]) & (b[1:]))[0] + 1
    falling = np.where((b[:-1]) & (~b[1:]))[0] + 1
    if rising.size == 0 or falling.size == 0:
        return np.array([], float), np.array([], float), np.array([], float)

    on = []
    off = []
    dur = []
    fi = 0
    for ri in rising:
        while fi < falling.size and falling[fi] <= ri:
            fi += 1
        if fi >= falling.size:
            break
        t0 = float(t[ri])
        t1 = float(t[falling[fi]])
        if t1 > t0:
            on.append(t0)
            off.append(t1)
            dur.append(t1 - t0)
        fi += 1
    return np.asarray(on, float), np.asarray(off, float), np.asarray(dur, float)

def _binary_columns_from_df(df) -> Tuple[str, Dict[str, np.ndarray]]:
    cols = [c.strip() for c in df.columns]
    time_col = None
    for c in df.columns:
        if str(c).strip().lower() in {"time", "trial time", "recording time"}:
            time_col = c
            break
    if time_col is None and cols:
        time_col = df.columns[0]

    t = np.asarray(df[time_col], float)
    behaviors: Dict[str, np.ndarray] = {}

    for c in df.columns:
        if c == time_col:
            continue
        arr = np.asarray(df[c], float)
        if arr.size == 0:
            continue
        finite = arr[np.isfinite(arr)]
        if finite.size == 0:
            continue
        uniq = np.unique(finite)
        if np.all(np.isin(uniq, [0.0, 1.0])):
            behaviors[str(c)] = arr.astype(float)

    return time_col, behaviors


def _timestamp_columns_from_df(df) -> Dict[str, np.ndarray]:
    import pandas as pd

    behaviors: Dict[str, np.ndarray] = {}
    for c in df.columns:
        name = str(c).strip()
        if not name:
            continue
        if name.lower() in {"time", "trial time", "recording time", "timestamp", "timestamps"}:
            continue
        vals = pd.to_numeric(df[c], errors="coerce")
        arr = np.asarray(vals, float)
        arr = arr[np.isfinite(arr)]
        if arr.size == 0:
            continue
        behaviors[name] = np.sort(np.unique(arr))
    return behaviors


def _load_behavior_csv(path: str, parse_mode: str = _BEHAVIOR_PARSE_BINARY) -> Dict[str, Any]:
    import pandas as pd
    df = pd.read_csv(path)
    if str(parse_mode) == _BEHAVIOR_PARSE_TIMESTAMPS:
        return {"kind": _BEHAVIOR_PARSE_TIMESTAMPS, "time": np.array([], float), "behaviors": _timestamp_columns_from_df(df)}
    time_col, behaviors = _binary_columns_from_df(df)
    return {"kind": _BEHAVIOR_PARSE_BINARY, "time": np.asarray(df[time_col], float), "behaviors": behaviors}


def _load_behavior_ethovision(
    path: str,
    sheet_name: Optional[str] = None,
    parse_mode: str = _BEHAVIOR_PARSE_BINARY,
) -> Dict[str, Any]:
    import pandas as pd

    if sheet_name is None:
        xls = pd.ExcelFile(path, engine="openpyxl")
        sheet_name = xls.sheet_names[0] if xls.sheet_names else None
    if not sheet_name:
        return {"kind": _BEHAVIOR_PARSE_BINARY, "time": np.array([], float), "behaviors": {}}
    if str(parse_mode) == _BEHAVIOR_PARSE_TIMESTAMPS:
        df = pd.read_excel(path, sheet_name=sheet_name, engine="openpyxl")
        return {
            "kind": _BEHAVIOR_PARSE_TIMESTAMPS,
            "time": np.array([], float),
            "behaviors": _timestamp_columns_from_df(df),
            "sheet": sheet_name,
        }
    df = clean_sheet(Path(path), sheet_name, interpolate=True)
    time_col, behaviors = _binary_columns_from_df(df)
    return {"kind": _BEHAVIOR_PARSE_BINARY, "time": np.asarray(df[time_col], float), "behaviors": behaviors, "sheet": sheet_name}


def _compute_psth_matrix(
    t: np.ndarray,
    y: np.ndarray,
    event_times: np.ndarray,
    window: Tuple[float, float],
    baseline_win: Tuple[float, float],
    resample_hz: float,
    smooth_sigma_s: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns:
      tvec (relative time), mat (n_events x n_samples) with NaNs if missing
    """
    t = np.asarray(t, float)
    y = np.asarray(y, float)
    ev = np.asarray(event_times, float)
    ev = ev[np.isfinite(ev)]
    if ev.size == 0:
        return np.array([], float), np.zeros((0, 0), float)

    dt = 1.0 / float(resample_hz)
    tvec = np.arange(window[0], window[1] + 0.5 * dt, dt)

    mat = np.full((ev.size, tvec.size), np.nan, float)

    for i, et in enumerate(ev):
        # baseline
        bmask = (t >= et + baseline_win[0]) & (t <= et + baseline_win[1])
        base = y[bmask]
        if base.size < 5 or not np.any(np.isfinite(base)):
            continue
        bmean = np.nanmean(base)
        bstd = np.nanstd(base)
        if not np.isfinite(bstd) or bstd <= 1e-12:
            bstd = 1.0

        # extract window and interpolate onto tvec
        wmask = (t >= et + window[0]) & (t <= et + window[1])
        tw = t[wmask] - et
        yw = y[wmask]
        good = np.isfinite(tw) & np.isfinite(yw)
        if np.sum(good) < 5:
            continue
        # sparse interpolation
        mat[i, :] = np.interp(tvec, tw[good], (yw[good] - bmean) / bstd)

    if smooth_sigma_s and smooth_sigma_s > 0:
        # simple gaussian smoothing along time axis
        from scipy.ndimage import gaussian_filter1d
        sigma = smooth_sigma_s * resample_hz
        mat = gaussian_filter1d(mat, sigma=sigma, axis=1, mode="nearest")

    return tvec, mat


class PostProcessingPanel(QtWidgets.QWidget):
    # bridge signals to main
    requestCurrentProcessed = QtCore.Signal()
    requestDioList = QtCore.Signal()
    requestDioData = QtCore.Signal(str, str)  # (path, dio)
    statusUpdate = QtCore.Signal(str, int)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._processed: List[ProcessedTrial] = []
        self._dio_cache: Dict[Tuple[str, str], Tuple[np.ndarray, np.ndarray]] = {}  # (path,dio)->(t,x)
        self._behavior_sources: Dict[str, Dict[str, Any]] = {}  # stem->behavior data
        self._last_mat: Optional[np.ndarray] = None
        self._last_tvec: Optional[np.ndarray] = None
        self._last_events: Optional[np.ndarray] = None
        self._last_durations: Optional[np.ndarray] = None
        self._last_metrics: Optional[Dict[str, float]] = None
        self._last_global_metrics: Optional[Dict[str, float]] = None
        self._last_event_rows: List[Dict[str, object]] = []
        self.last_signal_events: Optional[Dict[str, object]] = None
        self.last_behavior_analysis: Optional[Dict[str, object]] = None
        self._event_labels: List[pg.TextItem] = []
        self._event_regions: List[pg.LinearRegionItem] = []
        self._signal_peak_lines: List[pg.InfiniteLine] = []
        self._pre_region: Optional[pg.LinearRegionItem] = None
        self._post_region: Optional[pg.LinearRegionItem] = None
        self._settings = QtCore.QSettings("FiberPhotometryApp", "DoricProcessor")
        self._style = {
            "trace": (90, 190, 255),
            "behavior": (220, 180, 80),
            "avg": (90, 190, 255),
            "heatmap_cmap": "viridis",
            "heatmap_min": None,
            "heatmap_max": None,
        }
        self._section_popups: Dict[str, QtWidgets.QDockWidget] = {}
        self._section_scroll_hosts: Dict[str, QtWidgets.QScrollArea] = {}
        self._section_buttons: Dict[str, QtWidgets.QPushButton] = {}
        self._section_popup_initialized: set[str] = set()
        self._is_restoring_panel_layout: bool = False
        self._panel_layout_persistence_ready: bool = False
        self._last_opened_section: Optional[str] = None
        self._suspend_panel_layout_persistence: bool = False
        self._post_docks_hidden_for_tab_switch: bool = False
        self._post_section_visibility_before_hide: Dict[str, bool] = {}
        self._post_section_state_before_hide: Dict[str, Dict[str, object]] = {}
        self._dock_host: Optional[QtWidgets.QMainWindow] = None
        self._dock_layout_restored: bool = False
        self._app_closing: bool = False
        self._post_snapshot_applied: bool = False
        self._force_fixed_default_layout: bool = False
        self._build_ui()
        self._restore_settings()
        self._panel_layout_persistence_ready = True
        app = QtWidgets.QApplication.instance()
        if app is not None:
            app.aboutToQuit.connect(self._on_about_to_quit)

    def _build_ui(self) -> None:
        root = QtWidgets.QVBoxLayout(self)
        root.setContentsMargins(10, 10, 10, 10)
        root.setSpacing(8)

        grp_src = QtWidgets.QGroupBox("Signal Source")
        vsrc = QtWidgets.QVBoxLayout(grp_src)

        self.tab_sources = QtWidgets.QTabWidget()
        tab_single = QtWidgets.QWidget()
        tab_group = QtWidgets.QWidget()
        self.tab_sources.addTab(tab_single, "Single")
        self.tab_sources.addTab(tab_group, "Group")

        single_layout = QtWidgets.QVBoxLayout(tab_single)
        self.lbl_current = QtWidgets.QLabel("Current: (none)")
        self.btn_use_current = QtWidgets.QPushButton("Use current preprocessed selection")
        self.btn_use_current.setProperty("class", "compactPrimary")
        self.btn_use_current.setSizePolicy(QtWidgets.QSizePolicy.Policy.Ignored, QtWidgets.QSizePolicy.Policy.Fixed)
        self.btn_load_processed_single = QtWidgets.QPushButton("Load processed file (CSV/H5)")
        self.btn_load_processed_single.setProperty("class", "compactSmall")
        self.btn_load_processed_single.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Ignored,
            QtWidgets.QSizePolicy.Policy.Fixed,
        )
        single_layout.addWidget(self.lbl_current)
        single_layout.addWidget(self.btn_use_current)
        single_layout.addWidget(self.btn_load_processed_single)
        single_layout.addStretch(1)

        group_layout = QtWidgets.QVBoxLayout(tab_group)
        self.btn_load_processed = QtWidgets.QPushButton("Load processed files (CSV/H5)")
        self.btn_load_processed.setProperty("class", "compactSmall")
        self.btn_load_processed.setSizePolicy(QtWidgets.QSizePolicy.Policy.Ignored, QtWidgets.QSizePolicy.Policy.Fixed)
        self.lbl_group = QtWidgets.QLabel("(none)")
        self.lbl_group.setProperty("class", "hint")
        group_layout.addWidget(self.btn_load_processed)
        group_layout.addWidget(self.lbl_group)
        group_layout.addStretch(1)

        self.btn_refresh_dio = QtWidgets.QPushButton("Refresh A/D channel list")
        self.btn_refresh_dio.setProperty("class", "compactSmall")
        self.btn_refresh_dio.setSizePolicy(QtWidgets.QSizePolicy.Policy.Ignored, QtWidgets.QSizePolicy.Policy.Fixed)

        vsrc.addWidget(self.tab_sources)
        vsrc.addWidget(self.btn_refresh_dio)

        grp_align = QtWidgets.QGroupBox("Behavior / Events")
        fal = QtWidgets.QFormLayout(grp_align)
        fal.setRowWrapPolicy(QtWidgets.QFormLayout.RowWrapPolicy.WrapLongRows)
        fal.setLabelAlignment(QtCore.Qt.AlignmentFlag.AlignLeft | QtCore.Qt.AlignmentFlag.AlignTop)

        self.combo_align = QtWidgets.QComboBox()
        self.combo_align.addItems(["Analog/Digital channel (from Doric)", "Behavior (CSV/XLSX)"])
        self.combo_align.setCurrentIndex(1)
        _compact_combo(self.combo_align, min_chars=6)

        self.combo_dio = QtWidgets.QComboBox()
        _compact_combo(self.combo_dio, min_chars=6)
        self.combo_dio_polarity = QtWidgets.QComboBox()
        self.combo_dio_polarity.addItems(["Event high (0->1)", "Event low (1->0)"])
        _compact_combo(self.combo_dio_polarity, min_chars=6)
        self.combo_dio_align = QtWidgets.QComboBox()
        self.combo_dio_align.addItems(["Align to onset", "Align to offset"])
        _compact_combo(self.combo_dio_align, min_chars=6)

        self.btn_load_beh = QtWidgets.QPushButton("Load behavior CSV/XLSX...")
        self.btn_load_beh.setProperty("class", "compactSmall")
        self.btn_load_beh.setSizePolicy(QtWidgets.QSizePolicy.Policy.Ignored, QtWidgets.QSizePolicy.Policy.Fixed)
        self.lbl_behavior_file_type = QtWidgets.QLabel("Behavior file type")
        self.combo_behavior_file_type = QtWidgets.QComboBox()
        self.combo_behavior_file_type.addItems(
            [
                "Binary states (time + 0/1 columns)",
                "Timestamps per behavior (columns = behaviors)",
            ]
        )
        self.combo_behavior_file_type.setToolTip(
            "Choose how behavior files are parsed.\n"
            "Binary states expects a time column and 0/1 behavior columns.\n"
            "Timestamps mode expects one column per behavior containing event times."
        )
        _compact_combo(self.combo_behavior_file_type, min_chars=10)
        self.lbl_beh = QtWidgets.QLabel("(none)")
        self.lbl_beh.setProperty("class", "hint")

        # Preprocessed files list
        self.list_preprocessed = FileDropList()
        self.list_preprocessed.setMaximumHeight(120)
        self.list_preprocessed.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.MultiSelection)

        # Behaviors list
        self.list_behaviors = FileDropList()
        self.list_behaviors.setMaximumHeight(120)
        self.list_behaviors.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.MultiSelection)

        # Control buttons for ordering
        order_layout = QtWidgets.QHBoxLayout()
        self.btn_move_up = QtWidgets.QPushButton("Move Up")
        self.btn_move_down = QtWidgets.QPushButton("Move Down")
        self.btn_auto_match = QtWidgets.QPushButton("Auto Match")
        self.btn_remove_pre = QtWidgets.QPushButton("Remove selected")
        self.btn_remove_beh = QtWidgets.QPushButton("Remove selected")
        for b in (self.btn_move_up, self.btn_move_down, self.btn_auto_match, self.btn_remove_pre, self.btn_remove_beh):
            b.setProperty("class", "compactSmall")
            b.setSizePolicy(QtWidgets.QSizePolicy.Policy.Ignored, QtWidgets.QSizePolicy.Policy.Fixed)
        self.btn_move_up.clicked.connect(self._move_selected_up)
        self.btn_move_down.clicked.connect(self._move_selected_down)
        self.btn_auto_match.clicked.connect(self._auto_match_files)
        self.btn_remove_pre.clicked.connect(self._remove_selected_preprocessed)
        self.btn_remove_beh.clicked.connect(self._remove_selected_behaviors)
        order_layout.addWidget(self.btn_move_up)
        order_layout.addWidget(self.btn_move_down)
        order_layout.addWidget(self.btn_auto_match)
        order_layout.addStretch(1)

        files_layout = QtWidgets.QHBoxLayout()
        files_layout.addWidget(QtWidgets.QLabel("Preprocessed Files:"))
        files_layout.addWidget(QtWidgets.QLabel("Behaviors:"))

        lists_layout = QtWidgets.QHBoxLayout()
        pre_col = QtWidgets.QVBoxLayout()
        pre_col.addWidget(self.list_preprocessed)
        pre_col.addWidget(self.btn_remove_pre)
        beh_col = QtWidgets.QVBoxLayout()
        beh_col.addWidget(self.list_behaviors)
        beh_col.addWidget(self.btn_remove_beh)
        lists_layout.addLayout(pre_col)
        lists_layout.addLayout(beh_col)

        fal.addRow("Align source", self.combo_align)
        fal.addRow("A/D channel", self.combo_dio)
        fal.addRow("A/D polarity", self.combo_dio_polarity)
        fal.addRow("A/D align", self.combo_dio_align)
        fal.addRow(self.lbl_behavior_file_type, self.combo_behavior_file_type)
        fal.addRow(self.btn_load_beh)
        fal.addRow("Loaded files", self.lbl_beh)
        fal.addRow(files_layout)
        fal.addRow(lists_layout)
        fal.addRow(order_layout)

        # Behavior controls
        self.combo_behavior_name = QtWidgets.QComboBox()
        _compact_combo(self.combo_behavior_name, min_chars=6)
        self.combo_behavior_align = QtWidgets.QComboBox()
        self.combo_behavior_align.addItems(["Align to onset", "Align to offset", "Transition A->B"])
        _compact_combo(self.combo_behavior_align, min_chars=6)
        self.combo_behavior_from = QtWidgets.QComboBox()
        self.combo_behavior_to = QtWidgets.QComboBox()
        _compact_combo(self.combo_behavior_from, min_chars=6)
        _compact_combo(self.combo_behavior_to, min_chars=6)
        self.spin_transition_gap = QtWidgets.QDoubleSpinBox()
        self.spin_transition_gap.setRange(0, 60)
        self.spin_transition_gap.setValue(1.0)
        self.spin_transition_gap.setDecimals(2)

        self.lbl_trans_from = QtWidgets.QLabel("Transition from")
        self.lbl_trans_to = QtWidgets.QLabel("Transition to")
        self.lbl_trans_gap = QtWidgets.QLabel("Transition gap (s)")
        fal.addRow("Behavior name", self.combo_behavior_name)
        fal.addRow("Behavior align", self.combo_behavior_align)
        fal.addRow(self.lbl_trans_from, self.combo_behavior_from)
        fal.addRow(self.lbl_trans_to, self.combo_behavior_to)
        fal.addRow(self.lbl_trans_gap, self.spin_transition_gap)


        grp_opt = QtWidgets.QGroupBox("PSTH Options")
        fopt = QtWidgets.QFormLayout(grp_opt)
        fopt.setRowWrapPolicy(QtWidgets.QFormLayout.RowWrapPolicy.WrapLongRows)
        fopt.setLabelAlignment(QtCore.Qt.AlignmentFlag.AlignLeft | QtCore.Qt.AlignmentFlag.AlignTop)

        self.spin_pre = QtWidgets.QDoubleSpinBox(); self.spin_pre.setRange(0.1, 60); self.spin_pre.setValue(2.0); self.spin_pre.setDecimals(2)
        self.spin_post= QtWidgets.QDoubleSpinBox(); self.spin_post.setRange(0.1, 120); self.spin_post.setValue(5.0); self.spin_post.setDecimals(2)
        self.spin_b0  = QtWidgets.QDoubleSpinBox(); self.spin_b0.setRange(-60, 0); self.spin_b0.setValue(-1.0); self.spin_b0.setDecimals(2)
        self.spin_b1  = QtWidgets.QDoubleSpinBox(); self.spin_b1.setRange(-60, 0); self.spin_b1.setValue(0.0); self.spin_b1.setDecimals(2)

        self.spin_resample = QtWidgets.QDoubleSpinBox(); self.spin_resample.setRange(1, 1000); self.spin_resample.setValue(50); self.spin_resample.setDecimals(1)
        self.spin_smooth = QtWidgets.QDoubleSpinBox(); self.spin_smooth.setRange(0, 5); self.spin_smooth.setValue(0.0); self.spin_smooth.setDecimals(2)

        self.cb_filter_events = QtWidgets.QCheckBox("Enable event filters")
        self.cb_filter_events.setChecked(True)
        self.btn_hide_filters = QtWidgets.QToolButton()
        self.btn_hide_filters.setText("Hide")
        self.btn_hide_filters.setCheckable(True)
        self.spin_event_start = QtWidgets.QSpinBox(); self.spin_event_start.setRange(1, 1000000); self.spin_event_start.setValue(1)
        self.spin_event_end = QtWidgets.QSpinBox(); self.spin_event_end.setRange(0, 1000000); self.spin_event_end.setValue(0)
        self.spin_group_window = QtWidgets.QDoubleSpinBox(); self.spin_group_window.setRange(0.0, 1e6); self.spin_group_window.setValue(0.0); self.spin_group_window.setDecimals(3)
        self.spin_dur_min = QtWidgets.QDoubleSpinBox(); self.spin_dur_min.setRange(0, 1e6); self.spin_dur_min.setValue(0.0); self.spin_dur_min.setDecimals(2)
        self.spin_dur_max = QtWidgets.QDoubleSpinBox(); self.spin_dur_max.setRange(0, 1e6); self.spin_dur_max.setValue(0.0); self.spin_dur_max.setDecimals(2)
        self.cb_metrics = QtWidgets.QCheckBox("Enable PSTH metrics")
        self.cb_metrics.setChecked(True)
        self.btn_hide_metrics = QtWidgets.QToolButton()
        self.btn_hide_metrics.setText("Hide")
        self.btn_hide_metrics.setCheckable(True)
        self.combo_metric = QtWidgets.QComboBox()
        self.combo_metric.addItems(["AUC", "Mean z"])
        _compact_combo(self.combo_metric, min_chars=6)
        self.spin_metric_pre0 = QtWidgets.QDoubleSpinBox(); self.spin_metric_pre0.setRange(-120, 0); self.spin_metric_pre0.setValue(-1.0); self.spin_metric_pre0.setDecimals(2)
        self.spin_metric_pre1 = QtWidgets.QDoubleSpinBox(); self.spin_metric_pre1.setRange(-120, 0); self.spin_metric_pre1.setValue(0.0); self.spin_metric_pre1.setDecimals(2)
        self.spin_metric_post0 = QtWidgets.QDoubleSpinBox(); self.spin_metric_post0.setRange(0, 120); self.spin_metric_post0.setValue(0.0); self.spin_metric_post0.setDecimals(2)
        self.spin_metric_post1 = QtWidgets.QDoubleSpinBox(); self.spin_metric_post1.setRange(0, 120); self.spin_metric_post1.setValue(1.0); self.spin_metric_post1.setDecimals(2)

        for w in (
            self.spin_pre, self.spin_post, self.spin_b0, self.spin_b1,
            self.spin_resample, self.spin_smooth,
            self.spin_event_start, self.spin_event_end, self.spin_group_window, self.spin_dur_min, self.spin_dur_max,
            self.spin_metric_pre0, self.spin_metric_pre1, self.spin_metric_post0, self.spin_metric_post1,
        ):
            w.setMinimumWidth(60)
            w.setSizePolicy(QtWidgets.QSizePolicy.Policy.Ignored, QtWidgets.QSizePolicy.Policy.Fixed)

        win_row = QtWidgets.QGridLayout()
        win_row.setHorizontalSpacing(6)
        win_row.setContentsMargins(0, 0, 0, 0)
        win_pre = QtWidgets.QLabel("Pre:")
        win_post = QtWidgets.QLabel("Post:")
        win_pre.setMinimumWidth(35)
        win_post.setMinimumWidth(35)
        win_row.addWidget(win_pre, 0, 0)
        win_row.addWidget(self.spin_pre, 0, 1)
        win_row.addWidget(win_post, 0, 2)
        win_row.addWidget(self.spin_post, 0, 3)
        win_row.setColumnStretch(1, 1)
        win_row.setColumnStretch(3, 1)
        win_widget = QtWidgets.QWidget(); win_widget.setLayout(win_row)

        base_row = QtWidgets.QGridLayout()
        base_row.setHorizontalSpacing(6)
        base_row.setContentsMargins(0, 0, 0, 0)
        base_start = QtWidgets.QLabel("Start:")
        base_end = QtWidgets.QLabel("End:")
        base_start.setMinimumWidth(45)
        base_end.setMinimumWidth(35)
        base_row.addWidget(base_start, 0, 0)
        base_row.addWidget(self.spin_b0, 0, 1)
        base_row.addWidget(base_end, 0, 2)
        base_row.addWidget(self.spin_b1, 0, 3)
        base_row.setColumnStretch(1, 1)
        base_row.setColumnStretch(3, 1)
        base_widget = QtWidgets.QWidget(); base_widget.setLayout(base_row)

        metric_pre_row = QtWidgets.QGridLayout()
        metric_pre_row.setHorizontalSpacing(6)
        metric_pre_row.setContentsMargins(0, 0, 0, 0)
        metric_pre_start = QtWidgets.QLabel("Start:")
        metric_pre_end = QtWidgets.QLabel("End:")
        metric_pre_start.setMinimumWidth(45)
        metric_pre_end.setMinimumWidth(35)
        metric_pre_row.addWidget(metric_pre_start, 0, 0)
        metric_pre_row.addWidget(self.spin_metric_pre0, 0, 1)
        metric_pre_row.addWidget(metric_pre_end, 0, 2)
        metric_pre_row.addWidget(self.spin_metric_pre1, 0, 3)
        metric_pre_row.setColumnStretch(1, 1)
        metric_pre_row.setColumnStretch(3, 1)
        metric_pre_widget = QtWidgets.QWidget(); metric_pre_widget.setLayout(metric_pre_row)

        metric_post_row = QtWidgets.QGridLayout()
        metric_post_row.setHorizontalSpacing(6)
        metric_post_row.setContentsMargins(0, 0, 0, 0)
        metric_post_start = QtWidgets.QLabel("Start:")
        metric_post_end = QtWidgets.QLabel("End:")
        metric_post_start.setMinimumWidth(45)
        metric_post_end.setMinimumWidth(35)
        metric_post_row.addWidget(metric_post_start, 0, 0)
        metric_post_row.addWidget(self.spin_metric_post0, 0, 1)
        metric_post_row.addWidget(metric_post_end, 0, 2)
        metric_post_row.addWidget(self.spin_metric_post1, 0, 3)
        metric_post_row.setColumnStretch(1, 1)
        metric_post_row.setColumnStretch(3, 1)
        metric_post_widget = QtWidgets.QWidget(); metric_post_widget.setLayout(metric_post_row)

        fopt.addRow("Window (s)", win_widget)
        fopt.addRow("Baseline (s)", base_widget)
        fopt.addRow("Resample (Hz)", self.spin_resample)
        filt_row = QtWidgets.QHBoxLayout()
        filt_row.addWidget(self.cb_filter_events)
        filt_row.addStretch(1)
        filt_row.addWidget(self.btn_hide_filters)
        filt_widget = QtWidgets.QWidget(); filt_widget.setLayout(filt_row)
        fopt.addRow(filt_widget)
        self.lbl_event_start = QtWidgets.QLabel("Event index start (1-based)")
        self.lbl_event_end = QtWidgets.QLabel("Event index end (0=all)")
        self.lbl_group_window = QtWidgets.QLabel("Group events within (s) (0=off)")
        self.lbl_dur_min = QtWidgets.QLabel("Event duration min (s)")
        self.lbl_dur_max = QtWidgets.QLabel("Event duration max (s)")
        fopt.addRow(self.lbl_event_start, self.spin_event_start)
        fopt.addRow(self.lbl_event_end, self.spin_event_end)
        fopt.addRow(self.lbl_group_window, self.spin_group_window)
        fopt.addRow(self.lbl_dur_min, self.spin_dur_min)
        fopt.addRow(self.lbl_dur_max, self.spin_dur_max)
        fopt.addRow("Gaussian smooth sigma (s)", self.spin_smooth)
        met_row = QtWidgets.QHBoxLayout()
        met_row.addWidget(self.cb_metrics)
        met_row.addStretch(1)
        met_row.addWidget(self.btn_hide_metrics)
        met_widget = QtWidgets.QWidget(); met_widget.setLayout(met_row)
        fopt.addRow(met_widget)
        self.lbl_metric = QtWidgets.QLabel("Metric")
        self.lbl_metric_pre = QtWidgets.QLabel("Metric pre (s)")
        self.lbl_metric_post = QtWidgets.QLabel("Metric post (s)")
        fopt.addRow(self.lbl_metric, self.combo_metric)
        fopt.addRow(self.lbl_metric_pre, metric_pre_widget)
        fopt.addRow(self.lbl_metric_post, metric_post_widget)

        self.cb_global_metrics = QtWidgets.QCheckBox("Enable global metrics")
        self.cb_global_metrics.setChecked(True)
        fopt.addRow(self.cb_global_metrics)

        self.spin_global_start = QtWidgets.QDoubleSpinBox()
        self.spin_global_start.setRange(-1e6, 1e6)
        self.spin_global_start.setValue(0.0)
        self.spin_global_start.setDecimals(2)
        self.spin_global_end = QtWidgets.QDoubleSpinBox()
        self.spin_global_end.setRange(-1e6, 1e6)
        self.spin_global_end.setValue(0.0)
        self.spin_global_end.setDecimals(2)
        self.spin_global_start.setMinimumWidth(60)
        self.spin_global_end.setMinimumWidth(60)
        self.spin_global_start.setSizePolicy(QtWidgets.QSizePolicy.Policy.Ignored, QtWidgets.QSizePolicy.Policy.Fixed)
        self.spin_global_end.setSizePolicy(QtWidgets.QSizePolicy.Policy.Ignored, QtWidgets.QSizePolicy.Policy.Fixed)

        global_row = QtWidgets.QGridLayout()
        global_row.setHorizontalSpacing(6)
        global_row.setContentsMargins(0, 0, 0, 0)
        global_row.addWidget(QtWidgets.QLabel("Start:"), 0, 0)
        global_row.addWidget(self.spin_global_start, 0, 1)
        global_row.addWidget(QtWidgets.QLabel("End:"), 0, 2)
        global_row.addWidget(self.spin_global_end, 0, 3)
        global_row.setColumnStretch(1, 1)
        global_row.setColumnStretch(3, 1)
        global_widget = QtWidgets.QWidget()
        global_widget.setLayout(global_row)
        fopt.addRow("Global range (s)", global_widget)

        self.cb_global_amp = QtWidgets.QCheckBox("Peak amplitude")
        self.cb_global_amp.setChecked(True)
        self.cb_global_freq = QtWidgets.QCheckBox("Transient frequency")
        self.cb_global_freq.setChecked(True)
        global_opts = QtWidgets.QHBoxLayout()
        global_opts.addWidget(self.cb_global_amp)
        global_opts.addWidget(self.cb_global_freq)
        global_opts.addStretch(1)
        global_opts_widget = QtWidgets.QWidget()
        global_opts_widget.setLayout(global_opts)
        fopt.addRow("Global metrics", global_opts_widget)

        self.lbl_global_metrics = QtWidgets.QLabel("Global metrics: -")
        self.lbl_global_metrics.setProperty("class", "hint")
        fopt.addRow("", self.lbl_global_metrics)

        for w in (self.spin_global_start, self.spin_global_end):
            w.setMinimumWidth(60)
            w.setSizePolicy(QtWidgets.QSizePolicy.Policy.Ignored, QtWidgets.QSizePolicy.Policy.Fixed)

        self.btn_compute = QtWidgets.QPushButton("Post-process (compute PSTH)")
        self.btn_compute.setProperty("class", "compactPrimarySmall")
        self.btn_compute.setSizePolicy(QtWidgets.QSizePolicy.Policy.Ignored, QtWidgets.QSizePolicy.Policy.Fixed)
        self.btn_update = QtWidgets.QPushButton("Update Preview")
        self.btn_update.setProperty("class", "compactSmall")
        self.btn_update.setSizePolicy(QtWidgets.QSizePolicy.Policy.Ignored, QtWidgets.QSizePolicy.Policy.Fixed)
        self.btn_export = QtWidgets.QPushButton("Export results")
        self.btn_export.setProperty("class", "compactPrimarySmall")
        self.btn_export.setSizePolicy(QtWidgets.QSizePolicy.Policy.Ignored, QtWidgets.QSizePolicy.Policy.Fixed)
        self.btn_export_img = QtWidgets.QPushButton("Export images")
        self.btn_export_img.setProperty("class", "compactSmall")
        self.btn_export_img.setSizePolicy(QtWidgets.QSizePolicy.Policy.Ignored, QtWidgets.QSizePolicy.Policy.Fixed)
        self.btn_style = QtWidgets.QPushButton("Styling")
        self.btn_style.setProperty("class", "compactSmall")
        self.btn_style.setSizePolicy(QtWidgets.QSizePolicy.Policy.Ignored, QtWidgets.QSizePolicy.Policy.Fixed)
        self.btn_save_cfg = QtWidgets.QPushButton("Save config")
        self.btn_save_cfg.setProperty("class", "compactSmall")
        self.btn_save_cfg.setSizePolicy(QtWidgets.QSizePolicy.Policy.Ignored, QtWidgets.QSizePolicy.Policy.Fixed)
        self.btn_load_cfg = QtWidgets.QPushButton("Load config")
        self.btn_load_cfg.setProperty("class", "compactSmall")
        self.btn_load_cfg.setSizePolicy(QtWidgets.QSizePolicy.Policy.Ignored, QtWidgets.QSizePolicy.Policy.Fixed)

        grp_signal = QtWidgets.QGroupBox("Signal Event Analyzer")
        f_signal = QtWidgets.QFormLayout(grp_signal)
        f_signal.setRowWrapPolicy(QtWidgets.QFormLayout.RowWrapPolicy.WrapLongRows)
        f_signal.setLabelAlignment(QtCore.Qt.AlignmentFlag.AlignLeft | QtCore.Qt.AlignmentFlag.AlignTop)
        self.combo_signal_source = QtWidgets.QComboBox()
        self.combo_signal_source.addItems(["Use processed output trace (loaded file)", "Use PSTH input trace"])
        _compact_combo(self.combo_signal_source, min_chars=10)
        self.combo_signal_scope = QtWidgets.QComboBox()
        self.combo_signal_scope.addItems(["Per file", "Pooled"])
        _compact_combo(self.combo_signal_scope, min_chars=8)
        self.combo_signal_file = QtWidgets.QComboBox()
        _compact_combo(self.combo_signal_file, min_chars=8)
        self.combo_signal_method = QtWidgets.QComboBox()
        self.combo_signal_method.addItems(["SciPy find_peaks"])
        self.spin_peak_prominence = QtWidgets.QDoubleSpinBox()
        self.spin_peak_prominence.setRange(0.0, 1e6)
        self.spin_peak_prominence.setValue(0.5)
        self.spin_peak_prominence.setDecimals(4)
        self.spin_peak_height = QtWidgets.QDoubleSpinBox()
        self.spin_peak_height.setRange(0.0, 1e6)
        self.spin_peak_height.setValue(0.0)
        self.spin_peak_height.setDecimals(4)
        self.spin_peak_distance = QtWidgets.QDoubleSpinBox()
        self.spin_peak_distance.setRange(0.0, 3600.0)
        self.spin_peak_distance.setValue(0.5)
        self.spin_peak_distance.setDecimals(3)
        self.spin_peak_smooth = QtWidgets.QDoubleSpinBox()
        self.spin_peak_smooth.setRange(0.0, 30.0)
        self.spin_peak_smooth.setValue(0.0)
        self.spin_peak_smooth.setDecimals(3)
        self.combo_peak_baseline = QtWidgets.QComboBox()
        self.combo_peak_baseline.addItems(["Use trace as-is", "Detrend with rolling median", "Detrend with rolling mean"])
        self.spin_peak_baseline_window = QtWidgets.QDoubleSpinBox()
        self.spin_peak_baseline_window.setRange(0.1, 3600.0)
        self.spin_peak_baseline_window.setValue(10.0)
        self.spin_peak_baseline_window.setDecimals(2)
        self.spin_peak_rate_bin = QtWidgets.QDoubleSpinBox()
        self.spin_peak_rate_bin.setRange(0.5, 3600.0)
        self.spin_peak_rate_bin.setValue(60.0)
        self.spin_peak_rate_bin.setDecimals(2)
        self.spin_peak_auc_window = QtWidgets.QDoubleSpinBox()
        self.spin_peak_auc_window.setRange(0.0, 30.0)
        self.spin_peak_auc_window.setValue(0.5)
        self.spin_peak_auc_window.setDecimals(3)
        self.cb_peak_overlay = QtWidgets.QCheckBox("Show detected peaks on trace")
        self.cb_peak_overlay.setChecked(True)
        self.btn_detect_peaks = QtWidgets.QPushButton("Detect peaks")
        self.btn_detect_peaks.setProperty("class", "compactPrimarySmall")
        self.btn_export_peaks = QtWidgets.QPushButton("Export peaks CSV")
        self.btn_export_peaks.setProperty("class", "compactSmall")
        self.lbl_signal_msg = QtWidgets.QLabel("")
        self.lbl_signal_msg.setProperty("class", "hint")
        f_signal.addRow("Signal source", self.combo_signal_source)
        f_signal.addRow("Group mode", self.combo_signal_scope)
        f_signal.addRow("File", self.combo_signal_file)
        f_signal.addRow("Method", self.combo_signal_method)
        f_signal.addRow("Min prominence", self.spin_peak_prominence)
        f_signal.addRow("Min height (0=off)", self.spin_peak_height)
        f_signal.addRow("Min distance (s)", self.spin_peak_distance)
        f_signal.addRow("Smooth sigma (s)", self.spin_peak_smooth)
        f_signal.addRow("Baseline handling", self.combo_peak_baseline)
        f_signal.addRow("Baseline window (s)", self.spin_peak_baseline_window)
        f_signal.addRow("Rate bin (s)", self.spin_peak_rate_bin)
        f_signal.addRow("AUC window (+/- s)", self.spin_peak_auc_window)
        f_signal.addRow(self.cb_peak_overlay)
        signal_btn_row = QtWidgets.QHBoxLayout()
        signal_btn_row.addWidget(self.btn_detect_peaks)
        signal_btn_row.addWidget(self.btn_export_peaks)
        signal_btn_row.addStretch(1)
        signal_btn_wrap = QtWidgets.QWidget()
        signal_btn_wrap.setLayout(signal_btn_row)
        f_signal.addRow(signal_btn_wrap)

        self.tbl_signal_metrics = QtWidgets.QTableWidget(0, 2)
        self.tbl_signal_metrics.setHorizontalHeaderLabels(["Metric", "Value"])
        self.tbl_signal_metrics.verticalHeader().setVisible(False)
        self.tbl_signal_metrics.horizontalHeader().setStretchLastSection(True)
        self.tbl_signal_metrics.setEditTriggers(QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers)
        self.tbl_signal_metrics.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.NoSelection)

        grp_behavior_analysis = QtWidgets.QGroupBox("Behavior Analysis")
        f_behavior = QtWidgets.QFormLayout(grp_behavior_analysis)
        f_behavior.setRowWrapPolicy(QtWidgets.QFormLayout.RowWrapPolicy.WrapLongRows)
        f_behavior.setLabelAlignment(QtCore.Qt.AlignmentFlag.AlignLeft | QtCore.Qt.AlignmentFlag.AlignTop)
        self.combo_behavior_analysis = QtWidgets.QComboBox()
        _compact_combo(self.combo_behavior_analysis, min_chars=8)
        self.spin_behavior_bin = QtWidgets.QDoubleSpinBox()
        self.spin_behavior_bin.setRange(0.5, 3600.0)
        self.spin_behavior_bin.setValue(30.0)
        self.spin_behavior_bin.setDecimals(2)
        self.cb_behavior_aligned = QtWidgets.QCheckBox("Use aligned timeline when possible")
        self.cb_behavior_aligned.setChecked(False)
        self.btn_compute_behavior = QtWidgets.QPushButton("Compute behavior metrics")
        self.btn_compute_behavior.setProperty("class", "compactPrimarySmall")
        self.btn_export_behavior_metrics = QtWidgets.QPushButton("Export behavior metrics")
        self.btn_export_behavior_metrics.setProperty("class", "compactSmall")
        self.btn_export_behavior_events = QtWidgets.QPushButton("Export event list")
        self.btn_export_behavior_events.setProperty("class", "compactSmall")
        self.lbl_behavior_msg = QtWidgets.QLabel("")
        self.lbl_behavior_msg.setProperty("class", "hint")
        f_behavior.addRow("Behavior", self.combo_behavior_analysis)
        f_behavior.addRow("Bin size (s)", self.spin_behavior_bin)
        f_behavior.addRow(self.cb_behavior_aligned)
        behavior_btn_row = QtWidgets.QHBoxLayout()
        behavior_btn_row.addWidget(self.btn_compute_behavior)
        behavior_btn_row.addWidget(self.btn_export_behavior_metrics)
        behavior_btn_row.addWidget(self.btn_export_behavior_events)
        behavior_btn_row.addStretch(1)
        behavior_btn_wrap = QtWidgets.QWidget()
        behavior_btn_wrap.setLayout(behavior_btn_row)
        f_behavior.addRow(behavior_btn_wrap)

        self.tbl_behavior_metrics = QtWidgets.QTableWidget(0, 8)
        self.tbl_behavior_metrics.setHorizontalHeaderLabels(
            [
                "file_id",
                "event_count",
                "total_time",
                "mean_duration",
                "median_duration",
                "std_duration",
                "rate_per_min",
                "fraction_time",
            ]
        )
        self.tbl_behavior_metrics.verticalHeader().setVisible(False)
        self.tbl_behavior_metrics.horizontalHeader().setStretchLastSection(True)
        self.tbl_behavior_metrics.setEditTriggers(QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers)
        self.tbl_behavior_metrics.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows)
        self.lbl_behavior_summary = QtWidgets.QLabel("Group metrics: -")
        self.lbl_behavior_summary.setProperty("class", "hint")

        self.section_setup = QtWidgets.QWidget()
        setup_layout = QtWidgets.QVBoxLayout(self.section_setup)
        setup_layout.setContentsMargins(6, 6, 6, 6)
        setup_layout.setSpacing(8)
        setup_layout.addWidget(grp_src)
        setup_layout.addWidget(grp_align)
        setup_btn_row = QtWidgets.QHBoxLayout()
        self.btn_setup_load = QtWidgets.QPushButton("Load")
        self.btn_setup_load.setProperty("class", "compactPrimarySmall")
        self.btn_setup_refresh = QtWidgets.QPushButton("Refresh A/D")
        self.btn_setup_refresh.setProperty("class", "compactSmall")
        setup_btn_row.addWidget(self.btn_setup_load)
        setup_btn_row.addWidget(self.btn_setup_refresh)
        setup_btn_row.addStretch(1)
        setup_wrap = QtWidgets.QWidget()
        setup_wrap.setLayout(setup_btn_row)
        setup_layout.addWidget(setup_wrap)
        setup_layout.addStretch(1)

        self.section_psth = QtWidgets.QWidget()
        psth_layout = QtWidgets.QVBoxLayout(self.section_psth)
        psth_layout.setContentsMargins(6, 6, 6, 6)
        psth_layout.setSpacing(8)
        psth_layout.addWidget(grp_opt)
        psth_btn_row = QtWidgets.QHBoxLayout()
        psth_btn_row.addWidget(self.btn_compute)
        psth_btn_row.addWidget(self.btn_update)
        psth_btn_row.addStretch(1)
        psth_btn_wrap = QtWidgets.QWidget()
        psth_btn_wrap.setLayout(psth_btn_row)
        psth_layout.addWidget(psth_btn_wrap)
        psth_layout.addStretch(1)

        self.section_signal = QtWidgets.QWidget()
        signal_layout = QtWidgets.QVBoxLayout(self.section_signal)
        signal_layout.setContentsMargins(6, 6, 6, 6)
        signal_layout.setSpacing(8)
        signal_layout.addWidget(grp_signal)
        signal_layout.addWidget(self.tbl_signal_metrics)
        signal_layout.addStretch(1)

        self.section_behavior = QtWidgets.QWidget()
        behavior_layout = QtWidgets.QVBoxLayout(self.section_behavior)
        behavior_layout.setContentsMargins(6, 6, 6, 6)
        behavior_layout.setSpacing(8)
        behavior_layout.addWidget(grp_behavior_analysis)
        behavior_layout.addWidget(self.tbl_behavior_metrics)
        behavior_layout.addWidget(self.lbl_behavior_summary)
        behavior_layout.addStretch(1)

        self.section_export = QtWidgets.QWidget()
        export_layout = QtWidgets.QVBoxLayout(self.section_export)
        export_layout.setContentsMargins(6, 6, 6, 6)
        export_layout.setSpacing(8)
        export_layout.addWidget(self.btn_export)
        export_layout.addWidget(self.btn_export_img)
        export_layout.addWidget(self.btn_style)
        export_layout.addWidget(self.btn_save_cfg)
        export_layout.addWidget(self.btn_load_cfg)
        export_layout.addStretch(1)

        action_row = QtWidgets.QHBoxLayout()
        action_row.setContentsMargins(0, 0, 0, 0)
        action_row.setSpacing(6)

        self.btn_action_load = QtWidgets.QPushButton("Load")
        self.btn_action_load.setProperty("class", "compactPrimarySmall")
        self.menu_action_load = QtWidgets.QMenu(self.btn_action_load)
        self.act_load_current = self.menu_action_load.addAction("Use current preprocessed selection")
        self.act_load_single = self.menu_action_load.addAction("Load processed file (single)")
        self.act_load_group = self.menu_action_load.addAction("Load processed files (group)")
        self.menu_action_load.addSeparator()
        self.act_load_behavior = self.menu_action_load.addAction("Load behavior CSV/XLSX")
        self.act_refresh_dio = self.menu_action_load.addAction("Refresh A/D channel list")
        self.btn_action_load.setMenu(self.menu_action_load)

        self.btn_action_compute = QtWidgets.QPushButton("Compute PSTH")
        self.btn_action_compute.setProperty("class", "compactPrimarySmall")
        self.btn_action_export = QtWidgets.QPushButton("Export")
        self.btn_action_export.setProperty("class", "compactPrimarySmall")
        self.btn_action_hide = QtWidgets.QPushButton("Hide Panels")
        self.btn_action_hide.setProperty("class", "compactSmall")

        self.btn_panel_setup = QtWidgets.QPushButton("Setup")
        self.btn_panel_psth = QtWidgets.QPushButton("PSTH")
        self.btn_panel_signal = QtWidgets.QPushButton("Signal")
        self.btn_panel_behavior = QtWidgets.QPushButton("Behavior")
        self.btn_panel_export = QtWidgets.QPushButton("Export")
        self._section_buttons = {
            "setup": self.btn_panel_setup,
            "psth": self.btn_panel_psth,
            "signal": self.btn_panel_signal,
            "behavior": self.btn_panel_behavior,
            "export": self.btn_panel_export,
        }
        for b in self._section_buttons.values():
            b.setCheckable(True)
            b.setProperty("class", "compactSmall")

        action_row.addWidget(self.btn_action_load)
        action_row.addWidget(self.btn_action_compute)
        action_row.addWidget(self.btn_action_export)
        action_row.addWidget(self.btn_action_hide)
        action_row.addSpacing(8)
        action_row.addWidget(QtWidgets.QLabel("Panels:"))
        action_row.addWidget(self.btn_panel_setup)
        action_row.addWidget(self.btn_panel_psth)
        action_row.addWidget(self.btn_panel_signal)
        action_row.addWidget(self.btn_panel_behavior)
        action_row.addWidget(self.btn_panel_export)
        action_row.addStretch(1)
        root.addLayout(action_row)

        # Right plots: trace preview + heatmap + avg
        right = QtWidgets.QWidget()
        self._right_panel = right
        right.setSizePolicy(QtWidgets.QSizePolicy.Policy.Expanding, QtWidgets.QSizePolicy.Policy.Expanding)
        rv = QtWidgets.QVBoxLayout(right)
        rv.setSpacing(10)

        rv.setContentsMargins(0, 0, 0, 0)
        rv.setSpacing(8)

        header_row = QtWidgets.QHBoxLayout()
        self.lbl_plot_file = QtWidgets.QLabel("File: (none)")
        header_font = self.lbl_plot_file.font()
        header_font.setBold(True)
        self.lbl_plot_file.setFont(header_font)
        header_row.addWidget(self.lbl_plot_file)
        header_row.addStretch(1)
        rv.addLayout(header_row)

        view_row = QtWidgets.QHBoxLayout()
        view_row.addWidget(QtWidgets.QLabel("View layout"))
        self.combo_view_layout = QtWidgets.QComboBox()
        self.combo_view_layout.addItems(["Standard", "Heatmap focus", "Trace focus", "Metrics focus", "All"])
        _compact_combo(self.combo_view_layout, min_chars=9)
        view_row.addWidget(self.combo_view_layout)
        view_row.addStretch(1)
        rv.addLayout(view_row)

        self.plot_trace = pg.PlotWidget(title="Trace preview")
        self.plot_heat = pg.PlotWidget(title="Heatmap")
        self.plot_dur = pg.PlotWidget(title="Event duration")
        self.plot_avg = pg.PlotWidget(title="Average PSTH +/- SEM")
        self.plot_metrics = pg.PlotWidget(title="PSTH metrics")
        self.plot_global = pg.PlotWidget(title="Global metrics")
        self.plot_peak_amp = pg.PlotWidget(title="Peak amplitudes")
        self.plot_peak_ibi = pg.PlotWidget(title="Inter-peak intervals")
        self.plot_peak_rate = pg.PlotWidget(title="Peak rate over time")
        self.plot_behavior_raster = pg.PlotWidget(title="Behavior raster")
        self.plot_behavior_rate = pg.PlotWidget(title="Behavior frequency over time")
        self.plot_behavior_duration = pg.PlotWidget(title="Behavior duration distribution")
        self.plot_behavior_starts = pg.PlotWidget(title="Behavior start times")

        for w in (
            self.plot_trace,
            self.plot_heat,
            self.plot_dur,
            self.plot_avg,
            self.plot_metrics,
            self.plot_global,
            self.plot_peak_amp,
            self.plot_peak_ibi,
            self.plot_peak_rate,
            self.plot_behavior_raster,
            self.plot_behavior_rate,
            self.plot_behavior_duration,
            self.plot_behavior_starts,
        ):
            _opt_plot(w)

        self.curve_trace = self.plot_trace.plot(pen=pg.mkPen(self._style["trace"], width=1.1))
        self.curve_behavior = self.plot_trace.plot(pen=pg.mkPen(self._style["behavior"], width=1.0))
        self.curve_peak_markers = self.plot_trace.plot(
            pen=None,
            symbol="o",
            symbolSize=6,
            symbolBrush=pg.mkBrush(240, 120, 80),
            symbolPen=pg.mkPen((240, 120, 80), width=1.0),
        )
        self.event_lines: List[pg.InfiniteLine] = []

        self.img = pg.ImageItem()
        self.plot_heat.addItem(self.img)
        self.plot_heat.setLabel("bottom", "Time (s)")
        self.plot_heat.setLabel("left", "Trials / Recordings")
        self.plot_dur.setLabel("bottom", "Duration (s)")
        self.plot_dur.setLabel("left", "Count")
        self.plot_peak_amp.setLabel("bottom", "Amplitude")
        self.plot_peak_amp.setLabel("left", "Count")
        self.plot_peak_ibi.setLabel("bottom", "Interval (s)")
        self.plot_peak_ibi.setLabel("left", "Count")
        self.plot_peak_rate.setLabel("bottom", "Time (s)")
        self.plot_peak_rate.setLabel("left", "Peaks/min")
        self.plot_behavior_raster.setLabel("bottom", "Time (s)")
        self.plot_behavior_raster.setLabel("left", "File index")
        self.plot_behavior_rate.setLabel("bottom", "Time (s)")
        self.plot_behavior_rate.setLabel("left", "Events/min")
        self.plot_behavior_duration.setLabel("bottom", "Duration (s)")
        self.plot_behavior_duration.setLabel("left", "Count")
        self.plot_behavior_starts.setLabel("bottom", "Start time (s)")
        self.plot_behavior_starts.setLabel("left", "Count")

        self.curve_avg = self.plot_avg.plot(pen=pg.mkPen(self._style["avg"], width=1.3))
        self.curve_sem_hi = self.plot_avg.plot(pen=pg.mkPen((220, 220, 220), width=1.0))
        self.curve_sem_lo = self.plot_avg.plot(pen=pg.mkPen((220, 220, 220), width=1.0))
        self.plot_avg.addLine(x=0, pen=pg.mkPen((200, 200, 200), style=QtCore.Qt.PenStyle.DashLine))
        self.metrics_bar_pre = pg.BarGraphItem(x=[0], height=[0], width=0.6, brush=(90, 143, 214))
        self.metrics_bar_post = pg.BarGraphItem(x=[1], height=[0], width=0.6, brush=(214, 122, 90))
        self.plot_metrics.addItem(self.metrics_bar_pre)
        self.plot_metrics.addItem(self.metrics_bar_post)
        self.plot_metrics.setXRange(-0.5, 1.5, padding=0)
        self.plot_metrics.getAxis("bottom").setTicks([[(0, "pre"), (1, "post")]])

        self.global_bar_amp = pg.BarGraphItem(x=[0], height=[0], width=0.6, brush=(120, 180, 220))
        self.global_bar_freq = pg.BarGraphItem(x=[1], height=[0], width=0.6, brush=(220, 160, 120))
        self.plot_global.addItem(self.global_bar_amp)
        self.plot_global.addItem(self.global_bar_freq)
        self.plot_global.setXRange(-0.5, 1.5, padding=0)
        self.plot_global.getAxis("bottom").setTicks([[(0, "amp"), (1, "freq")]])

        self.row_heat = QtWidgets.QWidget()
        heat_row = QtWidgets.QHBoxLayout(self.row_heat)
        heat_row.setContentsMargins(0, 0, 0, 0)
        heat_row.setSpacing(8)
        heat_row.addWidget(self.plot_heat, stretch=4)
        heat_row.addWidget(self.plot_dur, stretch=1)

        self.row_avg = QtWidgets.QWidget()
        avg_row = QtWidgets.QHBoxLayout(self.row_avg)
        avg_row.setContentsMargins(0, 0, 0, 0)
        avg_row.setSpacing(8)
        avg_row.addWidget(self.plot_avg, stretch=4)
        avg_row.addWidget(self.plot_metrics, stretch=1)
        avg_row.addWidget(self.plot_global, stretch=1)

        self.row_signal = QtWidgets.QWidget()
        signal_row = QtWidgets.QHBoxLayout(self.row_signal)
        signal_row.setContentsMargins(0, 0, 0, 0)
        signal_row.setSpacing(8)
        signal_row.addWidget(self.plot_peak_amp, stretch=1)
        signal_row.addWidget(self.plot_peak_ibi, stretch=1)
        signal_row.addWidget(self.plot_peak_rate, stretch=1)

        self.row_behavior = QtWidgets.QWidget()
        behavior_grid = QtWidgets.QGridLayout(self.row_behavior)
        behavior_grid.setContentsMargins(0, 0, 0, 0)
        behavior_grid.setHorizontalSpacing(8)
        behavior_grid.setVerticalSpacing(8)
        behavior_grid.addWidget(self.plot_behavior_raster, 0, 0, 1, 3)
        behavior_grid.addWidget(self.plot_behavior_rate, 1, 0)
        behavior_grid.addWidget(self.plot_behavior_duration, 1, 1)
        behavior_grid.addWidget(self.plot_behavior_starts, 1, 2)
        behavior_grid.setColumnStretch(0, 1)
        behavior_grid.setColumnStretch(1, 1)
        behavior_grid.setColumnStretch(2, 1)

        # Keep a visible minimum plot footprint even with aggressive docking/resizing.
        self.plot_trace.setMinimumHeight(140)
        self.row_heat.setMinimumHeight(180)
        self.row_avg.setMinimumHeight(140)

        rv.addWidget(self.plot_trace, stretch=1)
        rv.addWidget(self.row_heat, stretch=2)
        rv.addWidget(self.row_avg, stretch=1)
        rv.addWidget(self.row_signal, stretch=1)
        rv.addWidget(self.row_behavior, stretch=1)
        root.addWidget(right, stretch=1)
        root.setStretch(0, 0)
        root.setStretch(1, 1)
        self._setup_section_popups()

        # Wiring
        self.act_load_current.triggered.connect(self.requestCurrentProcessed.emit)
        self.act_load_single.triggered.connect(self._load_processed_files_single)
        self.act_load_group.triggered.connect(self._load_processed_files)
        self.act_load_behavior.triggered.connect(self._load_behavior_files)
        self.act_refresh_dio.triggered.connect(self.requestDioList.emit)
        self.btn_action_compute.clicked.connect(self._compute_psth)
        self.btn_action_export.clicked.connect(self._export_results)
        self.btn_action_hide.clicked.connect(self._hide_all_section_popups)
        for key, btn in self._section_buttons.items():
            btn.toggled.connect(lambda checked, section_key=key: self._toggle_section_popup(section_key, checked))

        self.btn_use_current.clicked.connect(self.requestCurrentProcessed.emit)
        self.btn_refresh_dio.clicked.connect(self.requestDioList.emit)
        self.btn_setup_refresh.clicked.connect(self.requestDioList.emit)
        self.btn_setup_load.clicked.connect(self._load_processed_files)
        self.btn_load_beh.clicked.connect(self._load_behavior_files)
        self.btn_load_processed.clicked.connect(self._load_processed_files)
        self.btn_load_processed_single.clicked.connect(self._load_processed_files_single)
        self.list_preprocessed.filesDropped.connect(self._on_preprocessed_files_dropped)
        self.list_preprocessed.orderChanged.connect(self._sync_processed_order_from_list)
        self.list_behaviors.filesDropped.connect(self._on_behavior_files_dropped)
        self.list_behaviors.orderChanged.connect(self._sync_behavior_order_from_list)
        self.btn_compute.clicked.connect(self._compute_psth)
        self.btn_update.clicked.connect(self._compute_psth)
        self.btn_detect_peaks.clicked.connect(self._detect_signal_events)
        self.btn_export_peaks.clicked.connect(self._export_signal_events_csv)
        self.btn_compute_behavior.clicked.connect(self._compute_behavior_analysis)
        self.btn_export_behavior_metrics.clicked.connect(self._export_behavior_metrics_csv)
        self.btn_export_behavior_events.clicked.connect(self._export_behavior_events_csv)
        self.btn_export.clicked.connect(self._export_results)
        self.btn_export_img.clicked.connect(self._export_images)
        self.btn_style.clicked.connect(self._open_style_dialog)
        self.btn_save_cfg.clicked.connect(self._save_config_file)
        self.btn_load_cfg.clicked.connect(self._load_config_file)
        self.cb_filter_events.stateChanged.connect(self._update_event_filter_enabled)
        self.cb_metrics.stateChanged.connect(self._update_metrics_enabled)
        self.cb_global_metrics.stateChanged.connect(self._update_global_metrics_enabled)
        self.btn_hide_filters.toggled.connect(self._toggle_filter_panel)
        self.btn_hide_metrics.toggled.connect(self._toggle_metrics_panel)
        self.combo_view_layout.currentIndexChanged.connect(self._apply_view_layout)
        self.combo_view_layout.currentIndexChanged.connect(lambda *_: self._save_settings())
        self.cb_peak_overlay.toggled.connect(self._refresh_signal_overlay)
        self.combo_signal_source.currentIndexChanged.connect(self._refresh_signal_file_combo)
        self.combo_signal_scope.currentIndexChanged.connect(self._refresh_signal_file_combo)
        self.tab_sources.currentChanged.connect(self._refresh_signal_file_combo)

        self.combo_align.currentIndexChanged.connect(self._update_align_ui)
        self.combo_behavior_file_type.currentIndexChanged.connect(self._update_align_ui)
        self.combo_behavior_align.currentIndexChanged.connect(self._update_align_ui)
        self.combo_align.currentIndexChanged.connect(self._refresh_behavior_list)
        self.combo_align.currentIndexChanged.connect(self._compute_psth)
        for w in (
            self.combo_dio,
            self.combo_dio_polarity,
            self.combo_dio_align,
            self.combo_behavior_name,
            self.combo_behavior_align,
            self.combo_behavior_from,
            self.combo_behavior_to,
        ):
            w.currentIndexChanged.connect(self._compute_psth)
        self.spin_transition_gap.valueChanged.connect(self._compute_psth)
        for w in (
            self.spin_event_start,
            self.spin_event_end,
            self.spin_group_window,
            self.spin_dur_min,
            self.spin_dur_max,
            self.spin_metric_pre0,
            self.spin_metric_pre1,
            self.spin_metric_post0,
            self.spin_metric_post1,
            self.spin_global_start,
            self.spin_global_end,
        ):
            w.valueChanged.connect(self._compute_psth)
        self.combo_metric.currentIndexChanged.connect(self._compute_psth)
        self.cb_global_amp.stateChanged.connect(self._compute_psth)
        self.cb_global_freq.stateChanged.connect(self._compute_psth)
        for w in (self.spin_metric_pre0, self.spin_metric_pre1, self.spin_metric_post0, self.spin_metric_post1):
            w.valueChanged.connect(self._update_metric_regions)

        self._update_align_ui()
        self._update_event_filter_enabled()
        self._update_metrics_enabled()
        self._update_global_metrics_enabled()
        self._toggle_filter_panel(False)
        self._toggle_metrics_panel(False)
        self._apply_view_layout()
        self._refresh_signal_file_combo()
        self._update_data_availability()
        self._update_status_strip()

        QtGui.QShortcut(QtGui.QKeySequence("Ctrl+S"), self, activated=self._export_results)
        QtGui.QShortcut(QtGui.QKeySequence("F5"), self, activated=self._compute_psth)

    def _dock_main_window(self) -> Optional[QtWidgets.QMainWindow]:
        host = self.window()
        return host if isinstance(host, QtWidgets.QMainWindow) else None

    def _setup_section_popups(self) -> None:
        if self._section_popups:
            return
        host = self._dock_main_window()
        if host is None:
            # The tab can be created before it is attached to the main window.
            # Retry from showEvent once host is available.
            return
        self._dock_host = host
        section_map: Dict[str, Tuple[str, QtWidgets.QWidget]] = {
            "setup": ("Setup", self.section_setup),
            "psth": ("PSTH", self.section_psth),
            "signal": ("Signal Event Analyzer", self.section_signal),
            "behavior": ("Behavior Analysis", self.section_behavior),
            "export": ("Export", self.section_export),
        }
        for key, (title, widget) in section_map.items():
            scroll = QtWidgets.QScrollArea()
            scroll.setWidgetResizable(True)
            scroll.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
            scroll.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
            scroll.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAsNeeded)
            # Keep viewport painted with dock background to avoid dark paint gaps
            # when section rows are dynamically shown/hidden.
            scroll.viewport().setAutoFillBackground(True)
            scroll.viewport().setStyleSheet("background: #242a34;")
            # Keep section widgets shrinkable so dock stacks do not clip content.
            widget.setMinimumSize(0, 0)
            widget.setSizePolicy(QtWidgets.QSizePolicy.Policy.Preferred, QtWidgets.QSizePolicy.Policy.Preferred)
            scroll.setWidget(widget)
            self._section_scroll_hosts[key] = scroll

            dock = QtWidgets.QDockWidget(title, host)
            dock.setObjectName(f"post.{key}.dock")
            dock.setAllowedAreas(QtCore.Qt.DockWidgetArea.AllDockWidgetAreas)
            dock.setMinimumWidth(320)
            dock.setFeatures(
                QtWidgets.QDockWidget.DockWidgetFeature.DockWidgetClosable
                | QtWidgets.QDockWidget.DockWidgetFeature.DockWidgetMovable
                | QtWidgets.QDockWidget.DockWidgetFeature.DockWidgetFloatable
            )
            dock.setWidget(scroll)
            dock.visibilityChanged.connect(
                lambda visible, section_key=key: self._on_section_popup_visibility(section_key, visible)
            )
            dock.topLevelChanged.connect(lambda *_: self._save_panel_layout_state())
            dock.dockLocationChanged.connect(lambda *_: self._save_panel_layout_state())
            host.addDockWidget(QtCore.Qt.DockWidgetArea.RightDockWidgetArea, dock)
            # Match preprocessing behavior: popups open floating by default.
            dock.setFloating(True)
            dock.hide()
            self._section_popups[key] = dock

        # If popups become available after delayed host attachment, restore once here.
        if self._panel_layout_persistence_ready and not self._dock_layout_restored:
            self._restore_panel_layout_state()
            self._dock_layout_restored = True
        # Main window can use this callback to retry pending dock restores.
        try:
            if hasattr(host, "on_post_docks_ready"):
                host.on_post_docks_ready()
            elif hasattr(host, "onPostDocksReady"):
                host.onPostDocksReady()
        except Exception:
            pass

    def _set_section_button_checked(self, key: str, checked: bool) -> None:
        btn = self._section_buttons.get(key)
        if btn is None:
            return
        btn.blockSignals(True)
        btn.setChecked(bool(checked))
        btn.blockSignals(False)

    def _toggle_section_popup(self, key: str, checked: bool) -> None:
        if not self._section_popups:
            self._setup_section_popups()
        dock = self._section_popups.get(key)
        if dock is None:
            return
        if checked:
            # Keep user-selected docking mode. Only reposition if currently floating off-screen.
            if dock.isFloating():
                if key not in self._section_popup_initialized or not self._is_popup_on_screen(dock):
                    self._position_section_popup(dock, key)
                    self._section_popup_initialized.add(key)
            dock.show()
            dock.raise_()
            dock.activateWindow()
            self._last_opened_section = key
        else:
            dock.hide()
        self._save_panel_layout_state()

    def _on_section_popup_visibility(self, key: str, visible: bool) -> None:
        self._set_section_button_checked(key, visible)
        if visible:
            self._last_opened_section = key
        elif self._last_opened_section == key:
            self._last_opened_section = None
        self._save_panel_layout_state()

    def _hide_all_section_popups(self) -> None:
        for key, dock in self._section_popups.items():
            dock.hide()
            self._set_section_button_checked(key, False)
        self._last_opened_section = None
        self._save_panel_layout_state()

    def _default_popup_size(self, key: str) -> Tuple[int, int]:
        size_map = {
            "setup": (420, 620),
            "psth": (420, 640),
            "signal": (420, 640),
            "behavior": (500, 620),
            "export": (340, 300),
        }
        return size_map.get(key, (420, 620))

    def _active_screen_geometry(self) -> QtCore.QRect:
        handle = self.windowHandle()
        screen = handle.screen() if handle else None
        if screen is None:
            screen = QtGui.QGuiApplication.screenAt(self.mapToGlobal(self.rect().center()))
        if screen is None:
            screen = QtGui.QGuiApplication.primaryScreen()
        if screen is None:
            return QtCore.QRect(0, 0, 1920, 1080)
        return screen.availableGeometry()

    def _position_section_popup(self, dock: QtWidgets.QDockWidget, key: str) -> None:
        panel_global = self.mapToGlobal(self.rect().topLeft())
        panel_rect = QtCore.QRect(panel_global, self.size())
        screen_rect = self._active_screen_geometry()
        pref_w, pref_h = self._default_popup_size(key)
        w = min(pref_w, max(320, screen_rect.width() - 40))
        h = min(pref_h, max(240, screen_rect.height() - 40))

        x_right = panel_rect.x() + panel_rect.width() + 12
        x_left = panel_rect.x() - w - 12
        y_pref = panel_rect.y() + 60

        x_min = screen_rect.x() + 8
        x_max = screen_rect.x() + max(8, screen_rect.width() - w - 8)
        y_min = screen_rect.y() + 8
        y_max = screen_rect.y() + max(8, screen_rect.height() - h - 8)

        if x_right <= x_max:
            x = x_right
        elif x_left >= x_min:
            x = x_left
        else:
            x = x_max
        y = min(max(y_pref, y_min), y_max)

        dock.resize(int(w), int(h))
        dock.move(int(x), int(y))

    def _is_popup_on_screen(self, dock: QtWidgets.QDockWidget) -> bool:
        rect = dock.frameGeometry()
        if rect.width() <= 0 or rect.height() <= 0:
            return False
        for screen in QtGui.QGuiApplication.screens():
            if screen.availableGeometry().intersects(rect):
                return True
        return False

    def _dock_area_from_settings(
        self,
        value: object,
        default: QtCore.Qt.DockWidgetArea = QtCore.Qt.DockWidgetArea.RightDockWidgetArea,
    ) -> QtCore.Qt.DockWidgetArea:
        left_i = _dock_area_to_int(QtCore.Qt.DockWidgetArea.LeftDockWidgetArea, 1)
        right_i = _dock_area_to_int(QtCore.Qt.DockWidgetArea.RightDockWidgetArea, 2)
        top_i = _dock_area_to_int(QtCore.Qt.DockWidgetArea.TopDockWidgetArea, 4)
        bottom_i = _dock_area_to_int(QtCore.Qt.DockWidgetArea.BottomDockWidgetArea, 8)
        area_int = _dock_area_to_int(value, _dock_area_to_int(default, right_i))
        area_map: Dict[int, QtCore.Qt.DockWidgetArea] = {
            left_i: QtCore.Qt.DockWidgetArea.LeftDockWidgetArea,
            right_i: QtCore.Qt.DockWidgetArea.RightDockWidgetArea,
            top_i: QtCore.Qt.DockWidgetArea.TopDockWidgetArea,
            bottom_i: QtCore.Qt.DockWidgetArea.BottomDockWidgetArea,
        }
        if area_int in area_map:
            return area_map[area_int]
        return default

    def _to_qbytearray(self, value: object) -> Optional[QtCore.QByteArray]:
        if isinstance(value, QtCore.QByteArray):
            return value
        if isinstance(value, (bytes, bytearray)):
            return QtCore.QByteArray(bytes(value))
        if isinstance(value, str):
            try:
                return QtCore.QByteArray.fromBase64(value.encode("utf-8"))
            except Exception:
                return None
        return None

    def _sync_section_button_states_from_docks(self) -> None:
        self._last_opened_section = None
        for key, dock in self._section_popups.items():
            visible = bool(dock.isVisible())
            self._set_section_button_checked(key, visible)
            if visible:
                self._last_opened_section = key

    def _has_saved_layout_state(self) -> bool:
        try:
            if self._settings.contains(_POST_DOCK_STATE_KEY):
                return True
            keys = list(self._section_popups.keys()) or ["setup", "psth", "signal", "behavior", "export"]
            for key in keys:
                if self._settings.contains(f"post_section_docks/{key}/visible"):
                    return True
        except Exception:
            pass
        return False

    def _save_panel_layout_state(self) -> None:
        if not self._panel_layout_persistence_ready:
            return
        if self._is_restoring_panel_layout:
            return
        if self._suspend_panel_layout_persistence:
            return
        # Do not overwrite stored layout while post panels are hidden for tab switching.
        if self._post_docks_hidden_for_tab_switch:
            return
        host = self._dock_host or self._dock_main_window()
        if host is None:
            return
        self._dock_host = host
        if not self._section_popups:
            return
        for key, dock in self._section_popups.items():
            try:
                base = f"post_section_docks/{key}"
                # Preserve pre-hide state while switching tabs so settings are not overwritten
                # with temporary hidden values.
                cached = self._post_section_state_before_hide.get(key, {}) if self._post_docks_hidden_for_tab_switch else {}
                visible = bool(cached.get("visible", dock.isVisible()))
                floating = bool(cached.get("floating", dock.isFloating()))
                right_i = _dock_area_to_int(QtCore.Qt.DockWidgetArea.RightDockWidgetArea, 2)
                area_val = _dock_area_to_int(cached.get("area", host.dockWidgetArea(dock)), right_i)
                geom = cached.get("geometry", dock.saveGeometry())
                self._settings.setValue(f"{base}/visible", visible)
                self._settings.setValue(f"{base}/floating", floating)
                self._settings.setValue(f"{base}/area", area_val)
                self._settings.setValue(f"{base}/geometry", geom)
            except Exception:
                continue
        try:
            self._settings.sync()
        except Exception:
            pass

    def _persist_hidden_layout_state_from_cache(self) -> None:
        """Persist cached post layout captured when tab-switch hiding post docks."""
        if not self._post_docks_hidden_for_tab_switch:
            return
        right_i = _dock_area_to_int(QtCore.Qt.DockWidgetArea.RightDockWidgetArea, 2)
        if not self._section_popups:
            return
        for key in self._section_popups.keys():
            try:
                state = self._post_section_state_before_hide.get(key, {})
                base = f"post_section_docks/{key}"
                self._settings.setValue(f"{base}/visible", bool(state.get("visible", False)))
                self._settings.setValue(f"{base}/floating", bool(state.get("floating", True)))
                self._settings.setValue(f"{base}/area", _dock_area_to_int(state.get("area", right_i), right_i))
                geom = state.get("geometry")
                if isinstance(geom, QtCore.QByteArray) and not geom.isEmpty():
                    self._settings.setValue(f"{base}/geometry", geom)
            except Exception:
                continue
        try:
            self._settings.sync()
        except Exception:
            pass

    def flush_post_section_state_to_settings(self) -> None:
        """Flush latest post section visibility/layout into QSettings immediately."""
        if self._post_docks_hidden_for_tab_switch:
            self._persist_hidden_layout_state_from_cache()
            return
        self._save_panel_layout_state()

    def persist_layout_state_snapshot(self) -> None:
        """
        Persist post dock state safely.
        Uses cached tab-switch state while hidden, otherwise captures current host topology.
        """
        if self._post_docks_hidden_for_tab_switch:
            self._persist_hidden_layout_state_from_cache()
            return

        host = self._dock_host or self._dock_main_window()
        if host is not None:
            self._dock_host = host
            try:
                state = None
                if hasattr(host, "captureDockSnapshotForTab"):
                    state = host.captureDockSnapshotForTab("post")
                elif hasattr(host, "saveState"):
                    state = host.saveState(_DOCK_STATE_VERSION)
                if state is not None and not state.isEmpty():
                    self._settings.setValue(_POST_DOCK_STATE_KEY, state)
            except Exception:
                pass
        self.flush_post_section_state_to_settings()

    def _restore_panel_layout_state(self) -> None:
        if self._force_fixed_default_layout:
            self.apply_fixed_default_layout()
            return
        self._is_restoring_panel_layout = True
        try:
            if not self._section_popups:
                self._setup_section_popups()
            host = self._dock_host or self._dock_main_window()
            if host is None or not self._section_popups:
                return
            self._dock_host = host
            has_saved_layout = any(
                bool(self._settings.contains(f"post_section_docks/{key}/visible"))
                for key in self._section_popups.keys()
            )
            for key, dock in self._section_popups.items():
                base = f"post_section_docks/{key}"
                default_visible = (key == "setup") if not has_saved_layout else False
                visible = _to_bool(self._settings.value(f"{base}/visible", default_visible), bool(default_visible))
                floating = _to_bool(self._settings.value(f"{base}/floating", True), True)
                area_val = self._settings.value(
                    f"{base}/area",
                    _dock_area_to_int(QtCore.Qt.DockWidgetArea.RightDockWidgetArea, 2),
                )
                area = self._dock_area_from_settings(area_val, QtCore.Qt.DockWidgetArea.RightDockWidgetArea)
                geom = self._to_qbytearray(self._settings.value(f"{base}/geometry", None))

                dock.blockSignals(True)
                try:
                    if floating:
                        dock.setFloating(True)
                    else:
                        host.addDockWidget(area, dock)
                        dock.setFloating(False)

                    # Apply geometry for both floating and docked states.
                    if geom is not None and not geom.isEmpty():
                        dock.restoreGeometry(geom)
                        self._section_popup_initialized.add(key)

                    if visible:
                        if dock.isFloating() and not self._is_popup_on_screen(dock):
                            self._position_section_popup(dock, key)
                        dock.show()
                    else:
                        dock.hide()
                finally:
                    dock.blockSignals(False)
        except Exception:
            pass
        finally:
            self._sync_section_button_states_from_docks()
            self._is_restoring_panel_layout = False

    # ---- bridge reception ----

    def set_current_source_label(self, filename: str, channel: str) -> None:
        self.lbl_current.setText(f"Current: {filename} [{channel}]")
        if hasattr(self, "lbl_plot_file"):
            self.lbl_plot_file.setText(f"File: {filename}")
        self._update_status_strip()

    def notify_preprocessing_updated(self, _processed: ProcessedTrial) -> None:
        # no-op; user presses compute or update
        pass

    def _set_resample_from_processed(self) -> None:
        if not self._processed:
            return
        proc = self._processed[0]
        fs = float(proc.fs_used) if np.isfinite(proc.fs_used) else float(proc.fs_actual)
        if not np.isfinite(fs) or fs <= 0:
            t = np.asarray(proc.time, float) if proc.time is not None else np.array([], float)
            if t.size > 2:
                dt = np.nanmedian(np.diff(t))
                fs = 1.0 / float(dt) if np.isfinite(dt) and dt > 0 else np.nan
        if np.isfinite(fs) and fs > 0:
            fs = max(1.0, min(1000.0, float(fs)))
            self.spin_resample.setValue(fs)
        self._update_status_strip()

    @QtCore.Slot(list)
    def receive_current_processed(self, processed_list: List[ProcessedTrial]) -> None:
        self._processed = processed_list or []
        # update trace preview with first entry
        self._refresh_behavior_list()
        self._set_resample_from_processed()
        self._update_trace_preview()
        self._update_data_availability()
        self._update_status_strip()

    def append_processed(self, processed_list: List[ProcessedTrial]) -> None:
        if not processed_list:
            return
        self._processed.extend(processed_list)
        self._refresh_behavior_list()
        self._set_resample_from_processed()
        self._update_trace_preview()
        self._update_data_availability()
        self._update_status_strip()

    @QtCore.Slot(list)
    def receive_dio_list(self, dio_list: List[str]) -> None:
        self.combo_dio.clear()
        for d in dio_list or []:
            self.combo_dio.addItem(d)
        self._update_status_strip()

    @QtCore.Slot(str, str, object, object)
    def receive_dio_data(self, path: str, dio_name: str, t: Optional[np.ndarray], x: Optional[np.ndarray]) -> None:
        if t is None or x is None:
            return
        self._dio_cache[(path, dio_name)] = (np.asarray(t, float), np.asarray(x, float))

    def _load_behavior_paths(self, paths: List[str], replace: bool) -> None:
        if replace:
            self._behavior_sources.clear()
        parse_mode = self._current_behavior_parse_mode()
        for p in paths:
            stem = os.path.splitext(os.path.basename(p))[0]
            ext = os.path.splitext(p)[1].lower()
            try:
                if ext == ".csv":
                    info = _load_behavior_csv(p, parse_mode=parse_mode)
                elif ext == ".xlsx":
                    import pandas as pd
                    xls = pd.ExcelFile(p, engine="openpyxl")
                    sheet = None
                    if len(xls.sheet_names) > 1:
                        sheet, ok = QtWidgets.QInputDialog.getItem(
                            self,
                            "Select sheet",
                            f"{os.path.basename(p)}: choose sheet",
                            xls.sheet_names,
                            0,
                            False,
                        )
                        if not ok:
                            continue
                    info = _load_behavior_ethovision(p, sheet_name=sheet, parse_mode=parse_mode)
                else:
                    continue
                if not (info.get("behaviors") or {}):
                    QtWidgets.QMessageBox.warning(
                        self,
                        "Behavior load warning",
                        f"No behavior columns detected in {os.path.basename(p)} for the selected file type.",
                    )
                self._behavior_sources[stem] = info
            except Exception as exc:
                QtWidgets.QMessageBox.warning(
                    self,
                    "Behavior load failed",
                    f"Could not load {os.path.basename(p)}:\n{exc}",
                )
                continue
        mode_label = "timestamps" if parse_mode == _BEHAVIOR_PARSE_TIMESTAMPS else "binary"
        self.lbl_beh.setText(f"{len(self._behavior_sources)} file(s) loaded [{mode_label}]")
        self._update_data_availability()
        self._update_status_strip()

    def _load_processed_paths(self, paths: List[str], replace: bool) -> None:
        loaded: List[ProcessedTrial] = []
        for p in paths:
            ext = os.path.splitext(p)[1].lower()
            if ext == ".csv":
                trial = self._load_processed_csv(p)
            elif ext in (".h5", ".hdf5"):
                trial = self._load_processed_h5(p)
            else:
                trial = None
            if trial is not None:
                loaded.append(trial)
        if not loaded:
            return
        if replace:
            self._processed = loaded
        else:
            self._processed.extend(loaded)
        self.lbl_group.setText(f"{len(self._processed)} file(s) loaded")
        self._update_file_lists()
        self._set_resample_from_processed()
        self._compute_psth()
        self._update_data_availability()
        self._update_status_strip()

    def _on_preprocessed_files_dropped(self, paths: List[str]) -> None:
        allowed = {".csv", ".h5", ".hdf5"}
        keep = [p for p in paths if os.path.splitext(p)[1].lower() in allowed]
        if not keep:
            return
        self._load_processed_paths(keep, replace=False)

    def _on_behavior_files_dropped(self, paths: List[str]) -> None:
        allowed = {".csv", ".xlsx"}
        keep = [p for p in paths if os.path.splitext(p)[1].lower() in allowed]
        if not keep:
            return
        self._load_behavior_paths(keep, replace=False)
        self._refresh_behavior_list()

    # ---- behavior files ----

    def _update_align_ui(self) -> None:
        use_dio = _is_doric_channel_align(self.combo_align.currentText())
        for w in (self.combo_dio, self.combo_dio_polarity, self.combo_dio_align):
            w.setEnabled(use_dio)
            w.setVisible(use_dio)

        use_beh = not use_dio
        self.lbl_behavior_file_type.setEnabled(use_beh)
        self.lbl_behavior_file_type.setVisible(use_beh)
        self.combo_behavior_file_type.setEnabled(use_beh)
        self.combo_behavior_file_type.setVisible(use_beh)
        self.btn_load_beh.setEnabled(use_beh)
        self.btn_load_beh.setVisible(use_beh)
        self.combo_behavior_name.setEnabled(use_beh)
        self.combo_behavior_name.setVisible(use_beh)
        self.combo_behavior_align.setEnabled(use_beh)
        self.combo_behavior_align.setVisible(use_beh)

        is_transition = self.combo_behavior_align.currentText().startswith("Transition") and use_beh
        for w in (
            self.combo_behavior_from,
            self.combo_behavior_to,
            self.spin_transition_gap,
            self.lbl_trans_from,
            self.lbl_trans_to,
            self.lbl_trans_gap,
        ):
            w.setVisible(is_transition)

        if use_beh:
            self.combo_behavior_from.setEnabled(is_transition)
            self.combo_behavior_to.setEnabled(is_transition)
            self.spin_transition_gap.setEnabled(is_transition)
        self._update_trace_preview()
        self._update_status_strip()

    def _current_behavior_parse_mode(self) -> str:
        text = self.combo_behavior_file_type.currentText().strip().lower() if hasattr(self, "combo_behavior_file_type") else ""
        if "timestamp" in text:
            return _BEHAVIOR_PARSE_TIMESTAMPS
        return _BEHAVIOR_PARSE_BINARY

    def _apply_view_layout(self) -> None:
        layout = self.combo_view_layout.currentText() if hasattr(self, "combo_view_layout") else "Standard"
        show_trace = True
        show_heat = True
        show_avg = True
        show_signal = False
        show_behavior = False

        self.plot_dur.setVisible(True)
        self.plot_metrics.setVisible(True)
        self.plot_global.setVisible(self.cb_global_metrics.isChecked())

        if layout == "Heatmap focus":
            show_signal = False
            show_behavior = False
            self.plot_dur.setVisible(False)
            self.plot_metrics.setVisible(False)
            self.plot_global.setVisible(False)
        elif layout == "Trace focus":
            show_heat = False
            show_signal = False
            show_behavior = False
            self.plot_metrics.setVisible(False)
            self.plot_global.setVisible(False)
        elif layout == "Metrics focus":
            show_heat = False
            show_signal = False
            show_behavior = False
            self.plot_metrics.setVisible(True)
            self.plot_global.setVisible(True)
        elif layout == "All":
            show_signal = True
            show_behavior = True

        self.plot_trace.setVisible(show_trace)
        self.row_heat.setVisible(show_heat)
        self.row_avg.setVisible(show_avg)
        self.row_signal.setVisible(show_signal)
        self.row_behavior.setVisible(show_behavior)

    def _refresh_signal_file_combo(self) -> None:
        if not hasattr(self, "combo_signal_file"):
            return
        prev = self.combo_signal_file.currentText().strip()
        self.combo_signal_file.blockSignals(True)
        self.combo_signal_file.clear()
        for proc in self._processed:
            stem = os.path.splitext(os.path.basename(proc.path))[0] if proc.path else "import"
            self.combo_signal_file.addItem(stem)
        if prev:
            idx = self.combo_signal_file.findText(prev)
            if idx >= 0:
                self.combo_signal_file.setCurrentIndex(idx)
        self.combo_signal_file.blockSignals(False)
        has_multi = self.combo_signal_file.count() > 1
        self.combo_signal_scope.setEnabled(has_multi)
        self.combo_signal_file.setEnabled(self.combo_signal_scope.currentText() == "Per file")

    def _update_data_availability(self) -> None:
        has_processed = bool(self._processed)
        has_behavior = bool(self._behavior_sources)
        for w in (
            self.btn_compute,
            self.btn_update,
            self.btn_export,
            self.btn_export_img,
            self.btn_detect_peaks,
            self.btn_export_peaks,
        ):
            w.setEnabled(has_processed)
        for w in (
            self.combo_signal_source,
            self.combo_signal_scope,
            self.combo_signal_file,
            self.combo_signal_method,
            self.spin_peak_prominence,
            self.spin_peak_height,
            self.spin_peak_distance,
            self.spin_peak_smooth,
            self.combo_peak_baseline,
            self.spin_peak_baseline_window,
            self.spin_peak_rate_bin,
            self.spin_peak_auc_window,
            self.cb_peak_overlay,
        ):
            w.setEnabled(has_processed)
        for w in (
            self.btn_compute_behavior,
            self.btn_export_behavior_metrics,
            self.btn_export_behavior_events,
            self.combo_behavior_analysis,
            self.spin_behavior_bin,
            self.cb_behavior_aligned,
        ):
            w.setEnabled(has_behavior)
        self._refresh_signal_file_combo()

    def _update_status_strip(self) -> None:
        if not hasattr(self, "lbl_status"):
            return
        n_files = len(self._processed)
        src_mode = "Group" if self.tab_sources.currentIndex() == 1 else "Single"
        if self._processed:
            proc0 = self._processed[0]
            file_txt = os.path.basename(proc0.path) if proc0.path else "import"
            self.lbl_plot_file.setText(f"File: {file_txt}")
            fs_actual = float(proc0.fs_actual) if np.isfinite(proc0.fs_actual) else np.nan
            fs_used = float(proc0.fs_used) if np.isfinite(proc0.fs_used) else np.nan
            fs_txt = f"{fs_actual:.3g}->{fs_used:.3g}" if np.isfinite(fs_actual) and np.isfinite(fs_used) else "-"
            output_label = proc0.output_label or "output"
        else:
            self.lbl_plot_file.setText("File: (none)")
            fs_txt = "-"
            output_label = "-"
        align_src = self.combo_align.currentText()
        if _is_doric_channel_align(align_src):
            align_detail = self.combo_dio.currentText().strip() or "(none)"
            align_mode = self.combo_dio_align.currentText()
        else:
            align_detail = self.combo_behavior_name.currentText().strip() or "(none)"
            align_mode = self.combo_behavior_align.currentText()
        ev_count = int(self._last_events.size) if isinstance(self._last_events, np.ndarray) else 0
        win_txt = f"{float(self.spin_pre.value()):.3g}/{float(self.spin_post.value()):.3g}s"
        rs_txt = f"{float(self.spin_resample.value()):.3g}Hz"
        status_msg = (
            f"Source: {src_mode} ({n_files}) | Output: {output_label} | Align: {align_src} [{align_detail}] {align_mode} | Events: {ev_count} | Window: {win_txt} | Resample: {rs_txt} | Fs: {fs_txt}"
        )
        self.statusUpdate.emit(status_msg, 30000)

    def _update_event_filter_enabled(self) -> None:
        enabled = self.cb_filter_events.isChecked()
        for w in (self.spin_event_start, self.spin_event_end, self.spin_group_window, self.spin_dur_min, self.spin_dur_max):
            w.setEnabled(enabled)
        self._save_settings()

    def _update_metrics_enabled(self) -> None:
        enabled = self.cb_metrics.isChecked()
        for w in (
            self.combo_metric,
            self.spin_metric_pre0,
            self.spin_metric_pre1,
            self.spin_metric_post0,
            self.spin_metric_post1,
        ):
            w.setEnabled(enabled)
        self._update_metric_regions()
        self._save_settings()

    def _update_global_metrics_enabled(self) -> None:
        enabled = self.cb_global_metrics.isChecked()
        self.plot_global.setVisible(enabled)
        for w in (
            self.spin_global_start,
            self.spin_global_end,
            self.cb_global_amp,
            self.cb_global_freq,
        ):
            w.setEnabled(enabled)
        self._render_global_metrics()
        self._apply_view_layout()
        self._save_settings()

    def _refresh_section_scroll(self, key: str) -> None:
        scroll = self._section_scroll_hosts.get(key)
        if scroll is None:
            return
        try:
            content = scroll.widget()
            if content is not None:
                if content.layout() is not None:
                    content.layout().activate()
                content.updateGeometry()
            scroll.updateGeometry()
            scroll.viewport().update()
            scroll.update()
        except Exception:
            pass

    def _toggle_filter_panel(self, hide: bool) -> None:
        self.btn_hide_filters.setText("Show" if hide else "Hide")
        for w in (
            self.lbl_event_start,
            self.lbl_event_end,
            self.lbl_group_window,
            self.lbl_dur_min,
            self.lbl_dur_max,
            self.spin_event_start,
            self.spin_event_end,
            self.spin_group_window,
            self.spin_dur_min,
            self.spin_dur_max,
        ):
            w.setVisible(not hide)
        self._refresh_section_scroll("psth")

    def _toggle_metrics_panel(self, hide: bool) -> None:
        self.btn_hide_metrics.setText("Show" if hide else "Hide")
        for w in (
            self.lbl_metric,
            self.lbl_metric_pre,
            self.lbl_metric_post,
            self.combo_metric,
            self.spin_metric_pre0,
            self.spin_metric_pre1,
            self.spin_metric_post0,
            self.spin_metric_post1,
        ):
            w.setVisible(not hide)
        self._refresh_section_scroll("psth")

    def _load_behavior_files(self) -> None:
        start_dir = self._settings.value("postprocess_last_dir", os.getcwd(), type=str)
        if self._processed:
            try:
                start_dir = os.path.dirname(self._processed[0].path) or start_dir
            except Exception:
                pass
        paths, _ = QtWidgets.QFileDialog.getOpenFileNames(
            self,
            "Load behavior CSV/XLSX",
            start_dir,
            "Behavior files (*.csv *.xlsx)",
        )
        if not paths:
            return
        self._load_behavior_paths(paths, replace=True)
        try:
            self._settings.setValue("postprocess_last_dir", os.path.dirname(paths[0]))
        except Exception:
            pass
        self._refresh_behavior_list()

    def _load_processed_files(self) -> None:
        start_dir = self._settings.value("postprocess_last_dir", os.getcwd(), type=str)
        paths, _ = QtWidgets.QFileDialog.getOpenFileNames(
            self,
            "Load processed files",
            start_dir,
            "Processed files (*.csv *.h5 *.hdf5)",
        )
        if not paths:
            return
        self._load_processed_paths(paths, replace=True)
        try:
            self._settings.setValue("postprocess_last_dir", os.path.dirname(paths[0]))
        except Exception:
            pass

    def _load_processed_files_single(self) -> None:
        self._load_processed_files()
        if not self._processed:
            return
        proc = self._processed[0]
        filename = os.path.basename(proc.path) if proc.path else "import"
        self.set_current_source_label(filename, proc.channel_id or "import")

    def _load_processed_csv(self, path: str) -> Optional[ProcessedTrial]:
        import csv
        try:
            with open(path, "r", newline="") as f:
                reader = csv.reader(f)
                rows = list(reader)
        except Exception:
            return None
        if not rows:
            return None

        output_context = ""
        for r in rows:
            if not r:
                continue
            cell0 = str(r[0]).strip()
            if cell0.lower().startswith("# output_context:"):
                output_context = cell0.split(":", 1)[1].strip()
                break

        # Drop empty rows and metadata/comment rows (e.g., "# key: value")
        rows = [r for r in rows if r and any(cell.strip() for cell in r)]
        data_rows = [r for r in rows if not (r and r[0].lstrip().startswith("#"))]
        if not data_rows:
            return None

        header = [h.strip().lower() for h in data_rows[0]]

        def _find_col(names: List[str]) -> Optional[int]:
            for name in names:
                if name in header:
                    return header.index(name)
            return None

        time_idx = header.index("time") if "time" in header else None
        output_idx = _find_col(["dff", "z-score", "zscore", "z score", "output", "raw_signal", "raw_465"])
        has_header = time_idx is not None and output_idx is not None

        raw_idx = _find_col(["raw", "raw_465", "signal", "signal_465"]) if has_header else None
        iso_idx = _find_col(["isobestic", "isosbestic", "raw_405", "reference", "reference_405", "ref"]) if has_header else None
        dio_idx = _find_col(["dio"]) if has_header else None

        data_rows = data_rows[1:] if has_header else data_rows

        time = []
        output = []
        raw_vals = []
        iso_vals = []
        dio_vals = []

        for r in data_rows:
            if time_idx is None or output_idx is None:
                continue
            if len(r) <= max(time_idx, output_idx):
                continue
            try:
                tval = float(r[time_idx])
                oval = float(r[output_idx])
            except Exception:
                continue
            time.append(tval)
            output.append(oval)

            if raw_idx is not None:
                try:
                    raw_vals.append(float(r[raw_idx]) if len(r) > raw_idx else np.nan)
                except Exception:
                    raw_vals.append(np.nan)
            if iso_idx is not None:
                try:
                    iso_vals.append(float(r[iso_idx]) if len(r) > iso_idx else np.nan)
                except Exception:
                    iso_vals.append(np.nan)
            if dio_idx is not None:
                try:
                    dio_vals.append(float(r[dio_idx]) if len(r) > dio_idx else np.nan)
                except Exception:
                    dio_vals.append(np.nan)

        if not time:
            return None

        t = np.asarray(time, float)
        out = np.asarray(output, float)
        raw = np.asarray(raw_vals, float) if raw_idx is not None and len(raw_vals) == len(time) else np.full_like(t, np.nan)
        iso = np.asarray(iso_vals, float) if iso_idx is not None and len(iso_vals) == len(time) else np.full_like(t, np.nan)
        dio_arr = np.asarray(dio_vals, float) if dio_idx is not None and len(dio_vals) == len(time) else None

        output_label = "Imported CSV"
        if has_header and output_idx is not None:
            col = header[output_idx]
            if col and col != "output":
                if col == "zscore":
                    col = "z-score"
                elif col == "dff":
                    col = "dFF"
                elif col == "raw_signal":
                    col = "Raw signal (465)"
                output_label = f"Imported CSV ({col})"

        return ProcessedTrial(
            path=path,
            channel_id="import",
            time=t,
            raw_signal=raw,
            raw_reference=iso,
            dio=dio_arr,
            dio_name="",
            sig_f=None,
            ref_f=None,
            baseline_sig=None,
            baseline_ref=None,
            output=out,
            output_label=output_label,
            output_context=output_context,
            artifact_regions_sec=None,
            fs_actual=np.nan,
            fs_target=np.nan,
            fs_used=np.nan,
        )

    def _load_processed_h5(self, path: str) -> Optional[ProcessedTrial]:
        try:
            with h5py.File(path, "r") as f:
                if "data" not in f:
                    return None
                g = f["data"]
                if "time" not in g:
                    return None
                t = np.asarray(g["time"][()], float)
                if "output" in g:
                    out = np.asarray(g["output"][()], float)
                elif "dFF" in g:
                    out = np.asarray(g["dFF"][()], float)
                elif "z-score" in g:
                    out = np.asarray(g["z-score"][()], float)
                elif "zscore" in g:
                    out = np.asarray(g["zscore"][()], float)
                else:
                    return None
                raw_sig = np.asarray(g["raw_465"][()], float) if "raw_465" in g else (
                    np.asarray(g["raw"][()], float) if "raw" in g else np.full_like(t, np.nan)
                )
                raw_ref = np.asarray(g["raw_405"][()], float) if "raw_405" in g else (
                    np.asarray(g["isobestic"][()], float) if "isobestic" in g else np.full_like(t, np.nan)
                )
                dio = np.asarray(g["dio"][()], float) if "dio" in g else None
                dio_name = str(g.attrs.get("dio_name", "")) if hasattr(g, "attrs") else ""
                output_label = str(g.attrs.get("output_label", "Imported H5")) if hasattr(g, "attrs") else "Imported H5"
                output_context = str(g.attrs.get("output_context", "")) if hasattr(g, "attrs") else ""
                fs_actual = float(g.attrs.get("fs_actual", np.nan)) if hasattr(g, "attrs") else np.nan
                fs_target = float(g.attrs.get("fs_target", np.nan)) if hasattr(g, "attrs") else np.nan
                fs_used = float(g.attrs.get("fs_used", np.nan)) if hasattr(g, "attrs") else np.nan
        except Exception:
            return None

        return ProcessedTrial(
            path=path,
            channel_id="import",
            time=t,
            raw_signal=raw_sig,
            raw_reference=raw_ref,
            dio=dio,
            dio_name=dio_name,
            sig_f=None,
            ref_f=None,
            baseline_sig=None,
            baseline_ref=None,
            output=out,
            output_label=output_label,
            output_context=output_context,
            artifact_regions_sec=None,
            fs_actual=fs_actual,
            fs_target=fs_target,
            fs_used=fs_used,
        )

    def _refresh_behavior_list(self) -> None:
        self.combo_behavior_name.clear()
        if hasattr(self, "combo_behavior_analysis"):
            self.combo_behavior_analysis.clear()
        if not self._behavior_sources:
            self._update_data_availability()
            return
        behavior_names: set[str] = set()
        for info in self._behavior_sources.values():
            behaviors = info.get("behaviors") or {}
            behavior_names.update(str(k) for k in behaviors.keys())
        behaviors = sorted(list(behavior_names))
        for name in behaviors:
            self.combo_behavior_name.addItem(name)
            if hasattr(self, "combo_behavior_analysis"):
                self.combo_behavior_analysis.addItem(name)
        self.combo_behavior_from.clear()
        self.combo_behavior_to.clear()
        for name in behaviors:
            self.combo_behavior_from.addItem(name)
            self.combo_behavior_to.addItem(name)

        # Update the lists with numbered items
        self._update_file_lists()
        self._update_data_availability()
        self._update_status_strip()

    def _update_file_lists(self) -> None:
        """Update the preprocessed files and behaviors lists with numbered entries."""
        self.list_preprocessed.clear()
        for i, proc in enumerate(self._processed, 1):
            filename = os.path.splitext(os.path.basename(proc.path))[0]
            filename_clean = _strip_ain_suffix(filename)
            item = QtWidgets.QListWidgetItem(f"{i}. {filename_clean}")
            item.setData(QtCore.Qt.ItemDataRole.UserRole, id(proc))
            self.list_preprocessed.addItem(item)

        self.list_behaviors.clear()
        for i, (stem, _) in enumerate(self._behavior_sources.items(), 1):
            stem_clean = _strip_ain_suffix(stem)
            item = QtWidgets.QListWidgetItem(f"{i}. {stem_clean}")
            item.setData(QtCore.Qt.ItemDataRole.UserRole, stem)
            self.list_behaviors.addItem(item)
        self._refresh_signal_file_combo()

    def _sync_processed_order_from_list(self) -> None:
        new_order: List[ProcessedTrial] = []
        id_map = {id(proc): proc for proc in self._processed}
        for i in range(self.list_preprocessed.count()):
            item = self.list_preprocessed.item(i)
            proc_id = item.data(QtCore.Qt.ItemDataRole.UserRole) if item else None
            proc = id_map.get(proc_id)
            if proc is not None:
                new_order.append(proc)
        if new_order and len(new_order) == len(self._processed):
            self._processed = new_order
            self._update_file_lists()
            self._update_status_strip()

    def _sync_behavior_order_from_list(self) -> None:
        keys: List[str] = []
        for i in range(self.list_behaviors.count()):
            item = self.list_behaviors.item(i)
            key = item.data(QtCore.Qt.ItemDataRole.UserRole) if item else None
            if isinstance(key, str) and key in self._behavior_sources:
                keys.append(key)
        if keys and len(keys) == len(self._behavior_sources):
            self._behavior_sources = {k: self._behavior_sources[k] for k in keys}
            self._update_file_lists()
            self._update_status_strip()

    def _move_selected_up(self) -> None:
        """Move selected items up in the list."""
        # Get selected items from both lists
        selected_preproc = self.list_preprocessed.selectedItems()
        selected_behav = self.list_behaviors.selectedItems()

        if selected_preproc:
            self._move_items_up(self.list_preprocessed, selected_preproc)
            self._sync_processed_order_from_list()

        if selected_behav:
            self._move_items_up(self.list_behaviors, selected_behav)
            self._sync_behavior_order_from_list()

    def _move_selected_down(self) -> None:
        """Move selected items down in the list."""
        # Get selected items from both lists
        selected_preproc = self.list_preprocessed.selectedItems()
        selected_behav = self.list_behaviors.selectedItems()

        if selected_preproc:
            self._move_items_down(self.list_preprocessed, selected_preproc)
            self._sync_processed_order_from_list()

        if selected_behav:
            self._move_items_down(self.list_behaviors, selected_behav)
            self._sync_behavior_order_from_list()

    def _move_items_up(self, list_widget: QtWidgets.QListWidget, selected_items: List[QtWidgets.QListWidgetItem]) -> None:
        """Move selected items up in a QListWidget."""
        for item in selected_items:
            row = list_widget.row(item)
            if row > 0:
                list_widget.takeItem(row)
                list_widget.insertItem(row - 1, item)
                item.setSelected(True)

    def _move_items_down(self, list_widget: QtWidgets.QListWidget, selected_items: List[QtWidgets.QListWidgetItem]) -> None:
        """Move selected items down in a QListWidget."""
        for item in selected_items:
            row = list_widget.row(item)
            if row < list_widget.count() - 1:
                list_widget.takeItem(row)
                list_widget.insertItem(row + 1, item)
                item.setSelected(True)

    def _auto_match_files(self) -> None:
        """Auto-match preprocessed files with behavior files based on cleaned names."""
        if not self._processed or not self._behavior_sources:
            return

        # Create mapping of cleaned names to original names
        proc_names = []
        for proc in self._processed:
            filename = os.path.splitext(os.path.basename(proc.path))[0]
            filename_clean = _strip_ain_suffix(filename)
            proc_names.append(filename_clean)

        beh_names = []
        for stem in self._behavior_sources.keys():
            stem_clean = _strip_ain_suffix(stem)
            beh_names.append(stem_clean)

        # Simple auto-matching: sort both lists and try to match by position
        # For more complex matching, we could implement fuzzy string matching
        proc_sorted = sorted(proc_names)
        beh_sorted = sorted(beh_names)

        # Reorder processed files to match behavior files
        new_processed_order = []
        for beh_clean in beh_sorted:
            for i, proc_clean in enumerate(proc_names):
                if proc_clean == beh_clean and i < len(self._processed):
                    new_processed_order.append(self._processed[i])
                    break

        # Fill in any unmatched processed files at the end
        for proc in self._processed:
            if proc not in new_processed_order:
                new_processed_order.append(proc)

        self._processed = new_processed_order

        # Reorder behavior sources to match processed files
        new_behavior_order = []
        for proc in self._processed:
            proc_clean = _strip_ain_suffix(os.path.splitext(os.path.basename(proc.path))[0])
            for stem, data in self._behavior_sources.items():
                stem_clean = _strip_ain_suffix(stem)
                if stem_clean == proc_clean:
                    new_behavior_order.append((stem, data))
                    break

        # Fill in any unmatched behavior files at the end
        for stem, data in self._behavior_sources.items():
            if stem not in [s for s, _ in new_behavior_order]:
                new_behavior_order.append((stem, data))

        self._behavior_sources = dict(new_behavior_order)
        self._update_file_lists()
        self._update_status_strip()

    def _remove_selected_preprocessed(self) -> None:
        selected = self.list_preprocessed.selectedItems()
        if not selected:
            return
        rows = sorted({self.list_preprocessed.row(item) for item in selected}, reverse=True)
        for row in rows:
            if 0 <= row < len(self._processed):
                del self._processed[row]
        self._update_file_lists()
        self._update_data_availability()
        self._update_status_strip()

    def _remove_selected_behaviors(self) -> None:
        selected = self.list_behaviors.selectedItems()
        if not selected:
            return
        keys = list(self._behavior_sources.keys())
        rows = sorted({self.list_behaviors.row(item) for item in selected}, reverse=True)
        for row in rows:
            if 0 <= row < len(keys):
                key = keys[row]
                if key in self._behavior_sources:
                    del self._behavior_sources[key]
        self._refresh_behavior_list()
        self._update_data_availability()
        self._update_status_strip()

    # ---- PSTH compute ----

    def _match_behavior_source(self, proc: ProcessedTrial) -> Optional[Dict[str, Any]]:
        stem = os.path.splitext(os.path.basename(proc.path))[0]
        info = self._behavior_sources.get(stem, None)
        if info is None:
            stem_clean = _strip_ain_suffix(stem)
            for key, val in self._behavior_sources.items():
                key_clean = _strip_ain_suffix(key)
                if key_clean == stem_clean:
                    info = val
                    break
        if info is None and self._processed and self._behavior_sources:
            try:
                idx = next(i for i, p in enumerate(self._processed) if (p is proc) or (p.path == proc.path))
            except StopIteration:
                idx = None
            if idx is not None:
                keys = list(self._behavior_sources.keys())
                if 0 <= idx < len(keys):
                    info = self._behavior_sources.get(keys[idx])
        if info is None and len(self._behavior_sources) == 1:
            info = next(iter(self._behavior_sources.values()))
        return info

    def _extract_behavior_events(self, info: Dict[str, Any], behavior_name: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        behaviors = info.get("behaviors") or {}
        if behavior_name not in behaviors:
            return np.array([], float), np.array([], float), np.array([], float)
        kind = str(info.get("kind", _BEHAVIOR_PARSE_BINARY))
        if kind == _BEHAVIOR_PARSE_TIMESTAMPS:
            on = np.asarray(behaviors[behavior_name], float)
            on = on[np.isfinite(on)]
            if on.size == 0:
                return np.array([], float), np.array([], float), np.array([], float)
            on = np.sort(np.unique(on))
            off = on.copy()
            dur = np.full(on.shape, np.nan, dtype=float)
            return on, off, dur
        t = np.asarray(info.get("time", np.array([], float)), float)
        if t.size == 0:
            return np.array([], float), np.array([], float), np.array([], float)
        return _extract_onsets_offsets(t, np.asarray(behaviors[behavior_name], float), threshold=0.5)

    def _get_events_for_proc(self, proc: ProcessedTrial) -> Tuple[np.ndarray, np.ndarray]:
        align = self.combo_align.currentText()

        if _is_doric_channel_align(align):
            dio_name = self.combo_dio.currentText().strip()
            if not dio_name and proc.dio is None:
                return np.array([], float), np.array([], float)

            use_embedded = (
                proc.dio is not None
                and proc.time is not None
                and (not dio_name or dio_name == (proc.dio_name or ""))
            )

            if use_embedded:
                t = np.asarray(proc.time, float)
                x = np.asarray(proc.dio, float)
            else:
                # request if not cached
                key = (proc.path, dio_name)
                if key not in self._dio_cache:
                    self.requestDioData.emit(proc.path, dio_name)
                    return np.array([], float), np.array([], float)
                t, x = self._dio_cache[key]
            polarity = self.combo_dio_polarity.currentText()
            align_edge = self.combo_dio_align.currentText()
            sig = (np.asarray(x, float) > 0.5).astype(float)
            if polarity.startswith("Event low"):
                sig = 1.0 - sig
            on, off, dur = _extract_onsets_offsets(t, sig, threshold=0.5)
            if align_edge.endswith("offset"):
                return off, dur
            return on, dur

        # Behavior binary columns
        info = self._match_behavior_source(proc)
        if not info:
            return np.array([], float), np.array([], float)
        behaviors = info.get("behaviors") or {}
        if not behaviors:
            return np.array([], float), np.array([], float)

        align_mode = self.combo_behavior_align.currentText()
        if align_mode.startswith("Transition"):
            beh_a = self.combo_behavior_from.currentText().strip()
            beh_b = self.combo_behavior_to.currentText().strip()
            if beh_a not in behaviors or beh_b not in behaviors:
                return np.array([], float), np.array([], float)
            on_a, off_a, _ = self._extract_behavior_events(info, beh_a)
            on_b, _, dur_b = self._extract_behavior_events(info, beh_b)
            if on_a.size == 0 or on_b.size == 0:
                return np.array([], float), np.array([], float)
            gap = float(self.spin_transition_gap.value())
            times = []
            durs = []
            bi = 0
            for a_off in off_a:
                while bi < on_b.size and on_b[bi] < a_off:
                    bi += 1
                if bi >= on_b.size:
                    break
                if 0 <= on_b[bi] - a_off <= gap:
                    times.append(on_b[bi])
                    durs.append(dur_b[bi] if bi < dur_b.size else np.nan)
            return np.asarray(times, float), np.asarray(durs, float)

        beh = self.combo_behavior_name.currentText().strip()
        if not beh and behaviors:
            beh = next(iter(behaviors.keys()))
        on, off, dur = self._extract_behavior_events(info, beh)
        if on.size == 0:
            return np.array([], float), np.array([], float)
        if align_mode.endswith("offset"):
            return off, dur
        return on, dur

    def _group_close_events(
        self,
        times: np.ndarray,
        durations: np.ndarray,
        window_s: float,
    ) -> Tuple[np.ndarray, np.ndarray]:
        t = np.asarray(times, float)
        d = np.asarray(durations, float)
        if t.size < 2 or window_s <= 0:
            return t, d

        grouped_t: List[float] = []
        grouped_d: List[float] = []

        start = float(t[0])
        prev = float(t[0])
        d0 = float(d[0]) if d.size else np.nan
        cluster_end = (start + max(0.0, d0)) if np.isfinite(d0) else np.nan

        for i in range(1, t.size):
            ti = float(t[i])
            di = float(d[i]) if i < d.size else np.nan
            if (ti - prev) <= window_s:
                if np.isfinite(di):
                    end_i = ti + max(0.0, di)
                    cluster_end = end_i if not np.isfinite(cluster_end) else max(cluster_end, end_i)
                prev = ti
                continue

            grouped_t.append(start)
            grouped_d.append(max(0.0, cluster_end - start) if np.isfinite(cluster_end) else np.nan)
            start = ti
            prev = ti
            cluster_end = (ti + max(0.0, di)) if np.isfinite(di) else np.nan

        grouped_t.append(start)
        grouped_d.append(max(0.0, cluster_end - start) if np.isfinite(cluster_end) else np.nan)
        return np.asarray(grouped_t, float), np.asarray(grouped_d, float)

    def _filter_events(self, times: np.ndarray, durations: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if not self.cb_filter_events.isChecked():
            return np.asarray(times, float), np.asarray(durations, float)
        times = np.asarray(times, float)
        durations = np.asarray(durations, float) if durations is not None else np.array([], float)
        if durations.size != times.size:
            durations = np.full_like(times, np.nan, dtype=float)
        if times.size == 0:
            return times, durations

        finite = np.isfinite(times)
        if not np.any(finite):
            return np.array([], float), np.array([], float)
        times = times[finite]
        durations = durations[finite]
        order = np.argsort(times)
        times = times[order]
        durations = durations[order]

        group_window_s = max(0.0, float(self.spin_group_window.value()))
        times, durations = self._group_close_events(times, durations, group_window_s)

        start_idx = int(self.spin_event_start.value())
        end_idx = int(self.spin_event_end.value())
        if start_idx < 1:
            start_idx = 1
        if end_idx <= 0 or end_idx > times.size:
            end_idx = times.size
        if start_idx > end_idx:
            return np.array([], float), np.array([], float)

        times = times[start_idx - 1:end_idx]
        durations = durations[start_idx - 1:end_idx]

        min_dur = float(self.spin_dur_min.value())
        max_dur = float(self.spin_dur_max.value())
        if np.any(np.isfinite(durations)) and (min_dur > 0 or max_dur > 0):
            mask = np.ones_like(durations, dtype=bool)
            if min_dur > 0:
                mask &= durations >= min_dur
            if max_dur > 0:
                mask &= durations <= max_dur
            times = times[mask]
            durations = durations[mask]

        return times, durations

    def _update_trace_preview(self) -> None:
        # show first processed trace
        if not self._processed:
            self.curve_trace.setData([], [])
            self.curve_behavior.setData([], [])
            self.curve_peak_markers.setData([], [])
            for ln in self.event_lines:
                self.plot_trace.removeItem(ln)
            self.event_lines = []
            for lab in self._event_labels:
                self.plot_trace.removeItem(lab)
            self._event_labels = []
            for reg in self._event_regions:
                self.plot_trace.removeItem(reg)
            self._event_regions = []
            for ln in self._signal_peak_lines:
                self.plot_trace.removeItem(ln)
            self._signal_peak_lines = []
            return

        proc = self._processed[0]
        t = proc.time
        y = proc.output if proc.output is not None else np.full_like(t, np.nan)

        self.curve_trace.setData(t, y, connect="finite", skipFiniteCheck=True)
        self._update_behavior_overlay(proc)

        # draw event lines if possible
        for ln in self.event_lines:
            self.plot_trace.removeItem(ln)
        self.event_lines = []
        for lab in self._event_labels:
            self.plot_trace.removeItem(lab)
        self._event_labels = []
        for reg in self._event_regions:
            self.plot_trace.removeItem(reg)
        self._event_regions = []

        ev, dur = self._get_events_for_proc(proc)
        ev, dur = self._filter_events(ev, dur)
        if ev.size:
            # limit lines to a reasonable amount for UI
            ev = ev[:200]
            max_y = float(np.nanmax(y)) if np.isfinite(np.nanmax(y)) else 0.0
            for i, et in enumerate(ev, start=1):
                ln = pg.InfiniteLine(pos=float(et), angle=90, pen=pg.mkPen((220, 220, 220), width=1.0, style=QtCore.Qt.PenStyle.DashLine))
                self.plot_trace.addItem(ln)
                self.event_lines.append(ln)
                label = pg.TextItem(str(i), color=(200, 200, 200))
                label.setPos(float(et), max_y)
                self.plot_trace.addItem(label)
                self._event_labels.append(label)
                if dur is not None and i - 1 < dur.size and np.isfinite(dur[i - 1]):
                    t0 = float(et)
                    t1 = float(et + dur[i - 1])
                    reg = pg.LinearRegionItem(values=(t0, t1), brush=(200, 200, 200, 40), movable=False)
                    self.plot_trace.addItem(reg)
                    self._event_regions.append(reg)
        self._refresh_signal_overlay()

    def _update_behavior_overlay(self, proc: ProcessedTrial) -> None:
        if not proc or proc.time is None:
            self.curve_behavior.setData([], [])
            return
        if not self.combo_align.currentText().startswith("Behavior"):
            self.curve_behavior.setData([], [])
            return
        info = self._match_behavior_source(proc)
        if not info:
            self.curve_behavior.setData([], [])
            return
        behaviors = info.get("behaviors") or {}
        beh = self.combo_behavior_name.currentText().strip()
        if not beh and behaviors:
            beh = next(iter(behaviors.keys()))
        if beh not in behaviors:
            self.curve_behavior.setData([], [])
            return
        t_proc = np.asarray(proc.time, float)
        if t_proc.size == 0:
            self.curve_behavior.setData([], [])
            return
        kind = str(info.get("kind", _BEHAVIOR_PARSE_BINARY))
        if kind == _BEHAVIOR_PARSE_TIMESTAMPS:
            events = np.asarray(behaviors[beh], float)
            events = events[np.isfinite(events)]
            if events.size == 0:
                self.curve_behavior.setData([], [])
                return
            events = np.sort(np.unique(events))
            marker = np.zeros_like(t_proc, dtype=float)
            for ev in events:
                pos = int(np.searchsorted(t_proc, ev, side="left"))
                if pos <= 0:
                    idx = 0
                elif pos >= t_proc.size:
                    idx = t_proc.size - 1
                else:
                    idx = pos if abs(float(t_proc[pos] - ev)) <= abs(float(t_proc[pos - 1] - ev)) else (pos - 1)
                marker[idx] = 1.0
            self.curve_behavior.setData(t_proc, marker, connect="finite", skipFiniteCheck=True)
            return

        t = np.asarray(info.get("time", np.array([], float)), float)
        if t.size == 0:
            self.curve_behavior.setData([], [])
            return
        b = np.asarray(behaviors[beh], float)
        b_interp = np.interp(t_proc, t, b)
        self.curve_behavior.setData(t_proc, b_interp, connect="finite", skipFiniteCheck=True)

    def _resolve_signal_detection_targets(self) -> List[Tuple[str, np.ndarray, np.ndarray]]:
        targets: List[Tuple[str, np.ndarray, np.ndarray]] = []
        source_mode = self.combo_signal_source.currentText()
        if source_mode.startswith("Use PSTH input trace"):
            if self._last_mat is None or self._last_tvec is None or self._last_mat.size == 0:
                return []
            trace = np.nanmean(self._last_mat, axis=0)
            targets.append(("psth_trace", np.asarray(self._last_tvec, float), np.asarray(trace, float)))
            return targets

        if not self._processed:
            return []

        pooled = self.combo_signal_scope.currentText() == "Pooled"
        if pooled:
            for proc in self._processed:
                if proc.time is None or proc.output is None:
                    continue
                file_id = os.path.splitext(os.path.basename(proc.path))[0] if proc.path else "import"
                targets.append((file_id, np.asarray(proc.time, float), np.asarray(proc.output, float)))
            return targets

        if self.combo_signal_file.count() == 0:
            return []
        idx = self.combo_signal_file.currentIndex()
        idx = max(0, min(idx, len(self._processed) - 1))
        proc = self._processed[idx]
        if proc.time is None or proc.output is None:
            return []
        file_id = os.path.splitext(os.path.basename(proc.path))[0] if proc.path else "import"
        targets.append((file_id, np.asarray(proc.time, float), np.asarray(proc.output, float)))
        return targets

    def _preprocess_signal_for_peaks(self, t: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        t = np.asarray(t, float)
        y = np.asarray(y, float)
        m = np.isfinite(t) & np.isfinite(y)
        t = t[m]
        y = y[m]
        if t.size < 3:
            return np.array([], float), np.array([], float)

        dt = float(np.nanmedian(np.diff(t))) if t.size > 2 else np.nan
        if not np.isfinite(dt) or dt <= 0:
            dt = 1.0

        baseline_mode = self.combo_peak_baseline.currentText()
        baseline_window_sec = max(0.1, float(self.spin_peak_baseline_window.value()))
        win = max(3, int(round(baseline_window_sec / dt)))
        if win % 2 == 0:
            win += 1

        y_proc = y.copy()
        if baseline_mode.endswith("rolling median"):
            try:
                from scipy.ndimage import median_filter
                y_proc = y - median_filter(y, size=win, mode="nearest")
            except Exception:
                pass
        elif baseline_mode.endswith("rolling mean"):
            try:
                from scipy.ndimage import uniform_filter1d
                y_proc = y - uniform_filter1d(y, size=win, mode="nearest")
            except Exception:
                pass

        sigma_sec = max(0.0, float(self.spin_peak_smooth.value()))
        if sigma_sec > 0:
            try:
                from scipy.ndimage import gaussian_filter1d
                sigma_samp = sigma_sec / dt
                y_proc = gaussian_filter1d(y_proc, sigma=sigma_samp, mode="nearest")
            except Exception:
                pass

        return t, y_proc

    def _refresh_signal_overlay(self) -> None:
        for ln in self._signal_peak_lines:
            try:
                self.plot_trace.removeItem(ln)
            except Exception:
                pass
        self._signal_peak_lines = []
        self.curve_peak_markers.setData([], [])

        if not self.cb_peak_overlay.isChecked():
            return
        if not self.last_signal_events or not self._processed:
            return

        current_file = os.path.splitext(os.path.basename(self._processed[0].path))[0] if self._processed[0].path else "import"
        file_ids = self.last_signal_events.get("file_ids", [])
        times = np.asarray(self.last_signal_events.get("peak_times_sec", np.array([], float)), float)
        heights = np.asarray(self.last_signal_events.get("peak_heights", np.array([], float)), float)
        if times.size == 0 or heights.size == 0:
            return
        if file_ids and len(file_ids) == times.size:
            mask = np.asarray([fid == current_file or fid == "psth_trace" for fid in file_ids], bool)
            times = times[mask]
            heights = heights[mask]
        if times.size == 0:
            return
        self.curve_peak_markers.setData(times, heights, connect="finite", skipFiniteCheck=True)
        for t0 in times[:600]:
            ln = pg.InfiniteLine(
                pos=float(t0),
                angle=90,
                pen=pg.mkPen((240, 120, 80), width=1.0, style=QtCore.Qt.PenStyle.DotLine),
            )
            self.plot_trace.addItem(ln)
            self._signal_peak_lines.append(ln)

    def _detect_signal_events(self) -> None:
        self.last_signal_events = None
        targets = self._resolve_signal_detection_targets()
        if not targets:
            self.statusUpdate.emit("No signal available for peak detection.", 5000)
            self.tbl_signal_metrics.setRowCount(0)
            self._refresh_signal_overlay()
            return

        all_times: List[float] = []
        all_idx: List[int] = []
        all_heights: List[float] = []
        all_proms: List[float] = []
        all_widths_sec: List[float] = []
        all_auc: List[float] = []
        all_file_ids: List[str] = []

        for file_id, t_raw, y_raw in targets:
            t, y = self._preprocess_signal_for_peaks(t_raw, y_raw)
            if t.size < 5:
                continue
            dt = float(np.nanmedian(np.diff(t)))
            if not np.isfinite(dt) or dt <= 0:
                continue

            prominence = max(0.0, float(self.spin_peak_prominence.value()))
            min_height = float(self.spin_peak_height.value())
            min_distance_sec = max(0.0, float(self.spin_peak_distance.value()))
            min_dist_samples = max(1, int(round(min_distance_sec / dt))) if min_distance_sec > 0 else None

            kwargs: Dict[str, object] = {}
            if prominence > 0:
                kwargs["prominence"] = prominence
            if min_height > 0:
                kwargs["height"] = min_height
            if min_dist_samples is not None:
                kwargs["distance"] = min_dist_samples

            try:
                from scipy.signal import find_peaks, peak_widths
                peaks, props = find_peaks(y, **kwargs)
            except Exception as exc:
                self.statusUpdate.emit(f"Peak detection failed: {exc}", 5000)
                self.tbl_signal_metrics.setRowCount(0)
                self._refresh_signal_overlay()
                return

            if peaks.size == 0:
                continue

            p_heights = y[peaks]
            p_proms = np.asarray(props.get("prominences", np.full(peaks.size, np.nan)), float)
            try:
                widths_samp = peak_widths(y, peaks, rel_height=0.5)[0]
                widths_sec = np.asarray(widths_samp, float) * dt
            except Exception:
                widths_sec = np.full(peaks.size, np.nan, float)

            auc_half_win = max(0.0, float(self.spin_peak_auc_window.value()))
            auc_samp = int(round(auc_half_win / dt))
            auc_vals: List[float] = []
            for pk in peaks:
                if auc_samp <= 0:
                    auc_vals.append(np.nan)
                    continue
                i0 = max(0, int(pk - auc_samp))
                i1 = min(y.size, int(pk + auc_samp + 1))
                if i1 - i0 < 2:
                    auc_vals.append(np.nan)
                    continue
                auc_vals.append(float(np.trapz(y[i0:i1], t[i0:i1])))

            all_times.extend(t[peaks].tolist())
            all_idx.extend(peaks.tolist())
            all_heights.extend(np.asarray(p_heights, float).tolist())
            all_proms.extend(np.asarray(p_proms, float).tolist())
            all_widths_sec.extend(np.asarray(widths_sec, float).tolist())
            all_auc.extend(np.asarray(auc_vals, float).tolist())
            all_file_ids.extend([file_id] * peaks.size)

        if not all_times:
            self.statusUpdate.emit("No peaks detected with current settings.", 5000)
            self.tbl_signal_metrics.setRowCount(0)
            self._refresh_signal_overlay()
            return

        peak_times = np.asarray(all_times, float)
        peak_idx = np.asarray(all_idx, int)
        peak_heights = np.asarray(all_heights, float)
        peak_proms = np.asarray(all_proms, float)
        peak_widths_sec = np.asarray(all_widths_sec, float)
        peak_auc = np.asarray(all_auc, float)

        sort_idx = np.argsort(peak_times)
        peak_times = peak_times[sort_idx]
        peak_idx = peak_idx[sort_idx]
        peak_heights = peak_heights[sort_idx]
        peak_proms = peak_proms[sort_idx]
        peak_widths_sec = peak_widths_sec[sort_idx]
        peak_auc = peak_auc[sort_idx]
        all_file_ids = [all_file_ids[i] for i in sort_idx]

        ipi = np.diff(peak_times) if peak_times.size >= 2 else np.array([], float)
        freq_per_min = float(peak_times.size) / max((float(np.nanmax(peak_times) - np.nanmin(peak_times)) / 60.0), 1e-12)
        metrics = {
            "number_of_peaks": float(peak_times.size),
            "mean_amplitude": float(np.nanmean(peak_heights)),
            "median_amplitude": float(np.nanmedian(peak_heights)),
            "amplitude_std": float(np.nanstd(peak_heights)),
            "mean_prominence": float(np.nanmean(peak_proms)),
            "mean_width_half_prom_s": float(np.nanmean(peak_widths_sec)) if peak_widths_sec.size else np.nan,
            "peak_frequency_per_min": freq_per_min,
            "mean_inter_peak_interval_s": float(np.nanmean(ipi)) if ipi.size else np.nan,
            "mean_auc": float(np.nanmean(peak_auc)) if np.any(np.isfinite(peak_auc)) else np.nan,
        }

        self.last_signal_events = {
            "peak_times_sec": peak_times,
            "peak_indices": peak_idx,
            "peak_heights": peak_heights,
            "peak_prominences": peak_proms,
            "peak_widths_sec": peak_widths_sec,
            "peak_auc": peak_auc,
            "file_ids": all_file_ids,
            "derived_metrics": metrics,
            "params": {
                "method": self.combo_signal_method.currentText(),
                "prominence": float(self.spin_peak_prominence.value()),
                "min_height": float(self.spin_peak_height.value()),
                "min_distance_sec": float(self.spin_peak_distance.value()),
                "smooth_sigma_sec": float(self.spin_peak_smooth.value()),
                "baseline_mode": self.combo_peak_baseline.currentText(),
                "baseline_window_sec": float(self.spin_peak_baseline_window.value()),
                "rate_bin_sec": float(self.spin_peak_rate_bin.value()),
                "auc_half_window_sec": float(self.spin_peak_auc_window.value()),
            },
        }

        self.statusUpdate.emit(f"Detected {peak_times.size} peak(s).", 5000)
        self._refresh_signal_overlay()
        self._render_signal_event_plots()
        self._update_signal_metrics_table()
        self._save_settings()
        self._update_status_strip()

    def _render_signal_event_plots(self) -> None:
        for pw in (self.plot_peak_amp, self.plot_peak_ibi, self.plot_peak_rate):
            pw.clear()
        if not self.last_signal_events:
            return
        peak_times = np.asarray(self.last_signal_events.get("peak_times_sec", np.array([], float)), float)
        peak_heights = np.asarray(self.last_signal_events.get("peak_heights", np.array([], float)), float)
        if peak_times.size == 0 or peak_heights.size == 0:
            return

        def _bar_hist(plot: pg.PlotWidget, values: np.ndarray, color: Tuple[int, int, int]) -> None:
            vals = np.asarray(values, float)
            vals = vals[np.isfinite(vals)]
            if vals.size == 0:
                return
            bins = min(40, max(8, int(np.sqrt(vals.size))))
            hist, edges = np.histogram(vals, bins=bins)
            bars = pg.BarGraphItem(x=edges[:-1], height=hist, width=np.diff(edges), brush=color)
            plot.addItem(bars)

        _bar_hist(self.plot_peak_amp, peak_heights, (240, 120, 80))
        ipi = np.diff(np.sort(peak_times))
        _bar_hist(self.plot_peak_ibi, ipi, (120, 180, 220))

        bin_sec = max(0.5, float(self.spin_peak_rate_bin.value()))
        t0 = float(np.nanmin(peak_times))
        t1 = float(np.nanmax(peak_times))
        if t1 > t0:
            edges = np.arange(t0, t1 + bin_sec, bin_sec)
            hist, edges = np.histogram(peak_times, bins=edges)
            rate = hist / (bin_sec / 60.0)
            bars = pg.BarGraphItem(x=edges[:-1], height=rate, width=np.diff(edges), brush=(180, 180, 120))
            self.plot_peak_rate.addItem(bars)

    def _update_signal_metrics_table(self) -> None:
        self.tbl_signal_metrics.setRowCount(0)
        if not self.last_signal_events:
            return
        metrics = self.last_signal_events.get("derived_metrics", {}) or {}
        rows = [
            ("number of peaks", metrics.get("number_of_peaks", np.nan)),
            ("mean amplitude", metrics.get("mean_amplitude", np.nan)),
            ("median amplitude", metrics.get("median_amplitude", np.nan)),
            ("amplitude std", metrics.get("amplitude_std", np.nan)),
            ("mean prominence", metrics.get("mean_prominence", np.nan)),
            ("mean width at half prom (s)", metrics.get("mean_width_half_prom_s", np.nan)),
            ("peak frequency (per min)", metrics.get("peak_frequency_per_min", np.nan)),
            ("mean inter-peak interval (s)", metrics.get("mean_inter_peak_interval_s", np.nan)),
            ("mean AUC", metrics.get("mean_auc", np.nan)),
        ]
        for key, value in rows:
            r = self.tbl_signal_metrics.rowCount()
            self.tbl_signal_metrics.insertRow(r)
            self.tbl_signal_metrics.setItem(r, 0, QtWidgets.QTableWidgetItem(str(key)))
            if isinstance(value, (int, float)) and np.isfinite(value):
                txt = f"{float(value):.6g}"
            else:
                txt = "nan"
            self.tbl_signal_metrics.setItem(r, 1, QtWidgets.QTableWidgetItem(txt))

    def _export_signal_events_csv(self) -> None:
        if not self.last_signal_events:
            return
        start_dir = self._export_start_dir()
        out_dir = QtWidgets.QFileDialog.getExistingDirectory(self, "Select export folder", start_dir)
        if not out_dir:
            return
        self._remember_export_dir(out_dir)
        peak_times = np.asarray(self.last_signal_events.get("peak_times_sec", np.array([], float)), float)
        peak_heights = np.asarray(self.last_signal_events.get("peak_heights", np.array([], float)), float)
        peak_proms = np.asarray(self.last_signal_events.get("peak_prominences", np.array([], float)), float)
        peak_widths = np.asarray(self.last_signal_events.get("peak_widths_sec", np.array([], float)), float)
        file_ids = self.last_signal_events.get("file_ids", [])
        if peak_times.size == 0:
            return
        prefix = "signal_events"
        if self._processed:
            prefix = os.path.splitext(os.path.basename(self._processed[0].path))[0]
        out_path = os.path.join(out_dir, f"{prefix}_peaks.csv")
        import csv
        with open(out_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["peak_time_sec", "height", "prominence", "width_sec", "file_id"])
            for i in range(peak_times.size):
                fid = file_ids[i] if isinstance(file_ids, list) and i < len(file_ids) else ""
                w.writerow(
                    [
                        float(peak_times[i]),
                        float(peak_heights[i]) if i < peak_heights.size else np.nan,
                        float(peak_proms[i]) if i < peak_proms.size else np.nan,
                        float(peak_widths[i]) if i < peak_widths.size else np.nan,
                        fid,
                    ]
                )

    def _compute_behavior_analysis(self) -> None:
        self.last_behavior_analysis = None
        self.tbl_behavior_metrics.setRowCount(0)
        self.lbl_behavior_summary.setText("Group metrics: -")
        for pw in (
            self.plot_behavior_raster,
            self.plot_behavior_rate,
            self.plot_behavior_duration,
            self.plot_behavior_starts,
        ):
            pw.clear()

        if not self._behavior_sources:
            self.statusUpdate.emit("No behavior files loaded.", 5000)
            return

        behavior_name = self.combo_behavior_analysis.currentText().strip()
        if not behavior_name:
            self.statusUpdate.emit("Select a behavior to analyze.", 5000)
            return

        per_file_events: List[Dict[str, object]] = []
        per_file_metrics: List[Dict[str, float]] = []
        all_durations: List[float] = []
        all_starts: List[float] = []

        if self._processed:
            iter_rows: List[Tuple[str, Dict[str, Any], int]] = []
            for idx, proc in enumerate(self._processed):
                info = self._match_behavior_source(proc)
                if not info:
                    continue
                file_id = os.path.splitext(os.path.basename(proc.path))[0] if proc.path else f"file_{idx + 1}"
                iter_rows.append((file_id, info, idx))
        else:
            iter_rows = [(stem, info, idx) for idx, (stem, info) in enumerate(self._behavior_sources.items())]

        for file_id, info, idx in iter_rows:
            behaviors = info.get("behaviors") or {}
            if behavior_name not in behaviors:
                continue
            kind = str(info.get("kind", _BEHAVIOR_PARSE_BINARY))
            on, off, dur = self._extract_behavior_events(info, behavior_name)
            if on.size == 0:
                continue
            if kind == _BEHAVIOR_PARSE_BINARY:
                t = np.asarray(info.get("time", np.array([], float)), float)
                session_dur = float(t[-1] - t[0]) if t.size > 1 else 0.0
                total_time = float(np.nansum(dur))
                frac_time = total_time / session_dur if session_dur > 0 else np.nan
            else:
                session_dur = np.nan
                if self._processed and 0 <= idx < len(self._processed):
                    t_proc = np.asarray(self._processed[idx].time, float) if self._processed[idx].time is not None else np.array([], float)
                    if t_proc.size > 1:
                        session_dur = float(t_proc[-1] - t_proc[0])
                if (not np.isfinite(session_dur) or session_dur <= 0) and on.size > 1:
                    session_dur = float(np.nanmax(on) - np.nanmin(on))
                total_time = np.nan
                frac_time = np.nan
            rate_per_min = float(on.size) / (session_dur / 60.0) if np.isfinite(session_dur) and session_dur > 0 else np.nan
            finite_dur = dur[np.isfinite(dur)]

            metric_row = {
                "file_id": file_id,
                "event_count": float(on.size),
                "total_time": total_time,
                "mean_duration": float(np.nanmean(finite_dur)) if finite_dur.size else np.nan,
                "median_duration": float(np.nanmedian(finite_dur)) if finite_dur.size else np.nan,
                "std_duration": float(np.nanstd(finite_dur)) if finite_dur.size else np.nan,
                "rate_per_min": rate_per_min,
                "fraction_time": frac_time,
            }
            per_file_metrics.append(metric_row)
            all_durations.extend(np.asarray(dur, float).tolist())
            all_starts.extend(np.asarray(on, float).tolist())
            per_file_events.append(
                {
                    "file_id": file_id,
                    "start_sec": np.asarray(on, float),
                    "end_sec": np.asarray(off, float),
                    "duration_sec": np.asarray(dur, float),
                    "row_index": idx + 1,
                }
            )

        if not per_file_metrics:
            self.statusUpdate.emit("No behavior events found.", 5000)
            return

        def _mstd(key: str) -> Tuple[float, float]:
            vals = np.asarray([r.get(key, np.nan) for r in per_file_metrics], float)
            vals = vals[np.isfinite(vals)]
            if vals.size == 0:
                return np.nan, np.nan
            return float(np.nanmean(vals)), float(np.nanstd(vals))

        gm_count, gs_count = _mstd("event_count")
        gm_total, gs_total = _mstd("total_time")
        gm_mean, gs_mean = _mstd("mean_duration")
        gm_rate, gs_rate = _mstd("rate_per_min")
        gm_frac, gs_frac = _mstd("fraction_time")
        group_metrics = {
            "event_count_mean": gm_count,
            "event_count_std": gs_count,
            "total_time_mean": gm_total,
            "total_time_std": gs_total,
            "mean_duration_mean": gm_mean,
            "mean_duration_std": gs_mean,
            "rate_per_min_mean": gm_rate,
            "rate_per_min_std": gs_rate,
            "fraction_time_mean": gm_frac,
            "fraction_time_std": gs_frac,
        }

        self.last_behavior_analysis = {
            "behavior_name": behavior_name,
            "per_file_events": per_file_events,
            "per_file_metrics": per_file_metrics,
            "group_metrics": group_metrics,
            "params": {
                "bin_sec": float(self.spin_behavior_bin.value()),
                "aligned_view": bool(self.cb_behavior_aligned.isChecked()),
            },
        }
        self.statusUpdate.emit(f"Analyzed {len(per_file_metrics)} file(s).", 5000)
        self._render_behavior_analysis_outputs()
        self._save_settings()

    def _render_behavior_analysis_outputs(self) -> None:
        self.tbl_behavior_metrics.setRowCount(0)
        if not self.last_behavior_analysis:
            return
        per_file_metrics = self.last_behavior_analysis.get("per_file_metrics", []) or []
        for row in per_file_metrics:
            r = self.tbl_behavior_metrics.rowCount()
            self.tbl_behavior_metrics.insertRow(r)
            values = [
                row.get("file_id", ""),
                row.get("event_count", np.nan),
                row.get("total_time", np.nan),
                row.get("mean_duration", np.nan),
                row.get("median_duration", np.nan),
                row.get("std_duration", np.nan),
                row.get("rate_per_min", np.nan),
                row.get("fraction_time", np.nan),
            ]
            for c, v in enumerate(values):
                if isinstance(v, (int, float)) and np.isfinite(v):
                    txt = f"{float(v):.6g}"
                else:
                    txt = str(v)
                self.tbl_behavior_metrics.setItem(r, c, QtWidgets.QTableWidgetItem(txt))

        gm = self.last_behavior_analysis.get("group_metrics", {}) or {}
        self.lbl_behavior_summary.setText(
            "Group metrics: "
            f"count={gm.get('event_count_mean', np.nan):.4g}+-{gm.get('event_count_std', np.nan):.3g} | "
            f"rate={gm.get('rate_per_min_mean', np.nan):.4g}+-{gm.get('rate_per_min_std', np.nan):.3g} per min | "
            f"frac={gm.get('fraction_time_mean', np.nan):.4g}+-{gm.get('fraction_time_std', np.nan):.3g}"
        )

        for pw in (
            self.plot_behavior_raster,
            self.plot_behavior_rate,
            self.plot_behavior_duration,
            self.plot_behavior_starts,
        ):
            pw.clear()

        per_file_events = self.last_behavior_analysis.get("per_file_events", []) or []
        all_starts: List[float] = []
        all_durations: List[float] = []
        for ev_row in per_file_events:
            starts = np.asarray(ev_row.get("start_sec", np.array([], float)), float)
            ends = np.asarray(ev_row.get("end_sec", np.array([], float)), float)
            row_idx = float(ev_row.get("row_index", 0))
            if starts.size != ends.size:
                continue
            all_starts.extend(starts.tolist())
            all_durations.extend(np.asarray(ev_row.get("duration_sec", np.array([], float)), float).tolist())
            for s, e in zip(starts, ends):
                s_f = float(s)
                e_f = float(e)
                if np.isfinite(e_f) and e_f > s_f:
                    self.plot_behavior_raster.plot([s_f, e_f], [row_idx, row_idx], pen=pg.mkPen((220, 180, 90), width=2.0))
                else:
                    self.plot_behavior_raster.plot(
                        [s_f],
                        [row_idx],
                        pen=None,
                        symbol="o",
                        symbolSize=5,
                        symbolBrush=(220, 180, 90),
                    )

        starts_arr = np.asarray(all_starts, float)
        dur_arr = np.asarray(all_durations, float)
        starts_arr = starts_arr[np.isfinite(starts_arr)]
        dur_arr = dur_arr[np.isfinite(dur_arr)]

        if starts_arr.size:
            bin_sec = max(0.5, float(self.spin_behavior_bin.value()))
            t0 = float(np.nanmin(starts_arr))
            t1 = float(np.nanmax(starts_arr))
            if t1 > t0:
                edges = np.arange(t0, t1 + bin_sec, bin_sec)
                hist, edges = np.histogram(starts_arr, bins=edges)
                rate = hist / (bin_sec / 60.0)
                bars = pg.BarGraphItem(x=edges[:-1], height=rate, width=np.diff(edges), brush=(180, 200, 120))
                self.plot_behavior_rate.addItem(bars)

            bins = min(40, max(8, int(np.sqrt(starts_arr.size))))
            hist, edges = np.histogram(starts_arr, bins=bins)
            bars = pg.BarGraphItem(x=edges[:-1], height=hist, width=np.diff(edges), brush=(140, 190, 220))
            self.plot_behavior_starts.addItem(bars)

        if dur_arr.size:
            bins = min(40, max(8, int(np.sqrt(dur_arr.size))))
            hist, edges = np.histogram(dur_arr, bins=bins)
            bars = pg.BarGraphItem(x=edges[:-1], height=hist, width=np.diff(edges), brush=(220, 150, 110))
            self.plot_behavior_duration.addItem(bars)

    def _export_behavior_metrics_csv(self) -> None:
        if not self.last_behavior_analysis:
            return
        start_dir = self._export_start_dir()
        out_dir = QtWidgets.QFileDialog.getExistingDirectory(self, "Select export folder", start_dir)
        if not out_dir:
            return
        self._remember_export_dir(out_dir)
        behavior_name = str(self.last_behavior_analysis.get("behavior_name", "behavior"))
        safe_beh = re.sub(r"[^A-Za-z0-9_\\-]+", "_", behavior_name).strip("_") or "behavior"
        out_path = os.path.join(out_dir, f"behavior_metrics_{safe_beh}.csv")
        import csv
        with open(out_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["file_id", "behavior_name", "event_count", "total_time", "mean_duration", "median_duration", "rate_per_min", "fraction_time"])
            for row in self.last_behavior_analysis.get("per_file_metrics", []) or []:
                w.writerow(
                    [
                        row.get("file_id", ""),
                        behavior_name,
                        row.get("event_count", np.nan),
                        row.get("total_time", np.nan),
                        row.get("mean_duration", np.nan),
                        row.get("median_duration", np.nan),
                        row.get("rate_per_min", np.nan),
                        row.get("fraction_time", np.nan),
                    ]
                )

    def _export_behavior_events_csv(self) -> None:
        if not self.last_behavior_analysis:
            return
        start_dir = self._export_start_dir()
        out_dir = QtWidgets.QFileDialog.getExistingDirectory(self, "Select export folder", start_dir)
        if not out_dir:
            return
        self._remember_export_dir(out_dir)
        behavior_name = str(self.last_behavior_analysis.get("behavior_name", "behavior"))
        safe_beh = re.sub(r"[^A-Za-z0-9_\\-]+", "_", behavior_name).strip("_") or "behavior"
        out_path = os.path.join(out_dir, f"behavior_events_{safe_beh}.csv")
        import csv
        with open(out_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["start_sec", "end_sec", "duration_sec", "file_id", "behavior_name"])
            for ev_row in self.last_behavior_analysis.get("per_file_events", []) or []:
                file_id = ev_row.get("file_id", "")
                starts = np.asarray(ev_row.get("start_sec", np.array([], float)), float)
                ends = np.asarray(ev_row.get("end_sec", np.array([], float)), float)
                durs = np.asarray(ev_row.get("duration_sec", np.array([], float)), float)
                n = min(starts.size, ends.size, durs.size)
                for i in range(n):
                    w.writerow([float(starts[i]), float(ends[i]), float(durs[i]), file_id, behavior_name])

    def _compute_psth(self) -> None:
        if not self._processed:
            self.statusUpdate.emit("No processed data loaded.", 5000)
            self._last_global_metrics = None
            self._last_events = np.array([], float)
            self._last_event_rows = []
            if hasattr(self, "lbl_global_metrics"):
                self.lbl_global_metrics.setText("Global metrics: -")
            self._update_status_strip()
            return

        # Update trace preview each time (also updates event lines)
        self._update_trace_preview()

        pre = float(self.spin_pre.value())
        post = float(self.spin_post.value())
        b0 = float(self.spin_b0.value())
        b1 = float(self.spin_b1.value())
        res_hz = float(self.spin_resample.value())
        smooth = float(self.spin_smooth.value())

        window = (-pre, post)
        baseline = (b0, b1)

        try:
            group_mode = self.tab_sources.currentIndex() == 1
            mats: List[np.ndarray] = []
            animal_rows: List[np.ndarray] = []
            all_dur = []
            all_events: List[float] = []
            event_rows: List[Dict[str, object]] = []
            total_events = 0
            tvec = None

            for proc in self._processed:
                ev, dur = self._get_events_for_proc(proc)
                ev, dur = self._filter_events(ev, dur)
                if ev.size == 0:
                    continue
                file_id = os.path.splitext(os.path.basename(proc.path))[0] if proc.path else "import"
                all_events.extend(np.asarray(ev, float).tolist())
                if dur is None or len(dur) != ev.size:
                    dur_row = np.full(ev.shape, np.nan, dtype=float)
                else:
                    dur_row = np.asarray(dur, float)
                for i in range(ev.size):
                    event_rows.append(
                        {
                            "file_id": file_id,
                            "event_time_sec": float(ev[i]),
                            "duration_sec": float(dur_row[i]) if i < dur_row.size else np.nan,
                        }
                    )
                tvec, mat = _compute_psth_matrix(proc.time, proc.output, ev, window, baseline, res_hz, smooth_sigma_s=smooth)
                if mat.size == 0:
                    continue
                mats.append(mat)
                total_events += mat.shape[0]
                if group_mode:
                    row = np.nanmean(mat, axis=0)
                    if np.any(np.isfinite(row)):
                        animal_rows.append(row)
                if dur is not None and len(dur):
                    all_dur.append(np.asarray(dur, float))

            self._render_global_metrics()

            if not mats or tvec is None:
                self.statusUpdate.emit("No events found for the current alignment.", 5000)
                self._last_events = np.array([], float)
                self._last_event_rows = []
                self._last_durations = np.array([], float)
                self._update_status_strip()
                return

            mat_events = np.vstack(mats)
            if group_mode:
                if not animal_rows:
                    self.lbl_log.setText("No events found for the current alignment.")
                    self._last_events = np.array([], float)
                    self._last_event_rows = []
                    self._last_durations = np.array([], float)
                    self._update_status_strip()
                    return
                mat_display = np.vstack(animal_rows)
            else:
                mat_display = mat_events

            self._render_heatmap(mat_display, tvec)
            self._render_avg(mat_display, tvec)
            dur_all = np.concatenate(all_dur) if all_dur else np.array([], float)
            self._render_duration_hist(dur_all)
            self._render_metrics(mat_display, tvec)
            self._last_mat = mat_display
            self._last_tvec = tvec
            self._last_events = np.asarray(all_events, float) if all_events else np.array([], float)
            self._last_durations = dur_all
            self._last_event_rows = event_rows
            if group_mode:
                self.statusUpdate.emit(f"Computed PSTH for {total_events} event(s) across {mat_display.shape[0]} animal(s).", 5000)
            else:
                self.statusUpdate.emit(f"Computed PSTH for {total_events} event(s).", 5000)
            self._update_metric_regions()
            self._update_status_strip()
            self._save_settings()
        except Exception as e:
            self.statusUpdate.emit(f"Post-processing error: {e}", 5000)
            self._update_status_strip()

    def _render_heatmap(self, mat: np.ndarray, tvec: np.ndarray) -> None:
        if mat.size == 0:
            self.img.setImage(np.zeros((1, 1)))
            return

        # ImageItem maps axis-0 -> x and axis-1 -> y; transpose so time is x, trials are y.
        img = np.asarray(mat, float).T
        cmap_name = str(self._style.get("heatmap_cmap", "viridis"))
        try:
            cmap = pg.colormap.get(cmap_name)
            lut = cmap.getLookupTable()
            self.img.setLookupTable(lut)
        except Exception:
            pass
        hmin = self._style.get("heatmap_min", None)
        hmax = self._style.get("heatmap_max", None)
        # set image (auto-level)
        self.img.setImage(img, autoLevels=True)
        if hmin is not None and hmax is not None:
            self.img.setLevels([float(hmin), float(hmax)])

        # Map image to time axis using a rect (avoids scale() incompatibilities)
        x0 = float(tvec[0]) if tvec.size else 0.0
        x1 = float(tvec[-1]) if tvec.size else 1.0
        if x1 == x0:
            x1 = x0 + 1.0
        self.img.setRect(QtCore.QRectF(x0, 0, x1 - x0, img.shape[1]))
        self.plot_heat.setXRange(x0, x1, padding=0)
        self.plot_heat.setYRange(0, float(img.shape[1]), padding=0)

    def _render_duration_hist(self, durations: np.ndarray) -> None:
        self.plot_dur.clear()
        if durations is None or durations.size == 0:
            txt = pg.TextItem("No durations", color=(170, 180, 196))
            txt.setPos(0, 0)
            self.plot_dur.addItem(txt)
            self.plot_dur.setXRange(0, 1, padding=0)
            self.plot_dur.setYRange(0, 1, padding=0)
            return
        d = np.asarray(durations, float)
        d = d[np.isfinite(d)]
        if d.size == 0:
            txt = pg.TextItem("No durations", color=(170, 180, 196))
            txt.setPos(0, 0)
            self.plot_dur.addItem(txt)
            self.plot_dur.setXRange(0, 1, padding=0)
            self.plot_dur.setYRange(0, 1, padding=0)
            return
        bins = min(20, max(5, int(np.sqrt(d.size))))
        hist, edges = np.histogram(d, bins=bins)
        bg = pg.BarGraphItem(x=edges[:-1], height=hist, width=np.diff(edges), brush=pg.mkBrush(90, 143, 214))
        self.plot_dur.addItem(bg)
        self.plot_dur.setXRange(float(edges[0]), float(edges[-1]), padding=0.05)
        self.plot_dur.setYRange(0, float(np.max(hist)) if hist.size else 1.0, padding=0.1)
        self.plot_dur.enableAutoRange(axis=pg.ViewBox.YAxis, enable=False)
        self.plot_dur.enableAutoRange(axis=pg.ViewBox.XAxis, enable=False)

    def _render_avg(self, mat: np.ndarray, tvec: np.ndarray) -> None:
        if mat.size == 0:
            self.curve_avg.setData([], [])
            self.curve_sem_hi.setData([], [])
            self.curve_sem_lo.setData([], [])
            return

        avg = np.nanmean(mat, axis=0)
        sem = np.nanstd(mat, axis=0) / np.sqrt(max(1, np.sum(np.any(np.isfinite(mat), axis=1))))

        self.curve_avg.setData(tvec, avg, connect="finite", skipFiniteCheck=True)
        self.curve_sem_hi.setData(tvec, avg + sem, connect="finite", skipFiniteCheck=True)
        self.curve_sem_lo.setData(tvec, avg - sem, connect="finite", skipFiniteCheck=True)
        self.plot_avg.setXRange(float(tvec[0]), float(tvec[-1]), padding=0)

    def _render_metrics(self, mat: np.ndarray, tvec: np.ndarray) -> None:
        if mat.size == 0 or not self.cb_metrics.isChecked():
            self.metrics_bar_pre.setOpts(height=[0])
            self.metrics_bar_post.setOpts(height=[0])
            self._last_metrics = None
            return
        metric = self.combo_metric.currentText()
        pre0 = float(self.spin_metric_pre0.value())
        pre1 = float(self.spin_metric_pre1.value())
        post0 = float(self.spin_metric_post0.value())
        post1 = float(self.spin_metric_post1.value())

        def _window_vals(a: float, b: float) -> np.ndarray:
            mask = (tvec >= a) & (tvec <= b)
            if not np.any(mask):
                return np.array([], float)
            return mat[:, mask]

        pre = _window_vals(pre0, pre1)
        post = _window_vals(post0, post1)
        if pre.size == 0 or post.size == 0:
            self.metrics_bar_pre.setOpts(height=[0])
            self.metrics_bar_post.setOpts(height=[0])
            self._last_metrics = None
            return

        def _metric_vals(win: np.ndarray, duration: float) -> np.ndarray:
            if win.size == 0:
                return np.array([], float)
            if metric.startswith("AUC"):
                with np.errstate(all="ignore"):
                    vals = np.nanmean(win, axis=1) * float(abs(duration))
            else:
                with np.errstate(all="ignore"):
                    vals = np.nanmean(win, axis=1)
            vals = np.asarray(vals, float)
            vals = vals[np.isfinite(vals)]
            return vals

        pre_vals = _metric_vals(pre, pre1 - pre0)
        post_vals = _metric_vals(post, post1 - post0)
        pre_mean = float(np.nanmean(pre_vals)) if pre_vals.size else 0.0
        post_mean = float(np.nanmean(post_vals)) if post_vals.size else 0.0
        self.metrics_bar_pre.setOpts(height=[pre_mean])
        self.metrics_bar_post.setOpts(height=[post_mean])
        ymin = min(pre_mean, post_mean, 0.0)
        ymax = max(pre_mean, post_mean, 0.0)
        if ymin == ymax:
            ymax = ymin + 1.0
        self.plot_metrics.setYRange(ymin, ymax, padding=0.2)
        self._last_metrics = {"pre": pre_mean, "post": post_mean, "metric": metric}

    def _compute_global_metrics_for_trace(
        self,
        t: np.ndarray,
        y: np.ndarray,
        start_s: float,
        end_s: float,
    ) -> Optional[Dict[str, float]]:
        tt = np.asarray(t, float)
        yy = np.asarray(y, float)
        m = np.isfinite(tt) & np.isfinite(yy)
        tt = tt[m]
        yy = yy[m]
        if tt.size < 3:
            return None

        if np.isfinite(start_s) and np.isfinite(end_s) and end_s > start_s:
            mask = (tt >= start_s) & (tt <= end_s)
            tt = tt[mask]
            yy = yy[mask]
            if tt.size < 3:
                return None

        med = float(np.nanmedian(yy))
        mad = float(np.nanmedian(np.abs(yy - med)))
        hi_thr = med + 2.0 * mad
        if yy.size >= 3:
            peak_idx = np.where((yy[1:-1] > yy[:-2]) & (yy[1:-1] > yy[2:]))[0] + 1
            hi_idx = peak_idx[yy[peak_idx] > hi_thr]
            mask = np.ones(yy.size, dtype=bool)
            mask[hi_idx] = False
            yy_filt = yy[mask]
        else:
            yy_filt = yy
        med_filt = float(np.nanmedian(yy_filt)) if yy_filt.size else med
        thr = 3.0 * med_filt

        if yy.size < 3:
            return None
        peak_idx = np.where((yy[1:-1] > yy[:-2]) & (yy[1:-1] > yy[2:]) & (yy[1:-1] >= thr))[0] + 1
        peak_vals = yy[peak_idx]
        amp = float(np.nanmean(peak_vals)) if peak_vals.size else 0.0
        duration = float(tt[-1] - tt[0]) if tt.size > 1 else 0.0
        freq = float(peak_vals.size) / duration if duration > 0 else 0.0
        return {
            "amp": amp,
            "freq": freq,
            "thr": thr,
            "peaks": float(peak_vals.size),
            "duration": duration,
        }

    def _render_global_metrics(self) -> None:
        if not self.cb_global_metrics.isChecked():
            self._last_global_metrics = None
            self.lbl_global_metrics.setText("Global metrics: -")
            self.global_bar_amp.setOpts(height=[0])
            self.global_bar_freq.setOpts(height=[0])
            return

        if not (self.cb_global_amp.isChecked() or self.cb_global_freq.isChecked()):
            self._last_global_metrics = None
            self.lbl_global_metrics.setText("Global metrics: -")
            self.global_bar_amp.setOpts(height=[0])
            self.global_bar_freq.setOpts(height=[0])
            return

        start_s = float(self.spin_global_start.value())
        end_s = float(self.spin_global_end.value())

        amps = []
        freqs = []
        peaks = []
        durations = []
        thrs = []

        for proc in self._processed:
            if proc.output is None or proc.time is None:
                continue
            res = self._compute_global_metrics_for_trace(proc.time, proc.output, start_s, end_s)
            if not res:
                continue
            amps.append(res["amp"])
            freqs.append(res["freq"])
            peaks.append(res["peaks"])
            durations.append(res["duration"])
            thrs.append(res["thr"])

        if not amps and not freqs:
            self._last_global_metrics = None
            self.lbl_global_metrics.setText("Global metrics: -")
            self.global_bar_amp.setOpts(height=[0])
            self.global_bar_freq.setOpts(height=[0])
            return

        avg_amp = float(np.nanmean(amps)) if amps else 0.0
        avg_freq = float(np.nanmean(freqs)) if freqs else 0.0
        total_peaks = float(np.nansum(peaks)) if peaks else 0.0
        avg_thr = float(np.nanmean(thrs)) if thrs else 0.0
        avg_dur = float(np.nanmean(durations)) if durations else 0.0

        self._last_global_metrics = {
            "amp": avg_amp,
            "freq": avg_freq,
            "peaks": total_peaks,
            "thr": avg_thr,
            "duration": avg_dur,
            "start": start_s,
            "end": end_s,
        }

        parts = []
        if self.cb_global_amp.isChecked():
            parts.append(f"amp={avg_amp:.4g}")
        if self.cb_global_freq.isChecked():
            parts.append(f"freq={avg_freq:.4g} Hz")
        parts.append(f"peaks={int(total_peaks)}")
        self.lbl_global_metrics.setText("Global metrics: " + " | ".join(parts))

        self.global_bar_amp.setOpts(height=[avg_amp if self.cb_global_amp.isChecked() else 0.0])
        self.global_bar_freq.setOpts(height=[avg_freq if self.cb_global_freq.isChecked() else 0.0])
        ymin = min(0.0, avg_amp if self.cb_global_amp.isChecked() else 0.0, avg_freq if self.cb_global_freq.isChecked() else 0.0)
        ymax = max(avg_amp if self.cb_global_amp.isChecked() else 0.0, avg_freq if self.cb_global_freq.isChecked() else 0.0, 0.0)
        if ymin == ymax:
            ymax = ymin + 1.0
        self.plot_global.setYRange(ymin, ymax, padding=0.2)

    def _update_metric_regions(self) -> None:
        if self._pre_region is not None:
            self.plot_avg.removeItem(self._pre_region)
        if self._post_region is not None:
            self.plot_avg.removeItem(self._post_region)

        if not self.cb_metrics.isChecked():
            self._pre_region = None
            self._post_region = None
            return

        pre0 = float(self.spin_metric_pre0.value())
        pre1 = float(self.spin_metric_pre1.value())
        post0 = float(self.spin_metric_post0.value())
        post1 = float(self.spin_metric_post1.value())

        self._pre_region = pg.LinearRegionItem(values=(pre0, pre1), brush=(90, 143, 214, 60), movable=False)
        self._post_region = pg.LinearRegionItem(values=(post0, post1), brush=(214, 122, 90, 60), movable=False)
        self.plot_avg.addItem(self._pre_region)
        self.plot_avg.addItem(self._post_region)

    def _open_style_dialog(self) -> None:
        dlg = StyleDialog(self._style, self)
        if dlg.exec() != QtWidgets.QDialog.DialogCode.Accepted:
            return
        self._style = dlg.get_style()
        self.curve_trace.setPen(pg.mkPen(self._style["trace"], width=1.1))
        self.curve_behavior.setPen(pg.mkPen(self._style["behavior"], width=1.0))
        self.curve_avg.setPen(pg.mkPen(self._style["avg"], width=1.3))
        self._render_heatmap(self._last_mat if self._last_mat is not None else np.zeros((1, 1)), self._last_tvec if self._last_tvec is not None else np.array([0.0, 1.0]))
        self._save_settings()

    def _save_config_file(self) -> None:
        start_dir = self._settings.value("postprocess_last_dir", os.getcwd(), type=str)
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Save config", os.path.join(start_dir, "postprocess_config.json"), "JSON (*.json)")
        if not path:
            return
        data = self._collect_settings()
        try:
            import json
            with open(path, "w") as f:
                json.dump(data, f, indent=2)
            self._settings.setValue("postprocess_last_dir", os.path.dirname(path))
        except Exception:
            pass

    def _load_config_file(self) -> None:
        start_dir = self._settings.value("postprocess_last_dir", os.getcwd(), type=str)
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Load config", start_dir, "JSON (*.json)")
        if not path:
            return
        try:
            import json
            with open(path, "r") as f:
                data = json.load(f)
            self._apply_settings(data)
            self._compute_psth()
            self._settings.setValue("postprocess_last_dir", os.path.dirname(path))
        except Exception:
            pass

    def _collect_settings(self) -> Dict[str, object]:
        return {
            "align": self.combo_align.currentText(),
            "dio_channel": self.combo_dio.currentText(),
            "dio_polarity": self.combo_dio_polarity.currentText(),
            "dio_align": self.combo_dio_align.currentText(),
            "behavior_file_type": self.combo_behavior_file_type.currentText(),
            "behavior": self.combo_behavior_name.currentText(),
            "behavior_align": self.combo_behavior_align.currentText(),
            "behavior_from": self.combo_behavior_from.currentText(),
            "behavior_to": self.combo_behavior_to.currentText(),
            "transition_gap": float(self.spin_transition_gap.value()),
            "window_pre": float(self.spin_pre.value()),
            "window_post": float(self.spin_post.value()),
            "baseline_start": float(self.spin_b0.value()),
            "baseline_end": float(self.spin_b1.value()),
            "resample": float(self.spin_resample.value()),
            "smooth": float(self.spin_smooth.value()),
            "filter_enabled": self.cb_filter_events.isChecked(),
            "event_start": int(self.spin_event_start.value()),
            "event_end": int(self.spin_event_end.value()),
            "group_window_s": float(self.spin_group_window.value()),
            "dur_min": float(self.spin_dur_min.value()),
            "dur_max": float(self.spin_dur_max.value()),
            "metrics_enabled": self.cb_metrics.isChecked(),
            "metric": self.combo_metric.currentText(),
            "metric_pre0": float(self.spin_metric_pre0.value()),
            "metric_pre1": float(self.spin_metric_pre1.value()),
            "metric_post0": float(self.spin_metric_post0.value()),
            "metric_post1": float(self.spin_metric_post1.value()),
            "global_metrics_enabled": self.cb_global_metrics.isChecked(),
            "global_start": float(self.spin_global_start.value()),
            "global_end": float(self.spin_global_end.value()),
            "global_amp": self.cb_global_amp.isChecked(),
            "global_freq": self.cb_global_freq.isChecked(),
            "view_layout": self.combo_view_layout.currentText(),
            "signal_source": self.combo_signal_source.currentText(),
            "signal_scope": self.combo_signal_scope.currentText(),
            "signal_file": self.combo_signal_file.currentText(),
            "signal_method": self.combo_signal_method.currentText(),
            "signal_prominence": float(self.spin_peak_prominence.value()),
            "signal_height": float(self.spin_peak_height.value()),
            "signal_distance": float(self.spin_peak_distance.value()),
            "signal_smooth": float(self.spin_peak_smooth.value()),
            "signal_baseline": self.combo_peak_baseline.currentText(),
            "signal_baseline_window": float(self.spin_peak_baseline_window.value()),
            "signal_rate_bin": float(self.spin_peak_rate_bin.value()),
            "signal_auc_window": float(self.spin_peak_auc_window.value()),
            "signal_overlay": self.cb_peak_overlay.isChecked(),
            "behavior_analysis_name": self.combo_behavior_analysis.currentText(),
            "behavior_analysis_bin": float(self.spin_behavior_bin.value()),
            "behavior_analysis_aligned": self.cb_behavior_aligned.isChecked(),
            "style": dict(self._style),
        }

    def _apply_settings(self, data: Dict[str, object]) -> None:
        def _set_combo(combo: QtWidgets.QComboBox, val: object) -> None:
            if val is None:
                return
            if combo is self.combo_align and str(val) == "DIO (from Doric)":
                val = "Analog/Digital channel (from Doric)"
            idx = combo.findText(str(val))
            if idx >= 0:
                combo.setCurrentIndex(idx)

        _set_combo(self.combo_align, data.get("align"))
        _set_combo(self.combo_dio, data.get("dio_channel"))
        _set_combo(self.combo_dio_polarity, data.get("dio_polarity"))
        _set_combo(self.combo_dio_align, data.get("dio_align"))
        _set_combo(self.combo_behavior_file_type, data.get("behavior_file_type"))
        _set_combo(self.combo_behavior_name, data.get("behavior"))
        _set_combo(self.combo_behavior_align, data.get("behavior_align"))
        _set_combo(self.combo_behavior_from, data.get("behavior_from"))
        _set_combo(self.combo_behavior_to, data.get("behavior_to"))
        if "transition_gap" in data:
            self.spin_transition_gap.setValue(float(data["transition_gap"]))
        if "window_pre" in data:
            self.spin_pre.setValue(float(data["window_pre"]))
        if "window_post" in data:
            self.spin_post.setValue(float(data["window_post"]))
        if "baseline_start" in data:
            self.spin_b0.setValue(float(data["baseline_start"]))
        if "baseline_end" in data:
            self.spin_b1.setValue(float(data["baseline_end"]))
        if "resample" in data:
            self.spin_resample.setValue(float(data["resample"]))
        if "smooth" in data:
            self.spin_smooth.setValue(float(data["smooth"]))
        self.cb_filter_events.setChecked(bool(data.get("filter_enabled", True)))
        if "event_start" in data:
            self.spin_event_start.setValue(int(data["event_start"]))
        if "event_end" in data:
            self.spin_event_end.setValue(int(data["event_end"]))
        if "group_window_s" in data:
            self.spin_group_window.setValue(float(data["group_window_s"]))
        if "dur_min" in data:
            self.spin_dur_min.setValue(float(data["dur_min"]))
        if "dur_max" in data:
            self.spin_dur_max.setValue(float(data["dur_max"]))
        self.cb_metrics.setChecked(bool(data.get("metrics_enabled", True)))
        _set_combo(self.combo_metric, data.get("metric"))
        if "metric_pre0" in data:
            self.spin_metric_pre0.setValue(float(data["metric_pre0"]))
        if "metric_pre1" in data:
            self.spin_metric_pre1.setValue(float(data["metric_pre1"]))
        if "metric_post0" in data:
            self.spin_metric_post0.setValue(float(data["metric_post0"]))
        if "metric_post1" in data:
            self.spin_metric_post1.setValue(float(data["metric_post1"]))
        self.cb_global_metrics.setChecked(bool(data.get("global_metrics_enabled", True)))
        if "global_start" in data:
            self.spin_global_start.setValue(float(data["global_start"]))
        if "global_end" in data:
            self.spin_global_end.setValue(float(data["global_end"]))
        if "global_amp" in data:
            self.cb_global_amp.setChecked(bool(data["global_amp"]))
        if "global_freq" in data:
            self.cb_global_freq.setChecked(bool(data["global_freq"]))
        _set_combo(self.combo_view_layout, data.get("view_layout"))
        _set_combo(self.combo_signal_source, data.get("signal_source"))
        _set_combo(self.combo_signal_scope, data.get("signal_scope"))
        self._refresh_signal_file_combo()
        _set_combo(self.combo_signal_file, data.get("signal_file"))
        _set_combo(self.combo_signal_method, data.get("signal_method"))
        if "signal_prominence" in data:
            self.spin_peak_prominence.setValue(float(data["signal_prominence"]))
        if "signal_height" in data:
            self.spin_peak_height.setValue(float(data["signal_height"]))
        if "signal_distance" in data:
            self.spin_peak_distance.setValue(float(data["signal_distance"]))
        if "signal_smooth" in data:
            self.spin_peak_smooth.setValue(float(data["signal_smooth"]))
        _set_combo(self.combo_peak_baseline, data.get("signal_baseline"))
        if "signal_baseline_window" in data:
            self.spin_peak_baseline_window.setValue(float(data["signal_baseline_window"]))
        if "signal_rate_bin" in data:
            self.spin_peak_rate_bin.setValue(float(data["signal_rate_bin"]))
        if "signal_auc_window" in data:
            self.spin_peak_auc_window.setValue(float(data["signal_auc_window"]))
        if "signal_overlay" in data:
            self.cb_peak_overlay.setChecked(bool(data["signal_overlay"]))
        _set_combo(self.combo_behavior_analysis, data.get("behavior_analysis_name"))
        if "behavior_analysis_bin" in data:
            self.spin_behavior_bin.setValue(float(data["behavior_analysis_bin"]))
        if "behavior_analysis_aligned" in data:
            self.cb_behavior_aligned.setChecked(bool(data["behavior_analysis_aligned"]))
        style = data.get("style")
        if isinstance(style, dict):
            self._style.update(style)
            self.curve_trace.setPen(pg.mkPen(self._style["trace"], width=1.1))
            self.curve_behavior.setPen(pg.mkPen(self._style["behavior"], width=1.0))
            self.curve_avg.setPen(pg.mkPen(self._style["avg"], width=1.3))
        self._update_event_filter_enabled()
        self._update_metrics_enabled()
        self._update_global_metrics_enabled()
        self._update_metric_regions()
        self._apply_view_layout()
        self._refresh_signal_file_combo()
        self._update_data_availability()
        self._update_status_strip()

    def _save_settings(self) -> None:
        try:
            data = self._collect_settings()
            self._settings.setValue("postprocess_json", json.dumps(data))
        except Exception:
            pass
        self._save_panel_layout_state()
        try:
            self._settings.sync()
        except Exception:
            pass

    def _restore_settings(self) -> None:
        try:
            raw = self._settings.value("postprocess_json", "", type=str)
            if raw:
                data = json.loads(raw)
                self._apply_settings(data)
        except Exception:
            pass

    def _export_origin_dir(self) -> str:
        if self._processed:
            p = self._processed[0].path
            if p:
                d = os.path.dirname(p)
                if d and os.path.isdir(d):
                    return d
        return ""

    def _export_start_dir(self) -> str:
        origin_dir = self._export_origin_dir()
        last_dir = self._settings.value("postprocess_last_dir", "", type=str)
        override = self._settings.value("postprocess_last_dir_override", False, type=bool)

        def _valid(p: str) -> bool:
            return bool(p) and os.path.isdir(p)

        if override and _valid(last_dir):
            return last_dir
        if _valid(origin_dir):
            return origin_dir
        if _valid(last_dir):
            return last_dir
        return os.getcwd()

    def _remember_export_dir(self, out_dir: str) -> None:
        origin_dir = self._export_origin_dir()
        try:
            self._settings.setValue("postprocess_last_dir", out_dir)
            out_norm = os.path.normcase(os.path.abspath(out_dir)) if out_dir else ""
            origin_norm = os.path.normcase(os.path.abspath(origin_dir)) if origin_dir else ""
            override = bool(out_norm) and (not origin_norm or out_norm != origin_norm)
            self._settings.setValue("postprocess_last_dir_override", override)
        except Exception:
            pass

    def _export_results(self) -> None:
        if self._last_mat is None or self._last_tvec is None:
            return
        dlg = ExportDialog(self)
        if dlg.exec() != QtWidgets.QDialog.DialogCode.Accepted:
            return
        choices = dlg.choices()
        start_dir = self._export_start_dir()
        out_dir = QtWidgets.QFileDialog.getExistingDirectory(self, "Select export folder", start_dir)
        if not out_dir:
            return
        self._remember_export_dir(out_dir)
        prefix = "postprocess"
        if self._processed:
            prefix = os.path.splitext(os.path.basename(self._processed[0].path))[0]
        beh_suffix = self._behavior_suffix()
        if beh_suffix:
            prefix = f"{prefix}_{beh_suffix}"

        if choices.get("heatmap"):
            np.savetxt(os.path.join(out_dir, f"{prefix}_heatmap.csv"), self._last_mat, delimiter=",")
            np.savetxt(os.path.join(out_dir, f"{prefix}_heatmap_tvec.csv"), self._last_tvec, delimiter=",")
        if choices.get("avg"):
            avg = np.nanmean(self._last_mat, axis=0)
            sem = np.nanstd(self._last_mat, axis=0) / np.sqrt(max(1, np.sum(np.any(np.isfinite(self._last_mat), axis=1))))
            arr = np.vstack([self._last_tvec, avg, sem]).T
            np.savetxt(os.path.join(out_dir, f"{prefix}_avg_psth.csv"), arr, delimiter=",", header="time,avg,sem", comments="")
        if choices.get("events"):
            event_path = os.path.join(out_dir, f"{prefix}_events.csv")
            if self._last_event_rows:
                import csv
                with open(event_path, "w", newline="") as f:
                    w = csv.writer(f)
                    w.writerow(["file_id", "event_time_sec", "duration_sec"])
                    for row in self._last_event_rows:
                        w.writerow([row.get("file_id", ""), row.get("event_time_sec", np.nan), row.get("duration_sec", np.nan)])
            elif self._last_events is not None:
                np.savetxt(event_path, self._last_events, delimiter=",")
        if choices.get("durations") and self._last_durations is not None:
            np.savetxt(os.path.join(out_dir, f"{prefix}_durations.csv"), self._last_durations, delimiter=",")
        if choices.get("metrics") and (self._last_metrics or self._last_global_metrics):
            import csv
            with open(os.path.join(out_dir, f"{prefix}_metrics.csv"), "w", newline="") as f:
                w = csv.writer(f)
                if self._last_metrics:
                    w.writerow(["metric", "pre", "post"])
                    w.writerow([self._last_metrics.get("metric", ""), self._last_metrics.get("pre", ""), self._last_metrics.get("post", "")])
                if self._last_global_metrics:
                    if self._last_metrics:
                        w.writerow([])
                    w.writerow(["global_amp", "global_freq_hz", "global_start_s", "global_end_s", "global_peaks", "global_threshold", "global_duration_s"])
                    w.writerow([
                        self._last_global_metrics.get("amp", ""),
                        self._last_global_metrics.get("freq", ""),
                        self._last_global_metrics.get("start", ""),
                        self._last_global_metrics.get("end", ""),
                        self._last_global_metrics.get("peaks", ""),
                        self._last_global_metrics.get("thr", ""),
                        self._last_global_metrics.get("duration", ""),
                    ])

    def _behavior_suffix(self) -> str:
        if not self.combo_align.currentText().startswith("Behavior"):
            return ""
        align_mode = self.combo_behavior_align.currentText()
        if align_mode.startswith("Transition"):
            a = self.combo_behavior_from.currentText().strip()
            b = self.combo_behavior_to.currentText().strip()
            name = f"{a}_to_{b}" if a and b else ""
        else:
            name = self.combo_behavior_name.currentText().strip()
        if not name:
            return ""
        cleaned = re.sub(r"\s+", "_", name)
        cleaned = re.sub(r"[^A-Za-z0-9_\-]+", "", cleaned)
        return cleaned

    def _export_images(self) -> None:
        start_dir = self._export_start_dir()
        out_dir = QtWidgets.QFileDialog.getExistingDirectory(self, "Select export folder", start_dir)
        if not out_dir:
            return
        self._remember_export_dir(out_dir)
        prefix = "postprocess"
        if self._processed:
            prefix = os.path.splitext(os.path.basename(self._processed[0].path))[0]

        targets = {
            "overview": self._right_panel,
            "trace": self.plot_trace,
            "heatmap": self.plot_heat,
            "duration": self.plot_dur,
            "avg": self.plot_avg,
            "metrics": self.plot_metrics,
            "global": self.plot_global,
            "peaks_amp": self.plot_peak_amp,
            "peaks_ibi": self.plot_peak_ibi,
            "peaks_rate": self.plot_peak_rate,
            "behavior_raster": self.plot_behavior_raster,
            "behavior_rate": self.plot_behavior_rate,
            "behavior_duration": self.plot_behavior_duration,
            "behavior_starts": self.plot_behavior_starts,
        }
        for name, widget in targets.items():
            try:
                pix = widget.grab()
                pix.save(os.path.join(out_dir, f"{prefix}_{name}.png"))
            except Exception:
                pass

    def hideEvent(self, event: QtGui.QHideEvent) -> None:
        super().hideEvent(event)
        if self._app_closing:
            return
        self.hide_section_popups_for_tab_switch()

    def hide_section_popups_for_tab_switch(self) -> None:
        """Hide and detach post-processing docks when tab is inactive."""
        if not self._section_popups:
            return
        if self._post_docks_hidden_for_tab_switch:
            return
        host = self._dock_host or self._dock_main_window()
        self._post_section_visibility_before_hide = {}
        self._post_section_state_before_hide = {}
        for key, dock in self._section_popups.items():
            visible = bool(dock.isVisible())
            self._post_section_visibility_before_hide[key] = visible
            area = (
                _dock_area_to_int(host.dockWidgetArea(dock), _dock_area_to_int(QtCore.Qt.DockWidgetArea.RightDockWidgetArea, 2))
                if host is not None
                else _dock_area_to_int(QtCore.Qt.DockWidgetArea.RightDockWidgetArea, 2)
            )
            self._post_section_state_before_hide[key] = {
                "visible": visible,
                "floating": bool(dock.isFloating()),
                "area": area,
                "geometry": dock.saveGeometry(),
            }
        try:
            if host is not None:
                state = None
                if hasattr(host, "captureDockSnapshotForTab"):
                    state = host.captureDockSnapshotForTab("post")
                else:
                    state = host.saveState(_DOCK_STATE_VERSION)
                if state is not None and not state.isEmpty():
                    self._settings.setValue(_POST_DOCK_STATE_KEY, state)
                    self._settings.sync()
        except Exception:
            pass
        # Mark tab-switch hide before any dock visibility changes so delayed signals
        # do not persist temporary hidden/default states.
        self._post_docks_hidden_for_tab_switch = True
        self._persist_hidden_layout_state_from_cache()
        self._suspend_panel_layout_persistence = True
        try:
            for key, dock in self._section_popups.items():
                dock.hide()
                # Detach post docks while tab is inactive so preprocessing and postprocessing
                # layouts cannot mutate each other in the shared main-window dock host.
                if host is not None:
                    try:
                        host.removeDockWidget(dock)
                    except Exception:
                        pass
                self._set_section_button_checked(key, False)
        finally:
            self._suspend_panel_layout_persistence = False

    def _ensure_plot_rows_visible(self) -> None:
        """Guarantee the plot area remains visible after tab switches and dock operations."""
        if hasattr(self, "_right_panel"):
            self._right_panel.setVisible(True)
        if not (self.plot_trace.isVisible() or self.row_heat.isVisible() or self.row_avg.isVisible()):
            self.combo_view_layout.blockSignals(True)
            self.combo_view_layout.setCurrentText("Standard")
            self.combo_view_layout.blockSignals(False)
            self._apply_view_layout()

    def showEvent(self, event: QtGui.QShowEvent) -> None:
        super().showEvent(event)
        self._setup_section_popups()
        if not self._section_popups:
            # Defer until the widget is fully attached to a main-window host.
            QtCore.QTimer.singleShot(0, self._setup_section_popups)
        if self._force_fixed_default_layout and self._section_popups:
            # Always enforce fixed docking on show to override any late floating restores.
            self.apply_fixed_default_layout()
            self._dock_layout_restored = True
        elif not self._dock_layout_restored and self._section_popups:
            self._restore_panel_layout_state()
            self._dock_layout_restored = True

        if not self._force_fixed_default_layout:
            self._apply_post_main_dock_snapshot_if_needed()
        self._enforce_only_post_docks_visible()
        self._ensure_plot_rows_visible()
        if self._force_fixed_default_layout:
            # Fixed mode ignores cached tab-switch floating/visibility state.
            self._post_docks_hidden_for_tab_switch = False
            self._post_section_visibility_before_hide.clear()
            self._post_section_state_before_hide.clear()
            return
        if not self._section_popups:
            return
        if not self._post_docks_hidden_for_tab_switch:
            return
        host = self._dock_host or self._dock_main_window()
        self._suspend_panel_layout_persistence = True
        try:
            for key, dock in self._section_popups.items():
                state = self._post_section_state_before_hide.get(key, {})
                visible = bool(state.get("visible", self._post_section_visibility_before_hide.get(key, False)))
                floating = bool(state.get("floating", dock.isFloating()))
                area = self._dock_area_from_settings(
                    state.get("area", _dock_area_to_int(QtCore.Qt.DockWidgetArea.RightDockWidgetArea, 2)),
                    QtCore.Qt.DockWidgetArea.RightDockWidgetArea,
                )
                geom = state.get("geometry")
                dock.blockSignals(True)
                try:
                    if floating:
                        dock.setFloating(True)
                        if isinstance(geom, QtCore.QByteArray) and not geom.isEmpty():
                            dock.restoreGeometry(geom)
                            self._section_popup_initialized.add(key)
                        if key not in self._section_popup_initialized or not self._is_popup_on_screen(dock):
                            self._position_section_popup(dock, key)
                            self._section_popup_initialized.add(key)
                    else:
                        if host is not None:
                            host.addDockWidget(area, dock)
                        dock.setFloating(False)
                        if isinstance(geom, QtCore.QByteArray) and not geom.isEmpty():
                            dock.restoreGeometry(geom)
                finally:
                    dock.blockSignals(False)
                if visible:
                    dock.show()
                    self._set_section_button_checked(key, True)
                    self._last_opened_section = key
                else:
                    dock.hide()
                    self._set_section_button_checked(key, False)
        finally:
            self._suspend_panel_layout_persistence = False
        self._post_docks_hidden_for_tab_switch = False
        self._post_section_visibility_before_hide.clear()
        self._post_section_state_before_hide.clear()
        self._enforce_only_post_docks_visible()
        self._ensure_plot_rows_visible()
        self._save_panel_layout_state()

    def _apply_post_main_dock_snapshot_if_needed(self) -> None:
        if self._post_snapshot_applied:
            return
        host = self._dock_host or self._dock_main_window()
        if host is None:
            return
        try:
            raw = self._settings.value(_POST_DOCK_STATE_KEY, None)
            state = self._to_qbytearray(raw)
            if state is not None and not state.isEmpty():
                if hasattr(host, "restoreDockSnapshotForTab"):
                    ok = bool(host.restoreDockSnapshotForTab("post", state))
                else:
                    ok = bool(host.restoreState(state, _DOCK_STATE_VERSION))
                if ok:
                    self._post_snapshot_applied = True
                    _LOG.info("Post dock snapshot applied successfully")
                else:
                    _LOG.warning("Post dock snapshot restore failed")
                self._sync_section_button_states_from_docks()
                self._enforce_only_post_docks_visible()
        except Exception:
            _LOG.exception("Post dock snapshot restore crashed")

    def _on_about_to_quit(self) -> None:
        self._app_closing = True
        self.persist_layout_state_snapshot()
        self._save_settings()

    def _enforce_only_post_docks_visible(self) -> None:
        """
        Ensure preprocessing docks cannot stay visible while postprocessing tab is active.
        """
        if not self.isVisible():
            return
        host = self._dock_host or self._dock_main_window()
        if host is None:
            return
        tabs = getattr(host, "tabs", None)
        if isinstance(tabs, QtWidgets.QTabWidget) and tabs.currentWidget() is not self:
            return
        for dock in host.findChildren(QtWidgets.QDockWidget):
            name = str(dock.objectName() or "")
            if name.startswith(_PRE_DOCK_PREFIX):
                dock.hide()

    def mark_app_closing(self) -> None:
        self._app_closing = True

    def set_force_fixed_default_layout(self, enabled: bool) -> None:
        self._force_fixed_default_layout = bool(enabled)

    def apply_fixed_default_layout(self) -> None:
        """
        Apply deterministic Post Processing docking:
        right-side tab stack with Setup active, and Export docked as bottom strip.
        """
        self._setup_section_popups()
        host = self._dock_host or self._dock_main_window()
        if host is None or not self._section_popups:
            return
        self._dock_host = host

        setup = self._section_popups.get("setup")
        ordered_right_keys = ["setup", "psth", "signal", "behavior"]
        export = self._section_popups.get("export")

        self._suspend_panel_layout_persistence = True
        try:
            # Reset previous split/tab topology before rebuilding the fixed stack.
            for key in ("setup", "psth", "signal", "behavior", "export"):
                dock = self._section_popups.get(key)
                if dock is None:
                    continue
                try:
                    host.removeDockWidget(dock)
                except Exception:
                    pass

            for key in ordered_right_keys:
                dock = self._section_popups.get(key)
                if dock is None:
                    continue
                dock.blockSignals(True)
                try:
                    dock.setFloating(False)
                    host.addDockWidget(QtCore.Qt.DockWidgetArea.RightDockWidgetArea, dock)
                    dock.show()
                finally:
                    dock.blockSignals(False)

            if export is not None:
                export.blockSignals(True)
                try:
                    export.setFloating(False)
                    host.addDockWidget(QtCore.Qt.DockWidgetArea.BottomDockWidgetArea, export)
                    export.show()
                finally:
                    export.blockSignals(False)

            if setup is not None:
                for key in ("psth", "signal", "behavior"):
                    dock = self._section_popups.get(key)
                    if dock is None:
                        continue
                    try:
                        host.tabifyDockWidget(setup, dock)
                    except Exception:
                        continue

            if setup is not None:
                setup.show()
                setup.raise_()
                setup.setFocus(QtCore.Qt.FocusReason.OtherFocusReason)
                self._last_opened_section = "setup"
            if export is not None:
                export.raise_()

            self._sync_section_button_states_from_docks()
            self._post_docks_hidden_for_tab_switch = False
            self._post_section_visibility_before_hide.clear()
            self._post_section_state_before_hide.clear()
        finally:
            self._suspend_panel_layout_persistence = False

        self._enforce_only_post_docks_visible()

    def ensure_section_popups_initialized(self) -> None:
        self._setup_section_popups()

    def get_section_dock_widgets(self) -> List[QtWidgets.QDockWidget]:
        self._setup_section_popups()
        return list(self._section_popups.values())

    def get_section_popup_keys(self) -> List[str]:
        self._setup_section_popups()
        return list(self._section_popups.keys())

    def mark_dock_layout_restored(self) -> None:
        self._dock_layout_restored = True


class ExportDialog(QtWidgets.QDialog):
    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Export Results")
        self.setModal(True)
        layout = QtWidgets.QVBoxLayout(self)

        self.cb_heatmap = QtWidgets.QCheckBox("Heatmap matrix")
        self.cb_avg = QtWidgets.QCheckBox("Average PSTH")
        self.cb_events = QtWidgets.QCheckBox("Event times")
        self.cb_durations = QtWidgets.QCheckBox("Event durations")
        self.cb_metrics = QtWidgets.QCheckBox("Metrics table")
        for cb in (self.cb_heatmap, self.cb_avg, self.cb_events, self.cb_durations, self.cb_metrics):
            cb.setChecked(True)
            layout.addWidget(cb)

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

    def choices(self) -> Dict[str, bool]:
        return {
            "heatmap": self.cb_heatmap.isChecked(),
            "avg": self.cb_avg.isChecked(),
            "events": self.cb_events.isChecked(),
            "durations": self.cb_durations.isChecked(),
            "metrics": self.cb_metrics.isChecked(),
        }


class StyleDialog(QtWidgets.QDialog):
    def __init__(self, style: Dict[str, object], parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Plot Styling")
        self.setModal(True)
        self._style = dict(style)

        layout = QtWidgets.QFormLayout(self)

        self.btn_trace = QtWidgets.QPushButton("Pick")
        self.btn_behavior = QtWidgets.QPushButton("Pick")
        self.btn_avg = QtWidgets.QPushButton("Pick")
        self.combo_cmap = QtWidgets.QComboBox()
        self.combo_cmap.addItems(["viridis", "plasma", "inferno", "magma", "cividis", "turbo", "gray"])
        if self._style.get("heatmap_cmap"):
            self.combo_cmap.setCurrentText(str(self._style.get("heatmap_cmap")))

        self.spin_hmin = QtWidgets.QDoubleSpinBox(); self.spin_hmin.setRange(-1e9, 1e9); self.spin_hmin.setDecimals(3)
        self.spin_hmax = QtWidgets.QDoubleSpinBox(); self.spin_hmax.setRange(-1e9, 1e9); self.spin_hmax.setDecimals(3)
        if self._style.get("heatmap_min") is not None:
            self.spin_hmin.setValue(float(self._style["heatmap_min"]))
        if self._style.get("heatmap_max") is not None:
            self.spin_hmax.setValue(float(self._style["heatmap_max"]))

        layout.addRow("Trace color", self.btn_trace)
        layout.addRow("Behavior color", self.btn_behavior)
        layout.addRow("Avg color", self.btn_avg)
        layout.addRow("Heatmap colormap", self.combo_cmap)
        layout.addRow("Heatmap min", self.spin_hmin)
        layout.addRow("Heatmap max", self.spin_hmax)

        row = QtWidgets.QHBoxLayout()
        row.addStretch(1)
        btn_ok = QtWidgets.QPushButton("OK")
        btn_cancel = QtWidgets.QPushButton("Cancel")
        btn_ok.setDefault(True)
        row.addWidget(btn_ok)
        row.addWidget(btn_cancel)
        layout.addRow(row)

        self.btn_trace.clicked.connect(lambda *_: self._pick_color("trace"))
        self.btn_behavior.clicked.connect(lambda *_: self._pick_color("behavior"))
        self.btn_avg.clicked.connect(lambda *_: self._pick_color("avg"))
        btn_ok.clicked.connect(self.accept)
        btn_cancel.clicked.connect(self.reject)

    def _pick_color(self, key: str) -> None:
        col = QtWidgets.QColorDialog.getColor(parent=self)
        if not col.isValid():
            return
        self._style[key] = (col.red(), col.green(), col.blue())

    def get_style(self) -> Dict[str, object]:
        self._style["heatmap_cmap"] = self.combo_cmap.currentText()
        self._style["heatmap_min"] = float(self.spin_hmin.value()) if self.spin_hmin.value() != 0.0 else None
        self._style["heatmap_max"] = float(self.spin_hmax.value()) if self.spin_hmax.value() != 0.0 else None
        return dict(self._style)
