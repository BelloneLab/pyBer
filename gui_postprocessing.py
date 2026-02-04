# gui_postprocessing.py
from __future__ import annotations

import os
import re
import json
from pathlib import Path
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PySide6 import QtCore, QtWidgets, QtGui
import pyqtgraph as pg
import h5py

from analysis_core import ProcessedTrial
from ethovision_process_gui import clean_sheet


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


def _load_behavior_csv(path: str) -> Dict[str, Any]:
    import pandas as pd
    df = pd.read_csv(path)
    time_col, behaviors = _binary_columns_from_df(df)
    return {"kind": "binary_columns", "time": np.asarray(df[time_col], float), "behaviors": behaviors}


def _load_behavior_ethovision(path: str, sheet_name: Optional[str] = None) -> Dict[str, Any]:
    if sheet_name is None:
        import pandas as pd
        xls = pd.ExcelFile(path, engine="openpyxl")
        sheet_name = xls.sheet_names[0] if xls.sheet_names else None
    if not sheet_name:
        return {"kind": "binary_columns", "time": np.array([], float), "behaviors": {}}
    df = clean_sheet(Path(path), sheet_name, interpolate=True)
    time_col, behaviors = _binary_columns_from_df(df)
    return {"kind": "binary_columns", "time": np.asarray(df[time_col], float), "behaviors": behaviors, "sheet": sheet_name}


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
        self._event_labels: List[pg.TextItem] = []
        self._event_regions: List[pg.LinearRegionItem] = []
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
        self._build_ui()
        self._restore_settings()

    def _build_ui(self) -> None:
        root = QtWidgets.QHBoxLayout(self)
        root.setContentsMargins(10, 10, 10, 10)
        root.setSpacing(10)

        # Left controls (scroll)
        left = QtWidgets.QWidget()
        lv = QtWidgets.QVBoxLayout(left)
        lv.setSpacing(10)

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

        self.btn_refresh_dio = QtWidgets.QPushButton("Refresh DIO list")
        self.btn_refresh_dio.setProperty("class", "compactSmall")
        self.btn_refresh_dio.setSizePolicy(QtWidgets.QSizePolicy.Policy.Ignored, QtWidgets.QSizePolicy.Policy.Fixed)

        vsrc.addWidget(self.tab_sources)
        vsrc.addWidget(self.btn_refresh_dio)

        grp_align = QtWidgets.QGroupBox("Behavior / Events")
        fal = QtWidgets.QFormLayout(grp_align)
        fal.setRowWrapPolicy(QtWidgets.QFormLayout.RowWrapPolicy.WrapLongRows)
        fal.setLabelAlignment(QtCore.Qt.AlignmentFlag.AlignLeft | QtCore.Qt.AlignmentFlag.AlignTop)

        self.combo_align = QtWidgets.QComboBox()
        self.combo_align.addItems(["DIO (from Doric)", "Behavior (CSV/XLSX)"])
        self.combo_align.setCurrentIndex(1)
        _compact_combo(self.combo_align, min_chars=6)

        self.combo_dio = QtWidgets.QComboBox()
        _compact_combo(self.combo_dio, min_chars=6)
        self.combo_dio_polarity = QtWidgets.QComboBox()
        self.combo_dio_polarity.addItems(["Event high (0→1)", "Event low (1→0)"])
        _compact_combo(self.combo_dio_polarity, min_chars=6)
        self.combo_dio_align = QtWidgets.QComboBox()
        self.combo_dio_align.addItems(["Align to onset", "Align to offset"])
        _compact_combo(self.combo_dio_align, min_chars=6)

        self.btn_load_beh = QtWidgets.QPushButton("Load behavior CSV/XLSX…")
        self.btn_load_beh.setProperty("class", "compactSmall")
        self.btn_load_beh.setSizePolicy(QtWidgets.QSizePolicy.Policy.Ignored, QtWidgets.QSizePolicy.Policy.Fixed)
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
        self.btn_move_up = QtWidgets.QPushButton("↑ Move Up")
        self.btn_move_down = QtWidgets.QPushButton("↓ Move Down")
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
        fal.addRow("DIO channel", self.combo_dio)
        fal.addRow("DIO polarity", self.combo_dio_polarity)
        fal.addRow("DIO align", self.combo_dio_align)
        fal.addRow(self.btn_load_beh)
        fal.addRow("Loaded files", self.lbl_beh)
        fal.addRow(files_layout)
        fal.addRow(lists_layout)
        fal.addRow(order_layout)

        # Legacy behavior controls (hidden by default but kept for compatibility)
        self.combo_behavior_name = QtWidgets.QComboBox()
        _compact_combo(self.combo_behavior_name, min_chars=6)
        self.combo_behavior_align = QtWidgets.QComboBox()
        self.combo_behavior_align.addItems(["Align to onset", "Align to offset", "Transition A→B"])
        _compact_combo(self.combo_behavior_align, min_chars=6)
        self.combo_behavior_from = QtWidgets.QComboBox()
        self.combo_behavior_to = QtWidgets.QComboBox()
        _compact_combo(self.combo_behavior_from, min_chars=6)
        _compact_combo(self.combo_behavior_to, min_chars=6)
        self.spin_transition_gap = QtWidgets.QDoubleSpinBox()
        self.spin_transition_gap.setRange(0, 60)
        self.spin_transition_gap.setValue(1.0)
        self.spin_transition_gap.setDecimals(2)

        # Hide legacy controls initially
        legacy_group = QtWidgets.QGroupBox("Legacy Behavior Selection (deprecated)")
        legacy_group.setCheckable(True)
        legacy_group.setChecked(False)
        legacy_layout = QtWidgets.QFormLayout(legacy_group)
        legacy_layout.addRow("Behavior name", self.combo_behavior_name)
        legacy_layout.addRow("Behavior align", self.combo_behavior_align)
        self.lbl_trans_from = QtWidgets.QLabel("Transition from")
        self.lbl_trans_to = QtWidgets.QLabel("Transition to")
        self.lbl_trans_gap = QtWidgets.QLabel("Transition gap (s)")
        legacy_layout.addRow(self.lbl_trans_from, self.combo_behavior_from)
        legacy_layout.addRow(self.lbl_trans_to, self.combo_behavior_to)
        legacy_layout.addRow(self.lbl_trans_gap, self.spin_transition_gap)

        fal.addRow(legacy_group)

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
            self.spin_event_start, self.spin_event_end, self.spin_dur_min, self.spin_dur_max,
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
        self.lbl_dur_min = QtWidgets.QLabel("Event duration min (s)")
        self.lbl_dur_max = QtWidgets.QLabel("Event duration max (s)")
        fopt.addRow(self.lbl_event_start, self.spin_event_start)
        fopt.addRow(self.lbl_event_end, self.spin_event_end)
        fopt.addRow(self.lbl_dur_min, self.spin_dur_min)
        fopt.addRow(self.lbl_dur_max, self.spin_dur_max)
        fopt.addRow("Gaussian smooth σ (s)", self.spin_smooth)
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

        lv.addWidget(grp_src)
        lv.addWidget(grp_align)
        lv.addWidget(grp_opt)
        lv.addWidget(self.btn_compute)
        lv.addWidget(self.btn_update)
        lv.addWidget(self.btn_export)
        lv.addWidget(self.btn_export_img)
        lv.addWidget(self.btn_style)
        lv.addWidget(self.btn_save_cfg)
        lv.addWidget(self.btn_load_cfg)
        lv.addStretch(1)

        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(left)

        # Right plots: trace preview + heatmap + avg
        right = QtWidgets.QWidget()
        self._right_panel = right
        rv = QtWidgets.QVBoxLayout(right)
        rv.setSpacing(10)

        self.plot_trace = pg.PlotWidget(title="Trace preview (events as vertical lines)")
        self.plot_heat = pg.PlotWidget(title="Heatmap (trials or recordings)")
        self.plot_dur = pg.PlotWidget(title="Event duration")
        self.plot_avg = pg.PlotWidget(title="Average PSTH ± SEM")
        self.plot_metrics = pg.PlotWidget(title="Metrics (pre vs post)")
        self.plot_global = pg.PlotWidget(title="Global metrics")

        for w in (self.plot_trace, self.plot_heat, self.plot_dur, self.plot_avg, self.plot_metrics, self.plot_global):
            _opt_plot(w)

        self.curve_trace = self.plot_trace.plot(pen=pg.mkPen(self._style["trace"], width=1.1))
        self.curve_behavior = self.plot_trace.plot(pen=pg.mkPen(self._style["behavior"], width=1.0))
        self.event_lines: List[pg.InfiniteLine] = []

        self.img = pg.ImageItem()
        self.plot_heat.addItem(self.img)
        self.plot_heat.setLabel("bottom", "Time (s)")
        self.plot_heat.setLabel("left", "Trials / Recordings")
        self.plot_dur.setLabel("bottom", "Duration (s)")
        self.plot_dur.setLabel("left", "Count")

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

        heat_row = QtWidgets.QHBoxLayout()
        heat_row.addWidget(self.plot_heat, stretch=4)
        heat_row.addWidget(self.plot_dur, stretch=1)

        avg_row = QtWidgets.QHBoxLayout()
        avg_row.addWidget(self.plot_avg, stretch=4)
        avg_row.addWidget(self.plot_metrics, stretch=1)
        avg_row.addWidget(self.plot_global, stretch=1)

        rv.addWidget(self.plot_trace, stretch=1)
        rv.addLayout(heat_row, stretch=2)
        rv.addLayout(avg_row, stretch=1)
        self.lbl_log = QtWidgets.QLabel("")
        self.lbl_log.setProperty("class", "hint")
        rv.addWidget(self.lbl_log)

        # Layout
        splitter = QtWidgets.QSplitter(QtCore.Qt.Orientation.Horizontal)
        splitter.addWidget(scroll)
        splitter.addWidget(right)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([260, 1240])

        root.addWidget(splitter)

        # Wiring
        self.btn_use_current.clicked.connect(self.requestCurrentProcessed.emit)
        self.btn_refresh_dio.clicked.connect(self.requestDioList.emit)
        self.btn_load_beh.clicked.connect(self._load_behavior_files)
        self.btn_load_processed.clicked.connect(self._load_processed_files)
        self.btn_load_processed_single.clicked.connect(self._load_processed_files_single)
        self.list_preprocessed.filesDropped.connect(self._on_preprocessed_files_dropped)
        self.list_preprocessed.orderChanged.connect(self._sync_processed_order_from_list)
        self.list_behaviors.filesDropped.connect(self._on_behavior_files_dropped)
        self.list_behaviors.orderChanged.connect(self._sync_behavior_order_from_list)
        self.btn_compute.clicked.connect(self._compute_psth)
        self.btn_update.clicked.connect(self._compute_psth)
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

        self.combo_align.currentIndexChanged.connect(self._update_align_ui)
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

        QtGui.QShortcut(QtGui.QKeySequence("Ctrl+S"), self, activated=self._export_results)
        QtGui.QShortcut(QtGui.QKeySequence("F5"), self, activated=self._compute_psth)

    # ---- bridge reception ----

    def set_current_source_label(self, filename: str, channel: str) -> None:
        self.lbl_current.setText(f"Current: {filename} [{channel}]")

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

    @QtCore.Slot(list)
    def receive_current_processed(self, processed_list: List[ProcessedTrial]) -> None:
        self._processed = processed_list or []
        # update trace preview with first entry
        self._refresh_behavior_list()
        self._set_resample_from_processed()
        self._update_trace_preview()

    def append_processed(self, processed_list: List[ProcessedTrial]) -> None:
        if not processed_list:
            return
        self._processed.extend(processed_list)
        self._refresh_behavior_list()
        self._set_resample_from_processed()
        self._update_trace_preview()

    @QtCore.Slot(list)
    def receive_dio_list(self, dio_list: List[str]) -> None:
        self.combo_dio.clear()
        for d in dio_list or []:
            self.combo_dio.addItem(d)

    @QtCore.Slot(str, str, object, object)
    def receive_dio_data(self, path: str, dio_name: str, t: Optional[np.ndarray], x: Optional[np.ndarray]) -> None:
        if t is None or x is None:
            return
        self._dio_cache[(path, dio_name)] = (np.asarray(t, float), np.asarray(x, float))

    def _load_behavior_paths(self, paths: List[str], replace: bool) -> None:
        if replace:
            self._behavior_sources.clear()
        for p in paths:
            stem = os.path.splitext(os.path.basename(p))[0]
            ext = os.path.splitext(p)[1].lower()
            try:
                if ext == ".csv":
                    info = _load_behavior_csv(p)
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
                    info = _load_behavior_ethovision(p, sheet_name=sheet)
                else:
                    continue
                self._behavior_sources[stem] = info
            except Exception as exc:
                QtWidgets.QMessageBox.warning(
                    self,
                    "Behavior load failed",
                    f"Could not load {os.path.basename(p)}:\n{exc}",
                )
                continue
        self.lbl_beh.setText(f"{len(self._behavior_sources)} file(s) loaded")

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
        use_dio = self.combo_align.currentText().startswith("DIO")
        for w in (self.combo_dio, self.combo_dio_polarity, self.combo_dio_align):
            w.setEnabled(use_dio)

        use_beh = not use_dio
        self.btn_load_beh.setEnabled(use_beh)
        self.combo_behavior_name.setEnabled(use_beh)
        self.combo_behavior_align.setEnabled(use_beh)

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

    def _update_event_filter_enabled(self) -> None:
        enabled = self.cb_filter_events.isChecked()
        for w in (self.spin_event_start, self.spin_event_end, self.spin_dur_min, self.spin_dur_max):
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
        self._save_settings()

    def _toggle_filter_panel(self, hide: bool) -> None:
        self.btn_hide_filters.setText("Show" if hide else "Hide")
        for w in (
            self.lbl_event_start,
            self.lbl_event_end,
            self.lbl_dur_min,
            self.lbl_dur_max,
            self.spin_event_start,
            self.spin_event_end,
            self.spin_dur_min,
            self.spin_dur_max,
        ):
            w.setVisible(not hide)

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

        header = [h.strip().lower() for h in rows[0]]
        has_header = "time" in header and "output" in header
        data_rows = rows[1:] if has_header else rows
        time = []
        output = []
        dio = []
        has_dio = has_header and "dio" in header

        for r in data_rows:
            if len(r) < 2:
                continue
            try:
                tval = float(r[0])
                oval = float(r[1])
            except Exception:
                continue
            time.append(tval)
            output.append(oval)
            if has_dio and len(r) > 2:
                try:
                    dio.append(float(r[2]))
                except Exception:
                    dio.append(np.nan)

        if not time:
            return None

        t = np.asarray(time, float)
        out = np.asarray(output, float)
        raw = np.full_like(t, np.nan, dtype=float)
        dio_arr = np.asarray(dio, float) if has_dio and len(dio) == len(time) else None

        return ProcessedTrial(
            path=path,
            channel_id="import",
            time=t,
            raw_signal=raw,
            raw_reference=raw.copy(),
            dio=dio_arr,
            dio_name="",
            sig_f=None,
            ref_f=None,
            baseline_sig=None,
            baseline_ref=None,
            output=out,
            output_label="Imported CSV",
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
                if "time" not in g or "output" not in g:
                    return None
                t = np.asarray(g["time"][()], float)
                out = np.asarray(g["output"][()], float)
                raw_sig = np.asarray(g["raw_465"][()], float) if "raw_465" in g else np.full_like(t, np.nan)
                raw_ref = np.asarray(g["raw_405"][()], float) if "raw_405" in g else np.full_like(t, np.nan)
                dio = np.asarray(g["dio"][()], float) if "dio" in g else None
                dio_name = str(g.attrs.get("dio_name", "")) if hasattr(g, "attrs") else ""
                output_label = str(g.attrs.get("output_label", "Imported H5")) if hasattr(g, "attrs") else "Imported H5"
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
            artifact_regions_sec=None,
            fs_actual=fs_actual,
            fs_target=fs_target,
            fs_used=fs_used,
        )

    def _refresh_behavior_list(self) -> None:
        self.combo_behavior_name.clear()
        if not self._behavior_sources:
            return
        any_info = next(iter(self._behavior_sources.values()))
        behaviors = sorted(list((any_info.get("behaviors") or {}).keys()))
        for name in behaviors:
            self.combo_behavior_name.addItem(name)
        self.combo_behavior_from.clear()
        self.combo_behavior_to.clear()
        for name in behaviors:
            self.combo_behavior_from.addItem(name)
            self.combo_behavior_to.addItem(name)

        # Update the lists with numbered items
        self._update_file_lists()

    def _update_file_lists(self) -> None:
        """Update the preprocessed files and behaviors lists with numbered entries."""
        self.list_preprocessed.clear()
        for i, proc in enumerate(self._processed, 1):
            filename = os.path.splitext(os.path.basename(proc.path))[0]
            # Remove _AIN01/_AIN02 suffix for matching
            filename_clean = filename.replace('_AIN01', '').replace('_AIN02', '')
            item = QtWidgets.QListWidgetItem(f"{i}. {filename_clean}")
            item.setData(QtCore.Qt.ItemDataRole.UserRole, id(proc))
            self.list_preprocessed.addItem(item)

        self.list_behaviors.clear()
        for i, (stem, _) in enumerate(self._behavior_sources.items(), 1):
            # Remove _AIN01/_AIN02 suffix for matching
            stem_clean = stem.replace('_AIN01', '').replace('_AIN02', '')
            item = QtWidgets.QListWidgetItem(f"{i}. {stem_clean}")
            item.setData(QtCore.Qt.ItemDataRole.UserRole, stem)
            self.list_behaviors.addItem(item)

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
            filename_clean = filename.replace('_AIN01', '').replace('_AIN02', '')
            proc_names.append(filename_clean)

        beh_names = []
        for stem in self._behavior_sources.keys():
            stem_clean = stem.replace('_AIN01', '').replace('_AIN02', '')
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
            proc_clean = os.path.splitext(os.path.basename(proc.path))[0].replace('_AIN01', '').replace('_AIN02', '')
            for stem, data in self._behavior_sources.items():
                stem_clean = stem.replace('_AIN01', '').replace('_AIN02', '')
                if stem_clean == proc_clean:
                    new_behavior_order.append((stem, data))
                    break

        # Fill in any unmatched behavior files at the end
        for stem, data in self._behavior_sources.items():
            if stem not in [s for s, _ in new_behavior_order]:
                new_behavior_order.append((stem, data))

        self._behavior_sources = dict(new_behavior_order)
        self._update_file_lists()

    def _remove_selected_preprocessed(self) -> None:
        selected = self.list_preprocessed.selectedItems()
        if not selected:
            return
        rows = sorted({self.list_preprocessed.row(item) for item in selected}, reverse=True)
        for row in rows:
            if 0 <= row < len(self._processed):
                del self._processed[row]
        self._update_file_lists()

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

    # ---- PSTH compute ----

    def _match_behavior_source(self, proc: ProcessedTrial) -> Optional[Dict[str, Any]]:
        stem = os.path.splitext(os.path.basename(proc.path))[0]
        info = self._behavior_sources.get(stem, None)
        if info is None:
            stem_clean = stem.replace('_AIN01', '').replace('_AIN02', '')
            for key, val in self._behavior_sources.items():
                key_clean = key.replace('_AIN01', '').replace('_AIN02', '')
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

    def _get_events_for_proc(self, proc: ProcessedTrial) -> Tuple[np.ndarray, np.ndarray]:
        align = self.combo_align.currentText()

        if align.startswith("DIO"):
            dio_name = self.combo_dio.currentText().strip()
            if not dio_name and proc.dio is None:
                return np.array([], float), np.array([], float)

            if proc.dio is not None and proc.time is not None:
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
            sig = np.asarray(x, float)
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
        t = np.asarray(info.get("time", np.array([], float)), float)
        behaviors = info.get("behaviors") or {}
        if t.size == 0 or not behaviors:
            return np.array([], float), np.array([], float)

        align_mode = self.combo_behavior_align.currentText()
        if align_mode.startswith("Transition"):
            beh_a = self.combo_behavior_from.currentText().strip()
            beh_b = self.combo_behavior_to.currentText().strip()
            if beh_a not in behaviors or beh_b not in behaviors:
                return np.array([], float), np.array([], float)
            on_a, off_a, _ = _extract_onsets_offsets(t, behaviors[beh_a], threshold=0.5)
            on_b, _, dur_b = _extract_onsets_offsets(t, behaviors[beh_b], threshold=0.5)
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
        if beh not in behaviors:
            return np.array([], float), np.array([], float)
        on, off, dur = _extract_onsets_offsets(t, behaviors[beh], threshold=0.5)
        if align_mode.endswith("offset"):
            return off, dur
        return on, dur

    def _filter_events(self, times: np.ndarray, durations: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if not self.cb_filter_events.isChecked():
            return np.asarray(times, float), np.asarray(durations, float)
        times = np.asarray(times, float)
        durations = np.asarray(durations, float) if durations is not None else np.array([], float)
        if durations.size != times.size:
            durations = np.full_like(times, np.nan, dtype=float)
        if times.size == 0:
            return times, durations

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
            for ln in self.event_lines:
                self.plot_trace.removeItem(ln)
            self.event_lines = []
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
        t = np.asarray(info.get("time", np.array([], float)), float)
        behaviors = info.get("behaviors") or {}
        beh = self.combo_behavior_name.currentText().strip()
        if not beh and behaviors:
            beh = next(iter(behaviors.keys()))
        if beh not in behaviors or t.size == 0:
            self.curve_behavior.setData([], [])
            return
        b = np.asarray(behaviors[beh], float)
        # Resample to processed time for overlay
        t_proc = np.asarray(proc.time, float)
        if t_proc.size == 0:
            self.curve_behavior.setData([], [])
            return
        b_interp = np.interp(t_proc, t, b)
        self.curve_behavior.setData(t_proc, b_interp, connect="finite", skipFiniteCheck=True)

    def _compute_psth(self) -> None:
        if not self._processed:
            self.lbl_log.setText("No processed data loaded.")
            self._last_global_metrics = None
            if hasattr(self, "lbl_global_metrics"):
                self.lbl_global_metrics.setText("Global metrics: -")
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
            total_events = 0
            tvec = None

            for proc in self._processed:
                ev, dur = self._get_events_for_proc(proc)
                ev, dur = self._filter_events(ev, dur)
                if ev.size == 0:
                    continue
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
                self.lbl_log.setText("No events found for the current alignment.")
                return

            mat_events = np.vstack(mats)
            if group_mode:
                if not animal_rows:
                    self.lbl_log.setText("No events found for the current alignment.")
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
            self._last_events = None
            self._last_durations = dur_all
            if group_mode:
                self.lbl_log.setText(f"Computed PSTH for {total_events} event(s) across {mat_display.shape[0]} animal(s).")
            else:
                self.lbl_log.setText(f"Computed PSTH for {total_events} event(s).")
            self._update_metric_regions()
            self._save_settings()
        except Exception as e:
            self.lbl_log.setText(f"Post-processing error: {e}")

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
            "style": dict(self._style),
        }

    def _apply_settings(self, data: Dict[str, object]) -> None:
        def _set_combo(combo: QtWidgets.QComboBox, val: object) -> None:
            if val is None:
                return
            idx = combo.findText(str(val))
            if idx >= 0:
                combo.setCurrentIndex(idx)

        _set_combo(self.combo_align, data.get("align"))
        _set_combo(self.combo_dio, data.get("dio_channel"))
        _set_combo(self.combo_dio_polarity, data.get("dio_polarity"))
        _set_combo(self.combo_dio_align, data.get("dio_align"))
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

    def _save_settings(self) -> None:
        try:
            data = self._collect_settings()
            self._settings.setValue("postprocess_json", json.dumps(data))
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

    def _export_results(self) -> None:
        if self._last_mat is None or self._last_tvec is None:
            return
        dlg = ExportDialog(self)
        if dlg.exec() != QtWidgets.QDialog.DialogCode.Accepted:
            return
        choices = dlg.choices()
        start_dir = self._settings.value("postprocess_last_dir", os.getcwd(), type=str)
        out_dir = QtWidgets.QFileDialog.getExistingDirectory(self, "Select export folder", start_dir)
        if not out_dir:
            return
        try:
            self._settings.setValue("postprocess_last_dir", out_dir)
        except Exception:
            pass
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
        if choices.get("events") and self._last_events is not None:
            np.savetxt(os.path.join(out_dir, f"{prefix}_events.csv"), self._last_events, delimiter=",")
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
        start_dir = self._settings.value("postprocess_last_dir", os.getcwd(), type=str)
        out_dir = QtWidgets.QFileDialog.getExistingDirectory(self, "Select export folder", start_dir)
        if not out_dir:
            return
        try:
            self._settings.setValue("postprocess_last_dir", out_dir)
        except Exception:
            pass
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
        }
        for name, widget in targets.items():
            try:
                pix = widget.grab()
                pix.save(os.path.join(out_dir, f"{prefix}_{name}.png"))
            except Exception:
                pass


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
