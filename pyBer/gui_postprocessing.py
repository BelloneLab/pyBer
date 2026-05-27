# gui_postprocessing.py
from __future__ import annotations

import os
import re
import json
import copy
import logging
from pathlib import Path
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
from PySide6 import QtCore, QtWidgets, QtGui
import pyqtgraph as pg
from pyqtgraph.dockarea import DockArea, Dock
import h5py

from analysis_core import ProcessedTrial, coerce_time_value
from ethovision_process_gui import clean_sheet
from time_sync import SyncResult, align_timebase, extract_sync_events
from temporal_modeling import TemporalModelingWidget
from onboarding import (
    PanelHeader as _PyberPanelHeader,
    Subsection as _PyberSubsection,
    FooterActions as _PyberFooterActions,
    InlineStatus as _PyberInlineStatus,
)

_DOCK_STATE_VERSION = 3
_POST_DOCK_STATE_KEY = "post_main_dock_state_v4"
_POST_DOCKAREA_STATE_KEY = "post_dockarea_state_v1"
_POST_DOCKAREA_VISIBLE_KEY = "post_dockarea_visible_v1"
_POST_DOCKAREA_ACTIVE_KEY = "post_dockarea_active_v1"
_POST_DOCK_PREFIX = "post."
_PRE_DOCK_PREFIX = "pre."
_BEHAVIOR_PARSE_BINARY = "binary_columns"
_BEHAVIOR_PARSE_TIMESTAMPS = "timestamp_columns"
_FIXED_POST_RIGHT_SECTIONS = frozenset({"setup", "spatial", "psth", "export", "temporal"})
_FIXED_POST_VISIBLE_SECTIONS = frozenset({"setup", "spatial", "psth", "export", "temporal"})
_FIXED_POST_RIGHT_TAB_ORDER = ("setup", "psth", "spatial", "temporal", "export")
_POST_RIGHT_PANEL_MIN_WIDTH = 420
_FIXED_POST_RIGHT_TAB_TITLES: Dict[str, str] = {
    "setup": "Setup",
    "psth": "PSTH",
    "spatial": "Spatial",
    "temporal": "Modeling",
    "export": "Export",
}
_USE_PG_DOCKAREA_POST_LAYOUT = True
_LOG = logging.getLogger(__name__)


def _opt_plot(w: pg.PlotWidget) -> None:
    w.setMenuEnabled(True)
    w.showGrid(x=True, y=True, alpha=0.25)
    w.setMouseEnabled(x=True, y=True)
    pi = w.getPlotItem()
    pi.getViewBox().setMouseMode(pg.ViewBox.PanMode)
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

def _detect_time_column(df, fallback_to_first: bool = False) -> Optional[str]:
    for c in df.columns:
        if str(c).strip().lower() in {"time", "trial time", "recording time"}:
            return str(c)
    if fallback_to_first and len(df.columns):
        return str(df.columns[0])
    return None


def _numeric_column_array(df, col_name: str) -> np.ndarray:
    import pandas as pd

    col_key = None
    for c in df.columns:
        if str(c) == str(col_name):
            col_key = c
            break
    if col_key is None:
        return np.array([], float)
    vals = pd.to_numeric(df[col_key], errors="coerce")
    if vals.isna().all():
        vals = df[col_key].apply(lambda v: coerce_time_value(str(v)))
    return np.asarray(vals, float)


def _generated_time_array(n_rows: int, fps: float) -> np.ndarray:
    n = int(max(0, n_rows))
    if n <= 0 or not np.isfinite(float(fps)) or float(fps) <= 0:
        return np.array([], float)
    return np.arange(n, dtype=float) / float(fps)


def _binary_columns_from_df(df) -> Tuple[str, Dict[str, np.ndarray]]:
    time_col = _detect_time_column(df, fallback_to_first=False)

    behaviors: Dict[str, np.ndarray] = {}
    for c in df.columns:
        name = str(c)
        if time_col and name == time_col:
            continue
        arr = _numeric_column_array(df, name)
        if arr.size == 0:
            continue
        finite = arr[np.isfinite(arr)]
        if finite.size == 0:
            continue
        uniq = np.unique(finite)
        if np.all(np.isin(uniq, [0.0, 1.0])):
            behaviors[name] = arr.astype(float)
    return time_col, behaviors


def _trajectory_columns_from_df(df, time_col: Optional[str]) -> Dict[str, np.ndarray]:
    trajectory: Dict[str, np.ndarray] = {}
    for c in df.columns:
        name = str(c).strip()
        if not name:
            continue
        if time_col and name == time_col:
            continue
        arr = _numeric_column_array(df, name)
        if arr.size == 0:
            continue
        finite = arr[np.isfinite(arr)]
        if finite.size < 8:
            continue
        uniq = np.unique(finite)
        # Skip classic binary behavior columns from trajectory candidates.
        if uniq.size <= 2 and np.all(np.isin(uniq, [0.0, 1.0])):
            continue
        trajectory[name] = arr
    return trajectory


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


def _load_behavior_csv(path: str, parse_mode: str = _BEHAVIOR_PARSE_BINARY, fps: float = 0.0) -> Dict[str, Any]:
    import pandas as pd

    df = pd.read_csv(path)
    row_count = int(len(df.index))
    time_col = _detect_time_column(df)
    trajectory = _trajectory_columns_from_df(df, time_col=time_col)
    trajectory_time = _numeric_column_array(df, time_col) if time_col else np.array([], float)
    if str(parse_mode) == _BEHAVIOR_PARSE_TIMESTAMPS:
        return {
            "kind": _BEHAVIOR_PARSE_TIMESTAMPS,
            "time": np.array([], float),
            "behaviors": _timestamp_columns_from_df(df),
            "trajectory": trajectory,
            "trajectory_time": trajectory_time if trajectory_time.size else _generated_time_array(row_count, fps),
            "trajectory_time_col": time_col or "",
            "row_count": row_count,
            "needs_generated_time": bool(not time_col),
        }
    time_col, behaviors = _binary_columns_from_df(df)
    time = _numeric_column_array(df, time_col) if time_col else _generated_time_array(row_count, fps)
    return {
        "kind": _BEHAVIOR_PARSE_BINARY,
        "time": time,
        "behaviors": behaviors,
        "trajectory": trajectory,
        "trajectory_time": trajectory_time if trajectory_time.size else (time if time.size else _generated_time_array(row_count, fps)),
        "trajectory_time_col": time_col or "",
        "row_count": row_count,
        "needs_generated_time": bool(not time_col),
    }


def _load_behavior_ethovision(
    path: str,
    sheet_name: Optional[str] = None,
    parse_mode: str = _BEHAVIOR_PARSE_BINARY,
    fps: float = 0.0,
) -> Dict[str, Any]:
    import pandas as pd

    if sheet_name is None:
        xls = pd.ExcelFile(path, engine="openpyxl")
        sheet_name = xls.sheet_names[0] if xls.sheet_names else None
    if not sheet_name:
        return {
            "kind": _BEHAVIOR_PARSE_BINARY,
            "time": np.array([], float),
            "behaviors": {},
            "trajectory": {},
            "trajectory_time": np.array([], float),
            "trajectory_time_col": "",
            "row_count": 0,
            "needs_generated_time": False,
        }
    if str(parse_mode) == _BEHAVIOR_PARSE_TIMESTAMPS:
        df = pd.read_excel(path, sheet_name=sheet_name, engine="openpyxl")
        time_col = _detect_time_column(df)
        row_count = int(len(df.index))
        return {
            "kind": _BEHAVIOR_PARSE_TIMESTAMPS,
            "time": np.array([], float),
            "behaviors": _timestamp_columns_from_df(df),
            "trajectory": _trajectory_columns_from_df(df, time_col=time_col),
            "trajectory_time": _numeric_column_array(df, time_col) if time_col else _generated_time_array(row_count, fps),
            "trajectory_time_col": time_col or "",
            "sheet": sheet_name,
            "row_count": row_count,
            "needs_generated_time": bool(not time_col),
        }
    df = clean_sheet(Path(path), sheet_name, interpolate=True)
    row_count = int(len(df.index))
    time_col, behaviors = _binary_columns_from_df(df)
    time = _numeric_column_array(df, time_col) if time_col else _generated_time_array(row_count, fps)
    return {
        "kind": _BEHAVIOR_PARSE_BINARY,
        "time": time,
        "behaviors": behaviors,
        "trajectory": _trajectory_columns_from_df(df, time_col=time_col),
        "trajectory_time": _numeric_column_array(df, time_col) if time_col else _generated_time_array(row_count, fps),
        "trajectory_time_col": time_col or "",
        "sheet": sheet_name,
        "row_count": row_count,
        "needs_generated_time": bool(not time_col),
    }


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
    helpRequested = QtCore.Signal()

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
        self._last_spatial_occupancy_map: Optional[np.ndarray] = None
        self._last_spatial_activity_map: Optional[np.ndarray] = None
        self._last_spatial_velocity_map: Optional[np.ndarray] = None
        self._last_spatial_extent: Optional[Tuple[float, float, float, float]] = None
        self._last_event_rows: List[Dict[str, object]] = []
        self._last_display_labels: List[str] = []
        self._last_psth_display_level: str = "trials"
        self._psth_excluded_files: Dict[str, Dict[str, object]] = {}
        self._sync_results_by_file: Dict[str, Dict[str, object]] = {}
        self._last_sync_preview: Optional[SyncResult] = None
        self._known_dio_channels: List[str] = []
        # Per-file / group data for Individual vs Group visual modes
        self._per_file_mats: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}  # file_id -> (tvec, mat)
        self._per_file_labels: Dict[str, List[str]] = {}  # file_id -> trial labels
        self._group_mat: Optional[np.ndarray] = None
        self._group_tvec: Optional[np.ndarray] = None
        self._group_labels: List[str] = []
        self._group_trial_mat: Optional[np.ndarray] = None
        self._group_trial_tvec: Optional[np.ndarray] = None
        self._group_trial_labels: List[str] = []
        self._all_file_ids: List[str] = []
        self.last_signal_events: Optional[Dict[str, object]] = None
        self.last_behavior_analysis: Optional[Dict[str, object]] = None
        self._trace_preview_events = np.array([], float)
        self._trace_preview_durations = np.array([], float)
        self._trace_preview_y_bounds: Tuple[float, float] = (-1.0, 1.0)
        self._event_labels: List[pg.TextItem] = []
        self._event_regions: List[pg.LinearRegionItem] = []
        self._signal_peak_lines: List[pg.InfiniteLine] = []
        self._signal_noise_items: List[object] = []
        self._pre_region: Optional[pg.LinearRegionItem] = None
        self._post_region: Optional[pg.LinearRegionItem] = None
        self._settings = QtCore.QSettings("FiberPhotometryApp", "DoricProcessor")
        self._style = {
            "trace": (90, 190, 255),
            "behavior": (220, 180, 80),
            "avg": (90, 190, 255),
            "sem_edge": (152, 201, 143),
            "sem_fill": (188, 230, 178, 96),
            "plot_bg": (36, 42, 52),
            "grid_enabled": True,
            "grid_alpha": 0.25,
            "heatmap_cmap": "viridis",
            "heatmap_min": None,
            "heatmap_max": None,
            "heatmap_levels_manual": False,
        }
        self._section_popups: Dict[str, QtWidgets.QDockWidget] = {}
        self._section_scroll_hosts: Dict[str, QtWidgets.QScrollArea] = {}
        self._section_buttons: Dict[str, QtWidgets.QPushButton] = {}
        self._use_pg_dockarea_layout: bool = bool(_USE_PG_DOCKAREA_POST_LAYOUT)
        self._dockarea: Optional[DockArea] = None
        self._dockarea_docks: Dict[str, Dock] = {}
        self._dockarea_splitter: Optional[QtWidgets.QSplitter] = None
        self._fixed_right_tab_widget: Optional[QtWidgets.QTabWidget] = None
        self._section_popup_initialized: set[str] = set()
        self._is_restoring_settings: bool = True
        self._settings_save_timer = QtCore.QTimer(self)
        self._settings_save_timer.setSingleShot(True)
        self._settings_save_timer.setInterval(250)
        self._settings_save_timer.timeout.connect(self._save_settings)
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
        self._app_theme_mode: str = "dark"
        self._applying_fixed_default_layout: bool = False
        self._pending_fixed_layout_retry: bool = False
        self._pending_project_recompute_from_current: bool = False
        self._project_dirty: bool = False
        self._project_clean_key: str = ""
        self._autosave_restoring: bool = False
        self._project_recovered_from_autosave: bool = False
        self._suppress_heatmap_level_store: bool = False
        self._history_undo: List[Dict[str, object]] = []
        self._history_redo: List[Dict[str, object]] = []
        self._history_current: Optional[Dict[str, object]] = None
        self._history_key: str = ""
        self._history_restoring: bool = False
        self._history_limit: int = 60
        try:
            self._build_ui()
            self._restore_settings()
        finally:
            self._is_restoring_settings = False
        self._panel_layout_persistence_ready = True
        # Parameter drawer is hidden until the user explicitly
        # clicks one of the rail panel buttons.
        self._force_hide_post_drawer_initially()
        app = QtWidgets.QApplication.instance()
        if app is not None:
            app.aboutToQuit.connect(self._on_about_to_quit)
        QtCore.QTimer.singleShot(0, self._restore_project_autosave_if_needed)
        QtCore.QTimer.singleShot(0, self._reset_history_snapshot)
        QtCore.QTimer.singleShot(0, self._mark_project_clean)

    def _build_ui(self) -> None:
        root = QtWidgets.QVBoxLayout(self)
        root.setContentsMargins(10, 10, 10, 10)
        root.setSpacing(8)

        grp_src = QtWidgets.QGroupBox("Signal Source")
        grp_src.setSizePolicy(QtWidgets.QSizePolicy.Policy.Preferred, QtWidgets.QSizePolicy.Policy.Maximum)
        vsrc = QtWidgets.QVBoxLayout(grp_src)

        self.tab_sources = QtWidgets.QTabWidget()
        self.tab_sources.setObjectName("postSourceTabs")
        self.tab_sources.setSizePolicy(QtWidgets.QSizePolicy.Policy.Expanding, QtWidgets.QSizePolicy.Policy.Maximum)
        self.tab_sources.setStyleSheet(
            "QTabWidget::pane { border: 0px; background: transparent; padding: 0px; }"
            "QTabBar::tab { margin-right: 0px; }"
        )
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

        group_layout = QtWidgets.QVBoxLayout(tab_group)
        self.btn_load_processed = QtWidgets.QPushButton("Load processed files (CSV/H5)")
        self.btn_load_processed.setProperty("class", "compactSmall")
        self.btn_load_processed.setSizePolicy(QtWidgets.QSizePolicy.Policy.Ignored, QtWidgets.QSizePolicy.Policy.Fixed)
        self.lbl_group = QtWidgets.QLabel("(none)")
        self.lbl_group.setProperty("class", "hint")
        group_layout.addWidget(self.btn_load_processed)
        group_layout.addWidget(self.lbl_group)

        self.btn_refresh_dio = QtWidgets.QPushButton("Refresh A/D channel list")
        self.btn_refresh_dio.setProperty("class", "compactSmall")
        self.btn_refresh_dio.setSizePolicy(QtWidgets.QSizePolicy.Policy.Ignored, QtWidgets.QSizePolicy.Policy.Fixed)

        vsrc.addWidget(self.tab_sources)
        vsrc.addWidget(self.btn_refresh_dio)

        grp_align = QtWidgets.QGroupBox("Behavior / Events")
        grp_align.setSizePolicy(QtWidgets.QSizePolicy.Policy.Preferred, QtWidgets.QSizePolicy.Policy.Expanding)
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
        self.lbl_dio_channel = QtWidgets.QLabel("A/D channel")
        self.lbl_dio_polarity = QtWidgets.QLabel("A/D polarity")
        self.lbl_dio_align = QtWidgets.QLabel("A/D align")

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
        self.grp_behavior_time = QtWidgets.QGroupBox("Time")
        time_layout = QtWidgets.QHBoxLayout(self.grp_behavior_time)
        time_layout.setContentsMargins(6, 6, 6, 6)
        time_layout.setSpacing(6)
        self.lbl_behavior_time_hint = QtWidgets.QLabel("No time column detected. Generate time from FPS.")
        self.lbl_behavior_time_hint.setProperty("class", "hint")
        self.spin_behavior_fps = QtWidgets.QDoubleSpinBox()
        self.spin_behavior_fps.setRange(0.01, 10000.0)
        self.spin_behavior_fps.setDecimals(3)
        self.spin_behavior_fps.setValue(30.0)
        self.spin_behavior_fps.setSuffix(" fps")
        self.spin_behavior_fps.setMinimumWidth(90)
        self.btn_apply_behavior_time = QtWidgets.QPushButton("Apply FPS")
        self.btn_apply_behavior_time.setProperty("class", "compactSmall")
        time_layout.addWidget(self.lbl_behavior_time_hint, stretch=1)
        time_layout.addWidget(self.spin_behavior_fps, stretch=0)
        time_layout.addWidget(self.btn_apply_behavior_time, stretch=0)
        self.grp_behavior_time.setVisible(False)
        self.lbl_beh = QtWidgets.QLabel("(none)")
        self.lbl_beh.setProperty("class", "hint")

        # Preprocessed files list
        self.list_preprocessed = FileDropList()
        self.list_preprocessed.setMinimumHeight(180)
        self.list_preprocessed.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Expanding,
            QtWidgets.QSizePolicy.Policy.Expanding,
        )
        self.list_preprocessed.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.MultiSelection)

        # Behaviors list
        self.list_behaviors = FileDropList()
        self.list_behaviors.setMinimumHeight(180)
        self.list_behaviors.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Expanding,
            QtWidgets.QSizePolicy.Policy.Expanding,
        )
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
        pre_col.setStretch(0, 1)
        beh_col.setStretch(0, 1)
        lists_layout.addLayout(pre_col)
        lists_layout.addLayout(beh_col)

        fal.addRow("Align source", self.combo_align)
        fal.addRow(self.lbl_dio_channel, self.combo_dio)
        fal.addRow(self.lbl_dio_polarity, self.combo_dio_polarity)
        fal.addRow(self.lbl_dio_align, self.combo_dio_align)
        fal.addRow(self.lbl_behavior_file_type, self.combo_behavior_file_type)
        fal.addRow(self.grp_behavior_time)
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


        # ── Shared QSS for PSTH subsection headers ──
        _psth_section_qss = ""

        # ── Widget creation (unchanged logic, reordered for sections) ──
        self.spin_pre = QtWidgets.QDoubleSpinBox(); self.spin_pre.setRange(0.1, 60); self.spin_pre.setValue(2.0); self.spin_pre.setDecimals(2)
        self.spin_post = QtWidgets.QDoubleSpinBox(); self.spin_post.setRange(0.1, 120); self.spin_post.setValue(5.0); self.spin_post.setDecimals(2)
        self.spin_b0 = QtWidgets.QDoubleSpinBox(); self.spin_b0.setRange(-60, 0); self.spin_b0.setValue(-1.0); self.spin_b0.setDecimals(2)
        self.spin_b1 = QtWidgets.QDoubleSpinBox(); self.spin_b1.setRange(-60, 0); self.spin_b1.setValue(0.0); self.spin_b1.setDecimals(2)
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
        self.cb_exclude_low_event_animals = QtWidgets.QCheckBox("Exclude animals below minimum")
        self.cb_exclude_low_event_animals.setChecked(False)
        self.cb_exclude_low_event_animals.setToolTip(
            "When Group mode is used, omit animals/files with fewer accepted events than the threshold below."
        )
        self.spin_min_events_per_animal = QtWidgets.QSpinBox()
        self.spin_min_events_per_animal.setRange(1, 1000000)
        self.spin_min_events_per_animal.setValue(5)
        self.spin_min_events_per_animal.setToolTip(
            "Minimum number of accepted events required for an animal/file to enter grouped PSTH analysis."
        )
        self.cb_group_keep_trials = QtWidgets.QCheckBox("Keep trial rows in Group view")
        self.cb_group_keep_trials.setToolTip(
            "When Group is selected, show every accepted trial/event row instead of one averaged row per animal."
        )

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

        self.cb_global_metrics = QtWidgets.QCheckBox("Enable global metrics")
        self.cb_global_metrics.setChecked(True)
        self.spin_global_start = QtWidgets.QDoubleSpinBox(); self.spin_global_start.setRange(-1e6, 1e6); self.spin_global_start.setValue(0.0); self.spin_global_start.setDecimals(2)
        self.spin_global_end = QtWidgets.QDoubleSpinBox(); self.spin_global_end.setRange(-1e6, 1e6); self.spin_global_end.setValue(0.0); self.spin_global_end.setDecimals(2)
        self.cb_global_amp = QtWidgets.QCheckBox("Peak amplitude")
        self.cb_global_amp.setChecked(True)
        self.cb_global_freq = QtWidgets.QCheckBox("Transient frequency")
        self.cb_global_freq.setChecked(True)
        self.lbl_global_metrics = QtWidgets.QLabel("Global metrics: -")
        self.lbl_global_metrics.setProperty("class", "hint")

        for w in (
            self.spin_pre, self.spin_post, self.spin_b0, self.spin_b1,
            self.spin_resample, self.spin_smooth,
            self.spin_event_start, self.spin_event_end, self.spin_group_window,
            self.spin_dur_min, self.spin_dur_max, self.spin_min_events_per_animal,
            self.spin_metric_pre0, self.spin_metric_pre1,
            self.spin_metric_post0, self.spin_metric_post1,
            self.spin_global_start, self.spin_global_end,
        ):
            w.setMinimumWidth(60)
            w.setSizePolicy(QtWidgets.QSizePolicy.Policy.Ignored, QtWidgets.QSizePolicy.Policy.Fixed)

        # ── Helper: dual-spin row ──
        def _dual_row(lbl_a: str, w_a, lbl_b: str, w_b):
            g = QtWidgets.QGridLayout()
            g.setHorizontalSpacing(6); g.setContentsMargins(0, 0, 0, 0)
            la = QtWidgets.QLabel(lbl_a); la.setMinimumWidth(35)
            lb = QtWidgets.QLabel(lbl_b); lb.setMinimumWidth(35)
            g.addWidget(la, 0, 0); g.addWidget(w_a, 0, 1)
            g.addWidget(lb, 0, 2); g.addWidget(w_b, 0, 3)
            g.setColumnStretch(1, 1); g.setColumnStretch(3, 1)
            w = QtWidgets.QWidget(); w.setLayout(g); return w

        win_widget = _dual_row("Pre:", self.spin_pre, "Post:", self.spin_post)
        base_widget = _dual_row("Start:", self.spin_b0, "End:", self.spin_b1)
        metric_pre_widget = _dual_row("Start:", self.spin_metric_pre0, "End:", self.spin_metric_pre1)
        metric_post_widget = _dual_row("Start:", self.spin_metric_post0, "End:", self.spin_metric_post1)
        global_widget = _dual_row("Start:", self.spin_global_start, "End:", self.spin_global_end)

        # ═══════════════════════════════════════════════════════
        # Section 1 — Window & Baseline
        # ═══════════════════════════════════════════════════════
        grp_window = QtWidgets.QGroupBox("Window && baseline")
        grp_window.setStyleSheet(_psth_section_qss)
        fw = QtWidgets.QFormLayout(grp_window)
        fw.setRowWrapPolicy(QtWidgets.QFormLayout.RowWrapPolicy.WrapLongRows)
        fw.setLabelAlignment(QtCore.Qt.AlignmentFlag.AlignLeft | QtCore.Qt.AlignmentFlag.AlignTop)
        fw.addRow("Window (s)", win_widget)
        fw.addRow("Baseline (s)", base_widget)
        fw.addRow("Resample (Hz)", self.spin_resample)
        fw.addRow("Smooth sigma (s)", self.spin_smooth)

        # ═══════════════════════════════════════════════════════
        # Section 2 — Event filters
        # ═══════════════════════════════════════════════════════
        grp_filt = QtWidgets.QGroupBox("Event filters")
        grp_filt.setStyleSheet(_psth_section_qss)
        ff = QtWidgets.QFormLayout(grp_filt)
        ff.setRowWrapPolicy(QtWidgets.QFormLayout.RowWrapPolicy.WrapLongRows)
        ff.setLabelAlignment(QtCore.Qt.AlignmentFlag.AlignLeft | QtCore.Qt.AlignmentFlag.AlignTop)
        filt_row = QtWidgets.QHBoxLayout()
        filt_row.setContentsMargins(0, 0, 0, 0); filt_row.setSpacing(6)
        filt_row.addWidget(self.cb_filter_events); filt_row.addStretch(1)
        filt_row.addWidget(self.btn_hide_filters)
        filt_widget = QtWidgets.QWidget(); filt_widget.setLayout(filt_row)
        ff.addRow(filt_widget)
        self.lbl_event_start = QtWidgets.QLabel("Start index (1-based)")
        self.lbl_event_end = QtWidgets.QLabel("End index (0 = all)")
        self.lbl_group_window = QtWidgets.QLabel("Group within (s)")
        self.lbl_dur_min = QtWidgets.QLabel("Duration min (s)")
        self.lbl_dur_max = QtWidgets.QLabel("Duration max (s)")
        ff.addRow(self.lbl_event_start, self.spin_event_start)
        ff.addRow(self.lbl_event_end, self.spin_event_end)
        ff.addRow(self.lbl_group_window, self.spin_group_window)
        ff.addRow(self.lbl_dur_min, self.spin_dur_min)
        ff.addRow(self.lbl_dur_max, self.spin_dur_max)

        # ═══════════════════════════════════════════════════════
        # Section 3 — PSTH metrics
        # ═══════════════════════════════════════════════════════
        grp_include = QtWidgets.QGroupBox("Animal inclusion")
        grp_include.setStyleSheet(_psth_section_qss)
        fi = QtWidgets.QFormLayout(grp_include)
        fi.setRowWrapPolicy(QtWidgets.QFormLayout.RowWrapPolicy.WrapLongRows)
        fi.setLabelAlignment(QtCore.Qt.AlignmentFlag.AlignLeft | QtCore.Qt.AlignmentFlag.AlignTop)
        fi.addRow("", self.cb_exclude_low_event_animals)
        fi.addRow("Min events / animal", self.spin_min_events_per_animal)
        fi.addRow("", self.cb_group_keep_trials)

        grp_met = QtWidgets.QGroupBox("PSTH metrics")
        grp_met.setStyleSheet(_psth_section_qss)
        fm = QtWidgets.QFormLayout(grp_met)
        fm.setRowWrapPolicy(QtWidgets.QFormLayout.RowWrapPolicy.WrapLongRows)
        fm.setLabelAlignment(QtCore.Qt.AlignmentFlag.AlignLeft | QtCore.Qt.AlignmentFlag.AlignTop)
        met_row = QtWidgets.QHBoxLayout()
        met_row.setContentsMargins(0, 0, 0, 0); met_row.setSpacing(6)
        met_row.addWidget(self.cb_metrics); met_row.addStretch(1)
        met_row.addWidget(self.btn_hide_metrics)
        met_widget = QtWidgets.QWidget(); met_widget.setLayout(met_row)
        fm.addRow(met_widget)
        self.lbl_metric = QtWidgets.QLabel("Metric")
        self.lbl_metric_pre = QtWidgets.QLabel("Pre window (s)")
        self.lbl_metric_post = QtWidgets.QLabel("Post window (s)")
        fm.addRow(self.lbl_metric, self.combo_metric)
        fm.addRow(self.lbl_metric_pre, metric_pre_widget)
        fm.addRow(self.lbl_metric_post, metric_post_widget)

        # ═══════════════════════════════════════════════════════
        # Section 4 — Global metrics
        # ═══════════════════════════════════════════════════════
        grp_global = QtWidgets.QGroupBox("Global metrics")
        grp_global.setStyleSheet(_psth_section_qss)
        fg = QtWidgets.QFormLayout(grp_global)
        fg.setRowWrapPolicy(QtWidgets.QFormLayout.RowWrapPolicy.WrapLongRows)
        fg.setLabelAlignment(QtCore.Qt.AlignmentFlag.AlignLeft | QtCore.Qt.AlignmentFlag.AlignTop)
        fg.addRow(self.cb_global_metrics)
        fg.addRow("Range (s)", global_widget)
        global_opts = QtWidgets.QHBoxLayout()
        global_opts.setContentsMargins(0, 0, 0, 0); global_opts.setSpacing(6)
        global_opts.addWidget(self.cb_global_amp)
        global_opts.addWidget(self.cb_global_freq)
        global_opts.addStretch(1)
        global_opts_widget = QtWidgets.QWidget(); global_opts_widget.setLayout(global_opts)
        fg.addRow("Compute", global_opts_widget)
        fg.addRow("", self.lbl_global_metrics)

        # Container for all subsections (replaces old grp_opt)
        grp_opt = QtWidgets.QWidget()
        _psth_vbox = QtWidgets.QVBoxLayout(grp_opt)
        _psth_vbox.setContentsMargins(0, 0, 0, 0)
        _psth_vbox.setSpacing(4)
        _psth_vbox.addWidget(grp_window)
        _psth_vbox.addWidget(grp_filt)
        _psth_vbox.addWidget(grp_include)
        _psth_vbox.addWidget(grp_met)
        _psth_vbox.addWidget(grp_global)

        self.btn_compute = QtWidgets.QPushButton("Postprocessing (compute PSTH)")
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
        self.btn_style = QtWidgets.QPushButton("Plot style")
        self.btn_style.setProperty("class", "ghost")
        self.btn_style.setToolTip("Open plot styling options")
        self.btn_style.setSizePolicy(QtWidgets.QSizePolicy.Policy.Minimum, QtWidgets.QSizePolicy.Policy.Fixed)
        self.btn_save_cfg = QtWidgets.QPushButton("Save config")
        self.btn_save_cfg.setProperty("class", "compactSmall")
        self.btn_save_cfg.setSizePolicy(QtWidgets.QSizePolicy.Policy.Ignored, QtWidgets.QSizePolicy.Policy.Fixed)
        self.btn_load_cfg = QtWidgets.QPushButton("Load config")
        self.btn_load_cfg.setProperty("class", "compactSmall")
        self.btn_load_cfg.setSizePolicy(QtWidgets.QSizePolicy.Policy.Ignored, QtWidgets.QSizePolicy.Policy.Fixed)
        self.btn_reset_cfg = QtWidgets.QPushButton("Reset defaults")
        self.btn_reset_cfg.setProperty("class", "compactSmall")
        self.btn_reset_cfg.setSizePolicy(QtWidgets.QSizePolicy.Policy.Ignored, QtWidgets.QSizePolicy.Policy.Fixed)
        self.btn_new_project = QtWidgets.QPushButton("New project")
        self.btn_new_project.setProperty("class", "compactSmall")
        self.btn_new_project.setSizePolicy(QtWidgets.QSizePolicy.Policy.Ignored, QtWidgets.QSizePolicy.Policy.Fixed)
        self.btn_save_project = QtWidgets.QPushButton("Save project (.h5)")
        self.btn_save_project.setProperty("class", "compactSmall")
        self.btn_save_project.setSizePolicy(QtWidgets.QSizePolicy.Policy.Ignored, QtWidgets.QSizePolicy.Policy.Fixed)
        self.btn_load_project = QtWidgets.QPushButton("Load project (.h5)")
        self.btn_load_project.setProperty("class", "compactSmall")
        self.btn_load_project.setSizePolicy(QtWidgets.QSizePolicy.Policy.Ignored, QtWidgets.QSizePolicy.Policy.Fixed)

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
        self.cb_peak_auto_mad = QtWidgets.QCheckBox("Auto transient threshold (MAD noise)")
        self.cb_peak_auto_mad.setChecked(False)
        self.cb_peak_auto_mad.setToolTip(
            "Estimate trace noise as 1.4826 x MAD after baseline/smoothing and use "
            "the multiplier below as the minimum peak prominence."
        )
        self.spin_peak_mad_multiplier = QtWidgets.QDoubleSpinBox()
        self.spin_peak_mad_multiplier.setRange(0.5, 50.0)
        self.spin_peak_mad_multiplier.setValue(5.0)
        self.spin_peak_mad_multiplier.setDecimals(2)
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
        self.cb_peak_norm_prominence = QtWidgets.QCheckBox()
        self.cb_peak_norm_prominence.setChecked(False)
        self.cb_peak_norm_prominence.setToolTip(
            "Report peak amplitude after scaling by the top baseline peak prominences."
        )

        # ---- Baseline-prominence configuration ------------------------------
        self.combo_signal_baseline_source = QtWidgets.QComboBox()
        self.combo_signal_baseline_source.addItem("Time window (e.g. before task)", "window")
        self.combo_signal_baseline_source.addItem("Exclude behavior bouts", "exclude")
        self.combo_signal_baseline_source.addItem("Whole recording (MAD noise)", "whole")
        self.combo_signal_baseline_source.setToolTip(
            "How to define the 'quiet' segment used to set the prominence "
            "normalization scale and (when enabled) MAD noise estimate.\n\n"
            "- Time window: an explicit [start, end] interval, ideal for "
            "  pre-task recordings (e.g. 5 min before social interaction).\n"
            "- Exclude behavior bouts: load behavior files in Setup and tick the "
            "  behaviors you want removed; everything OUTSIDE those bouts becomes "
            "  baseline.\n"
            "- Whole recording: fall back to MAD noise across the whole trace."
        )
        _compact_combo(self.combo_signal_baseline_source, min_chars=12)

        self.spin_signal_baseline_start = QtWidgets.QDoubleSpinBox()
        self.spin_signal_baseline_start.setRange(0.0, 1e6)
        self.spin_signal_baseline_start.setDecimals(2)
        self.spin_signal_baseline_start.setSuffix(" s")
        self.spin_signal_baseline_start.setValue(0.0)
        self.spin_signal_baseline_start.setMaximumWidth(150)

        self.spin_signal_baseline_end = QtWidgets.QDoubleSpinBox()
        self.spin_signal_baseline_end.setRange(0.0, 1e6)
        self.spin_signal_baseline_end.setDecimals(2)
        self.spin_signal_baseline_end.setSuffix(" s")
        self.spin_signal_baseline_end.setValue(300.0)  # 5 min default
        self.spin_signal_baseline_end.setMaximumWidth(150)
        self.spin_signal_baseline_end.setToolTip("Set to 0 to extend until the end of the recording.")

        self.combo_signal_baseline_behavior_file = QtWidgets.QComboBox()
        self.combo_signal_baseline_behavior_file.setToolTip(
            "Which loaded behavior file should provide the exclusion intervals "
            "(behavior files are loaded in the Setup panel)."
        )
        _compact_combo(self.combo_signal_baseline_behavior_file, min_chars=10)

        self.list_signal_baseline_exclude = QtWidgets.QListWidget()
        self.list_signal_baseline_exclude.setSelectionMode(
            QtWidgets.QAbstractItemView.SelectionMode.NoSelection
        )
        self.list_signal_baseline_exclude.setMaximumHeight(110)
        self.list_signal_baseline_exclude.setToolTip(
            "Tick the behaviors whose intervals should be removed from baseline."
        )

        self.spin_signal_baseline_pad = QtWidgets.QDoubleSpinBox()
        self.spin_signal_baseline_pad.setRange(0.0, 600.0)
        self.spin_signal_baseline_pad.setDecimals(2)
        self.spin_signal_baseline_pad.setSuffix(" s")
        self.spin_signal_baseline_pad.setValue(1.0)
        self.spin_signal_baseline_pad.setMaximumWidth(150)
        self.spin_signal_baseline_pad.setToolTip(
            "Also exclude this many seconds before and after each behavior event "
            "(catches the rise / decay of evoked transients)."
        )

        self.lbl_signal_baseline_status = QtWidgets.QLabel("")
        self.lbl_signal_baseline_status.setProperty("class", "hint")
        self.lbl_signal_baseline_status.setWordWrap(True)
        self.cb_peak_overlay = QtWidgets.QCheckBox("Show detected peaks on trace")
        self.cb_peak_overlay.setChecked(True)
        self.cb_peak_noise_overlay = QtWidgets.QCheckBox("Show noise trace / MAD threshold overlay")
        self.cb_peak_noise_overlay.setChecked(False)
        self.cb_peak_noise_overlay.setToolTip(
            "After peak detection, overlay the preprocessed detection trace, robust noise band, "
            "and effective prominence threshold used for the visible file."
        )
        self.btn_detect_peaks = QtWidgets.QPushButton("Detect peaks")
        self.btn_detect_peaks.setProperty("class", "compactPrimarySmall")
        self.btn_export_peaks = QtWidgets.QPushButton("Export peaks CSV")
        self.btn_export_peaks.setProperty("class", "compactSmall")
        self.lbl_signal_msg = QtWidgets.QLabel("")
        self.lbl_signal_msg.setProperty("class", "hint")
        # Build into Subsections instead of a 17-row flat form.
        SIG_LABEL_W = 110

        def _sig_form() -> QtWidgets.QFormLayout:
            f = QtWidgets.QFormLayout()
            f.setContentsMargins(0, 2, 0, 2)
            f.setHorizontalSpacing(10)
            f.setVerticalSpacing(8)
            f.setLabelAlignment(QtCore.Qt.AlignmentFlag.AlignLeft | QtCore.Qt.AlignmentFlag.AlignVCenter)
            f.setRowWrapPolicy(QtWidgets.QFormLayout.RowWrapPolicy.DontWrapRows)
            f.setFieldGrowthPolicy(QtWidgets.QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)
            return f

        def _sig_add(form: QtWidgets.QFormLayout, label: str, field: QtWidgets.QWidget) -> None:
            lbl = QtWidgets.QLabel(label)
            lbl.setMinimumWidth(SIG_LABEL_W)
            lbl.setMaximumWidth(SIG_LABEL_W)
            form.addRow(lbl, field)

        # Source subsection.
        sub_sig_source = _PyberSubsection(
            "Source",
            "Pick the trace and scope for peak detection.",
        )
        fs_src = _sig_form()
        _sig_add(fs_src, "Signal source", self.combo_signal_source)
        _sig_add(fs_src, "Group mode", self.combo_signal_scope)
        _sig_add(fs_src, "File", self.combo_signal_file)
        sub_sig_source.add_layout(fs_src)

        # Detection subsection.
        # Suffixes baked into spinboxes for tighter forms.
        self.spin_peak_prominence.setMaximumWidth(150)
        self.spin_peak_mad_multiplier.setSuffix(" sigma")
        self.spin_peak_mad_multiplier.setMaximumWidth(150)
        self.spin_peak_height.setMaximumWidth(150)
        self.spin_peak_distance.setSuffix(" s")
        self.spin_peak_distance.setMaximumWidth(150)
        self.spin_peak_smooth.setSuffix(" s")
        self.spin_peak_smooth.setMaximumWidth(150)
        # Strip the redundant text from cb_peak_auto_mad - form row now labels it.
        self.cb_peak_auto_mad.setText("")
        sub_sig_detect = _PyberSubsection(
            "Detection",
            "Where to draw the peak threshold (manual prominence or MAD-noise auto).",
        )
        fs_det = _sig_form()
        _sig_add(fs_det, "Method", self.combo_signal_method)
        _sig_add(fs_det, "Auto MAD threshold", self.cb_peak_auto_mad)
        _sig_add(fs_det, "MAD multiplier", self.spin_peak_mad_multiplier)
        _sig_add(fs_det, "Min prominence", self.spin_peak_prominence)
        _sig_add(fs_det, "Min height (0=off)", self.spin_peak_height)
        _sig_add(fs_det, "Min distance", self.spin_peak_distance)
        _sig_add(fs_det, "Smooth sigma", self.spin_peak_smooth)
        sub_sig_detect.add_layout(fs_det)

        # Baseline subsection (the headline fix).
        self.spin_peak_baseline_window.setSuffix(" s")
        self.spin_peak_baseline_window.setMaximumWidth(150)
        sub_sig_baseline = _PyberSubsection(
            "Baseline prominence",
            "Define the 'quiet' segment used as the normalization reference and "
            "(optionally) for the MAD noise floor.",
        )
        fs_base = _sig_form()
        _sig_add(fs_base, "Baseline handling", self.combo_peak_baseline)
        _sig_add(fs_base, "Detrend window", self.spin_peak_baseline_window)
        _sig_add(fs_base, "Normalize amplitude", self.cb_peak_norm_prominence)
        _sig_add(fs_base, "Baseline source", self.combo_signal_baseline_source)

        # Time window inputs - one row.
        win_row = QtWidgets.QHBoxLayout()
        win_row.setContentsMargins(0, 0, 0, 0)
        win_row.setSpacing(6)
        self.spin_signal_baseline_start.setPrefix("Start ")
        self.spin_signal_baseline_end.setPrefix("End ")
        win_row.addWidget(self.spin_signal_baseline_start)
        win_row.addWidget(self.spin_signal_baseline_end)
        win_row.addStretch(1)
        self._signal_baseline_window_wrap = QtWidgets.QWidget()
        self._signal_baseline_window_wrap.setLayout(win_row)
        _sig_add(fs_base, "Window", self._signal_baseline_window_wrap)

        # Exclude-behaviors inputs (file picker + list + pad).
        _sig_add(fs_base, "Behavior file", self.combo_signal_baseline_behavior_file)
        _sig_add(fs_base, "Exclude bouts", self.list_signal_baseline_exclude)
        _sig_add(fs_base, "Pad before/after", self.spin_signal_baseline_pad)

        fs_base.addRow(self.lbl_signal_baseline_status)
        sub_sig_baseline.add_layout(fs_base)

        # Output subsection.
        self.spin_peak_rate_bin.setSuffix(" s")
        self.spin_peak_rate_bin.setMaximumWidth(150)
        self.spin_peak_auc_window.setSuffix(" s")
        self.spin_peak_auc_window.setMaximumWidth(150)
        # The two overlay checkboxes' labels are descriptive; keep them, no row label.
        sub_sig_output = _PyberSubsection(
            "Output",
            "Rate aggregation, AUC window, and trace overlays.",
        )
        fs_out = _sig_form()
        _sig_add(fs_out, "Rate bin", self.spin_peak_rate_bin)
        _sig_add(fs_out, "AUC window +/-", self.spin_peak_auc_window)
        fs_out.addRow(self.cb_peak_overlay)
        fs_out.addRow(self.cb_peak_noise_overlay)
        sub_sig_output.add_layout(fs_out)

        # Stack subsections inside the group box.
        grp_signal_layout = QtWidgets.QVBoxLayout()
        grp_signal_layout.setContentsMargins(0, 0, 0, 0)
        grp_signal_layout.setSpacing(4)
        for sub in (sub_sig_source, sub_sig_detect, sub_sig_baseline, sub_sig_output):
            grp_signal_layout.addWidget(sub)
        # Detach the now-empty f_signal layout and install the new one.
        _old_sig_layout = grp_signal.layout()
        if _old_sig_layout is not None:
            QtWidgets.QWidget().setLayout(_old_sig_layout)
        grp_signal.setLayout(grp_signal_layout)

        # Footer: primary Detect + ghost Export, status banner above.
        self.signal_status = _PyberInlineStatus("", "info")
        self.signal_footer = _PyberFooterActions()
        self.signal_footer.add_secondary(self.btn_export_peaks)
        self.signal_footer.add_primary(self.btn_detect_peaks)

        # Mirror existing self.lbl_signal_msg.setText -> banner with severity.
        self.lbl_signal_msg.hide()
        _orig_sig_msg_set_text = self.lbl_signal_msg.setText
        def _sig_msg_set_text(text: str) -> None:
            _orig_sig_msg_set_text(text)
            try:
                low = str(text or "").lower()
                if not text:
                    sev = "info"
                elif "fail" in low or "error" in low:
                    sev = "error"
                elif "no peaks" in low or "skip" in low or "no " in low:
                    sev = "warn"
                elif "detect" in low or "saved" in low or "export" in low:
                    sev = "ok"
                else:
                    sev = "info"
                self.signal_status.set(text, sev)
            except Exception:
                pass
        self.lbl_signal_msg.setText = _sig_msg_set_text  # type: ignore[assignment]

        self.tbl_signal_metrics = QtWidgets.QTableWidget(0, 2)
        self.tbl_signal_metrics.setHorizontalHeaderLabels(["Metric", "Value"])
        self.tbl_signal_metrics.verticalHeader().setVisible(False)
        self.tbl_signal_metrics.horizontalHeader().setStretchLastSection(True)
        self.tbl_signal_metrics.setEditTriggers(QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers)
        self.tbl_signal_metrics.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.NoSelection)

        # Plain frame holds the behavior subsections - no outer GroupBox chip,
        # the panel header already says "Behavior".
        grp_behavior_analysis = QtWidgets.QFrame()
        grp_behavior_analysis.setStyleSheet("background: transparent; border: 0;")
        self.combo_behavior_analysis = QtWidgets.QComboBox()
        _compact_combo(self.combo_behavior_analysis, min_chars=8)
        self.spin_behavior_bin = QtWidgets.QDoubleSpinBox()
        self.spin_behavior_bin.setRange(0.5, 3600.0)
        self.spin_behavior_bin.setValue(30.0)
        self.spin_behavior_bin.setDecimals(2)
        self.spin_behavior_bin.setSuffix(" s")
        self.spin_behavior_bin.setMaximumWidth(160)
        self.cb_behavior_aligned = QtWidgets.QCheckBox()  # label text moved to form row
        self.cb_behavior_aligned.setChecked(False)
        self.cb_behavior_aligned.setToolTip(
            "When the recordings include an aligned timeline, use it instead of "
            "the raw behavior timestamps."
        )
        self.btn_compute_behavior = QtWidgets.QPushButton("Compute metrics")
        self.btn_compute_behavior.setToolTip("Recompute behavior metrics for the loaded recordings (Ctrl+R)")
        self.btn_export_behavior_metrics = QtWidgets.QPushButton("Export metrics CSV")
        self.btn_export_behavior_metrics.setToolTip("Export the per-file behavior metrics table to CSV")
        self.btn_export_behavior_events = QtWidgets.QPushButton("Export events CSV")
        self.btn_export_behavior_events.setToolTip("Export the underlying behavior event list to CSV")
        self.lbl_behavior_msg = QtWidgets.QLabel("")
        self.lbl_behavior_msg.setProperty("class", "hint")

        # Build into clean Subsections instead of a flat 4-row form.
        BEH_LABEL_W = 96

        def _beh_form() -> QtWidgets.QFormLayout:
            f = QtWidgets.QFormLayout()
            f.setContentsMargins(0, 2, 0, 2)
            f.setHorizontalSpacing(10)
            f.setVerticalSpacing(8)
            f.setLabelAlignment(QtCore.Qt.AlignmentFlag.AlignLeft | QtCore.Qt.AlignmentFlag.AlignVCenter)
            f.setRowWrapPolicy(QtWidgets.QFormLayout.RowWrapPolicy.DontWrapRows)
            f.setFieldGrowthPolicy(QtWidgets.QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)
            return f

        def _beh_add(form: QtWidgets.QFormLayout, label: str, field: QtWidgets.QWidget) -> None:
            lbl = QtWidgets.QLabel(label)
            lbl.setMinimumWidth(BEH_LABEL_W)
            lbl.setMaximumWidth(BEH_LABEL_W)
            form.addRow(lbl, field)

        # Subsection: which behavior + bin size
        sub_beh_pick = _PyberSubsection(
            "Selection",
            "Behavior to analyze and bin width for over-time rate.",
        )
        f_pick = _beh_form()
        _beh_add(f_pick, "Behavior", self.combo_behavior_analysis)
        _beh_add(f_pick, "Bin size", self.spin_behavior_bin)
        _beh_add(f_pick, "Aligned timeline", self.cb_behavior_aligned)
        sub_beh_pick.add_layout(f_pick)

        grp_beh_layout = QtWidgets.QVBoxLayout()
        grp_beh_layout.setContentsMargins(0, 0, 0, 0)
        grp_beh_layout.setSpacing(4)
        grp_beh_layout.addWidget(sub_beh_pick)
        old_beh_layout = grp_behavior_analysis.layout()
        if old_beh_layout is not None:
            QtWidgets.QWidget().setLayout(old_beh_layout)
        grp_behavior_analysis.setLayout(grp_beh_layout)

        # Footer: primary Compute + two ghost Exports.
        self.behavior_status = _PyberInlineStatus("", "info")
        self.behavior_footer = _PyberFooterActions()
        self.behavior_footer.add_secondary(self.btn_export_behavior_events)
        self.behavior_footer.add_secondary(self.btn_export_behavior_metrics)
        self.behavior_footer.add_primary(self.btn_compute_behavior)

        # Mirror existing self.lbl_behavior_msg.setText -> banner with severity.
        self.lbl_behavior_msg.hide()
        _orig_beh_set_text = self.lbl_behavior_msg.setText
        def _beh_set_text(text: str) -> None:
            _orig_beh_set_text(text)
            try:
                low = str(text or "").lower()
                if not text:
                    sev = "info"
                elif "fail" in low or "error" in low:
                    sev = "error"
                elif "skip" in low or "no " in low or "missing" in low:
                    sev = "warn"
                elif "compute" in low or "export" in low or "ok" in low or "saved" in low:
                    sev = "ok"
                else:
                    sev = "info"
                self.behavior_status.set(text, sev)
            except Exception:
                pass
        self.lbl_behavior_msg.setText = _beh_set_text  # type: ignore[assignment]

        # NB: lbl_behavior_summary is created later (after the metrics table).
        # The mirror that re-routes its setText into the InlineStatus banner
        # is wired further down, after the label exists.

        self.tbl_behavior_metrics = QtWidgets.QTableWidget(0, 8)
        # Short, glance-able column names so the table fits the drawer.
        self.tbl_behavior_metrics.setHorizontalHeaderLabels(
            ["File", "n", "Total", "Mean", "Med", "SD", "Rate/min", "Frac"]
        )
        self.tbl_behavior_metrics.verticalHeader().setVisible(False)
        self.tbl_behavior_metrics.setEditTriggers(QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers)
        self.tbl_behavior_metrics.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows)
        self.tbl_behavior_metrics.setAlternatingRowColors(True)
        self.tbl_behavior_metrics.setHorizontalScrollMode(QtWidgets.QAbstractItemView.ScrollMode.ScrollPerPixel)
        # First column (File) takes available width; rest size to contents.
        _hdr = self.tbl_behavior_metrics.horizontalHeader()
        _hdr.setSectionResizeMode(QtWidgets.QHeaderView.ResizeMode.ResizeToContents)
        _hdr.setSectionResizeMode(0, QtWidgets.QHeaderView.ResizeMode.Stretch)
        _hdr.setStretchLastSection(False)

        self.lbl_behavior_summary = QtWidgets.QLabel("Group metrics: -")
        self.lbl_behavior_summary.hide()

        # Mirror the (now hidden) summary label's text into the InlineStatus
        # banner so existing self.lbl_behavior_summary.setText(...) calls
        # surface as a properly styled status above the footer.
        _orig_beh_summary_set_text = self.lbl_behavior_summary.setText
        def _beh_summary_set_text(text: str) -> None:
            _orig_beh_summary_set_text(text)
            try:
                txt = str(text or "").strip()
                if not txt or txt == "Group metrics: -":
                    self.behavior_status.set("", "info")  # hides
                else:
                    self.behavior_status.set(txt, "info")
            except Exception:
                pass
        self.lbl_behavior_summary.setText = _beh_summary_set_text  # type: ignore[assignment]
        self.lbl_behavior_summary.setProperty("class", "hint")

        grp_spatial = QtWidgets.QGroupBox("Spatial")
        f_spatial = QtWidgets.QFormLayout(grp_spatial)
        f_spatial.setRowWrapPolicy(QtWidgets.QFormLayout.RowWrapPolicy.DontWrapRows)
        f_spatial.setLabelAlignment(QtCore.Qt.AlignmentFlag.AlignLeft | QtCore.Qt.AlignmentFlag.AlignTop)

        self.combo_spatial_x = QtWidgets.QComboBox()
        self.combo_spatial_y = QtWidgets.QComboBox()
        _compact_combo(self.combo_spatial_x, min_chars=10)
        _compact_combo(self.combo_spatial_y, min_chars=10)
        self.spin_spatial_bins_x = QtWidgets.QSpinBox()
        self.spin_spatial_bins_x.setRange(8, 512)
        self.spin_spatial_bins_x.setValue(64)
        self.spin_spatial_bins_y = QtWidgets.QSpinBox()
        self.spin_spatial_bins_y.setRange(8, 512)
        self.spin_spatial_bins_y.setValue(64)
        self.combo_spatial_weight = QtWidgets.QComboBox()
        self.combo_spatial_weight.addItems(
            [
                "Occupancy (samples)",
                "Occupancy time (s)",
                "Probability (% of time)",
            ]
        )
        _compact_combo(self.combo_spatial_weight, min_chars=10)
        self.cb_spatial_clip = QtWidgets.QCheckBox()
        self.cb_spatial_clip.setChecked(True)
        self.spin_spatial_clip_low = QtWidgets.QDoubleSpinBox()
        self.spin_spatial_clip_low.setRange(0.0, 49.0)
        self.spin_spatial_clip_low.setValue(1.0)
        self.spin_spatial_clip_low.setDecimals(2)
        self.spin_spatial_clip_high = QtWidgets.QDoubleSpinBox()
        self.spin_spatial_clip_high.setRange(51.0, 100.0)
        self.spin_spatial_clip_high.setValue(99.0)
        self.spin_spatial_clip_high.setDecimals(2)
        self.spin_spatial_smooth = QtWidgets.QDoubleSpinBox()
        self.spin_spatial_smooth.setRange(0.0, 20.0)
        self.spin_spatial_smooth.setValue(0.0)
        self.spin_spatial_smooth.setDecimals(2)
        self.cb_spatial_log = QtWidgets.QCheckBox()
        self.cb_spatial_log.setChecked(False)
        self.cb_spatial_invert_y = QtWidgets.QCheckBox()
        self.cb_spatial_invert_y.setChecked(False)
        self.combo_spatial_activity_mode = QtWidgets.QComboBox()
        self.combo_spatial_activity_mode.addItems(
            [
                "Mean z-score/bin (occupancy normalized)",
                "Mean z-score/bin (velocity normalized)",
                "Sum z-score/bin (no normalization)",
            ]
        )
        self.combo_spatial_activity_mode.setCurrentIndex(0)
        _compact_combo(self.combo_spatial_activity_mode, min_chars=12)
        self.cb_spatial_time_filter = QtWidgets.QCheckBox()
        self.cb_spatial_time_filter.setChecked(False)
        self.spin_spatial_time_min = QtWidgets.QDoubleSpinBox()
        self.spin_spatial_time_min.setRange(-1e9, 1e9)
        self.spin_spatial_time_min.setDecimals(3)
        self.spin_spatial_time_min.setValue(0.0)
        self.spin_spatial_time_max = QtWidgets.QDoubleSpinBox()
        self.spin_spatial_time_max.setRange(-1e9, 1e9)
        self.spin_spatial_time_max.setDecimals(3)
        self.spin_spatial_time_max.setValue(0.0)
        self.btn_spatial_help = QtWidgets.QToolButton()
        self.btn_spatial_help.setText("?")
        self.btn_spatial_help.setToolTip(
            "Spatial heatmap help:\n"
            "- Top plot: occupancy map.\n"
            "- Middle plot: activity map (mode selected below).\n"
            "- Bottom plot: velocity map (mean speed/bin).\n"
            "- Mean z-score/bin (occupancy normalized) = sum(z*weight) / sum(weight).\n"
            "- Mean z-score/bin (velocity normalized) = sum(z*weight) / sum(speed*weight).\n"
            "- Enable Time filter to restrict trajectory/activity samples to [min,max] seconds.\n"
            "- Use right-side color cursors on each plot to set min/max display range."
        )
        self.btn_compute_spatial = QtWidgets.QPushButton("Compute spatial heatmap")
        self.btn_compute_spatial.setProperty("class", "compactPrimarySmall")
        self.btn_compute_spatial.setToolTip("Render occupancy + activity + velocity heatmaps from the trajectory")
        self.btn_compute_spatial.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Minimum, QtWidgets.QSizePolicy.Policy.Fixed
        )
        self.btn_compute_spatial.setMinimumWidth(220)
        self.lbl_spatial_msg = QtWidgets.QLabel("")
        self.lbl_spatial_msg.setProperty("class", "hint")
        self.lbl_spatial_msg.setWordWrap(True)

        # Compact paired inputs: one row, max-width spinboxes, prefix labels
        # baked into the input via setPrefix so they don't compete for space.
        for sb, prefix in (
            (self.spin_spatial_bins_x, "X "),
            (self.spin_spatial_bins_y, "Y "),
        ):
            sb.setPrefix(prefix)
            sb.setMaximumWidth(110)
            sb.setMinimumWidth(80)
        for sb, prefix in (
            (self.spin_spatial_clip_low, "Lo "),
            (self.spin_spatial_clip_high, "Hi "),
        ):
            sb.setPrefix(prefix)
            sb.setSuffix(" %")
            sb.setMaximumWidth(120)
            sb.setMinimumWidth(90)
        for sb, prefix in (
            (self.spin_spatial_time_min, "Start "),
            (self.spin_spatial_time_max, "End "),
        ):
            sb.setPrefix(prefix)
            sb.setSuffix(" s")
            sb.setMaximumWidth(140)
            sb.setMinimumWidth(110)
        self.spin_spatial_smooth.setSuffix(" bins")
        self.spin_spatial_smooth.setMaximumWidth(140)

        def _pair(a: QtWidgets.QWidget, b: QtWidgets.QWidget) -> QtWidgets.QWidget:
            box = QtWidgets.QWidget()
            row = QtWidgets.QHBoxLayout(box)
            row.setContentsMargins(0, 0, 0, 0)
            row.setSpacing(6)
            row.addWidget(a)
            row.addWidget(b)
            row.addStretch(1)
            return box

        bins_widget = _pair(self.spin_spatial_bins_x, self.spin_spatial_bins_y)
        clip_widget = _pair(self.spin_spatial_clip_low, self.spin_spatial_clip_high)
        time_widget = _pair(self.spin_spatial_time_min, self.spin_spatial_time_max)

        help_row = QtWidgets.QHBoxLayout()
        help_row.setContentsMargins(0, 0, 0, 0)
        help_row.addStretch(1)
        help_row.addWidget(self.btn_spatial_help)
        help_widget = QtWidgets.QWidget()
        help_widget.setLayout(help_row)

        # Build the form into clearly-labelled subsections instead of a 15-row flat list.
        # All form labels share a fixed minimum width so columns line up across subsections.
        FORM_LABEL_W = 96

        def _form() -> QtWidgets.QFormLayout:
            f = QtWidgets.QFormLayout()
            f.setContentsMargins(0, 2, 0, 2)
            f.setHorizontalSpacing(10)
            f.setVerticalSpacing(8)
            f.setLabelAlignment(QtCore.Qt.AlignmentFlag.AlignLeft | QtCore.Qt.AlignmentFlag.AlignVCenter)
            f.setFormAlignment(QtCore.Qt.AlignmentFlag.AlignLeft | QtCore.Qt.AlignmentFlag.AlignTop)
            f.setRowWrapPolicy(QtWidgets.QFormLayout.RowWrapPolicy.DontWrapRows)
            f.setFieldGrowthPolicy(QtWidgets.QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)
            return f

        def _add_row(form: QtWidgets.QFormLayout, label: str, field: QtWidgets.QWidget) -> None:
            lbl = QtWidgets.QLabel(label)
            lbl.setMinimumWidth(FORM_LABEL_W)
            lbl.setMaximumWidth(FORM_LABEL_W)
            form.addRow(lbl, field)

        # ---- Subsection: trajectory source ----
        sub_traj = _PyberSubsection(
            "Trajectory",
            "X / Y position columns from the loaded behavior file.",
        )
        sub_traj.add_head_action(self.btn_spatial_help)
        f1 = _form()
        _add_row(f1, "X column", self.combo_spatial_x)
        _add_row(f1, "Y column", self.combo_spatial_y)
        sub_traj.add_layout(f1)

        # ---- Subsection: occupancy map ----
        sub_occ = _PyberSubsection(
            "Occupancy",
            "Spatial bin grid and what each bin counts.",
        )
        f2 = _form()
        _add_row(f2, "Bins", bins_widget)
        _add_row(f2, "Map value", self.combo_spatial_weight)
        _add_row(f2, "Smoothing", self.spin_spatial_smooth)
        _add_row(f2, "Log scale", self.cb_spatial_log)
        _add_row(f2, "Invert Y axis", self.cb_spatial_invert_y)
        sub_occ.add_layout(f2)

        # ---- Subsection: range clipping ----
        sub_clip = _PyberSubsection(
            "Display range",
            "Clip extreme percentiles for readable colors.",
        )
        f3 = _form()
        _add_row(f3, "Enable", self.cb_spatial_clip)
        _add_row(f3, "Percentiles", clip_widget)
        sub_clip.add_layout(f3)

        # ---- Subsection: time filter ----
        sub_time = _PyberSubsection(
            "Time filter",
            "Restrict trajectory samples to a window.",
        )
        f4 = _form()
        _add_row(f4, "Enable", self.cb_spatial_time_filter)
        _add_row(f4, "Range", time_widget)
        sub_time.add_layout(f4)

        # ---- Subsection: activity map ----
        sub_act = _PyberSubsection(
            "Activity map",
            "How per-bin signal values are normalized.",
        )
        f5 = _form()
        _add_row(f5, "Mode", self.combo_spatial_activity_mode)
        sub_act.add_layout(f5)

        # Stack subsections inside the existing group box.
        grp_spatial_layout = QtWidgets.QVBoxLayout()
        grp_spatial_layout.setContentsMargins(0, 0, 0, 0)
        grp_spatial_layout.setSpacing(4)
        for sub in (sub_traj, sub_occ, sub_clip, sub_time, sub_act):
            grp_spatial_layout.addWidget(sub)

        # Replace the existing group box's layout (was f_spatial / QFormLayout).
        # We drop f_spatial entirely; it was assigned but is no longer attached.
        old_layout = grp_spatial.layout()
        if old_layout is not None:
            QtWidgets.QWidget().setLayout(old_layout)  # detach safely
        grp_spatial.setLayout(grp_spatial_layout)

        # Inline status banner + footer with the primary Compute action.
        self.spatial_status = _PyberInlineStatus("", "info")
        self.spatial_footer = _PyberFooterActions()
        self.spatial_footer.add_primary(self.btn_compute_spatial)

        # Mirror existing `self.lbl_spatial_msg.setText(...)` calls into the new
        # banner without changing every call site: monkey-patch setText.
        self.lbl_spatial_msg.hide()
        _orig_set_text = self.lbl_spatial_msg.setText
        def _set_text_with_banner(text: str) -> None:
            _orig_set_text(text)
            try:
                low = str(text or "").lower()
                if not text:
                    sev = "info"
                elif "skip" in low or "no " in low or "missing" in low:
                    sev = "warn"
                elif "fail" in low or "error" in low:
                    sev = "error"
                elif "render" in low or "compute" in low or "ok" in low:
                    sev = "ok"
                else:
                    sev = "info"
                self.spatial_status.set(text, sev)
            except Exception:
                pass
        self.lbl_spatial_msg.setText = _set_text_with_banner  # type: ignore[assignment]

        # ---- Time synchronization panel ----
        self.combo_sync_behavior_file = QtWidgets.QComboBox()
        self.combo_sync_behavior_file.addItem("Auto-match behavior file", "")
        _compact_combo(self.combo_sync_behavior_file, min_chars=12)
        self.combo_sync_camera_column = QtWidgets.QComboBox()
        _compact_combo(self.combo_sync_camera_column, min_chars=12)
        self.combo_sync_camera_mode = QtWidgets.QComboBox()
        self.combo_sync_camera_mode.addItems(["TTL rising edges", "TTL falling edges", "Barcode / value changes"])
        _compact_combo(self.combo_sync_camera_mode, min_chars=12)
        self.combo_sync_fiber_source = QtWidgets.QComboBox()
        self.combo_sync_fiber_source.addItem("Embedded DIO in processed file", "__embedded__")
        _compact_combo(self.combo_sync_fiber_source, min_chars=12)
        self.combo_sync_fiber_mode = QtWidgets.QComboBox()
        self.combo_sync_fiber_mode.addItems(["TTL rising edges", "TTL falling edges", "Barcode / value changes"])
        _compact_combo(self.combo_sync_fiber_mode, min_chars=12)
        self.combo_sync_method = QtWidgets.QComboBox()
        self.combo_sync_method.addItems(["Linear regression", "Interpolation"])
        _compact_combo(self.combo_sync_method, min_chars=12)
        self.cb_sync_auto_threshold = QtWidgets.QCheckBox("Auto threshold")
        self.cb_sync_auto_threshold.setChecked(True)
        self.spin_sync_threshold = QtWidgets.QDoubleSpinBox()
        self.spin_sync_threshold.setRange(-1e9, 1e9)
        self.spin_sync_threshold.setDecimals(4)
        self.spin_sync_threshold.setValue(0.5)
        self.spin_sync_threshold.setEnabled(False)
        self.spin_sync_min_interval = QtWidgets.QDoubleSpinBox()
        self.spin_sync_min_interval.setRange(0.0, 3600.0)
        self.spin_sync_min_interval.setDecimals(3)
        self.spin_sync_min_interval.setValue(0.2)
        self.spin_sync_min_interval.setSuffix(" s")
        self.spin_sync_max_offset = QtWidgets.QSpinBox()
        self.spin_sync_max_offset.setRange(0, 1000)
        self.spin_sync_max_offset.setValue(5)
        self.cb_sync_use_aligned = QtWidgets.QCheckBox("Use aligned time for postprocessing")
        self.cb_sync_use_aligned.setChecked(False)
        self.cb_sync_auto_recompute = QtWidgets.QCheckBox("Recompute PSTH after apply")
        self.cb_sync_auto_recompute.setChecked(True)
        self.combo_sync_export_format = QtWidgets.QComboBox()
        self.combo_sync_export_format.addItems(["CSV + H5", "CSV only", "H5 only"])
        _compact_combo(self.combo_sync_export_format, min_chars=8)
        self.btn_sync_refresh = QtWidgets.QPushButton("Refresh sources")
        self.btn_sync_refresh.setProperty("class", "compactSmall")
        self.btn_sync_preview = QtWidgets.QPushButton("Preview selected file")
        self.btn_sync_preview.setProperty("class", "compactSmall")
        self.btn_sync_apply_selected = QtWidgets.QPushButton("Apply selected")
        self.btn_sync_apply_selected.setProperty("class", "compactPrimarySmall")
        self.btn_sync_apply_batch = QtWidgets.QPushButton("Apply batch")
        self.btn_sync_apply_batch.setProperty("class", "compactPrimarySmall")
        self.btn_sync_export = QtWidgets.QPushButton("Export aligned files")
        self.btn_sync_export.setProperty("class", "compactSmall")
        self.sync_status = _PyberInlineStatus("", "info")
        self.txt_sync_report = QtWidgets.QTextEdit()
        self.txt_sync_report.setReadOnly(True)
        self.txt_sync_report.setMinimumHeight(92)
        self.txt_sync_report.setPlaceholderText("Preview or apply time synchronization to see fit diagnostics.")
        self.tbl_sync_results = QtWidgets.QTableWidget(0, 6)
        self.tbl_sync_results.setHorizontalHeaderLabels(["File", "Pairs", "RMS ms", "Median lag ms", "Drift ppm", "Status"])
        self.tbl_sync_results.verticalHeader().setVisible(False)
        self.tbl_sync_results.setEditTriggers(QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers)
        self.tbl_sync_results.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows)
        self.tbl_sync_results.setMinimumHeight(120)
        try:
            self.tbl_sync_results.horizontalHeader().setStretchLastSection(True)
            self.tbl_sync_results.horizontalHeader().setSectionResizeMode(0, QtWidgets.QHeaderView.ResizeMode.Stretch)
        except Exception:
            pass
        self.plot_sync_map = pg.PlotWidget(title="Sync event mapping")
        self.plot_sync_residual = pg.PlotWidget(title="Residual lag")
        for pw in (self.plot_sync_map, self.plot_sync_residual):
            _opt_plot(pw)
            pw.setMinimumHeight(150)
        self.plot_sync_map.setLabel("bottom", "Photometry sync time (s)")
        self.plot_sync_map.setLabel("left", "Camera / behavior sync time (s)")
        self.plot_sync_residual.setLabel("bottom", "Matched pulse")
        self.plot_sync_residual.setLabel("left", "Residual (ms)")

        grp_sync = QtWidgets.QGroupBox("Time synchronization")
        sync_layout = QtWidgets.QVBoxLayout(grp_sync)
        sync_layout.setContentsMargins(8, 8, 8, 8)
        sync_layout.setSpacing(8)

        sub_sync_sources = _PyberSubsection(
            "Sources",
            "Pair an external camera/behavior TTL or barcode column with the photometry DIO sync signal.",
        )
        f_sync_sources = _form()
        _add_row(f_sync_sources, "Behavior file", self.combo_sync_behavior_file)
        _add_row(f_sync_sources, "Camera sync", self.combo_sync_camera_column)
        _add_row(f_sync_sources, "Camera mode", self.combo_sync_camera_mode)
        _add_row(f_sync_sources, "Photometry sync", self.combo_sync_fiber_source)
        _add_row(f_sync_sources, "Photometry mode", self.combo_sync_fiber_mode)
        sub_sync_sources.add_layout(f_sync_sources)

        sub_sync_fit = _PyberSubsection(
            "Alignment",
            "Linear regression estimates global clock drift; interpolation follows pulse-to-pulse timing.",
        )
        f_sync_fit = _form()
        _add_row(f_sync_fit, "Method", self.combo_sync_method)
        _add_row(f_sync_fit, "Threshold", self.cb_sync_auto_threshold)
        _add_row(f_sync_fit, "Manual level", self.spin_sync_threshold)
        _add_row(f_sync_fit, "Min interval", self.spin_sync_min_interval)
        _add_row(f_sync_fit, "Pulse offset", self.spin_sync_max_offset)
        _add_row(f_sync_fit, "Use result", self.cb_sync_use_aligned)
        _add_row(f_sync_fit, "Auto recompute", self.cb_sync_auto_recompute)
        sub_sync_fit.add_layout(f_sync_fit)

        sub_sync_batch = _PyberSubsection(
            "Batch and export",
            "The selected settings are reused across matched files in the queue and saved in the project.",
        )
        sync_btn_grid = QtWidgets.QGridLayout()
        sync_btn_grid.setContentsMargins(0, 0, 0, 0)
        sync_btn_grid.setHorizontalSpacing(6)
        sync_btn_grid.setVerticalSpacing(6)
        sync_btn_grid.addWidget(self.btn_sync_refresh, 0, 0)
        sync_btn_grid.addWidget(self.btn_sync_preview, 0, 1)
        sync_btn_grid.addWidget(self.btn_sync_apply_selected, 1, 0)
        sync_btn_grid.addWidget(self.btn_sync_apply_batch, 1, 1)
        sync_export_row = QtWidgets.QHBoxLayout()
        sync_export_row.setContentsMargins(0, 0, 0, 0)
        sync_export_row.setSpacing(6)
        sync_export_row.addWidget(self.combo_sync_export_format, 1)
        sync_export_row.addWidget(self.btn_sync_export, 1)
        sub_sync_batch.add_layout(sync_btn_grid)
        sub_sync_batch.add_layout(sync_export_row)

        sync_layout.addWidget(sub_sync_sources)
        sync_layout.addWidget(sub_sync_fit)
        sync_layout.addWidget(sub_sync_batch)
        sync_layout.addWidget(self.sync_status)
        sync_layout.addWidget(self.txt_sync_report)
        sync_layout.addWidget(self.tbl_sync_results)
        sync_layout.addWidget(self.plot_sync_map)
        sync_layout.addWidget(self.plot_sync_residual)

        self.section_setup = QtWidgets.QWidget()
        setup_layout = QtWidgets.QVBoxLayout(self.section_setup)
        setup_layout.setContentsMargins(6, 6, 6, 6)
        setup_layout.setSpacing(8)
        setup_layout.addWidget(grp_src)
        setup_layout.addWidget(grp_align, stretch=1)
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
        signal_layout.addWidget(self.signal_status)
        signal_layout.addWidget(self.tbl_signal_metrics, 1)
        signal_layout.addWidget(self.signal_footer)

        self.section_behavior = QtWidgets.QWidget()
        behavior_layout = QtWidgets.QVBoxLayout(self.section_behavior)
        behavior_layout.setContentsMargins(6, 6, 6, 6)
        behavior_layout.setSpacing(8)
        behavior_layout.addWidget(grp_behavior_analysis)
        behavior_layout.addWidget(self.behavior_status)
        behavior_layout.addWidget(self.tbl_behavior_metrics, 1)
        behavior_layout.addWidget(self.lbl_behavior_summary)
        behavior_layout.addWidget(self.behavior_footer)

        self.section_sync = QtWidgets.QWidget()
        sync_section_layout = QtWidgets.QVBoxLayout(self.section_sync)
        sync_section_layout.setContentsMargins(6, 6, 6, 6)
        sync_section_layout.setSpacing(8)
        sync_section_layout.addWidget(grp_sync)

        self.section_temporal = TemporalModelingWidget()
        self.section_temporal.statusMessage.connect(
            lambda msg, ms: self.statusUpdate.emit(msg, ms)
        )

        self.section_spatial = QtWidgets.QWidget()
        spatial_layout = QtWidgets.QVBoxLayout(self.section_spatial)
        spatial_layout.setContentsMargins(6, 6, 6, 6)
        spatial_layout.setSpacing(8)
        spatial_layout.addWidget(grp_spatial)
        spatial_layout.addWidget(self.spatial_status)
        spatial_layout.addStretch(1)
        spatial_layout.addWidget(self.spatial_footer)

        self.section_export = QtWidgets.QWidget()
        export_layout = QtWidgets.QVBoxLayout(self.section_export)
        export_layout.setContentsMargins(6, 6, 6, 6)
        export_layout.setSpacing(8)
        export_layout.addWidget(self.btn_export)
        export_layout.addWidget(self.btn_export_img)
        export_layout.addWidget(self.btn_save_cfg)
        export_layout.addWidget(self.btn_load_cfg)
        export_layout.addWidget(self.btn_reset_cfg)
        export_layout.addWidget(self.btn_new_project)
        export_layout.addWidget(self.btn_save_project)
        export_layout.addWidget(self.btn_load_project)
        export_layout.addStretch(1)

        action_row = QtWidgets.QHBoxLayout()
        action_row.setContentsMargins(0, 0, 0, 0)
        action_row.setSpacing(6)

        self.btn_action_load = QtWidgets.QPushButton("File")
        self.btn_action_load.setProperty("class", "compactPrimarySmall")
        self.menu_action_load = QtWidgets.QMenu(self.btn_action_load)
        self.act_load_current = self.menu_action_load.addAction("Use current preprocessed selection")
        self.act_load_single = self.menu_action_load.addAction("Load processed file (single)")
        self.act_load_group = self.menu_action_load.addAction("Load processed files (group)")
        self.menu_action_load.addSeparator()
        self.act_new_project = self.menu_action_load.addAction("New project")
        self.act_reset_blank_project = self.menu_action_load.addAction("Reset blank project")
        self.act_save_project = self.menu_action_load.addAction("Save project (.h5)")
        self.act_load_project = self.menu_action_load.addAction("Load project (.h5)")
        self.menu_action_load.addSeparator()
        self.act_load_behavior = self.menu_action_load.addAction("Load behavior CSV/XLSX")
        self.menu_action_recent = self.menu_action_load.addMenu("Load recent")
        self.menu_recent_processed = self.menu_action_recent.addMenu("Processed files")
        self.menu_recent_behavior = self.menu_action_recent.addMenu("Behavior files")
        self.menu_recent_projects = self.menu_action_recent.addMenu("Projects")
        self.menu_action_recent.aboutToShow.connect(self._refresh_recent_postprocessing_menus)
        self.act_refresh_dio = self.menu_action_load.addAction("Refresh A/D channel list")
        self.menu_action_load.addSeparator()
        self.act_open_plot_style = self.menu_action_load.addAction("Plot style...")
        self.btn_action_load.setMenu(self.menu_action_load)

        self.btn_action_compute = QtWidgets.QPushButton("Compute PSTH")
        self.btn_action_compute.setProperty("class", "compactPrimarySmall")
        self.btn_action_export = QtWidgets.QPushButton("Export")
        self.btn_action_export.setProperty("class", "compactPrimarySmall")
        self.btn_action_reset = QtWidgets.QPushButton("Reset")
        self.btn_action_reset.setProperty("class", "ghost")
        self.btn_action_reset.setToolTip(
            "Clear postprocessing to a blank project and stop reopening the autosaved project"
        )
        self.btn_action_undo = QtWidgets.QToolButton()
        self.btn_action_undo.setObjectName("toolbarIconButton")
        self.btn_action_undo.setText("↶")
        self.btn_action_undo.setToolTip("Undo last postprocessing setting or view action (Ctrl+Z)")
        self.btn_action_undo.setFixedSize(34, 30)
        self.btn_action_redo = QtWidgets.QToolButton()
        self.btn_action_redo.setObjectName("toolbarIconButton")
        self.btn_action_redo.setText("↷")
        self.btn_action_redo.setToolTip("Redo last undone postprocessing setting or view action (Ctrl+Y)")
        self.btn_action_redo.setFixedSize(34, 30)
        self.btn_action_help = QtWidgets.QToolButton()
        self.btn_action_help.setObjectName("toolbarIconButton")
        self.btn_action_help.setText("?")
        self.btn_action_help.setToolTip("Show onboarding and panel tutorials")
        self.btn_action_help.setFixedSize(34, 30)
        self.btn_action_hide = QtWidgets.QPushButton("Hide Panels")
        self.btn_action_hide.setProperty("class", "ghost")

        self.btn_panel_setup = QtWidgets.QToolButton(); self.btn_panel_setup.setText("Setup")
        self.btn_panel_psth = QtWidgets.QToolButton(); self.btn_panel_psth.setText("PSTH")
        self.btn_panel_spatial = QtWidgets.QToolButton(); self.btn_panel_spatial.setText("Spatial")
        self.btn_panel_export = QtWidgets.QToolButton(); self.btn_panel_export.setText("Export")
        self.btn_panel_signal = QtWidgets.QToolButton(); self.btn_panel_signal.setText("Signal")
        self.btn_panel_behavior = QtWidgets.QToolButton(); self.btn_panel_behavior.setText("Behavior")
        self.btn_panel_sync = QtWidgets.QToolButton(); self.btn_panel_sync.setText("Sync")
        self.btn_panel_temporal = QtWidgets.QToolButton(); self.btn_panel_temporal.setText("Temporal")
        self._section_buttons = {
            "setup": self.btn_panel_setup,
            "psth": self.btn_panel_psth,
            "spatial": self.btn_panel_spatial,
            "export": self.btn_panel_export,
            "signal": self.btn_panel_signal,
            "behavior": self.btn_panel_behavior,
            "sync": self.btn_panel_sync,
            "temporal": self.btn_panel_temporal,
        }
        for b in self._section_buttons.values():
            b.setCheckable(True)
            b.setProperty("class", "compactSmall")

        # ----- Modern shell: convert section buttons to a left icon rail
        # and put workflow actions in a thin transport bar above the plots.
        from styles import (
            _make_icon, _paint_sliders, _paint_chart, _paint_grid,
            _paint_export, _paint_pulse, _paint_paw, _paint_temporal,
            _paint_sync,
        )
        _post_rail_meta = {
            "setup":    ("Setup",     "Load processed traces and behavior", _paint_sliders),
            "psth":     ("PSTH",      "Trial windows, baselines and metrics", _paint_chart),
            "spatial":  ("Spatial",   "Occupancy and activity heatmaps", _paint_grid),
            "export":   ("Export",    "Export PSTH, peaks, behavior", _paint_export),
            "signal":   ("Events",    "Peak / event detection", _paint_pulse),
            "behavior": ("Behavior",  "Behavior alignment and per-state analysis", _paint_paw),
            "sync":     ("Sync",      "Align photometry time to camera or behavior time", _paint_sync),
            "temporal": ("Modeling",  "GLM / FLMM modeling", _paint_temporal),
        }
        self._post_rail_icon_painters = {key: meta[2] for key, meta in _post_rail_meta.items()}
        for key, btn in self._section_buttons.items():
            label, hint, painter = _post_rail_meta[key]
            btn.setObjectName("railButton")
            btn.setProperty("class", "")
            btn.setText(label)
            btn.setToolTip(f"{label} - {hint}")
            btn.setStatusTip(f"{label} - {hint}")
            btn.setIcon(_make_icon(painter))
            btn.setIconSize(QtCore.QSize(22, 22))
            btn.setToolButtonStyle(QtCore.Qt.ToolButtonStyle.ToolButtonTextUnderIcon)
            btn.setFixedSize(76, 60)
            btn.setCheckable(True)

        self._post_side_rail = QtWidgets.QFrame()
        self._post_side_rail.setObjectName("sideRail")
        rail_layout = QtWidgets.QVBoxLayout(self._post_side_rail)
        rail_layout.setContentsMargins(8, 10, 8, 10)
        rail_layout.setSpacing(6)
        for key in ("setup", "sync", "psth", "spatial", "temporal", "signal", "behavior", "export"):
            rail_layout.addWidget(self._section_buttons[key], 0,
                                  QtCore.Qt.AlignmentFlag.AlignHCenter)
        rail_layout.addStretch(1)
        self._post_side_rail.setFixedWidth(96)

        # Transport bar holds the workflow actions.
        self._post_transport_bar = QtWidgets.QFrame()
        self._post_transport_bar.setObjectName("transportBar")
        tb_layout = QtWidgets.QHBoxLayout(self._post_transport_bar)
        tb_layout.setContentsMargins(12, 8, 12, 8)
        tb_layout.setSpacing(8)
        self.btn_action_compute.setText("Compute PSTH")
        self.btn_action_export.setText("Run Export")
        self.btn_action_hide.setText("Hide drawer")

        # Primary group: File / Compute / Export / Reset.
        for b in (self.btn_action_load, self.btn_action_compute, self.btn_action_export, self.btn_action_reset):
            tb_layout.addWidget(b)

        # Visual hairline separator between primary actions and utilities.
        sep = QtWidgets.QFrame()
        sep.setFixedWidth(1)
        sep.setFixedHeight(22)
        sep.setStyleSheet("background: #2c3240; border: 0;")
        tb_layout.addSpacing(4)
        tb_layout.addWidget(sep)
        tb_layout.addSpacing(4)

        # Utility group: Undo / Redo / Plot style / Help.
        for b in (self.btn_action_undo, self.btn_action_redo, self.btn_style, self.btn_action_help):
            tb_layout.addWidget(b)
        self._update_history_buttons()
        tb_layout.addStretch(1)
        tb_layout.addWidget(self.btn_action_hide)
        # action_row is no longer used; left in scope so any later reference
        # (none expected) does not crash. The transport bar replaces it.
        del action_row

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

        # --- Visual mode tabs: Individual / Group ---
        visual_bar = QtWidgets.QHBoxLayout()
        visual_bar.setContentsMargins(0, 0, 0, 0)
        visual_bar.setSpacing(8)
        self.tab_visual_mode = QtWidgets.QTabBar()
        self.tab_visual_mode.setObjectName("visualModeBar")
        self.tab_visual_mode.addTab("Individual")
        self.tab_visual_mode.addTab("Group")
        self.tab_visual_mode.setDrawBase(False)
        self.tab_visual_mode.setExpanding(False)
        visual_bar.addWidget(self.tab_visual_mode)
        visual_bar.addWidget(QtWidgets.QLabel("  File:"))
        self.combo_individual_file = QtWidgets.QComboBox()
        _compact_combo(self.combo_individual_file, min_chars=12)
        self.combo_individual_file.setToolTip("Select file for individual view (each row = trial)")
        visual_bar.addWidget(self.combo_individual_file, stretch=1)
        visual_bar.addStretch(1)
        rv.addLayout(visual_bar)

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
        self.plot_spatial_occupancy = pg.PlotWidget(title="Spatial occupancy")
        self.plot_spatial_activity = pg.PlotWidget(title="Spatial activity (mean z-score)")
        self.plot_spatial_velocity = pg.PlotWidget(title="Spatial velocity (mean speed)")

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
            self.plot_spatial_occupancy,
            self.plot_spatial_activity,
            self.plot_spatial_velocity,
        ):
            _opt_plot(w)

        # Empty-state hints removed by design - keep plots visually clean.
        self._postproc_empty_hints: List[Any] = []

        self.curve_trace = self.plot_trace.plot(pen=pg.mkPen(self._style["trace"], width=1.1))
        self.curve_behavior = self.plot_trace.plot(pen=pg.mkPen(self._style["behavior"], width=1.0))
        self.curve_behavior.setVisible(False)
        self.curve_peak_markers = self.plot_trace.plot(
            pen=None,
            symbol="o",
            symbolSize=6,
            symbolBrush=pg.mkBrush(240, 120, 80),
            symbolPen=pg.mkPen((240, 120, 80), width=1.0),
        )
        self.curve_event_markers = self.plot_trace.plot(
            pen=pg.mkPen((235, 218, 80, 150), width=1.0)
        )
        self.curve_event_markers.setZValue(4)
        self.event_lines: List[pg.InfiniteLine] = []
        self.plot_trace.getViewBox().sigXRangeChanged.connect(self._render_visible_event_annotations)

        self.img = pg.ImageItem()
        self.plot_heat.addItem(self.img)
        self.heat_zero_line = pg.InfiniteLine(
            pos=0.0,
            angle=90,
            movable=False,
            pen=pg.mkPen((245, 245, 245), width=1.0, style=QtCore.Qt.PenStyle.DotLine),
        )
        self.heat_zero_line.setZValue(20)
        self.plot_heat.addItem(self.heat_zero_line)
        self.heat_lut = pg.HistogramLUTWidget()
        self.heat_lut.setMinimumWidth(110)
        self.heat_lut.setMaximumWidth(150)
        self.heat_lut.setImageItem(self.img)
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
        self.plot_spatial_occupancy.setLabel("bottom", "X")
        self.plot_spatial_occupancy.setLabel("left", "Y")
        self.plot_spatial_activity.setLabel("bottom", "X")
        self.plot_spatial_activity.setLabel("left", "Y")
        self.plot_spatial_velocity.setLabel("bottom", "X")
        self.plot_spatial_velocity.setLabel("left", "Y")
        self.img_spatial_occupancy = pg.ImageItem()
        self.img_spatial_activity = pg.ImageItem()
        self.img_spatial_velocity = pg.ImageItem()
        self.plot_spatial_occupancy.addItem(self.img_spatial_occupancy)
        self.plot_spatial_activity.addItem(self.img_spatial_activity)
        self.plot_spatial_velocity.addItem(self.img_spatial_velocity)
        self.spatial_lut_occupancy = pg.HistogramLUTWidget()
        self.spatial_lut_activity = pg.HistogramLUTWidget()
        self.spatial_lut_velocity = pg.HistogramLUTWidget()
        self.spatial_lut_occupancy.setMinimumWidth(110)
        self.spatial_lut_occupancy.setMaximumWidth(150)
        self.spatial_lut_activity.setMinimumWidth(110)
        self.spatial_lut_activity.setMaximumWidth(150)
        self.spatial_lut_velocity.setMinimumWidth(110)
        self.spatial_lut_velocity.setMaximumWidth(150)
        self.spatial_lut_occupancy.setImageItem(self.img_spatial_occupancy)
        self.spatial_lut_activity.setImageItem(self.img_spatial_activity)
        self.spatial_lut_velocity.setImageItem(self.img_spatial_velocity)

        self.curve_avg = self.plot_avg.plot(pen=pg.mkPen(self._style["avg"], width=1.3))
        self.curve_sem_hi = self.plot_avg.plot(pen=pg.mkPen((152, 201, 143), width=1.0))
        self.curve_sem_lo = self.plot_avg.plot(pen=pg.mkPen((152, 201, 143), width=1.0))
        self.sem_band = pg.FillBetweenItem(
            self.curve_sem_hi,
            self.curve_sem_lo,
            brush=pg.mkBrush(188, 230, 178, 96),
        )
        self.plot_avg.addItem(self.sem_band)
        self.plot_avg.addLine(x=0, pen=pg.mkPen((200, 200, 200), style=QtCore.Qt.PenStyle.DashLine))
        self.metrics_bar_pre = pg.BarGraphItem(x=[0], height=[0], width=0.6, brush=(90, 143, 214))
        self.metrics_bar_post = pg.BarGraphItem(x=[1], height=[0], width=0.6, brush=(214, 122, 90))
        self.plot_metrics.addItem(self.metrics_bar_pre)
        self.plot_metrics.addItem(self.metrics_bar_post)
        # Overlay paired trial/event points (pre vs post) and links.
        self.metrics_pairs_curve = self.plot_metrics.plot(
            pen=pg.mkPen((210, 215, 225, 130), width=1.0),
            connect="finite",
            skipFiniteCheck=True,
        )
        self.metrics_scatter_pre = self.plot_metrics.plot(
            pen=None,
            symbol="o",
            symbolSize=5,
            symbolBrush=pg.mkBrush(90, 143, 214, 220),
            symbolPen=pg.mkPen((90, 143, 214), width=0.8),
        )
        self.metrics_scatter_post = self.plot_metrics.plot(
            pen=None,
            symbol="o",
            symbolSize=5,
            symbolBrush=pg.mkBrush(214, 122, 90, 220),
            symbolPen=pg.mkPen((214, 122, 90), width=0.8),
        )
        self.metrics_err_pre = pg.ErrorBarItem(
            x=np.array([0.0], float),
            y=np.array([0.0], float),
            top=np.array([0.0], float),
            bottom=np.array([0.0], float),
            beam=0.22,
            pen=pg.mkPen((230, 236, 246), width=2.0),
        )
        self.metrics_err_post = pg.ErrorBarItem(
            x=np.array([1.0], float),
            y=np.array([0.0], float),
            top=np.array([0.0], float),
            bottom=np.array([0.0], float),
            beam=0.22,
            pen=pg.mkPen((230, 236, 246), width=2.0),
        )
        self.plot_metrics.addItem(self.metrics_err_pre)
        self.plot_metrics.addItem(self.metrics_err_post)
        self.metrics_p_text = pg.TextItem("", color=(230, 236, 246), anchor=(0.5, 1.0))
        self.metrics_p_text.setZValue(20)
        self.metrics_p_text.setVisible(False)
        self.plot_metrics.addItem(self.metrics_p_text)
        self.plot_metrics.setXRange(-0.5, 1.5, padding=0)
        self.plot_metrics.getAxis("bottom").setTicks([[(0, "pre"), (1, "post")]])

        self.global_bar_amp = pg.BarGraphItem(x=[0], height=[0], width=0.6, brush=(120, 180, 220))
        self.global_bar_freq = pg.BarGraphItem(x=[1], height=[0], width=0.6, brush=(220, 160, 120))
        self.plot_global.addItem(self.global_bar_amp)
        self.plot_global.addItem(self.global_bar_freq)
        self.global_scatter_amp = self.plot_global.plot(
            pen=None,
            symbol="o",
            symbolSize=6,
            symbolBrush=pg.mkBrush(120, 180, 220, 220),
            symbolPen=pg.mkPen((120, 180, 220), width=0.9),
        )
        self.global_scatter_freq = self.plot_global.plot(
            pen=None,
            symbol="o",
            symbolSize=6,
            symbolBrush=pg.mkBrush(220, 160, 120, 220),
            symbolPen=pg.mkPen((220, 160, 120), width=0.9),
        )
        self.global_err_amp = pg.ErrorBarItem(
            x=np.array([0.0], float),
            y=np.array([0.0], float),
            top=np.array([0.0], float),
            bottom=np.array([0.0], float),
            beam=0.22,
            pen=pg.mkPen((230, 236, 246), width=2.0),
        )
        self.global_err_freq = pg.ErrorBarItem(
            x=np.array([1.0], float),
            y=np.array([0.0], float),
            top=np.array([0.0], float),
            bottom=np.array([0.0], float),
            beam=0.22,
            pen=pg.mkPen((230, 236, 246), width=2.0),
        )
        self.plot_global.addItem(self.global_err_amp)
        self.plot_global.addItem(self.global_err_freq)
        self.plot_global.setXRange(-0.5, 1.5, padding=0)
        self.plot_global.getAxis("bottom").setTicks([[(0, "amp"), (1, "freq")]])

        self.row_heat = QtWidgets.QWidget()
        heat_row = QtWidgets.QHBoxLayout(self.row_heat)
        heat_row.setContentsMargins(0, 0, 0, 0)
        heat_row.setSpacing(8)
        heat_row.addWidget(self.plot_heat, stretch=4)
        heat_row.addWidget(self.heat_lut, stretch=0)
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

        self.spatial_plot_dialog = QtWidgets.QDialog(self)
        self.spatial_plot_dialog.setWindowTitle("Spatial")
        self.spatial_plot_dialog.setModal(False)
        self.spatial_plot_dialog.resize(980, 920)
        spatial_dialog_layout = QtWidgets.QVBoxLayout(self.spatial_plot_dialog)
        spatial_dialog_layout.setContentsMargins(8, 8, 8, 8)
        spatial_dialog_layout.setSpacing(8)
        self.spatial_plot_content = QtWidgets.QWidget(self.spatial_plot_dialog)
        spatial_content_layout = QtWidgets.QVBoxLayout(self.spatial_plot_content)
        spatial_content_layout.setContentsMargins(0, 0, 0, 0)
        spatial_content_layout.setSpacing(8)
        spatial_top_row = QtWidgets.QHBoxLayout()
        spatial_top_row.setContentsMargins(0, 0, 0, 0)
        spatial_top_row.setSpacing(8)
        spatial_top_row.addWidget(self.plot_spatial_occupancy, stretch=1)
        spatial_top_row.addWidget(self.spatial_lut_occupancy, stretch=0)
        spatial_bottom_row = QtWidgets.QHBoxLayout()
        spatial_bottom_row.setContentsMargins(0, 0, 0, 0)
        spatial_bottom_row.setSpacing(8)
        spatial_bottom_row.addWidget(self.plot_spatial_activity, stretch=1)
        spatial_bottom_row.addWidget(self.spatial_lut_activity, stretch=0)
        spatial_velocity_row = QtWidgets.QHBoxLayout()
        spatial_velocity_row.setContentsMargins(0, 0, 0, 0)
        spatial_velocity_row.setSpacing(8)
        spatial_velocity_row.addWidget(self.plot_spatial_velocity, stretch=1)
        spatial_velocity_row.addWidget(self.spatial_lut_velocity, stretch=0)
        spatial_content_layout.addLayout(spatial_top_row, stretch=1)
        spatial_content_layout.addLayout(spatial_bottom_row, stretch=1)
        spatial_content_layout.addLayout(spatial_velocity_row, stretch=1)
        self.lbl_spatial_cursor_hint = QtWidgets.QLabel("Use the right-side color cursors to set min/max display range for each map.")
        self.lbl_spatial_cursor_hint.setProperty("class", "hint")
        spatial_content_layout.addWidget(self.lbl_spatial_cursor_hint)
        spatial_dialog_layout.addWidget(self.spatial_plot_content, stretch=1)
        self.btn_export_spatial_img = QtWidgets.QPushButton("Export image")
        self.btn_export_spatial_img.setProperty("class", "compactSmall")
        spatial_export_row = QtWidgets.QHBoxLayout()
        spatial_export_row.setContentsMargins(0, 0, 0, 0)
        spatial_export_row.addStretch(1)
        spatial_export_row.addWidget(self.btn_export_spatial_img)
        spatial_dialog_layout.addLayout(spatial_export_row)
        self.spatial_plot_dialog.hide()

        # Keep a visible minimum plot footprint even with aggressive docking/resizing.
        self.plot_trace.setMinimumHeight(140)
        self.row_heat.setMinimumHeight(180)
        self.row_avg.setMinimumHeight(140)

        rv.addWidget(self.plot_trace, stretch=1)
        rv.addWidget(self.row_heat, stretch=2)
        rv.addWidget(self.row_avg, stretch=1)
        rv.addWidget(self.row_signal, stretch=1)
        rv.addWidget(self.row_behavior, stretch=1)
        if self._use_pg_dockarea_layout:
            workspace = QtWidgets.QSplitter(QtCore.Qt.Orientation.Horizontal)
            workspace.setChildrenCollapsible(False)
            # Wrap the central plots in a rounded center panel.
            self._post_center_panel = QtWidgets.QFrame()
            self._post_center_panel.setObjectName("centerPanel")
            _cp_layout = QtWidgets.QVBoxLayout(self._post_center_panel)
            _cp_layout.setContentsMargins(10, 10, 10, 10)
            _cp_layout.setSpacing(8)
            _cp_layout.addWidget(self._post_transport_bar)
            _cp_layout.addWidget(right, stretch=1)
            self._dockarea = DockArea()
            self._dockarea.setMinimumWidth(_POST_RIGHT_PANEL_MIN_WIDTH)
            # Wrap dockarea in a drawer frame; start hidden.
            self._post_drawer = QtWidgets.QFrame()
            self._post_drawer.setObjectName("drawerPanel")
            _drw = QtWidgets.QVBoxLayout(self._post_drawer)
            _drw.setContentsMargins(12, 10, 12, 10)
            _drw.setSpacing(8)
            # Rich panel header (badge + title + subtitle); set per active section.
            self._post_drawer_header = _PyberPanelHeader()
            _drw.addWidget(self._post_drawer_header)
            # Hidden compat label so legacy lookups don't crash.
            self._post_drawer_title = QtWidgets.QLabel("")
            self._post_drawer_title.setVisible(False)
            _drw.addWidget(self._dockarea, stretch=1)
            self._post_drawer.setVisible(False)
            workspace.addWidget(self._post_drawer)
            workspace.addWidget(self._post_center_panel)
            workspace.setStretchFactor(0, 0)
            workspace.setStretchFactor(1, 1)
            workspace.setSizes([0, 1600])
            self._dockarea_splitter = workspace

            # Outer container: rail | workspace
            container = QtWidgets.QWidget()
            ch = QtWidgets.QHBoxLayout(container)
            ch.setContentsMargins(0, 0, 0, 0)
            ch.setSpacing(8)
            ch.addWidget(self._post_side_rail)
            ch.addWidget(workspace, stretch=1)
            root.addWidget(container, stretch=1)
        else:
            # Fallback non-DockArea path: still apply rail + transport bar.
            self._post_center_panel = QtWidgets.QFrame()
            self._post_center_panel.setObjectName("centerPanel")
            _cp_layout = QtWidgets.QVBoxLayout(self._post_center_panel)
            _cp_layout.setContentsMargins(10, 10, 10, 10)
            _cp_layout.setSpacing(8)
            _cp_layout.addWidget(self._post_transport_bar)
            _cp_layout.addWidget(right, stretch=1)
            container = QtWidgets.QWidget()
            ch = QtWidgets.QHBoxLayout(container)
            ch.setContentsMargins(0, 0, 0, 0)
            ch.setSpacing(8)
            ch.addWidget(self._post_side_rail)
            ch.addWidget(self._post_center_panel, stretch=1)
            root.addWidget(container, stretch=1)
        root.setStretch(0, 0)
        root.setStretch(1, 1)
        if self._use_pg_dockarea_layout:
            self._setup_dockarea_sections()
        else:
            self._setup_section_popups()
        self._apply_widget_theme_mode()

        # Wiring
        self.act_load_current.triggered.connect(self.requestCurrentProcessed.emit)
        self.act_load_single.triggered.connect(self._load_processed_files_single)
        self.act_load_group.triggered.connect(self._load_processed_files)
        self.act_new_project.triggered.connect(self._new_project)
        self.act_reset_blank_project.triggered.connect(self._reset_blank_project)
        self.act_save_project.triggered.connect(self._save_project_file)
        self.act_load_project.triggered.connect(self._load_project_file)
        self.act_load_behavior.triggered.connect(self._load_behavior_files)
        self.act_refresh_dio.triggered.connect(self.requestDioList.emit)
        self.act_open_plot_style.triggered.connect(self._open_style_dialog)
        self.btn_action_compute.clicked.connect(self._compute_psth)
        self.btn_action_export.clicked.connect(self._export_results)
        self.btn_action_reset.clicked.connect(self._reset_blank_project)
        self.btn_action_undo.clicked.connect(self._undo_post_action)
        self.btn_action_redo.clicked.connect(self._redo_post_action)
        self.btn_action_help.clicked.connect(lambda _checked=False: self.helpRequested.emit())
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
        self.btn_sync_refresh.clicked.connect(self._refresh_sync_sources)
        self.btn_sync_preview.clicked.connect(lambda _checked=False: self._preview_time_sync())
        self.btn_sync_apply_selected.clicked.connect(lambda _checked=False: self._apply_time_sync_selected())
        self.btn_sync_apply_batch.clicked.connect(lambda _checked=False: self._apply_time_sync_batch())
        self.btn_sync_export.clicked.connect(lambda _checked=False: self._export_sync_aligned_files())
        self.cb_sync_auto_threshold.toggled.connect(self._on_sync_auto_threshold_toggled)
        self.combo_sync_behavior_file.currentIndexChanged.connect(self._refresh_sync_camera_columns)
        self.cb_sync_use_aligned.toggled.connect(lambda _checked=False: self._on_sync_use_aligned_changed())
        self.list_preprocessed.filesDropped.connect(self._on_preprocessed_files_dropped)
        self.list_preprocessed.orderChanged.connect(self._sync_processed_order_from_list)
        self.list_preprocessed.itemSelectionChanged.connect(self._compute_spatial_heatmap)
        self.list_behaviors.filesDropped.connect(self._on_behavior_files_dropped)
        self.list_behaviors.orderChanged.connect(self._sync_behavior_order_from_list)
        self.list_behaviors.itemSelectionChanged.connect(self._compute_spatial_heatmap)
        self.btn_compute.clicked.connect(self._compute_psth)
        self.btn_update.clicked.connect(self._compute_psth)
        self.btn_detect_peaks.clicked.connect(self._detect_signal_events)
        self.btn_export_peaks.clicked.connect(self._export_signal_events_csv)
        self.btn_compute_behavior.clicked.connect(self._compute_behavior_analysis)
        self.btn_export_behavior_metrics.clicked.connect(self._export_behavior_metrics_csv)
        self.btn_export_behavior_events.clicked.connect(self._export_behavior_events_csv)
        self.btn_compute_spatial.clicked.connect(self._on_compute_spatial_clicked)
        self.btn_export_spatial_img.clicked.connect(self._export_spatial_figure)
        self.btn_export.clicked.connect(self._export_results)
        self.btn_export_img.clicked.connect(self._export_images)
        self.btn_style.clicked.connect(self._open_style_dialog)
        self.btn_save_cfg.clicked.connect(self._save_config_file)
        self.btn_load_cfg.clicked.connect(self._load_config_file)
        self.btn_reset_cfg.clicked.connect(self._reset_config_defaults)
        self.btn_new_project.clicked.connect(self._new_project)
        self.btn_save_project.clicked.connect(self._save_project_file)
        self.btn_load_project.clicked.connect(self._load_project_file)
        self.btn_apply_behavior_time.clicked.connect(self._apply_behavior_time_settings)
        self.cb_filter_events.stateChanged.connect(self._update_event_filter_enabled)
        self.cb_metrics.stateChanged.connect(self._update_metrics_enabled)
        self.cb_global_metrics.stateChanged.connect(self._update_global_metrics_enabled)
        self.btn_hide_filters.toggled.connect(self._toggle_filter_panel)
        self.btn_hide_metrics.toggled.connect(self._toggle_metrics_panel)
        self.combo_view_layout.currentIndexChanged.connect(self._apply_view_layout)
        self.combo_view_layout.currentIndexChanged.connect(self._queue_view_settings_save)
        self.cb_peak_overlay.toggled.connect(self._refresh_signal_overlay)
        self.combo_signal_source.currentIndexChanged.connect(self._refresh_signal_file_combo)
        self.combo_signal_scope.currentIndexChanged.connect(self._refresh_signal_file_combo)
        self.combo_signal_file.currentIndexChanged.connect(self._on_signal_file_changed)
        self.cb_peak_auto_mad.toggled.connect(self._update_peak_auto_mad_enabled)

        # Baseline-prominence source plumbing.
        self.combo_signal_baseline_source.currentIndexChanged.connect(self._update_signal_baseline_source_visibility)
        self.cb_peak_norm_prominence.toggled.connect(self._update_signal_baseline_source_visibility)
        self.spin_signal_baseline_start.valueChanged.connect(self._refresh_signal_baseline_status)
        self.spin_signal_baseline_end.valueChanged.connect(self._refresh_signal_baseline_status)
        self.spin_signal_baseline_pad.valueChanged.connect(self._refresh_signal_baseline_status)
        self.combo_signal_baseline_behavior_file.currentIndexChanged.connect(
            lambda *_: (self._rebuild_signal_baseline_exclude_list(set()), self._refresh_signal_baseline_status())
        )
        self.list_signal_baseline_exclude.itemChanged.connect(lambda *_: self._refresh_signal_baseline_status())
        self.cb_peak_noise_overlay.toggled.connect(self._refresh_signal_overlay)
        self.cb_peak_norm_prominence.toggled.connect(lambda _checked=False: self._save_settings())
        self.tab_sources.currentChanged.connect(self._refresh_signal_file_combo)
        self.tab_visual_mode.currentChanged.connect(self._on_visual_mode_changed)
        self.tab_visual_mode.currentChanged.connect(self._queue_view_settings_save)
        self.combo_individual_file.currentIndexChanged.connect(self._on_individual_file_changed)
        self.combo_individual_file.currentIndexChanged.connect(self._queue_view_settings_save)

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
            self.spin_min_events_per_animal,
            self.spin_metric_pre0,
            self.spin_metric_pre1,
            self.spin_metric_post0,
            self.spin_metric_post1,
            self.spin_global_start,
            self.spin_global_end,
        ):
            w.valueChanged.connect(self._compute_psth)
        self.combo_metric.currentIndexChanged.connect(self._compute_psth)
        self.cb_exclude_low_event_animals.toggled.connect(self._update_psth_inclusion_controls)
        self.cb_exclude_low_event_animals.toggled.connect(self._compute_psth)
        self.cb_group_keep_trials.toggled.connect(self._compute_psth)
        self.cb_global_amp.stateChanged.connect(self._compute_psth)
        self.cb_global_freq.stateChanged.connect(self._compute_psth)
        for w in (self.spin_metric_pre0, self.spin_metric_pre1, self.spin_metric_post0, self.spin_metric_post1):
            w.valueChanged.connect(self._update_metric_regions)
        for w in (self.combo_spatial_x, self.combo_spatial_y, self.combo_spatial_weight):
            w.currentIndexChanged.connect(self._compute_spatial_heatmap)
        for w in (
            self.spin_spatial_bins_x,
            self.spin_spatial_bins_y,
            self.spin_spatial_clip_low,
            self.spin_spatial_clip_high,
            self.spin_spatial_time_min,
            self.spin_spatial_time_max,
            self.spin_spatial_smooth,
        ):
            w.valueChanged.connect(self._compute_spatial_heatmap)
        self.cb_spatial_clip.toggled.connect(self._compute_spatial_heatmap)
        self.cb_spatial_clip.toggled.connect(self._update_spatial_clip_enabled)
        self.cb_spatial_time_filter.toggled.connect(self._compute_spatial_heatmap)
        self.cb_spatial_time_filter.toggled.connect(self._update_spatial_time_filter_enabled)
        self.cb_spatial_log.toggled.connect(self._compute_spatial_heatmap)
        self.cb_spatial_invert_y.toggled.connect(self._compute_spatial_heatmap)
        self.combo_spatial_activity_mode.currentIndexChanged.connect(self._compute_spatial_heatmap)
        self.btn_spatial_help.clicked.connect(self._show_spatial_help)
        if hasattr(self, "heat_lut") and getattr(self.heat_lut, "item", None) is not None:
            level_signal = getattr(self.heat_lut.item, "sigLevelChangeFinished", None)
            if level_signal is None:
                level_signal = getattr(self.heat_lut.item, "sigLevelsChangeFinished", None)
            if level_signal is not None:
                level_signal.connect(self._on_heatmap_levels_changed)
        self._wire_settings_autosave()

        self._apply_plot_style()
        self._update_align_ui()
        self._update_event_filter_enabled()
        self._update_psth_inclusion_controls()
        self._update_metrics_enabled()
        self._update_global_metrics_enabled()
        self._toggle_filter_panel(False)
        self._toggle_metrics_panel(False)
        self._apply_view_layout()
        self._refresh_signal_file_combo()
        self._update_data_availability()
        self._update_spatial_clip_enabled()
        self._update_spatial_time_filter_enabled()
        self._update_behavior_time_panel()
        self._refresh_spatial_columns()
        self._compute_spatial_heatmap()
        self._update_status_strip()

        QtGui.QShortcut(QtGui.QKeySequence("Ctrl+S"), self, activated=self._export_results)
        QtGui.QShortcut(QtGui.QKeySequence("F5"), self, activated=self._compute_psth)

    def _dock_main_window(self) -> Optional[QtWidgets.QMainWindow]:
        host = self.window()
        return host if isinstance(host, QtWidgets.QMainWindow) else None

    def _normalize_app_theme_mode(self, value: object) -> str:
        mode = str(value or "").strip().lower()
        if mode in {"light", "white", "l", "w"}:
            return "light"
        return "dark"

    def _scroll_background_color(self) -> str:
        if self._app_theme_mode == "light":
            return "#f6f8fc"
        return "#242a34"

    def _apply_scroll_theme(self, scroll: QtWidgets.QScrollArea) -> None:
        bg = self._scroll_background_color()
        scroll.setStyleSheet(f"QScrollArea {{ background: {bg}; border: none; }}")
        # Keep viewport painted with dock background so dynamic row visibility
        # does not produce unstyled gaps.
        scroll.viewport().setAutoFillBackground(True)
        scroll.viewport().setStyleSheet(f"background: {bg};")

    def _apply_widget_theme_mode(self) -> None:
        if hasattr(self, "tab_sources"):
            self.tab_sources.setStyleSheet(
                "QTabWidget::pane { border: 0px; background: transparent; padding: 0px; }"
                "QTabBar::tab { margin-right: 0px; }"
            )
        for scroll in self._section_scroll_hosts.values():
            try:
                self._apply_scroll_theme(scroll)
            except Exception:
                continue

    def _refresh_post_rail_icons(self) -> None:
        try:
            from styles import _make_icon
        except Exception:
            return
        icon_color = "#334155" if self._app_theme_mode == "light" else "#cdd6f4"
        buttons = getattr(self, "_section_buttons", {}) or {}
        for key, painter in (getattr(self, "_post_rail_icon_painters", {}) or {}).items():
            btn = buttons.get(key)
            if btn is None:
                continue
            try:
                btn.setIcon(_make_icon(painter, color=icon_color))
            except Exception:
                continue

    def set_app_theme_mode(self, theme_mode: object) -> None:
        self._app_theme_mode = self._normalize_app_theme_mode(theme_mode)
        self._apply_widget_theme_mode()
        self._refresh_post_rail_icons()
        temporal = getattr(self, "section_temporal", None)
        if hasattr(temporal, "set_app_theme_mode"):
            try:
                temporal.set_app_theme_mode(self._app_theme_mode)
            except Exception:
                pass
        self._style["plot_bg"] = self._theme_plot_background()
        try:
            self._apply_plot_style()
        except Exception:
            pass

    def _theme_plot_background(self) -> Tuple[int, int, int]:
        if self._app_theme_mode == "light":
            return (248, 250, 255)
        return (36, 42, 52)

    def _section_widget_map(self) -> Dict[str, Tuple[str, QtWidgets.QWidget]]:
        return {
            "setup": ("Setup", self.section_setup),
            "sync": ("Time Synchronization", self.section_sync),
            "psth": ("PSTH", self.section_psth),
            "spatial": ("Spatial", self.section_spatial),
            "temporal": ("Temporal Modeling", self.section_temporal),
            "export": ("Export", self.section_export),
            "signal": ("Signal Event Analyzer", self.section_signal),
            "behavior": ("Behavior Analysis", self.section_behavior),
        }

    def _setup_dockarea_sections(self) -> None:
        if not self._use_pg_dockarea_layout:
            return
        if self._dockarea is None:
            return
        if self._dockarea_docks:
            return
        for key, (title, widget) in self._section_widget_map().items():
            scroll = QtWidgets.QScrollArea()
            scroll.setWidgetResizable(True)
            scroll.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
            scroll.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
            scroll.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAsNeeded)
            self._apply_scroll_theme(scroll)
            scroll.setMinimumWidth(_POST_RIGHT_PANEL_MIN_WIDTH)
            widget.setMinimumSize(0, 0)
            widget.setSizePolicy(QtWidgets.QSizePolicy.Policy.Preferred, QtWidgets.QSizePolicy.Policy.Preferred)
            scroll.setWidget(widget)
            self._section_scroll_hosts[key] = scroll
            dock = Dock(title, area=self._dockarea, closable=False)
            dock.setObjectName(f"post.da.{key}.dock")
            dock.addWidget(scroll)
            try:
                dock.label.setFixedHeight(0)
                dock.label.setVisible(False)
            except Exception:
                pass
            self._lock_pg_dock_interactions(dock)
            try:
                dock.sigClosed.connect(lambda *_, section_key=key: self._on_dockarea_dock_closed(section_key))
            except Exception:
                pass
            self._dockarea_docks[key] = dock
        if self._force_fixed_default_layout:
            self._apply_fixed_dockarea_layout()
        else:
            self._restore_dockarea_layout_state()
        self._dock_layout_restored = True
        host = self._dock_main_window()
        if host is not None:
            self._dock_host = host
            try:
                if hasattr(host, "on_post_docks_ready"):
                    host.on_post_docks_ready()
                elif hasattr(host, "onPostDocksReady"):
                    host.onPostDocksReady()
            except Exception:
                pass

    def _dockarea_dock(self, key: str) -> Optional[Dock]:
        return self._dockarea_docks.get(key)

    def _dockarea_default_visibility_map(self) -> Dict[str, bool]:
        visible_map = {key: False for key in self._dockarea_docks.keys()}
        for key in _FIXED_POST_RIGHT_TAB_ORDER:
            if key in visible_map:
                visible_map[key] = True
        return visible_map

    def _lock_pg_dock_interactions(self, dock: Dock) -> None:
        label = getattr(dock, "label", None)
        if label is None:
            return
        if not self._force_fixed_default_layout:
            self._style_pg_dock_label_buttons(dock, label)
            return
        if bool(getattr(label, "_pyber_fixed_interaction_lock", False)):
            return

        def _ignore_drag(event: QtGui.QMouseEvent) -> None:
            event.ignore()

        def _ignore_double_click(event: QtGui.QMouseEvent) -> None:
            event.accept()

        try:
            label.mouseMoveEvent = _ignore_drag
            label.mouseDoubleClickEvent = _ignore_double_click
            label.setCursor(QtCore.Qt.CursorShape.ArrowCursor)
            label._pyber_fixed_interaction_lock = True
        except Exception:
            pass
        self._style_pg_dock_label_buttons(dock, label)

    def _style_pg_dock_label_buttons(self, dock: Dock, label: object) -> None:
        if label is None:
            return
        try:
            buttons = label.findChildren(QtWidgets.QToolButton)
        except Exception:
            buttons = []
        for btn in buttons:
            try:
                btn.setText("x")
                btn.setIcon(QtGui.QIcon())
                btn.setAutoRaise(True)
                btn.setFixedSize(13, 13)
                btn.setToolTip("Close")
                if not bool(btn.property("_pyber_hide_wired")):
                    try:
                        btn.clicked.disconnect()
                    except Exception:
                        pass
                    btn.clicked.connect(lambda _checked=False, section_dock=dock: self._hide_dockarea_dock(section_dock))
                    btn.setProperty("_pyber_hide_wired", True)
                light = str(getattr(self, "_app_theme_mode", "dark")).lower() == "light"
                fg = "#4a5568" if light else "#f3f5f8"
                fg_hover = "#172033" if light else "#ffffff"
                btn.setStyleSheet(
                    "QToolButton {"
                    " background: transparent;"
                    f" color: {fg};"
                    " border: none;"
                    " padding: 0px;"
                    " margin: 0px;"
                    " font-size: 8pt;"
                    " font-weight: 700;"
                    " }"
                    "QToolButton:hover {"
                    " background: transparent;"
                    f" color: {fg_hover};"
                    " border: none;"
                    " }"
                )
            except Exception:
                continue

    def _hide_dockarea_dock(self, dock: Dock) -> None:
        if dock is None:
            return
        try:
            dock.hide()
        except Exception:
            return
        for key, candidate in self._dockarea_docks.items():
            if candidate is dock:
                self._set_section_button_checked(key, False)
                if self._last_opened_section == key:
                    self._last_opened_section = None
                break
        self._update_post_drawer_visibility()
        self._save_panel_layout_state()

    def _dockarea_active_key(self) -> Optional[str]:
        active = self._last_opened_section if self._last_opened_section in self._dockarea_docks else None
        if active is not None:
            return active
        for key in _FIXED_POST_RIGHT_TAB_ORDER:
            dock = self._dockarea_dock(key)
            if dock is not None and dock.isVisible():
                return key
        for key, dock in self._dockarea_docks.items():
            if dock.isVisible():
                return key
        return None

    def _set_dockarea_visible(self, key: str, visible: bool) -> None:
        dock = self._dockarea_dock(key)
        if dock is None:
            return
        if visible:
            self._ensure_dock_in_stack(key)
            dock.show()
        else:
            dock.hide()

    # Section docks that intentionally live outside the main tab stack and
    # open as floating top-level windows when their side-rail button is
    # clicked. _ensure_dock_in_stack must skip these so we don't pull them
    # back into the drawer.
    _FLOATING_DOCKAREA_SECTIONS = frozenset({"temporal"})

    def _ensure_dock_in_stack(self, key: str) -> None:
        """Reattach a section dock to the main tab stack if pyqtgraph orphaned it.

        Calling dock.hide() on a tabbed pyqtgraph Dock removes the widget from
        its TabContainer; a subsequent show() returns visibility=True but
        leaves the dock in a VContainer / stray splitter with zero height. We
        detect that state and moveDock() the section back into the tab stack
        with another fixed section as the anchor.

        Floating sections (see _FLOATING_DOCKAREA_SECTIONS) are exempt: they
        live as separate top-level windows by design.
        """
        if key in self._FLOATING_DOCKAREA_SECTIONS:
            return
        dock = self._dockarea_dock(key)
        if dock is None or self._dockarea is None:
            return
        try:
            parent_cls = type(dock.parent()).__name__ if dock.parent() else ""
        except Exception:
            parent_cls = ""
        if parent_cls == "StackedWidget":
            return
        anchor = None
        for other_key in _FIXED_POST_RIGHT_TAB_ORDER:
            if other_key == key:
                continue
            other = self._dockarea_dock(other_key)
            if other is None:
                continue
            try:
                if type(other.parent()).__name__ == "StackedWidget":
                    anchor = other
                    break
            except Exception:
                continue
        if anchor is None:
            for other_key, other in self._dockarea_docks.items():
                if other_key == key or other is None:
                    continue
                try:
                    if type(other.parent()).__name__ == "StackedWidget":
                        anchor = other
                        break
                except Exception:
                    continue
        try:
            if anchor is not None:
                self._dockarea.moveDock(dock, "above", anchor)
            else:
                self._dockarea.addDock(dock, "left")
        except Exception as exc:
            _LOG.debug("ensure_dock_in_stack(%s) failed: %s", key, exc)

    def _arrange_dockarea_default(self) -> None:
        if self._dockarea is None:
            return
        setup = self._dockarea_dock("setup")
        psth = self._dockarea_dock("psth")
        spatial = self._dockarea_dock("spatial")
        sync = self._dockarea_dock("sync")
        signal = self._dockarea_dock("signal")
        behavior = self._dockarea_dock("behavior")
        export = self._dockarea_dock("export")
        if setup is None:
            return
        self._dockarea.addDock(setup, "left")
        # Keep section panels in one tab stack to avoid layout churn/floating glitches.
        if psth is not None:
            self._dockarea.addDock(psth, "above", setup)
        if sync is not None:
            self._dockarea.addDock(sync, "above", setup)
        if spatial is not None:
            self._dockarea.addDock(spatial, "above", setup)
        if export is not None:
            self._dockarea.addDock(export, "above", setup)
        if signal is not None:
            self._dockarea.addDock(signal, "above", setup)
        if behavior is not None:
            self._dockarea.addDock(behavior, "above", setup)

    def _dockarea_state_payload(self) -> Dict[str, object]:
        if self._dockarea is None:
            return {}
        try:
            return dict(self._dockarea.saveState() or {})
        except Exception:
            return {}

    def _dockarea_apply_visibility_map(self, visible_map: Dict[str, bool]) -> None:
        for key, dock in self._dockarea_docks.items():
            vis = bool(visible_map.get(key, False))
            if vis:
                dock.show()
            else:
                dock.hide()
        self._sync_section_button_states_from_docks()

    def _save_dockarea_layout_state(self) -> None:
        if self._dockarea is None:
            return
        state = self._dockarea_state_payload()
        visible = {key: bool(dock.isVisible()) for key, dock in self._dockarea_docks.items()}
        active = self._dockarea_active_key() or ""
        try:
            self._settings.setValue(_POST_DOCKAREA_STATE_KEY, json.dumps(state))
            self._settings.setValue(_POST_DOCKAREA_VISIBLE_KEY, json.dumps(visible))
            self._settings.setValue(_POST_DOCKAREA_ACTIVE_KEY, active)
            left_i = _dock_area_to_int(QtCore.Qt.DockWidgetArea.LeftDockWidgetArea, 1)
            for key, dock in self._dockarea_docks.items():
                base = f"post_section_docks/{key}"
                self._settings.setValue(f"{base}/visible", bool(dock.isVisible()))
                self._settings.setValue(f"{base}/floating", False)
                self._settings.setValue(f"{base}/area", left_i)
            self._settings.sync()
        except Exception:
            pass

    def _restore_dockarea_layout_state(self) -> None:
        if self._dockarea is None:
            return
        self._arrange_dockarea_default()
        raw_state = self._settings.value(_POST_DOCKAREA_STATE_KEY, "")
        try:
            if isinstance(raw_state, str) and raw_state.strip():
                parsed = json.loads(raw_state)
                if isinstance(parsed, dict):
                    self._dockarea.restoreState(parsed, missing="ignore", extra="bottom")
        except Exception:
            pass
        visible_map: Dict[str, bool] = {}
        raw_vis = self._settings.value(_POST_DOCKAREA_VISIBLE_KEY, "")
        try:
            if isinstance(raw_vis, str) and raw_vis.strip():
                parsed_vis = json.loads(raw_vis)
                if isinstance(parsed_vis, dict):
                    visible_map = {str(k): bool(v) for k, v in parsed_vis.items()}
        except Exception:
            visible_map = {}
        if not visible_map:
            visible_map = self._dockarea_default_visibility_map()
        self._dockarea_apply_visibility_map(visible_map)
        active = str(self._settings.value(_POST_DOCKAREA_ACTIVE_KEY, "setup") or "setup")
        dock = self._dockarea_dock(active)
        if dock is not None and dock.isVisible():
            try:
                dock.raiseDock()
            except Exception:
                pass
            self._last_opened_section = active
            self._sync_section_button_states_from_docks()
            self._update_post_drawer_visibility()
            return
        self._last_opened_section = None
        for key in _FIXED_POST_RIGHT_TAB_ORDER:
            fallback = self._dockarea_dock(key)
            if fallback is not None and fallback.isVisible():
                try:
                    fallback.raiseDock()
                except Exception:
                    pass
                self._last_opened_section = key
                break
        self._update_post_drawer_visibility()

    def _apply_fixed_dockarea_layout(self) -> None:
        if not self._use_pg_dockarea_layout:
            return
        if self._dockarea is None or not self._dockarea_docks:
            return
        self._apply_fixed_dock_features()
        visible_map = {key: bool(dock.isVisible()) for key, dock in self._dockarea_docks.items()}
        if not any(visible_map.values()):
            visible_map = self._dockarea_default_visibility_map()
        self._suspend_panel_layout_persistence = True
        try:
            self._arrange_dockarea_default()
            for key in self._dockarea_docks.keys():
                should_show = bool(visible_map.get(key, False))
                self._set_dockarea_visible(key, should_show)
            active = self._last_opened_section if bool(visible_map.get(self._last_opened_section or "", False)) else None
            if active is None:
                for key in _FIXED_POST_RIGHT_TAB_ORDER:
                    if bool(visible_map.get(key, False)):
                        active = key
                        break
            if active is None:
                for key in self._dockarea_docks.keys():
                    if bool(visible_map.get(key, False)):
                        active = key
                        break
            dock = self._dockarea_dock(active) if active else None
            if dock is not None and dock.isVisible():
                try:
                    dock.raiseDock()
                except Exception:
                    pass
            self._last_opened_section = active
            self._sync_section_button_states_from_docks()
            self._update_post_drawer_visibility()
            self._dock_layout_restored = True
        finally:
            self._suspend_panel_layout_persistence = False
        self._save_dockarea_layout_state()

    def _activate_dockarea_tab(self, key: str) -> None:
        if key not in self._dockarea_docks:
            return
        self._set_dockarea_visible(key, True)
        dock = self._dockarea_dock(key)
        if dock is not None:
            try:
                dock.raiseDock()
            except Exception:
                pass
        self._last_opened_section = key

    def _on_dockarea_dock_closed(self, key: str) -> None:
        self._set_section_button_checked(key, False)
        if self._last_opened_section == key:
            self._last_opened_section = None
        self._save_panel_layout_state()

    def _setup_section_popups(self) -> None:
        if self._use_pg_dockarea_layout:
            self._setup_dockarea_sections()
            return
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
            "sync": ("Time Synchronization", self.section_sync),
            "psth": ("PSTH", self.section_psth),
            "signal": ("Signal Event Analyzer", self.section_signal),
            "behavior": ("Behavior Analysis", self.section_behavior),
            "spatial": ("Spatial", self.section_spatial),
            "temporal": ("Temporal Modeling", self.section_temporal),
            "export": ("Export", self.section_export),
        }
        for key, (title, widget) in section_map.items():
            scroll = QtWidgets.QScrollArea()
            scroll.setWidgetResizable(True)
            scroll.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
            scroll.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
            scroll.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAsNeeded)
            self._apply_scroll_theme(scroll)
            # Keep section widgets shrinkable so dock stacks do not clip content.
            widget.setMinimumSize(0, 0)
            widget.setSizePolicy(QtWidgets.QSizePolicy.Policy.Preferred, QtWidgets.QSizePolicy.Policy.Preferred)
            scroll.setWidget(widget)
            self._section_scroll_hosts[key] = scroll

            dock = QtWidgets.QDockWidget(title, host)
            dock.setObjectName(f"post.{key}.dock")
            dock.setWindowModality(QtCore.Qt.WindowModality.NonModal)
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
            host.addDockWidget(QtCore.Qt.DockWidgetArea.LeftDockWidgetArea, dock)
            # Match preprocessing behavior: popups open floating by default.
            dock.setFloating(True)
            dock.hide()
            self._section_popups[key] = dock

        self._apply_fixed_dock_features()

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

    def _apply_fixed_dock_features(self) -> None:
        if self._use_pg_dockarea_layout:
            for key, dock in self._dockarea_docks.items():
                if dock is None or not hasattr(dock, "label"):
                    continue
                closable = not (self._force_fixed_default_layout and key in _FIXED_POST_RIGHT_SECTIONS)
                try:
                    dock.label.setClosable(bool(closable))
                except Exception:
                    pass
            return
        if not self._section_popups:
            return
        for key, dock in self._section_popups.items():
            if dock is None:
                continue
            if self._force_fixed_default_layout:
                if key in _FIXED_POST_RIGHT_SECTIONS:
                    features = QtWidgets.QDockWidget.DockWidgetFeature.NoDockWidgetFeatures
                else:
                    features = (
                        QtWidgets.QDockWidget.DockWidgetFeature.DockWidgetClosable
                        | QtWidgets.QDockWidget.DockWidgetFeature.DockWidgetMovable
                    )
                allowed = QtCore.Qt.DockWidgetArea.LeftDockWidgetArea
            else:
                features = (
                    QtWidgets.QDockWidget.DockWidgetFeature.DockWidgetClosable
                    | QtWidgets.QDockWidget.DockWidgetFeature.DockWidgetMovable
                    | QtWidgets.QDockWidget.DockWidgetFeature.DockWidgetFloatable
                )
                allowed = QtCore.Qt.DockWidgetArea.AllDockWidgetAreas
            try:
                dock.setAllowedAreas(allowed)
                dock.setFeatures(features)
            except Exception:
                pass

    _POST_SECTION_TITLES = {
        "setup": "Setup",
        "psth": "PSTH analysis",
        "spatial": "Spatial maps",
        "export": "Export",
        "signal": "Signal events",
        "behavior": "Behavior",
        "temporal": "Temporal Modeling",
    }

    def _update_post_drawer_visibility(self) -> None:
        drawer = getattr(self, "_post_drawer", None)
        if drawer is None:
            return
        any_checked = any(b.isChecked() for b in self._section_buttons.values())
        active_key = next((k for k, b in self._section_buttons.items() if b.isChecked()), None)
        # Compat title (kept hidden, used by tests).
        title_lbl = getattr(self, "_post_drawer_title", None)
        if title_lbl is not None:
            title_lbl.setText(self._POST_SECTION_TITLES.get(active_key or "", ""))
        # Rich header (badge + title + subtitle).
        header = getattr(self, "_post_drawer_header", None)
        if header is not None:
            try:
                header.set_postprocess_section(active_key or "")
            except Exception:
                pass
        drawer.setVisible(any_checked)
        splitter = self._dockarea_splitter
        if splitter is None:
            return
        try:
            sizes = splitter.sizes()
            if len(sizes) >= 2:
                if any_checked:
                    if sizes[0] < 60:
                        total = sum(sizes) or 1
                        drawer_w = max(420, int(total * 0.30))
                        sizes[0] = drawer_w
                        sizes[1] = max(420, sizes[1] - drawer_w)
                        splitter.setSizes(sizes)
                else:
                    if sizes[0] > 0:
                        sizes[1] += sizes[0]
                        sizes[0] = 0
                        splitter.setSizes(sizes)
        except Exception:
            pass

    def _force_hide_post_drawer_initially(self) -> None:
        for key, btn in self._section_buttons.items():
            if btn.isChecked():
                blocked = btn.blockSignals(True)
                try:
                    btn.setChecked(False)
                finally:
                    btn.blockSignals(blocked)
            try:
                if self._use_pg_dockarea_layout:
                    self._set_dockarea_visible(key, False)
            except Exception:
                pass
        drawer = getattr(self, "_post_drawer", None)
        if drawer is not None:
            drawer.setVisible(False)
        if self._dockarea_splitter is not None:
            try:
                sizes = self._dockarea_splitter.sizes()
                if len(sizes) >= 2:
                    sizes[1] += sizes[0]
                    sizes[0] = 0
                    self._dockarea_splitter.setSizes(sizes)
            except Exception:
                pass

    def _toggle_section_popup(self, key: str, checked: bool) -> None:
        if self._use_pg_dockarea_layout:
            self._setup_dockarea_sections()
            dock = self._dockarea_dock(key)
            if dock is None:
                return
            if checked:
                # Radio: hide siblings.
                for other_key, other_btn in self._section_buttons.items():
                    if other_key == key:
                        continue
                    if other_btn.isChecked():
                        blocked = other_btn.blockSignals(True)
                        try:
                            other_btn.setChecked(False)
                        finally:
                            other_btn.blockSignals(blocked)
                    self._set_dockarea_visible(other_key, False)
                self._activate_dockarea_tab(key)
            else:
                self._set_dockarea_visible(key, False)
            self._update_post_drawer_visibility()
            self._sync_section_button_states_from_docks()
            self._save_panel_layout_state()
            return
        if not self._section_popups:
            self._setup_section_popups()
        dock = self._section_popups.get(key)
        if dock is None:
            return
        host = self._dock_host or self._dock_main_window()
        tabs = getattr(host, "tabs", None) if host is not None else None
        post_active = not isinstance(tabs, QtWidgets.QTabWidget) or tabs.currentWidget() is self
        fixed_required = (
            self._force_fixed_default_layout
            and key in _FIXED_POST_RIGHT_SECTIONS
            and post_active
            and not self._post_docks_hidden_for_tab_switch
        )
        if fixed_required:
            self._activate_fixed_right_tab(key)
            self._save_panel_layout_state()
            return
        if checked:
            # Keep user-selected docking mode. Only reposition if currently floating off-screen.
            if dock.isFloating():
                if key not in self._section_popup_initialized or not self._is_popup_on_screen(dock):
                    self._position_section_popup(dock, key)
                    self._section_popup_initialized.add(key)
            dock.show()
            dock.raise_()
            if key != "temporal":
                dock.activateWindow()
            self._last_opened_section = key
        else:
            dock.hide()
        self._save_panel_layout_state()

    def _on_section_popup_visibility(self, key: str, visible: bool) -> None:
        if self._use_pg_dockarea_layout:
            self._set_section_button_checked(key, visible)
            if visible:
                self._last_opened_section = key
            elif self._last_opened_section == key:
                self._last_opened_section = None
            self._save_panel_layout_state()
            return
        host = self._dock_host or self._dock_main_window()
        tabs = getattr(host, "tabs", None) if host is not None else None
        post_active = not isinstance(tabs, QtWidgets.QTabWidget) or tabs.currentWidget() is self
        fixed_required = (
            self._force_fixed_default_layout
            and key in _FIXED_POST_RIGHT_SECTIONS
            and post_active
            and not self._post_docks_hidden_for_tab_switch
        )
        if fixed_required and not visible:
            # Spatial/PSTH are represented as tabs inside Setup dock in fixed mode.
            if key == "setup":
                self._activate_fixed_right_tab("setup")
            else:
                self._set_section_button_checked(key, True)
            self._save_panel_layout_state()
            return
        self._set_section_button_checked(key, visible)
        if visible:
            self._last_opened_section = key
        elif self._last_opened_section == key:
            self._last_opened_section = None
        self._save_panel_layout_state()

    def _hide_all_section_popups(self) -> None:
        if self._use_pg_dockarea_layout:
            self._setup_dockarea_sections()
            for key in self._dockarea_docks.keys():
                self._set_dockarea_visible(key, False)
                self._set_section_button_checked(key, False)
            self._last_opened_section = None
            self._update_post_drawer_visibility()
            self._save_panel_layout_state()
            return
        host = self._dock_host or self._dock_main_window()
        if self._force_fixed_default_layout:
            if host is not None:
                self._apply_fixed_right_tabs_as_single_dock(host, active_key="setup")
            for key in ("signal", "behavior", "export"):
                dock = self._section_popups.get(key)
                if dock is None:
                    continue
                dock.hide()
                self._set_section_button_checked(key, False)
            self._last_opened_section = "setup"
            self._save_panel_layout_state()
            return
        for key, dock in self._section_popups.items():
            keep_visible = self._force_fixed_default_layout and key in _FIXED_POST_RIGHT_SECTIONS
            if keep_visible:
                dock.blockSignals(True)
                try:
                    if host is not None:
                        host.addDockWidget(QtCore.Qt.DockWidgetArea.LeftDockWidgetArea, dock)
                    dock.setFloating(False)
                    dock.show()
                    dock.raise_()
                finally:
                    dock.blockSignals(False)
                self._set_section_button_checked(key, True)
            else:
                dock.hide()
                self._set_section_button_checked(key, False)
        if self._force_fixed_default_layout:
            self._last_opened_section = "setup"
        else:
            self._last_opened_section = None
        self._save_panel_layout_state()

    def _default_popup_size(self, key: str) -> Tuple[int, int]:
        size_map = {
            "setup": (420, 620),
            "sync": (620, 820),
            "psth": (420, 640),
            "signal": (420, 640),
            "behavior": (500, 620),
            "spatial": (420, 520),
            "temporal": (1320, 880),
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

        if x_left >= x_min:
            x = x_left
        elif x_right <= x_max:
            x = x_right
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
        default: QtCore.Qt.DockWidgetArea = QtCore.Qt.DockWidgetArea.LeftDockWidgetArea,
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
        if self._use_pg_dockarea_layout:
            self._last_opened_section = None
            for key, dock in self._dockarea_docks.items():
                visible = bool(dock.isVisible())
                self._set_section_button_checked(key, visible)
                if visible and self._last_opened_section is None:
                    self._last_opened_section = key
            return
        self._last_opened_section = None
        for key, dock in self._section_popups.items():
            visible = bool(dock.isVisible())
            if self._force_fixed_default_layout and key in _FIXED_POST_RIGHT_SECTIONS:
                # In fixed mode these sections are always present as left-side tabs.
                visible = True
            self._set_section_button_checked(key, visible)
            if visible:
                self._last_opened_section = key

    def _has_saved_layout_state(self) -> bool:
        if self._use_pg_dockarea_layout:
            try:
                return bool(self._settings.contains(_POST_DOCKAREA_STATE_KEY) or self._settings.contains(_POST_DOCKAREA_VISIBLE_KEY))
            except Exception:
                return False
        try:
            if self._settings.contains(_POST_DOCK_STATE_KEY):
                return True
            keys = list(self._section_popups.keys()) or ["setup", "psth", "signal", "behavior", "spatial", "export"]
            for key in keys:
                if self._settings.contains(f"post_section_docks/{key}/visible"):
                    return True
        except Exception:
            pass
        return False

    def _save_panel_layout_state(self) -> None:
        if not self._panel_layout_persistence_ready:
            return
        if self._use_pg_dockarea_layout:
            if self._is_restoring_panel_layout or self._suspend_panel_layout_persistence:
                return
            self._save_dockarea_layout_state()
            return
        if self._force_fixed_default_layout:
            self._persist_fixed_post_default_state()
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
                left_i = _dock_area_to_int(QtCore.Qt.DockWidgetArea.LeftDockWidgetArea, 1)
                area_val = _dock_area_to_int(cached.get("area", host.dockWidgetArea(dock)), left_i)
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
        if self._use_pg_dockarea_layout:
            return
        if self._force_fixed_default_layout:
            self._persist_fixed_post_default_state()
            return
        if not self._post_docks_hidden_for_tab_switch:
            return
        left_i = _dock_area_to_int(QtCore.Qt.DockWidgetArea.LeftDockWidgetArea, 1)
        if not self._section_popups:
            return
        for key in self._section_popups.keys():
            try:
                state = self._post_section_state_before_hide.get(key, {})
                base = f"post_section_docks/{key}"
                self._settings.setValue(f"{base}/visible", bool(state.get("visible", False)))
                self._settings.setValue(f"{base}/floating", bool(state.get("floating", True)))
                self._settings.setValue(f"{base}/area", _dock_area_to_int(state.get("area", left_i), left_i))
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
        if self._use_pg_dockarea_layout:
            self._save_panel_layout_state()
            return
        if self._post_docks_hidden_for_tab_switch:
            self._persist_hidden_layout_state_from_cache()
            return
        self._save_panel_layout_state()

    def persist_layout_state_snapshot(self) -> None:
        """
        Persist post dock state safely.
        Uses cached tab-switch state while hidden, otherwise captures current host topology.
        """
        if self._use_pg_dockarea_layout:
            self._save_panel_layout_state()
            return
        if self._force_fixed_default_layout:
            self._persist_fixed_post_default_state()
            return
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
        if self._use_pg_dockarea_layout:
            self._setup_dockarea_sections()
            if self._force_fixed_default_layout:
                self._apply_fixed_dockarea_layout()
            else:
                self._restore_dockarea_layout_state()
            return
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
                    _dock_area_to_int(QtCore.Qt.DockWidgetArea.LeftDockWidgetArea, 1),
                )
                area = self._dock_area_from_settings(area_val, QtCore.Qt.DockWidgetArea.LeftDockWidgetArea)
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
        if not self._autosave_restoring:
            self._project_dirty = True
        # update trace preview with first entry
        self._refresh_behavior_list()
        self._refresh_sync_sources()
        self._set_resample_from_processed()
        self._update_trace_preview()
        self._compute_spatial_heatmap()
        self._update_data_availability()
        self._update_status_strip()
        if self._pending_project_recompute_from_current:
            self._pending_project_recompute_from_current = False
            self._compute_psth()
            self._compute_spatial_heatmap()

    def append_processed(self, processed_list: List[ProcessedTrial]) -> None:
        if not processed_list:
            return
        self._processed.extend(processed_list)
        if not self._autosave_restoring:
            self._project_dirty = True
        self._refresh_behavior_list()
        self._refresh_sync_sources()
        self._set_resample_from_processed()
        self._update_trace_preview()
        self._compute_spatial_heatmap()
        self._update_data_availability()
        self._update_status_strip()

    @QtCore.Slot(list)
    def receive_dio_list(self, dio_list: List[str]) -> None:
        self._known_dio_channels = [str(d) for d in (dio_list or []) if str(d)]
        self.combo_dio.clear()
        for d in dio_list or []:
            self.combo_dio.addItem(d)
        self._refresh_sync_fiber_sources()
        self._update_status_strip()

    @QtCore.Slot(str, str, object, object)
    def receive_dio_data(self, path: str, dio_name: str, t: Optional[np.ndarray], x: Optional[np.ndarray]) -> None:
        if t is None or x is None:
            return
        self._dio_cache[(path, dio_name)] = (np.asarray(t, float), np.asarray(x, float))

    def _load_recent_paths(self, key: str) -> List[str]:
        raw = self._settings.value(key, "[]", type=str)
        try:
            data = json.loads(raw) if raw else []
        except Exception:
            data = []
        out: List[str] = []
        if isinstance(data, list):
            for item in data:
                p = str(item or "").strip()
                if p:
                    out.append(p)
        return out

    def _save_recent_paths(self, key: str, paths: List[str]) -> None:
        try:
            self._settings.setValue(key, json.dumps(paths))
        except Exception:
            pass

    def _push_recent_paths(self, key: str, paths: List[str], max_items: int = 15) -> None:
        if not paths:
            return
        existing = self._load_recent_paths(key)
        merged: List[str] = []
        for p in paths:
            sp = str(p or "").strip()
            if not sp:
                continue
            if sp in merged:
                continue
            merged.append(sp)
        for p in existing:
            if p not in merged:
                merged.append(p)
        self._save_recent_paths(key, merged[:max_items])

    def _prune_recent_paths(self, key: str) -> None:
        recent = self._load_recent_paths(key)
        kept = [p for p in recent if os.path.isfile(p)]
        self._save_recent_paths(key, kept)

    def _refresh_recent_postprocessing_menus(self) -> None:
        self._refresh_recent_menu(
            self.menu_recent_processed,
            key="postprocess_recent_processed_paths",
            loader=self._load_recent_processed_path,
        )
        self._refresh_recent_menu(
            self.menu_recent_behavior,
            key="postprocess_recent_behavior_paths",
            loader=self._load_recent_behavior_path,
        )
        self._refresh_recent_menu(
            self.menu_recent_projects,
            key="postprocess_recent_project_paths",
            loader=self._load_recent_project_path,
        )

    def _refresh_recent_menu(
        self,
        menu: QtWidgets.QMenu,
        key: str,
        loader: Callable[[str], None],
    ) -> None:
        if menu is None:
            return
        menu.clear()
        recent = self._load_recent_paths(key)
        if not recent:
            act_empty = menu.addAction("(No recent files)")
            act_empty.setEnabled(False)
            return

        missing: List[str] = []
        for path in recent:
            label = os.path.basename(path) or path
            if not os.path.isfile(path):
                label = f"{label} (missing)"
            act = menu.addAction(label)
            act.setToolTip(path)
            act.setEnabled(os.path.isfile(path))
            if os.path.isfile(path):
                act.triggered.connect(lambda _checked=False, p=path: loader(p))
            else:
                missing.append(path)
        menu.addSeparator()
        act_clear = menu.addAction("Clear recent")
        act_clear.triggered.connect(lambda: self._save_recent_paths(key, []))
        if missing:
            act_prune = menu.addAction("Remove missing")
            act_prune.triggered.connect(lambda: self._prune_recent_paths(key))

    def _load_recent_processed_path(self, path: str) -> None:
        if not path or not os.path.isfile(path):
            QtWidgets.QMessageBox.warning(self, "Load recent", "Selected recent processed file is missing.")
            return
        self._load_processed_paths([path], replace=True)
        try:
            self._settings.setValue("postprocess_last_dir", os.path.dirname(path))
        except Exception:
            pass

    def _load_recent_behavior_path(self, path: str) -> None:
        if not path or not os.path.isfile(path):
            QtWidgets.QMessageBox.warning(self, "Load recent", "Selected recent behavior file is missing.")
            return
        self._load_behavior_paths([path], replace=True)
        self._refresh_behavior_list()
        try:
            self._settings.setValue("postprocess_last_dir", os.path.dirname(path))
        except Exception:
            pass

    def _load_recent_project_path(self, path: str) -> None:
        if not path or not os.path.isfile(path):
            QtWidgets.QMessageBox.warning(self, "Load recent", "Selected recent project file is missing.")
            return
        self._load_project_from_path(path)

    def _load_behavior_paths(self, paths: List[str], replace: bool) -> None:
        if replace:
            self._behavior_sources.clear()
        parse_mode = self._current_behavior_parse_mode()
        fps = float(self.spin_behavior_fps.value()) if hasattr(self, "spin_behavior_fps") else 0.0
        loaded_any = False
        for p in paths:
            stem = os.path.splitext(os.path.basename(p))[0]
            ext = os.path.splitext(p)[1].lower()
            try:
                if ext == ".csv":
                    info = _load_behavior_csv(p, parse_mode=parse_mode, fps=fps)
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
                    info = _load_behavior_ethovision(p, sheet_name=sheet, parse_mode=parse_mode, fps=fps)
                else:
                    continue
                has_behaviors = bool(info.get("behaviors") or {})
                has_trajectory = bool(info.get("trajectory") or {})
                if not has_behaviors and not has_trajectory:
                    QtWidgets.QMessageBox.warning(
                        self,
                        "Behavior load warning",
                        f"No behavior or trajectory numeric columns detected in {os.path.basename(p)} for the selected file type.",
                    )
                info["source_path"] = str(p)
                self._behavior_sources[stem] = info
                loaded_any = True
            except Exception as exc:
                QtWidgets.QMessageBox.warning(
                    self,
                    "Behavior load failed",
                    f"Could not load {os.path.basename(p)}:\n{exc}",
                )
                continue
        mode_label = "timestamps" if parse_mode == _BEHAVIOR_PARSE_TIMESTAMPS else "binary"
        self.lbl_beh.setText(f"{len(self._behavior_sources)} file(s) loaded [{mode_label}]")
        self._update_behavior_time_panel()
        self._push_recent_paths("postprocess_recent_behavior_paths", paths)
        if loaded_any and not self._autosave_restoring:
            self._project_dirty = True
        self._update_data_availability()
        self._update_status_strip()
        self._sync_temporal_modeling_context()
        self._refresh_sync_sources()

    def _sync_temporal_modeling_context(self) -> None:
        if not hasattr(self, "section_temporal"):
            return
        try:
            self.section_temporal.set_data(
                processed_trials=self._processed,
                psth_mat=self._last_mat,
                psth_tvec=self._last_tvec,
                event_times=self._last_events,
                file_ids=self._all_file_ids,
                per_file_mats=self._per_file_mats,
                behavior_sources=self._behavior_sources,
                event_rows=self._last_event_rows,
                group_mat=self._group_mat,
                group_tvec=self._group_tvec,
                group_labels=self._group_labels,
                visual_mode=int(self.tab_visual_mode.currentIndex()) if hasattr(self, "tab_visual_mode") else 0,
                group_mode=bool(self.tab_sources.currentIndex() == 1) if hasattr(self, "tab_sources") else False,
            )
        except Exception:
            _LOG.debug("Could not sync temporal modeling context", exc_info=True)

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
        if not self._autosave_restoring:
            self._project_dirty = True
        # File to surface as the newly active selection. On replace, that's the
        # first of the loaded batch; on append, the most recently added.
        target_trial = loaded[0] if replace else loaded[-1]
        target_id = os.path.splitext(os.path.basename(target_trial.path))[0] if target_trial.path else ""
        self._pending_active_file_id = target_id

        self.lbl_group.setText(f"{len(self._processed)} file(s) loaded")
        self._push_recent_paths("postprocess_recent_processed_paths", paths)
        self._update_file_lists()
        self._refresh_sync_sources()
        self._set_resample_from_processed()
        self._compute_psth()
        self._compute_spatial_heatmap()
        self._update_data_availability()
        self._update_status_strip()
        # _compute_psth -> _refresh_individual_file_combo picks up the pending id.
        self._pending_active_file_id = ""

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
        for w in (
            self.lbl_dio_channel,
            self.combo_dio,
            self.lbl_dio_polarity,
            self.combo_dio_polarity,
            self.lbl_dio_align,
            self.combo_dio_align,
        ):
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
        show_time_panel = bool(use_beh and self._behavior_sources_need_generated_time())
        self.grp_behavior_time.setEnabled(show_time_panel)
        self.grp_behavior_time.setVisible(show_time_panel)

        # Behavior align combo + transition settings
        self.combo_behavior_align.setEnabled(use_beh)
        self.combo_behavior_align.setVisible(use_beh)
        is_transition = use_beh and self.combo_behavior_align.currentText().startswith("Transition")
        for w in (
            self.combo_behavior_from,
            self.combo_behavior_to,
            self.spin_transition_gap,
            self.lbl_trans_from,
            self.lbl_trans_to,
            self.lbl_trans_gap,
        ):
            w.setVisible(is_transition)
            w.setEnabled(is_transition)

    def _behavior_sources_need_generated_time(self) -> bool:
        for info in self._behavior_sources.values():
            if bool(info.get("needs_generated_time", False)):
                return True
        return False

    def _update_behavior_time_panel(self) -> None:
        need_time = self._behavior_sources_need_generated_time()
        count = sum(1 for info in self._behavior_sources.values() if bool(info.get("needs_generated_time", False)))
        if need_time:
            self.lbl_behavior_time_hint.setText(
                f"No time column detected in {count} behavior file(s). Generate time from FPS."
            )
        else:
            self.lbl_behavior_time_hint.setText("No missing time columns detected.")
        self._update_align_ui()

    def _apply_behavior_time_settings(self) -> None:
        fps = float(self.spin_behavior_fps.value()) if hasattr(self, "spin_behavior_fps") else 0.0
        updated = False
        for info in self._behavior_sources.values():
            if not bool(info.get("needs_generated_time", False)):
                continue
            row_count = int(info.get("row_count", 0) or 0)
            t = _generated_time_array(row_count, fps)
            info["time"] = t.copy()
            traj_t = np.asarray(info.get("trajectory_time", np.array([], float)), float)
            if traj_t.size == 0:
                info["trajectory_time"] = t.copy()
            updated = True
        if not updated:
            return
        if not self._autosave_restoring:
            self._project_dirty = True
        self._refresh_behavior_list()
        self._compute_psth()
        self._compute_spatial_heatmap()
        self._update_behavior_time_panel()
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
        self._refresh_signal_overlay()

    def _on_signal_file_changed(self, _index: int = 0) -> None:
        if not hasattr(self, "combo_signal_file"):
            return
        if self.combo_signal_source.currentText().startswith("Use PSTH input trace"):
            self._refresh_signal_overlay()
            return
        if self.combo_signal_scope.currentText() != "Per file":
            self._refresh_signal_overlay()
            return

        file_id = self.combo_signal_file.currentText().strip()
        if not file_id:
            self._refresh_signal_overlay()
            return
        try:
            idx = self.combo_individual_file.findText(file_id)
            if idx >= 0 and self.combo_individual_file.currentIndex() != idx:
                self.combo_individual_file.setCurrentIndex(idx)
            else:
                self._update_trace_preview()
        except Exception:
            self._update_trace_preview()
        self._refresh_signal_overlay()

    def _current_signal_overlay_file_id(self) -> str:
        if self.combo_signal_source.currentText().startswith("Use PSTH input trace"):
            return "psth_trace"
        if self.combo_signal_scope.currentText() == "Per file":
            file_id = self.combo_signal_file.currentText().strip()
            if file_id:
                return file_id
        try:
            if self.tab_visual_mode.currentIndex() == 0:
                file_id = self.combo_individual_file.currentText().strip()
                if file_id:
                    return file_id
        except Exception:
            pass
        if self._processed:
            proc = self._processed[0]
            return os.path.splitext(os.path.basename(proc.path))[0] if proc.path else "import"
        return ""

    def _refresh_individual_file_combo(self) -> None:
        if not hasattr(self, "combo_individual_file"):
            return
        prev = self.combo_individual_file.currentText().strip()
        # A `_pending_active_file_id` set by file loading wins over the previous selection.
        pending = str(getattr(self, "_pending_active_file_id", "") or "").strip()
        self.combo_individual_file.blockSignals(True)
        self.combo_individual_file.clear()
        for fid in self._all_file_ids:
            self.combo_individual_file.addItem(fid)

        chosen_idx = -1
        if pending:
            chosen_idx = self.combo_individual_file.findText(pending)
        if chosen_idx < 0 and prev:
            chosen_idx = self.combo_individual_file.findText(prev)
        if chosen_idx < 0 and self.combo_individual_file.count() > 0:
            chosen_idx = 0  # default to first file when nothing else matches
        if chosen_idx >= 0:
            self.combo_individual_file.setCurrentIndex(chosen_idx)
        self.combo_individual_file.blockSignals(False)

        # Fire the slot so downstream (PSTH heatmap, Temporal panel) follows.
        if chosen_idx >= 0:
            try:
                self._on_individual_file_changed(chosen_idx)
            except Exception:
                pass

        # Show file selector only in Individual mode
        is_individual = self.tab_visual_mode.currentIndex() == 0
        self.combo_individual_file.setVisible(is_individual)

    def _on_visual_mode_changed(self, index: int) -> None:
        is_individual = index == 0
        self.combo_individual_file.setVisible(is_individual)
        self._rerender_visual_from_cache()
        # Mirror to the Temporal Modeling scope: Individual -> Active file, Group -> Per-file batch.
        try:
            section = getattr(self, "section_temporal", None)
            if section is not None and hasattr(section, "combo_fit_scope"):
                target = "active" if is_individual else "batch"
                idx = section.combo_fit_scope.findData(target)
                if idx >= 0 and section.combo_fit_scope.currentIndex() != idx:
                    section.combo_fit_scope.blockSignals(True)
                    section.combo_fit_scope.setCurrentIndex(idx)
                    section.combo_fit_scope.blockSignals(False)
                    section._fit_mode = target
                    section._update_fit_state_label()
        except Exception:
            pass

    def _on_individual_file_changed(self, _index: int = 0) -> None:
        if self.tab_visual_mode.currentIndex() == 0:
            self._rerender_visual_from_cache()
        # Push the new active file into the Temporal Modeling section so its
        # scope strip stays in sync with the global Individual file picker.
        try:
            file_id = self.combo_individual_file.currentText().strip()
            section = getattr(self, "section_temporal", None)
            if section is None or not file_id:
                return
            combo = getattr(section, "combo_active_file", None)
            if combo is not None:
                idx = combo.findData(file_id)
                if idx < 0:
                    idx = combo.findText(file_id)
                if idx >= 0 and combo.currentIndex() != idx:
                    combo.blockSignals(True)
                    combo.setCurrentIndex(idx)
                    combo.blockSignals(False)
                    section._active_file_id = file_id
                    section._on_active_file_changed()
        except Exception:
            pass

    def _psth_min_events_per_animal(self) -> int:
        spin = getattr(self, "spin_min_events_per_animal", None)
        if spin is None:
            return 1
        try:
            return max(1, int(spin.value()))
        except Exception:
            return 1

    def _psth_exclude_low_event_animals_enabled(self) -> bool:
        chk = getattr(self, "cb_exclude_low_event_animals", None)
        return bool(chk is not None and chk.isChecked())

    def _psth_group_trial_view_enabled(self) -> bool:
        chk = getattr(self, "cb_group_keep_trials", None)
        return bool(chk is not None and chk.isChecked())

    def _stack_psth_trial_rows(
        self,
        per_file_mats: Optional[Dict[str, Tuple[np.ndarray, np.ndarray]]] = None,
        per_file_labels: Optional[Dict[str, List[str]]] = None,
        file_ids_order: Optional[List[str]] = None,
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], List[str]]:
        mats_by_file = per_file_mats if per_file_mats is not None else self._per_file_mats
        labels_by_file = per_file_labels if per_file_labels is not None else self._per_file_labels
        ordered_ids = list(file_ids_order or self._all_file_ids or mats_by_file.keys())
        rows: List[np.ndarray] = []
        labels: List[str] = []
        tvec_ref: Optional[np.ndarray] = None
        min_cols: Optional[int] = None

        for file_id in ordered_ids:
            if file_id not in mats_by_file:
                continue
            tvec, mat = mats_by_file[file_id]
            arr = np.asarray(mat, float)
            if arr.ndim != 2 or arr.shape[0] == 0 or arr.shape[1] == 0:
                continue
            if tvec_ref is None:
                tvec_ref = np.asarray(tvec, float).copy()
            min_cols = arr.shape[1] if min_cols is None else min(min_cols, arr.shape[1])
            base_labels = labels_by_file.get(file_id, [])
            for row_idx in range(arr.shape[0]):
                label = base_labels[row_idx] if row_idx < len(base_labels) else f"Trial {row_idx + 1}"
                labels.append(f"{file_id} | {label}")
            rows.append(arr)

        if not rows or tvec_ref is None or min_cols is None or min_cols <= 0:
            return None, None, []
        stacked = np.vstack([row[:, :min_cols] for row in rows])
        return stacked, tvec_ref[:min_cols], labels[: stacked.shape[0]]

    def _rerender_visual_from_cache(self) -> None:
        visual_mode = self.tab_visual_mode.currentIndex()
        if visual_mode == 1:
            # Group view
            if self._psth_group_trial_view_enabled():
                mat = self._group_trial_mat
                tvec = self._group_trial_tvec
                labels = list(self._group_trial_labels)
                if mat is None or tvec is None:
                    mat, tvec, labels = self._stack_psth_trial_rows()
                if mat is not None and tvec is not None:
                    self._last_psth_display_level = "trials"
                    self._last_display_labels = labels
                    self._render_heatmap(mat, tvec, labels=labels)
                    self._render_avg(mat, tvec)
                    self._render_metrics(mat, tvec)
                    self._last_mat = mat
                    self._last_tvec = tvec
                    self.lbl_plot_file.setText(f"Group trials: {mat.shape[0]} trial(s)")
            elif self._group_mat is not None and self._group_tvec is not None:
                self._last_psth_display_level = "animals"
                self._last_display_labels = list(self._group_labels)
                self._render_heatmap(self._group_mat, self._group_tvec, labels=self._group_labels)
                self._render_avg(self._group_mat, self._group_tvec)
                self._render_metrics(self._group_mat, self._group_tvec)
                self._last_mat = self._group_mat
                self._last_tvec = self._group_tvec
                self.lbl_plot_file.setText(f"Group: {len(self._group_labels)} animal(s)")
        else:
            # Individual view
            sel_id = self.combo_individual_file.currentText().strip()
            if not sel_id and self._all_file_ids:
                sel_id = self._all_file_ids[0]
            if sel_id and sel_id in self._per_file_mats:
                tvec, mat = self._per_file_mats[sel_id]
                labels = self._per_file_labels.get(sel_id, [])
                self._last_psth_display_level = "trials"
                self._last_display_labels = list(labels)
                self._render_heatmap(mat, tvec, labels=labels)
                self._render_avg(mat, tvec)
                self._render_metrics(mat, tvec)
                self._last_mat = mat
                self._last_tvec = tvec
                self.lbl_plot_file.setText(f"File: {sel_id}")
                self.plot_avg.setTitle("Average across trials +/- SEM")
        # Always refresh the trace preview to match the selected file
        self._update_trace_preview()
        self._sync_temporal_modeling_context()

    def _update_data_availability(self) -> None:
        has_processed = bool(self._processed)
        has_behavior = bool(self._behavior_sources)
        spatial_ready = has_processed and has_behavior
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
            self.cb_peak_auto_mad,
            self.spin_peak_mad_multiplier,
            self.spin_peak_height,
            self.spin_peak_distance,
            self.spin_peak_smooth,
            self.combo_peak_baseline,
            self.spin_peak_baseline_window,
            self.cb_peak_norm_prominence,
            self.spin_peak_rate_bin,
            self.spin_peak_auc_window,
            self.cb_peak_overlay,
            self.cb_peak_noise_overlay,
        ):
            w.setEnabled(has_processed)
        self._update_peak_auto_mad_enabled(queue=False)
        # Surface baseline-source widget visibility + status preview.
        try:
            self._update_signal_baseline_source_visibility()
        except Exception:
            pass
        for w in (
            self.btn_compute_behavior,
            self.btn_export_behavior_metrics,
            self.btn_export_behavior_events,
            self.combo_behavior_analysis,
            self.spin_behavior_bin,
            self.cb_behavior_aligned,
        ):
            w.setEnabled(has_behavior)
        for w in (
            self.combo_spatial_x,
            self.combo_spatial_y,
            self.spin_spatial_bins_x,
            self.spin_spatial_bins_y,
            self.combo_spatial_weight,
            self.cb_spatial_clip,
            self.spin_spatial_clip_low,
            self.spin_spatial_clip_high,
            self.cb_spatial_time_filter,
            self.spin_spatial_time_min,
            self.spin_spatial_time_max,
            self.spin_spatial_smooth,
            self.combo_spatial_activity_mode,
            self.cb_spatial_log,
            self.cb_spatial_invert_y,
            self.btn_compute_spatial,
            self.btn_export_spatial_img,
        ):
            w.setEnabled(spatial_ready)
        self.btn_spatial_help.setEnabled(True)
        self._update_spatial_clip_enabled()
        self._update_spatial_time_filter_enabled()
        self._refresh_signal_file_combo()
        has_aligned_time = any(
            getattr(proc, "sync_aligned_time", None) is not None
            and np.asarray(getattr(proc, "sync_aligned_time"), float).size == np.asarray(getattr(proc, "time", []), float).size
            for proc in (self._processed or [])
        )
        sync_ready = has_processed and has_behavior
        for w in (
            self.combo_sync_behavior_file,
            self.combo_sync_camera_column,
            self.combo_sync_camera_mode,
            self.combo_sync_fiber_source,
            self.combo_sync_fiber_mode,
            self.combo_sync_method,
            self.cb_sync_auto_threshold,
            self.spin_sync_threshold,
            self.spin_sync_min_interval,
            self.spin_sync_max_offset,
            self.cb_sync_auto_recompute,
            self.btn_sync_preview,
            self.btn_sync_apply_selected,
            self.btn_sync_apply_batch,
        ):
            w.setEnabled(sync_ready)
        self.cb_sync_use_aligned.setEnabled(has_aligned_time)
        self.combo_sync_export_format.setEnabled(has_aligned_time)
        self.btn_sync_export.setEnabled(has_aligned_time)
        self.btn_sync_refresh.setEnabled(True)
        self._on_sync_auto_threshold_toggled(self.cb_sync_auto_threshold.isChecked())

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
        self._queue_settings_save()

    def _update_psth_inclusion_controls(self, *_args: object) -> None:
        enabled = self._psth_exclude_low_event_animals_enabled()
        self.spin_min_events_per_animal.setEnabled(enabled)
        self._queue_settings_save()

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
        self._queue_settings_save()

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
        self._queue_settings_save()

    def _update_peak_auto_mad_enabled(self, _checked: object = None, *, queue: bool = True) -> None:
        has_processed = bool(self._processed)
        auto_mad = bool(self.cb_peak_auto_mad.isChecked())
        self.spin_peak_prominence.setEnabled(has_processed and not auto_mad)
        self.spin_peak_mad_multiplier.setEnabled(has_processed and auto_mad)
        if queue:
            self._queue_settings_save()

    # ------------------------------------------------------------------
    # Baseline-prominence UI plumbing
    # ------------------------------------------------------------------

    def _update_signal_baseline_source_visibility(self, _idx: int = 0) -> None:
        """Show only the controls relevant to the chosen baseline source mode."""
        if not hasattr(self, "combo_signal_baseline_source"):
            return
        mode = self.combo_signal_baseline_source.currentData() or "window"
        is_window = (mode == "window")
        is_exclude = (mode == "exclude")
        norm_on = bool(self.cb_peak_norm_prominence.isChecked())

        # Window inputs
        for w, lbl_row_widget in self._signal_baseline_window_rows():
            w.setVisible(is_window and norm_on)
            if lbl_row_widget is not None:
                lbl_row_widget.setVisible(is_window and norm_on)
        # Exclude inputs
        for w, lbl_row_widget in self._signal_baseline_exclude_rows():
            w.setVisible(is_exclude and norm_on)
            if lbl_row_widget is not None:
                lbl_row_widget.setVisible(is_exclude and norm_on)

        # The source dropdown itself is only meaningful when normalization is on.
        for attr in ("combo_signal_baseline_source",):
            w = getattr(self, attr, None)
            if w is not None:
                w.setEnabled(norm_on)
                # Find form row label for it via parent form layout - just enable.

        # Refresh the live status preview.
        self._refresh_signal_baseline_status()

    def _form_row_label_widget(self, field: QtWidgets.QWidget) -> Optional[QtWidgets.QWidget]:
        """Locate the QLabel paired with `field` inside the enclosing QFormLayout."""
        parent = field.parentWidget()
        if parent is None:
            return None
        layout = parent.layout()
        if not isinstance(layout, QtWidgets.QFormLayout):
            return None
        for r in range(layout.rowCount()):
            li = layout.itemAt(r, QtWidgets.QFormLayout.ItemRole.FieldRole)
            if li is not None and li.widget() is field:
                lr = layout.itemAt(r, QtWidgets.QFormLayout.ItemRole.LabelRole)
                if lr is not None:
                    return lr.widget()
        return None

    def _signal_baseline_window_rows(self) -> List[Tuple[QtWidgets.QWidget, Optional[QtWidgets.QWidget]]]:
        return [
            (
                self._signal_baseline_window_wrap,
                self._form_row_label_widget(self._signal_baseline_window_wrap),
            )
        ]

    def _signal_baseline_exclude_rows(self) -> List[Tuple[QtWidgets.QWidget, Optional[QtWidgets.QWidget]]]:
        return [
            (
                self.combo_signal_baseline_behavior_file,
                self._form_row_label_widget(self.combo_signal_baseline_behavior_file),
            ),
            (
                self.list_signal_baseline_exclude,
                self._form_row_label_widget(self.list_signal_baseline_exclude),
            ),
            (
                self.spin_signal_baseline_pad,
                self._form_row_label_widget(self.spin_signal_baseline_pad),
            ),
        ]

    def _refresh_signal_baseline_behavior_lists(self) -> None:
        """Rebuild the behavior-file combo and the exclude-bouts list from the
        loaded `_behavior_sources`. Preserves prior selections where possible."""
        if not hasattr(self, "combo_signal_baseline_behavior_file"):
            return
        prev_stem = self.combo_signal_baseline_behavior_file.currentData()
        ticked_prev: set = set()
        for i in range(self.list_signal_baseline_exclude.count()):
            it = self.list_signal_baseline_exclude.item(i)
            if it.checkState() == QtCore.Qt.CheckState.Checked:
                ticked_prev.add(it.data(QtCore.Qt.ItemDataRole.UserRole) or it.text())

        self.combo_signal_baseline_behavior_file.blockSignals(True)
        try:
            self.combo_signal_baseline_behavior_file.clear()
            self.combo_signal_baseline_behavior_file.addItem("Auto-match by file id", "")
            for stem in (self._behavior_sources or {}).keys():
                self.combo_signal_baseline_behavior_file.addItem(str(stem), str(stem))
            # Restore selection
            if isinstance(prev_stem, str) and prev_stem:
                idx = self.combo_signal_baseline_behavior_file.findData(prev_stem)
                if idx >= 0:
                    self.combo_signal_baseline_behavior_file.setCurrentIndex(idx)
        finally:
            self.combo_signal_baseline_behavior_file.blockSignals(False)

        self._rebuild_signal_baseline_exclude_list(ticked_prev)

    def _rebuild_signal_baseline_exclude_list(self, ticked_prev: set) -> None:
        """Populate the exclude-bouts list with behaviors from the currently
        selected file (or union across all files when 'Auto-match' is chosen)."""
        self.list_signal_baseline_exclude.blockSignals(True)
        try:
            self.list_signal_baseline_exclude.clear()
            stem = self.combo_signal_baseline_behavior_file.currentData() if hasattr(self, "combo_signal_baseline_behavior_file") else ""
            sources: Dict[str, Dict[str, Any]] = self._behavior_sources or {}
            names: set = set()
            if isinstance(stem, str) and stem and stem in sources:
                names.update((sources[stem].get("behaviors") or {}).keys())
            else:
                # Union across all behavior files.
                for info in sources.values():
                    names.update((info.get("behaviors") or {}).keys())
            for name in sorted(names):
                item = QtWidgets.QListWidgetItem(str(name))
                item.setData(QtCore.Qt.ItemDataRole.UserRole, str(name))
                item.setFlags(item.flags() | QtCore.Qt.ItemFlag.ItemIsUserCheckable)
                item.setCheckState(
                    QtCore.Qt.CheckState.Checked if str(name) in ticked_prev else QtCore.Qt.CheckState.Unchecked
                )
                self.list_signal_baseline_exclude.addItem(item)
        finally:
            self.list_signal_baseline_exclude.blockSignals(False)

    def _refresh_signal_baseline_status(self) -> None:
        """Live preview of the resulting baseline duration (independent of fit)."""
        if not hasattr(self, "lbl_signal_baseline_status"):
            return
        if not self._processed:
            self.lbl_signal_baseline_status.setText("Load a recording first.")
            return
        proc = self._processed[0]
        t = np.asarray(getattr(proc, "time", np.array([], float)), float)
        if t.size < 3:
            self.lbl_signal_baseline_status.setText("Active trace has no time vector.")
            return
        finite = np.isfinite(t)
        file_id = os.path.splitext(os.path.basename(getattr(proc, "path", "") or ""))[0] or "import"
        try:
            mask, source, explanation = self._signal_baseline_mask(t, finite, file_id=file_id)
        except Exception as exc:
            self.lbl_signal_baseline_status.setText(f"Baseline preview error: {exc}")
            return
        n = int(np.sum(mask))
        t_finite = t[mask]
        if t_finite.size >= 2:
            dur = float(np.nanmax(t_finite) - np.nanmin(t_finite))
        else:
            dur = 0.0
        total_dur = float(np.nanmax(t[finite]) - np.nanmin(t[finite])) if np.sum(finite) >= 2 else 0.0
        pct = (dur / total_dur * 100.0) if total_dur > 0 else 0.0
        self.lbl_signal_baseline_status.setText(
            f"Baseline: {n} samples, {dur:.1f} s ({pct:.1f}% of trace) - {explanation}."
        )

    def _history_snapshot(self) -> Dict[str, object]:
        state = self._collect_settings()
        try:
            state["visual_mode"] = int(self.tab_visual_mode.currentIndex())
        except Exception:
            state["visual_mode"] = 0
        try:
            state["individual_file"] = self.combo_individual_file.currentText().strip()
        except Exception:
            state["individual_file"] = ""
        return copy.deepcopy(state)

    def _history_state_key(self, state: Dict[str, object]) -> str:
        try:
            return json.dumps(state, sort_keys=True, separators=(",", ":"), default=str)
        except Exception:
            return repr(state)

    def _project_dirty_fingerprint(self) -> str:
        """
        Stable, lightweight fingerprint of content saved in a postprocessing
        project. View-only navigation is intentionally excluded so browsing
        Individual/Group or changing the active displayed file does not trigger
        an exit warning.
        """
        try:
            settings = dict(self._collect_settings())
        except Exception:
            settings = {}
        for key in ("view_layout", "visual_mode", "individual_file", "signal_file"):
            settings.pop(key, None)

        processed: List[Dict[str, object]] = []
        for proc in getattr(self, "_processed", []) or []:
            try:
                output = np.asarray(getattr(proc, "output", np.array([], float)), float)
                time = np.asarray(getattr(proc, "time", np.array([], float)), float)
                sync_raw = getattr(proc, "sync_aligned_time", None)
                sync_time = np.asarray(sync_raw, float) if sync_raw is not None else np.array([], float)
            except Exception:
                output = np.array([], float)
                time = np.array([], float)
                sync_time = np.array([], float)
            sync_report = getattr(proc, "sync_report", {}) or {}
            processed.append(
                {
                    "path": str(getattr(proc, "path", "") or ""),
                    "channel_id": str(getattr(proc, "channel_id", "") or ""),
                    "dio_name": str(getattr(proc, "dio_name", "") or ""),
                    "output_label": str(getattr(proc, "output_label", "") or ""),
                    "output_context": str(getattr(proc, "output_context", "") or ""),
                    "fs_actual": float(getattr(proc, "fs_actual", np.nan)),
                    "fs_target": float(getattr(proc, "fs_target", np.nan)),
                    "fs_used": float(getattr(proc, "fs_used", np.nan)),
                    "n_time": int(time.size),
                    "n_output": int(output.size),
                    "sync_n": int(sync_time.size),
                    "sync_start": float(sync_time[0]) if sync_time.size else None,
                    "sync_end": float(sync_time[-1]) if sync_time.size else None,
                    "sync_report": sync_report if isinstance(sync_report, dict) else {},
                }
            )

        behavior: List[Dict[str, object]] = []
        for name, info in sorted((getattr(self, "_behavior_sources", {}) or {}).items()):
            data = info if isinstance(info, dict) else {}
            behavior.append(
                {
                    "name": str(name),
                    "source_path": str(data.get("source_path", "") or ""),
                    "kind": str(data.get("kind", "") or ""),
                    "row_count": int(data.get("row_count", 0) or 0),
                    "behaviors": sorted(str(k) for k in (data.get("behaviors", {}) or {}).keys()),
                    "trajectory": sorted(str(k) for k in (data.get("trajectory", {}) or {}).keys()),
                }
            )

        signal = getattr(self, "last_signal_events", None)
        signal_summary: Dict[str, object] = {}
        if isinstance(signal, dict) and signal:
            signal_summary = {
                "params": signal.get("params", {}) or {},
                "derived_metrics": signal.get("derived_metrics", {}) or {},
                "n_peaks": int(np.asarray(signal.get("peak_times_sec", np.array([], float))).size),
                "file_ids": [str(v) for v in (signal.get("file_ids", []) or [])],
            }

        behavior_analysis = getattr(self, "last_behavior_analysis", None)
        behavior_summary: Dict[str, object] = {}
        if isinstance(behavior_analysis, dict) and behavior_analysis:
            behavior_summary = {
                "behavior_name": str(behavior_analysis.get("behavior_name", "") or ""),
                "params": behavior_analysis.get("params", {}) or {},
                "per_file_metrics": behavior_analysis.get("per_file_metrics", []) or [],
                "group_metrics": behavior_analysis.get("group_metrics", {}) or {},
                "n_event_rows": len(behavior_analysis.get("per_file_events", []) or []),
            }

        temporal_summary: Dict[str, object] = {}
        section = getattr(self, "section_temporal", None)
        if section is not None:
            try:
                glm = getattr(section, "_glm_result", None)
                by_file = getattr(section, "_glm_results_by_file", {}) or {}
                temporal_summary = {
                    "fit_mode": str(getattr(section, "_fit_mode", "") or ""),
                    "active_file_id": str(getattr(section, "_active_file_id", "") or ""),
                    "glm_r2": float(getattr(glm, "r2", np.nan)) if glm is not None else None,
                    "glm_predictors": list(getattr(glm, "predictor_names", []) or []) if glm is not None else [],
                    "by_file": {
                        str(fid): {
                            "r2": float(getattr(res, "r2", np.nan)),
                            "predictors": list(getattr(res, "predictor_names", []) or []),
                        }
                        for fid, res in sorted(by_file.items())
                    },
                }
            except Exception:
                temporal_summary = {}

        payload = {
            "settings": settings,
            "tab_sources_index": int(self.tab_sources.currentIndex()) if hasattr(self, "tab_sources") else 0,
            "processed": processed,
            "behavior": behavior,
            "signal_events": signal_summary,
            "behavior_analysis": behavior_summary,
            "temporal": temporal_summary,
        }
        try:
            return json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
        except Exception:
            return repr(payload)

    def _mark_project_clean(self) -> None:
        try:
            self._project_clean_key = self._project_dirty_fingerprint()
        except Exception:
            self._project_clean_key = ""
        self._project_dirty = False

    def is_project_dirty(self) -> bool:
        clean_key = str(getattr(self, "_project_clean_key", "") or "")
        if clean_key:
            try:
                if self._project_dirty_fingerprint() == clean_key:
                    self._project_dirty = False
                    self._project_recovered_from_autosave = False
                    return False
            except Exception:
                pass
        if not bool(getattr(self, "_project_dirty", False)):
            self._project_recovered_from_autosave = False
            return False
        return True

    def _update_history_buttons(self) -> None:
        for button, enabled in (
            (getattr(self, "btn_action_undo", None), bool(self._history_undo)),
            (getattr(self, "btn_action_redo", None), bool(self._history_redo)),
        ):
            if button is not None:
                button.setEnabled(enabled)

    def _reset_history_snapshot(self) -> None:
        if not hasattr(self, "combo_align"):
            return
        self._history_undo.clear()
        self._history_redo.clear()
        self._history_current = self._history_snapshot()
        self._history_key = self._history_state_key(self._history_current)
        self._update_history_buttons()

    def _record_history_change(self) -> None:
        if self._is_restoring_settings or self._history_restoring:
            return
        state = self._history_snapshot()
        key = self._history_state_key(state)
        if self._history_current is None:
            self._history_current = copy.deepcopy(state)
            self._history_key = key
            self._update_history_buttons()
            return
        if key == self._history_key:
            return
        self._history_undo.append(copy.deepcopy(self._history_current))
        if len(self._history_undo) > self._history_limit:
            self._history_undo = self._history_undo[-self._history_limit:]
        self._history_redo.clear()
        self._history_current = copy.deepcopy(state)
        self._history_key = key
        self._update_history_buttons()

    def _restore_history_state(self, state: Dict[str, object]) -> None:
        was_restoring = self._is_restoring_settings
        self._history_restoring = True
        self._is_restoring_settings = True
        try:
            self._apply_settings(copy.deepcopy(state))
            if "visual_mode" in state:
                idx = int(state.get("visual_mode") or 0)
                if 0 <= idx < self.tab_visual_mode.count():
                    self.tab_visual_mode.setCurrentIndex(idx)
            if "individual_file" in state:
                file_id = str(state.get("individual_file") or "").strip()
                if file_id:
                    combo_idx = self.combo_individual_file.findText(file_id)
                    if combo_idx >= 0:
                        self.combo_individual_file.setCurrentIndex(combo_idx)
            self._rerender_visual_from_cache()
            self._update_trace_preview()
            self._save_settings()
        finally:
            self._is_restoring_settings = was_restoring
            self._history_restoring = False

    def _undo_post_action(self) -> None:
        if not self._history_undo:
            return
        current = self._history_snapshot()
        previous = self._history_undo.pop()
        self._history_redo.append(copy.deepcopy(current))
        self._restore_history_state(previous)
        self._history_current = copy.deepcopy(previous)
        self._history_key = self._history_state_key(previous)
        self._update_history_buttons()
        self.statusUpdate.emit("Undid postprocessing action.", 2500)

    def _redo_post_action(self) -> None:
        if not self._history_redo:
            return
        current = self._history_snapshot()
        next_state = self._history_redo.pop()
        self._history_undo.append(copy.deepcopy(current))
        self._restore_history_state(next_state)
        self._history_current = copy.deepcopy(next_state)
        self._history_key = self._history_state_key(next_state)
        self._update_history_buttons()
        self.statusUpdate.emit("Redid postprocessing action.", 2500)

    def _queue_settings_save(self, *_args: object) -> None:
        if self._is_restoring_settings:
            return
        self._record_history_change()
        if not self._autosave_restoring:
            self._project_dirty = True
        timer = getattr(self, "_settings_save_timer", None)
        if timer is None:
            self._save_settings()
            return
        timer.start()

    def _queue_view_settings_save(self, *_args: object) -> None:
        """Persist navigation/view state without marking the project unsaved."""
        if self._is_restoring_settings:
            return
        timer = getattr(self, "_settings_save_timer", None)
        if timer is None:
            self._save_settings()
            return
        timer.start()

    def _wire_settings_autosave(self) -> None:
        for combo in (
            self.combo_align,
            self.combo_dio,
            self.combo_dio_polarity,
            self.combo_dio_align,
            self.combo_behavior_file_type,
            self.combo_behavior_name,
            self.combo_behavior_align,
            self.combo_behavior_from,
            self.combo_behavior_to,
            self.combo_sync_behavior_file,
            self.combo_sync_camera_column,
            self.combo_sync_camera_mode,
            self.combo_sync_fiber_source,
            self.combo_sync_fiber_mode,
            self.combo_sync_method,
            self.combo_sync_export_format,
            self.combo_metric,
            self.combo_signal_source,
            self.combo_signal_scope,
            self.combo_signal_file,
            self.combo_signal_method,
            self.combo_peak_baseline,
            self.combo_behavior_analysis,
            self.combo_spatial_x,
            self.combo_spatial_y,
            self.combo_spatial_weight,
            self.combo_spatial_activity_mode,
        ):
            combo.currentIndexChanged.connect(self._queue_settings_save)

        for spin in (
            self.spin_transition_gap,
            self.spin_sync_threshold,
            self.spin_sync_min_interval,
            self.spin_sync_max_offset,
            self.spin_pre,
            self.spin_post,
            self.spin_b0,
            self.spin_b1,
            self.spin_resample,
            self.spin_smooth,
            self.spin_event_start,
            self.spin_event_end,
            self.spin_group_window,
            self.spin_dur_min,
            self.spin_dur_max,
            self.spin_min_events_per_animal,
            self.spin_metric_pre0,
            self.spin_metric_pre1,
            self.spin_metric_post0,
            self.spin_metric_post1,
            self.spin_global_start,
            self.spin_global_end,
            self.spin_peak_prominence,
            self.spin_peak_mad_multiplier,
            self.spin_peak_height,
            self.spin_peak_distance,
            self.spin_peak_smooth,
            self.spin_peak_baseline_window,
            self.spin_peak_rate_bin,
            self.spin_peak_auc_window,
            self.spin_behavior_bin,
            self.spin_spatial_bins_x,
            self.spin_spatial_bins_y,
            self.spin_spatial_clip_low,
            self.spin_spatial_clip_high,
            self.spin_spatial_time_min,
            self.spin_spatial_time_max,
            self.spin_spatial_smooth,
        ):
            spin.valueChanged.connect(self._queue_settings_save)

        for chk in (
            self.cb_filter_events,
            self.cb_group_keep_trials,
            self.cb_sync_auto_threshold,
            self.cb_sync_use_aligned,
            self.cb_sync_auto_recompute,
            self.cb_metrics,
            self.cb_exclude_low_event_animals,
            self.cb_global_metrics,
            self.cb_global_amp,
            self.cb_global_freq,
            self.cb_peak_auto_mad,
            self.cb_peak_norm_prominence,
            self.cb_peak_overlay,
            self.cb_peak_noise_overlay,
            self.cb_behavior_aligned,
            self.cb_spatial_clip,
            self.cb_spatial_time_filter,
            self.cb_spatial_log,
            self.cb_spatial_invert_y,
        ):
            chk.toggled.connect(self._queue_settings_save)

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

        raw_header = [str(h).strip() for h in data_rows[0]]
        header = [h.lower() for h in raw_header]

        def _find_col(names: List[str]) -> Optional[int]:
            for name in names:
                if name in header:
                    return header.index(name)
            return None

        def _looks_like_photometry_sync_col(name: str) -> bool:
            key = str(name or "").strip().lower().replace(" ", "").replace("_", "")
            return (
                key == "dio"
                or key.startswith("dio")
                or key.startswith("ttl")
                or key.startswith("trigger")
                or "sync" in key
                or "barcode" in key
            )

        time_idx = header.index("time") if "time" in header else None
        output_idx = _find_col([
            "dff",
            "z-score",
            "zscore",
            "z score",
            "prominence",
            "output",
            "raw_signal",
            "raw_465",
            "raw",
            "isobestic",
            "raw_405",
            "reference",
            "reference_405",
            "ref",
            "dio",
            "baseline_465",
            "baseline_405",
        ])
        has_header = time_idx is not None and output_idx is not None

        raw_idx = _find_col(["raw", "raw_465", "signal", "signal_465"]) if has_header else None
        iso_idx = _find_col(["isobestic", "isosbestic", "raw_405", "reference", "reference_405", "ref"]) if has_header else None
        dio_idx = _find_col(["dio"]) if has_header else None
        aligned_idx = _find_col(["time_aligned", "aligned_time", "sync_aligned_time"]) if has_header else None
        trigger_cols: List[Tuple[int, str]] = []
        if has_header:
            seen_trigger_names: set[str] = set()
            for idx, name in enumerate(raw_header):
                if idx == time_idx:
                    continue
                label = str(name or header[idx] or f"column_{idx + 1}").strip()
                if not _looks_like_photometry_sync_col(label):
                    continue
                key = label.lower()
                if key in seen_trigger_names:
                    continue
                seen_trigger_names.add(key)
                trigger_cols.append((idx, label))
            if dio_idx is None and len(trigger_cols) == 1:
                dio_idx = trigger_cols[0][0]

        data_rows = data_rows[1:] if has_header else data_rows

        time = []
        output = []
        raw_vals = []
        iso_vals = []
        dio_vals = []
        aligned_vals = []
        trigger_vals: Dict[str, List[float]] = {name: [] for _, name in trigger_cols}

        for r in data_rows:
            if time_idx is None or output_idx is None:
                continue
            if len(r) <= max(time_idx, output_idx):
                continue
            try:
                tval = coerce_time_value(r[time_idx])
                oval = float(r[output_idx])
            except Exception:
                continue
            if not np.isfinite(tval):
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
            if aligned_idx is not None:
                try:
                    aligned_vals.append(float(r[aligned_idx]) if len(r) > aligned_idx else np.nan)
                except Exception:
                    aligned_vals.append(np.nan)
            for trig_idx, trig_name in trigger_cols:
                try:
                    trigger_vals[trig_name].append(float(r[trig_idx]) if len(r) > trig_idx else np.nan)
                except Exception:
                    trigger_vals[trig_name].append(np.nan)

        if not time:
            return None

        t = np.asarray(time, float)
        out = np.asarray(output, float)
        raw = np.asarray(raw_vals, float) if raw_idx is not None and len(raw_vals) == len(time) else np.full_like(t, np.nan)
        iso = np.asarray(iso_vals, float) if iso_idx is not None and len(iso_vals) == len(time) else np.full_like(t, np.nan)
        dio_arr = np.asarray(dio_vals, float) if dio_idx is not None and len(dio_vals) == len(time) else None
        dio_name = ""
        if dio_arr is not None and dio_idx is not None and 0 <= dio_idx < len(raw_header):
            dio_name = str(raw_header[dio_idx] or "DIO").strip() or "DIO"
        triggers: Dict[str, np.ndarray] = {}
        for trig_name, vals in trigger_vals.items():
            arr = np.asarray(vals, float)
            if arr.size == t.size and int(np.sum(np.isfinite(arr))) >= 2:
                triggers[str(trig_name)] = arr
        if dio_arr is not None and int(np.sum(np.isfinite(dio_arr))) >= 2:
            triggers.setdefault(dio_name or "DIO", dio_arr)
        sync_aligned = (
            np.asarray(aligned_vals, float)
            if aligned_idx is not None and len(aligned_vals) == len(time)
            else None
        )

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
                elif col == "prominence":
                    col = "Prominence normalized"
                output_label = f"Imported CSV ({col})"

        return ProcessedTrial(
            path=path,
            channel_id="import",
            time=t,
            raw_signal=raw,
            raw_reference=iso,
            dio=dio_arr,
            dio_name=dio_name,
            triggers=triggers,
            sig_f=None,
            ref_f=None,
            baseline_sig=None,
            baseline_ref=None,
            output=out,
            output_label=output_label,
            output_context=output_context,
            sync_aligned_time=sync_aligned,
            sync_report={"status": "imported", "method": "imported time_aligned"} if sync_aligned is not None else {},
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
                sync_aligned = None
                if "time_aligned" in g:
                    sync_aligned = np.asarray(g["time_aligned"][()], float)
                elif "sync_aligned_time" in g:
                    sync_aligned = np.asarray(g["sync_aligned_time"][()], float)
                sync_report = {}
                raw_report = g.attrs.get("sync_report_json", "") if hasattr(g, "attrs") else ""
                if raw_report:
                    try:
                        sync_report = json.loads(str(raw_report))
                    except Exception:
                        sync_report = {}
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
            sync_aligned_time=sync_aligned,
            sync_report=sync_report if isinstance(sync_report, dict) else {},
            artifact_regions_sec=None,
            fs_actual=fs_actual,
            fs_target=fs_target,
            fs_used=fs_used,
        )

    def _refresh_behavior_list(self) -> None:
        self.combo_behavior_name.clear()
        if hasattr(self, "combo_behavior_analysis"):
            self.combo_behavior_analysis.clear()
        # Also refresh the Signal Events baseline-exclusion list, since it pulls
        # from the same loaded _behavior_sources.
        try:
            self._refresh_signal_baseline_behavior_lists()
            self._refresh_signal_baseline_status()
        except Exception:
            pass
        if not self._behavior_sources:
            self._refresh_spatial_columns()
            self._compute_spatial_heatmap()
            self._update_data_availability()
            self._sync_temporal_modeling_context()
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
        self._refresh_spatial_columns()
        self._compute_spatial_heatmap()
        self._update_data_availability()
        self._update_status_strip()
        self._sync_temporal_modeling_context()

    def _guess_spatial_column(self, columns: List[str], axis: str) -> Optional[str]:
        if not columns:
            return None
        axis_l = axis.lower()
        def _norm(value: str) -> str:
            s = str(value).strip().lower()
            s = re.sub(r"[\s_\-:/\(\)\[\]]+", " ", s)
            return re.sub(r"\s+", " ", s).strip()

        # First choice: explicit EthoVision center labels (set as default).
        preferred_center = {
            "x": ["x center", "x_center", "x-center", "center x", "center_x", "center-x"],
            "y": ["y center", "y_center", "y-center", "center y", "center_y", "center-y"],
        }.get(axis_l, [])
        preferred_center_norm = {_norm(v) for v in preferred_center}
        for col in columns:
            name_norm = _norm(str(col))
            if name_norm in preferred_center_norm:
                return col
        # Fallback center match for labels like "Center X (cm)" / "X center px".
        for col in columns:
            name_norm = _norm(str(col))
            if "center" in name_norm and axis_l in name_norm:
                return col
        # Second choice: exact column labels "X" / "Y" (case-insensitive).
        for col in columns:
            if _norm(str(col)) == axis_l:
                return col
        # Third choice: labels starting with X/Y token (e.g. "X nose", "Y_center").
        token_re = re.compile(rf"^{re.escape(axis_l)}(?:$|[\s_\-:/\(\[])")
        for col in columns:
            name = str(col).strip().lower()
            if token_re.match(name):
                return col
        patterns = {
            "x": ["x", "center x", "centrex", "nose x", "body x", "position x", "x center"],
            "y": ["y", "center y", "centrey", "nose y", "body y", "position y", "y center"],
        }.get(axis_l, [])
        # Third choice: known exact labels.
        for col in columns:
            name = str(col).strip().lower()
            for pat in patterns:
                if pat == name:
                    return col
        # Last choice: known partial labels.
        for col in columns:
            name = str(col).strip().lower()
            for pat in patterns:
                if pat in name:
                    return col
        return columns[0]

    def _refresh_spatial_columns(self) -> None:
        if not hasattr(self, "combo_spatial_x") or not hasattr(self, "combo_spatial_y"):
            return

        prev_x = self.combo_spatial_x.currentText().strip()
        prev_y = self.combo_spatial_y.currentText().strip()
        cols: set[str] = set()
        for info in self._behavior_sources.values():
            trajectory = info.get("trajectory") or {}
            cols.update(str(k) for k in trajectory.keys())
        ordered = sorted(cols)

        for combo in (self.combo_spatial_x, self.combo_spatial_y):
            combo.blockSignals(True)
            combo.clear()
            combo.addItems(ordered)
            combo.blockSignals(False)

        if not ordered:
            return

        if prev_x:
            ix = self.combo_spatial_x.findText(prev_x)
            if ix >= 0:
                self.combo_spatial_x.setCurrentIndex(ix)
        if prev_y:
            iy = self.combo_spatial_y.findText(prev_y)
            if iy >= 0:
                self.combo_spatial_y.setCurrentIndex(iy)

        if not self.combo_spatial_x.currentText().strip():
            gx = self._guess_spatial_column(ordered, "x")
            if gx:
                self.combo_spatial_x.setCurrentText(gx)
        if not self.combo_spatial_y.currentText().strip():
            gy = self._guess_spatial_column(ordered, "y")
            if gy:
                self.combo_spatial_y.setCurrentText(gy)

        if self.combo_spatial_x.currentText().strip() == self.combo_spatial_y.currentText().strip() and len(ordered) > 1:
            for col in ordered:
                if col != self.combo_spatial_x.currentText().strip():
                    self.combo_spatial_y.setCurrentText(col)
                    break

    def _update_spatial_clip_enabled(self) -> None:
        enabled = bool(self.cb_spatial_clip.isChecked() and self.cb_spatial_clip.isEnabled())
        self.spin_spatial_clip_low.setEnabled(enabled)
        self.spin_spatial_clip_high.setEnabled(enabled)

    def _update_spatial_time_filter_enabled(self) -> None:
        enabled = bool(self.cb_spatial_time_filter.isChecked() and self.cb_spatial_time_filter.isEnabled())
        self.spin_spatial_time_min.setEnabled(enabled)
        self.spin_spatial_time_max.setEnabled(enabled)

    def _show_spatial_help(self) -> None:
        QtWidgets.QMessageBox.information(
            self,
            "Spatial Heatmap Help",
            "Top plot: occupancy heatmap.\n"
            "Middle plot: activity heatmap (mean occupancy-normalized, mean velocity-normalized, or sum z-score/bin).\n"
            "Bottom plot: velocity heatmap (mean speed per bin).\n"
            "Mean z-score/bin (occupancy normalized) = sum(z*weight) / sum(weight).\n"
            "Mean z-score/bin (velocity normalized) = sum(z*weight) / sum(speed*weight).\n"
            "Enable time filter to include only samples within [start, end] seconds.\n"
            "Use the right-side color cursors on each plot to set min/max display range interactively.",
        )

    def _selected_processed_for_spatial(self) -> List[ProcessedTrial]:
        if not self._processed:
            return []
        selected = self.list_preprocessed.selectedItems() if hasattr(self, "list_preprocessed") else []
        if not selected:
            return list(self._processed)
        id_map = {id(proc): proc for proc in self._processed}
        picked: List[ProcessedTrial] = []
        for item in selected:
            proc_id = item.data(QtCore.Qt.ItemDataRole.UserRole) if item else None
            proc = id_map.get(proc_id)
            if proc is not None:
                picked.append(proc)
        return picked or list(self._processed)

    def _spatial_weight_values(
        self,
        x: np.ndarray,
        y: np.ndarray,
        t: np.ndarray,
        mode: str,
    ) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
        xx = np.asarray(x, float)
        yy = np.asarray(y, float)
        n = min(xx.size, yy.size)
        if n <= 0:
            return np.array([], float), np.array([], float), None
        xx = xx[:n]
        yy = yy[:n]
        weights: Optional[np.ndarray] = None
        if "time" in mode.lower() or "probability" in mode.lower():
            tt = np.asarray(t, float) if t is not None else np.array([], float)
            if tt.size >= n:
                tt = tt[:n]
            else:
                tt = np.array([], float)
            if tt.size == n:
                dt = np.diff(tt)
                dt = dt[np.isfinite(dt) & (dt > 0)]
                default_dt = float(np.nanmedian(dt)) if dt.size else 1.0
                w = np.full(n, default_dt, dtype=float)
                if n > 1:
                    step = np.diff(tt, prepend=tt[0])
                    step[~np.isfinite(step)] = default_dt
                    step[step <= 0] = default_dt
                    w = step
                weights = w
            else:
                weights = np.ones(n, dtype=float)
        return xx, yy, weights

    def _spatial_velocity_values(
        self,
        x: np.ndarray,
        y: np.ndarray,
        t: np.ndarray,
    ) -> np.ndarray:
        xx = np.asarray(x, float)
        yy = np.asarray(y, float)
        n = min(xx.size, yy.size)
        if n <= 1:
            return np.array([], float)
        xx = xx[:n]
        yy = yy[:n]
        tt = np.asarray(t, float) if t is not None else np.array([], float)
        if tt.size >= n:
            tt = tt[:n]
            dt = np.diff(tt)
            use_time = True
        else:
            dt = np.ones(n - 1, dtype=float)
            use_time = False
        dx = np.diff(xx)
        dy = np.diff(yy)
        speed_steps = np.full(n - 1, np.nan, dtype=float)
        valid = np.isfinite(dx) & np.isfinite(dy) & np.isfinite(dt) & (dt > 0)
        if np.any(valid):
            dist = np.sqrt(dx[valid] * dx[valid] + dy[valid] * dy[valid])
            speed_steps[valid] = dist / dt[valid] if use_time else dist
        speed = np.full(n, np.nan, dtype=float)
        speed[1:] = speed_steps
        if np.isfinite(speed_steps).any():
            speed[0] = float(speed_steps[np.where(np.isfinite(speed_steps))[0][0]])
        return speed

    def _render_spatial_map(
        self,
        plot_widget: pg.PlotWidget,
        image_item: pg.ImageItem,
        heat: Optional[np.ndarray],
        extent: Optional[Tuple[float, float, float, float]],
        x_label: str,
        y_label: str,
        title: str,
    ) -> None:
        plot_widget.setTitle(title)
        plot_widget.setLabel("bottom", x_label or "X")
        plot_widget.setLabel("left", y_label or "Y")
        invert_y = bool(self.cb_spatial_invert_y.isChecked()) if hasattr(self, "cb_spatial_invert_y") else False
        vb = plot_widget.getViewBox()
        if vb is not None:
            vb.invertY(invert_y)

        if heat is None or heat.size == 0 or extent is None:
            image_item.setImage(np.zeros((1, 1)), autoLevels=True)
            image_item.setRect(QtCore.QRectF(0.0, 0.0, 1.0, 1.0))
            plot_widget.setXRange(0.0, 1.0, padding=0.0)
            plot_widget.setYRange(0.0, 1.0, padding=0.0)
            return

        cmap_name = str(self._style.get("heatmap_cmap", "viridis"))
        try:
            cmap = pg.colormap.get(cmap_name)
            image_item.setLookupTable(cmap.getLookupTable())
            if image_item is getattr(self, "img_spatial_occupancy", None):
                self.spatial_lut_occupancy.item.gradient.setColorMap(cmap)
            elif image_item is getattr(self, "img_spatial_activity", None):
                self.spatial_lut_activity.item.gradient.setColorMap(cmap)
            elif image_item is getattr(self, "img_spatial_velocity", None):
                self.spatial_lut_velocity.item.gradient.setColorMap(cmap)
        except Exception:
            pass

        arr = np.asarray(heat, float)
        image_item.setImage(arr, autoLevels=True)
        hmin = self._style.get("heatmap_min", None)
        hmax = self._style.get("heatmap_max", None)
        if hmin is not None and hmax is not None:
            image_item.setLevels([float(hmin), float(hmax)])
        xmin, xmax, ymin, ymax = extent
        dx = max(1e-9, float(xmax - xmin))
        dy = max(1e-9, float(ymax - ymin))
        image_item.setRect(QtCore.QRectF(float(xmin), float(ymin), dx, dy))
        plot_widget.setXRange(float(xmin), float(xmax), padding=0.0)
        plot_widget.setYRange(float(ymin), float(ymax), padding=0.0)

    def _render_spatial_heatmap(
        self,
        occupancy_heat: Optional[np.ndarray],
        activity_heat: Optional[np.ndarray],
        velocity_heat: Optional[np.ndarray],
        extent: Optional[Tuple[float, float, float, float]],
        x_label: str,
        y_label: str,
        occupancy_title: str,
        activity_title: str,
        velocity_title: str,
    ) -> None:
        self._render_spatial_map(
            self.plot_spatial_occupancy,
            self.img_spatial_occupancy,
            occupancy_heat,
            extent,
            x_label,
            y_label,
            occupancy_title,
        )
        self._render_spatial_map(
            self.plot_spatial_activity,
            self.img_spatial_activity,
            activity_heat,
            extent,
            x_label,
            y_label,
            activity_title,
        )
        self._render_spatial_map(
            self.plot_spatial_velocity,
            self.img_spatial_velocity,
            velocity_heat,
            extent,
            x_label,
            y_label,
            velocity_title,
        )

    def _show_spatial_heatmap_panel(self) -> None:
        if not hasattr(self, "spatial_plot_dialog"):
            return
        try:
            self.spatial_plot_dialog.show()
            self.spatial_plot_dialog.raise_()
            self.spatial_plot_dialog.activateWindow()
        except Exception:
            pass

    def _on_compute_spatial_clicked(self) -> None:
        self._compute_spatial_heatmap(show_panel=True)

    def _compute_spatial_heatmap(self, *_args: object, show_panel: bool = False) -> None:
        self._queue_settings_save()
        if not hasattr(self, "plot_spatial_occupancy"):
            return
        if show_panel:
            self._show_spatial_heatmap_panel()
        activity_mode_text = self.combo_spatial_activity_mode.currentText().strip().lower()
        activity_mode = "mean"
        if "velocity" in activity_mode_text:
            activity_mode = "velocity"
        elif "sum" in activity_mode_text:
            activity_mode = "sum"

        def _activity_title_default() -> str:
            if activity_mode == "velocity":
                return "Spatial activity (mean z-score/bin, velocity normalized)"
            if activity_mode == "sum":
                return "Spatial activity (sum z-score/bin)"
            return "Spatial activity (mean z-score/bin, occupancy normalized)"

        def _clear_spatial(msg: str, x_label: str = "", y_label: str = "") -> None:
            self._last_spatial_occupancy_map = None
            self._last_spatial_activity_map = None
            self._last_spatial_velocity_map = None
            self._last_spatial_extent = None
            self.lbl_spatial_msg.setText(msg)
            self._render_spatial_heatmap(
                None,
                None,
                None,
                None,
                x_label,
                y_label,
                "Spatial occupancy",
                _activity_title_default(),
                "Spatial velocity (mean speed/bin)",
            )

        if not self._processed:
            _clear_spatial("Load processed files to compute spatial heatmap.")
            return
        if not self._behavior_sources:
            _clear_spatial("Load behavior file(s) with trajectory columns.")
            return

        x_col = self.combo_spatial_x.currentText().strip()
        y_col = self.combo_spatial_y.currentText().strip()
        if not x_col or not y_col:
            _clear_spatial("Select X and Y trajectory columns.", x_col, y_col)
            return

        selected = self._selected_processed_for_spatial()
        mode = self.combo_spatial_weight.currentText().strip()
        use_time_filter = bool(self.cb_spatial_time_filter.isChecked())
        time_min = float(self.spin_spatial_time_min.value())
        time_max = float(self.spin_spatial_time_max.value())
        if use_time_filter and time_max <= time_min:
            _clear_spatial("Invalid time range: end must be greater than start.", x_col, y_col)
            return
        bins_x = int(self.spin_spatial_bins_x.value())
        bins_y = int(self.spin_spatial_bins_y.value())

        rows: List[Dict[str, np.ndarray]] = []
        all_x: List[np.ndarray] = []
        all_y: List[np.ndarray] = []
        skipped_missing_time = 0
        for proc in selected:
            info = self._match_behavior_source(proc)
            if not info:
                continue
            traj = info.get("trajectory") or {}
            if x_col not in traj or y_col not in traj:
                continue
            x = np.asarray(traj.get(x_col, np.array([], float)), float)
            y = np.asarray(traj.get(y_col, np.array([], float)), float)
            t = np.asarray(info.get("trajectory_time", np.array([], float)), float)
            if t.size == 0:
                t = np.asarray(info.get("time", np.array([], float)), float)
            n = min(x.size, y.size)
            if n <= 2:
                continue
            xx = x[:n]
            yy = y[:n]
            tt_full = t[:n] if t.size >= n else np.array([], float)
            finite = np.isfinite(xx) & np.isfinite(yy)
            if use_time_filter:
                if tt_full.size != n:
                    skipped_missing_time += 1
                    continue
                finite = finite & np.isfinite(tt_full) & (tt_full >= time_min) & (tt_full <= time_max)
            if not np.any(finite):
                continue

            xx_occ = xx[finite]
            yy_occ = yy[finite]
            tt_occ = tt_full[finite] if tt_full.size == n else np.array([], float)
            vel_occ = self._spatial_velocity_values(xx_occ, yy_occ, tt_occ)
            if vel_occ.size != xx_occ.size:
                vel_occ = np.full(xx_occ.shape, np.nan, dtype=float)

            act_vals = np.array([], float)
            if proc.output is not None:
                yy_out = np.asarray(proc.output, float)
                is_zscore_output = "zscore" in (proc.output_label or "").lower() or "z-score" in (proc.output_label or "").lower()
                if is_zscore_output:
                    yy_z = yy_out
                else:
                    yy_z = np.asarray(yy_out, float).copy()
                    m_out = np.isfinite(yy_z)
                    if np.any(m_out):
                        mu = float(np.nanmean(yy_z[m_out]))
                        sd = float(np.nanstd(yy_z[m_out]))
                        if not np.isfinite(sd) or sd <= 1e-12:
                            sd = 1.0
                        yy_z = (yy_z - mu) / sd
                tt_candidate = self._proc_time(proc)
                if tt_candidate is not None and np.asarray(tt_candidate, float).size == yy_out.size and t.size >= n:
                    tt_proc = np.asarray(tt_candidate, float)
                    act_full = np.interp(t[:n], tt_proc, yy_z, left=np.nan, right=np.nan)
                    act_vals = np.asarray(act_full[finite], float)
                elif yy_z.size >= n:
                    act_vals = np.asarray(yy_z[:n][finite], float)

            rows.append(
                {
                    "x_occ": xx_occ,
                    "y_occ": yy_occ,
                    "t_occ": tt_occ,
                    "vel_occ": vel_occ,
                    "x_act": xx_occ.copy(),
                    "y_act": yy_occ.copy(),
                    "t_act": tt_occ.copy(),
                    "vel_act": vel_occ.copy(),
                    "act": act_vals,
                }
            )
            all_x.append(xx_occ)
            all_y.append(yy_occ)

        if not rows or not all_x or not all_y:
            if use_time_filter:
                _clear_spatial("No trajectory samples in the selected time range.", x_col, y_col)
            else:
                _clear_spatial("No matching trajectory data found for selected files.", x_col, y_col)
            return

        x_all = np.concatenate(all_x)
        y_all = np.concatenate(all_y)
        if self.cb_spatial_clip.isChecked():
            lo = float(self.spin_spatial_clip_low.value())
            hi = float(self.spin_spatial_clip_high.value())
            if hi <= lo:
                hi = min(100.0, lo + 1.0)
            xmin, xmax = np.nanpercentile(x_all, [lo, hi])
            ymin, ymax = np.nanpercentile(y_all, [lo, hi])
        else:
            xmin, xmax = float(np.nanmin(x_all)), float(np.nanmax(x_all))
            ymin, ymax = float(np.nanmin(y_all)), float(np.nanmax(y_all))
        if not np.isfinite(xmin) or not np.isfinite(xmax) or xmin == xmax:
            xmin, xmax = 0.0, 1.0
        if not np.isfinite(ymin) or not np.isfinite(ymax) or ymin == ymax:
            ymin, ymax = 0.0, 1.0

        occ_maps: List[np.ndarray] = []
        act_maps: List[np.ndarray] = []
        vel_maps: List[np.ndarray] = []
        for row in rows:
            xx, yy, weights = self._spatial_weight_values(row["x_occ"], row["y_occ"], row["t_occ"], mode)
            if xx.size == 0 or yy.size == 0:
                continue
            occ_hist, _, _ = np.histogram2d(
                xx,
                yy,
                bins=[bins_x, bins_y],
                range=[[xmin, xmax], [ymin, ymax]],
                weights=weights,
            )
            occ_hist = np.asarray(occ_hist, float)
            if "probability" in mode.lower():
                denom = float(np.nansum(occ_hist))
                if denom > 0:
                    occ_hist = (occ_hist / denom) * 100.0
            occ_maps.append(occ_hist)

            vel = np.asarray(row.get("vel_occ", np.array([], float)), float)
            if vel.size > 0:
                xv = np.asarray(row["x_occ"], float)
                yv = np.asarray(row["y_occ"], float)
                tv = np.asarray(row["t_occ"], float)
                m_vel = np.isfinite(vel) & np.isfinite(xv) & np.isfinite(yv)
                if np.any(m_vel):
                    xv = xv[m_vel]
                    yv = yv[m_vel]
                    vel = np.clip(vel[m_vel], a_min=0.0, a_max=None)
                    tv = tv[m_vel] if tv.size == m_vel.size else np.array([], float)
                    xx_v, yy_v, w_v = self._spatial_weight_values(xv, yv, tv, mode)
                    n_v = min(xx_v.size, yy_v.size, vel.size)
                    if n_v > 0:
                        xx_v = xx_v[:n_v]
                        yy_v = yy_v[:n_v]
                        vel = vel[:n_v]
                        vel_w = w_v[:n_v] if w_v is not None and w_v.size >= n_v else np.ones(n_v, dtype=float)
                        vel_num, _, _ = np.histogram2d(
                            xx_v,
                            yy_v,
                            bins=[bins_x, bins_y],
                            range=[[xmin, xmax], [ymin, ymax]],
                            weights=vel * vel_w,
                        )
                        vel_den, _, _ = np.histogram2d(
                            xx_v,
                            yy_v,
                            bins=[bins_x, bins_y],
                            range=[[xmin, xmax], [ymin, ymax]],
                            weights=vel_w,
                        )
                        with np.errstate(divide="ignore", invalid="ignore"):
                            vel_hist = np.divide(vel_num, vel_den, out=np.full_like(vel_num, np.nan), where=vel_den > 0)
                        vel_maps.append(np.asarray(vel_hist, float))

            act = np.asarray(row.get("act", np.array([], float)), float)
            if act.size == 0:
                continue
            xa = np.asarray(row["x_act"], float)
            ya = np.asarray(row["y_act"], float)
            ta = np.asarray(row["t_act"], float)
            va = np.asarray(row.get("vel_act", np.array([], float)), float)
            m_act = np.isfinite(act) & np.isfinite(xa) & np.isfinite(ya)
            if not np.any(m_act):
                continue
            xa = xa[m_act]
            ya = ya[m_act]
            act = act[m_act]
            ta = ta[m_act] if ta.size == m_act.size else np.array([], float)
            if va.size == m_act.size:
                va = va[m_act]
            else:
                va = np.full(act.shape, np.nan, dtype=float)
            xx_a, yy_a, w_a = self._spatial_weight_values(xa, ya, ta, mode)
            if xx_a.size == 0 or yy_a.size == 0 or act.size == 0:
                continue
            n_a = min(xx_a.size, yy_a.size, act.size)
            xx_a = xx_a[:n_a]
            yy_a = yy_a[:n_a]
            act = act[:n_a]
            va = va[:n_a] if va.size >= n_a else np.full(n_a, np.nan, dtype=float)
            den_w = w_a[:n_a] if w_a is not None and w_a.size >= n_a else np.ones(n_a, dtype=float)
            num_hist, _, _ = np.histogram2d(
                xx_a,
                yy_a,
                bins=[bins_x, bins_y],
                range=[[xmin, xmax], [ymin, ymax]],
                weights=act * den_w,
            )

            if activity_mode == "sum":
                act_hist = np.asarray(num_hist, float)
            elif activity_mode == "velocity":
                vel_weights = np.nan_to_num(np.clip(va, a_min=0.0, a_max=None), nan=0.0, posinf=0.0, neginf=0.0) * den_w
                vel_den, _, _ = np.histogram2d(
                    xx_a,
                    yy_a,
                    bins=[bins_x, bins_y],
                    range=[[xmin, xmax], [ymin, ymax]],
                    weights=vel_weights,
                )
                with np.errstate(divide="ignore", invalid="ignore"):
                    act_hist = np.divide(num_hist, vel_den, out=np.full_like(num_hist, np.nan), where=vel_den > 1e-12)
            else:
                den_hist, _, _ = np.histogram2d(
                    xx_a,
                    yy_a,
                    bins=[bins_x, bins_y],
                    range=[[xmin, xmax], [ymin, ymax]],
                    weights=den_w,
                )
                with np.errstate(divide="ignore", invalid="ignore"):
                    act_hist = np.divide(num_hist, den_hist, out=np.full_like(num_hist, np.nan), where=den_hist > 0)
            act_maps.append(np.asarray(act_hist, float))

        if not occ_maps:
            _clear_spatial("Could not compute heatmap for selected trajectories.", x_col, y_col)
            return

        def _nanmean_stack_no_warning(maps: List[np.ndarray]) -> Optional[np.ndarray]:
            if not maps:
                return None
            stack = np.asarray(np.stack(maps, axis=0), float)
            valid = np.isfinite(stack)
            counts = np.sum(valid, axis=0)
            sums = np.nansum(stack, axis=0)
            return np.divide(sums, counts, out=np.full_like(sums, np.nan, dtype=float), where=counts > 0)

        occupancy_map = _nanmean_stack_no_warning(occ_maps)
        if occupancy_map is None:
            _clear_spatial("Could not compute heatmap for selected trajectories.", x_col, y_col)
            return
        activity_map = _nanmean_stack_no_warning(act_maps)
        velocity_map = _nanmean_stack_no_warning(vel_maps)
        sigma = float(self.spin_spatial_smooth.value())
        if sigma > 0:
            from scipy.ndimage import gaussian_filter

            def _smooth_map(arr: np.ndarray) -> np.ndarray:
                a = np.asarray(arr, float)
                if not np.any(~np.isfinite(a)):
                    return gaussian_filter(a, sigma=max(0.0, sigma), mode="nearest")
                valid = np.isfinite(a).astype(float)
                num = gaussian_filter(np.nan_to_num(a, nan=0.0), sigma=max(0.0, sigma), mode="nearest")
                den = gaussian_filter(valid, sigma=max(0.0, sigma), mode="nearest")
                return np.divide(num, den, out=np.full_like(num, np.nan), where=den > 1e-9)

            occupancy_map = _smooth_map(occupancy_map)
            if activity_map is not None:
                activity_map = _smooth_map(activity_map)
            if velocity_map is not None:
                velocity_map = _smooth_map(velocity_map)
        if self.cb_spatial_log.isChecked():
            occupancy_map = np.log1p(np.clip(occupancy_map, a_min=0.0, a_max=None))
            if activity_map is not None:
                activity_map = np.sign(activity_map) * np.log1p(np.abs(activity_map))
            if velocity_map is not None:
                velocity_map = np.log1p(np.clip(velocity_map, a_min=0.0, a_max=None))

        self._last_spatial_occupancy_map = occupancy_map
        self._last_spatial_activity_map = activity_map
        self._last_spatial_velocity_map = velocity_map
        self._last_spatial_extent = (float(xmin), float(xmax), float(ymin), float(ymax))
        occ_title = f"Spatial occupancy ({mode.lower()})"
        if len(occ_maps) > 1:
            occ_title += f" - avg {len(occ_maps)} files"
        act_title = _activity_title_default()
        if len(act_maps) > 1:
            act_title += f" - avg {len(act_maps)} files"
        vel_title = "Spatial velocity (mean speed/bin)"
        if len(vel_maps) > 1:
            vel_title += f" - avg {len(vel_maps)} files"
        self._render_spatial_heatmap(
            occupancy_map,
            activity_map,
            velocity_map,
            self._last_spatial_extent,
            x_col,
            y_col,
            occ_title,
            act_title,
            vel_title,
        )
        time_txt = f", time={time_min:g}-{time_max:g}s" if use_time_filter else ""
        skip_txt = f", skipped {skipped_missing_time} file(s) without valid time" if skipped_missing_time > 0 else ""
        self.lbl_spatial_msg.setText(
            f"Rendered occupancy {len(occ_maps)} map(s), activity {len(act_maps)} map(s), velocity {len(vel_maps)} map(s), bins={bins_x}x{bins_y}{time_txt}{skip_txt}."
        )

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
            self._compute_spatial_heatmap()
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
            self._compute_spatial_heatmap()
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
        self._compute_spatial_heatmap()
        self._update_status_strip()

    def _remove_selected_preprocessed(self) -> None:
        selected = self.list_preprocessed.selectedItems()
        if not selected:
            return
        rows = sorted({self.list_preprocessed.row(item) for item in selected}, reverse=True)
        for row in rows:
            if 0 <= row < len(self._processed):
                del self._processed[row]
        if not self._autosave_restoring:
            self._project_dirty = True

        # If nothing is left, run the same teardown as Reset so every plot
        # clears (trace, heatmap, avg, metrics, global, duration, peaks,
        # behavior plots) and the temporal-modeling cache is dropped.
        if not self._processed:
            try:
                self._clear_cached_analysis_outputs()
            except Exception:
                pass
            try:
                section = getattr(self, "section_temporal", None)
                if section is not None:
                    section._on_clear_cached_fits()
                    section._glm_result = None
                    section._flmm_result = None
                    if hasattr(section, "txt_summary"):
                        section.txt_summary.clear()
                    for plot_attr in (
                        "plot_kernel", "plot_prediction", "plot_residuals",
                        "plot_importance", "plot_illustration", "plot_coeff",
                        "plot_group_kernels", "plot_group_importance",
                    ):
                        pw = getattr(section, plot_attr, None)
                        if pw is not None:
                            try:
                                pw.clear()
                            except Exception:
                                pass
            except Exception:
                pass
            self.lbl_group.setText("(none)")

        self._update_file_lists()
        # _compute_psth handles both the populated and empty-list paths; in the
        # empty case it now nulls cached matrices and clears the PSTH plots.
        self._compute_psth()
        self._update_trace_preview()
        self._compute_spatial_heatmap()
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
        if not self._autosave_restoring:
            self._project_dirty = True
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

    def _file_id_for_proc(self, proc: ProcessedTrial) -> str:
        return os.path.splitext(os.path.basename(getattr(proc, "path", "") or ""))[0] or "import"

    def _proc_time(self, proc: ProcessedTrial) -> np.ndarray:
        base = np.asarray(getattr(proc, "time", np.array([], float)), float)
        if hasattr(self, "cb_sync_use_aligned") and self.cb_sync_use_aligned.isChecked():
            aligned = getattr(proc, "sync_aligned_time", None)
            if aligned is not None:
                arr = np.asarray(aligned, float)
                if arr.size == base.size and arr.size > 0 and np.any(np.isfinite(arr)):
                    return arr
        return base

    def _map_photometry_times_for_proc(self, proc: ProcessedTrial, values: np.ndarray) -> np.ndarray:
        vals = np.asarray(values, float)
        if not (hasattr(self, "cb_sync_use_aligned") and self.cb_sync_use_aligned.isChecked()):
            return vals
        base = np.asarray(getattr(proc, "time", np.array([], float)), float)
        aligned = getattr(proc, "sync_aligned_time", None)
        if aligned is None:
            return vals
        mapped = np.asarray(aligned, float)
        if base.size != mapped.size or base.size < 2:
            return vals
        finite = np.isfinite(base) & np.isfinite(mapped)
        if np.sum(finite) < 2:
            return vals
        return np.interp(vals, base[finite], mapped[finite], left=np.nan, right=np.nan)

    def _sync_mode_key(self, combo_text: str) -> str:
        text = str(combo_text or "").lower()
        if "barcode" in text or "value" in text:
            return "barcode"
        if "fall" in text:
            return "ttl_falling"
        return "ttl_rising"

    def _on_sync_auto_threshold_toggled(self, checked: bool) -> None:
        ready = bool(getattr(self, "_processed", None)) and bool(getattr(self, "_behavior_sources", None))
        self.spin_sync_threshold.setEnabled(ready and not bool(checked))

    def _on_sync_use_aligned_changed(self) -> None:
        if self._is_restoring_settings:
            return
        self._queue_settings_save()
        self._update_trace_preview()
        self._compute_spatial_heatmap()
        if self.cb_sync_auto_recompute.isChecked():
            self._compute_psth()
        self._update_status_strip()

    def _refresh_sync_sources(self) -> None:
        if not hasattr(self, "combo_sync_behavior_file"):
            return
        prev_beh = self.combo_sync_behavior_file.currentData()
        self.combo_sync_behavior_file.blockSignals(True)
        try:
            self.combo_sync_behavior_file.clear()
            self.combo_sync_behavior_file.addItem("Auto-match behavior file", "")
            for stem in (self._behavior_sources or {}).keys():
                self.combo_sync_behavior_file.addItem(str(stem), str(stem))
            if isinstance(prev_beh, str):
                idx = self.combo_sync_behavior_file.findData(prev_beh)
                if idx >= 0:
                    self.combo_sync_behavior_file.setCurrentIndex(idx)
        finally:
            self.combo_sync_behavior_file.blockSignals(False)
        self._refresh_sync_camera_columns()
        self._refresh_sync_fiber_sources()
        self._refresh_sync_results_table()

    def _refresh_sync_camera_columns(self) -> None:
        if not hasattr(self, "combo_sync_camera_column"):
            return
        prev = self.combo_sync_camera_column.currentData()
        self.combo_sync_camera_column.blockSignals(True)
        try:
            self.combo_sync_camera_column.clear()
            selected_stem = self.combo_sync_behavior_file.currentData() if hasattr(self, "combo_sync_behavior_file") else ""
            sources: Dict[str, Dict[str, Any]] = {}
            if isinstance(selected_stem, str) and selected_stem and selected_stem in self._behavior_sources:
                sources[selected_stem] = self._behavior_sources[selected_stem]
            else:
                sources = dict(self._behavior_sources or {})
            names: List[Tuple[str, str, str]] = []
            seen = set()
            for info in sources.values():
                for name in (info.get("behaviors") or {}).keys():
                    key = ("behavior", str(name))
                    if key not in seen:
                        seen.add(key)
                        names.append((f"Behavior: {name}", "behavior", str(name)))
                for name in (info.get("trajectory") or {}).keys():
                    key = ("trajectory", str(name))
                    if key not in seen:
                        seen.add(key)
                        names.append((f"Column: {name}", "trajectory", str(name)))
            if not names:
                self.combo_sync_camera_column.addItem("Load behavior or camera CSV/XLSX first", "")
            else:
                for label, kind, name in names:
                    self.combo_sync_camera_column.addItem(label, f"{kind}::{name}")
                if isinstance(prev, str):
                    idx = self.combo_sync_camera_column.findData(prev)
                    if idx >= 0:
                        self.combo_sync_camera_column.setCurrentIndex(idx)
        finally:
            self.combo_sync_camera_column.blockSignals(False)

    def _refresh_sync_fiber_sources(self) -> None:
        if not hasattr(self, "combo_sync_fiber_source"):
            return
        prev = self.combo_sync_fiber_source.currentData()
        self.combo_sync_fiber_source.blockSignals(True)
        try:
            self.combo_sync_fiber_source.clear()
            self.combo_sync_fiber_source.addItem("Embedded DIO in processed file", "__embedded__")
            embedded_available = any(
                getattr(proc, "dio", None) is not None
                and np.asarray(getattr(proc, "dio"), float).size >= 2
                for proc in self._processed
            )
            entries: List[Tuple[str, str]] = []
            seen_data: set[str] = set()

            def _add_entry(label: str, name: str) -> None:
                name = str(name or "").strip()
                if not name:
                    return
                data = f"dio::{name}"
                if data in seen_data:
                    return
                seen_data.add(data)
                entries.append((label, data))

            names = list(getattr(self, "_known_dio_channels", []) or [])
            for name in names:
                _add_entry(f"A/D channel: {name}", str(name))
            for proc in self._processed:
                name = str(getattr(proc, "dio_name", "") or "").strip()
                if name:
                    _add_entry(f"Embedded DIO: {name}", name)
                for trig_name in (getattr(proc, "triggers", {}) or {}).keys():
                    _add_entry(f"Processed column: {trig_name}", str(trig_name))
            for label, data in entries:
                self.combo_sync_fiber_source.addItem(label, data)
            if isinstance(prev, str):
                idx = self.combo_sync_fiber_source.findData(prev)
                if idx >= 0:
                    self.combo_sync_fiber_source.setCurrentIndex(idx)
                elif prev == "__embedded__" and not embedded_available and entries:
                    self.combo_sync_fiber_source.setCurrentIndex(1)
                elif (
                    prev == "__embedded__"
                    and len(entries) == 1
                    and str(entries[0][0]).startswith(("Embedded DIO:", "Processed column:"))
                ):
                    self.combo_sync_fiber_source.setCurrentIndex(1)
            elif not embedded_available and entries:
                self.combo_sync_fiber_source.setCurrentIndex(1)
        finally:
            self.combo_sync_fiber_source.blockSignals(False)

    def _sync_behavior_source_for_proc(self, proc: ProcessedTrial) -> Optional[Dict[str, Any]]:
        stem = self.combo_sync_behavior_file.currentData() if hasattr(self, "combo_sync_behavior_file") else ""
        if isinstance(stem, str) and stem and stem in self._behavior_sources:
            return self._behavior_sources.get(stem)
        return self._match_behavior_source(proc)

    def _sync_camera_events_for_proc(self, proc: ProcessedTrial) -> np.ndarray:
        info = self._sync_behavior_source_for_proc(proc)
        if not info:
            raise ValueError("No behavior/camera file matched this recording.")
        data = self.combo_sync_camera_column.currentData() if hasattr(self, "combo_sync_camera_column") else ""
        if not isinstance(data, str) or "::" not in data:
            raise ValueError("Choose a camera sync behavior or column.")
        kind, name = data.split("::", 1)
        min_interval = float(self.spin_sync_min_interval.value())
        mode = self._sync_mode_key(self.combo_sync_camera_mode.currentText())
        threshold = None if self.cb_sync_auto_threshold.isChecked() else float(self.spin_sync_threshold.value())
        if kind == "behavior":
            behaviors = info.get("behaviors") or {}
            if name not in behaviors:
                raise ValueError(f"Behavior sync column not found: {name}")
            if str(info.get("kind", _BEHAVIOR_PARSE_BINARY)) == _BEHAVIOR_PARSE_TIMESTAMPS:
                return extract_sync_events(np.asarray(behaviors[name], float), None, min_interval_s=min_interval)
            t = np.asarray(info.get("time", np.array([], float)), float)
            x = np.asarray(behaviors[name], float)
            return extract_sync_events(t, x, mode=mode, threshold=threshold, min_interval_s=min_interval)
        if kind == "trajectory":
            traj = info.get("trajectory") or {}
            if name not in traj:
                raise ValueError(f"Camera sync column not found: {name}")
            t = np.asarray(info.get("trajectory_time", np.array([], float)), float)
            if t.size == 0:
                t = np.asarray(info.get("time", np.array([], float)), float)
            x = np.asarray(traj[name], float)
            return extract_sync_events(t, x, mode=mode, threshold=threshold, min_interval_s=min_interval)
        raise ValueError(f"Unsupported camera sync source: {kind}")

    def _sync_fiber_trace_for_proc(self, proc: ProcessedTrial) -> Tuple[np.ndarray, np.ndarray]:
        source = self.combo_sync_fiber_source.currentData() if hasattr(self, "combo_sync_fiber_source") else "__embedded__"
        if not isinstance(source, str):
            source = "__embedded__"
        if source == "__embedded__":
            t = np.asarray(getattr(proc, "time", np.array([], float)), float)
            x = getattr(proc, "dio", None)
            if x is None:
                triggers = getattr(proc, "triggers", {}) or {}
                usable: List[Tuple[str, np.ndarray]] = []
                for name, vals in triggers.items():
                    arr = np.asarray(vals, float)
                    if arr.size == t.size and int(np.sum(np.isfinite(arr))) >= 2:
                        usable.append((str(name), arr))
                if len(usable) == 1:
                    return t, usable[0][1]
                if usable:
                    names = ", ".join(name for name, _vals in usable[:6])
                    raise ValueError(
                        "Choose the photometry sync column for this CSV "
                        f"(available: {names})."
                    )
                raise ValueError("This processed file has no embedded DIO sync trace.")
            return t, np.asarray(x, float)
        if source.startswith("dio::"):
            name = source.split("::", 1)[1]
            triggers = getattr(proc, "triggers", {}) or {}
            if name in triggers and getattr(proc, "time", None) is not None:
                return np.asarray(proc.time, float), np.asarray(triggers[name], float)
            if getattr(proc, "dio", None) is not None and name == str(getattr(proc, "dio_name", "") or ""):
                return np.asarray(proc.time, float), np.asarray(proc.dio, float)
            key = (proc.path, name)
            if key not in self._dio_cache:
                self.requestDioData.emit(proc.path, name)
                raise RuntimeError(f"Requested A/D channel '{name}'. Click Preview/Apply again after it loads.")
            return self._dio_cache[key]
        raise ValueError(f"Unsupported photometry sync source: {source}")

    def _compute_time_sync_for_proc(self, proc: ProcessedTrial) -> SyncResult:
        camera_events = self._sync_camera_events_for_proc(proc)
        fiber_time, fiber_sync = self._sync_fiber_trace_for_proc(proc)
        mode = self._sync_mode_key(self.combo_sync_fiber_mode.currentText())
        threshold = None if self.cb_sync_auto_threshold.isChecked() else float(self.spin_sync_threshold.value())
        fiber_events = extract_sync_events(
            fiber_time,
            fiber_sync,
            mode=mode,
            threshold=threshold,
            min_interval_s=float(self.spin_sync_min_interval.value()),
        )
        method = "interpolation" if "interp" in self.combo_sync_method.currentText().lower() else "linear"
        return align_timebase(
            np.asarray(getattr(proc, "time", np.array([], float)), float),
            camera_events,
            fiber_events,
            method=method,
            max_offset=int(self.spin_sync_max_offset.value()),
            min_pairs=2,
        )

    def _selected_proc_for_sync(self) -> Optional[ProcessedTrial]:
        if not self._processed:
            return None
        file_id = ""
        if hasattr(self, "combo_individual_file"):
            file_id = self.combo_individual_file.currentText().strip()
        if file_id:
            for proc in self._processed:
                if self._file_id_for_proc(proc) == file_id:
                    return proc
        return self._processed[0]

    def _sync_report_text(self, proc: ProcessedTrial, result: SyncResult) -> str:
        rep = result.summary_dict()
        file_id = self._file_id_for_proc(proc)
        lines = [
            f"File: {file_id}",
            f"Status: {rep.get('status')} | method: {rep.get('method')}",
            f"Matched pulses: {rep.get('n_matched')} / camera {rep.get('n_camera_events')} / photometry {rep.get('n_fiber_events')}",
            f"Clock mapping: camera_time = {float(rep.get('slope', np.nan)):.9g} * photometry_time + {float(rep.get('intercept', np.nan)):.9g}",
            f"Median lag: {float(rep.get('median_lag_s', np.nan)) * 1000:.3f} ms",
            f"Residual RMS: {float(rep.get('rms_error_s', np.nan)) * 1000:.3f} ms",
            f"Max residual: {float(rep.get('max_abs_error_s', np.nan)) * 1000:.3f} ms",
            f"Clock drift: {float(rep.get('drift_ppm', np.nan)):.3f} ppm",
        ]
        warnings = rep.get("warnings", [])
        if isinstance(warnings, list) and warnings:
            lines.append("Warnings:")
            lines.extend(f"- {w}" for w in warnings)
        lines.append("")
        lines.append("Interpretation:")
        if str(rep.get("status")) == "ok":
            lines.append("The shared sync events support a usable camera-time column for this photometry trace.")
        else:
            lines.append("Inspect the pulse pairing and residual plot before using this alignment for analysis.")
        return "\n".join(lines)

    def _render_sync_result(self, proc: ProcessedTrial, result: SyncResult) -> None:
        self._last_sync_preview = result
        self.txt_sync_report.setPlainText(self._sync_report_text(proc, result))
        self.plot_sync_map.clear()
        self.plot_sync_residual.clear()
        self.plot_sync_map.setLabel("bottom", "Photometry sync time (s)")
        self.plot_sync_map.setLabel("left", "Camera / behavior sync time (s)")
        self.plot_sync_residual.setLabel("bottom", "Matched pulse")
        self.plot_sync_residual.setLabel("left", "Residual (ms)")
        cam = np.asarray(result.matched_camera_events, float)
        fib = np.asarray(result.matched_fiber_events, float)
        fit = np.asarray(result.fitted_camera_events, float)
        resid = np.asarray(result.residuals, float)
        if cam.size and fib.size:
            self.plot_sync_map.plot(fib, cam, pen=None, symbol="o", symbolSize=7,
                                    symbolBrush=pg.mkBrush(90, 190, 255, 220),
                                    symbolPen=pg.mkPen((90, 190, 255), width=1.0))
        if fib.size and fit.size:
            order = np.argsort(fib)
            self.plot_sync_map.plot(fib[order], fit[order], pen=pg.mkPen((255, 190, 90), width=1.5))
        if resid.size:
            x = np.arange(1, resid.size + 1, dtype=float)
            self.plot_sync_residual.plot(x, resid * 1000.0, pen=pg.mkPen((245, 210, 80), width=1.2),
                                         symbol="o", symbolSize=5,
                                         symbolBrush=pg.mkBrush(245, 210, 80, 180))
            self.plot_sync_residual.addLine(y=0.0, pen=pg.mkPen((180, 190, 210), style=QtCore.Qt.PenStyle.DashLine))
        sev = "ok" if result.status == "ok" else ("error" if result.status == "failed" else "warn")
        self.sync_status.set(
            f"{self._file_id_for_proc(proc)}: {result.status}, {cam.size} matched sync event(s), "
            f"RMS {result.rms_error_s * 1000:.2f} ms",
            sev,
        )

    def _store_sync_result(self, proc: ProcessedTrial, result: SyncResult) -> None:
        base = np.asarray(getattr(proc, "time", np.array([], float)), float)
        aligned = np.asarray(result.aligned_time, float)
        if aligned.size != base.size or aligned.size == 0 or not np.any(np.isfinite(aligned)):
            raise ValueError("Alignment did not produce a valid aligned time vector.")
        proc.sync_aligned_time = aligned
        report = result.summary_dict()
        report["camera_source"] = self.combo_sync_camera_column.currentText()
        report["fiber_source"] = self.combo_sync_fiber_source.currentText()
        report["created_utc"] = datetime.now(timezone.utc).isoformat()
        proc.sync_report = report
        self._sync_results_by_file[self._file_id_for_proc(proc)] = report

    def _preview_time_sync(self) -> None:
        proc = self._selected_proc_for_sync()
        if proc is None:
            self.sync_status.set("Load a processed recording first.", "warn")
            return
        try:
            result = self._compute_time_sync_for_proc(proc)
            self._render_sync_result(proc, result)
        except RuntimeError as exc:
            self.sync_status.set(str(exc), "info")
            self.statusUpdate.emit(str(exc), 5000)
        except Exception as exc:
            self.sync_status.set(f"Sync preview failed: {exc}", "error")
            self.statusUpdate.emit(f"Sync preview failed: {exc}", 5000)

    def _apply_time_sync_selected(self) -> None:
        proc = self._selected_proc_for_sync()
        if proc is None:
            self.sync_status.set("Load a processed recording first.", "warn")
            return
        try:
            result = self._compute_time_sync_for_proc(proc)
            self._store_sync_result(proc, result)
            self._render_sync_result(proc, result)
            self._refresh_sync_results_table()
            self._update_data_availability()
            if not self._autosave_restoring:
                self._project_dirty = True
            self._save_settings()
            if self.cb_sync_auto_recompute.isChecked():
                self._compute_psth()
                self._compute_spatial_heatmap()
            self.statusUpdate.emit(f"Aligned time saved for {self._file_id_for_proc(proc)}.", 5000)
        except RuntimeError as exc:
            self.sync_status.set(str(exc), "info")
            self.statusUpdate.emit(str(exc), 5000)
        except Exception as exc:
            self.sync_status.set(f"Sync apply failed: {exc}", "error")
            self.statusUpdate.emit(f"Sync apply failed: {exc}", 5000)

    def _apply_time_sync_batch(self) -> None:
        if not self._processed:
            self.sync_status.set("Load processed recordings first.", "warn")
            return
        progress = QtWidgets.QProgressDialog("Synchronizing files...", "Cancel", 0, len(self._processed), self)
        progress.setWindowModality(QtCore.Qt.WindowModality.WindowModal)
        progress.setMinimumDuration(300)
        ok_count = 0
        errors: List[str] = []
        for idx, proc in enumerate(self._processed, start=1):
            progress.setValue(idx - 1)
            progress.setLabelText(f"Synchronizing {self._file_id_for_proc(proc)}...")
            QtWidgets.QApplication.processEvents()
            if progress.wasCanceled():
                break
            try:
                result = self._compute_time_sync_for_proc(proc)
                self._store_sync_result(proc, result)
                ok_count += 1
                if proc is self._selected_proc_for_sync():
                    self._render_sync_result(proc, result)
            except RuntimeError as exc:
                errors.append(f"{self._file_id_for_proc(proc)}: {exc}")
            except Exception as exc:
                errors.append(f"{self._file_id_for_proc(proc)}: {exc}")
        progress.setValue(len(self._processed))
        self._refresh_sync_results_table()
        self._update_data_availability()
        if ok_count:
            if not self._autosave_restoring:
                self._project_dirty = True
            self._save_settings()
            if self.cb_sync_auto_recompute.isChecked():
                self._compute_psth()
                self._compute_spatial_heatmap()
        if errors:
            self.sync_status.set(f"Aligned {ok_count} file(s); {len(errors)} file(s) need attention.", "warn")
            self.txt_sync_report.setPlainText("Batch synchronization warnings:\n" + "\n".join(errors[:30]))
        else:
            self.sync_status.set(f"Aligned {ok_count} file(s).", "ok")
        self.statusUpdate.emit(f"Time synchronization batch finished: {ok_count} file(s).", 5000)

    def _refresh_sync_results_table(self) -> None:
        if not hasattr(self, "tbl_sync_results"):
            return
        rows: List[Tuple[str, Dict[str, object]]] = []
        self._sync_results_by_file = {}
        for proc in self._processed:
            fid = self._file_id_for_proc(proc)
            report = getattr(proc, "sync_report", {}) or {}
            if report:
                self._sync_results_by_file[fid] = dict(report)
            rows.append((fid, dict(report)))
        self.tbl_sync_results.setRowCount(len(rows))
        for r, (fid, report) in enumerate(rows):
            values = [
                fid,
                str(report.get("n_matched", "")) if report else "",
                f"{float(report.get('rms_error_s', np.nan)) * 1000:.2f}" if report else "",
                f"{float(report.get('median_lag_s', np.nan)) * 1000:.2f}" if report else "",
                f"{float(report.get('drift_ppm', np.nan)):.2f}" if report else "",
                str(report.get("status", "")) if report else "not aligned",
            ]
            for c, val in enumerate(values):
                item = QtWidgets.QTableWidgetItem(val)
                if c == 5 and str(val).lower() == "ok":
                    item.setForeground(QtGui.QBrush(QtGui.QColor("#5ee0a1")))
                elif c == 5 and str(val).lower() not in ("ok", "not aligned"):
                    item.setForeground(QtGui.QBrush(QtGui.QColor("#ffd166")))
                self.tbl_sync_results.setItem(r, c, item)
        try:
            self.tbl_sync_results.resizeColumnsToContents()
            self.tbl_sync_results.horizontalHeader().setSectionResizeMode(0, QtWidgets.QHeaderView.ResizeMode.Stretch)
        except Exception:
            pass

    def _export_sync_aligned_files(self) -> None:
        ready = [
            proc for proc in self._processed
            if getattr(proc, "sync_aligned_time", None) is not None
            and np.asarray(getattr(proc, "sync_aligned_time"), float).size == np.asarray(getattr(proc, "time", []), float).size
        ]
        if not ready:
            self.sync_status.set("No aligned time columns to export. Apply synchronization first.", "warn")
            return
        out_dir = QtWidgets.QFileDialog.getExistingDirectory(self, "Select aligned export folder", self._export_start_dir())
        if not out_dir:
            return
        self._remember_export_dir(out_dir)
        fmt = self.combo_sync_export_format.currentText().lower()
        do_csv = "csv" in fmt
        do_h5 = "h5" in fmt
        if "csv + h5" in fmt:
            do_csv = do_h5 = True
        import csv
        count = 0
        for proc in ready:
            fid = re.sub(r"[^A-Za-z0-9_\\-]+", "_", self._file_id_for_proc(proc)).strip("_") or "aligned"
            t = np.asarray(proc.time, float)
            aligned = np.asarray(proc.sync_aligned_time, float)
            out = np.asarray(proc.output if proc.output is not None else np.full_like(t, np.nan), float)
            raw = np.asarray(proc.raw_signal if proc.raw_signal is not None else np.full_like(t, np.nan), float)
            ref = np.asarray(proc.raw_reference if proc.raw_reference is not None else np.full_like(t, np.nan), float)
            dio = np.asarray(proc.dio if proc.dio is not None else np.full_like(t, np.nan), float)
            n = min(t.size, aligned.size, out.size, raw.size, ref.size, dio.size)
            report = getattr(proc, "sync_report", {}) or {}
            if do_csv:
                csv_path = os.path.join(out_dir, f"{fid}_time_aligned.csv")
                with open(csv_path, "w", newline="") as f:
                    w = csv.writer(f)
                    w.writerow(["# sync_report_json", json.dumps(report, default=str)])
                    w.writerow(["time", "time_aligned", "output", "raw_465", "raw_405", "dio"])
                    for i in range(n):
                        w.writerow([float(t[i]), float(aligned[i]), float(out[i]), float(raw[i]), float(ref[i]), float(dio[i])])
            if do_h5:
                h5_path = os.path.join(out_dir, f"{fid}_time_aligned.h5")
                with h5py.File(h5_path, "w") as h5:
                    h5.attrs["project_type"] = "pyber_time_aligned_processed"
                    g = h5.create_group("data")
                    g.attrs["output_label"] = str(getattr(proc, "output_label", "") or "")
                    g.attrs["output_context"] = str(getattr(proc, "output_context", "") or "")
                    g.attrs["dio_name"] = str(getattr(proc, "dio_name", "") or "")
                    g.attrs["sync_report_json"] = json.dumps(report, default=str)
                    self._write_h5_numeric(g, "time", t[:n])
                    self._write_h5_numeric(g, "time_aligned", aligned[:n])
                    self._write_h5_numeric(g, "output", out[:n])
                    self._write_h5_numeric(g, "raw_465", raw[:n])
                    self._write_h5_numeric(g, "raw_405", ref[:n])
                    self._write_h5_numeric(g, "dio", dio[:n])
            count += 1
        self.sync_status.set(f"Exported aligned time files for {count} recording(s).", "ok")
        self.statusUpdate.emit(f"Aligned time export complete: {out_dir}", 5000)

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
            on = self._map_photometry_times_for_proc(proc, on)
            off = self._map_photometry_times_for_proc(proc, off)
            if on.size and off.size and on.size == off.size:
                dur = off - on
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

    def _clear_event_annotations(self) -> None:
        for ln in list(self.event_lines):
            try:
                self.plot_trace.removeItem(ln)
            except Exception:
                pass
        self.event_lines = []
        for lab in list(self._event_labels):
            try:
                self.plot_trace.removeItem(lab)
            except Exception:
                pass
        self._event_labels = []
        for reg in list(self._event_regions):
            try:
                self.plot_trace.removeItem(reg)
            except Exception:
                pass
        self._event_regions = []

    def _render_visible_event_annotations(self, *_args: object) -> None:
        self._clear_event_annotations()
        ev = np.asarray(getattr(self, "_trace_preview_events", np.array([], float)), float)
        if ev.size == 0:
            return
        dur = np.asarray(getattr(self, "_trace_preview_durations", np.array([], float)), float)
        if dur.size != ev.size:
            dur = np.full(ev.shape, np.nan, dtype=float)
        min_y, max_y = getattr(self, "_trace_preview_y_bounds", (-1.0, 1.0))
        try:
            x0, x1 = self.plot_trace.getViewBox().viewRange()[0]
        except Exception:
            x0, x1 = float(np.nanmin(ev)), float(np.nanmax(ev))
        if x1 < x0:
            x0, x1 = x1, x0
        finite_ev = np.isfinite(ev)
        finite_dur = np.isfinite(dur)
        overlaps = finite_ev & (ev <= x1) & (
            (ev >= x0) | (finite_dur & ((ev + np.maximum(dur, 0.0)) >= x0))
        )
        visible_idx = np.where(overlaps)[0]
        for idx in visible_idx:
            et = float(ev[idx])
            label = pg.TextItem(str(int(idx) + 1), color=(200, 200, 200))
            label.setPos(et, float(max_y))
            self.plot_trace.addItem(label)
            self._event_labels.append(label)
            if idx < dur.size and np.isfinite(dur[idx]):
                t1 = et + max(0.0, float(dur[idx]))
                if t1 > et:
                    reg = pg.LinearRegionItem(values=(et, t1), brush=(200, 200, 200, 40), movable=False)
                    reg.setZValue(1)
                    self.plot_trace.addItem(reg)
                    self._event_regions.append(reg)

    def _clear_trace_preview(self) -> None:
        self.curve_trace.setData([], [])
        self.curve_behavior.setData([], [])
        self.curve_peak_markers.setData([], [])
        if hasattr(self, "curve_event_markers"):
            self.curve_event_markers.setData([], [])
        self._trace_preview_events = np.array([], float)
        self._trace_preview_durations = np.array([], float)
        self._clear_event_annotations()
        for ln in list(self._signal_peak_lines):
            try:
                self.plot_trace.removeItem(ln)
            except Exception:
                pass
        self._signal_peak_lines = []
        try:
            self.plot_trace.setXRange(0.0, 1.0, padding=0)
            self.plot_trace.setYRange(0.0, 1.0, padding=0)
            self.plot_trace.setTitle("Trace preview")
        except Exception:
            pass

    def _clear_psth_cache(self) -> None:
        self._last_mat = None
        self._last_tvec = None
        self._last_global_metrics = None
        self._last_events = np.array([], float)
        self._last_event_rows = []
        self._per_file_mats = {}
        self._per_file_labels = {}
        self._group_mat = None
        self._group_tvec = None
        self._group_labels = []
        self._group_trial_mat = None
        self._group_trial_tvec = None
        self._group_trial_labels = []
        self._all_file_ids = []
        self._last_display_labels = []
        self._last_psth_display_level = "trials"
        self._psth_excluded_files = {}

    def _clear_psth_visuals(self) -> None:
        self._clear_trace_preview()
        self._render_heatmap(np.zeros((0, 0), dtype=float), np.array([], dtype=float), labels=[])
        self._render_avg(np.zeros((0, 0), dtype=float), np.array([], dtype=float))
        self._render_metrics(np.zeros((0, 0), dtype=float), np.array([], dtype=float))
        self._render_duration_hist(np.array([], dtype=float))
        self._render_global_metrics()
        if hasattr(self, "lbl_global_metrics"):
            self.lbl_global_metrics.setText("Global metrics: -")
        if hasattr(self, "lbl_plot_file"):
            self.lbl_plot_file.setText("File: (none)")
        if hasattr(self, "combo_individual_file"):
            try:
                self.combo_individual_file.blockSignals(True)
                self.combo_individual_file.clear()
            finally:
                self.combo_individual_file.blockSignals(False)
        try:
            self.plot_avg.setXRange(0.0, 1.0, padding=0)
            self.plot_metrics.setYRange(0.0, 1.0, padding=0)
            self.plot_global.setYRange(0.0, 1.0, padding=0)
        except Exception:
            pass

    def _update_trace_preview(self) -> None:
        # show first processed trace
        if not self._processed:
            self._clear_trace_preview()
            return

        # In individual mode, show the selected file's trace
        proc = self._processed[0]
        if hasattr(self, "tab_visual_mode") and self.tab_visual_mode.currentIndex() == 0:
            sel_id = self.combo_individual_file.currentText().strip() if hasattr(self, "combo_individual_file") else ""
            if sel_id:
                for p in self._processed:
                    fid = os.path.splitext(os.path.basename(p.path))[0] if p.path else "import"
                    if fid == sel_id:
                        proc = p
                        break
        t = self._proc_time(proc)
        y = proc.output if proc.output is not None else np.full_like(t, np.nan)

        self.curve_trace.setData(t, y, connect="finite", skipFiniteCheck=True)
        self._update_behavior_overlay(proc)

        # draw event lines if possible
        self._clear_event_annotations()
        if hasattr(self, "curve_event_markers"):
            self.curve_event_markers.setData([], [])
        self._trace_preview_events = np.array([], float)
        self._trace_preview_durations = np.array([], float)

        ev, dur = self._get_events_for_proc(proc)
        ev, dur = self._filter_events(ev, dur)
        if ev.size:
            dur = np.asarray(dur, float) if dur is not None else np.array([], float)
            if dur.size != ev.size:
                dur = np.full(ev.shape, np.nan, dtype=float)
            y_arr = np.asarray(y, float)
            finite_y = y_arr[np.isfinite(y_arr)]
            if finite_y.size:
                min_y = float(np.nanmin(finite_y))
                max_y = float(np.nanmax(finite_y))
                pad = 0.03 * max(1e-12, max_y - min_y)
                min_y -= pad
                max_y += pad
            else:
                min_y, max_y = -1.0, 1.0
            ev_arr = np.asarray(ev, float)
            line_x = np.empty(ev_arr.size * 3, dtype=float)
            line_y = np.empty(ev_arr.size * 3, dtype=float)
            line_x[0::3] = ev_arr
            line_x[1::3] = ev_arr
            line_x[2::3] = np.nan
            line_y[0::3] = min_y
            line_y[1::3] = max_y
            line_y[2::3] = np.nan
            if hasattr(self, "curve_event_markers"):
                self.curve_event_markers.setData(line_x, line_y, connect="finite")
            self._trace_preview_events = ev_arr
            self._trace_preview_durations = dur
            self._trace_preview_y_bounds = (float(min_y), float(max_y))
            self._render_visible_event_annotations()
            self.plot_trace.setTitle(f"Trace preview ({int(ev.size)} events)")
        else:
            self._trace_preview_events = np.array([], float)
            self._trace_preview_durations = np.array([], float)
            self.plot_trace.setTitle("Trace preview")
        self._refresh_signal_overlay()

    def _update_behavior_overlay(self, proc: ProcessedTrial) -> None:
        # The trace preview should only show the processed signal trace.
        # Behavior/event data remain available for alignment and analysis,
        # but the binary edge overlay is intentionally not rendered here.
        self.curve_behavior.setData([], [])

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
                targets.append((file_id, np.asarray(self._proc_time(proc), float), np.asarray(proc.output, float)))
            return targets

        if self.combo_signal_file.count() == 0:
            return []
        idx = self.combo_signal_file.currentIndex()
        idx = max(0, min(idx, len(self._processed) - 1))
        proc = self._processed[idx]
        if proc.time is None or proc.output is None:
            return []
        file_id = os.path.splitext(os.path.basename(proc.path))[0] if proc.path else "import"
        targets.append((file_id, np.asarray(self._proc_time(proc), float), np.asarray(proc.output, float)))
        return targets

    def _preprocess_signal_for_peaks(self, t: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        t = np.asarray(t, float)
        y = np.asarray(y, float)
        m = np.isfinite(t) & np.isfinite(y)
        t = t[m]
        y = y[m]
        if t.size < 3:
            return np.array([], float), np.array([], float), np.array([], float)

        dt = float(np.nanmedian(np.diff(t))) if t.size > 2 else np.nan
        if not np.isfinite(dt) or dt <= 0:
            dt = 1.0

        baseline_mode = self.combo_peak_baseline.currentText()
        baseline_window_sec = max(0.1, float(self.spin_peak_baseline_window.value()))
        win = max(3, int(round(baseline_window_sec / dt)))
        if win % 2 == 0:
            win += 1

        y_trace = y.copy()
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

        return t, y_proc, y_trace

    # ------------------------------------------------------------------
    # Baseline-prominence helpers
    # ------------------------------------------------------------------

    def _signal_baseline_mask(
        self,
        t: np.ndarray,
        finite: np.ndarray,
        file_id: str = "",
    ) -> Tuple[np.ndarray, str, str]:
        """Compute the baseline boolean mask for `t` according to the user's
        Baseline-source choice. Returns (mask, source_label, explanation)."""
        t = np.asarray(t, float)
        finite = np.asarray(finite, bool)

        # Default: full recording.
        source = "whole"
        if hasattr(self, "combo_signal_baseline_source"):
            data = self.combo_signal_baseline_source.currentData()
            if isinstance(data, str) and data:
                source = data

        if source == "window":
            start_s = float(self.spin_signal_baseline_start.value())
            end_s = float(self.spin_signal_baseline_end.value())
            t_finite = t[finite]
            if t_finite.size == 0:
                return finite.copy(), "whole", "no finite samples; fell back to whole trace"
            t_lo = float(np.nanmin(t_finite))
            # User-friendly: start/end are interpreted as seconds from the
            # start of the trace. Setting end=0 means "until the end".
            abs_start = t_lo + max(0.0, start_s)
            abs_end = t_lo + max(0.0, end_s) if end_s > 0 else float(np.nanmax(t_finite))
            if abs_end <= abs_start:
                return finite.copy(), "whole", f"window {start_s:g}->{end_s:g}s is empty; fell back to whole trace"
            mask = finite & (t >= abs_start) & (t <= abs_end)
            n = int(np.sum(mask))
            if n < 3:
                return finite.copy(), "whole", "window has too few samples; fell back to whole trace"
            dur = float(abs_end - abs_start)
            return mask, "window", f"window [{start_s:g}, {end_s:g}] s -> {n} samples ({dur:.1f} s)"

        if source == "exclude":
            intervals = self._signal_baseline_exclusion_intervals(file_id)
            if not intervals:
                return finite.copy(), "whole", "no behavior bouts ticked; baseline = whole trace"
            pad = max(0.0, float(self.spin_signal_baseline_pad.value()))
            mask = finite.copy()
            total_excluded = 0.0
            for lo, hi in intervals:
                lo_p, hi_p = lo - pad, hi + pad
                in_bout = (t >= lo_p) & (t <= hi_p)
                mask &= ~in_bout
                total_excluded += max(0.0, hi_p - lo_p)
            if int(np.sum(mask)) < 3:
                return finite.copy(), "whole", "behavior exclusion left too few samples; fell back to whole trace"
            return mask, "exclude_behaviors", (
                f"removed {len(intervals)} bout(s) (~{total_excluded:.1f} s with pad {pad:g} s) -> "
                f"{int(np.sum(mask))} baseline samples"
            )

        # Whole recording.
        return finite.copy(), "whole", "baseline = whole recording (MAD-noise fallback)"

    def _signal_baseline_exclusion_intervals(self, file_id: str) -> List[Tuple[float, float]]:
        """Return [(lo, hi), ...] intervals to remove from baseline, based on
        the user's tick boxes in `list_signal_baseline_exclude` and the loaded
        `_behavior_sources`. Handles both binary-state and timestamp behavior
        files."""
        intervals: List[Tuple[float, float]] = []
        if not hasattr(self, "list_signal_baseline_exclude") or self.list_signal_baseline_exclude.count() == 0:
            return intervals
        if not getattr(self, "_behavior_sources", None):
            return intervals

        ticked: List[str] = []
        for i in range(self.list_signal_baseline_exclude.count()):
            item = self.list_signal_baseline_exclude.item(i)
            if item.checkState() == QtCore.Qt.CheckState.Checked:
                ticked.append(item.data(QtCore.Qt.ItemDataRole.UserRole) or item.text())
        if not ticked:
            return intervals

        # Resolve which behavior source to use.
        info: Optional[Dict[str, Any]] = None
        if hasattr(self, "combo_signal_baseline_behavior_file"):
            stem = self.combo_signal_baseline_behavior_file.currentData()
            if isinstance(stem, str) and stem and stem in self._behavior_sources:
                info = self._behavior_sources[stem]
        if info is None:
            # Auto-match by file_id stem.
            key = str(file_id or "").strip().lower()
            for stem, val in self._behavior_sources.items():
                if str(stem).strip().lower() in key or key in str(stem).strip().lower():
                    info = val
                    break
            if info is None and self._behavior_sources:
                info = next(iter(self._behavior_sources.values()))
        if not isinstance(info, dict):
            return intervals

        kind = str(info.get("kind", _BEHAVIOR_PARSE_BINARY))
        behaviors = info.get("behaviors") or {}
        beh_time = np.asarray(info.get("time", np.array([], float)), float)

        for name in ticked:
            values = behaviors.get(name)
            if values is None:
                continue
            if kind == _BEHAVIOR_PARSE_TIMESTAMPS:
                # Timestamps style: each entry is a single event time.
                arr = np.asarray(values, float)
                arr = arr[np.isfinite(arr)]
                for ts in arr:
                    intervals.append((float(ts), float(ts)))
                continue
            # Binary states style: extract on/off intervals.
            x = np.asarray(values, float)
            if beh_time.size != x.size or beh_time.size < 2:
                continue
            b = x > 0.5
            rising = np.where((~b[:-1]) & (b[1:]))[0] + 1
            falling = np.where(b[:-1] & (~b[1:]))[0] + 1
            if b.size and bool(b[0]):
                rising = np.concatenate([[0], rising])
            if b.size and bool(b[-1]):
                falling = np.concatenate([falling, [b.size]])
            n_pairs = min(rising.size, falling.size)
            for i in range(n_pairs):
                lo_i = int(rising[i])
                hi_i = int(min(falling[i], beh_time.size - 1))
                if 0 <= lo_i <= hi_i < beh_time.size:
                    intervals.append((float(beh_time[lo_i]), float(beh_time[hi_i])))
        return intervals

    def _signal_baseline_prominence_stats(
        self,
        t: np.ndarray,
        y: np.ndarray,
        min_prominence: float,
        file_id: str = "",
    ) -> Dict[str, float]:
        t = np.asarray(t, float)
        y = np.asarray(y, float)
        finite = np.isfinite(t) & np.isfinite(y)
        if t.size < 3 or y.size != t.size or not np.any(finite):
            return {
                "scale": np.nan,
                "baseline_median": np.nan,
                "n_baseline_peaks": 0.0,
                "baseline_duration_s": 0.0,
                "scale_source": "unavailable",
                "mad_noise_sigma": np.nan,
                "baseline_mask_source": "unavailable",
                "baseline_explanation": "trace has no finite samples",
            }

        keep, mask_source, explanation = self._signal_baseline_mask(t, finite, file_id=file_id)
        if int(np.sum(keep)) < 3:
            keep = finite
            mask_source = "whole"
            explanation = "fell back to whole recording"

        baseline = y[keep]
        baseline_median = float(np.nanmedian(baseline)) if baseline.size else np.nan
        if not np.isfinite(baseline_median):
            return {
                "scale": np.nan,
                "baseline_median": np.nan,
                "n_baseline_peaks": 0.0,
                "baseline_duration_s": 0.0,
                "scale_source": "unavailable",
                "mad_noise_sigma": np.nan,
                "baseline_mask_source": mask_source,
                "baseline_explanation": explanation,
            }

        centered_baseline = np.asarray(baseline, float) - baseline_median
        mad_stats = self._signal_mad_noise_stats(centered_baseline)
        mad_sigma = float(mad_stats.get("noise_sigma", np.nan))
        if not np.isfinite(mad_sigma) or mad_sigma <= 1e-12:
            full_centered = np.asarray(y[finite], float) - float(np.nanmedian(y[finite]))
            full_mad_stats = self._signal_mad_noise_stats(full_centered)
            mad_sigma = float(full_mad_stats.get("noise_sigma", np.nan))
        try:
            from scipy.signal import find_peaks
            peaks, props = find_peaks(centered_baseline, prominence=max(0.0, float(min_prominence)))
        except Exception:
            peaks = np.array([], int)
            props = {}
        proms = np.asarray(props.get("prominences", np.array([], float)), float)
        proms = proms[np.isfinite(proms) & (proms > 1e-12)]
        if proms.size == 0:
            scale = mad_sigma if np.isfinite(mad_sigma) and mad_sigma > 1e-12 else np.nan
            scale_source = "mad_noise_fallback" if np.isfinite(scale) else "unavailable"
        else:
            top_count = max(1, int(np.ceil(proms.size * 0.10)))
            scale = float(np.nanmean(np.sort(proms)[-top_count:]))
            scale_source = "baseline_peak_prominence"

        duration = float(np.nanmax(t[keep]) - np.nanmin(t[keep])) if np.sum(keep) >= 2 else 0.0
        return {
            "scale": scale,
            "baseline_median": baseline_median,
            "n_baseline_peaks": float(proms.size),
            "baseline_duration_s": duration,
            "scale_source": scale_source,
            "mad_noise_sigma": mad_sigma,
            "baseline_mask_source": mask_source,
            "baseline_explanation": explanation,
            "baseline_n_samples": float(int(np.sum(keep))),
        }

    @staticmethod
    def _signal_mad_noise_stats(y: np.ndarray) -> Dict[str, float]:
        arr = np.asarray(y, float)
        arr = arr[np.isfinite(arr)]
        if arr.size < 5:
            return {"center": np.nan, "mad": np.nan, "noise_sigma": np.nan, "n_samples": float(arr.size)}

        center = float(np.nanmedian(arr))
        abs_dev = np.abs(arr - center)
        mad = float(np.nanmedian(abs_dev))
        sigma = 1.4826 * mad

        # Re-estimate from the central mass so large transients do not inflate the noise estimate.
        if np.isfinite(sigma) and sigma > 1e-12:
            keep = abs_dev <= (3.0 * sigma)
            if np.sum(keep) >= max(5, int(0.10 * arr.size)):
                core = arr[keep]
                center = float(np.nanmedian(core))
                mad = float(np.nanmedian(np.abs(core - center)))
                sigma = 1.4826 * mad

        if not np.isfinite(sigma) or sigma <= 1e-12:
            q25, q75 = np.nanpercentile(arr, [25.0, 75.0])
            sigma = float((q75 - q25) / 1.349) if np.isfinite(q25) and np.isfinite(q75) else np.nan
        if not np.isfinite(sigma) or sigma <= 1e-12:
            sigma = float(np.nanstd(arr))
        if not np.isfinite(sigma) or sigma <= 1e-12:
            sigma = np.nan

        return {
            "center": center,
            "mad": mad,
            "noise_sigma": sigma,
            "n_samples": float(arr.size),
        }

    @staticmethod
    def _trapz_area(y: np.ndarray, x: np.ndarray) -> float:
        try:
            return float(np.trapezoid(y, x))
        except AttributeError:
            return float(np.trapz(y, x))

    def _refresh_signal_overlay(self) -> None:
        for ln in self._signal_peak_lines:
            try:
                self.plot_trace.removeItem(ln)
            except Exception:
                pass
        self._signal_peak_lines = []
        for item in self._signal_noise_items:
            try:
                self.plot_trace.removeItem(item)
            except Exception:
                pass
        self._signal_noise_items = []
        self.curve_peak_markers.setData([], [])

        if not self.last_signal_events or not self._processed:
            return

        current_file = self._current_signal_overlay_file_id()
        self._draw_signal_noise_overlay(current_file)

        if not self.cb_peak_overlay.isChecked():
            return

        file_ids = self.last_signal_events.get("file_ids", [])
        times = np.asarray(self.last_signal_events.get("peak_times_sec", np.array([], float)), float)
        heights = np.asarray(self.last_signal_events.get("peak_trace_values", np.array([], float)), float)
        if heights.size != times.size:
            heights = np.asarray(self.last_signal_events.get("peak_heights", np.array([], float)), float)
        if times.size == 0 or heights.size == 0:
            return
        if current_file and file_ids and len(file_ids) == times.size:
            mask = np.asarray([str(fid) == current_file for fid in file_ids], bool)
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

    def _draw_signal_noise_overlay(self, current_file: str) -> None:
        if not getattr(self, "cb_peak_noise_overlay", None) or not self.cb_peak_noise_overlay.isChecked():
            return
        overlays = self.last_signal_events.get("noise_overlay_by_file", {}) if self.last_signal_events else {}
        if not isinstance(overlays, dict) or not overlays:
            return
        overlay = overlays.get(str(current_file or ""))
        if overlay is None and len(overlays) == 1:
            overlay = next(iter(overlays.values()))
        if not isinstance(overlay, dict):
            return

        t = np.asarray(overlay.get("time", np.array([], float)), float)
        y = np.asarray(overlay.get("detection_trace", np.array([], float)), float)
        if t.size != y.size or t.size < 2:
            return

        step = max(1, int(np.ceil(t.size / 5000)))
        trace_item = pg.PlotDataItem(
            t[::step],
            y[::step],
            pen=pg.mkPen((80, 220, 220, 150), width=1.0, style=QtCore.Qt.PenStyle.DashLine),
            name="detection trace",
        )
        trace_item.setZValue(8)
        self.plot_trace.addItem(trace_item)
        self._signal_noise_items.append(trace_item)

        center = float(overlay.get("center", np.nan))
        sigma = float(overlay.get("noise_sigma", np.nan))
        used_prominence = float(overlay.get("used_prominence", np.nan))
        for y0, color, width, style in (
            (center, (80, 220, 220, 170), 1.0, QtCore.Qt.PenStyle.DotLine),
            (center + sigma, (80, 220, 220, 110), 0.8, QtCore.Qt.PenStyle.DotLine),
            (center - sigma, (80, 220, 220, 110), 0.8, QtCore.Qt.PenStyle.DotLine),
            (center + used_prominence, (255, 210, 80, 190), 1.2, QtCore.Qt.PenStyle.DashLine),
        ):
            if not np.isfinite(y0):
                continue
            ln = pg.InfiniteLine(
                pos=float(y0),
                angle=0,
                pen=pg.mkPen(color, width=width, style=style),
                movable=False,
            )
            ln.setZValue(9)
            self.plot_trace.addItem(ln)
            self._signal_noise_items.append(ln)

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
        all_signal_heights: List[float] = []
        all_trace_values: List[float] = []
        all_proms: List[float] = []
        all_norm_proms: List[float] = []
        all_norm_scales: List[float] = []
        all_mad_sigmas: List[float] = []
        all_auto_prominence_thresholds: List[float] = []
        all_widths_sec: List[float] = []
        all_auc: List[float] = []
        all_file_ids: List[str] = []
        normalization_by_file: Dict[str, Dict[str, float]] = {}
        mad_threshold_by_file: Dict[str, Dict[str, float]] = {}
        noise_overlay_by_file: Dict[str, Dict[str, object]] = {}
        normalize_amplitude = bool(self.cb_peak_norm_prominence.isChecked())
        auto_mad = bool(self.cb_peak_auto_mad.isChecked())
        mad_multiplier = float(self.spin_peak_mad_multiplier.value())

        for file_id, t_raw, y_raw in targets:
            t, y, y_trace = self._preprocess_signal_for_peaks(t_raw, y_raw)
            if t.size < 5:
                continue
            dt = float(np.nanmedian(np.diff(t)))
            if not np.isfinite(dt) or dt <= 0:
                continue

            manual_prominence = max(0.0, float(self.spin_peak_prominence.value()))
            prominence = manual_prominence
            mad_stats = self._signal_mad_noise_stats(y)
            mad_sigma = float(mad_stats.get("noise_sigma", np.nan))
            auto_prominence = np.nan
            if auto_mad:
                if np.isfinite(mad_sigma) and mad_sigma > 1e-12:
                    auto_prominence = max(0.0, mad_multiplier * mad_sigma)
                    prominence = auto_prominence
                mad_threshold_by_file[str(file_id)] = {
                    **mad_stats,
                    "multiplier": mad_multiplier,
                    "auto_prominence": auto_prominence,
                    "fallback_manual_prominence": manual_prominence,
                    "used_prominence": prominence,
                }
            noise_overlay_by_file[str(file_id)] = {
                "time": t.copy(),
                "detection_trace": y.copy(),
                "center": float(mad_stats.get("center", np.nan)),
                "mad": float(mad_stats.get("mad", np.nan)),
                "noise_sigma": mad_sigma,
                "auto_prominence": auto_prominence,
                "manual_prominence": manual_prominence,
                "used_prominence": prominence,
            }
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

            p_signal_heights = np.asarray(y[peaks], float)
            p_trace_values = np.asarray(y_trace[peaks], float) if y_trace.size == y.size else p_signal_heights.copy()
            p_proms = np.asarray(props.get("prominences", np.full(peaks.size, np.nan)), float)
            p_heights = p_signal_heights.copy()
            p_norm_proms = np.full(peaks.size, np.nan, float)
            p_norm_scales = np.full(peaks.size, np.nan, float)

            if normalize_amplitude:
                norm_stats = self._signal_baseline_prominence_stats(t, y, prominence, file_id=str(file_id))
                normalization_by_file[str(file_id)] = norm_stats
                scale = float(norm_stats.get("scale", np.nan))
                baseline_median = float(norm_stats.get("baseline_median", np.nan))
                if np.isfinite(scale) and scale > 1e-12 and np.isfinite(baseline_median):
                    p_heights = (p_signal_heights - baseline_median) / scale
                    p_norm_proms = p_proms / scale
                    p_norm_scales[:] = scale
                else:
                    p_heights = np.full(peaks.size, np.nan, float)
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
                auc_vals.append(self._trapz_area(y[i0:i1], t[i0:i1]))

            all_times.extend(t[peaks].tolist())
            all_idx.extend(peaks.tolist())
            all_heights.extend(np.asarray(p_heights, float).tolist())
            all_signal_heights.extend(np.asarray(p_signal_heights, float).tolist())
            all_trace_values.extend(np.asarray(p_trace_values, float).tolist())
            all_proms.extend(np.asarray(p_proms, float).tolist())
            all_norm_proms.extend(np.asarray(p_norm_proms, float).tolist())
            all_norm_scales.extend(np.asarray(p_norm_scales, float).tolist())
            all_mad_sigmas.extend([mad_sigma] * peaks.size)
            all_auto_prominence_thresholds.extend([auto_prominence] * peaks.size)
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
        peak_signal_heights = np.asarray(all_signal_heights, float)
        peak_trace_values = np.asarray(all_trace_values, float)
        peak_proms = np.asarray(all_proms, float)
        peak_norm_proms = np.asarray(all_norm_proms, float)
        peak_norm_scales = np.asarray(all_norm_scales, float)
        peak_mad_sigmas = np.asarray(all_mad_sigmas, float)
        peak_auto_prominence_thresholds = np.asarray(all_auto_prominence_thresholds, float)
        peak_widths_sec = np.asarray(all_widths_sec, float)
        peak_auc = np.asarray(all_auc, float)

        sort_idx = np.argsort(peak_times)
        peak_times = peak_times[sort_idx]
        peak_idx = peak_idx[sort_idx]
        peak_heights = peak_heights[sort_idx]
        peak_signal_heights = peak_signal_heights[sort_idx]
        peak_trace_values = peak_trace_values[sort_idx]
        peak_proms = peak_proms[sort_idx]
        peak_norm_proms = peak_norm_proms[sort_idx]
        peak_norm_scales = peak_norm_scales[sort_idx]
        peak_mad_sigmas = peak_mad_sigmas[sort_idx]
        peak_auto_prominence_thresholds = peak_auto_prominence_thresholds[sort_idx]
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
            "baseline_prominence_normalized": bool(normalize_amplitude),
            "mad_auto_threshold_enabled": bool(auto_mad),
        }
        if auto_mad:
            metrics.update(
                {
                    "mad_multiplier": float(mad_multiplier),
                    "mean_mad_noise_sigma": (
                        float(np.nanmean(peak_mad_sigmas)) if np.any(np.isfinite(peak_mad_sigmas)) else np.nan
                    ),
                    "mean_auto_prominence_threshold": (
                        float(np.nanmean(peak_auto_prominence_thresholds))
                        if np.any(np.isfinite(peak_auto_prominence_thresholds))
                        else np.nan
                    ),
                    "mad_threshold_files_with_estimate": float(
                        sum(
                            1
                            for stats in mad_threshold_by_file.values()
                            if np.isfinite(float(stats.get("noise_sigma", np.nan)))
                        )
                    ),
                }
            )
        if normalize_amplitude:
            metrics.update(
                {
                    "mean_raw_amplitude": float(np.nanmean(peak_signal_heights)),
                    "median_raw_amplitude": float(np.nanmedian(peak_signal_heights)),
                    "mean_normalized_prominence": (
                        float(np.nanmean(peak_norm_proms)) if np.any(np.isfinite(peak_norm_proms)) else np.nan
                    ),
                    "mean_baseline_prominence_scale": (
                        float(np.nanmean(peak_norm_scales)) if np.any(np.isfinite(peak_norm_scales)) else np.nan
                    ),
                    "baseline_prominence_files_with_scale": float(
                        sum(
                            1
                            for stats in normalization_by_file.values()
                            if np.isfinite(float(stats.get("scale", np.nan)))
                        )
                    ),
                    "baseline_prominence_files_with_peak_scale": float(
                        sum(
                            1
                            for stats in normalization_by_file.values()
                            if str(stats.get("scale_source", "")) == "baseline_peak_prominence"
                        )
                    ),
                    "baseline_prominence_files_with_mad_fallback": float(
                        sum(
                            1
                            for stats in normalization_by_file.values()
                            if str(stats.get("scale_source", "")) == "mad_noise_fallback"
                        )
                    ),
                }
            )

        self.last_signal_events = {
            "peak_times_sec": peak_times,
            "peak_indices": peak_idx,
            "peak_heights": peak_heights,
            "peak_signal_heights": peak_signal_heights,
            "peak_trace_values": peak_trace_values,
            "peak_prominences": peak_proms,
            "peak_normalized_prominences": peak_norm_proms,
            "peak_baseline_prominence_scale": peak_norm_scales,
            "peak_mad_noise_sigma": peak_mad_sigmas,
            "peak_auto_prominence_threshold": peak_auto_prominence_thresholds,
            "peak_widths_sec": peak_widths_sec,
            "peak_auc": peak_auc,
            "file_ids": all_file_ids,
            "derived_metrics": metrics,
            "normalization_by_file": normalization_by_file,
            "mad_threshold_by_file": mad_threshold_by_file,
            "noise_overlay_by_file": noise_overlay_by_file,
            "params": {
                "method": self.combo_signal_method.currentText(),
                "prominence": float(self.spin_peak_prominence.value()),
                "auto_mad_threshold": bool(auto_mad),
                "mad_multiplier": float(mad_multiplier),
                "min_height": float(self.spin_peak_height.value()),
                "min_distance_sec": float(self.spin_peak_distance.value()),
                "smooth_sigma_sec": float(self.spin_peak_smooth.value()),
                "baseline_mode": self.combo_peak_baseline.currentText(),
                "baseline_window_sec": float(self.spin_peak_baseline_window.value()),
                "baseline_prominence_normalized": bool(normalize_amplitude),
                "rate_bin_sec": float(self.spin_peak_rate_bin.value()),
                "auc_half_window_sec": float(self.spin_peak_auc_window.value()),
            },
        }

        msg = f"Detected {peak_times.size} peak(s)."
        if auto_mad:
            auto_vals = [
                float(stats.get("auto_prominence", np.nan))
                for stats in mad_threshold_by_file.values()
                if np.isfinite(float(stats.get("auto_prominence", np.nan)))
            ]
            if auto_vals:
                msg += f" Auto-MAD prominence ~{float(np.nanmean(auto_vals)):.4g}."
            else:
                msg += " Auto-MAD estimate unavailable; manual prominence used."
        if normalize_amplitude and not any(np.isfinite(float(s.get("scale", np.nan))) for s in normalization_by_file.values()):
            msg += " Baseline prominence scale unavailable."
        elif normalize_amplitude and any(str(s.get("scale_source", "")) == "mad_noise_fallback" for s in normalization_by_file.values()):
            msg += " Normalization used MAD fallback for files without baseline peaks."
        self.statusUpdate.emit(msg, 5000)
        self._refresh_signal_overlay()
        self._render_signal_event_plots()
        self._update_signal_metrics_table()
        if not self._autosave_restoring:
            self._project_dirty = True
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
        metrics = self.last_signal_events.get("derived_metrics", {}) or {}
        if bool(metrics.get("baseline_prominence_normalized", False)):
            self.plot_peak_amp.setLabel("bottom", "Prominence-normalized amplitude")
        else:
            self.plot_peak_amp.setLabel("bottom", "Amplitude")

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
        normalized = bool(metrics.get("baseline_prominence_normalized", False))
        amp_label = "mean amplitude (prom-norm)" if normalized else "mean amplitude"
        med_amp_label = "median amplitude (prom-norm)" if normalized else "median amplitude"
        rows = [
            ("number of peaks", metrics.get("number_of_peaks", np.nan)),
            (amp_label, metrics.get("mean_amplitude", np.nan)),
            (med_amp_label, metrics.get("median_amplitude", np.nan)),
            ("amplitude std", metrics.get("amplitude_std", np.nan)),
            ("mean prominence", metrics.get("mean_prominence", np.nan)),
            ("mean width at half prom (s)", metrics.get("mean_width_half_prom_s", np.nan)),
            ("peak frequency (per min)", metrics.get("peak_frequency_per_min", np.nan)),
            ("mean inter-peak interval (s)", metrics.get("mean_inter_peak_interval_s", np.nan)),
            ("mean AUC", metrics.get("mean_auc", np.nan)),
        ]
        if normalized:
            rows.extend(
                [
                    ("mean raw amplitude", metrics.get("mean_raw_amplitude", np.nan)),
                    ("mean normalized prominence", metrics.get("mean_normalized_prominence", np.nan)),
                    ("baseline prominence scale", metrics.get("mean_baseline_prominence_scale", np.nan)),
                    ("files with scale", metrics.get("baseline_prominence_files_with_scale", np.nan)),
                    ("files using baseline peaks", metrics.get("baseline_prominence_files_with_peak_scale", np.nan)),
                    ("files using MAD fallback", metrics.get("baseline_prominence_files_with_mad_fallback", np.nan)),
                ]
            )
        if bool(metrics.get("mad_auto_threshold_enabled", False)):
            rows.extend(
                [
                    ("MAD multiplier", metrics.get("mad_multiplier", np.nan)),
                    ("mean MAD noise sigma", metrics.get("mean_mad_noise_sigma", np.nan)),
                    ("mean auto prominence", metrics.get("mean_auto_prominence_threshold", np.nan)),
                    ("files with MAD estimate", metrics.get("mad_threshold_files_with_estimate", np.nan)),
                ]
            )
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
        peak_signal_heights = np.asarray(
            self.last_signal_events.get("peak_signal_heights", peak_heights),
            float,
        )
        peak_trace_values = np.asarray(
            self.last_signal_events.get("peak_trace_values", peak_signal_heights),
            float,
        )
        peak_proms = np.asarray(self.last_signal_events.get("peak_prominences", np.array([], float)), float)
        peak_norm_proms = np.asarray(
            self.last_signal_events.get("peak_normalized_prominences", np.full_like(peak_proms, np.nan)),
            float,
        )
        peak_norm_scales = np.asarray(
            self.last_signal_events.get("peak_baseline_prominence_scale", np.full_like(peak_heights, np.nan)),
            float,
        )
        peak_mad_sigmas = np.asarray(
            self.last_signal_events.get("peak_mad_noise_sigma", np.full_like(peak_heights, np.nan)),
            float,
        )
        peak_auto_prominence = np.asarray(
            self.last_signal_events.get("peak_auto_prominence_threshold", np.full_like(peak_heights, np.nan)),
            float,
        )
        peak_widths = np.asarray(self.last_signal_events.get("peak_widths_sec", np.array([], float)), float)
        peak_auc = np.asarray(self.last_signal_events.get("peak_auc", np.array([], float)), float)
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
            w.writerow(
                [
                    "peak_time_sec",
                    "height",
                    "prominence",
                    "width_sec",
                    "auc",
                    "trace_value",
                    "signal_height",
                    "normalized_prominence",
                    "baseline_prominence_scale",
                    "mad_noise_sigma",
                    "auto_prominence_threshold",
                    "file_id",
                ]
            )
            for i in range(peak_times.size):
                fid = file_ids[i] if isinstance(file_ids, list) and i < len(file_ids) else ""
                w.writerow(
                    [
                        float(peak_times[i]),
                        float(peak_heights[i]) if i < peak_heights.size else np.nan,
                        float(peak_proms[i]) if i < peak_proms.size else np.nan,
                        float(peak_widths[i]) if i < peak_widths.size else np.nan,
                        float(peak_auc[i]) if i < peak_auc.size else np.nan,
                        float(peak_trace_values[i]) if i < peak_trace_values.size else np.nan,
                        float(peak_signal_heights[i]) if i < peak_signal_heights.size else np.nan,
                        float(peak_norm_proms[i]) if i < peak_norm_proms.size else np.nan,
                        float(peak_norm_scales[i]) if i < peak_norm_scales.size else np.nan,
                        float(peak_mad_sigmas[i]) if i < peak_mad_sigmas.size else np.nan,
                        float(peak_auto_prominence[i]) if i < peak_auto_prominence.size else np.nan,
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
        if not self._autosave_restoring:
            self._project_dirty = True
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
        self._queue_settings_save()
        if not self._processed:
            self.statusUpdate.emit("No processed data loaded.", 5000)
            self._clear_psth_cache()
            self._clear_psth_visuals()
            self._update_status_strip()
            self._sync_temporal_modeling_context()
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
            animal_labels: List[str] = []
            per_file_mats: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
            per_file_labels: Dict[str, List[str]] = {}
            group_per_file_mats: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
            group_per_file_labels: Dict[str, List[str]] = {}
            file_ids_order: List[str] = []
            group_file_ids_order: List[str] = []
            all_dur = []
            all_events: List[float] = []
            event_rows: List[Dict[str, object]] = []
            total_events = 0
            tvec = None
            min_events = self._psth_min_events_per_animal()
            exclude_low_events = bool(group_mode and self._psth_exclude_low_event_animals_enabled())
            excluded_files: Dict[str, Dict[str, object]] = {}

            for proc in self._processed:
                ev, dur = self._get_events_for_proc(proc)
                ev, dur = self._filter_events(ev, dur)
                file_id = os.path.splitext(os.path.basename(proc.path))[0] if proc.path else "import"
                if ev.size == 0:
                    if exclude_low_events:
                        excluded_files[file_id] = {
                            "event_count": 0,
                            "min_events": int(min_events),
                        }
                    continue
                if exclude_low_events and ev.size < min_events:
                    excluded_files[file_id] = {
                        "event_count": int(ev.size),
                        "min_events": int(min_events),
                    }
                if dur is None or len(dur) != ev.size:
                    dur_row = np.full(ev.shape, np.nan, dtype=float)
                else:
                    dur_row = np.asarray(dur, float)
                tvec, mat = _compute_psth_matrix(self._proc_time(proc), proc.output, ev, window, baseline, res_hz, smooth_sigma_s=smooth)
                if mat.size == 0:
                    continue
                # Keep all per-file matrices so Individual view can inspect excluded animals.
                per_file_mats[file_id] = (tvec.copy(), mat.copy())
                per_file_labels[file_id] = [f"Trial {j + 1}" for j in range(mat.shape[0])]
                if file_id not in file_ids_order:
                    file_ids_order.append(file_id)
                if exclude_low_events and mat.shape[0] < min_events:
                    excluded_files[file_id] = {
                        "event_count": int(mat.shape[0]),
                        "min_events": int(min_events),
                    }
                    continue
                all_events.extend(np.asarray(ev, float).tolist())
                for i in range(ev.size):
                    event_rows.append(
                        {
                            "file_id": file_id,
                            "event_time_sec": float(ev[i]),
                            "duration_sec": float(dur_row[i]) if i < dur_row.size else np.nan,
                        }
                    )
                mats.append(mat)
                total_events += mat.shape[0]
                group_per_file_mats[file_id] = (tvec.copy(), mat.copy())
                group_per_file_labels[file_id] = per_file_labels[file_id]
                if file_id not in group_file_ids_order:
                    group_file_ids_order.append(file_id)
                # Build group (animal-level) row
                row = np.nanmean(mat, axis=0)
                if np.any(np.isfinite(row)):
                    animal_rows.append(row)
                    animal_labels.append(file_id)
                if dur is not None and len(dur):
                    all_dur.append(np.asarray(dur, float))

            # Persist per-file and group data for visual tab switching
            self._per_file_mats = per_file_mats
            self._per_file_labels = per_file_labels
            self._all_file_ids = file_ids_order
            self._psth_excluded_files = excluded_files
            if animal_rows:
                self._group_mat = np.vstack(animal_rows)
                self._group_tvec = tvec.copy() if tvec is not None else None
                self._group_labels = animal_labels
            else:
                self._group_mat = None
                self._group_tvec = None
                self._group_labels = []
            self._group_trial_mat, self._group_trial_tvec, self._group_trial_labels = self._stack_psth_trial_rows(
                per_file_mats=group_per_file_mats,
                per_file_labels=group_per_file_labels,
                file_ids_order=group_file_ids_order,
            )
            self._refresh_individual_file_combo()

            self._render_global_metrics()

            if (not mats or tvec is None) and not per_file_mats:
                msg = "No events found for the current alignment."
                if excluded_files:
                    msg = f"No animal/file met the minimum event criterion ({min_events} event(s))."
                self.statusUpdate.emit(msg, 5000)
                self._last_events = np.array([], float)
                self._last_event_rows = []
                self._last_durations = np.array([], float)
                self._update_status_strip()
                self._sync_temporal_modeling_context()
                return

            if tvec is None and per_file_mats:
                first_tvec, _first_mat = next(iter(per_file_mats.values()))
                tvec = first_tvec
            mat_events = np.vstack(mats) if mats else np.empty((0, int(np.asarray(tvec).size) if tvec is not None else 0), dtype=float)
            # Determine what to display based on visual mode tab
            visual_mode = self.tab_visual_mode.currentIndex()  # 0=Individual, 1=Group
            display_level = "trials"
            if visual_mode == 1 and group_mode:
                if self._psth_group_trial_view_enabled():
                    trial_mat = self._group_trial_mat
                    trial_tvec = self._group_trial_tvec
                    trial_labels = list(self._group_trial_labels)
                    if trial_mat is None or trial_tvec is None:
                        self.statusUpdate.emit(f"No animal/file met the minimum event criterion ({min_events} event(s)).", 5000)
                        self._last_events = np.array([], float)
                        self._last_event_rows = []
                        self._last_durations = np.array([], float)
                        self._update_status_strip()
                        self._sync_temporal_modeling_context()
                        return
                    mat_display = trial_mat
                    tvec = trial_tvec
                    display_labels = trial_labels
                    display_level = "trials"
                elif not animal_rows:
                    self.statusUpdate.emit(f"No animal/file met the minimum event criterion ({min_events} event(s)).", 5000)
                    self._last_events = np.array([], float)
                    self._last_event_rows = []
                    self._last_durations = np.array([], float)
                    self._update_status_strip()
                    self._sync_temporal_modeling_context()
                    return
                else:
                    # Group view: each row = animal
                    mat_display = self._group_mat
                    display_labels = self._group_labels
                    display_level = "animals"
            elif visual_mode == 0 and per_file_mats:
                # Individual view: show selected file's trials
                sel_id = self.combo_individual_file.currentText().strip()
                if sel_id and sel_id in per_file_mats:
                    _, mat_display = per_file_mats[sel_id]
                    display_labels = per_file_labels.get(sel_id, [])
                else:
                    # Default to first file
                    first_id = file_ids_order[0] if file_ids_order else None
                    if first_id and first_id in per_file_mats:
                        _, mat_display = per_file_mats[first_id]
                        display_labels = per_file_labels.get(first_id, [])
                    else:
                        mat_display = mat_events
                        display_labels = [f"Trial {j + 1}" for j in range(mat_events.shape[0])]
            elif group_mode:
                mat_display = np.vstack(animal_rows) if animal_rows else mat_events
                display_labels = animal_labels if animal_rows else [f"Trial {j + 1}" for j in range(mat_events.shape[0])]
                display_level = "animals" if animal_rows else "trials"
            else:
                mat_display = mat_events
                display_labels = [f"Trial {j + 1}" for j in range(mat_events.shape[0])]

            self._last_display_labels = list(display_labels)
            self._last_psth_display_level = display_level
            self._render_heatmap(mat_display, tvec, labels=display_labels)
            self._render_avg(mat_display, tvec)
            dur_all = np.concatenate(all_dur) if all_dur else np.array([], float)
            self._render_duration_hist(dur_all)
            self._render_metrics(mat_display, tvec)
            self._last_mat = mat_display
            self._last_tvec = tvec
            self._last_events = np.asarray(all_events, float) if all_events else np.array([], float)
            self._last_durations = dur_all
            self._last_event_rows = event_rows
            excluded_suffix = ""
            if excluded_files:
                excluded_suffix = f" Excluded {len(excluded_files)} animal/file(s) below {min_events} event(s)."
            if visual_mode == 1 and group_mode and display_level == "animals":
                self.statusUpdate.emit(
                    f"Group view: {total_events} event(s) across {mat_display.shape[0]} animal(s).{excluded_suffix}", 5000)
            elif visual_mode == 1 and group_mode:
                self.statusUpdate.emit(
                    f"Group trial view: {mat_display.shape[0]} trial(s) from {len(group_per_file_mats)} animal/file(s).{excluded_suffix}", 5000)
            elif visual_mode == 0:
                sel = self.combo_individual_file.currentText().strip() or "(first)"
                self.statusUpdate.emit(
                    f"Individual view [{sel}]: {mat_display.shape[0]} trial(s) displayed, {total_events} included event(s).{excluded_suffix}", 5000)
            else:
                self.statusUpdate.emit(f"Computed PSTH for {total_events} event(s).{excluded_suffix}", 5000)
            self._update_metric_regions()
            self._update_status_strip()
            self._save_settings()
            self._sync_temporal_modeling_context()
        except Exception as e:
            self.statusUpdate.emit(f"Postprocessing error: {e}", 5000)
            self._update_status_strip()

    @staticmethod
    def _short_heatmap_axis_label(value: object, max_chars: int = 18) -> str:
        text = str(value).strip()
        if len(text) <= max_chars:
            return text
        head = max(5, (max_chars - 3) // 2)
        tail = max(5, max_chars - head - 3)
        return f"{text[:head]}...{text[-tail:]}"

    def _build_heatmap_y_ticks(
        self,
        labels: Optional[List[str]],
        n_rows: int,
        show_names: bool,
    ) -> List[Tuple[float, str]]:
        if n_rows <= 0:
            return []
        dense_limit = 12
        if n_rows <= dense_limit:
            indices = list(range(n_rows))
        else:
            target_ticks = min(8, n_rows)
            indices = sorted({int(round(v)) for v in np.linspace(0, n_rows - 1, target_ticks)})
            if indices[0] != 0:
                indices.insert(0, 0)
            if indices[-1] != n_rows - 1:
                indices.append(n_rows - 1)
        ticks: List[Tuple[float, str]] = []
        for idx in indices:
            if show_names and labels and idx < len(labels):
                label = self._short_heatmap_axis_label(labels[idx])
            else:
                label = str(idx + 1)
            ticks.append((float(idx) + 0.5, label))
        return ticks

    def _render_heatmap(self, mat: np.ndarray, tvec: np.ndarray, labels: Optional[List[str]] = None) -> None:
        if mat.size == 0:
            self._suppress_heatmap_level_store = True
            try:
                self.img.setImage(np.zeros((1, 1)), autoLevels=False)
                self.img.setLevels([0.0, 1.0])
                self.img.setRect(QtCore.QRectF(0.0, 0.0, 1.0, 1.0))
                if hasattr(self, "heat_lut") and getattr(self.heat_lut, "item", None) is not None:
                    try:
                        self.heat_lut.item.setLevels(0.0, 1.0)
                    except TypeError:
                        self.heat_lut.item.setLevels((0.0, 1.0))
            finally:
                self._suppress_heatmap_level_store = False
            self.heat_zero_line.setVisible(False)
            self.plot_heat.setXRange(0.0, 1.0, padding=0)
            self.plot_heat.setYRange(0.0, 1.0, padding=0)
            self.plot_heat.setLabel("left", "Trials")
            try:
                self.plot_heat.getAxis("left").setTicks([[]])
            except Exception:
                pass
            return

        # ImageItem maps axis-0 -> x and axis-1 -> y; transpose so time is x, trials are y.
        img = np.asarray(mat, float).T
        cmap_name = str(self._style.get("heatmap_cmap", "viridis"))
        try:
            cmap = pg.colormap.get(cmap_name)
            lut = cmap.getLookupTable()
            self.img.setLookupTable(lut)
            if hasattr(self, "heat_lut") and getattr(self.heat_lut, "item", None) is not None:
                self.heat_lut.item.gradient.setColorMap(cmap)
        except Exception:
            pass
        finite = img[np.isfinite(img)]
        if finite.size:
            lo = float(np.nanmin(finite))
            hi = float(np.nanmax(finite))
        else:
            lo, hi = 0.0, 1.0
        manual_levels = bool(self._style.get("heatmap_levels_manual", False))
        if not manual_levels:
            self._style["heatmap_min"] = None
            self._style["heatmap_max"] = None
        hmin = self._style.get("heatmap_min", None)
        hmax = self._style.get("heatmap_max", None)
        if manual_levels and hmin is not None and hmax is not None:
            try:
                lo = float(hmin)
                hi = float(hmax)
            except Exception:
                pass
        if (not np.isfinite(lo)) or (not np.isfinite(hi)) or hi <= lo:
            center = lo if np.isfinite(lo) else 0.0
            lo, hi = center - 0.5, center + 0.5
        self._suppress_heatmap_level_store = True
        try:
            self.img.setImage(img, autoLevels=False)
            self.img.setLevels([lo, hi])
            if hasattr(self, "heat_lut") and getattr(self.heat_lut, "item", None) is not None:
                try:
                    self.heat_lut.item.setLevels(lo, hi)
                except TypeError:
                    self.heat_lut.item.setLevels((lo, hi))
        finally:
            self._suppress_heatmap_level_store = False

        # Map image to time axis using a rect (avoids scale() incompatibilities)
        x0 = float(tvec[0]) if tvec.size else 0.0
        x1 = float(tvec[-1]) if tvec.size else 1.0
        if x1 == x0:
            x1 = x0 + 1.0
        n_rows = img.shape[1]
        self.img.setRect(QtCore.QRectF(x0, 0, x1 - x0, n_rows))
        self.plot_heat.setXRange(x0, x1, padding=0)
        self.plot_heat.setYRange(0, float(n_rows), padding=0)
        self.heat_zero_line.setPos(0.0)
        self.heat_zero_line.setVisible(bool(x0 <= 0.0 <= x1))

        # Keep row labels readable: show all only for small heatmaps, otherwise
        # use a sparse trial/animal ruler rather than overprinting every row.
        y_axis = self.plot_heat.getAxis("left")
        display_level = str(getattr(self, "_last_psth_display_level", "trials") or "trials")
        is_animal_level = display_level == "animals"
        has_labels = bool(labels and len(labels) == n_rows)
        show_names = bool(has_labels and (is_animal_level or n_rows <= 12))
        ticks = self._build_heatmap_y_ticks(labels if has_labels else None, n_rows, show_names)
        y_axis.setTicks([ticks])
        try:
            y_axis.setStyle(tickTextOffset=6, autoExpandTextSpace=False)
            y_axis.setWidth(58 if n_rows > 12 else 76)
        except Exception:
            pass
        label_kind = "Animals" if is_animal_level else "Trials"
        self.plot_heat.setLabel("left", f"{label_kind} (n={n_rows})")
        if n_rows > 12:
            self.plot_heat.setToolTip(
                f"Heatmap rows: {n_rows} {label_kind.lower()}. "
                "The y-axis shows sparse row numbers to keep the display readable."
            )
        else:
            self.plot_heat.setToolTip("")

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
        display_level = str(getattr(self, "_last_psth_display_level", "trials") or "trials")
        if display_level == "animals":
            self.plot_avg.setTitle("Average across animals \u00b1 SEM")
        else:
            self.plot_avg.setTitle("Average across trials \u00b1 SEM")

    @staticmethod
    def _finite_mean_sem(values: np.ndarray) -> Tuple[float, float, int]:
        vals = np.asarray(values, float)
        vals = vals[np.isfinite(vals)]
        n = int(vals.size)
        if n == 0:
            return 0.0, 0.0, 0
        mean = float(np.nanmean(vals))
        if n < 2:
            return mean, 0.0, n
        sem = float(np.nanstd(vals, ddof=1) / np.sqrt(float(n)))
        if not np.isfinite(sem):
            sem = 0.0
        return mean, sem, n

    @staticmethod
    def _jittered_x(center: float, n: int, half_width: float = 0.16) -> np.ndarray:
        if n <= 0:
            return np.array([], float)
        if n == 1:
            return np.array([float(center)], float)
        return float(center) + np.linspace(-abs(float(half_width)), abs(float(half_width)), n)

    @staticmethod
    def _set_error_bar(item: pg.ErrorBarItem, x: float, y: float, err: float) -> None:
        yy = float(y) if np.isfinite(y) else 0.0
        ee = float(err) if np.isfinite(err) and err > 0 else 0.0
        item.setData(
            x=np.array([float(x)], float),
            y=np.array([yy], float),
            top=np.array([ee], float),
            bottom=np.array([ee], float),
        )

    def _render_metrics(self, mat: np.ndarray, tvec: np.ndarray) -> None:
        if mat.size == 0 or not self.cb_metrics.isChecked():
            self.metrics_bar_pre.setOpts(height=[0])
            self.metrics_bar_post.setOpts(height=[0])
            self.metrics_pairs_curve.setData([], [])
            self.metrics_scatter_pre.setData([], [])
            self.metrics_scatter_post.setData([], [])
            self._set_error_bar(self.metrics_err_pre, 0.0, 0.0, 0.0)
            self._set_error_bar(self.metrics_err_post, 1.0, 0.0, 0.0)
            self.metrics_p_text.setVisible(False)
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
            self.metrics_pairs_curve.setData([], [])
            self.metrics_scatter_pre.setData([], [])
            self.metrics_scatter_post.setData([], [])
            self._set_error_bar(self.metrics_err_pre, 0.0, 0.0, 0.0)
            self._set_error_bar(self.metrics_err_post, 1.0, 0.0, 0.0)
            self.metrics_p_text.setVisible(False)
            self._last_metrics = None
            return

        def _metric_vals(win: np.ndarray, duration: float) -> np.ndarray:
            if win.size == 0:
                return np.array([], float)
            w = np.asarray(win, float)
            valid = np.isfinite(w)
            counts = np.sum(valid, axis=1)
            sums = np.nansum(w, axis=1)
            vals = np.divide(sums, counts, out=np.full(w.shape[0], np.nan, dtype=float), where=counts > 0)
            if metric.startswith("AUC"):
                vals = vals * float(abs(duration))
            return np.asarray(vals, float)

        pre_vals_all = _metric_vals(pre, pre1 - pre0)
        post_vals_all = _metric_vals(post, post1 - post0)
        pre_vals_finite = pre_vals_all[np.isfinite(pre_vals_all)]
        post_vals_finite = post_vals_all[np.isfinite(post_vals_all)]
        pre_mean, pre_sem, pre_n = self._finite_mean_sem(pre_vals_finite)
        post_mean, post_sem, post_n = self._finite_mean_sem(post_vals_finite)
        self.metrics_bar_pre.setOpts(height=[pre_mean])
        self.metrics_bar_post.setOpts(height=[post_mean])
        group_mode = self.tab_sources.currentIndex() == 1

        pair_mask = np.isfinite(pre_vals_all) & np.isfinite(post_vals_all)
        pre_pair = pre_vals_all[pair_mask]
        post_pair = post_vals_all[pair_mask]
        p_value = np.nan
        n_pair = int(min(pre_pair.size, post_pair.size))
        if pre_pair.size and post_pair.size:
            pre_pair = pre_pair[:n_pair]
            post_pair = post_pair[:n_pair]
            if n_pair >= 2:
                diffs = post_pair - pre_pair
                diffs = diffs[np.isfinite(diffs)]
                if diffs.size >= 2:
                    if float(np.nanstd(diffs, ddof=1)) == 0.0:
                        p_value = 1.0 if float(np.nanmean(diffs)) == 0.0 else 0.0
                    else:
                        try:
                            from scipy import stats
                            result = stats.ttest_rel(pre_pair, post_pair, nan_policy="omit")
                            p_value = float(result.pvalue)
                        except Exception:
                            p_value = np.nan
            # Build segmented polyline: (0, pre_i) -> (1, post_i), NaN separator.
            x_line = np.empty(n_pair * 3, dtype=float)
            y_line = np.empty(n_pair * 3, dtype=float)
            x_line[0::3] = 0.0
            x_line[1::3] = 1.0
            x_line[2::3] = np.nan
            y_line[0::3] = pre_pair
            y_line[1::3] = post_pair
            y_line[2::3] = np.nan
            self.metrics_pairs_curve.setData(x_line, y_line, connect="finite", skipFiniteCheck=True)
            if group_mode:
                x_pre = self._jittered_x(0.0, n_pair, half_width=0.16)
                x_post = self._jittered_x(1.0, n_pair, half_width=0.16)
            else:
                x_pre = np.zeros(n_pair, dtype=float)
                x_post = np.ones(n_pair, dtype=float)
            self.metrics_scatter_pre.setData(x_pre, pre_pair)
            self.metrics_scatter_post.setData(x_post, post_pair)
        else:
            self.metrics_pairs_curve.setData([], [])
            self.metrics_scatter_pre.setData([], [])
            self.metrics_scatter_post.setData([], [])

        if group_mode:
            self._set_error_bar(self.metrics_err_pre, 0.0, pre_mean, pre_sem)
            self._set_error_bar(self.metrics_err_post, 1.0, post_mean, post_sem)
        else:
            self._set_error_bar(self.metrics_err_pre, 0.0, 0.0, 0.0)
            self._set_error_bar(self.metrics_err_post, 1.0, 0.0, 0.0)

        finite_all = np.concatenate(
            [
                pre_vals_finite if pre_vals_finite.size else np.array([], float),
                post_vals_finite if post_vals_finite.size else np.array([], float),
                np.array(
                    [
                        pre_mean,
                        post_mean,
                        pre_mean - pre_sem,
                        pre_mean + pre_sem,
                        post_mean - post_sem,
                        post_mean + post_sem,
                        0.0,
                    ],
                    float,
                ),
            ]
        )
        finite_all = finite_all[np.isfinite(finite_all)]
        if finite_all.size:
            ymin = float(np.nanmin(finite_all))
            ymax = float(np.nanmax(finite_all))
        else:
            ymin, ymax = 0.0, 1.0
        if ymin == ymax:
            ymax = ymin + 1.0
        self.plot_metrics.setYRange(ymin, ymax, padding=0.2)
        span = float(ymax - ymin) if np.isfinite(ymax - ymin) and ymax != ymin else 1.0
        if n_pair >= 2:
            if np.isfinite(p_value):
                p_label = "paired p < 1e-4" if p_value < 1e-4 else f"paired p = {p_value:.4g}"
            else:
                p_label = "paired p = n/a"
            self.metrics_p_text.setText(p_label)
            self.metrics_p_text.setPos(0.5, ymax + 0.14 * span)
            self.metrics_p_text.setVisible(True)
        else:
            self.metrics_p_text.setVisible(False)
        self._last_metrics = {
            "pre": pre_mean,
            "post": post_mean,
            "pre_sem": pre_sem,
            "post_sem": post_sem,
            "pre_n": float(pre_n),
            "post_n": float(post_n),
            "paired_n": float(n_pair),
            "paired_p": float(p_value) if np.isfinite(p_value) else np.nan,
            "metric": metric,
        }

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
            self.global_scatter_amp.setData([], [])
            self.global_scatter_freq.setData([], [])
            self._set_error_bar(self.global_err_amp, 0.0, 0.0, 0.0)
            self._set_error_bar(self.global_err_freq, 1.0, 0.0, 0.0)
            return

        if not (self.cb_global_amp.isChecked() or self.cb_global_freq.isChecked()):
            self._last_global_metrics = None
            self.lbl_global_metrics.setText("Global metrics: -")
            self.global_bar_amp.setOpts(height=[0])
            self.global_bar_freq.setOpts(height=[0])
            self.global_scatter_amp.setData([], [])
            self.global_scatter_freq.setData([], [])
            self._set_error_bar(self.global_err_amp, 0.0, 0.0, 0.0)
            self._set_error_bar(self.global_err_freq, 1.0, 0.0, 0.0)
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
            res = self._compute_global_metrics_for_trace(self._proc_time(proc), proc.output, start_s, end_s)
            if not res:
                continue
            amps.append(res["amp"])
            freqs.append(res["freq"])
            peaks.append(res["peaks"])
            durations.append(res["duration"])
            thrs.append(res["thr"])

        amp_vals = np.asarray(amps, float)
        freq_vals = np.asarray(freqs, float)
        amp_vals = amp_vals[np.isfinite(amp_vals)]
        freq_vals = freq_vals[np.isfinite(freq_vals)]
        if amp_vals.size == 0 and freq_vals.size == 0:
            self._last_global_metrics = None
            self.lbl_global_metrics.setText("Global metrics: -")
            self.global_bar_amp.setOpts(height=[0])
            self.global_bar_freq.setOpts(height=[0])
            self.global_scatter_amp.setData([], [])
            self.global_scatter_freq.setData([], [])
            self._set_error_bar(self.global_err_amp, 0.0, 0.0, 0.0)
            self._set_error_bar(self.global_err_freq, 1.0, 0.0, 0.0)
            return

        avg_amp, sem_amp, n_amp = self._finite_mean_sem(amp_vals)
        avg_freq, sem_freq, n_freq = self._finite_mean_sem(freq_vals)
        total_peaks = float(np.nansum(peaks)) if peaks else 0.0
        avg_thr = float(np.nanmean(thrs)) if thrs else 0.0
        avg_dur = float(np.nanmean(durations)) if durations else 0.0
        group_mode = self.tab_sources.currentIndex() == 1

        self._last_global_metrics = {
            "amp": avg_amp,
            "amp_sem": sem_amp,
            "amp_n": float(n_amp),
            "freq": avg_freq,
            "freq_sem": sem_freq,
            "freq_n": float(n_freq),
            "peaks": total_peaks,
            "thr": avg_thr,
            "duration": avg_dur,
            "start": start_s,
            "end": end_s,
        }

        parts = []
        if self.cb_global_amp.isChecked():
            if group_mode:
                parts.append(f"amp={avg_amp:.4g}+-{sem_amp:.3g}")
            else:
                parts.append(f"amp={avg_amp:.4g}")
        if self.cb_global_freq.isChecked():
            if group_mode:
                parts.append(f"freq={avg_freq:.4g}+-{sem_freq:.3g} Hz")
            else:
                parts.append(f"freq={avg_freq:.4g} Hz")
        parts.append(f"peaks={int(total_peaks)}")
        self.lbl_global_metrics.setText("Global metrics: " + " | ".join(parts))

        self.global_bar_amp.setOpts(height=[avg_amp if self.cb_global_amp.isChecked() else 0.0])
        self.global_bar_freq.setOpts(height=[avg_freq if self.cb_global_freq.isChecked() else 0.0])
        if group_mode:
            if self.cb_global_amp.isChecked() and amp_vals.size:
                self.global_scatter_amp.setData(self._jittered_x(0.0, int(amp_vals.size), half_width=0.16), amp_vals)
                self._set_error_bar(self.global_err_amp, 0.0, avg_amp, sem_amp)
            else:
                self.global_scatter_amp.setData([], [])
                self._set_error_bar(self.global_err_amp, 0.0, 0.0, 0.0)
            if self.cb_global_freq.isChecked() and freq_vals.size:
                self.global_scatter_freq.setData(self._jittered_x(1.0, int(freq_vals.size), half_width=0.16), freq_vals)
                self._set_error_bar(self.global_err_freq, 1.0, avg_freq, sem_freq)
            else:
                self.global_scatter_freq.setData([], [])
                self._set_error_bar(self.global_err_freq, 1.0, 0.0, 0.0)
        else:
            self.global_scatter_amp.setData([], [])
            self.global_scatter_freq.setData([], [])
            self._set_error_bar(self.global_err_amp, 0.0, 0.0, 0.0)
            self._set_error_bar(self.global_err_freq, 1.0, 0.0, 0.0)
        y_candidates: List[float] = [0.0]
        if self.cb_global_amp.isChecked():
            y_candidates.extend([avg_amp, avg_amp - sem_amp, avg_amp + sem_amp])
            if group_mode and amp_vals.size:
                y_candidates.extend(np.asarray(amp_vals, float).tolist())
        if self.cb_global_freq.isChecked():
            y_candidates.extend([avg_freq, avg_freq - sem_freq, avg_freq + sem_freq])
            if group_mode and freq_vals.size:
                y_candidates.extend(np.asarray(freq_vals, float).tolist())
        y_arr = np.asarray(y_candidates, float)
        y_arr = y_arr[np.isfinite(y_arr)]
        if y_arr.size:
            ymin = float(np.nanmin(y_arr))
            ymax = float(np.nanmax(y_arr))
        else:
            ymin, ymax = 0.0, 1.0
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

    def _style_color_tuple(self, key: str, fallback: Tuple[int, ...]) -> Tuple[int, ...]:
        raw = self._style.get(key, fallback)
        if isinstance(raw, np.ndarray):
            vals = raw.tolist()
        elif isinstance(raw, (list, tuple)):
            vals = list(raw)
        else:
            vals = list(fallback)
        out: List[int] = []
        for i, default in enumerate(list(fallback)):
            try:
                v = int(vals[i]) if i < len(vals) else int(default)
            except Exception:
                v = int(default)
            out.append(max(0, min(255, v)))
        return tuple(out)

    def _apply_plot_style(self) -> None:
        self.curve_trace.setPen(pg.mkPen(self._style_color_tuple("trace", (90, 190, 255)), width=1.1))
        self.curve_behavior.setPen(pg.mkPen(self._style_color_tuple("behavior", (220, 180, 80)), width=1.0))
        self.curve_avg.setPen(pg.mkPen(self._style_color_tuple("avg", (90, 190, 255)), width=1.3))
        sem_edge = self._style_color_tuple("sem_edge", (152, 201, 143))
        sem_fill = self._style_color_tuple("sem_fill", (188, 230, 178, 96))
        self.curve_sem_hi.setPen(pg.mkPen(sem_edge, width=1.0))
        self.curve_sem_lo.setPen(pg.mkPen(sem_edge, width=1.0))
        if hasattr(self, "sem_band"):
            self.sem_band.setBrush(pg.mkBrush(*sem_fill))
        bg = self._style_color_tuple("plot_bg", (36, 42, 52))
        grid_enabled = bool(self._style.get("grid_enabled", True))
        try:
            grid_alpha = float(self._style.get("grid_alpha", 0.25))
        except Exception:
            grid_alpha = 0.25
        grid_alpha = max(0.0, min(1.0, grid_alpha))
        light_plot = (sum(bg[:3]) / 3.0) >= 180.0
        axis_color = (45, 55, 72) if light_plot else (182, 194, 212)
        text_color = (31, 42, 55) if light_plot else (200, 211, 226)
        grid_draw_alpha = max(grid_alpha, 0.22) if light_plot else grid_alpha
        for pw in (
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
            self.plot_spatial_occupancy,
            self.plot_spatial_activity,
            self.plot_spatial_velocity,
        ):
            try:
                pw.setBackground(QtGui.QColor(*bg[:3]))
            except Exception:
                pass
            try:
                pw.showGrid(x=grid_enabled, y=grid_enabled, alpha=grid_draw_alpha)
            except Exception:
                pass
            try:
                plot_item = pw.getPlotItem()
                axis_pen = pg.mkPen(axis_color)
                text_pen = pg.mkPen(text_color)
                for axis_name in ("left", "right", "bottom", "top"):
                    axis = plot_item.getAxis(axis_name)
                    if axis is None:
                        continue
                    axis.setPen(axis_pen)
                    axis.setTextPen(text_pen)
                title_label = getattr(plot_item, "titleLabel", None)
                title_item = getattr(title_label, "item", None)
                if title_item is not None:
                    title_item.setDefaultTextColor(QtGui.QColor(*text_color))
            except Exception:
                pass
        cmap_name = str(self._style.get("heatmap_cmap", "viridis"))
        try:
            cmap = pg.colormap.get(cmap_name)
            if hasattr(self, "heat_lut") and getattr(self.heat_lut, "item", None) is not None:
                self.heat_lut.item.gradient.setColorMap(cmap)
        except Exception:
            pass

    def _on_heatmap_levels_changed(self) -> None:
        if self._is_restoring_settings or self._suppress_heatmap_level_store:
            return
        if not hasattr(self, "heat_lut") or getattr(self.heat_lut, "item", None) is None:
            return
        try:
            levels = self.heat_lut.item.getLevels()
        except Exception:
            return
        if not isinstance(levels, (list, tuple)) or len(levels) < 2:
            return
        lo = float(levels[0])
        hi = float(levels[1])
        if np.isfinite(lo) and np.isfinite(hi) and hi > lo:
            self._style["heatmap_min"] = lo
            self._style["heatmap_max"] = hi
            self._style["heatmap_levels_manual"] = True
            self._queue_settings_save()

    def _open_style_dialog(self) -> None:
        dlg = StyleDialog(self._style, self)
        if dlg.exec() != QtWidgets.QDialog.DialogCode.Accepted:
            return
        self._style = dlg.get_style()
        self._apply_plot_style()
        self._record_history_change()
        if self._last_mat is not None and self._last_tvec is not None:
            self._render_heatmap(self._last_mat, self._last_tvec, labels=self._last_display_labels)
        else:
            self._render_heatmap(np.zeros((0, 0), dtype=float), np.array([], dtype=float), labels=[])
        self._render_spatial_heatmap(
            self._last_spatial_occupancy_map,
            self._last_spatial_activity_map,
            self._last_spatial_velocity_map,
            self._last_spatial_extent,
            self.combo_spatial_x.currentText().strip(),
            self.combo_spatial_y.currentText().strip(),
            "Spatial occupancy",
            "Spatial activity",
            "Spatial velocity",
        )
        self._save_settings()

    def _h5_text(self, value: object, default: str = "") -> str:
        if value is None:
            return default
        if isinstance(value, bytes):
            try:
                return value.decode("utf-8", errors="ignore")
            except Exception:
                return default
        return str(value)

    def _write_h5_json(self, group: h5py.Group, name: str, payload: Dict[str, object]) -> None:
        text = json.dumps(payload or {})
        dtype = h5py.string_dtype(encoding="utf-8")
        if name in group:
            del group[name]
        group.create_dataset(name, data=text, dtype=dtype)

    def _write_h5_json_any(self, group: h5py.Group, name: str, payload: object) -> None:
        text = json.dumps(payload)
        dtype = h5py.string_dtype(encoding="utf-8")
        if name in group:
            del group[name]
        group.create_dataset(name, data=text, dtype=dtype)

    def _read_h5_json(self, group: Optional[h5py.Group], name: str) -> Dict[str, object]:
        if group is None or name not in group:
            return {}
        try:
            raw = group[name][()]
        except Exception:
            return {}
        if isinstance(raw, np.ndarray):
            try:
                raw = raw.item()
            except Exception:
                raw = raw.tolist()
        if isinstance(raw, bytes):
            try:
                raw = raw.decode("utf-8", errors="ignore")
            except Exception:
                raw = ""
        text = str(raw or "")
        if not text:
            return {}
        try:
            data = json.loads(text)
        except Exception:
            return {}
        return data if isinstance(data, dict) else {}

    def _read_h5_json_any(self, group: Optional[h5py.Group], name: str, default: object) -> object:
        if group is None or name not in group:
            return default
        try:
            raw = group[name][()]
        except Exception:
            return default
        if isinstance(raw, np.ndarray):
            try:
                raw = raw.item()
            except Exception:
                raw = raw.tolist()
        if isinstance(raw, bytes):
            try:
                raw = raw.decode("utf-8", errors="ignore")
            except Exception:
                raw = ""
        text = str(raw or "")
        if not text:
            return default
        try:
            return json.loads(text)
        except Exception:
            return default

    def _write_h5_str_list(self, group: h5py.Group, name: str, values: List[str]) -> None:
        dtype = h5py.string_dtype(encoding="utf-8")
        arr = np.asarray([str(v) for v in (values or [])], dtype=dtype)
        if name in group:
            del group[name]
        group.create_dataset(name, data=arr, dtype=dtype)

    def _read_h5_str_list(self, group: Optional[h5py.Group], name: str) -> List[str]:
        if group is None or name not in group:
            return []
        try:
            raw = group[name][()]
        except Exception:
            return []
        if isinstance(raw, bytes):
            return [raw.decode("utf-8", errors="ignore")]
        if isinstance(raw, str):
            return [raw]
        if isinstance(raw, np.ndarray):
            out: List[str] = []
            for item in raw.tolist():
                if isinstance(item, bytes):
                    out.append(item.decode("utf-8", errors="ignore"))
                else:
                    out.append(str(item))
            return out
        return []

    def _write_h5_numeric(self, group: h5py.Group, name: str, values: np.ndarray) -> None:
        arr = np.asarray(values, float)
        if name in group:
            del group[name]
        kwargs: Dict[str, object] = {}
        if arr.size > 0:
            kwargs["compression"] = "gzip"
        group.create_dataset(name, data=arr, **kwargs)

    def _read_h5_numeric(self, group: h5py.Group, name: str) -> Optional[np.ndarray]:
        if name not in group:
            return None
        try:
            return np.asarray(group[name][()], float)
        except Exception:
            return None

    def _save_signal_events_h5(self, parent: h5py.Group) -> None:
        if not isinstance(self.last_signal_events, dict) or not self.last_signal_events:
            return
        group = parent.create_group("signal_events")
        for key in (
            "peak_times_sec",
            "peak_indices",
            "peak_heights",
            "peak_signal_heights",
            "peak_trace_values",
            "peak_prominences",
            "peak_normalized_prominences",
            "peak_baseline_prominence_scale",
            "peak_mad_noise_sigma",
            "peak_auto_prominence_threshold",
            "peak_widths_sec",
            "peak_auc",
        ):
            self._write_h5_numeric(group, key, np.asarray(self.last_signal_events.get(key, np.array([], float)), float))
        self._write_h5_str_list(group, "file_ids", [str(v) for v in self.last_signal_events.get("file_ids", []) or []])
        self._write_h5_json(group, "derived_metrics_json", dict(self.last_signal_events.get("derived_metrics", {}) or {}))
        self._write_h5_json(group, "params_json", dict(self.last_signal_events.get("params", {}) or {}))
        self._write_h5_json_any(
            group,
            "normalization_by_file_json",
            dict(self.last_signal_events.get("normalization_by_file", {}) or {}),
        )
        self._write_h5_json_any(
            group,
            "mad_threshold_by_file_json",
            dict(self.last_signal_events.get("mad_threshold_by_file", {}) or {}),
        )

    def _load_signal_events_h5(self, parent: Optional[h5py.Group]) -> Optional[Dict[str, object]]:
        if parent is None:
            return None
        group = parent.get("signal_events")
        if not isinstance(group, h5py.Group):
            return None

        def _num(name: str) -> np.ndarray:
            arr = self._read_h5_numeric(group, name)
            if arr is None:
                return np.array([], float)
            return np.asarray(arr, float)

        out: Dict[str, object] = {
            "peak_times_sec": _num("peak_times_sec"),
            "peak_indices": _num("peak_indices"),
            "peak_heights": _num("peak_heights"),
            "peak_signal_heights": _num("peak_signal_heights"),
            "peak_trace_values": _num("peak_trace_values"),
            "peak_prominences": _num("peak_prominences"),
            "peak_normalized_prominences": _num("peak_normalized_prominences"),
            "peak_baseline_prominence_scale": _num("peak_baseline_prominence_scale"),
            "peak_mad_noise_sigma": _num("peak_mad_noise_sigma"),
            "peak_auto_prominence_threshold": _num("peak_auto_prominence_threshold"),
            "peak_widths_sec": _num("peak_widths_sec"),
            "peak_auc": _num("peak_auc"),
            "file_ids": self._read_h5_str_list(group, "file_ids"),
            "derived_metrics": self._read_h5_json(group, "derived_metrics_json"),
            "params": self._read_h5_json(group, "params_json"),
            "normalization_by_file": self._read_h5_json_any(group, "normalization_by_file_json", {}),
            "mad_threshold_by_file": self._read_h5_json_any(group, "mad_threshold_by_file_json", {}),
        }
        return out

    def _save_behavior_analysis_h5(self, parent: h5py.Group) -> None:
        if not isinstance(self.last_behavior_analysis, dict) or not self.last_behavior_analysis:
            return
        group = parent.create_group("behavior_analysis")
        group.attrs["behavior_name"] = str(self.last_behavior_analysis.get("behavior_name", "") or "")
        self._write_h5_json_any(group, "per_file_metrics_json", self.last_behavior_analysis.get("per_file_metrics", []) or [])
        self._write_h5_json(group, "group_metrics_json", dict(self.last_behavior_analysis.get("group_metrics", {}) or {}))
        self._write_h5_json(group, "params_json", dict(self.last_behavior_analysis.get("params", {}) or {}))

        events_group = group.create_group("per_file_events")
        for idx, row in enumerate(self.last_behavior_analysis.get("per_file_events", []) or []):
            if not isinstance(row, dict):
                continue
            entry = events_group.create_group(f"item_{idx:04d}")
            entry.attrs["file_id"] = str(row.get("file_id", "") or "")
            try:
                entry.attrs["row_index"] = int(row.get("row_index", idx + 1))
            except Exception:
                entry.attrs["row_index"] = int(idx + 1)
            self._write_h5_numeric(entry, "start_sec", np.asarray(row.get("start_sec", np.array([], float)), float))
            self._write_h5_numeric(entry, "end_sec", np.asarray(row.get("end_sec", np.array([], float)), float))
            self._write_h5_numeric(entry, "duration_sec", np.asarray(row.get("duration_sec", np.array([], float)), float))

    def _load_behavior_analysis_h5(self, parent: Optional[h5py.Group]) -> Optional[Dict[str, object]]:
        if parent is None:
            return None
        group = parent.get("behavior_analysis")
        if not isinstance(group, h5py.Group):
            return None

        per_file_metrics_raw = self._read_h5_json_any(group, "per_file_metrics_json", [])
        per_file_metrics = per_file_metrics_raw if isinstance(per_file_metrics_raw, list) else []

        per_file_events: List[Dict[str, object]] = []
        events_group = group.get("per_file_events")
        if isinstance(events_group, h5py.Group):
            for key in sorted(events_group.keys()):
                entry = events_group.get(key)
                if not isinstance(entry, h5py.Group):
                    continue
                try:
                    row_index = int(entry.attrs.get("row_index", 0))
                except Exception:
                    row_index = 0
                per_file_events.append(
                    {
                        "file_id": self._h5_text(entry.attrs.get("file_id", ""), ""),
                        "row_index": row_index,
                        "start_sec": np.asarray(
                            self._read_h5_numeric(entry, "start_sec") if "start_sec" in entry else np.array([], float),
                            float,
                        ),
                        "end_sec": np.asarray(
                            self._read_h5_numeric(entry, "end_sec") if "end_sec" in entry else np.array([], float),
                            float,
                        ),
                        "duration_sec": np.asarray(
                            self._read_h5_numeric(entry, "duration_sec") if "duration_sec" in entry else np.array([], float),
                            float,
                        ),
                    }
                )

        return {
            "behavior_name": self._h5_text(group.attrs.get("behavior_name", ""), ""),
            "per_file_metrics": per_file_metrics,
            "per_file_events": per_file_events,
            "group_metrics": self._read_h5_json(group, "group_metrics_json"),
            "params": self._read_h5_json(group, "params_json"),
        }

    def _clear_cached_analysis_outputs(self) -> None:
        self.last_signal_events = None
        self.last_behavior_analysis = None
        self.tbl_signal_metrics.setRowCount(0)
        self.tbl_behavior_metrics.setRowCount(0)
        self.lbl_behavior_summary.setText("Group metrics: -")
        for pw in (self.plot_peak_amp, self.plot_peak_ibi, self.plot_peak_rate):
            pw.clear()
        for pw in (
            self.plot_behavior_raster,
            self.plot_behavior_rate,
            self.plot_behavior_duration,
            self.plot_behavior_starts,
        ):
            pw.clear()
        self._refresh_signal_overlay()

    def _restore_cached_analysis_outputs(self, payload: Dict[str, object]) -> None:
        self._clear_cached_analysis_outputs()

        signal_events = payload.get("signal_events")
        if isinstance(signal_events, dict) and signal_events:
            self.last_signal_events = signal_events
            self._refresh_signal_overlay()
            self._render_signal_event_plots()
            self._update_signal_metrics_table()

        behavior_analysis = payload.get("behavior_analysis")
        if isinstance(behavior_analysis, dict) and behavior_analysis:
            self.last_behavior_analysis = behavior_analysis
            self._render_behavior_analysis_outputs()

        temporal_state = payload.get("temporal_modeling")
        if isinstance(temporal_state, dict) and temporal_state and hasattr(self, "section_temporal"):
            try:
                self.section_temporal.deserialize_state(temporal_state)
            except Exception:
                _LOG.debug("Could not restore temporal modeling state", exc_info=True)

    def _save_project_h5(self, path: str) -> None:
        with h5py.File(path, "w") as f:
            f.attrs["project_type"] = "pyber_postprocessing_project"
            f.attrs["project_version"] = 1
            f.attrs["created_utc"] = datetime.now(timezone.utc).isoformat()

            ui_group = f.create_group("ui")
            self._write_h5_json(ui_group, "settings_json", self._collect_settings())
            ui_group.attrs["tab_sources_index"] = int(self.tab_sources.currentIndex())

            meta_group = f.create_group("meta")
            processed_paths = [str(getattr(proc, "path", "") or "").strip() for proc in self._processed]
            processed_paths = [p for p in processed_paths if p]
            behavior_paths: List[str] = []
            for info in self._behavior_sources.values():
                src = str((info or {}).get("source_path", "") or "").strip()
                if src:
                    behavior_paths.append(src)
            self._write_h5_json(
                meta_group,
                "recent_paths_json",
                {
                    "processed_paths": processed_paths,
                    "behavior_paths": behavior_paths,
                },
            )

            processed_group = f.create_group("processed")
            processed_group.attrs["count"] = int(len(self._processed))
            for idx, proc in enumerate(self._processed):
                entry = processed_group.create_group(f"item_{idx:04d}")
                entry.attrs["path"] = str(getattr(proc, "path", "") or "")
                entry.attrs["channel_id"] = str(getattr(proc, "channel_id", "") or "")
                entry.attrs["dio_name"] = str(getattr(proc, "dio_name", "") or "")
                entry.attrs["output_label"] = str(getattr(proc, "output_label", "") or "")
                entry.attrs["output_context"] = str(getattr(proc, "output_context", "") or "")
                entry.attrs["fs_actual"] = float(getattr(proc, "fs_actual", np.nan))
                entry.attrs["fs_target"] = float(getattr(proc, "fs_target", np.nan))
                entry.attrs["fs_used"] = float(getattr(proc, "fs_used", np.nan))

                self._write_h5_numeric(entry, "time", np.asarray(getattr(proc, "time", np.array([], float)), float))
                self._write_h5_numeric(entry, "raw_signal", np.asarray(getattr(proc, "raw_signal", np.array([], float)), float))
                self._write_h5_numeric(entry, "raw_reference", np.asarray(getattr(proc, "raw_reference", np.array([], float)), float))

                for field in (
                    "raw_thr_hi",
                    "raw_thr_lo",
                    "dio",
                    "sig_f",
                    "ref_f",
                    "baseline_sig",
                    "baseline_ref",
                    "output",
                    "sync_aligned_time",
                ):
                    value = getattr(proc, field, None)
                    if value is None:
                        continue
                    self._write_h5_numeric(entry, field, np.asarray(value, float))
                sync_report = getattr(proc, "sync_report", {}) or {}
                if sync_report:
                    self._write_h5_json_any(entry, "sync_report_json", sync_report)

                triggers = getattr(proc, "triggers", {}) or {}
                if isinstance(triggers, dict) and triggers:
                    trig_group = entry.create_group("triggers")
                    for trig_idx, (trig_name, trig_values) in enumerate(triggers.items()):
                        arr = np.asarray(trig_values, float).reshape(-1)
                        if arr.size == 0:
                            continue
                        ds = trig_group.create_dataset(f"trigger_{trig_idx:04d}", data=arr)
                        ds.attrs["name"] = str(trig_name)

                artifact_regions = getattr(proc, "artifact_regions_sec", None)
                if artifact_regions:
                    arr = np.asarray(artifact_regions, float).reshape(-1, 2)
                    self._write_h5_numeric(entry, "artifact_regions_sec", arr)
                artifact_regions_auto = getattr(proc, "artifact_regions_auto_sec", None)
                if artifact_regions_auto:
                    arr_auto = np.asarray(artifact_regions_auto, float).reshape(-1, 2)
                    self._write_h5_numeric(entry, "artifact_regions_auto_sec", arr_auto)

            behavior_group = f.create_group("behavior_sources")
            behavior_group.attrs["count"] = int(len(self._behavior_sources))
            for idx, (stem, info) in enumerate(self._behavior_sources.items()):
                source = info or {}
                entry = behavior_group.create_group(f"item_{idx:04d}")
                entry.attrs["stem"] = str(stem)
                entry.attrs["kind"] = str(source.get("kind", _BEHAVIOR_PARSE_BINARY))
                entry.attrs["trajectory_time_col"] = str(source.get("trajectory_time_col", "") or "")
                if source.get("sheet") is not None:
                    entry.attrs["sheet"] = str(source.get("sheet"))
                if source.get("source_path") is not None:
                    entry.attrs["source_path"] = str(source.get("source_path"))

                self._write_h5_numeric(entry, "time", np.asarray(source.get("time", np.array([], float)), float))
                self._write_h5_numeric(
                    entry,
                    "trajectory_time",
                    np.asarray(source.get("trajectory_time", np.array([], float)), float),
                )

                behaviors_group = entry.create_group("behaviors")
                behaviors = source.get("behaviors") or {}
                for b_idx, (name, values) in enumerate(behaviors.items()):
                    data = np.asarray(values, float)
                    kwargs: Dict[str, object] = {}
                    if data.size > 0:
                        kwargs["compression"] = "gzip"
                    ds = behaviors_group.create_dataset(f"item_{b_idx:04d}", data=data, **kwargs)
                    ds.attrs["name"] = str(name)

                trajectory_group = entry.create_group("trajectory")
                trajectory = source.get("trajectory") or {}
                for t_idx, (name, values) in enumerate(trajectory.items()):
                    data = np.asarray(values, float)
                    kwargs: Dict[str, object] = {}
                    if data.size > 0:
                        kwargs["compression"] = "gzip"
                    ds = trajectory_group.create_dataset(f"item_{t_idx:04d}", data=data, **kwargs)
                    ds.attrs["name"] = str(name)

            analysis_group = f.create_group("analysis")
            self._save_signal_events_h5(analysis_group)
            self._save_behavior_analysis_h5(analysis_group)
            try:
                if hasattr(self, "section_temporal") and self.section_temporal is not None:
                    state = self.section_temporal.serialize_state()
                    if state:
                        self._write_h5_json_any(analysis_group, "temporal_modeling_json", state)
            except Exception:
                _LOG.debug("Could not save temporal modeling state to project", exc_info=True)

    def _load_project_h5(self, path: str) -> Dict[str, object]:
        payload: Dict[str, object] = {
            "settings": {},
            "tab_sources_index": 0,
            "processed": [],
            "behavior_sources": {},
            "recent_paths": {},
            "signal_events": None,
            "behavior_analysis": None,
        }
        with h5py.File(path, "r") as f:
            project_type = self._h5_text(f.attrs.get("project_type", ""), "")
            if project_type and project_type != "pyber_postprocessing_project":
                raise ValueError("This H5 file is not a pyBer postprocessing project.")
            if not project_type and "processed" not in f and "ui" not in f:
                raise ValueError("This H5 file does not contain a pyBer postprocessing project.")

            ui_group = f.get("ui")
            if isinstance(ui_group, h5py.Group):
                payload["settings"] = self._read_h5_json(ui_group, "settings_json")
                try:
                    payload["tab_sources_index"] = int(ui_group.attrs.get("tab_sources_index", 0))
                except Exception:
                    payload["tab_sources_index"] = 0

            meta_group = f.get("meta")
            if isinstance(meta_group, h5py.Group):
                payload["recent_paths"] = self._read_h5_json(meta_group, "recent_paths_json")

            loaded_processed: List[ProcessedTrial] = []
            processed_group = f.get("processed")
            if isinstance(processed_group, h5py.Group):
                for key in sorted(processed_group.keys()):
                    entry = processed_group.get(key)
                    if not isinstance(entry, h5py.Group):
                        continue
                    time = self._read_h5_numeric(entry, "time")
                    if time is None or time.size == 0:
                        continue
                    t = np.asarray(time, float).reshape(-1)
                    n = int(t.size)

                    def _aligned(values: Optional[np.ndarray], fill_nan: bool = True) -> np.ndarray:
                        if values is None:
                            return np.full(n, np.nan, dtype=float) if fill_nan else np.array([], float)
                        arr = np.asarray(values, float).reshape(-1)
                        if arr.size == n:
                            return arr
                        if arr.size == 0:
                            return np.full(n, np.nan, dtype=float) if fill_nan else np.array([], float)
                        out = np.full(n, np.nan, dtype=float) if fill_nan else np.array([], float)
                        if fill_nan:
                            m = min(n, arr.size)
                            out[:m] = arr[:m]
                            return out
                        return arr

                    raw_signal = _aligned(self._read_h5_numeric(entry, "raw_signal"), fill_nan=True)
                    raw_reference = _aligned(self._read_h5_numeric(entry, "raw_reference"), fill_nan=True)
                    output_arr = _aligned(self._read_h5_numeric(entry, "output"), fill_nan=True)
                    triggers: Dict[str, np.ndarray] = {}
                    trig_group = entry.get("triggers")
                    if isinstance(trig_group, h5py.Group):
                        for trig_key in sorted(trig_group.keys()):
                            ds = trig_group.get(trig_key)
                            if ds is None:
                                continue
                            trig_name = self._h5_text(getattr(ds, "attrs", {}).get("name", trig_key), trig_key)
                            trig_arr = _aligned(np.asarray(ds[()], float), fill_nan=False)
                            if trig_arr.size:
                                triggers[str(trig_name)] = trig_arr

                    trial = ProcessedTrial(
                        path=self._h5_text(entry.attrs.get("path", ""), ""),
                        channel_id=self._h5_text(entry.attrs.get("channel_id", ""), "import"),
                        time=t,
                        raw_signal=raw_signal,
                        raw_reference=raw_reference,
                        raw_thr_hi=_aligned(self._read_h5_numeric(entry, "raw_thr_hi"), fill_nan=False)
                        if "raw_thr_hi" in entry
                        else None,
                        raw_thr_lo=_aligned(self._read_h5_numeric(entry, "raw_thr_lo"), fill_nan=False)
                        if "raw_thr_lo" in entry
                        else None,
                        dio=_aligned(self._read_h5_numeric(entry, "dio"), fill_nan=False) if "dio" in entry else None,
                        dio_name=self._h5_text(entry.attrs.get("dio_name", ""), ""),
                        triggers=triggers,
                        sig_f=_aligned(self._read_h5_numeric(entry, "sig_f"), fill_nan=False) if "sig_f" in entry else None,
                        ref_f=_aligned(self._read_h5_numeric(entry, "ref_f"), fill_nan=False) if "ref_f" in entry else None,
                        baseline_sig=_aligned(self._read_h5_numeric(entry, "baseline_sig"), fill_nan=False)
                        if "baseline_sig" in entry
                        else None,
                        baseline_ref=_aligned(self._read_h5_numeric(entry, "baseline_ref"), fill_nan=False)
                        if "baseline_ref" in entry
                        else None,
                        output=output_arr,
                        output_label=self._h5_text(entry.attrs.get("output_label", "output"), "output"),
                        output_context=self._h5_text(entry.attrs.get("output_context", ""), ""),
                        sync_aligned_time=_aligned(self._read_h5_numeric(entry, "sync_aligned_time"), fill_nan=False)
                        if "sync_aligned_time" in entry
                        else None,
                        sync_report=self._read_h5_json_any(entry, "sync_report_json", {})
                        if "sync_report_json" in entry
                        else {},
                        artifact_regions_sec=None,
                        artifact_regions_auto_sec=None,
                        fs_actual=float(entry.attrs.get("fs_actual", np.nan)),
                        fs_target=float(entry.attrs.get("fs_target", np.nan)),
                        fs_used=float(entry.attrs.get("fs_used", np.nan)),
                    )

                    regions = self._read_h5_numeric(entry, "artifact_regions_sec")
                    if regions is not None and regions.size:
                        rr = np.asarray(regions, float).reshape(-1, 2)
                        trial.artifact_regions_sec = [(float(a), float(b)) for a, b in rr]
                    regions_auto = self._read_h5_numeric(entry, "artifact_regions_auto_sec")
                    if regions_auto is not None and regions_auto.size:
                        ra = np.asarray(regions_auto, float).reshape(-1, 2)
                        trial.artifact_regions_auto_sec = [(float(a), float(b)) for a, b in ra]

                    loaded_processed.append(trial)

            loaded_behavior: Dict[str, Dict[str, Any]] = {}
            behavior_group = f.get("behavior_sources")
            if isinstance(behavior_group, h5py.Group):
                for key in sorted(behavior_group.keys()):
                    entry = behavior_group.get(key)
                    if not isinstance(entry, h5py.Group):
                        continue
                    stem = self._h5_text(entry.attrs.get("stem", key), key)
                    info: Dict[str, Any] = {
                        "kind": self._h5_text(entry.attrs.get("kind", _BEHAVIOR_PARSE_BINARY), _BEHAVIOR_PARSE_BINARY),
                        "time": np.asarray(
                            self._read_h5_numeric(entry, "time") if "time" in entry else np.array([], float),
                            float,
                        ),
                        "behaviors": {},
                        "trajectory": {},
                        "trajectory_time": np.asarray(
                            self._read_h5_numeric(entry, "trajectory_time")
                            if "trajectory_time" in entry
                            else np.array([], float),
                            float,
                        ),
                        "trajectory_time_col": self._h5_text(entry.attrs.get("trajectory_time_col", ""), ""),
                    }
                    if "sheet" in entry.attrs:
                        info["sheet"] = self._h5_text(entry.attrs.get("sheet", ""), "")
                    if "source_path" in entry.attrs:
                        info["source_path"] = self._h5_text(entry.attrs.get("source_path", ""), "")

                    behaviors_group = entry.get("behaviors")
                    if isinstance(behaviors_group, h5py.Group):
                        for b_key in sorted(behaviors_group.keys()):
                            ds = behaviors_group.get(b_key)
                            if ds is None:
                                continue
                            name = self._h5_text(ds.attrs.get("name", b_key), b_key)
                            info["behaviors"][name] = np.asarray(ds[()], float)

                    trajectory_group = entry.get("trajectory")
                    if isinstance(trajectory_group, h5py.Group):
                        for t_key in sorted(trajectory_group.keys()):
                            ds = trajectory_group.get(t_key)
                            if ds is None:
                                continue
                            name = self._h5_text(ds.attrs.get("name", t_key), t_key)
                            info["trajectory"][name] = np.asarray(ds[()], float)

                    loaded_behavior[stem] = info

            analysis_group = f.get("analysis")
            if isinstance(analysis_group, h5py.Group):
                payload["signal_events"] = self._load_signal_events_h5(analysis_group)
                payload["behavior_analysis"] = self._load_behavior_analysis_h5(analysis_group)
                try:
                    payload["temporal_modeling"] = self._read_h5_json_any(analysis_group, "temporal_modeling_json", {})
                except Exception:
                    payload["temporal_modeling"] = {}

            payload["processed"] = loaded_processed
            payload["behavior_sources"] = loaded_behavior
        return payload

    def _autosave_project_cache_path(self) -> str:
        cache_root = QtCore.QStandardPaths.writableLocation(QtCore.QStandardPaths.StandardLocation.CacheLocation)
        if not cache_root:
            cache_root = QtCore.QStandardPaths.writableLocation(QtCore.QStandardPaths.StandardLocation.AppDataLocation)
        if not cache_root:
            cache_root = os.path.join(os.getcwd(), "cache")
        cache_dir = os.path.join(cache_root, "pyber_postprocessing")
        os.makedirs(cache_dir, exist_ok=True)
        return os.path.join(cache_dir, "autosave_project.h5")

    def _has_project_state_for_autosave(self) -> bool:
        if self._processed or self._behavior_sources:
            return True
        if isinstance(self.last_signal_events, dict) and bool(self.last_signal_events):
            return True
        if isinstance(self.last_behavior_analysis, dict) and bool(self.last_behavior_analysis):
            return True
        return False

    def _clear_project_autosave_cache(self, delete_file: bool = True) -> None:
        path = self._settings.value("postprocess_autosave_project_path", "", type=str).strip()
        if not path:
            path = self._autosave_project_cache_path()
        if delete_file and path and os.path.isfile(path):
            try:
                os.remove(path)
            except Exception:
                pass
        self._settings.setValue("postprocess_autosave_project_available", False)
        self._settings.setValue("postprocess_autosave_project_path", "")
        self._settings.setValue("postprocess_autosave_project_utc", "")
        self._project_recovered_from_autosave = False

    def _autosave_project_to_cache(self) -> None:
        if self._autosave_restoring:
            return
        if not self._has_project_state_for_autosave():
            self._clear_project_autosave_cache(delete_file=True)
            return
        should_write = bool(self.is_project_dirty() or self._project_recovered_from_autosave)
        if not should_write:
            self._clear_project_autosave_cache(delete_file=True)
            return
        path = self._autosave_project_cache_path()
        try:
            self._save_project_h5(path)
            self._settings.setValue("postprocess_autosave_project_available", True)
            self._settings.setValue("postprocess_autosave_project_path", path)
            self._settings.setValue("postprocess_autosave_project_utc", datetime.now(timezone.utc).isoformat())
        except Exception:
            _LOG.exception("Failed to autosave postprocessing project to cache")

    def _restore_project_autosave_if_needed(self) -> None:
        restore_on_startup = bool(
            self._settings.value("postprocess_restore_autosave_on_startup", False, type=bool)
        )
        if not restore_on_startup:
            self._clear_project_autosave_cache(delete_file=True)
            return
        available = bool(self._settings.value("postprocess_autosave_project_available", False, type=bool))
        path = self._settings.value("postprocess_autosave_project_path", "", type=str).strip()
        if not path:
            path = self._autosave_project_cache_path()
        if not available:
            return
        if not path or not os.path.isfile(path):
            self._clear_project_autosave_cache(delete_file=False)
            return

        try:
            self._autosave_restoring = True
            ok = self._load_project_from_path(path, from_autosave=True)
        finally:
            self._autosave_restoring = False
        if not ok:
            self._clear_project_autosave_cache(delete_file=False)

    def _save_project_file(self) -> None:
        start_dir = self._settings.value("postprocess_last_dir", os.getcwd(), type=str)
        default_name = f"{self._default_export_prefix()}_project.h5"
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self,
            "Save postprocessing project",
            os.path.join(start_dir, default_name),
            "HDF5 project (*.h5)",
        )
        if not path:
            return
        if not path.lower().endswith((".h5", ".hdf5")):
            path = f"{path}.h5"
        try:
            self._save_project_h5(path)
            self._push_recent_paths("postprocess_recent_project_paths", [path])
            self._settings.setValue("postprocess_last_dir", os.path.dirname(path))
            self._mark_project_clean()
            self._project_recovered_from_autosave = False
            self._clear_project_autosave_cache(delete_file=True)
            self.statusUpdate.emit(f"Project saved: {os.path.basename(path)}", 5000)
        except Exception as exc:
            QtWidgets.QMessageBox.warning(self, "Save project", f"Could not save project:\n{exc}")

    def _load_project_file(self) -> None:
        start_dir = self._settings.value("postprocess_last_dir", os.getcwd(), type=str)
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Load postprocessing project",
            start_dir,
            "HDF5 project (*.h5 *.hdf5)",
        )
        if not path:
            return
        self._load_project_from_path(path)

    def _confirm_discard_current_project(self) -> bool:
        if not self.is_project_dirty() and not self._has_project_state_for_autosave():
            return True
        ask = QtWidgets.QMessageBox.question(
            self,
            "New project",
            "Discard the current postprocessing project and start a new one?",
            QtWidgets.QMessageBox.StandardButton.Yes | QtWidgets.QMessageBox.StandardButton.No,
            QtWidgets.QMessageBox.StandardButton.No,
        )
        return ask == QtWidgets.QMessageBox.StandardButton.Yes

    def discard_unsaved_project_for_close(self) -> bool:
        """
        Mark the current postprocessing state as intentionally discarded.
        Called from the main close confirmation so Discard does not create a new
        autosave that reopens the same project on the next launch.
        """
        self._project_dirty = False
        self._project_recovered_from_autosave = False
        self._mark_project_clean()
        self._clear_project_autosave_cache(delete_file=True)
        return True

    def _reset_project_state(self) -> None:
        was_restoring = self._is_restoring_settings
        self._is_restoring_settings = True
        try:
            self._clear_cached_analysis_outputs()
            self._processed = []
            self._behavior_sources = {}
            self._sync_results_by_file = {}
            self._last_sync_preview = None
            self._pending_project_recompute_from_current = False
            self._dio_cache.clear()
            self.lbl_group.setText("(none)")
            self.lbl_beh.setText("(none)")
            self.lbl_behavior_msg.setText("")
            self.lbl_signal_msg.setText("")
            if hasattr(self, "txt_sync_report"):
                self.txt_sync_report.clear()
            if hasattr(self, "tbl_sync_results"):
                self.tbl_sync_results.setRowCount(0)
            if hasattr(self, "plot_sync_map"):
                self.plot_sync_map.clear()
            if hasattr(self, "plot_sync_residual"):
                self.plot_sync_residual.clear()
            # Some legacy code expected a self.lbl_status that was never created
            # in the modern UI; clear it defensively if a host adds one later.
            _legacy_status = getattr(self, "lbl_status", None)
            if _legacy_status is not None:
                try:
                    _legacy_status.setText("")
                except Exception:
                    pass
            self.tab_sources.setCurrentIndex(0)
            # Drop cached behavior + signal-event analyses too.
            self.last_signal_events = None
            self.last_behavior_analysis = None
            # Wipe cached temporal-modeling fits and group aggregates.
            try:
                if hasattr(self, "section_temporal"):
                    self.section_temporal._on_clear_cached_fits()
                    self.section_temporal._glm_result = None
                    self.section_temporal._flmm_result = None
                    if hasattr(self.section_temporal, "txt_summary"):
                        self.section_temporal.txt_summary.clear()
                    for plot_attr in (
                        "plot_kernel", "plot_prediction", "plot_residuals",
                        "plot_importance", "plot_illustration", "plot_coeff",
                        "plot_group_kernels", "plot_group_importance",
                    ):
                        pw = getattr(self.section_temporal, plot_attr, None)
                        if pw is not None:
                            try:
                                pw.clear()
                            except Exception:
                                pass
            except Exception:
                pass
            self._update_file_lists()
            self._refresh_behavior_list()
            self._update_trace_preview()
            self._update_behavior_time_panel()
            self._update_data_availability()
        finally:
            self._is_restoring_settings = was_restoring

        self._compute_psth()
        self._compute_spatial_heatmap()
        self._save_settings()
        self._mark_project_clean()
        self._project_recovered_from_autosave = False
        self._clear_project_autosave_cache(delete_file=True)
        self._update_status_strip()

    def reset_for_new_preprocessing_project(self) -> None:
        self._reset_project_state()
        self._reset_history_snapshot()
        self.statusUpdate.emit("Cleared postprocessing project state.", 5000)

    def _new_project(self) -> None:
        if not self._confirm_discard_current_project():
            return

        self._reset_project_state()
        self._reset_history_snapshot()
        self.statusUpdate.emit("Started a new postprocessing project.", 5000)

    def _reset_blank_project(self) -> None:
        if not self._confirm_discard_current_project():
            return
        self._reset_project_state()
        self._reset_history_snapshot()
        self.statusUpdate.emit(
            "Postprocessing reset to a blank project; autosaved project was forgotten.",
            5000,
        )

    def _import_project_source_paths(self, recent_paths: Dict[str, object]) -> bool:
        proc_raw = recent_paths.get("processed_paths", []) if isinstance(recent_paths, dict) else []
        beh_raw = recent_paths.get("behavior_paths", []) if isinstance(recent_paths, dict) else []
        proc_paths = [str(p).strip() for p in (proc_raw if isinstance(proc_raw, list) else []) if str(p).strip()]
        beh_paths = [str(p).strip() for p in (beh_raw if isinstance(beh_raw, list) else []) if str(p).strip()]
        proc_existing = [p for p in proc_paths if os.path.isfile(p)]
        beh_existing = [p for p in beh_paths if os.path.isfile(p)]
        if not proc_existing and not beh_existing:
            return False

        if proc_existing:
            self._load_processed_paths(proc_existing, replace=True)
        if beh_existing:
            self._load_behavior_paths(beh_existing, replace=True)
            self._refresh_behavior_list()
        return bool(proc_existing or beh_existing)

    def _load_project_from_path(self, path: str, from_autosave: bool = False) -> bool:
        try:
            payload = self._load_project_h5(path)
        except Exception as exc:
            QtWidgets.QMessageBox.warning(self, "Load project", f"Could not load project:\n{exc}")
            return False

        settings_data = payload.get("settings", {})
        processed = payload.get("processed", [])
        behavior_sources = payload.get("behavior_sources", {})
        tab_idx = payload.get("tab_sources_index", 0)
        recent_paths = payload.get("recent_paths", {}) if isinstance(payload.get("recent_paths", {}), dict) else {}

        was_restoring = self._is_restoring_settings
        self._is_restoring_settings = True
        try:
            self._clear_cached_analysis_outputs()
            self._processed = list(processed) if isinstance(processed, list) else []
            self._behavior_sources = dict(behavior_sources) if isinstance(behavior_sources, dict) else {}

            self.lbl_group.setText(f"{len(self._processed)} file(s) loaded")
            kinds = {
                str((info or {}).get("kind", "")).strip()
                for info in self._behavior_sources.values()
                if isinstance(info, dict)
            }
            if len(kinds) == 1 and _BEHAVIOR_PARSE_TIMESTAMPS in kinds:
                mode_label = "timestamps"
            elif len(kinds) == 1 and _BEHAVIOR_PARSE_BINARY in kinds:
                mode_label = "binary"
            elif len(kinds) > 1:
                mode_label = "mixed"
            else:
                mode_label = "timestamps" if self._current_behavior_parse_mode() == _BEHAVIOR_PARSE_TIMESTAMPS else "binary"
            self.lbl_beh.setText(f"{len(self._behavior_sources)} file(s) loaded [{mode_label}]")

            self._update_file_lists()
            self._refresh_behavior_list()
            self._refresh_sync_sources()
            self._set_resample_from_processed()
            if isinstance(settings_data, dict) and settings_data:
                self._apply_settings(settings_data)
            if isinstance(tab_idx, int) and 0 <= tab_idx < self.tab_sources.count():
                self.tab_sources.setCurrentIndex(tab_idx)
            self._update_trace_preview()
            self._update_data_availability()
        finally:
            self._is_restoring_settings = was_restoring

        proc_paths = recent_paths.get("processed_paths", []) if isinstance(recent_paths, dict) else []
        beh_paths = recent_paths.get("behavior_paths", []) if isinstance(recent_paths, dict) else []
        if isinstance(proc_paths, list) and proc_paths:
            self._push_recent_paths("postprocess_recent_processed_paths", [str(p) for p in proc_paths if str(p).strip()])
        if isinstance(beh_paths, list) and beh_paths:
            self._push_recent_paths("postprocess_recent_behavior_paths", [str(p) for p in beh_paths if str(p).strip()])

        if not from_autosave:
            self._push_recent_paths("postprocess_recent_project_paths", [path])
            try:
                self._settings.setValue("postprocess_last_dir", os.path.dirname(path))
            except Exception:
                pass

        proc_raw = recent_paths.get("processed_paths", []) if isinstance(recent_paths, dict) else []
        beh_raw = recent_paths.get("behavior_paths", []) if isinstance(recent_paths, dict) else []
        proc_existing = [str(p).strip() for p in (proc_raw if isinstance(proc_raw, list) else []) if str(p).strip() and os.path.isfile(str(p).strip())]
        beh_existing = [str(p).strip() for p in (beh_raw if isinstance(beh_raw, list) else []) if str(p).strip() and os.path.isfile(str(p).strip())]
        has_referenced_sources = bool(proc_existing or beh_existing)

        imported_sources = False
        if has_referenced_sources and not from_autosave:
            ask_sources = QtWidgets.QMessageBox.question(
                self,
                "Load project",
                "Import linked source files from this project (last opened data)?",
                QtWidgets.QMessageBox.StandardButton.Yes | QtWidgets.QMessageBox.StandardButton.No,
                QtWidgets.QMessageBox.StandardButton.No,
            )
            if ask_sources == QtWidgets.QMessageBox.StandardButton.Yes:
                imported_sources = self._import_project_source_paths(recent_paths)

        if self._processed:
            self._compute_psth()
            self._compute_spatial_heatmap()
        elif not imported_sources and not from_autosave:
            ask = QtWidgets.QMessageBox.question(
                self,
                "Load project",
                "Project loaded without processed traces. Import current preprocessing selection now?",
                QtWidgets.QMessageBox.StandardButton.Yes | QtWidgets.QMessageBox.StandardButton.No,
                QtWidgets.QMessageBox.StandardButton.Yes,
            )
            if ask == QtWidgets.QMessageBox.StandardButton.Yes:
                self._pending_project_recompute_from_current = True
                self.requestCurrentProcessed.emit()

        self._restore_cached_analysis_outputs(payload)
        self._save_settings()
        self._update_status_strip()
        self._mark_project_clean()
        self._project_recovered_from_autosave = bool(from_autosave)
        if from_autosave:
            self.statusUpdate.emit("Recovered autosaved postprocessing project.", 5000)
        else:
            self.statusUpdate.emit(f"Project loaded: {os.path.basename(path)}", 5000)
        self._reset_history_snapshot()
        return True

    def _default_settings_payload(self) -> Dict[str, object]:
        return {
            "align": "Behavior (CSV/XLSX)",
            "dio_channel": self.combo_dio.currentText(),
            "dio_polarity": "Event high (0->1)",
            "dio_align": "Align to onset",
            "sync_behavior_file": "",
            "sync_camera_column": "",
            "sync_camera_mode": "TTL rising edges",
            "sync_fiber_source": "__embedded__",
            "sync_fiber_mode": "TTL rising edges",
            "sync_method": "Linear regression",
            "sync_auto_threshold": True,
            "sync_threshold": 0.5,
            "sync_min_interval": 0.2,
            "sync_max_offset": 5,
            "sync_use_aligned": False,
            "sync_auto_recompute": True,
            "sync_export_format": "CSV + H5",
            "behavior_file_type": "Binary states (time + 0/1 columns)",
            "behavior_time_fps": 30.0,
            "behavior": self.combo_behavior_name.currentText(),
            "behavior_align": self.combo_behavior_align.currentText(),
            "behavior_from": self.combo_behavior_from.currentText(),
            "behavior_to": self.combo_behavior_to.currentText(),
            "transition_gap": 1.0,
            "window_pre": 2.0,
            "window_post": 5.0,
            "baseline_start": -1.0,
            "baseline_end": 0.0,
            "resample": 50.0,
            "smooth": 0.0,
            "filter_enabled": True,
            "event_start": 1,
            "event_end": 0,
            "group_window_s": 0.0,
            "dur_min": 0.0,
            "dur_max": 0.0,
            "psth_exclude_low_event_animals": False,
            "psth_min_events_per_animal": 5,
            "psth_group_keep_trials": False,
            "metrics_enabled": True,
            "metric": "AUC",
            "metric_pre0": -1.0,
            "metric_pre1": 0.0,
            "metric_post0": 0.0,
            "metric_post1": 1.0,
            "global_metrics_enabled": True,
            "global_start": 0.0,
            "global_end": 0.0,
            "global_amp": True,
            "global_freq": True,
            "view_layout": "Standard",
            "visual_mode": 0,
            "individual_file": self.combo_individual_file.currentText().strip(),
            "signal_source": "Use processed output trace (loaded file)",
            "signal_scope": "Per file",
            "signal_file": self.combo_signal_file.currentText(),
            "signal_method": "SciPy find_peaks",
            "signal_prominence": 0.5,
            "signal_auto_mad": False,
            "signal_mad_multiplier": 5.0,
            "signal_height": 0.0,
            "signal_distance": 0.5,
            "signal_smooth": 0.0,
            "signal_baseline": "Use trace as-is",
            "signal_baseline_window": 10.0,
            "signal_norm_prominence": False,
            "signal_rate_bin": 60.0,
            "signal_auc_window": 0.5,
            "signal_overlay": True,
            "signal_noise_overlay": False,
            "behavior_analysis_name": self.combo_behavior_analysis.currentText(),
            "behavior_analysis_bin": 30.0,
            "behavior_analysis_aligned": False,
            "spatial_x": self.combo_spatial_x.currentText(),
            "spatial_y": self.combo_spatial_y.currentText(),
            "spatial_bins_x": 64,
            "spatial_bins_y": 64,
            "spatial_weight": "Occupancy (samples)",
            "spatial_clip": True,
            "spatial_clip_low": 1.0,
            "spatial_clip_high": 99.0,
            "spatial_time_filter": False,
            "spatial_time_min": 0.0,
            "spatial_time_max": 0.0,
            "spatial_smooth": 0.0,
            "spatial_activity_mode": "Mean z-score/bin (occupancy normalized)",
            "spatial_activity_norm": True,
            "spatial_log": False,
            "spatial_invert_y": False,
            "style": {
                "trace": (90, 190, 255),
                "behavior": (220, 180, 80),
                "avg": (90, 190, 255),
                "sem_edge": (152, 201, 143),
                "sem_fill": (188, 230, 178, 96),
                "plot_bg": (248, 250, 255) if self._app_theme_mode == "light" else (36, 42, 52),
                "grid_enabled": True,
                "grid_alpha": 0.25,
                "heatmap_cmap": "viridis",
                "heatmap_min": None,
                "heatmap_max": None,
                "heatmap_levels_manual": False,
            },
        }

    def _reset_config_defaults(self) -> None:
        previous_restoring = self._is_restoring_settings
        previous_history_restoring = self._history_restoring
        self._is_restoring_settings = True
        self._history_restoring = True
        try:
            self._apply_settings(self._default_settings_payload())
        finally:
            self._is_restoring_settings = previous_restoring
            self._history_restoring = previous_history_restoring
        self._record_history_change()
        self._rerender_visual_from_cache()
        self._compute_spatial_heatmap()
        self._save_settings()
        self.statusUpdate.emit("Reset postprocessing parameters to defaults.", 3000)

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
            self._compute_spatial_heatmap()
            self._settings.setValue("postprocess_last_dir", os.path.dirname(path))
        except Exception:
            pass

    def _collect_settings(self) -> Dict[str, object]:
        return {
            "align": self.combo_align.currentText(),
            "dio_channel": self.combo_dio.currentText(),
            "dio_polarity": self.combo_dio_polarity.currentText(),
            "dio_align": self.combo_dio_align.currentText(),
            "sync_behavior_file": self.combo_sync_behavior_file.currentData(),
            "sync_camera_column": self.combo_sync_camera_column.currentData(),
            "sync_camera_mode": self.combo_sync_camera_mode.currentText(),
            "sync_fiber_source": self.combo_sync_fiber_source.currentData(),
            "sync_fiber_mode": self.combo_sync_fiber_mode.currentText(),
            "sync_method": self.combo_sync_method.currentText(),
            "sync_auto_threshold": self.cb_sync_auto_threshold.isChecked(),
            "sync_threshold": float(self.spin_sync_threshold.value()),
            "sync_min_interval": float(self.spin_sync_min_interval.value()),
            "sync_max_offset": int(self.spin_sync_max_offset.value()),
            "sync_use_aligned": self.cb_sync_use_aligned.isChecked(),
            "sync_auto_recompute": self.cb_sync_auto_recompute.isChecked(),
            "sync_export_format": self.combo_sync_export_format.currentText(),
            "behavior_file_type": self.combo_behavior_file_type.currentText(),
            "behavior_time_fps": float(self.spin_behavior_fps.value()),
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
            "psth_exclude_low_event_animals": self.cb_exclude_low_event_animals.isChecked(),
            "psth_min_events_per_animal": int(self.spin_min_events_per_animal.value()),
            "psth_group_keep_trials": self.cb_group_keep_trials.isChecked(),
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
            "visual_mode": int(self.tab_visual_mode.currentIndex()),
            "individual_file": self.combo_individual_file.currentText().strip(),
            "signal_source": self.combo_signal_source.currentText(),
            "signal_scope": self.combo_signal_scope.currentText(),
            "signal_file": self.combo_signal_file.currentText(),
            "signal_method": self.combo_signal_method.currentText(),
            "signal_prominence": float(self.spin_peak_prominence.value()),
            "signal_auto_mad": self.cb_peak_auto_mad.isChecked(),
            "signal_mad_multiplier": float(self.spin_peak_mad_multiplier.value()),
            "signal_height": float(self.spin_peak_height.value()),
            "signal_distance": float(self.spin_peak_distance.value()),
            "signal_smooth": float(self.spin_peak_smooth.value()),
            "signal_baseline": self.combo_peak_baseline.currentText(),
            "signal_baseline_window": float(self.spin_peak_baseline_window.value()),
            "signal_norm_prominence": self.cb_peak_norm_prominence.isChecked(),
            "signal_rate_bin": float(self.spin_peak_rate_bin.value()),
            "signal_auc_window": float(self.spin_peak_auc_window.value()),
            "signal_overlay": self.cb_peak_overlay.isChecked(),
            "signal_noise_overlay": self.cb_peak_noise_overlay.isChecked(),
            "behavior_analysis_name": self.combo_behavior_analysis.currentText(),
            "behavior_analysis_bin": float(self.spin_behavior_bin.value()),
            "behavior_analysis_aligned": self.cb_behavior_aligned.isChecked(),
            "spatial_x": self.combo_spatial_x.currentText(),
            "spatial_y": self.combo_spatial_y.currentText(),
            "spatial_bins_x": int(self.spin_spatial_bins_x.value()),
            "spatial_bins_y": int(self.spin_spatial_bins_y.value()),
            "spatial_weight": self.combo_spatial_weight.currentText(),
            "spatial_clip": self.cb_spatial_clip.isChecked(),
            "spatial_clip_low": float(self.spin_spatial_clip_low.value()),
            "spatial_clip_high": float(self.spin_spatial_clip_high.value()),
            "spatial_time_filter": self.cb_spatial_time_filter.isChecked(),
            "spatial_time_min": float(self.spin_spatial_time_min.value()),
            "spatial_time_max": float(self.spin_spatial_time_max.value()),
            "spatial_smooth": float(self.spin_spatial_smooth.value()),
            "spatial_activity_mode": self.combo_spatial_activity_mode.currentText(),
            "spatial_activity_norm": self.combo_spatial_activity_mode.currentText().strip().lower().startswith("mean"),
            "spatial_log": self.cb_spatial_log.isChecked(),
            "spatial_invert_y": self.cb_spatial_invert_y.isChecked(),
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

        def _set_combo_data(combo: QtWidgets.QComboBox, val: object) -> None:
            if val is None:
                return
            idx = combo.findData(val)
            if idx < 0:
                idx = combo.findText(str(val))
            if idx >= 0:
                combo.setCurrentIndex(idx)

        _set_combo(self.combo_align, data.get("align"))
        _set_combo(self.combo_dio, data.get("dio_channel"))
        _set_combo(self.combo_dio_polarity, data.get("dio_polarity"))
        _set_combo(self.combo_dio_align, data.get("dio_align"))
        self._refresh_sync_sources()
        _set_combo_data(self.combo_sync_behavior_file, data.get("sync_behavior_file"))
        self._refresh_sync_camera_columns()
        _set_combo_data(self.combo_sync_camera_column, data.get("sync_camera_column"))
        _set_combo(self.combo_sync_camera_mode, data.get("sync_camera_mode"))
        self._refresh_sync_fiber_sources()
        _set_combo_data(self.combo_sync_fiber_source, data.get("sync_fiber_source"))
        _set_combo(self.combo_sync_fiber_mode, data.get("sync_fiber_mode"))
        _set_combo(self.combo_sync_method, data.get("sync_method"))
        if "sync_auto_threshold" in data:
            self.cb_sync_auto_threshold.setChecked(bool(data["sync_auto_threshold"]))
        if "sync_threshold" in data:
            self.spin_sync_threshold.setValue(float(data["sync_threshold"]))
        if "sync_min_interval" in data:
            self.spin_sync_min_interval.setValue(float(data["sync_min_interval"]))
        if "sync_max_offset" in data:
            self.spin_sync_max_offset.setValue(int(data["sync_max_offset"]))
        if "sync_use_aligned" in data:
            self.cb_sync_use_aligned.setChecked(bool(data["sync_use_aligned"]))
        if "sync_auto_recompute" in data:
            self.cb_sync_auto_recompute.setChecked(bool(data["sync_auto_recompute"]))
        _set_combo(self.combo_sync_export_format, data.get("sync_export_format"))
        self._on_sync_auto_threshold_toggled(self.cb_sync_auto_threshold.isChecked())
        _set_combo(self.combo_behavior_file_type, data.get("behavior_file_type"))
        if "behavior_time_fps" in data:
            self.spin_behavior_fps.setValue(float(data["behavior_time_fps"]))
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
        if "psth_exclude_low_event_animals" in data:
            self.cb_exclude_low_event_animals.setChecked(bool(data["psth_exclude_low_event_animals"]))
        if "psth_min_events_per_animal" in data:
            self.spin_min_events_per_animal.setValue(max(1, int(data["psth_min_events_per_animal"])))
        if "psth_group_keep_trials" in data:
            self.cb_group_keep_trials.setChecked(bool(data["psth_group_keep_trials"]))
        self._update_psth_inclusion_controls()
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
        if "visual_mode" in data:
            try:
                idx = int(data.get("visual_mode") or 0)
                if 0 <= idx < self.tab_visual_mode.count():
                    self.tab_visual_mode.setCurrentIndex(idx)
            except Exception:
                pass
        if "individual_file" in data:
            file_id = str(data.get("individual_file") or "").strip()
            if file_id:
                idx = self.combo_individual_file.findText(file_id)
                if idx >= 0:
                    self.combo_individual_file.setCurrentIndex(idx)
        _set_combo(self.combo_signal_source, data.get("signal_source"))
        _set_combo(self.combo_signal_scope, data.get("signal_scope"))
        self._refresh_signal_file_combo()
        _set_combo(self.combo_signal_file, data.get("signal_file"))
        _set_combo(self.combo_signal_method, data.get("signal_method"))
        if "signal_prominence" in data:
            self.spin_peak_prominence.setValue(float(data["signal_prominence"]))
        if "signal_auto_mad" in data:
            self.cb_peak_auto_mad.setChecked(bool(data["signal_auto_mad"]))
        if "signal_mad_multiplier" in data:
            self.spin_peak_mad_multiplier.setValue(float(data["signal_mad_multiplier"]))
        self._update_peak_auto_mad_enabled(queue=False)
        if "signal_height" in data:
            self.spin_peak_height.setValue(float(data["signal_height"]))
        if "signal_distance" in data:
            self.spin_peak_distance.setValue(float(data["signal_distance"]))
        if "signal_smooth" in data:
            self.spin_peak_smooth.setValue(float(data["signal_smooth"]))
        _set_combo(self.combo_peak_baseline, data.get("signal_baseline"))
        if "signal_baseline_window" in data:
            self.spin_peak_baseline_window.setValue(float(data["signal_baseline_window"]))
        if "signal_norm_prominence" in data:
            self.cb_peak_norm_prominence.setChecked(bool(data["signal_norm_prominence"]))
        if "signal_rate_bin" in data:
            self.spin_peak_rate_bin.setValue(float(data["signal_rate_bin"]))
        if "signal_auc_window" in data:
            self.spin_peak_auc_window.setValue(float(data["signal_auc_window"]))
        if "signal_overlay" in data:
            self.cb_peak_overlay.setChecked(bool(data["signal_overlay"]))
        if "signal_noise_overlay" in data:
            self.cb_peak_noise_overlay.setChecked(bool(data["signal_noise_overlay"]))
        _set_combo(self.combo_behavior_analysis, data.get("behavior_analysis_name"))
        if "behavior_analysis_bin" in data:
            self.spin_behavior_bin.setValue(float(data["behavior_analysis_bin"]))
        if "behavior_analysis_aligned" in data:
            self.cb_behavior_aligned.setChecked(bool(data["behavior_analysis_aligned"]))
        _set_combo(self.combo_spatial_x, data.get("spatial_x"))
        _set_combo(self.combo_spatial_y, data.get("spatial_y"))
        if "spatial_bins_x" in data:
            self.spin_spatial_bins_x.setValue(int(data["spatial_bins_x"]))
        if "spatial_bins_y" in data:
            self.spin_spatial_bins_y.setValue(int(data["spatial_bins_y"]))
        _set_combo(self.combo_spatial_weight, data.get("spatial_weight"))
        if "spatial_clip" in data:
            self.cb_spatial_clip.setChecked(bool(data["spatial_clip"]))
        if "spatial_clip_low" in data:
            self.spin_spatial_clip_low.setValue(float(data["spatial_clip_low"]))
        if "spatial_clip_high" in data:
            self.spin_spatial_clip_high.setValue(float(data["spatial_clip_high"]))
        if "spatial_time_filter" in data:
            self.cb_spatial_time_filter.setChecked(bool(data["spatial_time_filter"]))
        if "spatial_time_min" in data:
            self.spin_spatial_time_min.setValue(float(data["spatial_time_min"]))
        if "spatial_time_max" in data:
            self.spin_spatial_time_max.setValue(float(data["spatial_time_max"]))
        if "spatial_smooth" in data:
            self.spin_spatial_smooth.setValue(float(data["spatial_smooth"]))
        if "spatial_activity_mode" in data:
            _set_combo(self.combo_spatial_activity_mode, data.get("spatial_activity_mode"))
        elif "spatial_activity_norm" in data:
            if bool(data["spatial_activity_norm"]):
                _set_combo(self.combo_spatial_activity_mode, "Mean z-score/bin (occupancy normalized)")
            else:
                _set_combo(self.combo_spatial_activity_mode, "Sum z-score/bin")
        if "spatial_log" in data:
            self.cb_spatial_log.setChecked(bool(data["spatial_log"]))
        if "spatial_invert_y" in data:
            self.cb_spatial_invert_y.setChecked(bool(data["spatial_invert_y"]))
        style = data.get("style")
        if isinstance(style, dict):
            self._style.update(style)
            self._apply_plot_style()
        self._apply_behavior_time_settings()
        self._update_event_filter_enabled()
        self._update_metrics_enabled()
        self._update_global_metrics_enabled()
        self._update_spatial_clip_enabled()
        self._update_spatial_time_filter_enabled()
        self._update_metric_regions()
        self._apply_view_layout()
        self._refresh_signal_file_combo()
        self._refresh_sync_sources()
        self._compute_spatial_heatmap()
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
        was_restoring = self._is_restoring_settings
        self._is_restoring_settings = True
        try:
            raw = self._settings.value("postprocess_json", "", type=str)
            if raw:
                data = json.loads(raw)
                self._apply_settings(data)
            if self._app_theme_mode == "dark":
                bg = self._style_color_tuple("plot_bg", self._theme_plot_background())
                if (sum(bg[:3]) / 3.0) >= 180.0:
                    self._style["plot_bg"] = self._theme_plot_background()
                    self._apply_plot_style()
        except Exception:
            pass
        finally:
            self._is_restoring_settings = was_restoring

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

    def _is_group_export_context(self) -> bool:
        return bool(hasattr(self, "tab_sources") and self.tab_sources.currentIndex() == 1)

    def _group_export_prefix(self, prefix: str) -> str:
        clean = str(prefix or "").strip() or "postprocess"
        if self._is_group_export_context() and not clean.lower().startswith("group_"):
            return f"group_{clean}"
        return clean

    def _default_export_prefix(self) -> str:
        prefix = "postprocess"
        if self._processed:
            prefix = os.path.splitext(os.path.basename(self._processed[0].path))[0]
        beh_suffix = self._behavior_suffix()
        if beh_suffix:
            prefix = f"{prefix}_{beh_suffix}"
        return self._group_export_prefix(prefix)

    def _format_export_param_value(self, value: object) -> str:
        if isinstance(value, bool):
            return "yes" if value else "no"
        if isinstance(value, (int, np.integer)):
            return str(int(value))
        if isinstance(value, (float, np.floating)):
            return f"{float(value):.6g}" if np.isfinite(value) else "nan"
        if value is None:
            return ""
        return str(value)

    def _write_export_parameter_file(
        self,
        base_path: str,
        sections: List[Tuple[str, List[Tuple[str, object]]]],
    ) -> Optional[str]:
        if not base_path:
            return None
        out_path = f"{base_path}_params.txt"
        lines = [
            "pyBer postprocessing export parameters",
            f"generated_utc: {datetime.now(timezone.utc).isoformat()}",
            f"export_mode: {'group' if self._is_group_export_context() else 'single'}",
            f"processed_files: {len(self._processed)}",
            f"behavior_files: {len(self._behavior_sources)}",
        ]
        for title, items in sections:
            if not items:
                continue
            lines.append("")
            lines.append(f"[{title}]")
            for key, value in items:
                lines.append(f"{key}: {self._format_export_param_value(value)}")
        try:
            with open(out_path, "w", encoding="utf-8") as f:
                f.write("\n".join(lines) + "\n")
        except Exception:
            return None
        return out_path

    def _collect_psth_parameter_sections(self, include_heatmap: bool = False) -> List[Tuple[str, List[Tuple[str, object]]]]:
        data = self._collect_settings()
        align_items: List[Tuple[str, object]] = [("align_source", data.get("align", ""))]
        align_text = str(data.get("align", ""))
        if align_text.startswith("Analog/Digital"):
            align_items.extend(
                [
                    ("dio_channel", data.get("dio_channel", "")),
                    ("dio_polarity", data.get("dio_polarity", "")),
                    ("dio_align", data.get("dio_align", "")),
                ]
            )
        elif align_text.startswith("Behavior"):
            align_items.extend(
                [
                    ("behavior_file_type", data.get("behavior_file_type", "")),
                    ("behavior", data.get("behavior", "")),
                    ("behavior_align", data.get("behavior_align", "")),
                    ("behavior_from", data.get("behavior_from", "")),
                    ("behavior_to", data.get("behavior_to", "")),
                    ("transition_gap_s", data.get("transition_gap", 0.0)),
                    ("generated_time_fps", data.get("behavior_time_fps", 0.0)),
                ]
            )

        psth_items: List[Tuple[str, object]] = [
            ("window_pre_s", data.get("window_pre", 0.0)),
            ("window_post_s", data.get("window_post", 0.0)),
            ("baseline_start_s", data.get("baseline_start", 0.0)),
            ("baseline_end_s", data.get("baseline_end", 0.0)),
            ("resample_hz", data.get("resample", 0.0)),
            ("gaussian_smooth_sigma_s", data.get("smooth", 0.0)),
            ("event_filters_enabled", data.get("filter_enabled", False)),
            ("event_index_start", data.get("event_start", 0)),
            ("event_index_end", data.get("event_end", 0)),
            ("group_events_within_s", data.get("group_window_s", 0.0)),
            ("event_duration_min_s", data.get("dur_min", 0.0)),
            ("event_duration_max_s", data.get("dur_max", 0.0)),
            ("exclude_animals_below_min_events", data.get("psth_exclude_low_event_animals", False)),
            ("min_events_per_animal", data.get("psth_min_events_per_animal", 1)),
            ("group_view_keeps_trial_rows", data.get("psth_group_keep_trials", False)),
            ("excluded_animals_or_files", len(getattr(self, "_psth_excluded_files", {}) or {})),
            ("metrics_enabled", data.get("metrics_enabled", False)),
            ("metric", data.get("metric", "")),
            ("metric_pre_start_s", data.get("metric_pre0", 0.0)),
            ("metric_pre_end_s", data.get("metric_pre1", 0.0)),
            ("metric_post_start_s", data.get("metric_post0", 0.0)),
            ("metric_post_end_s", data.get("metric_post1", 0.0)),
            ("global_metrics_enabled", data.get("global_metrics_enabled", False)),
            ("global_start_s", data.get("global_start", 0.0)),
            ("global_end_s", data.get("global_end", 0.0)),
            ("global_peak_amplitude", data.get("global_amp", False)),
            ("global_transient_frequency", data.get("global_freq", False)),
        ]
        sections: List[Tuple[str, List[Tuple[str, object]]]] = [
            ("Alignment", align_items),
            ("PSTH", psth_items),
        ]
        if include_heatmap:
            rows = int(self._last_mat.shape[0]) if isinstance(self._last_mat, np.ndarray) and self._last_mat.ndim >= 1 else 0
            cols = int(self._last_mat.shape[1]) if isinstance(self._last_mat, np.ndarray) and self._last_mat.ndim >= 2 else 0
            sections.append(
                (
                    "Heatmap",
                    [
                        ("rows", rows),
                        ("columns", cols),
                        ("color_map", self._style.get("heatmap_cmap", "viridis")),
                        ("display_min", self._style.get("heatmap_min", None)),
                        ("display_max", self._style.get("heatmap_max", None)),
                    ],
                )
            )
        return sections

    def _collect_spatial_parameter_sections(self) -> List[Tuple[str, List[Tuple[str, object]]]]:
        data = self._collect_settings()
        return [
            (
                "Spatial",
                [
                    ("x_column", data.get("spatial_x", "")),
                    ("y_column", data.get("spatial_y", "")),
                    ("bins_x", data.get("spatial_bins_x", 0)),
                    ("bins_y", data.get("spatial_bins_y", 0)),
                    ("occupancy_map_value", data.get("spatial_weight", "")),
                    ("clip_enabled", data.get("spatial_clip", False)),
                    ("clip_low_percentile", data.get("spatial_clip_low", 0.0)),
                    ("clip_high_percentile", data.get("spatial_clip_high", 0.0)),
                    ("time_filter_enabled", data.get("spatial_time_filter", False)),
                    ("time_min_s", data.get("spatial_time_min", 0.0)),
                    ("time_max_s", data.get("spatial_time_max", 0.0)),
                    ("smooth_bins", data.get("spatial_smooth", 0.0)),
                    ("activity_map_mode", data.get("spatial_activity_mode", "")),
                    ("log_scale", data.get("spatial_log", False)),
                    ("invert_y", data.get("spatial_invert_y", False)),
                    ("color_map", self._style.get("heatmap_cmap", "viridis")),
                    ("display_min", self._style.get("heatmap_min", None)),
                    ("display_max", self._style.get("heatmap_max", None)),
                ],
            )
        ]

    def _render_widget_image(
        self,
        widget: QtWidgets.QWidget,
        transparent: bool = True,
    ) -> Optional[QtGui.QImage]:
        if widget is None:
            return None
        size = widget.size()
        if not size.isValid() or size.width() <= 0 or size.height() <= 0:
            size = widget.sizeHint()
        if not size.isValid() or size.width() <= 0 or size.height() <= 0:
            size = QtCore.QSize(1280, 720)
        dpr = 1.0
        try:
            dpr = max(1.0, float(widget.devicePixelRatioF()))
        except Exception:
            dpr = 1.0
        image = QtGui.QImage(
            max(1, int(round(size.width() * dpr))),
            max(1, int(round(size.height() * dpr))),
            QtGui.QImage.Format.Format_ARGB32,
        )
        image.setDevicePixelRatio(dpr)
        if transparent:
            image.fill(QtCore.Qt.GlobalColor.transparent)
        else:
            image.fill(QtGui.QColor(255, 255, 255))
        painter = QtGui.QPainter(image)
        try:
            painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing, True)
            painter.setRenderHint(QtGui.QPainter.RenderHint.TextAntialiasing, True)
            # Keep alpha background while rendering children.
            widget.render(
                painter,
                QtCore.QPoint(),
                QtGui.QRegion(),
                QtWidgets.QWidget.RenderFlag.DrawChildren,
            )
        finally:
            painter.end()
        return image

    def _write_widget_pdf(self, widget: QtWidgets.QWidget, path: str) -> bool:
        if widget is None:
            return False
        size = widget.size()
        if not size.isValid() or size.width() <= 0 or size.height() <= 0:
            size = widget.sizeHint()
        if not size.isValid() or size.width() <= 0 or size.height() <= 0:
            size = QtCore.QSize(1280, 720)
        writer = QtGui.QPdfWriter(path)
        writer.setResolution(300)
        width_px = float(size.width())
        height_px = float(size.height())
        width_pt = max(1.0, width_px * 72.0 / 96.0)
        height_pt = max(1.0, height_px * 72.0 / 96.0)
        writer.setPageSize(QtGui.QPageSize(QtCore.QSizeF(width_pt, height_pt), QtGui.QPageSize.Unit.Point))
        writer.setPageMargins(QtCore.QMarginsF(0.0, 0.0, 0.0, 0.0))
        painter = QtGui.QPainter(writer)
        try:
            painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing, True)
            painter.setRenderHint(QtGui.QPainter.RenderHint.TextAntialiasing, True)
            widget.render(
                painter,
                QtCore.QPoint(),
                QtGui.QRegion(),
                QtWidgets.QWidget.RenderFlag.DrawChildren,
            )
        finally:
            painter.end()
        return True

    def _export_widget_png_pdf(self, widget: QtWidgets.QWidget, base_path: str, transparent: bool = True) -> Tuple[bool, Optional[str], Optional[str]]:
        image = self._render_widget_image(widget, transparent=transparent)
        if image is None or image.isNull():
            return False, None, None
        png_path = f"{base_path}.png"
        pdf_path = f"{base_path}.pdf"
        ok_png = bool(image.save(png_path, "PNG"))
        ok_pdf = self._write_widget_pdf(widget, pdf_path)
        return (ok_png and ok_pdf), (png_path if ok_png else None), (pdf_path if ok_pdf else None)

    def _export_results(self) -> None:
        if self._last_mat is None or self._last_tvec is None:
            return
        is_group = (
            self.tab_visual_mode.currentIndex() == 1
            and str(getattr(self, "_last_psth_display_level", "trials")) == "animals"
            and bool(self._group_labels)
        )
        dlg = ExportDialog(self, group_labels=self._group_labels if is_group else None)
        if dlg.exec() != QtWidgets.QDialog.DialogCode.Accepted:
            return
        choices = dlg.choices()
        start_dir = self._export_start_dir()
        out_dir = QtWidgets.QFileDialog.getExistingDirectory(self, "Select export folder", start_dir)
        if not out_dir:
            return
        self._remember_export_dir(out_dir)
        prefix = self._default_export_prefix()
        do_csv = bool(choices.get("csv", True))
        do_h5 = bool(choices.get("h5", False))
        do_png = bool(choices.get("png", True))
        do_pdf = bool(choices.get("pdf", True))

        # Determine row labels for heatmap columns
        display_labels = list(getattr(self, "_last_display_labels", []) or [])
        if display_labels and len(display_labels) == int(self._last_mat.shape[0]):
            row_labels = display_labels
        elif is_group and self._group_labels:
            row_labels = self._group_labels
        else:
            row_labels = [f"trial_{i + 1}" for i in range(self._last_mat.shape[0])]

        if choices.get("heatmap"):
            heat_base = os.path.join(out_dir, f"{prefix}_heatmap")
            mat = np.asarray(self._last_mat, float)
            if mat.ndim == 1:
                mat = mat[np.newaxis, :]
            time = np.asarray(self._last_tvec, float)
            n_time = min(time.size, mat.shape[1])
            time = time[:n_time]
            mat = mat[:, :n_time]
            arr = np.column_stack([time, mat.T])
            header_cols = ["time"] + list(row_labels[:mat.shape[0]])
            if do_csv:
                np.savetxt(f"{heat_base}.csv", arr, delimiter=",",
                           header=",".join(header_cols), comments="")
            if do_h5:
                with h5py.File(f"{heat_base}.h5", "w") as hf:
                    hf.create_dataset("time", data=time)
                    hf.create_dataset("matrix", data=mat)
                    hf.attrs["row_labels"] = row_labels[:mat.shape[0]]
            self._write_export_parameter_file(heat_base, self._collect_psth_parameter_sections(include_heatmap=True))

        if choices.get("heatmap_aligned"):
            # Export all per-file heatmaps stacked with file_id column
            aligned_base = os.path.join(out_dir, f"{prefix}_heatmap_aligned")
            if self._per_file_mats and do_csv:
                import csv as csv_mod
                with open(f"{aligned_base}.csv", "w", newline="") as f:
                    w = csv_mod.writer(f)
                    first_id = next(iter(self._per_file_mats))
                    tvec_ref = self._per_file_mats[first_id][0]
                    w.writerow(["file_id", "trial"] + [f"{t:.4f}" for t in tvec_ref])
                    for fid, (tvec_f, mat_f) in self._per_file_mats.items():
                        for j in range(mat_f.shape[0]):
                            row_data = [fid, f"trial_{j + 1}"] + [f"{v:.6f}" for v in mat_f[j, :min(tvec_ref.size, mat_f.shape[1])]]
                            w.writerow(row_data)
            if self._per_file_mats and do_h5:
                with h5py.File(f"{aligned_base}.h5", "w") as hf:
                    for fid, (tvec_f, mat_f) in self._per_file_mats.items():
                        grp = hf.create_group(fid)
                        grp.create_dataset("time", data=tvec_f)
                        grp.create_dataset("matrix", data=mat_f)

        if choices.get("avg"):
            avg_base = os.path.join(out_dir, f"{prefix}_avg_psth")
            avg = np.nanmean(self._last_mat, axis=0)
            n_valid = max(1, np.sum(np.any(np.isfinite(self._last_mat), axis=1)))
            sem = np.nanstd(self._last_mat, axis=0) / np.sqrt(n_valid)
            arr = np.vstack([self._last_tvec, avg, sem]).T
            if do_csv:
                np.savetxt(f"{avg_base}.csv", arr, delimiter=",",
                           header="time,average_psth,sem", comments="")
            if do_h5:
                with h5py.File(f"{avg_base}.h5", "w") as hf:
                    hf.create_dataset("time", data=self._last_tvec)
                    hf.create_dataset("average", data=avg)
                    hf.create_dataset("sem", data=sem)
            self._write_export_parameter_file(avg_base, self._collect_psth_parameter_sections(include_heatmap=False))

        if choices.get("events"):
            event_base = os.path.join(out_dir, f"{prefix}_events")
            if self._last_event_rows:
                import csv
                if do_csv:
                    with open(f"{event_base}.csv", "w", newline="") as f:
                        w = csv.writer(f)
                        w.writerow(["file_id", "event_time_sec", "duration_sec"])
                        for row in self._last_event_rows:
                            w.writerow([row.get("file_id", ""), row.get("event_time_sec", np.nan), row.get("duration_sec", np.nan)])
            elif self._last_events is not None and do_csv:
                np.savetxt(f"{event_base}.csv", self._last_events, delimiter=",")

        if choices.get("durations") and self._last_durations is not None:
            dur_base = os.path.join(out_dir, f"{prefix}_durations")
            if do_csv:
                np.savetxt(f"{dur_base}.csv", self._last_durations, delimiter=",")

        if choices.get("metrics") and (self._last_metrics or self._last_global_metrics):
            import csv
            met_base = os.path.join(out_dir, f"{prefix}_metrics")
            if do_csv:
                with open(f"{met_base}.csv", "w", newline="") as f:
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

        # --- Plot exports ---
        if choices.get("plot_heatmap") and hasattr(self, "row_heat"):
            base = os.path.join(out_dir, f"{prefix}_plot_heatmap")
            self._export_widget_selective(self.row_heat, base, do_png, do_pdf)
        if choices.get("plot_avg") and hasattr(self, "row_avg"):
            base = os.path.join(out_dir, f"{prefix}_plot_avg")
            self._export_widget_selective(self.row_avg, base, do_png, do_pdf)
        if choices.get("plot_trace") and hasattr(self, "plot_trace"):
            base = os.path.join(out_dir, f"{prefix}_plot_trace")
            self._export_widget_selective(self.plot_trace, base, do_png, do_pdf)

        # --- Publication figure ---
        if choices.get("pub_figure"):
            pub_content = str(choices.get("pub_content", "Heatmap + Avg PSTH + Metrics"))
            self._export_publication_figure(out_dir, prefix, pub_content)

        self.statusUpdate.emit(f"Export complete \u2192 {out_dir}", 5000)

    def _export_widget_selective(self, widget: QtWidgets.QWidget, base_path: str,
                                  do_png: bool, do_pdf: bool) -> None:
        if do_png:
            image = self._render_widget_image(widget, transparent=True)
            if image and not image.isNull():
                image.save(f"{base_path}.png", "PNG")
        if do_pdf:
            self._write_widget_pdf(widget, f"{base_path}.pdf")

    def _get_all_behavior_names(self) -> List[str]:
        names: List[str] = []
        for info in self._behavior_sources.values():
            behaviors = info.get("behaviors") or {}
            for beh in behaviors:
                if beh not in names:
                    names.append(beh)
        return names

    def _compute_psth_for_behavior(self, behavior_name: str) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], List[str]]:
        pre = float(self.spin_pre.value())
        post = float(self.spin_post.value())
        b0 = float(self.spin_b0.value())
        b1 = float(self.spin_b1.value())
        res_hz = float(self.spin_resample.value())
        smooth = float(self.spin_smooth.value())
        window = (-pre, post)
        baseline = (b0, b1)
        is_group = self.tab_visual_mode.currentIndex() == 1 and len(self._processed) > 1
        keep_trial_rows = bool(is_group and self._psth_group_trial_view_enabled())
        min_events = self._psth_min_events_per_animal()
        exclude_low_events = bool(is_group and self._psth_exclude_low_event_animals_enabled())
        animal_rows: List[np.ndarray] = []
        animal_labels: List[str] = []
        all_mats: List[np.ndarray] = []
        all_labels: List[str] = []
        tvec = None
        for proc in self._processed:
            info = self._match_behavior_source(proc)
            if not info:
                continue
            on, off, dur = self._extract_behavior_events(info, behavior_name)
            if on.size == 0:
                continue
            align_mode = self.combo_behavior_align.currentText()
            if align_mode.endswith("offset"):
                events = off
            else:
                events = on
            events, dur = self._filter_events(events, dur)
            file_id = os.path.splitext(os.path.basename(proc.path))[0] if proc.path else "import"
            if exclude_low_events and events.size < min_events:
                continue
            tvec, mat = _compute_psth_matrix(self._proc_time(proc), proc.output, events, window, baseline, res_hz, smooth_sigma_s=smooth)
            if mat.size == 0:
                continue
            all_mats.append(mat)
            all_labels.extend([f"{file_id} | Trial {i + 1}" for i in range(mat.shape[0])])
            row = np.nanmean(mat, axis=0)
            if np.any(np.isfinite(row)):
                animal_rows.append(row)
                animal_labels.append(file_id)
        if not all_mats or tvec is None:
            return None, None, []
        if is_group and animal_rows and not keep_trial_rows:
            return np.vstack(animal_rows), tvec, animal_labels
        mat_all = np.vstack(all_mats)
        labels = all_labels[: mat_all.shape[0]] if all_labels else [f"Trial {i + 1}" for i in range(mat_all.shape[0])]
        return mat_all, tvec, labels

    def _export_publication_figure(self, out_dir: str, prefix: str, pub_content: str) -> None:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            from matplotlib.gridspec import GridSpec
            from scipy import stats
        except ImportError as e:
            self.statusUpdate.emit(f"Publication figure requires matplotlib and scipy: {e}", 5000)
            return

        behaviors = self._get_all_behavior_names()
        if not behaviors:
            self.statusUpdate.emit("No behaviors found for publication figure.", 5000)
            return

        show_heat = "Heatmap" in pub_content
        show_avg = "Avg" in pub_content or "PSTH" in pub_content
        show_metrics = "Metrics" in pub_content
        n_cols = int(show_heat) + int(show_avg) + int(show_metrics)
        if n_cols == 0:
            n_cols = 3
            show_heat = show_avg = show_metrics = True
        n_rows = len(behaviors)

        # Publication styling
        plt.rcParams.update({
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "axes.linewidth": 0.8,
            "xtick.major.width": 0.6,
            "ytick.major.width": 0.6,
            "figure.dpi": 300,
        })

        width_ratios = []
        if show_heat:
            width_ratios.append(3)
        if show_avg:
            width_ratios.append(2.5)
        if show_metrics:
            width_ratios.append(1.2)

        fig_w = sum(width_ratios) * 1.8
        fig_h = max(3.5, n_rows * 1.8)
        fig = plt.figure(figsize=(fig_w, fig_h), facecolor="white")
        gs = GridSpec(n_rows, n_cols, figure=fig, width_ratios=width_ratios,
                      hspace=0.45, wspace=0.4, left=0.08, right=0.95, top=0.93, bottom=0.08)

        pre0 = float(self.spin_metric_pre0.value())
        pre1 = float(self.spin_metric_pre1.value())
        post0 = float(self.spin_metric_post0.value())
        post1 = float(self.spin_metric_post1.value())
        metric_name = self.combo_metric.currentText()
        cmap_name = str(self._style.get("heatmap_cmap", "viridis"))

        for row_i, beh_name in enumerate(behaviors):
            mat, tvec, labels = self._compute_psth_for_behavior(beh_name)
            col = 0

            if mat is None or tvec is None or mat.size == 0:
                # Empty row — still show behavior name as title
                titled = False
                if show_heat:
                    ax = fig.add_subplot(gs[row_i, col])
                    ax.set_title(beh_name, fontweight="bold", fontsize=10, loc="left", pad=8)
                    ax.text(0.5, 0.5, "No data", ha="center", va="center",
                            transform=ax.transAxes, fontsize=8, color="#999")
                    ax.set_yticks([]); ax.set_xticks([])
                    titled = True
                    col += 1
                if show_avg:
                    ax = fig.add_subplot(gs[row_i, col])
                    if not titled:
                        ax.set_title(beh_name, fontweight="bold", fontsize=10, loc="left", pad=8)
                        titled = True
                    ax.text(0.5, 0.5, "No data", ha="center", va="center",
                            transform=ax.transAxes, fontsize=8, color="#999")
                    ax.set_yticks([]); ax.set_xticks([])
                    col += 1
                if show_metrics:
                    ax = fig.add_subplot(gs[row_i, col])
                    if not titled:
                        ax.set_title(beh_name, fontweight="bold", fontsize=10, loc="left", pad=8)
                    ax.text(0.5, 0.5, "N/A", ha="center", va="center",
                            transform=ax.transAxes, fontsize=8, color="#999")
                    ax.set_yticks([]); ax.set_xticks([])
                continue

            avg = np.nanmean(mat, axis=0)
            n_valid = max(1, np.sum(np.any(np.isfinite(mat), axis=1)))
            sem = np.nanstd(mat, axis=0) / np.sqrt(n_valid)

            # --- Row title (behavior name) spanning all columns ---
            # Place as a left-aligned text annotation on the first panel
            first_ax_placed = False

            # --- Heatmap ---
            if show_heat:
                ax_heat = fig.add_subplot(gs[row_i, col])
                if not first_ax_placed:
                    ax_heat.set_title(beh_name, fontweight="bold", fontsize=10, loc="left", pad=8)
                    first_ax_placed = True
                extent = [float(tvec[0]), float(tvec[-1]), 0, mat.shape[0]]
                im = ax_heat.imshow(mat, aspect="auto", origin="lower", extent=extent,
                                     cmap=cmap_name, interpolation="nearest")
                ax_heat.axvline(0, color="white", linewidth=0.7, linestyle="--", alpha=0.8)
                if labels:
                    n = min(len(labels), mat.shape[0])
                    tick_pos = [i + 0.5 for i in range(n)]
                    if n > 15:
                        step = max(1, n // 10)
                        tick_pos = tick_pos[::step]
                        tick_labels = labels[::step]
                    else:
                        tick_labels = labels[:n]
                    ax_heat.set_yticks(tick_pos)
                    ax_heat.set_yticklabels(tick_labels, fontsize=6)
                else:
                    ax_heat.set_yticks([])
                if row_i == n_rows - 1:
                    ax_heat.set_xlabel("Time (s)")
                else:
                    ax_heat.set_xticklabels([])
                cbar = fig.colorbar(im, ax=ax_heat, fraction=0.04, pad=0.02)
                cbar.ax.tick_params(labelsize=6)
                col += 1

            # --- Average PSTH ---
            if show_avg:
                ax_avg = fig.add_subplot(gs[row_i, col])
                if not first_ax_placed:
                    ax_avg.set_title(beh_name, fontweight="bold", fontsize=10, loc="left", pad=8)
                    first_ax_placed = True
                elif row_i == 0:
                    ax_avg.set_title("Average PSTH \u00b1 SEM", fontweight="bold")
                ax_avg.fill_between(tvec, avg - sem, avg + sem,
                                     color="#7BC8A4", alpha=0.3, linewidth=0)
                ax_avg.plot(tvec, avg, color="#3B82F6", linewidth=1.2)
                ax_avg.axvline(0, color="#666", linewidth=0.7, linestyle="--", alpha=0.7)
                ax_avg.axhline(0, color="#999", linewidth=0.4, alpha=0.5)
                ax_avg.axvspan(pre0, pre1, color="#5B8CD6", alpha=0.08)
                ax_avg.axvspan(post0, post1, color="#D67B5B", alpha=0.08)
                ax_avg.set_xlim(float(tvec[0]), float(tvec[-1]))
                if row_i == n_rows - 1:
                    ax_avg.set_xlabel("Time (s)")
                else:
                    ax_avg.set_xticklabels([])
                ax_avg.set_ylabel("z-score")
                ax_avg.spines["top"].set_visible(False)
                ax_avg.spines["right"].set_visible(False)
                col += 1

            # --- Metrics bar with t-test ---
            if show_metrics:
                ax_met = fig.add_subplot(gs[row_i, col])
                if not first_ax_placed:
                    ax_met.set_title(beh_name, fontweight="bold", fontsize=10, loc="left", pad=8)
                    first_ax_placed = True
                elif row_i == 0:
                    ax_met.set_title(f"{metric_name} (paired t-test)", fontweight="bold")
                pre_mask = (tvec >= pre0) & (tvec <= pre1)
                post_mask = (tvec >= post0) & (tvec <= post1)
                if np.any(pre_mask) and np.any(post_mask):
                    pre_vals = mat[:, pre_mask]
                    post_vals = mat[:, post_mask]
                    if metric_name == "AUC":
                        dt = float(tvec[1] - tvec[0]) if tvec.size > 1 else 1.0
                        per_row_pre = np.nansum(pre_vals, axis=1) * dt
                        per_row_post = np.nansum(post_vals, axis=1) * dt
                    else:
                        per_row_pre = np.nanmean(pre_vals, axis=1)
                        per_row_post = np.nanmean(post_vals, axis=1)

                    # Filter valid paired data
                    valid = np.isfinite(per_row_pre) & np.isfinite(per_row_post)
                    pre_v = per_row_pre[valid]
                    post_v = per_row_post[valid]
                    n_pairs = int(pre_v.size)

                    mean_pre = float(np.mean(pre_v)) if n_pairs else 0.0
                    mean_post = float(np.mean(post_v)) if n_pairs else 0.0
                    sem_pre = float(np.std(pre_v, ddof=1) / np.sqrt(n_pairs)) if n_pairs > 1 else 0.0
                    sem_post = float(np.std(post_v, ddof=1) / np.sqrt(n_pairs)) if n_pairs > 1 else 0.0

                    colors = ["#5B8CD6", "#D67B5B"]
                    bars = ax_met.bar([0, 1], [mean_pre, mean_post], width=0.55,
                                       color=colors, edgecolor="white", linewidth=0.5, alpha=0.85)
                    ax_met.errorbar([0, 1], [mean_pre, mean_post], yerr=[sem_pre, sem_post],
                                     fmt="none", ecolor="#333", elinewidth=1.0, capsize=3, capthick=0.8)

                    # Paired lines + scatter
                    if n_pairs > 0 and n_pairs <= 50:
                        for j in range(n_pairs):
                            ax_met.plot([0, 1], [pre_v[j], post_v[j]],
                                         color="#888", linewidth=0.4, alpha=0.5, zorder=1)
                        jitter_pre = np.random.default_rng(42).uniform(-0.08, 0.08, n_pairs)
                        jitter_post = np.random.default_rng(43).uniform(-0.08, 0.08, n_pairs)
                        ax_met.scatter(jitter_pre, pre_v, s=12, color="#3B6CB0",
                                        edgecolors="white", linewidths=0.3, zorder=3, alpha=0.8)
                        ax_met.scatter(1 + jitter_post, post_v, s=12, color="#B05B3B",
                                        edgecolors="white", linewidths=0.3, zorder=3, alpha=0.8)

                    # Paired t-test
                    p_val = np.nan
                    if n_pairs >= 2:
                        try:
                            _, p_val = stats.ttest_rel(pre_v, post_v)
                        except Exception:
                            pass

                    # Significance annotation
                    y_max = max(mean_pre + sem_pre, mean_post + sem_post)
                    if np.isfinite(pre_v).any() and np.isfinite(post_v).any():
                        y_max = max(y_max, float(np.nanmax(np.concatenate([pre_v, post_v]))))
                    bar_y = y_max * 1.08
                    if np.isfinite(p_val):
                        if p_val < 0.001:
                            sig_str = "***"
                        elif p_val < 0.01:
                            sig_str = "**"
                        elif p_val < 0.05:
                            sig_str = "*"
                        else:
                            sig_str = "n.s."
                        ax_met.plot([0, 0, 1, 1], [bar_y, bar_y * 1.03, bar_y * 1.03, bar_y],
                                     color="#333", linewidth=0.8)
                        ax_met.text(0.5, bar_y * 1.05, f"{sig_str}\np={p_val:.3g}",
                                     ha="center", va="bottom", fontsize=7, color="#333")
                else:
                    ax_met.text(0.5, 0.5, "N/A", ha="center", va="center",
                                transform=ax_met.transAxes, fontsize=8, color="#999")

                ax_met.set_xticks([0, 1])
                ax_met.set_xticklabels(["Pre", "Post"], fontsize=8)
                ax_met.set_ylabel(metric_name, fontsize=8)
                ax_met.spines["top"].set_visible(False)
                ax_met.spines["right"].set_visible(False)

        # Save
        fig_base = os.path.join(out_dir, f"{prefix}_publication_figure")
        fig.savefig(f"{fig_base}.pdf", format="pdf", bbox_inches="tight", dpi=300)
        fig.savefig(f"{fig_base}.png", format="png", bbox_inches="tight", dpi=300)
        plt.close(fig)
        self.statusUpdate.emit(f"Publication figure saved: {prefix}_publication_figure.pdf/.png", 5000)

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
        if not hasattr(self, "_right_panel"):
            return
        dlg = ExportImageDialog(self)
        if dlg.exec() != QtWidgets.QDialog.DialogCode.Accepted:
            return
        choices = dlg.choices()

        export_targets: List[Tuple[str, QtWidgets.QWidget]] = []
        if choices.get("all", False):
            export_targets.append(("psth_figure", self._right_panel))
        else:
            if choices.get("trace"):
                export_targets.append(("trace", self.plot_trace))
            if choices.get("heat"):
                export_targets.append(("heatmap", self.row_heat))
            if choices.get("avg"):
                export_targets.append(("avg_metrics", self.row_avg))
            if choices.get("signal"):
                export_targets.append(("signal", self.row_signal))
            if choices.get("behavior"):
                export_targets.append(("behavior", self.row_behavior))
            if choices.get("spatial"):
                target_spatial = getattr(self, "spatial_plot_content", None) or getattr(self, "spatial_plot_dialog", None)
                if target_spatial is not None:
                    export_targets.append(("spatial", target_spatial))

        if not export_targets:
            QtWidgets.QMessageBox.information(self, "Export images", "Select at least one panel to export.")
            return

        start_dir = self._export_start_dir()
        prefix = self._default_export_prefix()
        if len(export_targets) == 1:
            suffix, widget = export_targets[0]
            default_path = os.path.join(start_dir, f"{prefix}_{suffix}.png")
            path, _ = QtWidgets.QFileDialog.getSaveFileName(
                self,
                "Export image (PNG + PDF)",
                default_path,
                "PNG image (*.png);;All files (*.*)",
            )
            if not path:
                return
            if not os.path.splitext(path)[1]:
                path = f"{path}.png"
            base_path, _ = os.path.splitext(path)
            out_dir = os.path.dirname(base_path)
            if out_dir:
                self._remember_export_dir(out_dir)
            ok, png_path, pdf_path = self._export_widget_png_pdf(widget, base_path, transparent=True)
            if ok:
                if suffix == "heatmap":
                    self._write_export_parameter_file(base_path, self._collect_psth_parameter_sections(include_heatmap=True))
                elif suffix in {"avg_metrics", "psth_figure"}:
                    self._write_export_parameter_file(base_path, self._collect_psth_parameter_sections(include_heatmap=(suffix == "psth_figure")))
                elif suffix == "spatial":
                    self._write_export_parameter_file(base_path, self._collect_spatial_parameter_sections())
                self.statusUpdate.emit(
                    f"Exported image: {os.path.basename(png_path or '')}, {os.path.basename(pdf_path or '')}",
                    5000,
                )
            else:
                QtWidgets.QMessageBox.warning(self, "Export failed", "Could not export selected panel as PNG/PDF.")
            return

        out_dir = QtWidgets.QFileDialog.getExistingDirectory(self, "Select export folder", start_dir)
        if not out_dir:
            return
        self._remember_export_dir(out_dir)
        ok_count = 0
        failed: List[str] = []
        for suffix, widget in export_targets:
            base_path = os.path.join(out_dir, f"{prefix}_{suffix}")
            ok, _png_path, _pdf_path = self._export_widget_png_pdf(widget, base_path, transparent=True)
            if ok:
                if suffix == "heatmap":
                    self._write_export_parameter_file(base_path, self._collect_psth_parameter_sections(include_heatmap=True))
                elif suffix in {"avg_metrics", "psth_figure"}:
                    self._write_export_parameter_file(base_path, self._collect_psth_parameter_sections(include_heatmap=(suffix == "psth_figure")))
                elif suffix == "spatial":
                    self._write_export_parameter_file(base_path, self._collect_spatial_parameter_sections())
                ok_count += 1
            else:
                failed.append(suffix)
        if failed:
            QtWidgets.QMessageBox.warning(
                self,
                "Export images",
                f"Exported {ok_count}/{len(export_targets)} panel(s).\nFailed: {', '.join(failed)}",
            )
        else:
            self.statusUpdate.emit(f"Exported {ok_count} panel image set(s) (PNG + PDF).", 5000)

    def _export_spatial_figure(self) -> None:
        if self._last_spatial_occupancy_map is None or self._last_spatial_extent is None:
            QtWidgets.QMessageBox.information(self, "Spatial heatmap", "Compute spatial heatmap first.")
            return
        target = getattr(self, "spatial_plot_content", None) or getattr(self, "spatial_plot_dialog", None)
        if target is None:
            return
        start_dir = self._export_start_dir()
        prefix = self._default_export_prefix()
        default_path = os.path.join(start_dir, f"{prefix}_spatial_figure.png")
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self,
            "Export spatial figure (PNG + PDF)",
            default_path,
            "PNG image (*.png);;All files (*.*)",
        )
        if not path:
            return
        if not os.path.splitext(path)[1]:
            path = f"{path}.png"
        base_path, _ = os.path.splitext(path)
        out_dir = os.path.dirname(base_path)
        if out_dir:
            self._remember_export_dir(out_dir)
        ok, png_path, pdf_path = self._export_widget_png_pdf(target, base_path, transparent=True)
        if ok:
            self._write_export_parameter_file(base_path, self._collect_spatial_parameter_sections())
            self.statusUpdate.emit(f"Exported spatial figure: {os.path.basename(png_path or '')}, {os.path.basename(pdf_path or '')}", 5000)
        else:
            QtWidgets.QMessageBox.warning(self, "Export failed", "Could not export spatial figure as PNG/PDF.")

    def hideEvent(self, event: QtGui.QHideEvent) -> None:
        super().hideEvent(event)
        if self._app_closing:
            return
        try:
            if hasattr(self, "spatial_plot_dialog"):
                self.spatial_plot_dialog.hide()
        except Exception:
            pass
        self.hide_section_popups_for_tab_switch()

    def hide_section_popups_for_tab_switch(self) -> None:
        """Hide and detach post-processing docks when tab is inactive."""
        if self._use_pg_dockarea_layout:
            # DockArea lives inside the post tab widget; no main-window dock mutation is needed.
            return
        if not self._section_popups:
            return
        if self._applying_fixed_default_layout:
            return
        if self._force_fixed_default_layout:
            self._suspend_panel_layout_persistence = True
            try:
                for key, dock in self._section_popups.items():
                    dock.hide()
                    self._set_section_button_checked(key, False)
            finally:
                self._suspend_panel_layout_persistence = False
            # Fixed mode does not use cached hidden-state restore.
            self._post_docks_hidden_for_tab_switch = False
            self._post_section_visibility_before_hide.clear()
            self._post_section_state_before_hide.clear()
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
                _dock_area_to_int(host.dockWidgetArea(dock), _dock_area_to_int(QtCore.Qt.DockWidgetArea.LeftDockWidgetArea, 1))
                if host is not None
                else _dock_area_to_int(QtCore.Qt.DockWidgetArea.LeftDockWidgetArea, 1)
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
        if not (
            self.plot_trace.isVisible()
            or self.row_heat.isVisible()
            or self.row_avg.isVisible()
        ):
            self.combo_view_layout.blockSignals(True)
            self.combo_view_layout.setCurrentText("Standard")
            self.combo_view_layout.blockSignals(False)
            self._apply_view_layout()

    def showEvent(self, event: QtGui.QShowEvent) -> None:
        super().showEvent(event)
        if self._use_pg_dockarea_layout:
            self._setup_dockarea_sections()
            if self._force_fixed_default_layout:
                if not self._dock_layout_restored:
                    self._apply_fixed_dockarea_layout()
                else:
                    self._sync_section_button_states_from_docks()
            elif not self._dock_layout_restored:
                self._restore_dockarea_layout_state()
                self._dock_layout_restored = True
            self._ensure_plot_rows_visible()
            return
        self._setup_section_popups()
        if not self._section_popups:
            # Defer until the widget is fully attached to a main-window host.
            QtCore.QTimer.singleShot(0, self._setup_section_popups)
        if self._force_fixed_default_layout and self._section_popups:
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
                    state.get("area", _dock_area_to_int(QtCore.Qt.DockWidgetArea.LeftDockWidgetArea, 1)),
                    QtCore.Qt.DockWidgetArea.LeftDockWidgetArea,
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
        if self._use_pg_dockarea_layout:
            return
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
        try:
            if hasattr(self, "spatial_plot_dialog"):
                self.spatial_plot_dialog.hide()
        except Exception:
            pass
        self._autosave_project_to_cache()
        self.persist_layout_state_snapshot()
        self._save_settings()

    def _enforce_only_post_docks_visible(self) -> None:
        """
        Ensure preprocessing docks cannot stay visible while postprocessing tab is active.
        """
        if self._use_pg_dockarea_layout:
            return
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
        self._apply_fixed_dock_features()
        if self._use_pg_dockarea_layout:
            self._setup_dockarea_sections()
            if self._force_fixed_default_layout:
                self._apply_fixed_dockarea_layout()
            else:
                self._restore_dockarea_layout_state()
            return

    def _schedule_fixed_layout_retry(self) -> None:
        if self._pending_fixed_layout_retry:
            return
        self._pending_fixed_layout_retry = True
        QtCore.QTimer.singleShot(0, self._retry_apply_fixed_default_layout)

    def _retry_apply_fixed_default_layout(self) -> None:
        self._pending_fixed_layout_retry = False
        try:
            self.apply_fixed_default_layout()
        except Exception:
            _LOG.exception("Deferred fixed post layout apply failed")

    def _ensure_fixed_right_tab_widget(self, host: QtWidgets.QMainWindow) -> Optional[QtWidgets.QTabWidget]:
        setup_dock = self._section_popups.get("setup")
        if setup_dock is None:
            return None
        tabw = self._fixed_right_tab_widget
        if tabw is None:
            tabw = QtWidgets.QTabWidget()
            tabw.setObjectName("post.fixed.right.tabs")
            tabw.setDocumentMode(True)
            tabw.setTabPosition(QtWidgets.QTabWidget.TabPosition.South)
            self._fixed_right_tab_widget = tabw
        for key in _FIXED_POST_RIGHT_TAB_ORDER:
            scroll = self._section_scroll_hosts.get(key)
            if scroll is None:
                continue
            idx = tabw.indexOf(scroll)
            title = _FIXED_POST_RIGHT_TAB_TITLES.get(key, key.title())
            if idx < 0:
                idx = tabw.addTab(scroll, title)
            else:
                tabw.setTabText(idx, title)
        if setup_dock.widget() is not tabw:
            setup_dock.setWidget(tabw)
        return tabw

    def _apply_fixed_right_tabs_as_single_dock(self, host: QtWidgets.QMainWindow, active_key: str = "setup") -> None:
        setup_dock = self._section_popups.get("setup")
        if setup_dock is None:
            return
        tabw = self._ensure_fixed_right_tab_widget(host)
        if tabw is None:
            return
        # Setup dock hosts the fixed tabs; the individual PSTH/Spatial docks stay hidden.
        setup_dock.blockSignals(True)
        try:
            host.addDockWidget(QtCore.Qt.DockWidgetArea.LeftDockWidgetArea, setup_dock)
            setup_dock.setFloating(False)
            setup_dock.show()
        finally:
            setup_dock.blockSignals(False)
        for key in ("psth", "spatial", "signal", "behavior"):
            dock = self._section_popups.get(key)
            if dock is None:
                continue
            dock.blockSignals(True)
            try:
                host.addDockWidget(QtCore.Qt.DockWidgetArea.LeftDockWidgetArea, dock)
                dock.setFloating(False)
                dock.hide()
            finally:
                dock.blockSignals(False)
        active = active_key if active_key in _FIXED_POST_RIGHT_SECTIONS else "setup"
        page = self._section_scroll_hosts.get(active)
        idx = tabw.indexOf(page) if page is not None else -1
        if idx >= 0:
            tabw.setCurrentIndex(idx)
        try:
            setup_dock.raise_()
            setup_dock.setFocus(QtCore.Qt.FocusReason.OtherFocusReason)
        except Exception:
            pass
        for key in _FIXED_POST_RIGHT_SECTIONS:
            self._set_section_button_checked(key, True)
        self._last_opened_section = active

    def _activate_fixed_right_tab(self, key: str) -> None:
        host = self._dock_host or self._dock_main_window()
        if host is None:
            return
        self._apply_fixed_right_tabs_as_single_dock(host, active_key=key)

    def _enforce_fixed_post_default_visibility(self) -> None:
        if not self._force_fixed_default_layout:
            return
        if self._use_pg_dockarea_layout:
            self._save_dockarea_layout_state()
            return
        host = self._dock_host or self._dock_main_window()
        if host is None or not self._section_popups:
            return
        tabs = getattr(host, "tabs", None)
        if isinstance(tabs, QtWidgets.QTabWidget) and tabs.currentWidget() is not self:
            return
        visible_keys = {"setup", "export"}
        for key, dock in self._section_popups.items():
            if dock is None:
                continue
            dock.blockSignals(True)
            try:
                area = QtCore.Qt.DockWidgetArea.LeftDockWidgetArea
                host.addDockWidget(area, dock)
                dock.setFloating(False)
                if key in visible_keys:
                    dock.show()
                else:
                    dock.hide()
            finally:
                dock.blockSignals(False)
        self._apply_fixed_right_tabs_as_single_dock(host, active_key=self._last_opened_section or "setup")
        setup = self._section_popups.get("setup")
        if setup is not None:
            try:
                setup.raise_()
                setup.setFocus(QtCore.Qt.FocusReason.OtherFocusReason)
                self._last_opened_section = "setup"
            except Exception:
                pass
        try:
            self._sync_section_button_states_from_docks()
        except Exception:
            pass

    def _persist_fixed_post_default_state(self) -> None:
        if not self._force_fixed_default_layout:
            return
        if self._use_pg_dockarea_layout:
            self._save_dockarea_layout_state()
            return
        left_i = _dock_area_to_int(QtCore.Qt.DockWidgetArea.LeftDockWidgetArea, 1)
        for key in ("setup", "psth", "signal", "behavior", "spatial", "export"):
            dock = self._section_popups.get(key)
            if dock is None:
                continue
            base = f"post_section_docks/{key}"
            visible = key in _FIXED_POST_VISIBLE_SECTIONS
            area_i = left_i
            try:
                self._settings.setValue(f"{base}/visible", visible)
                self._settings.setValue(f"{base}/floating", False)
                self._settings.setValue(f"{base}/area", area_i)
                self._settings.setValue(f"{base}/geometry", dock.saveGeometry())
            except Exception:
                continue
        # Fixed mode should not depend on snapshot restore blobs.
        try:
            self._settings.remove(_POST_DOCK_STATE_KEY)
        except Exception:
            pass
        try:
            self._settings.sync()
        except Exception:
            pass

    def apply_fixed_default_layout(self) -> None:
        """
        Apply deterministic Post Processing docking default:
        Setup, PSTH, Spatial, and Export as fixed left-side tabs.
        """
        if self._use_pg_dockarea_layout:
            self._setup_dockarea_sections()
            if not self._dock_layout_restored:
                self._apply_fixed_dockarea_layout()
            else:
                self._apply_fixed_dock_features()
                self._sync_section_button_states_from_docks()
                self._save_dockarea_layout_state()
            return
        if self._applying_fixed_default_layout:
            return
        self._setup_section_popups()
        host = self._dock_host or self._dock_main_window()
        if host is None or not self._section_popups:
            return
        tabs = getattr(host, "tabs", None)
        if isinstance(tabs, QtWidgets.QTabWidget) and tabs.currentWidget() is not self:
            return
        self._pending_fixed_layout_retry = False
        self._dock_host = host
        self._applying_fixed_default_layout = True

        setup = self._section_popups.get("setup")
        ordered_right_keys = ["setup", "spatial", "psth", "signal", "behavior"]
        visible_right_keys = {"setup"}
        export = self._section_popups.get("export")

        self._suspend_panel_layout_persistence = True
        try:
            self._apply_fixed_dock_features()
            # Reset post dock topology first so stale tab groups from previous
            # sessions cannot override the enforced default.
            for key in ("setup", "psth", "signal", "behavior", "spatial", "export"):
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
                    host.addDockWidget(QtCore.Qt.DockWidgetArea.LeftDockWidgetArea, dock)
                    dock.setFloating(False)
                    if key in visible_right_keys:
                        dock.show()
                    else:
                        dock.hide()
                finally:
                    dock.blockSignals(False)

            if export is not None:
                export.blockSignals(True)
                try:
                    host.addDockWidget(QtCore.Qt.DockWidgetArea.LeftDockWidgetArea, export)
                    export.setFloating(False)
                    export.show()
                finally:
                    export.blockSignals(False)

            self._apply_fixed_right_tabs_as_single_dock(host, active_key=self._last_opened_section or "setup")

            if setup is not None:
                setup.show()
                setup.raise_()
                setup.setFocus(QtCore.Qt.FocusReason.OtherFocusReason)
                self._last_opened_section = "setup"
            if export is not None:
                export.raise_()

            # Final hard enforcement: all post docks must be docked (non-floating).
            for key in ("setup", "psth", "signal", "behavior", "spatial", "export"):
                dock = self._section_popups.get(key)
                if dock is None:
                    continue
                try:
                    if dock.isFloating():
                        dock.setFloating(False)
                except Exception:
                    pass

            self._sync_section_button_states_from_docks()
            self._post_docks_hidden_for_tab_switch = False
            self._post_section_visibility_before_hide.clear()
            self._post_section_state_before_hide.clear()
            self._dock_layout_restored = True
        finally:
            self._suspend_panel_layout_persistence = False
            self._applying_fixed_default_layout = False

        # Re-apply once after queued dock events for extra stability.
        QtCore.QTimer.singleShot(0, self._enforce_fixed_post_default_visibility)
        self._persist_fixed_post_default_state()
        self._enforce_only_post_docks_visible()

    def ensure_section_popups_initialized(self) -> None:
        if self._use_pg_dockarea_layout:
            self._setup_dockarea_sections()
            return
        self._setup_section_popups()

    def get_section_dock_widgets(self) -> List[QtWidgets.QDockWidget]:
        if self._use_pg_dockarea_layout:
            self._setup_dockarea_sections()
            return []
        self._setup_section_popups()
        return list(self._section_popups.values())

    def get_section_popup_keys(self) -> List[str]:
        if self._use_pg_dockarea_layout:
            self._setup_dockarea_sections()
            return list(self._dockarea_docks.keys()) or list(self._section_widget_map().keys())
        self._setup_section_popups()
        return list(self._section_popups.keys())

    def mark_dock_layout_restored(self) -> None:
        self._dock_layout_restored = True


class ExportDialog(QtWidgets.QDialog):
    def __init__(self, parent=None, group_labels: Optional[List[str]] = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Export Results")
        self.setModal(True)
        self.setMinimumWidth(420)
        layout = QtWidgets.QVBoxLayout(self)
        layout.setSpacing(10)

        # --- Data exports ---
        grp_data = QtWidgets.QGroupBox("Data to export")
        data_layout = QtWidgets.QVBoxLayout(grp_data)
        self.cb_heatmap = QtWidgets.QCheckBox("Heatmap matrix")
        self.cb_heatmap_aligned = QtWidgets.QCheckBox("Heatmap aligned (time-locked matrix)")
        self.cb_avg = QtWidgets.QCheckBox("Average PSTH")
        self.cb_events = QtWidgets.QCheckBox("Event times")
        self.cb_durations = QtWidgets.QCheckBox("Event durations")
        self.cb_metrics = QtWidgets.QCheckBox("Metrics table")
        for cb in (self.cb_heatmap, self.cb_heatmap_aligned, self.cb_avg,
                   self.cb_events, self.cb_durations, self.cb_metrics):
            cb.setChecked(True)
            data_layout.addWidget(cb)
        self.cb_heatmap_aligned.setChecked(False)
        layout.addWidget(grp_data)

        # --- Format ---
        grp_fmt = QtWidgets.QGroupBox("Export format")
        fmt_layout = QtWidgets.QHBoxLayout(grp_fmt)
        self.cb_csv = QtWidgets.QCheckBox("CSV")
        self.cb_csv.setChecked(True)
        self.cb_h5 = QtWidgets.QCheckBox("HDF5 (.h5)")
        self.cb_h5.setChecked(False)
        fmt_layout.addWidget(self.cb_csv)
        fmt_layout.addWidget(self.cb_h5)
        fmt_layout.addStretch(1)
        layout.addWidget(grp_fmt)

        # --- Plot exports ---
        grp_plots = QtWidgets.QGroupBox("Plot exports")
        plot_layout = QtWidgets.QVBoxLayout(grp_plots)
        fmt_row = QtWidgets.QHBoxLayout()
        self.cb_png = QtWidgets.QCheckBox("PNG")
        self.cb_png.setChecked(True)
        self.cb_pdf = QtWidgets.QCheckBox("PDF")
        self.cb_pdf.setChecked(True)
        fmt_row.addWidget(self.cb_png)
        fmt_row.addWidget(self.cb_pdf)
        fmt_row.addStretch(1)
        plot_layout.addLayout(fmt_row)
        self.cb_plot_heatmap = QtWidgets.QCheckBox("Heatmap + durations")
        self.cb_plot_avg = QtWidgets.QCheckBox("Average PSTH + metrics")
        self.cb_plot_trace = QtWidgets.QCheckBox("Trace preview")
        for cb in (self.cb_plot_heatmap, self.cb_plot_avg, self.cb_plot_trace):
            cb.setChecked(False)
            plot_layout.addWidget(cb)
        layout.addWidget(grp_plots)

        # --- Publication figure ---
        grp_pub = QtWidgets.QGroupBox("Publication figure")
        pub_layout = QtWidgets.QVBoxLayout(grp_pub)
        self.cb_pub_figure = QtWidgets.QCheckBox("Multi-panel figure (all behaviors)")
        self.cb_pub_figure.setChecked(False)
        self.cb_pub_figure.setToolTip(
            "Export a publication-ready figure with one row per behavior:\n"
            "heatmap | average PSTH | pre/post metrics with paired t-test p-value"
        )
        pub_layout.addWidget(self.cb_pub_figure)
        self.combo_pub_content = QtWidgets.QComboBox()
        self.combo_pub_content.addItems(["Heatmap + Avg PSTH + Metrics", "Heatmap + Metrics", "Avg PSTH + Metrics"])
        _compact_combo(self.combo_pub_content, min_chars=12)
        pub_content_row = QtWidgets.QHBoxLayout()
        pub_content_row.addWidget(QtWidgets.QLabel("Panels:"))
        pub_content_row.addWidget(self.combo_pub_content, stretch=1)
        pub_layout.addLayout(pub_content_row)
        layout.addWidget(grp_pub)

        # Show group info if applicable
        if group_labels and len(group_labels) > 1:
            info = QtWidgets.QLabel(f"Group mode: {len(group_labels)} animals — columns will be labeled by animal ID")
            info.setProperty("class", "hint")
            info.setStyleSheet("color: #5abeFF; font-style: italic; padding: 4px 0;")
            layout.addWidget(info)

        # --- Buttons ---
        row = QtWidgets.QHBoxLayout()
        row.addStretch(1)
        btn_ok = QtWidgets.QPushButton("Export")
        btn_ok.setProperty("class", "compactPrimary")
        btn_cancel = QtWidgets.QPushButton("Cancel")
        btn_ok.setDefault(True)
        row.addWidget(btn_ok)
        row.addWidget(btn_cancel)
        layout.addLayout(row)

        btn_ok.clicked.connect(self.accept)
        btn_cancel.clicked.connect(self.reject)

    def choices(self) -> Dict[str, object]:
        return {
            "heatmap": self.cb_heatmap.isChecked(),
            "heatmap_aligned": self.cb_heatmap_aligned.isChecked(),
            "avg": self.cb_avg.isChecked(),
            "events": self.cb_events.isChecked(),
            "durations": self.cb_durations.isChecked(),
            "metrics": self.cb_metrics.isChecked(),
            "csv": self.cb_csv.isChecked(),
            "h5": self.cb_h5.isChecked(),
            "png": self.cb_png.isChecked(),
            "pdf": self.cb_pdf.isChecked(),
            "plot_heatmap": self.cb_plot_heatmap.isChecked(),
            "plot_avg": self.cb_plot_avg.isChecked(),
            "plot_trace": self.cb_plot_trace.isChecked(),
            "pub_figure": self.cb_pub_figure.isChecked(),
            "pub_content": self.combo_pub_content.currentText(),
        }


class ExportImageDialog(QtWidgets.QDialog):
    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Export Images")
        self.setModal(True)
        layout = QtWidgets.QVBoxLayout(self)

        self.cb_all = QtWidgets.QCheckBox("All panels")
        self.cb_trace = QtWidgets.QCheckBox("Trace preview")
        self.cb_heat = QtWidgets.QCheckBox("Heatmap + durations")
        self.cb_avg = QtWidgets.QCheckBox("Average + metrics")
        self.cb_signal = QtWidgets.QCheckBox("Signal analyzer")
        self.cb_behavior = QtWidgets.QCheckBox("Behavior analysis")
        self.cb_spatial = QtWidgets.QCheckBox("Spatial window")

        self.cb_all.setChecked(True)
        for cb in (self.cb_all, self.cb_trace, self.cb_heat, self.cb_avg, self.cb_signal, self.cb_behavior, self.cb_spatial):
            layout.addWidget(cb)
        self._set_individual_enabled(self.cb_all.isChecked())

        row = QtWidgets.QHBoxLayout()
        row.addStretch(1)
        btn_ok = QtWidgets.QPushButton("OK")
        btn_cancel = QtWidgets.QPushButton("Cancel")
        btn_ok.setDefault(True)
        row.addWidget(btn_ok)
        row.addWidget(btn_cancel)
        layout.addLayout(row)

        self.cb_all.toggled.connect(self._set_individual_enabled)
        btn_ok.clicked.connect(self.accept)
        btn_cancel.clicked.connect(self.reject)

    def _set_individual_enabled(self, all_checked: bool) -> None:
        enabled = not bool(all_checked)
        for cb in (self.cb_trace, self.cb_heat, self.cb_avg, self.cb_signal, self.cb_behavior, self.cb_spatial):
            cb.setEnabled(enabled)

    def choices(self) -> Dict[str, bool]:
        return {
            "all": self.cb_all.isChecked(),
            "trace": self.cb_trace.isChecked(),
            "heat": self.cb_heat.isChecked(),
            "avg": self.cb_avg.isChecked(),
            "signal": self.cb_signal.isChecked(),
            "behavior": self.cb_behavior.isChecked(),
            "spatial": self.cb_spatial.isChecked(),
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
        self.btn_sem_edge = QtWidgets.QPushButton("Pick")
        self.btn_sem_fill = QtWidgets.QPushButton("Pick")
        self.btn_plot_bg = QtWidgets.QPushButton("Pick")
        self.cb_grid = QtWidgets.QCheckBox("Show grid on plots")
        self.cb_grid.setChecked(bool(self._style.get("grid_enabled", True)))
        self.spin_grid_alpha = QtWidgets.QDoubleSpinBox()
        self.spin_grid_alpha.setRange(0.0, 1.0)
        self.spin_grid_alpha.setSingleStep(0.05)
        self.spin_grid_alpha.setDecimals(2)
        try:
            self.spin_grid_alpha.setValue(float(self._style.get("grid_alpha", 0.25)))
        except Exception:
            self.spin_grid_alpha.setValue(0.25)
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
        layout.addRow("SEM edge color", self.btn_sem_edge)
        layout.addRow("SEM fill color", self.btn_sem_fill)
        layout.addRow("Plot background", self.btn_plot_bg)
        layout.addRow("Grid", self.cb_grid)
        layout.addRow("Grid alpha", self.spin_grid_alpha)
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
        self.btn_sem_edge.clicked.connect(lambda *_: self._pick_color("sem_edge"))
        self.btn_sem_fill.clicked.connect(lambda *_: self._pick_color("sem_fill", with_alpha=True))
        self.btn_plot_bg.clicked.connect(lambda *_: self._pick_color("plot_bg"))
        btn_ok.clicked.connect(self.accept)
        btn_cancel.clicked.connect(self.reject)

    def _pick_color(self, key: str, with_alpha: bool = False) -> None:
        current = self._style.get(key, (255, 255, 255, 255) if with_alpha else (255, 255, 255))
        if isinstance(current, np.ndarray):
            current_vals = current.tolist()
        elif isinstance(current, (list, tuple)):
            current_vals = list(current)
        else:
            current_vals = [255, 255, 255, 255] if with_alpha else [255, 255, 255]
        clean_vals: List[int] = []
        for idx, default in enumerate(([255, 255, 255, 255] if with_alpha else [255, 255, 255])):
            try:
                raw = current_vals[idx] if idx < len(current_vals) else default
                val = int(float(raw))
            except Exception:
                val = int(default)
            clean_vals.append(max(0, min(255, val)))
        if with_alpha:
            qcol = QtGui.QColor(clean_vals[0], clean_vals[1], clean_vals[2], clean_vals[3])
        else:
            qcol = QtGui.QColor(clean_vals[0], clean_vals[1], clean_vals[2])
        dlg = QtWidgets.QColorDialog(self)
        dlg.setWindowTitle("Select color")
        dlg.setCurrentColor(qcol)
        dlg.setOption(QtWidgets.QColorDialog.ColorDialogOption.DontUseNativeDialog, True)
        if with_alpha:
            dlg.setOption(QtWidgets.QColorDialog.ColorDialogOption.ShowAlphaChannel, True)
        if dlg.exec() != QtWidgets.QDialog.DialogCode.Accepted:
            return
        col = dlg.selectedColor()
        if not col.isValid():
            return
        if with_alpha:
            self._style[key] = (col.red(), col.green(), col.blue(), col.alpha())
        else:
            self._style[key] = (col.red(), col.green(), col.blue())

    def get_style(self) -> Dict[str, object]:
        self._style["heatmap_cmap"] = self.combo_cmap.currentText()
        manual_levels = self.spin_hmin.value() != 0.0 or self.spin_hmax.value() != 0.0
        self._style["heatmap_min"] = float(self.spin_hmin.value()) if manual_levels else None
        self._style["heatmap_max"] = float(self.spin_hmax.value()) if manual_levels else None
        self._style["heatmap_levels_manual"] = bool(manual_levels)
        self._style["grid_enabled"] = bool(self.cb_grid.isChecked())
        self._style["grid_alpha"] = float(self.spin_grid_alpha.value())
        self._style.setdefault("plot_bg", (36, 42, 52))
        self._style.setdefault("sem_edge", (152, 201, 143))
        self._style.setdefault("sem_fill", (188, 230, 178, 96))
        return dict(self._style)
