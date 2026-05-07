# main.py
"""
Fiber Photometry Processor (Doric .doric) - PySide6 + pyqtgraph

Run:
    python main.py

Dependencies:
    pip install PySide6 pyqtgraph h5py numpy scipy scikit-learn pybaselines
"""

from __future__ import annotations

import os
import json
import logging
import sys
from typing import Callable, Dict, List, Optional, Tuple


_DLL_DIR_HANDLES = []


def _bootstrap_windows_conda_runtime() -> None:
    if os.name != "nt":
        return

    os.environ.setdefault("PYTHONNOUSERSITE", "1")

    try:
        import site
        user_site = os.path.normcase(os.path.abspath(site.getusersitepackages()))
    except Exception:
        user_site = ""

    appdata_python = ""
    appdata = os.environ.get("APPDATA", "")
    if appdata:
        appdata_python = os.path.normcase(os.path.abspath(os.path.join(appdata, "Python")))

    def _is_user_site_path(path: str) -> bool:
        if not path:
            return False
        try:
            norm = os.path.normcase(os.path.abspath(path))
        except Exception:
            return False
        if user_site and (norm == user_site or norm.startswith(user_site + os.sep)):
            return True
        return bool(appdata_python and (norm == appdata_python or norm.startswith(appdata_python + os.sep)))

    sys.path[:] = [path for path in sys.path if not _is_user_site_path(path)]
    script_dir = os.path.dirname(os.path.abspath(__file__))
    if script_dir and script_dir not in sys.path:
        sys.path.insert(0, script_dir)

    prefix = os.environ.get("CONDA_PREFIX") or sys.prefix
    dll_dirs = [
        prefix,
        os.path.join(prefix, "Library", "mingw-w64", "bin"),
        os.path.join(prefix, "Library", "usr", "bin"),
        os.path.join(prefix, "Library", "bin"),
        os.path.join(prefix, "Scripts"),
    ]
    existing = [path for path in dll_dirs if path and os.path.isdir(path)]

    if hasattr(os, "add_dll_directory"):
        for path in existing:
            try:
                _DLL_DIR_HANDLES.append(os.add_dll_directory(path))
            except Exception:
                pass

    old_path = os.environ.get("PATH", "")
    old_parts = [os.path.normcase(os.path.abspath(p)) for p in old_path.split(os.pathsep) if p]
    prepend = [p for p in existing if os.path.normcase(os.path.abspath(p)) not in old_parts]
    if prepend:
        os.environ["PATH"] = os.pathsep.join(prepend + [old_path])


_bootstrap_windows_conda_runtime()

from PySide6 import QtCore, QtGui, QtWidgets
import pyqtgraph as pg
from pyqtgraph.dockarea import DockArea, Dock
import h5py

from analysis_core import (
    ExportSelection,
    OUTPUT_MODES,
    PhotometryProcessor,
    ProcessingParams,
    LoadedDoricFile,
    LoadedTrial,
    ProcessedTrial,
    export_processed_csv,
    export_processed_h5,
    safe_stem_from_metadata,
    detect_artifacts_adaptive,
    interpolate_nans,
    zscore_median_std,
    safe_divide,
    _lowpass_sos,
    coerce_time_value,
)
from gui_preprocessing import (
    FileQueuePanel,
    ParameterPanel,
    PlotDashboard,
    MetadataDialog,
    ArtifactPanel,
    AdvancedOptionsDialog,
)
from gui_postprocessing import PostProcessingPanel
from numeric_controls import install_spinbox_scrubbers
from styles import (
    apply_app_palette,
    app_qss,
    _make_icon,
    _paint_database,
    _paint_list,
    _paint_sliders,
    _paint_filter,
    _paint_wave,
    _paint_chart,
    _paint_badge,
    _paint_export,
    _paint_gear,
)
import numpy as np


# Icon painters now live in styles.py and are imported above.


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


_DOCK_STATE_VERSION = 3
# Dock snapshot keys are versioned because object names changed from pre_/post_
# to pre./post. and old blobs are incompatible with restoreState.
_PRE_DOCK_STATE_KEY = "pre_main_dock_state_v4"
_POST_DOCK_STATE_KEY = "post_main_dock_state_v4"
_PRE_TAB_GROUPS_KEY = "pre_tab_groups_v1"
_PRE_DOCKAREA_STATE_KEY = "pre_dockarea_state_v1"
_PRE_DOCKAREA_VISIBLE_KEY = "pre_dockarea_visible_v1"
_PRE_DOCKAREA_ACTIVE_KEY = "pre_dockarea_active_v1"
_PRE_DOCK_PREFIX = "pre."
_POST_DOCK_PREFIX = "post."
_FORCE_FIXED_DOCK_LAYOUTS = False
_USE_PG_DOCKAREA_PRE_LAYOUT = True
_PRE_DOCKAREA_PRIMARY_ORDER = ("artifacts_list", "artifacts", "filtering", "baseline", "output", "export")
_PRE_DOCKAREA_OPTIONAL_ORDER = ("qc", "config")
_PRE_DOCKAREA_DEFAULT_VISIBLE = frozenset(_PRE_DOCKAREA_PRIMARY_ORDER)
_CSV_NONE_LABEL = "(none)"
_PRE_PROJECT_TYPE = "pyber_preprocessing_project"
_PRE_PROJECT_VERSION = 1

_LOG = logging.getLogger(__name__)


def _pyber_icon_path() -> str:
    if getattr(sys, "frozen", False):
        base_dir = str(getattr(sys, "_MEIPASS", "")) or os.path.dirname(sys.executable)
        candidates = [
            os.path.join(base_dir, "assets", "pyBer_logo_big.png"),
            os.path.join(os.path.dirname(sys.executable), "assets", "pyBer_logo_big.png"),
        ]
    else:
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        candidates = [os.path.join(base_dir, "assets", "pyBer_logo_big.png")]

    for path in candidates:
        if os.path.isfile(path):
            return path
    return candidates[0]


def _rolling_corr(x: np.ndarray, y: np.ndarray, win: int) -> Tuple[np.ndarray, np.ndarray]:
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    if win <= 10 or x.size < win or y.size < win:
        return np.array([], float), np.array([], int)
    step = max(10, win // 2)
    rs = []
    centers = []
    for i in range(0, x.size - win + 1, step):
        xx = x[i:i + win]
        yy = y[i:i + win]
        m = np.isfinite(xx) & np.isfinite(yy)
        if np.sum(m) < 10:
            r = np.nan
        else:
            r = float(np.corrcoef(xx[m], yy[m])[0, 1])
        rs.append(r)
        centers.append(i + win // 2)
    return np.asarray(rs, float), np.asarray(centers, int)


class QcDialog(QtWidgets.QDialog):
    def __init__(self, qc: Dict[str, object], parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Quality Check (z-score)")
        self.resize(1100, 800)
        self._qc = qc
        layout = QtWidgets.QVBoxLayout(self)

        self.plot_z = pg.PlotWidget(title="z_sig / z_ref")
        self.plot_corr = pg.PlotWidget(title="Correlation (z_ref vs z_sig)")
        self.plot_zdist = pg.PlotWidget(title="Z distribution")
        self.plot_roll = pg.PlotWidget(title="Rolling corr(z_ref, z_sig)")
        for w in (self.plot_z, self.plot_corr, self.plot_zdist, self.plot_roll):
            w.showGrid(x=True, y=True, alpha=0.25)
            w.showAxis("top", False)
            w.showAxis("right", False)

        self.plot_z.plot(qc["t"], qc["z_sig"], pen=pg.mkPen((90, 190, 255), width=1.0), name="z_sig")
        self.plot_z.plot(qc["t"], qc["z_ref"], pen=pg.mkPen((240, 180, 80), width=1.0), name="z_ref")
        self.plot_z.setLabel("left", "z units")

        # Z distribution
        Zf = qc["Zf"]
        if Zf.size:
            hist, edges = np.histogram(Zf, bins=80)
            bg = pg.BarGraphItem(x=edges[:-1], height=hist, width=np.diff(edges), brush=pg.mkBrush(90, 143, 214, 80))
            self.plot_zdist.addItem(bg)
            q25 = float(qc.get("q25", np.nan))
            q50 = float(qc.get("q50", np.nan))
            q75 = float(qc.get("q75", np.nan))
            if np.isfinite(q25) and np.isfinite(q75):
                region = pg.LinearRegionItem(values=(q25, q75), brush=(90, 143, 214, 50), movable=False)
                self.plot_zdist.addItem(region)
            if np.isfinite(q50):
                self.plot_zdist.addItem(pg.InfiniteLine(pos=q50, angle=90, pen=pg.mkPen((220, 220, 220), width=1.0)))
            iqr = float(qc.get("iqr", np.nan))
            if np.isfinite(q50) and np.isfinite(iqr):
                self._add_plot_text_topleft(self.plot_zdist, f"median={q50:.3g}  IQR={iqr:.3g}")
        self.plot_zdist.setLabel("left", "count")

        # Correlation scatter + fit
        z_ref = qc["z_ref"]
        z_sig = qc["z_sig"]
        m = np.isfinite(z_ref) & np.isfinite(z_sig)
        if np.sum(m) >= 10:
            self.plot_corr.plot(z_ref[m], z_sig[m], pen=None, symbol="o", symbolSize=4, symbolBrush=(120, 180, 220, 80))
            a, b = np.polyfit(z_ref[m], z_sig[m], 1)
            xs = np.linspace(np.nanmin(z_ref[m]), np.nanmax(z_ref[m]), 200)
            self.plot_corr.plot(xs, a * xs + b, pen=pg.mkPen((220, 120, 120), width=1.2))
            r = float(qc.get("r", np.nan))
            r2 = r * r if np.isfinite(r) else np.nan
            if np.isfinite(r):
                self._add_plot_text_topleft(self.plot_corr, f"r={r:.3g}  r2={r2:.3g}")
        self.plot_corr.setLabel("left", "z_sig")
        self.plot_corr.setLabel("bottom", "z_ref")

        # Rolling corr
        if qc["r_roll"].size:
            t_cent = qc["t"][qc["r_centers"]]
            self.plot_roll.plot(t_cent, qc["r_roll"], pen=pg.mkPen((180, 200, 120), width=1.0))
            self.plot_roll.addItem(pg.InfiniteLine(pos=0.5, angle=0, pen=pg.mkPen((200, 200, 200), width=1.0, style=QtCore.Qt.PenStyle.DashLine)))
            r_avg = float(np.nanmean(qc["r_roll"])) if qc["r_roll"].size else np.nan
            if np.isfinite(r_avg):
                self._add_plot_text_topleft(self.plot_roll, f"avg r={r_avg:.3g}")
        self.plot_roll.setLabel("left", "r")

        stats = qc["stats"]
        self.lbl_stats = QtWidgets.QLabel(stats)
        self.lbl_stats.setProperty("class", "hint")

        btn_row = QtWidgets.QHBoxLayout()
        btn_row.addStretch(1)
        self.btn_save = QtWidgets.QPushButton("Save report images")
        self.btn_close = QtWidgets.QPushButton("Close")
        btn_row.addWidget(self.btn_save)
        btn_row.addWidget(self.btn_close)

        layout.addWidget(self.plot_z, stretch=2)
        row = QtWidgets.QHBoxLayout()
        row.addWidget(self.plot_corr, stretch=1)
        row.addWidget(self.plot_zdist, stretch=1)
        layout.addLayout(row, stretch=2)
        layout.addWidget(self.plot_roll, stretch=1)
        layout.addWidget(self.lbl_stats)
        layout.addLayout(btn_row)

        self.btn_close.clicked.connect(self.close)
        self.btn_save.clicked.connect(self._save_images)

    def _add_plot_text_topleft(self, plot: pg.PlotWidget, text: str) -> None:
        if not text:
            return
        vb = plot.getViewBox()
        if not vb:
            return
        (x0, x1), (y0, y1) = vb.viewRange()
        if not np.isfinite(x0) or not np.isfinite(y1):
            return
        pad_x = (x1 - x0) * 0.02
        pad_y = (y1 - y0) * 0.05
        item = pg.TextItem(text, color=(220, 220, 220), anchor=(0, 1))
        item.setPos(x0 + pad_x, y1 - pad_y)
        plot.addItem(item)

    def _save_images(self) -> None:
        self.save_report()

    def save_report(self, out_dir: Optional[str] = None) -> None:
        path = self._qc.get("path", "")
        channel = self._qc.get("channel", "")
        stem = os.path.splitext(os.path.basename(path))[0] if path else "quality"
        if channel:
            stem = f"{stem}_{channel}"
        out_dir = out_dir or (os.path.dirname(path) if path else os.getcwd())
        img_path = os.path.join(out_dir, f"{stem}_quality.png")
        txt_path = os.path.join(out_dir, f"{stem}_quality.txt")
        try:
            pix = self.grab()
            pix.save(img_path)
        except Exception:
            pass
        try:
            with open(txt_path, "w") as f:
                f.write(str(self._qc.get("stats", "")))
        except Exception:
            pass


class CsvChannelMappingDialog(QtWidgets.QDialog):
    def __init__(
        self,
        headers: List[str],
        numeric_headers: List[str],
        defaults: Optional[Dict[str, object]] = None,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("CSV channel mapping")
        self.setModal(True)
        self.resize(520, 260)
        self._headers = list(headers or [])
        self._numeric_headers = list(numeric_headers or [])
        self._defaults = defaults or {}

        layout = QtWidgets.QVBoxLayout(self)
        hint = QtWidgets.QLabel(
            "Choose how this CSV maps to preprocessing inputs. The same column names will be reused for CSV files in this session."
        )
        hint.setWordWrap(True)
        hint.setProperty("class", "hint")
        layout.addWidget(hint)

        form = QtWidgets.QFormLayout()
        self.combo_time = QtWidgets.QComboBox()
        self.combo_time_unit = QtWidgets.QComboBox()
        self.combo_raw1 = QtWidgets.QComboBox()
        self.combo_raw2 = QtWidgets.QComboBox()
        self.combo_ref = QtWidgets.QComboBox()
        self.combo_trigger = QtWidgets.QComboBox()

        self.combo_time.addItems(self._headers)
        self.combo_time_unit.addItems(["Auto", "Seconds", "Milliseconds"])
        self.combo_raw1.addItems(self._numeric_headers)
        self.combo_raw2.addItem(_CSV_NONE_LABEL)
        self.combo_raw2.addItems(self._numeric_headers)
        self.combo_ref.addItems(self._numeric_headers)
        self.combo_trigger.addItem(_CSV_NONE_LABEL)
        self.combo_trigger.addItems(self._numeric_headers)

        form.addRow("Time column", self.combo_time)
        form.addRow("Time unit", self.combo_time_unit)
        form.addRow("Raw signal 1", self.combo_raw1)
        form.addRow("Raw signal 2 (optional)", self.combo_raw2)
        form.addRow("Isobestic / reference", self.combo_ref)
        form.addRow("Event / DIO (optional)", self.combo_trigger)
        layout.addLayout(form)

        btn_row = QtWidgets.QHBoxLayout()
        btn_row.addStretch(1)
        self.btn_ok = QtWidgets.QPushButton("OK")
        self.btn_cancel = QtWidgets.QPushButton("Cancel")
        self.btn_ok.setDefault(True)
        btn_row.addWidget(self.btn_ok)
        btn_row.addWidget(self.btn_cancel)
        layout.addLayout(btn_row)

        self.btn_ok.clicked.connect(self._accept_if_valid)
        self.btn_cancel.clicked.connect(self.reject)
        self._apply_defaults()

    def _set_combo_text(self, combo: QtWidgets.QComboBox, value: object) -> None:
        text = str(value or "").strip()
        if not text:
            return
        idx = combo.findText(text, QtCore.Qt.MatchFlag.MatchFixedString)
        if idx >= 0:
            combo.setCurrentIndex(idx)

    def _apply_defaults(self) -> None:
        self._set_combo_text(self.combo_time, self._defaults.get("time"))
        self._set_combo_text(self.combo_time_unit, self._defaults.get("time_unit") or "Auto")
        self._set_combo_text(self.combo_raw1, self._defaults.get("raw1"))
        self._set_combo_text(self.combo_raw2, self._defaults.get("raw2") or _CSV_NONE_LABEL)
        self._set_combo_text(self.combo_ref, self._defaults.get("reference"))
        self._set_combo_text(self.combo_trigger, self._defaults.get("trigger") or _CSV_NONE_LABEL)

    def mapping(self) -> Dict[str, str]:
        raw2 = self.combo_raw2.currentText().strip()
        trigger = self.combo_trigger.currentText().strip()
        return {
            "time": self.combo_time.currentText().strip(),
            "time_unit": self.combo_time_unit.currentText().strip() or "Auto",
            "raw1": self.combo_raw1.currentText().strip(),
            "raw2": "" if raw2 == _CSV_NONE_LABEL else raw2,
            "reference": self.combo_ref.currentText().strip(),
            "trigger": "" if trigger == _CSV_NONE_LABEL else trigger,
        }

    def _accept_if_valid(self) -> None:
        m = self.mapping()
        raw1 = m.get("raw1", "")
        raw2 = m.get("raw2", "")
        ref = m.get("reference", "")
        if not m.get("time") or not raw1 or not ref:
            QtWidgets.QMessageBox.warning(self, "CSV mapping", "Choose a time column, raw signal 1, and isobestic/reference column.")
            return
        if raw1 == ref:
            QtWidgets.QMessageBox.warning(self, "CSV mapping", "Raw signal 1 and isobestic/reference must use different columns.")
            return
        if raw2 and raw2 in {raw1, ref}:
            QtWidgets.QMessageBox.warning(self, "CSV mapping", "Raw signal 2 must use a different column.")
            return
        self.accept()

class MainWindow(QtWidgets.QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Pyber - Fiber Photometry")
        try:
            icon_path = _pyber_icon_path()
            if os.path.isfile(icon_path):
                icon = QtGui.QIcon(icon_path)
                if not icon.isNull():
                    self.setWindowIcon(icon)
        except Exception:
            pass
        self._set_initial_window_size()
        self.setDockOptions(
            QtWidgets.QMainWindow.DockOption.AllowNestedDocks
            | QtWidgets.QMainWindow.DockOption.AllowTabbedDocks
            | QtWidgets.QMainWindow.DockOption.AnimatedDocks
        )
        self.setDockNestingEnabled(True)

        # Core
        self.processor = PhotometryProcessor()

        # State
        self._loaded_files: Dict[str, LoadedDoricFile] = {}
        self._current_path: Optional[str] = None
        self._current_channel: Optional[str] = None
        self._current_trigger: Optional[str] = None
        self._pre_project_path: Optional[str] = None
        self._csv_channel_mapping_session: Optional[Dict[str, str]] = None
        self._csv_mappings_by_path: Dict[str, Dict[str, str]] = {}

        self._manual_regions_by_key: Dict[Tuple[str, str], List[Tuple[float, float]]] = {}
        self._manual_exclude_by_key: Dict[Tuple[str, str], List[Tuple[float, float]]] = {}
        self._auto_regions_by_key: Dict[Tuple[str, str], List[Tuple[float, float]]] = {}
        self._metadata_by_key: Dict[Tuple[str, str], Dict[str, str]] = {}
        self._cutout_regions_by_key: Dict[Tuple[str, str], List[Tuple[float, float]]] = {}
        self._sections_by_key: Dict[Tuple[str, str], List[Dict[str, object]]] = {}
        self._pending_box_region_by_key: Dict[Tuple[str, str], Tuple[float, float]] = {}

        self._last_processed: Dict[Tuple[str, str], ProcessedTrial] = {}
        self._advanced_dialog: Optional[AdvancedOptionsDialog] = None
        self._box_select_callback: Optional[Callable[[float, float], None]] = None
        self._last_artifact_params: Optional[Tuple[object, ...]] = None
        self._section_docks: Dict[str, QtWidgets.QDockWidget] = {}
        self._use_pg_dockarea_pre_layout: bool = bool(_USE_PG_DOCKAREA_PRE_LAYOUT)
        self._pre_dockarea: Optional[DockArea] = None
        self._pre_drawer_splitter: Optional[QtWidgets.QSplitter] = None
        self._pre_dockarea_docks: Dict[str, Dock] = {}
        self._pre_section_scroll_hosts: Dict[str, QtWidgets.QScrollArea] = {}
        self._pre_dockarea_fixed_layout_applied: bool = False
        self._shortcuts: List[QtGui.QShortcut] = []
        self._last_opened_section: Optional[str] = None
        self._section_popup_initialized: set[str] = set()
        self._is_restoring_panel_layout: bool = False
        # Prevent startup widget initialization from overwriting previously saved panel layout.
        self._panel_layout_persistence_ready: bool = False
        # Prevent temporary tab-switch popup visibility changes from overwriting persisted layout.
        self._suspend_panel_layout_persistence: bool = False
        self._pre_popups_hidden_by_tab_switch: bool = False
        self._pre_section_visibility_before_tab_switch: Dict[str, bool] = {}
        self._pre_section_state_before_tab_switch: Dict[str, Dict[str, object]] = {}
        self._pre_artifact_visible_before_tab_switch: bool = False
        self._pre_artifact_state_before_tab_switch: Dict[str, object] = {}
        self._pre_advanced_visible_before_tab_switch: bool = False
        self._pre_main_dock_state_before_tab_switch: Optional[QtCore.QByteArray] = None
        self._pre_tab_groups_before_tab_switch: List[Dict[str, object]] = []
        self._pre_last_interacted_dock_name: Optional[str] = None
        self._pre_snapshot_applied: bool = False
        self._pre_snapshot_retry_attempts: int = 0
        self._pre_snapshot_retry_scheduled: bool = False
        self._pre_snapshot_max_retries: int = 6
        self._post_docks_ready: bool = False
        self._handling_main_tab_change: bool = False
        self._pending_main_tab_index: Optional[int] = None
        self._force_fixed_dock_layouts: bool = bool(_FORCE_FIXED_DOCK_LAYOUTS)
        self._app_theme_mode: str = "dark"

        # Worker infra (stable)
        self._pool = QtCore.QThreadPool.globalInstance()
        self._job_counter = 0
        self._latest_job_id = 0
        self._preview_preserve_view_pending: bool = False
        self._preview_preserve_view_by_job: Dict[int, bool] = {}

        # Debounce
        self._preview_timer = QtCore.QTimer(self)
        self._preview_timer.setSingleShot(True)
        self._preview_timer.setInterval(180)
        self._preview_timer.timeout.connect(self._start_preview_processing)

        # Settings (persist folder + params)
        self.settings = QtCore.QSettings("FiberPhotometryApp", "DoricProcessor")
        self._migrate_legacy_dock_state_settings()
        # Load panel layout JSON into QSettings before UI is built.
        self._load_panel_config_json_into_settings()

        self._build_ui()
        self._restore_settings()
        self._panel_layout_persistence_ready = True
        # Enforce: preprocessing drawer is hidden until the user
        # explicitly clicks a rail section button (overrides any saved state).
        self._force_hide_pre_drawer_initially()

    # ---------------- UI ----------------

    def _set_initial_window_size(self) -> None:
        """Choose a sensible non-fullscreen default size relative to the active screen."""
        screen = QtGui.QGuiApplication.primaryScreen()
        if screen is None:
            self.resize(1280, 780)
            return
        rect = screen.availableGeometry()
        width = max(1024, min(1500, int(rect.width() * 0.86)))
        height = max(680, min(900, int(rect.height() * 0.84)))
        min_w = max(860, min(980, int(rect.width() * 0.65)))
        min_h = max(560, min(640, int(rect.height() * 0.60)))
        self.setMinimumSize(min_w, min_h)
        self.resize(width, height)

    def _build_ui(self) -> None:
        self.setStyleSheet(app_qss(self._app_theme_mode))

        self.tabs = QtWidgets.QTabWidget()
        self.setCentralWidget(self.tabs)
        self._status_bar = QtWidgets.QStatusBar(self)
        self.setStatusBar(self._status_bar)
        self.btn_app_theme = QtWidgets.QPushButton("Theme")
        self.btn_app_theme.setProperty("class", "blueSecondarySmall")
        self.btn_app_theme.setSizePolicy(QtWidgets.QSizePolicy.Policy.Minimum, QtWidgets.QSizePolicy.Policy.Fixed)
        self.btn_app_theme.setFocusPolicy(QtCore.Qt.FocusPolicy.NoFocus)
        self.menu_app_theme = QtWidgets.QMenu(self.btn_app_theme)
        self._app_theme_group = QtGui.QActionGroup(self)
        self._app_theme_group.setExclusive(True)
        self.act_app_theme_dark = self.menu_app_theme.addAction("Dark mode")
        self.act_app_theme_dark.setCheckable(True)
        self.act_app_theme_light = self.menu_app_theme.addAction("Light mode")
        self.act_app_theme_light.setCheckable(True)
        self._app_theme_group.addAction(self.act_app_theme_dark)
        self._app_theme_group.addAction(self.act_app_theme_light)
        self.act_app_theme_dark.setChecked(True)
        self.btn_app_theme.setMenu(self.menu_app_theme)

        self._status_bar.addPermanentWidget(QtWidgets.QLabel("App theme"))
        self._status_bar.addPermanentWidget(self.btn_app_theme)

        # Preprocessing tab
        self.pre_tab = QtWidgets.QWidget()
        self.tabs.addTab(self.pre_tab, "Preprocessing")

        self.file_panel = FileQueuePanel(self.pre_tab)
        self.param_panel = ParameterPanel(self.pre_tab)
        self.param_panel.setVisible(False)
        self.plots = PlotDashboard(self.pre_tab)
        self.artifact_panel = ArtifactPanel(self.pre_tab)

        self.art_dock: Optional[QtWidgets.QDockWidget] = None
        # Legacy artifact list dock (kept for non-DockArea preprocessing mode).
        if not self._use_pg_dockarea_pre_layout:
            self.art_dock = QtWidgets.QDockWidget("Artifact list", self)
            self.art_dock.setObjectName("pre.artifact.dock")
            self.art_dock.setWidget(self.artifact_panel)
            self.art_dock.setAllowedAreas(QtCore.Qt.DockWidgetArea.LeftDockWidgetArea)
            self.art_dock.setVisible(False)
            self.art_dock.visibilityChanged.connect(lambda *_: self._save_panel_layout_state())
            self.art_dock.topLevelChanged.connect(lambda *_: self._save_panel_layout_state())
            self.art_dock.dockLocationChanged.connect(lambda *_: self._save_panel_layout_state())
            self.art_dock.installEventFilter(self)
            self.artifact_panel.installEventFilter(self)
            self.addDockWidget(QtCore.Qt.DockWidgetArea.LeftDockWidgetArea, self.art_dock)

        # Left pane: data browser
        self.file_panel.setMinimumWidth(260)
        self.file_panel.setMaximumWidth(340)
        self.file_panel.setSizePolicy(QtWidgets.QSizePolicy.Policy.Fixed, QtWidgets.QSizePolicy.Policy.Expanding)

        # Center pane: workflow toolbar + plots
        self.btn_workflow_load = QtWidgets.QPushButton("File")
        self.btn_workflow_load.setProperty("class", "blueSecondarySmall")
        self.menu_workflow_load = QtWidgets.QMenu(self.btn_workflow_load)
        self.act_pre_new_project = self.menu_workflow_load.addAction("New Project")
        self.act_pre_open_project = self.menu_workflow_load.addAction("Open Project...")
        self.act_pre_save_project = self.menu_workflow_load.addAction("Save Project...")
        self.menu_workflow_load.addSeparator()
        self.act_open_file = self.menu_workflow_load.addAction("Open File...")
        self.act_add_folder = self.menu_workflow_load.addAction("Add Folder...")
        self.menu_workflow_load_recent = self.menu_workflow_load.addMenu("Recent Files")
        self.menu_workflow_load_recent.aboutToShow.connect(self._refresh_recent_preprocessing_menu)
        self.menu_workflow_load.addSeparator()
        self.act_focus_data = self.menu_workflow_load.addAction("Focus Data Browser")
        self.btn_workflow_load.setMenu(self.menu_workflow_load)

        self.btn_workflow_artifacts = QtWidgets.QPushButton("Detected artifacts")
        self.btn_workflow_qc = QtWidgets.QPushButton("QC")
        self.btn_workflow_export = QtWidgets.QPushButton("Export")
        self.btn_plot_style = QtWidgets.QPushButton("Plot style")
        self.btn_toggle_data = QtWidgets.QPushButton("Data")
        self.btn_toggle_data.setCheckable(True)
        self.btn_toggle_data.setChecked(True)
        self.btn_toggle_data.setProperty("class", "blueSecondarySmall")
        self.btn_workflow_export.setProperty("class", "bluePrimarySmall")
        for b in (
            self.btn_toggle_data,
            self.btn_workflow_load,
            self.btn_workflow_artifacts,
            self.btn_workflow_qc,
            self.btn_workflow_export,
            self.btn_plot_style,
        ):
            b.setSizePolicy(QtWidgets.QSizePolicy.Policy.Minimum, QtWidgets.QSizePolicy.Policy.Fixed)
            b.setFocusPolicy(QtCore.Qt.FocusPolicy.NoFocus)
        for b in (self.btn_workflow_artifacts, self.btn_workflow_qc, self.btn_plot_style):
            b.setProperty("class", "blueSecondarySmall")

        self.menu_plot_style = QtWidgets.QMenu(self.btn_plot_style)
        self._plot_bg_group = QtGui.QActionGroup(self)
        self._plot_bg_group.setExclusive(True)
        self.act_plot_bg_dark = self.menu_plot_style.addAction("Dark background")
        self.act_plot_bg_dark.setCheckable(True)
        self.act_plot_bg_white = self.menu_plot_style.addAction("White background")
        self.act_plot_bg_white.setCheckable(True)
        self._plot_bg_group.addAction(self.act_plot_bg_dark)
        self._plot_bg_group.addAction(self.act_plot_bg_white)
        self.menu_plot_style.addSeparator()
        self.act_plot_grid = self.menu_plot_style.addAction("Show grid")
        self.act_plot_grid.setCheckable(True)
        self.act_plot_bg_dark.setChecked(True)
        self.act_plot_grid.setChecked(True)
        self.btn_plot_style.setMenu(self.menu_plot_style)

        # Inline parameter section buttons (same row as workflow actions).
        self.btn_section_artifacts_list = QtWidgets.QPushButton("Artifact list")
        self.btn_section_artifacts = QtWidgets.QPushButton("Artifact setup")
        self.btn_section_filtering = QtWidgets.QPushButton("Filtering")
        self.btn_section_baseline = QtWidgets.QPushButton("Baseline")
        self.btn_section_output = QtWidgets.QPushButton("Output")
        self.btn_section_qc = QtWidgets.QPushButton("QC")
        self.btn_section_export = QtWidgets.QPushButton("Export")
        self.btn_section_config = QtWidgets.QPushButton("Configuration")
        self._section_buttons: Dict[str, QtWidgets.QPushButton] = {
            "artifacts_list": self.btn_section_artifacts_list,
            "artifacts": self.btn_section_artifacts,
            "filtering": self.btn_section_filtering,
            "baseline": self.btn_section_baseline,
            "output": self.btn_section_output,
            "export": self.btn_section_export,
            "qc": self.btn_section_qc,
            "config": self.btn_section_config,
        }
        for btn in self._section_buttons.values():
            btn.setCheckable(True)
            btn.setProperty("class", "blueSecondarySmall")
            btn.setSizePolicy(QtWidgets.QSizePolicy.Policy.Minimum, QtWidgets.QSizePolicy.Policy.Fixed)
            btn.setFocusPolicy(QtCore.Qt.FocusPolicy.NoFocus)

        # ----- Modern shell: vertical icon rail + thin transport bar ------
        # Configure section buttons as icon-only rail buttons.
        _rail_section_meta = {
            "artifacts_list": ("Artifact list", _paint_list),
            "artifacts":      ("Artifact setup", _paint_sliders),
            "filtering":      ("Filtering", _paint_filter),
            "baseline":       ("Baseline", _paint_wave),
            "output":         ("Output", _paint_chart),
            "qc":             ("Quality control", _paint_badge),
            "export":         ("Export", _paint_export),
            "config":         ("Configuration", _paint_gear),
        }
        for key, btn in self._section_buttons.items():
            tip, painter = _rail_section_meta[key]
            btn.setObjectName("railButton")
            btn.setProperty("class", "")
            btn.setText("")
            btn.setToolTip(tip)
            btn.setStatusTip(tip)
            btn.setIcon(_make_icon(painter))
            btn.setIconSize(QtCore.QSize(22, 22))
            btn.setFixedSize(44, 44)

        # Data-browser toggle as a rail toggle button.
        self.btn_toggle_data.setObjectName("railToggleButton")
        self.btn_toggle_data.setProperty("class", "")
        self.btn_toggle_data.setText("")
        self.btn_toggle_data.setToolTip("Show or hide data browser")
        self.btn_toggle_data.setStatusTip("Show or hide data browser")
        self.btn_toggle_data.setIcon(_make_icon(_paint_database))
        self.btn_toggle_data.setIconSize(QtCore.QSize(22, 22))
        self.btn_toggle_data.setFixedSize(44, 44)

        side_rail = QtWidgets.QFrame()
        side_rail.setObjectName("sideRail")
        rail_layout = QtWidgets.QVBoxLayout(side_rail)
        rail_layout.setContentsMargins(8, 10, 8, 10)
        rail_layout.setSpacing(6)
        rail_layout.addWidget(self.btn_toggle_data, 0, QtCore.Qt.AlignmentFlag.AlignHCenter)
        sep = QtWidgets.QFrame()
        sep.setObjectName("railSeparator")
        sep.setFrameShape(QtWidgets.QFrame.Shape.HLine)
        rail_layout.addWidget(sep)
        for key in ("artifacts_list", "artifacts", "filtering", "baseline",
                    "output", "qc", "export", "config"):
            rail_layout.addWidget(self._section_buttons[key], 0,
                                  QtCore.Qt.AlignmentFlag.AlignHCenter)
        rail_layout.addStretch(1)
        side_rail.setFixedWidth(64)

        # Transport bar: workflow actions + status meta. Compact, single row.
        transport_bar = QtWidgets.QFrame()
        transport_bar.setObjectName("transportBar")
        transport_layout = QtWidgets.QHBoxLayout(transport_bar)
        transport_layout.setContentsMargins(12, 8, 12, 8)
        transport_layout.setSpacing(8)
        # Rename action buttons to clearer verbs to avoid confusion with rail.
        self.btn_workflow_load.setText("File")
        self.btn_workflow_qc.setText("Run QC")
        self.btn_workflow_export.setText("Run Export")
        transport_layout.addWidget(self.btn_workflow_load)
        transport_layout.addWidget(self.btn_workflow_qc)
        transport_layout.addWidget(self.btn_workflow_export)
        transport_layout.addSpacing(8)
        transport_layout.addWidget(self.btn_plot_style)
        transport_layout.addStretch(1)
        # Redundant duplicate: 'Detected artifacts' workflow button is covered
        # by the artifact-list rail button. Hide from layout but keep instance
        # so existing wiring (signals, references) remains intact.
        self.btn_workflow_artifacts.setVisible(False)

        center_panel = QtWidgets.QFrame()
        center_panel.setObjectName("centerPanel")
        center_panel_layout = QtWidgets.QVBoxLayout(center_panel)
        center_panel_layout.setContentsMargins(10, 10, 10, 10)
        center_panel_layout.setSpacing(8)
        center_panel_layout.addWidget(transport_bar)
        center_panel_layout.addWidget(self.plots, stretch=1)

        center_widget = QtWidgets.QWidget()
        center_h = QtWidgets.QHBoxLayout(center_widget)
        center_h.setContentsMargins(0, 0, 0, 0)
        center_h.setSpacing(8)
        center_h.addWidget(side_rail)
        if self._use_pg_dockarea_pre_layout:
            self._pre_drawer_splitter = QtWidgets.QSplitter(QtCore.Qt.Orientation.Horizontal)
            self._pre_drawer_splitter.setChildrenCollapsible(False)
            self._pre_dockarea = DockArea()
            # Wrap the DockArea in a rounded drawer frame so it matches the
            # modern shell, and start hidden. It sits beside the left rail so
            # tool panels open on the same side as the data browser.
            self._pre_drawer = QtWidgets.QFrame()
            self._pre_drawer.setObjectName("drawerPanel")
            _drawer_l = QtWidgets.QVBoxLayout(self._pre_drawer)
            _drawer_l.setContentsMargins(12, 10, 12, 10)
            _drawer_l.setSpacing(8)
            self._pre_drawer_title = QtWidgets.QLabel("")
            self._pre_drawer_title.setObjectName("panelTitle")
            _drawer_l.addWidget(self._pre_drawer_title)
            _drawer_l.addWidget(self._pre_dockarea, stretch=1)
            self._pre_drawer.setVisible(False)
            self._pre_drawer_splitter.addWidget(self._pre_drawer)
            self._pre_drawer_splitter.addWidget(center_panel)
            self._pre_drawer_splitter.setStretchFactor(0, 0)
            self._pre_drawer_splitter.setStretchFactor(1, 1)
            self._pre_drawer_splitter.setSizes([0, 1400])
            center_h.addWidget(self._pre_drawer_splitter, stretch=1)
        else:
            center_h.addWidget(center_panel, stretch=1)

        # Main splitter: data panel + visuals. Parameter popups are floating by default.
        self.pre_splitter = QtWidgets.QSplitter(QtCore.Qt.Orientation.Horizontal)
        self.pre_splitter.setObjectName("preprocessing_splitter")
        self.pre_splitter.addWidget(self.file_panel)
        self.pre_splitter.addWidget(center_widget)
        self.pre_splitter.setChildrenCollapsible(False)
        self.pre_splitter.setStretchFactor(0, 0)
        self.pre_splitter.setStretchFactor(1, 1)
        self.pre_splitter.setSizes([350, 1350])
        self.pre_splitter.splitterMoved.connect(self._save_splitter_sizes)

        pre_layout = QtWidgets.QVBoxLayout(self.pre_tab)
        pre_layout.setContentsMargins(10, 10, 10, 10)
        pre_layout.addWidget(self.pre_splitter)

        # Postprocessing tab
        self.post_tab = PostProcessingPanel()
        if hasattr(self.post_tab, "set_app_theme_mode"):
            try:
                self.post_tab.set_app_theme_mode(self._app_theme_mode)
            except Exception:
                pass
        if hasattr(self.post_tab, "set_force_fixed_default_layout"):
            try:
                self.post_tab.set_force_fixed_default_layout(self._force_fixed_dock_layouts)
            except Exception:
                pass
        self.tabs.addTab(self.post_tab, "Postprocessing")
        self.post_tab.statusUpdate.connect(self._show_status_message)

        # Wiring - file panel
        self.file_panel.openFileRequested.connect(self._open_files_dialog)
        self.file_panel.openFolderRequested.connect(self._open_folder_dialog)
        self.file_panel.selectionChanged.connect(self._on_file_selection_changed)
        self.file_panel.channelChanged.connect(self._on_channel_changed)
        self.file_panel.triggerChanged.connect(self._on_trigger_changed)
        self.file_panel.timeWindowChanged.connect(self._on_time_window_changed)

        self.file_panel.updatePreviewRequested.connect(self._trigger_preview)
        self.file_panel.metadataRequested.connect(self._edit_metadata_for_current)
        self.file_panel.exportRequested.connect(self._export_selected_or_all)
        self.file_panel.toggleArtifactsRequested.connect(self._toggle_artifacts_panel)
        self.file_panel.advancedOptionsRequested.connect(self._open_advanced_options)
        self.file_panel.qcRequested.connect(self._run_qc_dialog)
        self.file_panel.batchQcRequested.connect(self._run_batch_qc)

        # Parameters: changes and actions
        self.param_panel.paramsChanged.connect(self._on_params_changed)
        self.param_panel.paramsChanged.connect(self._update_export_summary_label)
        self.param_panel.previewRequested.connect(self._trigger_preview)
        self.param_panel.metadataRequested.connect(self._edit_metadata_for_current)
        self.param_panel.exportRequested.connect(self._export_selected_or_all)
        self.param_panel.artifactsRequested.connect(self._toggle_artifacts_panel)
        self.param_panel.artifactOverlayToggled.connect(self._on_artifact_overlay_toggled)
        self.param_panel.advancedOptionsRequested.connect(self._open_advanced_options)
        self.param_panel.qcRequested.connect(self._run_qc_dialog)
        self.param_panel.batchQcRequested.connect(self._run_batch_qc)
        self.param_panel.set_config_state_hooks(
            self._export_preprocessing_ui_state_for_config,
            self._import_preprocessing_ui_state_from_config,
        )

        # Workflow toolbar
        self.act_pre_new_project.triggered.connect(self._new_preprocessing_project)
        self.act_pre_open_project.triggered.connect(self._open_preprocessing_project_file)
        self.act_pre_save_project.triggered.connect(self._save_preprocessing_project_file)
        self.act_open_file.triggered.connect(self._open_files_dialog)
        self.act_add_folder.triggered.connect(self._open_folder_dialog)
        self.act_focus_data.triggered.connect(self._focus_data_browser)
        self.act_plot_bg_dark.triggered.connect(self._on_pre_plot_style_changed)
        self.act_plot_bg_white.triggered.connect(self._on_pre_plot_style_changed)
        self.act_plot_grid.toggled.connect(self._on_pre_plot_style_changed)
        self.act_app_theme_dark.triggered.connect(self._on_app_theme_changed)
        self.act_app_theme_light.triggered.connect(self._on_app_theme_changed)
        self.btn_toggle_data.toggled.connect(self._set_data_panel_visible)
        self.btn_workflow_artifacts.clicked.connect(self._toggle_artifacts_panel)
        self.btn_workflow_qc.clicked.connect(self._run_qc_dialog)
        self.btn_workflow_export.clicked.connect(self._export_selected_or_all)

        # Section popup controls
        self._setup_section_popups()
        for key, btn in self._section_buttons.items():
            btn.toggled.connect(lambda checked, section_key=key: self._toggle_section_popup(section_key, checked))

        # Plot sync
        self.plots.xRangeChanged.connect(self.plots.set_xrange_all)

        # Manual artifacts
        self.plots.manualRegionFromSelectorRequested.connect(self._add_manual_region_from_selector)
        self.plots.manualRegionFromDragRequested.connect(self._add_manual_region_from_drag)
        self.plots.clearManualRegionsRequested.connect(self._clear_manual_regions_current)
        self.plots.showArtifactsRequested.connect(self._toggle_artifacts_panel)
        self.plots.boxSelectionCleared.connect(self._cancel_box_select_request)
        self.plots.boxSelectionContextRequested.connect(self._show_box_selection_context_menu)
        self.plots.artifactThresholdsToggled.connect(self._on_artifact_thresholds_toggled)

        self.artifact_panel.regionsChanged.connect(self._artifact_regions_changed)
        self.artifact_panel.selectionChanged.connect(self.plots.highlight_artifact_regions)

        # Postprocessing needs access to "current processed"
        self.post_tab.requestCurrentProcessed.connect(self._post_get_current_processed)
        self.post_tab.requestDioList.connect(self._post_get_current_dio_list)
        self.post_tab.requestDioData.connect(self._post_get_dio_data_for_path)
        self.tabs.currentChanged.connect(self._on_main_tab_changed)

        self._init_shortcuts()
        self.plots.set_artifact_overlay_visible(self.param_panel.artifact_overlay_visible())
        self.plots.set_artifact_thresholds_visible(True)
        self._update_plot_status()
        self.setAcceptDrops(True)

    def _setup_section_popups(self) -> None:
        """Create preprocessing section panels using DockArea or legacy floating docks."""
        if self._use_pg_dockarea_pre_layout and self._pre_dockarea_docks:
            return
        if (not self._use_pg_dockarea_pre_layout) and self._section_docks:
            return

        # Move section cards out of the hidden ParameterPanel container and into docks.
        root_layout = self.param_panel.layout()
        section_cards = [
            self.param_panel.card_artifacts,
            self.param_panel.card_filtering,
            self.param_panel.card_baseline,
            self.param_panel.card_output,
            self.param_panel.card_actions,
        ]
        if root_layout is not None:
            for w in section_cards:
                root_layout.removeWidget(w)
                w.setParent(None)
        self.param_panel.card_actions.setVisible(False)

        section_widgets: Dict[str, QtWidgets.QWidget] = {
            "artifacts_list": self.artifact_panel,
            "artifacts": self.param_panel.card_artifacts,
            "filtering": self.param_panel.card_filtering,
            "baseline": self.param_panel.card_baseline,
            "output": self.param_panel.card_output,
            "export": self._build_export_actions_widget(),
            "qc": self._build_qc_actions_widget(),
            "config": self._build_config_actions_widget(),
        }
        section_titles: Dict[str, str] = {
            "artifacts_list": "Artifact list",
            "artifacts": "Artifact setup",
            "filtering": "Filtering",
            "baseline": "Baseline",
            "output": "Output",
            "export": "Export",
            "qc": "QC",
            "config": "Configuration",
        }

        if self._use_pg_dockarea_pre_layout:
            if self._pre_dockarea is None:
                return
            for key, title in section_titles.items():
                widget = section_widgets[key]
                scroll = QtWidgets.QScrollArea()
                scroll.setWidgetResizable(True)
                scroll.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
                scroll.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
                scroll.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAsNeeded)
                widget.setMinimumSize(0, 0)
                widget.setSizePolicy(QtWidgets.QSizePolicy.Policy.Preferred, QtWidgets.QSizePolicy.Policy.Preferred)
                scroll.setWidget(widget)
                self._pre_section_scroll_hosts[key] = scroll

                dock = Dock(title, area=self._pre_dockarea, closable=False)
                dock.setObjectName(f"pre.da.{key}.dock")
                dock.addWidget(scroll)
                # Collapse the per-dock label/tab to 0px without deleting it
                # (pyqtgraph still references dock.label when restacking).
                try:
                    dock.label.setMaximumHeight(0)
                    dock.label.setMinimumHeight(0)
                    dock.label.setFixedHeight(0)
                    dock.label.setVisible(False)
                except Exception:
                    pass
                self._lock_pre_pg_dock_interactions(dock)
                try:
                    dock.sigClosed.connect(lambda *_, section_key=key: self._on_pre_dockarea_dock_closed(section_key))
                except Exception:
                    pass
                self._pre_dockarea_docks[key] = dock

            self._restore_pre_dockarea_layout_state()
            self._pre_dockarea_fixed_layout_applied = False
            return

        for key, title in section_titles.items():
            widget = section_widgets[key]
            dock = QtWidgets.QDockWidget(title, self)
            dock.setObjectName(f"pre.{key}.dock")
            dock.setAllowedAreas(QtCore.Qt.DockWidgetArea.AllDockWidgetAreas)
            dock.setFeatures(
                QtWidgets.QDockWidget.DockWidgetFeature.DockWidgetClosable
                | QtWidgets.QDockWidget.DockWidgetFeature.DockWidgetMovable
                | QtWidgets.QDockWidget.DockWidgetFeature.DockWidgetFloatable
            )
            dock.setWidget(widget)
            dock.visibilityChanged.connect(lambda visible, section_key=key: self._on_section_dock_visibility(section_key, visible))
            dock.topLevelChanged.connect(lambda *_: self._save_panel_layout_state())
            dock.dockLocationChanged.connect(lambda *_: self._save_panel_layout_state())
            # Register with main window once; each popup opens floating by default.
            self.addDockWidget(QtCore.Qt.DockWidgetArea.LeftDockWidgetArea, dock)
            dock.setFloating(True)
            dock.hide()
            dock.installEventFilter(self)
            widget.installEventFilter(self)
            self._section_docks[key] = dock

    def _pre_dockarea_dock(self, key: str) -> Optional[Dock]:
        return self._pre_dockarea_docks.get(key)

    def _pre_dockarea_ordered_keys(self) -> List[str]:
        ordered = list(_PRE_DOCKAREA_PRIMARY_ORDER) + list(_PRE_DOCKAREA_OPTIONAL_ORDER)
        return [key for key in ordered if self._pre_dockarea_dock(key) is not None]

    def _pre_dockarea_default_visible_map(self) -> Dict[str, bool]:
        return {key: (key in _PRE_DOCKAREA_DEFAULT_VISIBLE) for key in self._pre_dockarea_docks.keys()}

    def _lock_pre_pg_dock_interactions(self, dock: Dock) -> None:
        label = getattr(dock, "label", None)
        if label is None:
            return
        if not self._force_fixed_dock_layouts:
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
                    btn.clicked.connect(lambda _checked=False, section_dock=dock: self._hide_pre_dockarea_dock(section_dock))
                    btn.setProperty("_pyber_hide_wired", True)
                btn.setStyleSheet(
                    "QToolButton {"
                    " background: transparent;"
                    " color: #f3f5f8;"
                    " border: none;"
                    " padding: 0px;"
                    " margin: 0px;"
                    " font-size: 8pt;"
                    " font-weight: 700;"
                    " }"
                    "QToolButton:hover {"
                    " background: transparent;"
                    " color: #ffffff;"
                    " border: none;"
                    " }"
                )
            except Exception:
                continue

    def _hide_pre_dockarea_dock(self, dock: Dock) -> None:
        if dock is None:
            return
        try:
            dock.hide()
        except Exception:
            return
        for key, candidate in self._pre_dockarea_docks.items():
            if candidate is dock:
                self._set_section_button_checked(key, False)
                if self._last_opened_section == key:
                    self._last_opened_section = None
                break
        self._update_pre_drawer_visibility()
        self._save_panel_layout_state()

    def _arrange_pre_dockarea_default(self) -> None:
        if self._pre_dockarea is None:
            return
        ordered = self._pre_dockarea_ordered_keys()
        root = self._pre_dockarea_dock("artifacts_list")
        if root is None and ordered:
            root = self._pre_dockarea_dock(ordered[0])
        if root is None:
            return
        self._pre_dockarea.addDock(root, "left")
        for key in ordered:
            dock = self._pre_dockarea_dock(key)
            if dock is not None and dock is not root:
                self._pre_dockarea.addDock(dock, "above", root)

    def _pre_dockarea_active_key(self) -> Optional[str]:
        active = self._last_opened_section
        if isinstance(active, str) and active in self._pre_dockarea_docks:
            return active
        for key in self._pre_dockarea_ordered_keys():
            dock = self._pre_dockarea_dock(key)
            if dock is not None and dock.isVisible():
                return key
        return None

    def _set_pre_dockarea_visible(self, key: str, visible: bool) -> None:
        dock = self._pre_dockarea_dock(key)
        if dock is None:
            return
        if visible:
            self._arrange_pre_dockarea_default()
            dock.show()
            try:
                dock.raiseDock()
            except Exception:
                pass
        else:
            dock.hide()

    def _save_pre_dockarea_layout_state(self) -> None:
        if self._pre_dockarea is None:
            return
        try:
            state = dict(self._pre_dockarea.saveState() or {})
        except Exception:
            state = {}
        visible = {key: bool(dock.isVisible()) for key, dock in self._pre_dockarea_docks.items()}
        active = self._pre_dockarea_active_key() or ""
        try:
            self.settings.setValue(_PRE_DOCKAREA_STATE_KEY, json.dumps(state))
            self.settings.setValue(_PRE_DOCKAREA_VISIBLE_KEY, json.dumps(visible))
            self.settings.setValue(_PRE_DOCKAREA_ACTIVE_KEY, active)
            self.settings.remove(_PRE_DOCK_STATE_KEY)
            self.settings.remove(_PRE_TAB_GROUPS_KEY)
        except Exception:
            pass

        left_i = _dock_area_to_int(QtCore.Qt.DockWidgetArea.LeftDockWidgetArea, 1)
        for key, dock in self._pre_dockarea_docks.items():
            if key == "artifacts_list":
                continue
            try:
                base = f"pre_section_docks/{key}"
                self.settings.setValue(f"{base}/visible", bool(dock.isVisible()))
                self.settings.setValue(f"{base}/floating", False)
                self.settings.setValue(f"{base}/area", left_i)
            except Exception:
                continue
        try:
            art_base = "pre_artifact_dock_state"
            art_vis = bool(visible.get("artifacts_list", False))
            self.settings.setValue(f"{art_base}/visible", art_vis)
            self.settings.setValue(f"{art_base}/floating", False)
            self.settings.setValue(f"{art_base}/area", left_i)
        except Exception:
            pass

    def _restore_pre_dockarea_layout_state(self) -> None:
        if self._pre_dockarea is None or not self._pre_dockarea_docks:
            return
        self._pre_dockarea_fixed_layout_applied = False
        self._arrange_pre_dockarea_default()

        # The left rail drawer behaves as a single-section stack. Restoring old
        # DockArea splitter topology can strand the active dock in a zero-height slot.
        visible_map: Dict[str, bool] = {}
        raw_vis = self.settings.value(_PRE_DOCKAREA_VISIBLE_KEY, "")
        try:
            if isinstance(raw_vis, str) and raw_vis.strip():
                parsed = json.loads(raw_vis)
                if isinstance(parsed, dict):
                    visible_map = {str(k): bool(v) for k, v in parsed.items()}
        except Exception:
            visible_map = {}

        if not visible_map:
            for key in self._pre_dockarea_docks.keys():
                if key == "artifacts_list":
                    raw = self.settings.value("pre_artifact_dock_state/visible", None)
                    if raw is not None:
                        visible_map[key] = _to_bool(raw, False)
                    continue
                raw = self.settings.value(f"pre_section_docks/{key}/visible", None)
                if raw is not None:
                    visible_map[key] = _to_bool(raw, False)
        if not visible_map:
            visible_map = self._pre_dockarea_default_visible_map()

        active = str(self.settings.value(_PRE_DOCKAREA_ACTIVE_KEY, "artifacts_list") or "artifacts_list")
        if not bool(visible_map.get(active, False)):
            active = next((key for key in self._pre_dockarea_ordered_keys() if bool(visible_map.get(key, False))), "")

        for key in self._pre_dockarea_docks.keys():
            self._set_pre_dockarea_visible(key, bool(active and key == active))

        active_dock = self._pre_dockarea_dock(active)
        if active_dock is not None and active_dock.isVisible():
            try:
                active_dock.raiseDock()
            except Exception:
                pass
            if active in self._section_buttons:
                self._last_opened_section = active
        else:
            self._last_opened_section = None
            for key in self._pre_dockarea_ordered_keys():
                dock = self._pre_dockarea_dock(key)
                if dock is not None and dock.isVisible():
                    try:
                        dock.raiseDock()
                    except Exception:
                        pass
                    self._last_opened_section = key
                    break

        self._sync_section_button_states_from_docks()
        self._update_pre_drawer_visibility()

    def _apply_pre_fixed_dockarea_layout(self) -> None:
        if self._pre_dockarea is None or not self._pre_dockarea_docks:
            return
        visible_map = {key: bool(dock.isVisible()) for key, dock in self._pre_dockarea_docks.items()}
        if not any(visible_map.values()):
            visible_map = self._pre_dockarea_default_visible_map()
        self._arrange_pre_dockarea_default()
        for key in self._pre_dockarea_docks.keys():
            self._set_pre_dockarea_visible(key, bool(visible_map.get(key, False)))

        active = self._last_opened_section if bool(visible_map.get(self._last_opened_section or "", False)) else None
        if active is None:
            for key in self._pre_dockarea_ordered_keys():
                if bool(visible_map.get(key, False)):
                    active = key
                    break
        dock = self._pre_dockarea_dock(active) if active else None
        if dock is not None and dock.isVisible():
            try:
                dock.raiseDock()
            except Exception:
                pass
        self._last_opened_section = active
        self._sync_section_button_states_from_docks()
        self._update_pre_drawer_visibility()
        self._save_pre_dockarea_layout_state()
        self._pre_dockarea_fixed_layout_applied = True

    def _on_pre_dockarea_dock_closed(self, key: str) -> None:
        if key in self._section_buttons:
            self._set_section_button_checked(key, False)
            if self._last_opened_section == key:
                self._last_opened_section = None
        self._save_panel_layout_state()

    def _build_qc_actions_widget(self) -> QtWidgets.QWidget:
        panel = QtWidgets.QWidget()
        v = QtWidgets.QVBoxLayout(panel)
        v.setContentsMargins(8, 8, 8, 8)
        v.setSpacing(6)
        self.param_panel.btn_qc.setProperty("class", "blueSecondarySmall")
        self.param_panel.btn_qc_batch.setProperty("class", "blueSecondarySmall")
        self.param_panel.btn_artifacts_panel.setProperty("class", "blueSecondarySmall")
        v.addWidget(self.param_panel.btn_qc)
        v.addWidget(self.param_panel.btn_qc_batch)
        v.addWidget(self.param_panel.btn_artifacts_panel)
        v.addWidget(self.param_panel.lbl_fs)
        v.addStretch(1)
        return panel

    def _build_export_actions_widget(self) -> QtWidgets.QWidget:
        panel = QtWidgets.QWidget()
        v = QtWidgets.QVBoxLayout(panel)
        v.setContentsMargins(8, 8, 8, 8)
        v.setSpacing(6)
        self.lbl_export_summary = QtWidgets.QLabel("")
        self.lbl_export_summary.setWordWrap(True)
        self.lbl_export_summary.setProperty("class", "hint")
        v.addWidget(self.lbl_export_summary)
        self.param_panel.btn_export.setProperty("class", "bluePrimarySmall")
        v.addWidget(self.param_panel.btn_export)
        v.addStretch(1)
        self._update_export_summary_label()
        return panel

    def _build_config_actions_widget(self) -> QtWidgets.QWidget:
        panel = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(panel)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)
        row = QtWidgets.QHBoxLayout()
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(6)
        self.param_panel.btn_metadata.setProperty("class", "blueSecondarySmall")
        self.param_panel.btn_save_config.setProperty("class", "blueSecondarySmall")
        self.param_panel.btn_load_config.setProperty("class", "blueSecondarySmall")
        for btn in (
            self.param_panel.btn_metadata,
            self.param_panel.btn_save_config,
            self.param_panel.btn_load_config,
        ):
            btn.setSizePolicy(QtWidgets.QSizePolicy.Policy.Ignored, QtWidgets.QSizePolicy.Policy.Fixed)
            btn.setMinimumWidth(90)
            btn.setFocusPolicy(QtCore.Qt.FocusPolicy.NoFocus)
            row.addWidget(btn)
        layout.addLayout(row)
        if hasattr(self.param_panel, "export_options_group"):
            self.param_panel.export_options_group.setSizePolicy(
                QtWidgets.QSizePolicy.Policy.Expanding,
                QtWidgets.QSizePolicy.Policy.Fixed,
            )
            layout.addWidget(self.param_panel.export_options_group)
        layout.addStretch(1)
        return panel

    def _update_export_summary_label(self) -> None:
        if hasattr(self, "lbl_export_summary") and self.lbl_export_summary is not None:
            self.lbl_export_summary.setText(self.param_panel.export_selection_summary())

    def _set_section_button_checked(self, key: str, checked: bool) -> None:
        btn = self._section_buttons.get(key)
        if btn is None:
            return
        btn.blockSignals(True)
        btn.setChecked(bool(checked))
        btn.blockSignals(False)

    def _force_hide_pre_drawer_initially(self) -> None:
        """Hide every preprocessing section dock and the left drawer at startup."""
        for key, btn in self._section_buttons.items():
            if btn.isChecked():
                blocked = btn.blockSignals(True)
                try:
                    btn.setChecked(False)
                finally:
                    btn.blockSignals(blocked)
            try:
                dock = self._pre_dockarea_dock(key) if self._use_pg_dockarea_pre_layout else self._section_docks.get(key)
                if dock is not None:
                    dock.hide()
            except Exception:
                pass
        drawer = getattr(self, "_pre_drawer", None)
        if drawer is not None:
            drawer.setVisible(False)
        splitter = self._pre_drawer_splitter
        if splitter is not None:
            try:
                sizes = splitter.sizes()
                if len(sizes) >= 2:
                    sizes[1] += sizes[0]
                    sizes[0] = 0
                    splitter.setSizes(sizes)
            except Exception:
                pass

    _PRE_SECTION_TITLES = {
        "artifacts_list": "Artifact list",
        "artifacts": "Artifact setup",
        "filtering": "Filtering",
        "baseline": "Baseline",
        "output": "Output",
        "qc": "Quality control",
        "export": "Export",
        "config": "Configuration",
    }

    def _update_pre_drawer_visibility(self) -> None:
        """Show the left preprocessing drawer iff at least one section is active."""
        drawer = getattr(self, "_pre_drawer", None)
        if drawer is None:
            return
        any_checked = any(btn.isChecked() for btn in self._section_buttons.values())
        # Update header label to the active section title.
        title_lbl = getattr(self, "_pre_drawer_title", None)
        if title_lbl is not None:
            active_key = next((k for k, b in self._section_buttons.items() if b.isChecked()), None)
            title_lbl.setText(self._PRE_SECTION_TITLES.get(active_key or "", ""))
        drawer.setVisible(any_checked)
        splitter = self._pre_drawer_splitter
        if splitter is None:
            return
        try:
            sizes = splitter.sizes()
            if len(sizes) >= 2:
                if any_checked:
                    if sizes[0] < 60:
                        total = sum(sizes) or 1
                        drawer_w = max(420, int(total * 0.28))
                        sizes[0] = drawer_w
                        sizes[1] = max(400, sizes[1] - drawer_w)
                        splitter.setSizes(sizes)
                else:
                    if sizes[0] > 0:
                        sizes[1] += sizes[0]
                        sizes[0] = 0
                        splitter.setSizes(sizes)
        except Exception:
            pass

    def _toggle_section_popup(self, key: str, checked: bool) -> None:
        if self._use_pg_dockarea_pre_layout:
            dock = self._pre_dockarea_dock(key)
            if dock is None:
                return
            if checked:
                self._arrange_pre_dockarea_default()
                # Radio behavior: hide all other section docks and uncheck
                # their rail buttons so only one drawer section is visible.
                for other_key, other_btn in self._section_buttons.items():
                    if other_key == key:
                        continue
                    if other_btn.isChecked():
                        blocked = other_btn.blockSignals(True)
                        try:
                            other_btn.setChecked(False)
                        finally:
                            other_btn.blockSignals(blocked)
                    other_dock = self._pre_dockarea_dock(other_key)
                    if other_dock is not None:
                        try:
                            other_dock.hide()
                        except Exception:
                            pass
                dock.show()
                try:
                    dock.raiseDock()
                except Exception:
                    pass
                scroll = self._pre_section_scroll_hosts.get(key)
                self._focus_first_editable(scroll.widget() if scroll is not None else None)
                self._last_opened_section = key
            else:
                dock.hide()
            self._update_pre_drawer_visibility()
            self._save_panel_layout_state()
            return
        dock = self._section_docks.get(key)
        if dock is None:
            return
        if checked:
            if key not in self._section_popup_initialized or not self._is_popup_on_screen(dock):
                dock.setFloating(True)
                self._position_section_popup(dock)
                self._section_popup_initialized.add(key)
            dock.show()
            dock.raise_()
            dock.activateWindow()
            self._focus_first_editable(dock.widget())
            self._last_opened_section = key
        else:
            dock.hide()

    def _on_section_dock_visibility(self, key: str, visible: bool) -> None:
        if self._use_pg_dockarea_pre_layout:
            if key in self._section_buttons:
                self._set_section_button_checked(key, visible)
                if not visible and self._last_opened_section == key:
                    self._last_opened_section = None
                if visible:
                    self._last_opened_section = key
            self._update_pre_drawer_visibility()
            self._save_panel_layout_state()
            return
        self._set_section_button_checked(key, visible)
        if not visible and self._last_opened_section == key:
            self._last_opened_section = None
        if visible:
            self._last_opened_section = key
        self._save_panel_layout_state()

    def _position_section_popup(self, dock: QtWidgets.QDockWidget) -> None:
        """Place floating popups near the window while keeping them inside visible screen bounds."""
        geom = self.frameGeometry()
        screen_rect = self._active_screen_geometry()

        pref_w, pref_h = self._default_popup_size(dock)
        max_w = max(320, screen_rect.width() - 40)
        max_h = max(260, screen_rect.height() - 40)
        width = min(pref_w, max_w)
        height = min(pref_h, max_h)

        # Prefer the left side of the main window, then fall back to right, then clamp.
        x_right = geom.x() + geom.width() + 12
        x_left = geom.x() - width - 12
        y_pref = geom.y() + 60

        x_min = screen_rect.x() + 10
        y_min = screen_rect.y() + 10
        x_max = screen_rect.x() + max(10, screen_rect.width() - width - 10)
        y_max = screen_rect.y() + max(10, screen_rect.height() - height - 10)

        if x_left >= x_min:
            x = x_left
        elif x_right <= x_max:
            x = x_right
        else:
            x = x_max
        y = min(max(y_pref, y_min), y_max)

        dock.resize(width, height)
        dock.move(int(x), int(y))

    def _default_popup_size(self, dock: QtWidgets.QDockWidget) -> Tuple[int, int]:
        """Compact default popup sizes, with smaller heights per section."""
        geom = self.frameGeometry()
        name = str(dock.objectName() or "")
        if name.startswith(f"{_PRE_DOCK_PREFIX}") and name.endswith(".dock"):
            key = name[len(_PRE_DOCK_PREFIX):-len(".dock")]
        else:
            key = name
        height_by_section = {
            "artifacts": 380,
            "filtering": 340,
            "baseline": 380,
            "output": 410,
            "qc": 300,
            "export": 270,
            "config": 280,
        }
        pref_h = int(height_by_section.get(key, 340))
        pref_w = max(360, int(geom.width() * 0.24))
        return pref_w, pref_h

    def _active_screen_geometry(self) -> QtCore.QRect:
        handle = self.windowHandle()
        screen = handle.screen() if handle else None
        if screen is None:
            screen = QtGui.QGuiApplication.screenAt(self.frameGeometry().center())
        if screen is None:
            screen = QtGui.QGuiApplication.primaryScreen()
        if screen is None:
            return QtCore.QRect(0, 0, 1920, 1080)
        return screen.availableGeometry()

    def _is_popup_on_screen(self, dock: QtWidgets.QDockWidget) -> bool:
        rect = dock.frameGeometry()
        if rect.width() <= 0 or rect.height() <= 0:
            return False
        for screen in QtGui.QGuiApplication.screens():
            if screen.availableGeometry().intersects(rect):
                return True
        return False

    def eventFilter(self, obj: QtCore.QObject, event: QtCore.QEvent) -> bool:
        try:
            if event.type() in (QtCore.QEvent.Type.MouseButtonPress, QtCore.QEvent.Type.FocusIn):
                dock: Optional[QtWidgets.QDockWidget] = None
                if isinstance(obj, QtWidgets.QDockWidget):
                    dock = obj
                elif isinstance(obj, QtWidgets.QWidget):
                    parent = obj
                    while parent is not None and not isinstance(parent, QtWidgets.QDockWidget):
                        parent = parent.parentWidget()
                    if isinstance(parent, QtWidgets.QDockWidget):
                        dock = parent
                if dock is not None:
                    name = str(dock.objectName() or "")
                    if name.startswith(_PRE_DOCK_PREFIX):
                        self._pre_last_interacted_dock_name = name
        except Exception:
            pass
        return super().eventFilter(obj, event)

    def getPreDockWidgets(self) -> List[QtWidgets.QDockWidget]:
        if self._use_pg_dockarea_pre_layout:
            docks: List[QtWidgets.QDockWidget] = []
            if isinstance(self.art_dock, QtWidgets.QDockWidget):
                docks.append(self.art_dock)
        else:
            docks = list(self._section_docks.values())
            if isinstance(self.art_dock, QtWidgets.QDockWidget):
                docks.append(self.art_dock)
        seen: set[int] = set()
        out: List[QtWidgets.QDockWidget] = []
        for dock in docks:
            did = id(dock)
            if did in seen:
                continue
            seen.add(did)
            out.append(dock)
        return out

    def getPostDockWidgets(self) -> List[QtWidgets.QDockWidget]:
        docks: List[QtWidgets.QDockWidget] = []
        try:
            self.post_tab.ensure_section_popups_initialized()
            docks = list(self.post_tab.get_section_dock_widgets())
        except Exception:
            docks = []
        if docks:
            return docks
        # Fallback for legacy sessions where post docks may already exist but not registered.
        return [
            d for d in self.findChildren(QtWidgets.QDockWidget)
            if str(d.objectName() or "").startswith(_POST_DOCK_PREFIX)
        ]

    def _hide_dock_widgets(self, docks: List[QtWidgets.QDockWidget], *, remove: bool = True) -> None:
        for dock in docks:
            if dock is None:
                continue
            try:
                dock.hide()
            except Exception:
                pass
            if remove:
                try:
                    self.removeDockWidget(dock)
                except Exception:
                    pass

    def hideOtherTabDocks(self, tab_name: str) -> None:
        if tab_name == "pre":
            remove_post = not self._force_fixed_dock_layouts
            self._hide_dock_widgets(self.getPostDockWidgets(), remove=remove_post)
        elif tab_name == "post":
            self._hide_dock_widgets(self.getPreDockWidgets(), remove=True)
            # Final guard: keep post dock registry initialized before post restore paths run.
            try:
                self.post_tab.ensure_section_popups_initialized()
            except Exception:
                pass

    def _enforce_only_tab_docks_visible(self, tab_name: str) -> None:
        self.hideOtherTabDocks(tab_name)

    def captureDockSnapshotForTab(self, tab_name: str) -> Optional[QtCore.QByteArray]:
        """
        Capture a tab-scoped dock snapshot. Other-tab docks are hidden first so
        QMainWindow.saveState() cannot serialize mixed-tab layouts.
        """
        if tab_name not in {"pre", "post"}:
            return None
        try:
            self.hideOtherTabDocks(tab_name)
            QtWidgets.QApplication.processEvents(QtCore.QEventLoop.ProcessEventsFlag.ExcludeUserInputEvents)
            state = self.saveState(_DOCK_STATE_VERSION)
            self.hideOtherTabDocks(tab_name)
            if state is None or state.isEmpty():
                _LOG.warning("Dock snapshot capture returned empty state for tab=%s", tab_name)
                return None
            if not self._is_tab_scoped_dock_state(tab_name, state):
                _LOG.warning("Discarding contaminated dock snapshot for tab=%s", tab_name)
                return None
            return state
        except Exception:
            _LOG.exception("Dock snapshot capture failed for tab=%s", tab_name)
            return None

    def restoreDockSnapshotForTab(self, tab_name: str, state: QtCore.QByteArray) -> bool:
        """
        Restore a tab-scoped dock snapshot with hard post-restore enforcement so
        foreign tab docks cannot leak back into the active tab.
        """
        if tab_name not in {"pre", "post"}:
            return False
        if state is None or state.isEmpty():
            return False
        if not self._is_tab_scoped_dock_state(tab_name, state):
            _LOG.warning("Rejecting invalid/contaminated dock snapshot for tab=%s", tab_name)
            try:
                if tab_name == "pre":
                    self.settings.remove(_PRE_DOCK_STATE_KEY)
                else:
                    self.settings.remove(_POST_DOCK_STATE_KEY)
            except Exception:
                pass
            return False
        try:
            self.hideOtherTabDocks(tab_name)
            ok = bool(self.restoreState(state, _DOCK_STATE_VERSION))
            self.hideOtherTabDocks(tab_name)
            self._enforce_only_tab_docks_visible(tab_name)
            if not ok:
                _LOG.warning("Dock snapshot restore failed for tab=%s", tab_name)
            else:
                _LOG.info("Dock snapshot restore succeeded for tab=%s", tab_name)
            return ok
        except Exception:
            _LOG.exception("Dock snapshot restore crashed for tab=%s", tab_name)
            return False

    def _capture_pre_tab_groups_state(self) -> List[Dict[str, object]]:
        """
        Capture tabified pre-dock groups + active tab candidate for fallback restore.
        """
        docks = [d for d in self.getPreDockWidgets() if not d.isFloating()]
        by_name: Dict[str, QtWidgets.QDockWidget] = {
            str(d.objectName()): d for d in docks if str(d.objectName() or "")
        }
        groups: List[Dict[str, object]] = []
        visited: set[str] = set()
        for dock in docks:
            name = str(dock.objectName() or "")
            if not name or name in visited:
                continue
            members = [dock] + [d for d in self.tabifiedDockWidgets(dock) if d in docks]
            member_names = sorted({str(d.objectName() or "") for d in members if str(d.objectName() or "")})
            if len(member_names) < 2:
                continue
            visited.update(member_names)
            active = ""
            if self._pre_last_interacted_dock_name and self._pre_last_interacted_dock_name in member_names:
                active = self._pre_last_interacted_dock_name
            if not active:
                for n in member_names:
                    d = by_name.get(n)
                    if d is not None and d.isVisible():
                        active = n
                        break
            if not active:
                active = member_names[0]
            groups.append({"members": member_names, "active": active})
        return groups

    def _save_pre_tab_groups_to_settings(self, groups: List[Dict[str, object]]) -> None:
        try:
            self.settings.setValue(_PRE_TAB_GROUPS_KEY, json.dumps(groups))
        except Exception:
            pass

    def _load_pre_tab_groups_from_settings(self) -> List[Dict[str, object]]:
        try:
            raw = self.settings.value(_PRE_TAB_GROUPS_KEY, "", type=str)
            if not raw:
                return []
            data = json.loads(raw)
            if isinstance(data, list):
                return [g for g in data if isinstance(g, dict)]
        except Exception:
            pass
        return []

    def _restore_pre_tab_groups_fallback(self, groups: List[Dict[str, object]]) -> None:
        if not groups:
            return
        by_name = {
            str(d.objectName() or ""): d for d in self.getPreDockWidgets() if str(d.objectName() or "")
        }
        for group in groups:
            members_raw = group.get("members", [])
            if not isinstance(members_raw, list):
                continue
            members = [by_name.get(str(n)) for n in members_raw]
            members = [d for d in members if isinstance(d, QtWidgets.QDockWidget)]
            members = [d for d in members if not d.isFloating()]
            if len(members) < 2:
                continue
            root = members[0]
            for d in members[1:]:
                try:
                    self.tabifyDockWidget(root, d)
                except Exception:
                    continue
            active_name = str(group.get("active", ""))
            active = by_name.get(active_name)
            if isinstance(active, QtWidgets.QDockWidget) and active in members:
                try:
                    active.show()
                    active.raise_()
                    active.setFocus(QtCore.Qt.FocusReason.OtherFocusReason)
                    self._pre_last_interacted_dock_name = active_name
                except Exception:
                    pass

    def _schedule_pre_snapshot_retry(self, delay_ms: int) -> None:
        if self._pre_snapshot_retry_scheduled:
            return
        self._pre_snapshot_retry_scheduled = True
        QtCore.QTimer.singleShot(max(0, int(delay_ms)), self._retry_pre_snapshot_restore)

    def _retry_pre_snapshot_restore(self) -> None:
        self._pre_snapshot_retry_scheduled = False
        self._apply_pre_main_dock_snapshot_if_needed()

    def onPostDocksReady(self) -> None:
        self.on_post_docks_ready()

    def on_post_docks_ready(self) -> None:
        self._post_docks_ready = True
        if self._force_fixed_dock_layouts:
            self._pre_snapshot_applied = True
            return
        if not self._pre_snapshot_applied:
            self._schedule_pre_snapshot_retry(0)

    def _set_data_panel_visible(self, visible: bool, persist: bool = True) -> None:
        vis = bool(visible)
        self.file_panel.setVisible(vis)
        self.btn_toggle_data.blockSignals(True)
        self.btn_toggle_data.setChecked(vis)
        self.btn_toggle_data.blockSignals(False)
        self._save_splitter_sizes()
        if persist:
            self._save_settings()

    def _toggle_data_panel_shortcut(self) -> None:
        self._set_data_panel_visible(not self.file_panel.isVisible())

    def _toggle_all_parameter_popups_shortcut(self) -> None:
        if self._use_pg_dockarea_pre_layout:
            any_open = any(
                bool(dock.isVisible())
                for key, dock in self._pre_dockarea_docks.items()
                if key in self._section_buttons
            )
            if any_open:
                for key in self._section_buttons.keys():
                    self._set_pre_dockarea_visible(key, False)
                    self._set_section_button_checked(key, False)
                self._last_opened_section = None
                self._update_pre_drawer_visibility()
                self._save_panel_layout_state()
                return
            self._toggle_section_shortcut("output")
            return
        any_open = any(d.isVisible() for d in self._section_docks.values())
        if any_open:
            for key, dock in self._section_docks.items():
                dock.hide()
                self._set_section_button_checked(key, False)
            self._last_opened_section = None
            return
        self._toggle_section_shortcut("output")

    def _toggle_section_shortcut(self, key: str) -> None:
        btn = self._section_buttons.get(key)
        if btn is None:
            return
        next_state = not btn.isChecked()
        self._set_section_button_checked(key, next_state)
        self._toggle_section_popup(key, next_state)

    def _close_focused_popup(self) -> None:
        if self._use_pg_dockarea_pre_layout:
            if self._last_opened_section:
                dock = self._pre_dockarea_dock(self._last_opened_section)
                if dock is not None and dock.isVisible():
                    dock.hide()
                    self._set_section_button_checked(self._last_opened_section, False)
                    self._last_opened_section = None
                    self._update_pre_drawer_visibility()
                    self._save_panel_layout_state()
            return
        fw = QtWidgets.QApplication.focusWidget()
        while fw is not None and not isinstance(fw, QtWidgets.QDockWidget):
            fw = fw.parentWidget()
        if isinstance(fw, QtWidgets.QDockWidget):
            fw.close()
            return
        if self._last_opened_section:
            dock = self._section_docks.get(self._last_opened_section)
            if dock is not None and dock.isVisible():
                dock.close()

    def _is_text_entry_focused(self) -> bool:
        fw = QtWidgets.QApplication.focusWidget()
        if fw is None:
            return False
        if isinstance(fw, QtWidgets.QAbstractButton):
            return False
        if isinstance(fw, (QtWidgets.QLineEdit, QtWidgets.QPlainTextEdit, QtWidgets.QTextEdit)):
            return True
        if isinstance(fw, (QtWidgets.QSpinBox, QtWidgets.QDoubleSpinBox, QtWidgets.QAbstractSpinBox)):
            return True
        if isinstance(fw, QtWidgets.QComboBox) and fw.isEditable():
            return True
        parent = fw.parentWidget()
        while parent is not None:
            if isinstance(parent, (QtWidgets.QSpinBox, QtWidgets.QDoubleSpinBox, QtWidgets.QAbstractSpinBox)):
                return True
            if isinstance(parent, QtWidgets.QComboBox) and parent.isEditable():
                return True
            parent = parent.parentWidget()
        return False

    def _focus_first_editable(self, root: Optional[QtWidgets.QWidget]) -> None:
        if root is None:
            return
        editable_types = (
            QtWidgets.QLineEdit,
            QtWidgets.QPlainTextEdit,
            QtWidgets.QTextEdit,
            QtWidgets.QSpinBox,
            QtWidgets.QDoubleSpinBox,
            QtWidgets.QAbstractSpinBox,
            QtWidgets.QComboBox,
        )
        for w in root.findChildren(QtWidgets.QWidget):
            if not isinstance(w, editable_types):
                continue
            if not w.isVisible() or not w.isEnabled():
                continue
            if isinstance(w, QtWidgets.QComboBox) and not w.isEditable():
                continue
            try:
                w.setFocus(QtCore.Qt.FocusReason.TabFocusReason)
            except Exception:
                continue
            if isinstance(w, QtWidgets.QAbstractSpinBox):
                le = w.lineEdit()
                if le is not None:
                    le.selectAll()
            elif isinstance(w, QtWidgets.QLineEdit):
                w.selectAll()
            return

    def _bind_shortcut(
        self,
        sequence: str,
        callback: Callable[[], None],
        *,
        require_non_text_focus: bool = False,
    ) -> None:
        shortcut = QtGui.QShortcut(QtGui.QKeySequence(sequence), self)
        shortcut.setContext(QtCore.Qt.ShortcutContext.WidgetWithChildrenShortcut)

        def _on_activated() -> None:
            if self.tabs.currentWidget() is not self.pre_tab:
                return
            if self._is_text_entry_focused():
                return
            callback()

        shortcut.activated.connect(_on_activated)
        self._shortcuts.append(shortcut)

    def _init_shortcuts(self) -> None:
        # Keyboard-first workflow for preprocessing actions.
        self._bind_shortcut("Ctrl+O", self._open_files_dialog)
        self._bind_shortcut("Ctrl+Shift+O", self._open_folder_dialog)
        self._bind_shortcut("Delete", self.file_panel._remove_selected_files, require_non_text_focus=True)
        self._bind_shortcut("Ctrl+Return", self._trigger_preview, require_non_text_focus=True)
        self._bind_shortcut("Ctrl+Enter", self._trigger_preview, require_non_text_focus=True)
        self._bind_shortcut("Ctrl+E", self._export_selected_or_all, require_non_text_focus=True)
        self._bind_shortcut("Ctrl+K", lambda: self._toggle_section_shortcut("artifacts"), require_non_text_focus=True)
        self._bind_shortcut("Ctrl+F", lambda: self._toggle_section_shortcut("filtering"), require_non_text_focus=True)
        self._bind_shortcut("Ctrl+B", lambda: self._toggle_section_shortcut("baseline"), require_non_text_focus=True)
        self._bind_shortcut("Ctrl+M", lambda: self._toggle_section_shortcut("output"), require_non_text_focus=True)
        self._bind_shortcut("Ctrl+Q", self._run_qc_dialog, require_non_text_focus=True)
        self._bind_shortcut("Ctrl+Shift+Q", self._run_batch_qc, require_non_text_focus=True)
        self._bind_shortcut("Ctrl+L", self.param_panel._load_config, require_non_text_focus=True)
        self._bind_shortcut("Ctrl+S", self.param_panel._save_config, require_non_text_focus=True)
        self._bind_shortcut("Ctrl+D", self._toggle_data_panel_shortcut, require_non_text_focus=True)
        self._bind_shortcut("Ctrl+P", self._toggle_all_parameter_popups_shortcut, require_non_text_focus=True)
        self._bind_shortcut("A", self._assign_pending_box_to_artifact, require_non_text_focus=True)
        self._bind_shortcut("C", self._assign_pending_box_to_cut, require_non_text_focus=True)
        self._bind_shortcut("S", self._assign_pending_box_to_section, require_non_text_focus=True)
        self._bind_shortcut("Escape", self._close_focused_popup, require_non_text_focus=True)

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
            # QSettings may return serialized bytearrays as text with some backends.
            try:
                return QtCore.QByteArray.fromBase64(value.encode("utf-8"))
            except Exception:
                return None
        return None

    def _qbytearray_to_b64(self, value: Optional[QtCore.QByteArray]) -> str:
        if value is None:
            return ""
        try:
            if value.isEmpty():
                return ""
            return bytes(value.toBase64()).decode("ascii")
        except Exception:
            return ""

    def _b64_to_qbytearray(self, value: object) -> Optional[QtCore.QByteArray]:
        if not value:
            return None
        if isinstance(value, QtCore.QByteArray):
            return value
        if isinstance(value, (bytes, bytearray)):
            try:
                return QtCore.QByteArray.fromBase64(bytes(value))
            except Exception:
                return None
        if isinstance(value, str):
            try:
                return QtCore.QByteArray.fromBase64(value.encode("ascii"))
            except Exception:
                return None
        return None

    def _dock_state_prefix_presence(self, state: QtCore.QByteArray) -> Tuple[bool, bool]:
        """
        Return (has_pre_prefix, has_post_prefix) for a Qt dock-state blob.
        Object names may be serialized as ASCII or UTF-16LE.
        """
        try:
            raw = bytes(state)
        except Exception:
            return False, False
        pre_ascii = b"pre."
        post_ascii = b"post."
        pre_utf16 = "pre.".encode("utf-16-le")
        post_utf16 = "post.".encode("utf-16-le")
        has_pre = (pre_ascii in raw) or (pre_utf16 in raw)
        has_post = (post_ascii in raw) or (post_utf16 in raw)
        return has_pre, has_post

    def _is_tab_scoped_dock_state(self, tab_name: str, state: QtCore.QByteArray) -> bool:
        if tab_name not in {"pre", "post"}:
            return False
        if state is None or state.isEmpty():
            return False
        has_pre, has_post = self._dock_state_prefix_presence(state)
        if tab_name == "pre" and has_post:
            return False
        if tab_name == "post" and has_pre:
            return False
        return True

    def _migrate_legacy_dock_state_settings(self) -> None:
        """
        Drop legacy full-window dock blobs that reference old object names.
        New snapshots are tab-scoped and use pre./post. dock prefixes.
        """
        try:
            if self.settings.contains("pre_main_dock_state_v3") and not self.settings.contains(_PRE_DOCK_STATE_KEY):
                self.settings.remove("pre_main_dock_state_v3")
            if self.settings.contains("post_main_dock_state_v3") and not self.settings.contains(_POST_DOCK_STATE_KEY):
                self.settings.remove("post_main_dock_state_v3")
            pre_state = self._to_qbytearray(self.settings.value(_PRE_DOCK_STATE_KEY, None))
            if pre_state is not None and not pre_state.isEmpty():
                if not self._is_tab_scoped_dock_state("pre", pre_state):
                    self.settings.remove(_PRE_DOCK_STATE_KEY)
            post_state = self._to_qbytearray(self.settings.value(_POST_DOCK_STATE_KEY, None))
            if post_state is not None and not post_state.isEmpty():
                if not self._is_tab_scoped_dock_state("post", post_state):
                    self.settings.remove(_POST_DOCK_STATE_KEY)
        except Exception:
            pass

    def _panel_config_json_path(self) -> str:
        base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        return os.path.join(base_dir, "panel_layout.json")

    def _load_panel_config_json_into_settings(self) -> None:
        """Load panel layout JSON into QSettings so existing restore logic can use it."""
        path = self._panel_config_json_path()
        if not os.path.isfile(path):
            return
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            return
        self._load_panel_config_payload_into_settings(data)

    def _load_panel_config_payload_into_settings(self, data: object) -> None:
        """Load a panel layout payload (same schema as panel_layout.json) into QSettings."""
        if not isinstance(data, dict):
            return

        try:
            layout_version = int(data.get("version", 1))
        except Exception:
            layout_version = 1
        # Snapshot blobs are considered stable starting from layout schema v3.
        # v2 files may contain mixed pre/post docks from legacy full-window capture.
        allow_snapshot_blobs = layout_version >= 3

        pre = data.get("pre", {}) if isinstance(data.get("pre"), dict) else {}
        post = data.get("post", {}) if isinstance(data.get("post"), dict) else {}

        try:
            if "pre_data_panel_visible" in pre:
                self.settings.setValue("pre_data_panel_visible", bool(pre["pre_data_panel_visible"]))
            if "pre_splitter_sizes" in pre and isinstance(pre["pre_splitter_sizes"], list):
                self.settings.setValue("pre_splitter_sizes", [int(x) for x in pre["pre_splitter_sizes"]])
            if allow_snapshot_blobs and "pre_main_dock_state" in pre:
                ba = self._b64_to_qbytearray(pre.get("pre_main_dock_state"))
                if ba is not None and self._is_tab_scoped_dock_state("pre", ba):
                    self.settings.setValue(_PRE_DOCK_STATE_KEY, ba)
                else:
                    self.settings.remove(_PRE_DOCK_STATE_KEY)
            if allow_snapshot_blobs and "post_main_dock_state" in post:
                ba = self._b64_to_qbytearray(post.get("post_main_dock_state"))
                if ba is not None and self._is_tab_scoped_dock_state("post", ba):
                    self.settings.setValue(_POST_DOCK_STATE_KEY, ba)
                else:
                    self.settings.remove(_POST_DOCK_STATE_KEY)
            if "tab_groups" in pre and isinstance(pre.get("tab_groups"), list):
                self.settings.setValue(_PRE_TAB_GROUPS_KEY, json.dumps(pre.get("tab_groups")))
        except Exception:
            pass

        def _apply_section_settings(prefix: str, section_map: object) -> None:
            if not isinstance(section_map, dict):
                return
            for key, sec in section_map.items():
                if not isinstance(sec, dict):
                    continue
                base = f"{prefix}/{key}"
                if "visible" in sec:
                    self.settings.setValue(f"{base}/visible", bool(sec["visible"]))
                if "floating" in sec:
                    self.settings.setValue(f"{base}/floating", bool(sec["floating"]))
                if "area" in sec:
                    try:
                        self.settings.setValue(f"{base}/area", int(sec["area"]))
                    except Exception:
                        pass
                if "geometry" in sec:
                    ba = self._b64_to_qbytearray(sec.get("geometry"))
                    if ba is not None:
                        self.settings.setValue(f"{base}/geometry", ba)

        _apply_section_settings("pre_section_docks", pre.get("sections"))
        _apply_section_settings("post_section_docks", post.get("sections"))

        art = pre.get("artifact", {}) if isinstance(pre.get("artifact"), dict) else {}
        if art:
            base = "pre_artifact_dock_state"
            if "visible" in art:
                self.settings.setValue(f"{base}/visible", bool(art["visible"]))
            if "floating" in art:
                self.settings.setValue(f"{base}/floating", bool(art["floating"]))
            if "area" in art:
                try:
                    self.settings.setValue(f"{base}/area", int(art["area"]))
                except Exception:
                    pass
            if "geometry" in art:
                ba = self._b64_to_qbytearray(art.get("geometry"))
                if ba is not None:
                    self.settings.setValue(f"{base}/geometry", ba)

        try:
            self.settings.sync()
        except Exception:
            pass

    def _collect_panel_layout_payload(self) -> Dict[str, object]:
        """Build the panel layout payload used for both JSON persistence and config export."""
        # Ensure QSettings has the latest dock values
        try:
            self._save_panel_layout_state()
        except Exception:
            pass
        try:
            self.post_tab.flush_post_section_state_to_settings()
            self.post_tab._save_panel_layout_state()
        except Exception:
            pass

        splitter_sizes: Optional[List[int]] = None
        try:
            raw_sizes = self.settings.value("pre_splitter_sizes", None)
            if raw_sizes is not None and hasattr(raw_sizes, "__len__"):
                splitter_sizes = [int(x) for x in raw_sizes]
        except Exception:
            splitter_sizes = None

        def _read_section_settings(prefix: str, keys: List[str]) -> Dict[str, Dict[str, object]]:
            out: Dict[str, Dict[str, object]] = {}
            for key in keys:
                base = f"{prefix}/{key}"
                visible = self.settings.value(f"{base}/visible", None)
                floating = self.settings.value(f"{base}/floating", None)
                area = self.settings.value(f"{base}/area", None)
                geom = self._to_qbytearray(self.settings.value(f"{base}/geometry", None))
                left_i = _dock_area_to_int(QtCore.Qt.DockWidgetArea.LeftDockWidgetArea, 1)
                out[key] = {
                    "visible": _to_bool(visible, False) if visible is not None else False,
                    "floating": _to_bool(floating, True) if floating is not None else True,
                    "area": _dock_area_to_int(area, left_i) if area is not None else left_i,
                    "geometry": self._qbytearray_to_b64(geom),
                }
            return out

        if self._use_pg_dockarea_pre_layout and self._pre_dockarea_docks:
            pre_sections = [k for k in self._pre_dockarea_docks.keys() if k != "artifacts_list"]
        else:
            pre_sections = list(self._section_docks.keys())
        post_sections = []
        try:
            self.post_tab.ensure_section_popups_initialized()
            post_sections = list(self.post_tab.get_section_popup_keys())
        except Exception:
            post_sections = []

        pre_main = self._to_qbytearray(self.settings.value(_PRE_DOCK_STATE_KEY, None))
        post_main = self._to_qbytearray(self.settings.value(_POST_DOCK_STATE_KEY, None))
        pre_tab_groups = self._load_pre_tab_groups_from_settings()

        art_geom = self._to_qbytearray(self.settings.value("pre_artifact_dock_state/geometry", None))
        art_visible = self.settings.value("pre_artifact_dock_state/visible", None)
        art_floating = self.settings.value("pre_artifact_dock_state/floating", None)
        art_area = self.settings.value("pre_artifact_dock_state/area", None)

        data = {
            "version": 3,
            "pre": {
                "pre_data_panel_visible": _to_bool(self.settings.value("pre_data_panel_visible", True), True),
                "pre_splitter_sizes": splitter_sizes,
                "pre_main_dock_state": self._qbytearray_to_b64(pre_main),
                "tab_groups": pre_tab_groups,
                "sections": _read_section_settings("pre_section_docks", pre_sections),
                "artifact": {
                    "visible": _to_bool(art_visible, False) if art_visible is not None else False,
                    "floating": _to_bool(art_floating, False) if art_floating is not None else False,
                    "area": int(art_area) if art_area is not None else _dock_area_to_int(QtCore.Qt.DockWidgetArea.LeftDockWidgetArea, 1),
                    "geometry": self._qbytearray_to_b64(art_geom),
                },
            },
            "post": {
                "post_main_dock_state": self._qbytearray_to_b64(post_main),
                "sections": _read_section_settings("post_section_docks", post_sections),
            },
        }
        return data

    def _save_panel_config_json(self) -> None:
        """Persist current panel layout into a JSON file."""
        try:
            data = self._collect_panel_layout_payload()
            path = self._panel_config_json_path()
            with open(path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
        except Exception:
            _LOG.exception("Failed to write panel layout JSON")

    def _export_preprocessing_ui_state_for_config(self) -> Dict[str, object]:
        """Extra UI payload stored in preprocessing_config.json."""
        return {
            "artifact_overlay_visible": bool(self.param_panel.artifact_overlay_visible()),
            "artifact_thresholds_visible": bool(self.plots.artifact_thresholds_visible()),
            "export_selection": self.param_panel.export_selection().to_dict(),
            "export_channel_names": self.param_panel.export_channel_names(),
            "export_trigger_names": self.param_panel.export_trigger_names(),
            "auto_export_to_source_dir": bool(self.param_panel.auto_export_enabled()),
            "panel_layout": self._collect_panel_layout_payload(),
        }

    def _apply_panel_layout_from_settings(self) -> None:
        """
        Apply dock disposition from current QSettings values.
        Used after importing panel layout from preprocessing config files.
        """
        if self._force_fixed_dock_layouts:
            self._apply_pre_fixed_layout()
            try:
                self.post_tab.ensure_section_popups_initialized()
                if hasattr(self.post_tab, "apply_fixed_default_layout"):
                    self.post_tab.apply_fixed_default_layout()
            except Exception:
                pass
            if self.tabs.currentWidget() is self.pre_tab:
                self._enforce_only_tab_docks_visible("pre")
            else:
                self._enforce_only_tab_docks_visible("post")
            return

        self._restore_panel_layout_state()
        self._pre_snapshot_applied = False
        self._pre_snapshot_retry_attempts = 0
        self._pre_snapshot_retry_scheduled = False
        self._apply_pre_main_dock_snapshot_if_needed()
        try:
            self.post_tab.ensure_section_popups_initialized()
            self.post_tab._restore_panel_layout_state()
            self.post_tab._post_snapshot_applied = False
            self.post_tab._apply_post_main_dock_snapshot_if_needed()
        except Exception:
            pass
        if self.tabs.currentWidget() is self.pre_tab:
            self._enforce_only_tab_docks_visible("pre")
        else:
            self._enforce_only_tab_docks_visible("post")

    def _import_preprocessing_ui_state_from_config(self, ui_state: Dict[str, object]) -> None:
        if not isinstance(ui_state, dict):
            return
        if "artifact_overlay_visible" in ui_state:
            visible = bool(ui_state.get("artifact_overlay_visible"))
            self.param_panel.set_artifact_overlay_visible(visible)
            self.plots.set_artifact_overlay_visible(visible)
        if "artifact_thresholds_visible" in ui_state:
            self.plots.set_artifact_thresholds_visible(bool(ui_state.get("artifact_thresholds_visible")))
        if "export_selection" in ui_state:
            self.param_panel.set_export_selection(ExportSelection.from_dict(ui_state.get("export_selection")))
        if "export_channel_names" in ui_state:
            self.param_panel.set_export_channel_names(list(ui_state.get("export_channel_names") or []))
        if "export_trigger_names" in ui_state:
            self.param_panel.set_export_trigger_names(list(ui_state.get("export_trigger_names") or []))
        if "auto_export_to_source_dir" in ui_state:
            self.param_panel.set_auto_export_enabled(_to_bool(ui_state.get("auto_export_to_source_dir"), False))
        self._update_export_summary_label()
        panel_layout = ui_state.get("panel_layout")
        if isinstance(panel_layout, dict):
            self._load_panel_config_payload_into_settings(panel_layout)
            self._apply_panel_layout_from_settings()
            self._save_panel_config_json()
        self._save_settings()

    def _sync_section_button_states_from_docks(self) -> None:
        if self._use_pg_dockarea_pre_layout:
            self._last_opened_section = None
            for key in self._section_buttons.keys():
                dock = self._pre_dockarea_dock(key)
                visible = bool(dock.isVisible()) if dock is not None else False
                self._set_section_button_checked(key, visible)
                if visible and self._last_opened_section is None:
                    self._last_opened_section = key
            return
        self._last_opened_section = None
        for key, dock in self._section_docks.items():
            vis = bool(dock.isVisible())
            self._set_section_button_checked(key, vis)
            if vis:
                self._last_opened_section = key

    def _save_panel_layout_state(self) -> None:
        """Persist popup/artifact panel visibility, docking mode, area, and geometry."""
        if not self._panel_layout_persistence_ready:
            return
        if self._is_restoring_panel_layout:
            return
        if self._suspend_panel_layout_persistence:
            return
        # Do not overwrite stored layout while preprocessing panels are hidden for tab switching.
        if self._pre_popups_hidden_by_tab_switch:
            return

        if self._use_pg_dockarea_pre_layout:
            self._save_pre_dockarea_layout_state()
            try:
                self.settings.sync()
            except Exception:
                pass
            return

        # Per-dock persistence is isolated so one faulty dock payload cannot drop all others.
        for key, dock in self._section_docks.items():
            try:
                base = f"pre_section_docks/{key}"
                cached = (
                    self._pre_section_state_before_tab_switch.get(key, {})
                    if self._pre_popups_hidden_by_tab_switch
                    else {}
                )
                visible = bool(cached.get("visible", dock.isVisible()))
                floating = bool(cached.get("floating", dock.isFloating()))
                left_i = _dock_area_to_int(QtCore.Qt.DockWidgetArea.LeftDockWidgetArea, 1)
                area_val = _dock_area_to_int(cached.get("area", self.dockWidgetArea(dock)), left_i)
                geom = cached.get("geometry", dock.saveGeometry())
                self.settings.setValue(f"{base}/visible", visible)
                self.settings.setValue(f"{base}/floating", floating)
                self.settings.setValue(f"{base}/area", area_val)
                self.settings.setValue(f"{base}/geometry", geom)
            except Exception:
                continue

        if isinstance(self.art_dock, QtWidgets.QDockWidget):
            try:
                base = "pre_artifact_dock_state"
                cached = self._pre_artifact_state_before_tab_switch if self._pre_popups_hidden_by_tab_switch else {}
                visible = bool(cached.get("visible", self.art_dock.isVisible()))
                floating = bool(cached.get("floating", self.art_dock.isFloating()))
                left_i = _dock_area_to_int(QtCore.Qt.DockWidgetArea.LeftDockWidgetArea, 1)
                area_val = _dock_area_to_int(cached.get("area", self.dockWidgetArea(self.art_dock)), left_i)
                geom = cached.get("geometry", self.art_dock.saveGeometry())
                self.settings.setValue(f"{base}/visible", visible)
                self.settings.setValue(f"{base}/floating", floating)
                self.settings.setValue(f"{base}/area", area_val)
                self.settings.setValue(f"{base}/geometry", geom)
            except Exception:
                pass
        try:
            self._save_pre_tab_groups_to_settings(self._capture_pre_tab_groups_state())
        except Exception:
            pass
        try:
            self.settings.sync()
        except Exception:
            pass

    def _save_full_main_dock_state(self) -> None:
        """
        Save full main-window dock disposition (tabified/split relationships).
        This complements per-dock visibility settings.
        """
        try:
            self.settings.setValue("main_dock_state_v2", self.saveState(_DOCK_STATE_VERSION))
            self.settings.sync()
        except Exception:
            pass

    def _restore_full_main_dock_state(self) -> None:
        """
        Restore full main-window dock disposition after all docks are registered.
        """
        try:
            raw = self.settings.value("main_dock_state_v2", None)
            state = self._to_qbytearray(raw)
            if state is None or state.isEmpty():
                return
            ok = self.restoreState(state, _DOCK_STATE_VERSION)
            if not ok:
                # Drop invalid payload and fall back to per-dock restore.
                self.settings.remove("main_dock_state_v2")
                return
            try:
                self.post_tab.mark_dock_layout_restored()
            except Exception:
                pass
        except Exception:
            pass

    def _restore_panel_layout_state(self) -> None:
        """Restore popup/artifact panel layout from the previous app session."""
        if self._use_pg_dockarea_pre_layout:
            self._is_restoring_panel_layout = True
            try:
                self._setup_section_popups()
                self._restore_pre_dockarea_layout_state()
            finally:
                self._is_restoring_panel_layout = False
            return

        self._is_restoring_panel_layout = True
        for key, dock in self._section_docks.items():
            base = f"pre_section_docks/{key}"
            try:
                visible = _to_bool(self.settings.value(f"{base}/visible", False), False)
                floating = _to_bool(self.settings.value(f"{base}/floating", True), True)
                area_val = self.settings.value(
                    f"{base}/area",
                    _dock_area_to_int(QtCore.Qt.DockWidgetArea.LeftDockWidgetArea, 1),
                )
                area = self._dock_area_from_settings(area_val, QtCore.Qt.DockWidgetArea.LeftDockWidgetArea)
                geom = self._to_qbytearray(self.settings.value(f"{base}/geometry", None))

                dock.blockSignals(True)
                if bool(floating):
                    dock.setFloating(True)
                else:
                    self.addDockWidget(area, dock)
                    dock.setFloating(False)

                if geom is not None and not geom.isEmpty():
                    dock.restoreGeometry(geom)
                    self._section_popup_initialized.add(key)

                if visible:
                    dock.show()
                    if dock.isFloating() and not self._is_popup_on_screen(dock):
                        self._position_section_popup(dock)
                    self._set_section_button_checked(key, True)
                    self._last_opened_section = key
                else:
                    dock.hide()
                    self._set_section_button_checked(key, False)
            except Exception:
                continue
            finally:
                try:
                    dock.blockSignals(False)
                except Exception:
                    pass

        if isinstance(self.art_dock, QtWidgets.QDockWidget):
            try:
                base = "pre_artifact_dock_state"
                visible = _to_bool(self.settings.value(f"{base}/visible", False), False)
                floating = _to_bool(self.settings.value(f"{base}/floating", False), False)
                area_val = self.settings.value(
                    f"{base}/area",
                    _dock_area_to_int(QtCore.Qt.DockWidgetArea.LeftDockWidgetArea, 1),
                )
                area = self._dock_area_from_settings(area_val, QtCore.Qt.DockWidgetArea.LeftDockWidgetArea)
                geom = self._to_qbytearray(self.settings.value(f"{base}/geometry", None))

                if bool(floating):
                    self.art_dock.setFloating(True)
                else:
                    self.addDockWidget(area, self.art_dock)
                    self.art_dock.setFloating(False)

                if geom is not None and not geom.isEmpty():
                    self.art_dock.restoreGeometry(geom)
                self.art_dock.setVisible(bool(visible))
            except Exception:
                pass

        self._sync_section_button_states_from_docks()
        self._restore_pre_tab_groups_fallback(self._load_pre_tab_groups_from_settings())
        self._is_restoring_panel_layout = False

    def _has_saved_pre_layout_state(self) -> bool:
        try:
            if self._use_pg_dockarea_pre_layout:
                if self.settings.contains(_PRE_DOCKAREA_STATE_KEY) or self.settings.contains(_PRE_DOCKAREA_VISIBLE_KEY):
                    return True
            if self.settings.contains(_PRE_DOCK_STATE_KEY):
                return True
            if self.settings.contains("pre_artifact_dock_state/visible"):
                return True
            keys = list(self._section_docks.keys())
            if self._use_pg_dockarea_pre_layout and self._pre_dockarea_docks:
                keys = [k for k in self._pre_dockarea_docks.keys() if k != "artifacts_list"]
            for key in keys:
                if self.settings.contains(f"pre_section_docks/{key}/visible"):
                    return True
        except Exception:
            pass
        return False

    def _has_saved_post_layout_state(self) -> bool:
        try:
            if self.settings.contains(_POST_DOCK_STATE_KEY):
                return True
            self.post_tab.ensure_section_popups_initialized()
            for key in self.post_tab.get_section_popup_keys():
                if self.settings.contains(f"post_section_docks/{key}/visible"):
                    return True
        except Exception:
            pass
        return False

    # ---------------- Settings persistence ----------------

    def _restore_settings(self) -> None:
        last_dir = self.settings.value("last_open_dir", "", type=str)
        if last_dir and os.path.isdir(last_dir):
            self.file_panel.set_path_hint(last_dir)

        try:
            app_theme = self.settings.value("app_theme_mode", "dark", type=str)
        except Exception:
            app_theme = "dark"
        self._apply_app_theme(app_theme, persist=False)

        # restore params
        try:
            raw = self.settings.value("params_json", "", type=str)
            if raw:
                d = json.loads(raw)
                # One-time migration: ensure invert polarity defaults to off
                migrated = self.settings.value("invert_polarity_migrated", False, type=bool)
                if not migrated:
                    d["invert_polarity"] = False
                    self.settings.setValue("invert_polarity_migrated", True)
                    self.settings.setValue("params_json", json.dumps(d))
                p = ProcessingParams.from_dict(d)
                self.param_panel.set_params(p)
                self._update_plot_status(fs_target=float(p.target_fs_hz))
        except Exception:
            pass

        try:
            show_overlay = self.settings.value("artifact_overlay_visible", True, type=bool)
            self.param_panel.set_artifact_overlay_visible(bool(show_overlay))
            self.plots.set_artifact_overlay_visible(bool(show_overlay))
        except Exception:
            pass
        try:
            show_thresholds = self.settings.value("artifact_thresholds_visible", True, type=bool)
            self.plots.set_artifact_thresholds_visible(bool(show_thresholds))
        except Exception:
            pass
        try:
            auto_export = _to_bool(self.settings.value("auto_export_to_source_dir", False), False)
            self.param_panel.set_auto_export_enabled(auto_export)
            self._update_export_summary_label()
        except Exception:
            pass
        try:
            default_bg = "white" if self._app_theme_mode == "light" else "dark"
            plot_bg = self.settings.value("pre_plot_background", default_bg, type=str)
        except Exception:
            plot_bg = "dark"
        try:
            plot_grid = _to_bool(self.settings.value("pre_plot_grid", True), True)
        except Exception:
            plot_grid = True
        self._apply_pre_plot_style(plot_bg, plot_grid, persist=False)

        if self._force_fixed_dock_layouts:
            # Fixed mode: always enforce deterministic defaults.
            try:
                self.pre_splitter.setSizes([300, 1200])
            except Exception:
                pass
            try:
                splitter_sizes = self.settings.value("pre_splitter_sizes", None)
                if splitter_sizes is None:
                    splitter_sizes = self.settings.value("splitter_sizes", None)
                if splitter_sizes and hasattr(splitter_sizes, "__len__"):
                    vals = [int(x) for x in splitter_sizes]
                    if self._use_pg_dockarea_pre_layout:
                        if len(vals) >= 3:
                            self.pre_splitter.setSizes([vals[0], max(640, vals[1] + vals[2])])
                        elif len(vals) == 2:
                            self.pre_splitter.setSizes(vals[:2])
                    elif len(vals) >= 3:
                        left = max(260, vals[0])
                        center = max(640, vals[1] + vals[2])
                        self.pre_splitter.setSizes([left, center])
                    elif len(vals) == 2:
                        self.pre_splitter.setSizes(vals[:2])
            except Exception:
                pass
            try:
                show_data = self.settings.value("pre_data_panel_visible", False, type=bool)
            except Exception:
                show_data = False
            self._set_data_panel_visible(bool(show_data), persist=False)

            self._apply_pre_fixed_layout()

            try:
                self.post_tab.ensure_section_popups_initialized()
                if hasattr(self.post_tab, "apply_fixed_default_layout"):
                    self.post_tab.apply_fixed_default_layout()
                # Keep post docks detached while Preprocessing is active at startup.
                self._hide_dock_widgets(self.getPostDockWidgets(), remove=True)
            except Exception:
                pass
        else:
            # restore splitter sizes (2-pane layout; migrate older layouts)
            try:
                splitter_sizes = self.settings.value("pre_splitter_sizes", None)
                if splitter_sizes is None:
                    splitter_sizes = self.settings.value("splitter_sizes", None)
                if splitter_sizes and hasattr(splitter_sizes, "__len__"):
                    vals = [int(x) for x in splitter_sizes]
                    if self._use_pg_dockarea_pre_layout:
                        if len(vals) >= 3:
                            self.pre_splitter.setSizes([vals[0], max(640, vals[1] + vals[2])])
                        elif len(vals) == 2:
                            self.pre_splitter.setSizes(vals[:2])
                    elif len(vals) >= 3:
                        # Migrate old 3-pane [left, center, right] into [left, center+right].
                        left = max(260, vals[0])
                        center = max(640, vals[1] + vals[2])
                        self.pre_splitter.setSizes([left, center])
                    elif len(vals) == 2:
                        self.pre_splitter.setSizes(vals[:2])
            except Exception:
                pass

            # restore data panel visibility
            try:
                show_data = self.settings.value("pre_data_panel_visible", True, type=bool)
                self._set_data_panel_visible(bool(show_data), persist=False)
            except Exception:
                pass

            # restore panel layout/disposition (floating popups + artifacts dock).
            self._restore_panel_layout_state()
            # Apply default preprocessing dock layout if no saved layout exists.
            self._apply_pre_default_layout_if_missing()
            # Apply saved preprocessing dock snapshot at startup.
            self._apply_pre_main_dock_snapshot_if_needed()

        # restore last selected main tab
        try:
            idx = self.settings.value("main_current_tab", 0, type=int)
            if isinstance(idx, int) and 0 <= idx < self.tabs.count():
                self.tabs.setCurrentIndex(idx)
        except Exception:
            pass

    def _save_settings(self) -> None:
        try:
            last_dir = self.file_panel.current_dir_hint()
            if last_dir:
                self.settings.setValue("last_open_dir", last_dir)
        except Exception:
            pass

        try:
            p = self.param_panel.get_params()
            self.settings.setValue("params_json", json.dumps(p.to_dict()))
        except Exception:
            pass

        try:
            self.settings.setValue("artifact_overlay_visible", bool(self.param_panel.artifact_overlay_visible()))
        except Exception:
            pass
        try:
            self.settings.setValue("artifact_thresholds_visible", bool(self.plots.artifact_thresholds_visible()))
        except Exception:
            pass
        try:
            self.settings.setValue("auto_export_to_source_dir", bool(self.param_panel.auto_export_enabled()))
        except Exception:
            pass
        try:
            self.settings.setValue("pre_plot_background", str(self.plots.plot_background_mode()))
            self.settings.setValue("pre_plot_grid", bool(self.plots.plot_grid_visible()))
        except Exception:
            pass
        try:
            self.settings.setValue("app_theme_mode", str(self._app_theme_mode))
        except Exception:
            pass

        try:
            self.settings.setValue("pre_data_panel_visible", bool(self.file_panel.isVisible()))
        except Exception:
            pass
        try:
            self.settings.setValue("main_current_tab", int(self.tabs.currentIndex()))
        except Exception:
            pass
        try:
            self.settings.sync()
        except Exception:
            pass

    def _save_splitter_sizes(self, *_args) -> None:
        """Save the current splitter sizes to settings."""
        try:
            if hasattr(self, "pre_splitter") and self.pre_splitter:
                sizes = self.pre_splitter.sizes()
                self.settings.setValue("pre_splitter_sizes", sizes)
                self.settings.setValue("splitter_sizes", sizes)
        except Exception:
            pass

    # ---------------- File loading ----------------

    def _load_recent_preprocessing_files(self) -> List[str]:
        raw = self.settings.value("recent_pre_files", "[]", type=str)
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

    def _save_recent_preprocessing_files(self, paths: List[str]) -> None:
        try:
            self.settings.setValue("recent_pre_files", json.dumps(paths))
        except Exception:
            pass

    def _push_recent_preprocessing_files(self, paths: List[str], max_items: int = 15) -> None:
        if not paths:
            return
        existing = self._load_recent_preprocessing_files()
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
        self._save_recent_preprocessing_files(merged[:max_items])

    def _refresh_recent_preprocessing_menu(self) -> None:
        if not hasattr(self, "menu_workflow_load_recent"):
            return
        menu = self.menu_workflow_load_recent
        menu.clear()
        recent = self._load_recent_preprocessing_files()
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
                act.triggered.connect(lambda _checked=False, p=path: self._add_files([p]))
            else:
                missing.append(path)
        menu.addSeparator()
        act_clear = menu.addAction("Clear recent")
        act_clear.triggered.connect(lambda: self._save_recent_preprocessing_files([]))
        if missing:
            act_prune = menu.addAction("Remove missing")
            act_prune.triggered.connect(self._prune_recent_preprocessing_files)

    def _prune_recent_preprocessing_files(self) -> None:
        recent = self._load_recent_preprocessing_files()
        kept = [p for p in recent if os.path.isfile(p)]
        self._save_recent_preprocessing_files(kept)

    # ---------------- Preprocessing projects ----------------

    def _preprocessing_project_state_exists(self) -> bool:
        return bool(
            self._loaded_files
            or self._manual_regions_by_key
            or self._manual_exclude_by_key
            or self._metadata_by_key
            or self._cutout_regions_by_key
            or self._sections_by_key
        )

    def _confirm_discard_preprocessing_project(self, title: str) -> bool:
        if not self._preprocessing_project_state_exists():
            return True
        reply = QtWidgets.QMessageBox.question(
            self,
            title,
            "Discard the current preprocessing project state?",
            QtWidgets.QMessageBox.StandardButton.Yes | QtWidgets.QMessageBox.StandardButton.No,
            QtWidgets.QMessageBox.StandardButton.No,
        )
        return reply == QtWidgets.QMessageBox.StandardButton.Yes

    def _clear_preprocessing_project_state(self) -> None:
        try:
            self._preview_timer.stop()
        except Exception:
            pass
        self._job_counter += 1
        self._latest_job_id = self._job_counter
        self._preview_preserve_view_by_job.clear()
        self._loaded_files.clear()
        self._current_path = None
        self._current_channel = None
        self._current_trigger = None
        self._manual_regions_by_key.clear()
        self._manual_exclude_by_key.clear()
        self._auto_regions_by_key.clear()
        self._metadata_by_key.clear()
        self._cutout_regions_by_key.clear()
        self._sections_by_key.clear()
        self._pending_box_region_by_key.clear()
        self._last_processed.clear()
        self._csv_channel_mapping_session = None
        self._csv_mappings_by_path.clear()

        self.file_panel.list_files.clear()
        self.file_panel.set_available_channels([])
        self.file_panel.set_available_triggers([])
        self.param_panel.set_available_export_channels([])
        self.param_panel.set_available_export_triggers([])
        for ed in (self.file_panel.edit_time_start, self.file_panel.edit_time_end):
            ed.blockSignals(True)
            try:
                ed.clear()
            finally:
                ed.blockSignals(False)
        self.artifact_panel.set_regions([])
        self.artifact_panel.set_auto_regions([])
        self.plots.set_title("No file loaded")
        self.plots.set_log("")
        self.plots.show_raw()
        self._update_plot_status()
        self.post_tab.set_current_source_label("", "")

    def _new_preprocessing_project(self) -> None:
        if not self._confirm_discard_preprocessing_project("New preprocessing project"):
            return
        self._clear_preprocessing_project_state()
        self._pre_project_path = None
        try:
            self.post_tab.reset_for_new_preprocessing_project()
        except Exception:
            pass
        self._show_status_message("Started a new preprocessing project.", 5000)

    def _keyed_regions_to_project(
        self,
        mapping: Dict[Tuple[str, str], List[Tuple[float, float]]],
    ) -> List[Dict[str, object]]:
        out: List[Dict[str, object]] = []
        for (path, channel), regions in mapping.items():
            clean_regions = []
            for a, b in regions or []:
                try:
                    clean_regions.append([float(a), float(b)])
                except Exception:
                    continue
            if clean_regions:
                out.append({"path": path, "channel": channel, "regions": clean_regions})
        return out

    def _project_to_keyed_regions(self, data: object) -> Dict[Tuple[str, str], List[Tuple[float, float]]]:
        out: Dict[Tuple[str, str], List[Tuple[float, float]]] = {}
        if not isinstance(data, list):
            return out
        for entry in data:
            if not isinstance(entry, dict):
                continue
            path = str(entry.get("path") or "").strip()
            channel = str(entry.get("channel") or "").strip()
            if not path or not channel:
                continue
            regions: List[Tuple[float, float]] = []
            for item in entry.get("regions") or []:
                try:
                    a, b = item
                    regions.append((float(a), float(b)))
                except Exception:
                    continue
            if regions:
                regions.sort(key=lambda x: x[0])
                out[(path, channel)] = regions
        return out

    def _keyed_dict_to_project(self, mapping: Dict[Tuple[str, str], Dict[str, str]]) -> List[Dict[str, object]]:
        out: List[Dict[str, object]] = []
        for (path, channel), value in mapping.items():
            if isinstance(value, dict) and value:
                out.append({"path": path, "channel": channel, "value": dict(value)})
        return out

    def _project_to_keyed_dict(self, data: object) -> Dict[Tuple[str, str], Dict[str, str]]:
        out: Dict[Tuple[str, str], Dict[str, str]] = {}
        if not isinstance(data, list):
            return out
        for entry in data:
            if not isinstance(entry, dict):
                continue
            path = str(entry.get("path") or "").strip()
            channel = str(entry.get("channel") or "").strip()
            value = entry.get("value")
            if path and channel and isinstance(value, dict):
                out[(path, channel)] = {str(k): str(v) for k, v in value.items()}
        return out

    def _sections_to_project(self) -> List[Dict[str, object]]:
        out: List[Dict[str, object]] = []
        for (path, channel), sections in self._sections_by_key.items():
            if not sections:
                continue
            try:
                clean_sections = json.loads(json.dumps(sections))
            except Exception:
                clean_sections = []
            if clean_sections:
                out.append({"path": path, "channel": channel, "sections": clean_sections})
        return out

    def _project_to_sections(self, data: object) -> Dict[Tuple[str, str], List[Dict[str, object]]]:
        out: Dict[Tuple[str, str], List[Dict[str, object]]] = {}
        if not isinstance(data, list):
            return out
        for entry in data:
            if not isinstance(entry, dict):
                continue
            path = str(entry.get("path") or "").strip()
            channel = str(entry.get("channel") or "").strip()
            sections = entry.get("sections")
            if path and channel and isinstance(sections, list):
                out[(path, channel)] = [s for s in sections if isinstance(s, dict)]
        return out

    def _preprocessing_config_payload(self) -> Dict[str, object]:
        params = self.param_panel.get_params()
        return {
            "artifact_detection_enabled": bool(self.param_panel.cb_artifact.isChecked()),
            "artifact_overlay_visible": bool(self.param_panel.cb_show_artifact_overlay.isChecked()),
            "filtering_enabled": bool(self.param_panel.cb_filtering.isChecked()),
            "parameters": params.to_dict(),
            "ui_state": self._export_preprocessing_ui_state_for_config(),
        }

    def _apply_preprocessing_config_payload(self, config: object) -> None:
        if not isinstance(config, dict):
            return
        try:
            params = config.get("parameters")
            if isinstance(params, dict):
                self.param_panel.set_params(ProcessingParams.from_dict(params))
            if "artifact_detection_enabled" in config:
                self.param_panel.cb_artifact.setChecked(bool(config.get("artifact_detection_enabled")))
            if "artifact_overlay_visible" in config:
                visible = bool(config.get("artifact_overlay_visible"))
                self.param_panel.cb_show_artifact_overlay.setChecked(visible)
                self.param_panel.set_artifact_overlay_visible(visible)
                self.plots.set_artifact_overlay_visible(visible)
            if "filtering_enabled" in config:
                self.param_panel.cb_filtering.setChecked(bool(config.get("filtering_enabled")))
            ui_state = config.get("ui_state")
            if isinstance(ui_state, dict):
                self._import_preprocessing_ui_state_from_config(ui_state)
        except Exception:
            _LOG.exception("Failed to apply preprocessing project config")

    def _collect_preprocessing_project_payload(self) -> Dict[str, object]:
        selected_paths = self._selected_paths()
        start_s, end_s = self._time_window_bounds()
        return {
            "project_type": _PRE_PROJECT_TYPE,
            "project_version": _PRE_PROJECT_VERSION,
            "source_paths": self.file_panel.all_paths(),
            "selected_paths": selected_paths,
            "current_path": self._current_path or "",
            "current_channel": self._current_channel or "",
            "current_trigger": self._current_trigger or "",
            "time_window": {"start_s": start_s, "end_s": end_s},
            "preprocessing_config": self._preprocessing_config_payload(),
            "manual_regions": self._keyed_regions_to_project(self._manual_regions_by_key),
            "manual_exclude_regions": self._keyed_regions_to_project(self._manual_exclude_by_key),
            "auto_regions": self._keyed_regions_to_project(self._auto_regions_by_key),
            "metadata": self._keyed_dict_to_project(self._metadata_by_key),
            "cutout_regions": self._keyed_regions_to_project(self._cutout_regions_by_key),
            "sections": self._sections_to_project(),
            "csv_mapping_session": dict(self._csv_channel_mapping_session or {}),
            "csv_mappings_by_path": [
                {"path": path, "mapping": dict(mapping)}
                for path, mapping in self._csv_mappings_by_path.items()
                if path and isinstance(mapping, dict)
            ],
        }

    def _save_preprocessing_project_file(self) -> None:
        start_dir = (
            os.path.dirname(self._pre_project_path)
            if self._pre_project_path
            else (self.file_panel.current_dir_hint() or self.settings.value("last_open_dir", "", type=str) or os.getcwd())
        )
        default_name = os.path.basename(self._pre_project_path) if self._pre_project_path else "pyber_preprocessing_project.json"
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self,
            "Save preprocessing project",
            os.path.join(start_dir, default_name),
            "pyBer preprocessing project (*.json)",
        )
        if not path:
            return
        if not path.lower().endswith(".json"):
            path = f"{path}.json"
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(self._collect_preprocessing_project_payload(), f, indent=2)
        except Exception as exc:
            QtWidgets.QMessageBox.warning(self, "Save project", f"Could not save preprocessing project:\n{exc}")
            return
        self._pre_project_path = path
        self._show_status_message(f"Preprocessing project saved: {os.path.basename(path)}", 5000)

    def _open_preprocessing_project_file(self) -> None:
        start_dir = self.file_panel.current_dir_hint() or self.settings.value("last_open_dir", "", type=str) or os.getcwd()
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Open preprocessing project",
            start_dir,
            "pyBer preprocessing project (*.json);;All files (*.*)",
        )
        if not path:
            return
        self._load_preprocessing_project_from_path(path)

    def _load_preprocessing_project_from_path(self, path: str) -> None:
        if not self._confirm_discard_preprocessing_project("Open preprocessing project"):
            return
        try:
            with open(path, "r", encoding="utf-8") as f:
                payload = json.load(f)
        except Exception as exc:
            QtWidgets.QMessageBox.warning(self, "Open project", f"Could not read preprocessing project:\n{exc}")
            return
        if not isinstance(payload, dict) or payload.get("project_type") != _PRE_PROJECT_TYPE:
            QtWidgets.QMessageBox.warning(self, "Open project", "This file is not a pyBer preprocessing project.")
            return

        self._clear_preprocessing_project_state()
        self._pre_project_path = path
        try:
            self.post_tab.reset_for_new_preprocessing_project()
        except Exception:
            pass
        self._apply_preprocessing_config_payload(payload.get("preprocessing_config"))

        session_mapping = payload.get("csv_mapping_session")
        if isinstance(session_mapping, dict):
            self._csv_channel_mapping_session = {str(k): str(v) for k, v in session_mapping.items()}
        for entry in payload.get("csv_mappings_by_path") or []:
            if not isinstance(entry, dict):
                continue
            src_path = str(entry.get("path") or "").strip()
            mapping = entry.get("mapping")
            if src_path and isinstance(mapping, dict):
                self._csv_mappings_by_path[src_path] = {str(k): str(v) for k, v in mapping.items()}

        source_paths = [str(p) for p in payload.get("source_paths") or [] if str(p or "").strip()]
        existing_paths = [p for p in source_paths if os.path.isfile(p)]
        missing_paths = [p for p in source_paths if p not in existing_paths]
        if existing_paths:
            self._add_files(existing_paths, select_after=False)

        self._manual_regions_by_key = self._project_to_keyed_regions(payload.get("manual_regions"))
        self._manual_exclude_by_key = self._project_to_keyed_regions(payload.get("manual_exclude_regions"))
        self._auto_regions_by_key = self._project_to_keyed_regions(payload.get("auto_regions"))
        self._metadata_by_key = self._project_to_keyed_dict(payload.get("metadata"))
        self._cutout_regions_by_key = self._project_to_keyed_regions(payload.get("cutout_regions"))
        self._sections_by_key = self._project_to_sections(payload.get("sections"))

        tw = payload.get("time_window") if isinstance(payload.get("time_window"), dict) else {}
        for ed, value in (
            (self.file_panel.edit_time_start, tw.get("start_s")),
            (self.file_panel.edit_time_end, tw.get("end_s")),
        ):
            ed.blockSignals(True)
            try:
                ed.setText("" if value is None else f"{float(value):.6g}")
            except Exception:
                ed.setText("")
            finally:
                ed.blockSignals(False)

        self._current_path = str(payload.get("current_path") or "") or None
        self._current_channel = str(payload.get("current_channel") or "") or None
        self._current_trigger = str(payload.get("current_trigger") or "") or None
        selected_paths = [str(p) for p in payload.get("selected_paths") or [] if str(p or "").strip()]
        self._restore_file_selection(selected_paths, self._current_path)
        self._push_recent_preprocessing_files(existing_paths)
        self._on_file_selection_changed()

        if missing_paths:
            QtWidgets.QMessageBox.warning(
                self,
                "Open project",
                "Some linked input files are missing and were skipped:\n" + "\n".join(missing_paths[:12]),
            )
        self._show_status_message(f"Preprocessing project loaded: {os.path.basename(path)}", 5000)

    def _restore_file_selection(self, selected_paths: List[str], current_path: Optional[str]) -> None:
        selected = set(selected_paths or [])
        if current_path:
            selected.add(current_path)
        list_widget = self.file_panel.list_files
        list_widget.blockSignals(True)
        try:
            target_row = -1
            for i in range(list_widget.count()):
                item = list_widget.item(i)
                if item is None:
                    continue
                path = str(item.data(QtCore.Qt.ItemDataRole.UserRole) or "")
                item.setSelected(path in selected)
                if current_path and path == current_path:
                    target_row = i
            if target_row >= 0:
                list_widget.setCurrentRow(target_row)
            elif list_widget.count() and not selected:
                list_widget.setCurrentRow(0)
                item0 = list_widget.item(0)
                if item0 is not None:
                    item0.setSelected(True)
        finally:
            list_widget.blockSignals(False)

    # ---------------- Raw CSV preprocessing import ----------------

    def _normalize_csv_column_name(self, value: object) -> str:
        return "".join(ch.lower() for ch in str(value or "") if ch.isalnum())

    def _is_csv_time_column(self, value: object) -> bool:
        norm = self._normalize_csv_column_name(value)
        return norm in {"time", "t", "timestamp", "times", "timesec", "times", "timems"} or "timestamp" in norm

    def _parse_csv_float(self, value: object) -> float:
        text = str(value or "").strip()
        if not text or text.lower() in {"nan", "none", "null", "na"}:
            return np.nan
        try:
            return float(text)
        except Exception:
            pass
        try:
            return float(text.replace(" ", "").replace(",", "."))
        except Exception:
            pass
        return coerce_time_value(text)

    def _clean_csv_row(self, row: List[str]) -> List[str]:
        out = [str(cell or "").strip() for cell in row]
        while out and not out[-1]:
            out.pop()
        return out

    def _read_csv_rows(self, path: str) -> List[List[str]]:
        import csv

        last_error: Optional[Exception] = None
        for encoding in ("utf-8-sig", "utf-8", "cp1252"):
            try:
                with open(path, "r", newline="", encoding=encoding) as f:
                    return [self._clean_csv_row(row) for row in csv.reader(f)]
            except UnicodeDecodeError as exc:
                last_error = exc
                continue
        if last_error is not None:
            raise last_error
        return []

    def _find_raw_csv_table(self, rows: List[List[str]]) -> Tuple[List[str], List[List[str]]]:
        cleaned = [row for row in (self._clean_csv_row(r) for r in rows) if row and any(cell for cell in row)]
        for idx, row in enumerate(cleaned):
            if len(row) < 2:
                continue
            if any(self._is_csv_time_column(cell) for cell in row):
                headers = [h.strip() or f"Column {i + 1}" for i, h in enumerate(row)]
                return headers, cleaned[idx + 1 :]

        # Fallback for CSVs without a canonical time header: find the first row whose
        # following line looks numeric in at least two columns.
        for idx, row in enumerate(cleaned[:-1]):
            if len(row) < 2:
                continue
            next_row = cleaned[idx + 1]
            numeric_count = 0
            for col_idx in range(min(len(row), len(next_row))):
                if np.isfinite(self._parse_csv_float(next_row[col_idx])):
                    numeric_count += 1
            if numeric_count >= 2:
                headers = [h.strip() or f"Column {i + 1}" for i, h in enumerate(row)]
                return headers, cleaned[idx + 1 :]

        raise ValueError("Could not find a CSV header row with a time column.")

    def _csv_numeric_headers(self, headers: List[str], rows: List[List[str]]) -> List[str]:
        out: List[str] = []
        sample = rows[: min(len(rows), 1000)]
        min_count = 1 if len(sample) < 10 else 3
        for idx, name in enumerate(headers):
            if self._is_csv_time_column(name):
                continue
            count = 0
            for row in sample:
                if idx < len(row) and np.isfinite(self._parse_csv_float(row[idx])):
                    count += 1
            if count >= min_count:
                out.append(name)
        return out

    def _resolve_csv_column_name(self, headers: List[str], wanted: object) -> str:
        text = str(wanted or "").strip()
        if not text:
            return ""
        for h in headers:
            if h == text:
                return h
        for h in headers:
            if h.lower() == text.lower():
                return h
        norm = self._normalize_csv_column_name(text)
        for h in headers:
            if self._normalize_csv_column_name(h) == norm:
                return h
        return ""

    def _sanitize_csv_mapping_for_headers(
        self,
        mapping: object,
        headers: List[str],
        *,
        require_all: bool = True,
    ) -> Optional[Dict[str, str]]:
        if not isinstance(mapping, dict):
            return None

        def _resolve(name_key: str, index_key: str) -> str:
            col = self._resolve_csv_column_name(headers, mapping.get(name_key))
            if col:
                return col
            try:
                idx = int(mapping.get(index_key))
            except Exception:
                idx = -1
            if 0 <= idx < len(headers):
                return headers[idx]
            return ""

        time_col = _resolve("time", "time_index")
        raw1 = _resolve("raw1", "raw1_index")
        ref = _resolve("reference", "reference_index")
        if require_all and (not time_col or not raw1 or not ref):
            return None
        raw2 = _resolve("raw2", "raw2_index")
        trigger = _resolve("trigger", "trigger_index")
        unit = str(mapping.get("time_unit") or "Auto").strip()
        if unit.lower().startswith("milli"):
            unit = "Milliseconds"
        elif unit.lower().startswith("sec"):
            unit = "Seconds"
        else:
            unit = "Auto"
        return {
            "time": time_col,
            "time_unit": unit,
            "raw1": raw1,
            "raw2": raw2,
            "reference": ref,
            "trigger": trigger,
            "time_index": str(self._csv_column_index(headers, time_col)),
            "raw1_index": str(self._csv_column_index(headers, raw1)),
            "raw2_index": str(self._csv_column_index(headers, raw2)),
            "reference_index": str(self._csv_column_index(headers, ref)),
            "trigger_index": str(self._csv_column_index(headers, trigger)),
        }

    def _infer_csv_mapping_defaults(self, headers: List[str], numeric_headers: List[str]) -> Dict[str, str]:
        time_col = next((h for h in headers if self._is_csv_time_column(h)), headers[0] if headers else "")

        def _has_any(name: str, terms: Tuple[str, ...]) -> bool:
            norm = self._normalize_csv_column_name(name)
            return any(term in norm for term in terms)

        candidates = [h for h in numeric_headers if h != time_col]
        ref = next((h for h in candidates if _has_any(h, ("410", "405", "isob", "isos", "ref"))), "")
        raw_priority = [h for h in candidates if h != ref and _has_any(h, ("470", "465", "signal", "sig"))]
        raw_rest = [h for h in candidates if h != ref and h not in raw_priority and not _has_any(h, ("event", "dio", "ttl", "digital"))]
        raw_candidates = raw_priority + raw_rest
        if not ref and len(candidates) >= 2:
            ref = candidates[1] if raw_candidates and candidates[1] != raw_candidates[0] else candidates[0]
        raw1 = raw_candidates[0] if raw_candidates else next((h for h in candidates if h != ref), "")
        raw2 = raw_candidates[1] if len(raw_candidates) > 1 else ""
        trigger = next((h for h in candidates if _has_any(h, ("event", "dio", "ttl", "digital"))), "")
        return {
            "time": time_col,
            "time_unit": "Auto",
            "raw1": raw1,
            "raw2": raw2,
            "reference": ref,
            "trigger": trigger,
        }

    def _csv_mapping_for_file(
        self,
        path: str,
        headers: List[str],
        numeric_headers: List[str],
    ) -> Optional[Dict[str, str]]:
        for candidate in (
            self._csv_mappings_by_path.get(path),
            self._csv_channel_mapping_session,
        ):
            resolved = self._sanitize_csv_mapping_for_headers(candidate, headers)
            if resolved is not None:
                self._csv_mappings_by_path[path] = dict(resolved)
                if self._csv_channel_mapping_session is None:
                    self._csv_channel_mapping_session = dict(resolved)
                return resolved

        defaults = self._infer_csv_mapping_defaults(headers, numeric_headers)
        partial = self._sanitize_csv_mapping_for_headers(self._csv_channel_mapping_session, headers, require_all=False)
        if partial:
            for key, value in partial.items():
                if value:
                    defaults[key] = value
        if len(numeric_headers) < 2:
            raise ValueError("CSV must contain at least two numeric columns for raw signal and isobestic/reference.")

        dlg = CsvChannelMappingDialog(headers, numeric_headers, defaults, self)
        if dlg.exec() != QtWidgets.QDialog.DialogCode.Accepted:
            return None
        mapping = self._sanitize_csv_mapping_for_headers(dlg.mapping(), headers)
        if mapping is None:
            raise ValueError("Invalid CSV channel mapping.")
        self._csv_channel_mapping_session = dict(mapping)
        self._csv_mappings_by_path[path] = dict(mapping)
        return mapping

    def _csv_column_index(self, headers: List[str], column: str) -> int:
        try:
            return headers.index(column)
        except ValueError:
            return -1

    def _csv_time_seconds(self, time: np.ndarray, time_col: str, unit: str) -> np.ndarray:
        t = np.asarray(time, float)
        unit_l = str(unit or "Auto").strip().lower()
        if unit_l.startswith("milli"):
            return t / 1000.0
        if unit_l.startswith("sec"):
            return t
        finite = t[np.isfinite(t)]
        if finite.size > 2:
            dt = float(np.nanmedian(np.abs(np.diff(finite))))
        else:
            dt = np.nan
        norm = self._normalize_csv_column_name(time_col)
        if "ms" in norm or "millisecond" in norm or (np.isfinite(dt) and dt >= 10.0):
            return t / 1000.0
        return t

    def _load_raw_csv_as_pre_file(self, path: str) -> Optional[LoadedDoricFile]:
        rows = self._read_csv_rows(path)
        if not rows:
            raise ValueError("CSV file is empty.")
        headers, data_rows = self._find_raw_csv_table(rows)
        if not data_rows:
            raise ValueError("CSV file has no data rows.")
        numeric_headers = self._csv_numeric_headers(headers, data_rows)
        mapping = self._csv_mapping_for_file(path, headers, numeric_headers)
        if mapping is None:
            return None

        idx_time = self._csv_column_index(headers, mapping["time"])
        idx_raw1 = self._csv_column_index(headers, mapping["raw1"])
        idx_raw2 = self._csv_column_index(headers, mapping.get("raw2", ""))
        idx_ref = self._csv_column_index(headers, mapping["reference"])
        idx_trig = self._csv_column_index(headers, mapping.get("trigger", ""))
        if min(idx_time, idx_raw1, idx_ref) < 0:
            raise ValueError("CSV channel mapping refers to a missing column.")

        time_vals: List[float] = []
        raw1_vals: List[float] = []
        raw2_vals: List[float] = []
        ref_vals: List[float] = []
        trig_vals: List[float] = []
        has_raw2 = idx_raw2 >= 0
        has_trig = idx_trig >= 0

        for row in data_rows:
            tval = self._parse_csv_float(row[idx_time] if idx_time < len(row) else "")
            if not np.isfinite(tval):
                continue
            time_vals.append(tval)
            raw1_vals.append(self._parse_csv_float(row[idx_raw1] if idx_raw1 < len(row) else ""))
            ref_vals.append(self._parse_csv_float(row[idx_ref] if idx_ref < len(row) else ""))
            if has_raw2:
                raw2_vals.append(self._parse_csv_float(row[idx_raw2] if idx_raw2 < len(row) else ""))
            if has_trig:
                trig_vals.append(self._parse_csv_float(row[idx_trig] if idx_trig < len(row) else ""))

        if len(time_vals) < 2:
            raise ValueError("CSV file has fewer than two valid time samples.")

        t = self._csv_time_seconds(np.asarray(time_vals, float), mapping["time"], mapping.get("time_unit", "Auto"))
        raw1 = np.asarray(raw1_vals, float)
        ref = np.asarray(ref_vals, float)
        if not np.isfinite(raw1).any():
            raise ValueError(f"Raw signal column '{mapping['raw1']}' has no numeric values.")
        if not np.isfinite(ref).any():
            raise ValueError(f"Isobestic/reference column '{mapping['reference']}' has no numeric values.")

        order = np.argsort(t)
        if not np.all(order == np.arange(t.size)):
            t = t[order]
            raw1 = raw1[order]
            ref = ref[order]
            if has_raw2:
                raw2_vals = list(np.asarray(raw2_vals, float)[order])
            if has_trig:
                trig_vals = list(np.asarray(trig_vals, float)[order])

        channels = [mapping["raw1"]]
        time_by = {mapping["raw1"]: t.copy()}
        signal_by = {mapping["raw1"]: raw1.copy()}
        reference_by = {mapping["raw1"]: ref.copy()}

        if has_raw2:
            raw2 = np.asarray(raw2_vals, float)
            if np.isfinite(raw2).any():
                channels.append(mapping["raw2"])
                time_by[mapping["raw2"]] = t.copy()
                signal_by[mapping["raw2"]] = raw2.copy()
                reference_by[mapping["raw2"]] = ref.copy()

        trigger_by: Dict[str, np.ndarray] = {}
        trigger_time_by: Dict[str, np.ndarray] = {}
        digital_time: Optional[np.ndarray] = None
        if has_trig:
            trig = np.asarray(trig_vals, float)
            if trig.size == t.size and np.isfinite(trig).any():
                trig_name = mapping.get("trigger", "") or "Events"
                digital_time = t.copy()
                trigger_by[trig_name] = trig.copy()
                trigger_time_by[trig_name] = t.copy()

        return LoadedDoricFile(
            path=path,
            channels=channels,
            time_by_channel=time_by,
            signal_by_channel=signal_by,
            reference_by_channel=reference_by,
            digital_time=digital_time,
            digital_by_name={k: v.copy() for k, v in trigger_by.items()},
            trigger_time_by_name=trigger_time_by,
            trigger_by_name=trigger_by,
        )

    def _open_files_dialog(self) -> None:
        start_dir = self.file_panel.current_dir_hint() or self.settings.value("last_open_dir", "", type=str) or os.getcwd()
        paths, _ = QtWidgets.QFileDialog.getOpenFileNames(
            self,
            "Open files",
            start_dir,
            "Data files (*.doric *.h5 *.hdf5 *.csv);;Doric/HDF5 files (*.doric *.h5 *.hdf5);;CSV files (*.csv);;All files (*.*)",
        )
        if not paths:
            return

        self.settings.setValue("last_open_dir", os.path.dirname(paths[0]))
        self._push_recent_preprocessing_files(paths)
        self._add_files(paths)

    def _open_folder_dialog(self) -> None:
        start_dir = self.file_panel.current_dir_hint() or self.settings.value("last_open_dir", "", type=str) or os.getcwd()
        folder = QtWidgets.QFileDialog.getExistingDirectory(self, "Add folder with data files", start_dir)
        if not folder:
            return
        self.settings.setValue("last_open_dir", folder)

        paths: List[str] = []
        for fn in os.listdir(folder):
            if fn.lower().endswith((".doric", ".h5", ".hdf5", ".csv")):
                paths.append(os.path.join(folder, fn))
        paths.sort()
        self._push_recent_preprocessing_files(paths)
        self._add_files(paths)

    def _add_files(self, paths: List[str], select_after: bool = True) -> None:
        for p in paths:
            if p in self._loaded_files:
                continue
            ext = os.path.splitext(p)[1].lower()
            if ext == ".csv":
                try:
                    loaded_from_csv = self._load_raw_csv_as_pre_file(p)
                except Exception as e:
                    QtWidgets.QMessageBox.critical(self, "Load error", f"Failed to load CSV:\n{p}\n\n{e}")
                    continue
                if loaded_from_csv is None:
                    continue
                self._loaded_files[p] = loaded_from_csv
                self.file_panel.add_file(p)
                self._show_status_message(f"Loaded CSV: {os.path.basename(p)}", 5000)
                continue
            try:
                doric = self.processor.load_file(p)
                self._loaded_files[p] = doric
                self.file_panel.add_file(p)
                self._show_status_message(f"Loaded: {os.path.basename(p)}", 5000)
            except Exception as e:
                loaded_from_processed: Optional[LoadedDoricFile] = None
                if ext in (".h5", ".hdf5"):
                    loaded_from_processed = self._load_processed_h5_as_pre_file(p)
                if loaded_from_processed is not None:
                    self._loaded_files[p] = loaded_from_processed
                    self.file_panel.add_file(p)
                    self._show_status_message(
                        f"Loaded processed H5 as preprocessing source: {os.path.basename(p)}",
                        6000,
                    )
                    continue
                QtWidgets.QMessageBox.critical(self, "Load error", f"Failed to load:\n{p}\n\n{e}")

        self._push_recent_preprocessing_files(paths)

        # set current selection -> triggers preview
        if select_after:
            self._on_file_selection_changed()

    # ---------------- Current selection ----------------

    def _selected_paths(self) -> List[str]:
        return self.file_panel.selected_paths()

    def _current_key(self) -> Optional[Tuple[str, str]]:
        if not self._current_path or not self._current_channel:
            return None
        return (self._current_path, self._current_channel)

    def _focus_data_browser(self) -> None:
        if not self.file_panel.isVisible():
            self._set_data_panel_visible(True)
        self.file_panel.setFocus()
        self.file_panel.list_files.setFocus()

    def _hide_preprocessing_popups_for_tab_switch(self) -> None:
        if self._use_pg_dockarea_pre_layout:
            # DockArea lives inside the Preprocessing tab widget; avoid costly hide/remove
            # churn during main-tab switches for smoother transitions.
            return
        if self._pre_popups_hidden_by_tab_switch:
            # Re-apply hide in case late dock events re-show a preprocessing dock.
            self._enforce_preprocessing_popups_hidden()
            return
        host = self
        self._pre_section_visibility_before_tab_switch = {
            key: bool(dock.isVisible()) for key, dock in self._section_docks.items()
        }
        self._pre_section_state_before_tab_switch = {}
        for key, dock in self._section_docks.items():
            area = _dock_area_to_int(host.dockWidgetArea(dock), _dock_area_to_int(QtCore.Qt.DockWidgetArea.LeftDockWidgetArea, 1))
            self._pre_section_state_before_tab_switch[key] = {
                "visible": bool(dock.isVisible()),
                "floating": bool(dock.isFloating()),
                "area": area,
                "geometry": dock.saveGeometry(),
            }
        self._pre_artifact_visible_before_tab_switch = bool(self.art_dock.isVisible())
        self._pre_artifact_state_before_tab_switch = {
            "visible": bool(self.art_dock.isVisible()),
            "floating": bool(self.art_dock.isFloating()),
            "area": _dock_area_to_int(host.dockWidgetArea(self.art_dock), _dock_area_to_int(QtCore.Qt.DockWidgetArea.LeftDockWidgetArea, 1)),
            "geometry": self.art_dock.saveGeometry(),
        }
        self._pre_advanced_visible_before_tab_switch = bool(
            self._advanced_dialog is not None and self._advanced_dialog.isVisible()
        )
        self._pre_tab_groups_before_tab_switch = self._capture_pre_tab_groups_state()
        self._pre_main_dock_state_before_tab_switch = self.captureDockSnapshotForTab("pre")
        self._store_pre_main_dock_snapshot()
        # Mark switch-hide state before any dock visibility changes so asynchronous
        # visibility signals cannot persist temporary hidden defaults.
        self._pre_popups_hidden_by_tab_switch = True
        # Persist cached state now; dock hide/remove operations below are temporary.
        self._persist_hidden_preprocessing_layout_state()

        self._suspend_panel_layout_persistence = True
        try:
            for key in self._section_docks.keys():
                self._set_section_button_checked(key, False)
            self._hide_dock_widgets(self.getPreDockWidgets(), remove=True)
            if self._advanced_dialog is not None:
                self._advanced_dialog.hide()
        finally:
            self._suspend_panel_layout_persistence = False

    def _enforce_preprocessing_popups_hidden(self) -> None:
        """
        Hard-hide preprocessing docks/dialogs while Post Processing is active.
        This protects against late Qt dock re-show events when dock tab stacks are rebuilt.
        """
        if self._use_pg_dockarea_pre_layout:
            if isinstance(self.art_dock, QtWidgets.QDockWidget):
                try:
                    self.art_dock.hide()
                    self.removeDockWidget(self.art_dock)
                except Exception:
                    pass
            return
        if hasattr(self, "tabs") and self.tabs.currentWidget() is self.pre_tab:
            return
        self._suspend_panel_layout_persistence = True
        try:
            for key in self._section_docks.keys():
                self._set_section_button_checked(key, False)
            self._hide_dock_widgets(self.getPreDockWidgets(), remove=True)
            if self._advanced_dialog is not None:
                self._advanced_dialog.hide()
            # Extra safety: hide any dock that belongs to preprocessing by object name prefix.
            for dock in self.findChildren(QtWidgets.QDockWidget):
                name = str(dock.objectName() or "")
                if name.startswith(_PRE_DOCK_PREFIX):
                    dock.hide()
                    try:
                        self.removeDockWidget(dock)
                    except Exception:
                        pass
        finally:
            self._suspend_panel_layout_persistence = False

    def _enforce_postprocessing_popups_hidden(self) -> None:
        """Hide post-processing docks while Preprocessing is active."""
        if hasattr(self, "tabs") and self.tabs.currentWidget() is not self.pre_tab:
            return
        remove_post = not self._force_fixed_dock_layouts
        self._hide_dock_widgets(self.getPostDockWidgets(), remove=remove_post)

    def _store_pre_main_dock_snapshot(self) -> None:
        """Persist the current preprocessing dock arrangement."""
        if self._use_pg_dockarea_pre_layout:
            self._save_panel_layout_state()
            return
        try:
            state = self.captureDockSnapshotForTab("pre")
            if state is not None and not state.isEmpty():
                self.settings.setValue(_PRE_DOCK_STATE_KEY, state)
            self._save_pre_tab_groups_to_settings(self._capture_pre_tab_groups_state())
            self.settings.sync()
        except Exception:
            pass

    def _persist_hidden_preprocessing_layout_state(self) -> None:
        """
        Persist cached preprocessing layout while preprocessing docks are hidden
        during a main-tab switch.
        """
        if self._use_pg_dockarea_pre_layout:
            self._save_panel_layout_state()
            return
        if not self._pre_popups_hidden_by_tab_switch:
            return
        left_i = _dock_area_to_int(QtCore.Qt.DockWidgetArea.LeftDockWidgetArea, 1)
        try:
            for key in self._section_docks.keys():
                state = self._pre_section_state_before_tab_switch.get(key, {})
                base = f"pre_section_docks/{key}"
                self.settings.setValue(f"{base}/visible", bool(state.get("visible", False)))
                self.settings.setValue(f"{base}/floating", bool(state.get("floating", True)))
                self.settings.setValue(f"{base}/area", _dock_area_to_int(state.get("area", left_i), left_i))
                geom = state.get("geometry")
                if isinstance(geom, QtCore.QByteArray) and not geom.isEmpty():
                    self.settings.setValue(f"{base}/geometry", geom)
        except Exception:
            pass

        try:
            art_state = self._pre_artifact_state_before_tab_switch or {}
            base = "pre_artifact_dock_state"
            self.settings.setValue(f"{base}/visible", bool(art_state.get("visible", False)))
            self.settings.setValue(f"{base}/floating", bool(art_state.get("floating", False)))
            self.settings.setValue(f"{base}/area", _dock_area_to_int(art_state.get("area", left_i), left_i))
            art_geom = art_state.get("geometry")
            if isinstance(art_geom, QtCore.QByteArray) and not art_geom.isEmpty():
                self.settings.setValue(f"{base}/geometry", art_geom)
        except Exception:
            pass

        try:
            state = self._pre_main_dock_state_before_tab_switch
            if isinstance(state, QtCore.QByteArray) and not state.isEmpty():
                self.settings.setValue(_PRE_DOCK_STATE_KEY, state)
        except Exception:
            pass
        self._save_pre_tab_groups_to_settings(self._pre_tab_groups_before_tab_switch)

        try:
            self.settings.sync()
        except Exception:
            pass

    def _apply_pre_main_dock_snapshot_if_needed(self) -> None:
        if self._use_pg_dockarea_pre_layout:
            self._pre_snapshot_applied = True
            return
        if self._force_fixed_dock_layouts:
            self._pre_snapshot_applied = True
            return
        if self._pre_snapshot_applied:
            return
        try:
            raw = self.settings.value(_PRE_DOCK_STATE_KEY, None)
            state = self._to_qbytearray(raw)
            if state is None or state.isEmpty():
                self._pre_snapshot_applied = True
                return

            ok = self.restoreDockSnapshotForTab("pre", state)
            if ok:
                self._pre_snapshot_applied = True
                self._pre_snapshot_retry_attempts = 0
                self._sync_section_button_states_from_docks()
                _LOG.info("Pre dock snapshot applied successfully")
                return

            self._pre_snapshot_retry_attempts += 1
            _LOG.warning(
                "Pre dock snapshot restore failed (attempt %s/%s)",
                self._pre_snapshot_retry_attempts,
                self._pre_snapshot_max_retries,
            )
            if self._pre_snapshot_retry_attempts >= self._pre_snapshot_max_retries:
                # Incompatible payload (old object names or stale version): drop and continue
                # with per-dock fallback settings.
                self.settings.remove(_PRE_DOCK_STATE_KEY)
                self._pre_snapshot_applied = True
                return

            delay = 0 if self._post_docks_ready else 120
            self._schedule_pre_snapshot_retry(delay)
        except Exception:
            _LOG.exception("Pre dock snapshot restore raised unexpectedly")

    def _apply_pre_default_layout_if_missing(self) -> None:
        """Set a sensible preprocessing dock layout when no saved layout exists."""
        if self._use_pg_dockarea_pre_layout:
            self._setup_section_popups()
            self._restore_pre_dockarea_layout_state()
            self._save_panel_layout_state()
            return
        try:
            if self.settings.contains(_PRE_DOCK_STATE_KEY):
                return
            has_any = False
            for key in self._section_docks.keys():
                if self.settings.contains(f"pre_section_docks/{key}/visible"):
                    has_any = True
                    break
            if self.settings.contains("pre_artifact_dock_state/visible"):
                has_any = True
            if has_any:
                return
        except Exception:
            return

        if not self._section_docks:
            return

        left = QtCore.Qt.DockWidgetArea.LeftDockWidgetArea
        self._suspend_panel_layout_persistence = True
        try:
            self._enforce_postprocessing_popups_hidden()
            # Default preprocessing layout:
            # - top tab group: Artifacts list / Artifacts / Filtering / Baseline / Output
            # - middle: QC
            # - bottom: Export
            # - left tab: Configuration
            artifacts = self._section_docks.get("artifacts")
            filtering = self._section_docks.get("filtering")
            baseline = self._section_docks.get("baseline")
            output = self._section_docks.get("output")
            qc = self._section_docks.get("qc")
            export = self._section_docks.get("export")
            config = self._section_docks.get("config")

            for dock in (self.art_dock, artifacts, filtering, baseline, output, qc, export, config):
                if dock is None:
                    continue
                dock.setFloating(False)
                dock.show()

            self.addDockWidget(left, self.art_dock)
            for dock in (artifacts, filtering, baseline, output, qc, export):
                if dock is not None:
                    self.addDockWidget(left, dock)

            if qc is not None:
                self.splitDockWidget(self.art_dock, qc, QtCore.Qt.Orientation.Vertical)
            if export is not None:
                if qc is not None:
                    self.splitDockWidget(qc, export, QtCore.Qt.Orientation.Vertical)
                else:
                    self.splitDockWidget(self.art_dock, export, QtCore.Qt.Orientation.Vertical)

            if config is not None:
                self.addDockWidget(left, config)
                config.raise_()

            for dock in (artifacts, filtering, baseline, output, config):
                if dock is not None:
                    self.tabifyDockWidget(self.art_dock, dock)
            self.art_dock.raise_()
            if qc is not None:
                qc.raise_()
            if export is not None:
                export.raise_()

            self._sync_section_button_states_from_docks()
        finally:
            self._suspend_panel_layout_persistence = False

        self._save_panel_layout_state()
        self._store_pre_main_dock_snapshot()

    def _apply_pre_fixed_layout(self) -> None:
        """
        Force a deterministic preprocessing dock layout matching the project default:
        - Left column top: Artifacts list tab group
          (Artifacts list / Artifacts / Filtering / Baseline / Output)
        - Left column middle: QC
        - Left column bottom: Export
        - Left tab: Configuration
        """
        if self._use_pg_dockarea_pre_layout:
            self._setup_section_popups()
            if not self._pre_dockarea_fixed_layout_applied:
                self._apply_pre_fixed_dockarea_layout()
            else:
                self._sync_section_button_states_from_docks()
            return
        if not self._section_docks:
            return

        host = self
        left = QtCore.Qt.DockWidgetArea.LeftDockWidgetArea
        artifacts = self._section_docks.get("artifacts")
        filtering = self._section_docks.get("filtering")
        baseline = self._section_docks.get("baseline")
        output = self._section_docks.get("output")
        qc = self._section_docks.get("qc")
        export = self._section_docks.get("export")
        config = self._section_docks.get("config")

        self._suspend_panel_layout_persistence = True
        try:
            self._hide_dock_widgets(self.getPostDockWidgets(), remove=True)
            # Attach all preprocessing docks in a deterministic non-floating state first.
            ordered_left: List[QtWidgets.QDockWidget] = []
            if isinstance(self.art_dock, QtWidgets.QDockWidget):
                ordered_left.append(self.art_dock)
            for dock in (artifacts, filtering, baseline, output, qc, export, config):
                if isinstance(dock, QtWidgets.QDockWidget):
                    ordered_left.append(dock)

            for dock in ordered_left:
                dock.blockSignals(True)
                try:
                    dock.setFloating(False)
                    host.addDockWidget(left, dock)
                    dock.show()
                finally:
                    dock.blockSignals(False)

            # Vertical stack in left area: top tab group -> QC -> Export.
            if qc is not None:
                host.splitDockWidget(self.art_dock, qc, QtCore.Qt.Orientation.Vertical)
            if export is not None:
                if qc is not None:
                    host.splitDockWidget(qc, export, QtCore.Qt.Orientation.Vertical)
                else:
                    host.splitDockWidget(self.art_dock, export, QtCore.Qt.Orientation.Vertical)

            # Top tab group: Artifacts list + Artifacts + Filtering + Baseline + Output.
            if artifacts is not None:
                host.tabifyDockWidget(self.art_dock, artifacts)
            if filtering is not None:
                host.tabifyDockWidget(self.art_dock, filtering)
            if baseline is not None:
                host.tabifyDockWidget(self.art_dock, baseline)
            if output is not None:
                host.tabifyDockWidget(self.art_dock, output)
            if config is not None:
                host.tabifyDockWidget(self.art_dock, config)

            # Keep active tabs consistent with the default arrangement.
            try:
                self.art_dock.raise_()
            except Exception:
                pass
            if qc is not None:
                qc.raise_()
            if export is not None:
                export.raise_()
            if config is not None:
                config.raise_()

            # Approximate default height proportions for left-column groups.
            try:
                vdocks: List[QtWidgets.QDockWidget] = []
                sizes: List[int] = []
                if isinstance(self.art_dock, QtWidgets.QDockWidget):
                    vdocks.append(self.art_dock)
                    sizes.append(560)
                if qc is not None:
                    vdocks.append(qc)
                    sizes.append(220)
                if export is not None:
                    vdocks.append(export)
                    sizes.append(120)
                if vdocks and sizes:
                    host.resizeDocks(vdocks, sizes, QtCore.Qt.Orientation.Vertical)
            except Exception:
                pass

            self._sync_section_button_states_from_docks()
        finally:
            self._suspend_panel_layout_persistence = False

    def _restore_preprocessing_popups_after_tab_switch(self) -> None:
        if self._use_pg_dockarea_pre_layout:
            return
        if not self._pre_popups_hidden_by_tab_switch:
            return

        host = self
        restored_from_snapshot = False
        self._suspend_panel_layout_persistence = True
        try:
            snapshot = self._pre_main_dock_state_before_tab_switch
            if (
                not self._force_fixed_dock_layouts
                and isinstance(snapshot, QtCore.QByteArray)
                and not snapshot.isEmpty()
            ):
                try:
                    restored_from_snapshot = self.restoreDockSnapshotForTab("pre", snapshot)
                except Exception:
                    restored_from_snapshot = False
            if restored_from_snapshot:
                self._sync_section_button_states_from_docks()
                if self._pre_advanced_visible_before_tab_switch and self._advanced_dialog is not None:
                    self._advanced_dialog.show()
                    self._advanced_dialog.raise_()
                    self._advanced_dialog.activateWindow()
            else:
                for key, dock in self._section_docks.items():
                    state = self._pre_section_state_before_tab_switch.get(key, {})
                    visible = bool(state.get("visible", self._pre_section_visibility_before_tab_switch.get(key, False)))
                    floating = bool(state.get("floating", dock.isFloating()))
                    area = self._dock_area_from_settings(
                        state.get("area", _dock_area_to_int(QtCore.Qt.DockWidgetArea.LeftDockWidgetArea, 1)),
                        QtCore.Qt.DockWidgetArea.LeftDockWidgetArea,
                    )
                    geom = state.get("geometry", None)

                    dock.blockSignals(True)
                    try:
                        if floating:
                            dock.setFloating(True)
                        else:
                            host.addDockWidget(area, dock)
                            dock.setFloating(False)
                        if isinstance(geom, QtCore.QByteArray) and not geom.isEmpty():
                            dock.restoreGeometry(geom)
                            self._section_popup_initialized.add(key)
                        if visible:
                            if dock.isFloating() and not self._is_popup_on_screen(dock):
                                self._position_section_popup(dock)
                            dock.show()
                            self._set_section_button_checked(key, True)
                            self._last_opened_section = key
                        else:
                            dock.hide()
                            self._set_section_button_checked(key, False)
                    finally:
                        dock.blockSignals(False)

                art_state = self._pre_artifact_state_before_tab_switch or {}
                art_visible = bool(art_state.get("visible", self._pre_artifact_visible_before_tab_switch))
                art_floating = bool(art_state.get("floating", self.art_dock.isFloating()))
                art_area = self._dock_area_from_settings(
                    art_state.get("area", _dock_area_to_int(QtCore.Qt.DockWidgetArea.LeftDockWidgetArea, 1)),
                    QtCore.Qt.DockWidgetArea.LeftDockWidgetArea,
                )
                art_geom = art_state.get("geometry", None)
                if art_floating:
                    self.art_dock.setFloating(True)
                else:
                    host.addDockWidget(art_area, self.art_dock)
                    self.art_dock.setFloating(False)
                if isinstance(art_geom, QtCore.QByteArray) and not art_geom.isEmpty():
                    self.art_dock.restoreGeometry(art_geom)
                self.art_dock.setVisible(art_visible)

                if self._pre_advanced_visible_before_tab_switch and self._advanced_dialog is not None:
                    self._advanced_dialog.show()
                    self._advanced_dialog.raise_()
                    self._advanced_dialog.activateWindow()
                self._restore_pre_tab_groups_fallback(self._pre_tab_groups_before_tab_switch)
        finally:
            self._suspend_panel_layout_persistence = False

        self._pre_popups_hidden_by_tab_switch = False
        self._pre_section_visibility_before_tab_switch.clear()
        self._pre_section_state_before_tab_switch.clear()
        self._pre_artifact_visible_before_tab_switch = False
        self._pre_artifact_state_before_tab_switch.clear()
        self._pre_advanced_visible_before_tab_switch = False
        self._pre_main_dock_state_before_tab_switch = None
        self._pre_tab_groups_before_tab_switch = []
        self._enforce_postprocessing_popups_hidden()
        self._save_panel_layout_state()

    def _restore_window_state_after_tab_switch(self, was_fullscreen: bool, was_maximized: bool) -> None:
        """
        Keep the top-level window mode stable across heavy dock add/remove/tabify
        operations triggered by main-tab switches.
        """
        if was_fullscreen:
            if self.isFullScreen():
                return

            def _ensure_fullscreen() -> None:
                if not self.isFullScreen():
                    self.showFullScreen()

            try:
                _ensure_fullscreen()
                QtCore.QTimer.singleShot(0, _ensure_fullscreen)
                QtCore.QTimer.singleShot(120, _ensure_fullscreen)
            except Exception:
                pass
            return

        if was_maximized and not self.isMaximized() and not self.isFullScreen():
            try:
                self.showMaximized()
            except Exception:
                pass

    def _apply_fixed_post_layout_deferred(self) -> None:
        if not hasattr(self, "tabs") or self.tabs.currentWidget() is not self.post_tab:
            return
        try:
            self.post_tab.ensure_section_popups_initialized()
            if hasattr(self.post_tab, "apply_fixed_default_layout"):
                self.post_tab.apply_fixed_default_layout()
        except Exception:
            _LOG.exception("Failed to apply fixed post layout on tab switch")
            return
        try:
            self._enforce_only_tab_docks_visible("post")
        except Exception:
            pass

    def _enforce_fixed_layout_for_active_tab(self) -> None:
        if not self._force_fixed_dock_layouts or not hasattr(self, "tabs"):
            return
        try:
            current = self.tabs.currentWidget()
        except Exception:
            return
        try:
            if current is self.pre_tab:
                self._apply_pre_fixed_layout()
                self._enforce_only_tab_docks_visible("pre")
            elif current is self.post_tab:
                self._apply_fixed_post_layout_deferred()
                self._enforce_only_tab_docks_visible("post")
        except Exception:
            _LOG.exception("Failed to enforce fixed layout for active tab")

    def _on_main_tab_changed(self, index: int) -> None:
        if self._handling_main_tab_change:
            self._pending_main_tab_index = int(index)
            return
        self._handling_main_tab_change = True
        was_fullscreen = bool(self.isFullScreen())
        was_maximized = bool(self.isMaximized())
        try:
            current = self.tabs.widget(index)
            if self._force_fixed_dock_layouts:
                try:
                    if current is self.pre_tab:
                        try:
                            self.post_tab.hide_section_popups_for_tab_switch()
                        except Exception:
                            pass
                        self._enforce_postprocessing_popups_hidden()
                        self._restore_preprocessing_popups_after_tab_switch()
                        self._apply_pre_fixed_layout()
                        self._enforce_only_tab_docks_visible("pre")
                    else:
                        self._hide_preprocessing_popups_for_tab_switch()
                        self._enforce_preprocessing_popups_hidden()
                        QtCore.QTimer.singleShot(0, self._enforce_preprocessing_popups_hidden)
                        self._apply_fixed_post_layout_deferred()
                        QtCore.QTimer.singleShot(0, self._apply_fixed_post_layout_deferred)
                        QtCore.QTimer.singleShot(120, self._apply_fixed_post_layout_deferred)
                        self._enforce_only_tab_docks_visible("post")
                    try:
                        self._save_panel_layout_state()
                    except Exception:
                        pass
                    if current is self.pre_tab:
                        try:
                            self._store_pre_main_dock_snapshot()
                        except Exception:
                            pass
                    else:
                        try:
                            self._persist_hidden_preprocessing_layout_state()
                        except Exception:
                            pass
                    if current is self.pre_tab:
                        try:
                            self.post_tab.persist_layout_state_snapshot()
                        except Exception:
                            pass
                    self._save_settings()
                    self._save_panel_config_json()
                except Exception:
                    _LOG.exception("Failed to handle fixed-layout tab switch")
                return
            try:
                if current is self.pre_tab:
                    try:
                        self.post_tab.hide_section_popups_for_tab_switch()
                    except Exception:
                        pass
                    self._enforce_postprocessing_popups_hidden()
                    self._restore_preprocessing_popups_after_tab_switch()
                    self._apply_pre_main_dock_snapshot_if_needed()
                    self._enforce_only_tab_docks_visible("pre")
                else:
                    self._hide_preprocessing_popups_for_tab_switch()
                    # Run once now and once after queued dock events so preprocessing panels
                    # cannot bleed into Post Processing.
                    self._enforce_preprocessing_popups_hidden()
                    QtCore.QTimer.singleShot(0, self._enforce_preprocessing_popups_hidden)
                    self._enforce_only_tab_docks_visible("post")
                # Persist active main tab immediately.
                self._save_settings()
                # Persist panel layout JSON on each tab switch.
                self._save_panel_config_json()
            except Exception:
                _LOG.exception("Failed to handle main tab switch")
        finally:
            self._restore_window_state_after_tab_switch(was_fullscreen, was_maximized)
            self._handling_main_tab_change = False
            if self._force_fixed_dock_layouts:
                QtCore.QTimer.singleShot(0, self._enforce_fixed_layout_for_active_tab)
                QtCore.QTimer.singleShot(80, self._enforce_fixed_layout_for_active_tab)
            if self._pending_main_tab_index is not None:
                pending = int(self._pending_main_tab_index)
                self._pending_main_tab_index = None
                QtCore.QTimer.singleShot(0, lambda idx=pending: self._on_main_tab_changed(idx))

    def _on_artifact_overlay_toggled(self, visible: bool) -> None:
        self.plots.set_artifact_overlay_visible(bool(visible))
        self._save_settings()

    def _on_artifact_thresholds_toggled(self, visible: bool) -> None:
        self.plots.set_artifact_thresholds_visible(bool(visible))
        self._save_settings()

    def _normalize_app_theme_mode(self, value: object) -> str:
        mode = str(value or "").strip().lower()
        if mode in {"light", "white", "l", "w"}:
            return "light"
        return "dark"

    def _selected_app_theme_mode(self) -> str:
        if hasattr(self, "act_app_theme_light") and self.act_app_theme_light.isChecked():
            return "light"
        return "dark"

    def _apply_app_theme(self, theme_mode: object, persist: bool = True) -> None:
        mode = self._normalize_app_theme_mode(theme_mode)
        self._app_theme_mode = mode

        if hasattr(self, "act_app_theme_dark"):
            self.act_app_theme_dark.blockSignals(True)
            self.act_app_theme_dark.setChecked(mode == "dark")
            self.act_app_theme_dark.blockSignals(False)
        if hasattr(self, "act_app_theme_light"):
            self.act_app_theme_light.blockSignals(True)
            self.act_app_theme_light.setChecked(mode == "light")
            self.act_app_theme_light.blockSignals(False)

        try:
            apply_app_palette(QtWidgets.QApplication.instance(), mode)
            self.setStyleSheet(app_qss(mode))
        except Exception:
            pass

        pre_bg = "white" if mode == "light" else "dark"
        pre_grid = self.act_plot_grid.isChecked() if hasattr(self, "act_plot_grid") else True
        self._apply_pre_plot_style(pre_bg, pre_grid, persist=False)

        try:
            if hasattr(self.post_tab, "set_app_theme_mode"):
                self.post_tab.set_app_theme_mode(mode)
        except Exception:
            pass

        if persist:
            self._save_settings()

    def _on_app_theme_changed(self, *_args) -> None:
        self._apply_app_theme(self._selected_app_theme_mode(), persist=True)

    def _normalize_pre_plot_background(self, value: object) -> str:
        mode = str(value or "").strip().lower()
        if mode in {"white", "light", "w"}:
            return "white"
        return "dark"

    def _selected_pre_plot_background(self) -> str:
        if hasattr(self, "act_plot_bg_white") and self.act_plot_bg_white.isChecked():
            return "white"
        return "dark"

    def _apply_pre_plot_style(self, background: object, show_grid: object, persist: bool = True) -> None:
        mode = self._normalize_pre_plot_background(background)
        grid = bool(show_grid)
        if hasattr(self, "act_plot_bg_dark"):
            self.act_plot_bg_dark.blockSignals(True)
            self.act_plot_bg_dark.setChecked(mode == "dark")
            self.act_plot_bg_dark.blockSignals(False)
        if hasattr(self, "act_plot_bg_white"):
            self.act_plot_bg_white.blockSignals(True)
            self.act_plot_bg_white.setChecked(mode == "white")
            self.act_plot_bg_white.blockSignals(False)
        if hasattr(self, "act_plot_grid"):
            self.act_plot_grid.blockSignals(True)
            self.act_plot_grid.setChecked(grid)
            self.act_plot_grid.blockSignals(False)
        try:
            self.plots.set_plot_appearance(mode, grid)
        except Exception:
            pass
        if persist:
            self._save_settings()

    def _on_pre_plot_style_changed(self, *_args) -> None:
        self._apply_pre_plot_style(
            self._selected_pre_plot_background(),
            self.act_plot_grid.isChecked() if hasattr(self, "act_plot_grid") else True,
            persist=True,
        )

    def _auto_range_for_processed(self, processed: ProcessedTrial) -> None:
        try:
            start_s, end_s = self._time_window_bounds()
            t = np.asarray(processed.time, float)
            if t.size > 1:
                x0 = float(np.nanmin(t)) if start_s is None else float(start_s)
                x1 = float(np.nanmax(t)) if end_s is None else float(end_s)
                if np.isfinite(x0) and np.isfinite(x1) and x1 > x0:
                    self.plots.auto_range_all(x0=x0, x1=x1)
                else:
                    self.plots.auto_range_all()
            else:
                self.plots.auto_range_all()
        except Exception:
            self.plots.auto_range_all()

    def _fmt_fs(self, fs: Optional[float]) -> str:
        if fs is None or not np.isfinite(float(fs)):
            return "-"
        return f"{float(fs):.2f}"

    def _current_fs_actual(self) -> Optional[float]:
        key = self._current_key()
        if key:
            proc = self._last_processed.get(key)
            if proc is not None and np.isfinite(float(getattr(proc, "fs_actual", np.nan))):
                return float(proc.fs_actual)
        if not self._current_path or not self._current_channel:
            return None
        doric = self._loaded_files.get(self._current_path)
        if doric is None:
            return None
        try:
            trial = doric.make_trial(self._current_channel, trigger_name=self._current_trigger)
            trial = self._apply_time_window(trial)
            fs = float(trial.sampling_rate)
            return fs if np.isfinite(fs) else None
        except Exception:
            return None

    def _show_status_message(self, message: str, timeout_ms: int = 0) -> None:
        sb = getattr(self, "_status_bar", None)
        if not isinstance(sb, QtWidgets.QStatusBar):
            attr = getattr(self, "statusBar", None)
            if callable(attr):
                try:
                    sb = attr()
                except Exception:
                    sb = None
            elif isinstance(attr, QtWidgets.QStatusBar):
                sb = attr
        if not isinstance(sb, QtWidgets.QStatusBar):
            return
        try:
            sb.showMessage(str(message), int(timeout_ms))
        except Exception:
            pass

    def _update_plot_status(self, fs_actual: Optional[float] = None, fs_target: Optional[float] = None) -> None:
        channel = self._current_channel or "-"
        trig = self._current_trigger or "None"
        mode = "-"
        target = fs_target
        try:
            p = self.param_panel.get_params()
            mode = str(p.output_mode)
            if target is None:
                target = float(p.target_fs_hz)
        except Exception:
            pass
        if fs_actual is None:
            fs_actual = self._current_fs_actual()

        status = (
            f"Channel: {channel} | A/D: {trig} | Fs: {self._fmt_fs(fs_actual)} -> "
            f"{self._fmt_fs(target)} Hz | Mode: {mode}"
        )
        self._show_status_message(status, 30000)

    def _on_file_selection_changed(self) -> None:
        sel = self._selected_paths()
        if not sel:
            all_paths = self.file_panel.all_paths()
            if all_paths:
                self.file_panel.list_files.setCurrentRow(0)
                item0 = self.file_panel.list_files.item(0)
                if item0 is not None:
                    item0.setSelected(True)
                sel = self._selected_paths()
            if not sel:
                self._current_path = None
                self._current_channel = None
                self._current_trigger = None
                self.plots.set_title("No file loaded")
                self._update_plot_status()
            return

        # preview shows first selected
        path = sel[0]
        self._current_path = path

        doric = self._loaded_files.get(path)
        if not doric:
            return

        self.file_panel.set_available_channels(doric.channels)
        self.file_panel.set_available_triggers(sorted(doric.trigger_by_name.keys()))
        self.param_panel.set_available_export_channels(doric.channels)
        self.param_panel.set_available_export_triggers(sorted(doric.trigger_by_name.keys()))
        self._update_export_summary_label()

        # keep channel if still valid
        if self._current_channel in doric.channels:
            self.file_panel.set_channel(self._current_channel)
        else:
            self._current_channel = doric.channels[0] if doric.channels else None
            if self._current_channel:
                self.file_panel.set_channel(self._current_channel)

        # keep trigger if still valid
        if self._current_trigger and self._current_trigger not in doric.trigger_by_name:
            self._current_trigger = None
            self.file_panel.set_trigger("")
        self._update_export_summary_label()

        self._update_raw_plot()
        self._trigger_preview()

        # update post tab selection context
        self.post_tab.set_current_source_label(os.path.basename(path), self._current_channel or "")
        self._update_plot_status()

    def _on_channel_changed(self, ch: str) -> None:
        self._current_channel = ch
        if self._current_path:
            doric = self._loaded_files.get(self._current_path)
            if doric is not None:
                self.param_panel.set_available_export_channels(
                    doric.channels,
                    preferred=self.param_panel.export_channel_names(),
                )
                self._update_export_summary_label()
        self._update_raw_plot()
        self._trigger_preview()
        self.post_tab.set_current_source_label(os.path.basename(self._current_path or ""), self._current_channel or "")
        self._update_plot_status()

    def _on_trigger_changed(self, trig: str) -> None:
        self._current_trigger = trig if trig else None
        if self._current_path:
            doric = self._loaded_files.get(self._current_path)
            if doric is not None:
                self.param_panel.set_available_export_triggers(
                    sorted(doric.trigger_by_name.keys()),
                    preferred=self.param_panel.export_trigger_names(),
                )
                self._update_export_summary_label()
        self._update_raw_plot()
        self._update_plot_status()

    def _on_time_window_changed(self) -> None:
        self._last_processed.clear()
        key = self._current_key()
        if key:
            start_s, end_s = self._time_window_bounds()
            manual_win = self._clip_regions_to_window(self._manual_regions_by_key.get(key, []), start_s, end_s)
            ignore_win = self._clip_regions_to_window(self._manual_exclude_by_key.get(key, []), start_s, end_s)
            auto_win = self._clip_regions_to_window(self._auto_regions_by_key.get(key, []), start_s, end_s)
            checked_auto = [r for r in auto_win if not any(self._regions_match(r, ig) for ig in ignore_win)]
            self.artifact_panel.set_auto_regions(auto_win, checked_regions=checked_auto)
            self.artifact_panel.set_regions(manual_win)
        self._update_raw_plot()
        self._trigger_preview()
        self._update_plot_status()

    def _open_advanced_options(self) -> None:
        key = self._current_key()
        if not key:
            return
        if self._advanced_dialog and self._advanced_dialog.isVisible():
            self._advanced_dialog.raise_()
            self._advanced_dialog.activateWindow()
            return
        cutouts = self._cutout_regions_by_key.get(key, [])
        sections = self._sections_by_key.get(key, [])
        dlg = AdvancedOptionsDialog(
            cutouts,
            sections,
            self.param_panel.get_params(),
            request_box_select=self._request_box_select,
            parent=self,
        )
        self._advanced_dialog = dlg

        def _cleanup() -> None:
            if self._advanced_dialog is dlg:
                self._advanced_dialog = None
            self._cancel_box_select_request()

        def _apply() -> None:
            if self._advanced_dialog is not dlg:
                return
            self._cutout_regions_by_key[key] = dlg.get_cutouts()
            self._sections_by_key[key] = dlg.get_sections()
            self._last_processed.clear()
            self._update_raw_plot()
            self._trigger_preview()
            _cleanup()

        dlg.accepted.connect(_apply)
        dlg.rejected.connect(_cleanup)
        dlg.finished.connect(_cleanup)
        dlg.show()

    def _run_qc_dialog(self) -> None:
        if not self._current_path or not self._current_channel:
            return
        doric = self._loaded_files.get(self._current_path)
        if not doric:
            return
        trial = doric.make_trial(self._current_channel, trigger_name=self._current_trigger)
        trial = self._apply_time_window(trial)
        key = (self._current_path, self._current_channel)
        cutouts = self._cutout_regions_by_key.get(key, [])
        trial = self._apply_cutouts(trial, cutouts)

        qc = self._compute_qc(trial)
        if qc is None:
            return
        dlg = QcDialog(qc, self)
        dlg.exec()

    def _run_batch_qc(self) -> None:
        paths = self._selected_paths()
        if not paths:
            return
        for p in paths:
            doric = self._loaded_files.get(p)
            if not doric:
                continue
            if self._current_channel and self._current_channel in doric.channels:
                ch = self._current_channel
            else:
                ch = doric.channels[0] if doric.channels else None
            if not ch:
                continue
            trial = doric.make_trial(ch, trigger_name=self._current_trigger)
            trial = self._apply_time_window(trial)
            key = (p, ch)
            cutouts = self._cutout_regions_by_key.get(key, [])
            trial = self._apply_cutouts(trial, cutouts)
            qc = self._compute_qc(trial)
            if qc is None:
                continue
            dlg = QcDialog(qc, self)
            dlg.save_report()
            dlg.close()

    def _compute_qc(self, trial: LoadedTrial) -> Optional[Dict[str, object]]:
        t = np.asarray(trial.time, float)
        sig = np.asarray(trial.signal_465, float)
        ref = np.asarray(trial.reference_405, float)
        if t.size < 10:
            return None
        fs = float(trial.sampling_rate) if np.isfinite(trial.sampling_rate) else (
            1.0 / float(np.nanmedian(np.diff(t))) if t.size > 2 else np.nan
        )
        m = np.isfinite(t) & np.isfinite(sig) & np.isfinite(ref)
        t = t[m]; sig = sig[m]; ref = ref[m]
        if t.size < 10:
            return None

        # Artifact removal (adaptive MAD)
        mask_sig = detect_artifacts_adaptive(t, sig, k=6.0, window_s=1.0, pad_s=0.2)
        mask_ref = detect_artifacts_adaptive(t, ref, k=6.0, window_s=1.0, pad_s=0.2)
        mask = mask_sig | mask_ref
        sig_clean = sig.copy()
        ref_clean = ref.copy()
        sig_clean[mask] = np.nan
        ref_clean[mask] = np.nan
        sig_clean = interpolate_nans(sig_clean)
        ref_clean = interpolate_nans(ref_clean)
        art_frac = float(np.mean(mask)) if mask.size else 0.0

        # Baseline + dff
        cutoff = 0.01
        sig_base = _lowpass_sos(sig_clean, fs, cutoff, 3)
        ref_base = _lowpass_sos(ref_clean, fs, cutoff, 3)
        dff_sig = safe_divide(sig_clean - sig_base, sig_base)
        dff_ref = safe_divide(ref_clean - ref_base, ref_base)

        # z-score
        z_sig = zscore_median_std(dff_sig)
        z_ref = zscore_median_std(dff_ref)
        Z = z_sig - z_ref
        Zf = Z[np.isfinite(Z)]

        # Correlation
        m2 = np.isfinite(z_sig) & np.isfinite(z_ref)
        r = float(np.corrcoef(z_ref[m2], z_sig[m2])[0, 1]) if np.sum(m2) >= 10 else np.nan
        win = int(max(10, round(fs * 10.0))) if np.isfinite(fs) and fs > 0 else 5000
        r_roll, centers = _rolling_corr(z_ref, z_sig, win)

        # Distribution stats
        if Zf.size:
            q25, q50, q75 = np.quantile(Zf, [0.25, 0.5, 0.75])
            frac_gt3 = float(np.mean(np.abs(Zf) > 3.0) * 100.0)
            frac_gt5 = float(np.mean(np.abs(Zf) > 5.0) * 100.0)
            iqr = float(q75 - q25)
        else:
            q25 = q50 = q75 = frac_gt3 = frac_gt5 = iqr = np.nan

        stats = (
            f"artifact_frac={art_frac*100:.2f}% | r={r:.3f} | "
            f"Z median={q50:.3g} IQR=({q25:.3g},{q75:.3g}) | "
            f"|Z|>3: {frac_gt3:.2f}% | |Z|>5: {frac_gt5:.2f}%"
        )

        return {
            "path": trial.path,
            "channel": trial.channel_id,
            "t": t,
            "z_sig": z_sig,
            "z_ref": z_ref,
            "Z": Z,
            "Zf": Zf,
            "r_roll": r_roll,
            "r_centers": centers,
            "r": r,
            "q25": q25,
            "q50": q50,
            "q75": q75,
            "iqr": iqr,
            "stats": stats,
        }

    # ---------------- Raw plot update ----------------

    def _apply_time_window(self, trial: LoadedTrial) -> LoadedTrial:
        start_s, end_s = self.file_panel.time_window()
        if start_s is None and end_s is None:
            return trial

        t = np.asarray(trial.time, float)
        if start_s is None:
            mask = t <= float(end_s)
        elif end_s is None:
            mask = t >= float(start_s)
        else:
            if end_s <= start_s:
                return trial
            mask = (t >= float(start_s)) & (t <= float(end_s))
        if np.sum(mask) < 2:
            return trial

        def _mask_arr(arr: Optional[np.ndarray], use_time_mask: bool) -> Optional[np.ndarray]:
            if arr is None:
                return None
            if use_time_mask and arr.size == t.size:
                return np.asarray(arr, float)[mask]
            return np.asarray(arr, float)

        time = t[mask]
        sig = np.asarray(trial.signal_465, float)[mask]
        ref = np.asarray(trial.reference_405, float)[mask]

        trig_time = trial.trigger_time
        trig = trial.trigger
        if trig_time is not None and trig is not None:
            if trig_time.size == t.size:
                trig_time = _mask_arr(trig_time, True)
                trig = _mask_arr(trig, True)
            else:
                if start_s is None:
                    tmask = np.asarray(trig_time, float) <= float(end_s)
                elif end_s is None:
                    tmask = np.asarray(trig_time, float) >= float(start_s)
                else:
                    tmask = (trig_time >= float(start_s)) & (trig_time <= float(end_s))
                trig_time = np.asarray(trig_time, float)[tmask]
                trig = np.asarray(trig, float)[tmask]

        fs = 1.0 / float(np.nanmedian(np.diff(time))) if time.size > 2 else np.nan

        new_triggers = {}
        new_trigger_times = {}
        if hasattr(trial, "triggers") and trial.triggers:
            for name, val in trial.triggers.items():
                vt = trial.trigger_times.get(name)
                if vt is not None:
                    if vt.size == t.size:
                        new_triggers[name] = np.asarray(val, float)[mask]
                        new_trigger_times[name] = np.asarray(vt, float)[mask]
                    else:
                        tmask = (vt >= float(start_s)) & (vt <= float(end_s))
                        new_triggers[name] = np.asarray(val, float)[tmask]
                        new_trigger_times[name] = np.asarray(vt, float)[tmask]

        return LoadedTrial(
            path=trial.path,
            channel_id=trial.channel_id,
            time=time,
            signal_465=sig,
            reference_405=ref,
            sampling_rate=float(fs) if np.isfinite(fs) else np.nan,
            trigger_time=trig_time,
            trigger=trig,
            trigger_name=trial.trigger_name,
            triggers=new_triggers,
            trigger_times=new_trigger_times,
        )

    def _apply_cutouts(self, trial: LoadedTrial, cutouts: List[Tuple[float, float]]) -> LoadedTrial:
        if not cutouts:
            return trial
        t = np.asarray(trial.time, float)
        sig = np.asarray(trial.signal_465, float).copy()
        ref = np.asarray(trial.reference_405, float).copy()
        for (a, b) in cutouts:
            mask = (t >= float(a)) & (t <= float(b))
            sig[mask] = np.nan
            ref[mask] = np.nan
        return LoadedTrial(
            path=trial.path,
            channel_id=trial.channel_id,
            time=t,
            signal_465=sig,
            reference_405=ref,
            sampling_rate=trial.sampling_rate,
            trigger_time=trial.trigger_time,
            trigger=trial.trigger,
            trigger_name=trial.trigger_name,
            triggers=dict(trial.triggers) if hasattr(trial, "triggers") else {},
            trigger_times=dict(trial.trigger_times) if hasattr(trial, "trigger_times") else {},
        )

    def _apply_cutouts_to_processed(self, processed: ProcessedTrial, cutouts: List[Tuple[float, float]]) -> ProcessedTrial:
        if not cutouts or processed.time is None:
            return processed
        t = np.asarray(processed.time, float)
        mask = np.zeros_like(t, dtype=bool)
        for (a, b) in cutouts:
            mask |= (t >= float(a)) & (t <= float(b))
        if not np.any(mask):
            return processed

        def _mask_arr(arr: Optional[np.ndarray]) -> Optional[np.ndarray]:
            if arr is None:
                return None
            y = np.asarray(arr, float).copy()
            if y.size == t.size:
                y[mask] = np.nan
            return y

        processed.raw_signal = _mask_arr(processed.raw_signal)
        processed.raw_reference = _mask_arr(processed.raw_reference)
        processed.sig_f = _mask_arr(processed.sig_f)
        processed.ref_f = _mask_arr(processed.ref_f)
        processed.baseline_sig = _mask_arr(processed.baseline_sig)
        processed.baseline_ref = _mask_arr(processed.baseline_ref)
        processed.output = _mask_arr(processed.output)
        if hasattr(processed, "outputs") and processed.outputs:
            masked_outputs = {}
            for label, values in processed.outputs.items():
                masked = _mask_arr(values)
                if masked is not None:
                    masked_outputs[str(label)] = masked
            processed.outputs = masked_outputs
        
        # Mask triggers too if requested by convention, but here we keep them as-is or NaN them
        if hasattr(processed, "triggers") and processed.triggers:
            new_triggers = {}
            for name, val in processed.triggers.items():
                new_triggers[name] = _mask_arr(val)
            processed.triggers = new_triggers

        return processed

    def _slice_trial(self, trial: LoadedTrial, start_s: float, end_s: float) -> Optional[LoadedTrial]:
        t = np.asarray(trial.time, float)
        mask = (t >= float(start_s)) & (t <= float(end_s))
        if np.sum(mask) < 2:
            return None
        time = t[mask]
        sig = np.asarray(trial.signal_465, float)[mask]
        ref = np.asarray(trial.reference_405, float)[mask]
        trig_time = trial.trigger_time
        trig = trial.trigger
        if trig_time is not None and trig is not None:
            if trig_time.size == t.size:
                trig_time = np.asarray(trig_time, float)[mask]
                trig = np.asarray(trig, float)[mask]
            else:
                tmask = (trig_time >= float(start_s)) & (trig_time <= float(end_s))
                trig_time = np.asarray(trig_time, float)[tmask]
                trig = np.asarray(trig, float)[tmask]
        
        new_triggers = {}
        new_trigger_times = {}
        if hasattr(trial, "triggers") and trial.triggers:
            for name, val in trial.triggers.items():
                vt = trial.trigger_times.get(name)
                if vt is not None:
                    if vt.size == t.size:
                        new_triggers[name] = np.asarray(val, float)[mask]
                        new_trigger_times[name] = np.asarray(vt, float)[mask]
                    else:
                        tmask = (vt >= float(start_s)) & (vt <= float(end_s))
                        new_triggers[name] = np.asarray(val, float)[tmask]
                        new_trigger_times[name] = np.asarray(vt, float)[tmask]

        fs = 1.0 / float(np.nanmedian(np.diff(time))) if time.size > 2 else np.nan
        return LoadedTrial(
            path=trial.path,
            channel_id=trial.channel_id,
            time=time,
            signal_465=sig,
            reference_405=ref,
            sampling_rate=float(fs) if np.isfinite(fs) else np.nan,
            trigger_time=trig_time,
            trigger=trig,
            trigger_name=trial.trigger_name,
            triggers=new_triggers,
            trigger_times=new_trigger_times,
        )

    def _update_raw_plot(self, preserve_view: bool = False) -> None:
        if not self._current_path or not self._current_channel:
            return
        doric = self._loaded_files.get(self._current_path)
        if not doric:
            return

        trial = doric.make_trial(self._current_channel, trigger_name=self._current_trigger)
        trial = self._apply_time_window(trial)
        key = (self._current_path, self._current_channel)
        cutouts = self._cutout_regions_by_key.get(key, [])
        trial = self._apply_cutouts(trial, cutouts)
        start_s, end_s = self._time_window_bounds()
        manual = self._clip_regions_to_window(self._manual_regions_by_key.get(key, []), start_s, end_s)
        params = self.param_panel.get_params()

        raw465 = trial.signal_465
        raw405 = trial.reference_405
        if bool(getattr(params, "invert_polarity", False)):
            raw465 = -np.asarray(raw465, float)
            raw405 = -np.asarray(raw405, float)

        self.plots.set_title("raw signal")
        self.plots.show_raw(
            time=trial.time,
            raw465=raw465,
            raw405=raw405,
            trig_time=trial.trigger_time,
            trig=trial.trigger,
            trig_label=self._current_trigger or "",
            manual_regions=manual,
            preserve_view=preserve_view,
        )
        self._update_plot_status(fs_actual=float(trial.sampling_rate), fs_target=float(params.target_fs_hz))

    # ---------------- Preview processing (worker) ----------------

    def _artifact_param_signature(self, params: ProcessingParams) -> Tuple[object, ...]:
        return (
            bool(getattr(params, "artifact_detection_enabled", True)),
            str(params.artifact_mode),
            float(params.mad_k),
            float(params.adaptive_window_s),
            float(params.artifact_pad_s),
        )

    def _on_params_changed(self) -> None:
        try:
            params = self.param_panel.get_params()
        except Exception:
            self._trigger_preview(preserve_view=True)
            return
        self._update_plot_status(fs_target=float(params.target_fs_hz))
        sig = self._artifact_param_signature(params)
        if self._last_artifact_params is None:
            self._last_artifact_params = sig
        elif sig != self._last_artifact_params:
            self._last_artifact_params = sig
            # Reset auto artifact selections when detection params change
            self._manual_exclude_by_key.clear()
            key = self._current_key()
            if key:
                auto = self._auto_regions_by_key.get(key, [])
                if auto:
                    self.artifact_panel.set_auto_regions(auto, checked_regions=auto)
        # Update raw display for toggles like polarity inversion
        try:
            self._update_raw_plot(preserve_view=True)
        except Exception:
            pass
        self._trigger_preview(preserve_view=True)

    def _trigger_preview(self, preserve_view: bool = False) -> None:
        # persist params quickly
        self._save_settings()
        self._preview_preserve_view_pending = bool(preserve_view)
        self._preview_timer.start()

    def _start_preview_processing(self) -> None:
        if not self._current_path or not self._current_channel:
            return
        doric = self._loaded_files.get(self._current_path)
        if not doric:
            return

        params = self.param_panel.get_params()
        trial = doric.make_trial(self._current_channel, trigger_name=self._current_trigger)
        trial = self._apply_time_window(trial)
        key = (self._current_path, self._current_channel)
        cutouts = self._cutout_regions_by_key.get(key, [])
        trial = self._apply_cutouts(trial, cutouts)

        start_s, end_s = self._time_window_bounds()
        manual = self._clip_regions_to_window(self._manual_regions_by_key.get(key, []), start_s, end_s)
        manual_exclude = self._clip_regions_to_window(self._manual_exclude_by_key.get(key, []), start_s, end_s)
        self._job_counter += 1
        job_id = self._job_counter
        self._latest_job_id = job_id
        preserve_view = bool(self._preview_preserve_view_pending)
        self._preview_preserve_view_pending = False
        self._preview_preserve_view_by_job[job_id] = preserve_view

        self._show_status_message(
            f"Processing preview... (fs={trial.sampling_rate:.2f} Hz -> target {params.target_fs_hz:.1f} Hz, "
            f"baseline={params.baseline_method})"
        )
        self._update_plot_status(fs_actual=float(trial.sampling_rate), fs_target=float(params.target_fs_hz))

        task = self.processor.make_preview_task(
            trial=trial,
            params=params,
            manual_regions_sec=manual,
            manual_exclude_regions_sec=manual_exclude,
            job_id=job_id,
        )
        task.signals.finished.connect(self._on_preview_finished)
        task.signals.failed.connect(self._on_preview_failed)
        self._pool.start(task)

    @QtCore.Slot(object, int, float)
    def _on_preview_finished(self, processed: ProcessedTrial, job_id: int, elapsed_s: float) -> None:
        preserve_view = bool(self._preview_preserve_view_by_job.pop(job_id, False))
        if job_id != self._latest_job_id:
            return  # ignore stale jobs

        key = (processed.path, processed.channel_id)
        cutouts = self._cutout_regions_by_key.get(key, [])
        processed = self._apply_cutouts_to_processed(processed, cutouts)
        self._last_processed[key] = processed

        # Update artifact panel regions list
        start_s, end_s = self._time_window_bounds()
        auto_regs = processed.artifact_regions_auto_sec or []
        auto_regs = self._clip_regions_to_window(auto_regs, start_s, end_s)
        self._auto_regions_by_key[key] = auto_regs
        ignore = self._clip_regions_to_window(self._manual_exclude_by_key.get(key, []), start_s, end_s)
        # build checked list by excluding ignored
        checked_auto = [r for r in auto_regs if not any(self._regions_match(r, ig) for ig in ignore)]
        self.artifact_panel.set_auto_regions(auto_regs, checked_regions=checked_auto)
        manual_regs = self._clip_regions_to_window(self._manual_regions_by_key.get(key, []), start_s, end_s)
        self.artifact_panel.set_regions(manual_regs)

        # Update plots (decimated signals)
        self.plots.update_plots(processed, preserve_view=preserve_view)
        if not preserve_view:
            # Auto-range on each update so file/time-window changes do not require manual reset.
            self._auto_range_for_processed(processed)

        log_msg = (
            f"Preview updated: {processed.output_label} | fs={processed.fs_actual:.2f}->{processed.fs_used:.2f} Hz "
            f"(target {processed.fs_target:.2f}) | n={processed.time.size} | {elapsed_s*1000:.0f} ms"
        )
        self._show_status_message(log_msg, 10000)
        self.param_panel.set_fs_info(processed.fs_actual, processed.fs_target, processed.fs_used)
        self._update_plot_status(fs_actual=float(processed.fs_actual), fs_target=float(processed.fs_target))

        # Inform post tab that current processed changed
        self.post_tab.notify_preprocessing_updated(processed)

    @QtCore.Slot(str, int)
    def _on_preview_failed(self, err: str, job_id: int) -> None:
        self._preview_preserve_view_by_job.pop(job_id, None)
        if job_id != self._latest_job_id:
            return
        self._show_status_message(f"Preview error: {err}")

    # ---------------- Manual artifacts ----------------

    def _regions_match(self, a: Tuple[float, float], b: Tuple[float, float], tol: float = 1e-3) -> bool:
        return (abs(a[0] - b[0]) <= tol) and (abs(a[1] - b[1]) <= tol)

    def _time_window_bounds(self) -> Tuple[Optional[float], Optional[float]]:
        start_s, end_s = self.file_panel.time_window()
        if start_s is not None and end_s is not None and end_s <= start_s:
            return None, None
        return start_s, end_s

    def _clip_regions_to_window(
        self,
        regions: List[Tuple[float, float]],
        start_s: Optional[float],
        end_s: Optional[float],
    ) -> List[Tuple[float, float]]:
        if start_s is None and end_s is None:
            return list(regions)
        lo = -np.inf if start_s is None else float(start_s)
        hi = np.inf if end_s is None else float(end_s)
        out: List[Tuple[float, float]] = []
        for a, b in regions or []:
            t0, t1 = (min(a, b), max(a, b))
            if t1 < lo or t0 > hi:
                continue
            out.append((max(t0, lo), min(t1, hi)))
        return out

    def _merge_regions_with_window(
        self,
        original: List[Tuple[float, float]],
        windowed: List[Tuple[float, float]],
        start_s: Optional[float],
        end_s: Optional[float],
    ) -> List[Tuple[float, float]]:
        if start_s is None and end_s is None:
            out = list(windowed)
            out.sort(key=lambda x: x[0])
            return out
        lo = -np.inf if start_s is None else float(start_s)
        hi = np.inf if end_s is None else float(end_s)
        kept: List[Tuple[float, float]] = []
        for a, b in original or []:
            t0, t1 = (min(a, b), max(a, b))
            if t1 < lo or t0 > hi:
                kept.append((t0, t1))
        out = kept + list(windowed)
        out.sort(key=lambda x: x[0])
        return out

    def _add_manual_region_from_selector(self) -> None:
        key = self._current_key()
        if not key:
            return
        t0, t1 = self.plots.selector_region()
        regs = self._manual_regions_by_key.get(key, [])
        regs.append((min(t0, t1), max(t0, t1)))
        self._manual_regions_by_key[key] = regs
        start_s, end_s = self._time_window_bounds()
        self.artifact_panel.set_regions(self._clip_regions_to_window(regs, start_s, end_s))
        self._trigger_preview(preserve_view=True)

    def _add_manual_region_from_drag(self, t0: float, t1: float) -> None:
        if self._box_select_callback:
            cb = self._box_select_callback
            self._box_select_callback = None
            self.plots.btn_box_select.setChecked(False)
            cb(float(min(t0, t1)), float(max(t0, t1)))
            return
        key = self._current_key()
        if not key:
            return
        if not np.isfinite(t0) or not np.isfinite(t1) or t0 == t1:
            return
        region = (float(min(t0, t1)), float(max(t0, t1)))
        self._pending_box_region_by_key[key] = region
        self.plots.set_selector_region(*region, visible=True)
        self._show_status_message("Selection ready: press A=artifact, C=cut, S=section, or right-click for actions.")

    def _clear_manual_regions_current(self) -> None:
        key = self._current_key()
        if not key:
            return
        self._manual_regions_by_key[key] = []
        self._manual_exclude_by_key[key] = []
        self._pending_box_region_by_key.pop(key, None)
        self.artifact_panel.set_regions([])
        self._trigger_preview(preserve_view=True)

    def _request_box_select(self, callback: Callable[[float, float], None]) -> None:
        self._box_select_callback = callback
        self.plots.btn_box_select.setChecked(True)
        self._show_status_message("Box select: drag on the raw plot to set the time window; right-click to cancel.")

    def _cancel_box_select_request(self) -> None:
        key = self._current_key()
        self._box_select_callback = None
        if key:
            self._pending_box_region_by_key.pop(key, None)
        self.plots.set_selector_region(0.0, 1.0, visible=False)
        self.plots.btn_box_select.setChecked(False)

    def _pending_box_region(self) -> Optional[Tuple[float, float]]:
        key = self._current_key()
        if not key:
            return None
        region = self._pending_box_region_by_key.get(key)
        if not region:
            if not self.plots.selector_visible():
                return None
            t0, t1 = self.plots.selector_region()
            return (float(min(t0, t1)), float(max(t0, t1)))
        return (float(min(region)), float(max(region)))

    def _consume_pending_box_region(self) -> Optional[Tuple[float, float]]:
        key = self._current_key()
        if not key:
            return None
        
        is_tool_active = self.plots.btn_box_select.isChecked()
        region = self._pending_box_region_by_key.pop(key, None)
        
        if not region:
            if not self.plots.selector_visible():
                return None
            t0, t1 = self.plots.selector_region()
            region = (float(min(t0, t1)), float(max(t0, t1)))
            # If not using the box-select tool, we do NOT hide the persistent selector.
            if not is_tool_active:
                return region

        # Cleanup if we were in tool mode or had a temporary drag selection.
        self.plots.set_selector_region(0.0, 1.0, visible=False)
        self.plots.btn_box_select.setChecked(False)
        return (float(min(region)), float(max(region)))

    def _assign_pending_box_to_artifact(self) -> None:
        region = self._consume_pending_box_region()
        key = self._current_key()
        if not region or not key:
            return
        regs = self._manual_regions_by_key.get(key, [])
        regs.append(region)
        self._manual_regions_by_key[key] = regs
        start_s, end_s = self._time_window_bounds()
        self.artifact_panel.set_regions(self._clip_regions_to_window(regs, start_s, end_s))
        self._trigger_preview(preserve_view=True)

    def _assign_pending_box_to_cut(self) -> None:
        region = self._consume_pending_box_region()
        key = self._current_key()
        if not region or not key:
            return
        regs = self._cutout_regions_by_key.get(key, [])
        regs.append(region)
        regs.sort(key=lambda x: x[0])
        self._cutout_regions_by_key[key] = regs
        self._last_processed.clear()
        self._update_raw_plot()
        self._trigger_preview()

    def _assign_pending_box_to_section(self) -> None:
        region = self._consume_pending_box_region()
        key = self._current_key()
        if not region or not key:
            return
        sections = self._sections_by_key.get(key, [])
        sections.append({
            "start": float(region[0]),
            "end": float(region[1]),
            "params": self.param_panel.get_params().to_dict(),
        })
        sections.sort(key=lambda sec: float(sec.get("start", 0.0)))
        self._sections_by_key[key] = sections
        self._show_status_message(f"Section added: {region[0]:.3f}s to {region[1]:.3f}s")

    def _show_box_selection_context_menu(self) -> None:
        region = self._pending_box_region()
        if region is None:
            self._cancel_box_select_request()
            return
        menu = QtWidgets.QMenu(self)
        act_art = menu.addAction("Set as artifact")
        act_cut = menu.addAction("Set as cut")
        act_sec = menu.addAction("Set as section")
        menu.addSeparator()
        act_cancel = menu.addAction("Cancel selection")
        chosen = menu.exec(QtGui.QCursor.pos())
        if chosen is act_art:
            self._assign_pending_box_to_artifact()
        elif chosen is act_cut:
            self._assign_pending_box_to_cut()
        elif chosen is act_sec:
            self._assign_pending_box_to_section()
        elif chosen is act_cancel:
            self._cancel_box_select_request()

    def _artifact_regions_changed(self, regions: List[Tuple[float, float]]) -> None:
        key = self._current_key()
        if not key:
            return
        start_s, end_s = self._time_window_bounds()
        auto = self._clip_regions_to_window(self._auto_regions_by_key.get(key, []), start_s, end_s)

        def _contains(target: Tuple[float, float], arr: List[Tuple[float, float]]) -> bool:
            return any(self._regions_match(target, other) for other in arr)

        manual_add = [r for r in regions if not _contains(r, auto)]
        manual_ignore = [r for r in auto if not _contains(r, regions)]

        prev_manual = self._manual_regions_by_key.get(key, [])
        prev_ignore = self._manual_exclude_by_key.get(key, [])
        self._manual_regions_by_key[key] = self._merge_regions_with_window(prev_manual, manual_add, start_s, end_s)
        self._manual_exclude_by_key[key] = self._merge_regions_with_window(prev_ignore, manual_ignore, start_s, end_s)
        self._trigger_preview(preserve_view=True)

    def _toggle_artifacts_panel(self) -> None:
        if self._use_pg_dockarea_pre_layout:
            self._setup_section_popups()
            dock = self._pre_dockarea_dock("artifacts_list")
            if dock is None:
                return
            if dock.isVisible():
                dock.hide()
            else:
                dock.show()
                try:
                    dock.raiseDock()
                except Exception:
                    pass
                self._last_opened_section = "artifacts_list"
            self._sync_section_button_states_from_docks()
            self._save_panel_layout_state()
            return

        if isinstance(self.art_dock, QtWidgets.QDockWidget):
            if self.art_dock.isVisible():
                self.art_dock.setVisible(False)
                self._save_panel_layout_state()
                return
            self.artifact_panel.show()
            self.art_dock.setVisible(True)
        self._save_panel_layout_state()

    # ---------------- Metadata ----------------

    def _edit_metadata_for_current(self) -> None:
        if not self._current_path:
            QtWidgets.QMessageBox.information(self, "Metadata", "Select a file first.")
            return
        doric = self._loaded_files.get(self._current_path)
        if not doric:
            QtWidgets.QMessageBox.warning(self, "Metadata", "Current file is not loaded.")
            return
        if not doric.channels:
            QtWidgets.QMessageBox.warning(self, "Metadata", "No channels available for metadata editing.")
            return

        # existing per channel
        existing: Dict[str, Dict[str, str]] = {}
        for ch in doric.channels:
            existing[ch] = self._metadata_by_key.get((self._current_path, ch), {})

        defaults: Dict[str, str] = {}
        try:
            raw = self.settings.value("last_metadata_template", "", type=str)
            if raw:
                defaults = json.loads(raw)
        except Exception:
            defaults = {}

        dlg = MetadataDialog(channels=doric.channels, existing=existing, defaults=defaults, parent=self)
        if dlg.exec() != QtWidgets.QDialog.DialogCode.Accepted:
            return
        meta = dlg.get_metadata()
        for ch, md in meta.items():
            self._metadata_by_key[(self._current_path, ch)] = md
        try:
            if self._current_channel and self._current_channel in meta:
                self.settings.setValue("last_metadata_template", json.dumps(meta[self._current_channel]))
            elif meta:
                first = next(iter(meta.values()))
                self.settings.setValue("last_metadata_template", json.dumps(first))
        except Exception:
            pass

    # ---------------- Export (multi-file) ----------------

    def _export_origin_dir(self, selected_paths: List[str]) -> str:
        if selected_paths:
            d = os.path.dirname(selected_paths[0])
            if d and os.path.isdir(d):
                return d
        hint = self.file_panel.current_dir_hint()
        if hint and os.path.isdir(hint):
            return hint
        return ""

    def _export_start_dir(self, selected_paths: List[str]) -> str:
        origin_dir = self._export_origin_dir(selected_paths)
        last_dir = self.settings.value("last_save_dir", "", type=str)
        override = self.settings.value("last_save_dir_override", False, type=bool)

        def _valid(p: str) -> bool:
            return bool(p) and os.path.isdir(p)

        if override and _valid(last_dir):
            return last_dir
        if _valid(origin_dir):
            return origin_dir
        if _valid(last_dir):
            return last_dir
        return os.getcwd()

    def _remember_export_dir(self, out_dir: str, origin_dir: str) -> None:
        try:
            self.settings.setValue("last_save_dir", out_dir)
            out_norm = os.path.normcase(os.path.abspath(out_dir)) if out_dir else ""
            origin_norm = os.path.normcase(os.path.abspath(origin_dir)) if origin_dir else ""
            override = bool(out_norm) and (not origin_norm or out_norm != origin_norm)
            self.settings.setValue("last_save_dir_override", override)
        except Exception:
            pass

    def _process_trial_for_export(
        self,
        trial: LoadedTrial,
        params: ProcessingParams,
        export_selection: ExportSelection,
        manual_regions_sec: List[Tuple[float, float]],
        manual_exclude_regions_sec: List[Tuple[float, float]],
    ) -> ProcessedTrial:
        modes: List[str] = []
        if export_selection.output:
            for mode in export_selection.output_modes or [params.output_mode]:
                mode = str(mode or "").strip()
                if mode in OUTPUT_MODES and mode not in modes:
                    modes.append(mode)
        if not modes:
            modes = [params.output_mode if params.output_mode in OUTPUT_MODES else OUTPUT_MODES[0]]

        primary = params.output_mode if params.output_mode in modes else modes[0]
        ordered_modes = [primary] + [mode for mode in modes if mode != primary]
        base_processed: Optional[ProcessedTrial] = None
        outputs: Dict[str, np.ndarray] = {}

        for mode in ordered_modes:
            mode_params = ProcessingParams.from_dict(params.to_dict())
            mode_params.output_mode = mode
            processed = self.processor.process_trial(
                trial=trial,
                params=mode_params,
                manual_regions_sec=manual_regions_sec,
                manual_exclude_regions_sec=manual_exclude_regions_sec,
                preview_mode=False,
            )
            if base_processed is None:
                base_processed = processed
            if processed.output is not None:
                outputs[str(processed.output_label or mode)] = np.asarray(processed.output, float)

        if base_processed is None:
            fallback_params = ProcessingParams.from_dict(params.to_dict())
            base_processed = self.processor.process_trial(
                trial=trial,
                params=fallback_params,
                manual_regions_sec=manual_regions_sec,
                manual_exclude_regions_sec=manual_exclude_regions_sec,
                preview_mode=False,
            )
        if export_selection.output:
            base_processed.outputs = outputs
        return base_processed

    def _export_selected_or_all(self) -> None:
        selected = self._selected_paths()
        if not selected:
            selected = self.file_panel.all_paths()
        if not selected:
            return

        auto_export = bool(self.param_panel.auto_export_enabled())
        origin_dir = self._export_origin_dir(selected)
        out_dir = ""
        if not auto_export:
            start_dir = self._export_start_dir(selected)
            out_dir = QtWidgets.QFileDialog.getExistingDirectory(self, "Select export folder", start_dir)
            if not out_dir:
                return
            self._remember_export_dir(out_dir, origin_dir)

        params = self.param_panel.get_params()
        export_selection = self.param_panel.export_selection()
        export_channel_names = [] if auto_export else self.param_panel.export_channel_names()
        export_trigger_names = self.param_panel.export_trigger_names()

        # Process/export each selected file. Auto export writes beside each source
        # file and intentionally exports every analog channel with the same params.
        n_total = 0
        exported_dirs = set()
        for path in selected:
            doric = self._loaded_files.get(path)
            if not doric:
                continue
            if auto_export:
                channels = list(doric.channels)
            else:
                channels = [name for name in export_channel_names if name in doric.channels]
                if not channels:
                    fallback = self._current_channel if (self._current_channel in doric.channels) else (doric.channels[0] if doric.channels else None)
                    channels = [fallback] if fallback else []
            path_out_dir = out_dir
            if auto_export:
                path_out_dir = os.path.dirname(path)
                if not path_out_dir or not os.path.isdir(path_out_dir):
                    path_out_dir = origin_dir if origin_dir and os.path.isdir(origin_dir) else os.getcwd()
            dio_names = [name for name in export_trigger_names if name in doric.trigger_by_name]
            if not export_selection.dio:
                dio_names = [None]
            elif not dio_names:
                dio_names = [self._current_trigger] if self._current_trigger else [None]

            for ch in channels:
                if not ch:
                    continue
                key = (path, ch)
                cutouts = self._cutout_regions_by_key.get(key, [])
                start_s, end_s = self._time_window_bounds()
                manual = self._clip_regions_to_window(self._manual_regions_by_key.get(key, []), start_s, end_s)
                manual_exclude = self._clip_regions_to_window(self._manual_exclude_by_key.get(key, []), start_s, end_s)
                meta = self._metadata_by_key.get(key, {})
                sections = self._sections_by_key.get(key, [])

                # Use all selected triggers for one export per channel
                # If current trigger is in dio_names, use it as primary for alignment.
                # Otherwise, pick first available as primary.
                primary_trigger = None
                if export_selection.dio and dio_names:
                    primary_trigger = (self._current_trigger if self._current_trigger in dio_names else (dio_names[0] if dio_names[0] else None))

                trial = doric.make_trial(ch, trigger_name=primary_trigger, trigger_names=(dio_names if export_selection.dio else None))
                trial = self._apply_time_window(trial)
                trial = self._apply_cutouts(trial, cutouts)

                def _export_one(proc: ProcessedTrial, suffix: str = "") -> None:
                    nonlocal n_total
                    proc = self._apply_cutouts_to_processed(proc, cutouts)
                    stem = safe_stem_from_metadata(path, ch, meta)
                    if suffix:
                        stem = f"{stem}_{suffix}"
                    csv_path = os.path.join(path_out_dir, f"{stem}.csv")
                    h5_path = os.path.join(path_out_dir, f"{stem}.h5")
                    export_processed_csv(csv_path, proc, metadata=meta, selection=export_selection)
                    export_processed_h5(h5_path, proc, metadata=meta, selection=export_selection)
                    exported_dirs.add(path_out_dir)
                    n_total += 1

                try:
                    if sections:
                        for i, sec in enumerate(sections, start=1):
                            s0 = float(sec.get("start", 0.0))
                            s1 = float(sec.get("end", 0.0))
                            sec_trial = self._slice_trial(trial, s0, s1)
                            if sec_trial is None:
                                continue
                            sec_params = ProcessingParams.from_dict(sec.get("params", {})) if isinstance(sec.get("params"), dict) else params
                            processed = self._process_trial_for_export(
                                trial=sec_trial,
                                params=sec_params,
                                export_selection=export_selection,
                                manual_regions_sec=manual,
                                manual_exclude_regions_sec=manual_exclude,
                            )
                            _export_one(processed, suffix=f"sec{i}_{s0:.2f}_{s1:.2f}")
                    else:
                        processed = self._process_trial_for_export(
                            trial=trial,
                            params=params,
                            export_selection=export_selection,
                            manual_regions_sec=manual,
                            manual_exclude_regions_sec=manual_exclude,
                        )
                        _export_one(processed)
                except Exception as e:
                    QtWidgets.QMessageBox.warning(
                        self,
                        "Export error",
                        f"Failed export:\n{path} [{ch}] [{primary_trigger or 'no DIO'}]\n\n{e}",
                    )
        if auto_export:
            if len(exported_dirs) == 1:
                target = next(iter(exported_dirs))
            elif exported_dirs:
                target = f"{len(exported_dirs)} source folders"
            else:
                target = "source folders"
        else:
            target = out_dir
        self._show_status_message(f"Export complete: {n_total} recording(s) written to {target}")

        # optional: update post tab list by loading exported results? (user can load later)

    # ---------------- Postprocessing bridge ----------------

    @QtCore.Slot()
    def _post_get_current_processed(self):
        # Determine selection context: if multiple selected, provide multiple processed outputs if available
        paths = self._selected_paths()
        if not paths:
            paths = [self._current_path] if self._current_path else []

        out: List[ProcessedTrial] = []
        for p in paths:
            doric = self._loaded_files.get(p)
            if not doric:
                continue
            # Use current channel when available for all selected files
            if self._current_channel and self._current_channel in doric.channels:
                ch = self._current_channel
            else:
                ch = doric.channels[0] if doric.channels else "AIN01"
            key = (p, ch)
            if key in self._last_processed:
                out.append(self._last_processed[key])
            else:
                # compute on-demand (fast due to decimation), using current params
                try:
                    params = self.param_panel.get_params()
                    trial = doric.make_trial(ch, trigger_name=self._current_trigger)
                    trial = self._apply_time_window(trial)
                    start_s, end_s = self._time_window_bounds()
                    manual = self._clip_regions_to_window(self._manual_regions_by_key.get(key, []), start_s, end_s)
                    manual_exclude = self._clip_regions_to_window(self._manual_exclude_by_key.get(key, []), start_s, end_s)
                    proc = self.processor.process_trial(
                        trial,
                        params,
                        manual_regions_sec=manual,
                        manual_exclude_regions_sec=manual_exclude,
                        preview_mode=False,
                    )
                    cutouts = self._cutout_regions_by_key.get(key, [])
                    proc = self._apply_cutouts_to_processed(proc, cutouts)
                    self._last_processed[key] = proc
                    out.append(proc)
                except Exception:
                    pass

        self.post_tab.receive_current_processed(out)

    @QtCore.Slot()
    def _post_get_current_dio_list(self):
        # Analog/digital channel list for current/selected files: union.
        paths = self._selected_paths()
        if not paths:
            paths = [self._current_path] if self._current_path else []

        dio = set()
        for p in paths:
            f = self._loaded_files.get(p)
            if f:
                dio |= set(f.trigger_by_name.keys())
        self.post_tab.receive_dio_list(sorted(dio))

    @QtCore.Slot(str, str)
    def _post_get_dio_data_for_path(self, path: str, dio_name: str):
        """
        Returns (t_dio, y_dio) for the requested dio_name for a given *raw* path
        currently loaded/parsed in the cache.

        Fixes numpy array truth-value ambiguity by checking None/len explicitly.
        """
        f = self._loaded_files.get(path, None)

        if f is None:
            return

        trigger_map = getattr(f, "trigger_by_name", None)
        if not isinstance(trigger_map, dict) or dio_name not in trigger_map:
            return

        y_dio = np.asarray(trigger_map[dio_name], float)
        if y_dio.size == 0:
            return

        t_map = getattr(f, "trigger_time_by_name", None)
        t_dio = None
        if isinstance(t_map, dict):
            t_dio = t_map.get(dio_name)
        if t_dio is None and getattr(f, "digital_time", None) is not None and dio_name in getattr(f, "digital_by_name", {}):
            t_dio = f.digital_time
        if t_dio is None:
            # Fallback: use any analog channel timebase with matching length.
            for t_candidate in getattr(f, "time_by_channel", {}).values():
                arr = np.asarray(t_candidate, float)
                if arr.size == y_dio.size:
                    t_dio = arr
                    break
        if t_dio is None:
            return
        t_dio = np.asarray(t_dio, float)
        if t_dio.size == 0:
            return

        # Ensure same length.
        n = min(t_dio.size, y_dio.size)
        t_dio = t_dio[:n]
        y_dio = y_dio[:n]

        self.post_tab.receive_dio_data(path, dio_name, t_dio, y_dio)
        return

    # ---------------- Drag and drop ----------------

    def dragEnterEvent(self, event) -> None:
        if event.mimeData().hasUrls():
            paths = [u.toLocalFile() for u in event.mimeData().urls()]
            if self._can_accept_drop(paths):
                event.acceptProposedAction()
                return
        event.ignore()

    def dropEvent(self, event) -> None:
        if not event.mimeData().hasUrls():
            event.ignore()
            return
        paths = [u.toLocalFile() for u in event.mimeData().urls()]
        self._handle_drop(paths)
        event.acceptProposedAction()

    def _can_accept_drop(self, paths: List[str]) -> bool:
        known_ext = (".doric", ".h5", ".hdf5", ".csv")
        return any(p.lower().endswith(known_ext) for p in paths)

    def _handle_drop(self, paths: List[str]) -> None:
        doric_paths: List[str] = []
        processed: List[ProcessedTrial] = []
        pre_active = bool(hasattr(self, "tabs") and self.tabs.currentWidget() is self.pre_tab)

        for p in paths:
            if not p:
                continue
            ext = os.path.splitext(p)[1].lower()
            if ext == ".doric":
                doric_paths.append(p)
                continue
            if ext == ".csv":
                if pre_active:
                    doric_paths.append(p)
                    continue
                trial = self._load_processed_csv(p)
                if trial is not None:
                    processed.append(trial)
                continue
            if ext in (".h5", ".hdf5"):
                if pre_active:
                    # On preprocessing tab, treat dropped H5 as a raw-input candidate.
                    # _add_files() will load Doric H5 natively and falls back to processed H5 import.
                    doric_paths.append(p)
                    continue
                trial = self._load_processed_h5(p)
                if trial is not None:
                    processed.append(trial)
                else:
                    doric_paths.append(p)

        if doric_paths:
            self._add_files(doric_paths)
        if processed:
            self.post_tab.append_processed(processed)

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
                tval = self._parse_csv_float(r[time_idx])
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
                elif "prominence" in g:
                    out = np.asarray(g["prominence"][()], float)
                elif "raw_465" in g:
                    out = np.asarray(g["raw_465"][()], float)
                elif "raw" in g:
                    out = np.asarray(g["raw"][()], float)
                elif "raw_405" in g:
                    out = np.asarray(g["raw_405"][()], float)
                elif "isobestic" in g:
                    out = np.asarray(g["isobestic"][()], float)
                elif "dio" in g:
                    out = np.asarray(g["dio"][()], float)
                elif "baseline_465" in g:
                    out = np.asarray(g["baseline_465"][()], float)
                elif "baseline_405" in g:
                    out = np.asarray(g["baseline_405"][()], float)
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

    def _processed_trial_to_loaded_doric(self, processed: ProcessedTrial) -> Optional[LoadedDoricFile]:
        t = np.asarray(processed.time if processed.time is not None else np.array([], float), float)
        if t.size < 2:
            return None

        raw_sig = np.asarray(
            processed.raw_signal if processed.raw_signal is not None else np.array([], float),
            float,
        )
        raw_ref = np.asarray(
            processed.raw_reference if processed.raw_reference is not None else np.array([], float),
            float,
        )
        out = np.asarray(processed.output if processed.output is not None else np.array([], float), float)

        if raw_sig.size != t.size:
            if out.size == t.size:
                raw_sig = out.copy()
            else:
                return None
        if raw_ref.size != t.size or not np.isfinite(raw_ref).any():
            # Keep preprocessing numerically stable even if original H5 has no raw_405.
            raw_ref = raw_sig.copy()

        if not np.isfinite(raw_sig).any():
            return None

        channel = str(processed.channel_id or "AIN01").strip() or "AIN01"
        dio_map: Dict[str, np.ndarray] = {}
        dio_time_map: Dict[str, np.ndarray] = {}
        digital_time: Optional[np.ndarray] = None
        dio = np.asarray(processed.dio, float) if processed.dio is not None else np.array([], float)
        if dio.size == t.size:
            dio_name = str(processed.dio_name or "DIO_import").strip() or "DIO_import"
            digital_time = t.copy()
            dio_map[dio_name] = dio.copy()
            dio_time_map[dio_name] = t.copy()

        return LoadedDoricFile(
            path=str(processed.path or ""),
            channels=[channel],
            time_by_channel={channel: t.copy()},
            signal_by_channel={channel: raw_sig.copy()},
            reference_by_channel={channel: raw_ref.copy()},
            digital_time=digital_time,
            digital_by_name={k: v.copy() for k, v in dio_map.items()},
            trigger_time_by_name={k: v.copy() for k, v in dio_time_map.items()},
            trigger_by_name={k: v.copy() for k, v in dio_map.items()},
        )

    def _load_processed_h5_as_pre_file(self, path: str) -> Optional[LoadedDoricFile]:
        processed = self._load_processed_h5(path)
        if processed is None:
            return None
        loaded = self._processed_trial_to_loaded_doric(processed)
        if loaded is None:
            return None
        loaded.path = path
        return loaded

    def closeEvent(self, event):
        try:
            self.post_tab.mark_app_closing()
        except Exception:
            pass
        try:
            current = self.tabs.currentWidget() if hasattr(self, "tabs") else None
            if current is self.pre_tab:
                # Closing on preprocessing: capture the live preprocessing dock topology.
                self._store_pre_main_dock_snapshot()
            else:
                # Closing on postprocessing: preprocessing docks are hidden by tab switch.
                self._persist_hidden_preprocessing_layout_state()
        except Exception:
            pass
        try:
            # Persist post layout from live state or cached tab-switch state without
            # overwriting it with preprocessing topology.
            self.post_tab.persist_layout_state_snapshot()
        except Exception:
            pass
        try:
            self.post_tab._on_about_to_quit()
        except Exception:
            pass
        self._save_panel_layout_state()
        self._save_panel_config_json()
        self._save_settings()
        try:
            self.settings.sync()
        except Exception:
            pass
        super().closeEvent(event)


def main() -> None:
    pg.setConfigOptions(antialias=False)
    smoke_test = str(os.environ.get("PYBER_SMOKE_TEST", "")).strip().lower() in {"1", "true", "yes", "on"}
    app = QtWidgets.QApplication([])
    apply_app_palette(app, "dark")
    spinbox_scrubber = install_spinbox_scrubbers(app)
    icon_path = _pyber_icon_path()
    try:
        if os.path.isfile(icon_path):
            app_icon = QtGui.QIcon(icon_path)
            if not app_icon.isNull():
                app.setWindowIcon(app_icon)
    except Exception:
        pass
    splash = None
    if not smoke_test:
        try:
            if os.path.isfile(icon_path):
                pix = QtGui.QPixmap(icon_path)
                if not pix.isNull():
                    splash = QtWidgets.QSplashScreen(pix, QtCore.Qt.WindowType.WindowStaysOnTopHint)
                    splash.show()
                    app.processEvents(QtCore.QEventLoop.ProcessEventsFlag.AllEvents)
        except Exception:
            splash = None
    w = MainWindow()
    spinbox_scrubber.scan(w)

    if smoke_test:
        try:
            w.show()
            app.processEvents(QtCore.QEventLoop.ProcessEventsFlag.AllEvents)
            if hasattr(w, "tabs") and hasattr(w, "post_tab"):
                idx = w.tabs.indexOf(w.post_tab)
                if idx >= 0:
                    w.tabs.setCurrentIndex(idx)
                for _ in range(8):
                    app.processEvents(QtCore.QEventLoop.ProcessEventsFlag.AllEvents)
            try:
                if hasattr(w, "post_tab") and hasattr(w.post_tab, "ensure_section_popups_initialized"):
                    w.post_tab.ensure_section_popups_initialized()
            except Exception:
                pass
            for _ in range(8):
                app.processEvents(QtCore.QEventLoop.ProcessEventsFlag.AllEvents)
        finally:
            try:
                w.close()
            except Exception:
                pass
            app.processEvents(QtCore.QEventLoop.ProcessEventsFlag.AllEvents)
        return

    w.show()
    if splash is not None:
        splash.finish(w)
    app.exec()


if __name__ == "__main__":
    main()
