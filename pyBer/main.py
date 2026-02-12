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
from typing import Callable, Dict, List, Optional, Tuple

from PySide6 import QtCore, QtGui, QtWidgets
import pyqtgraph as pg
import h5py

from analysis_core import (
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
from styles import APP_QSS
import numpy as np


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
_PRE_DOCK_PREFIX = "pre."
_POST_DOCK_PREFIX = "post."
_FORCE_FIXED_DOCK_LAYOUTS = True

_LOG = logging.getLogger(__name__)


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

class MainWindow(QtWidgets.QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Pyber - Fiber Photometry")
        self.resize(1500, 900)
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

        self._manual_regions_by_key: Dict[Tuple[str, str], List[Tuple[float, float]]] = {}
        self._manual_exclude_by_key: Dict[Tuple[str, str], List[Tuple[float, float]]] = {}
        self._auto_regions_by_key: Dict[Tuple[str, str], List[Tuple[float, float]]] = {}
        self._metadata_by_key: Dict[Tuple[str, str], Dict[str, str]] = {}
        self._cutout_regions_by_key: Dict[Tuple[str, str], List[Tuple[float, float]]] = {}
        self._sections_by_key: Dict[Tuple[str, str], List[Dict[str, object]]] = {}

        self._last_processed: Dict[Tuple[str, str], ProcessedTrial] = {}
        self._advanced_dialog: Optional[AdvancedOptionsDialog] = None
        self._box_select_callback: Optional[Callable[[float, float], None]] = None
        self._last_artifact_params: Optional[Tuple[object, ...]] = None
        self._section_docks: Dict[str, QtWidgets.QDockWidget] = {}
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
        self._force_fixed_dock_layouts: bool = bool(_FORCE_FIXED_DOCK_LAYOUTS)

        # Worker infra (stable)
        self._pool = QtCore.QThreadPool.globalInstance()
        self._job_counter = 0
        self._latest_job_id = 0

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

    # ---------------- UI ----------------

    def _build_ui(self) -> None:
        self.setStyleSheet(APP_QSS)

        self.tabs = QtWidgets.QTabWidget()
        self.setCentralWidget(self.tabs)
        self._status_bar = QtWidgets.QStatusBar(self)
        self.setStatusBar(self._status_bar)

        # Preprocessing tab
        self.pre_tab = QtWidgets.QWidget()
        self.tabs.addTab(self.pre_tab, "Preprocessing")

        self.file_panel = FileQueuePanel(self.pre_tab)
        self.param_panel = ParameterPanel(self.pre_tab)
        self.param_panel.setVisible(False)
        self.plots = PlotDashboard(self.pre_tab)
        self.artifact_panel = ArtifactPanel(self.pre_tab)

        # Right artifact panel dock (hidden by default)
        self.art_dock = QtWidgets.QDockWidget("Artifacts list", self)
        self.art_dock.setObjectName("pre.artifact.dock")
        self.art_dock.setWidget(self.artifact_panel)
        self.art_dock.setAllowedAreas(QtCore.Qt.DockWidgetArea.RightDockWidgetArea)
        self.art_dock.setVisible(False)
        self.art_dock.visibilityChanged.connect(lambda *_: self._save_panel_layout_state())
        self.art_dock.topLevelChanged.connect(lambda *_: self._save_panel_layout_state())
        self.art_dock.dockLocationChanged.connect(lambda *_: self._save_panel_layout_state())
        self.art_dock.installEventFilter(self)
        self.artifact_panel.installEventFilter(self)
        self.addDockWidget(QtCore.Qt.DockWidgetArea.RightDockWidgetArea, self.art_dock)

        # Left pane: data browser
        self.file_panel.setMinimumWidth(320)
        self.file_panel.setMaximumWidth(380)
        self.file_panel.setSizePolicy(QtWidgets.QSizePolicy.Policy.Fixed, QtWidgets.QSizePolicy.Policy.Expanding)

        # Center pane: workflow toolbar + plots
        self.btn_workflow_load = QtWidgets.QPushButton("Load")
        self.btn_workflow_load.setProperty("class", "blueSecondarySmall")
        self.menu_workflow_load = QtWidgets.QMenu(self.btn_workflow_load)
        self.act_open_file = self.menu_workflow_load.addAction("Open File...")
        self.act_add_folder = self.menu_workflow_load.addAction("Add Folder...")
        self.menu_workflow_load.addSeparator()
        self.act_focus_data = self.menu_workflow_load.addAction("Focus Data Browser")
        self.btn_workflow_load.setMenu(self.menu_workflow_load)

        self.btn_workflow_artifacts = QtWidgets.QPushButton("Detected artifacts")
        self.btn_workflow_qc = QtWidgets.QPushButton("QC")
        self.btn_workflow_export = QtWidgets.QPushButton("Export")
        self.btn_toggle_data = QtWidgets.QPushButton("Data")
        self.btn_toggle_data.setCheckable(True)
        self.btn_toggle_data.setChecked(True)
        self.btn_toggle_data.setProperty("class", "blueSecondarySmall")
        self.btn_workflow_export.setProperty("class", "bluePrimarySmall")
        for b in (self.btn_workflow_artifacts, self.btn_workflow_qc):
            b.setProperty("class", "blueSecondarySmall")

        # Inline parameter section buttons (same row as workflow actions).
        self.btn_section_artifacts = QtWidgets.QPushButton("Artifacts")
        self.btn_section_filtering = QtWidgets.QPushButton("Filtering")
        self.btn_section_baseline = QtWidgets.QPushButton("Baseline")
        self.btn_section_output = QtWidgets.QPushButton("Output")
        self.btn_section_qc = QtWidgets.QPushButton("QC")
        self.btn_section_export = QtWidgets.QPushButton("Export")
        self.btn_section_config = QtWidgets.QPushButton("Configuration")
        self._section_buttons: Dict[str, QtWidgets.QPushButton] = {
            "artifacts": self.btn_section_artifacts,
            "filtering": self.btn_section_filtering,
            "baseline": self.btn_section_baseline,
            "output": self.btn_section_output,
            "qc": self.btn_section_qc,
            "export": self.btn_section_export,
            "config": self.btn_section_config,
        }
        for btn in self._section_buttons.values():
            btn.setCheckable(True)
            btn.setProperty("class", "blueSecondarySmall")
            btn.setSizePolicy(QtWidgets.QSizePolicy.Policy.Fixed, QtWidgets.QSizePolicy.Policy.Fixed)

        workflow_row = QtWidgets.QHBoxLayout()
        workflow_row.setContentsMargins(0, 0, 0, 0)
        workflow_row.setSpacing(6)
        workflow_row.addWidget(self.btn_toggle_data)
        workflow_row.addWidget(self.btn_workflow_load)
        workflow_row.addWidget(self.btn_workflow_artifacts)
        workflow_row.addWidget(self.btn_workflow_qc)
        workflow_row.addWidget(self.btn_workflow_export)
        workflow_row.addSpacing(8)
        workflow_row.addWidget(QtWidgets.QLabel("Parameters:"))
        for btn in self._section_buttons.values():
            workflow_row.addWidget(btn)
        workflow_row.addStretch(1)

        center_widget = QtWidgets.QWidget()
        center_v = QtWidgets.QVBoxLayout(center_widget)
        center_v.setContentsMargins(0, 0, 0, 0)
        center_v.setSpacing(6)
        center_v.addLayout(workflow_row)
        center_v.addWidget(self.plots, stretch=1)

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
        if hasattr(self.post_tab, "set_force_fixed_default_layout"):
            try:
                self.post_tab.set_force_fixed_default_layout(self._force_fixed_dock_layouts)
            except Exception:
                pass
        self.tabs.addTab(self.post_tab, "Post Processing")
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
        self.act_open_file.triggered.connect(self._open_files_dialog)
        self.act_add_folder.triggered.connect(self._open_folder_dialog)
        self.act_focus_data.triggered.connect(self._focus_data_browser)
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
        """Create one floating popup per processing section and reuse existing controls."""
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
            "artifacts": self.param_panel.card_artifacts,
            "filtering": self.param_panel.card_filtering,
            "baseline": self.param_panel.card_baseline,
            "output": self.param_panel.card_output,
            "qc": self._build_qc_actions_widget(),
            "export": self._build_export_actions_widget(),
            "config": self._build_config_actions_widget(),
        }
        section_titles: Dict[str, str] = {
            "artifacts": "Artifacts",
            "filtering": "Filtering",
            "baseline": "Baseline",
            "output": "Output",
            "qc": "QC",
            "export": "Export",
            "config": "Configuration",
        }

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
            self.addDockWidget(QtCore.Qt.DockWidgetArea.RightDockWidgetArea, dock)
            dock.setFloating(True)
            dock.hide()
            dock.installEventFilter(self)
            widget.installEventFilter(self)
            self._section_docks[key] = dock

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
        self.param_panel.btn_export.setProperty("class", "bluePrimarySmall")
        v.addWidget(self.param_panel.btn_export)
        v.addStretch(1)
        return panel

    def _build_config_actions_widget(self) -> QtWidgets.QWidget:
        panel = QtWidgets.QWidget()
        row = QtWidgets.QHBoxLayout(panel)
        row.setContentsMargins(8, 8, 8, 8)
        row.setSpacing(6)
        self.param_panel.btn_metadata.setProperty("class", "blueSecondarySmall")
        self.param_panel.btn_advanced.setProperty("class", "blueSecondarySmall")
        self.param_panel.btn_save_config.setProperty("class", "blueSecondarySmall")
        self.param_panel.btn_load_config.setProperty("class", "blueSecondarySmall")
        for btn in (
            self.param_panel.btn_metadata,
            self.param_panel.btn_advanced,
            self.param_panel.btn_save_config,
            self.param_panel.btn_load_config,
        ):
            btn.setSizePolicy(QtWidgets.QSizePolicy.Policy.Expanding, QtWidgets.QSizePolicy.Policy.Fixed)
            row.addWidget(btn)
        return panel

    def _set_section_button_checked(self, key: str, checked: bool) -> None:
        btn = self._section_buttons.get(key)
        if btn is None:
            return
        btn.blockSignals(True)
        btn.setChecked(bool(checked))
        btn.blockSignals(False)

    def _toggle_section_popup(self, key: str, checked: bool) -> None:
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
            self._last_opened_section = key
        else:
            dock.hide()

    def _on_section_dock_visibility(self, key: str, visible: bool) -> None:
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

        # Prefer right side of main window, then fall back to left, then clamp.
        x_right = geom.x() + geom.width() + 12
        x_left = geom.x() - width - 12
        y_pref = geom.y() + 60

        x_min = screen_rect.x() + 10
        y_min = screen_rect.y() + 10
        x_max = screen_rect.x() + max(10, screen_rect.width() - width - 10)
        y_max = screen_rect.y() + max(10, screen_rect.height() - height - 10)

        if x_right <= x_max:
            x = x_right
        elif x_left >= x_min:
            x = x_left
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
        docks: List[QtWidgets.QDockWidget] = list(self._section_docks.values())
        if hasattr(self, "art_dock") and isinstance(self.art_dock, QtWidgets.QDockWidget):
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
            self._hide_dock_widgets(self.getPostDockWidgets(), remove=True)
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
        if isinstance(fw, (QtWidgets.QLineEdit, QtWidgets.QPlainTextEdit, QtWidgets.QTextEdit)):
            return True
        if isinstance(fw, (QtWidgets.QSpinBox, QtWidgets.QDoubleSpinBox, QtWidgets.QAbstractSpinBox)):
            return True
        if isinstance(fw, QtWidgets.QComboBox) and fw.isEditable():
            return True
        return False

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
            if require_non_text_focus and self._is_text_entry_focused():
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
        self._bind_shortcut("Escape", self._close_focused_popup, require_non_text_focus=True)

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
                right_i = _dock_area_to_int(QtCore.Qt.DockWidgetArea.RightDockWidgetArea, 2)
                out[key] = {
                    "visible": _to_bool(visible, False) if visible is not None else False,
                    "floating": _to_bool(floating, True) if floating is not None else True,
                    "area": _dock_area_to_int(area, right_i) if area is not None else right_i,
                    "geometry": self._qbytearray_to_b64(geom),
                }
            return out

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
                    "area": int(art_area) if art_area is not None else _dock_area_to_int(QtCore.Qt.DockWidgetArea.RightDockWidgetArea, 2),
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
        panel_layout = ui_state.get("panel_layout")
        if isinstance(panel_layout, dict):
            self._load_panel_config_payload_into_settings(panel_layout)
            self._apply_panel_layout_from_settings()
            self._save_panel_config_json()
        self._save_settings()

    def _sync_section_button_states_from_docks(self) -> None:
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
                right_i = _dock_area_to_int(QtCore.Qt.DockWidgetArea.RightDockWidgetArea, 2)
                area_val = _dock_area_to_int(cached.get("area", self.dockWidgetArea(dock)), right_i)
                geom = cached.get("geometry", dock.saveGeometry())
                self.settings.setValue(f"{base}/visible", visible)
                self.settings.setValue(f"{base}/floating", floating)
                self.settings.setValue(f"{base}/area", area_val)
                self.settings.setValue(f"{base}/geometry", geom)
            except Exception:
                continue

        try:
            base = "pre_artifact_dock_state"
            cached = self._pre_artifact_state_before_tab_switch if self._pre_popups_hidden_by_tab_switch else {}
            visible = bool(cached.get("visible", self.art_dock.isVisible()))
            floating = bool(cached.get("floating", self.art_dock.isFloating()))
            right_i = _dock_area_to_int(QtCore.Qt.DockWidgetArea.RightDockWidgetArea, 2)
            area_val = _dock_area_to_int(cached.get("area", self.dockWidgetArea(self.art_dock)), right_i)
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
        self._is_restoring_panel_layout = True
        for key, dock in self._section_docks.items():
            base = f"pre_section_docks/{key}"
            try:
                visible = _to_bool(self.settings.value(f"{base}/visible", False), False)
                floating = _to_bool(self.settings.value(f"{base}/floating", True), True)
                area_val = self.settings.value(
                    f"{base}/area",
                    _dock_area_to_int(QtCore.Qt.DockWidgetArea.RightDockWidgetArea, 2),
                )
                area = self._dock_area_from_settings(area_val, QtCore.Qt.DockWidgetArea.RightDockWidgetArea)
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

        try:
            base = "pre_artifact_dock_state"
            visible = _to_bool(self.settings.value(f"{base}/visible", False), False)
            floating = _to_bool(self.settings.value(f"{base}/floating", False), False)
            area_val = self.settings.value(
                f"{base}/area",
                _dock_area_to_int(QtCore.Qt.DockWidgetArea.RightDockWidgetArea, 2),
            )
            area = self._dock_area_from_settings(area_val, QtCore.Qt.DockWidgetArea.RightDockWidgetArea)
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
            if self.settings.contains(_PRE_DOCK_STATE_KEY):
                return True
            if self.settings.contains("pre_artifact_dock_state/visible"):
                return True
            for key in self._section_docks.keys():
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
                    if len(vals) >= 3:
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
                    if len(vals) >= 3:
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

    def _open_files_dialog(self) -> None:
        start_dir = self.file_panel.current_dir_hint() or self.settings.value("last_open_dir", "", type=str) or os.getcwd()
        paths, _ = QtWidgets.QFileDialog.getOpenFileNames(
            self,
            "Open files",
            start_dir,
            "Doric files (*.doric *.h5 *.hdf5);;All files (*.*)",
        )
        if not paths:
            return

        self.settings.setValue("last_open_dir", os.path.dirname(paths[0]))
        self._add_files(paths)

    def _open_folder_dialog(self) -> None:
        start_dir = self.file_panel.current_dir_hint() or self.settings.value("last_open_dir", "", type=str) or os.getcwd()
        folder = QtWidgets.QFileDialog.getExistingDirectory(self, "Add folder with .doric", start_dir)
        if not folder:
            return
        self.settings.setValue("last_open_dir", folder)

        paths: List[str] = []
        for fn in os.listdir(folder):
            if fn.lower().endswith((".doric", ".h5", ".hdf5")):
                paths.append(os.path.join(folder, fn))
        paths.sort()
        self._add_files(paths)

    def _add_files(self, paths: List[str]) -> None:
        for p in paths:
            if p in self._loaded_files:
                continue
            try:
                doric = self.processor.load_file(p)
                self._loaded_files[p] = doric
                self.file_panel.add_file(p)
                self._show_status_message(f"Loaded: {os.path.basename(p)}", 5000)
            except Exception as e:
                QtWidgets.QMessageBox.critical(self, "Load error", f"Failed to load:\n{p}\n\n{e}")

        # set current selection -> triggers preview
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
            area = _dock_area_to_int(host.dockWidgetArea(dock), _dock_area_to_int(QtCore.Qt.DockWidgetArea.RightDockWidgetArea, 2))
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
            "area": _dock_area_to_int(host.dockWidgetArea(self.art_dock), _dock_area_to_int(QtCore.Qt.DockWidgetArea.RightDockWidgetArea, 2)),
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
        self._hide_dock_widgets(self.getPostDockWidgets(), remove=True)

    def _store_pre_main_dock_snapshot(self) -> None:
        """Persist the current preprocessing dock arrangement."""
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
        if not self._pre_popups_hidden_by_tab_switch:
            return
        right_i = _dock_area_to_int(QtCore.Qt.DockWidgetArea.RightDockWidgetArea, 2)
        try:
            for key in self._section_docks.keys():
                state = self._pre_section_state_before_tab_switch.get(key, {})
                base = f"pre_section_docks/{key}"
                self.settings.setValue(f"{base}/visible", bool(state.get("visible", False)))
                self.settings.setValue(f"{base}/floating", bool(state.get("floating", True)))
                self.settings.setValue(f"{base}/area", _dock_area_to_int(state.get("area", right_i), right_i))
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
            self.settings.setValue(f"{base}/area", _dock_area_to_int(art_state.get("area", right_i), right_i))
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

        right = QtCore.Qt.DockWidgetArea.RightDockWidgetArea
        self._suspend_panel_layout_persistence = True
        try:
            self._enforce_postprocessing_popups_hidden()
            # Default dock layout: Artifacts on top, tabbed Filtering/Baseline/Output,
            # QC below, Export at bottom. Config remains hidden.
            artifacts = self._section_docks.get("artifacts")
            filtering = self._section_docks.get("filtering")
            baseline = self._section_docks.get("baseline")
            output = self._section_docks.get("output")
            qc = self._section_docks.get("qc")
            export = self._section_docks.get("export")
            config = self._section_docks.get("config")

            for dock in (artifacts, filtering, baseline, output, qc, export, config):
                if dock is None:
                    continue
                dock.setFloating(False)
                dock.show()

            if artifacts is not None:
                self.addDockWidget(right, artifacts)
            if filtering is not None:
                self.addDockWidget(right, filtering)
                if artifacts is not None:
                    self.splitDockWidget(artifacts, filtering, QtCore.Qt.Orientation.Vertical)

            if baseline is not None and filtering is not None:
                self.addDockWidget(right, baseline)
                self.tabifyDockWidget(filtering, baseline)
            if output is not None and filtering is not None:
                self.addDockWidget(right, output)
                self.tabifyDockWidget(filtering, output)
                filtering.raise_()

            if qc is not None:
                self.addDockWidget(right, qc)
                if filtering is not None:
                    self.splitDockWidget(filtering, qc, QtCore.Qt.Orientation.Vertical)
                elif artifacts is not None:
                    self.splitDockWidget(artifacts, qc, QtCore.Qt.Orientation.Vertical)
            if export is not None:
                self.addDockWidget(right, export)
                if qc is not None:
                    self.splitDockWidget(qc, export, QtCore.Qt.Orientation.Vertical)
                elif filtering is not None:
                    self.splitDockWidget(filtering, export, QtCore.Qt.Orientation.Vertical)

            if config is not None:
                config.hide()

            self._sync_section_button_states_from_docks()
        finally:
            self._suspend_panel_layout_persistence = False

        self._save_panel_layout_state()
        self._store_pre_main_dock_snapshot()

    def _apply_pre_fixed_layout(self) -> None:
        """
        Force a deterministic preprocessing dock layout matching the project default:
        - Right column top: Artifacts list tab group (Artifacts list / Filtering / Artifacts)
        - Right column middle: Baseline tab group (Baseline / Output / QC)
        - Right column bottom: Export
        - Bottom strip: Configuration
        """
        if not self._section_docks:
            return

        host = self
        right = QtCore.Qt.DockWidgetArea.RightDockWidgetArea
        bottom = QtCore.Qt.DockWidgetArea.BottomDockWidgetArea

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
            ordered_right: List[QtWidgets.QDockWidget] = []
            if isinstance(self.art_dock, QtWidgets.QDockWidget):
                ordered_right.append(self.art_dock)
            for dock in (artifacts, filtering, baseline, output, qc, export):
                if isinstance(dock, QtWidgets.QDockWidget):
                    ordered_right.append(dock)

            for dock in ordered_right:
                dock.blockSignals(True)
                try:
                    dock.setFloating(False)
                    host.addDockWidget(right, dock)
                    dock.show()
                finally:
                    dock.blockSignals(False)

            if isinstance(config, QtWidgets.QDockWidget):
                config.blockSignals(True)
                try:
                    config.setFloating(False)
                    host.addDockWidget(bottom, config)
                    config.show()
                finally:
                    config.blockSignals(False)

            # Vertical stack in right area: artifacts list (top) -> baseline group (middle) -> export (bottom).
            if baseline is not None:
                host.splitDockWidget(self.art_dock, baseline, QtCore.Qt.Orientation.Vertical)
            if export is not None:
                if baseline is not None:
                    host.splitDockWidget(baseline, export, QtCore.Qt.Orientation.Vertical)
                else:
                    host.splitDockWidget(self.art_dock, export, QtCore.Qt.Orientation.Vertical)

            # Top tab group: Artifacts list + Filtering + Artifacts.
            if artifacts is not None:
                host.tabifyDockWidget(self.art_dock, artifacts)
            if filtering is not None:
                host.tabifyDockWidget(self.art_dock, filtering)

            # Middle tab group: Baseline + Output + QC.
            if baseline is not None and output is not None:
                host.tabifyDockWidget(baseline, output)
            if baseline is not None and qc is not None:
                host.tabifyDockWidget(baseline, qc)

            # Keep active tabs consistent with the default arrangement.
            try:
                self.art_dock.raise_()
            except Exception:
                pass
            if baseline is not None:
                baseline.raise_()
            if export is not None:
                export.raise_()
            if config is not None:
                config.raise_()

            # Approximate default height proportions for right-column groups.
            try:
                vdocks: List[QtWidgets.QDockWidget] = []
                sizes: List[int] = []
                if isinstance(self.art_dock, QtWidgets.QDockWidget):
                    vdocks.append(self.art_dock)
                    sizes.append(520)
                if baseline is not None:
                    vdocks.append(baseline)
                    sizes.append(260)
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
                        state.get("area", _dock_area_to_int(QtCore.Qt.DockWidgetArea.RightDockWidgetArea, 2)),
                        QtCore.Qt.DockWidgetArea.RightDockWidgetArea,
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
                    art_state.get("area", _dock_area_to_int(QtCore.Qt.DockWidgetArea.RightDockWidgetArea, 2)),
                    QtCore.Qt.DockWidgetArea.RightDockWidgetArea,
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

    def _on_main_tab_changed(self, index: int) -> None:
        if self._handling_main_tab_change:
            return
        self._handling_main_tab_change = True
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
                        try:
                            self.post_tab.ensure_section_popups_initialized()
                            if hasattr(self.post_tab, "apply_fixed_default_layout"):
                                self.post_tab.apply_fixed_default_layout()
                                # Re-apply after queued dock events from tab switch.
                                QtCore.QTimer.singleShot(0, self.post_tab.apply_fixed_default_layout)
                        except Exception:
                            _LOG.exception("Failed to apply fixed post layout on tab switch")
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
            self._handling_main_tab_change = False

    def _on_artifact_overlay_toggled(self, visible: bool) -> None:
        self.plots.set_artifact_overlay_visible(bool(visible))
        self._save_settings()

    def _on_artifact_thresholds_toggled(self, visible: bool) -> None:
        self.plots.set_artifact_thresholds_visible(bool(visible))
        self._save_settings()

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

        self._update_raw_plot()
        self._trigger_preview()

        # update post tab selection context
        self.post_tab.set_current_source_label(os.path.basename(path), self._current_channel or "")
        self._update_plot_status()

    def _on_channel_changed(self, ch: str) -> None:
        self._current_channel = ch
        self._update_raw_plot()
        self._trigger_preview()
        self.post_tab.set_current_source_label(os.path.basename(self._current_path or ""), self._current_channel or "")
        self._update_plot_status()

    def _on_trigger_changed(self, trig: str) -> None:
        self._current_trigger = trig if trig else None
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

        raw_sig = _mask_arr(processed.raw_signal)
        raw_ref = _mask_arr(processed.raw_reference)
        processed.raw_signal = raw_sig if raw_sig is not None else processed.raw_signal
        processed.raw_reference = raw_ref if raw_ref is not None else processed.raw_reference
        processed.sig_f = _mask_arr(processed.sig_f)
        processed.ref_f = _mask_arr(processed.ref_f)
        processed.baseline_sig = _mask_arr(processed.baseline_sig)
        processed.baseline_ref = _mask_arr(processed.baseline_ref)
        processed.output = _mask_arr(processed.output)
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
        )

    def _update_raw_plot(self) -> None:
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

        self.plots.set_title(os.path.basename(self._current_path))
        self.plots.show_raw(
            time=trial.time,
            raw465=raw465,
            raw405=raw405,
            trig_time=trial.trigger_time,
            trig=trial.trigger,
            trig_label=self._current_trigger or "",
            manual_regions=manual,
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
            self._trigger_preview()
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
            self._update_raw_plot()
        except Exception:
            pass
        self._trigger_preview()

    def _trigger_preview(self) -> None:
        # persist params quickly
        self._save_settings()
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
        self.plots.update_plots(processed)
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
        self._trigger_preview()

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
        regs = self._manual_regions_by_key.get(key, [])
        regs.append((min(t0, t1), max(t0, t1)))
        self._manual_regions_by_key[key] = regs
        start_s, end_s = self._time_window_bounds()
        self.artifact_panel.set_regions(self._clip_regions_to_window(regs, start_s, end_s))
        self._trigger_preview()

    def _clear_manual_regions_current(self) -> None:
        key = self._current_key()
        if not key:
            return
        self._manual_regions_by_key[key] = []
        self._manual_exclude_by_key[key] = []
        self.artifact_panel.set_regions([])
        self._trigger_preview()

    def _request_box_select(self, callback: Callable[[float, float], None]) -> None:
        self._box_select_callback = callback
        self.plots.btn_box_select.setChecked(True)
        self._show_status_message("Box select: drag on the raw plot to set the time window; right-click to cancel.")

    def _cancel_box_select_request(self) -> None:
        if not self._box_select_callback:
            return
        self._box_select_callback = None
        self.plots.btn_box_select.setChecked(False)

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
        self._trigger_preview()

    def _toggle_artifacts_panel(self) -> None:
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

    def _export_selected_or_all(self) -> None:
        selected = self._selected_paths()
        if not selected:
            selected = self.file_panel.all_paths()
        if not selected:
            return

        origin_dir = self._export_origin_dir(selected)
        start_dir = self._export_start_dir(selected)
        out_dir = QtWidgets.QFileDialog.getExistingDirectory(self, "Select export folder", start_dir)
        if not out_dir:
            return
        self._remember_export_dir(out_dir, origin_dir)

        params = self.param_panel.get_params()

        # Process/export each selected file, for the currently selected channel.
        n_total = 0
        for path in selected:
            doric = self._loaded_files.get(path)
            if not doric:
                continue
            ch = self._current_channel if (self._current_channel in doric.channels) else (doric.channels[0] if doric.channels else None)
            if not ch:
                continue
            key = (path, ch)
            trial = doric.make_trial(ch, trigger_name=self._current_trigger)  # export uses current trigger selection
            trial = self._apply_time_window(trial)
            cutouts = self._cutout_regions_by_key.get(key, [])
            trial = self._apply_cutouts(trial, cutouts)
            start_s, end_s = self._time_window_bounds()
            manual = self._clip_regions_to_window(self._manual_regions_by_key.get(key, []), start_s, end_s)
            manual_exclude = self._clip_regions_to_window(self._manual_exclude_by_key.get(key, []), start_s, end_s)
            meta = self._metadata_by_key.get(key, {})
            sections = self._sections_by_key.get(key, [])

            def _export_one(proc: ProcessedTrial, suffix: str = "") -> None:
                nonlocal n_total
                proc = self._apply_cutouts_to_processed(proc, cutouts)
                stem = safe_stem_from_metadata(path, ch, meta)
                if suffix:
                    stem = f"{stem}_{suffix}"
                csv_path = os.path.join(out_dir, f"{stem}.csv")
                h5_path = os.path.join(out_dir, f"{stem}.h5")
                export_processed_csv(csv_path, proc, metadata=meta)
                export_processed_h5(h5_path, proc, metadata=meta)
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
                        processed = self.processor.process_trial(
                            trial=sec_trial,
                            params=sec_params,
                            manual_regions_sec=manual,
                            manual_exclude_regions_sec=manual_exclude,
                            preview_mode=False,
                        )
                        _export_one(processed, suffix=f"sec{i}_{s0:.2f}_{s1:.2f}")
                else:
                    processed = self.processor.process_trial(
                        trial=trial,
                        params=params,
                        manual_regions_sec=manual,
                        manual_exclude_regions_sec=manual_exclude,
                        preview_mode=False,
                    )
                    _export_one(processed)
            except Exception as e:
                QtWidgets.QMessageBox.warning(self, "Export error", f"Failed export:\n{path} [{ch}]\n\n{e}")

        self._show_status_message(f"Export complete: {n_total} recording(s) written to {out_dir}")

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

        for p in paths:
            if not p:
                continue
            ext = os.path.splitext(p)[1].lower()
            if ext == ".doric":
                doric_paths.append(p)
                continue
            if ext == ".csv":
                trial = self._load_processed_csv(p)
                if trial is not None:
                    processed.append(trial)
                continue
            if ext in (".h5", ".hdf5"):
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
    pg.setConfigOptions(antialias=True)
    app = QtWidgets.QApplication([])
    w = MainWindow()
    w.show()
    app.exec()


if __name__ == "__main__":
    main()
