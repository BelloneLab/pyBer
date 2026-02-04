# main.py
"""
Fiber Photometry Processor (Doric .doric) — PySide6 + pyqtgraph

Run:
    python main.py

Dependencies:
    pip install PySide6 pyqtgraph h5py numpy scipy scikit-learn pybaselines
"""

from __future__ import annotations

import os
import json
from typing import Callable, Dict, List, Optional, Tuple

from PySide6 import QtCore, QtWidgets
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
from gui_preprocessing import FileQueuePanel, ParameterPanel, PlotDashboard, MetadataDialog, ArtifactPanel, AdvancedOptionsDialog
from gui_postprocessing import PostProcessingPanel
from styles import APP_QSS
import numpy as np

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

        self._build_ui()
        self._restore_settings()

    # ---------------- UI ----------------

    def _build_ui(self) -> None:
        self.setStyleSheet(APP_QSS)

        self.tabs = QtWidgets.QTabWidget()
        self.setCentralWidget(self.tabs)

        # Preprocessing tab
        pre = QtWidgets.QWidget()
        self.tabs.addTab(pre, "Preprocessing")

        self.file_panel = FileQueuePanel()
        self.param_panel = ParameterPanel()
        self.plots = PlotDashboard()
        self.artifact_panel = ArtifactPanel()

        # Right artifact panel dock (hidden by default)
        self.art_dock = QtWidgets.QDockWidget("Artifacts", self)
        self.art_dock.setWidget(self.artifact_panel)
        self.art_dock.setAllowedAreas(QtCore.Qt.DockWidgetArea.RightDockWidgetArea)
        self.art_dock.setVisible(False)
        self.addDockWidget(QtCore.Qt.DockWidgetArea.RightDockWidgetArea, self.art_dock)

        # Left column: file + selection + buttons + parameters, all scrollable
        left_container = QtWidgets.QWidget()
        left_v = QtWidgets.QVBoxLayout(left_container)
        left_v.setContentsMargins(0, 0, 0, 0)
        left_v.setSpacing(10)
        left_v.addWidget(self.file_panel)
        left_v.addWidget(self.param_panel, stretch=1)

        left_scroll = QtWidgets.QScrollArea()
        left_scroll.setWidgetResizable(True)
        left_scroll.setMinimumWidth(160)
        left_scroll.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        left_scroll.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        left_scroll.setWidget(left_container)

        # Make panels resizable: left | plots (right dock already resizable by Qt)
        splitter = QtWidgets.QSplitter(QtCore.Qt.Orientation.Horizontal)
        splitter.addWidget(left_scroll)
        splitter.addWidget(self.plots)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        # Set initial sizes: smaller left panel, larger visualization
        splitter.setSizes([160, 1540])  # Narrower left panel to favor plots
        # Connect splitter size changes to save settings
        splitter.splitterMoved.connect(self._save_splitter_sizes)

        pre_layout = QtWidgets.QVBoxLayout(pre)
        pre_layout.setContentsMargins(10, 10, 10, 10)
        pre_layout.addWidget(splitter)

        # Postprocessing tab
        self.post_tab = PostProcessingPanel()
        self.tabs.addTab(self.post_tab, "Post Processing")

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

        # Parameters -> debounce preview
        self.param_panel.paramsChanged.connect(self._on_params_changed)

        # Plot sync
        self.plots.xRangeChanged.connect(self.plots.set_xrange_all)

        # Manual artifacts
        self.plots.manualRegionFromSelectorRequested.connect(self._add_manual_region_from_selector)
        self.plots.manualRegionFromDragRequested.connect(self._add_manual_region_from_drag)
        self.plots.clearManualRegionsRequested.connect(self._clear_manual_regions_current)
        self.plots.showArtifactsRequested.connect(self._toggle_artifacts_panel)
        self.plots.boxSelectionCleared.connect(self._cancel_box_select_request)

        self.artifact_panel.regionsChanged.connect(self._artifact_regions_changed)
        self.artifact_panel.selectionChanged.connect(self.plots.highlight_artifact_regions)

        # Postprocessing needs access to "current processed"
        self.post_tab.requestCurrentProcessed.connect(self._post_get_current_processed)
        self.post_tab.requestDioList.connect(self._post_get_current_dio_list)
        self.post_tab.requestDioData.connect(self._post_get_dio_data_for_path)

        self.setAcceptDrops(True)

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
        except Exception:
            pass

        # restore splitter sizes
        try:
            splitter_sizes = self.settings.value("splitter_sizes", None)
            if splitter_sizes and hasattr(splitter_sizes, '__len__') and len(splitter_sizes) >= 2:
                # Find the splitter in the preprocessing tab
                pre_tab = self.tabs.widget(0)  # Assuming preprocessing is tab 0
                if pre_tab:
                    splitter = pre_tab.findChild(QtWidgets.QSplitter)
                    if splitter:
                        splitter.setSizes([int(splitter_sizes[0]), int(splitter_sizes[1])])
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

    def _save_splitter_sizes(self) -> None:
        """Save the current splitter sizes to settings."""
        try:
            # Find the splitter in the preprocessing tab
            pre_tab = self.tabs.widget(0)  # Assuming preprocessing is tab 0
            if pre_tab:
                splitter = pre_tab.findChild(QtWidgets.QSplitter)
                if splitter:
                    sizes = splitter.sizes()
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
                self.plots.set_log(f"Loaded: {p}")
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

    def _on_file_selection_changed(self) -> None:
        sel = self._selected_paths()
        if not sel:
            return

        # preview shows first selected
        path = sel[0]
        self._current_path = path

        doric = self._loaded_files.get(path)
        if not doric:
            return

        self.file_panel.set_available_channels(doric.channels)
        self.file_panel.set_available_triggers(sorted(doric.digital_by_name.keys()))

        # keep channel if still valid
        if self._current_channel in doric.channels:
            self.file_panel.set_channel(self._current_channel)
        else:
            self._current_channel = doric.channels[0] if doric.channels else None
            if self._current_channel:
                self.file_panel.set_channel(self._current_channel)

        # keep trigger if still valid
        if self._current_trigger and self._current_trigger not in doric.digital_by_name:
            self._current_trigger = None
            self.file_panel.set_trigger("")

        self._update_raw_plot()
        self._trigger_preview()

        # update post tab selection context
        self.post_tab.set_current_source_label(os.path.basename(path), self._current_channel or "")

    def _on_channel_changed(self, ch: str) -> None:
        self._current_channel = ch
        self._update_raw_plot()
        self._trigger_preview()
        self.post_tab.set_current_source_label(os.path.basename(self._current_path or ""), self._current_channel or "")

    def _on_trigger_changed(self, trig: str) -> None:
        self._current_trigger = trig if trig else None
        self._update_raw_plot()

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

        self.plots.set_log(
            f"Processing preview… (fs={trial.sampling_rate:.2f} Hz → target {params.target_fs_hz:.1f} Hz, "
            f"baseline={params.baseline_method})"
        )

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

        # Preserve current x/y ranges before updating plots
        current_xrange = None
        try:
            # Get current x range from the first plot
            view_box = self.plots.plot_raw.getViewBox()
            if view_box:
                x_range = view_box.viewRange()[0]  # x axis range
                current_xrange = (x_range[0], x_range[1])
        except Exception:
            pass

        # Update plots (decimated signals)
        self.plots.update_plots(processed)

        # Restore x range if it was preserved
        if current_xrange is not None:
            try:
                self.plots.set_xrange_all(current_xrange[0], current_xrange[1])
            except Exception:
                pass

        self.plots.set_log(
            f"Preview updated: {processed.output_label} | fs={processed.fs_actual:.2f}→{processed.fs_used:.2f} Hz "
            f"(target {processed.fs_target:.2f}) | n={processed.time.size} | {elapsed_s*1000:.0f} ms"
        )

        # Inform post tab that current processed changed
        self.post_tab.notify_preprocessing_updated(processed)

    @QtCore.Slot(str, int)
    def _on_preview_failed(self, err: str, job_id: int) -> None:
        if job_id != self._latest_job_id:
            return
        self.plots.set_log(f"Preview error: {err}")

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
        self.plots.set_log("Box select: drag on the raw plot to set the time window; right-click to cancel.")

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
            return
        self.artifact_panel.show()
        self.art_dock.setVisible(True)

    # ---------------- Metadata ----------------

    def _edit_metadata_for_current(self) -> None:
        if not self._current_path:
            return
        doric = self._loaded_files.get(self._current_path)
        if not doric:
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

        self.plots.set_log(f"Export complete: {n_total} recording(s) written to {out_dir}")

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
        # DIO list for current/selected files: intersection or union; easiest = union
        paths = self._selected_paths()
        if not paths:
            paths = [self._current_path] if self._current_path else []

        dio = set()
        for p in paths:
            f = self._loaded_files.get(p)
            if f:
                dio |= set(f.digital_by_name.keys())
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

        # digital_time may be a numpy array
        if getattr(f, "digital_time", None) is None:
            return

        t_dio = np.asarray(f.digital_time)
        if t_dio.size == 0:
            return

        digital_map = getattr(f, "digital_by_name", None)
        if not isinstance(digital_map, dict) or dio_name not in digital_map:
            return

        y_dio = np.asarray(digital_map[dio_name])
        if y_dio.size == 0:
            return

        # Ensure same length
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
        output_idx = _find_col(["dff", "z-score", "zscore", "z score", "output"])
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

    def closeEvent(self, event):
        self._save_settings()
        super().closeEvent(event)


def main() -> None:
    pg.setConfigOptions(antialias=True)
    app = QtWidgets.QApplication([])
    w = MainWindow()
    w.show()
    app.exec()


if __name__ == "__main__":
    main()
