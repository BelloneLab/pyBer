"""Generate the README screenshots from a real Doric recording.

The script intentionally drives the same Qt widgets and analysis routines used
by pyBer.  It does not mock traces or paint a separate marketing image.

Example
-------
python scripts/generate_readme_screenshots.py \
    --doric C:\\data\\trial_0010.doric \
    --behavior C:\\data\\trial_0010_with_time.csv
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys
import tempfile
import time


os.environ.setdefault("QT_QPA_PLATFORM", "windows" if os.name == "nt" else "offscreen")
os.environ.setdefault("QT_SCALE_FACTOR", "1")
os.environ.setdefault("PYBER_SMOKE_TEST", "1")
os.environ.setdefault("PYTHONNOUSERSITE", "1")

ROOT = Path(__file__).resolve().parents[1]
PACKAGE_DIR = ROOT / "pyBer"
if str(PACKAGE_DIR) not in sys.path:
    sys.path.insert(0, str(PACKAGE_DIR))

from PySide6 import QtCore, QtWidgets  # noqa: E402

from analysis_core import PhotometryProcessor, ProcessingParams, recommend_preprocessing_settings  # noqa: E402
from main import MainWindow, QcDialog  # noqa: E402
from styles import apply_app_palette  # noqa: E402


DEFAULT_DORIC = Path(r"C:\Analysis\fiberPhotometry\pyber_test_data\trial_0010.doric")
DEFAULT_BEHAVIOR = Path(r"C:\Analysis\fiberPhotometry\pyber_test_data\trial_0010_with_time.csv")
DEFAULT_OUTPUT = ROOT / "assets" / "screenshots"


def _pump(app: QtWidgets.QApplication, milliseconds: int = 350) -> None:
    deadline = time.monotonic() + milliseconds / 1000.0
    while time.monotonic() < deadline:
        app.processEvents(QtCore.QEventLoop.ProcessEventsFlag.AllEvents, 50)
        time.sleep(0.01)


def _save(app: QtWidgets.QApplication, widget: QtWidgets.QWidget, path: Path) -> None:
    _pump(app, 650)
    pixmap = widget.grab()
    if pixmap.isNull() or not pixmap.save(str(path), "PNG"):
        raise RuntimeError(f"Could not save screenshot: {path}")
    print(f"saved {path.relative_to(ROOT)} ({pixmap.width()}x{pixmap.height()})")


def _disable_toasts(window: MainWindow) -> None:
    """Remove ephemeral notifications that would obscure a reproducible capture."""
    toaster = getattr(window, "_toaster", None)
    if toaster is not None:
        for toast in list(getattr(toaster, "_toasts", [])):
            toast.close()
        getattr(toaster, "_toasts", []).clear()
    window._toaster = None


def _set_combo(combo: QtWidgets.QComboBox, text: str) -> None:
    index = combo.findText(text)
    if index < 0:
        raise RuntimeError(f"Combo value not found: {text!r}")
    combo.setCurrentIndex(index)


def _open_preprocessing_drawer(window: MainWindow, key: str) -> None:
    window._set_section_button_checked(key, True)
    window._toggle_section_popup(key, True)


def _open_postprocessing_drawer(window: MainWindow, key: str) -> None:
    post = window.post_tab
    post._set_section_button_checked(key, True)
    post._toggle_section_popup(key, True)


def _prepare_preprocessing(window: MainWindow, doric_path: Path):
    processor = PhotometryProcessor()
    loaded = processor.load_file(str(doric_path))
    channel = loaded.channels[0]
    trial = loaded.make_trial(channel)
    recommendation = recommend_preprocessing_settings(trial, ProcessingParams())
    processed = processor.process_trial(trial, recommendation.params, preview_mode=False)

    path = str(doric_path)
    window._loaded_files[path] = loaded
    window.file_panel.add_file(path)
    window._on_file_selection_changed()
    window._preview_timer.stop()

    window._current_path = path
    window._current_channel = channel
    window._current_trigger = None
    window.param_panel.set_params(recommendation.params)
    window.param_panel.set_recommendation(recommendation)
    window._preview_timer.stop()

    key = (path, channel)
    window._last_processed[key] = processed
    auto_regions = list(processed.artifact_regions_auto_sec or [])
    window._auto_regions_by_key[key] = auto_regions
    window.artifact_panel.set_auto_regions(
        auto_regions,
        checked_regions=auto_regions,
        sources=list(processed.artifact_regions_auto_source or []),
        core_regions=list(processed.artifact_regions_auto_core_sec or []),
    )
    window.plots.update_plots(processed, preserve_view=False)
    window.plots.set_title(doric_path.name)
    window._auto_range_for_processed(processed)
    window.param_panel.set_fs_info(processed.fs_actual, processed.fs_target, processed.fs_used)
    window._update_plot_status(processed.fs_actual, processed.fs_target)
    return loaded, trial, processed, recommendation


def _prepare_postprocessing(window: MainWindow, processed, behavior_path: Path) -> None:
    post = window.post_tab
    post.receive_current_processed([processed])
    post._load_behavior_paths([str(behavior_path)], replace=True)
    post._refresh_behavior_list()

    _set_combo(post.combo_align, "Behavior (CSV/XLSX)")
    _set_combo(post.combo_behavior_name, "social_contacts")
    _set_combo(post.combo_behavior_align, "Align to onset")
    post.spin_pre.setValue(3.0)
    post.spin_post.setValue(6.0)
    post.spin_b0.setValue(-2.0)
    post.spin_b1.setValue(0.0)
    post.spin_resample.setValue(50.0)
    post.spin_smooth.setValue(0.08)
    post._compute_psth()
    if getattr(post, "_last_mat", None) is None:
        raise RuntimeError("The behavior file did not produce a PSTH matrix.")

    window.tabs.setCurrentWidget(post)
    _set_combo(post.combo_view_layout, "Standard")
    post.plot_trace.autoRange()
    post._force_hide_post_drawer_initially()


def generate(doric_path: Path, behavior_path: Path, output_dir: Path) -> None:
    for path in (doric_path, behavior_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    output_dir.mkdir(parents=True, exist_ok=True)

    settings_dir = tempfile.TemporaryDirectory(prefix="pyber-readme-")
    QtCore.QSettings.setDefaultFormat(QtCore.QSettings.Format.IniFormat)
    QtCore.QSettings.setPath(
        QtCore.QSettings.Format.IniFormat,
        QtCore.QSettings.Scope.UserScope,
        settings_dir.name,
    )

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    app.setApplicationName("pyBer README capture")
    apply_app_palette(app, "dark")

    window = MainWindow()
    window.resize(1920, 1080)
    window.show()
    _pump(app, 500)
    _disable_toasts(window)

    _loaded, trial, processed, _recommendation = _prepare_preprocessing(window, doric_path)

    # Overview: the complete raw-to-output stack plus the sensor-aware output advice.
    window.tabs.setCurrentIndex(0)
    _open_preprocessing_drawer(window, "output")
    _save(app, window, output_dir / "preprocessing_overview.png")

    # Artifact panel: automatic regions remain connected to the live trace selection.
    _open_preprocessing_drawer(window, "artifacts")
    _save(app, window, output_dir / "artifact_review.png")

    # Strict QC uses the exact same recording and preprocessing state.
    qc = window._compute_qc(trial)
    if qc is None:
        raise RuntimeError("QC could not be computed for the demo recording.")
    qc_dialog = QcDialog(qc, window)
    qc_dialog.resize(1800, 1000)
    qc_dialog.show()
    _save(app, qc_dialog, output_dir / "quality_control.png")
    qc_dialog.close()

    _prepare_postprocessing(window, processed, behavior_path)
    _save(app, window, output_dir / "behavior_psth.png")

    # Signal-event analysis with a noise-adaptive threshold derived from the trace.
    post = window.post_tab
    _set_combo(post.combo_view_layout, "All")
    post.cb_peak_auto_mad.setChecked(True)
    post.spin_peak_mad_multiplier.setValue(5.0)
    post.spin_peak_distance.setValue(0.75)
    post.spin_peak_smooth.setValue(0.05)
    post._detect_signal_events()
    post.plot_trace.autoRange()
    _open_postprocessing_drawer(window, "signal")
    _save(app, window, output_dir / "signal_event_analysis.png")

    # Behavior summaries are computed from the supplied binary annotation table.
    _set_combo(post.combo_behavior_analysis, "social_contacts")
    post._compute_behavior_analysis()
    _open_postprocessing_drawer(window, "behavior")
    _save(app, window, output_dir / "behavior_analysis.png")

    # Show the model workbench with the current processed and behavioral context loaded.
    post._sync_temporal_modeling_context()
    temporal = post.section_temporal
    temporal.list_predictors.clear()
    for predictor_key in ("behavior_event::social_contacts", "behavior_state::social_contacts"):
        if predictor_key in temporal._predictor_catalog:
            temporal._add_predictor_item(predictor_key)
    temporal._remember_current_predictors_for_file()
    _set_combo(temporal.combo_model_type, "Continuous GLM")
    _set_combo(temporal.combo_basis, "Raised cosine")
    _set_combo(temporal.combo_reg, "Ridge")
    temporal.spin_n_basis.setValue(6)
    temporal.spin_glm_bootstrap.setValue(0)
    scope_index = temporal.combo_fit_scope.findData("active")
    temporal.combo_fit_scope.setCurrentIndex(scope_index)
    temporal._fit_mode = "active"
    temporal._on_fit_clicked()
    temporal_dialog = QtWidgets.QDialog(window)
    temporal_dialog.setWindowTitle("pyBer - Temporal Modeling")
    temporal_layout = QtWidgets.QVBoxLayout(temporal_dialog)
    temporal_layout.setContentsMargins(0, 0, 0, 0)
    temporal_layout.addWidget(temporal)
    temporal.setFixedSize(1580, 900)
    temporal_dialog.setFixedSize(1600, 920)
    temporal_dialog.show()
    temporal_layout.activate()
    temporal.tabs_workspace.setCurrentIndex(0)
    _save(app, temporal_dialog, output_dir / "temporal_modeling.png")
    kernels_index = next(
        (i for i in range(temporal.tabs_workspace.count()) if temporal.tabs_workspace.tabText(i) == "Kernels"),
        -1,
    )
    if kernels_index >= 0:
        temporal.tabs_workspace.setCurrentIndex(kernels_index)
        _save(app, temporal_dialog, output_dir / "temporal_model_kernels.png")
    temporal_dialog.close()

    window.hide()
    window.deleteLater()
    app.processEvents()
    QtCore.QThreadPool.globalInstance().clear()
    QtCore.QThreadPool.globalInstance().waitForDone(5000)
    app.quit()
    settings_dir.cleanup()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--doric", type=Path, default=DEFAULT_DORIC)
    parser.add_argument("--behavior", type=Path, default=DEFAULT_BEHAVIOR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    generate(args.doric.resolve(), args.behavior.resolve(), args.output_dir.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
