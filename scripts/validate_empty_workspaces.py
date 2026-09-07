"""Capture real empty/load/clear transitions with isolated application settings.

Run from an IDE or with conda run -n pyBer python scripts/validate_empty_workspaces.py.
Inputs are only read. Generated screenshots and integrity checks stay in _test.
"""

from contextlib import ExitStack
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "_test" / "empty_workspaces"
REQUESTED_DATA = ROOT / "_test" / "_data"
FALLBACK_DATA = ROOT.parent / "pyBer_test_data"
WINDOW_SIZE = (1500, 950)
os.environ.setdefault("QT_QPA_PLATFORM", "windows" if os.name == "nt" else "offscreen")
os.environ.setdefault("PYBER_SMOKE_TEST", "1")
sys.path.insert(0, str(ROOT / "pyBer"))

from PySide6 import QtCore, QtWidgets, QtTest
from analysis_core import PhotometryProcessor
from gui_postprocessing import PostProcessingPanel
from main import MainWindow
from styles import apply_app_palette


def digest(path):
    """Read a source hash without changing its content or metadata."""
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def capture(window, name):
    """Wait for real layout events and save the actual application surface."""
    QtTest.QTest.qWait(250)
    if not window.grab().save(str(OUTPUT / f"{name}.png")):
        raise RuntimeError(f"Could not save {name}")
    print(f"Captured {name}", flush=True)


def main():
    """Verify sparse initial views and restore valid plots from real recordings."""
    data = REQUESTED_DATA if REQUESTED_DATA.is_dir() else FALLBACK_DATA
    print(f"Reading test data from {data}", flush=True)
    paths = [data / name for name in ("trial_0027_with_artefacts.doric",
                                      "trial_0027_AIN01.csv", "trial_0027_behavior_with_time.csv")]
    hashes = {str(path): digest(path) for path in paths}
    OUTPUT.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="pyber-empty-validation-") as temporary, ExitStack() as stack:
        QtCore.QSettings.setDefaultFormat(QtCore.QSettings.Format.IniFormat)
        QtCore.QSettings.setPath(QtCore.QSettings.Format.IniFormat, QtCore.QSettings.Scope.UserScope, temporary)
        stack.enter_context(patch.object(MainWindow, "_panel_config_json_path", return_value=str(Path(temporary) / "layout.json")))
        stack.enter_context(patch.object(PostProcessingPanel, "_autosave_project_cache_path", return_value=str(Path(temporary) / "autosave.h5")))
        stack.enter_context(patch.object(PostProcessingPanel, "_restore_project_autosave_if_needed"))
        app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
        apply_app_palette(app, "dark")
        window = MainWindow()
        window.resize(*WINDOW_SIZE)
        window.show()
        QtTest.QTest.qWait(300)
        toaster = getattr(window, "_toaster", None)
        if toaster is not None:
            for toast in list(getattr(toaster, "_toasts", [])):
                toast.close()
        window._toaster = None
        post = window.post_tab
        for theme in ("dark", "light"):
            window._apply_app_theme(theme, persist=False)
            for tab, name in ((0, "preprocessing"), (1, "postprocessing")):
                window.tabs.setCurrentIndex(tab)
                capture(window, f"{theme}_{name}_empty")
            assert window.plots.plot_workspace.currentWidget() is window.plots.plot_empty_state
            assert post._results_stack.currentWidget() is post._empty_results
        window._apply_app_theme("dark", persist=False)
        window.tabs.setCurrentIndex(0)
        loaded = PhotometryProcessor().load_file(str(paths[0]))
        trial = loaded.make_trial(loaded.channels[0])
        window.plots.show_raw(trial.time, trial.signal_465, trial.reference_405)
        window.plots.set_title(paths[0].name)
        capture(window, "preprocessing_raw_loaded")
        assert window.plots.plot_raw.getPlotItem().isVisible()
        assert not window.plots.plot_proc.getPlotItem().isVisible()
        processed = post._load_processed_csv(str(paths[1]))
        assert processed is not None
        window.plots.update_plots(processed)
        capture(window, "preprocessing_results_loaded")
        assert window.plots.plot_out.getPlotItem().isVisible()
        window.plots.clear_plots()
        capture(window, "preprocessing_cleared")
        assert window.plots.plot_workspace.currentWidget() is window.plots.plot_empty_state
        window.tabs.setCurrentIndex(1)
        post.receive_current_processed([processed])
        capture(window, "postprocessing_trace_loaded")
        assert post.plot_trace.getPlotItem().isVisible()
        assert not post.plot_heat.getPlotItem().isVisible()
        post._load_behavior_paths([str(paths[2])], replace=True)
        post._refresh_behavior_list()
        post.combo_align.setCurrentText("Behavior (CSV/XLSX)")
        post.combo_behavior_name.setCurrentText("social_contacts")
        assert post.combo_behavior_name.currentText() == "social_contacts"
        post.spin_pre.setValue(3)
        post.spin_post.setValue(6)
        post.spin_b0.setValue(-2)
        post.spin_b1.setValue(0)
        post._compute_psth()
        assert post._last_mat is not None
        capture(window, "postprocessing_results_loaded")
        assert post.plot_heat.getPlotItem().isVisible()
        assert post.plot_avg.getPlotItem().isVisible()
        assert not post.heat_colorbar_widget.isHidden()
        post.receive_current_processed([])
        capture(window, "postprocessing_cleared")
        assert post._results_stack.currentWidget() is post._empty_results
        assert post._last_mat is None
        post._project_dirty = False
        window.close()
        app.processEvents()
    assert hashes == {str(path): digest(path) for path in paths}, "Input integrity failure"
    (OUTPUT / "validation.json").write_text(json.dumps({
        "requested_data": str(REQUESTED_DATA), "actual_data": str(data),
        "inputs_unchanged": True, "input_sha256": hashes,
        "transitions": "Empty, raw-only, computed and cleared states passed in both workspaces.",
    }, indent=2), encoding="utf-8")
    print(f"Validation passed: {OUTPUT}", flush=True)


if __name__ == "__main__":
    main()
