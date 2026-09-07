"""Regression coverage for result identity, invalidation, and complete exports.

These tests execute production panel methods with a small UI state double so
they exercise scientific/export contracts without constructing the large GUI.
"""

import csv
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import MethodType, SimpleNamespace
from unittest.mock import MagicMock, patch

import h5py
import numpy as np

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("PYBER_SMOKE_TEST", "1")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "pyBer"))

import gui_postprocessing as gui


class _Control:
    """Expose the read interface of the controls consumed by panel methods."""

    def __init__(self, value=0):
        self.state = value

    def value(self):
        return self.state

    def currentIndex(self):
        return self.state

    def currentText(self):
        return str(self.state)

    def isChecked(self):
        return bool(self.state)


def _bind(panel, *names):
    """Bind the exact panel implementation to a deliberately small test state."""
    for name in names:
        setattr(panel, name, MethodType(getattr(gui.PostProcessingPanel, name), panel))


def _panel():
    """Create an existing successful result and the settings that produced it."""
    panel = SimpleNamespace(
        _processed=[SimpleNamespace(path="mouse_01.csv", output=np.arange(100.0))],
        tab_sources=_Control(0),
        tab_visual_mode=_Control(0),
        combo_individual_file=_Control("mouse_01"),
        combo_align=_Control("Behavior (CSV/XLSX)"),
        combo_behavior_name=_Control("Contact"),
        combo_behavior_align=_Control("Align to onset"),
        combo_behavior_from=_Control("Approach"),
        combo_behavior_to=_Control("Contact"),
        spin_pre=_Control(1.0), spin_post=_Control(1.0),
        spin_b0=_Control(-1.0), spin_b1=_Control(0.0),
        spin_resample=_Control(10.0), spin_smooth=_Control(0.0),
        _last_mat=np.array([[1.0, 2.0], [3.0, 4.0]]),
        _last_tvec=np.array([-1.0, 1.0]),
        _last_metrics={"metric": "Mean z", "pre": 2.0, "post": 3.0,
                       "pre_sem": 1.0, "post_sem": 1.0,
                       "paired_n": 2.0, "paired_p": 0.25},
        _last_global_metrics=None,
        _last_events=np.array([5.0, 15.0]),
        _last_durations=np.array([1.0, 2.0]),
        _last_event_rows=[
            {"file_id": "mouse_01", "event_time_sec": 5.0, "duration_sec": 1.0},
            {"file_id": "mouse_01", "event_time_sec": 15.0, "duration_sec": 2.0},
        ],
        _last_display_labels=["Trial 1", "Trial 2"],
        _last_psth_display_level="trials",
        _per_file_mats={}, _per_file_labels={}, _all_file_ids=[],
        _group_mat=None, _group_tvec=None, _group_labels=[],
        _group_trial_mat=None, _group_trial_tvec=None, _group_trial_labels=[],
        _psth_excluded_files={},
        statusUpdate=MagicMock(), exportProgress=MagicMock(),
        lbl_plot_file=MagicMock(), plot_avg=MagicMock(),
    )
    for name in (
        "_queue_settings_save", "_update_trace_preview", "_update_status_strip", "_refresh_psth_duration_view",
        "_sync_temporal_modeling_context", "_refresh_individual_file_combo",
        "_render_global_metrics", "_clear_psth_visuals", "_render_heatmap",
        "_render_avg", "_render_metrics", "_render_duration_hist",
        "_update_metric_regions", "_save_settings", "_remember_export_dir",
        "_write_export_parameter_file", "_update_data_availability",
    ):
        setattr(panel, name, MagicMock())
    panel._collect_settings = lambda: {}
    panel._collect_psth_parameter_sections = lambda **kwargs: []
    panel._psth_min_events_per_animal = lambda: 3
    panel._psth_exclude_low_event_animals_enabled = lambda: True
    panel._psth_group_trial_view_enabled = lambda: False
    panel._get_events_for_proc = lambda proc: (np.array([], float), np.array([], float))
    panel._filter_events = lambda events, durations: (events, durations)
    panel._proc_time = lambda proc: np.arange(100.0)
    _bind(panel, "_compute_psth", "_compute_psth_impl", "_ensure_current_psth",
          "_clear_psth_result_view", "_clear_psth_cache", "_stack_psth_trial_rows", "_behavior_suffix",
          "_alignment_export_suffix", "_is_group_export_context", "_group_export_prefix",
          "_default_export_prefix", "_psth_normalization")
    return panel


def _export_to_folder(panel, folder, **overrides):
    """Run the production export with explicit choices and a temporary folder."""
    choices = {"csv": True, "h5": True, "png": False, "pdf": False,
               "events": True, "durations": True}
    choices.update(overrides)
    dialog = MagicMock()
    dialog.exec.return_value = gui.QtWidgets.QDialog.DialogCode.Accepted
    dialog.choices.return_value = choices
    panel._export_start_dir = lambda: folder
    with patch.object(gui, "ExportDialog", return_value=dialog), patch.object(
        gui.QtWidgets.QFileDialog, "getExistingDirectory", return_value=folder
    ):
        gui.PostProcessingPanel._export_results(panel)


class ResultInvalidationTests(unittest.TestCase):
    def assert_no_exportable_result(self, panel):
        """An unsuccessful computation must not retain an old exportable result."""
        self.assertIsNone(panel._last_mat)
        self.assertIsNone(panel._last_tvec)
        self.assertIsNone(panel._last_metrics)
        self.assertEqual(panel._last_event_rows, [])

    def test_no_events_invalidates_previous_result(self):
        panel = _panel()
        gui.PostProcessingPanel._compute_psth(panel)
        self.assert_no_exportable_result(panel)

    def test_failed_computation_invalidates_previous_result(self):
        panel = _panel()
        panel._get_events_for_proc = MagicMock(side_effect=ValueError("invalid event data"))
        with self.assertLogs(gui._LOG, level="ERROR"):
            gui.PostProcessingPanel._compute_psth(panel)
        self.assert_no_exportable_result(panel)

    def test_pending_edits_with_no_events_cannot_export_old_results(self):
        panel = _panel()
        panel._psth_pending = True
        # Switching to an alignment with no events must be evaluated before
        # the export dialog can act on the previously displayed result.
        panel.combo_behavior_align.state = "Align to offset"
        with patch.object(gui, "ExportDialog") as dialog:
            gui.PostProcessingPanel._export_results(panel)
        dialog.assert_not_called()
        self.assert_no_exportable_result(panel)

    def test_missing_individual_clears_previous_file_display(self):
        panel = _panel()
        panel.combo_individual_file.state = "mouse_without_events"
        gui.PostProcessingPanel._rerender_visual_from_cache(panel)
        self.assertIsNone(panel._last_mat)
        self.assertIsNone(panel._last_tvec)
        self.assertIsNone(panel._last_metrics)

    def test_excluded_trials_cannot_reappear_when_switching_to_group(self):
        panel = _panel()
        panel.tab_sources.state = panel.tab_visual_mode.state = 1
        panel._psth_group_trial_view_enabled = lambda: True
        panel._per_file_mats = {"mouse_01": (panel._last_tvec.copy(), panel._last_mat.copy())}
        panel._all_file_ids = ["mouse_01"]
        panel._psth_excluded_files = {"mouse_01": {"event_count": 2, "min_events": 3}}
        gui.PostProcessingPanel._rerender_visual_from_cache(panel)
        self.assertIsNone(panel._last_mat)
        self.assertFalse(panel._render_heatmap.called and
                         any(np.asarray(call.args[0]).size for call in panel._render_heatmap.call_args_list))


class ExportIdentityTests(unittest.TestCase):
    def test_individual_events_and_durations_use_selected_file_in_both_formats(self):
        panel = _panel()
        panel.tab_sources.state = 1
        panel.combo_individual_file.state = "mouse_02"
        panel._processed.append(SimpleNamespace(path="mouse_02.csv"))
        # This file was excluded from group summaries but remains inspectable
        # individually. Its rows must come from the per-file accepted cache.
        panel._per_file_event_rows = {"mouse_02": [
            {"file_id": "mouse_02", "event_time_sec": 25.0, "duration_sec": 3.0}
        ]}
        panel._psth_excluded_files = {"mouse_02": {"event_count": 1, "min_events": 3}}
        with tempfile.TemporaryDirectory(prefix="pyber_scope_test_") as folder:
            _export_to_folder(panel, folder)
            prefix = panel._default_export_prefix()
            with open(Path(folder) / f"{prefix}_events.csv", newline="", encoding="utf-8") as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["file_id"], "mouse_02")
            self.assertEqual(float(rows[0]["event_time_sec"]), 25.0)
            np.testing.assert_allclose(np.loadtxt(Path(folder) / f"{prefix}_durations.csv", skiprows=1), 3.0)
            with h5py.File(Path(folder) / f"{prefix}_events.h5", "r") as handle:
                np.testing.assert_allclose(handle["event_time_sec"][:], [25.0])
                self.assertEqual(handle["file_id"].asstr()[:].tolist(), ["mouse_02"])
            with h5py.File(Path(folder) / f"{prefix}_durations.h5", "r") as handle:
                np.testing.assert_allclose(handle["duration_sec"][:], [3.0])
            manifest = json.loads((Path(folder) / f"{prefix}_manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["scope"], "individual")
            self.assertEqual(manifest["event_count"], 1)
            self.assertEqual(manifest["selected_file"], "mouse_02")
            self.assertEqual(len(manifest["outputs"]), 4)

    def test_repeated_export_preserves_previous_bundle(self):
        panel = _panel()
        with tempfile.TemporaryDirectory(prefix="pyber_repeat_test_") as folder:
            _export_to_folder(panel, folder)
            prefix = panel._default_export_prefix()
            original = Path(folder) / f"{prefix}_events.csv"
            original_bytes = original.read_bytes()
            panel._last_event_rows[0]["event_time_sec"] = 99.0
            _export_to_folder(panel, folder)
            self.assertEqual(original.read_bytes(), original_bytes)
            repeat = Path(folder) / f"{prefix}_run_2_events.csv"
            self.assertTrue(repeat.is_file())
            self.assertNotEqual(repeat.read_bytes(), original_bytes)
            manifest = json.loads((Path(folder) / f"{prefix}_run_2_manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["status"], "complete")
            self.assertTrue(all(name.startswith(f"{prefix}_run_2_") for name in manifest["outputs"]))

    def test_failed_writer_does_not_report_export_complete(self):
        panel = _panel()
        with tempfile.TemporaryDirectory(prefix="pyber_failed_test_") as folder:
            with patch.object(gui.np, "savetxt", side_effect=OSError("disk full")), self.assertLogs(gui._LOG, level="ERROR"):
                _export_to_folder(panel, folder, heatmap=True)
            messages = [str(call.args[0]) for call in panel.statusUpdate.emit.call_args_list]
            self.assertTrue(any("Export failed" in message for message in messages))
            self.assertFalse(any("Export complete" in message for message in messages))
            prefix = panel._default_export_prefix()
            manifest = json.loads((Path(folder) / f"{prefix}_manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["status"], "failed")
            self.assertIn("disk full", manifest["error"])

    def test_selected_individual_has_its_own_export_prefix(self):
        panel = _panel()
        panel.tab_sources.state = 1
        panel._processed.append(SimpleNamespace(path="mouse_02.csv"))
        first = panel._default_export_prefix()
        panel.combo_individual_file.state = "mouse_02"
        second = panel._default_export_prefix()
        self.assertNotEqual(first, second)
        self.assertIn("mouse_02", second)

    def test_group_animals_and_trials_have_distinct_export_prefixes(self):
        panel = _panel()
        panel.tab_sources.state = panel.tab_visual_mode.state = 1
        panel._last_psth_display_level = "animals"
        animals = panel._default_export_prefix()
        panel._last_psth_display_level = "trials"
        panel._psth_group_trial_view_enabled = lambda: True
        trials = panel._default_export_prefix()
        self.assertNotEqual(animals, trials)

    def test_hdf5_only_exports_events_durations_and_metric_statistics(self):
        panel = _panel()
        choices = {"csv": False, "h5": True, "png": False, "pdf": False,
                   "events": True, "durations": True, "metrics": True}
        dialog = MagicMock()
        dialog.exec.return_value = gui.QtWidgets.QDialog.DialogCode.Accepted
        dialog.choices.return_value = choices
        with tempfile.TemporaryDirectory(prefix="pyber_export_test_") as folder:
            panel._export_start_dir = lambda: folder
            with patch.object(gui, "ExportDialog", return_value=dialog), patch.object(
                gui.QtWidgets.QFileDialog, "getExistingDirectory", return_value=folder
            ):
                gui.PostProcessingPanel._export_results(panel)
            prefix = panel._default_export_prefix()
            for suffix, datasets in {
                "events": ("file_id", "event_time_sec", "duration_sec"),
                "durations": ("duration_sec",),
                "metrics": ("pre", "post", "pre_sem", "post_sem", "paired_n", "paired_p"),
            }.items():
                path = Path(folder) / f"{prefix}_{suffix}.h5"
                self.assertTrue(path.is_file(), f"Missing requested HDF5 export: {path.name}")
                with h5py.File(path, "r") as handle:
                    for name in datasets:
                        self.assertIn(name, handle)
                    if suffix == "events":
                        np.testing.assert_allclose(handle["event_time_sec"][:], [5.0, 15.0])
                    if suffix == "metrics":
                        self.assertAlmostEqual(float(handle["paired_p"][()]), 0.25)


if __name__ == "__main__":
    unittest.main()
