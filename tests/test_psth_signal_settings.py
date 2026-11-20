"""PSTH choices act on stored samples, retain legacy inputs, and explain the actual math."""
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import h5py
import numpy as np
import test_postprocessing_empty_state as fixture
from analysis_core import ProcessedTrial, load_processed_csv, load_processed_h5, zscore_median_std
from test_postprocessing_export_pipeline import _export_to_folder


class PsthSignalSettingsTests(unittest.TestCase):
    setUpClass = classmethod(fixture.PostprocessingEmptyStateTests.setUpClass.__func__)
    setUp = fixture.PostprocessingEmptyStateTests.setUp
    tearDown = fixture.PostprocessingEmptyStateTests.tearDown

    def trial(self, name, with_z=True):
        t = np.arange(101) / 10
        dff = .01 * t ** 2
        outputs = {"dFF": dff}
        if with_z:
            outputs["zscore (non motion corrected)"] = zscore_median_std(dff)
        return ProcessedTrial(path=name, channel_id="AIN01", time=t, raw_signal=dff,
                              raw_reference=dff, output=dff, output_label="dFF", outputs=outputs,
                              injection_seconds=5.)

    def load(self, files):
        p = self.panel
        p._processed = files
        p._update_file_lists()
        p.tab_sources.setCurrentIndex(1)
        p.tab_visual_mode.setCurrentIndex(1)
        p.psth_align_source.setCurrentText("Event marker")
        p.spin_resample.setValue(10)
        p.spin_pre.setValue(2)
        p.spin_post.setValue(2)
        p.spin_b0.setValue(-2)
        p.spin_b1.setValue(-1)
        return p

    def choose(self, source, method):
        p = self.panel
        p.combo_psth_input.setCurrentIndex(p.combo_psth_input.findData(source))
        p.combo_psth_normalization.setCurrentIndex(p.combo_psth_normalization.findData(method))
        p._compute_psth()

    def test_four_choices_against_native_baseline_and_settings_only_formula(self):
        r = self.trial("one.csv")
        p = self.load([r])
        self.assertTrue(p.lbl_psth_formula.isHidden())
        p.btn_psth_formula.click()
        self.assertFalse(p.lbl_psth_formula.isHidden())
        self.assertTrue(p.psth_subsections.isAncestorOf(p.lbl_psth_formula))
        self.assertFalse(p.psth_shared_card.isAncestorOf(p.lbl_psth_formula))
        results = {}
        for source, label in (("dFF", "dFF"), ("z-score", "zscore (non motion corrected)")):
            for mode in ("none", "zscore"):
                self.choose(source, mode)
                x = r.outputs[label]
                baseline = x[30:41]
                expected = x[30:71] if mode == "none" else (x[30:71] - baseline.mean()) / baseline.std()
                np.testing.assert_allclose(p._last_mat[0], expected, atol=1e-12)
                results[source, mode] = p._last_mat[0].copy()
                self.assertIn("Y = X." if mode == "none" else "(X − mean(B)) / SD(B)", p.lbl_psth_formula.text())
                self.assertEqual(p._psth_baseline_row.isHidden(), mode == "none")
                self.assertNotIn("mean(B)", p.lbl_psth_plot_summary.text())
                self.assertIn(label, p.lbl_psth_plot_summary.text())
        np.testing.assert_allclose(results["dFF", "zscore"], results["z-score", "zscore"], atol=1e-12)
        p.btn_psth_advanced.click()
        p.cb_psth_subtract.setChecked(True)
        p._compute_psth()
        z = r.outputs["zscore (non motion corrected)"]
        np.testing.assert_allclose(p._last_mat[0], z[30:71] - z[30:41].mean())
        self.assertIn("Y = X − mean(B)", p.lbl_psth_formula.text())
        self.assertEqual(r.output_label, "dFF")

    def test_missing_file_requires_explicit_subset_and_roundtrips(self):
        p = self.load([self.trial("a.csv"), self.trial("b.csv", with_z=False)])
        self.choose("z-score", "none")
        self.assertIn("b.csv", p.lbl_psth_availability.text())
        self.assertIsNone(p._last_mat)
        p.cb_psth_available_only.click()
        p._compute_psth()
        self.assertEqual(p._group_labels, ["a"])
        self.assertIn("b", p._psth_excluded_files)
        with tempfile.TemporaryDirectory() as folder:
            _export_to_folder(p, folder, avg=True)
            manifest = json.loads(next(Path(folder).glob("*_manifest.json")).read_text())
            self.assertEqual(manifest["input_signals"][0]["output_label"], "zscore (non motion corrected)")
            self.assertIsNone(manifest["input_signals"][1]["output_label"])
            project = str(Path(folder) / "project.h5")
            p._save_project_h5(project)
            self.choose("dFF", "zscore")
            self.assertTrue(p._load_project_from_path(project))
            self.assertEqual(p.combo_psth_input.currentData(), "z-score")
            self.assertTrue(p.cb_psth_available_only.isChecked())
            self.assertEqual(p._psth_normalization(), "none")
            self.assertEqual([r.injection_seconds for r in p._processed], [5., 5.])
            self.assertEqual([r.output_label for r in p._processed], ["dFF", "dFF"])
            p._apply_settings({"psth_normalization": "Subtract baseline"})
            self.assertEqual(p.combo_psth_input.currentData(), "saved")
            self.assertEqual(p._psth_normalization(), "subtract")

    def test_multiple_variants_require_an_explicit_choice(self):
        r = self.trial("variants.csv")
        alternate = "zscore (motion corrected with fitted ref)"
        r.outputs[alternate] = -r.outputs["zscore (non motion corrected)"]
        p = self.load([r])
        self.choose("z-score", "none")
        self.assertIsNone(p._last_mat)
        self.assertIn(r.path, p.lbl_psth_availability.text())
        p.btn_psth_advanced.click()
        self.choose("saved", "none")
        picker = p.list_outputs.itemWidget(p.list_outputs.item(0)).findChild(fixture.QtWidgets.QComboBox)
        picker.setCurrentText(alternate)
        np.testing.assert_allclose(p._last_mat[0], r.outputs[alternate][30:71])
        self.assertEqual(r.output_label, "dFF")
        self.assertIn(alternate, p.lbl_psth_plot_summary.text())

    def test_old_project_recovers_exact_linked_zscore_when_selected(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "linked.csv"
            trial = self.trial(str(path), with_z=False)
            trial.output_label = "Imported CSV (dFF)"
            trial.outputs = {}
            saved_z = np.cos(trial.time) + 17
            np.savetxt(path, np.column_stack([trial.time, trial.output, saved_z]),
                       delimiter=",", header="time,dFF,z-score", comments="")
            p = self.load([trial])
            group = p.recording_groups.add_group("Drug")
            p.recording_groups.assign(["linked"], group)
            project = str(Path(folder) / "old.h5")
            p._save_project_h5(project)
            self.assertTrue(p._load_project_from_path(project))
            self.choose("z-score", "none")
            np.testing.assert_array_equal(p._psth_input(p._processed[0])[1], saved_z)
            np.testing.assert_allclose(p._last_mat[0], saved_z[30:71], atol=1e-12)
            self.assertEqual(p._processed[0].injection_seconds, 5.)
            self.assertEqual(p.recording_groups.assignments()["linked"], group)
            self.assertEqual(p._processed[0].output_label, "Imported CSV (dFF)")

    def test_open_saved_csv_from_preprocessing_preserves_samples_and_seconds(self):
        from analysis_core import load_rwd_csv
        from main import MainWindow
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "processed.csv"
            path.write_text("time,raw_465,raw_405,dFF,z-score,baseline_465,baseline_405\n"
                            "30.0,8,4,0.01,1.25,5,2\n30.02,9,3,0.05,4.5,5,2\n")
            self.assertIsNone(load_rwd_csv(str(path)))
            tabs = fixture.QtWidgets.QTabWidget()
            tabs.addTab(fixture.QtWidgets.QWidget(), "Preprocessing")
            tabs.addTab(self.panel, "Postprocessing")
            try:
                def use_saved():
                    prompt = fixture.QtWidgets.QApplication.activeModalWidget()
                    prompt.defaultButton().click()
                fixture.QtCore.QTimer.singleShot(0, use_saved)
                window = MainWindow.__new__(MainWindow)
                fixture.QtWidgets.QMainWindow.__init__(window)
                window.post_tab = self.panel
                window.tabs = tabs
                window._add_files([str(path)])
                self.assertIs(tabs.currentWidget(), self.panel)
                self.choose("z-score", "none")
                trial, = self.panel._processed
                np.testing.assert_array_equal(trial.time, [30., 30.02])
                np.testing.assert_array_equal(self.panel._psth_input(trial)[1], [1.25, 4.5])
            finally:
                self.panel.setParent(None)

    def test_recover_original_csv_into_primary_only_project(self):
        r = self.trial("recording.csv", with_z=False)
        other = self.trial("different.csv", with_z=False)
        p = self.load([r, other])
        group = p.recording_groups.add_group("Injected")
        p.recording_groups.assign(["recording"], group)
        with tempfile.TemporaryDirectory() as folder:
            project = str(Path(folder) / "old.h5")
            p._save_project_h5(project)
            self.assertTrue(p._load_project_from_path(project))
            self.choose("z-score", "none")
            self.assertIsNone(p._last_mat)
            self.assertFalse(p.btn_recover_psth_signal.isHidden())
            self.assertNotIn("recording.csv", p.lbl_psth_availability.text())
            self.assertIn("recording.csv", p.lbl_psth_availability.toolTip())
            z = np.arange(r.time.size, dtype=float) + 7
            paths = []
            for name, dff in (("recording (1).csv", r.output), ("different.csv", other.output + 1)):
                path = str(Path(folder) / name)
                np.savetxt(path, np.column_stack([r.time, dff, z]), delimiter=",",
                           header="time,dFF,z-score", comments="")
                paths.append(path)
            with patch.object(fixture.QtWidgets.QFileDialog, "getOpenFileNames", return_value=(paths, "")), \
                 patch.object(fixture.QtWidgets.QMessageBox, "information") as report:
                p.btn_recover_psth_signal.click()
            self.assertIn("1 of 2", report.call_args.args[2])
            self.assertIsNone(p._last_mat)
            p.cb_psth_available_only.click()
            p._compute_psth()
            np.testing.assert_allclose(p._last_mat[0], z[30:71], rtol=0, atol=1e-12)
            updated = str(Path(folder) / "updated.h5")
            p._save_project_h5(updated)
            self.assertTrue(p._load_project_from_path(updated))
            self.assertEqual(p.recording_groups.assignments().get("recording"), group)
            self.assertEqual([f.injection_seconds for f in p._processed], [5., 5.])
            self.assertEqual([f.output_label for f in p._processed], ["dFF", "dFF"])
            np.testing.assert_equal(p._processed[0].outputs["z-score"], z)
            self.assertNotIn("z-score", p._processed[1].outputs)

    def test_legacy_full_output_headers_in_csv_and_h5(self):
        with tempfile.TemporaryDirectory() as folder:
            csv, h5 = Path(folder)/"r.csv", Path(folder)/"r.h5"
            label = "zscore (non motion corrected)"
            csv.write_text(f"time,dFF,{label}\n0,1,7\n1,2,8\n")
            with h5py.File(h5, "w") as f:
                data = f.create_group("data")
                for key, vals in (("time", [0, 1]), ("dFF", [1, 2]), (label, [7, 8])):
                    data.create_dataset(key, data=vals)
            for trial in (load_processed_csv(str(csv)), load_processed_h5(str(h5))):
                np.testing.assert_equal(trial.outputs[label], [7, 8])
