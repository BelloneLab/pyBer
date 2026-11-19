"""Select actual stored samples independently for each imported recording."""
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import h5py
import numpy as np
import test_postprocessing_empty_state as fixture
from PySide6 import QtWidgets
from analysis_core import export_processed_csv, export_processed_h5, load_processed_csv, load_processed_h5
from test_export_format import _make_proc
from test_postprocessing_export_pipeline import _export_to_folder


class RecordingOutputTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

    setUp = fixture.PostprocessingEmptyStateTests.setUp
    tearDown = fixture.PostprocessingEmptyStateTests.tearDown

    def picker(self, row):
        listing = self.panel.list_outputs
        return listing.itemWidget(listing.item(row)).findChild(QtWidgets.QComboBox)

    def test_primary_switch_psth_export_project_and_source_hashes(self):
        p = self.panel
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            trial = _make_proc(1000, multi=True)
            dff = trial.output_label
            zscore = "zscore (non motion corrected)"
            samples = np.sin(trial.time) + 3
            trial.outputs[zscore] = samples
            csv, h5 = root / "mouse1.csv", root / "mouse2.h5"
            export_processed_csv(str(csv), trial)
            export_processed_h5(str(h5), trial)
            # Metadata also wins when z-score, rather than dFF, is saved as primary.
            ztrial = _make_proc(1000)
            ztrial.output_label = zscore
            ztrial.output = samples
            ztrial.outputs[zscore] = samples
            zfile = root / "zprimary.csv"
            export_processed_csv(str(zfile), ztrial)
            self.assertEqual(load_processed_csv(str(zfile)).output_label, zscore)
            hashes = {path: hashlib.sha256(path.read_bytes()).digest() for path in root.iterdir()}
            p._load_processed_paths([str(csv), str(h5)], replace=True)
            for row in (0, 1):
                picker = self.picker(row)
                self.assertEqual(picker.currentText(), dff)
                self.assertEqual({picker.itemText(i) for i in range(picker.count())}, set(trial.outputs))
                p._processed[row].injection_seconds = 30.
            p.tab_sources.setCurrentIndex(1)
            p.psth_align_source.setCurrentText("Event marker")
            p.combo_psth_normalization.setCurrentIndex(p.combo_psth_normalization.findData("none"))
            p.tab_sources.setCurrentIndex(0)
            p._compute_psth()
            original_matrix = p._per_file_mats["mouse1"][1].copy()
            p.list_preprocessed.item(0).setSelected(True)
            self.picker(0).setCurrentText(zscore)
            self.assertTrue(p.list_preprocessed.item(0).isSelected())
            self.assertEqual(self.picker(1).currentText(), dff)
            np.testing.assert_allclose(p._processed[0].output, samples)
            self.assertEqual(p._processed[0].injection_seconds, 30.)
            self.assertFalse(np.allclose(original_matrix, p._per_file_mats["mouse1"][1], equal_nan=True))
            self.picker(0).setCurrentText(dff)
            np.testing.assert_allclose(p._per_file_mats["mouse1"][1], original_matrix, equal_nan=True)
            self.picker(1).setCurrentText(zscore)
            p._processed.reverse()
            p._update_file_lists()
            self.assertEqual(self.picker(0).currentText(), zscore)
            project = root / "project.h5"
            p._save_project_h5(str(project))
            self.assertTrue(p._load_project_from_path(str(project)))
            self.assertEqual(self.picker(0).currentText(), zscore)
            self.assertEqual(self.picker(0).count(), 3)
            np.testing.assert_allclose(p._processed[0].output, samples)
            self.picker(0).setCurrentText(dff)
            np.testing.assert_allclose(p._processed[0].output, trial.outputs[dff])
            self.picker(0).setCurrentText(zscore)
            _export_to_folder(p, str(root))
            manifest = json.loads(next(root.glob("*_manifest.json")).read_text())
            self.assertEqual(manifest["input_signals"], [
                {"file": str(h5), "output_label": zscore}, {"file": str(csv), "output_label": dff}])
            for path, digest in hashes.items():
                self.assertEqual(hashlib.sha256(path.read_bytes()).digest(), digest)

    def test_legacy_columns_without_sidecars_and_missing_samples(self):
        with tempfile.TemporaryDirectory() as folder:
            csv = Path(folder) / "old.csv"
            csv.write_text("time,dFF,z-score,DIO01\n0,1,7,0\n1,2,,1\n2,,9,0\n")
            data = load_processed_csv(str(csv))
            self.assertEqual(set(data.outputs), {"dFF", "z-score"})
            np.testing.assert_equal(data.outputs["z-score"], [7, np.nan, 9])
            h5 = Path(folder) / "old.h5"
            with h5py.File(h5, "w") as f:
                g = f.create_group("data")
                for name, values in {"time": [0,1,2], "dFF": [1,2,3], "z-score": [7,8,9],
                                     "DIO01": [0,1,0]}.items():
                    g.create_dataset(name, data=values)
            data = load_processed_h5(str(h5))
            self.assertEqual(set(data.outputs), {"dFF", "z-score"})
            np.testing.assert_equal(data.output, [1,2,3])
