"""Tests for the pyBer processed-trace v1.0 export/import contract.

Covers the writer (clean CSV, family column names, no generic "output" column,
sidecar JSON, self-contained HDF5), and the postprocessing loaders (deterministic
column-role selection from the sidecar, plus backward compatibility with the
legacy comment-line CSV / "data/output" HDF5 layout).
"""
import csv
import json
import os
import sys
import tempfile
import unittest

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "pyBer"))

import analysis_core as ac  # noqa: E402
from analysis_core import (  # noqa: E402
    ExportSelection,
    ProcessedTrial,
    ProcessingParams,
    export_processed_csv,
    export_processed_h5,
    read_processed_sidecar,
)

INVERTED_FIT_MODE = "dFF (motion corrected with inverted isobestic fit)"
BAND_LIMITED_INVERTED_MODE = ac.BAND_LIMITED_INVERTED_ISO_MODE


def _make_proc(n=40, multi=False):
    rng = np.arange(n, dtype=float)
    outputs = {"dFF (motion corrected with fitted ref)": 0.01 * rng}
    if multi:
        outputs["dFF (non motion corrected)"] = 0.02 * rng
    return ProcessedTrial(
        path=r"C:\data\M12_raw.doric",
        channel_id="AIN01",
        time=np.linspace(0.0, (n - 1) / 10.0, n),
        raw_signal=100.0 + rng,
        raw_reference=50.0 + 0.5 * rng,
        baseline_sig=99.0 + rng,
        baseline_ref=49.5 + 0.5 * rng,
        dio=(rng % 10 < 2).astype(float),
        dio_name="DIO01",
        triggers={"DIO01": (rng % 10 < 2).astype(float), "DIO02": (rng % 7 < 1).astype(float)},
        output=0.01 * rng,
        output_label="dFF (motion corrected with fitted ref)",
        output_context="Fit: OLS (recommended) | Baseline: airpls",
        outputs=outputs,
        sync_aligned_time=np.linspace(0.0, (n - 1) / 10.0, n) + 0.123,
        sync_report={"status": "ok", "lag_s": 0.123},
        fs_actual=100.0, fs_target=100.0, fs_used=100.0,
    )


def _full_selection(modes):
    return ExportSelection(
        raw=True, isobestic=True, output=True, dio=True,
        baseline_sig=True, baseline_ref=True, output_modes=list(modes),
    )


class ColumnNamingTests(unittest.TestCase):
    def test_family_and_variant_mapping(self):
        self.assertEqual(ac.output_family("dFF (motion corrected with fitted ref)"), "dFF")
        self.assertEqual(ac.output_variant_key("dFF (motion corrected with fitted ref)"), "fitref")
        self.assertEqual(ac.output_family(INVERTED_FIT_MODE), "dFF")
        self.assertEqual(ac.output_variant_key(INVERTED_FIT_MODE), "invfitref")
        self.assertEqual(ac.output_family(BAND_LIMITED_INVERTED_MODE), "dFF")
        self.assertEqual(ac.output_variant_key(BAND_LIMITED_INVERTED_MODE), "bandinvfitref")
        self.assertEqual(ac.output_family("zscore (subtractions)"), "z-score")
        self.assertEqual(ac.output_variant_key("zscore (subtractions)"), "zdiff")
        self.assertEqual(ac.output_family("Raw signal (465)"), "signal_465")

    def test_primary_gets_bare_family_extras_disambiguated(self):
        names = ac.assign_output_column_names([
            "dFF (motion corrected with fitted ref)",
            "dFF (non motion corrected)",
            "zscore (motion corrected with fitted ref)",
        ])
        self.assertEqual([n for _, n in names], ["dFF", "dFF__nomc", "z-score"])

    def test_inverted_fit_disambiguates_same_family_outputs(self):
        names = ac.assign_output_column_names([
            INVERTED_FIT_MODE,
            "dFF (motion corrected with fitted ref)",
            BAND_LIMITED_INVERTED_MODE,
        ])
        self.assertEqual([n for _, n in names], ["dFF", "dFF__fitref", "dFF__bandinvfitref"])


class CsvWriterTests(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp(prefix="pyber_csv_")

    def _header(self, path):
        with open(path, newline="") as f:
            return next(csv.reader(f))

    def test_single_output_clean_header_no_comments(self):
        proc = _make_proc()
        path = os.path.join(self.dir, "rec.csv")
        export_processed_csv(path, proc, selection=_full_selection(
            ["dFF (motion corrected with fitted ref)"]), params=ProcessingParams())
        header = self._header(path)
        self.assertEqual(header, ["time", "time_aligned", "raw_465", "raw_405",
                                  "dFF", "baseline_465", "baseline_405", "DIO01", "DIO02"])
        with open(path) as f:
            self.assertFalse(f.readline().startswith("#"), "CSV must have no comment lines")
        self.assertNotIn("output", header, "no generic 'output' column")

    def test_multi_output_no_duplicate_generic_column(self):
        proc = _make_proc(multi=True)
        path = os.path.join(self.dir, "rec.csv")
        export_processed_csv(path, proc, selection=_full_selection([
            "dFF (motion corrected with fitted ref)", "dFF (non motion corrected)"]),
            params=ProcessingParams())
        header = self._header(path)
        self.assertIn("dFF", header)
        self.assertIn("dFF__nomc", header)
        self.assertNotIn("output", header)
        self.assertEqual(header.count("dFF"), 1, "primary must not be duplicated")

    def test_no_output_when_output_deselected(self):
        proc = _make_proc()
        sel = ExportSelection(raw=True, isobestic=True, output=False, dio=False,
                              baseline_sig=False, baseline_ref=False)
        path = os.path.join(self.dir, "rec.csv")
        export_processed_csv(path, proc, selection=sel, params=ProcessingParams())
        header = self._header(path)
        self.assertEqual(header, ["time", "time_aligned", "raw_465", "raw_405"])
        self.assertNotIn("dFF", header)
        self.assertNotIn("output", header)


class SidecarTests(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp(prefix="pyber_side_")

    def test_sidecar_records_output_nature_and_params(self):
        proc = _make_proc(multi=True)
        params = ProcessingParams()
        params.reference_fit = "OLS (recommended)"
        meta = {"animal_id": "M12", "session": "day1", "custom:cohort": "A"}
        path = os.path.join(self.dir, "rec.csv")
        export_processed_csv(path, proc, metadata=meta, params=params,
                             selection=_full_selection([
                                 "dFF (motion corrected with fitted ref)",
                                 "dFF (non motion corrected)"]))
        side = read_processed_sidecar(path)
        self.assertIsNotNone(side)
        self.assertEqual(side["pyber_format_version"], "1.0")
        self.assertEqual(side["primary_output"], "dFF")
        self.assertEqual(side["outputs"]["dFF"]["label"],
                         "dFF (motion corrected with fitted ref)")
        self.assertEqual(side["outputs"]["dFF"]["variant"], "fitref")
        self.assertEqual(side["outputs"]["dFF"]["reference_fit"], "OLS (recommended)")
        self.assertEqual(side["outputs"]["dFF__nomc"]["variant"], "nomc")
        self.assertEqual(side["subject"]["animal_id"], "M12")
        self.assertEqual(side["subject"]["custom"]["cohort"], "A")
        self.assertEqual(side["processing"]["baseline_method"], params.baseline_method)
        roles = {c["name"]: c["role"] for c in side["columns"]}
        self.assertEqual(roles["dFF"], "output")
        self.assertEqual(roles["raw_465"], "raw_signal")
        self.assertEqual(roles["DIO01"], "trigger")
        self.assertEqual(side["triggers"], ["DIO01", "DIO02"])

    def test_sidecar_records_inverted_fit_variant_and_params(self):
        proc = _make_proc(multi=True)
        rng = np.arange(proc.time.size, dtype=float)
        proc.output = 0.03 * rng
        proc.output_label = INVERTED_FIT_MODE
        proc.outputs = {
            INVERTED_FIT_MODE: 0.03 * rng,
            "dFF (motion corrected with fitted ref)": 0.01 * rng,
        }
        params = ProcessingParams()
        params.reference_fit = "RLM (HuberT)"
        path = os.path.join(self.dir, "rec_inverted.csv")
        export_processed_csv(path, proc, params=params,
                             selection=_full_selection([
                                 INVERTED_FIT_MODE,
                                 "dFF (motion corrected with fitted ref)",
                             ]))
        side = read_processed_sidecar(path)
        self.assertIsNotNone(side)
        self.assertEqual(side["primary_output"], "dFF")
        self.assertEqual(side["outputs"]["dFF"]["label"], INVERTED_FIT_MODE)
        self.assertEqual(side["outputs"]["dFF"]["variant"], "invfitref")
        self.assertEqual(side["outputs"]["dFF"]["motion_correction"], "inverted_fitted_ref")
        self.assertEqual(side["outputs"]["dFF"]["reference_fit"], "RLM (HuberT)")
        self.assertEqual(side["outputs"]["dFF__fitref"]["variant"], "fitref")

    def test_sidecar_records_band_limited_inverted_variant_and_window(self):
        proc = _make_proc(multi=True)
        rng = np.arange(proc.time.size, dtype=float)
        proc.output = 0.04 * rng
        proc.output_label = BAND_LIMITED_INVERTED_MODE
        proc.outputs = {BAND_LIMITED_INVERTED_MODE: 0.04 * rng}
        params = ProcessingParams()
        params.band_limited_reference_window_s = 45.0
        path = os.path.join(self.dir, "rec_band_inverted.csv")
        export_processed_csv(path, proc, params=params,
                             selection=_full_selection([BAND_LIMITED_INVERTED_MODE]))
        side = read_processed_sidecar(path)
        self.assertIsNotNone(side)
        self.assertEqual(side["outputs"]["dFF"]["variant"], "bandinvfitref")
        self.assertEqual(side["outputs"]["dFF"]["motion_correction"], "band_limited_inverted_fitted_ref")
        self.assertEqual(side["outputs"]["dFF"]["band_limited_reference_window_s"], 45.0)


class H5WriterTests(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp(prefix="pyber_h5_")

    def test_h5_family_datasets_and_embedded_metadata(self):
        import h5py
        proc = _make_proc(multi=True)
        path = os.path.join(self.dir, "rec.h5")
        export_processed_h5(path, proc, params=ProcessingParams(),
                            selection=_full_selection([
                                "dFF (motion corrected with fitted ref)",
                                "dFF (non motion corrected)"]))
        with h5py.File(path, "r") as f:
            self.assertEqual(f.attrs["pyber_format_version"], "1.0")
            embedded = json.loads(f.attrs["pyber_meta_json"])
            self.assertEqual(embedded["primary_output"], "dFF")
            g = f["data"]
            self.assertIn("dFF", g)
            self.assertIn("dFF__nomc", g)
            self.assertNotIn("output", g)
            self.assertEqual(g.attrs["primary_output"], "dFF")
            self.assertEqual(g["dFF"].attrs["label"],
                             "dFF (motion corrected with fitted ref)")


class LoaderRoundTripTests(unittest.TestCase):
    """Deterministic column-role loading via the shared analysis_core reader."""

    def setUp(self):
        from analysis_core import load_processed_csv, load_processed_h5
        # Plain functions stored on the instance are not bound, so calling
        # self.load_csv(path) does not inject self.
        self.load_csv = load_processed_csv
        self.load_h5 = load_processed_h5
        self.dir = tempfile.mkdtemp(prefix="pyber_load_")

    def _rng(self, n=40):
        return np.arange(n, dtype=float)

    def test_v1_csv_and_h5_pick_primary_output(self):
        proc = _make_proc(multi=True)
        rng = self._rng()
        csv_path = os.path.join(self.dir, "v1.csv")
        h5_path = os.path.join(self.dir, "v1.h5")
        sel = _full_selection([
            "dFF (motion corrected with fitted ref)", "dFF (non motion corrected)"])
        export_processed_csv(csv_path, proc, selection=sel, params=ProcessingParams())
        export_processed_h5(h5_path, proc, selection=sel, params=ProcessingParams())

        pc = self.load_csv(csv_path)
        ph = self.load_h5(h5_path)
        for tag, p in (("csv", pc), ("h5", ph)):
            self.assertIsNotNone(p, tag)
            # Must be the fitted-ref primary (0.01*rng), NOT raw and NOT the nomc extra.
            np.testing.assert_allclose(p.output[:5], (0.01 * rng)[:5], err_msg=tag)
            np.testing.assert_allclose(p.raw_signal[:5], (100.0 + rng)[:5], err_msg=tag)
            np.testing.assert_allclose(p.raw_reference[:5], (50.0 + 0.5 * rng)[:5], err_msg=tag)
            self.assertIn("fitted ref", p.output_label, tag)
            self.assertIsNotNone(p.sync_aligned_time, tag)
            self.assertGreaterEqual(set(p.triggers.keys()), {"DIO01", "DIO02"}, tag)

    def test_legacy_csv_with_comment_lines(self):
        rng = self._rng()
        path = os.path.join(self.dir, "legacy.csv")
        with open(path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["# output_label: dFF (motion corrected with fitted ref)"])
            w.writerow(["# output_context: Fit: OLS"])
            w.writerow(["# animal_id: M9"])
            w.writerow(["time", "raw", "isobestic", "dFF", "DIO01"])
            for i in range(rng.size):
                w.writerow([i / 10.0, 100.0 + i, 50.0 + i, 0.01 * i, int(i % 10 < 2)])
        p = self.load_csv(path)
        self.assertIsNotNone(p)
        np.testing.assert_allclose(p.output[:5], (0.01 * rng)[:5])
        np.testing.assert_allclose(p.raw_signal[:5], (100.0 + rng)[:5])
        self.assertEqual(p.output_context, "Fit: OLS")
        self.assertIn("dFF", p.output_label)
        self.assertIn("DIO01", p.triggers)

    def test_legacy_h5_with_data_output(self):
        import h5py
        rng = self._rng()
        path = os.path.join(self.dir, "legacy.h5")
        with h5py.File(path, "w") as f:
            g = f.create_group("data")
            g.create_dataset("time", data=rng / 10.0)
            g.create_dataset("output", data=0.01 * rng)
            g.create_dataset("raw_465", data=100.0 + rng)
            g.create_dataset("raw_405", data=50.0 + rng)
            g.create_dataset("dio", data=(rng % 10 < 2).astype(float))
            g.attrs["output_label"] = "dFF (motion corrected with fitted ref)"
            g.attrs["dio_name"] = "DIO01"
            g.attrs["fs_used"] = 100.0
        p = self.load_h5(path)
        self.assertIsNotNone(p)
        np.testing.assert_allclose(p.output[:5], (0.01 * rng)[:5])
        np.testing.assert_allclose(p.raw_signal[:5], (100.0 + rng)[:5])
        self.assertIn("fitted ref", p.output_label)

    def test_signal_output_not_shadowed_by_raw(self):
        # A v1.0 export whose primary output is the processed 465 trace must load
        # signal_465 as the output, not raw_465, thanks to the sidecar.
        rng = self._rng()
        proc = _make_proc()
        proc.output = 7.0 + rng
        proc.output_label = "Raw signal (465)"
        proc.outputs = {"Raw signal (465)": 7.0 + rng}
        path = os.path.join(self.dir, "sig.csv")
        export_processed_csv(path, proc, selection=_full_selection(["Raw signal (465)"]),
                             params=ProcessingParams())
        p = self.load_csv(path)
        self.assertIsNotNone(p)
        np.testing.assert_allclose(p.output[:5], (7.0 + rng)[:5])
        np.testing.assert_allclose(p.raw_signal[:5], (100.0 + rng)[:5])

    def test_aligned_export_roundtrips_time_aligned(self):
        # The postprocessing "Export aligned files" path re-exports through the
        # same writer; a re-loaded aligned file must keep time_aligned + output.
        rng = self._rng()
        proc = _make_proc()
        sel = ExportSelection(raw=True, isobestic=True, output=True, dio=True,
                              baseline_sig=True, baseline_ref=True)
        path = os.path.join(self.dir, "rec_time_aligned.csv")
        export_processed_csv(path, proc, selection=sel)
        p = self.load_csv(path)
        self.assertIsNotNone(p)
        self.assertIsNotNone(p.sync_aligned_time)
        np.testing.assert_allclose(p.output[:5], (0.01 * rng)[:5])


class DelegationTests(unittest.TestCase):
    """The GUI tabs must delegate to the shared analysis_core reader."""

    def test_gui_loaders_delegate_to_analysis_core(self):
        try:
            import PySide6  # noqa: F401
            import gui_postprocessing as gp
            import main as m
        except Exception as exc:  # pragma: no cover - Qt optional in headless CI
            self.skipTest(f"Qt/gui modules unavailable: {exc}")
        import analysis_core as core
        # Both GUI copies should be thin wrappers over the shared functions.
        for cls in (gp.PostProcessingPanel,):
            src = cls._load_processed_csv.__code__.co_names
            self.assertIn("load_processed_csv", src)
        self.assertTrue(hasattr(core, "load_processed_csv"))
        self.assertTrue(hasattr(core, "load_processed_h5"))


if __name__ == "__main__":
    unittest.main()
