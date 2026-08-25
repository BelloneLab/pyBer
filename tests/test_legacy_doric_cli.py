import argparse
import os
import sys
import tempfile
import unittest
from pathlib import Path

import h5py
import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "pyBer"))

from analysis_core import PhotometryProcessor, ProcessingParams  # noqa: E402
from cli import apply_overrides, discover_raw_files, evaluate_qc, load_override_file, resolve_sensor  # noqa: E402


class LegacyDoricImportTests(unittest.TestCase):
    def test_legacy_per_channel_lockin_groups_load(self):
        with tempfile.TemporaryDirectory(prefix="pyber_legacy_doric_") as tmp:
            path = os.path.join(tmp, "legacy.doric")
            time = np.arange(200, dtype=float) / 20.0
            with h5py.File(path, "w") as handle:
                base = handle.create_group("DataAcquisition/FPConsole/Signals/Series0001")
                for output, values in ((1, 2.0 + 0.1 * np.sin(time)), (2, 4.0 + np.sin(time))):
                    group = base.create_group(f"AIN01xAOUT0{output}-LockIn")
                    group.create_dataset("Time", data=time)
                    group.create_dataset("Values", data=values)
                digital = base.create_group("DigitalIO")
                digital.create_dataset("Time", data=time)
                digital.create_dataset("DIO02", data=(time > 3.0).astype(float))

            loaded = PhotometryProcessor().load_file(path)
            self.assertEqual(loaded.channels, ["AIN01"])
            trial = loaded.make_trial("AIN01", trigger_name="DIO02")
            np.testing.assert_allclose(trial.time, time)
            np.testing.assert_allclose(trial.signal_465, 4.0 + np.sin(time))
            np.testing.assert_allclose(trial.reference_405, 2.0 + 0.1 * np.sin(time))
            self.assertAlmostEqual(trial.sampling_rate, 20.0)
            self.assertIn("DIO02", loaded.trigger_by_name)

    def test_series_group_can_be_nested_below_nonstandard_keys(self):
        with tempfile.TemporaryDirectory(prefix="pyber_nested_series_") as tmp:
            path = os.path.join(tmp, "nested.doric")
            time = np.arange(30, dtype=float) / 10.0
            with h5py.File(path, "w") as handle:
                base = handle.create_group("Acquisition/Signals/Series42")
                signal = base.create_group("LockInAOUT02")
                signal.create_dataset("Time", data=time)
                signal.create_dataset("AIN03", data=np.arange(30, dtype=float))
            loaded = PhotometryProcessor().load_file(path)
            self.assertEqual(loaded.channels, ["AIN03"])
            self.assertTrue(np.isnan(loaded.reference_by_channel["AIN03"]).all())


class CliTests(unittest.TestCase):
    def test_recursive_discovery_finds_deep_raw_files_and_skips_events(self):
        with tempfile.TemporaryDirectory(prefix="pyber_cli_discovery_") as tmp:
            root = Path(tmp)
            deep = root / "animal" / "day" / "run"
            deep.mkdir(parents=True)
            (deep / "recording.doric").write_bytes(b"placeholder")
            (deep / "Fluorescence.csv").write_text("placeholder\n", encoding="utf-8")
            (deep / "Events.csv").write_text("TimeStamp,Name,State\n", encoding="utf-8")
            paths = discover_raw_files([tmp])
            names = sorted(path.name for path in paths)
            self.assertEqual(names, ["Fluorescence.csv", "recording.doric"])

    def test_sensor_and_parameter_overrides(self):
        self.assertEqual(resolve_sensor("GCaMP6f"), "gcamp6f")
        params = apply_overrides(
            ProcessingParams(),
            ["lowpass_hz=5.5", "filter_order=4", "smoothing_enabled=true"],
        )
        self.assertEqual(params.lowpass_hz, 5.5)
        self.assertEqual(params.filter_order, 4)
        self.assertTrue(params.smoothing_enabled)

    def test_parameter_override_json(self):
        with tempfile.TemporaryDirectory(prefix="pyber_cli_params_") as tmp:
            path = Path(tmp) / "params.json"
            path.write_text('{"lowpass_hz": 7.5, "artifact_detection_enabled": false}', encoding="utf-8")
            params = apply_overrides(ProcessingParams(), load_override_file(str(path)))
            self.assertEqual(params.lowpass_hz, 7.5)
            self.assertFalse(params.artifact_detection_enabled)

    def test_qc_flags_high_artifact_recommendation(self):
        processed = argparse.Namespace(
            output=np.ones(100), raw_signal=np.linspace(0, 1, 100), sensor_check={"status": "ok"}
        )
        recommendation = argparse.Namespace(
            metrics={"artifact_fraction": 0.12}, confidence=0.9, warnings=[]
        )
        qc = evaluate_qc(processed, recommendation)
        self.assertEqual(qc["tier"], "FAIL")
        self.assertTrue(qc["flagged"])

    def test_cli_qc_rejects_common_mode_signal_removed_by_reference_fit(self):
        fs = 20.0
        time = np.arange(0.0, 180.0, 1.0 / fs)
        common = 0.012 * np.sin(2.0 * np.pi * 0.18 * time)
        reference = 0.08 * (1.0 + common)
        signal = 0.12 * (1.0 + 0.92 * common + 0.0002 * np.sin(2.0 * np.pi * 1.3 * time))
        processed = argparse.Namespace(
            time=time,
            output=np.zeros(time.size),
            raw_signal=signal,
            raw_reference=reference,
            sig_f=signal,
            ref_f=reference,
            baseline_sig=np.full(time.size, 0.12),
            baseline_ref=np.full(time.size, 0.08),
            sensor_check={"status": "ok"},
        )
        recommendation = argparse.Namespace(
            metrics={"artifact_fraction": 0.0}, confidence=0.9, warnings=[]
        )

        qc = evaluate_qc(processed, recommendation)

        self.assertEqual(qc["tier"], "FAIL")
        self.assertLess(qc["signal_retention"], 0.50)
        self.assertTrue(any("reference fitting" in reason for reason in qc["reasons"]))


if __name__ == "__main__":
    unittest.main()
