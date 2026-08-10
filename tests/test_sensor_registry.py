"""Tests for the sensor registry and sensor-aware preprocessing."""
import os
import sys
import tempfile
import unittest

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "pyBer"))

import analysis_core as ac  # noqa: E402
import sensor_registry as sr  # noqa: E402


def _trial(fs: float = 200.0, duration: float = 12.0) -> ac.LoadedTrial:
    t = np.arange(0.0, duration, 1.0 / fs)
    events = np.zeros_like(t)
    for center in (2.0, 5.0, 8.0):
        events += np.exp(-0.5 * ((t - center) / 0.12) ** 2)
    sig = 120.0 + 1.6 * events + 0.03 * np.sin(2.0 * np.pi * 0.3 * t)
    ref = 70.0 + 0.2 * np.sin(2.0 * np.pi * 0.3 * t)
    return ac.LoadedTrial(
        path=r"C:\data\sensor_demo.csv",
        channel_id="CH1",
        time=t,
        signal_465=sig,
        reference_405=ref,
        sampling_rate=fs,
    )


class SensorRegistryTests(unittest.TestCase):
    def test_registry_has_unique_ids_and_core_families(self):
        sensors = sr.all_sensors()
        ids = [sensor.sensor_id for sensor in sensors]
        self.assertEqual(len(ids), len(set(ids)))
        names = {sensor.name for sensor in sensors}
        for expected in ("GCaMP6f", "jGCaMP8f", "dLight1.2", "GRAB-5HT1.0", "GRAB-ACh3.0"):
            self.assertIn(expected, names)
        families = {sensor.family for sensor in sensors}
        for expected in ("Calcium", "Dopamine", "Serotonin", "Norepinephrine", "Glutamate"):
            self.assertIn(expected, families)

    def test_registry_has_precise_kinetic_values_and_context(self):
        j8f = sr.get_sensor("jgcamp8f")
        self.assertIn("7.1", j8f.rise)
        self.assertIn("67.4", j8f.decay)
        self.assertIn("cultured neuron", j8f.kinetics_context)

        dlight = sr.get_sensor("dlight12")
        self.assertIn("10 ms", dlight.rise)
        self.assertIn("100 ms", dlight.decay)
        self.assertIn("sensor speed", dlight.kinetics_context)

        ach = sr.get_sensor("grabach30")
        self.assertIn("0.09", ach.rise)
        self.assertIn("0.91", ach.decay)
        self.assertIn("sensor characterization", ach.kinetics_context)

    def test_trace_check_detects_sensor_polarity_mismatch(self):
        t = np.arange(0.0, 20.0, 0.02)
        upward = np.sin(2.0 * np.pi * 0.2 * t)
        upward += 4.0 * np.exp(-0.5 * ((t - 8.0) / 0.15) ** 2)

        gcamp = sr.assess_sensor_trace("gcamp6f", t, upward)
        dark = sr.assess_sensor_trace("sdarken", t, upward)

        self.assertEqual(gcamp["status"], "ok")
        self.assertEqual(dark["status"], "warn")
        self.assertIn("expected to report downward", dark["message"])

    def test_recommendation_uses_sensor_bandwidth(self):
        fast_params = ac.ProcessingParams()
        fast_params.sensor_id = "iglusnfr3"
        fast = ac.recommend_preprocessing_settings(_trial(fs=250.0), fast_params)

        slow_params = ac.ProcessingParams()
        slow_params.sensor_id = "gcamp6s"
        slow = ac.recommend_preprocessing_settings(_trial(fs=250.0), slow_params)

        self.assertGreater(fast.params.target_fs_hz, slow.params.target_fs_hz)
        self.assertGreater(fast.params.lowpass_hz, slow.params.lowpass_hz)
        self.assertIn("iGluSnFR3", fast.sections["sensor"])
        self.assertIn("GCaMP6s", slow.sections["filtering"])

    def test_darkening_sensor_recommends_polarity_inversion_on_upward_trace(self):
        params = ac.ProcessingParams()
        params.sensor_id = "sdarken"
        rec = ac.recommend_preprocessing_settings(_trial(fs=80.0), params)

        self.assertTrue(rec.params.invert_polarity)
        self.assertIn("expected to report downward", rec.sections["sensor"])
        self.assertTrue(any("sDarken" in warning for warning in rec.warnings))

    def test_processing_and_export_record_sensor_metadata(self):
        trial = _trial(fs=80.0)
        params = ac.ProcessingParams()
        params.sensor_id = "dlight12"
        params.artifact_detection_enabled = False
        params.target_fs_hz = 40.0
        params.lowpass_hz = 8.0
        params.baseline_max_iter = 10

        processed = ac.PhotometryProcessor().process_trial(trial, params)
        self.assertEqual(processed.sensor_label, "dLight1.2")
        self.assertIn("Sensor: dLight1.2", processed.output_context)

        with tempfile.TemporaryDirectory(prefix="pyber_sensor_") as tmp:
            path = os.path.join(tmp, "sensor.csv")
            ac.export_processed_csv(path, processed, params=params)
            sidecar = ac.read_processed_sidecar(path)
        self.assertIsNotNone(sidecar)
        self.assertEqual(sidecar["sensor"]["id"], "dlight12")
        self.assertEqual(sidecar["sensor"]["name"], "dLight1.2")
        self.assertIn("10 ms", sidecar["sensor"]["rise"])
        self.assertIn("kinetics_context", sidecar["sensor"])
        self.assertIn("trace_check", sidecar["sensor"])


if __name__ == "__main__":
    unittest.main()
