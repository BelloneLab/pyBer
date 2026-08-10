"""Tests for smart multi-evidence artifact detection."""
import os
import sys
import unittest

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "pyBer"))

import analysis_core as ac  # noqa: E402


def _base_trace(fs=40.0, duration=120.0):
    t = np.arange(0.0, duration, 1.0 / fs)
    signal = 250.0 + 2.0 * np.sin(2.0 * np.pi * 0.20 * t)
    reference = 80.0 + 0.5 * np.sin(2.0 * np.pi * 0.20 * t)
    return t, signal, reference, fs


class SmartArtifactDetectionTests(unittest.TestCase):
    def test_smooth_signal_only_transient_is_preserved(self):
        t, signal, reference, fs = _base_trace()
        signal = signal + 20.0 * np.exp(-0.5 * ((t - 30.0) / 1.5) ** 2)

        result = ac.detect_artifacts_smart(t, signal, reference, k=7.0, window_s=5.0, pad_s=0.25, fs=fs)

        self.assertEqual(result.regions, [])
        self.assertFalse(np.any(result.mask))

    def test_shared_motion_spike_is_detected_with_evidence_label(self):
        t, signal, reference, fs = _base_trace()
        idx = int(np.argmin(np.abs(t - 45.0)))
        signal[idx:idx + 2] += 80.0
        reference[idx:idx + 2] += 35.0

        result = ac.detect_artifacts_smart(t, signal, reference, k=7.0, window_s=5.0, pad_s=0.25, fs=fs)

        self.assertTrue(any(a <= 45.0 <= b for a, b in result.regions))
        self.assertTrue(any("shared" in src for src in result.region_sources))
        self.assertTrue(any("465:" in src and "405:" in src for src in result.region_sources))

    def test_reference_only_spike_is_detected(self):
        t, signal, reference, fs = _base_trace()
        idx = int(np.argmin(np.abs(t - 70.0)))
        reference[idx] -= 40.0

        result = ac.detect_artifacts_smart(t, signal, reference, k=7.0, window_s=5.0, pad_s=0.25, fs=fs)

        self.assertTrue(any(a <= 70.0 <= b for a, b in result.regions))
        self.assertTrue(any("405:" in src for src in result.region_sources))

    def test_processing_pipeline_uses_smart_artifacts(self):
        t, signal, reference, fs = _base_trace()
        idx = int(np.argmin(np.abs(t - 52.0)))
        signal[idx] += 100.0
        reference[idx] += 60.0
        trial = ac.LoadedTrial(
            path=r"C:\data\artifact.csv",
            channel_id="CH1",
            time=t,
            signal_465=signal,
            reference_405=reference,
            sampling_rate=fs,
        )
        params = ac.ProcessingParams()
        params.artifact_mode = ac.SMART_ARTIFACT_MODE
        params.mad_k = 7.0
        params.adaptive_window_s = 5.0
        params.artifact_pad_s = 0.25
        params.target_fs_hz = fs
        params.lowpass_hz = 8.0

        processed = ac.PhotometryProcessor().process_trial(trial, params, preview_mode=False)

        self.assertTrue(any(a <= 52.0 <= b for a, b in (processed.artifact_regions_auto_sec or [])))
        self.assertTrue(any("shared" in src for src in (processed.artifact_regions_auto_source or [])))
        self.assertIn("smart artifacts", processed.output_context)


if __name__ == "__main__":
    unittest.main()
