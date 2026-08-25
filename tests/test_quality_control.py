"""Regression tests for recording-level fiber-photometry QC."""

import os
import sys
import unittest

import numpy as np
from scipy.signal import lfilter


os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("PYBER_SMOKE_TEST", "1")

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "pyBer"))

from analysis_core import LoadedTrial  # noqa: E402
from main import MainWindow, _evaluate_qc, _qc_detect_bad_segments  # noqa: E402


def _labeled_synthetic_trials():
    """Return deterministic poor-common-mode and good-event-rich recordings."""
    rng = np.random.default_rng(4)
    fs = 20.0
    t = np.arange(0.0, 300.0, 1.0 / fs)
    common = (
        -0.05 * (1.0 - np.exp(-t / 180.0))
        + 0.006 * np.sin(2.0 * np.pi * 0.17 * t)
        + 0.003 * np.sin(2.0 * np.pi * 0.035 * t)
        + rng.normal(0.0, 0.0008, t.size)
    )
    reference = 0.085 * (1.0 + common)

    # Almost everything in 465 is a scaled copy of 405. Reference fitting
    # leaves no event-like sensor structure.
    poor_signal = 0.125 * (
        1.0 + 0.90 * common + rng.normal(0.0, 0.00045, t.size)
    )

    # Independent positive transients survive reference fitting.
    impulses = np.zeros(t.size, dtype=float)
    for event_t in np.arange(5.0, 298.0, 4.0):
        impulses[int(event_t * fs)] = rng.uniform(0.004, 0.018)
    activity = lfilter([1.0], [1.0, -np.exp(-1.0 / (fs * 0.8))], impulses)
    good_signal = 0.125 * (
        1.0 + 0.25 * common + activity + rng.normal(0.0, 0.0005, t.size)
    )

    def trial(name, signal):
        return LoadedTrial(
            path=name,
            channel_id="AIN01",
            time=t,
            signal_465=signal,
            reference_405=reference,
            sampling_rate=fs,
        )

    return trial("known_poor.csv", poor_signal), trial("known_good.csv", good_signal)


class QualityControlTests(unittest.TestCase):
    def test_common_mode_recording_is_excluded_but_event_rich_recording_is_kept(self):
        poor_trial, good_trial = _labeled_synthetic_trials()

        poor_qc = MainWindow._compute_qc(None, poor_trial)
        good_qc = MainWindow._compute_qc(None, good_trial)
        poor_verdict = _evaluate_qc(poor_qc)
        good_verdict = _evaluate_qc(good_qc)

        self.assertEqual(poor_verdict.tier, "POOR")
        self.assertEqual(poor_verdict.action_kind, "EXCLUDE")
        self.assertLess(poor_qc["signal_retention"], 0.50)

        self.assertEqual(good_verdict.tier, "GOOD")
        self.assertEqual(good_verdict.action_kind, "KEEP")
        self.assertGreater(good_qc["signal_retention"], 0.70)
        self.assertGreater(good_qc["usable_snr"], 6.0)

    def test_usable_snr_is_measured_after_reference_fitting(self):
        poor_trial, _good_trial = _labeled_synthetic_trials()
        qc = MainWindow._compute_qc(None, poor_trial)

        self.assertIn("corrected_dff_pct", qc)
        self.assertIn("corrected_noise_pct", qc)
        self.assertLess(qc["event_amp_pct"], qc["raw_event_amp_pct"])
        self.assertLess(qc["usable_snr"], 3.0)

    def test_mild_baseline_adaptation_is_not_called_bleach_in(self):
        fs = 10.0
        t = np.arange(0.0, 600.0, 1.0 / fs)

        def qc_for_drop(fraction):
            baseline = 1.0 - fraction * (1.0 - np.exp(-t / 20.0))
            return {
                "t": t,
                "sig_base": baseline,
                "fs": fs,
                "has_reference": False,
                "art_mask": np.zeros(t.size, dtype=bool),
                "hf_sig_pct": np.sin(t) * 0.01,
            }

        mild = _qc_detect_bad_segments(qc_for_drop(0.04))
        strong = _qc_detect_bad_segments(qc_for_drop(0.12))

        self.assertNotIn("bleach_in", {segment["kind"] for segment in mild})
        self.assertIn("bleach_in", {segment["kind"] for segment in strong})


if __name__ == "__main__":
    unittest.main()
