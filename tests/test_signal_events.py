"""Synthetic regression checks for gap-safe signal event analysis."""
import numpy as np
import unittest

from pyBer.signal_events import detect_peaks, estimate_noise, observed_intervals, preprocess_trace


class SignalEventsTests(unittest.TestCase):
    """Run numerical regressions without optional test-runner dependencies."""
    def test_preprocessing_never_bridges_nan_or_timestamp_cuts(self):
        """Independent constant segments cannot contaminate each other when filtered."""
        t = np.r_[np.arange(20), np.arange(40, 60)].astype(float)
        y = np.r_[np.zeros(20), np.ones(20) * 100]
        y[10] = np.nan
        original = y.copy()
        _, processed, trace = preprocess_trace(t, y, smooth_sigma_sec=3)
        np.testing.assert_equal(y, original)
        np.testing.assert_equal(trace, original)
        np.testing.assert_allclose(processed[:10], 0)
        np.testing.assert_allclose(processed[20:], 100)
        assert np.isnan(processed[10])
        np.testing.assert_equal(observed_intervals(t, y), [[0, 9], [11, 19], [40, 59]])


    def test_detection_width_and_auc_do_not_cross_gap(self):
        """AUC stops before a large value beyond a removed interval."""
        t = np.r_[np.arange(5), np.arange(20, 25)].astype(float)
        y = np.array([0, 1, 3, 1, 0, 100, 101, 104, 101, 100.])
        result = detect_peaks(t, y, prominence=1, auc_half_window_sec=2)
        np.testing.assert_equal(result["indices"], [2, 7])
        np.testing.assert_allclose(result["auc"], [5, 406])
        np.testing.assert_allclose(result["widths_sec"], [1.5, 4 / 3])
        assert result["duration_s"] == 8
        assert result["inter_peak_intervals_sec"].size == 0
        assert np.all(np.isnan(detect_peaks(t, y, 1, auc_half_window_sec=100)["auc"]))


    def test_distance_and_width_use_actual_timestamps(self):
        """Irregular sampling must not turn seconds into a global sample-count rule."""
        t = np.array([0, 1, 2, 2.1, 2.2, 3.2, 4.2])
        y = np.array([0, 3, 0, 4, 0, 0, 0.])
        result = detect_peaks(t, y, prominence=1, min_distance_sec=1.2)
        np.testing.assert_equal(result["indices"], [3])
        np.testing.assert_allclose(result["widths_sec"], [.1])


    def test_noise_is_robust_to_slow_drift_and_transients(self):
        """Known synthetic white-noise scale survives substantial slow baseline drift."""
        rng = np.random.default_rng(381)
        t = np.arange(0, 120, .02)
        baseline = .8 * np.sin(2 * np.pi * t / 100)
        y = baseline + rng.normal(0, .03, t.size)
        for event in [20, 45, 70, 95]:
            y += .4 * np.exp(-.5 * ((t - event) / .15) ** 2)
        stats = estimate_noise(t, y, baseline_window_sec=5)
        assert .024 < stats["noise_sigma"] < .038
        assert np.median(np.abs(stats["baseline"] - baseline)) < .02
        assert np.std(y) > 8 * stats["noise_sigma"]


    def test_noise_reports_unavailable_selection_and_preserves_cuts(self):
        """Empty baseline selections and constant traces cannot invent a noise scale."""
        t = np.arange(12.)
        y = np.ones(12)
        y[6] = np.nan
        assert np.isnan(estimate_noise(t, y)["noise_sigma"])
        stats = estimate_noise(t, y, baseline_mask=np.zeros(12, bool))
        assert np.isnan(stats["noise_sigma"])
        assert stats["n_samples"] == 0
        assert np.isnan(stats["baseline"][6])


    def test_empty_trace_and_invalid_shape(self):
        """Empty recordings are valid; misaligned arrays fail explicitly."""
        assert observed_intervals([], []).shape == (0, 2)
        assert detect_peaks([], [], 1)["indices"].size == 0
        with self.assertRaises(ValueError):
            preprocess_trace([0, 1], [1])


    def test_local_height_gate_uses_original_indices_across_cuts(self):
        """Local baseline-relative gates reject small peaks despite absolute drift."""
        t = np.arange(11.)
        y = np.array([0, 2, 0, np.nan, 10, 12, 10, 10, 14, 10, 10])
        gate = np.array([3, 3, 3, np.nan, 13, 13, 13, 13, 13, 13, 13])
        np.testing.assert_equal(detect_peaks(t, y, 1, min_height=gate)["indices"], [8])
        np.testing.assert_equal(detect_peaks(t, y, 1, min_height=0)["indices"], [1, 5, 8])
        with self.assertRaises(ValueError):
            detect_peaks(t, y, 1, min_height=[1, 2])

    def test_empty_and_all_nonfinite_arrays(self):
        """Empty arrays and entirely invalid recordings consistently have no events."""
        for t, y in [([], []), ([0, 1], [np.nan, np.inf]), ([np.nan, np.inf], [0, 1])]:
            self.assertEqual(observed_intervals(t, y).shape, (0, 2))
            result = detect_peaks(t, y, 1)
            self.assertEqual(result['indices'].size, 0)
            self.assertEqual(result['duration_s'], 0)
            noise = estimate_noise(t, y)
            self.assertTrue(np.isnan(noise['noise_sigma']))
            self.assertEqual(noise['n_samples'], 0)
            processed = preprocess_trace(t, y)[1]
            self.assertTrue(np.all(np.isnan(processed)))

    def test_auc_includes_exact_subsample_window_endpoints(self):
        """A triangle integrated over +/-0.5 s has analytically known area 0.75."""
        result = detect_peaks([0, 1, 2], [0, 1, 0], .1, auc_half_window_sec=.5)
        np.testing.assert_allclose(result["auc"], [.75])


if __name__ == '__main__':
    unittest.main()
