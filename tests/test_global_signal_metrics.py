"""Numerical contracts for global recording summaries and removed intervals."""

import unittest

import numpy as np

from pyBer.global_signal_metrics import GLOBAL_SIGNAL_METRICS, compute_global_signal_metrics


class GlobalSignalMetricsTests(unittest.TestCase):
    def test_distribution_and_integral_match_known_signal(self):
        result = compute_global_signal_metrics(np.arange(5.), np.arange(5.))
        self.assertEqual(result["median"], 2.)
        self.assertEqual(result["iqr"], 2.)
        self.assertAlmostEqual(result["std"], np.sqrt(2.5))
        self.assertAlmostEqual(result["rms"], np.sqrt(6.))
        self.assertAlmostEqual(result["dynamic_range"], 3.6)
        self.assertEqual(result["auc"], 8.)
        self.assertEqual(result["duration"], 4.)
        self.assertEqual(len(GLOBAL_SIGNAL_METRICS), 10)
        self.assertTrue(set(GLOBAL_SIGNAL_METRICS).issubset(result))

    def test_legacy_peak_detector_unchanged_on_complete_trace(self):
        y = np.array([0., 4., 0., 0., 6., 0., 0., 8., 0.])
        result = compute_global_signal_metrics(np.arange(y.size), y)
        self.assertEqual(result["amp"], 6.)
        self.assertEqual(result["peaks"], 3.)
        self.assertEqual(result["freq"], 3. / 8.)
        self.assertEqual(result["ibi"], 3.)
        self.assertEqual(result["thr"], 0.)

    def test_legacy_threshold_parity_across_offsets_and_distributions(self):
        """A new menu must not silently alter existing complete-trace results."""
        rng = np.random.default_rng(203)
        for offset in (-2., 0., 2.):
            for y in (rng.normal(size=1000) + offset, rng.exponential(size=1000) + offset):
                median = np.median(y)
                local = np.flatnonzero((y[1:-1] > y[:-2]) & (y[1:-1] > y[2:])) + 1
                high = local[y[local] > median + 2. * np.median(np.abs(y - median))]
                retained = np.ones(y.size, dtype=bool)
                retained[high] = False
                threshold = 3. * np.median(y[retained])
                detected = local[y[local] >= threshold]
                result = compute_global_signal_metrics(np.arange(y.size) * .1, y)
                self.assertEqual(result["thr"], threshold)
                self.assertEqual(result["peaks"], detected.size)
                self.assertAlmostEqual(result["amp"], np.mean(y[detected]) if detected.size else 0.)

    def test_nan_cut_never_creates_peak_or_bridges_integral(self):
        t = np.arange(7.)
        y = np.array([0., 0., 5., np.nan, 0., 0., 0.])
        result = compute_global_signal_metrics(t, y)
        self.assertEqual(result["peaks"], 0.)
        self.assertEqual(result["duration"], 4.)
        self.assertEqual(result["auc"], 2.5)
        self.assertTrue(np.isnan(result["ibi"]))
        self.assertTrue(np.isnan(y[3]))

    def test_clock_gap_excluded_from_frequency_and_intervals(self):
        t = np.array([0., 1., 2., 100., 101., 102.])
        y = np.array([0., 3., 0., 0., 5., 0.])
        result = compute_global_signal_metrics(t, y)
        self.assertEqual(result["duration"], 4.)
        self.assertEqual(result["freq"], .5)
        self.assertEqual(result["auc"], 8.)
        self.assertTrue(np.isnan(result["ibi"]))

    def test_selected_range_keeps_native_gap_detection(self):
        t = np.array([0., 1., 2., 3., 100., 101., 102., 103.])
        result = compute_global_signal_metrics(t, np.ones(8), 3., 101.)
        self.assertEqual(result["duration"], 1.)
        self.assertEqual(result["auc"], 1.)

    def test_invalid_and_empty_inputs(self):
        self.assertIsNone(compute_global_signal_metrics([], []))
        self.assertIsNone(compute_global_signal_metrics([0, 1, 2], [np.nan] * 3))
        with self.assertRaises(ValueError):
            compute_global_signal_metrics([0, 1, 1], [0, 1, 2])
        with self.assertRaises(ValueError):
            compute_global_signal_metrics([0, 1], [0])


if __name__ == "__main__":
    unittest.main()
