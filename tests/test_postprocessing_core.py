"""Regression checks for event identity, interpolation, and uncertainty."""

import os
import sys
import unittest
import warnings

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "pyBer"))

from postprocessing_core import (  # noqa: E402
    compute_psth_matrix, extract_complete_events, group_close_events,
    mean_sem, normalize_events, paired_summary, window_metrics,
)


class EventIntegrityTests(unittest.TestCase):
    """Onsets, offsets and durations must always describe the same event."""

    def test_sort_and_dedup_preserve_event_rows(self):
        on, off, dur = normalize_events([4, np.nan, 1, 4], [7, 99, 2, 8], [3, 5, 1, 4])
        np.testing.assert_array_equal(on, [1, 4])
        np.testing.assert_array_equal(off, [2, 7])
        np.testing.assert_array_equal(dur, [1, 3])

    def test_missing_offset_does_not_shift_following_rows(self):
        on, off, dur = normalize_events([1, 4], [np.nan, 7], [np.nan, 3])
        np.testing.assert_array_equal(on, [1, 4])
        np.testing.assert_allclose(off, [np.nan, 7], equal_nan=True)
        np.testing.assert_allclose(dur, [np.nan, 3], equal_nan=True)

    def test_complete_binary_event_uses_first_low_sample(self):
        on, off, dur = extract_complete_events(np.arange(8), [1, 0, 1, 1, 0, 0, 1, 1])
        np.testing.assert_array_equal(on, [2])
        np.testing.assert_array_equal(off, [4])
        np.testing.assert_array_equal(dur, [2])

    def test_missing_sample_does_not_invent_falling_edge(self):
        on, off, dur = extract_complete_events(np.arange(8), [0, 1, np.nan, 1, 0, 1, 0, 0])
        np.testing.assert_array_equal(on, [5])
        np.testing.assert_array_equal(off, [6])
        np.testing.assert_array_equal(dur, [1])

    def test_recording_gap_does_not_invent_complete_bout(self):
        on, off, dur = extract_complete_events([0, 1, 2, 8, 9, 10, 11], [0, 1, 1, 1, 0, 1, 0])
        np.testing.assert_array_equal(on, [10])
        np.testing.assert_array_equal(off, [11])
        np.testing.assert_array_equal(dur, [1])

    def test_offset_grouping_uses_latest_offset_and_earliest_onset(self):
        times, durations = group_close_events([5, 4, 10], [2, 3, 1], 1.1, alignment="offset")
        np.testing.assert_array_equal(times, [5, 10])
        np.testing.assert_array_equal(durations, [4, 1])

    def test_onset_grouping_preserves_longest_overlapping_bout(self):
        times, durations = group_close_events([1, 2, 8], [5, 1, 2], 1.1)
        np.testing.assert_array_equal(times, [1, 8])
        np.testing.assert_array_equal(durations, [5, 2])

    def test_unknown_group_duration_remains_unknown(self):
        _, durations = group_close_events([1, 2], [np.nan, 3], 2)
        self.assertTrue(np.isnan(durations[0]))


class PsthNumericsTests(unittest.TestCase):
    """Synthetic analytical traces expose padding and normalization mistakes."""

    def setUp(self):
        self.time = np.arange(0, 10.01, 0.1)
        self.signal = self.time.copy()

    def test_linear_trace_recovers_analytical_baseline_zscore(self):
        rel, mat = compute_psth_matrix(self.time, self.signal, [5], (-2, 2), (-2, -1), 10)
        baseline = self.signal[(self.time >= 3) & (self.time <= 4)]
        np.testing.assert_allclose(mat[0], ((5 + rel) - baseline.mean()) / baseline.std(), atol=1e-12)

    def test_recording_edges_are_missing_instead_of_constant_padding(self):
        rel, mat = compute_psth_matrix(self.time, self.signal, [9], (-2, 3), (-2, -1), 10)
        self.assertTrue(np.isnan(mat[0, rel > 1.01]).all())
        self.assertTrue(np.isfinite(mat[0, rel < 0.99]).all())

    def test_nan_gap_remains_missing_after_smoothing(self):
        self.signal[55:60] = np.nan
        rel, raw = compute_psth_matrix(self.time, self.signal, [5], (-2, 2), (-2, -1), 10)
        _, smooth = compute_psth_matrix(self.time, self.signal, [5], (-2, 2), (-2, -1), 10, 0.2)
        np.testing.assert_array_equal(np.isnan(raw), np.isnan(smooth))
        self.assertTrue(np.isnan(smooth[0, (rel >= 0.5) & (rel <= 0.8)]).all())
        self.assertTrue(np.isfinite(smooth[0, rel < 0.3]).all())

    def test_timestamp_gap_is_not_interpolated(self):
        keep = (self.time < 5.5) | (self.time > 6.5)
        rel, mat = compute_psth_matrix(self.time[keep], self.signal[keep], [5], (-2, 2), (-2, -1), 10)
        self.assertTrue(np.isnan(mat[0, (rel > 0.5) & (rel < 1.5)]).all())

    def test_five_rows_but_only_one_valid_baseline_sample_is_rejected(self):
        self.signal[30:40] = np.nan
        _, mat = compute_psth_matrix(self.time, self.signal, [5], (-2, 2), (-2, -1), 10)
        self.assertTrue(np.isnan(mat).all())

    def test_constant_baseline_requires_explicit_subtraction(self):
        flat = np.ones(self.time.shape) * 4
        _, zscore = compute_psth_matrix(self.time, flat, [5], (-2, 2), (-2, -1), 10)
        _, subtract = compute_psth_matrix(self.time, flat, [5], (-2, 2), (-2, -1), 10, normalization="subtract")
        self.assertTrue(np.isnan(zscore).all())
        np.testing.assert_array_equal(subtract, np.zeros(subtract.shape))

    def test_no_normalization_does_not_require_baseline_coverage(self):
        rel, mat = compute_psth_matrix(self.time, self.signal, [1], (-1, 1), (-10, -9), 10, normalization="none")
        np.testing.assert_allclose(mat[0], 1 + rel)

    def test_invalid_times_fail_with_actionable_message(self):
        for time in ([0, 2, 1], [0, 1, 1], [0, np.nan, 2]):
            with self.subTest(time=time):
                with self.assertRaisesRegex(ValueError, "strictly increasing"):
                    compute_psth_matrix(time, [1, 2, 3], [1], (-1, 1), (-1, 0), 10)

    def test_time_grid_stays_inside_requested_window(self):
        rel, _ = compute_psth_matrix(self.time, self.signal, [5], (-1, 1.06), (-2, -1), 10)
        self.assertLessEqual(rel[-1], 1.06)
        np.testing.assert_allclose(np.diff(rel), 0.1)


class SummaryAndMetricsTests(unittest.TestCase):
    """Sample uncertainty and time integration agree with analytical results."""

    def test_sem_uses_finite_count_at_each_time_and_sample_variance(self):
        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            avg, sem, n = mean_sem([[1, 1, 5, np.nan], [3, np.nan, np.nan, np.nan], [5, 5, np.nan, np.nan]])
        np.testing.assert_allclose(avg, [3, 3, 5, np.nan], equal_nan=True)
        np.testing.assert_allclose(sem, [2 / np.sqrt(3), 2, np.nan, np.nan], equal_nan=True)
        np.testing.assert_array_equal(n, [3, 2, 1, 0])

    def test_auc_integrates_exact_window_on_irregular_time_grid(self):
        time = np.array([0, 0.1, 0.7, 1.5, 2.0])
        result = window_metrics(np.array([2 * time + 1]), time, 0.2, 1.8, "auc")
        # Integral of 2t + 1 is t^2 + t, evaluated at the requested boundaries.
        np.testing.assert_allclose(result, [(1.8 ** 2 + 1.8) - (0.2 ** 2 + 0.2)])

    def test_auc_does_not_treat_missing_coverage_as_zero(self):
        result = window_metrics([[1, np.nan, 1]], [0, 1, 2], 0, 2, "auc")
        self.assertTrue(np.isnan(result[0]))
        outside = window_metrics([[1, 1, 1]], [0, 1, 2], -1, 2, "auc")
        self.assertTrue(np.isnan(outside[0]))

    def test_mean_metric_uses_available_finite_samples(self):
        result = window_metrics([[1, np.nan, 3], [np.nan, np.nan, np.nan]], [0, 1, 2], 0, 2)
        np.testing.assert_allclose(result, [2, np.nan], equal_nan=True)

    def test_constant_nonzero_differences_do_not_fabricate_zero_pvalue(self):
        result = paired_summary([1, 2, 3], [2, 3, 4])
        self.assertEqual(result["paired_p"], 0.25)
        self.assertEqual(result["paired_nonzero_n"], 3)
        self.assertIn("Fewer than six", result["assumption_note"])

    def test_sign_test_ignores_ties_and_missing_pairs(self):
        result = paired_summary([0, 0, 0, 0, np.nan], [1, 1, -1, 0, 100])
        self.assertEqual(result["paired_n"], 4)
        self.assertEqual(result["paired_nonzero_n"], 3)
        self.assertEqual(result["paired_ties_n"], 1)
        self.assertEqual(result["paired_p"], 1.0)

    def test_all_ties_provide_no_directional_evidence(self):
        result = paired_summary([1, 2, 3], [1, 2, 3])
        self.assertEqual(result["paired_p"], 1.0)
        self.assertEqual(result["paired_nonzero_n"], 0)

    def test_six_consistent_signs_have_exact_binomial_probability(self):
        result = paired_summary(np.zeros(6), np.arange(1, 7))
        self.assertEqual(result["paired_p"], 2 / (2 ** 6))

    def test_grouped_trials_cannot_be_treated_as_independent_animals(self):
        result = paired_summary(np.zeros(100), np.ones(100), independent_units=False)
        self.assertTrue(np.isnan(result["paired_p"]))
        self.assertEqual(result["method"], "Descriptive only")
        self.assertIn("pooled trials", result["assumption_note"])

    def test_empty_pairs_leave_inference_undefined(self):
        result = paired_summary([np.nan], [1])
        self.assertEqual(result["paired_n"], 0)
        self.assertTrue(np.isnan(result["paired_p"]))


if __name__ == "__main__":
    unittest.main()
