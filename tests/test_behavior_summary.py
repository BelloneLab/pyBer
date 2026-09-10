"""Exact interval accounting for selectable behavior plots and exports."""
import unittest
import csv
import json
from pathlib import Path
import tempfile

import numpy as np
import h5py

from pyBer.behavior_summary import summarize_behavior, export_behavior_summary


def recording(onsets, offsets, end=20., observed=None, name="a", start=0.):
    """Create a small native-time fixture with explicit observation coverage."""
    return {"file_id": name, "start": start, "end": end,
            "onsets": np.asarray(onsets, float), "offsets": np.asarray(offsets, float),
            "observed_intervals": np.asarray(observed if observed is not None else [[start, end]], float)}


class BehaviorSummaryTests(unittest.TestCase):
    def test_histogram_median_and_iqr_are_unbinned_and_bin_independent(self):
        source = recording([1, 5, 10], [2, 7, 14])
        for width in (.2, 1, 10):
            result = summarize_behavior([source], "duration", bin_s=width, auto_bins=False)
            self.assertEqual(result["statistics"]["median"], 2)
            self.assertEqual(result["statistics"]["q1"], 1.5)
            self.assertEqual(result["statistics"]["q3"], 3)
            self.assertEqual(result["statistics"]["n"], 3)

    def test_occupancy_integrates_union_and_divides_observed_exposure(self):
        source = recording([2, 4, 12], [7, 9, 14], observed=[[0, 5], [10, 20]])
        result = summarize_behavior([source], "occupancy", bin_s=5)
        np.testing.assert_allclose(result["values"], [60, np.nan, 40, 0])
        self.assertAlmostEqual(result["statistics"]["median"], 100 * 5 / 15)
        self.assertTrue(np.all(result["values"][np.isfinite(result["values"])] <= 100))
        point_only = recording([1], [np.nan])
        self.assertFalse(summarize_behavior([point_only], "occupancy")["has_data"])

    def test_cumulative_count_supports_point_events_and_fixed_group_cohort(self):
        first = recording([1, 4], [np.nan, np.nan], end=5)
        second = recording([12, 18], [np.nan, np.nan], name="b")
        result = summarize_behavior([first, second], "cumulative_count", bin_s=5)
        np.testing.assert_allclose(result["values"], [1, 1, 1.5, 2])
        np.testing.assert_equal(result["counts"], [2, 2, 2, 2])
        self.assertEqual(result["statistics"]["median"], 2)

    def test_segment_end_event_is_not_lost_in_following_empty_bin(self):
        source = recording([5, 12, 20], [np.nan] * 3, observed=[[0, 5], [10, 20]])
        result = summarize_behavior([source], "frequency", bin_s=5)
        np.testing.assert_allclose(result["values"], [12, np.nan, 12, 12])
        cumulative = summarize_behavior([source], "cumulative_count", bin_s=5)
        np.testing.assert_equal(cumulative["values"], [1, 1, 2, 3])

    def test_onset_interval_supports_unknown_ends_but_never_crosses_gaps(self):
        source = recording([1, 4, 12, 18], [np.nan] * 4, observed=[[0, 5], [10, 20]])
        result = summarize_behavior([source], "onset_interval", bin_s=1, auto_bins=False)
        self.assertEqual(result["values"].sum(), 2)
        self.assertEqual(result["statistics"]["median"], 4.5)
        self.assertEqual(result["statistics"]["q1"], 3.75)

    def test_duration_over_time_uses_complete_bouts_and_has_explicit_iqr(self):
        source = recording([1, 3, 5, 12], [2, 6, 15, 16], observed=[[0, 10], [11, 20]])
        result = summarize_behavior([source], "duration_time", bin_s=10)
        # The 5->15 bout crosses a gap and does not enter the duration statistic.
        np.testing.assert_equal(result["values"], [2, 4])
        np.testing.assert_equal(result["lower"], [1.5, 4])
        np.testing.assert_equal(result["upper"], [2.5, 4])
        self.assertTrue(np.all(np.isnan(result["sem"])))
        second = recording([1, 5], [7, 13], name="b")
        group = summarize_behavior([source, second], "duration_time", bin_s=10)
        np.testing.assert_equal(group["values"], [4.5, 4])
        np.testing.assert_equal(group["lower"], [3.25, 4])
        np.testing.assert_equal(group["upper"], [5.75, 4])
        empty = summarize_behavior([recording([1], [2])], "duration_time", bin_s=5)
        self.assertTrue(np.all(np.isnan(empty["values"][1:])))

    def test_new_statistics_and_iqr_export_match_numerical_summary(self):
        result = summarize_behavior([recording([1, 3], [2, 7])], "duration_time", bin_s=10)
        with tempfile.TemporaryDirectory() as folder:
            prefix = Path(folder) / "duration_time"
            export_behavior_summary(result, prefix, write_csv=True, write_h5=True)
            metadata = json.loads(Path(str(prefix) + ".json").read_text())
            self.assertEqual(metadata["statistics"]["median"], 2.5)
            with Path(str(prefix) + ".csv").open(encoding="utf-8-sig") as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(float(rows[0]["lower"]), 1.75)
            with h5py.File(str(prefix) + ".h5") as output:
                np.testing.assert_equal(output["upper"][:], result["upper"])
                self.assertIn("per_file_q1", output)

    def test_duration_histogram_manual_bins_and_constants(self):
        source = recording([1, 5, 10], [2, 7, 14])
        result = summarize_behavior([source], "duration", bin_s=1, auto_bins=False)
        np.testing.assert_equal(result["edges"], [0, 1, 2, 3, 4])
        np.testing.assert_equal(result["values"], [0, 1, 1, 1])
        constant = summarize_behavior([recording([1, 5], [2, 6])], "duration")
        self.assertEqual(sum(constant["values"]), 2)
        self.assertGreaterEqual(constant["edges"][0], 0)

    def test_ibi_is_offset_to_onset_with_no_cross_recording_intervals(self):
        first = recording([1, 6, 12], [3, 8, 14])
        second = recording([1, 10], [5, 13], name="b")
        result = summarize_behavior([first, second], "ibi", bin_s=1, auto_bins=False)
        self.assertEqual(sum(result["values"]), 3)
        # Actual gaps are 3, 4 and 5 s. A 5-s upper edge includes its endpoint.
        np.testing.assert_equal(result["values"], [0, 0, 0, 1, 2])

    def test_ibi_does_not_cross_gaps_or_skip_unknown_bout_ends(self):
        source = recording([1, 6, 12, 16], [3, np.nan, 14, 18], observed=[[0, 9], [10, 20]])
        result = summarize_behavior([source], "ibi", bin_s=1, auto_bins=False)
        self.assertEqual(sum(result["values"]), 2)  # gaps 3 and 2, no6->12 interval
        broken = recording([1, 12], [3, 14], observed=[[0, 5], [10, 20]])
        self.assertFalse(summarize_behavior([broken], "ibi")["has_data"])

    def test_frequency_uses_observed_seconds_and_partial_final_bin(self):
        source = recording([1, 12, 19, 22], [2, 13, 20, 23], end=23, observed=[[0, 5], [10, 23]])
        result = summarize_behavior([source], "frequency", bin_s=10)
        np.testing.assert_equal(result["edges"], [0, 10, 20, 23])
        np.testing.assert_allclose(result["values"], [12, 12, 20])
        np.testing.assert_equal(result["observed_seconds"], [[5, 10, 3]])

    def test_group_frequency_is_equal_recording_mean_with_missing_bins(self):
        first = recording([1], [2], end=10)
        second = recording([1, 2, 12], [1.5, 3, 13], name="b")
        result = summarize_behavior([first, second], "frequency", bin_s=10)
        np.testing.assert_equal(result["values"], [9, 6])
        np.testing.assert_equal(result["counts"], [2, 1])
        np.testing.assert_allclose(result["sem"], [3, 0])
        self.assertTrue(np.isnan(result["per_file_values"][0, 1]))

    def test_cumulative_duration_integrates_overlapping_bouts_across_bins(self):
        source = recording([2, 8], [12, 14])
        result = summarize_behavior([source], "cumulative", bin_s=5)
        np.testing.assert_equal(result["values"], [3, 8, 12, 12])
        alternate = summarize_behavior([source], "cumulative", bin_s=7)
        self.assertEqual(alternate["values"][-1], 12)

    def test_cumulative_duration_counts_only_observed_time(self):
        source = recording([2], [12], observed=[[0, 5], [10, 20]])
        result = summarize_behavior([source], "cumulative", bin_s=5)
        np.testing.assert_equal(result["values"], [3, 3, 5, 5])

    def test_group_cumulative_cannot_fall_when_a_recording_ends(self):
        first = recording([1], [9], end=10)
        second = recording([12], [14], name="b")
        result = summarize_behavior([first, second], "cumulative", bin_s=5)
        np.testing.assert_equal(result["values"], [2, 4, 5, 5])
        self.assertTrue(np.all(np.diff(result["values"]) >= 0))
        np.testing.assert_equal(result["counts"], [2, 2, 2, 2])

    def test_point_events_support_frequency_without_inventing_duration(self):
        source = recording([1, 4], [np.nan, np.nan])
        self.assertTrue(summarize_behavior([source], "frequency")["has_data"])
        for metric in ("duration", "ibi", "cumulative"):
            self.assertFalse(summarize_behavior([source], metric)["has_data"])

    def test_elapsed_origin_empty_cases_and_input_integrity(self):
        source = recording([102], [112], start=100, end=120)
        before = source["onsets"].copy()
        result = summarize_behavior([source], "cumulative", bin_s=5)
        np.testing.assert_equal(result["values"], [3, 8, 10, 10])
        np.testing.assert_equal(before, source["onsets"])
        self.assertFalse(summarize_behavior([])["has_data"])
        empty = summarize_behavior([recording([], [])], "frequency", bin_s=10)
        np.testing.assert_equal(empty["values"], [0, 0])
        with self.assertRaises(ValueError):
            summarize_behavior([source], bin_s=0)
        with self.assertRaises(ValueError):
            summarize_behavior([source], "frequency", bin_s=.0001)

    def test_export_retains_visible_metric_bins_and_group_values(self):
        """CSV and HDF5 must reproduce the selected chart and its actual units."""
        result = summarize_behavior([recording([2], [12])], "cumulative", bin_s=5)
        with tempfile.TemporaryDirectory() as folder:
            prefix = Path(folder) / "selected_behavior"
            paths = export_behavior_summary(result, prefix, write_csv=True, write_h5=True)
            self.assertEqual(len(paths), 3)
            with Path(str(prefix) + ".csv").open(encoding="utf-8-sig") as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual([float(row["value"]) for row in rows], [3, 8, 10, 10])
            with h5py.File(str(prefix) + ".h5") as source:
                np.testing.assert_equal(source["edges"][:], result["edges"])
                np.testing.assert_equal(source["per_file_values"][:], result["per_file_values"])
            metadata = json.loads(Path(str(prefix) + ".json").read_text())
            self.assertEqual(metadata["metric"], "Cumulative duration")
            self.assertEqual(metadata["bin_s"], 5)


if __name__ == "__main__":
    unittest.main()
