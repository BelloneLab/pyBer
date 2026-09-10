"""Fit-ranked inline baselines must stay pre-event, explicit and numerically sound."""
from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "pyBer"))
from baseline_advisor import BaselineRecording
from baseline_suggestions import (
    BaselineSuggestionConfig, suggest_baselines, _batch_window_statistics,
    _window_statistics, _candidate_windows,
)


class BaselineSuggestionsTests(unittest.TestCase):
    """Compare controlled inputs and independent invariants of the ranking."""

    def test_integer_grid_respects_fractional_limits_and_guard(self):
        """Scored endpoints are exact integers, inside both user and guard limits."""
        for pre in (1.9, 2., 4.7, 5., 60.):
            for guard in (.1, 1., 1.8):
                config = BaselineSuggestionConfig(pre_window_s=pre, guard_s=guard)
                windows = _candidate_windows(config)
                self.assertLessEqual(len(windows), 40)
                for start, end in windows:
                    self.assertEqual(start, int(start))
                    self.assertEqual(end, int(end))
                    self.assertGreaterEqual(start, -pre)
                    self.assertLessEqual(start + 1, end)
                    self.assertLess(end, -guard)
        self.assertEqual(_candidate_windows(BaselineSuggestionConfig(pre_window_s=1.9)), [])
        self.assertEqual(_candidate_windows(BaselineSuggestionConfig(pre_window_s=2.)), [(-2, -1)])

    def fixture(self, seed=5, events=None):
        """Create a deterministic stationary signal without altering any user data."""
        time = np.arange(0., 160., .05)
        signal = np.random.default_rng(seed).normal(0., .01, len(time))
        events = np.arange(10., 151., 10.) if events is None else np.asarray(events, float)
        return BaselineRecording("synthetic", time, signal, events)

    def test_sparse_events_offer_limited_choices_without_holdout_gate(self):
        result = suggest_baselines([self.fixture(events=[20, 50, 90])])
        self.assertEqual(result["status"], "ready")
        self.assertGreaterEqual(len(result["choices"]), 2)
        for choice in result["choices"]:
            self.assertEqual(choice["quality"], "Limited")
            self.assertLessEqual(choice["score"], 55)
            self.assertGreaterEqual(choice["start"], -5)
            self.assertLess(choice["start"], choice["end"])
            self.assertLess(choice["end"], 0)
            self.assertEqual(choice["diagnostics"]["event_free_fraction"], 1)

    def test_repeated_unit_and_post_event_invariance(self):
        """Only the pre-event inputs can affect fit ranking, across several seeds."""
        for seed in (5, 17, 81):
            recording = self.fixture(seed)
            expected = suggest_baselines([recording])
            changed = recording.signal.copy()
            for event in recording.events:
                changed[(recording.time >= event) & (recording.time < event + 3)] += 1000
            altered = BaselineRecording(recording.label, recording.time, changed, recording.events)
            scaled = BaselineRecording(recording.label, recording.time, recording.signal * 100, recording.events)
            for actual in (suggest_baselines([altered]), suggest_baselines([scaled])):
                self.assertEqual([choice["window"] for choice in actual["choices"]],
                                 [choice["window"] for choice in expected["choices"]])
                self.assertEqual([choice["score"] for choice in actual["choices"]],
                                 [choice["score"] for choice in expected["choices"]])

    def test_flat_missing_and_deterministic_trend_abstain(self):
        recording = self.fixture()
        for signal in (np.zeros_like(recording.signal), np.full_like(recording.signal, np.nan), recording.time.copy()):
            result = suggest_baselines([BaselineRecording(recording.label, recording.time, signal, recording.events)])
            self.assertEqual(result["choices"], [])

    def test_overlap_is_never_hidden_from_score_or_count(self):
        """Some clean target windows permit choices but prior bouts visibly limit them."""
        recording = self.fixture(events=[20, 40, 60, 80, 100, 120])
        exclusions = np.array([[15, 20], [55, 60], [95, 100]], float)
        result = suggest_baselines([BaselineRecording(recording.label, recording.time, recording.signal,
                                                      recording.events, exclusions)])
        self.assertTrue(result["choices"])
        for choice in result["choices"]:
            self.assertEqual(choice["quality"], "Limited")
            self.assertLessEqual(choice["score"], 49)
            self.assertAlmostEqual(choice["diagnostics"]["event_free_fraction"], .5)
            row = choice["diagnostics"]["per_recording"][0]
            self.assertEqual(row["events"], 6)
            self.assertEqual(row["overlap_events"], 3)
            self.assertEqual(row["sampled_events"], 3)
            self.assertIn("overlap", choice["summary"].lower())

    def test_gap_coverage_is_measured_on_every_event(self):
        recording = self.fixture(events=[20, 40, 60, 80])
        signal = recording.signal.copy()
        signal[(recording.time > 35) & (recording.time < 40)] = np.nan
        result = suggest_baselines([BaselineRecording(recording.label, recording.time, signal, recording.events)])
        self.assertTrue(result["choices"])
        for choice in result["choices"]:
            self.assertAlmostEqual(choice["diagnostics"]["observed_coverage"], .75)
            self.assertLessEqual(choice["score"], 49)

    def test_vectorized_statistics_match_scalar_across_mixed_lengths(self):
        """The speedup preserves scalar math for odd/even lengths and flat segments."""
        rng = np.random.default_rng(444)
        samples = []
        for length in (6, 7, 18, 33, 100, 256):
            for index in range(5):
                time = np.arange(length) / 20
                values = rng.normal(size=length) * (index + 1)
                if index == 0:
                    values[:] = 0
                samples.append((time, values))
        expected = [_window_statistics(t, y) for t, y in samples]
        expected = [row for row in expected if row is not None]
        actual = _batch_window_statistics(samples)
        # Batch grouping changes list order; compare independent unique SD keys.
        expected.sort(key=lambda row: row["sd"])
        actual.sort(key=lambda row: row["sd"])
        self.assertEqual(len(actual), len(expected))
        for first, second in zip(expected, actual):
            self.assertEqual(first["monotonic"], second["monotonic"])
            for key in first.keys() - {"monotonic"}:
                self.assertAlmostEqual(first[key], second[key], places=10)


if __name__ == "__main__":
    unittest.main()
