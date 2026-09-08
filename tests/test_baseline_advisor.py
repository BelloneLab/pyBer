"""Behavioral guarantees for baseline recommendation, independent of the GUI."""
from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
from scipy.signal import lfilter

from pyBer.baseline_advisor import (
    BaselineAdvisorConfig, BaselineRecording, export_baseline_report, recommend_baseline,
    _context, _evaluate, _prepare,
)


class BaselineAdvisorTests(unittest.TestCase):
    """Validate reference quality and abstention, not a preferred plot shape."""

    def setUp(self):
        self.time = np.arange(0, 1000, .05)
        self.events = np.arange(40, 990, 20.)
        self.signal = np.random.default_rng(10).normal(size=len(self.time))
        self.recording = BaselineRecording("recording", self.time, self.signal, self.events)

    def test_stationary_reference_passes_purged_holdout(self):
        """Stable data should produce a native-data-supported, pre-event window."""
        before = self.signal.copy()
        result = recommend_baseline([self.recording], current_window=(-.98, -.79))
        self.assertEqual(result["status"], "recommended")
        start, end = result["window"]
        self.assertLess(start, end)
        self.assertLess(end, -.25)
        self.assertEqual(round(start, 2), start)
        info = result["recordings"][0]
        self.assertGreater(info["purged_events"], 0)
        self.assertFalse(result["current"]["eligible"])
        winner = result["candidates"][0]
        self.assertTrue(winner["diagnostics"]["validation"]["eligible"])
        self.assertGreater(winner["diagnostics"]["training"]["effective_samples_p10"], 20)
        np.testing.assert_equal(before, self.signal)

    def test_excluded_event_response_cannot_drive_recommendation(self):
        """Increasing responses wholly inside excluded bouts leaves rankings intact."""
        bouts = np.column_stack((self.events, self.events + 2))
        original = replace(self.recording, exclusion_intervals=bouts)
        changed = self.signal.copy()
        for event in self.events:
            changed[(self.time >= event) & (self.time <= event + 2)] += 1e4
        first = recommend_baseline([original])
        second = recommend_baseline([replace(original, signal=changed)])
        self.assertEqual(first, second)

    def test_long_active_bouts_and_offset_alignment_are_excluded(self):
        """Negative time relative to an offset is still inside its active bout."""
        bouts = np.column_stack((self.events - 12, self.events))
        result = recommend_baseline([replace(self.recording, exclusion_intervals=bouts)])
        if result["status"] == "recommended":
            self.assertLess(result["window"][1], -12.25)
        else:
            self.assertTrue(any("coverage" in reason for reason in result["reasons"]))

    def test_dense_events_abstain_instead_of_crossing_neighbors(self):
        """No event-free duration means no valid recommendation."""
        result = recommend_baseline([replace(self.recording, events=np.arange(1, 999, .2))])
        self.assertEqual(result["status"], "unavailable")
        self.assertIsNone(result["window"])

    def test_unfiltered_event_exclusions_survive_filtered_target_list(self):
        """Unselected intermediate bouts must still protect the reference."""
        all_events = np.arange(1, 999, .2)
        result = recommend_baseline([replace(self.recording,
            exclusion_intervals=np.column_stack((all_events, all_events)))])
        self.assertEqual(result["status"], "unavailable")

    def test_flat_data_and_small_event_counts_abstain(self):
        """A quiet-looking trace or very few trials is not evidence of reliability."""
        flat = recommend_baseline([replace(self.recording, signal=np.ones(len(self.time)))])
        self.assertEqual(flat["status"], "unavailable")
        few = recommend_baseline([replace(self.recording, events=self.events[:8])])
        self.assertEqual(few["status"], "unavailable")
        self.assertTrue(any("too few" in reason for reason in few["reasons"]))

    def test_temporal_scale_change_fails_once_without_alternative_search(self):
        """Later variance instability cannot be hidden by picking another holdout winner."""
        changed = self.signal.copy()
        changed[self.time > 700] *= 10
        result = recommend_baseline([replace(self.recording, signal=changed)])
        self.assertEqual(result["status"], "unavailable")
        checked = [row for row in result["candidates"] if "validation" in row["diagnostics"]]
        self.assertEqual(len(checked), 1)
        self.assertTrue(any("later" in reason for reason in result["reasons"]))

    def test_groups_do_not_hide_an_unreliable_recording(self):
        """Many clean trials cannot override another file's invalid reference."""
        flat = replace(self.recording, label="flat", signal=np.ones(len(self.time)))
        result = recommend_baseline([self.recording, flat])
        self.assertEqual(result["status"], "unavailable")
        self.assertEqual(len(result["recordings"]), 2)

    def test_cuts_are_never_filled_or_bridged(self):
        """NaN cuts and equivalent removed timestamps receive conservative rejection."""
        keep = np.ones(len(self.time), bool)
        for event in self.events:
            keep[(self.time >= event - 19) & (self.time < event - .25)] = False
        cut = self.signal.copy()
        cut[~keep] = np.nan
        for recording in (replace(self.recording, signal=cut),
                          replace(self.recording, time=self.time[keep], signal=self.signal[keep])):
            self.assertEqual(recommend_baseline([recording])["status"], "unavailable")

    def test_units_and_offset_do_not_change_choice(self):
        """Selection is based on relative stability rather than absolute amplitude."""
        original = recommend_baseline([self.recording])
        converted = recommend_baseline([replace(self.recording, signal=self.signal * .001 + 2)])
        self.assertEqual(original["window"], converted["window"])
        self.assertEqual(original["status"], converted["status"])
        # Squared-signal autocorrelation must not fail through fourth-power
        # overflow/underflow merely because the signal is in different units.
        for factor in (1e-100, 1e100):
            scaled = recommend_baseline([replace(self.recording, signal=self.signal * factor)])
            self.assertEqual(original["window"], scaled["window"])

    def test_autocorrelation_reduces_effective_information(self):
        """Slow signals carry less information than equal-length white-noise traces."""
        slow = lfilter([1], [1, -.98], self.signal)
        white = recommend_baseline([self.recording])
        colored = recommend_baseline([replace(self.recording, signal=slow)])
        white_tau = white["recordings"][0]["training_context"]["information_s"]
        slow_tau = colored["recordings"][0]["training_context"]["information_s"]
        self.assertGreater(slow_tau, white_tau * 10)

    def test_smooth_candidate_cannot_borrow_information_from_noisy_context(self):
        """Candidate-local dependence must override a shorter global ACF estimate."""
        events = np.arange(40, 990, 40.)
        signal = .5 * self.signal.copy()
        for event in events:
            mask = (self.time >= event - 8.5) & (self.time <= event - .5)
            signal[mask] = np.sin(2 * np.pi * (self.time[mask] - event) / 4) / np.sqrt(2)
        config = BaselineAdvisorConfig()
        recording = _prepare(replace(self.recording, events=events, signal=signal), config)
        context = _context(recording, recording["train"], config)
        result = _evaluate(recording, recording["train"], context, (-8.5, -.5), config)
        self.assertLess(result["effective_samples_p10"], 10)
        self.assertFalse(result["eligible"])

    def test_reports_are_portable_and_invalid_inputs_raise(self):
        """Reports reproduce configuration and do not contain nonstandard JSON floats."""
        report = recommend_baseline([self.recording])
        with tempfile.TemporaryDirectory() as directory:
            json_path, csv_path = export_baseline_report(report, Path(directory) / "baseline")
            self.assertEqual(json.loads(Path(json_path).read_text()), report)
            self.assertIn("validation_passed", Path(csv_path).read_text(encoding="utf-8-sig"))
        with self.assertRaises(ValueError):
            recommend_baseline([self.recording], BaselineAdvisorConfig(min_window_s=0))
        with self.assertRaises(ValueError):
            recommend_baseline([replace(self.recording, time=self.time[::-1])])
        self.assertEqual(recommend_baseline([])["status"], "unavailable")


if __name__ == "__main__":
    unittest.main()
