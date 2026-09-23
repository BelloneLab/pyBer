"""MAMIR identity, zone timing, and group weighting regression checks."""

from pathlib import Path
from tempfile import TemporaryDirectory
import json
import unittest

import numpy as np
import pandas as pd

from behavior_zone_compare import compare_recording, group_recordings
from mamir_import import inspect_mamir, load_mamir, legacy_import_warning


class MamirImportTests(unittest.TestCase):
    def test_all_labels_match_across_wide_long_and_bout_exports(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            labels = ["z_custom", "walking", "attack", "a_sub_attack"]
            wide_rows, long_rows, bouts = [], [], []
            expected = {}
            for identity in (1, 2):
                frames = np.arange(40)
                time = 17.5 + frames / 20.
                flags = np.random.default_rng(identity).integers(0, 2, (40, len(labels)))
                for i, (frame, stamp) in enumerate(zip(frames, time)):
                    wide_rows.append([identity, *flags[i], stamp, frame])
                    for j, label in enumerate(labels):
                        if flags[i, j]:
                            long_rows.append([label, identity, stamp, frame])
                for j, label in enumerate(labels):
                    active = np.flatnonzero(flags[:, j])
                    runs = np.split(active, np.flatnonzero(np.diff(active) != 1) + 1)
                    on = [time[run[0]] for run in runs]
                    off = [time[run[-1]] + .05 for run in runs]
                    expected[(identity, label)] = (on, off)
                    bouts.extend([identity, label, start, stop] for start, stop in zip(on, off))
            pd.DataFrame(wide_rows, columns=["identity", *labels, "time_s", "frame"]).to_csv(root / "behavior_frames.csv", index=False)
            pd.DataFrame(long_rows, columns=["behavior", "identity", "time_s", "frame"]).to_csv(root / "behavior_frames_long.csv", index=False)
            pd.DataFrame(bouts, columns=["identity", "behavior", "start_s", "stop_s"]).to_csv(root / "behavior_bouts.csv", index=False)
            (root / "behavior_summary.json").write_text(json.dumps({"fps": 20}), encoding="utf-8")
            for filename in ("behavior_frames.csv", "behavior_frames_long.csv", "behavior_bouts.csv"):
                for identity in (1, 2):
                    events = load_mamir(root / filename, "behavior", str(identity))["event_behaviors"]
                    for label in labels:
                        with self.subTest(filename=filename, identity=identity, label=label):
                            on, off = expected[(identity, label)]
                            np.testing.assert_allclose(events[label]["on"], on)
                            np.testing.assert_allclose(events[label]["off"], off)

    def test_legacy_snapshots_are_warned_about_without_mutation(self):
        sources = {"mouse": {"import_report": {"format": "mamir", "category": "behavior"}}}
        before = json.dumps(sources, sort_keys=True)
        self.assertIn("column-order", legacy_import_warning(sources))
        self.assertEqual(json.dumps(sources, sort_keys=True), before)
        sources["mouse"]["import_report"]["adapter_version"] = 2
        self.assertEqual(legacy_import_warning(sources), "")

    def test_wide_columns_are_matched_by_name_not_csv_position(self):
        with TemporaryDirectory() as temporary:
            path = Path(temporary) / "behavior_frames.csv"
            # Deliberately scramble both metadata and alphabetic behavior order.
            pd.DataFrame({"walking": [1, 0, 0, 0], "identity": [1]*4,
                          "time_s": [100., 100.1, 100.2, 100.3],
                          "attack": [0, 0, 1, 0], "frame": [0, 1, 2, 3]}
                         ).to_csv(path, index=False)
            result = load_mamir(path, "behavior", "1")["event_behaviors"]
            np.testing.assert_allclose(result["walking"]["on"], [100.])
            np.testing.assert_allclose(result["walking"]["off"], [100.1])
            np.testing.assert_allclose(result["attack"]["on"], [100.2])
            np.testing.assert_allclose(result["attack"]["off"], [100.3])

    def test_selected_frame_file_is_not_replaced_by_neighboring_bouts(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            frames = root / "behavior_frames.csv"
            pd.DataFrame({"frame": [0, 1, 2], "time_s": [0., .1, .2],
                          "identity": [1, 1, 1], "attack": [0, 1, 0]}).to_csv(frames, index=False)
            pd.DataFrame({"behavior": ["attack"], "identity": [1], "start_s": [9.],
                          "stop_s": [10.]}).to_csv(root / "behavior_bouts.csv", index=False)
            result = load_mamir(frames, "behavior", "1")
            np.testing.assert_allclose(result["event_behaviors"]["attack"]["on"], [.1])
            self.assertEqual(result["import_report"]["adapter_version"], 2)
            pd.DataFrame({"behavior": ["attack"], "count": [1]}).to_csv(root / "behavior_summary.csv", index=False)
            with self.assertRaisesRegex(ValueError, "time-resolved"):
                inspect_mamir(root / "behavior_summary.csv", "behavior")

    def test_long_columns_and_nonzero_clock_do_not_change_frame_duration(self):
        with TemporaryDirectory() as temporary:
            path = Path(temporary) / "behavior_frames_long.csv"
            pd.DataFrame({"behavior": ["attack", "attack", "attack", "groom"],
                          "time_s": [100., 100., 100.1, 100.3], "identity": [1]*4,
                          "frame": [0, 0, 1, 3]}).to_csv(path, index=False)
            result = load_mamir(path, "behavior", "1")["event_behaviors"]
            np.testing.assert_allclose(result["attack"]["on"], [100.])
            np.testing.assert_allclose(result["attack"]["off"], [100.2])
            np.testing.assert_allclose(result["groom"]["off"], [100.4])

    def test_wide_disordered_or_invalid_clock_is_rejected(self):
        with TemporaryDirectory() as temporary:
            path = Path(temporary) / "behavior_frames.csv"
            for times in ([0., .2, .1], [0., .1, .1], [0., np.nan, .2]):
                pd.DataFrame({"frame": [0, 1, 2], "time_s": times,
                              "identity": [1, 1, 1], "attack": [0, 1, 0]}).to_csv(path, index=False)
                with self.assertRaises(ValueError):
                    load_mamir(path, "behavior", "1")

    def test_interleaved_frames_keep_animal_identity_and_custom_behavior(self):
        with TemporaryDirectory() as temporary:
            path = Path(temporary) / "behavior_frames.csv"
            pd.DataFrame([
                (0, 0.0, 1, 0), (0, 0.0, 2, 1),
                (1, 0.1, 1, 1), (1, 0.1, 2, 1),
                (2, 0.2, 1, 1), (2, 0.2, 2, 0),
                (3, 0.3, 1, 0), (3, 0.3, 2, 0),
            ], columns=["frame", "time_s", "identity", "custom_action"]).to_csv(path, index=False)
            overview = inspect_mamir(path, "behavior")
            self.assertEqual(overview["identities"], ["1", "2"])
            info = load_mamir(path, "behavior", "1")
            events = info["event_behaviors"]["custom_action"]
            np.testing.assert_allclose(events["on"], [0.1])
            np.testing.assert_allclose(events["off"], [0.3])

    def test_partner_filter_and_zone_inclusive_end_frame(self):
        with TemporaryDirectory() as temporary:
            directory = Path(temporary)
            pd.DataFrame([
                ("attack", 1, 2, 5., 6.),
                ("attack", 1, 3, 7., 8.),
                ("flee", 2, 1, 9., 10.),
            ], columns=["behavior", "identity", "partner", "start_s", "stop_s"]
            ).to_csv(directory / "behavior_bouts.csv", index=False)
            info = load_mamir(directory / "behavior_bouts.csv", "behavior", "1", "2")
            np.testing.assert_allclose(info["event_behaviors"]["attack"]["on"], [5.])
            self.assertEqual(info["event_behaviors"]["flee"]["on"].size, 0)

            pd.DataFrame([(1, "left", 1, 10, 12, 3, .15)],
                         columns=["animal_id", "zone", "bout", "start_frame",
                                  "end_frame", "observed_frames", "duration_s"]
                         ).to_csv(directory / "zone_bouts.csv", index=False)
            (directory / "summary.json").write_text(json.dumps({"fps": 20}), encoding="utf-8")
            zone = load_mamir(directory / "zone_bouts.csv", "zone", "1")
            np.testing.assert_allclose(zone["event_behaviors"]["left"]["on"], [.5])
            np.testing.assert_allclose(zone["event_behaviors"]["left"]["off"], [.65])

    def test_long_frames_preserve_simultaneous_behaviors_and_gaps(self):
        with TemporaryDirectory() as temporary:
            path = Path(temporary) / "behavior_frames_long.csv"
            pd.DataFrame([
                (0, 0., 1, "attack"), (0, 0., 2, "groom"),
                (1, .1, 1, "attack"), (1, .1, 1, "flee"),
                (2, .2, 1, "flee"), (4, .4, 1, "attack"),
                (5, .5, 1, "attack"),
            ], columns=["frame", "time_s", "identity", "behavior"]).to_csv(path, index=False)
            overview = inspect_mamir(path, "behavior")
            self.assertEqual(overview["labels"], ["attack", "flee", "groom"])
            events = load_mamir(path, "behavior", "1")["event_behaviors"]
            np.testing.assert_allclose(events["attack"]["on"], [0., .4])
            np.testing.assert_allclose(events["attack"]["off"], [.2, .6])
            np.testing.assert_allclose(events["flee"]["on"], [.1])
            np.testing.assert_allclose(events["flee"]["off"], [.3])
            self.assertEqual(events["groom"]["on"].size, 0)

    def test_single_active_frame_uses_mamir_fps_sidecar(self):
        with TemporaryDirectory() as temporary:
            directory = Path(temporary)
            pd.DataFrame([(0, 0., 1, "attack")],
                         columns=["frame", "time_s", "identity", "behavior"]
                         ).to_csv(directory / "behavior_frames_long.csv", index=False)
            (directory / "behavior_summary.json").write_text(json.dumps({"fps": 20}), encoding="utf-8")
            events = load_mamir(directory / "behavior_frames_long.csv", "behavior", "1")["event_behaviors"]
            np.testing.assert_allclose(events["attack"]["off"], [.05])


class BehaviorZoneCompareTests(unittest.TestCase):
    def test_before_during_after_use_bout_start_and_end_without_stretching(self):
        time = np.arange(0., 20., .1)
        signal = np.zeros_like(time)
        signal[(time >= 8) & (time < 10)] = 1.
        signal[(time >= 10) & (time < 13)] = 4.
        signal[(time >= 13) & (time < 15)] = 2.
        saved = signal.copy()
        events = {"attack": {"on": np.array([10.]), "off": np.array([13.])}}
        row = compare_recording("a", time, signal, events, ["attack"], pre_s=2., post_s=2.)[0]
        self.assertEqual((row["before"], row["during"], row["after"]), (1., 4., 2.))
        self.assertEqual((row["during_minus_before"], row["after_minus_before"]), (3., 1.))
        np.testing.assert_array_equal(signal, saved)
        shifted_events = {"attack": {"on": np.array([9.]), "off": np.array([12.])}}
        shifted = compare_recording("a", time, signal, shifted_events, ["attack"],
                                    pre_s=2., post_s=2., offset_s=1.)[0]
        self.assertEqual(shifted, row)

    def test_incomplete_recording_edges_nan_and_gaps_reject_whole_bout(self):
        events = {"attack": {"on": np.array([1.]), "off": np.array([2.])}}
        # A nearly complete window still begins outside the recording.
        time = np.arange(0., 4., .1)
        row = compare_recording("a", time, time, events, ["attack"], pre_s=1.05, post_s=1.)[0]
        self.assertEqual((row["events"], row["rejected"]), (0, 1))
        for missing in ("nan", "gap"):
            signal = time.copy()
            clock = time.copy()
            if missing == "nan":
                signal[15] = np.nan
            else:
                clock = np.delete(clock, np.arange(13, 18))
                signal = np.delete(signal, np.arange(13, 18))
            row = compare_recording("a", clock, signal, events, ["attack"], pre_s=.5, post_s=.5)[0]
            self.assertEqual((row["events"], row["rejected"]), (0, 1))

    def test_complete_windows_and_equal_recording_group_weight(self):
        time = np.arange(0., 10., .1)
        events = {"attack": {"on": np.array([.5, 4.]),
                             "off": np.array([1., 5.])}}
        rows = compare_recording("a", time, time, events, ["attack"], pre_s=2., post_s=2.)
        self.assertEqual(rows[0]["events"], 1)
        self.assertEqual(rows[0]["rejected"], 1)
        self.assertAlmostEqual(rows[0]["during"], 4.45)
        second = dict(rows[0], file_id="b", events=9, during=14.45)
        group = group_recordings([rows[0], second])[0]
        self.assertEqual(group["recordings"], 2)
        self.assertEqual(group["events"], 10)
        self.assertEqual(group["rejected"], 2)
        self.assertAlmostEqual(group["during"], 9.45)


if __name__ == "__main__":
    unittest.main()
