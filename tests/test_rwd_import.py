import os
import sys
import tempfile
import unittest

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "pyBer"))

from analysis_core import is_rwd_events_csv, load_rwd_csv  # noqa: E402


class RwdImportTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="pyber_rwd_")
        self.dir = self.tmp.name
        self.events_path = os.path.join(self.dir, "Events.csv")
        with open(self.events_path, "w", encoding="utf-8", newline="") as f:
            f.write("TimeStamp,Name,State\n")
            f.write("25,Cue,1\n")
            f.write("75,Cue,0\n")
            f.write("25,Dip,0\n")
            f.write("75,Dip,1\n")

    def tearDown(self):
        self.tmp.cleanup()

    def test_aligned_rwd_fluorescence_loads_with_events(self):
        path = os.path.join(self.dir, "Fluorescence.csv")
        with open(path, "w", encoding="utf-8", newline="") as f:
            f.write('{"Fps":80.0}\n')
            f.write("TimeStamp,Events,CH1-410,CH1-470,\n")
            f.write("0,,10,20,\n")
            f.write("25,,11,21,\n")
            f.write("50,,12,22,\n")
            f.write("75,,13,23,\n")
            f.write("100,,14,24,\n")

        loaded = load_rwd_csv(path)

        self.assertIsNotNone(loaded)
        self.assertEqual(loaded.channels, ["CH1"])
        np.testing.assert_allclose(loaded.time_by_channel["CH1"], [0.0, 0.025, 0.05, 0.075, 0.1])
        np.testing.assert_allclose(loaded.reference_by_channel["CH1"], [10, 11, 12, 13, 14])
        np.testing.assert_allclose(loaded.signal_by_channel["CH1"], [20, 21, 22, 23, 24])
        self.assertIn("Cue", loaded.trigger_by_name)
        np.testing.assert_allclose(loaded.trigger_by_name["Cue"], [0, 1, 1, 0, 0])
        self.assertIn("Dip", loaded.trigger_by_name)
        np.testing.assert_allclose(loaded.trigger_by_name["Dip"], [0, 1, 1, 0, 0])

        trial = loaded.make_trial("CH1", trigger_name="Cue")
        np.testing.assert_allclose(trial.time, loaded.time_by_channel["CH1"])
        np.testing.assert_allclose(trial.trigger, [0, 1, 1, 0, 0])

    def test_unaligned_rwd_fluorescence_pairs_alternating_lights(self):
        path = os.path.join(self.dir, "Fluorescence-unaligned.csv")
        with open(path, "w", encoding="utf-8", newline="") as f:
            f.write("TimeStamp,Lights,Channel1\n")
            f.write("0,410,10\n")
            f.write("12.5,470,20\n")
            f.write("25,410,11\n")
            f.write("37.5,470,21\n")
            f.write("50,410,12\n")
            f.write("62.5,470,22\n")
            f.write("75,410,13\n")
            f.write("87.5,470,23\n")
            f.write("100,410,14\n")
            f.write("112.5,470,24\n")

        loaded = load_rwd_csv(path)

        self.assertIsNotNone(loaded)
        self.assertEqual(loaded.channels, ["Channel1"])
        np.testing.assert_allclose(loaded.time_by_channel["Channel1"], [0.0, 0.025, 0.05, 0.075, 0.1])
        np.testing.assert_allclose(loaded.reference_by_channel["Channel1"], [10, 11, 12, 13, 14])
        np.testing.assert_allclose(loaded.signal_by_channel["Channel1"], [20, 21, 22, 23, 24])
        self.assertIn("Cue", loaded.trigger_by_name)
        np.testing.assert_allclose(loaded.trigger_by_name["Cue"], [0, 1, 1, 0, 0])
        self.assertIn("Dip", loaded.trigger_by_name)
        np.testing.assert_allclose(loaded.trigger_by_name["Dip"], [0, 1, 1, 0, 0])

    def test_events_csv_is_detected_but_not_loaded_as_fluorescence(self):
        self.assertTrue(is_rwd_events_csv(self.events_path))
        self.assertIsNone(load_rwd_csv(self.events_path))


if __name__ == "__main__":
    unittest.main()
