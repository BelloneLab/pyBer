"""Recording groups: model, persistence and group-aware PSTH presentation."""
import os
from pathlib import Path
import sys
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "pyBer"))

import numpy as np
from PySide6 import QtWidgets

from postprocessing_core import mean_sem
from psth_groups import block_summaries, block_ticks, build_group_display, legend_label, row_file_id
from recording_groups import GROUP_COLORS_DARK, RecordingGroups, UNGROUPED_LABEL


class RecordingGroupModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

    def test_assignment_rename_recolor_delete_and_signal(self):
        model = RecordingGroups()
        emitted = []
        model.changed.connect(lambda: emitted.append(1))
        wt = model.add_group("WT")
        het = model.add_group("Het")
        self.assertEqual([g["color"] for g in model.groups()], GROUP_COLORS_DARK[:2])
        self.assertEqual(model.assign(["a", "b"], wt), 2)
        self.assertEqual(model.assign(["b", "c"], het), 2)   # b moves to Het
        self.assertEqual(model.members(wt, ["a", "b", "c"]), ["a"])
        self.assertEqual(model.members(het, ["a", "b", "c"]), ["b", "c"])
        self.assertEqual(model.assign(["c"], None), 1)
        self.assertIsNone(model.group_id_of("c"))
        self.assertTrue(model.update_group(wt, name="Wild type", color="#123456"))
        self.assertEqual(model.group(wt)["name"], "Wild type")
        self.assertEqual(model.group(wt)["color"], "#123456")
        model.remove_group(het)
        self.assertIsNone(model.group_id_of("b"))
        self.assertGreater(len(emitted), 4)

    def test_unused_palette_slot_follows_the_entity(self):
        model = RecordingGroups()
        first = model.add_group("A")
        model.add_group("B")
        model.remove_group(first)
        # The freed first slot is reused rather than generating a new hue.
        self.assertEqual(model.group(model.add_group("C"))["color"], GROUP_COLORS_DARK[0])

    def test_state_roundtrip_and_malformed_state_is_ignored(self):
        model = RecordingGroups()
        wt = model.add_group("WT")
        model.assign(["m1", "m2"], wt)
        state = model.to_state()
        restored = RecordingGroups()
        restored.from_state(state)
        self.assertEqual(restored.to_state()["groups"], state["groups"])
        self.assertEqual(restored.assignments(), {"m1": wt, "m2": wt})
        self.assertNotEqual(restored.add_group("New"), wt)   # ids never collide
        broken = RecordingGroups()
        broken.from_state({"groups": [{"id": "", "name": "x"}, "junk", {"id": "g9", "name": "Ok", "color": "bad"}],
                           "assignments": {"f": "missing", "g": "g9"}})
        self.assertEqual([g["name"] for g in broken.groups()], ["Ok"])
        self.assertEqual(broken.assignments(), {"g": "g9"})
        broken.from_state("not a dict")
        self.assertEqual(broken.groups(), [])


class GroupDisplayTests(unittest.TestCase):
    groups = [{"id": "g1", "name": "WT", "color": "#3987e5"}, {"id": "g2", "name": "Het", "color": "#d95926"}]
    assignment = {"wt1": "g1", "wt2": "g1", "het1": "g2"}

    def matrix(self):
        rng = np.random.default_rng(1)
        return rng.normal(size=(4, 30))

    def test_all_groups_reorders_rows_without_changing_values(self):
        mat = self.matrix()
        original = mat.copy()
        labels = ["het1", "wt1", "loose", "wt2"]
        display = build_group_display(mat, labels, "all", self.groups, self.assignment.get)
        self.assertEqual(display.labels, ["wt1", "wt2", "het1", "loose"])
        self.assertEqual([(b.name, b.start, b.stop) for b in display.blocks],
                         [("WT", 0, 2), ("Het", 2, 3), (UNGROUPED_LABEL, 3, 4)])
        np.testing.assert_array_equal(display.matrix[0], mat[1])
        np.testing.assert_array_equal(mat, original)
        self.assertEqual(display.row_colors[:3], ["#3987e5", "#3987e5", "#d95926"])
        # Per-group statistics are exactly the pooled statistics of that subset.
        block, mean, sem, counts = block_summaries(display)[0]
        ref_mean, ref_sem, ref_counts = mean_sem(mat[[1, 3]])
        np.testing.assert_array_equal(mean, ref_mean)
        np.testing.assert_array_equal(sem, ref_sem)
        self.assertEqual(legend_label(block, "animals"), "WT (n=2)")
        self.assertEqual(block_ticks(display.blocks, "animals")[0], (1.0, "WT"))

    def test_only_one_group_and_trial_labels(self):
        mat = self.matrix()
        labels = ["wt1 | Trial 1", "het1 | Trial 1", "wt1 | Trial 2", "wt2 | Trial 1"]
        self.assertEqual(row_file_id(labels[1]), "het1")
        display = build_group_display(mat, labels, "only", self.groups, self.assignment.get,
                                      only_group="g1", level="trials")
        self.assertEqual(display.matrix.shape, (3, 30))
        self.assertEqual(display.blocks[0].n_files, 2)
        self.assertEqual(legend_label(display.blocks[0], "trials"), "WT (n=3 trials, 2 rec.)")
        empty = build_group_display(mat, labels, "only", self.groups, {}.get, only_group="g2")
        self.assertEqual(empty.matrix.shape[0], 0)

    def test_mismatched_labels_are_never_guessed(self):
        self.assertIsNone(build_group_display(self.matrix(), ["a"], "all", self.groups, self.assignment.get))
        self.assertIsNone(build_group_display(self.matrix(), ["a"] * 4, "only", self.groups,
                                              self.assignment.get, only_group="nope"))


if __name__ == "__main__":
    unittest.main()
