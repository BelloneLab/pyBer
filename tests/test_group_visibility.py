"""Groups menu: show or hide the visuals of individual groups."""
import unittest

import numpy as np

import test_group_view_dashboard as dashboard
from psth_groups import build_group_display

N = dashboard.N_PER_GROUP


class GroupDisplayHiddenTests(unittest.TestCase):
    def test_hidden_groups_are_left_out_and_values_are_untouched(self):
        matrix = np.arange(12, dtype=float).reshape(4, 3)
        labels = ["a1", "b1", "a2", "x"]
        groups = [{"id": "A", "name": "A", "color": "#111111"}, {"id": "B", "name": "B", "color": "#222222"}]
        owner = {"a1": "A", "a2": "A", "b1": "B"}
        full = build_group_display(matrix, labels, "all", groups, owner.get)
        self.assertEqual([b.name for b in full.blocks], ["A", "B", "Ungrouped"])
        shown = build_group_display(matrix, labels, "all", groups, owner.get, hidden_groups={"B", ""})
        self.assertEqual([b.name for b in shown.blocks], ["A"])
        self.assertEqual(shown.labels, ["a1", "a2"])
        np.testing.assert_array_equal(shown.matrix, matrix[[0, 2]])


class GroupsMenuTests(dashboard.GroupViewDashboardTests):
    """Reuse the synthetic WT / Het recordings of the group dashboard tests."""

    def _menu_actions(self):
        self.panel._rebuild_group_menu()
        return {action.text(): action for action in self.panel.menu_group_view.actions() if action.text()}

    def test_hiding_one_group_removes_only_its_visuals(self):
        self.load()
        panel = self.panel
        panel.cb_group_view.setChecked(True)
        self.app.processEvents()
        original = panel._group_mat.copy()
        self.assertFalse(panel.btn_group_view.isHidden())
        self.assertEqual(panel.btn_group_view.text(), "Hide/show groups")
        actions = self._menu_actions()
        self.assertEqual([name for name in actions][:3], ["WT", "Het", "Ungrouped"])
        self.assertTrue(actions["WT"].isChecked() and actions["Het"].isChecked())

        actions["Het"].setChecked(False)          # hide Het
        self.app.processEvents()
        panel._reflect_group_view()
        self.assertEqual(panel._hidden_groups(), {self.het})
        self.assertEqual([b.name for b in panel._group_display.blocks], ["WT"])
        self.assertEqual(panel._last_mat.shape[0], N)
        self.assertEqual(np.asarray(panel.img.image).T.shape[0], N)
        legend = [label.text for _sample, label in panel.psth_group_legend.items]
        self.assertEqual(legend, [f"WT (n={N})"])
        self.assertEqual(panel.btn_group_view.text(), "Hide/show groups (2/3)")
        self.assertFalse(panel._group_view_keeps("het0"))
        self.assertTrue(panel._group_view_keeps("wt0"))
        np.testing.assert_array_equal(panel._group_mat, original)   # data untouched

        state = panel._collect_settings()["group_view"]
        self.assertEqual(state["hidden"], [self.het])

        actions = self._menu_actions()
        self.assertFalse(actions["Het"].isChecked())
        actions["Show all groups"].trigger()
        self.app.processEvents()
        self.assertEqual(panel._hidden_groups(), set())
        self.assertEqual([b.name for b in panel._group_display.blocks], ["WT", "Het"])
        self.assertEqual(panel._last_mat.shape[0], 2 * N)

    def test_chip_eye_toggles_visibility_and_chip_body_still_assigns(self):
        from PySide6 import QtCore, QtTest
        from recording_groups import GroupChip
        self.load()
        panel = self.panel
        panel.cb_group_view.setChecked(True)
        panel._section_buttons["setup"].setChecked(True)
        for _ in range(10):
            self.app.processEvents()
        chips = {chip.text().split("  ")[0]: chip for chip in panel.group_chip_bar.findChildren(GroupChip)
                 if not chip.isHidden()}
        self.assertEqual(sorted(chips), ["Het", "WT"])          # Ungroup / New group have no eye
        het = chips["Het"]
        self.assertTrue(het.is_group_shown())
        self.assertLess(het.eye_rect().right(), het.width())
        self.assertGreater(het.eye_rect().left(), het.width() // 2)

        assigned = []
        panel.group_chip_bar.assignRequested.connect(assigned.append)
        QtTest.QTest.mouseClick(het, QtCore.Qt.MouseButton.LeftButton, pos=het.eye_rect().center())
        self.app.processEvents()
        self.assertEqual(assigned, [])                           # the eye does not assign
        self.assertEqual(panel._hidden_groups(), {self.het})
        self.assertFalse(het.is_group_shown())
        self.assertEqual([b.name for b in panel._group_display.blocks], ["WT"])
        self.assertEqual(panel.btn_group_view.text(), "Hide/show groups (2/3)")

        QtTest.QTest.mouseClick(het, QtCore.Qt.MouseButton.LeftButton, pos=het.eye_rect().center())
        self.app.processEvents()
        self.assertEqual(panel._hidden_groups(), set())
        self.assertEqual([b.name for b in panel._group_display.blocks], ["WT", "Het"])

        QtTest.QTest.mouseClick(het, QtCore.Qt.MouseButton.LeftButton, pos=QtCore.QPoint(12, het.height() // 2))
        self.app.processEvents()
        self.assertEqual(assigned, [self.het])                   # the chip body still assigns
        self.assertEqual(panel._hidden_groups(), set())

    def test_hidden_group_leaves_the_pooled_view_and_is_forgotten_when_deleted(self):
        self.load()
        panel = self.panel
        panel.cb_group_view.setChecked(True)
        panel.combo_group_display.setCurrentIndex(panel.combo_group_display.findData("pooled"))
        self.app.processEvents()
        self.assertEqual(panel._last_mat.shape[0], 2 * N)
        panel.set_group_hidden(self.wt, True)
        self.app.processEvents()
        self.assertEqual(panel._last_mat.shape[0], N)
        self.assertTrue(all(str(label).startswith("het") for label in panel._last_display_labels))
        panel.recording_groups.remove_group(self.wt)
        self.assertEqual(panel._hidden_groups(), set())


# The inherited dashboard tests already run in their own module.
for _name in [name for name in dir(dashboard.GroupViewDashboardTests) if name.startswith("test_")]:
    setattr(GroupsMenuTests, _name, None)


if __name__ == "__main__":
    unittest.main()
