"""Inline event list (chip '+'): events listed; hide/remove exclude them from PSTH; restore."""
import tempfile
import unittest

import numpy as np
from PySide6 import QtCore, QtTest

import test_behavior_import_gui as base


class EventListTests(unittest.TestCase):
    setUpClass = classmethod(base.gui_fixture.PostprocessingEmptyStateTests.setUpClass.__func__)
    setUp = base.gui_fixture.PostprocessingEmptyStateTests.setUp
    tearDown = base.gui_fixture.PostprocessingEmptyStateTests.tearDown
    _onset_fixture = base.BehaviorImportGuiTests._onset_fixture
    _load_clicked = base.BehaviorImportGuiTests._load_clicked

    def _rows(self, table):
        return [(table.item(r, 2).text(), table.item(r, 4).text(), table.item(r, 5).text())
                for r in range(table.rowCount())]

    def test_chip_plus_opens_inline_list_and_exclusions_apply(self):
        panel = self.panel
        with tempfile.TemporaryDirectory() as directory:
            _fiber, onsets, states = self._onset_fixture(directory)
            self._load_clicked(states)
            self._load_clicked(onsets)
            panel._set_event_category("behavior")
            panel.resize(1400, 900)
            panel.show()
            self.app.processEvents()
            view = panel.behavior_zone_panel
            listing = view.selectors["behavior"]
            item = listing.findItems("groom", QtCore.Qt.MatchFlag.MatchExactly)[0]
            checked = item.checkState()
            from behavior_zone_dialog import _plus_rect
            QtTest.QTest.mouseClick(listing.viewport(), QtCore.Qt.MouseButton.LeftButton,
                                    pos=_plus_rect(listing.visualItemRect(item)).center())
            self.assertEqual(item.checkState(), checked)          # + does not toggle the chip
            self.assertFalse(view.event_list.isHidden())
            self.assertEqual(self._rows(view.event_list.table), [("10.000", "3.000", "used")])
            view._toggle_event_list("groom")                     # same chip closes it
            self.assertTrue(view.event_list.isHidden())
            view._toggle_event_list("injection")
            table = view.event_list.table
            self.assertEqual([row[0] for row in self._rows(table)], ["5.600", "15.600"])
            panel.combo_align.setCurrentText("Behavior (CSV/XLSX)")
            panel.combo_behavior_name.setCurrentText("injection")
            table.selectRow(0)
            view.event_list._set_selected("hidden")
            times, _ = panel._get_events_for_proc(panel._processed[0])
            np.testing.assert_equal(times, [15.6])
            self.assertIn("1 hidden", view.event_list.summary.text())
            row = next(r for r in range(table.rowCount()) if table.item(r, 2).text() == "15.600")
            table.clearSelection()
            table.selectRow(row)
            view.event_list._set_selected("removed")
            times, _ = panel._get_events_for_proc(panel._processed[0])
            self.assertEqual(times.size, 0)
            self.assertNotIn("15.600", [r[0] for r in self._rows(table)])
            view.event_list._restore_all()
            times, _ = panel._get_events_for_proc(panel._processed[0])
            np.testing.assert_equal(times, [5.6, 15.6])
            source = next(iter(panel._behavior_sources.values()))
            np.testing.assert_equal(source["event_behaviors"]["injection"]["on"], [5.6, 15.6])


if __name__ == "__main__":
    unittest.main()
