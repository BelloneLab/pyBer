"""Event list window: all events listed; hide/remove exclude them from PSTH; restore."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from PySide6 import QtWidgets

import test_behavior_import_gui as base


class EventListTests(unittest.TestCase):
    setUpClass = classmethod(base.gui_fixture.PostprocessingEmptyStateTests.setUpClass.__func__)
    setUp = base.gui_fixture.PostprocessingEmptyStateTests.setUp
    tearDown = base.gui_fixture.PostprocessingEmptyStateTests.tearDown
    _onset_fixture = base.BehaviorImportGuiTests._onset_fixture
    _load_clicked = base.BehaviorImportGuiTests._load_clicked

    def test_hide_remove_restore_events(self):
        panel = self.panel
        with tempfile.TemporaryDirectory() as directory:
            _fiber, onsets, states = self._onset_fixture(directory)
            self._load_clicked(states)
            self._load_clicked(onsets)
            panel._open_event_list()
            window = panel._event_list_window
            table = window.table
            rows = {(table.item(r, 1).text(), table.item(r, 3).text()) for r in range(table.rowCount())}
            self.assertIn(("injection", "5.600"), rows)
            self.assertIn(("groom", "10.000"), rows)
            groom = next(r for r in range(table.rowCount()) if table.item(r, 1).text() == "groom")
            self.assertEqual(table.item(groom, 5).text(), "3.000")   # bout duration
            source = next(iter(panel._behavior_sources.values()))
            panel.combo_behavior_name.setCurrentText("injection")
            first = next(r for r in range(table.rowCount())
                         if table.item(r, 1).text() == "injection" and table.item(r, 3).text() == "5.600")
            table.selectRow(first)
            window._set_selected("hidden")
            times, _ = panel._get_events_for_proc(panel._processed[0])
            np.testing.assert_equal(times, [15.6])
            self.assertIn("1 hidden", window.summary.text())
            second = next(r for r in range(table.rowCount())
                          if table.item(r, 1).text() == "injection" and table.item(r, 3).text() == "15.600")
            table.clearSelection()
            table.selectRow(second)
            window._set_selected("removed")
            times, _ = panel._get_events_for_proc(panel._processed[0])
            self.assertEqual(times.size, 0)
            self.assertNotIn("15.600", [table.item(r, 3).text() for r in range(table.rowCount())
                                        if table.item(r, 1).text() == "injection"])
            window._restore_all()
            times, _ = panel._get_events_for_proc(panel._processed[0])
            np.testing.assert_equal(times, [5.6, 15.6])
            np.testing.assert_equal(source["event_behaviors"]["injection"]["on"], [5.6, 15.6])


if __name__ == "__main__":
    unittest.main()
