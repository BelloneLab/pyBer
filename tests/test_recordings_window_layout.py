"""Recording lists: stacked in a narrow drawer, side by side in their own window."""
import unittest

import test_postprocessing_empty_state as gui_fixture
from adaptive_density import AdaptiveDensity
from PySide6 import QtCore, QtWidgets

ROW = QtWidgets.QBoxLayout.Direction.LeftToRight
STACKED = QtWidgets.QBoxLayout.Direction.TopToBottom


class RecordingsWindowLayoutTests(unittest.TestCase):
    setUpClass = classmethod(gui_fixture.PostprocessingEmptyStateTests.setUpClass.__func__)
    setUp = gui_fixture.PostprocessingEmptyStateTests.setUp
    tearDown = gui_fixture.PostprocessingEmptyStateTests.tearDown

    def _settle(self):
        for _ in range(40):
            self.app.processEvents()
            QtCore.QThread.msleep(10)

    def test_lists_are_side_by_side_in_the_window_and_restack_when_docked(self):
        panel = self.panel
        panel.resize(1400, 900)
        panel.show()
        panel._section_buttons["setup"].setChecked(True)
        self._settle()
        splitter = panel._dockarea_splitter
        total = sum(splitter.sizes())
        splitter.setSizes([280, total - 280])
        self._settle()
        row = panel._recording_lists_layout
        density = panel._section_scroll_hosts["setup"].findChild(AdaptiveDensity)
        self.assertEqual(row.direction(), STACKED)          # narrow drawer: stacked, readable

        section = panel.recordings_section
        section.detach()
        self._settle()
        try:
            self.assertTrue(section.is_detached())
            self.assertEqual(row.direction(), ROW)           # own window: side by side
            self.assertFalse(density._stacked_first)
        finally:
            section.attach()
        self._settle()
        self.assertFalse(section.is_detached())
        self.assertEqual(row.direction(), STACKED)           # back in the narrow drawer

        splitter.setSizes([900, total - 900])
        self._settle()
        if panel._section_scroll_hosts["setup"].viewport().width() > 700:
            self.assertEqual(row.direction(), ROW)           # wide drawer: side by side again


if __name__ == "__main__":
    unittest.main()
