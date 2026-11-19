"""Recording lists stay side by side: narrow drawer, own window, docked back."""
import unittest

import test_postprocessing_empty_state as gui_fixture
from adaptive_density import AdaptiveDensity
from PySide6 import QtCore, QtWidgets

ROW = QtWidgets.QBoxLayout.Direction.LeftToRight


class RecordingsWindowLayoutTests(unittest.TestCase):
    setUpClass = classmethod(gui_fixture.PostprocessingEmptyStateTests.setUpClass.__func__)
    setUp = gui_fixture.PostprocessingEmptyStateTests.setUp
    tearDown = gui_fixture.PostprocessingEmptyStateTests.tearDown

    def _settle(self):
        for _ in range(40):
            self.app.processEvents()
            QtCore.QThread.msleep(10)

    def test_lists_never_stack_and_keep_a_readable_width(self):
        panel = self.panel
        panel.resize(1400, 900)
        panel.show()
        panel._section_buttons["setup"].setChecked(True)
        self._settle()
        splitter = panel._dockarea_splitter
        total = sum(splitter.sizes())
        row = panel._recording_lists_layout
        scroll = panel._section_scroll_hosts["setup"]
        density = scroll.findChild(AdaptiveDensity)
        for width in (600, 400, 280):
            splitter.setSizes([width, total - width])
            self._settle()
            self.assertEqual(row.direction(), ROW, width)
            self.assertFalse(density._stacked_first)
            for listing in (panel.list_preprocessed, panel.list_behaviors):
                self.assertGreaterEqual(listing.width(), 60, width)
        # Too narrow for the row: the page scrolls sideways, nothing is cut off.
        self.assertEqual(scroll.horizontalScrollBarPolicy(), QtCore.Qt.ScrollBarPolicy.ScrollBarAsNeeded)

        section = panel.recordings_section
        section.detach()
        self._settle()
        try:
            self.assertEqual(row.direction(), ROW)
        finally:
            section.attach()
        self._settle()
        self.assertEqual(row.direction(), ROW)


if __name__ == "__main__":
    unittest.main()
