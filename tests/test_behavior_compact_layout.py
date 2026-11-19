"""The behavior comparison takes only the height its content needs.

No behavior file: one toolbar row. Behaviors loaded: toolbar plus chip rows.
Results: plot and table appear and the plots below keep most of the height.
"""
import unittest

import numpy as np
import pandas as pd

import test_postprocessing_empty_state as gui_fixture
from analysis_core import ProcessedTrial
from gui_postprocessing import _behavior_table_info
from PySide6 import QtCore


class BehaviorCompactLayoutTests(unittest.TestCase):
    setUpClass = classmethod(gui_fixture.PostprocessingEmptyStateTests.setUpClass.__func__)
    setUp = gui_fixture.PostprocessingEmptyStateTests.setUp
    tearDown = gui_fixture.PostprocessingEmptyStateTests.tearDown

    def _settle(self):
        for _ in range(8):
            self.app.processEvents()
        QtCore.QThread.msleep(250)  # comparison debounce timer
        for _ in range(8):
            self.app.processEvents()

    def _show(self, width=2400, height=900):
        panel = self.panel
        time = np.arange(0, 60, 0.05)
        panel._processed = [ProcessedTrial(
            path="fiber.csv", channel_id="AIN01", time=time,
            raw_signal=np.sin(time), raw_reference=np.zeros_like(time),
            output=np.sin(time), output_label="dFF")]
        panel._set_event_category("behavior")
        panel.resize(width, height)
        panel.show()
        self._settle()
        return panel.behavior_zone_panel

    def _load_behaviors(self):
        frame = pd.DataFrame({"time": np.arange(0, 60, 0.5)})
        for index, name in enumerate(("attack", "grooming", "rearing")):
            frame[name] = ((frame["time"] // 5) % 3 == index).astype(int)
        self.panel._behavior_sources = {"fiber": _behavior_table_info(frame, "binary_columns", 0.)}
        self.panel._refresh_behavior_list()
        self.panel.behavior_zone_panel.reload()
        self._settle()

    def test_empty_state_is_a_single_toolbar_row(self):
        view = self._show()
        listing = view.selectors["behavior"]
        self.assertTrue(listing.isHidden())
        self.assertTrue(view.results_hidden())
        self.assertFalse(view._toolbar_folded)
        for widget in (view.searches["behavior"], view.select_visible_button,
                       view.clear_selection_button, view.combine_button):
            self.assertTrue(widget.isHidden())
        comparison, plots = self.panel._behavior_workspace.sizes()
        self.assertLess(comparison, 80)
        self.assertGreater(plots, 8 * comparison)
        self.assertIn("Add a behavior file", view.status.text())
        self.assertEqual(view.status.toolTip(), view.status.text())

    def test_chips_appear_with_behaviors_and_results_only_when_selected(self):
        view = self._show()
        self._load_behaviors()
        view.clear_selection_button.click()
        self._settle()
        listing = view.selectors["behavior"]
        self.assertFalse(listing.isHidden())
        self.assertFalse(view.searches["behavior"].isHidden())
        self.assertTrue(view.results_hidden())
        # Three chips fit on one row: the list is one chip row tall.
        self.assertLess(listing.height(), 2 * listing.gridSize().height())
        chips_only = self.panel._behavior_workspace.sizes()[0]
        self.assertLess(chips_only, 130)

        listing.findItems("grooming", QtCore.Qt.MatchFlag.MatchExactly)[0].setCheckState(
            QtCore.Qt.CheckState.Checked)
        self._settle()
        self.assertFalse(view.results_hidden())
        self.assertGreater(view.table.rowCount(), 0)
        comparison, plots = self.panel._behavior_workspace.sizes()
        self.assertGreater(comparison, chips_only + 150)
        self.assertGreater(plots, comparison)

        view.clear_selection_button.click()
        self._settle()
        self.assertTrue(view.results_hidden())
        self.assertLess(self.panel._behavior_workspace.sizes()[0], 130)

    def test_toolbar_folds_to_two_rows_when_narrow(self):
        view = self._show(width=2400)
        self._load_behaviors()
        self.assertFalse(view._toolbar_folded)
        self.panel.resize(760, 900)
        self._settle()
        self.assertTrue(view._toolbar_folded)
        # Folded controls stay inside the panel instead of forcing a scrollbar.
        self.assertLessEqual(view._action_group.geometry().right(), view.width())
        self.assertLessEqual(view._selection_group.sizeHint().width(), view.width())


if __name__ == "__main__":
    unittest.main()
