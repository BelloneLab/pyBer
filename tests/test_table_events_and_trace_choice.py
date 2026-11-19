"""Single events inside the comparison table; trace preview choice in Group scope."""
import unittest

import numpy as np
import pandas as pd

import test_group_view_dashboard as dashboard
import test_postprocessing_empty_state as gui_fixture
from analysis_core import ProcessedTrial
from behavior_zone_dialog import _EVENTS_COLUMN, _EXPANDED_ROLE
from event_list_dialog import EventListPanel
from gui_postprocessing import _behavior_table_info
from PySide6 import QtCore


class TableEventListTests(unittest.TestCase):
    setUpClass = classmethod(gui_fixture.PostprocessingEmptyStateTests.setUpClass.__func__)
    setUp = gui_fixture.PostprocessingEmptyStateTests.setUp
    tearDown = gui_fixture.PostprocessingEmptyStateTests.tearDown

    def _settle(self):
        for _ in range(8):
            self.app.processEvents()
        QtCore.QThread.msleep(260)   # comparison debounce
        for _ in range(8):
            self.app.processEvents()

    def _load(self):
        panel = self.panel
        time = np.arange(0, 60, 0.05)
        panel._processed = [ProcessedTrial(
            path="fiber.csv", channel_id="AIN01", time=time, raw_signal=np.sin(time),
            raw_reference=np.zeros_like(time), output=np.sin(time), output_label="dFF")]
        frame = pd.DataFrame({"time": np.arange(0, 60, 0.5)})
        # grooming: bouts at 5-10, 20-25, 35-40, 50-55 s.
        frame["grooming"] = ((frame["time"] // 5) % 3 == 1).astype(int)
        frame["rearing"] = ((frame["time"] // 5) % 3 == 2).astype(int)
        panel._behavior_sources = {"fiber": _behavior_table_info(frame, "binary_columns", 0.)}
        panel._refresh_behavior_list()
        panel._set_event_category("behavior")
        panel.resize(1500, 900)
        panel.show()
        view = panel.behavior_zone_panel
        view.reload()
        view.clear_selection_button.click()
        listing = view.selectors["behavior"]
        listing.findItems("grooming", QtCore.Qt.MatchFlag.MatchExactly)[0].setCheckState(
            QtCore.Qt.CheckState.Checked)
        self._settle()
        return view

    def test_events_cell_opens_the_rows_event_list_inside_the_table(self):
        view = self._load()
        table = view.table
        self.assertEqual(table.rowCount(), 1)
        cell = table.item(0, _EVENTS_COLUMN)
        n_events = int(cell.text())
        self.assertGreaterEqual(n_events, 3)
        self.assertFalse(cell.data(_EXPANDED_ROLE))

        view._table_cell_clicked(0, 0)                     # other cells do nothing
        self.assertEqual(table.rowCount(), 1)
        view._table_cell_clicked(0, _EVENTS_COLUMN)        # "+" in the Events cell
        self.assertEqual(table.rowCount(), 2)
        self.assertTrue(table.item(0, _EVENTS_COLUMN).data(_EXPANDED_ROLE))
        self.assertEqual(table.columnSpan(1, 0), table.columnCount())
        events = table.cellWidget(1, 0)
        self.assertIsInstance(events, EventListPanel)
        self.assertEqual(events.label, "grooming")
        self.assertGreaterEqual(events.table.rowCount(), n_events)
        self.assertGreater(table.rowHeight(1), table.rowHeight(0))
        view._table_cell_clicked(1, _EVENTS_COLUMN)        # the list row itself is not a toggle
        self.assertEqual(table.rowCount(), 2)

        # Manage a single event from the table: hide it, the count follows, the list stays open.
        events.table.selectRow(0)
        events._set_selected("hidden")
        self._settle()
        self.assertEqual(table.rowCount(), 2)
        self.assertEqual(int(table.item(0, _EVENTS_COLUMN).text()), n_events - 1)
        events = table.cellWidget(1, 0)
        self.assertIn("1 hidden", events.summary.text())
        events._restore_all()
        self._settle()
        self.assertEqual(int(view.table.item(0, _EVENTS_COLUMN).text()), n_events)

        view._table_cell_clicked(0, _EVENTS_COLUMN)        # same cell closes it
        self.assertEqual(view.table.rowCount(), 1)
        self.assertFalse(view.table.item(0, _EVENTS_COLUMN).data(_EXPANDED_ROLE))

    def test_individual_row_lists_only_its_own_recording(self):
        view = self._load()
        panel = self.panel
        other = pd.DataFrame({"time": np.arange(0, 60, 0.5)})
        other["grooming"] = ((other["time"] // 10) % 2 == 1).astype(int)
        panel._behavior_sources["elsewhere"] = _behavior_table_info(other, "binary_columns", 0.)
        source = view._source_for_file("fiber")
        self.assertIs(source, panel._behavior_sources["fiber"])
        events = EventListPanel(panel)
        events.set_label("grooming", only_source=source)
        self.assertEqual({events.table.item(r, 0).text() for r in range(events.table.rowCount())}, {"fiber"})
        events.set_label("grooming")
        self.assertEqual({events.table.item(r, 0).text() for r in range(events.table.rowCount())},
                         {"fiber", "elsewhere"})
        events.deleteLater()


class GroupTracePreviewChoiceTests(dashboard.GroupViewDashboardTests):
    """Reuse the six synthetic WT / Het recordings of the group dashboard tests."""

    def test_right_click_menu_chooses_the_previewed_recording(self):
        self.load()
        panel = self.panel
        panel.tab_visual_mode.setCurrentIndex(1)
        self.app.processEvents()
        self.assertIs(panel._group_trace_preview_proc(), panel._processed[0])
        self.assertIn("wt0", panel.plot_trace.plotItem.titleLabel.text)

        menu = panel.plot_trace.getPlotItem().getViewBox().menu
        panel._inject_trace_preview_menu()
        choices = menu._pyber_trace_preview_menu
        names = [action.text() for action in choices.actions()]
        self.assertEqual(names, [f"wt{i}" for i in range(dashboard.N_PER_GROUP)]
                         + [f"het{i}" for i in range(dashboard.N_PER_GROUP)])
        self.assertEqual([a.text() for a in choices.actions() if a.isChecked()], ["wt0"])
        self.assertIn(menu._pyber_trace_preview_action, menu.actions())

        next(a for a in choices.actions() if a.text() == "het1").trigger()
        self.app.processEvents()
        chosen = next(p for p in panel._processed if p.path == "het1.csv")
        self.assertIs(panel._group_trace_preview_proc(), chosen)
        self.assertIn("het1", panel.plot_trace.plotItem.titleLabel.text)
        panel._inject_trace_preview_menu()
        self.assertEqual([a.text() for a in menu._pyber_trace_preview_menu.actions() if a.isChecked()], ["het1"])
        self.assertEqual(sum(a is menu._pyber_trace_preview_action for a in menu.actions()), 1)

        panel.tab_visual_mode.setCurrentIndex(0)           # Individual scope: no such entry
        panel._inject_trace_preview_menu()
        self.assertIsNone(menu._pyber_trace_preview_action)


for _name in [name for name in dir(dashboard.GroupViewDashboardTests) if name.startswith("test_")]:
    setattr(GroupTracePreviewChoiceTests, _name, None)


if __name__ == "__main__":
    unittest.main()
