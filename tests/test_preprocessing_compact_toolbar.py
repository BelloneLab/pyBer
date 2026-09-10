"""Preprocessing toolbar layout and preservation of editing actions."""
import unittest
from types import SimpleNamespace
from PySide6 import QtCore, QtWidgets
import test_preprocessing_empty_state as fixture
from compact_toolbar_widgets import compact_preprocessing_toolbar


class CompactPreprocessingToolbarTests(unittest.TestCase):
    """Exercise the real dashboard with a minimal workflow-button owner."""
    setUpClass = classmethod(fixture.PreprocessingEmptyStateTests.setUpClass.__func__)

    def setUp(self):
        self.plots = fixture.PlotDashboard()
        self.bar = QtWidgets.QFrame()
        QtWidgets.QHBoxLayout(self.bar)
        self.owner = SimpleNamespace(plots=self.plots, menu_plot_style=QtWidgets.QMenu(self.bar))
        for name in ("btn_workflow_load", "btn_workflow_qc", "btn_workflow_export", "btn_sensor"):
            setattr(self.owner, name, QtWidgets.QPushButton(name, self.bar))
        self.owner.btn_workflow_load.setText("File")
        compact_preprocessing_toolbar(self.owner, self.bar)

    def tearDown(self):
        self.bar.close()
        self.plots.close()
        self.bar.deleteLater()
        self.plots.deleteLater()
        self.app.sendPostedEvents(None, QtCore.QEvent.Type.DeferredDelete)

    def test_controls_stay_on_one_line_without_duplicate_context(self):
        title = "Recording_" + "very_long_name_" * 12 + ".doric"
        self.plots.set_title(title)
        self.plots.set_status("Channel: AIN01 | A/D: DIO01 | Fs: 1000 -> 100 Hz | Mode: dFF")
        self.bar.resize(1050, 44)
        self.bar.show()
        self.app.processEvents()
        self.assertEqual(self.bar.height(), 44)
        self.assertEqual(self.plots.lbl_title.text(), title)
        self.assertEqual(self.plots.lbl_title.toolTip(), title)
        self.assertTrue(self.plots._plot_header.isHidden())
        self.assertTrue(self.plots._plot_tools.isHidden())
        self.assertTrue(self.plots.lbl_title.isHidden())
        self.assertTrue(self.plots.lbl_status.isHidden())
        for index in range(self.bar.layout().count()):
            widget = self.bar.layout().itemAt(index).widget()
            if widget is None:
                continue
            self.assertLessEqual(widget.geometry().right(), self.bar.width())

    def test_selection_and_view_actions_retain_original_signals(self):
        events = []
        self.plots.btn_add_region.setEnabled(True)
        self.plots.btn_clear_regions.setEnabled(True)
        self.plots.btn_box_select.setEnabled(True)
        self.plots.btn_thresholds.setEnabled(True)
        self.plots.manualRegionFromSelectorRequested.connect(lambda: events.append("add"))
        self.plots.clearManualRegionsRequested.connect(lambda: events.append("clear"))
        actions = {action.text(): action for action in self.owner.btn_pre_selection.menu().actions()}
        actions["Add from selector"].trigger()
        actions["Clear manual regions"].trigger()
        self.assertEqual(events, ["add", "clear"])
        actions["Box select"].trigger()
        self.assertTrue(self.plots.btn_box_select.isChecked())
        view = self.owner.btn_pre_view.menu()
        view.aboutToShow.emit()
        next(action for action in view.actions() if action.text() == "Show thresholds").trigger()
        self.assertFalse(self.plots.btn_thresholds.isChecked())


if __name__ == "__main__":
    unittest.main()
