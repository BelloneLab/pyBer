"""Exercise real settings, panel selection, clearing and theme behavior."""
import unittest
from unittest.mock import patch
import numpy as np
from PySide6 import QtCore, QtWidgets

import test_postprocessing_empty_state as empty_fixture


class MetricSelectionUITests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

    setUp = empty_fixture.PostprocessingEmptyStateTests.setUp
    tearDown = empty_fixture.PostprocessingEmptyStateTests.tearDown

    def test_metric_selection_reuses_cached_psth_without_recomputing_trials(self):
        panel = self.panel
        panel._last_tvec = np.linspace(-2, 2, 81)
        panel._last_mat = np.arange(6)[:, None] * .1 + np.sin(panel._last_tvec)[None, :]
        with patch.object(panel, "_schedule_psth") as recompute:
            panel._metric_actions["median"].setChecked(True)
            self.assertIn("median", panel._last_metric_panels)
            recompute.assert_not_called()

    def test_three_metric_panels_persist_and_clear_with_results(self):
        panel = self.panel
        with QtCore.QSignalBlocker(panel._metric_actions["median"]):
            panel._metric_actions["median"].setChecked(True)
        with QtCore.QSignalBlocker(panel._metric_actions["peak"]):
            panel._metric_actions["peak"].setChecked(True)
        time = np.linspace(-2, 2, 81)
        matrix = np.arange(6)[:, None] * .1 + np.sin(time)[None, :]
        panel._render_metrics(matrix, time)
        self.assertEqual(list(panel._last_metric_panels), ["auc", "median", "peak"])
        self.assertFalse(panel._extra_metric_plots["median"].isHidden())
        self.assertTrue(panel._extra_metric_plots["mean"].isHidden())
        self.assertEqual(panel._collect_settings()["extra_metrics"], ["median", "peak"])
        for theme in ("Paper", "Midnight"):
            panel._set_plot_preset(theme)
            panel._render_metrics(matrix, time)
            self.assertIsNotNone(panel._extra_metric_plots["median"].result)
        panel._clear_psth_result_view()
        self.assertEqual(panel._last_metric_panels, {})
        self.assertIsNone(panel.plot_metrics.result)
        self.assertIsNone(panel._extra_metric_plots["median"].result)

    def test_primary_comparison_stays_beside_global_and_extras_use_compact_grid(self):
        panel = self.panel
        with QtCore.QSignalBlocker(panel._metric_actions["mean"]):
            panel._metric_actions["mean"].setChecked(True)
        panel.metric_panels_widget.columns = 1
        panel._sync_metric_panel_layout()
        grid = panel.metric_panels_widget.layout()
        primary_card = panel._plot_card_by_widget[panel.plot_metrics]
        extra_card = panel._plot_card_by_widget[panel._extra_metric_plots["mean"]]
        self.assertEqual(panel.row_avg_trace.layout().indexOf(primary_card), 0)
        self.assertIs(grid.itemAtPosition(0, 0).widget(), extra_card)
        self.assertEqual(grid.indexOf(primary_card), -1)
        panel.cb_metrics.setChecked(False)
        panel._sync_metric_panel_layout()
        self.assertTrue(panel.metric_panels_widget.isHidden())


if __name__ == "__main__":
    unittest.main()
