"""Integration checks for independent dashboard choices and compact placement."""
import unittest
from unittest.mock import patch
import numpy as np
from PySide6 import QtWidgets
import test_postprocessing_empty_state as fixture
from global_signal_metrics import GLOBAL_SIGNAL_METRICS


class CompactDashboardTests(unittest.TestCase):
    """Use isolated settings, preserving the user's running application state."""

    @classmethod
    def setUpClass(cls):
        cls.app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

    setUp = fixture.PostprocessingEmptyStateTests.setUp
    tearDown = fixture.PostprocessingEmptyStateTests.tearDown

    def load_signals(self):
        """Two contrasting constant recordings make scope errors conspicuous."""
        t = np.arange(0., 20., .1)
        records = [fixture.ProcessedTrial(path=f"fixture{index}.csv", channel_id="AIN01", time=t,
                   raw_signal=np.full(t.size, value), raw_reference=np.zeros(t.size),
                   output=np.full(t.size, value), output_label="dFF") for index, value in enumerate((2., 8.))]
        self.panel.receive_current_processed(records)
        self.panel.combo_individual_file.addItems([self.panel._file_id_for_proc(proc) for proc in records])
        return records

    def test_standard_rows_have_three_plots_and_no_separate_primary_row(self):
        panel = self.panel
        panel._apply_view_layout()
        for row, plots in ((panel.row_heat, [panel.plot_heat, panel.plot_dur, panel.plot_bout_second]),
                           (panel.row_avg_trace, [panel.plot_avg, panel.plot_metrics, panel.plot_global])):
            for plot in plots:
                owner = panel.heat_figure if plot is panel.plot_heat else row
                self.assertGreaterEqual(owner.layout().indexOf(panel._plot_card_by_widget[plot]), 0)
        self.assertTrue(panel.metric_panels_widget.isHidden())
        self.assertLessEqual(panel.row_avg.minimumHeight(), 230)

    def test_global_menu_displays_one_metric_and_respects_individual_scope(self):
        records = self.load_signals()
        panel = self.panel
        panel.combo_individual_file.setCurrentText(panel._file_id_for_proc(records[1]))
        with patch.object(panel, "_schedule_psth") as recompute:
            panel.combo_global_metric.setCurrentIndex(panel.combo_global_metric.findData("median"))
            panel._render_global_metrics()
            recompute.assert_not_called()
        self.assertAlmostEqual(panel._last_global_metrics["median"], 8.)
        self.assertTrue(panel.global_bar_amp.isVisible() is False)
        self.assertTrue(panel.global_bar_freq.isVisible() is False)
        for key in GLOBAL_SIGNAL_METRICS:
            panel.combo_global_metric.setCurrentIndex(panel.combo_global_metric.findData(key))
            self.assertEqual(panel.plot_global.plotItem.titleLabel.text, GLOBAL_SIGNAL_METRICS[key]["label"])

    def test_group_global_values_use_recordings_and_new_choices_persist(self):
        self.load_signals()
        panel = self.panel
        panel.tab_visual_mode.setCurrentIndex(1)
        panel.combo_global_metric.setCurrentIndex(panel.combo_global_metric.findData("median"))
        panel._render_global_metrics()
        self.assertAlmostEqual(panel._last_global_metrics["median"], 5.)
        self.assertEqual(panel._last_global_metrics["median_n"], 2.)
        panel.combo_psth_behavior_metric_second.setCurrentIndex(4)
        settings = panel._collect_settings()
        self.assertEqual(settings["global_metric"], "median")
        self.assertEqual(settings["psth_behavior_metric_second"], "occupancy")

    def test_behavior_menus_render_two_distinct_summaries_without_recompute(self):
        panel = self.panel
        records = [{"file_id": "synthetic", "start": 0., "end": 20.,
                    "onsets": np.array([1., 5., 12.]), "offsets": np.array([2., 7., 15.]),
                    "observed_intervals": np.array([[0., 20.]])}]
        with patch.object(panel, "_psth_behavior_summary_recordings", return_value=records), \
                patch.object(panel, "_schedule_psth") as recompute:
            panel._refresh_psth_duration_view()
            self.assertTrue(panel.plot_dur.property("hasPlotData"))
            self.assertTrue(panel.plot_bout_second.property("hasPlotData"))
            self.assertNotEqual(panel.plot_dur.plotItem.titleLabel.text, panel.plot_bout_second.plotItem.titleLabel.text)
            recompute.assert_not_called()


if __name__ == "__main__":
    unittest.main()
