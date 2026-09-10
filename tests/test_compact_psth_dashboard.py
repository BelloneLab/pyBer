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

    def test_shared_cards_keep_related_plots_together(self):
        panel = self.panel
        panel._apply_view_layout()
        self.assertIs(panel._plot_card_by_widget[panel.plot_heat], panel._plot_card_by_widget[panel.plot_avg])
        self.assertIs(panel._plot_card_by_widget[panel.plot_dur], panel._plot_card_by_widget[panel.plot_bout_second])
        self.assertEqual(panel.bout_figure.layout().spacing(), 2)
        self.assertGreaterEqual(panel._view_splitters["comparison"].indexOf(panel._plot_card_by_widget[panel.plot_metrics]), 0)
        self.assertTrue(panel.metric_panels_widget.isHidden())

    def test_shared_time_axes_align_after_resize_zoom_and_scale_changes(self):
        from PySide6 import QtCore
        panel = self.panel
        t = np.linspace(-5., 5., 201)
        matrix = np.random.default_rng(5).normal(size=(27, t.size))
        panel._render_heatmap(matrix, t)
        panel._render_avg(matrix, t)
        panel.row_heat.setParent(None)
        panel.row_heat.show()
        try:
            for width in (1000, 1400):
                panel.row_heat.resize(width, 500)
                for detail in (False, True):
                    panel.btn_edit_scale.setChecked(detail)
                    self.app.processEvents()
                    panel.plot_heat.setXRange(-2., 3., padding=0)
                    self.app.processEvents()
                    np.testing.assert_allclose(panel.plot_heat.viewRange()[0], panel.plot_avg.viewRange()[0], atol=1e-9)
                    for time in (-2., 0., 3.):
                        positions = []
                        for plot in (panel.plot_heat, panel.plot_avg):
                            point = plot.plotItem.vb.mapViewToScene(QtCore.QPointF(time, 0.))
                            positions.append(plot.mapToGlobal(point.toPoint()).x())
                        self.assertLessEqual(abs(positions[0] - positions[1]), 1)
            panel.plot_avg.setXRange(-1., 1., padding=0)
            self.app.processEvents()
            np.testing.assert_allclose(panel.plot_heat.viewRange()[0], panel.plot_avg.viewRange()[0], atol=1e-9)
        finally:
            panel.row_heat.setParent(panel)

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
