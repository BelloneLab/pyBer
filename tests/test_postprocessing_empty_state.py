"""Exercise empty/result presentation without changing analysis or user data."""

from contextlib import ExitStack
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("PYBER_SMOKE_TEST", "1")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "pyBer"))

import numpy as np
from PySide6 import QtCore, QtWidgets
from analysis_core import ProcessedTrial
from gui_postprocessing import PostProcessingPanel


class PostprocessingEmptyStateTests(unittest.TestCase):
    """A plot becomes visible only when its own finite data is available."""

    @classmethod
    def setUpClass(cls):
        cls.app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

    def setUp(self):
        self.resources = ExitStack()
        temporary = self.resources.enter_context(tempfile.TemporaryDirectory(prefix="pyber-empty-test-"))
        self.previous_format = QtCore.QSettings.defaultFormat()
        QtCore.QSettings.setDefaultFormat(QtCore.QSettings.Format.IniFormat)
        QtCore.QSettings.setPath(QtCore.QSettings.Format.IniFormat, QtCore.QSettings.Scope.UserScope, temporary)
        self.resources.enter_context(patch.object(PostProcessingPanel, "_restore_project_autosave_if_needed"))
        self.resources.enter_context(patch.object(PostProcessingPanel, "_autosave_project_cache_path",
                                                 return_value=str(Path(temporary) / "autosave.h5")))
        self.panel = PostProcessingPanel()
        self.app.processEvents()

    def tearDown(self):
        self.panel._project_dirty = False
        self.panel._psth_timer.stop()
        self.panel.close()
        self.panel.deleteLater()
        self.app.sendPostedEvents(None, QtCore.QEvent.Type.DeferredDelete)
        self.resources.close()
        QtCore.QSettings.setDefaultFormat(self.previous_format)

    def test_initial_workspace_and_all_layouts_are_empty(self):
        panel = self.panel
        for layout in ("Standard", "All", "Metrics focus", "Heatmap focus"):
            panel.combo_view_layout.setCurrentText(layout)
            self.assertIs(panel._results_stack.currentWidget(), panel._empty_results)
            for plot in panel._plot_card_by_widget:
                self.assertFalse(plot.getPlotItem().isVisible())
            self.assertTrue(panel.heat_colorbar_widget.isHidden())
            self.assertTrue(panel.heat_lut.isHidden())
            self.assertFalse(panel.btn_edit_scale.isEnabled())

    def test_palette_changes_and_scale_toggle_do_not_reveal_placeholders(self):
        panel = self.panel
        for preset in ("Midnight", "Paper", "Sand"):
            panel._set_plot_preset(preset)
            panel.btn_edit_scale.setChecked(True)
            self.assertFalse(panel.plot_heat.getPlotItem().isVisible())
            self.assertTrue(panel.heat_lut.isHidden())
            self.assertTrue(panel.heat_colorbar_widget.isHidden())

    def test_loaded_trace_does_not_imply_computed_psth(self):
        panel = self.panel
        time = np.linspace(0, 10, 101)
        processed = ProcessedTrial(path="fixture.csv", channel_id="AIN01", time=time,
                                   raw_signal=np.sin(time), raw_reference=np.cos(time),
                                   output=np.sin(time), output_label="dFF")
        panel.receive_current_processed([processed])
        self.assertIs(panel._results_stack.currentWidget(), panel._results_scroll)
        self.assertTrue(panel.plot_trace.getPlotItem().isVisible())
        for plot in (panel.plot_heat, panel.plot_avg, panel.plot_metrics):
            self.assertFalse(plot.getPlotItem().isVisible())
        panel.receive_current_processed([])
        self.assertIs(panel._results_stack.currentWidget(), panel._empty_results)
        self.assertFalse(panel.plot_trace.getPlotItem().isVisible())

    def test_results_clear_and_reload_without_losing_color_scale_choice(self):
        panel = self.panel
        time = np.linspace(-2, 3, 101)
        matrix = np.array([np.sin(time), np.sin(time) + 0.2])
        panel.btn_edit_scale.setChecked(True)
        for _ in range(2):
            panel._render_heatmap(matrix, time)
            panel._render_avg(matrix, time)
            panel._render_metrics(matrix, time)
            panel._render_duration_hist(np.array([1.0, 2.0]))
            for plot in (panel.plot_heat, panel.plot_avg, panel.plot_metrics, panel.plot_dur):
                self.assertTrue(plot.getPlotItem().isVisible())
            self.assertFalse(panel.heat_lut.isHidden())
            self.assertTrue(panel.heat_colorbar_widget.isHidden())
            panel._clear_psth_result_view()
            for plot in (panel.plot_heat, panel.plot_avg, panel.plot_metrics, panel.plot_dur):
                self.assertFalse(plot.getPlotItem().isVisible())
            self.assertTrue(panel.heat_lut.isHidden())

    def test_nonfinite_results_do_not_create_fake_axes(self):
        panel = self.panel
        time = np.linspace(-2, 2, 21)
        matrix = np.full((2, 21), np.nan)
        panel._render_heatmap(matrix, time)
        panel._render_avg(matrix, time)
        panel._render_metrics(matrix, time)
        panel._render_duration_hist(np.array([np.nan]))
        panel._render_global_metrics()
        for plot in (panel.plot_heat, panel.plot_avg, panel.plot_metrics, panel.plot_dur, panel.plot_global):
            self.assertFalse(plot.getPlotItem().isVisible())


if __name__ == "__main__":
    unittest.main()
