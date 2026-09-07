"""Exercise the preprocessing workspace's empty, loading and ready states."""

import os
import sys
import unittest

import numpy as np

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("PYBER_SMOKE_TEST", "1")
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "pyBer"))

from PySide6 import QtWidgets  # noqa: E402
from analysis_core import ProcessedTrial  # noqa: E402
from gui_preprocessing import PlotDashboard  # noqa: E402


class PreprocessingEmptyStateTests(unittest.TestCase):
    """No sample axes or stale traces should imply a loaded recording."""

    @classmethod
    def setUpClass(cls):
        """Reuse one application for all offscreen widget checks."""
        cls.app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

    def setUp(self):
        """Build a real dashboard so graphics visibility is exercised."""
        self.dashboard = PlotDashboard()
        self.dashboard.resize(1100, 800)
        self.dashboard.show()
        self.app.processEvents()

    def tearDown(self):
        """Dispose the dashboard and any deferred graphics resources."""
        self.dashboard.hide()
        self.dashboard.deleteLater()
        self.app.processEvents()

    @staticmethod
    def _processed():
        """Construct a deterministic fixture without touching user data."""
        time = np.linspace(0, 10, 101)
        signal = 2 + np.sin(time)
        reference = 1 + 0.1 * np.cos(time)
        return ProcessedTrial(
            path="empty_state_fixture.doric", channel_id="AIN01", time=time,
            raw_signal=signal, raw_reference=reference,
            sig_f=signal, ref_f=reference,
            baseline_sig=np.ones_like(time), baseline_ref=np.ones_like(time),
            output=signal - 1, output_label="dFF",
            artifact_regions_sec=[(2, 3)],
            dio=(time >= 5).astype(float), dio_name="DIO1",
        )

    def _assert_empty(self):
        """Check the whole canvas, including independent digital overlays."""
        dashboard = self.dashboard
        self.assertIs(dashboard.plot_workspace.currentWidget(), dashboard.plot_empty_state)
        for plot in (dashboard.plot_raw, dashboard.plot_proc, dashboard.plot_out):
            self.assertFalse(plot.getPlotItem().isVisible())
        for view in (dashboard.vb_dio_raw, dashboard.vb_dio_proc, dashboard.vb_dio_out):
            self.assertFalse(view.isVisible())
        self.assertFalse(dashboard.selector.isVisible())
        self.assertFalse(dashboard.btn_add_region.isEnabled())
        self.assertFalse(dashboard.btn_box_select.isEnabled())

    def test_startup_has_one_quiet_empty_surface(self):
        """The initial scene contains neither default axes nor selection fill."""
        self._assert_empty()

    def test_theme_changes_do_not_reveal_empty_plots(self):
        """Appearance settings apply while keeping absent data hidden."""
        for background in ("white", "dark"):
            self.dashboard.set_plot_appearance(background, True)
            self.app.processEvents()
            self._assert_empty()

    def test_raw_loading_does_not_expose_uncomputed_panels(self):
        """Raw data can appear before the asynchronous processing result."""
        processed = self._processed()
        dashboard = self.dashboard
        dashboard.show_raw(
            processed.time, processed.raw_signal, processed.raw_reference,
            dio=processed.dio,
        )
        self.app.processEvents()
        self.assertIs(dashboard.plot_workspace.currentWidget(), dashboard.plot_splitter)
        self.assertTrue(dashboard.plot_raw.getPlotItem().isVisible())
        self.assertTrue(dashboard.vb_dio_raw.isVisible())
        for plot, overlay in (
            (dashboard.plot_proc, dashboard.vb_dio_proc),
            (dashboard.plot_out, dashboard.vb_dio_out),
        ):
            self.assertFalse(plot.getPlotItem().isVisible())
            self.assertFalse(overlay.isVisible())

    def test_load_clear_reload_restores_plots_and_appearance(self):
        """Clearing removes every previous trace, then loading restores charts."""
        dashboard = self.dashboard
        processed = self._processed()
        dashboard.update_plots(processed)
        self.app.processEvents()
        for plot in (dashboard.plot_raw, dashboard.plot_proc, dashboard.plot_out):
            self.assertTrue(plot.getPlotItem().isVisible())
        self.assertTrue(dashboard._artifact_regions)
        dashboard.btn_box_select.setChecked(True)
        dashboard.clear_plots()
        self._assert_empty()
        self.assertFalse(dashboard._artifact_regions)
        self.assertFalse(dashboard._raw_vb._drag_enabled)
        for plot in (dashboard.plot_raw, dashboard.plot_proc, dashboard.plot_out):
            for curve in plot.listDataItems():
                x, _ = curve.getData()
                self.assertTrue(x is None or len(x) == 0)
        dashboard.set_plot_appearance("white", False)
        dashboard.update_plots(processed)
        self.app.processEvents()
        self.assertEqual(dashboard.plot_background_mode(), "white")
        self.assertFalse(dashboard.plot_grid_visible())
        self.assertTrue(dashboard.plot_out.getPlotItem().isVisible())
        np.testing.assert_allclose(dashboard.curve_out.getData()[1], processed.output)
        self.assertTrue(dashboard.selector.isVisible())

    def test_legacy_no_data_call_clears_all_panels(self):
        """The existing project reset API must also discard processed output."""
        self.dashboard.update_plots(self._processed())
        self.dashboard.show_raw()
        self._assert_empty()
        self.assertIsNone(self.dashboard.curve_out.getData()[0])

    def test_empty_and_nonfinite_raw_inputs_stay_blank(self):
        """An empty selection or invalid recording cannot activate the canvas."""
        for time, values in (([], []), ([0, 1], [np.nan, np.nan])):
            self.dashboard.show_raw(time, values, values)
            self._assert_empty()


if __name__ == "__main__":
    unittest.main()
