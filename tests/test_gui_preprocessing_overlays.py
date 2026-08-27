"""Regression tests for preprocessing artifact overlays."""

import os
import sys
import unittest

import numpy as np


os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("PYBER_SMOKE_TEST", "1")

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "pyBer"))

from PySide6 import QtWidgets  # noqa: E402

from analysis_core import LoadedDoricFile  # noqa: E402
from gui_preprocessing import PlotDashboard  # noqa: E402
from main import MainWindow  # noqa: E402


class ArtifactOverlayTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

    def test_large_region_set_uses_supported_batched_qt_path(self) -> None:
        dashboard = PlotDashboard()
        time = np.linspace(0.0, 60.0, 600)
        signal = np.sin(time)
        regions = [(float(i), float(i) + 0.2) for i in range(30)]

        dashboard._update_artifact_overlays(time, signal, regions)

        self.assertEqual(len(dashboard._artifact_regions), 1)
        self.assertIsInstance(dashboard._artifact_regions[0], QtWidgets.QGraphicsPathItem)
        self.assertEqual(dashboard._artifact_region_bounds, regions)

        # Selection and cleanup must accept the mixed overlay item type too.
        dashboard.highlight_artifact_regions([regions[0]])
        dashboard._clear_artifact_overlays()
        self.assertEqual(dashboard._artifact_regions, [])
        self.assertEqual(dashboard._artifact_region_bounds, [])
        dashboard.deleteLater()

    def test_large_cut_overlay_keeps_label_positions_and_autorange_finite(self) -> None:
        dashboard = PlotDashboard()
        time = np.linspace(0.0, 60.0, 1201)
        signal = 0.12 + 0.005 * np.sin(time)
        regions = [(float(i), float(i) + 0.35) for i in range(30)]
        cut_signal = signal.copy()
        for start_s, end_s in regions:
            cut_signal[(time >= start_s) & (time <= end_s)] = np.nan

        dashboard.show_raw(time, cut_signal, cut_signal)
        dashboard._update_artifact_overlays(time, cut_signal, regions)

        self.assertEqual(len(dashboard._artifact_labels), len(regions))
        self.assertTrue(
            all(np.isfinite(label.pos().y()) for label in dashboard._artifact_labels)
        )
        dashboard.plot_raw.getViewBox().updateAutoRange()
        x_range, y_range = dashboard.plot_raw.getViewBox().viewRange()
        self.assertTrue(np.all(np.isfinite(x_range)))
        self.assertTrue(np.all(np.isfinite(y_range)))
        dashboard.deleteLater()

    def test_each_newly_opened_file_becomes_the_active_selection(self) -> None:
        window = MainWindow()
        first = os.path.abspath("first_open.doric")
        second = os.path.abspath("second_open.doric")

        def _loaded(path: str) -> LoadedDoricFile:
            time = np.linspace(0.0, 10.0, 501)
            signal = 1.0 + 0.01 * np.sin(time)
            reference = 0.5 + 0.005 * np.cos(time)
            return LoadedDoricFile(
                path=path,
                channels=["AIN01"],
                time_by_channel={"AIN01": time},
                signal_by_channel={"AIN01": signal},
                reference_by_channel={"AIN01": reference},
                digital_time=None,
                digital_by_name={},
                trigger_time_by_name={},
                trigger_by_name={},
            )

        window.processor.load_file = _loaded
        window._add_files([first])
        window._preview_timer.stop()
        self.assertEqual(window.file_panel.selected_paths(), [first])
        self.assertEqual(window._current_path, first)

        window._add_files([second])
        window._preview_timer.stop()
        self.assertEqual(window.file_panel.selected_paths(), [second])
        self.assertEqual(window._current_path, second)

        window._busy_poll.stop()
        window.hide()
        window.deleteLater()


if __name__ == "__main__":
    unittest.main()
