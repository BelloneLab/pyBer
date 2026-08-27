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

from analysis_core import LoadedDoricFile, ProcessedTrial  # noqa: E402
from gui_preprocessing import ArtifactPanel, PlotDashboard  # noqa: E402
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

    def test_disabled_auto_region_keeps_original_table_and_overlay_ids(self) -> None:
        panel = ArtifactPanel()
        regions = [(10.0, 10.2), (20.0, 20.2), (30.0, 30.2), (40.0, 40.2)]
        panel.set_auto_regions(
            regions,
            checked_regions=[regions[0], regions[2], regions[3]],
        )
        panel.set_regions([(25.0, 25.2)])

        self.assertEqual(
            panel.active_overlay_entries(),
            [
                ("1", 10.0, 10.2),
                ("M1", 25.0, 25.2),
                ("3", 30.0, 30.2),
                ("4", 40.0, 40.2),
            ],
        )
        self.assertEqual(
            [panel.table_auto.item(row, 0).text() for row in range(4)],
            ["1", "2", "3", "4"],
        )
        self.assertEqual(panel.table.item(0, 0).text(), "M1")

        dashboard = PlotDashboard()
        time = np.linspace(0.0, 50.0, 1001)
        signal = 0.12 + 0.002 * np.sin(time)
        entries = panel.active_overlay_entries()
        dashboard._update_artifact_overlays(
            time,
            signal,
            [(a, b) for _label, a, b in entries],
            [label for label, _a, _b in entries],
        )
        overlay_text = [label.toPlainText() for label in dashboard._artifact_labels]
        self.assertEqual(overlay_text, ["1", "M1", "3", "4"])

        # Re-enabling row 2 restores label 2 at the same interval. It does not
        # rename any of the later artifacts.
        panel.set_auto_regions(regions, checked_regions=regions)
        self.assertEqual(
            [label for label, _a, _b in panel.active_overlay_entries() if not label.startswith("M")],
            ["1", "2", "3", "4"],
        )
        dashboard.deleteLater()
        panel.deleteLater()

    def test_artifact_toggle_refresh_preserves_all_linked_x_ranges(self) -> None:
        dashboard = PlotDashboard()
        dashboard.resize(1400, 900)
        dashboard.show()
        self.app.processEvents()

        time = np.linspace(0.083, 599.783, 72000)
        signal = 0.12 + 0.002 * np.sin(time)
        reference = 0.08 + 0.001 * np.cos(time)
        output = 0.01 * np.sin(time / 5.0)
        processed = ProcessedTrial(
            path="toggle_test.doric",
            channel_id="AIN01",
            time=time,
            raw_signal=signal,
            raw_reference=reference,
            sig_f=signal,
            ref_f=reference,
            baseline_sig=np.full_like(time, 0.12),
            baseline_ref=np.full_like(time, 0.08),
            output=output,
            output_label="dFF",
            # A realistic detector context used to expand the output title to
            # several thousand pixels and corrupt linked-view scaling.
            output_context="Artifacts: Interpolate | " + ("shared artifact details; " * 80),
            artifact_regions_sec=[(14.0, 16.0), (487.0, 488.0), (499.0, 501.0)],
        )
        full_entries = [("1", 14.0, 16.0), ("2", 487.0, 488.0), ("3", 499.0, 501.0)]
        enabled_entries = [("1", 14.0, 16.0), ("3", 499.0, 501.0)]

        dashboard.update_plots(processed, artifact_overlay_entries=full_entries)
        self.app.processEvents()
        target = (10.63, 15.36)
        dashboard.set_xrange_all(*target)
        self.app.processEvents()

        # This is the plot-side equivalent of unchecking artifact 2 and
        # receiving the asynchronous preview result.
        dashboard.update_plots(
            processed,
            preserve_view=True,
            artifact_overlay_entries=enabled_entries,
        )
        self.app.processEvents()

        for plot in (dashboard.plot_raw, dashboard.plot_proc, dashboard.plot_out):
            self.assertTrue(
                np.allclose(plot.getViewBox().viewRange()[0], target, rtol=0.0, atol=1e-9),
                msg=f"range drifted: {plot.getViewBox().viewRange()[0]}",
            )
        widths = [
            plot.getViewBox().sceneBoundingRect().width()
            for plot in (dashboard.plot_raw, dashboard.plot_proc, dashboard.plot_out)
        ]
        self.assertLessEqual(max(widths) - min(widths), 1.0, msg=f"view widths={widths}")
        dashboard.hide()
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
