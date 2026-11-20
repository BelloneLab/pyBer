"""Several checked behaviors: one column each (heatmap, mean PSTH, metric) and colored trace events."""
import unittest

import numpy as np
import pandas as pd
import pyqtgraph as pg
from PySide6 import QtCore, QtWidgets

import test_postprocessing_empty_state as gui_fixture
from analysis_core import ProcessedTrial
from gui_postprocessing import _behavior_table_info
from multi_behavior_view import (BAND_SPACING, BEHAVIOR_COLORS, COLORBAR_WIDTH, MIN_COLUMN_WIDTH,
                                 MultiBehaviorView, behavior_color, columns_per_band, shared_range)
from postprocessing_core import mean_sem
from postprocessing_style import POSTPROCESSING_PRESETS


def _entries(count, rows=6, seed=2):
    """Synthetic behaviors with different response sizes (clearly not real data)."""
    rng = np.random.default_rng(seed)
    tvec = np.linspace(-2, 5, 141)
    out = []
    for index in range(count):
        matrix = rng.normal(0, 0.2, (rows + index, tvec.size)) + (index + 1) * (tvec > 0)
        out.append({"name": f"behavior {index + 1} with a rather long descriptive name", "color": behavior_color(index),
                    "matrix": matrix, "tvec": tvec, "labels": [f"Trial {i + 1}" for i in range(rows + index)],
                    "row_unit": "trials", "metric": None})
    return out


class LayoutRuleTests(unittest.TestCase):
    """Pure rules: how many columns fit, and the range shared by the mean PSTHs."""

    def test_columns_wrap_instead_of_shrinking_below_the_minimum(self):
        one = COLORBAR_WIDTH + MIN_COLUMN_WIDTH
        self.assertEqual(columns_per_band(one, 4), 1)
        self.assertEqual(columns_per_band(one + MIN_COLUMN_WIDTH + BAND_SPACING, 4), 2)
        self.assertEqual(columns_per_band(10_000, 4), 4)          # never more than the behaviors
        self.assertEqual(columns_per_band(10, 4), 1)               # never less than one
        self.assertEqual(columns_per_band(10_000, 0), 1)

    def test_shared_range_covers_every_mean_and_sem(self):
        entries = _entries(3)
        low, high = shared_range(entries, include_rows=False)
        for entry in entries:
            mean, sem, _ = mean_sem(entry["matrix"])
            self.assertLessEqual(low, float(np.nanmin(mean - sem)))
            self.assertGreaterEqual(high, float(np.nanmax(mean + sem)))
        wide_low, wide_high = shared_range(entries, include_rows=True)
        self.assertLessEqual(wide_low, low)
        self.assertGreaterEqual(wide_high, high)
        self.assertIsNone(shared_range([{"matrix": None}], include_rows=False))
        self.assertEqual(behavior_color(len(BEHAVIOR_COLORS)), BEHAVIOR_COLORS[0])


class MultiBehaviorWidgetTests(unittest.TestCase):
    """The widget alone: shared scales, wrapping, nothing cut."""

    @classmethod
    def setUpClass(cls):
        cls.app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

    def setUp(self):
        self.view = MultiBehaviorView()
        self.view.apply_theme(POSTPROCESSING_PRESETS["Midnight"])
        self.view.resize(1400, 700)
        self.view.show()
        self.app.processEvents()

    def tearDown(self):
        self.view.close()
        self.view.deleteLater()
        self.app.sendPostedEvents(None, QtCore.QEvent.Type.DeferredDelete)

    def show(self, count, **view):
        entries = _entries(count)
        self.view.show_entries(entries, color_map=pg.colormap.get("viridis"), levels=(-1.0, 4.0), units="z",
                               time_label="Time (s)", trace_view=view or {"show": "average"},
                               missing_mode="interpolate", missing_trim=True)
        for _ in range(6):
            self.app.processEvents()
        return entries

    def test_one_column_per_behavior_on_shared_scales(self):
        entries = self.show(3)
        columns = self.view.columns[:3]
        self.assertEqual([column.name for column in columns], [entry["name"] for entry in entries])
        self.assertTrue(all(column.isVisible() for column in columns))
        # Same color limits for every heatmap, shown by the single color bar.
        for column in columns:
            np.testing.assert_allclose(column.image.getLevels(), (-1.0, 4.0))
        np.testing.assert_allclose(self.view.colorbar.levels(), (-1.0, 4.0))
        self.assertTrue(self.view.colorbar_host.isVisible())
        # Same y range for every mean PSTH, each in its behavior color.
        ranges = [column.psth.getPlotItem().getViewBox().viewRange()[1] for column in columns]
        for y_range in ranges[1:]:
            np.testing.assert_allclose(y_range, ranges[0])
        for index, column in enumerate(columns):
            self.assertEqual(column.mean.opts["pen"].color().getRgb()[:3], behavior_color(index))
            mean, sem, _ = mean_sem(entries[index]["matrix"])
            np.testing.assert_allclose(column.mean.getData()[1], mean)
            self.assertFalse(column.sem_band.path().isEmpty())           # the SEM band is really drawn
        # The y label is written once per band; the header never loses its text.
        self.assertTrue(columns[0].psth.getAxis("left").labelText)
        self.assertFalse(columns[1].psth.getAxis("left").labelText)
        self.assertEqual(columns[0].header.toolTip(), columns[0].header.full_text())
        self.assertIn("n=6 trials", columns[0].header.full_text())

    def test_narrow_width_wraps_columns_and_reserves_their_height(self):
        self.show(4)
        self.assertEqual(self.view._per_band, 4)
        one_band = self.view.required_height()
        emitted = []
        self.view.heightChanged.connect(lambda: emitted.append(self.view.bands()))
        self.view.resize(COLORBAR_WIDTH + 2 * MIN_COLUMN_WIDTH + BAND_SPACING + 20, 700)
        for _ in range(6):
            self.app.processEvents()
        self.assertEqual(self.view._per_band, 2)
        self.assertEqual(self.view.bands(), 2)
        self.assertGreater(self.view.required_height(), 1.9 * one_band)
        self.assertEqual(self.view.minimumHeight(), self.view.required_height())
        self.assertEqual(emitted[-1], 2)
        for column in self.view.columns[:4]:
            self.assertGreaterEqual(column.width(), MIN_COLUMN_WIDTH - 2)
        # First column of each band carries the axis label.
        labelled = [bool(column.psth.getAxis("left").labelText) for column in self.view.columns[:4]]
        self.assertEqual(labelled, [True, False, True, False])

    def test_individual_traces_and_missing_behavior(self):
        entries = self.show(2, show="both", color="fade", style="dash", legend=False)
        column = self.view.columns[0]
        self.assertEqual(len(column.traces.curves), entries[0]["matrix"].shape[0])
        self.assertEqual({curve.opts["pen"].style() for curve in column.traces.curves},
                         {QtCore.Qt.PenStyle.DashLine})
        empty = {"name": "never happens", "color": behavior_color(1), "matrix": None, "tvec": None, "labels": [],
                 "row_unit": "trials", "metric": None}
        self.view.show_entries([entries[0], empty], color_map=pg.colormap.get("viridis"), levels=(0.0, 1.0),
                               units="z", time_label="Time (s)", trace_view={"show": "average"},
                               missing_mode="interpolate", missing_trim=True)
        self.app.processEvents()
        self.assertIn("no events", self.view.columns[1].header.full_text())
        self.assertFalse(self.view.columns[1].psth.property("hasPlotData"))
        self.assertTrue(self.view.columns[0].psth.property("hasPlotData"))
        self.view.clear()
        self.assertEqual(self.view.entries, [])


class MultiBehaviorPanelTests(unittest.TestCase):
    """Checking two behaviors swaps the standard row for the columns, and back."""

    setUpClass = classmethod(gui_fixture.PostprocessingEmptyStateTests.setUpClass.__func__)
    setUp = gui_fixture.PostprocessingEmptyStateTests.setUp
    tearDown = gui_fixture.PostprocessingEmptyStateTests.tearDown

    def settle(self):
        for _ in range(3):
            for _ in range(8):
                self.app.processEvents()
            QtCore.QThread.msleep(160)        # debounce timers
        for _ in range(8):
            self.app.processEvents()

    def load(self):
        panel = self.panel
        time = np.arange(0, 120, 0.05)
        signal = np.sin(time / 3.0) + 0.1 * np.random.default_rng(1).normal(size=time.size)
        panel._processed = [ProcessedTrial(path="fiber.csv", channel_id="AIN01", time=time, raw_signal=signal,
                                           raw_reference=np.zeros_like(time), output=signal, output_label="dFF")]
        panel._update_file_lists()
        panel._set_event_category("behavior")
        frame = pd.DataFrame({"time": np.arange(0, 120, 0.5)})
        for index, name in enumerate(("attack", "grooming", "rearing")):
            frame[name] = ((frame["time"] // 5) % 3 == index).astype(int)
        panel._behavior_sources = {"fiber": _behavior_table_info(frame, "binary_columns", 0.)}
        panel._refresh_behavior_list()
        panel.combo_align.setCurrentText("Behavior (CSV/XLSX)")
        panel.behavior_zone_panel.reload()
        panel.spin_pre.setValue(2.0)
        panel.spin_post.setValue(3.0)
        panel.spin_b0.setValue(-2.0)
        panel.spin_b1.setValue(0.0)
        panel.cb_filter_events.setChecked(False)
        panel.resize(1500, 950)
        panel.show()
        self.settle()
        return panel.behavior_zone_panel.selectors["behavior"]

    def check(self, listing, names):
        for index in range(listing.count()):
            item = listing.item(index)
            item.setCheckState(QtCore.Qt.CheckState.Checked if item.text() in names
                               else QtCore.Qt.CheckState.Unchecked)
        self.panel._compute_psth_impl()
        self.settle()

    def test_columns_replace_the_single_row_and_come_back(self):
        listing = self.load()
        panel = self.panel
        self.check(listing, {"attack"})
        self.assertFalse(panel._multi_behavior_active)
        self.assertTrue(panel.multi_behavior_view.isHidden())
        self.assertFalse(panel.row_heat_page.isHidden())
        self.check(listing, {"attack", "rearing"})
        view = panel.multi_behavior_view
        self.assertTrue(panel._multi_behavior_active)
        self.assertFalse(view.isHidden())
        self.assertTrue(panel.row_heat_page.isHidden())
        self.assertEqual([entry["name"] for entry in view.entries], ["attack", "rearing"])
        # Each column holds exactly what the single-behavior computation gives.
        for entry in view.entries:
            matrix, tvec, _labels = panel._compute_psth_for_behavior(entry["name"])
            np.testing.assert_array_equal(entry["matrix"], matrix)
            np.testing.assert_array_equal(entry["tvec"], tvec)
            self.assertIsNotNone(entry["metric"])
            self.assertEqual(entry["metric"]["summary"]["post_n"], matrix.shape[0])
        # The results area reserves the height of the columns.
        self.assertGreaterEqual(panel.row_heat_host.minimumHeight(), view.required_height())
        self.assertIs(panel._psth_dashboard_widget(), view)
        self.check(listing, {"rearing"})
        self.assertFalse(panel._multi_behavior_active)
        self.assertTrue(view.isHidden())
        self.assertFalse(panel.row_heat_page.isHidden())
        self.assertIs(panel._psth_dashboard_widget(), panel.row_heat)

    def test_trace_preview_colors_the_events_of_each_behavior(self):
        listing = self.load()
        panel = self.panel
        self.check(listing, {"attack", "grooming", "rearing"})
        lines = [item for item in panel._multi_event_items if isinstance(item, pg.PlotDataItem)]
        legends = [item for item in panel._multi_event_items if isinstance(item, pg.LegendItem)]
        self.assertEqual(len(lines), 3)
        self.assertEqual(len(legends), 1)
        colors = [line.opts["pen"].color().getRgb()[:3] for line in lines]
        self.assertEqual(colors, [behavior_color(0), behavior_color(1), behavior_color(2)])
        source = panel._behavior_sources["fiber"]
        for name, line in zip(("attack", "grooming", "rearing"), lines):
            onsets, _offsets, _durations = panel._extract_behavior_events(source, name)
            np.testing.assert_allclose(line.getData()[0][0::3], onsets)        # one vertical line per onset
        labels = [label.text for _sample, label in legends[0].items]
        self.assertTrue(labels[0].startswith("attack ("))
        # The single-behavior markers are not drawn underneath.
        self.assertEqual(panel._trace_preview_events.size, 0)
        self.check(listing, {"attack"})
        self.assertEqual(panel._multi_event_items, [])
        self.assertGreater(panel._trace_preview_events.size, 0)

    def test_event_marker_alignment_keeps_the_single_view(self):
        listing = self.load()
        panel = self.panel
        self.check(listing, {"attack", "rearing"})
        self.assertTrue(panel._multi_behavior_active)
        panel.combo_align.setCurrentText("Event marker")
        self.settle()
        panel._refresh_multi_behavior()
        self.assertFalse(panel._multi_behavior_active)
        self.assertEqual(panel._multi_behavior_names(), [])


if __name__ == "__main__":
    unittest.main()
