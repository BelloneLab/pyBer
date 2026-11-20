"""Several checked behaviors: one column each (heatmap, mean PSTH, metric) and colored trace events."""
import unittest

import numpy as np
import pandas as pd
import pyqtgraph as pg
from PySide6 import QtCore, QtWidgets

import test_postprocessing_empty_state as gui_fixture
from analysis_core import ProcessedTrial
from gui_postprocessing import _behavior_table_info
from metric_windows import result_key
from multi_behavior_view import (BAND_SPACING, BEHAVIOR_COLORS, COLORBAR_WIDTH, MIN_COLUMN_WIDTH,
                                 MultiBehaviorView, behavior_color, columns_per_band, shared_range)
from postprocessing_core import mean_sem
from postprocessing_style import POSTPROCESSING_PRESETS
from psth_metric_panels import MetricPanel
from psth_metrics import METRICS, metric_values


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

    def menu_action(self, plot, *path):
        """Open the plot's actual menu and each submenu, exactly as a right-click does."""
        menu = plot.getPlotItem().getViewBox().menu
        for index, label in enumerate(path):
            self.assertIsNotNone(menu, path[:index])
            menu.aboutToShow.emit()
            matches = [action for action in menu.actions() if action.text() == label]
            self.assertEqual(len(matches), 1, (label, [action.text() for action in menu.actions()]))
            action = matches[0]
            self.assertTrue(action.isEnabled(), path[:index + 1])
            menu = action.menu()
        return action

    def test_trace_modes_from_either_event_menu_redraw_both_columns(self):
        listing = self.load()
        panel = self.panel
        self.check(listing, {"attack", "rearing"})
        columns = panel.multi_behavior_view.columns[:2]
        entries = panel.multi_behavior_view.entries
        # Compare the whole trace submenu to the single-event control, so new
        # display settings automatically acquire coverage on both event plots.
        def options(plot):
            menu = self.menu_action(plot, "Individual traces").menu()
            menu.aboutToShow.emit()
            return [action.text() for action in menu.actions() if not action.isSeparator()]

        expected = options(panel.plot_avg)
        for column in columns:
            self.assertEqual(options(column.psth), expected)
        for index, (label, mode) in enumerate((("Individual traces only", "traces"),
                                                ("Average and individual traces", "both"),
                                                ("Average only", "average"))):
            self.menu_action(columns[index % 2].psth, "Individual traces", label).trigger()
            self.settle()
            self.assertEqual(panel.trace_view()["show"], mode)
            for entry, column in zip(entries, columns):
                self.assertTrue(self.menu_action(column.psth, "Individual traces", label).isChecked())
                self.assertEqual(column.mean.isVisible(), mode != "traces")
                self.assertEqual(column.sem_band.isVisible(), mode != "traces")
                self.assertEqual(len(column.traces.curves), 0 if mode == "average" else len(entry["matrix"]))

        self.check(listing, {"rearing"})
        self.assertFalse(panel._multi_behavior_active)
        self.assertTrue(self.menu_action(panel.plot_avg, "Individual traces", "Average only").isChecked())

    def test_additional_panel_shows_each_events_own_data_and_follows_selection(self):
        listing = self.load()
        panel = self.panel
        self.check(listing, {"attack", "rearing"})
        view = panel.multi_behavior_view
        columns = view.columns[:2]
        initial_height = view.required_height()
        self.menu_action(columns[1].psth, "Add metric panel", METRICS["peak"]).trigger()
        self.settle()
        self.assertGreater(view.required_height(), initial_height)
        for entry, column in zip(view.entries, columns):
            extra = column.extra_metrics["peak"]
            self.assertTrue(extra.isVisible())
            self.assertTrue(column.metric.isVisible())
            self.assertIs(extra.result, entry["metrics"]["peak"])
            post = metric_values(entry["matrix"], entry["tvec"], panel.spin_metric_post0.value(),
                                 panel.spin_metric_post1.value(), "peak")
            np.testing.assert_allclose(extra.result["post_values"], post, equal_nan=True)
        # The selected panel is available in the regular dashboard after
        # unchecking a behavior, and can be removed from the other event column.
        self.check(listing, {"attack"})
        self.assertTrue(panel._extra_metric_plots["peak"].isVisible())
        self.check(listing, {"attack", "rearing"})
        action = self.menu_action(columns[0].psth, "Add metric panel", METRICS["peak"])
        self.assertTrue(action.isChecked())
        action.trigger()
        self.settle()
        self.assertEqual(view.required_height(), initial_height)
        for column in columns:
            extra = column.extra_metrics.get("peak")
            self.assertTrue(extra is None or extra.isHidden())

    def test_remove_and_restore_event_plot_row_from_another_visible_menu(self):
        listing = self.load()
        self.check(listing, {"attack", "rearing"})
        columns = self.panel.multi_behavior_view.columns[:2]
        self.menu_action(columns[1].psth, "Remove this panel").trigger()
        self.settle()
        self.assertTrue(all(column.psth.isHidden() for column in columns))
        self.assertTrue(all(column.heat.isVisible() for column in columns))
        self.menu_action(columns[0].heat, "Restore removed panels", "Average PSTH").trigger()
        self.settle()
        self.assertTrue(all(column.psth.isVisible() for column in columns))
        self.check(listing, {"rearing"})
        self.assertTrue(self.panel.plot_avg.isVisible())

    def test_named_extra_window_uses_each_events_offset_and_selected_limits(self):
        listing = self.load()
        panel = self.panel
        self.check(listing, {"attack", "rearing"})
        view = panel.multi_behavior_view
        onset_means = {entry["name"]: mean_sem(entry["matrix"])[0] for entry in view.entries}
        panel.combo_behavior_align.setCurrentText("Align to offset")
        panel.metric_windows.add_button.click()
        row = panel.metric_windows._rows[0]
        row.label.set_name("After offset", emit=True)
        row.start.setValue(1.25)
        row.end.setValue(2.5)
        self.settle()
        key = result_key("auc", row.window_id)
        for entry, column in zip(view.entries, view.columns):
            extra = column.extra_metrics[key]
            self.assertTrue(extra.isVisible())
            matrix, tvec, _ = panel._compute_psth_for_behavior(entry["name"])
            self.assertFalse(np.allclose(mean_sem(matrix)[0], onset_means[entry["name"]]))
            np.testing.assert_allclose(extra.result["post_values"], metric_values(matrix, tvec, 1.25, 2.5, "auc"),
                                       equal_nan=True)
            summary = extra.result["summary"]
            self.assertEqual((summary["post_label"], summary["post_start_s"], summary["post_end_s"]),
                             ("After offset", 1.25, 2.5))
        # Editing just the limit must refresh both panels without another
        # behavior selection or a manual request to recompute the PSTH.
        row.end.setValue(2.0)
        self.settle()
        for column in view.columns[:2]:
            self.assertEqual(column.extra_metrics[key].result["summary"]["post_end_s"], 2.0)
        row.remove_button.click()
        self.settle()
        self.assertTrue(all(key not in column.extra_metrics for column in view.columns[:2]))

    def test_cluster_panels_share_each_behavior_columns_own_trial_fit(self):
        """Selecting the two PCA panels from a column uses only that event's recording trials."""
        from pca_trajectories import decimate_rows

        listing = self.load()
        self.check(listing, {"attack", "rearing"})
        panel = self.panel
        panel.tab_visual_mode.setCurrentIndex(0)
        panel.pca_trajectories.set_option("method", "trials")
        first = panel.multi_behavior_view.columns[0]
        self.menu_action(first.heat, "Change to", "State-space (PCA)", "PCA trajectories").trigger()
        self.menu_action(first.psth, "Change to", "State-space (PCA)", "Cluster separation").trigger()
        self.settle()
        entries = panel.multi_behavior_view.entries
        self.assertEqual([entry["name"] for entry in entries], ["attack", "rearing"])
        for column, entry in zip(panel.multi_behavior_view.columns, entries):
            report = entry["_column_pca"][1]
            self.assertEqual(report["behaviors"], [entry["name"]])
            self.assertEqual(report["individual_recording"], "fiber")
            model = report["trial_clusters"]
            rows, _ = decimate_rows(entry["matrix"], entry["tvec"])
            valid = np.isfinite(rows).all(axis=1)
            expected = (rows[valid] - rows[valid].mean(axis=0)) @ report["components"].T
            np.testing.assert_allclose(model["scores"][valid], expected, atol=1e-8)
            self.assertEqual(model["fit_count"], int(valid.sum()))
            for slot in ("heatmap", "psth"):
                self.assertTrue(column.dashboard_substitutes[(slot, "trajectory")].isVisible())

    def test_change_to_from_event_menu_renders_the_substitute_for_each_event(self):
        listing = self.load()
        self.check(listing, {"attack", "rearing"})
        panel = self.panel
        view = panel.multi_behavior_view
        columns = view.columns[:2]
        self.menu_action(columns[1].heat, "Change to", "Pre/post comparison", METRICS["peak"]).trigger()
        self.settle()
        substitutes = []
        for column, entry in zip(columns, view.entries):
            self.assertTrue(column.heat.isHidden())
            candidates = [plot for plot in column.findChildren(MetricPanel)
                          if plot is not column.metric and plot.isVisible() and plot.result is not None]
            self.assertEqual(len(candidates), 1)
            substitute = candidates[0]
            post = metric_values(entry["matrix"], entry["tvec"], panel.spin_metric_post0.value(),
                                 panel.spin_metric_post1.value(), "peak")
            np.testing.assert_allclose(substitute.result["post_values"], post, equal_nan=True)
            substitutes.append(substitute)
        self.menu_action(substitutes[0], "Change to", "Original: Heatmap").trigger()
        self.settle()
        self.assertTrue(all(column.heat.isVisible() for column in columns))

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
