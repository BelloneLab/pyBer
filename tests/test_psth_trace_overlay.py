"""Individual traces behind the mean PSTH: colors, thinning, hover, menus, persistence."""
import unittest
from unittest.mock import patch

import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets

import test_postprocessing_empty_state as fixture
# The module is imported, not the class: a TestCase imported by name would be
# collected and run again by this module.
import test_group_view_dashboard as dashboard
import test_multi_behavior_view as behavior

EVENTS, N_PER_GROUP = dashboard.EVENTS, dashboard.N_PER_GROUP
from psth_groups import LEGEND_AVOID_PENALTY, best_legend_position, legend_overlap_counts
from psth_trace_overlay import (DEFAULT_TRACE_VIEW, LINE_STYLES, MAX_TRACES, SHOW_MODES, distinct_colors,
                                legend_indices, nearest_trace, normalize_trace_view, response_amplitudes,
                                thinned_indices, trace_colors)


class TraceHelperTests(unittest.TestCase):
    """Pure functions that decide how traces look and which one the pointer selects."""

    def test_single_hue_rank_remains_visible_at_fixed_opacity_on_each_theme(self):
        for background, dark in (("#101520", True), ("#ffffff", False)):
            with self.subTest(background=background):
                colors = trace_colors(6, "#008080", "fade", background=background, opacity=40)
                self.assertEqual(len({color[:3] for color in colors}), 6)
                self.assertEqual({color[3] for color in colors}, {102})
                # Later trials gain contrast against either surface.
                lightness = [QtGui.QColor(*color).lightness() for color in colors]
                self.assertEqual(lightness, sorted(lightness, reverse=not dark))
                self.assertEqual(trace_colors(0, "#008080", "fade", background=background), [])

    def test_amplitude_uses_both_windows_and_both_response_polarities(self):
        time = np.array([-2., -1., 0., 1., 2.])
        rows = np.array([
            [1, 3, 3, 8, 99],                 # baseline 2, response ends at 1 s
            [10, np.nan, 4, 7, 99],           # downward response; finite Pre sample suffices
            [3, 3, np.nan, 3, -90],           # flat response has zero amplitude
            [np.nan, np.nan, 2, 4, 0],        # missing baseline
            [1, 5, np.nan, np.inf, 0],        # no finite response
        ])
        original = rows.copy()
        values = response_amplitudes(rows, time, (-2., -1.), (0., 1.))
        np.testing.assert_equal(values, [6., 6., 0., np.nan, np.nan])
        self.assertTrue(np.isnan(response_amplitudes(rows, time, (-2., -1.), (4., 5.))).all())
        np.testing.assert_array_equal(rows, original)

    def test_amplitude_palette_treats_ties_and_missing_values_consistently(self):
        # The colors are returned in original row order, even when rank is unsorted.
        ranks = np.array([1., np.nan, 0., 1.])
        for mode in ("fade", "distinct"):
            colors = trace_colors(4, "#008080", mode, ranks=ranks, rank_by="amplitude", opacity=40)
            self.assertEqual(colors[0], colors[3])
            self.assertNotEqual(colors[0][:3], colors[2][:3])
            red, green, blue, _alpha = colors[1]
            self.assertLessEqual(max(red, green, blue) - min(red, green, blue), 10)

    def test_distinct_colors_are_all_different(self):
        for count in (1, 3, 10, 11, 40):
            colors = distinct_colors(count)
            self.assertEqual(len(colors), count)
            self.assertEqual(len(set(colors)), count)
        self.assertEqual(len({color[:3] for color in trace_colors(7, "#336699", "distinct")}), 7)

    def test_thinning_and_legend_keep_both_ends(self):
        self.assertEqual(thinned_indices(5), [0, 1, 2, 3, 4])
        many = thinned_indices(5 * MAX_TRACES)
        self.assertLessEqual(len(many), MAX_TRACES)
        self.assertEqual((many[0], many[-1]), (0, 5 * MAX_TRACES - 1))
        self.assertEqual(legend_indices(4, 5), [0, 1, 2, 3])
        picked = legend_indices(100, 5)
        self.assertEqual((picked[0], picked[-1], len(picked)), (0, 99, 5))

    def test_nearest_trace_uses_pixel_distance_and_ignores_missing_rows(self):
        tvec = np.linspace(0, 10, 101)
        rows = np.vstack([np.zeros(101), np.full(101, 5.0), np.full(101, np.nan)])
        self.assertEqual(nearest_trace(rows, tvec, 4.0, 4.6, y_per_px=0.1), 1)       # 4 px away
        self.assertIsNone(nearest_trace(rows, tvec, 4.0, 2.5, y_per_px=0.1))          # 25 px from both
        self.assertIsNone(nearest_trace(rows, tvec, 11.0, 5.0, y_per_px=0.1))         # outside the time axis
        self.assertEqual(nearest_trace(rows, tvec, 4.0, 5.2, y_per_px=0.1, subset=[2, 1]), 1)
        self.assertIsNone(nearest_trace(rows, tvec, 4.0, 5.0, y_per_px=0.1, subset=[]))

    def test_hover_uses_visible_segments_and_does_not_interpolate_across_gaps(self):
        time = np.array([0., 1.])
        rows = np.array([[0., 10.], [3., 3.], [9., np.nan]])
        self.assertEqual(nearest_trace(rows, time, 0.4, 4., y_per_px=0.01), 0)
        self.assertIsNone(nearest_trace(rows, time, 0.4, 9., y_per_px=0.01))
        self.assertEqual(nearest_trace(rows, time, 0., 9., y_per_px=0.01), 2)

    def test_view_is_normalized(self):
        self.assertEqual(normalize_trace_view(None), DEFAULT_TRACE_VIEW)
        cleaned = normalize_trace_view({"show": "both", "color": "rainbow", "style": "dash", "legend": 1})
        self.assertEqual(cleaned, {**DEFAULT_TRACE_VIEW, "show": "both", "style": "dash", "legend": True})
        self.assertEqual((cleaned["rank"], cleaned["opacity"]), ("order", "auto"))
        self.assertEqual(normalize_trace_view({"rank": "unknown", "opacity": "broken"}), DEFAULT_TRACE_VIEW)
        custom = normalize_trace_view({"rank": "amplitude", "opacity": 37})
        self.assertEqual((custom["rank"], custom["opacity"]), ("amplitude", 37))

    def test_second_legend_avoids_the_first(self):
        counts = legend_overlap_counts([], ((0, 1), (0, 1)), (400, 300), (100, 60),
                                       avoid_px=(292, 8, 100, 60))               # top right is taken
        self.assertGreaterEqual(counts["top right"], LEGEND_AVOID_PENALTY)
        self.assertEqual(best_legend_position(counts)[0], "top left")


class TraceOverlayGuiTests(unittest.TestCase):
    """The overlay follows the displayed rows and never touches the analysis."""

    @classmethod
    def setUpClass(cls):
        cls.app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    setUp = fixture.PostprocessingEmptyStateTests.setUp
    tearDown = fixture.PostprocessingEmptyStateTests.tearDown
    load = dashboard.GroupViewDashboardTests.load

    def individual(self):
        """One recording selected: rows are its trials."""
        self.load()
        panel = self.panel
        panel.tab_visual_mode.setCurrentIndex(0)
        self.app.processEvents()
        panel._compute_psth_impl()
        self.app.processEvents()
        return panel

    def test_default_is_the_average_only(self):
        panel = self.individual()
        self.assertEqual(panel.trace_view(), DEFAULT_TRACE_VIEW)
        self.assertFalse(panel.psth_traces.has_traces())
        self.assertTrue(panel.curve_avg.isVisible())
        self.assertFalse(panel.psth_trace_legend.isVisible())

    def test_trials_use_ordered_shades_and_the_average_can_be_hidden(self):
        panel = self.individual()
        matrix = panel._last_mat.copy()
        mean = panel.curve_avg.getData()[1].copy()
        panel.set_trace_view("show", "both")
        overlay = panel.psth_traces
        self.assertEqual(len(overlay.curves), EVENTS)                 # one trace per trial
        for curve, row in zip(overlay.curves, panel._last_mat):
            np.testing.assert_array_equal(curve.getData()[1], row)
        colors = [curve.opts["pen"].color().getRgb() for curve in overlay.curves]
        self.assertEqual(len({color[:3] for color in colors}), EVENTS)
        self.assertEqual(len({color[3] for color in colors}), 1)
        self.assertTrue(panel.curve_avg.isVisible())
        self.assertTrue(panel.sem_band.isVisible())
        self.assertFalse(panel.sem_band.path().isEmpty())
        panel.set_trace_view("opacity", 40)
        self.assertEqual({curve.opts["pen"].color().alpha() for curve in overlay.curves}, {102})
        self.assertEqual([curve.opts["pen"].color().getRgb()[:3] for curve in overlay.curves],
                         [color[:3] for color in colors])
        panel.set_trace_view("show", "traces")
        self.assertFalse(panel.curve_avg.isVisible())
        self.assertFalse(panel.sem_band.isVisible())
        self.assertEqual(len(overlay.curves), EVENTS)
        panel.set_trace_view("show", "average")
        self.assertFalse(overlay.has_traces())
        self.assertTrue(panel.curve_avg.isVisible())
        np.testing.assert_array_equal(panel._last_mat, matrix)        # display only
        np.testing.assert_array_equal(panel.curve_avg.getData()[1], mean)

    def test_live_amplitude_ranking_follows_the_selected_windows(self):
        panel = self.individual()
        panel.set_trace_view("show", "both")
        panel.set_trace_view("rank", "amplitude")
        overlay = panel.psth_traces
        labels = list(overlay.labels)
        time = panel._last_tvec.copy()
        rows = panel._last_mat.copy()
        # The fixture delivers a response from 0 through 2 s. Move Post off
        # that response and compare against a direct calculation on the rows.
        for start, stop in ((0., 1.), (2.5, 3.5)):
            panel.spin_metric_pre0.setValue(-2.)
            panel.spin_metric_pre1.setValue(-1.)
            panel.spin_metric_post1.setValue(stop)
            panel.spin_metric_post0.setValue(start)
            self.app.processEvents()
            baseline = np.mean(rows[:, (time >= -2.) & (time <= -1.)], axis=1)
            response = rows[:, (time >= start) & (time <= stop)]
            expected = np.max(np.abs(response - baseline[:, None]), axis=1)
            np.testing.assert_allclose(overlay.amplitudes, expected)
            self.assertEqual(overlay.labels, labels)
            for curve, row in zip(overlay.curves, rows):
                np.testing.assert_array_equal(curve.getData()[1], row)

    def test_switching_plot_theme_updates_active_shades_and_mean(self):
        panel = self.individual()
        panel.set_trace_view("show", "both")
        for preset, dark in (("Midnight", True), ("Paper", False)):
            panel.combo_plot_preset.setCurrentText(preset)
            self.app.processEvents()
            curves = panel.psth_traces.curves
            first = curves[0].opts["pen"].color().lightness()
            last = curves[-1].opts["pen"].color().lightness()
            self.assertGreater((last - first) if dark else (first - last), 0)
            self.assertGreater(panel.curve_avg.opts["pen"].widthF(), 2.)
            self.assertFalse(panel.sem_band.path().isEmpty())

    def test_auto_opacity_adapts_to_density_and_keeps_the_mean_prominent(self):
        panel = self.individual()
        overlay = panel.psth_traces
        time = np.array([-1., 0., 1.])

        def drawn_alpha(count, show, opacity="auto"):
            """Draw a known density using the same overlay the menu controls."""
            rows = np.tile(np.arange(count, dtype=float)[:, None], (1, len(time)))
            overlay.draw(time, rows, [], [(0, count, "#008080", "")], {"show": show, "opacity": opacity})
            return overlay.curves[0].opts["pen"].color().alpha()

        self.assertGreater(drawn_alpha(3, "both"), drawn_alpha(100, "both"))
        self.assertGreater(drawn_alpha(20, "traces"), drawn_alpha(20, "both"))
        self.assertEqual(drawn_alpha(3, "both", 40), drawn_alpha(100, "both", 40))

    def test_amplitude_hover_preserves_identity_and_handles_missing_response(self):
        panel = self.individual()
        overlay = panel.psth_traces
        time = np.array([-2., -1., 0., 1.])
        rows = np.array([[2., 2., 4., -1.], [10., 10., np.nan, np.nan], [30., 30., 30., 33.]])
        labels = ["Trial A", "Trial B", "Trial C"]
        overlay.draw(time, rows, labels, [(0, 3, "#008080", "")], {"show": "both", "rank": "amplitude"},
                     pre_window=(-2., -1.), post_window=(0., 1.))
        self.assertEqual(overlay.labels, labels)
        self.assertEqual(overlay.row_of_curve, [0, 1, 2])
        self.assertEqual(overlay.colors[0], overlay.colors[2])
        self.assertNotEqual(overlay.colors[0], overlay.colors[1])
        view = panel.plot_avg.getViewBox()
        view.setYRange(-5, 35, padding=0)
        self.app.processEvents()
        for index in (0, 1):
            point = view.mapViewToScene(QtCore.QPointF(-1., rows[index, 1]))
            overlay.hover(point)
            tip = overlay.tip.toPlainText()
            self.assertIn(labels[index], tip)
            if index == 0:
                self.assertIn("3", tip)
            else:
                self.assertRegex(tip.lower(), "unavailable|n/a|no .*samples")

    def test_colors_styles_and_legend(self):
        panel = self.individual()
        panel.set_trace_view("show", "both")
        panel.set_trace_view("color", "distinct")
        overlay = panel.psth_traces
        self.assertEqual(len({curve.opts["pen"].color().getRgb()[:3] for curve in overlay.curves}), EVENTS)
        for code, qt_style in (("dash", QtCore.Qt.PenStyle.DashLine), ("dot", QtCore.Qt.PenStyle.DotLine),
                               ("dashdot", QtCore.Qt.PenStyle.DashDotLine), ("solid", QtCore.Qt.PenStyle.SolidLine)):
            panel.set_trace_view("style", code)
            self.assertEqual({curve.opts["pen"].style() for curve in panel.psth_traces.curves}, {qt_style})
        self.assertEqual(set(LINE_STYLES), {"solid", "dash", "dot", "dashdot"})
        self.assertFalse(panel.psth_trace_legend.isVisible())
        panel.set_trace_view("legend", True)
        legend = panel.psth_trace_legend
        self.assertTrue(legend.isVisible())
        self.assertEqual([label.text for _sample, label in legend.items], overlay.labels)
        panel.set_trace_view("legend", False)
        self.assertFalse(legend.isVisible())
        self.assertEqual(legend.items, [])

    def test_pointing_at_a_trace_highlights_and_names_it(self):
        panel = self.individual()
        panel.set_trace_view("show", "both")
        self.app.processEvents()
        overlay = panel.psth_traces
        view = panel.plot_avg.getPlotItem().getViewBox()
        row = 2
        column = panel._last_tvec.size // 2
        # Spread the traces so exactly one is under the pointer.
        overlay.rows = overlay.rows + np.arange(overlay.rows.shape[0])[:, None] * 100.0
        view.setYRange(-50, 100.0 * overlay.rows.shape[0], padding=0)
        self.app.processEvents()
        target = view.mapViewToScene(QtCore.QPointF(panel._last_tvec[column], overlay.rows[row, column]))
        self.assertEqual(overlay.hover(target), overlay.labels[row])
        self.assertTrue(overlay.tip.isVisible())
        self.assertGreater(overlay.curves[row].opts["pen"].widthF(), 2)
        self.assertEqual(overlay.curves[row].opts["pen"].color().alpha(), 255)
        outside = view.sceneBoundingRect().bottomRight() + QtCore.QPointF(50, 50)
        self.assertIsNone(overlay.hover(outside))
        self.assertFalse(overlay.tip.isVisible())
        self.assertLess(overlay.curves[row].opts["pen"].widthF(), 2)   # highlight removed

    def test_group_view_draws_each_animal_in_its_group_color(self):
        self.load()
        panel = self.panel
        panel.cb_group_view.setChecked(True)
        self.app.processEvents()
        panel._compute_psth_impl()
        self.app.processEvents()
        panel.set_trace_view("show", "both")
        overlay = panel.psth_traces
        self.assertEqual(len(overlay.curves), 2 * N_PER_GROUP)
        display = panel._group_display
        for block in display.blocks:
            pens = [overlay.curves[i].opts["pen"].color() for i in range(block.start, block.stop)]
            self.assertEqual(len({pen.getRgb()[:3] for pen in pens}), N_PER_GROUP)
            # Every group's ramp stays in its own hue family.
            for pen in pens:
                delta = abs(pen.hue() - QtGui.QColor(block.color).hue())
                self.assertLess(min(delta, 360 - delta), 10)
        self.assertTrue(all(item.isVisible() for item in panel._group_avg_items))
        panel.set_trace_view("show", "traces")
        # Means fade out but stay "visible", so the group legend keeps its colors.
        for item in panel._group_avg_items:
            is_mean = any(item is curve for curve in panel._group_mean_curves)
            self.assertEqual((item.isVisible(), item.opacity()), (True, 0.0) if is_mean else (False, 1.0))
        panel.set_trace_view("show", "both")
        self.assertTrue(all(curve.opacity() == 1.0 for curve in panel._group_mean_curves))
        self.assertTrue(all(curve.opts["pen"].widthF() >= 2.8 for curve in panel._group_mean_curves))
        panel.set_trace_view("show", "traces")
        panel.set_trace_view("legend", True)
        labels = [label.text for _sample, label in panel.psth_trace_legend.items]
        self.assertEqual(len(labels), 2 * N_PER_GROUP)
        self.assertTrue(labels[0].startswith("WT: "))
        # Group legend and trace legend never sit on top of each other.
        panel._place_group_legend()
        group_box = panel.psth_group_legend.mapRectToParent(panel.psth_group_legend.boundingRect())
        trace_box = panel.psth_trace_legend.mapRectToParent(panel.psth_trace_legend.boundingRect())
        if group_box.width() + trace_box.width() < panel.plot_avg.getPlotItem().getViewBox().size().width():
            self.assertFalse(group_box.intersects(trace_box))

    def test_menus_and_settings(self):
        panel = self.individual()
        panel.show()
        self.app.processEvents()
        self.assertFalse(panel.btn_group_view.isHidden())
        self.assertEqual(panel.btn_group_view.text(), "Hide/show traces")
        panel._rebuild_group_menu()
        actions = {action.text(): action for action in panel.menu_group_view.actions() if action.text()}
        self.assertTrue(set(SHOW_MODES.values()) <= set(actions))
        self.assertTrue(actions[SHOW_MODES["average"]].isChecked())
        actions[SHOW_MODES["both"]].trigger()
        self.assertEqual(panel.trace_view()["show"], "both")
        self.assertTrue(panel.psth_traces.has_traces())
        # Right-click menu of the plot offers the same options.
        panel._fill_trace_menu()
        texts = [action.text() for action in panel.menu_psth_traces.actions()]
        self.assertIn("Trace legend", texts)
        self.assertIn("Color by", texts)
        self.assertIn("Trace opacity", texts)
        rank_menu = next(action.menu() for action in panel.menu_psth_traces.actions() if action.text() == "Color by")
        amplitude_action = next(action for action in rank_menu.actions() if "amplitude" in action.text().lower())
        amplitude_action.trigger()
        self.assertEqual(panel.trace_view()["rank"], "amplitude")
        opacity_menu = next(action.menu() for action in panel.menu_psth_traces.actions()
                            if action.text() == "Trace opacity")
        next(action for action in opacity_menu.actions() if action.text() == "50%").trigger()
        self.assertEqual(panel.trace_view()["opacity"], 50)
        custom = next(action for action in opacity_menu.actions() if action.text().startswith("Custom"))
        with patch.object(QtWidgets.QInputDialog, "getInt", return_value=(37, True)):
            custom.trigger()
        self.assertEqual(panel.trace_view()["opacity"], 37)
        # Reopening the menu rebuilds its actions from the stored setting.
        panel._fill_trace_menu()
        reopened = next(action.menu() for action in panel.menu_psth_traces.actions()
                        if action.text() == "Trace opacity")
        checked = [action.text() for action in reopened.actions() if action.isChecked()]
        self.assertEqual(len(checked), 1)
        self.assertIn("37%", checked[0])
        self.assertIn(panel.menu_psth_traces.menuAction(),
                      panel.plot_avg.getPlotItem().getViewBox().menu.actions())
        # Group scope: the options sit in a submenu of the groups menu.
        panel.tab_visual_mode.setCurrentIndex(1)
        self.app.processEvents()
        self.assertTrue(panel.btn_group_view.text().startswith("Hide/show groups"))
        panel._rebuild_group_menu()
        submenus = [action.menu() for action in panel.menu_group_view.actions() if action.menu() is not None]
        self.assertEqual([menu.title() for menu in submenus], ["Individual traces"])
        # Persistence.
        panel.set_trace_view("style", "dot")
        settings = panel._collect_settings()
        self.assertEqual(settings["psth_trace_view"], panel.trace_view())
        panel.set_trace_view("show", "average")
        panel.set_trace_view("rank", "order")
        panel.set_trace_view("opacity", "auto")
        panel._is_restoring_settings = True
        panel._apply_settings(settings)
        panel._is_restoring_settings = False
        self.assertEqual(panel.trace_view()["show"], "both")
        self.assertEqual(panel.trace_view()["style"], "dot")
        self.assertEqual((panel.trace_view()["rank"], panel.trace_view()["opacity"]), ("amplitude", 37))
        self.assertEqual(panel._default_settings_payload()["psth_trace_view"], DEFAULT_TRACE_VIEW)


class MultiBehaviorTraceOverlayTests(unittest.TestCase):
    """Changing a display option reuses the aligned rows of each visible behavior."""

    setUpClass = classmethod(behavior.MultiBehaviorPanelTests.setUpClass.__func__)
    setUp = behavior.MultiBehaviorPanelTests.setUp
    tearDown = behavior.MultiBehaviorPanelTests.tearDown
    load = behavior.MultiBehaviorPanelTests.load
    check = behavior.MultiBehaviorPanelTests.check
    settle = behavior.MultiBehaviorPanelTests.settle

    def test_menu_changes_update_cached_columns_and_keep_zoom(self):
        listing = self.load()
        self.check(listing, {"attack", "rearing"})
        panel = self.panel
        self.assertTrue(panel._multi_behavior_active)
        self.assertEqual(len(panel.multi_behavior_view.entries), 2)
        with patch.object(panel, "_compute_psth_for_behavior", wraps=panel._compute_psth_for_behavior) as compute:
            panel.set_trace_view("show", "both")
            columns = panel.multi_behavior_view.columns[:2]
            for column in columns:
                column.psth.setRange(xRange=(-0.5, 1.0), yRange=(-1.0, 0.5), padding=0)
            panel.set_trace_view("rank", "amplitude")
            panel.set_trace_view("opacity", 40)
            self.settle()
            compute.assert_not_called()
        for entry, column in zip(panel.multi_behavior_view.entries, columns):
            time, rows = entry["tvec"], entry["matrix"]
            pre = (time >= panel.spin_metric_pre0.value()) & (time <= panel.spin_metric_pre1.value())
            post = (time >= panel.spin_metric_post0.value()) & (time <= panel.spin_metric_post1.value())
            baseline = np.nanmean(rows[:, pre], axis=1)
            expected = np.nanmax(np.abs(rows[:, post] - baseline[:, None]), axis=1)
            np.testing.assert_allclose(column.traces.amplitudes, expected)
            self.assertEqual({curve.opts["pen"].color().alpha() for curve in column.traces.curves}, {102})
            np.testing.assert_allclose(column.psth.viewRange(), [[-0.5, 1.0], [-1.0, 0.5]])


if __name__ == "__main__":
    unittest.main()
