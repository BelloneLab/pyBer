"""Individual traces behind the mean PSTH: colors, thinning, hover, menus, persistence."""
import unittest

import numpy as np
from PySide6 import QtCore, QtWidgets

import test_postprocessing_empty_state as fixture
# The module is imported, not the class: a TestCase imported by name would be
# collected and run again by this module.
import test_group_view_dashboard as dashboard

EVENTS, N_PER_GROUP = dashboard.EVENTS, dashboard.N_PER_GROUP
from psth_groups import LEGEND_AVOID_PENALTY, best_legend_position, legend_overlap_counts
from psth_trace_overlay import (ALPHA_FIRST, ALPHA_LAST, DEFAULT_TRACE_VIEW, LINE_STYLES, MAX_TRACES,
                                SHOW_MODES, distinct_colors, fade_alphas, legend_indices, nearest_trace,
                                normalize_trace_view, thinned_indices, trace_colors)


class TraceHelperTests(unittest.TestCase):
    """Pure functions that decide how traces look and which one the pointer selects."""

    def test_fade_runs_from_opaque_first_to_faint_last(self):
        alphas = fade_alphas(6)
        self.assertEqual((alphas[0], alphas[-1]), (ALPHA_FIRST, ALPHA_LAST))
        self.assertEqual(alphas, sorted(alphas, reverse=True))
        self.assertEqual(fade_alphas(1), [ALPHA_FIRST])
        self.assertEqual(fade_alphas(0), [])
        colors = trace_colors(4, "#336699", "fade")
        self.assertEqual({color[:3] for color in colors}, {(0x33, 0x66, 0x99)})   # one hue, only opacity changes
        self.assertGreater(colors[0][3], colors[-1][3])

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

    def test_view_is_normalized(self):
        self.assertEqual(normalize_trace_view(None), DEFAULT_TRACE_VIEW)
        cleaned = normalize_trace_view({"show": "both", "color": "rainbow", "style": "dash", "legend": 1})
        self.assertEqual(cleaned, {"show": "both", "color": "fade", "style": "dash", "legend": True})

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

    def test_trials_are_drawn_fading_and_the_average_can_be_hidden(self):
        panel = self.individual()
        matrix = panel._last_mat.copy()
        mean = panel.curve_avg.getData()[1].copy()
        panel.set_trace_view("show", "both")
        overlay = panel.psth_traces
        self.assertEqual(len(overlay.curves), EVENTS)                 # one trace per trial
        for curve, row in zip(overlay.curves, panel._last_mat):
            np.testing.assert_array_equal(curve.getData()[1], row)
        alphas = [curve.opts["pen"].color().alpha() for curve in overlay.curves]
        self.assertEqual(alphas, sorted(alphas, reverse=True))        # first trial darkest
        self.assertEqual((alphas[0], alphas[-1]), (ALPHA_FIRST, ALPHA_LAST))
        base = panel.curve_avg.opts["pen"].color().getRgb()[:3]
        self.assertEqual({curve.opts["pen"].color().getRgb()[:3] for curve in overlay.curves}, {base})
        self.assertTrue(panel.curve_avg.isVisible())
        self.assertTrue(panel.sem_band.isVisible())
        panel.set_trace_view("show", "traces")
        self.assertFalse(panel.curve_avg.isVisible())
        self.assertFalse(panel.sem_band.isVisible())
        self.assertEqual(len(overlay.curves), EVENTS)
        panel.set_trace_view("show", "average")
        self.assertFalse(overlay.has_traces())
        self.assertTrue(panel.curve_avg.isVisible())
        np.testing.assert_array_equal(panel._last_mat, matrix)        # display only
        np.testing.assert_array_equal(panel.curve_avg.getData()[1], mean)

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
            expected = tuple(int(block.color[i:i + 2], 16) for i in (1, 3, 5))
            pens = [overlay.curves[i].opts["pen"].color() for i in range(block.start, block.stop)]
            self.assertEqual({pen.getRgb()[:3] for pen in pens}, {expected})
            self.assertEqual(pens[0].alpha(), ALPHA_FIRST)             # the fade restarts in every group
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
        self.assertEqual(settings["psth_trace_view"]["style"], "dot")
        panel.set_trace_view("show", "average")
        panel._is_restoring_settings = True
        panel._apply_settings(settings)
        panel._is_restoring_settings = False
        self.assertEqual(panel.trace_view()["show"], "both")
        self.assertEqual(panel.trace_view()["style"], "dot")
        self.assertEqual(panel._default_settings_payload()["psth_trace_view"], DEFAULT_TRACE_VIEW)


if __name__ == "__main__":
    unittest.main()
