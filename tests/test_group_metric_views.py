"""Selectable group representations of PSTH metrics: statistics, drawing, GUI and exports."""
import csv
from pathlib import Path
import tempfile
import unittest

import numpy as np
from PySide6 import QtCore, QtWidgets
from scipy import stats

import test_postprocessing_empty_state as fixture
from test_group_view_dashboard import GroupViewDashboardTests, N_PER_GROUP
from group_metric_views import (GROUP_VIEWS, SUMMARY_STYLES, build_group_view, bracket_layout,
                                draw_group_view_matplotlib, paired_test, significance_stars,
                                view_bounds, view_from_comparison, view_note, within_rows)
from group_stats import compare_groups
from psth_groups import LEGEND_POSITIONS, best_legend_position, legend_overlap_counts


def _groups(seed=4, n=8):
    """Three synthetic groups: no change, a clear rise, a clear fall (values per animal)."""
    rng = np.random.default_rng(seed)
    out = []
    for name, color, shift in (("Ctrl", "#1f77b4", 0.0), ("Up", "#d62728", 3.0), ("Down", "#2ca02c", -3.0)):
        pre = rng.normal(0, 1, n)
        out.append({"name": name, "color": color, "pre": pre, "post": pre + shift + rng.normal(0, 0.4, n),
                    "ids": [f"{name}{i}" for i in range(n)]})
    return out


class GroupViewStatisticsTests(unittest.TestCase):
    """Each representation tests the quantity it shows."""

    def test_between_group_quantity_follows_the_view(self):
        groups = _groups()
        for mode, values in (("delta", lambda g: g["post"] - g["pre"]), ("post", lambda g: g["post"]),
                             ("pre", lambda g: g["pre"])):
            view = build_group_view(mode, groups, "AUC")
            expected = compare_groups([{"name": g["name"], "values": values(g)} for g in groups])
            self.assertEqual(view["between"]["test"], expected["test"])
            self.assertAlmostEqual(view["between"]["p"], expected["p"])
            self.assertEqual(len(view["columns"]), 3)
            for column, group in zip(view["columns"], groups):
                np.testing.assert_allclose(column["values"], values(group))
                self.assertAlmostEqual(column["mean"], float(np.mean(values(group))))
        self.assertTrue(build_group_view("delta", groups)["zero_line"])
        self.assertFalse(build_group_view("post", groups)["zero_line"])
        self.assertIn("Post − Pre", build_group_view("delta", groups, "AUC")["title"])
        self.assertEqual(build_group_view("unknown", groups)["mode"], "delta")

    def test_paired_view_tests_within_groups_with_holm_and_links_animals(self):
        groups = _groups()
        view = build_group_view("paired", groups, "AUC")
        self.assertEqual([c["label"] for c in view["columns"]], ["Pre", "Post"] * 3)
        self.assertEqual(len(view["links"]), 3)
        np.testing.assert_allclose(view["links"][1]["y0"], groups[1]["pre"])
        np.testing.assert_allclose(view["links"][1]["y1"], groups[1]["post"])
        raw = []
        for item, group in zip(view["within"], groups):
            self.assertEqual(item["n_pairs"], 8)
            self.assertIn(item["test"], {"Paired t-test", "Wilcoxon signed-rank"})
            if item["test"] == "Paired t-test":
                self.assertAlmostEqual(item["p"], float(stats.ttest_rel(group["post"], group["pre"]).pvalue))
            else:
                self.assertAlmostEqual(item["p"], float(stats.wilcoxon(group["post"] - group["pre"]).pvalue))
            raw.append(item["p"])
        # Holm: the smallest raw p is multiplied by the number of tests.
        order = np.argsort(raw)
        self.assertAlmostEqual(view["within"][order[0]]["p_holm"], min(1.0, raw[order[0]] * 3))
        self.assertTrue(all(item["p_holm"] >= item["p"] for item in view["within"]))
        self.assertLess(view["within"][1]["p_holm"], 0.05)       # clear rise
        self.assertGreater(view["within"][0]["p_holm"], 0.05)    # control unchanged
        self.assertEqual(len(view["brackets"]), 3)
        # The between-group part of the paired view tests the change.
        delta = build_group_view("delta", groups)
        self.assertAlmostEqual(view["between"]["p"], delta["between"]["p"])
        self.assertIn("Change between groups", view_note(view, "points"))
        rows = within_rows(view, metric_id="auc")
        self.assertEqual([row["group"] for row in rows], ["Ctrl", "Up", "Down"])

    def test_small_or_untested_groups_are_descriptive(self):
        tiny = paired_test([1.0, 2.0], [2.0, 3.5])
        self.assertEqual(tiny["test"], "Descriptive only")
        self.assertTrue(np.isnan(tiny["p"]))
        constant = paired_test([1, 2, 3, 4], [2, 3, 4, 5])
        self.assertTrue(np.isnan(constant["p"]))
        groups = _groups()
        groups.append({"name": "Ungrouped", "color": "#888888", "pre": np.zeros(5), "post": np.ones(5) * 9,
                       "tested": False})
        view = build_group_view("paired", groups)
        self.assertTrue(np.isnan(view["within"][-1]["p_holm"]))
        self.assertEqual(len(view["brackets"]), 3)               # no label above the untested group
        self.assertTrue(all("p=" in bracket["text"] for bracket in view["brackets"]))
        # Nothing testable at all: no brackets and no text on the plot.
        small = [dict(group, pre=group["pre"][:2], post=group["post"][:2]) for group in groups[:3]]
        for mode in GROUP_VIEWS:
            untested = build_group_view(mode, small, "AUC")
            self.assertEqual(untested["brackets"], [])
            self.assertEqual(view_note(untested, "points"), "")
        # The untested group does not enter the Holm family of the others.
        reference = build_group_view("paired", groups[:3])
        for a, b in zip(view["within"][:3], reference["within"]):
            self.assertAlmostEqual(a["p_holm"], b["p_holm"])

    def test_brackets_stars_and_bounds(self):
        self.assertEqual([significance_stars(p) for p in (0.0004, 0.004, 0.04, 0.4, np.nan)],
                         ["***", "**", "*", "ns", ""])
        view = build_group_view("post", _groups())
        self.assertLessEqual(len(view["brackets"]), 3)
        self.assertTrue(all(b["text"].startswith("*") for b in view["brackets"]))   # only significant pairs
        low, high = view_bounds(view, "points")
        placed, top = bracket_layout(view, low, high)
        self.assertEqual(len(placed), len(view["brackets"]))
        self.assertGreater(top, high)
        self.assertTrue(all(y > high for _x0, _x1, y, _text in placed))
        bars_low, bars_high = view_bounds(view, "bars")
        self.assertLessEqual(bars_low, 0.0)      # bars start at zero
        self.assertGreaterEqual(bars_high, high)
        no_difference = build_group_view("pre", _groups())
        self.assertEqual(no_difference["brackets"], [])

    def test_matplotlib_panels_draw_every_view_and_style(self):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        groups = _groups()
        for mode in GROUP_VIEWS:
            view = build_group_view(mode, groups, "AUC")
            for style in SUMMARY_STYLES:
                figure, axis = plt.subplots()
                draw_group_view_matplotlib(axis, view, style, units="z")
                self.assertEqual(axis.get_title(), view["title"])
                self.assertEqual(len(axis.get_xticks()), len(view["ticks"]))
                plt.close(figure)
        comparison = compare_groups([{"name": g["name"], "values": g["post"], "color": g["color"]} for g in groups])
        wrapped = view_from_comparison(comparison, "Peak rate")
        self.assertEqual(len(wrapped["columns"]), 3)
        self.assertIn("recordings", view_note(wrapped, "box"))


class LegendPlacementTests(unittest.TestCase):
    """The legend goes where it hides the fewest curve samples."""

    def test_least_covered_corner_is_chosen(self):
        x = np.linspace(0, 10, 500)
        rising = (x, x)                       # bottom-left to top-right
        counts = legend_overlap_counts([rising], ((0, 10), (0, 10)), (400, 300), (100, 60))
        self.assertGreater(counts["top right"], 0)
        self.assertGreater(counts["bottom left"], 0)
        self.assertEqual(counts["top left"], 0)
        self.assertEqual(best_legend_position(counts)[0], "top left")
        # No curve anywhere: the preferred position wins the tie.
        empty = legend_overlap_counts([], ((0, 10), (0, 10)), (400, 300), (100, 60))
        self.assertEqual(best_legend_position(empty)[0], LEGEND_POSITIONS[0][0])
        flat_top = (x, np.full_like(x, 9.5))
        counts = legend_overlap_counts([flat_top], ((0, 10), (0, 10)), (400, 300), (100, 60))
        self.assertIn(best_legend_position(counts)[0], {"bottom right", "bottom left", "bottom center"})


class GroupViewGuiTests(unittest.TestCase):
    """Selectors redraw the panels, persist and reach the exports."""

    @classmethod
    def setUpClass(cls):
        cls.app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    setUp = fixture.PostprocessingEmptyStateTests.setUp
    tearDown = fixture.PostprocessingEmptyStateTests.tearDown
    load = GroupViewDashboardTests.load

    def show_groups(self):
        self.load()
        panel = self.panel
        panel.cb_group_view.setChecked(True)
        self.app.processEvents()
        panel._compute_psth_impl()
        self.app.processEvents()
        return panel

    def select(self, combo, code):
        combo.setCurrentIndex(combo.findData(code))
        self.app.processEvents()

    def test_views_and_styles_redraw_without_recomputing_the_psth(self):
        panel = self.show_groups()
        matrix = panel._last_mat.copy()
        titles = {}
        for mode in GROUP_VIEWS:
            self.select(panel.combo_metric_group_view, mode)
            result = next(iter(panel._last_metric_panels.values()))
            self.assertEqual(result["group_view"]["mode"], mode)
            self.assertIs(result["group_comparison"], result["group_view"]["between"])
            titles[mode] = panel.plot_metrics.getPlotItem().titleLabel.text
            self.assertTrue(panel.plot_metrics.note.toPlainText())
            self.assertTrue(panel.plot_metrics._note_wanted)
            for style in SUMMARY_STYLES:
                self.select(panel.combo_metric_style, style)
                self.assertEqual(panel.plot_metrics.summary_style, style)
                self.assertTrue(panel.plot_metrics._group_items)
        np.testing.assert_array_equal(panel._last_mat, matrix)
        self.assertIn("Post − Pre", titles["delta"])
        self.assertTrue(titles["post"].endswith("Post"))
        self.assertIn("Pre vs Post", titles["paired"])
        # WT rises and Het falls after the event; Pre does not differ.
        self.select(panel.combo_metric_group_view, "paired")
        view = next(iter(panel._last_metric_panels.values()))["group_view"]
        self.assertEqual([item["n_pairs"] for item in view["within"]], [N_PER_GROUP, N_PER_GROUP])
        self.assertGreater(view["within"][0]["mean_difference"], 0)
        self.assertLess(view["within"][1]["mean_difference"], 0)
        self.assertEqual(len(view["links"]), 2)

    def test_right_click_menu_and_settings_roundtrip(self):
        panel = self.show_groups()
        menus = panel.plot_metrics.getPlotItem().getViewBox().menu._pyber_metric_view_menus
        self.assertEqual([m.title() for m in menus], ["Group comparison", "Summary style"])
        menus[0].actions()[list(GROUP_VIEWS).index("post")].trigger()
        menus[1].actions()[list(SUMMARY_STYLES).index("box")].trigger()
        self.app.processEvents()
        self.assertEqual(panel._metric_group_view(), "post")
        self.assertEqual(panel._metric_summary_style(), "box")
        menus[0].aboutToShow.emit()
        self.assertEqual([a.isChecked() for a in menus[0].actions()],
                         [code == "post" for code in GROUP_VIEWS])
        # The global panel offers the style only.
        global_menus = panel.plot_global.getPlotItem().getViewBox().menu._pyber_metric_view_menus
        self.assertEqual([m.title() for m in global_menus], ["Summary style"])
        settings = panel._collect_settings()
        self.assertEqual((settings["metric_group_view"], settings["metric_summary_style"]), ("post", "box"))
        self.select(panel.combo_metric_group_view, "delta")
        self.select(panel.combo_metric_style, "points")
        panel._is_restoring_settings = True
        panel._apply_settings(settings)
        panel._is_restoring_settings = False
        self.assertEqual(panel._metric_group_view(), "post")
        self.assertEqual(panel._metric_summary_style(), "box")
        defaults = panel._default_settings_payload()
        self.assertEqual((defaults["metric_group_view"], defaults["metric_summary_style"]), ("delta", "points"))

    def test_exports_follow_the_view(self):
        from psth_metrics import export_selected_metrics
        panel = self.show_groups()
        with tempfile.TemporaryDirectory() as directory:
            self.select(panel.combo_metric_group_view, "post")
            base = str(Path(directory) / "post")
            export_selected_metrics(panel._last_metric_panels, base, True, False)
            with open(base + "_groups.csv", encoding="utf-8-sig") as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual({row["quantity"] for row in rows}, {"post"})
            self.assertFalse(Path(base + "_group_paired.csv").exists())
            self.select(panel.combo_metric_group_view, "paired")
            base = str(Path(directory) / "paired")
            export_selected_metrics(panel._last_metric_panels, base, True, False)
            with open(base + "_group_paired.csv", encoding="utf-8-sig") as stream:
                paired = list(csv.DictReader(stream))
            self.assertEqual({row["group"] for row in paired}, {"WT", "Het"})
            self.assertTrue(all(int(row["n_pairs"]) == N_PER_GROUP for row in paired))
            with open(base + "_group_rows.csv", encoding="utf-8-sig") as stream:
                animals = list(csv.DictReader(stream))
            self.assertEqual(len(animals), 2 * N_PER_GROUP * len(panel._last_metric_panels))
            self.assertTrue({"pre", "post", "post_minus_pre"} <= set(animals[0]))

    def test_color_scale_is_as_tall_as_the_heatmap_and_psth_has_no_generic_title(self):
        panel = self.show_groups()
        for _ in range(6):
            self.app.processEvents()
        panel._align_heat_colorbar()
        panel.heat_colorbar_widget.ci.layout.activate()
        for _ in range(6):
            self.app.processEvents()

        def edges(widget, view_box):
            """Global y of the top and bottom of a view box."""
            rect = view_box.sceneBoundingRect()
            return (widget.mapToGlobal(widget.mapFromScene(rect.topLeft())).y(),
                    widget.mapToGlobal(widget.mapFromScene(rect.bottomLeft())).y())

        heat_top, heat_bottom = edges(panel.plot_heat, panel.plot_heat.getPlotItem().getViewBox())
        bar_top, bar_bottom = edges(panel.heat_colorbar_widget, panel.heat_colorbar.getViewBox())
        self.assertGreater(heat_bottom - heat_top, 40)
        self.assertLessEqual(abs(bar_top - heat_top), 2)
        self.assertLessEqual(abs(bar_bottom - heat_bottom), 2)
        # The mean PSTH carries no generic title; its description is the tooltip.
        self.assertFalse(panel.plot_avg.getPlotItem().titleLabel.isVisible())
        self.assertIn("PSTH", panel.plot_avg.toolTip())
        self.assertIn("groups", panel.plot_avg.toolTip())

    def test_untested_comparisons_write_nothing_on_the_panel(self):
        """Fewer than 3 animals per group: no 'descriptive only' text, no brackets."""
        panel = self.show_groups()
        groups = panel.recording_groups
        groups.assign(["wt0", "het0"], None)          # leaves 2 animals per group
        self.app.processEvents()
        panel._compute_psth_impl()
        self.app.processEvents()
        for mode in GROUP_VIEWS:
            self.select(panel.combo_metric_group_view, mode)
            result = next(iter(panel._last_metric_panels.values()))
            self.assertTrue(np.isnan(result["group_comparison"]["p"]))
            self.assertEqual(result["group_view"]["brackets"], [])
            self.assertFalse(panel.plot_metrics._note_wanted)
            self.assertFalse(panel.plot_metrics.note.isVisibleTo(panel.plot_metrics.getPlotItem()))
            self.assertTrue(panel.plot_metrics._group_items)     # the data are still drawn

    def test_group_legend_hides_from_the_menu_and_moves_clear_of_curves(self):
        panel = self.show_groups()
        legend = panel.psth_group_legend
        self.assertTrue(legend.isVisible())
        self.assertIn(panel.act_group_legend, panel.plot_avg.getPlotItem().getViewBox().menu.actions())
        panel.act_group_legend.setChecked(False)
        self.assertFalse(legend.isVisible())
        self.assertFalse(panel._collect_settings()["style"]["group_legend_visible"])
        panel._compute_psth_impl()               # a recompute keeps it hidden
        self.app.processEvents()
        self.assertFalse(legend.isVisible())
        panel.act_group_legend.setChecked(True)
        self.assertTrue(legend.isVisible())
        # The chosen position covers no more curve samples than any other.
        name = panel._place_group_legend()
        view = panel.plot_avg.getPlotItem().getViewBox()
        box = legend.boundingRect()
        counts = legend_overlap_counts(panel._group_legend_curves, view.viewRange(),
                                       (view.size().width(), view.size().height()), (box.width(), box.height()))
        self.assertEqual(counts[name], min(counts.values()))


if __name__ == "__main__":
    unittest.main()
