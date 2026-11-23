"""Named PSTH metric windows: renamable Pre / Post, extra windows, one Holm family."""
import os
import tempfile
import unittest

import numpy as np

import test_group_view_dashboard as dashboard
from metric_windows import RenamableLabel, is_window_key, result_key
from psth_metrics import export_selected_metrics, summarize_metrics
from PySide6 import QtCore, QtTest, QtWidgets


def _bimodal_matrix():
    """20 trials: flat before 0, +1 from 0 to 1 s (early), -2 from 2 to 3 s (late)."""
    time = np.arange(-2.0, 4.0, 0.05)
    rng = np.random.default_rng(1)
    matrix = rng.normal(0, 0.05, (20, time.size))
    matrix[:, (time >= 0) & (time < 1)] += 1.0
    matrix[:, (time >= 2) & (time < 3)] -= 2.0
    return matrix, time


class MetricWindowsCoreTests(unittest.TestCase):
    def test_extra_windows_are_compared_with_pre_in_one_holm_family(self):
        matrix, time = _bimodal_matrix()
        base = summarize_metrics(matrix, time, ["mean"], (-1.0, 0.0), (0.0, 1.0), "z")
        results = summarize_metrics(matrix, time, ["mean"], (-1.0, 0.0), (0.0, 1.0), "z",
                                    pre_label="Baseline", post_label="Early",
                                    extra_windows=[("w1", "Late", (2.0, 3.0))])
        self.assertEqual(list(results), ["mean", result_key("mean", "w1")])
        self.assertTrue(is_window_key("mean@w1") and not is_window_key("mean"))
        early, late = results["mean"]["summary"], results["mean@w1"]["summary"]
        # The standard window is unchanged by the presence of another window.
        np.testing.assert_array_equal(results["mean"]["post_values"], base["mean"]["post_values"])
        self.assertAlmostEqual(early["post"] - early["pre"], 1.0, delta=0.05)
        self.assertAlmostEqual(late["post"] - late["pre"], -2.0, delta=0.05)
        self.assertEqual((early["pre_label"], early["post_label"]), ("Baseline", "Early"))
        self.assertEqual((late["pre_label"], late["post_label"], late["window_id"]), ("Baseline", "Late", "w1"))
        self.assertEqual(late["metric"], "Mean signal · Late")
        self.assertEqual((late["post_start_s"], late["post_end_s"]), (2.0, 3.0))
        # One Holm family across both windows: each raw p is multiplied by at most 2.
        self.assertEqual(early["family_size"], 2)
        self.assertEqual(late["family_size"], 2)
        for summary in (early, late):
            self.assertGreaterEqual(summary["paired_p_holm"], summary["paired_p"])
            self.assertLessEqual(summary["paired_p_holm"], 2 * summary["paired_p"] + 1e-15)
        self.assertEqual(base["mean"]["summary"]["family_size"], 1)
        self.assertEqual(base["mean"]["summary"]["post_label"], "Post")

    def test_export_keeps_one_row_per_window_with_its_label(self):
        matrix, time = _bimodal_matrix()
        results = summarize_metrics(matrix, time, ["mean", "auc"], (-1.0, 0.0), (0.0, 1.0), "z",
                                    extra_windows=[("w1", "Late", (2.0, 3.0))])
        with tempfile.TemporaryDirectory() as directory:
            prefix = os.path.join(directory, "metrics")
            export_selected_metrics(results, prefix, write_csv=True, write_h5=True)
            import csv
            rows = list(csv.DictReader(open(prefix + "_selected.csv", encoding="utf-8-sig")))
            self.assertEqual([(row["metric_id"], row["window_label"]) for row in rows],
                             [("mean", "Post"), ("auc", "Post"), ("mean", "Late"), ("auc", "Late")])
            per_row = list(csv.DictReader(open(prefix + "_rows.csv", encoding="utf-8-sig")))
            self.assertEqual({row["metric_id"] for row in per_row}, {"mean", "auc", "mean@w1", "auc@w1"})
            import h5py
            with h5py.File(prefix + "_selected.h5", "r") as handle:
                self.assertIn("mean@w1", handle)
                self.assertEqual(handle["mean@w1"]["post_values"].shape[0], matrix.shape[0])


class AllWindowsComparisonTests(unittest.TestCase):
    """Pre vs Post vs Late in one panel: omnibus test, every pair, one Holm family."""

    PRE, POST, LATE = (-1.0, 0.0), (0.0, 1.0), ("w1", "Late", (2.0, 3.0))

    def _all(self, metrics=("mean",), late=None, **kwargs):
        matrix, time = _bimodal_matrix()
        return summarize_metrics(matrix, time, list(metrics), self.PRE, self.POST, "z",
                                 extra_windows=[late or self.LATE], compare="all", **kwargs), matrix, time

    def test_one_panel_per_metric_with_friedman_and_every_pair(self):
        from scipy.stats import binomtest, friedmanchisquare
        results, matrix, time = self._all()
        self.assertEqual(list(results), ["mean"])                 # no separate "mean@w1" panel
        result = results["mean"]
        windows = result["windows"]
        self.assertEqual([(w["id"], w["label"]) for w in windows], [("pre", "Pre"), ("post", "Post"), ("w1", "Late")])
        # Window values are the ones of the "each window with Pre" mode: nothing is recomputed differently.
        separate = summarize_metrics(matrix, time, ["mean"], self.PRE, self.POST, "z", extra_windows=[self.LATE])
        np.testing.assert_array_equal(windows[1]["values"], separate["mean"]["post_values"])
        np.testing.assert_array_equal(windows[2]["values"], separate["mean@w1"]["post_values"])
        # Omnibus: independent reference computed directly with SciPy.
        omnibus = result["window_comparison"]["omnibus"]
        reference = friedmanchisquare(*[w["values"] for w in windows])
        self.assertEqual(omnibus["test"], "Friedman test")
        self.assertAlmostEqual(omnibus["statistic"], reference.statistic)
        self.assertAlmostEqual(omnibus["p"], reference.pvalue)
        self.assertEqual((omnibus["n"], omnibus["df"]), (20, 2))
        self.assertAlmostEqual(omnibus["kendall_w"], 1.0)         # Late < Pre < Post in every trial
        # Pairwise: exact sign test of the three pairs, Holm across the three.
        pairs = result["window_comparison"]["pairs"]
        self.assertEqual([(p["a_label"], p["b_label"]) for p in pairs],
                         [("Pre", "Post"), ("Pre", "Late"), ("Post", "Late")])
        raw = binomtest(20, 20, 0.5).pvalue
        for pair in pairs:
            self.assertAlmostEqual(pair["p"], raw)
            self.assertAlmostEqual(pair["p_holm"], 3 * raw)
            self.assertEqual(pair["n_pairs"], 20)
        self.assertAlmostEqual(pairs[2]["mean_difference"], -3.0, delta=0.05)   # Late minus Post
        summary = result["summary"]
        self.assertEqual(summary["family_size"], 3)
        self.assertEqual(summary["paired_p_holm"], pairs[0]["p_holm"])
        self.assertEqual((summary["comparison_mode"], summary["window_labels"]), ("all windows", "Pre | Post | Late"))
        self.assertAlmostEqual(summary["omnibus_p"], reference.pvalue)
        # The drawable panel: three columns, rows joined, one bracket per pair on two levels.
        view = result["window_view"]
        self.assertEqual(view["title"], "Mean signal: Pre vs Post vs Late")
        self.assertEqual([text for _x, text in view["ticks"]], ["Pre", "Post", "Late"])
        self.assertEqual((len(view["columns"]), len(view["links"]), len(view["brackets"])), (3, 2, 3))
        self.assertEqual(sorted(b["level"] for b in view["brackets"]), [0, 0, 1])
        self.assertTrue(all(b["text"].startswith("***") for b in view["brackets"]))
        self.assertIn("Friedman", view["note"])
        self.assertIn("Holm", view["note"])

    def test_without_an_extra_window_the_mode_changes_nothing(self):
        matrix, time = _bimodal_matrix()
        plain = summarize_metrics(matrix, time, ["mean"], self.PRE, self.POST, "z")
        same = summarize_metrics(matrix, time, ["mean"], self.PRE, self.POST, "z", compare="all")
        np.testing.assert_equal(plain["mean"]["summary"], same["mean"]["summary"])   # NaN-aware
        self.assertNotIn("window_view", same["mean"])

    def test_missing_rows_unequal_lengths_and_pooled_trials(self):
        matrix, time = _bimodal_matrix()
        matrix[0, (time >= 2) & (time < 3)] = np.nan                 # trial 1 has no Late value
        results = summarize_metrics(matrix, time, ["median"], self.PRE, self.POST, "z",
                                    extra_windows=[self.LATE], compare="all")
        comparison = results["median"]["window_comparison"]
        self.assertEqual(comparison["omnibus"]["n"], 19)             # complete rows only
        self.assertEqual([p["n_pairs"] for p in comparison["pairs"]], [20, 19, 19])
        # AUC depends on the window length: a 2 s Late window is not tested against 1 s windows.
        results, _matrix, _time = self._all(metrics=("auc",), late=("w1", "Late", (2.0, 4.0)))
        comparison = results["auc"]["window_comparison"]
        self.assertTrue(np.isfinite(comparison["pairs"][0]["p"]))    # Pre vs Post, equal lengths
        self.assertTrue(np.isnan(comparison["pairs"][1]["p"]) and np.isnan(comparison["pairs"][2]["p"]))
        self.assertEqual(comparison["omnibus"]["test"], "Descriptive only")
        self.assertIn("Unequal window lengths", comparison["pairs"][2]["note"])
        self.assertEqual(results["auc"]["summary"]["family_size"], 1)
        self.assertEqual(len(results["auc"]["window_view"]["brackets"]), 1)
        # Pooled trials of several animals: nothing is tested and nothing is annotated.
        results, _matrix, _time = self._all(independent_units=False)
        comparison = results["mean"]["window_comparison"]
        self.assertTrue(all(np.isnan(p["p"]) for p in comparison["pairs"]))
        self.assertTrue(np.isnan(comparison["omnibus"]["p"]))
        self.assertEqual((results["mean"]["window_view"]["note"], results["mean"]["window_view"]["brackets"]),
                         ("", []))

    def test_holm_family_spans_the_metrics_and_many_windows_show_significant_pairs_only(self):
        results, matrix, time = self._all(metrics=("mean", "median"))
        self.assertEqual({r["summary"]["family_size"] for r in results.values()}, {6})
        self.assertEqual(results["mean"]["window_comparison"]["omnibus"]["family_size"], 2)
        self.assertIn("Holm p=", results["mean"]["window_view"]["note"])     # omnibus adjusted across metrics
        # Four windows: the flat "Rest" window does not differ from Pre, so that bracket is not drawn.
        rng = np.random.default_rng(5)
        flat = matrix.copy()
        flat[:, (time >= 3) & (time < 4)] = flat[:, (time >= -1) & (time < 0)] + rng.normal(0, 1e-3, (20, 20))
        results = summarize_metrics(flat, time, ["mean"], self.PRE, self.POST, "z", compare="all",
                                    extra_windows=[self.LATE, ("w2", "Rest", (3.0, 4.0))])
        view = results["mean"]["window_view"]
        self.assertEqual(view["title"], "Mean signal: 4 windows")
        self.assertEqual(len(results["mean"]["window_comparison"]["pairs"]), 6)
        self.assertEqual(len(view["brackets"]), 5)                           # Pre vs Rest is not significant

    def test_export_writes_window_tables_and_the_same_panel_as_on_screen(self):
        import csv
        import h5py
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from psth_metrics import draw_metric_matplotlib
        results, matrix, _time = self._all(metrics=("mean", "auc"))
        with tempfile.TemporaryDirectory() as directory:
            prefix = os.path.join(directory, "metrics")
            export_selected_metrics(results, prefix, write_csv=True, write_h5=True)

            def read(suffix):
                with open(prefix + suffix, encoding="utf-8-sig") as stream:
                    return list(csv.DictReader(stream))

            self.assertEqual([(r["metric_id"], r["window_label"]) for r in read("_windows.csv")],
                             [("mean", "Pre"), ("mean", "Post"), ("mean", "Late"),
                              ("auc", "Pre"), ("auc", "Post"), ("auc", "Late")])
            pairs = read("_window_pairs.csv")
            self.assertEqual(len(pairs), 6)
            self.assertEqual((pairs[2]["window_a"], pairs[2]["window_b"], pairs[2]["family_size"]),
                             ("Post", "Late", "6"))
            self.assertEqual(len(read("_window_rows.csv")), 2 * 3 * matrix.shape[0])
            self.assertEqual(read("_selected.csv")[0]["omnibus_test"], "Friedman test")
            with h5py.File(prefix + "_selected.h5", "r") as handle:
                self.assertEqual(handle["mean/windows/w1/values"].shape[0], matrix.shape[0])
                self.assertIn("post_vs_w1", handle["mean/window_pairs"])
        figure, axis = plt.subplots()
        results["mean"]["window_style"] = "bars"
        draw_metric_matplotlib(axis, results["mean"])
        self.assertEqual([label.get_text() for label in axis.get_xticklabels()], ["Pre", "Post", "Late"])
        self.assertEqual(axis.get_title(), "Mean signal: Pre vs Post vs Late")
        plt.close(figure)


class RenamableLabelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

    def test_double_click_renames_and_escape_or_empty_keeps_the_name(self):
        label = RenamableLabel("Post", " window")
        label.resize(120, 24)
        label.show()
        names = []
        label.renamed.connect(names.append)
        self.assertEqual(label.text(), "Post window")
        QtTest.QTest.mouseDClick(label, QtCore.Qt.MouseButton.LeftButton)
        editor = label.findChild(QtWidgets.QLineEdit)
        self.assertEqual(editor.text(), "Post")              # the suffix is not edited
        editor.setText("Early")
        editor.editingFinished.emit()
        self.assertEqual((label.name(), label.text(), names), ("Early", "Early window", ["Early"]))
        label.start_editing()
        editor = label.findChild(QtWidgets.QLineEdit)
        editor.setText("oops")
        QtTest.QTest.keyClick(editor, QtCore.Qt.Key.Key_Escape)
        self.assertEqual(label.name(), "Early")
        label.start_editing()
        editor = label._editor
        editor.setText("   ")
        editor.editingFinished.emit()
        self.assertEqual((label.name(), names), ("Early", ["Early"]))
        label.deleteLater()


class MetricWindowsPanelTests(dashboard.GroupViewDashboardTests):
    """Six synthetic recordings with a response in the 2 s after each event."""

    def _ready(self):
        self.load()
        panel = self.panel
        panel.cb_metrics.setChecked(True)
        panel._psth_timer.stop()
        panel._compute_psth_impl()
        self.app.processEvents()
        return panel, panel.metric_windows

    def test_plus_adds_a_labelled_window_with_its_own_panel(self):
        panel, windows = self._ready()
        self.assertEqual(panel.lbl_metric_pre.text(), "Pre window")
        self.assertEqual(panel.lbl_metric_post.text(), "Post window")
        self.assertEqual(windows.windows(), [])
        primary = next(iter(panel._last_metric_panels))
        self.assertEqual([key for key in panel._last_metric_panels if is_window_key(key)], [])

        windows.set_compare_mode("pre")                        # one panel per window, each with Pre
        windows.add_button.click()                             # the "+" below the last window
        panel._psth_timer.stop()
        panel._compute_psth_impl()
        self.app.processEvents()
        row = windows._rows[0]
        self.assertEqual(row.label.name(), "Late")
        # A real row of the metrics form, right below the Post row.
        form = windows._form
        self.assertEqual(form.getWidgetPosition(row)[0], form.getWidgetPosition(panel.lbl_metric_post)[0] + 1)
        self.assertIs(form.labelForField(row), row.label)
        self.assertEqual(row.start.value(), panel.spin_metric_post1.value())   # starts where Post ends
        key = result_key(primary, row.window_id)
        self.assertIn(key, panel._last_metric_panels)
        summary = panel._last_metric_panels[key]["summary"]
        self.assertEqual((summary["pre_label"], summary["post_label"]), ("Pre", "Late"))
        self.assertEqual(panel._last_metric_panels[primary]["summary"]["family_size"],
                         summary["family_size"])
        plot = panel._extra_metric_plots[key]
        self.assertIn(plot, windows.visible_plots())
        self.assertFalse(plot.isHidden())
        ticks = plot.getAxis("bottom")._tickLevels[0]
        self.assertEqual([text for _value, text in ticks], ["Pre", "Late"])
        self.assertEqual(len(windows._regions), 1)             # shaded on the mean PSTH

        # Rename by the labels: the standard Post and the new window.
        panel.lbl_metric_post.set_name("Early", emit=True)
        row.label.set_name("Rebound", emit=True)
        panel._psth_timer.stop()
        panel._compute_psth_impl()
        self.assertEqual(panel._last_metric_panels[primary]["summary"]["post_label"], "Early")
        self.assertEqual(panel._last_metric_panels[key]["summary"]["post_label"], "Rebound")
        self.assertEqual(panel.lbl_metric_post.text(), "Early window")

        # Saved with the settings and restored.
        state = panel._collect_settings()["metric_windows"]
        self.assertEqual((state["post_label"], state["windows"][0]["label"]), ("Early", "Rebound"))
        windows.set_state(None)
        self.assertEqual((windows.post_label(), windows.windows()), ("Post", []))
        windows.set_state(state)
        self.assertEqual(windows.post_label(), "Early")
        self.assertEqual([w["label"] for w in windows.windows()], ["Rebound"])

        # Removing the window removes its panel, shading and result.
        windows._rows[0].remove_button.click()
        panel._psth_timer.stop()
        panel._compute_psth_impl()
        self.app.processEvents()
        self.assertEqual(windows.windows(), [])
        self.assertEqual([k for k in panel._extra_metric_plots if is_window_key(k)], [])
        self.assertEqual([k for k in panel._last_metric_panels if is_window_key(k)], [])
        self.assertEqual(windows._regions, [])

    def test_add_button_follows_the_last_window_and_has_no_limit(self):
        panel, windows = self._ready()
        form = windows._form

        def row_of(widget):
            return form.getWidgetPosition(widget)[0]

        post_row = row_of(windows._anchor)
        self.assertEqual(row_of(windows.add_row), post_row + 1)          # right below Post
        self.assertIs(windows.add_button.parentWidget(), windows.add_row)
        self.assertIsNot(windows.add_button.parentWidget(), panel.metric_selection_widget)
        panel._psth_timer.stop()
        for count in range(1, 11):                                       # more than the former limit of 6
            windows.add_button.click()
            self.assertEqual(len(windows.windows()), count)
            self.assertEqual(row_of(windows._rows[-1]), post_row + count)
            self.assertEqual(row_of(windows.add_row), post_row + count + 1)   # still below the last window
        panel._psth_timer.stop()
        self.assertEqual(len({w["id"] for w in windows.windows()}), 10)
        self.assertEqual(len(windows.valid_windows()), 10)
        # Each new window starts where the previous one ends.
        ends = [w["end"] for w in windows.windows()]
        starts = [w["start"] for w in windows.windows()]
        self.assertEqual(starts[1:], ends[:-1])
        state = windows.state()
        windows.set_state({"windows": []})
        self.assertEqual(row_of(windows.add_row), post_row + 1)
        windows.set_state(state)
        self.assertEqual(len(windows.windows()), 10)                     # all restored, none dropped
        self.assertEqual(row_of(windows.add_row), post_row + 11)
        panel._toggle_metrics_panel(True)
        self.assertTrue(windows.add_row.isHidden())
        panel._toggle_metrics_panel(False)
        self.assertFalse(windows.add_row.isHidden())
        panel._psth_timer.stop()

    def test_added_window_is_compared_with_pre_and_post_in_one_panel(self):
        panel, windows = self._ready()
        form = windows._form
        self.assertFalse(form.isRowVisible(windows.compare_combo))       # nothing to choose yet
        primary = next(iter(panel._last_metric_panels))
        messages = []
        panel.statusUpdate.connect(lambda text, _ms: messages.append(text))

        windows.add_button.click()
        panel._psth_timer.stop()
        panel._compute_psth_impl()
        self.app.processEvents()
        # Proposed by default: the choice appears and the panel shows the three windows.
        self.assertTrue(form.isRowVisible(windows.compare_combo))
        self.assertEqual(form.getWidgetPosition(windows.compare_combo)[0],
                         form.getWidgetPosition(windows.add_row)[0] + 1)
        self.assertEqual(windows.compare_mode(), "all")
        self.assertTrue(any("Pre vs Post vs Late" in text for text in messages))
        self.assertEqual([key for key in panel._last_metric_panels if is_window_key(key)], [])
        result = panel._last_metric_panels[primary]
        self.assertEqual([w["label"] for w in result["windows"]], ["Pre", "Post", "Late"])
        self.assertEqual(len(result["window_comparison"]["pairs"]), 3)
        self.assertEqual(result["window_style"], panel._metric_summary_style())
        plot = panel.plot_metrics
        self.assertEqual([text for _x, text in plot.getAxis("bottom")._tickLevels[0]], ["Pre", "Post", "Late"])
        self.assertTrue(plot.getPlotItem().titleLabel.text.endswith("Pre vs Post vs Late"))
        self.assertEqual(windows.visible_plots(), [])                    # no separate Late panel
        self.assertEqual(len(windows._regions), 1)                       # Late is still shaded on the PSTH

        # The summary style applies to the panel (bars, box) without recomputing the statistics.
        before = result["window_comparison"]["pairs"][2]["p"]
        panel.combo_metric_style.setCurrentIndex(panel.combo_metric_style.findData("bars"))
        self.assertEqual(panel._last_metric_panels[primary]["window_style"], "bars")
        after = panel._last_metric_panels[primary]["window_comparison"]["pairs"][2]["p"]
        self.assertTrue(before == after or (np.isnan(before) and np.isnan(after)))

        # Right-click menu of the panel offers the same choice.
        menu = plot.getPlotItem().getViewBox().menu
        titles = [sub.title() for sub in menu._pyber_metric_view_menus]
        self.assertEqual(titles, ["Group comparison", "Compare windows", "Summary style"])
        menu.aboutToShow.emit()
        self.assertTrue(menu._pyber_metric_view_menus[1].menuAction().isVisible())

        # "Each window with Pre": back to one panel per window.
        windows.compare_combo.setCurrentIndex(windows.compare_combo.findData("pre"))
        panel._psth_timer.stop()
        panel._compute_psth_impl()
        self.app.processEvents()
        key = result_key(primary, windows._rows[0].window_id)
        self.assertIn(key, panel._last_metric_panels)
        self.assertNotIn("window_view", panel._last_metric_panels[primary])
        self.assertEqual([text for _x, text in plot.getAxis("bottom")._tickLevels[0]], ["Pre", "Post"])

        # Saved and restored with the settings.
        state = panel._collect_settings()["metric_windows"]
        self.assertEqual(state["compare"], "pre")
        windows.set_state(None)
        self.assertEqual(windows.compare_mode(), "all")
        self.assertFalse(form.isRowVisible(windows.compare_combo))       # no window left
        windows.set_state(state)
        self.assertEqual(windows.compare_mode(), "pre")
        self.assertTrue(form.isRowVisible(windows.compare_combo))
        panel._toggle_metrics_panel(True)
        self.assertFalse(form.isRowVisible(windows.compare_combo))
        panel._toggle_metrics_panel(False)
        self.assertTrue(form.isRowVisible(windows.compare_combo))
        panel._psth_timer.stop()

    def test_group_comparison_keeps_one_panel_per_window(self):
        panel, windows = self._ready()
        windows.add_window("Late", 2.0, 4.0)
        self.assertEqual(windows.compare_mode(), "all")
        panel.cb_group_view.setChecked(True)                             # All groups: WT vs Het
        panel._psth_timer.stop()
        panel._compute_psth_impl()
        self.app.processEvents()
        primary = next(iter(panel._last_metric_panels))
        key = result_key(primary, windows._rows[0].window_id)
        self.assertIn(key, panel._last_metric_panels)                    # one quantity per group: one Pre / Post pair
        for name in (primary, key):
            self.assertIn("group_view", panel._last_metric_panels[name])
            self.assertNotIn("window_view", panel._last_metric_panels[name])
        self.assertTrue(panel._last_metric_panels[key]["group_view"]["title"].endswith("Late − Pre"))
        panel._psth_timer.stop()

    def test_paired_group_view_shows_every_window_per_group(self):
        panel, windows = self._ready()
        windows.add_window("Late", 2.0, 4.0)
        panel.cb_group_view.setChecked(True)                             # All groups: WT vs Het
        panel.combo_metric_group_view.setCurrentIndex(panel.combo_metric_group_view.findData("paired"))
        panel._psth_timer.stop()
        panel._compute_psth_impl()
        self.app.processEvents()
        primary = next(iter(panel._last_metric_panels))
        self.assertEqual([k for k in panel._last_metric_panels if is_window_key(k)], [])   # one panel per metric
        result = panel._last_metric_panels[primary]
        view = result["group_view"]
        self.assertNotIn("window_view", result)
        self.assertEqual(view["mode"], "group_windows")
        self.assertTrue(view["title"].endswith("Pre vs Post vs Late"))
        # Pre, Post, Late for each of the two groups; the group caption sits under the middle window.
        self.assertEqual([text.split("\n")[0] for _x, text in view["ticks"]], ["Pre", "Post", "Late"] * 2)
        self.assertEqual([text for _x, text in view["ticks"] if "\n" in text],
                         [f"Post\nWT\nn={dashboard.N_PER_GROUP}", f"Post\nHet\nn={dashboard.N_PER_GROUP}"])
        self.assertEqual((len(view["columns"]), len(view["links"])), (6, 4))
        # One value per animal in every window column, taken from the same window values.
        late_columns = view["columns"][2::3]
        self.assertEqual([column["n"] for column in late_columns], [dashboard.N_PER_GROUP] * 2)
        late = np.asarray(result["windows"][2]["values"], float)
        if late.size == 2 * dashboard.N_PER_GROUP:                       # rows are animals: identical values
            np.testing.assert_allclose(np.sort(np.concatenate([column["values"] for column in late_columns])),
                                       np.sort(late))
        # Three pairs per group tested within group; change from Pre compared between groups per window.
        self.assertEqual(len(view["within"]), 6)
        self.assertEqual([entry["window"] for entry in view["between_windows"]], ["Post", "Late"])
        self.assertIn("Late", result["group_rows"][0])                   # per-animal export has the extra window
        ticks = [text.split("\n")[0] for _x, text in panel.plot_metrics.getAxis("bottom")._tickLevels[0]]
        self.assertEqual(ticks, ["Pre", "Post", "Late"] * 2)
        # Another group comparison shows one quantity per group: back to one panel per window.
        panel.combo_metric_group_view.setCurrentIndex(panel.combo_metric_group_view.findData("delta"))
        self.assertIn(result_key(primary, windows._rows[0].window_id), panel._last_metric_panels)
        panel._psth_timer.stop()

    def test_invalid_window_is_ignored_and_metrics_off_disables_the_rows(self):
        panel, windows = self._ready()
        row = windows.add_window("Backwards", 3.0, 1.0)        # end before start
        self.assertEqual(windows.valid_windows(), [])
        panel._psth_timer.stop()
        panel._compute_psth_impl()
        self.assertEqual([k for k in panel._last_metric_panels if is_window_key(k)], [])
        panel.cb_metrics.setChecked(False)
        self.assertFalse(row.isEnabled())
        self.assertFalse(windows.add_button.isEnabled())
        self.assertEqual(windows._regions, [])


for _name in [name for name in dir(dashboard.GroupViewDashboardTests) if name.startswith("test_")]:
    setattr(MetricWindowsPanelTests, _name, None)


if __name__ == "__main__":
    unittest.main()
