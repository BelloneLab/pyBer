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

        windows.add_button.click()                             # the "+" beside the metric selector
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
