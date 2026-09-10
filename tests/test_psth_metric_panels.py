"""Numerical, export and rendering contracts for independent PSTH metric panels."""
import csv
import os
from pathlib import Path
import sys
import tempfile
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("PYBER_SMOKE_TEST", "1")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "pyBer"))

import h5py
import numpy as np
from PySide6 import QtCore, QtWidgets
from psth_metrics import metric_values, summarize_metrics, export_selected_metrics, draw_metric_matplotlib
from psth_metric_panels import MetricPanel


def fixture():
    """Known rows have a consistent unit increase in matched pre/post windows."""
    time = np.linspace(-2, 2, 81)
    matrix = np.arange(6)[:, None] * 0.1 + np.where(time < 0, 0, 1)[None, :]
    return matrix, time


class MetricNumericalTests(unittest.TestCase):
    def test_median_extrema_and_sample_standard_deviation(self):
        matrix = np.array([[1, 4, 2, 8, 3]], float)
        time = np.arange(5.0)
        for key, expected in (("median", 3), ("peak", 8), ("trough", 1),
                              ("std", np.std(matrix, ddof=1)), ("peak_latency", 3)):
            np.testing.assert_allclose(metric_values(matrix, time, 0, 4, key), [expected])

    def test_peak_delay_is_relative_to_window_start_and_flat_is_missing(self):
        time = np.arange(-4.0, 1)
        matrix = np.array([[1, 4, 2, 8, 3], [1, 1, 1, 1, 1]], float)
        values = metric_values(matrix, time, -4, 0, "peak_latency")
        self.assertEqual(values[0], 3)
        self.assertTrue(np.isnan(values[1]))

    def test_new_metrics_reject_cuts_and_incomplete_requested_coverage(self):
        matrix = np.array([[1, np.nan, 3, 4]], float)
        time = np.arange(4.0)
        for key in ("median", "peak", "trough", "std", "peak_latency"):
            self.assertTrue(np.isnan(metric_values(matrix, time, 0, 3, key)[0]))
            self.assertTrue(np.isnan(metric_values(np.ones((1, 4)), time, -1, 3, key)[0]))

    def test_mean_retains_legacy_available_bins_and_auc_exact_boundaries(self):
        time = np.array([0, 1, 2.0])
        self.assertEqual(metric_values(np.array([[1, np.nan, 3]]), time, 0, 2, "mean")[0], 2)
        self.assertAlmostEqual(metric_values(np.array([[0, 1, 2.0]]), time, 0.25, 1.75, "auc")[0], 1.5)

    def test_holm_adjustment_has_explicit_family_and_raw_pvalues(self):
        matrix, time = fixture()
        results = summarize_metrics(matrix, time, ["auc", "mean", "median"], (-1, -.1), (.1, 1), "z-score")
        for result in results.values():
            summary = result["summary"]
            self.assertEqual(summary["family_size"], 3)
            self.assertEqual(summary["paired_p"], .03125)
            self.assertEqual(summary["paired_p_holm"], .09375)
            self.assertAlmostEqual(summary["pre_median"], .225 if summary["metric_id"] == "auc" else .25)
            self.assertEqual(summary["paired_n"], 6)

    def test_pooled_group_rows_and_unequal_delay_windows_are_descriptive(self):
        matrix, time = fixture()
        pooled = summarize_metrics(matrix, time, ["mean"], (-1, -.1), (.1, 1), "z", independent_units=False)
        self.assertTrue(np.isnan(pooled["mean"]["summary"]["paired_p"]))
        unequal = summarize_metrics(matrix + time, time, ["peak_latency", "auc"], (-1.5, -.1), (.1, 1), "z")
        for result in unequal.values():
            self.assertTrue(np.isnan(result["summary"]["paired_p"]))
            self.assertIn("Unequal window", result["summary"]["assumption_note"])

    def test_csv_h5_export_all_metrics_units_and_complete_row_values(self):
        matrix, time = fixture()
        units = "Normalized fluorescence in original processed units"
        results = summarize_metrics(matrix, time, ["mean", "auc", "median"], (-1, -.1), (.1, 1), units,
                                    row_unit="files")
        with tempfile.TemporaryDirectory() as temporary:
            prefix = str(Path(temporary) / "metrics")
            export_selected_metrics(results, prefix)
            with open(prefix + "_selected.csv", encoding="utf-8-sig") as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(len(rows), 3)
            self.assertEqual(rows[0]["units"], units)
            with h5py.File(prefix + "_selected.h5", "r") as file:
                self.assertEqual(set(file), {"mean", "auc", "median"})
                self.assertEqual(file["auc"]["units"].asstr()[()], units + " · s")
                np.testing.assert_allclose(file["mean"]["post_values"][:], results["mean"]["post_values"])
                self.assertIn("averaged trial waveform", file["mean"]["reduction_level"].asstr()[()])


class MetricRenderTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

    def test_live_empty_result_and_populated_median_markers(self):
        matrix, time = fixture()
        plot = MetricPanel()
        result = summarize_metrics(matrix, time, ["median"], (-1, -.1), (.1, 1), "z")["median"]
        plot.show_result(result)
        np.testing.assert_allclose(plot.medians[0].getData()[1], [.25, .25])
        empty = summarize_metrics(np.empty((0, len(time))), time, ["median"], (-1, -.1), (.1, 1), "z")["median"]
        plot.show_result(empty)
        self.assertFalse(plot.note.isVisible())
        plot.show_result(None)
        self.assertIsNone(plot.result)
        plot.close()

    def test_matplotlib_accepts_empty_result(self):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        _, time = fixture()
        result = summarize_metrics(np.empty((0, len(time))), time, ["mean"], (-1, -.1), (.1, 1), "z")["mean"]
        figure, axis = plt.subplots()
        draw_metric_matplotlib(axis, result)
        with tempfile.TemporaryDirectory() as temporary:
            figure.savefig(Path(temporary) / "empty.svg")
        plt.close(figure)

    def test_summary_annotation_stays_inside_compact_plot(self):
        matrix, time = fixture()
        plot = MetricPanel()
        plot.setAttribute(QtCore.Qt.WidgetAttribute.WA_DontShowOnScreen)
        plot.resize(320, 250)
        plot.show()
        result = summarize_metrics(matrix, time, ["median"], (-1, -.1), (.1, 1), "z")["median"]
        plot.show_result(result)
        for _ in range(4):
            self.app.processEvents()
        view = plot.getViewBox().sceneBoundingRect()
        note = plot.note.sceneBoundingRect()
        self.assertGreaterEqual(note.top(), view.top())
        self.assertLessEqual(note.bottom(), view.bottom())
        self.assertGreaterEqual(note.left(), view.left())
        self.assertLessEqual(note.right(), view.right())
        plot.close()
        plot.deleteLater()


if __name__ == "__main__":
    unittest.main()
