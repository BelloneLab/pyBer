"""GUI event filtering and annotation checks without constructing the full panel."""

import os
import sys
import unittest
from types import MethodType, SimpleNamespace

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("PYBER_SMOKE_TEST", "1")
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "pyBer"))

import numpy as np
import pyqtgraph as pg
from PySide6 import QtWidgets

from gui_postprocessing import PostProcessingPanel


def _combo(text):
    """Return only the text accessor needed by alignment-aware methods."""
    return SimpleNamespace(currentText=lambda: text)


def _filter_panel(enabled=False, minimum=0.0, maximum=0.0):
    """Construct the independent control state used by event selection."""
    panel = SimpleNamespace(
        cb_filter_events=SimpleNamespace(isChecked=lambda: enabled),
        spin_group_window=SimpleNamespace(value=lambda: 0.0),
        spin_event_start=SimpleNamespace(value=lambda: 1),
        spin_event_end=SimpleNamespace(value=lambda: 0),
        spin_dur_min=SimpleNamespace(value=lambda: minimum),
        spin_dur_max=SimpleNamespace(value=lambda: maximum),
        combo_align=_combo("Behavior (CSV/XLSX)"),
        combo_behavior_align=_combo("Align to onset"),
    )
    panel._group_close_events = MethodType(PostProcessingPanel._group_close_events, panel)
    return panel


class EventFilterTests(unittest.TestCase):
    """Disabling optional filters must never disable tuple integrity."""

    def test_disabled_filters_remove_nonfinite_times_and_sort_complete_tuples(self):
        times, durations = PostProcessingPanel._filter_events(
            _filter_panel(), [3, np.nan, 1, np.inf, 2], [30, 99, 10, 88, 20],
        )
        np.testing.assert_array_equal(times, [1, 2, 3])
        np.testing.assert_array_equal(durations, [10, 20, 30])

    def test_missing_duration_array_produces_one_unknown_duration_per_event(self):
        times, durations = PostProcessingPanel._filter_events(_filter_panel(), [2, 1], None)
        np.testing.assert_array_equal(times, [1, 2])
        self.assertEqual(durations.shape, times.shape)
        self.assertTrue(np.isnan(durations).all())

    def test_mismatched_duration_array_does_not_borrow_another_events_duration(self):
        _, durations = PostProcessingPanel._filter_events(_filter_panel(), [2, 1], [10])
        self.assertTrue(np.isnan(durations).all())

    def test_explicit_duration_bounds_exclude_all_unknown_durations(self):
        for controls in (_filter_panel(True, minimum=1), _filter_panel(True, maximum=2)):
            times, durations = PostProcessingPanel._filter_events(controls, [1, 2], [np.nan, np.nan])
            self.assertEqual(times.size, 0)
            self.assertEqual(durations.size, 0)

    def test_invalid_durations_remain_unknown_when_unfiltered(self):
        _, durations = PostProcessingPanel._filter_events(_filter_panel(), [1, 2, 3], [-1, np.inf, 0])
        np.testing.assert_allclose(durations, [np.nan, np.nan, 0], equal_nan=True)

    def test_duration_bounds_keep_only_finite_qualifying_tuples(self):
        times, durations = PostProcessingPanel._filter_events(_filter_panel(True, minimum=1, maximum=3),
                                                              [4, 1, 2, 3], [2, np.nan, 0.5, 5])
        np.testing.assert_array_equal(times, [4])
        np.testing.assert_array_equal(durations, [2])


class EventAnnotationTests(unittest.TestCase):
    """Inspect actual plot items to verify offset regions and label spacing."""

    @classmethod
    def setUpClass(cls):
        cls.app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

    def setUp(self):
        self.plot = pg.PlotWidget()
        self.plot.resize(600, 250)
        self.plot.show()
        self.app.processEvents()
        self.panel = SimpleNamespace(
            plot_trace=self.plot,
            event_lines=[], _event_labels=[], _event_regions=[],
            _trace_preview_events=np.array([5.0]),
            _trace_preview_durations=np.array([2.0]),
            _trace_preview_y_bounds=(-1.0, 1.0),
            combo_align=_combo("Behavior (CSV/XLSX)"),
            combo_behavior_align=_combo("Align to offset"),
        )
        self.panel._clear_event_annotations = MethodType(PostProcessingPanel._clear_event_annotations, self.panel)

    def tearDown(self):
        self.plot.close()
        self.plot.deleteLater()
        self.app.processEvents()

    def test_offset_region_ends_at_alignment_time(self):
        self.plot.setXRange(0, 10, padding=0)
        PostProcessingPanel._render_visible_event_annotations(self.panel)
        self.assertEqual(self.panel._event_regions[0].getRegion(), (3.0, 5.0))

    def test_offset_region_visible_when_its_alignment_marker_is_outside_view(self):
        self.plot.setXRange(3.5, 4.5, padding=0)
        PostProcessingPanel._render_visible_event_annotations(self.panel)
        self.assertEqual(len(self.panel._event_regions), 1)
        self.assertEqual(len(self.panel._event_labels), 0)

    def test_dense_event_labels_are_spaced_and_more_are_revealed_by_zoom(self):
        self.panel._trace_preview_events = np.arange(0, 100, 0.25)
        self.panel._trace_preview_durations = np.full(400, np.nan)
        self.plot.setXRange(0, 100, padding=0)
        PostProcessingPanel._render_visible_event_annotations(self.panel)
        full_count = len(self.panel._event_labels)
        full_fraction = full_count / 400
        positions = [self.plot.getViewBox().mapViewToScene(label.pos()).x() for label in self.panel._event_labels]
        self.assertTrue(np.all(np.diff(positions) >= 35.9))
        self.assertLess(full_count, 25)
        self.plot.setXRange(0, 2, padding=0)
        PostProcessingPanel._render_visible_event_annotations(self.panel)
        self.assertGreater(len(self.panel._event_labels) / 9, full_fraction)
        self.assertEqual(len(self.panel._event_labels), 9)


if __name__ == "__main__":
    unittest.main()
