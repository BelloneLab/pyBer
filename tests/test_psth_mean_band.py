"""Check the actual filled PSTH geometry where trial coverage changes."""
import unittest

import numpy as np
import pyqtgraph as pg
from PySide6 import QtCore

import test_postprocessing_empty_state as fixture
from psth_groups import GroupBlock, GroupDisplay


class MeanBandTests(unittest.TestCase):
    """Use two known traces so mean ± sample SEM is exactly [0, 2]."""

    setUpClass = classmethod(fixture.PostprocessingEmptyStateTests.setUpClass.__func__)
    setUp = fixture.PostprocessingEmptyStateTests.setUp
    tearDown = fixture.PostprocessingEmptyStateTests.tearDown

    def response(self):
        """The second trial drops out at time zero, leaving its SEM undefined."""
        time = np.arange(-4, 5, dtype=float)
        rows = np.vstack([np.zeros_like(time), np.full_like(time, 2.0)])
        rows[1, 4] = np.nan
        return time, rows

    def assert_band(self, band, high, low):
        """Test the QPainterPath as well as its input, including the interior gap."""
        np.testing.assert_allclose(high.getOriginalDataset()[1], [2, 2, 2, 2, np.nan, 2, 2, 2, 2])
        np.testing.assert_allclose(low.getOriginalDataset()[1], [0, 0, 0, 0, np.nan, 0, 0, 0, 0])
        path = band.path()
        self.assertTrue(path.contains(QtCore.QPointF(-2, 1)))
        self.assertTrue(path.contains(QtCore.QPointF(2, 1)))
        self.assertFalse(path.contains(QtCore.QPointF(0, 1)))
        self.assertFalse(path.contains(QtCore.QPointF(2, 2.1)))
        self.assertNotEqual(band.pen().style(), QtCore.Qt.PenStyle.NoPen)

    def test_individual_band_leaves_single_observation_bins_empty(self):
        time, rows = self.response()
        panel = self.panel
        panel.set_trace_view("show", "both")
        panel._render_avg(rows, time)
        self.assert_band(panel.sem_band, panel.curve_sem_hi, panel.curve_sem_lo)
        # With just one remaining row the mean persists and the entire band vanishes.
        panel._render_avg(rows[:1], time)
        self.assertTrue(panel.sem_band.path().isEmpty())
        np.testing.assert_array_equal(panel.curve_avg.getOriginalDataset()[1], rows[0])

    def test_group_band_uses_within_group_coverage(self):
        time, rows = self.response()
        panel = self.panel
        matrix = np.vstack([rows, rows[0]])
        panel._group_display = GroupDisplay(
            "all", matrix, ["a", "b", "c"],
            [GroupBlock("g1", "Control", "#208060", 0, 2), GroupBlock("g2", "Test", "#903020", 2, 3)])
        panel.set_trace_view("show", "both")
        panel._render_avg(matrix, time)
        high, low, band = panel._group_avg_items[:3]
        self.assert_band(band, high, low)
        self.assertTrue(panel._group_mean_bands[1].path().isEmpty())

    def test_behavior_band_keeps_undefined_sem_when_the_mean_is_finite(self):
        time, rows = self.response()
        view = self.panel.multi_behavior_view
        entry = {"name": "Response", "matrix": rows, "tvec": time}
        view.show_entries([entry], color_map=pg.colormap.get("viridis"), levels=(-1, 2),
                          units="z", time_label="Time (s)", trace_view={"show": "both"},
                          missing_mode="interpolate", missing_trim=True)
        column = view.columns[0]
        self.assert_band(column.sem_band, column.sem_high, column.sem_low)
