"""Regression checks for heatmap contrast across changes in display units."""
import unittest
from unittest.mock import patch

import numpy as np
import test_postprocessing_empty_state as fixture


class HeatmapAutoscaleTests(unittest.TestCase):
    """Use real linked image, histogram and colorbar widgets with isolated settings."""

    @classmethod
    def setUpClass(cls):
        cls.app = fixture.QtWidgets.QApplication.instance() or fixture.QtWidgets.QApplication([])

    def setUp(self):
        fixture.PostprocessingEmptyStateTests.setUp(self)

    def tearDown(self):
        fixture.PostprocessingEmptyStateTests.tearDown(self)

    def render(self, matrix):
        """Provide a display matrix without changing any recording or metadata."""
        self.panel._last_mat = np.asarray(matrix, float)
        self.panel._last_tvec = np.linspace(-1, 1, self.panel._last_mat.shape[1])
        self.panel._render_heatmap(self.panel._last_mat, self.panel._last_tvec)

    def manual(self, low, high):
        self.panel.heat_lut.item.setLevels(low, high)
        self.panel._on_heatmap_levels_changed()
        self.assertTrue(self.panel._style['heatmap_levels_manual'])

    def test_normalization_change_discards_incompatible_limits(self):
        """The screenshot's old z-score range must not clip original/subtracted units."""
        panel = self.panel
        for normalization, matrix in (
            ('Original processed units', [[0.01, 0.02, 0.03], [0.04, 0.05, 0.06]]),
            ('Subtract baseline', [[-0.02, 0, 0.03], [0.01, -0.01, 0.02]]),
            ('Baseline z-score', [[-3, 0, 4], [2, -1, 6]]),
        ):
            self.render([[-15, -5, 0], [-10, -3, 1]])
            self.manual(-15, 0)
            panel.combo_psth_normalization.setCurrentText(normalization)
            self.assertFalse(panel._style['heatmap_levels_manual'])
            self.assertTrue(panel._psth_timer.isActive())
            panel._psth_timer.stop()
            panel._psth_pending = False
            self.render(matrix)
            np.testing.assert_allclose(panel.img.getLevels(), [np.min(matrix), np.max(matrix)])
            np.testing.assert_allclose(panel.heat_lut.item.getLevels(), panel.img.getLevels())

    def test_auto_button_and_contrast_modes_use_displayed_finite_values(self):
        matrix = np.arange(100, dtype=float).reshape(10, 10)
        matrix[0, 0] = np.nan
        self.render(matrix)
        original = matrix.copy()
        self.manual(-15, 0)
        self.panel.btn_auto_heat_scale.click()
        np.testing.assert_allclose(self.panel.img.getLevels(), [1, 99])
        self.manual(-15, 0)
        self.panel.combo_heat_scale.setCurrentIndex(1)
        np.testing.assert_allclose(self.panel.img.getLevels(), np.nanpercentile(matrix, [2, 98]))
        self.panel.combo_heat_scale.setCurrentIndex(2)
        np.testing.assert_allclose(self.panel.img.getLevels(), [-99, 99])
        np.testing.assert_equal(self.panel._last_mat, original)

    def test_programmatic_colormap_notifications_do_not_lock_scale(self):
        """A linked LUT can emit finished-level signals during style updates."""
        gradient = self.panel.heat_lut.item.gradient
        original = gradient.setColorMap
        def notifying_set(cmap):
            original(cmap)
            self.panel._on_heatmap_levels_changed()
        with patch.object(gradient, 'setColorMap', side_effect=notifying_set):
            self.render([[1, 2, 3]])
            self.panel._apply_plot_style()
            self.assertFalse(self.panel._style['heatmap_levels_manual'])
            self.render([[100, 200, 300]])
            np.testing.assert_allclose(self.panel.img.getLevels(), [100, 300])

    def test_manual_limits_survive_redraw_and_restoring_normalization(self):
        self.render([[1, 2, 3]])
        self.manual(1.2, 2.8)
        self.render([[1, 2, 3]])
        np.testing.assert_allclose(self.panel.img.getLevels(), [1.2, 2.8])
        self.panel._set_plot_preset('Paper')
        np.testing.assert_allclose(self.panel.img.getLevels(), [1.2, 2.8])
        self.panel._is_restoring_settings = True
        try:
            self.panel.combo_psth_normalization.setCurrentIndex(1)
            self.assertTrue(self.panel._style['heatmap_levels_manual'])
        finally:
            self.panel._is_restoring_settings = False

    def test_empty_and_constant_data_have_safe_scale_state(self):
        self.assertFalse(self.panel.btn_auto_heat_scale.isEnabled())
        self.render([[0, 0, 0]])
        low, high = self.panel.img.getLevels()
        self.assertLess(low, high)
        self.assertTrue(self.panel.btn_auto_heat_scale.isEnabled())
        self.render([[np.nan, np.nan]])
        self.assertFalse(self.panel.btn_auto_heat_scale.isEnabled())
