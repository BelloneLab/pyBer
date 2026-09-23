"""Heatmap color adjustments change rendering, never the underlying analysis."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from PySide6 import QtCore, QtWidgets
import test_postprocessing_empty_state as fixture
from heatmap_display import color_limits, color_map, matplotlib_color_map


class HeatmapColorTests(unittest.TestCase):
    setUpClass = classmethod(fixture.PostprocessingEmptyStateTests.setUpClass.__func__)
    setUp = fixture.PostprocessingEmptyStateTests.setUp
    tearDown = fixture.PostprocessingEmptyStateTests.tearDown

    def render(self, matrix):
        panel = self.panel
        panel._last_mat = np.asarray(matrix, float).copy()
        panel._last_tvec = np.linspace(-1, 1, panel._last_mat.shape[1])
        panel._render_heatmap(panel._last_mat, panel._last_tvec)

    def assert_levels(self, expected):
        panel = self.panel
        np.testing.assert_allclose(panel.img.getLevels(), expected)
        np.testing.assert_allclose(panel.heat_lut.item.getLevels(), expected)
        np.testing.assert_allclose(panel.heat_colorbar.values, expected)
        controls = panel.heatmap_color_controls
        np.testing.assert_allclose([controls.minimum.value(), controls.maximum.value()], expected, atol=1e-9)

    def test_colorbar_drag_and_histogram_keep_all_scales_linked(self):
        panel = self.panel
        self.render([[.001, .002, .003, .004]])
        original = panel._last_mat.copy()
        with patch.object(panel, '_compute_psth') as compute:
            self.assertTrue(panel.heat_colorbar.interactive)
            panel.heat_colorbar.region.setRegion((80, 176))
            levels = panel.heat_colorbar.values
            self.assertGreater(levels[0], .001)
            self.assertLess(levels[1], .004)
            self.assertTrue(panel._style['heatmap_levels_manual'])
            self.assert_levels(levels)
            panel.heat_lut.item.region.setRegion((.0015, .0035))
            self.assert_levels((.0015, .0035))
            compute.assert_not_called()
        np.testing.assert_array_equal(panel._last_mat, original)

    def test_exact_limits_validation_and_auto_do_not_change_analysis(self):
        panel = self.panel
        self.render([[1, 2, 3], [4, 5, 100]])
        original = panel._last_mat.copy()
        baseline = panel.spin_b0.value(), panel.spin_b1.value()
        controls = panel.heatmap_color_controls
        collapsed_height = panel.row_heat.minimumHeight()
        controls.adjust.click()
        self.assertFalse(panel.heat_lut.isHidden())
        self.assertGreater(panel.row_heat.minimumHeight(), collapsed_height)
        with patch.object(panel, '_compute_psth') as compute:
            controls.minimum.setValue(1.5)
            controls.maximum.setValue(5)
            controls.maximum.editingFinished.emit()
            self.assert_levels((1.5, 5))
            self.assertIn('Fixed', controls.summary.text())
            controls.minimum.setValue(7)
            controls.minimum.editingFinished.emit()
            self.assertFalse(controls.error.isHidden())
            np.testing.assert_allclose(panel.img.getLevels(), (1.5, 5))
            self.assertFalse(panel._set_heatmap_display_levels(float('nan'), 10))
            self.assertFalse(panel._set_heatmap_display_levels(10, 10))
            panel._psth_pending = False
            controls.auto.click()
            self.assert_levels((1, 100))
            self.assertIn('Auto', controls.summary.text())
            compute.assert_not_called()
        np.testing.assert_array_equal(panel._last_mat, original)
        self.assertEqual((panel.spin_b0.value(), panel.spin_b1.value()), baseline)

    def test_color_controls_work_in_both_modes_and_follow_view_menu(self):
        panel = self.panel
        self.render(np.arange(100).reshape(10, 10))
        controls = panel.heatmap_color_controls
        for category in ('zone', 'behavior'):
            panel._set_event_category(category)
            self.assertFalse(controls.isHidden())
            controls.mode.setCurrentIndex(1)
            controls.mode.activated.emit(1)
            self.assertEqual(panel.combo_heat_scale.currentIndex(), 1)
            self.assert_levels(np.percentile(np.arange(100), [2, 98]))
            panel.combo_heat_scale.setCurrentIndex(2)
            self.assertEqual(controls.mode.currentIndex(), 2)
            self.assert_levels((-99, 99))
        panel.combo_view_layout.setCurrentText('Trace focus')
        self.assertTrue(controls.isHidden())
        panel.combo_view_layout.setCurrentText('Standard')
        self.assertFalse(controls.isHidden())

    def test_psth_palette_and_limits_do_not_bleed_into_spatial_maps(self):
        panel = self.panel
        data = np.array([[20., 25.], [30., 40.]])
        def spatial():
            panel._render_spatial_map(panel.plot_spatial_activity, panel.img_spatial_activity,
                                      data, (0, 1, 0, 1), 'X', 'Y', 'Activity')
        spatial()
        spatial_levels = panel.img_spatial_activity.getLevels().copy()
        spatial_lut = panel.spatial_lut_activity.item.gradient.getLookupTable(256).copy()
        self.render([[-2, 0, 2]])
        panel._set_heatmap_display_levels(-1, 1)
        panel._set_heatmap_palette('CET-D1')
        spatial()
        np.testing.assert_array_equal(panel.img_spatial_activity.getLevels(), spatial_levels)
        np.testing.assert_array_equal(panel.spatial_lut_activity.item.gradient.getLookupTable(256), spatial_lut)
        np.testing.assert_array_equal(panel.img_spatial_activity.image, data)
        self.assertEqual(panel._style['heatmap_cmap'], 'viridis')
        self.assertEqual(panel._style['psth_heatmap_cmap'], 'CET-D1')

    def test_palette_limits_and_editor_survive_project_roundtrip(self):
        panel = self.panel
        self.render([[-2, 0, 2]])
        panel._set_heatmap_display_levels(-.5, 1.5)
        panel._set_heatmap_palette('CET-D1')
        panel.heatmap_color_controls.adjust.click()
        with tempfile.TemporaryDirectory() as folder:
            path = str(Path(folder) / 'colors.h5')
            panel._save_project_h5(path)
            settings = panel._load_project_h5(path)['settings']
        panel._style.pop('psth_heatmap_cmap')
        panel._clear_heatmap_manual_limits()
        panel._is_restoring_settings = True
        try:
            panel._apply_settings(settings)
        finally:
            panel._is_restoring_settings = False
        self.render([[-20, 0, 20]])
        self.assert_levels((-.5, 1.5))
        self.assertEqual(panel.heatmap_color_controls.palette.currentData(), 'CET-D1')
        self.assertTrue(panel.heatmap_color_controls.adjust.isChecked())
        panel._set_plot_preset('Paper')
        self.assertEqual(panel.heatmap_color_controls.palette.currentData(), 'CET-D1')
        self.assert_levels((-.5, 1.5))
        np.testing.assert_array_equal(panel.img.lut / 255.,
                                      matplotlib_color_map(panel._style)(np.linspace(0, 1, 256)))

    def test_compact_controls_fit_and_empty_plots_hide_editor(self):
        controls = self.panel.heatmap_color_controls
        self.assertTrue(controls.isHidden())
        self.render([[1, 2, 3]])
        controls.adjust.click()
        controls.setParent(None)
        try:
            controls.resize(330, 160)
            controls.show()
            self.app.processEvents()
            for widget in (controls.mode, controls.auto, controls.adjust, controls.palette,
                           controls.minimum, controls.maximum):
                right = widget.mapTo(controls, QtCore.QPoint(widget.width(), 0)).x()
                self.assertLessEqual(right, controls.width())
        finally:
            controls.setParent(self.panel)
        self.render([[np.nan, np.nan, np.nan]])
        self.assertTrue(controls.isHidden())

    def test_plot_export_hides_editor_and_restores_it_even_on_failure(self):
        from heatmap_color_controls import plot_export_view
        panel = self.panel
        self.render([[-2, 0, 2]])
        panel.heatmap_color_controls.adjust.click()
        panel._set_heatmap_display_levels(-1, 1)
        original = panel._last_mat.copy()
        def check_view():
            self.assertTrue(panel.heatmap_color_controls.isHidden())
            self.assertTrue(panel.heat_lut.isHidden())
            self.assertFalse(panel.heat_colorbar_widget.isHidden())
            self.assert_levels((-1, 1))
        with tempfile.TemporaryDirectory() as folder:
            with patch.object(panel, '_write_widget_pdf', side_effect=lambda *_: check_view()):
                panel._export_widget_selective(panel.psth_shared_card, str(Path(folder) / 'plot'), False, True)
        self.assertFalse(panel.heatmap_color_controls.isHidden())
        with self.assertRaisesRegex(RuntimeError, 'export failure'):
            with plot_export_view(panel, panel.psth_shared_card):
                check_view()
                raise RuntimeError('export failure')
        self.assertFalse(panel.heatmap_color_controls.isHidden())
        self.assertFalse(panel.heat_lut.isHidden())
        self.assertTrue(panel.heat_colorbar_widget.isHidden())
        np.testing.assert_array_equal(panel._last_mat, original)


class ColorContractTests(unittest.TestCase):
    def test_auto_ranges_ignore_missing_values_and_reject_invalid_manual_limits(self):
        matrix = np.array([np.nan, -2., 0., 2., 100., np.inf])
        original = matrix.copy()
        self.assertEqual(color_limits(matrix), (-2., 100.))
        np.testing.assert_allclose(color_limits(matrix, 1), np.percentile(matrix[np.isfinite(matrix)], [2, 98]))
        self.assertEqual(color_limits(matrix, 2), (-100., 100.))
        self.assertEqual(color_limits(matrix, 0, dict(heatmap_levels_manual=True,
                         heatmap_min=10, heatmap_max=2)), (-2., 100.))
        self.assertEqual(color_limits([np.nan]), (0., 1.))
        self.assertEqual(color_limits([0., 0.]), (-.5, .5))
        np.testing.assert_array_equal(matrix, original)

    def test_screen_and_publication_share_identical_palette_samples(self):
        for name in ('viridis', 'CET-D1', 'gray'):
            style = {'psth_heatmap_cmap': name}
            qt = color_map(style).getLookupTable(nPts=256, alpha=True) / 255.
            mpl = matplotlib_color_map(style)(np.linspace(0, 1, 256))
            np.testing.assert_array_equal(qt, mpl)


if __name__ == '__main__':
    unittest.main()
