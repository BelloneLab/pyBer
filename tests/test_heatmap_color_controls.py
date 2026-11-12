"""Heatmap color adjustments (color-bar right-click menu) change rendering, never the analysis."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets
import test_postprocessing_empty_state as fixture
from heatmap_display import color_limits, color_map, matplotlib_color_map
from heatmap_color_controls import HeatmapLimitsDialog, find_action


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

    def menu(self):
        return self.panel.heatmap_color_menu.build_menu()

    def trigger(self, *path):
        action = find_action(self.menu(), *path)
        self.assertIsNotNone(action, path)
        action.trigger()
        return action

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

    def test_exact_limits_dialog_validation_and_auto_do_not_change_analysis(self):
        panel = self.panel
        self.render([[1, 2, 3], [4, 5, 100]])
        original = panel._last_mat.copy()
        baseline = panel.spin_b0.value(), panel.spin_b1.value()
        with patch.object(panel, '_compute_psth') as compute:
            dialog = HeatmapLimitsDialog(panel)
            self.assertEqual((dialog.minimum.value(), dialog.maximum.value()), (1, 100))
            dialog.minimum.setValue(1.5)
            dialog.maximum.setValue(5)
            self.assertTrue(dialog.apply())
            self.assert_levels((1.5, 5))
            self.assertTrue(find_action(self.menu(), 'Fixed: 1.5 to 5 · ' + panel._psth_units()) is not None)
            invalid = HeatmapLimitsDialog(panel)
            invalid.minimum.setValue(7)
            self.assertFalse(invalid.apply())
            self.assertFalse(invalid.error.isHidden())
            np.testing.assert_allclose(panel.img.getLevels(), (1.5, 5))
            self.assertFalse(panel._set_heatmap_display_levels(float('nan'), 10))
            self.assertFalse(panel._set_heatmap_display_levels(10, 10))
            panel._psth_pending = False
            self.trigger('Auto scale')
            self.assert_levels((1, 100))
            self.assertTrue(self.menu().actions()[0].text().startswith('Auto:'))
            compute.assert_not_called()
        np.testing.assert_array_equal(panel._last_mat, original)
        self.assertEqual((panel.spin_b0.value(), panel.spin_b1.value()), baseline)

    def test_contrast_menu_works_in_both_modes_and_follows_view_menu(self):
        panel = self.panel
        self.render(np.arange(100).reshape(10, 10))
        for category in ('zone', 'behavior'):
            panel._set_event_category(category)
            self.assertFalse(panel.heat_colorbar_widget.isHidden())
            self.trigger('Contrast', panel.combo_heat_scale.itemText(1))
            self.assertEqual(panel.combo_heat_scale.currentIndex(), 1)
            self.assert_levels(np.percentile(np.arange(100), [2, 98]))
            panel.combo_heat_scale.setCurrentIndex(2)
            self.assertTrue(find_action(self.menu(), 'Contrast', panel.combo_heat_scale.itemText(2)).isChecked())
            self.assert_levels((-99, 99))
        panel.combo_view_layout.setCurrentText('Trace focus')
        self.assertTrue(panel.psth_scale_container.isHidden())
        panel.combo_view_layout.setCurrentText('Standard')
        self.assertFalse(panel.psth_scale_container.isHidden())

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
        self.trigger('Palette', 'Blue-white-red')
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
        self.trigger('Palette', 'Blue-white-red')
        self.trigger('Histogram editor')
        self.assertTrue(panel.btn_edit_scale.isChecked())
        self.assertFalse(panel.heat_lut.isHidden())
        with tempfile.TemporaryDirectory() as folder:
            path = str(Path(folder) / 'colors.h5')
            panel._save_project_h5(path)
            settings = panel._load_project_h5(path)['settings']
        panel._style.pop('psth_heatmap_cmap')
        panel.btn_edit_scale.setChecked(False)
        panel._clear_heatmap_manual_limits()
        panel._is_restoring_settings = True
        try:
            panel._apply_settings(settings)
        finally:
            panel._is_restoring_settings = False
        self.render([[-20, 0, 20]])
        self.assert_levels((-.5, 1.5))
        self.assertTrue(find_action(self.menu(), 'Palette', 'Blue-white-red').isChecked())
        self.assertTrue(find_action(self.menu(), 'Histogram editor').isChecked())
        panel._set_plot_preset('Paper')
        self.assertTrue(find_action(self.menu(), 'Palette', 'Blue-white-red').isChecked())
        self.assert_levels((-.5, 1.5))
        np.testing.assert_array_equal(panel.img.lut / 255.,
                                      matplotlib_color_map(panel._style)(np.linspace(0, 1, 256)))

    def test_no_editor_row_and_right_click_opens_menu_on_both_scales(self):
        panel = self.panel
        layout = panel.psth_shared_card.layout()
        children = [layout.itemAt(i).widget() for i in range(layout.count())]
        self.assertNotIn('HeatmapColorControls', [type(w).__name__ for w in children])
        self.assertIn(panel.heat_figure, children)
        self.assertTrue(all(not action.isEnabled() for action in self.menu().actions()
                            if action.text() in ('Contrast', 'Auto scale', 'Set exact limits...')))
        self.render([[1, 2, 3]])
        shown = []
        panel.heatmap_color_menu.exec_menu = lambda menu, pos: shown.append([a.text() for a in menu.actions()])
        for view in (panel.heat_colorbar_widget, panel.heat_lut):
            point = QtCore.QPoint(5, 5)
            event = QtGui.QContextMenuEvent(QtGui.QContextMenuEvent.Reason.Mouse, point,
                                            view.viewport().mapToGlobal(point))
            QtWidgets.QApplication.sendEvent(view.viewport(), event)
        self.assertEqual(len(shown), 2)
        self.assertIn('Contrast', shown[0])
        self.assertIn('Set exact limits...', shown[0])
        self.assertIn('Histogram editor', shown[0])
        self.assertIn('Auto: 1 to 3', panel.heat_colorbar_widget.toolTip())

    def test_plot_export_shows_color_bar_and_restores_editor_even_on_failure(self):
        from heatmap_color_controls import plot_export_view
        panel = self.panel
        self.render([[-2, 0, 2]])
        panel.btn_edit_scale.setChecked(True)
        panel._set_heatmap_display_levels(-1, 1)
        original = panel._last_mat.copy()
        def check_view():
            self.assertTrue(panel.heat_lut.isHidden())
            self.assertFalse(panel.heat_colorbar_widget.isHidden())
            self.assert_levels((-1, 1))
        with tempfile.TemporaryDirectory() as folder:
            with patch.object(panel, '_write_widget_pdf', side_effect=lambda *_: check_view()):
                panel._export_widget_selective(panel.psth_shared_card, str(Path(folder) / 'plot'), False, True)
        self.assertFalse(panel.heat_lut.isHidden())
        with self.assertRaisesRegex(RuntimeError, 'export failure'):
            with plot_export_view(panel, panel.psth_shared_card):
                check_view()
                raise RuntimeError('export failure')
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
