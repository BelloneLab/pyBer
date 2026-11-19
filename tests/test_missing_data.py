"""Missing samples in heatmaps: common-span trim, gap handling, preference, group toggle."""
import unittest

import numpy as np

import test_postprocessing_empty_state as gui_fixture
import missing_data
from missing_data import common_span, interpolate_interior_gaps, prepare_heatmap
from onboarding import PreferencesDialog


def _matrix():
    """Three rows of unequal length with an interior gap (values = column index)."""
    mat = np.tile(np.arange(20, dtype=float), (3, 1))
    mat[0, :2] = np.nan            # starts late
    mat[1, 17:] = np.nan           # ends early
    mat[2, 8:11] = np.nan          # cut artifact
    return mat, np.linspace(-2.0, 7.5, 20)


class MissingDataCoreTests(unittest.TestCase):
    def test_common_span_is_latest_start_to_earliest_end(self):
        mat, _ = _matrix()
        start, stop, kept = common_span(mat)
        self.assertEqual((start, stop), (2, 17))
        self.assertAlmostEqual(kept, 15 / 20)

    def test_interior_gaps_are_interpolated_and_edges_are_not(self):
        mat, _ = _matrix()
        filled, count = interpolate_interior_gaps(mat)
        self.assertEqual(count, 3)
        np.testing.assert_allclose(filled[2], np.arange(20.0))   # linear ramp restored exactly
        self.assertTrue(np.isnan(filled[0, :2]).all())           # no extrapolation
        self.assertTrue(np.isnan(filled[1, 17:]).all())
        self.assertTrue(np.isnan(mat[2, 8:11]).all())            # input untouched

    def test_default_display_has_equal_rows_and_no_missing_sample(self):
        mat, tvec = _matrix()
        shown, t_shown, report = prepare_heatmap(mat, tvec)
        self.assertEqual(shown.shape, (3, 15))
        self.assertTrue(np.isfinite(shown).all())
        np.testing.assert_allclose(t_shown, tvec[2:17])
        self.assertTrue(report["trimmed"])
        self.assertEqual((report["filled"], report["missing"]), (3, 0))
        self.assertIn("Display only", missing_data.describe(report))

    def test_modes_gray_and_blank_keep_gaps(self):
        mat, tvec = _matrix()
        shown, _t, report = prepare_heatmap(mat, tvec, missing_data.MODE_GRAY, trim=False)
        self.assertEqual(shown.shape, mat.shape)
        np.testing.assert_array_equal(np.isnan(shown), np.isnan(mat))
        self.assertEqual(report["missing"], 8)
        self.assertEqual(missing_data.describe({"missing": 0}), "")

    def test_one_very_short_row_does_not_hide_the_others(self):
        mat = np.tile(np.arange(100, dtype=float), (4, 1))
        mat[3, 10:] = np.nan                                   # 10 % of the others
        shown, _t, report = prepare_heatmap(mat, np.arange(100.0))
        self.assertEqual(shown.shape, mat.shape)
        self.assertFalse(report["trimmed"])
        self.assertTrue(report["trim_skipped"])
        self.assertIn("not trimmed", missing_data.describe(report))


class MissingDataGuiTests(unittest.TestCase):
    setUpClass = classmethod(gui_fixture.PostprocessingEmptyStateTests.setUpClass.__func__)
    setUp = gui_fixture.PostprocessingEmptyStateTests.setUp
    tearDown = gui_fixture.PostprocessingEmptyStateTests.tearDown

    def test_heatmap_draws_prepared_matrix_and_leaves_analysis_matrix_alone(self):
        panel = self.panel
        mat, tvec = _matrix()
        original = mat.copy()
        panel._render_heatmap(mat, tvec, labels=["a", "b", "c"])
        np.testing.assert_array_equal(mat, original)
        drawn = np.asarray(panel.img.image).T
        self.assertEqual(drawn.shape, (3, 15))
        self.assertTrue(np.isfinite(drawn).all())
        self.assertIn("Display only", panel.plot_heat.toolTip())

        panel._settings.setValue(missing_data.SETTINGS_KEY_MODE, missing_data.MODE_BLANK)
        panel._settings.setValue(missing_data.SETTINGS_KEY_TRIM, False)
        panel._render_heatmap(mat, tvec, labels=["a", "b", "c"])
        drawn = np.asarray(panel.img.image).T
        np.testing.assert_array_equal(np.isnan(drawn), np.isnan(original))

    def test_preferences_store_missing_data_choices(self):
        settings = self.panel._settings
        dialog = PreferencesDialog(self.panel, settings)
        self.assertEqual(dialog.combo_missing.currentData(), missing_data.DEFAULT_MODE)
        self.assertTrue(dialog.chk_missing_trim.isChecked())
        dialog.combo_missing.setCurrentIndex(dialog.combo_missing.findData(missing_data.MODE_GRAY))
        dialog.chk_missing_trim.setChecked(False)
        dialog._apply()
        self.assertEqual(settings.value(missing_data.SETTINGS_KEY_MODE), missing_data.MODE_GRAY)
        self.assertFalse(settings.value(missing_data.SETTINGS_KEY_TRIM, True, type=bool))
        dialog.deleteLater()

    def test_groups_menu_replaces_hide_drawer_in_group_scope(self):
        panel = self.panel
        panel.show()
        self.app.processEvents()
        self.assertTrue(panel.btn_action_hide.isHidden())
        panel.tab_visual_mode.setCurrentIndex(0)
        self.assertTrue(panel.btn_group_view.isHidden())
        panel.tab_visual_mode.setCurrentIndex(1)
        self.assertFalse(panel.btn_group_view.isHidden())
        self.assertEqual(panel.btn_group_view.text(), "Groups")


if __name__ == "__main__":
    unittest.main()
