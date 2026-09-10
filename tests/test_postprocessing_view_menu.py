"""Verify view menu routing, portable sizes and usable trace geometry."""
import json
import tempfile
import unittest
from pathlib import Path
from PySide6 import QtCore, QtWidgets
import numpy as np
import test_postprocessing_empty_state as fixture


class ViewMenuTests(unittest.TestCase):
    """Use real widgets with isolated preferences and synthetic signals."""
    setUpClass = classmethod(fixture.PostprocessingEmptyStateTests.setUpClass.__func__)
    setUp = fixture.PostprocessingEmptyStateTests.setUp
    tearDown = fixture.PostprocessingEmptyStateTests.tearDown

    def test_menu_replaces_strip_and_updates_existing_controls(self):
        panel = self.panel
        menu = panel.btn_view_menu.menu()
        layout_menu = next(action.menu() for action in menu.actions() if action.text() == "Layout")
        next(action for action in layout_menu.actions() if action.text() == "Trace focus").trigger()
        self.assertEqual(panel.combo_view_layout.currentText(), "Trace focus")
        panel._processed = [fixture.ProcessedTrial(path="synthetic.csv", channel_id="1", time=np.arange(5.),
                            raw_signal=np.ones(5), raw_reference=np.ones(5), output=np.ones(5), output_label="dFF")]
        panel._update_status_strip()
        self.assertTrue(panel._plot_view_controls.isHidden())
        self.assertTrue(panel._plot_file_context.isHidden())
        self.assertTrue(panel._plot_scope_controls.isHidden())

    def test_single_row_scope_binding_and_secondary_actions(self):
        """The compact controls preserve old signals, shortcuts and project state."""
        panel = self.panel
        panel.combo_toolbar_scope.setCurrentIndex(1)
        self.assertEqual(panel.tab_visual_mode.currentIndex(), 1)
        panel.tab_visual_mode.setCurrentIndex(0)
        self.assertEqual(panel.combo_toolbar_scope.currentIndex(), 0)
        self.assertIn("Plot style...", [action.text() for action in panel.btn_view_menu.menu().actions()])
        self.assertIn("Reset analysis...", [action.text() for action in panel.menu_action_load.actions()])
        toolbar = panel._post_transport_bar
        toolbar.setParent(None)
        try:
            toolbar.resize(1050, 44)
            toolbar.show()
            self.app.processEvents()
            controls = (panel.btn_action_load, panel.btn_action_compute, panel.btn_action_export,
                        panel.btn_view_menu, panel.combo_toolbar_scope, panel.combo_individual_file,
                        panel.btn_action_hide)
            centers = [widget.geometry().center().y() for widget in controls]
            self.assertLessEqual(max(centers) - min(centers), 1)
            self.assertEqual(toolbar.height(), 44)
            for widget in controls:
                self.assertLessEqual(widget.geometry().right(), toolbar.width())
        finally:
            toolbar.setParent(panel)

    def test_named_sizes_and_paper_theme_survive_preferences_and_project_settings(self):
        panel = self.panel
        sizes = {"rows": [300, 600, 220, 360], "columns": [700, 400],
                 "details": [350, 230], "bouts": [200, 300], "comparison": [280, 200]}
        panel.plot_splitter_preferences.restore(sizes)
        panel._set_plot_preset("Paper")
        panel.btn_edit_scale.setChecked(True)
        panel._save_settings()
        stored = json.loads(panel._settings.value("postprocess_json"))
        self.assertEqual(stored["plot_splitters"], sizes)
        panel.plot_splitter_preferences.reset()
        panel._restore_settings()
        self.assertEqual(panel.plot_splitter_preferences.snapshot(), sizes)
        self.assertEqual(panel._style["postprocessing_preset"], "Paper")
        self.assertGreater(np.mean(panel._style["plot_bg"][:3]), 180)
        self.assertTrue(panel.btn_edit_scale.isChecked())
        with tempfile.TemporaryDirectory() as directory:
            path = str(Path(directory) / "view.h5")
            panel._save_project_h5(path)
            saved = panel._load_project_h5(path)["settings"]
            self.assertEqual(saved["plot_splitters"], sizes)
            self.assertTrue(saved["heatmap_scale_editor"])

    def test_hidden_panels_do_not_erase_sizes_and_invalid_values_are_ignored(self):
        prefs = self.panel.plot_splitter_preferences
        prefs.restore({"columns": [700, 400]})
        prefs.restore({"columns": [0, 100], "rows": [1], "bouts": [float("nan"), 1]})
        self.assertEqual(prefs.snapshot()["columns"], [700, 400])
        prefs.reset()
        self.assertEqual(prefs.snapshot()["columns"], [1000, 1000])

    def test_dragged_splitter_sizes_are_saved_and_restored_on_show(self):
        """Exercise a real divider, rather than only serializing supplied numbers."""
        panel = self.panel
        panel.row_heat.setParent(None)
        try:
            panel.row_heat.resize(1400, 500)
            panel.row_heat.show()
            self.app.processEvents()
            splitter = panel._view_splitters["columns"]
            splitter.moveSplitter(820, 1)
            expected = splitter.sizes()
            self.assertGreater(expected[0], expected[1])
            saved = panel.plot_splitter_preferences.snapshot()
            self.assertEqual(saved["columns"], expected)
            panel.row_heat.hide()
            panel.plot_splitter_preferences.restore(saved)
            panel.row_heat.show()
            self.app.processEvents()
            actual = splitter.sizes()
            self.assertAlmostEqual(actual[0] / sum(actual), expected[0] / sum(expected), places=2)
        finally:
            panel.row_heat.setParent(panel)

    def test_trace_axis_remains_inside_resized_preview_with_overlay_caption(self):
        panel = self.panel
        card = panel.trace_card
        panel.plot_trace.plot(np.arange(100.), np.sin(np.arange(100.)))
        panel.plot_trace.plotItem.show()
        card.subtitle_label.setText("Noise band | Prominence guide | Height threshold")
        card.subtitle_label.show()
        card.setParent(None)
        try:
            card.resize(1000, card.minimumHeight())
            card.show()
            self.app.processEvents()
            self.app.processEvents()
            axis = panel.plot_trace.getAxis("bottom")
            self.assertLessEqual(axis.sceneBoundingRect().bottom(), panel.plot_trace.viewport().height() + 1)
            self.assertGreaterEqual(panel.plot_trace.height(), 190)
        finally:
            card.setParent(panel)


if __name__ == "__main__":
    unittest.main()
