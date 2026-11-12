"""GUI integration: recording groups, group PSTH view and editable dashboard panels."""
import csv
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
from PySide6 import QtCore, QtWidgets

import test_postprocessing_empty_state as fixture
from postprocessing_core import mean_sem

N_PER_GROUP = 3
EVENTS = 6


class GroupViewDashboardTests(unittest.TestCase):
    """Isolated settings; synthetic recordings with known opposite responses."""

    @classmethod
    def setUpClass(cls):
        cls.app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

    setUp = fixture.PostprocessingEmptyStateTests.setUp
    tearDown = fixture.PostprocessingEmptyStateTests.tearDown

    def load(self):
        panel = self.panel
        rng = np.random.default_rng(3)
        t = np.arange(0, 120, 0.05)
        records, events = [], {}
        for prefix, amplitude in (("wt", 1.0), ("het", -1.0)):
            for index in range(N_PER_GROUP):
                ev = np.linspace(15, 100, EVENTS) + index
                y = rng.normal(0, 0.05, t.size)
                for e in ev:
                    window = (t >= e) & (t < e + 2)
                    y[window] += amplitude
                name = f"{prefix}{index}"
                records.append(fixture.ProcessedTrial(path=f"{name}.csv", channel_id="AIN01", time=t,
                                                      raw_signal=y, raw_reference=np.zeros_like(y),
                                                      output=y, output_label="dFF"))
                events[name] = ev
        panel._get_events_for_proc = lambda proc: (events[Path(proc.path).stem], np.full(EVENTS, 1.0))
        panel.tab_sources.setCurrentIndex(1)
        panel._processed = records
        panel._update_file_lists()
        panel.combo_psth_normalization.setCurrentIndex(panel.combo_psth_normalization.findData("subtract"))
        panel.spin_pre.setValue(3.0)
        panel.spin_post.setValue(4.0)
        panel.spin_b0.setValue(-3.0)
        panel.spin_b1.setValue(-1.0)
        panel.resize(1500, 950)
        panel.show()
        self.app.processEvents()
        panel._compute_psth_impl()
        panel.tab_visual_mode.setCurrentIndex(1)
        groups = panel.recording_groups
        self.wt, self.het = groups.add_group("WT"), groups.add_group("Het")
        for row in range(panel.list_preprocessed.count()):
            item = panel.list_preprocessed.item(row)
            item.setSelected(item.text().split(". ", 1)[1].startswith("wt"))
        panel._assign_selected_recordings_to_group(self.wt)
        groups.assign([f"het{i}" for i in range(N_PER_GROUP)], self.het)
        self.app.processEvents()
        return records

    def test_group_view_modes_split_heatmap_and_psth(self):
        self.load()
        panel = self.panel
        pooled_rows = panel._last_mat.shape[0]
        panel.cb_group_view.setChecked(True)
        self.app.processEvents()
        display = panel._group_display
        self.assertEqual(display.mode, "all")
        self.assertEqual([b.name for b in display.blocks], ["WT", "Het"])
        labels = [label.text for _sample, label in panel.psth_group_legend.items]
        self.assertEqual(labels, [f"WT (n={N_PER_GROUP})", f"Het (n={N_PER_GROUP})"])
        self.assertEqual(len(panel.heat_group_bands._rects), 2)
        self.assertFalse(panel.sem_band.isVisible())
        # The WT overlay equals mean_sem of the WT animal rows of the cache.
        wt_rows = [i for i, label in enumerate(panel._group_labels) if label.startswith("wt")]
        overlay = panel._group_avg_items[3].getData()[1]
        np.testing.assert_array_equal(overlay, mean_sem(panel._group_mat[wt_rows])[0])
        self.assertGreater(np.nanmax(overlay), 0.5)
        # Metric points follow group colors.
        self.assertEqual(panel.plot_metrics._row_colors, display.row_colors)
        # Only one group: every downstream summary uses that subset.
        panel.combo_group_display.setCurrentIndex(panel.combo_group_display.findData(f"only:{self.het}"))
        self.app.processEvents()
        self.assertEqual(panel._last_mat.shape[0], N_PER_GROUP)
        self.assertTrue(all(label.startswith("het") for label in panel._last_display_labels))
        self.assertEqual(panel._last_global_results["file_ids"], [f"het{i}" for i in range(N_PER_GROUP)])
        self.assertLess(np.nanmin(panel.curve_avg.getData()[1]), -0.5)
        panel.combo_group_display.setCurrentIndex(panel.combo_group_display.findData("pooled"))
        self.app.processEvents()
        self.assertIsNone(panel._group_display)
        self.assertEqual(panel._last_mat.shape[0], pooled_rows)
        self.assertTrue(panel.sem_band.isVisible())

    def test_groups_view_and_panels_persist_in_settings_and_projects(self):
        self.load()
        panel = self.panel
        panel.cb_group_view.setChecked(True)
        panel.combo_group_display.setCurrentIndex(panel.combo_group_display.findData(f"only:{self.wt}"))
        panel.dashboard_panels.remove("bout2")
        panel.dashboard_panels.set_spec("global", "behavior:duration")
        panel.psth_time_handle.setSizes([1400, 600])
        panel._save_settings()
        stored = json.loads(panel._settings.value("postprocess_json"))
        self.assertEqual(stored["group_view"], {"enabled": True, "display": f"only:{self.wt}"})
        self.assertEqual(stored["dashboard_panels"]["removed"], ["bout2"])
        self.assertEqual(stored["dashboard_panels"]["time_ratio"], [1400, 600])
        panel.recording_groups.clear()
        panel.dashboard_panels.reset()
        panel.cb_group_view.setChecked(False)
        panel._restore_settings()
        self.assertEqual([g["name"] for g in panel.recording_groups.groups()], ["WT", "Het"])
        self.assertTrue(panel.cb_group_view.isChecked())
        self.assertEqual(panel.combo_group_display.currentData(), f"only:{self.wt}")
        self.assertEqual(panel.dashboard_panels.substitutes, {"global": "behavior:duration"})
        self.assertEqual(panel.psth_time_handle.sizes(), [1400, 600])
        with tempfile.TemporaryDirectory() as folder:
            path = str(Path(folder) / "project.h5")
            panel._save_project_h5(path)
            saved = panel._load_project_h5(path)["settings"]
        self.assertEqual(saved["recording_groups"]["assignments"]["het0"], self.het)

    def test_remove_change_and_restore_panels(self):
        self.load()
        panel = self.panel
        panels = panel.dashboard_panels
        width_before = panel.plot_dur.width()
        panels.remove("bout2")
        self.app.processEvents()
        self.assertTrue(panel.plot_bout_second.isHidden())
        self.assertGreater(panel.plot_dur.width(), width_before * 1.5)   # neighbor takes the space
        panels.remove("bout1")
        self.app.processEvents()
        self.assertTrue(panel.bout_shared_card.isHidden())
        panels.restore_all()
        self.app.processEvents()
        self.assertFalse(panel.bout_shared_card.isHidden())
        # Same family: the slot's own selector changes, no substitute.
        panels.set_spec("bout1", "behavior:ibi")
        self.assertEqual(panel.combo_psth_behavior_metric.currentData(), "ibi")
        self.assertNotIn("bout1", panels.substitutes)
        # Another family: a substitute takes exactly the same slot.
        index = panel._view_splitters["bouts"].indexOf(panel.plot_bout_second)
        panels.set_spec("bout2", "metric:peak")
        substitute = panels.displayed("bout2")
        self.assertIsNot(substitute, panel.plot_bout_second)
        self.assertEqual(panel._view_splitters["bouts"].indexOf(substitute), index)
        self.assertIn("peak", panel._last_metric_panels)
        self.assertTrue(substitute.property("hasPlotData"))
        panels.set_spec("heatmap", "global:median")
        self.app.processEvents()
        self.assertTrue(panel.psth_scale_container.isHidden())
        panels.restore_original("heatmap")
        self.app.processEvents()
        self.assertIs(panels.displayed("heatmap"), panel.plot_heat)
        self.assertFalse(panel.psth_scale_container.isHidden())
        # Right-click menu gains panel actions on top of pyqtgraph's own.
        menu = panel.plot_avg.getPlotItem().getViewBox().menu
        menu.aboutToShow.emit()
        names = [action.text() for action in menu.actions()]
        self.assertEqual(names[:5], ["Panel: Average PSTH", "Remove this panel", "Change to",
                                     "Restore removed panels", "Reset dashboard layout"])
        panels.reset()
        self.assertEqual(panels.substitutes, {})
        self.assertIs(panels.displayed("bout2"), panel.plot_bout_second)
        # A layout preset still hides panels after a reset.
        panel.combo_view_layout.setCurrentText("Heatmap focus")
        self.assertTrue(panel.bout_shared_card.isHidden())
        panel.combo_view_layout.setCurrentText("Standard")
        self.assertFalse(panel.bout_shared_card.isHidden())

    def test_detach_window_keeps_lists_working_and_exports_groups(self):
        self.load()
        panel = self.panel
        section = panel.recordings_section
        section.detach()
        self.app.processEvents()
        self.assertTrue(section.is_detached())
        self.assertIs(panel.list_preprocessed.window(), section.window_widget)
        panel.list_preprocessed.clearSelection()
        panel.list_preprocessed.item(0).setSelected(True)
        panel._assign_selected_recordings_to_group(None)
        self.assertIsNone(panel.recording_groups.group_id_of("wt0"))
        section.window_widget.close()
        self.app.processEvents()
        self.assertFalse(section.is_detached())
        self.assertIsNot(panel.list_preprocessed.window(), section.window_widget)
        panel.cb_group_view.setChecked(True)
        self.app.processEvents()
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "groups.csv"
            panel._export_group_avg_csv(str(path), panel._group_display, panel._last_tvec)
            with path.open(encoding="utf-8") as stream:
                rows = list(csv.reader(stream))
            groups_path = Path(folder) / "recording_groups.csv"
            panel._export_recording_groups_csv(str(groups_path))
            with groups_path.open(encoding="utf-8") as stream:
                table = list(csv.DictReader(stream))
        self.assertEqual(rows[0][:4], ["time", "WT_mean", "WT_sem", "WT_n"])
        self.assertEqual(len(rows) - 1, panel._last_tvec.size)
        self.assertIn("Ungrouped_mean", rows[0])   # wt0 was ungrouped above
        self.assertEqual({row["file_id"]: row["group"] for row in table}["het1"], "Het")

    def test_right_click_targets_row_without_toggling_selection(self):
        self.load()
        panel = self.panel
        lst = panel.list_preprocessed
        lst.clearSelection()
        lst.item(0).setSelected(True)
        rect = lst.visualItemRect(lst.item(2))
        from PySide6 import QtGui
        from PySide6.QtTest import QTest
        QTest.mouseClick(lst.viewport(), QtCore.Qt.MouseButton.RightButton, QtCore.Qt.KeyboardModifier.NoModifier,
                         rect.center())
        self.assertEqual([lst.row(item) for item in lst.selectedItems()], [2])
        QTest.mouseClick(lst.viewport(), QtCore.Qt.MouseButton.RightButton, QtCore.Qt.KeyboardModifier.NoModifier,
                         rect.center())
        self.assertEqual([lst.row(item) for item in lst.selectedItems()], [2])   # not toggled off


if __name__ == "__main__":
    unittest.main()
