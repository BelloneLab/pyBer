"""Recording lists: + / x headers, per-row remove cross, reordering, menus, line pairing."""
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np
from PySide6 import QtCore, QtWidgets
from PySide6.QtTest import QTest

import test_postprocessing_empty_state as fixture


class RecordingListTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

    setUp = fixture.PostprocessingEmptyStateTests.setUp
    tearDown = fixture.PostprocessingEmptyStateTests.tearDown

    def load(self, behavior_keys=("rec0_beh", "rec1_beh", "rec2_beh")):
        panel = self.panel
        t = np.arange(0, 10, 0.1)
        panel._processed = [fixture.ProcessedTrial(path=f"rec{i}.csv", channel_id="AIN01", time=t,
                                                   raw_signal=np.zeros_like(t), raw_reference=np.zeros_like(t),
                                                   output=np.zeros_like(t), output_label="dFF")
                            for i in range(3)]
        panel._behavior_sources = {key: {"source_path": f"{key}.csv", "behaviors": {}} for key in behavior_keys}
        panel._update_file_lists()
        panel.resize(1000, 900)
        panel.recordings_section.detach()
        self.app.processEvents()
        return panel

    def processed_names(self):
        return [Path(proc.path).stem for proc in self.panel._processed]

    def test_headers_rows_and_line_pairing(self):
        panel = self.load()
        self.assertEqual(panel.lbl_processed_list.text(), "Processed recordings (3)")
        self.assertEqual(panel.lbl_behavior_list.text(), "Behaviors (3)")
        self.assertTrue(panel.btn_remove_pre.isHidden())
        delegate = panel._behavior_row_delegate
        index = panel.list_behaviors.model().index(0, 0)
        # Positional (line-wise) pairing: no note when row i pairs with row i.
        self.assertIsNone(panel._behavior_item_note(index))
        heights = {panel.list_preprocessed.visualItemRect(panel.list_preprocessed.item(i)).height() for i in range(3)}
        heights |= {panel.list_behaviors.visualItemRect(panel.list_behaviors.item(i)).height() for i in range(3)}
        self.assertEqual(len(heights), 1)   # identical rows keep lines aligned
        self.assertIsNotNone(delegate)

    def test_name_match_on_another_line_is_flagged(self):
        panel = self.load(("x_beh", "rec0", "y_beh"))
        model = panel.list_behaviors.model()
        # "rec0" is named after recording 1, so it pairs there, not on line 2,
        # and the line-wise fallback must not give it to rec1 as well.
        self.assertEqual(panel._behavior_item_note(model.index(1, 0))[0], "pairs with #1")
        self.assertIsNone(panel._match_behavior_source(panel._processed[1]))
        self.assertEqual(panel._recording_item_note(panel.list_preprocessed.model().index(1, 0))[0], "no behavior")
        self.assertEqual(panel._behavior_item_note(model.index(0, 0))[0], "unpaired")
        self.assertIsNone(panel._behavior_item_note(model.index(2, 0)))   # y_beh pairs line-wise with rec2

    def test_row_cross_removes_only_that_row(self):
        panel = self.load()
        lst = panel.list_preprocessed
        lst.item(0).setSelected(True)
        rect = lst.visualItemRect(lst.item(1))
        with patch.object(panel, "_compute_psth"):
            QTest.mouseClick(lst.viewport(), QtCore.Qt.MouseButton.LeftButton, QtCore.Qt.KeyboardModifier.NoModifier,
                             QtCore.QPoint(rect.right() - 10, rect.center().y()))
        self.assertEqual(self.processed_names(), ["rec0", "rec2"])
        bl = panel.list_behaviors
        rect = bl.visualItemRect(bl.item(2))
        QTest.mouseClick(bl.viewport(), QtCore.Qt.MouseButton.LeftButton, QtCore.Qt.KeyboardModifier.NoModifier,
                         QtCore.QPoint(rect.right() - 10, rect.center().y()))
        self.assertEqual(list(panel._behavior_sources), ["rec0_beh", "rec1_beh"])

    def test_plus_button_appends_several_files(self):
        panel = self.load()
        with patch.object(QtWidgets.QFileDialog, "getOpenFileNames", return_value=(["a.csv", "b.h5"], "")), \
                patch.object(panel, "_load_processed_paths") as load:
            panel.btn_add_processed.click()
        load.assert_called_once_with(["a.csv", "b.h5"], replace=False)
        self.assertEqual(panel.tab_sources.currentIndex(), 1)
        with patch.object(panel, "_load_behavior_files") as behaviors:
            panel.btn_add_behaviors.clicked.disconnect()
            panel.btn_add_behaviors.clicked.connect(panel._load_behavior_files)
            panel.btn_add_behaviors.click()
        behaviors.assert_called_once()

    def test_header_cross_and_menu_reorder(self):
        panel = self.load()
        lst = panel.list_preprocessed
        lst.item(2).setSelected(True)
        panel._move_list_rows(lst, "top")
        self.assertEqual(self.processed_names(), ["rec2", "rec0", "rec1"])
        self.assertEqual([lst.row(item) for item in lst.selectedItems()], [0])
        panel._move_list_rows(lst, "down")
        self.assertEqual(self.processed_names(), ["rec0", "rec2", "rec1"])
        panel._move_list_rows(lst, "bottom")
        self.assertEqual(self.processed_names(), ["rec0", "rec1", "rec2"])
        menu = panel._build_recording_list_menu(lst)
        names = [action.text() for action in menu.actions()]
        for expected in ("Add processed files...", "Remove selected (1)", "Move up", "Move down", "Move to top",
                         "Move to bottom", "Auto match rows by name", "Open containing folder",
                         "Attribute to group", "Remove from group"):
            self.assertIn(expected, names)
        with patch.object(panel, "_compute_psth"):
            panel.btn_remove_processed_list.click()
        self.assertEqual(self.processed_names(), ["rec0", "rec1"])

    def test_lists_scroll_together(self):
        panel = self.load()
        t = np.arange(0, 1, 0.1)
        panel._processed = [fixture.ProcessedTrial(path=f"r{i:02d}.csv", channel_id="A", time=t, raw_signal=t,
                                                   raw_reference=t, output=t, output_label="dFF") for i in range(40)]
        panel._behavior_sources = {f"r{i:02d}_b": {"source_path": ""} for i in range(40)}
        panel._update_file_lists()
        self.app.processEvents()
        panel.list_preprocessed.verticalScrollBar().setValue(5)
        self.assertEqual(panel.list_behaviors.verticalScrollBar().value(), 5)


if __name__ == "__main__":
    unittest.main()
