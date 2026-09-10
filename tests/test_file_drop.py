"""Exercise real Qt file-drop delivery without touching source recordings."""

from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import Mock, patch

import test_postprocessing_empty_state as fixture
from PySide6 import QtCore, QtGui, QtWidgets
from file_drop import (expand_paths, install_file_drop, local_paths,
                       create_drop_privilege_notice, ELEVATED_DROP_HELP)
from gui_preprocessing import FileQueuePanel


def send_drop(widget, urls, actions=None):
    """Deliver Explorer's event sequence to the actual receiving Qt widget."""
    if actions is None:
        actions = QtCore.Qt.DropAction.CopyAction | QtCore.Qt.DropAction.MoveAction
    mime = QtCore.QMimeData()
    mime.setUrls(urls)
    point = QtCore.QPoint(8, 8)
    buttons = QtCore.Qt.MouseButton.LeftButton
    modifiers = QtCore.Qt.KeyboardModifier.ShiftModifier
    enter = QtGui.QDragEnterEvent(point, actions, mime, buttons, modifiers)
    QtWidgets.QApplication.sendEvent(widget, enter)
    move = QtGui.QDragMoveEvent(point, actions, mime, buttons, modifiers)
    QtWidgets.QApplication.sendEvent(widget, move)
    drop = QtGui.QDropEvent(QtCore.QPointF(point), actions, mime, buttons, modifiers)
    QtWidgets.QApplication.sendEvent(widget, drop)
    return enter, move, drop


class FileDropTests(unittest.TestCase):
    """Direct widget targets remain valid when their dock becomes a window."""

    @classmethod
    def setUpClass(cls):
        cls.app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="pyber-drop-")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.source = self.root / "souris été 01.CSV"
        self.source.write_bytes(b"time,signal\n0,1\n1,2\n")

    def dispose(self, widget):
        """Delete owned native Qt objects before their temporary files vanish."""
        widget.close()
        widget.deleteLater()
        self.app.sendPostedEvents(None, QtCore.QEvent.Type.DeferredDelete)

    def test_preprocessing_viewport_and_buttons_copy_after_native_drop(self):
        panel = FileQueuePanel()
        self.addCleanup(self.dispose, panel)
        received = []
        panel.filesDropped.connect(received.append)
        original = self.source.read_bytes()
        for target in (panel.list_files.viewport(), panel.btn_open, panel.btn_folder):
            with self.subTest(target=target):
                received.clear()
                events = send_drop(target, [QtCore.QUrl.fromLocalFile(str(self.source))])
                self.assertTrue(all(event.isAccepted() for event in events))
                self.assertTrue(all(event.dropAction() == QtCore.Qt.DropAction.CopyAction for event in events))
                self.assertEqual(received, [], "Loading must wait until the native drop returns")
                self.app.processEvents()
                self.assertEqual(len(received), 1)
                self.assertEqual(Path(received[0][0]), self.source)
                self.assertEqual(self.source.read_bytes(), original)

    def test_elevated_mode_explains_drop_restriction_after_show(self):
        """The real Windows restriction is visible even when no drop arrives."""
        with patch("file_drop.is_process_elevated", return_value=True):
            panel = FileQueuePanel()
            notice = create_drop_privilege_notice(panel)
        self.addCleanup(self.dispose, panel)
        panel.setAttribute(QtCore.Qt.WidgetAttribute.WA_DontShowOnScreen)
        panel.btn_open.setToolTip("Choose a recording")
        panel.show()
        self.app.processEvents()
        for target in (panel.list_files.viewport(), panel.btn_open, panel.btn_folder):
            self.assertIn(ELEVATED_DROP_HELP, target.toolTip())
        self.assertTrue(panel.btn_open.toolTip().startswith("Choose a recording"))
        self.assertIn("administrator mode", notice.text())
        self.assertIn("Save your work", notice.toolTip())
        panel.hide()
        panel.show()
        self.app.processEvents()
        self.assertEqual(panel.btn_open.toolTip().count(ELEVATED_DROP_HELP), 1)

    def test_normal_mode_keeps_drop_targets_and_status_uncluttered(self):
        """Ordinary launches retain their original tooltips and no warning."""
        button = QtWidgets.QPushButton()
        self.addCleanup(self.dispose, button)
        button.setAttribute(QtCore.Qt.WidgetAttribute.WA_DontShowOnScreen)
        button.setToolTip("Choose a recording")
        received = []
        with patch("file_drop.is_process_elevated", return_value=False):
            install_file_drop(button, received.append, (".csv",))
            self.assertIsNone(create_drop_privilege_notice(button))
        button.show()
        self.app.processEvents()
        self.assertEqual(button.toolTip(), "Choose a recording")
        self.assertTrue(send_drop(button, [QtCore.QUrl.fromLocalFile(str(self.source))])[-1].isAccepted())
        self.app.processEvents()
        self.assertEqual(len(received), 1)

    def test_detached_preprocessing_dock_receives_viewport_drop(self):
        window = QtWidgets.QMainWindow()
        self.addCleanup(self.dispose, window)
        window.setAttribute(QtCore.Qt.WidgetAttribute.WA_DontShowOnScreen)
        dock = QtWidgets.QDockWidget("Data", window)
        panel = FileQueuePanel()
        dock.setWidget(panel)
        window.addDockWidget(QtCore.Qt.DockWidgetArea.LeftDockWidgetArea, dock)
        dock.setFloating(True)
        received = []
        panel.filesDropped.connect(received.append)
        self.assertTrue(dock.isFloating())
        events = send_drop(panel.list_files.viewport(), [QtCore.QUrl.fromLocalFile(str(self.source))])
        self.assertTrue(events[-1].isAccepted())
        self.app.processEvents()
        self.assertEqual(Path(received[0][0]), self.source)

    def test_remote_unsupported_missing_and_move_only_drops_are_rejected(self):
        button = QtWidgets.QPushButton()
        self.addCleanup(self.dispose, button)
        received = []
        install_file_drop(button, received.append, (".csv",))
        unsupported = self.root / "image.png"
        unsupported.write_bytes(b"not a recording")
        for url, actions in (
            (QtCore.QUrl("https://example.org/trace.csv"), QtCore.Qt.DropAction.CopyAction),
            (QtCore.QUrl.fromLocalFile(str(unsupported)), QtCore.Qt.DropAction.CopyAction),
            (QtCore.QUrl.fromLocalFile(str(self.root / "missing.csv")), QtCore.Qt.DropAction.CopyAction),
            (QtCore.QUrl.fromLocalFile(str(self.source)), QtCore.Qt.DropAction.MoveAction),
        ):
            with self.subTest(url=url.toString()):
                self.assertFalse(send_drop(button, [url], actions)[0].isAccepted())
        self.app.processEvents()
        self.assertEqual(received, [])

    def test_folder_expansion_retains_order_and_deduplicates_sources(self):
        nested = self.root / "nested"
        nested.mkdir()
        second = nested / "second.H5"
        second.write_bytes(b"fixture")
        (nested / "skip.png").write_bytes(b"fixture")
        paths = expand_paths([str(self.source), str(self.root), str(second)], (".csv", ".h5"))
        self.assertEqual([Path(path) for path in paths], [self.source, second])
        mime = QtCore.QMimeData()
        mime.setUrls([QtCore.QUrl.fromLocalFile(str(self.source))] * 2)
        self.assertEqual([Path(path) for path in local_paths(mime, (".csv",))], [self.source])

    def test_preprocessing_dispatch_uses_raw_loader_independent_of_active_tab(self):
        """A detached Data drawer must retain preprocessing semantics."""
        from main import MainWindow

        owner = Mock()
        owner._expand_dropped_url_paths.side_effect = lambda urls: MainWindow._expand_dropped_url_paths(owner, urls)
        MainWindow._on_preprocessing_files_dropped(owner, [str(self.source)])
        owner._add_files.assert_called_once()
        self.assertEqual([Path(path) for path in owner._add_files.call_args.args[0]], [self.source])
        owner._handle_drop.assert_not_called()
        owner._push_recent_preprocessing_files.assert_called_once()

    def test_ethovision_metadata_viewport_routes_to_workbook_loader(self):
        """The read-only text viewport must not swallow Explorer URLs."""
        from ethovision_process_gui import MainWindow

        window = MainWindow()
        self.addCleanup(self.dispose, window)
        source = self.root / "tracking été.XLSX"
        source.write_bytes(b"unchanged workbook fixture")
        with patch.object(window, "load_workbook") as loader:
            events = send_drop(window.meta_text.viewport(), [QtCore.QUrl.fromLocalFile(str(source))])
            self.assertTrue(all(event.isAccepted() for event in events))
            self.assertEqual(events[-1].dropAction(), QtCore.Qt.DropAction.CopyAction)
            loader.assert_not_called()
            self.app.processEvents()
            loader.assert_called_once_with(source)
        self.assertEqual(source.read_bytes(), b"unchanged workbook fixture")


class PostprocessingFileDropTests(unittest.TestCase):
    """Route processed and behavior drops through the same loaders as buttons."""

    @classmethod
    def setUpClass(cls):
        cls.app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

    def setUp(self):
        fixture.PostprocessingEmptyStateTests.setUp(self)
        temporary = self.resources.enter_context(tempfile.TemporaryDirectory(prefix="pyber-post-drop-"))
        self.root = Path(temporary)

    def tearDown(self):
        fixture.PostprocessingEmptyStateTests.tearDown(self)

    def test_processed_buttons_and_list_route_to_processed_loader(self):
        source = self.root / "processed trace.H5"
        source.write_bytes(b"source unchanged")
        with patch.object(self.panel, "_load_processed_paths") as loader:
            for target in (self.panel.list_preprocessed.viewport(), self.panel.btn_load_processed_single,
                           self.panel.btn_load_processed):
                loader.reset_mock()
                self.assertTrue(send_drop(target, [QtCore.QUrl.fromLocalFile(str(source))])[-1].isAccepted())
                loader.assert_not_called()
                self.app.processEvents()
                loader.assert_called_once()
                self.assertEqual([Path(path) for path in loader.call_args.args[0]], [source])
                self.assertEqual(loader.call_args.kwargs, {"replace": False})
        self.assertEqual(source.read_bytes(), b"source unchanged")

    def test_behavior_list_and_button_expand_folder_without_processed_routing(self):
        source = self.root / "behavior été.XLSX"
        source.write_bytes(b"source unchanged")
        with patch.object(self.panel, "_load_behavior_paths") as behavior, \
                patch.object(self.panel, "_load_processed_paths") as processed:
            for target in (self.panel.list_behaviors.viewport(), self.panel.btn_load_beh):
                behavior.reset_mock()
                events = send_drop(target, [QtCore.QUrl.fromLocalFile(str(self.root))])
                self.assertTrue(events[-1].isAccepted())
                self.app.processEvents()
                behavior.assert_called_once()
                self.assertEqual([Path(path) for path in behavior.call_args.args[0]], [source])
                self.assertEqual(behavior.call_args.kwargs, {"replace": False})
            processed.assert_not_called()
        self.assertEqual(source.read_bytes(), b"source unchanged")

    def test_global_postprocessing_folder_separates_behavior_and_processed_loads(self):
        """General drops dispatch XLSX separately, after the native copy ends."""
        from main import MainWindow

        behavior = self.root / "events.XLSX"
        processed = self.root / "processed.CSV"
        for source in (behavior, processed):
            source.write_bytes(b"unchanged source")
        owner = SimpleNamespace(post_tab=self.panel, tabs=SimpleNamespace(currentWidget=lambda: self.panel))
        mime = QtCore.QMimeData()
        mime.setUrls([QtCore.QUrl.fromLocalFile(str(self.root))])
        event = QtGui.QDropEvent(
            QtCore.QPointF(8, 8), QtCore.Qt.DropAction.CopyAction | QtCore.Qt.DropAction.MoveAction,
            mime, QtCore.Qt.MouseButton.LeftButton, QtCore.Qt.KeyboardModifier.ShiftModifier,
        )
        with patch.object(self.panel, "_on_behavior_files_dropped") as load_behavior, \
                patch.object(self.panel, "_on_preprocessed_files_dropped") as load_processed:
            MainWindow.dropEvent(owner, event)
            self.assertTrue(event.isAccepted())
            self.assertEqual(event.dropAction(), QtCore.Qt.DropAction.CopyAction)
            load_behavior.assert_not_called()
            load_processed.assert_not_called()
            self.app.processEvents()
            load_behavior.assert_called_once()
            load_processed.assert_called_once()
            self.assertEqual([Path(path) for path in load_behavior.call_args.args[0]], [behavior])
            self.assertEqual([Path(path) for path in load_processed.call_args.args[0]], [processed])
        self.assertTrue(all(source.read_bytes() == b"unchanged source" for source in (behavior, processed)))


if __name__ == "__main__":
    unittest.main()
