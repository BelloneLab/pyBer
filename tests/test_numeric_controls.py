"""Regression tests for the application-wide numeric control event filter."""

import os
import sys
import unittest


os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "pyBer"))

from PySide6 import QtCore, QtWidgets  # noqa: E402
from shiboken6 import delete, isValid  # noqa: E402

from numeric_controls import SpinBoxScrubber  # noqa: E402


class SpinBoxScrubberLifecycleTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

    def setUp(self) -> None:
        self.scrubber = SpinBoxScrubber()

    def tearDown(self) -> None:
        self.scrubber.deleteLater()
        self.app.processEvents()

    def test_late_event_for_deleted_spinbox_editor_is_ignored(self) -> None:
        """A queued dialog teardown event must not dereference stale wrappers."""
        spin = QtWidgets.QDoubleSpinBox()
        editor = spin.lineEdit()
        self.assertTrue(isValid(editor))

        delete(spin)
        self.assertFalse(isValid(editor))

        event = QtCore.QEvent(QtCore.QEvent.Type.FocusOut)
        self.assertFalse(self.scrubber.eventFilter(editor, event))

    def test_deleted_drag_target_clears_scrubber_state_and_cursor(self) -> None:
        spin = QtWidgets.QSpinBox()
        self.scrubber._press_spin = spin
        self.scrubber._dragging = True
        self.scrubber._last_steps = 4
        self.scrubber._override_cursor = True
        QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.CursorShape.SizeHorCursor)

        delete(spin)
        live_widget = QtWidgets.QWidget()
        event = QtCore.QEvent(QtCore.QEvent.Type.FocusIn)
        self.assertFalse(self.scrubber.eventFilter(live_widget, event))

        self.assertIsNone(self.scrubber._press_spin)
        self.assertFalse(self.scrubber._dragging)
        self.assertEqual(self.scrubber._last_steps, 0)
        self.assertFalse(self.scrubber._override_cursor)
        self.assertIsNone(QtWidgets.QApplication.overrideCursor())
        live_widget.deleteLater()
