"""Regression tests for the application-wide numeric control event filter."""

import os
import sys
import unittest


os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "pyBer"))

from PySide6 import QtCore, QtGui, QtTest, QtWidgets  # noqa: E402
from shiboken6 import delete, isValid  # noqa: E402

from numeric_controls import SpinBoxScrubber, with_slider  # noqa: E402


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

    def test_keyboard_commits_exact_value_once_and_retains_native_step(self):
        """Typing, selecting and pasting must keep ordinary editor semantics."""
        spin = QtWidgets.QDoubleSpinBox()
        spin.setRange(-100, 100)
        spin.setDecimals(6)
        spin.setSingleStep(0.25)
        self.scrubber.scan(spin)
        self.app.installEventFilter(self.scrubber)
        try:
            spin.show()
            spin.setFocus()
            changes = []
            spin.valueChanged.connect(changes.append)
            spin.selectAll()
            QtTest.QTest.keyClicks(spin, "-0.123456")
            self.assertEqual(changes, [])
            QtTest.QTest.keyClick(spin, QtCore.Qt.Key.Key_Return)
            self.assertEqual(changes, [-0.123456])
            self.assertEqual(spin.lineEdit().cursor().shape(), QtCore.Qt.CursorShape.IBeamCursor)
            self.assertEqual(spin.singleStep(), 0.25)
            spin.selectAll()
            self.app.clipboard().setText("4.567891")
            QtTest.QTest.keyClick(spin, QtCore.Qt.Key.Key_V, QtCore.Qt.KeyboardModifier.ControlModifier)
            QtTest.QTest.keyClick(spin, QtCore.Qt.Key.Key_Return)
            self.assertEqual(spin.value(), 4.567891)
        finally:
            self.app.removeEventFilter(self.scrubber)
            delete(spin)

    def test_plain_mouse_drag_does_not_start_value_scrubbing(self):
        spin = QtWidgets.QDoubleSpinBox()
        self.scrubber.scan(spin)
        event = QtGui.QMouseEvent(
            QtCore.QEvent.Type.MouseButtonPress, QtCore.QPointF(10, 10),
            QtCore.QPointF(10, 10), QtCore.Qt.MouseButton.LeftButton,
            QtCore.Qt.MouseButton.LeftButton, QtCore.Qt.KeyboardModifier.NoModifier,
        )
        self.assertFalse(self.scrubber.eventFilter(spin.lineEdit(), event))
        self.assertIsNone(self.scrubber._press_spin)
        delete(spin)

    def test_slider_preserves_typed_precision_and_full_range(self):
        """A coarse slider must never quantize a typed or restored parameter."""
        for logarithmic in (False, True):
            spin = QtWidgets.QDoubleSpinBox()
            spin.setDecimals(8)
            spin.setRange(0, 3600)
            control = with_slider(spin, logarithmic=logarithmic)
            spin.setValue(0.12345678)
            self.assertEqual(spin.value(), 0.12345678)
            control.slider.setValue(control.RESOLUTION)
            self.assertEqual(spin.value(), 3600)
            control.slider.setValue(0)
            self.assertEqual(spin.value(), 0)
            spin.setDisabled(True)
            self.assertFalse(control.slider.isEnabled())
            spin.setEnabled(True)
            self.assertTrue(control.slider.isEnabled())
            spin.setReadOnly(True)
            self.assertFalse(control.slider.isEnabled())
            delete(control)
