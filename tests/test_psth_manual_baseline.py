"""Manual entry of PSTH windows, the on-request baseline button and heatmap gap rendering."""
import unittest
from unittest.mock import patch

import numpy as np
from PySide6 import QtCore, QtGui, QtTest, QtWidgets

import missing_data
import test_postprocessing_empty_state as fixture
from numeric_controls import DurationSpinBox


class DurationEntryTests(unittest.TestCase):
    """Typing is never blocked and any finite duration is accepted."""

    @classmethod
    def setUpClass(cls):
        cls.app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

    def setUp(self):
        self.spin = DurationSpinBox(-1e9, 1e9, -1)
        self.spin.show()
        self.values = []
        self.spin.valueChanged.connect(self.values.append)

    def tearDown(self):
        self.spin.close()
        self.spin.deleteLater()

    def type(self, text):
        """Replace the whole entry like a user (select all, type, Enter)."""
        self.spin.setFocus()
        self.spin.lineEdit().selectAll()
        QtTest.QTest.keyClicks(self.spin, text)
        QtTest.QTest.keyClick(self.spin, QtCore.Qt.Key.Key_Return)
        self.app.processEvents()
        return self.spin.value()

    def test_any_sign_and_size_can_be_typed(self):
        self.assertEqual(self.type("-600"), -600)
        self.assertEqual(self.type("5"), 5)            # a baseline after the event is allowed
        self.assertEqual(self.type("2700"), 2700)
        self.assertEqual(self.spin.text(), "2700 s")   # no thousands separator to trip over
        self.assertEqual(self.type("-123456.5"), -123456.5)
        self.assertEqual(self.values[-1], -123456.5)

    def test_editing_inside_a_large_value_is_not_blocked(self):
        self.type("2700")
        editor = self.spin.lineEdit()
        editor.setCursorPosition(1)                    # 2|700 -> 25700
        QtTest.QTest.keyClicks(self.spin, "5")
        self.assertEqual(editor.text(), "25700 s")
        QtTest.QTest.keyClick(self.spin, QtCore.Qt.Key.Key_Backspace)
        QtTest.QTest.keyClick(self.spin, QtCore.Qt.Key.Key_Return)
        self.assertEqual(self.spin.value(), 2700)

    def test_separators_and_decimal_marks_are_understood(self):
        state = QtGui.QValidator.State
        for text, seconds in (("2'700 s", 2700), ("2 700 s", 2700), ("1 234,5 s", 1234.5),
                              ("-0.25 s", -0.25), ("−3 s", -3)):
            self.assertEqual(self.spin.validate(text, 0)[0], state.Acceptable, text)
            self.assertEqual(self.spin.valueFromText(text), seconds, text)
        # Partial or unusable entries never swallow keystrokes ...
        for text in ("", "-", "abc s", "1e", "--2"):
            self.assertEqual(self.spin.validate(text, 0)[0], state.Intermediate, text)
        # ... and restore the previous value when committed.
        self.type("42")
        self.assertEqual(self.type("abc"), 42)

    def test_units_keep_seconds(self):
        self.spin.set_unit("min")
        self.assertEqual(self.type("-10"), -600)
        self.assertEqual(self.spin.text(), "-10 min")
        self.spin.set_unit("s")
        self.assertEqual(self.spin.text(), "-600 s")


class ManualBaselineTests(unittest.TestCase):
    """The PSTH panel never edits or evaluates the baseline on its own."""

    @classmethod
    def setUpClass(cls):
        cls.app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    setUp = fixture.PostprocessingEmptyStateTests.setUp
    tearDown = fixture.PostprocessingEmptyStateTests.tearDown

    def test_windows_accept_every_value(self):
        panel = self.panel
        for spin in (panel.spin_b0, panel.spin_b1, panel.spin_metric_pre0, panel.spin_metric_pre1,
                     panel.spin_metric_post0, panel.spin_metric_post1):
            for value in (-5000.0, 0.0, 1234.5):
                spin.setValue(value)
                self.assertEqual(spin.value(), value)
        panel.spin_b0.setValue(-9)
        panel.spin_b1.setValue(0)
        settings = panel._collect_settings()
        self.assertEqual((settings["baseline_start"], settings["baseline_end"]), (-9, 0))

    def test_auto_button_sits_in_the_baseline_row_and_only_runs_on_click(self):
        panel = self.panel
        widget = panel.baseline_suggestions
        self.assertIs(widget.parentWidget(), panel.spin_b0.parentWidget())      # same row as the fields
        self.assertEqual(widget.menu_button.text(), "Auto")
        self.assertLess(widget.sizeHint().width(), 120)                         # compact
        with patch("baseline_suggestions_widget.suggest_baselines") as estimate, \
                patch.object(widget, "queue", wraps=widget.queue) as queued:
            before = (panel.spin_b0.value(), panel.spin_b1.value())
            panel._is_restoring_settings = True       # schedule without starting a PSTH computation
            for _ in range(3):
                panel._schedule_psth()
                panel._update_trace_preview()
            panel._is_restoring_settings = False
            QtTest.QTest.qWait(40)
            queued.assert_not_called()
            estimate.assert_not_called()
            self.assertEqual((panel.spin_b0.value(), panel.spin_b1.value()), before)
            self.assertTrue(widget.menu_button.isEnabled())
            widget.menu_button.click()                # explicit request
            self.assertEqual(queued.call_count, 1)
        widget.invalidate()

    def test_requested_evaluation_survives_a_recompute(self):
        """A PSTH redraw during the evaluation re-evaluates instead of cancelling the click."""
        widget = self.panel.baseline_suggestions
        with patch.object(widget, "queue") as queued:
            widget.request_menu()
            self.assertEqual(queued.call_count, 1)
            self.assertFalse(widget.menu_button.isEnabled())
            widget.invalidate()                       # inputs changed while waiting
            self.assertEqual(queued.call_count, 2)
            self.assertTrue(widget._popup_when_ready)
        widget._popup_when_ready = False
        widget.invalidate()
        self.assertTrue(widget.menu_button.isEnabled())


class HeatmapGapRenderingTests(unittest.TestCase):
    """No gray block: the plot background shows wherever the heatmap has nothing to draw."""

    @classmethod
    def setUpClass(cls):
        cls.app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    setUp = fixture.PostprocessingEmptyStateTests.setUp
    tearDown = fixture.PostprocessingEmptyStateTests.tearDown

    def render(self, mode):
        panel = self.panel
        panel._settings.setValue(missing_data.SETTINGS_KEY_MODE, mode)
        matrix = np.arange(40, dtype=float).reshape(4, 10)
        matrix[1, 7:] = np.nan          # one row ends early
        matrix[2, 3:5] = np.nan         # interior gap
        tvec = np.linspace(-2, 7, 10)
        original = matrix.copy()
        panel._render_heatmap(matrix, tvec)
        np.testing.assert_array_equal(matrix, original)      # display only
        return panel

    def background_alpha(self, panel):
        """Opacity of the heatmap view background (0 = the plot background shows through)."""
        view = panel.plot_heat.getViewBox()
        background = view.background
        if not background.isVisibleTo(view) or background.brush().style() == QtCore.Qt.BrushStyle.NoBrush:
            return 0
        return background.brush().color().alpha()

    def test_default_mode_has_no_gray_anywhere(self):
        panel = self.render(missing_data.MODE_INTERPOLATE)
        self.assertEqual(self.background_alpha(panel), 0)
        self.assertFalse(panel._heat_gap_backing.isVisible())
        # Rows are trimmed to the span they share: the image stops where the short row stops.
        self.assertTrue(panel._heatmap_missing_report["trimmed"])
        self.assertLess(panel._heat_gap_backing.rect().right(), 7)

    def test_gray_mode_marks_gaps_inside_the_image_only(self):
        panel = self.render(missing_data.MODE_GRAY)
        self.assertEqual(self.background_alpha(panel), 0)
        backing = panel._heat_gap_backing
        self.assertTrue(backing.isVisible())
        image = panel.img.mapRectToParent(panel.img.boundingRect())
        for side in ("left", "right", "top", "bottom"):
            self.assertAlmostEqual(getattr(backing.rect(), side)(), getattr(image, side)(), places=6)
        self.assertLess(backing.zValue(), panel.img.zValue())
        panel = self.render(missing_data.MODE_BLANK)
        self.assertFalse(panel._heat_gap_backing.isVisible())
        panel._render_heatmap(np.empty((0, 0)), np.empty(0))
        self.assertFalse(panel._heat_gap_backing.isVisible())


if __name__ == "__main__":
    unittest.main()
