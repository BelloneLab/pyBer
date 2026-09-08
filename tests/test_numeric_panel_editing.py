"""Verify exact entry and slider wiring in the actual analysis panels."""

import unittest

import test_postprocessing_empty_state as fixture
from PySide6 import QtCore, QtTest, QtWidgets

from gui_preprocessing import ParameterPanel
from numeric_controls import NumericSlider, SpinBoxScrubber


class NumericPanelEditingTests(unittest.TestCase):
    """Numeric interaction must preserve parameter precision and availability."""

    @classmethod
    def setUpClass(cls):
        """Share the existing offscreen Qt application."""
        cls.app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

    def setUp(self):
        """Reuse the fixture that redirects even named QSettings to a temp INI."""
        fixture.PostprocessingEmptyStateTests.setUp(self)
        self.preprocessing = ParameterPanel()
        self.scrubber = SpinBoxScrubber()
        self.scrubber.scan(self.preprocessing)
        self.scrubber.scan(self.panel)
        self.app.installEventFilter(self.scrubber)

    def tearDown(self):
        """Remove the global filter before destroying panels and temp settings."""
        self.app.removeEventFilter(self.scrubber)
        self.scrubber.deleteLater()
        self.preprocessing.close()
        self.preprocessing.deleteLater()
        fixture.PostprocessingEmptyStateTests.tearDown(self)

    def enter_value(self, spin, text, expected):
        """Send real keyboard events and commit once, as an exact-value edit."""
        self.assertTrue(spin.isEnabled())
        spin.selectAll()
        QtTest.QTest.keyClicks(spin, text.replace(".", spin.locale().decimalPoint()))
        QtTest.QTest.keyClick(spin, QtCore.Qt.Key.Key_Return)
        self.assertAlmostEqual(spin.value(), expected, places=12)

    def load_trace(self):
        """Use synthetic in-memory data solely to enable postprocessing fields."""
        clock = fixture.np.linspace(0, 10, 101)
        signal = fixture.np.sin(clock)
        self.panel.receive_current_processed([fixture.ProcessedTrial(
            path="numeric-fixture.csv", channel_id="AIN01", time=clock,
            raw_signal=signal, raw_reference=signal.copy(),
            output=signal.copy(), output_label="dFF",
        )])
        self.panel._psth_timer.stop()

    def test_preprocessing_exact_values_and_smallest_tolerances(self):
        """Decimal editors retain all accepted precision, including tiny values."""
        pre = self.preprocessing
        self.enter_value(pre.spin_mad, "7.125", 7.125)
        self.enter_value(pre.spin_filt_order, "5", 5)
        self.enter_value(pre.spin_tol, "0.00000001", 1e-8)
        self.enter_value(pre.spin_rlm_tol, "0.000000000001", 1e-12)
        self.assertEqual(pre.spin_tol.minimum(), 1e-8)
        self.assertEqual(pre.spin_rlm_tol.minimum(), 1e-12)

    def test_postprocessing_keyboard_values_and_auto_availability(self):
        """Manual prominence remains explicit while other enabled values type freely."""
        self.assertFalse(self.panel.spin_peak_mad_multiplier.isEnabled())
        self.load_trace()
        self.assertFalse(self.panel.spin_peak_prominence.isEnabled())
        self.enter_value(self.panel.spin_peak_mad_multiplier, "3.75", 3.75)
        self.enter_value(self.panel.spin_peak_distance, "0.123", 0.123)
        self.enter_value(self.panel.spin_b0, "-1.25", -1.25)
        self.panel.cb_peak_auto_mad.setChecked(False)
        self.enter_value(self.panel.spin_peak_prominence, "0.01234567", 0.01234567)

    def test_sliders_follow_enabled_state_and_emit_original_parameter_signals(self):
        """A slider edit reaches the same settings path as the existing spinbox."""
        pre = self.preprocessing
        control = pre.spin_mad.parentWidget()
        self.assertIsInstance(control, NumericSlider)
        changes = QtTest.QSignalSpy(pre.paramsChanged)
        control.slider.setValue(control.slider.maximum())
        self.assertEqual(pre.spin_mad.value(), pre.spin_mad.maximum())
        self.assertGreater(changes.count(), 0)
        pre.cb_artifact.setChecked(False)
        self.assertFalse(control.slider.isEnabled())
        pre.cb_artifact.setChecked(True)
        self.assertTrue(control.slider.isEnabled())

        self.load_trace()
        self.panel._signal_preview_requested = True
        control = self.panel.spin_peak_mad_multiplier.parentWidget()
        changes = QtTest.QSignalSpy(self.panel.spin_peak_mad_multiplier.valueChanged)
        control.slider.setValue(control.slider.maximum())
        self.assertEqual(changes.count(), 1)
        self.assertTrue(self.panel._signal_preview_timer.isActive())

    def test_all_numeric_editors_keep_native_text_entry(self):
        """The shared scan must cover regular and advanced fields in both panels."""
        for panel in (self.preprocessing, self.panel):
            fields = panel.findChildren(QtWidgets.QAbstractSpinBox)
            self.assertGreater(len(fields), 20)
            for field in fields:
                self.assertFalse(field.isReadOnly())
                self.assertFalse(field.lineEdit().isReadOnly())
                self.assertEqual(field.lineEdit().cursor().shape(), QtCore.Qt.CursorShape.IBeamCursor)
                self.assertFalse(field.keyboardTracking())


if __name__ == "__main__":
    unittest.main()
