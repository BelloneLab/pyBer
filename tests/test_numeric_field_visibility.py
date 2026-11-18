"""Regression tests: numeric values stay visible, sliders pair with form fields,
and footer actions never force a drawer to scroll sideways.

Background: the v0.57 Windows build bundled Qt 6.11.3, which lays out the text
editor of a style-sheeted spin box without arrow buttons as a 1 px strip, so
every parameter value in the app looked empty.
"""
import os
from pathlib import Path
import sys
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "pyBer"))

from PySide6 import QtWidgets
from numeric_controls import (NumericSlider, ensure_editor_visible, wrap_form_spinbox,
                              add_sliders_to_forms, NO_SLIDER_PROP)
from onboarding import FooterActions
from ui_audit import audit_spinboxes
from styles import app_qss


class NumericFieldVisibilityTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
        cls._previous_qss = cls.app.styleSheet()
        cls.app.setStyleSheet(app_qss("dark"))

    @classmethod
    def tearDownClass(cls):
        # Other test modules share this QApplication and expect no app stylesheet.
        cls.app.setStyleSheet(cls._previous_qss)

    def _spin(self, value=3.5):
        spin = QtWidgets.QDoubleSpinBox()
        spin.setButtonSymbols(QtWidgets.QAbstractSpinBox.ButtonSymbols.NoButtons)
        spin.setValue(value)
        spin.resize(200, 34)
        spin.show()
        self.app.processEvents()
        return spin

    def test_collapsed_editor_is_restored(self):
        spin = self._spin()
        editor = spin.lineEdit()
        editor.setGeometry(0, 6, 1, 22)  # exactly what Qt 6.11.3 produces
        self.assertTrue(ensure_editor_visible(spin))
        self.assertGreater(editor.width(), 150)
        self.assertGreater(editor.width(), editor.fontMetrics().horizontalAdvance(spin.text()))
        self.assertTrue(editor.isVisible())
        spin.close()

    def test_healthy_editor_is_left_alone(self):
        spin = self._spin()
        ensure_editor_visible(spin)  # repairs a native collapse (Qt 6.11.3) if any
        before = spin.lineEdit().geometry()
        self.assertGreater(before.width(), 150)
        self.assertFalse(ensure_editor_visible(spin))
        self.assertEqual(spin.lineEdit().geometry(), before)
        spin.close()

    def test_form_spinbox_gets_slider_but_paired_rows_do_not(self):
        page = QtWidgets.QWidget()
        form = QtWidgets.QFormLayout(page)
        single = QtWidgets.QDoubleSpinBox(); single.setRange(0, 10); single.setValue(2)
        start, end = QtWidgets.QDoubleSpinBox(), QtWidgets.QDoubleSpinBox()
        pair = QtWidgets.QWidget(); row = QtWidgets.QHBoxLayout(pair)
        row.addWidget(start); row.addWidget(end)
        opted_out = QtWidgets.QSpinBox(); opted_out.setProperty(NO_SLIDER_PROP, True)
        unbounded = QtWidgets.QDoubleSpinBox(); unbounded.setRange(-1e6, 1e6)
        form.addRow("Single", single)
        form.addRow("Window", pair)
        form.addRow("Opted out", opted_out)
        form.addRow("Unbounded", unbounded)
        page.show(); self.app.processEvents()
        self.assertEqual(add_sliders_to_forms(page), 1)
        self.assertIsInstance(single.parentWidget(), NumericSlider)
        self.assertIs(start.parentWidget(), pair)
        self.assertNotIsInstance(opted_out.parentWidget(), NumericSlider)
        self.assertNotIsInstance(unbounded.parentWidget(), NumericSlider)
        # Already wrapped: nothing more to do, and the value path is unchanged.
        self.assertFalse(wrap_form_spinbox(single))
        single.parentWidget().slider.setValue(1000)
        self.assertAlmostEqual(single.value(), 10.0)
        page.close()

    def test_hidden_row_follows_its_spinbox(self):
        page = QtWidgets.QWidget()
        form = QtWidgets.QFormLayout(page)
        spin = QtWidgets.QDoubleSpinBox()
        form.addRow("Conditional", spin)
        page.show(); self.app.processEvents()
        self.assertTrue(wrap_form_spinbox(spin))
        wrapper = spin.parentWidget()
        spin.setVisible(False); self.app.processEvents()
        self.assertTrue(wrapper.isHidden())
        spin.setVisible(True); self.app.processEvents()
        self.assertFalse(wrapper.isHidden())
        self.assertTrue(wrapper.slider.isVisible())
        page.close()

    def test_footer_buttons_stack_instead_of_forcing_width(self):
        # Hosted like a drawer: a parent whose layout follows the footer's hints.
        host = QtWidgets.QWidget()
        column = QtWidgets.QVBoxLayout(host)
        footer = FooterActions()
        for text in ("Export events CSV", "Export metrics CSV", "Compute metrics"):
            footer.add_secondary(QtWidgets.QPushButton(text))
        column.addWidget(footer)
        host.show(); self.app.processEvents()
        widest = max(b.minimumSizeHint().width() for b in footer._buttons())
        self.assertLess(footer.minimumSizeHint().width(), widest + 40)
        host.resize(widest + 60, 200); self.app.processEvents()
        self.assertLessEqual(footer.width(), widest + 60)
        self.assertEqual(footer._right.direction(), QtWidgets.QBoxLayout.Direction.TopToBottom)
        host.resize(900, 200); self.app.processEvents()
        self.assertEqual(footer._right.direction(), QtWidgets.QBoxLayout.Direction.LeftToRight)
        host.close()

    def test_audit_flags_hidden_values_and_passes_healthy_fields(self):
        """The release gate must fail on the exact v0.57 symptom."""
        page = QtWidgets.QWidget()
        form = QtWidgets.QFormLayout(page)
        spin = QtWidgets.QDoubleSpinBox(); spin.setValue(8.742)
        form.addRow("MAD threshold", spin)
        page.resize(300, 80); page.show(); self.app.processEvents()
        ensure_editor_visible(spin)
        self.assertEqual(audit_spinboxes(page, "healthy"), [])
        spin.lineEdit().setGeometry(0, 6, 1, 22)  # Qt 6.11.3 collapse
        failures = audit_spinboxes(page, "collapsed")
        self.assertEqual(len(failures), 1)
        self.assertEqual(failures[0]["problem"], "collapsed editor")
        self.assertEqual(failures[0]["text"], spin.text())
        page.close()


if __name__ == "__main__":
    unittest.main()
