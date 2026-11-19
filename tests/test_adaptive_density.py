"""Tool pages get denser, then reflow, instead of being clipped by a narrow drawer."""
import os
import sys
import unittest
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "pyBer"))

from PySide6 import QtCore, QtWidgets  # noqa: E402

from adaptive_density import (  # noqa: E402
    DENSITY_LEVELS,
    KEEP_ROW_PROPERTY,
    STACK_FIRST_PROPERTY,
    AdaptiveDensity,
    install_adaptive_density,
)


class AdaptiveDensityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

    def setUp(self):
        self.page = QtWidgets.QWidget()
        self.root = QtWidgets.QVBoxLayout(self.page)
        # A form row: label + field + button, with an explicit field width.
        self.form_row = QtWidgets.QHBoxLayout()
        self.form_row.setSpacing(10)
        self.field = QtWidgets.QLineEdit()
        self.field.setMinimumWidth(160)
        self.form_row.addWidget(QtWidgets.QLabel("Alignment source"))
        self.form_row.addWidget(self.field)
        self.form_row.addWidget(QtWidgets.QPushButton("Load processed file"))
        self.root.addLayout(self.form_row)
        # Side-by-side lists that must stay aligned line by line.
        self.lists_row = QtWidgets.QHBoxLayout()
        self.lists_row.setProperty(KEEP_ROW_PROPERTY, True)
        self.captions, self.lists = [], []
        for title in ("Processed recordings (12)", "Behavior files (12)"):
            column = QtWidgets.QVBoxLayout()
            caption = QtWidgets.QLabel(title)
            listing = QtWidgets.QListWidget()
            listing.setMinimumWidth(150)
            column.addWidget(caption)
            column.addWidget(listing)
            self.lists_row.addLayout(column)
            self.captions.append(caption)
            self.lists.append(listing)
        self.root.addLayout(self.lists_row)

        self.scroll = QtWidgets.QScrollArea()
        self.scroll.setWidgetResizable(True)
        self.scroll.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.scroll.setWidget(self.page)
        self.density = install_adaptive_density(self.scroll)
        self.scroll.resize(900, 500)
        self.scroll.show()
        self._settle()
        self.natural = self.page.minimumSizeHint().width()

    def tearDown(self):
        self.scroll.close()
        self.scroll.deleteLater()
        self.app.processEvents()

    def _settle(self):
        for _ in range(40):
            self.app.processEvents()
            QtCore.QThread.msleep(10)

    def _resize(self, width):
        self.scroll.resize(width, 500)
        self._settle()

    def _clipped(self):
        return max(0, self.page.minimumSizeHint().width() - self.scroll.viewport().width())

    def test_wide_drawer_keeps_the_normal_presentation(self):
        self.assertEqual(self.density.level, 0)
        self.assertEqual(self.page.styleSheet(), "")
        self.assertEqual(self.field.minimumWidth(), 160)
        self.assertFalse(self.density._reflowed)
        self.assertFalse(self.density._squeezed)
        self.assertIs(install_adaptive_density(self.scroll), self.density)

    def test_narrowing_shrinks_content_until_it_fits(self):
        self.assertGreater(self.natural, 400)
        self._resize(int(self.natural * 0.8))
        self.assertEqual(self._clipped(), 0)
        # Mildest step first: the aligned lists got narrower.
        self.assertTrue(self.density._squeezed)
        self.assertLess(self.lists[0].minimumWidth(), 150)
        self.assertEqual(self.captions[0].toolTip(), "Processed recordings (12)")

        self._resize(260)
        self.assertEqual(self._clipped(), 0)
        self.assertGreater(self.density.level, 0)
        scale = DENSITY_LEVELS[self.density.level][0]
        self.assertEqual(self.field.minimumWidth(), round(160 * scale))
        self.assertIn("font-size", self.page.styleSheet())
        self.assertEqual(self.form_row.spacing(), max(1, round(10 * scale)))

    def test_rows_that_cannot_fit_stack_but_kept_rows_stay_horizontal(self):
        self._resize(150)
        self.assertEqual(self.density.level, len(DENSITY_LEVELS) - 1)
        self.assertEqual(self.form_row.direction(), QtWidgets.QBoxLayout.Direction.TopToBottom)
        self.assertEqual(self.lists_row.direction(), QtWidgets.QBoxLayout.Direction.LeftToRight)
        # Whatever still does not fit stays reachable by scrolling sideways.
        self.assertEqual(self.scroll.horizontalScrollBarPolicy(),
                         QtCore.Qt.ScrollBarPolicy.ScrollBarAsNeeded)

    def test_widening_restores_everything(self):
        self._resize(150)
        self._resize(self.natural + 200)
        self.assertEqual(self.density.level, 0)
        self.assertEqual(self.page.styleSheet(), "")
        self.assertEqual(self.form_row.direction(), QtWidgets.QBoxLayout.Direction.LeftToRight)
        self.assertEqual(self.form_row.spacing(), 10)
        self.assertEqual(self.field.minimumWidth(), 160)
        self.assertEqual(self.lists[0].minimumWidth(), 150)
        self.assertEqual(self.captions[0].toolTip(), "")
        self.assertEqual(self.page.minimumSizeHint().width(), self.natural)

    def test_stack_first_rows_stack_before_any_text_shrinks(self):
        self.lists_row.setProperty(KEEP_ROW_PROPERTY, False)
        self.lists_row.setProperty(STACK_FIRST_PROPERTY, True)
        self._resize(self.natural - 30)
        self.assertEqual(self.lists_row.direction(), QtWidgets.QBoxLayout.Direction.TopToBottom)
        self.assertTrue(self.density._stacked_first)
        self.assertFalse(self.density._squeezed)              # lists are not squeezed
        self._resize(self.natural + 200)
        self.assertEqual(self.lists_row.direction(), QtWidgets.QBoxLayout.Direction.LeftToRight)

    def test_page_own_style_rules_survive(self):
        self.page.setStyleSheet("QLabel { color: red; }")
        self._resize(260)
        self.assertIn("QLabel { color: red; }", self.page.styleSheet())
        self._resize(self.natural + 200)
        self.assertEqual(self.page.styleSheet(), "QLabel { color: red; }")


if __name__ == "__main__":
    unittest.main()
