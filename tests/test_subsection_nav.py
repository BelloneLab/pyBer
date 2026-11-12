"""Regression tests for the vertical subsection navigator used by the PSTH panel."""
import os
from pathlib import Path
import sys
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "pyBer"))

from PySide6 import QtWidgets
from subsection_nav import VerticalSubsections


def _card(title: str, rows: int, wide: bool = False) -> QtWidgets.QGroupBox:
    """Build a titled form card with ``rows`` line edits (optionally very wide)."""
    box = QtWidgets.QGroupBox(title)
    form = QtWidgets.QFormLayout(box)
    for i in range(rows):
        edit = QtWidgets.QLineEdit()
        if wide:
            edit.setMinimumWidth(900)
        form.addRow(f"row {i}", edit)
    return box


class VerticalSubsectionsTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

    def setUp(self):
        self.nav = VerticalSubsections()
        self.short = _card("Short", 2)
        self.tall = _card("Tall", 14)
        self.wide = _card("Wide", 1, wide=True)
        for title, page in (("Short", self.short), ("Tall", self.tall), ("Wide", self.wide)):
            self.nav.addSection(title, page)
        self.nav.resize(500, 600)
        self.nav.show()
        self.app.processEvents()

    def tearDown(self):
        self.nav.close()
        self.nav.deleteLater()

    def test_one_page_visible_and_titles_moved_to_rail(self):
        self.assertEqual(self.nav.count(), 3)
        self.assertEqual(self.nav.currentIndex(), 0)
        self.assertTrue(self.short.isVisible())
        self.assertFalse(self.tall.isVisible())
        self.assertEqual(self.short.title(), "")
        self.assertEqual(self.nav.sectionTitle(1), "Tall")
        self.assertEqual(self.nav.indexOfTitle("wide"), 2)

    def test_rail_click_switches_page_and_emits(self):
        seen = []
        self.nav.currentChanged.connect(seen.append)
        self.nav.button(1).click()
        self.app.processEvents()
        self.assertEqual(self.nav.currentIndex(), 1)
        self.assertTrue(self.tall.isVisible())
        self.assertTrue(self.nav.button(1).isChecked())
        self.assertFalse(self.nav.button(0).isChecked())
        self.assertEqual(seen, [1])

    def test_hidden_pages_do_not_inflate_size(self):
        # The visible short page must not inherit the hidden pages' height or width.
        self.assertLess(self.nav._stack.sizeHint().height(), self.tall.sizeHint().height())
        self.assertLess(self.nav.minimumSizeHint().width(), 900)
        self.nav.setCurrentIndex(1)
        self.app.processEvents()
        self.assertGreaterEqual(self.nav._stack.sizeHint().height(), self.tall.sizeHint().height())

    def test_reveal_widget_and_out_of_range(self):
        child = self.wide.findChild(QtWidgets.QLineEdit)
        self.assertTrue(self.nav.revealWidget(child))
        self.assertEqual(self.nav.currentIndex(), 2)
        self.nav.setCurrentIndex(99)
        self.assertEqual(self.nav.currentIndex(), 2)
        self.assertFalse(self.nav.revealWidget(QtWidgets.QLabel()))


if __name__ == "__main__":
    unittest.main()
