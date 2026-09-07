"""Guard vector legibility, white states and badge-free panel headings."""

import os
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "pyBer"))

from PySide6 import QtCore, QtGui, QtWidgets
from onboarding import PanelHeader, _POST_SECTION_META, _PRE_SECTION_META
from tool_icons import ICON_ART, _icon_painter, _make_icon, _renderer
from styles import apply_native_titlebar


class ToolIconTests(unittest.TestCase):
    """Inspect rendered pixels, including checked and high-DPI icon variants."""

    @classmethod
    def setUpClass(cls):
        cls.app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

    def test_all_artwork_is_valid_visible_white_and_unclipped(self):
        for name in ICON_ART:
            with self.subTest(tool=name):
                self.assertTrue(_renderer(name, "#ffffff").isValid())
                icon = _make_icon(_icon_painter(name))
                for edge in (16, 22, 44, 96):
                    image = icon.pixmap(edge).toImage()
                    pixels = [image.pixelColor(x, y) for y in range(edge) for x in range(edge)]
                    visible = [p for p in pixels if p.alpha() > 30]
                    self.assertGreater(len(visible), edge)
                    self.assertTrue(all(p.red() == p.green() == p.blue() == 255 for p in visible))
                    boundary = [image.pixelColor(x, y).alpha() for x in range(edge)
                                for y in (0, edge - 1)]
                    boundary += [image.pixelColor(x, y).alpha() for y in range(edge)
                                 for x in (0, edge - 1)]
                    self.assertLess(max(boundary), 180, "Drawing is clipped by its canvas")

    def test_selection_preserves_white_and_disabled_reduces_opacity(self):
        icon = _make_icon(_icon_painter("data"))
        normal = icon.pixmap(QtCore.QSize(24, 24), QtGui.QIcon.Mode.Normal).toImage()
        selected = icon.pixmap(QtCore.QSize(24, 24), QtGui.QIcon.Mode.Selected,
                               QtGui.QIcon.State.On).toImage()
        disabled = icon.pixmap(QtCore.QSize(24, 24), QtGui.QIcon.Mode.Disabled).toImage()
        self.assertEqual(normal, selected)
        alpha = lambda im: sum(im.pixelColor(x, y).alpha() for y in range(24) for x in range(24))
        self.assertLess(alpha(disabled), alpha(normal) * 0.4)
        self.assertGreater(alpha(disabled), 0)

    def test_light_theme_can_request_contrasting_ink(self):
        image = _make_icon(_icon_painter("setup"), color="#3b4763").pixmap(24).toImage()
        opaque = [image.pixelColor(x, y) for y in range(24) for x in range(24)
                  if image.pixelColor(x, y).alpha() == 255]
        self.assertTrue(opaque)
        self.assertTrue(all(p.name() == "#3b4763" for p in opaque))
        checked = _make_icon(_icon_painter("setup"), color="#3b4763").pixmap(
            QtCore.QSize(24, 24), QtGui.QIcon.Mode.Normal, QtGui.QIcon.State.On).toImage()
        self.assertTrue(all(checked.pixelColor(x, y).name() == "#ffffff"
                            for y in range(24) for x in range(24)
                            if checked.pixelColor(x, y).alpha() > 30))

    def test_every_panel_header_contains_only_title_and_description(self):
        header = PanelHeader()
        for meta in list(_PRE_SECTION_META.values()) + list(_POST_SECTION_META.values()):
            with self.subTest(title=meta[2]):
                header.set_from_meta(meta)
                self.assertEqual([label.text() for label in header.findChildren(QtWidgets.QLabel)],
                                 [meta[2], meta[3]])
                self.assertIsNone(header.findChild(QtWidgets.QLabel, "pyberPanelBadge"))
                self.assertTrue(header._title.wordWrap())
        header.set_from_meta(None)
        self.assertEqual(header._title.text(), "")
        header.deleteLater()

    def test_native_titlebar_does_not_create_an_offscreen_handle(self):
        """Native theming must not call winId on a non-Windows Qt backend."""
        widget = unittest.mock.Mock()
        with patch.object(QtGui.QGuiApplication, "platformName", return_value="offscreen"):
            apply_native_titlebar(widget, "dark")
        widget.winId.assert_not_called()


if __name__ == "__main__":
    unittest.main()
