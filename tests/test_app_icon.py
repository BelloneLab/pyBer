"""Regressions for application branding, asset fallback and native taskbar icons."""
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "pyBer"))

from PySide6 import QtCore, QtGui, QtWidgets
from shiboken6 import delete
import app_icon


class AppIconTests(unittest.TestCase):
    """Exercise real Qt image decoding without relying on the desktop theme."""

    @classmethod
    def setUpClass(cls):
        """Reuse the suite's QApplication when already present."""
        cls.app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

    def test_brand_is_visible_at_all_supported_sizes(self):
        """Every shell size contains visible, nonuniform pixels, not a blank icon."""
        icon = app_icon.build_icon()
        self.assertFalse(icon.isNull())
        for size in app_icon.ICON_SIZES:
            with self.subTest(size=size):
                image = icon.pixmap(QtCore.QSize(size, size), 1.0).toImage()
                self.assertEqual(image.size(), QtCore.QSize(size, size))
                colors = {image.pixelColor(x, y).rgba() for y in range(size)
                          for x in range(size)}
                self.assertGreater(len(colors), 10)
                visible = sum(image.pixelColor(x, y).alpha() > 0
                              for y in range(size) for x in range(size))
                # A frameless optical/mouse mark intentionally leaves empty space.
                self.assertGreater(visible, size * size * 0.3)
                for x, y in ((0, 0), (0, size - 1), (size - 1, 0), (size - 1, size - 1)):
                    self.assertLessEqual(image.pixelColor(x, y).alpha(), 32)

    def test_corrupt_ico_falls_back_to_original_brand_png(self):
        """A present but unreadable ICO must not prevent the usable PNG fallback."""
        with tempfile.TemporaryDirectory() as directory:
            damaged = Path(directory) / "damaged.ico"
            damaged.write_bytes(b"invalid icon")
            def candidates(filename):
                return [damaged] if filename.endswith(".ico") else [ROOT / "assets" / filename]
            with patch.object(app_icon, "asset_candidates", side_effect=candidates):
                self.assertFalse(app_icon.build_icon().isNull())

    def test_frozen_assets_include_bundle_and_executable_locations(self):
        """Support both PyInstaller's extraction folder and adjacent assets."""
        with patch.object(sys, "frozen", True, create=True), \
             patch.object(sys, "_MEIPASS", "D:/bundle", create=True), \
             patch.object(sys, "executable", "D:/install/pyBer.exe"):
            self.assertEqual(app_icon.asset_candidates("pyBer.ico"),
                             [Path("D:/bundle/assets/pyBer.ico"),
                              Path("D:/install/assets/pyBer.ico")])

    def test_install_is_idempotent_and_brands_detached_panels(self):
        """Repeated startup calls retain one listener and newly opened dialog icons."""
        previous_icon = self.app.windowIcon()
        previous_name = self.app.applicationDisplayName()
        previous_listener = getattr(self.app, "_pyber_window_icon_filter", None)
        app_icon.install_application_icon(self.app)
        listener = self.app._pyber_window_icon_filter
        app_icon.install_application_icon(self.app)
        self.assertIs(listener, self.app._pyber_window_icon_filter)
        window = QtWidgets.QDialog()
        try:
            self.app.sendEvent(window, QtCore.QEvent(QtCore.QEvent.Type.Show))
            self.assertEqual(window.windowIcon().cacheKey(), self.app.windowIcon().cacheKey())
            self.assertEqual(self.app.applicationDisplayName(), "pyBer")
        finally:
            delete(window)
            # Do not leave a test-installed global event filter or branding in
            # the shared QApplication used by unrelated GUI regression tests.
            if previous_listener is None:
                self.app.removeEventFilter(listener)
                del self.app._pyber_window_icon_filter
                delete(listener)
            self.app.setWindowIcon(previous_icon)
            self.app.setApplicationDisplayName(previous_name)

    def test_native_handle_destruction_never_sets_an_icon(self):
        """WinIdChange must not recreate a half-destroyed QWidget native handle."""
        listener = app_icon._WindowIconFilter(self.app)
        window = QtWidgets.QWidget()
        try:
            with patch.object(app_icon, "apply_native_window_icon") as native, \
                 patch.object(window, "setWindowIcon") as qt_icon:
                self.assertFalse(listener.eventFilter(
                    window, QtCore.QEvent(QtCore.QEvent.Type.WinIdChange)))
                native.assert_not_called()
                qt_icon.assert_not_called()
        finally:
            delete(window)
            delete(listener)

    def test_repeated_native_window_teardown_keeps_widget_registry_valid(self):
        """Exercise real handle creation/destruction while the lifecycle filter runs."""
        listener = app_icon._WindowIconFilter(self.app)
        self.app.installEventFilter(listener)
        try:
            for _ in range(24):
                window = QtWidgets.QWidget()
                # Creating a native handle does not display an interactive
                # window, but still exercises its WinIdChange teardown path.
                window.winId()
                self.app.sendEvent(window, QtCore.QEvent(QtCore.QEvent.Type.Show))
                delete(window)
                self.app.topLevelWidgets()
            self.app.processEvents()
        finally:
            self.app.removeEventFilter(listener)
            delete(listener)

    @unittest.skipUnless(os.name == "nt", "Windows native icon API")
    def test_native_icon_handles_are_valid_and_reused(self):
        """Keep cached HICONs alive and never feed a PNG to LoadImageW."""
        for size in (16, 32):
            handle = app_icon._native_icon(size)
            self.assertGreater(handle, 0)
            self.assertEqual(handle, app_icon._native_icon(size))
        with patch.object(app_icon, "icon_path", return_value="fallback.png"):
            self.assertEqual(app_icon._native_icon(32), 0)

    @unittest.skipUnless(os.name == "nt", "Windows native window API")
    def test_windows_hwnd_has_both_native_icon_sizes(self):
        """Read back WM_GETICON from a real HWND when run with the Windows backend."""
        if QtGui.QGuiApplication.platformName() != "windows":
            self.skipTest("Run with QT_QPA_PLATFORM=windows to verify real HWNDs")
        import ctypes
        from ctypes import wintypes
        window = QtWidgets.QWidget()
        try:
            app_icon.apply_native_window_icon(window)
            sender = ctypes.windll.user32.SendMessageW
            sender.argtypes = [wintypes.HWND, wintypes.UINT, wintypes.WPARAM, wintypes.LPARAM]
            sender.restype = wintypes.LPARAM
            hwnd = int(window.winId())
            for kind, size in enumerate(app_icon._native_window_sizes(hwnd)):
                self.assertEqual(sender(hwnd, 0x7F, kind, 0), app_icon._native_icon(size))
        finally:
            window.destroy()
            delete(window)

    @unittest.skipUnless(os.name == "nt", "Windows taskbar property store")
    def test_taskbar_properties_reference_durable_visible_artwork(self):
        """Verify shell branding independently of Qt/WM_GETICON handles."""
        if QtGui.QGuiApplication.platformName() != "windows":
            self.skipTest("Requires native HWND")
        import windows_taskbar
        window = QtWidgets.QWidget()
        try:
            app_icon.apply_native_window_icon(window)
            values = windows_taskbar.window_properties(int(window.winId()))
            self.assertEqual(values[5], app_icon.APP_USER_MODEL_ID)
            self.assertEqual(values[4], "pyBer")
            self.assertIn("main.py", values[2])
            self.assertTrue(values[3].endswith(",0"))
            asset = Path(values[3][:-2])
            self.assertTrue(asset.is_file())
            self.assertEqual(asset.read_bytes(), Path(app_icon.icon_path()).read_bytes())
            self.assertFalse(QtGui.QIcon(str(asset)).pixmap(32, 32).isNull())
        finally:
            window.destroy()
            delete(window)

    def test_shell_icon_cache_is_content_versioned_and_preserves_original(self):
        """New artwork gets a new durable resource name without deleting old pins."""
        import windows_taskbar
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "source.ico"
            source.write_bytes(Path(app_icon.icon_path()).read_bytes())
            windows_taskbar.persistent_icon.cache_clear()
            with patch.dict(os.environ, {"LOCALAPPDATA": directory}):
                first = Path(windows_taskbar.persistent_icon(str(source)))
                original = first.read_bytes()
                source.write_bytes(original + b"new-version")
                windows_taskbar.persistent_icon.cache_clear()
                second = Path(windows_taskbar.persistent_icon(str(source)))
                self.assertNotEqual(first, second)
                self.assertEqual(first.read_bytes(), original)
                self.assertEqual(second.read_bytes(), source.read_bytes())
            windows_taskbar.persistent_icon.cache_clear()


if __name__ == "__main__":
    unittest.main()
