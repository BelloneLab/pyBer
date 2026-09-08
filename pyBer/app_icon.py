"""Shared, import-light branding for Qt windows and Windows taskbar entries.

The application identity and default icon must be installed before the first
native window exists. Source launches may also own a separate console window;
brand that window without changing a terminal shared with another application.
"""
from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

from PySide6 import QtCore, QtGui, QtWidgets


APP_USER_MODEL_ID = "BelloneLab.pyBer.FiberPhotometry"
ICON_SIZES = (16, 20, 24, 32, 40, 48, 64, 96, 128, 256)
_NATIVE_HANDLES: dict[tuple[str, int], int] = {}


def asset_candidates(filename: str) -> list[Path]:
    """Resolve source, one-folder and one-file bundled assets independently of cwd."""
    if getattr(sys, "frozen", False):
        roots = (Path(getattr(sys, "_MEIPASS", Path(sys.executable).parent)),
                 Path(sys.executable).parent)
    else:
        roots = (Path(__file__).resolve().parent.parent,)
    return [root / "assets" / filename for root in roots]


def icon_path() -> str:
    """Prefer the shell-compatible ICO, retaining the original logo as fallback."""
    for filename in ("pyBer.ico", "pyBer_logo_big.png"):
        for path in asset_candidates(filename):
            if path.is_file():
                return str(path)
    return str(asset_candidates("pyBer.ico")[0])


def build_icon() -> QtGui.QIcon:
    """Populate explicit small and high-DPI pixmaps, skipping damaged assets."""
    for filename in ("pyBer.ico", "pyBer_logo_big.png"):
        for path in asset_candidates(filename):
            if not path.is_file():
                continue
            source = QtGui.QIcon(str(path))
            icon = QtGui.QIcon()
            for size in ICON_SIZES:
                # Explicit physical-pixel sizes avoid accidentally baking the
                # current screen's 175%/200% scale into the ICO size registry.
                pixmap = source.pixmap(QtCore.QSize(size, size), 1.0)
                if not pixmap.isNull():
                    icon.addPixmap(pixmap)
            if not icon.isNull():
                return icon
    logging.warning("No readable pyBer application icon was found")
    return QtGui.QIcon()


def set_windows_app_id() -> None:
    """Set a stable identity before QApplication or any splash is constructed."""
    if os.name != "nt":
        return
    try:
        import ctypes
        from ctypes import wintypes
        function = ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID
        function.argtypes = [wintypes.LPCWSTR]
        function.restype = ctypes.HRESULT
        result = function(APP_USER_MODEL_ID)
        if result:
            logging.warning("Windows application identity failed: 0x%08x", result & 0xFFFFFFFF)
    except (AttributeError, OSError) as exc:
        logging.warning("Windows application identity unavailable: %s", exc)


def _native_icon(size: int) -> int:
    """Load and retain one HICON per size for the lifetime of this process.

    Windows does not copy handles supplied to WM_SETICON. A process-lifetime
    cache keeps them valid and avoids leaking a new pair on each window show.
    Only ICO files may be passed to the native icon loader, never PNG files.
    """
    import ctypes
    from ctypes import wintypes
    path = icon_path()
    if Path(path).suffix.lower() != ".ico":
        return 0
    key = (path, size)
    if key not in _NATIVE_HANDLES:
        loader = ctypes.windll.user32.LoadImageW
        loader.argtypes = [wintypes.HINSTANCE, wintypes.LPCWSTR, wintypes.UINT,
                           ctypes.c_int, ctypes.c_int, wintypes.UINT]
        loader.restype = wintypes.HANDLE
        handle = loader(None, path, 1, size, size, 0x10)
        if handle:
            _NATIVE_HANDLES[key] = int(handle)
    return _NATIVE_HANDLES.get(key, 0)


def _native_window_sizes(hwnd: int) -> tuple[int, int]:
    """Ask Windows for physical small/large icon sizes at this window's DPI."""
    import ctypes
    from ctypes import wintypes
    try:
        user = ctypes.windll.user32
        user.GetDpiForWindow.argtypes = [wintypes.HWND]
        user.GetDpiForWindow.restype = wintypes.UINT
        dpi = user.GetDpiForWindow(hwnd) or 96
        user.GetSystemMetricsForDpi.argtypes = [ctypes.c_int, wintypes.UINT]
        user.GetSystemMetricsForDpi.restype = ctypes.c_int
        return (user.GetSystemMetricsForDpi(49, dpi) or 16,
                user.GetSystemMetricsForDpi(11, dpi) or 32)
    except (AttributeError, OSError):
        return (16, 32)


def apply_native_window_icon(window: QtWidgets.QWidget) -> None:
    """Refresh small and taskbar icons after creation or native-handle changes."""
    if os.name != "nt" or QtGui.QGuiApplication.platformName() != "windows":
        return
    try:
        import ctypes
        from ctypes import wintypes
        sender = ctypes.windll.user32.SendMessageW
        sender.argtypes = [wintypes.HWND, wintypes.UINT, wintypes.WPARAM, wintypes.LPARAM]
        sender.restype = wintypes.LPARAM
        hwnd = int(window.winId())
        for kind, size in enumerate(_native_window_sizes(hwnd)):
            handle = _native_icon(size)
            if handle:
                sender(hwnd, 0x80, kind, handle)
    except (AttributeError, OSError, RuntimeError) as exc:
        logging.warning("Native window icon update failed: %s", exc)


def _brand_private_console() -> None:
    """Brand an IDE/script-created private console, never a shared terminal."""
    if os.name != "nt":
        return
    try:
        import ctypes
        from ctypes import wintypes
        kernel = ctypes.windll.kernel32
        processes = (wintypes.DWORD * 2)()
        count = kernel.GetConsoleProcessList(processes, len(processes))
        if count != 1 or processes[0] != os.getpid():
            return
        handle = _native_icon(32)
        if handle:
            setter = kernel.SetConsoleIcon
            setter.argtypes = [wintypes.HANDLE]
            setter.restype = wintypes.BOOL
            setter(handle)
    except (AttributeError, OSError):
        # Not every Windows terminal provides the legacy console icon API.
        pass


class _WindowIconFilter(QtCore.QObject):
    """Brand detached panels when shown, never while native windows are torn down."""

    def __init__(self, app: QtWidgets.QApplication) -> None:
        super().__init__(app)
        self._busy = False

    def eventFilter(self, watched: QtCore.QObject, event: QtCore.QEvent) -> bool:
        """Refresh visible windows without reacting to destructive WinIdChange events.

        Qt emits WinIdChange both when creating and destroying native windows.
        Setting an icon or requesting winId during destruction can recreate a
        half-destroyed window. A recreated visible window receives Show again,
        which is the safe boundary to refresh its taskbar branding.
        """
        if self._busy or event.type() != QtCore.QEvent.Type.Show:
            return False
        if isinstance(watched, QtWidgets.QWidget) and watched.isWindow():
            self._busy = True
            try:
                watched.setWindowIcon(QtWidgets.QApplication.windowIcon())
                if watched.testAttribute(QtCore.Qt.WidgetAttribute.WA_WState_Created):
                    apply_native_window_icon(watched)
            finally:
                self._busy = False
        return False


def install_application_icon(app: QtWidgets.QApplication) -> None:
    """Install early default branding once, including future detached dialogs."""
    icon = build_icon()
    if not icon.isNull():
        app.setWindowIcon(icon)
    app.setApplicationDisplayName("pyBer")
    if not hasattr(app, "_pyber_window_icon_filter"):
        listener = _WindowIconFilter(app)
        app.installEventFilter(listener)
        app._pyber_window_icon_filter = listener
    _brand_private_console()
