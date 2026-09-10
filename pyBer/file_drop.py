"""Local-file drop targets shared by docked and detached Qt panels."""
from __future__ import annotations

import os
import sys
from pathlib import Path
from PySide6 import QtCore, QtWidgets


ELEVATED_DROP_HELP = (
    "Windows blocks file drops from ordinary File Explorer into an administrator application. "
    "Save your work, close pyBer, then open pyBer from File Explorer without Run as administrator. "
    "When running from VS Code, close and reopen VS Code normally first. "
    "Open File and the load buttons remain available in this session."
)


def is_process_elevated():
    """Read this process's Windows token, without changing privileges or policy.

    Qt cannot receive an Explorer drop across this privilege boundary. Testing
    the process token, rather than account membership, also catches elevation
    inherited from an IDE. Unknown/non-Windows states produce no warning.
    """
    if sys.platform != "win32":
        return False
    import ctypes
    from ctypes import wintypes
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    security = ctypes.WinDLL("advapi32", use_last_error=True)
    kernel.GetCurrentProcess.restype = wintypes.HANDLE
    kernel.CloseHandle.argtypes = [wintypes.HANDLE]
    security.OpenProcessToken.argtypes = [wintypes.HANDLE, wintypes.DWORD, ctypes.POINTER(wintypes.HANDLE)]
    security.GetTokenInformation.argtypes = [wintypes.HANDLE, ctypes.c_int, wintypes.LPVOID,
                                           wintypes.DWORD, ctypes.POINTER(wintypes.DWORD)]
    token = wintypes.HANDLE()
    if not security.OpenProcessToken(kernel.GetCurrentProcess(), 0x0008, ctypes.byref(token)):
        return False
    try:
        elevated, size = wintypes.DWORD(), wintypes.DWORD()
        # TokenElevation (20) returns one DWORD, not a pointer-sized integer.
        success = security.GetTokenInformation(token, 20, ctypes.byref(elevated),
                                              ctypes.sizeof(elevated), ctypes.byref(size))
        return bool(success and elevated.value)
    finally:
        kernel.CloseHandle(token)


def create_drop_privilege_notice(parent=None):
    """Explain an actual Windows drop restriction without interrupting work."""
    if not is_process_elevated():
        return None
    notice = QtWidgets.QLabel("File drops blocked: administrator mode", parent)
    notice.setToolTip(ELEVATED_DROP_HELP)
    notice.setAccessibleDescription(ELEVATED_DROP_HELP)
    notice.setStyleSheet("color: #dba94d; padding: 0 8px;")
    return notice


def local_paths(mime, extensions=()):
    """Decode Explorer URLs without losing spaces, Unicode or UNC paths."""
    allowed = {extension.lower() for extension in extensions}
    paths, seen = [], set()
    for url in mime.urls() if mime is not None and mime.hasUrls() else ():
        if not url.isLocalFile():
            continue
        path = url.toLocalFile()
        key = os.path.normcase(os.path.abspath(path))
        if path and key not in seen and (os.path.isdir(path) or (
            os.path.isfile(path) and (not allowed or Path(path).suffix.lower() in allowed)
        )):
            paths.append(path)
            seen.add(key)
    return paths


def expand_paths(paths, extensions):
    """Expand dropped folders deterministically, retaining only supported files."""
    allowed = {extension.lower() for extension in extensions}
    result, seen = [], set()
    for path in paths:
        candidates = []
        if os.path.isdir(path):
            for directory, folders, files in os.walk(path):
                folders.sort()
                candidates.extend(os.path.join(directory, name) for name in sorted(files))
        else:
            candidates = [path]
        for candidate in candidates:
            key = os.path.normcase(os.path.abspath(candidate))
            if key not in seen and os.path.isfile(candidate) and Path(candidate).suffix.lower() in allowed:
                result.append(candidate)
                seen.add(key)
    return result


class FileDropTarget(QtCore.QObject):
    """Accept files on the actual receiving widget, including list viewports.

    Detached drawers cannot bubble a drop to the main window. Each receiver
    therefore owns its handler. Always copy from Explorer and finish its native
    drag transaction before invoking loaders that may open modal dialogs.
    """

    def __init__(self, widget, callback, extensions=()):
        super().__init__(widget)
        self.callback = callback
        self.extensions = extensions
        self.elevated = is_process_elevated()
        targets = [widget]
        if isinstance(widget, QtWidgets.QAbstractScrollArea):
            targets.append(widget.viewport())
        for target in targets:
            target.setAcceptDrops(True)
            target.installEventFilter(self)

    def eventFilter(self, watched, event):
        if event.type() == QtCore.QEvent.Type.Show and self.elevated:
            # Run after setup so loader-specific tooltips remain intact. Repeat
            # shows/reparenting must not duplicate the explanation.
            tooltip = watched.toolTip()
            if ELEVATED_DROP_HELP not in tooltip:
                watched.setToolTip((tooltip + "\n\n" + ELEVATED_DROP_HELP).strip())
        if event.type() not in (QtCore.QEvent.Type.DragEnter, QtCore.QEvent.Type.DragMove,
                                QtCore.QEvent.Type.Drop):
            return False
        if not event.mimeData().hasUrls():
            return False  # Preserve native item reordering.
        paths = local_paths(event.mimeData(), self.extensions)
        if not paths or not event.possibleActions() & QtCore.Qt.DropAction.CopyAction:
            event.ignore()
            return True
        event.setDropAction(QtCore.Qt.DropAction.CopyAction)
        event.accept()
        if event.type() == QtCore.QEvent.Type.Drop:
            QtCore.QTimer.singleShot(0, self, lambda: self.callback(paths))
        return True


def install_file_drop(widget, callback, extensions=()):
    """Retain a target on its widget for as long as the receiving UI exists."""
    target = FileDropTarget(widget, callback, extensions)
    widget._file_drop_target = target
    return target
