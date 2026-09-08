"""Local-file drop targets shared by docked and detached Qt panels."""
from __future__ import annotations

import os
from pathlib import Path
from PySide6 import QtCore, QtWidgets


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
        targets = [widget]
        if isinstance(widget, QtWidgets.QAbstractScrollArea):
            targets.append(widget.viewport())
        for target in targets:
            target.setAcceptDrops(True)
            target.installEventFilter(self)

    def eventFilter(self, watched, event):
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
