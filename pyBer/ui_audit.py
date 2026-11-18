"""Self-audit of the settings UI, runnable inside the packaged executable.

Unit tests run against the developer's Qt; the released executable bundles
whatever Qt the build machine resolved. v0.57 shipped with a Qt whose numeric
fields hid their values, and no test noticed. This audit runs *inside* the
frozen app (``PYBER_UI_AUDIT=<folder>``), opens every settings drawer and
subsection, and checks every visible numeric field:

* its text editor is wide enough to show the value (not collapsed), and
* the field lies inside its drawer (not clipped / scrolled out of view).

It writes ``ui_audit.json`` plus one PNG per drawer, and returns the list of
failures so the caller can exit non-zero.
"""
from __future__ import annotations

import json
import os
from typing import Dict, List

from PySide6 import QtCore, QtWidgets


def _settle(app: QtWidgets.QApplication, rounds: int = 12) -> None:
    """Let deferred layout work (timers, slider wrapping, editor repair) run."""
    for _ in range(rounds):
        app.processEvents(QtCore.QEventLoop.ProcessEventsFlag.AllEvents)
        QtCore.QThread.msleep(15)


def _scroll_into_view(widget: QtWidgets.QWidget) -> None:
    """Scroll vertically like a user would; never sideways where users cannot.

    A field below the fold is fine (the drawer scrolls). A field beyond the
    right edge of a drawer whose horizontal scroll bar is hidden is not: the
    user cannot reach it, so the horizontal position is reset to 0.
    """
    host = widget.parentWidget()
    while host is not None and not isinstance(host, QtWidgets.QScrollArea):
        host = host.parentWidget()
    if host is None:
        return
    host.ensureWidgetVisible(widget, 0, 0)
    if host.horizontalScrollBarPolicy() == QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff:
        host.horizontalScrollBar().setValue(0)


def _visible_rect(widget: QtWidgets.QWidget) -> QtCore.QRect:
    """Part of ``widget`` (in its own coordinates) not clipped by any ancestor.

    Computed geometrically from the parent chain, so it is identical on every
    Qt platform (``visibleRegion()`` is empty on the offscreen test platform).
    """
    top = widget.window()
    origin = widget.mapTo(top, QtCore.QPoint(0, 0))
    rect = QtCore.QRect(origin, widget.size())
    ancestor = widget.parentWidget()
    while ancestor is not None:
        rect = rect.intersected(QtCore.QRect(ancestor.mapTo(top, QtCore.QPoint(0, 0)), ancestor.size()))
        if ancestor is top:
            break
        ancestor = ancestor.parentWidget()
    return rect.translated(-origin.x(), -origin.y())


def audit_spinboxes(root: QtWidgets.QWidget, where: str) -> List[Dict[str, object]]:
    """Return one failure record per visible spin box that hides its value.

    Uses each field's visible rectangle after clipping by every ancestor
    (scroll viewports, stacked pages, drawers), so a field pushed past any
    edge is caught, not only past its scroll area.
    """
    failures: List[Dict[str, object]] = []
    for spin in root.findChildren(QtWidgets.QAbstractSpinBox):
        if not spin.isVisible():
            continue
        _scroll_into_view(spin)
        editor = spin.lineEdit()
        text = spin.text()
        need = max(min(editor.fontMetrics().horizontalAdvance(text), 30) if text else 0, 20)
        record = {"where": where, "field": spin.objectName() or type(spin).__name__,
                  "text": text, "editor": list(editor.geometry().getRect())}
        if editor.width() < need:
            failures.append({**record, "problem": "collapsed editor"})
            continue
        shown = _visible_rect(editor)
        if shown.width() < need or shown.height() < editor.height() // 2:
            failures.append({**record, "problem": "value clipped or out of view",
                             "visible_editor": list(shown.getRect())})
            continue
        spin_shown = _visible_rect(spin)
        if spin_shown.width() < spin.width() - 2:
            failures.append({**record, "problem": "field partly clipped",
                             "visible_field": list(spin_shown.getRect()), "field_width": spin.width()})
    return failures


# Window sizes audited: a desktop monitor and a small laptop (1366x768 at
# 125% scaling, or GitHub's 1024x768 build machine). Fields must stay visible
# at both, since users resize and drawers get narrow.
AUDIT_SIZES = ((1500, 950), (1024, 700))


def _close_overlays(window: QtWidgets.QWidget) -> None:
    """Hide the first-run tutorial so it cannot cover the audited drawers."""
    for child in window.findChildren(QtWidgets.QWidget):
        if type(child).__name__ == "TutorialOverlay" and child.isVisible():
            child.hide()


def run_ui_audit(window: QtWidgets.QMainWindow, out_dir: str, sizes=AUDIT_SIZES) -> List[Dict[str, object]]:
    """Open every drawer/subsection of ``window`` at each size, audit, save evidence."""
    app = QtWidgets.QApplication.instance()
    os.makedirs(out_dir, exist_ok=True)
    failures: List[Dict[str, object]] = []
    visited: List[Dict[str, object]] = []

    def check(name: str) -> None:
        _settle(app)
        _close_overlays(window)
        found = audit_spinboxes(window, name)
        count = sum(1 for s in window.findChildren(QtWidgets.QAbstractSpinBox) if s.isVisible())
        visited.append({"view": name, "visible_numeric_fields": count, "failures": len(found),
                        "window": [window.width(), window.height()]})
        failures.extend(found)
        window.grab().save(os.path.join(out_dir, f"{name}.png"))

    window.show()
    for width, height in sizes:
        window.resize(width, height)
        _settle(app, 30)
        _close_overlays(window)
        tag = f"{window.width()}x{window.height()}"

        # Preprocessing drawers.
        if hasattr(window, "tabs"):
            window.tabs.setCurrentIndex(0)
        for key, button in getattr(window, "_section_buttons", {}).items():
            if not button.isChecked():
                button.click()
            check(f"{tag}_pre_{key}")

        # Postprocessing drawers, including every vertical subsection.
        post = getattr(window, "post_tab", None)
        if post is not None and hasattr(window, "tabs"):
            window.tabs.setCurrentIndex(window.tabs.indexOf(post))
            _settle(app, 20)
            for key, button in getattr(post, "_section_buttons", {}).items():
                if key == "sync":
                    continue  # opens a separate window
                button.setChecked(True)
                _settle(app)
                nav = getattr(post, f"{key}_subsections", None)
                if nav is None:
                    check(f"{tag}_post_{key}")
                    continue
                for index in range(nav.count()):
                    nav.setCurrentIndex(index)
                    check(f"{tag}_post_{key}_{nav.sectionTitle(index).lower()}")

    report = {
        "qt_version": QtCore.qVersion(),
        "views": visited,
        "numeric_fields_checked": sum(v["visible_numeric_fields"] for v in visited),
        "failures": failures,
    }
    with open(os.path.join(out_dir, "ui_audit.json"), "w", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2)
    return failures
