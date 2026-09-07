"""Render tool artwork and real panels without altering user preferences.

Run from an IDE or: conda run -n pyBer python scripts/validate_icon_design.py
The comparison uses the previous committed design, not a fabricated mockup.
"""

from contextlib import ExitStack
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "_test" / "icon_design"
BASELINE_REF = "7ac797a"
WINDOW_SIZE = (1500, 950)
os.environ.setdefault("QT_QPA_PLATFORM", "windows" if os.name == "nt" else "offscreen")
os.environ.setdefault("PYBER_SMOKE_TEST", "1")
sys.path.insert(0, str(ROOT / "pyBer"))

from PySide6 import QtCore, QtGui, QtWidgets, QtTest
from gui_postprocessing import PostProcessingPanel
from main import MainWindow
from onboarding import PanelHeader
from styles import apply_app_palette
from tool_icons import _icon_painter, _make_icon

TOOLS = [
    ("Data", "data", "database"), ("Artifacts", "artifacts", "sliders"),
    ("Filtering", "filtering", "filter"), ("Baseline", "baseline", "wave"),
    ("Output", "output", "chart"), ("Quality control", "qc", "badge"),
    ("Export", "export", "export"), ("Configuration", "config", "gear"),
    ("Setup", "setup", "sliders"), ("PSTH", "psth", "chart"),
    ("Spatial", "spatial", "grid"), ("Modeling", "modeling", "temporal"),
    ("Events", "events", "pulse"), ("Behavior", "behavior", "paw"),
    ("Synchronization", "sync", "sync"),
]


def contact_sheet():
    """Compare old small raster icons with new direct-size vector renders."""
    source = subprocess.check_output(["git", "show", f"{BASELINE_REF}:pyBer/styles.py"],
                                     cwd=ROOT).decode("utf-8")
    legacy = {}
    exec(source.split('# ===========================================================================')[0], legacy)
    canvas = QtGui.QPixmap(1100, 740)
    canvas.fill(QtGui.QColor("#10131c"))
    p = QtGui.QPainter(canvas)
    p.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
    p.setPen(QtGui.QColor("white"))
    p.setFont(QtGui.QFont("Segoe UI", 20, QtGui.QFont.Weight.DemiBold))
    p.drawText(32, 48, "Analysis tools, drawn with purpose")
    p.setPen(QtGui.QColor("#a9b3c9"))
    p.setFont(QtGui.QFont("Segoe UI", 10))
    p.drawText(32, 76, "Each card: previous icon at left, new icon at right. Enlarged artwork below.")
    for index, (label, name, previous) in enumerate(TOOLS):
        x, y = 24 + index % 5 * 216, 104 + index // 5 * 204
        p.setPen(QtGui.QColor("#232a3d"))
        p.setBrush(QtGui.QColor("#161a26"))
        p.drawRoundedRect(QtCore.QRectF(x, y, 204, 192), 12, 12)
        old_icon = legacy["_make_icon"](legacy[f"_paint_{previous}"])
        new_icon = _make_icon(_icon_painter(name))
        old_icon.paint(p, x + 52, y + 20, 24, 24)
        new_icon.paint(p, x + 128, y + 20, 24, 24)
        new_icon.paint(p, x + 78, y + 72, 48, 48)
        p.setPen(QtGui.QColor("#eef2f8"))
        p.drawText(QtCore.QRect(x, y + 142, 204, 28), QtCore.Qt.AlignmentFlag.AlignCenter, label)
    p.end()
    if not canvas.save(str(OUTPUT / "tool_icons_comparison.png")):
        raise RuntimeError("Could not save contact sheet")


def main():
    """Exercise actual panels and both themes with isolated layout/settings."""
    OUTPUT.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="pyber-icon-design-") as temporary, ExitStack() as stack:
        QtCore.QSettings.setDefaultFormat(QtCore.QSettings.Format.IniFormat)
        QtCore.QSettings.setPath(QtCore.QSettings.Format.IniFormat, QtCore.QSettings.Scope.UserScope, temporary)
        stack.enter_context(patch.object(MainWindow, "_panel_config_json_path", return_value=str(Path(temporary) / "layout.json")))
        stack.enter_context(patch.object(PostProcessingPanel, "_autosave_project_cache_path", return_value=str(Path(temporary) / "autosave.h5")))
        stack.enter_context(patch.object(PostProcessingPanel, "_restore_project_autosave_if_needed", return_value=None))
        app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
        apply_app_palette(app, "dark")
        contact_sheet()
        window = MainWindow()
        window.resize(*WINDOW_SIZE)
        window.show()
        QtTest.QTest.qWait(400)
        for theme in ("dark", "light"):
            window._apply_app_theme(theme, persist=False)
            for section in ("artifacts", "filtering", "temporal"):
                post = section == "temporal"
                window.tabs.setCurrentIndex(int(post))
                QtTest.QTest.qWait(150)
                owner = window.post_tab if post else window
                owner._section_buttons[section].setChecked(True)
                owner._toggle_section_popup(section, True)
                QtTest.QTest.qWait(250)
                assert not window.findChildren(QtWidgets.QLabel, "pyberPanelBadge")
                for header in window.findChildren(PanelHeader):
                    assert len(header.findChildren(QtWidgets.QLabel)) == 2
                target = OUTPUT / f"{theme}_{section}.png"
                if not window.grab().save(str(target)):
                    raise RuntimeError(f"Could not save {target}")
                print(f"Captured {target.name}", flush=True)
                if post:
                    # Modeling intentionally floats outside the main window;
                    # inspect that actual panel as well as its rail heading.
                    temporal = window.post_tab.section_temporal
                    host = temporal.window()
                    host.resize(1250, 850)
                    host.show()
                    QtTest.QTest.qWait(250)
                    header = temporal.findChild(QtWidgets.QFrame, "temporalHeader")
                    labels = header.findChildren(QtWidgets.QLabel)
                    assert all(label.text() != "T" for label in labels)
                    assert temporal.findChild(QtWidgets.QLabel, "temporalHeaderTitle") is not None
                    assert host.grab().save(str(OUTPUT / f"{theme}_temporal_panel.png"))
                    host.hide()
        window.post_tab._project_dirty = False
        window.close()
        app.processEvents()
    print(f"Visual validation complete: {OUTPUT}", flush=True)


if __name__ == "__main__":
    main()
