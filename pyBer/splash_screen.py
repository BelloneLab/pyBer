# splash_screen.py
"""Import-cheap startup splash for pyBer.

This module must stay light: it is imported before the heavy scientific
stack (numpy, scipy, h5py, pyqtgraph) precisely so the branded splash can
appear within about a second of launch, while the slow imports proceed
behind it. Keep the dependency set to os/sys/PySide6/version only.
"""
from __future__ import annotations

import math
import os
import sys
from typing import List, Optional

from PySide6 import QtCore, QtGui, QtWidgets

from version import __version__


def _asset_candidates(filename: str) -> List[str]:
    if getattr(sys, "frozen", False):
        base_dir = str(getattr(sys, "_MEIPASS", "")) or os.path.dirname(sys.executable)
        return [
            os.path.join(base_dir, "assets", filename),
            os.path.join(os.path.dirname(sys.executable), "assets", filename),
        ]
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return [os.path.join(base_dir, "assets", filename)]


def _first_existing_asset(filename: str) -> str:
    candidates = _asset_candidates(filename)
    for path in candidates:
        if os.path.isfile(path):
            return path
    return candidates[0] if candidates else filename


def splash_logo_path() -> str:
    return _first_existing_asset("pyBer_logo_big.png")


def build_branded_splash() -> Optional[QtGui.QPixmap]:
    """Compose the startup splash: rounded obsidian card, iris glow, logo,
    wordmark and version. Falls back to the raw logo if anything fails."""
    try:
        w, h = 620, 380
        pix = QtGui.QPixmap(w, h)
        pix.fill(QtCore.Qt.GlobalColor.transparent)
        p = QtGui.QPainter(pix)
        p.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
        p.setRenderHint(QtGui.QPainter.RenderHint.SmoothPixmapTransform)

        card = QtCore.QRectF(1, 1, w - 2, h - 2)
        path = QtGui.QPainterPath()
        path.addRoundedRect(card, 18, 18)
        p.setClipPath(path)

        ground = QtGui.QLinearGradient(0, 0, 0, h)
        ground.setColorAt(0.0, QtGui.QColor("#151a28"))
        ground.setColorAt(1.0, QtGui.QColor("#0b0e15"))
        p.fillPath(path, ground)

        # Iris glow rising behind the logo.
        glow = QtGui.QRadialGradient(w / 2.0, h * 0.42, w * 0.55)
        glow.setColorAt(0.0, QtGui.QColor(124, 92, 255, 66))
        glow.setColorAt(0.55, QtGui.QColor(124, 92, 255, 18))
        glow.setColorAt(1.0, QtGui.QColor(124, 92, 255, 0))
        p.fillPath(path, glow)

        # Faint oscilloscope trace along the lower third.
        trace = QtGui.QPainterPath()
        base_y = h * 0.80
        trace.moveTo(0, base_y)
        for x in range(0, w + 1, 2):
            t = x / w
            y = base_y - math.sin(t * math.pi * 6.0) * 7.0 * math.exp(-((t - 0.5) ** 2) * 6.0)
            if 0.44 < t < 0.50:
                y -= 26.0 * math.exp(-((t - 0.47) * 90.0) ** 2)
            trace.lineTo(x, y)
        p.setPen(QtGui.QPen(QtGui.QColor(80, 250, 160, 40), 4.0))
        p.drawPath(trace)
        p.setPen(QtGui.QPen(QtGui.QColor(80, 250, 160, 150), 1.4))
        p.drawPath(trace)

        # Logo.
        logo_path = splash_logo_path()
        logo_bottom = h * 0.40
        if os.path.isfile(logo_path):
            logo = QtGui.QPixmap(logo_path)
            if not logo.isNull():
                logo = logo.scaledToHeight(
                    110, QtCore.Qt.TransformationMode.SmoothTransformation)
                p.drawPixmap(int((w - logo.width()) / 2), int(h * 0.13), logo)
                logo_bottom = h * 0.13 + 110

        # Wordmark and version.
        title_font = QtGui.QFont("Segoe UI Variable Display", 27)
        title_font.setWeight(QtGui.QFont.Weight.ExtraBold)
        p.setFont(title_font)
        p.setPen(QtGui.QColor("#f7f9fd"))
        p.drawText(QtCore.QRectF(0, logo_bottom + 8, w, 52),
                   QtCore.Qt.AlignmentFlag.AlignHCenter, "pyBer")

        sub_font = QtGui.QFont("Segoe UI Variable Text", 10)
        p.setFont(sub_font)
        p.setPen(QtGui.QColor("#a9b3c9"))
        p.drawText(QtCore.QRectF(0, logo_bottom + 62, w, 24),
                   QtCore.Qt.AlignmentFlag.AlignHCenter,
                   f"Fiber Photometry Analysis  ·  v{__version__}")

        # Accent baseline.
        bar = QtGui.QLinearGradient(0, 0, w, 0)
        bar.setColorAt(0.0, QtGui.QColor(124, 92, 255, 0))
        bar.setColorAt(0.5, QtGui.QColor("#7c5cff"))
        bar.setColorAt(1.0, QtGui.QColor(124, 92, 255, 0))
        p.fillRect(QtCore.QRectF(0, h - 4, w, 3), bar)

        p.setClipping(False)
        p.setPen(QtGui.QPen(QtGui.QColor("#333c56"), 1.0))
        p.setBrush(QtCore.Qt.BrushStyle.NoBrush)
        p.drawPath(path)
        p.end()
        return pix
    except Exception:
        try:
            icon_path = splash_logo_path()
            if os.path.isfile(icon_path):
                return QtGui.QPixmap(icon_path)
        except Exception:
            pass
        return None


def _set_windows_app_user_model_id_early() -> None:
    if os.name != "nt":
        return
    try:
        import ctypes
        from ctypes import wintypes

        func = ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID
        func.argtypes = [wintypes.LPCWSTR]
        func.restype = ctypes.HRESULT
        func("BelloneLab.pyBer.FiberPhotometry")
    except Exception:
        pass


def show_early_splash() -> Optional[QtWidgets.QSplashScreen]:
    """Create the QApplication and show the splash immediately.

    Called from main.py before its heavy imports run. Returns the splash (or
    None); the QApplication is reachable via QApplication.instance().
    """
    try:
        _set_windows_app_user_model_id_early()
        app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv[:1])
        pix = build_branded_splash()
        if pix is None or pix.isNull():
            return None
        splash = QtWidgets.QSplashScreen(pix, QtCore.Qt.WindowType.WindowStaysOnTopHint)
        splash.setAttribute(QtCore.Qt.WidgetAttribute.WA_TranslucentBackground, True)
        icon_path = _first_existing_asset("pyBer.ico")
        if os.path.isfile(icon_path):
            splash.setWindowIcon(QtGui.QIcon(icon_path))
        splash.show()
        app.processEvents(QtCore.QEventLoop.ProcessEventsFlag.AllEvents)
        return splash
    except Exception:
        return None
