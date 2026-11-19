# splash_screen.py
"""Import-cheap animated startup splash for pyBer.

This module must stay light: it is imported before the heavy scientific
stack (numpy, scipy, h5py, pyqtgraph) precisely so the branded splash can
appear within about a second of launch, while the slow imports proceed
behind it. Keep the dependency set to the standard library, PySide6 and
``version`` only.

How the animation stays smooth
------------------------------
While pyBer loads, its main thread is busy importing and building widgets,
so it cannot repaint anything. The splash therefore has two parts:

1. the launching process shows a *static* frame immediately (QSplashScreen);
2. a tiny helper process (``splash_helper.py``; the frozen executable
   re-enters itself with ``--pyber-splash``) draws the *animated* window on
   top of it at 60 fps, with its own event loop.

The helper leaves when the main window is ready (a stop file appears) or
when the launching process disappears. If the helper cannot start, the
static frame simply remains: loading never depends on the animation.

What the animation shows
------------------------
Each simulated calcium transient launches a pulse of light at the implant
on the mouse's head. The pulse travels up the fiber and, when it reaches the
top, the transient appears at the writing head of the scrolling trace.
Events, amplitudes and timing are deterministic (fixed seed). The colour of
the light and of the trace is one of ``THEMES``, drawn at random at every
launch (never the same twice in a row).

Why it is crisp
---------------
Everything is vector drawing at the screen's own pixel density. The trace
is a chain of cubic curves built from the signal's exact value *and slope*
(Hermite interpolation), so it has no polyline corners; its glow is a stack
of real strokes, not an upscaled bitmap. The light in the fiber is a single
smooth stroke whose brightness follows a gradient along the fiber.
"""
from __future__ import annotations

import bisect
import math
import os
import random
import sys
from typing import Dict, List, NamedTuple, Optional, Sequence, Tuple

from PySide6 import QtCore, QtGui, QtWidgets

from version import __version__

# ---------------------------------------------------------------------------
# Parameters (logical pixels, seconds). Edit here to restyle the splash.
# ---------------------------------------------------------------------------
CARD_W, CARD_H = 640, 420          # splash card size
CARD_RADIUS = 18                   # rounded corners
LOGO_H = 172                       # logo height (square artwork)
LOGO_TOP = 24                      # logo top edge
TITLE_PX, SUBTITLE_PX = 36, 13     # wordmark / subtitle font sizes

TRACE_BASE_Y = 374                 # resting level of the signal trace
TRACE_GAIN = 44                    # pixels for a transient of amplitude 1
TRACE_HEAD_X = CARD_W - 46         # x of the writing head ("now")
TRACE_WINDOW_S = 5.2               # seconds of signal visible left of the head
TRACE_FADE_FRACTION = 0.24         # left part of the trace that fades out
# Trace strokes, widest first: (width, opacity 0-255, 0 = glow colour / 1 = core colour)
TRACE_LAYERS = ((15.0, 13, 0), (9.5, 24, 0), (5.6, 46, 0), (3.2, 96, 0), (1.7, 255, 1))

TAU_RISE_S, TAU_DECAY_S = 0.07, 0.60   # transient kinetics
EVENT_GAP_S = (0.75, 2.0)          # random interval between transients
DOUBLET_PROBABILITY = 0.22         # chance of a closely following transient
DOUBLET_GAP_S = (0.26, 0.40)
EVENT_AMPLITUDE = (0.55, 1.0)
MODEL_SEED = 11
KNOT_STEP_S = 0.20                 # spacing of curve knots on smooth stretches
# Extra knots after each transient onset, where the curve bends fastest.
KNOT_OFFSETS_S = (0.0, 0.03, 0.07, 0.12, 0.18, 0.26, 0.36)

FIBER_TRAVEL_S = 0.62              # time for light to climb the fiber
PULSE_TAIL = 0.46                  # comet tail, as a fraction of the fiber
PULSE_LEAD = 0.05                  # soft leading edge of the comet
FIBER_SAMPLES = 96                 # resolution of the brightness gradient
# Comet strokes, widest first: (width, opacity 0-255, 0 = body colour / 1 = hot core)
PULSE_LAYERS = ((14.0, 48, 0), (6.4, 120, 0), (2.5, 250, 1))
# Fiber centre line in the logo's own 500 x 500 coordinates, from the implant
# on the mouse's head up to the optics (two cubic Bezier segments), and the
# highlight drawn along it (both copied from assets/pyBer_logo.svg).
FIBER_LOGO_PATH = ((139, 318), (143, 216), (156, 181), (214, 152),
                   (253, 132), (270, 103), (269, 35))
FIBER_LOGO_HIGHLIGHT = ((142, 313), (145, 220), (158, 181), (211, 151),
                        (249, 132), (267, 105), (266, 39))
FIBER_LOGO_WIDTH, FIBER_LOGO_HIGHLIGHT_WIDTH = 12.0, 2.5
FIBER_LOGO_HEAD_Y = 315.2          # below this the fiber is hidden by the mouse's head
LOGO_VIEWBOX = 500.0

FRAME_INTERVAL_MS = 16             # ~60 fps
FADE_IN_S, FADE_OUT_S = 0.22, 0.26
START_T = 12.0                     # animation phase of the static first frame
HELPER_MAX_LIFETIME_S = 300.0      # safety net: the helper never outlives this
HELPER_FLAG = "--pyber-splash"


class Theme(NamedTuple):
    """Colour of the light and of the signal for one launch.

    ``deep`` is the saturated colour at the implant, ``mid`` the body of the
    light, ``light`` its pale end at the optics, ``hot`` the almost-white
    core of a pulse and of the trace's writing head.
    """
    name: str
    deep: Tuple[int, int, int]
    mid: Tuple[int, int, int]
    light: Tuple[int, int, int]
    hot: Tuple[int, int, int]


THEMES: Tuple[Theme, ...] = (
    Theme("emerald", (22, 226, 130), (96, 250, 166), (158, 255, 222), (240, 255, 248)),
    Theme("violet", (140, 84, 255), (178, 134, 255), (214, 192, 255), (247, 242, 255)),
    Theme("ruby", (255, 56, 84), (255, 104, 120), (255, 170, 168), (255, 242, 240)),
    Theme("pink", (255, 72, 166), (255, 124, 196), (255, 184, 226), (255, 242, 250)),
    Theme("amber", (255, 170, 28), (255, 202, 78), (255, 230, 150), (255, 250, 232)),
    Theme("cyan", (0, 206, 255), (84, 228, 255), (170, 245, 255), (238, 252, 255)),
    Theme("azure", (58, 122, 255), (108, 164, 255), (172, 208, 255), (238, 245, 255)),
    Theme("coral", (255, 112, 44), (255, 152, 92), (255, 198, 150), (255, 244, 234)),
)


def _theme_memory_path() -> str:
    base = os.environ.get("LOCALAPPDATA") or os.environ.get("TEMP") or os.path.expanduser("~")
    return os.path.join(base, "pyBer", "splash_theme.txt")


def pick_theme(remember: bool = True) -> int:
    """Random theme index for this launch, different from the previous launch."""
    choices = list(range(len(THEMES)))
    path = _theme_memory_path()
    if remember:
        try:
            with open(path, "r", encoding="utf-8") as stream:
                previous = int(stream.read().strip())
            if previous in choices and len(choices) > 1:
                choices.remove(previous)
        except (OSError, ValueError):
            pass
    index = random.SystemRandom().choice(choices)
    if remember:
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "w", encoding="utf-8") as stream:
                stream.write(str(index))
        except OSError:
            pass
    return index


# ---------------------------------------------------------------------------
# Assets
# ---------------------------------------------------------------------------
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


# ---------------------------------------------------------------------------
# Signal model: deterministic calcium-like transients
# ---------------------------------------------------------------------------
class SignalModel:
    """Simulated photometry trace: a train of transients plus a slow ripple.

    ``events`` are (arrival time, amplitude). A transient starts on the trace
    at its arrival time; the matching light pulse left the mouse
    ``FIBER_TRAVEL_S`` earlier. Everything is generated from a fixed seed, so
    every launch (and the static first frame) shows the same sequence.

    The transient kernel is ``(1 - exp(-d / tau_rise)) * exp(-d / tau_decay)``
    scaled to peak at 1; its slope is known in closed form, which lets the
    trace be drawn as smooth cubic curves instead of a polyline.
    """

    def __init__(self, seed: int = MODEL_SEED) -> None:
        self._rng = random.Random(seed)
        self._times: List[float] = []
        self._amps: List[float] = []
        self._next = 0.4
        self._knots: Dict[float, Tuple[float, float]] = {}
        # Normalize the kernel so a transient of amplitude 1 peaks at 1.
        peak_at = TAU_RISE_S * math.log(1.0 + TAU_DECAY_S / TAU_RISE_S)
        self._norm = (1.0 - math.exp(-peak_at / TAU_RISE_S)) * math.exp(-peak_at / TAU_DECAY_S)
        #: slope added by a transient of amplitude 1 at the instant it starts
        self.onset_slope = 1.0 / (TAU_RISE_S * self._norm)

    def _extend(self, until: float) -> None:
        rng = self._rng
        while self._next <= until:
            self._times.append(self._next)
            self._amps.append(rng.uniform(*EVENT_AMPLITUDE))
            if rng.random() < DOUBLET_PROBABILITY:
                self._next += rng.uniform(*DOUBLET_GAP_S)
            else:
                self._next += rng.uniform(*EVENT_GAP_S)

    def events_between(self, start: float, stop: float) -> List[Tuple[float, float]]:
        """Transients arriving in [start, stop], oldest first."""
        self._extend(stop)
        lo = bisect.bisect_left(self._times, start)
        hi = bisect.bisect_right(self._times, stop)
        return list(zip(self._times[lo:hi], self._amps[lo:hi]))

    def value_slope(self, t: float) -> Tuple[float, float]:
        """Signal and its time derivative at t (transients starting at t included)."""
        value = slope = 0.0
        for arrival, amp in self.events_between(t - 9.0 * TAU_DECAY_S, t):
            d = t - arrival
            rise, decay = math.exp(-d / TAU_RISE_S), math.exp(-d / TAU_DECAY_S)
            value += amp * (1.0 - rise) * decay / self._norm
            slope += amp * (rise * decay / TAU_RISE_S - (1.0 - rise) * decay / TAU_DECAY_S) / self._norm
        w1, w2 = 2.0 * math.pi * 0.9, 2.0 * math.pi * 2.3
        value += 0.022 * math.sin(w1 * t + 0.4) + 0.011 * math.sin(w2 * t + 1.9)
        slope += 0.022 * w1 * math.cos(w1 * t + 0.4) + 0.011 * w2 * math.cos(w2 * t + 1.9)
        return value, slope

    def value(self, t: float) -> float:
        return self.value_slope(t)[0]

    def knot(self, t: float) -> Tuple[float, float]:
        """Cached ``value_slope`` for curve knots, which recur from frame to frame."""
        cached = self._knots.get(t)
        if cached is None:
            cached = self.value_slope(t)
            if len(self._knots) > 3000:
                self._knots = {k: v for k, v in self._knots.items() if k > t - 2.0 * TRACE_WINDOW_S}
            self._knots[t] = cached
        return cached

    def knots_between(self, start: float, stop: float) -> List[Tuple[float, float]]:
        """Curve knots in [start, stop] as (time, amplitude of a transient starting there or 0).

        Knots sit on a regular grid in signal time (so they scroll with the
        trace and never shimmer) and cluster after each onset.
        """
        # Times stay exact (never rounded): an onset knot must not fall a hair
        # before its own transient, or its slope would miss the transient.
        marks: Dict[float, float] = {start: 0.0, stop: 0.0}
        first = math.ceil(start / KNOT_STEP_S)
        for index in range(first, int(math.floor(stop / KNOT_STEP_S)) + 1):
            marks.setdefault(index * KNOT_STEP_S, 0.0)
        for arrival, amp in self.events_between(start - KNOT_OFFSETS_S[-1], stop):
            for offset in KNOT_OFFSETS_S:
                when = arrival + offset
                if start <= when <= stop:
                    marks[when] = amp if offset == 0.0 else marks.get(when, 0.0)
        ordered = sorted(marks.items())
        kept = [ordered[0]]
        for when, amp in ordered[1:]:
            if when - kept[-1][0] < 0.012 and amp == 0.0 and when != ordered[-1][0]:
                continue            # too close to its neighbour to matter
            kept.append((when, amp))
        return kept


_CLIP_LIMIT, _CLIP_KNEE = 1.12, 0.22


def _soft_clip(value: float, slope: float = 0.0) -> Tuple[float, float]:
    """Keep summed transients inside the trace band without a hard corner."""
    if value <= _CLIP_LIMIT:
        return value, slope
    bend = math.tanh((value - _CLIP_LIMIT) / _CLIP_KNEE)
    return _CLIP_LIMIT + _CLIP_KNEE * bend, slope * (1.0 - bend * bend)


def _mix(a: Sequence[float], b: Sequence[float], w: float) -> Tuple[int, int, int]:
    w = 0.0 if w < 0.0 else 1.0 if w > 1.0 else w
    return (int(a[0] + (b[0] - a[0]) * w), int(a[1] + (b[1] - a[1]) * w), int(a[2] + (b[2] - a[2]) * w))


def fiber_color(s: float, theme: Theme) -> Tuple[int, int, int]:
    """Colour of the light at position s along the fiber (0 = mouse, 1 = top)."""
    s = 0.0 if s < 0.0 else 1.0 if s > 1.0 else s
    if s <= 0.55:
        return _mix(theme.deep, theme.mid, s / 0.55)
    return _mix(theme.mid, theme.light, (s - 0.55) / 0.45)


def _rgba(rgb: Sequence[int], alpha: float) -> QtGui.QColor:
    return QtGui.QColor(rgb[0], rgb[1], rgb[2], max(0, min(255, int(alpha))))


def _round_pen(brush, width: float) -> QtGui.QPen:
    pen = QtGui.QPen(brush, width)
    pen.setCapStyle(QtCore.Qt.PenCapStyle.RoundCap)
    pen.setJoinStyle(QtCore.Qt.PenJoinStyle.RoundJoin)
    return pen


# ---------------------------------------------------------------------------
# Artwork: one painter for the static frame and every animated frame
# ---------------------------------------------------------------------------
class SplashArt:
    """Draws the splash. ``paint(painter, t)`` renders the frame at time t."""

    def __init__(self, dpr: float = 1.0, theme: int = 0) -> None:
        self.dpr = max(1.0, float(dpr))
        self.theme_index = int(theme) % len(THEMES)
        self.theme = THEMES[self.theme_index]
        self.model = SignalModel()
        self.card_path = QtGui.QPainterPath()
        self.card_path.addRoundedRect(QtCore.QRectF(1, 1, CARD_W - 2, CARD_H - 2), CARD_RADIUS, CARD_RADIUS)
        self.glow_clip = QtGui.QPainterPath()
        self.glow_clip.addRoundedRect(QtCore.QRectF(2, 2, CARD_W - 4, CARD_H - 4), CARD_RADIUS - 1, CARD_RADIUS - 1)
        self.logo_scale = LOGO_H / LOGO_VIEWBOX
        self.logo_left = (CARD_W - LOGO_H) / 2.0
        self.fiber_path = self._logo_curve(FIBER_LOGO_PATH)
        length = self.fiber_path.length()
        self.fiber_points = [self.fiber_path.pointAtPercent(self.fiber_path.percentAtLength(length * i / FIBER_SAMPLES))
                             for i in range(FIBER_SAMPLES + 1)]
        # The fiber rises monotonically, so height tells where we are along it:
        # a vertical gradient can then paint brightness as a function of position.
        self.fiber_bottom, self.fiber_top = self.fiber_points[0].y(), self.fiber_points[-1].y()
        span = self.fiber_bottom - self.fiber_top
        self.fiber_heights = [(self.fiber_bottom - point.y()) / span for point in self.fiber_points]
        self.static = self._build_static()
        self.band_top = TRACE_BASE_Y - int(TRACE_GAIN * 1.5) - 12
        self.fade = self._build_left_fade()

    # -- construction --------------------------------------------------------
    def _new_pixmap(self, width: int, height: int) -> QtGui.QPixmap:
        pix = QtGui.QPixmap(int(round(width * self.dpr)), int(round(height * self.dpr)))
        pix.setDevicePixelRatio(self.dpr)
        pix.fill(QtCore.Qt.GlobalColor.transparent)
        return pix

    def _logo_point(self, x: float, y: float) -> QtCore.QPointF:
        return QtCore.QPointF(self.logo_left + x * self.logo_scale, LOGO_TOP + y * self.logo_scale)

    def _logo_curve(self, control_points) -> QtGui.QPainterPath:
        """Chain of cubic Beziers given in logo coordinates, mapped onto the card."""
        path = QtGui.QPainterPath(self._logo_point(*control_points[0]))
        for first in range(1, len(control_points), 3):
            path.cubicTo(self._logo_point(*control_points[first]), self._logo_point(*control_points[first + 1]),
                         self._logo_point(*control_points[first + 2]))
        return path

    def _paint_fiber_cable(self, p: QtGui.QPainter) -> None:
        """Repaint the logo's fiber in this launch's colour, exactly over the original."""
        theme, scale = self.theme, self.logo_scale
        head_y = LOGO_TOP + FIBER_LOGO_HEAD_Y * scale
        p.save()
        p.setClipRect(QtCore.QRectF(0, 0, CARD_W, head_y), QtCore.Qt.ClipOperation.IntersectClip)
        body = QtGui.QLinearGradient(0, self.fiber_top, 0, self.fiber_bottom)
        body.setColorAt(0.0, _rgba(_mix(theme.light, theme.hot, 0.25), 255))
        body.setColorAt(1.0, _rgba(theme.deep, 255))
        p.setBrush(QtCore.Qt.BrushStyle.NoBrush)
        p.setPen(_round_pen(QtGui.QBrush(body), (FIBER_LOGO_WIDTH + 0.9) * scale))
        p.drawPath(self.fiber_path)
        p.setPen(_round_pen(_rgba(theme.hot, 225), FIBER_LOGO_HIGHLIGHT_WIDTH * scale))
        p.drawPath(self._logo_curve(FIBER_LOGO_HIGHLIGHT))
        p.restore()

    def _build_static(self) -> QtGui.QPixmap:
        """Card, glows, logo, wordmark and version: everything that never moves."""
        theme = self.theme
        pix = self._new_pixmap(CARD_W, CARD_H)
        p = QtGui.QPainter(pix)
        p.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
        p.setRenderHint(QtGui.QPainter.RenderHint.SmoothPixmapTransform)
        p.setRenderHint(QtGui.QPainter.RenderHint.TextAntialiasing)
        p.setClipPath(self.card_path)

        ground = QtGui.QLinearGradient(0, 0, 0, CARD_H)
        ground.setColorAt(0.0, QtGui.QColor("#151a28"))
        ground.setColorAt(1.0, QtGui.QColor("#0b0e15"))
        p.fillPath(self.card_path, ground)

        # Iris glow rising behind the logo, and a faint themed one behind the trace.
        glow = QtGui.QRadialGradient(CARD_W / 2.0, CARD_H * 0.36, CARD_W * 0.55)
        glow.setColorAt(0.0, QtGui.QColor(124, 92, 255, 66))
        glow.setColorAt(0.55, QtGui.QColor(124, 92, 255, 18))
        glow.setColorAt(1.0, QtGui.QColor(124, 92, 255, 0))
        p.fillPath(self.card_path, glow)
        floor = QtGui.QRadialGradient(CARD_W * 0.62, CARD_H + 40.0, CARD_W * 0.55)
        floor.setColorAt(0.0, _rgba(theme.deep, 34))
        floor.setColorAt(1.0, _rgba(theme.deep, 0))
        p.fillPath(self.card_path, floor)

        logo_path = splash_logo_path()
        if os.path.isfile(logo_path):
            logo = QtGui.QPixmap(logo_path)
            if not logo.isNull():
                logo = logo.scaledToHeight(int(round(LOGO_H * self.dpr)),
                                           QtCore.Qt.TransformationMode.SmoothTransformation)
                logo.setDevicePixelRatio(self.dpr)
                p.drawPixmap(QtCore.QPointF(self.logo_left, LOGO_TOP), logo)
                self._paint_fiber_cable(p)

        title_top = LOGO_TOP + LOGO_H + 2
        title_font = QtGui.QFont("Segoe UI Variable Display")
        title_font.setPixelSize(TITLE_PX)
        title_font.setWeight(QtGui.QFont.Weight.ExtraBold)
        p.setFont(title_font)
        p.setPen(QtGui.QColor("#f7f9fd"))
        p.drawText(QtCore.QRectF(0, title_top, CARD_W, 50),
                   QtCore.Qt.AlignmentFlag.AlignHCenter | QtCore.Qt.AlignmentFlag.AlignVCenter, "pyBer")
        sub_font = QtGui.QFont("Segoe UI Variable Text")
        sub_font.setPixelSize(SUBTITLE_PX)
        p.setFont(sub_font)
        p.setPen(QtGui.QColor("#a9b3c9"))
        p.drawText(QtCore.QRectF(0, title_top + 52, CARD_W, 22),
                   QtCore.Qt.AlignmentFlag.AlignHCenter | QtCore.Qt.AlignmentFlag.AlignVCenter,
                   f"Fiber Photometry Analysis  ·  v{__version__}")

        p.setClipping(False)
        p.setPen(QtGui.QPen(QtGui.QColor("#333c56"), 1.0))
        p.setBrush(QtCore.Qt.BrushStyle.NoBrush)
        p.drawPath(self.card_path)
        p.end()
        return pix

    def _build_left_fade(self) -> QtGui.QPixmap:
        """Copy of the static background that dissolves the trace toward the left edge."""
        width = int(CARD_W * TRACE_FADE_FRACTION)
        height = CARD_H - self.band_top
        pix = self._new_pixmap(width, height)
        p = QtGui.QPainter(pix)
        p.drawPixmap(QtCore.QPointF(0, 0), self.static,
                     QtCore.QRectF(0, self.band_top * self.dpr, width * self.dpr, height * self.dpr))
        # Smooth (ease-in-out) dissolve: fully background at the edge, clear at the right.
        mask = QtGui.QLinearGradient(0, 0, width, 0)
        for step in range(11):
            x = step / 10.0
            ease = x * x * (3.0 - 2.0 * x)
            mask.setColorAt(x, QtGui.QColor(0, 0, 0, int(255 * (1.0 - ease))))
        p.setCompositionMode(QtGui.QPainter.CompositionMode.CompositionMode_DestinationIn)
        p.fillRect(QtCore.QRectF(0, 0, width, height), mask)
        p.end()
        return pix

    # -- frames --------------------------------------------------------------
    def paint(self, p: QtGui.QPainter, t: float) -> None:
        p.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
        p.drawPixmap(QtCore.QPointF(0, 0), self.static)
        p.save()
        p.setClipPath(self.glow_clip)
        self._paint_trace(p, t)
        self._paint_fiber(p, t)
        p.restore()

    def trace_path(self, t: float) -> Tuple[QtGui.QPainterPath, QtCore.QPointF, float]:
        """The visible signal as smooth cubic curves; also the head point and level.

        Between two knots the curve is the cubic Hermite interpolant of the
        signal's exact value and slope, so it follows the model to a small
        fraction of a pixel. At a transient's onset the slope jumps: the knot
        there keeps a true corner by using its left slope for the incoming
        curve and its right slope for the outgoing one.
        """
        model = self.model
        speed = TRACE_HEAD_X / TRACE_WINDOW_S           # pixels per second
        start = t - TRACE_WINDOW_S - 0.05
        knots = model.knots_between(start, t)
        path = QtGui.QPainterPath()
        previous = None
        for when, onset_amp in knots:
            value, slope_right = model.knot(when)
            slope_left = slope_right - onset_amp * model.onset_slope
            x = TRACE_HEAD_X - (t - when) * speed
            shown, _ = _soft_clip(value)
            y = TRACE_BASE_Y - TRACE_GAIN * shown
            if previous is None:
                path.moveTo(x, y)
            else:
                x0, y0, value0, slope0, when0 = previous
                third = (when - when0) / 3.0
                # Slopes in pixels: dy/dx = -gain * dvalue/dt / speed.
                d0 = -TRACE_GAIN * _soft_clip(value0, slope0)[1]
                d1 = -TRACE_GAIN * _soft_clip(value, slope_left)[1]
                path.cubicTo(x0 + third * speed, y0 + third * d0, x - third * speed, y - third * d1, x, y)
            previous = (x, y, value, slope_right, when)
        level = _soft_clip(model.knot(t)[0])[0]
        return path, QtCore.QPointF(TRACE_HEAD_X, TRACE_BASE_Y - TRACE_GAIN * level), level

    def _paint_trace(self, p: QtGui.QPainter, t: float) -> None:
        """Scrolling signal: new signal is written at the head and drifts left."""
        theme = self.theme
        line, head, level = self.trace_path(t)

        # Soft fill under the curve.
        area = QtGui.QPainterPath(line)
        area.lineTo(TRACE_HEAD_X, TRACE_BASE_Y + 12)
        area.lineTo(line.elementAt(0).x, TRACE_BASE_Y + 12)
        area.closeSubpath()
        fill = QtGui.QLinearGradient(0, TRACE_BASE_Y - TRACE_GAIN * 1.2, 0, TRACE_BASE_Y + 12)
        fill.setColorAt(0.0, _rgba(theme.deep, 58))
        fill.setColorAt(0.6, _rgba(theme.deep, 16))
        fill.setColorAt(1.0, _rgba(theme.deep, 0))
        p.fillPath(area, fill)

        # Glow: a stack of real strokes, wide and faint to narrow and bright,
        # added as light; then the crisp core line.
        p.setBrush(QtCore.Qt.BrushStyle.NoBrush)
        core = _mix(theme.light, theme.hot, 0.45)
        for width, alpha, is_core in TRACE_LAYERS:
            additive = not is_core
            p.setCompositionMode(QtGui.QPainter.CompositionMode.CompositionMode_Plus if additive
                                 else QtGui.QPainter.CompositionMode.CompositionMode_SourceOver)
            p.setPen(_round_pen(_rgba(core if is_core else theme.mid, alpha), width))
            p.drawPath(line)
        p.setCompositionMode(QtGui.QPainter.CompositionMode.CompositionMode_SourceOver)
        p.drawPixmap(QtCore.QPointF(0, self.band_top), self.fade)

        # Writing head: brighter while a transient is being written.
        level = max(0.0, min(1.0, level))
        radius = 14.0 + 11.0 * level
        bloom = QtGui.QRadialGradient(head, radius)
        bloom.setColorAt(0.0, _rgba(_mix(theme.light, theme.hot, 0.5), 120 + 115 * level))
        bloom.setColorAt(0.22, _rgba(theme.mid, 70 + 70 * level))
        bloom.setColorAt(0.55, _rgba(theme.deep, 20 + 26 * level))
        bloom.setColorAt(1.0, _rgba(theme.deep, 0))
        p.setCompositionMode(QtGui.QPainter.CompositionMode.CompositionMode_Plus)
        p.setPen(QtCore.Qt.PenStyle.NoPen)
        p.setBrush(bloom)
        p.drawEllipse(head, radius, radius)
        p.setCompositionMode(QtGui.QPainter.CompositionMode.CompositionMode_SourceOver)
        p.setBrush(_rgba(theme.hot, 255))
        p.drawEllipse(head, 2.6, 2.6)

    def _fiber_point(self, s: float) -> QtCore.QPointF:
        position = max(0.0, min(1.0, s)) * FIBER_SAMPLES
        index = min(FIBER_SAMPLES - 1, int(position))
        a, b = self.fiber_points[index], self.fiber_points[index + 1]
        w = position - index
        return QtCore.QPointF(a.x() + (b.x() - a.x()) * w, a.y() + (b.y() - a.y()) * w)

    def _bloom(self, p: QtGui.QPainter, center: QtCore.QPointF, radius: float,
               inner: Sequence[int], outer: Sequence[int], strength: float) -> None:
        if strength <= 0.01:
            return
        gradient = QtGui.QRadialGradient(center, radius)
        gradient.setColorAt(0.0, _rgba(inner, 215 * strength))
        gradient.setColorAt(0.18, _rgba(_mix(inner, outer, 0.6), 150 * strength))
        gradient.setColorAt(0.5, _rgba(outer, 46 * strength))
        gradient.setColorAt(1.0, _rgba(outer, 0))
        p.setPen(QtCore.Qt.PenStyle.NoPen)
        p.setBrush(gradient)
        p.drawEllipse(center, radius, radius)

    def _pulse_gradients(self, head: float, amp: float) -> List[QtGui.QLinearGradient]:
        """Brightness along the fiber for a comet whose head is at ``head``.

        One vertical gradient per stroke layer. Stops are placed at the
        heights of fiber samples, so brightness follows distance along the
        fiber even though the gradient itself is vertical.
        """
        theme = self.theme
        gradients = [QtGui.QLinearGradient(0, self.fiber_bottom, 0, self.fiber_top) for _ in PULSE_LAYERS]
        first = max(0, int(math.floor((head - PULSE_TAIL) * FIBER_SAMPLES)) - 1)
        last = min(FIBER_SAMPLES, int(math.ceil((head + PULSE_LEAD) * FIBER_SAMPLES)) + 1)
        stride = max(1, (last - first) // 28)
        indices = list(range(first, last + 1, stride))
        if indices and indices[-1] != last:
            indices.append(last)
        for gradient in gradients:
            gradient.setColorAt(0.0, QtGui.QColor(0, 0, 0, 0))
            gradient.setColorAt(1.0, QtGui.QColor(0, 0, 0, 0))
        for index in indices:
            s = index / FIBER_SAMPLES
            behind = head - s
            if behind < -PULSE_LEAD or behind > PULSE_TAIL:
                strength = 0.0
            elif behind < 0.0:
                strength = amp * (1.0 + behind / PULSE_LEAD) ** 2
            else:
                strength = amp * (1.0 - behind / PULSE_TAIL) ** 1.8
            body = fiber_color(s, theme)
            hot = _mix(body, theme.hot, 0.9 * (1.0 - abs(behind) / 0.14))
            position = min(1.0, max(0.0, self.fiber_heights[index]))
            for gradient, (_width, alpha, is_core) in zip(gradients, PULSE_LAYERS):
                gradient.setColorAt(position, _rgba(hot if is_core else body, alpha * strength))
        return gradients

    def _paint_fiber(self, p: QtGui.QPainter, t: float) -> None:
        """Light climbing the fiber: one comet per transient, mouse to optics."""
        theme = self.theme
        scale = LOGO_H / 172.0
        points = self.fiber_points
        p.setCompositionMode(QtGui.QPainter.CompositionMode.CompositionMode_Plus)
        p.setBrush(QtCore.Qt.BrushStyle.NoBrush)

        # The fiber itself glows faintly and breathes.
        breath = 0.82 + 0.18 * math.sin(2.0 * math.pi * t / 2.6)
        for width, rgb, alpha in ((9.0, theme.deep, 16), (5.0, theme.deep, 24), (1.8, theme.mid, 40)):
            p.setPen(_round_pen(_rgba(rgb, alpha * breath), width * scale))
            p.drawPath(self.fiber_path)

        for arrival, amp in self.model.events_between(t - 1.2, t + FIBER_TRAVEL_S + 0.3):
            launch = arrival - FIBER_TRAVEL_S
            since_launch = t - launch
            head = since_launch / FIBER_TRAVEL_S      # 0 = implant, 1 = optics

            # Activity flash at the implant: swells just before the pulse leaves.
            if -0.20 < since_launch < 0.0:
                flash = amp * (1.0 + since_launch / 0.20) ** 2
            elif 0.0 <= since_launch < 0.7:
                flash = amp * math.exp(-since_launch / 0.17)
            else:
                flash = 0.0
            self._bloom(p, points[0], 22.0 * scale, _mix(theme.mid, theme.hot, 0.5), theme.deep, flash)

            # The comet: white-hot head, coloured body, long soft tail.
            if -PULSE_LEAD < head < 1.0 + PULSE_TAIL:
                p.setBrush(QtCore.Qt.BrushStyle.NoBrush)
                for gradient, (width, _alpha, _core) in zip(self._pulse_gradients(head, amp), PULSE_LAYERS):
                    p.setPen(_round_pen(QtGui.QBrush(gradient), width * scale))
                    p.drawPath(self.fiber_path)
                if 0.0 <= head <= 1.0:
                    self._bloom(p, self._fiber_point(head), 18.0 * scale, theme.hot,
                                fiber_color(head, theme), 0.9 * amp)

            # Arrival at the optics: a short pale flash, in step with the trace.
            if t >= arrival:
                self._bloom(p, points[-1], 26.0 * scale, theme.hot, theme.light,
                            amp * math.exp(-(t - arrival) / 0.26))
        p.setCompositionMode(QtGui.QPainter.CompositionMode.CompositionMode_SourceOver)

    def render(self, t: float = START_T) -> QtGui.QPixmap:
        """One complete frame as a pixmap (used for the static splash)."""
        pix = self._new_pixmap(CARD_W, CARD_H)
        p = QtGui.QPainter(pix)
        self.paint(p, t)
        p.end()
        return pix


def _screen_dpr() -> float:
    try:
        screen = QtGui.QGuiApplication.primaryScreen()
        return float(screen.devicePixelRatio()) if screen is not None else 1.0
    except Exception:
        return 1.0


def build_branded_splash(theme: Optional[int] = None) -> Optional[QtGui.QPixmap]:
    """Static splash frame (the first frame of the animation) in the given or a
    random colour theme. Falls back to the raw logo if anything fails."""
    try:
        index = pick_theme(remember=False) if theme is None else int(theme)
        return SplashArt(_screen_dpr(), index).render(START_T)
    except Exception:
        try:
            icon_path = splash_logo_path()
            if os.path.isfile(icon_path):
                return QtGui.QPixmap(icon_path)
        except Exception:
            pass
        return None


# ---------------------------------------------------------------------------
# Animated window (runs in the helper process)
# ---------------------------------------------------------------------------
class ParentWatch:
    """Tells whether the launching process is still alive."""

    def __init__(self, pid: int) -> None:
        self.pid = int(pid)
        self._handle = None
        if self.pid > 0 and sys.platform == "win32":
            import ctypes
            self._kernel32 = ctypes.windll.kernel32
            self._kernel32.OpenProcess.restype = ctypes.c_void_p
            self._kernel32.WaitForSingleObject.argtypes = [ctypes.c_void_p, ctypes.c_uint32]
            self._handle = self._kernel32.OpenProcess(0x00100000, False, self.pid)  # SYNCHRONIZE

    def alive(self) -> bool:
        if self.pid <= 0:
            return True            # nothing to watch (tests, previews)
        if sys.platform == "win32":
            if not self._handle:
                return False
            return self._kernel32.WaitForSingleObject(self._handle, 0) == 0x102   # WAIT_TIMEOUT
        try:
            os.kill(self.pid, 0)
            return True
        except OSError:
            return False


class AnimatedSplashWindow(QtWidgets.QWidget):
    """Frameless, always-on-top window that plays the splash animation."""

    def __init__(self, parent_pid: int = 0, stop_path: str = "", theme: int = 0) -> None:
        super().__init__(None, QtCore.Qt.WindowType.SplashScreen
                         | QtCore.Qt.WindowType.FramelessWindowHint
                         | QtCore.Qt.WindowType.WindowStaysOnTopHint
                         | QtCore.Qt.WindowType.WindowDoesNotAcceptFocus)
        self.setAttribute(QtCore.Qt.WidgetAttribute.WA_TranslucentBackground, True)
        self.setAttribute(QtCore.Qt.WidgetAttribute.WA_ShowWithoutActivating, True)
        self.setFixedSize(CARD_W, CARD_H)
        self.art = SplashArt(_screen_dpr(), theme)
        self.stop_path = stop_path
        self.watch = ParentWatch(parent_pid)
        self.clock = QtCore.QElapsedTimer()
        self.clock.start()
        self._closing_at: Optional[float] = None
        self._ticks = 0
        self.frames_painted = 0
        self.setWindowOpacity(0.0)
        self.timer = QtCore.QTimer(self)
        self.timer.setTimerType(QtCore.Qt.TimerType.PreciseTimer)
        self.timer.timeout.connect(self._tick)
        self.timer.start(FRAME_INTERVAL_MS)

    def elapsed(self) -> float:
        return self.clock.nsecsElapsed() / 1e9

    def _should_stop(self) -> bool:
        if self.elapsed() > HELPER_MAX_LIFETIME_S:
            return True
        if self.stop_path and os.path.exists(self.stop_path):
            return True
        return not self.watch.alive()

    def begin_close(self) -> None:
        if self._closing_at is None:
            self._closing_at = self.elapsed()

    def _tick(self) -> None:
        now = self.elapsed()
        self._ticks += 1
        if self._closing_at is None and self._ticks % 3 == 0 and self._should_stop():
            self.begin_close()
        if self._closing_at is None:
            self.setWindowOpacity(min(1.0, now / FADE_IN_S))
        else:
            remaining = 1.0 - (now - self._closing_at) / FADE_OUT_S
            if remaining <= 0.0:
                self.timer.stop()
                self._write_stats()
                self.close()
                QtWidgets.QApplication.quit()
                return
            self.setWindowOpacity(min(self.windowOpacity(), remaining))
        self.update()

    def paintEvent(self, _event) -> None:
        painter = QtGui.QPainter(self)
        self.art.paint(painter, START_T + self.elapsed())
        painter.end()
        self.frames_painted += 1

    def _write_stats(self) -> None:
        """Optional diagnostics: PYBER_SPLASH_STATS=<file> records frames and seconds."""
        target = os.environ.get("PYBER_SPLASH_STATS", "")
        if not target:
            return
        try:
            with open(target, "w", encoding="utf-8") as stream:
                stream.write(f"{self.frames_painted} {self.elapsed():.3f}\n")
        except OSError:
            pass

    def mousePressEvent(self, event) -> None:
        event.accept()      # a click never dismisses the splash


def run_helper(argv: Sequence[str]) -> int:
    """Entry point of the helper process: play the animation until told to stop.

    Arguments: ``--pid N`` (launching process), ``--stop PATH`` (file whose
    appearance ends the animation), ``--pos X,Y`` (top-left corner),
    ``--theme N`` (colour theme; random when absent).
    """
    options = {"--pid": "0", "--stop": "", "--pos": "", "--theme": ""}
    args = list(argv)
    for index, token in enumerate(args):
        if token in options and index + 1 < len(args):
            options[token] = args[index + 1]
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv[:1])
    try:
        from app_icon import install_application_icon, set_windows_app_id
        set_windows_app_id()
        install_application_icon(app)
    except Exception:
        pass
    try:
        pid = int(options["--pid"])
    except ValueError:
        pid = 0
    try:
        theme = int(options["--theme"])
    except ValueError:
        theme = pick_theme()
    window = AnimatedSplashWindow(pid, options["--stop"], theme)
    try:
        x, y = (int(float(v)) for v in options["--pos"].split(","))
        window.move(x, y)
    except ValueError:
        screen = app.primaryScreen()
        if screen is not None:
            window.move(screen.availableGeometry().center() - window.rect().center())
    window.show()
    status = app.exec()
    try:
        if options["--stop"] and os.path.exists(options["--stop"]):
            os.remove(options["--stop"])
    except OSError:
        pass
    return int(status)


# ---------------------------------------------------------------------------
# Launching side: static splash + helper process
# ---------------------------------------------------------------------------
class StartupSplash(QtWidgets.QSplashScreen):
    """Static first frame; owns the helper process that animates on top of it."""

    def __init__(self, pixmap: QtGui.QPixmap, theme: int = 0) -> None:
        super().__init__(pixmap, QtCore.Qt.WindowType.WindowStaysOnTopHint)
        self.setAttribute(QtCore.Qt.WidgetAttribute.WA_TranslucentBackground, True)
        self.theme = int(theme)
        self._helper = None
        self._stop_path = ""

    def start_animation(self) -> bool:
        """Start the helper process over this splash. False if it could not start."""
        if self._helper is not None:
            return True
        try:
            import subprocess
            import tempfile
            self._stop_path = os.path.join(tempfile.gettempdir(), f"pyber-splash-{os.getpid()}.stop")
            if os.path.exists(self._stop_path):
                os.remove(self._stop_path)
            corner = self.geometry().topLeft()
            arguments = ["--pid", str(os.getpid()), "--stop", self._stop_path,
                         "--pos", f"{corner.x()},{corner.y()}", "--theme", str(self.theme)]
            if getattr(sys, "frozen", False):
                command = [sys.executable, HELPER_FLAG, *arguments]
            else:
                script = os.path.join(os.path.dirname(os.path.abspath(__file__)), "splash_helper.py")
                command = [sys.executable, script, HELPER_FLAG, *arguments]
            flags = 0x08000000 if sys.platform == "win32" else 0     # CREATE_NO_WINDOW
            self._helper = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                                            stderr=subprocess.DEVNULL, creationflags=flags, close_fds=True)
            import atexit
            atexit.register(self.stop_animation, True)
            return True
        except Exception:
            self._helper = None
            return False

    def stop_animation(self, wait: bool = False) -> None:
        """Ask the helper to fade out; optionally wait until it has left."""
        helper, self._helper = self._helper, None
        if helper is None:
            return
        try:
            with open(self._stop_path, "w", encoding="utf-8"):
                pass
        except OSError:
            try:
                helper.terminate()
            except Exception:
                pass
        if wait:
            try:
                helper.wait(timeout=2.0)
            except Exception:
                try:
                    helper.kill()
                except Exception:
                    pass
            try:
                if os.path.exists(self._stop_path):
                    os.remove(self._stop_path)
            except OSError:
                pass

    def finish(self, window) -> None:
        self.stop_animation()
        super().finish(window)

    def closeEvent(self, event) -> None:
        self.stop_animation()
        super().closeEvent(event)


def animation_allowed(app: QtWidgets.QApplication) -> bool:
    """The helper is skipped for automated runs and non-desktop platforms."""
    def enabled(name: str) -> bool:
        return str(os.environ.get(name, "")).strip().lower() not in {"", "0", "false", "no", "off"}

    if enabled("PYBER_NO_SPLASH_ANIMATION") or enabled("PYBER_SMOKE_TEST"):
        return False
    if str(os.environ.get("PYBER_UI_AUDIT", "")).strip():
        return False
    return app.platformName() in {"windows", "cocoa", "xcb"}


def _set_windows_app_user_model_id_early() -> None:
    """Share the identity used by normal and IDE launches."""
    from app_icon import set_windows_app_id
    set_windows_app_id()


def show_early_splash() -> Optional[QtWidgets.QSplashScreen]:
    """Create the QApplication and show the splash immediately.

    Called from main.py before its heavy imports run. Returns the splash (or
    None); the QApplication is reachable via QApplication.instance().
    """
    try:
        _set_windows_app_user_model_id_early()
        app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv[:1])
        # Windows may cache the first window's application icon. Install the
        # application default before creating even the startup splash.
        from app_icon import install_application_icon
        install_application_icon(app)
        theme = pick_theme()
        pix = build_branded_splash(theme)
        if pix is None or pix.isNull():
            return None
        splash = StartupSplash(pix, theme)
        splash.setWindowIcon(app.windowIcon())
        splash.show()
        app.processEvents(QtCore.QEventLoop.ProcessEventsFlag.AllEvents)
        if animation_allowed(app):
            splash.start_animation()
        return splash
    except Exception:
        return None
