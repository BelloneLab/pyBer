"""Build the animated README logo (assets/pyBer_logo_animated.svg) from the editable logo.

Run in the pyBer environment from an IDE or with: python scripts/build_animated_logo.py
(scripts/build_app_icon.py also calls it, so one command rebuilds every logo asset).

The animation is plain CSS inside the SVG (no script), which GitHub plays when
the file is shown as an image:

* a pulse of green light climbs the fiber from the mouse's head to the optics;
* the implant glows as the pulse leaves and the optics flash when it arrives;
* the three small dashes between the optics light up one after the other at
  each arrival and fade, in the rhythm of the transients.

One loop holds several transients of different strength at irregular
intervals. Viewers who ask for reduced motion get the still logo.
"""
from pathlib import Path
import os
import re

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
from PySide6 import QtCore, QtGui, QtSvg, QtWidgets

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "assets/pyBer_logo.svg"
TARGET = ROOT / "assets/pyBer_logo_animated.svg"

# ----------------------------------------------------------------- parameters
LOOP_S = 7.2                       # duration of one loop
# (time the pulse leaves the mouse, strength 0-1): irregular, like real transients
EVENTS = ((0.35, 0.70), (1.55, 1.00), (2.70, 0.55), (4.20, 1.00), (5.45, 0.85))
TRAVEL_S = 0.62                    # time for the head of the pulse to climb the fiber
# Fiber centre line from the implant up to the optics (logo coordinates).
FIBER = "M139,318C143,216 156,181 214,152C253,132 270,103 269,35"
FIBER_POINTS = ((139, 318), (143, 216), (156, 181), (214, 152), (253, 132), (270, 103), (269, 35))
# Comet layers: (name, length as a fraction of the fiber, stroke width, colour, opacity, blurred)
LAYERS = (("glow", 0.42, 30, "rgb(70,255,160)", 0.70, True),
          ("body", 0.27, 12, "rgb(140,255,190)", 1.0, False),
          ("core", 0.11, 5, "rgb(244,255,249)", 1.0, False))
# The three dashes ("sparks") between the optics.
SPARKS = (((223, 194), (223, 204)), ((264, 204), (259, 213)), ((287, 239), (277, 243)))
SPARK_STAGGER_S = 0.09             # delay between successive sparks
SPARK_ON_S, SPARK_HOLD_S, SPARK_OFF_S = 0.07, 0.30, 0.85   # appear, linger, gone (after arrival)
SPARK_COLOR, SPARK_LIT_COLOR = "rgb(152,199,238)", "rgb(190,255,220)"
DASH_ELEMENT = re.compile(r'[ \t]*<path d="M223,194L223,204M264,204L259,213M287,239L277,243"[^>]*/>\n')
HIGHLIGHT_ELEMENT = re.compile(r'([ \t]*<path d="M266,39C267,105[^>]*/>\n)')


def fiber_length() -> float:
    """Length of the fiber path, needed to size and move the dash that forms the pulse."""
    path = QtGui.QPainterPath(QtCore.QPointF(*FIBER_POINTS[0]))
    for first in range(1, len(FIBER_POINTS), 3):
        path.cubicTo(*(QtCore.QPointF(*FIBER_POINTS[first + k]) for k in range(3)))
    return path.length()


def percent(seconds: float) -> str:
    return f"{100.0 * seconds / LOOP_S:.3f}%"


def keyframes(name: str, frames) -> str:
    """CSS @keyframes from (time in s, declarations) pairs, sorted by time."""
    body = "".join(f"{percent(t)}{{{css}}}" for t, css in sorted(frames, key=lambda item: item[0]))
    return f"@keyframes {name}{{{body}}}"


def build_style(length: float) -> str:
    longest = max(layer[1] for layer in LAYERS) * length
    run_s = TRAVEL_S * (length + longest) / length      # until the longest tail has left the fiber
    rules = []
    for name, fraction, _width, _color, _opacity, _blur in LAYERS:
        dash = fraction * length
        start, end = dash, dash - (length + longest)    # dash hidden below the implant / above the optics
        frames = [(0.0, f"stroke-dashoffset:{start:.1f};opacity:0;animation-timing-function:step-end")]
        for launch, strength in EVENTS:
            frames.append((launch, f"stroke-dashoffset:{start:.1f};opacity:{strength:.2f};"
                                   "animation-timing-function:linear"))
            # step-end: hold "gone" until the next launch, then jump back unseen.
            frames.append((launch + run_s, f"stroke-dashoffset:{end:.1f};opacity:{strength:.2f};"
                                           "animation-timing-function:step-end"))
        frames.append((LOOP_S, f"stroke-dashoffset:{start:.1f};opacity:0"))
        rules.append(keyframes(f"pulse-{name}", frames))
        rules.append(f".pulse-{name}{{stroke-dasharray:{dash:.1f} {3 * length:.1f};"
                     f"stroke-dashoffset:{start:.1f};opacity:0;"
                     f"animation:pulse-{name} {LOOP_S}s linear infinite}}")

    implant, optics = [(0.0, "opacity:0")], [(0.0, "opacity:0")]
    for launch, strength in EVENTS:
        arrival = launch + TRAVEL_S
        implant += [(launch - 0.20, "opacity:0"), (launch, f"opacity:{strength:.2f}"), (launch + 0.45, "opacity:0")]
        optics += [(arrival - 0.04, "opacity:0"), (arrival + 0.05, f"opacity:{strength:.2f}"),
                   (arrival + 0.55, "opacity:0")]
    implant.append((LOOP_S, "opacity:0"))
    optics.append((LOOP_S, "opacity:0"))
    rules += [keyframes("implant", implant), keyframes("optics", optics),
              f".implant{{opacity:0;animation:implant {LOOP_S}s linear infinite}}",
              f".optics{{opacity:0;animation:optics {LOOP_S}s linear infinite}}"]

    for index in range(len(SPARKS)):
        frames = [(0.0, f"opacity:0;stroke:{SPARK_COLOR}")]
        for launch, strength in EVENTS:
            on = launch + TRAVEL_S + index * SPARK_STAGGER_S
            lit = min(1.0, 0.4 + 0.6 * strength)
            frames += [(on, f"opacity:0;stroke:{SPARK_LIT_COLOR}"),
                       (on + SPARK_ON_S, f"opacity:{lit:.2f};stroke:{SPARK_LIT_COLOR}"),
                       (on + SPARK_HOLD_S, f"opacity:{0.6 * lit:.2f};stroke:{SPARK_COLOR}"),
                       (on + SPARK_OFF_S, f"opacity:0;stroke:{SPARK_COLOR}")]
        frames.append((LOOP_S, f"opacity:0;stroke:{SPARK_COLOR}"))
        rules.append(keyframes(f"spark-{index + 1}", frames))
        rules.append(f".spark-{index + 1}{{animation:spark-{index + 1} {LOOP_S}s linear infinite}}")
    # Still logo for viewers who prefer reduced motion: dashes shown, no light.
    rules.append("@media (prefers-reduced-motion:reduce){.pulse-glow,.pulse-body,.pulse-core,.implant,.optics,"
                 ".spark-1,.spark-2,.spark-3{animation:none}}")
    return "\n        ".join(rules)


def build_animated_logo(target: Path = TARGET) -> Path:
    """Write the animated SVG (by default next to the source logo) and return its path."""
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])   # noqa: F841 (Qt needs it)
    source = SOURCE.read_text(encoding="utf-8")
    length = fiber_length()

    sparks = "".join(
        f'    <path class="spark-{index + 1}" d="M{x1},{y1}L{x2},{y2}" style="fill:none;stroke:{SPARK_COLOR};'
        f'stroke-width:12px;stroke-linecap:round;"/>\n'
        for index, ((x1, y1), (x2, y2)) in enumerate(SPARKS))
    animated, count = DASH_ELEMENT.subn(sparks, source)
    if count != 1:
        raise RuntimeError("The logo's three dashes were not found: update DASH_ELEMENT.")

    # Light layers sit right above the fiber, so the mouse still hides its lower end.
    light = ['    <g id="fiber-light" style="fill:none;stroke-linecap:round;">\n',
             '        <circle class="implant" cx="140" cy="302" r="34" style="fill:url(#_ImplantGlow);"/>\n']
    for name, _fraction, width, color, opacity, blurred in LAYERS:
        blur = ' filter="url(#_PulseBlur)"' if blurred else ""
        light.append(f'        <path class="pulse-{name}" d="{FIBER}"{blur} style="stroke:{color};'
                     f'stroke-width:{width}px;stroke-opacity:{opacity};"/>\n')
    light += ['        <circle class="optics" cx="269" cy="38" r="40" style="fill:url(#_OpticsGlow);"/>\n',
              "    </g>\n"]
    animated, count = HIGHLIGHT_ELEMENT.subn(lambda match: match.group(1) + "".join(light), animated)
    if count != 1:
        raise RuntimeError("The fiber highlight was not found: update HIGHLIGHT_ELEMENT.")

    extra_defs = (
        '        <filter id="_PulseBlur" x="-60%" y="-20%" width="220%" height="140%">'
        '<feGaussianBlur stdDeviation="5"/></filter>\n'
        '        <radialGradient id="_ImplantGlow"><stop offset="0" style="stop-color:rgb(190,255,220);stop-opacity:0.95"/>'
        '<stop offset="0.35" style="stop-color:rgb(60,240,150);stop-opacity:0.5"/>'
        '<stop offset="1" style="stop-color:rgb(60,240,150);stop-opacity:0"/></radialGradient>\n'
        '        <radialGradient id="_OpticsGlow"><stop offset="0" style="stop-color:rgb(244,255,249);stop-opacity:0.95"/>'
        '<stop offset="0.3" style="stop-color:rgb(150,255,215);stop-opacity:0.5"/>'
        '<stop offset="1" style="stop-color:rgb(150,255,215);stop-opacity:0"/></radialGradient>\n')
    animated = animated.replace("    <defs>\n", "    <defs>\n" + extra_defs, 1)
    style = f"    <style>\n        {build_style(length)}\n    </style>\n"
    animated, count = re.subn(r"(<svg[^>]*>\n)", lambda match: match.group(1) + style, animated, count=1)
    if count != 1 or "_PulseBlur" not in animated:
        raise RuntimeError("Unexpected logo structure: <svg> or <defs> not found.")
    target = Path(target)
    target.write_text(animated, encoding="utf-8", newline="\n")
    if not QtSvg.QSvgRenderer(str(target)).isValid():
        raise RuntimeError("The animated logo is not a valid SVG.")
    return target


if __name__ == "__main__":
    print(build_animated_logo())
