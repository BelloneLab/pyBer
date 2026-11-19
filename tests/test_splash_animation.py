"""Startup splash: deterministic signal, light/trace synchrony, helper lifecycle."""
import os
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "pyBer"))
from PySide6 import QtWidgets

import splash_screen as splash

HELPER = Path(splash.__file__).with_name("splash_helper.py")


def _frame(art, t):
    return art.render(t).toImage()


class SplashArtTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

    def test_signal_is_deterministic_with_several_transients_in_view(self):
        a, b = splash.SignalModel(), splash.SignalModel()
        self.assertEqual(a.events_between(0, 60), b.events_between(0, 60))
        for start in (5.0, 20.0, 40.0):
            visible = a.events_between(start, start + splash.TRACE_WINDOW_S)
            self.assertGreaterEqual(len(visible), 3)
        arrival, amp = a.events_between(12, 30)[0]
        before = a.value(arrival - 0.01)
        peak = max(a.value(arrival + k / 200.0) for k in range(120))
        self.assertGreater(peak - before, 0.6 * amp)      # the transient rises after its arrival

    def test_light_pulse_climbs_from_the_mouse_to_the_optics(self):
        """The fiber runs upward, and the logo region changes while a pulse travels."""
        art = splash.SplashArt(1.0)
        arrival, _amp = art.model.events_between(15, 30)[0]
        launch = arrival - splash.FIBER_TRAVEL_S
        bottom, top = art.fiber_points[0], art.fiber_points[-1]
        self.assertGreater(bottom.y(), top.y())
        self.assertAlmostEqual(art._fiber_point(0.0).y(), bottom.y())
        self.assertAlmostEqual(art._fiber_point(1.0).y(), top.y())
        region = (int(art.logo_left), splash.LOGO_TOP, splash.LOGO_H, splash.LOGO_H)
        early = _frame(art, launch + 0.15).copy(*region)
        late = _frame(art, launch + 0.50).copy(*region)
        self.assertNotEqual(early, late)

    def test_static_frame_is_the_first_animation_frame_and_respects_dpr(self):
        art = splash.SplashArt(2.0)
        pix = art.render(splash.START_T)
        self.assertEqual((pix.width(), pix.height()), (splash.CARD_W * 2, splash.CARD_H * 2))
        self.assertEqual(pix.devicePixelRatio(), 2.0)
        self.assertEqual(_frame(art, splash.START_T), _frame(splash.SplashArt(2.0), splash.START_T))
        self.assertNotEqual(_frame(art, splash.START_T), _frame(art, splash.START_T + 0.5))
        self.assertEqual(pix.toImage().pixelColor(0, 0).alpha(), 0)   # rounded card, clear corners

    def test_trace_curve_follows_the_signal_within_a_fraction_of_a_pixel(self):
        """Smooth cubic curves (no polyline corners) must still be the exact signal."""
        art = splash.SplashArt(1.0)
        speed = splash.TRACE_HEAD_X / splash.TRACE_WINDOW_S
        worst = 0.0
        for frame in range(0, 240, 11):
            t = splash.START_T + frame / 60.0
            path, head, _level = art.trace_path(t)
            self.assertAlmostEqual(head.x(), splash.TRACE_HEAD_X)
            for step in range(1, 200):
                point = path.pointAtPercent(step / 200.0)
                when = t - (splash.TRACE_HEAD_X - point.x()) / speed
                exact = splash.TRACE_BASE_Y - splash.TRACE_GAIN * splash._soft_clip(art.model.value(when))[0]
                worst = max(worst, abs(exact - point.y()))
        self.assertLess(worst, 0.25)

    def test_at_least_seven_distinct_colour_themes_recolour_light_and_trace(self):
        self.assertGreaterEqual(len(splash.THEMES), 7)
        self.assertEqual(len({theme.deep for theme in splash.THEMES}), len(splash.THEMES))
        names = {theme.name for theme in splash.THEMES}
        self.assertTrue({"emerald", "violet", "ruby", "pink"} <= names)
        frames = [_frame(splash.SplashArt(1.0, index), splash.START_T) for index in range(len(splash.THEMES))]
        for index in range(1, len(frames)):
            self.assertNotEqual(frames[0], frames[index])
        # The fiber of the logo itself takes the theme colour (sampled mid-fiber).
        art = splash.SplashArt(2.0, [t.name for t in splash.THEMES].index("ruby"))
        point = art.fiber_points[len(art.fiber_points) // 2]
        colour = art.static.toImage().pixelColor(int((point.x() + 1.2) * 2), int(point.y() * 2))   # beside the highlight
        self.assertGreater(colour.red(), colour.green() + 40)
        self.assertGreater(colour.red(), colour.blue() + 40)

    def test_theme_is_random_and_never_repeats_the_previous_launch(self):
        with tempfile.TemporaryDirectory() as directory:
            memory = str(Path(directory) / "theme.txt")
            original = splash._theme_memory_path
            splash._theme_memory_path = lambda: memory
            try:
                picks = [splash.pick_theme() for _ in range(300)]
            finally:
                splash._theme_memory_path = original
        self.assertEqual(set(picks), set(range(len(splash.THEMES))))
        self.assertTrue(all(a != b for a, b in zip(picks, picks[1:])))

    def test_animation_is_skipped_for_automated_runs(self):
        self.assertFalse(splash.animation_allowed(self.app))          # offscreen platform
        fake = type("App", (), {"platformName": lambda self: "windows"})()
        for name in ("PYBER_SMOKE_TEST", "PYBER_UI_AUDIT", "PYBER_NO_SPLASH_ANIMATION"):
            previous = os.environ.get(name)
            os.environ[name] = "1"
            try:
                self.assertFalse(splash.animation_allowed(fake), name)
            finally:
                if previous is None:
                    del os.environ[name]
                else:
                    os.environ[name] = previous


class SplashHelperProcessTests(unittest.TestCase):
    """The helper process plays frames, stops on request and when its launcher dies."""

    def _start(self, directory, pid):
        stop = Path(directory) / "stop"
        stats = Path(directory) / "stats.txt"
        env = dict(os.environ, QT_QPA_PLATFORM="offscreen", PYBER_SPLASH_STATS=str(stats))
        process = subprocess.Popen([sys.executable, str(HELPER), splash.HELPER_FLAG, "--pid", str(pid),
                                    "--stop", str(stop), "--pos", "10,10", "--theme", "3"], env=env)
        return process, stop, stats

    def test_helper_animates_then_stops_on_request(self):
        with tempfile.TemporaryDirectory() as directory:
            process, stop, stats = self._start(directory, os.getpid())
            try:
                time.sleep(2.5)
                self.assertIsNone(process.poll())                     # still animating
                stop.write_text("")
                self.assertEqual(process.wait(timeout=10), 0)
            finally:
                if process.poll() is None:
                    process.kill()
            frames, _seconds = stats.read_text().split()
            self.assertGreater(int(frames), 20)
            self.assertFalse(stop.exists())                           # helper removes its stop file

    def test_helper_leaves_when_its_launcher_is_gone(self):
        gone = subprocess.Popen([sys.executable, "-c", "pass"])
        gone.wait()
        with tempfile.TemporaryDirectory() as directory:
            process, _stop, _stats = self._start(directory, gone.pid)
            try:
                self.assertEqual(process.wait(timeout=15), 0)
            finally:
                if process.poll() is None:
                    process.kill()


if __name__ == "__main__":
    unittest.main()
