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
        speed = splash.TRACE_HEAD_X / splash.TRACE_WINDOW_S
        worst = 0.0
        for seed in (splash.MODEL_SEED, 3, 4):            # includes sharp-rise transients
            art = splash.SplashArt(1.0, 0, seed, 8.0)
            for frame in range(0, 720, 17):
                t = splash.START_T + frame / 60.0
                path, head, _level = art.trace_path(t)
                self.assertAlmostEqual(head.x(), splash.TRACE_HEAD_X)
                for step in range(1, 200):
                    point = path.pointAtPercent(step / 200.0)
                    when = t - (splash.TRACE_HEAD_X - point.x()) / speed
                    value, slope = splash._soft_clip(*art.model.value_slope(when))
                    exact = splash.TRACE_BASE_Y - splash.TRACE_GAIN * value
                    steepness = splash.TRACE_GAIN * slope / speed
                    # Distance from the true curve, measured across it (a near-vertical
                    # rise must not turn a hair's width into a large vertical error).
                    worst = max(worst, abs(exact - point.y()) / (1.0 + steepness ** 2) ** 0.5)
        self.assertLess(worst, 0.25)

    def test_some_transients_have_a_much_sharper_rise(self):
        model = splash.SignalModel(7, 8.0)
        events = model.events_between(0, 400)
        rises = model.rise_times_between(0, 400)
        self.assertEqual(len(events), len(rises))
        share = sum(r == splash.SHARP_TAU_RISE_S for r in rises) / len(rises)
        self.assertGreater(share, 0.12)                   # from time to time...
        self.assertLess(share, 0.40)                      # ...not most of them

        def rise_10_90(arrival, tau):
            lone = splash.SignalModel(1, 8.0)             # one isolated transient
            lone._times, lone._amps, lone._rises, lone._next = [arrival], [1.0], [tau], 1e9
            base = lone.value(arrival - 1e-6)
            samples = [(k / 5000.0, lone.value(arrival + k / 5000.0) - base) for k in range(1, 1500)]
            peak = max(v for _d, v in samples)
            t10 = next(d for d, v in samples if v >= 0.1 * peak)
            t90 = next(d for d, v in samples if v >= 0.9 * peak)
            return t90 - t10

        normal = rise_10_90(50.0, splash.TAU_RISE_S)
        sharp = rise_10_90(50.0, splash.SHARP_TAU_RISE_S)
        self.assertLess(sharp, 0.35 * normal)
        self.assertLess(sharp, 0.04)                      # tens of milliseconds

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

    def test_transient_sequence_is_random_per_launch_and_builds_up(self):
        """A new seed gives a new sequence; transients grow and quicken as loading ends."""
        seeds = {splash.new_seed() for _ in range(20)}
        self.assertGreater(len(seeds), 15)
        first = splash.SignalModel(101, 8.0).events_between(0, 40)
        self.assertEqual(first, splash.SignalModel(101, 8.0).events_between(0, 40))
        self.assertNotEqual(first, splash.SignalModel(202, 8.0).events_between(0, 40))
        start = splash.START_T
        for expected in (4.0, 8.0, 20.0):
            early, late, early_gaps, late_gaps = [], [], [], []
            for seed in range(1, 41):
                model = splash.SignalModel(seed, expected)
                before = model.events_between(start - 6.0, start + 0.15 * expected)
                after = model.events_between(start + 0.9 * expected, start + 0.9 * expected + 8.0)
                early += [amp for _t, amp in before]
                late += [amp for _t, amp in after]
                early_gaps.append(len(before) / (6.0 + 0.15 * expected))
                late_gaps.append(len(after) / 8.0)
            mean = lambda values: sum(values) / len(values)
            self.assertGreater(mean(late), 2.2 * mean(early), expected)       # bigger spikes
            self.assertGreater(max(late), 1.3)
            self.assertLessEqual(max(late), splash.TRACE_CEILING + 1e-9)     # stacked peaks stay in the band
            self.assertLess(max(early), 0.75)
            self.assertGreater(mean(late_gaps), 1.3 * mean(early_gaps), expected)   # and more of them
        self.assertEqual(splash.SignalModel(5, 8.0).progress(start - 1), 0.0)
        self.assertEqual(splash.SignalModel(5, 8.0).progress(start + 100), 1.0)

    def test_loading_time_is_remembered_within_sane_limits(self):
        with tempfile.TemporaryDirectory() as directory:
            original = splash._theme_memory_path
            splash._theme_memory_path = lambda: str(Path(directory) / "theme.txt")
            try:
                self.assertEqual(splash.expected_load_seconds(), splash.DEFAULT_LOAD_S)
                splash.remember_load_seconds(12.34)
                self.assertAlmostEqual(splash.expected_load_seconds(), 12.34)
                splash.remember_load_seconds(0.2)                 # absurdly short: clamped
                self.assertEqual(splash.expected_load_seconds(), splash.LOAD_S_LIMITS[0])
                splash.remember_load_seconds(5000)
                self.assertEqual(splash.expected_load_seconds(), splash.LOAD_S_LIMITS[1])
            finally:
                splash._theme_memory_path = original

    def test_logo_dashes_light_up_with_each_arrival_and_disappear_between(self):
        """The three dashes are drawn live: off before a pulse arrives, lit one
        after the other at the arrival, gone again once the transient has passed."""
        art = splash.SplashArt(2.0, 1, 4, 8.0)
        model = art.model
        events = model.events_between(splash.START_T - 8, splash.START_T + 1)
        # An isolated transient: nothing within 1.6 s before it.
        arrival = next(b for (a, _), (b, _) in zip(events, events[1:]) if b - a > 1.6)
        before = art.spark_levels(arrival - 0.01)
        self.assertTrue(all(level < 0.02 for level in before))
        first, second, third = art.spark_levels(arrival + 0.04)
        self.assertGreater(first, 0.25)
        self.assertGreater(first, second)                 # they appear one after the other
        self.assertEqual(third, 0.0)
        self.assertTrue(all(level > 0.25 for level in art.spark_levels(arrival + 0.26)))
        # The static layer no longer contains the dashes (they would never switch off).
        x1, y1 = splash.SPARK_LOGO_LINES[0][0]
        x2, y2 = splash.SPARK_LOGO_LINES[0][1]
        point = art._logo_point((x1 + x2) / 2.0, (y1 + y2) / 2.0)
        still = art.static.toImage().pixelColor(int(point.x() * 2), int(point.y() * 2))
        self.assertLess(still.blue(), 110)                # dark card, not the pale dash
        lit = _frame(art, arrival + 0.06).pixelColor(int(point.x() * 2), int(point.y() * 2))
        self.assertGreater(lit.blue() + lit.red() + lit.green(), still.blue() + still.red() + still.green() + 150)

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


class AnimatedReadmeLogoTests(unittest.TestCase):
    """The README logo is generated from the editable logo and stays in sync with it."""

    @classmethod
    def setUpClass(cls):
        cls.app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
        root = Path(__file__).resolve().parents[1]
        sys.path.insert(0, str(root / "scripts"))
        import build_animated_logo
        cls.builder = build_animated_logo
        cls.root = root

    def test_committed_asset_matches_the_generator_and_is_used_by_the_readme(self):
        with tempfile.TemporaryDirectory() as directory:
            fresh = self.builder.build_animated_logo(Path(directory) / "logo.svg").read_text(encoding="utf-8")
        committed = (self.root / "assets" / "pyBer_logo_animated.svg").read_text(encoding="utf-8")
        self.assertEqual(fresh.split(), committed.split())      # same content, any line endings
        self.assertIn("assets/pyBer_logo_animated.svg", (self.root / "README.md").read_text(encoding="utf-8"))

    def test_animation_has_pulses_sparks_and_a_still_fallback(self):
        text = (self.root / "assets" / "pyBer_logo_animated.svg").read_text(encoding="utf-8")
        for name in ("pulse-glow", "pulse-body", "pulse-core", "implant", "optics", "spark-1", "spark-2", "spark-3"):
            self.assertIn(f"@keyframes {name}", text)
            self.assertIn(f'class="{name}"', text)
        self.assertIn("prefers-reduced-motion", text)
        self.assertNotIn("<script", text)
        self.assertNotIn("M223,194L223,204M264,204L259,213M287,239L277,243", text)   # split into three sparks
        launches = [launch for launch, _strength in self.builder.EVENTS]
        gaps = [b - a for a, b in zip(launches, launches[1:])]
        self.assertGreaterEqual(len(launches), 4)                     # several transients per loop
        self.assertGreater(max(gaps) - min(gaps), 0.2)                # irregular rhythm
        self.assertGreater(min(gaps), self.builder.TRAVEL_S * 1.5)    # pulses never overlap
        last_spark_off = (launches[-1] + self.builder.TRAVEL_S + 2 * self.builder.SPARK_STAGGER_S
                          + self.builder.SPARK_OFF_S)
        self.assertLess(last_spark_off, self.builder.LOOP_S)          # the loop closes cleanly


class SplashHelperProcessTests(unittest.TestCase):
    """The helper process plays frames, stops on request and when its launcher dies."""

    def _start(self, directory, pid):
        stop = Path(directory) / "stop"
        stats = Path(directory) / "stats.txt"
        env = dict(os.environ, QT_QPA_PLATFORM="offscreen", PYBER_SPLASH_STATS=str(stats))
        process = subprocess.Popen([sys.executable, str(HELPER), splash.HELPER_FLAG, "--pid", str(pid),
                                    "--stop", str(stop), "--pos", "10,10", "--theme", "3", "--seed", "77", "--expect", "6"], env=env)
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
