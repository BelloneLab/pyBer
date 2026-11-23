"""Event-scale ("prominence normalized") output: numerics, event files and GUI wiring.

The synthetic recordings below have known spontaneous events, so every number
the normalization reports (event count, top-fraction mean, rates, scale) can
be checked against ground truth rather than against a previous run.
"""

from contextlib import ExitStack
import math
import os
from pathlib import Path
import sys
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("PYBER_SMOKE_TEST", "1")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "pyBer"))

import numpy as np
from scipy.signal import find_peaks

from analysis_core import ProcessingParams, ProcessedTrial, prominence_normalize
from event_scale_normalization import (
    PAPER_PRESET, EventTable, bouts_from_states, describe_scale, event_window_mask,
    normalize_by_event_scale, read_event_table,
)

FS = 100.0
TEST_DATA = Path(r"D:\Apps\pyBer_test_data")


def synthetic_recording(seed=0, duration=300.0, amplitudes=None, times=None, noise=0.0005, width=0.15):
    """dF/F-like trace: Gaussian noise plus Gaussian transients of known peak value."""
    rng = np.random.default_rng(seed)
    t = np.arange(0.0, duration, 1.0 / FS)
    y = rng.normal(0.0, noise, t.size)
    if times is None:
        times = np.arange(5.0, duration - 5.0, 5.0)
    if amplitudes is None:
        amplitudes = np.linspace(0.004, 0.04, len(times))
    for t0, a in zip(times, amplitudes):
        y += a * np.exp(-0.5 * ((t - t0) / width) ** 2)
    return t, y, np.asarray(times, float), np.asarray(amplitudes, float)


def params(**kw):
    base = dict(prominence_baseline_source="whole", prominence_percent_top=0.10)
    base.update(kw)
    return ProcessingParams.from_dict(base)


def legacy_scale(y, keep, top):
    """The pre-refactor algorithm (centre on median, all maxima, top fraction of prominences)."""
    med = float(np.nanmedian(y[keep]))
    d = y - med
    locs, props = find_peaks(d, prominence=0.0)
    proms = props["prominences"]
    inside = keep[locs]
    proms = proms[inside]
    n = int(np.ceil(proms.size * top))
    return float(np.mean(np.sort(proms)[::-1][:n])), n


class EventScaleNumericsTests(unittest.TestCase):
    """The scale must be the stated statistic of the stated events, nothing else."""

    def test_prominence_recipe_matches_legacy_algorithm(self):
        t, y, _, _ = synthetic_recording(seed=1)
        keep = t <= 150.0
        out, stats = normalize_by_event_scale(
            y, t, params(prominence_baseline_source="window", prominence_baseline_end_s=150.0), FS)
        scale, n = legacy_scale(y, keep, 0.10)
        self.assertEqual(stats["n_selected"], n)
        self.assertAlmostEqual(stats["scale"], scale, places=12)
        np.testing.assert_allclose(out, (y - np.median(y[keep])) / scale, rtol=0, atol=1e-12)

    def test_amplitude_recipe_recovers_known_top_events(self):
        # 58 known transients (0.4 % .. 4 % dF/F); with a 0.2 % floor and noise
        # sigma 0.05 % the top 10 % are the 6 largest known events.
        t, y, times, amps = synthetic_recording(seed=2)
        p = params(prominence_method="amplitude", prominence_min_amplitude=0.002,
                   prominence_min_distance_s=1.0)
        out, stats = normalize_by_event_scale(y, t, p, FS)
        self.assertEqual(stats["n_candidates"], times.size)
        expected = np.mean(np.sort(amps)[::-1][: math.ceil(times.size * 0.10)])
        self.assertAlmostEqual(stats["scale"], expected, delta=0.002 * expected + 0.0006)
        # Divide only: zero stays zero and the mean selected peak sits on 1.
        np.testing.assert_allclose(out, y / stats["scale"], atol=1e-12)
        self.assertAlmostEqual(float(np.mean(stats["selected_peak"])), 1.0, places=9)
        np.testing.assert_allclose(stats["selected_base"], 0.0)
        self.assertAlmostEqual(stats["candidate_rate_hz"], times.size / 300.0, delta=1e-3)

    def test_top_fraction_changes_scale_but_not_shape(self):
        t, y, _, _ = synthetic_recording(seed=3)
        out10, s10 = normalize_by_event_scale(y, t, params(prominence_method="amplitude",
                                                           prominence_min_amplitude=0.002), FS)
        out50, s50 = normalize_by_event_scale(y, t, params(prominence_method="amplitude",
                                                           prominence_min_amplitude=0.002,
                                                           prominence_percent_top=0.5), FS)
        self.assertGreater(s10["scale"], s50["scale"])
        # One global divisor: traces are exactly proportional.
        np.testing.assert_allclose(out10 * s10["scale"], out50 * s50["scale"], atol=1e-12)

    def test_noise_units_make_output_scale_free(self):
        t, y, _, _ = synthetic_recording(seed=4)
        p = params(prominence_method="amplitude", prominence_threshold_units="noise",
                   prominence_min_peak=3.0, prominence_min_amplitude=3.0)
        out1, s1 = normalize_by_event_scale(y, t, p, FS)
        out7, s7 = normalize_by_event_scale(7.0 * y, t, p, FS)
        self.assertEqual(s1["n_candidates"], s7["n_candidates"])
        np.testing.assert_allclose(out1, out7, atol=1e-9)
        # The same thresholds in absolute dF/F are NOT scale free.
        pa = params(prominence_method="amplitude", prominence_min_amplitude=0.01)
        _, a1 = normalize_by_event_scale(y, t, pa, FS)
        _, a7 = normalize_by_event_scale(7.0 * y, t, pa, FS)
        self.assertNotEqual(a1["n_candidates"], a7["n_candidates"])

    def test_floor_above_every_event_is_reported_not_silent(self):
        t, y, _, _ = synthetic_recording(seed=5)
        _, stats = normalize_by_event_scale(y, t, params(prominence_min_peak=4.0), FS)
        self.assertTrue(stats["scale_source"].endswith("floor_ignored"))
        self.assertTrue(any("floor was ignored" in w for w in stats["warnings"]))
        text, warn = describe_scale(stats)
        self.assertTrue(warn)
        self.assertIn("Warning:", text)

    def test_max_size_rejects_artifact_and_zero_means_no_limit(self):
        t, y, times, amps = synthetic_recording(seed=6)
        y = y + 0.5 * np.exp(-0.5 * ((t - 152.5) / 0.15) ** 2)  # one huge artifact
        p = dict(prominence_method="amplitude", prominence_min_amplitude=0.002)
        _, capped = normalize_by_event_scale(y, t, params(prominence_max_peak=0.1, **p), FS)
        _, free = normalize_by_event_scale(y, t, params(prominence_max_peak=0.0, **p), FS)
        self.assertLess(capped["scale"], 0.05)
        self.assertGreater(free["scale"], capped["scale"])

    def test_min_spacing_keeps_the_larger_event(self):
        t = np.arange(0.0, 60.0, 1.0 / FS)
        y = np.zeros_like(t)
        for t0, a in ((10.0, 0.01), (10.4, 0.02), (30.0, 0.015)):
            y += a * np.exp(-0.5 * ((t - t0) / 0.05) ** 2)
        p = params(prominence_method="amplitude", prominence_percent_top=1.0,
                   prominence_min_amplitude=0.002, prominence_min_distance_s=1.0)
        _, stats = normalize_by_event_scale(y, t, p, FS)
        np.testing.assert_allclose(np.sort(stats["selected_times"]), [10.4, 30.0], atol=0.011)

    def test_event_windows_exclude_or_include(self):
        t, y, times, _ = synthetic_recording(seed=7, noise=0.0002)  # 0.2 % floor = 10 sigma: no noise events
        onsets = np.array([51.0, 151.0])  # window edges (46, 91, 146, 191 s) fall between transients
        offsets = np.array([61.0, 161.0])
        # 1 s spacing: noise ripples riding on one transient are not separate events.
        p = dict(prominence_baseline_source="events", prominence_method="amplitude",
                 prominence_min_amplitude=0.002, prominence_percent_top=1.0, prominence_min_distance_s=1.0,
                 prominence_exclude_before_s=5.0, prominence_exclude_after_s=30.0)
        _, ex = normalize_by_event_scale(y, t, params(**p), FS, onsets, offsets)
        inside = event_window_mask(ex["candidate_times"], onsets, offsets, 5.0, 30.0)
        self.assertFalse(np.any(inside))
        _, inc = normalize_by_event_scale(y, t, params(prominence_event_role="include", **p), FS, onsets, offsets)
        self.assertTrue(np.all(event_window_mask(inc["candidate_times"], onsets, offsets, 5.0, 30.0)))
        expected_inside = np.sum(event_window_mask(times, onsets, offsets, 5.0, 30.0))
        self.assertEqual(inc["n_candidates"], expected_inside)
        self.assertEqual(ex["n_candidates"] + inc["n_candidates"], times.size)

    def test_paper_preset_values(self):
        self.assertEqual(PAPER_PRESET["prominence_method"], "amplitude")
        self.assertEqual(PAPER_PRESET["prominence_percent_top"], 0.10)
        self.assertEqual(PAPER_PRESET["prominence_min_amplitude"], 0.002)
        self.assertEqual((PAPER_PRESET["prominence_exclude_before_s"],
                          PAPER_PRESET["prominence_exclude_after_s"]), (5.0, 30.0))

    def test_missing_values_do_not_create_events(self):
        t, y, times, _ = synthetic_recording(seed=8)
        y = y.copy()
        y[(t > 99.0) & (t < 101.0)] = np.nan  # removes the transient at t=100
        p = params(prominence_method="amplitude", prominence_min_amplitude=0.002, prominence_percent_top=1.0,
                   prominence_min_distance_s=1.0)
        out, stats = normalize_by_event_scale(y, t, p, FS)
        self.assertEqual(stats["n_candidates"], times.size - 1)
        self.assertTrue(np.all(np.isnan(out[(t > 99.0) & (t < 101.0)])))

    def test_wrapper_keeps_legacy_interface(self):
        t, y, _, _ = synthetic_recording(seed=9)
        out, stats = prominence_normalize(y, t, np.array([100.0]), params(prominence_baseline_source="events"), FS)
        self.assertEqual(out.shape, y.shape)
        for key in ("mean_amplitude", "sem_amplitude", "n_peaks", "duration_s", "peak_times",
                    "peak_values", "baseline_intervals", "scale_source", "baseline_source"):
            self.assertIn(key, stats)

    def test_processing_pipeline_carries_statistics(self):
        from analysis_core import LoadedTrial, PhotometryProcessor

        t, y, _, _ = synthetic_recording(seed=10, duration=120.0)
        ref = 1.0 + 0.001 * np.sin(t / 7.0)
        sig = 2.0 * ref * (1.0 + y)
        trial = LoadedTrial(path="synthetic.doric", channel_id="AIN01", time=t, signal_465=sig,
                            reference_405=ref, sampling_rate=FS)
        p = params(output_mode="prominence normalized (motion corrected with fitted ref)",
                   prominence_method="amplitude", prominence_min_amplitude=0.002, lowpass_hz=20.0)
        processed = PhotometryProcessor().process_trial(trial, p)
        stats = processed.prominence_stats
        self.assertGreater(stats["n_candidates"], 10)
        self.assertIn("SE amplitude", processed.output_context)
        self.assertTrue(np.isfinite(stats["scale"]))


class EventFileTests(unittest.TestCase):
    """Event files are read with the postprocessing format rules."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="pyber-events-")
        self.dir = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def test_bouts_keep_boundary_runs(self):
        t = np.arange(10, dtype=float)
        on, off = bouts_from_states(t, [1, 1, 0, 0, 1, 1, 0, 0, 1, 1])
        np.testing.assert_array_equal(on, [0, 4, 8])
        np.testing.assert_array_equal(off, [2, 6, 9])

    def test_onset_table(self):
        path = self.dir / "onsets.csv"
        path.write_text("injection,cd1,death\n5.6,325,200\n15.6,340,210\n315,360,60\n", encoding="utf-8")
        table = read_event_table(str(path))
        self.assertEqual(table.format, "onset table")
        self.assertEqual(table.labels(), ["injection", "cd1", "death"])
        on, off = table.select("death")
        np.testing.assert_array_equal(on, [60, 200, 210])
        np.testing.assert_array_equal(off, on)
        self.assertEqual(table.count(), 9)

    def test_binary_states_with_time(self):
        path = self.dir / "states.csv"
        rows = ["time,rearing,grooming"] + [f"{i * 0.5},{int(4 <= i < 8)},{int(i >= 16)}" for i in range(20)]
        path.write_text("\n".join(rows) + "\n", encoding="utf-8")
        table = read_event_table(str(path))
        self.assertEqual(table.format, "binary states")
        on, off = table.select("rearing")
        np.testing.assert_allclose(on, [2.0])
        np.testing.assert_allclose(off, [4.0])
        on, off = table.select("grooming")  # still ongoing at the last sample
        np.testing.assert_allclose(on, [8.0])
        np.testing.assert_allclose(off, [9.5])

    def test_plain_time_list(self):
        # An increasing "time" column is a clock without states: a plain list.
        path = self.dir / "list.csv"
        path.write_text("time\n3.0\n12.5\n40\n", encoding="utf-8")
        table = read_event_table(str(path))
        self.assertEqual(table.format, "event list")
        np.testing.assert_allclose(table.select()[0], [3.0, 12.5, 40.0])
        # Unordered values form an onset column, exactly as in postprocessing.
        path.write_text("time\n12.5\n3.0\n40\n", encoding="utf-8")
        table = read_event_table(str(path))
        self.assertEqual(table.format, "onset table")
        np.testing.assert_allclose(table.select()[0], [3.0, 12.5, 40.0])

    def test_unusable_file_raises_readable_error(self):
        path = self.dir / "text.csv"
        path.write_text("name,comment\na,b\nc,d\n", encoding="utf-8")
        with self.assertRaises(ValueError):
            read_event_table(str(path))
        with self.assertRaises(ValueError):
            read_event_table(str(self.dir / "missing.csv"))

    def test_file_source_with_label_selection(self):
        path = self.dir / "onsets.csv"
        path.write_text("cue,reward\n100,140\n200,\n", encoding="utf-8")
        t, y, _, _ = synthetic_recording(seed=11)
        p = params(prominence_baseline_source="file", prominence_event_file_path=str(path),
                   prominence_event_label="cue", prominence_exclude_before_s=5.0,
                   prominence_exclude_after_s=30.0)
        _, stats = normalize_by_event_scale(y, t, p, FS)
        self.assertEqual(stats["baseline"]["n_events"], 2)
        self.assertAlmostEqual(stats["baseline_duration_s"], 300.0 - 2 * 35.0, delta=0.05)
        self.assertFalse(stats["warnings"])

    def test_events_off_the_recording_clock_are_flagged(self):
        path = self.dir / "late.csv"
        path.write_text("cue\n5000\n6000\n", encoding="utf-8")
        t, y, _, _ = synthetic_recording(seed=12)
        p = params(prominence_baseline_source="file", prominence_event_file_path=str(path))
        _, stats = normalize_by_event_scale(y, t, p, FS)
        self.assertTrue(any("recording clock" in w for w in stats["warnings"]))

    @unittest.skipUnless((TEST_DATA / "trial_0027_behavior_with_time.csv").is_file(), "test data absent")
    def test_real_binary_behavior_file(self):
        table = read_event_table(str(TEST_DATA / "trial_0027_behavior_with_time.csv"))
        self.assertEqual(table.format, "binary states")
        self.assertIn("social_contacts", table.labels())
        on, off = table.select("social_contacts")
        self.assertGreater(on.size, 0)
        self.assertTrue(np.all(off >= on))

    @unittest.skipUnless((TEST_DATA / "trial_0027_onsets.csv").is_file(), "test data absent")
    def test_real_onset_file(self):
        table = read_event_table(str(TEST_DATA / "trial_0027_onsets.csv"))
        self.assertEqual(table.format, "onset table")
        self.assertEqual(set(table.labels()), {"injection", "cd1", "death"})


class EventScaleGuiTests(unittest.TestCase):
    """Settings round-trip, label lists, summary text and the output-plot overlay."""

    @classmethod
    def setUpClass(cls):
        from PySide6 import QtWidgets

        cls.app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

    def setUp(self):
        from PySide6 import QtCore

        self.resources = ExitStack()
        temporary = self.resources.enter_context(tempfile.TemporaryDirectory(prefix="pyber-scale-gui-"))
        self.temporary = Path(temporary)

        class IsolatedSettings(QtCore.QSettings):
            def __init__(self, *_args, **_kwargs):
                super().__init__(str(Path(temporary) / "preferences.ini"), QtCore.QSettings.Format.IniFormat)

        self.resources.enter_context(patch.object(QtCore, "QSettings", IsolatedSettings))
        from gui_preprocessing import ParameterPanel, PlotDashboard

        self.panel = ParameterPanel()
        self.plots = PlotDashboard()

    def tearDown(self):
        for widget in (self.panel, self.plots):
            widget.close()
            widget.deleteLater()
        self.app.processEvents()
        self.resources.close()

    def test_new_parameters_round_trip(self):
        path = self.temporary / "events.csv"
        path.write_text("cue,reward\n10,20\n30,\n", encoding="utf-8")
        values = ProcessingParams().to_dict()
        values.update(PAPER_PRESET)
        values.update(output_mode="prominence normalized (motion corrected with fitted ref)",
                      prominence_baseline_source="file", prominence_event_file_path=str(path),
                      prominence_event_label="reward", prominence_min_distance_s=0.25,
                      prominence_event_role="include")
        self.panel.set_params(ProcessingParams.from_dict(values))
        got = self.panel.get_params()
        for key in list(PAPER_PRESET) + ["prominence_event_label", "prominence_min_distance_s",
                                         "prominence_event_role", "prominence_event_file_path"]:
            self.assertEqual(getattr(got, key), values[key], key)
        labels = [self.panel.combo_prominence_event_label.itemData(i)
                  for i in range(self.panel.combo_prominence_event_label.count())]
        self.assertEqual(labels, ["", "cue", "reward"])
        self.assertIn("onset table", self.panel.lbl_prominence_event_file.text())
        self.assertIn("dFF_fit / mean(top spontaneous-event amplitudes)", self.panel.ed_output_definition.text())

    def test_legacy_no_limit_and_units(self):
        values = ProcessingParams().to_dict()
        values.update(prominence_max_peak=1e6)  # legacy "no limit"
        self.panel.set_params(ProcessingParams.from_dict(values))
        self.assertEqual(self.panel.get_params().prominence_max_peak, 0.0)
        self.panel.combo_prominence_units.setCurrentIndex(self.panel.combo_prominence_units.findData("noise"))
        self.assertEqual(self.panel.spin_prominence_min.maximum(), 100.0)
        self.assertTrue(self.panel.spin_prominence_min.suffix().strip().endswith("sigma"))

    def test_preset_button_applies_paper_values(self):
        self.panel._apply_prominence_paper_preset()
        got = self.panel.get_params()
        for key, value in PAPER_PRESET.items():
            self.assertEqual(getattr(got, key), value, key)

    def test_summary_and_overlay_on_output_plot(self):
        t, y, _, _ = synthetic_recording(seed=13)
        p = params(prominence_method="amplitude", prominence_min_amplitude=0.002)
        out, stats = normalize_by_event_scale(y, t, p, FS)
        self.panel.set_prominence_summary(stats)
        self.assertIn("Events:", self.panel.lbl_prominence_status.text())
        processed = ProcessedTrial(
            path="x.doric", channel_id="AIN01", time=t, raw_signal=y + 1.0, raw_reference=np.ones_like(t),
            sig_f=y + 1.0, ref_f=np.ones_like(t), baseline_sig=np.ones_like(t), baseline_ref=np.ones_like(t),
            output=out, output_label="prominence normalized (motion corrected with fitted ref)",
            prominence_stats=stats,
        )
        self.plots.update_plots(processed)
        vb = self.plots.plot_out.getViewBox()
        self.assertIs(self.plots._prom_selected_stems.getViewBox(), vb)
        self.assertTrue(self.plots._prom_selected_stems.isVisible())
        self.assertTrue(self.plots._prom_candidate_marks.isVisible())
        self.assertTrue(self.plots._prom_unit_line.isVisible())
        x, yy = self.plots._prom_selected_dots.getData()
        np.testing.assert_allclose(np.sort(x), np.sort(stats["selected_times"]))
        self.assertAlmostEqual(float(np.mean(yy)), 1.0, places=9)
        self.assertEqual(len(self.plots._prom_baseline_regions), 1)
        # Overlay off: nothing drawn; other output modes: nothing drawn.
        stats_off = dict(stats, show_overlay=False)
        processed.prominence_stats = stats_off
        self.plots.update_plots(processed)
        self.assertFalse(self.plots._prom_selected_stems.isVisible())
        processed.prominence_stats = stats
        processed.output_label = "dFF (motion corrected with fitted ref)"
        self.plots.update_plots(processed)
        self.assertFalse(self.plots._prom_selected_dots.isVisible())


if __name__ == "__main__":
    unittest.main()
