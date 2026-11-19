"""Neurophotometrics (NPM) raw CSV import.

Fixtures reproduce the header and state layouts of real NPM recordings
(FP3002 LedState with two clocks, legacy Flags with a TTL bit, three-LED files
with blank header cells, per-LED split files, header-less event lists). Values
are synthetic so every expected number is known exactly.
"""
import os
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "pyBer"))

from analysis_core import PhotometryProcessor, ProcessingParams, load_rwd_csv  # noqa: E402
from npm_import import (  # noqa: E402
    NPM_EVENT_PULSE_S,
    is_npm_photometry_csv,
    load_npm_csv,
    npm_csv_role,
    npm_load_summary,
    npm_primary_for,
    region_color,
)

try:
    from main import _discover_preprocessing_data_files  # noqa: E402
    _MAIN_IMPORT_ERROR = None
except Exception as exc:  # pragma: no cover - GUI deps (PySide6) may be absent
    _MAIN_IMPORT_ERROR = exc


FRAME_DT = 0.025          # 40 frames/s in total, 20 Hz per LED for two LEDs
T_START = 100.0           # FP3002 hardware clock at the startup frame (s)
COMPUTER_OFFSET_S = 5000.0


def _write(path: str, lines) -> str:
    with open(path, "w", encoding="utf-8", newline="") as handle:
        handle.write("\n".join(lines) + "\n")
    return path


def _rising_edges(time: np.ndarray, trace: np.ndarray) -> np.ndarray:
    high = np.asarray(trace) > 0.5
    idx = np.where(high & ~np.r_[False, high[:-1]])[0]
    return np.asarray(time)[idx]


def _onsets(loaded, name):
    timebase = loaded.trigger_time_by_name[name]
    return _rising_edges(timebase, loaded.trigger_by_name[name])


def _fp3002_rows(n_frames: int = 401):
    """FP3002 rows: startup frame 7, then 470/415 alternating.

    470 nm frames carry a sine, 415 nm frames a linear ramp, so the
    interpolated reference must equal the ramp exactly at signal times.
    """
    rows = []
    for k in range(n_frames):
        t_sys = T_START + k * FRAME_DT
        jitter_ms = 3.0 if k % 2 else -3.0  # computer clock jitter
        t_comp_ms = (t_sys + COMPUTER_OFFSET_S) * 1000.0 + jitter_ms
        if k == 0:
            state, value = 7, 0.0039
        elif k % 2 == 1:
            state, value = 2, 1.0 + 0.1 * np.sin(2 * np.pi * 0.5 * t_sys)
        else:
            state, value = 1, 0.5 + 0.01 * t_sys
        rows.append((k, t_sys, state, t_comp_ms, value))
    return rows


class NpmFixtureMixin:
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="pyber_npm_")
        self.dir = self.tmp.name

    def tearDown(self):
        self.tmp.cleanup()

    def write_fp3002(self, folder=None, name="session_fp.csv", events=True):
        folder = folder or self.dir
        rows = _fp3002_rows()
        lines = ["FrameCounter,SystemTimestamp,LedState,ComputerTimestamp,G0"]
        lines += [f"{k},{t:.6f},{s},{c:.2f},{v:.9f}" for k, t, s, c, v in rows]
        path = _write(os.path.join(folder, name), lines)
        if events:
            # Header-less event list on the computer clock (milliseconds).
            ev = [(T_START + 2.0, "tone"), (T_START + 5.0, "shock"), (T_START + 7.0, "tone")]
            _write(
                os.path.join(folder, "session_stimuli.csv"),
                [f"{(t + COMPUTER_OFFSET_S) * 1000.0:.2f},{label}" for t, label in ev],
            )
        return path, rows


class Fp3002LedStateTests(NpmFixtureMixin, unittest.TestCase):
    def test_detection_and_roles(self):
        path, _ = self.write_fp3002()
        self.assertTrue(is_npm_photometry_csv(path))
        self.assertEqual(npm_csv_role(path), "photometry")
        self.assertEqual(npm_csv_role(os.path.join(self.dir, "session_stimuli.csv")), "event")

    def test_signal_is_untouched_and_reference_interpolated(self):
        path, rows = self.write_fp3002(events=False)
        loaded = load_npm_csv(path)
        self.assertIsNotNone(loaded)
        self.assertEqual(loaded.channels, ["G0"])

        md = loaded.metadata
        self.assertEqual(md["acquisition_system"], "Neurophotometrics")
        self.assertEqual(md["npm_time_column"], "SystemTimestamp")
        self.assertTrue(md["npm_startup_frame_dropped"])
        t0 = T_START + FRAME_DT  # first lit frame after the startup frame
        self.assertAlmostEqual(md["npm_time_origin_s"], t0, places=6)

        sig_rows = [(t, v) for _k, t, s, _c, v in rows if s == 2]
        ref_times = [t for _k, t, s, _c, _v in rows if s == 1]
        t_sig = np.array([t for t, _ in sig_rows]) - t0
        v_sig = np.array([v for _, v in sig_rows])
        # The first 470 frame (t = 0) precedes the first 415 frame: it is
        # trimmed instead of extrapolating the reference.
        inside = (t_sig >= ref_times[0] - t0 - 1e-12) & (t_sig <= ref_times[-1] - t0 + 1e-12)
        self.assertFalse(inside[0])
        self.assertEqual(int(np.sum(~inside)), 1)
        t = loaded.time_by_channel["G0"]
        np.testing.assert_allclose(t, t_sig[inside], atol=1e-9)
        self.assertAlmostEqual(float(t[0]), 2 * FRAME_DT, places=9)
        self.assertEqual(md["npm_channels"]["G0"]["trimmed_signal_samples"], 1)
        np.testing.assert_allclose(loaded.signal_by_channel["G0"], v_sig[inside], rtol=0, atol=1e-9)
        expected_ref = 0.5 + 0.01 * (t + t0)
        np.testing.assert_allclose(loaded.reference_by_channel["G0"], expected_ref, atol=1e-9)
        self.assertAlmostEqual(md["npm_channels"]["G0"]["fs_hz"], 20.0, places=6)

    def test_computer_clock_events_map_onto_hardware_time(self):
        path, _ = self.write_fp3002(events=True)
        loaded = load_npm_csv(path)
        t0 = T_START + FRAME_DT
        self.assertEqual(sorted(loaded.trigger_by_name), ["shock", "tone"])
        report = loaded.metadata["npm_event_files"][0]
        self.assertEqual(report["clock"], "ComputerTimestamp")
        self.assertEqual(report["n_in_recording"], 3)
        # Onsets: first signal sample at or after the event, i.e. within one 20 Hz period.
        tone = _onsets(loaded, "tone")
        shock = _onsets(loaded, "shock")
        np.testing.assert_allclose(tone, [T_START + 2.0 - t0, T_START + 7.0 - t0], atol=0.05 + 1e-6)
        np.testing.assert_allclose(shock, [T_START + 5.0 - t0], atol=0.05 + 1e-6)
        # Pulses are short and do not spill into neighbours.
        width = np.sum(loaded.trigger_by_name["shock"] > 0.5) / 20.0
        self.assertLessEqual(width, NPM_EVENT_PULSE_S + 0.05)
        self.assertIn("G0 (470/415 nm)", npm_load_summary(loaded))

    def test_make_trial_and_full_processing(self):
        path, _ = self.write_fp3002(events=True)
        loaded = load_npm_csv(path)
        trial = loaded.make_trial("G0", trigger_name="tone", trigger_names=["tone", "shock"])
        np.testing.assert_allclose(trial.time, loaded.time_by_channel["G0"])
        self.assertAlmostEqual(trial.sampling_rate, 20.0, places=6)
        processed = PhotometryProcessor().process_trial(trial, ProcessingParams(), preview_mode=False)
        self.assertIsNotNone(processed.output)
        self.assertTrue(np.isfinite(processed.output).sum() > 0.9 * processed.output.size)
        self.assertIn("tone", processed.triggers)
        self.assertEqual(_rising_edges(processed.time, processed.triggers["tone"]).size, 2)

    def test_event_file_on_unrelated_clock_is_not_attached(self):
        path, _ = self.write_fp3002(events=False)
        _write(os.path.join(self.dir, "other_session_ts.csv"), ["1.0,True", "2.0,False"])
        loaded = load_npm_csv(path)
        self.assertEqual(loaded.trigger_by_name, {})
        self.assertEqual(len(loaded.metadata["npm_unmatched_event_files"]), 1)


class LegacyFlagsTests(NpmFixtureMixin, unittest.TestCase):
    def write_legacy(self):
        lines = ["FrameCounter,Timestamp,Flags,Region0G,Region1G,Region2G"]
        ttl_on, ttl_off = T_START + 3.0, T_START + 6.0
        for k in range(601):
            t = T_START + k * FRAME_DT
            if k == 0:
                flags = 16  # unlit startup frame of FP3001 files
            else:
                flags = 18 if k % 2 == 1 else 17
                if ttl_on <= t < ttl_off:
                    flags += 256
            lines.append(f"{k},{t:.6f},{flags},{0.01 + 1e-4 * k:.6f},{0.02:.6f},{0.03 + 1e-5 * k:.6f}")
        path = _write(os.path.join(self.dir, "Fear_1442.csv"), lines)
        _write(os.path.join(self.dir, "Fear_1442_ts0.csv"), [f"{ttl_on:.6f},True", f"{ttl_off:.6f},False"])
        return path, ttl_on, ttl_off

    def test_ttl_bit_and_true_false_events_agree(self):
        path, ttl_on, ttl_off = self.write_legacy()
        loaded = load_npm_csv(path)
        self.assertEqual(loaded.channels, ["Region0G", "Region1G", "Region2G"])
        md = loaded.metadata
        self.assertEqual(md["npm_state_column"], "Flags")
        self.assertEqual(md["npm_unlit_frames_dropped"], 1)
        self.assertFalse(md["npm_startup_frame_dropped"])
        # Bit 4 (value 16) is constant in legacy files: not a trigger. Bit 8 (256) is the TTL.
        self.assertNotIn("Flags bit 4", loaded.trigger_by_name)
        self.assertIn("Flags bit 8", loaded.trigger_by_name)
        self.assertIn("Fear_1442_ts0", loaded.trigger_by_name)
        np.testing.assert_allclose(_onsets(loaded, "Flags bit 8"), _onsets(loaded, "Fear_1442_ts0"), atol=1e-9)
        t0 = md["npm_time_origin_s"]
        trace = loaded.trigger_by_name["Fear_1442_ts0"]
        tb = loaded.trigger_time_by_name["Fear_1442_ts0"]
        high_t = tb[trace > 0.5]
        self.assertGreaterEqual(high_t.min(), ttl_on - t0 - 1e-9)
        self.assertLess(high_t.max(), ttl_off - t0)


class ThreeLedAndSimultaneousTests(NpmFixtureMixin, unittest.TestCase):
    def test_three_led_cycle_with_blank_header_cells(self):
        # PhAT-style header: blank cells hold a frame index and copies of the state.
        lines = [",Timestamp,msTimestamp,,Region0R,Region1G,,,LedState"]
        lines.append(f",{T_START:.5f},{T_START * 1000:.2f},1,9.0,9.0,0,,0")
        cycle = (4, 1, 2)
        for k in range(1, 301):
            t = T_START + k * FRAME_DT
            led = cycle[(k - 1) % 3]
            red = {4: 300.0, 1: 100.0 + k, 2: 50.0}[led]
            green = {4: 20.0, 1: 200.0 + k, 2: 400.0 + 0.5 * k}[led]
            lines.append(f"{k - 1},{t:.5f},{t * 1000:.2f},{k},{red},{green},{led},{led},{led}")
        path = _write(os.path.join(self.dir, "Sample_NPM_1fiber.csv"), lines)
        loaded = load_npm_csv(path)
        self.assertEqual(loaded.channels, ["Region0R", "Region1G"])
        info = loaded.metadata["npm_channels"]
        self.assertEqual((info["Region0R"]["signal_nm"], info["Region0R"]["reference_nm"]), (560, 415))
        self.assertEqual((info["Region1G"]["signal_nm"], info["Region1G"]["reference_nm"]), (470, 415))
        np.testing.assert_allclose(loaded.signal_by_channel["Region0R"], 300.0)
        self.assertTrue(np.all(loaded.signal_by_channel["Region1G"] >= 400.0))
        # Reference ramps linearly with the frame index, so interpolation is exact.
        ref = loaded.reference_by_channel["Region1G"]
        t = loaded.time_by_channel["Region1G"]
        k_at_t = (t + loaded.metadata["npm_time_origin_s"] - T_START) / FRAME_DT
        np.testing.assert_allclose(ref, 200.0 + k_at_t, atol=1e-6)

    def test_simultaneous_470_560_frames_feed_green_and_red(self):
        lines = ["FrameCounter,SystemTimestamp,LedState,ComputerTimestamp,G0,R0"]
        for k in range(200):
            t = T_START + k * FRAME_DT
            state = 6 if k % 2 == 0 else 1
            g, r = (10.0, 20.0) if state == 6 else (1.0, 2.0)
            lines.append(f"{k},{t:.6f},{state},{t * 1000:.2f},{g},{r}")
        loaded = load_npm_csv(_write(os.path.join(self.dir, "dual_color.csv"), lines))
        self.assertEqual(loaded.channels, ["G0", "R0"])
        np.testing.assert_allclose(loaded.signal_by_channel["G0"], 10.0)
        np.testing.assert_allclose(loaded.signal_by_channel["R0"], 20.0)
        np.testing.assert_allclose(loaded.reference_by_channel["G0"], 1.0)
        np.testing.assert_allclose(loaded.reference_by_channel["R0"], 2.0)

    def test_signal_without_reference_keeps_nan_reference(self):
        lines = ["FrameCounter,SystemTimestamp,LedState,G0"]
        lines += [f"{k},{T_START + k * FRAME_DT:.6f},2,{1.0 + 0.001 * k}" for k in range(100)]
        loaded = load_npm_csv(_write(os.path.join(self.dir, "only470.csv"), lines))
        self.assertTrue(np.all(np.isnan(loaded.reference_by_channel["G0"])))
        self.assertIsNone(loaded.metadata["npm_channels"]["G0"]["reference_nm"])
        trial = loaded.make_trial("G0")
        self.assertEqual(trial.signal_465.size, trial.reference_405.size)

    def test_red_region_without_560_frames_is_skipped_with_reason(self):
        lines = ["FrameCounter,SystemTimestamp,LedState,Region0G,Region1R"]
        lines += [f"{k},{T_START + k * FRAME_DT:.6f},{2 if k % 2 else 1},1.0,2.0" for k in range(100)]
        loaded = load_npm_csv(_write(os.path.join(self.dir, "no560.csv"), lines))
        self.assertEqual(loaded.channels, ["Region0G"])
        self.assertIn("Region1R", loaded.metadata["npm_skipped_regions"])


class SplitFileTests(NpmFixtureMixin, unittest.TestCase):
    def write_split(self, folder):
        header = "FrameCounter,Timestamp,LedState,Stimulation,Output0,Output1,Input0,Input1,Region0G"
        by_led = {415: [header], 470: [header]}
        for k in range(301):
            t = T_START + k * FRAME_DT
            input0 = 1 if T_START + 2.0 <= t < T_START + 4.0 else 0
            if k == 0:
                for led in by_led:
                    by_led[led].append(f"0,{t:.6f},7,0,0,0,0,0,0.0039")
                continue
            led = 470 if k % 2 else 415
            state = 2 if led == 470 else 1
            value = 1.0 + 0.001 * k if led == 470 else 0.5 + 0.0005 * k
            by_led[led].append(f"{k},{t:.6f},{state},0,0,0,{input0},0,{value}")
        p415 = _write(os.path.join(folder, "FiberData415.csv"), by_led[415])
        p470 = _write(os.path.join(folder, "FiberData470.csv"), by_led[470])
        return p415, p470

    def test_split_files_merge_into_one_recording(self):
        p415, p470 = self.write_split(self.dir)
        self.assertEqual(npm_csv_role(p415), "split_companion")
        self.assertEqual(npm_csv_role(p470), "photometry")
        self.assertEqual(os.path.normcase(npm_primary_for(p415)), os.path.normcase(p470))
        loaded = load_npm_csv(p470)
        self.assertEqual(loaded.channels, ["Region0G"])
        self.assertEqual(len(loaded.metadata["npm_source_files"]), 2)
        self.assertTrue(np.all(np.isfinite(loaded.reference_by_channel["Region0G"])))
        # Constant digital columns are dropped; Input0 toggles and becomes a trigger.
        self.assertEqual(list(loaded.trigger_by_name), ["Input0"])
        t0 = loaded.metadata["npm_time_origin_s"]
        np.testing.assert_allclose(_onsets(loaded, "Input0"), [T_START + 2.0 - t0], atol=0.05 + 1e-6)

    def test_complete_recordings_named_like_wavelengths_are_not_merged(self):
        # Animal IDs that happen to be 415 / 470: both files hold both LEDs.
        a, _ = self.write_fp3002(name="mouse415.csv", events=False)
        b, _ = self.write_fp3002(name="mouse470.csv", events=False)
        self.assertEqual((npm_csv_role(a), npm_csv_role(b)), ("photometry", "photometry"))
        self.assertEqual(npm_primary_for(a), "")
        self.assertEqual(len(load_npm_csv(a).metadata["npm_source_files"]), 1)

    @unittest.skipIf(_MAIN_IMPORT_ERROR is not None, f"main import failed: {_MAIN_IMPORT_ERROR}")
    def test_folder_discovery_lists_only_photometry_tables(self):
        nested = os.path.join(self.dir, "mouse1", "day1")
        split = os.path.join(self.dir, "mouse2")
        os.makedirs(nested)
        os.makedirs(split)
        self.write_fp3002(folder=nested, name="bl72_fp.csv", events=True)
        self.write_split(split)
        found = _discover_preprocessing_data_files(self.dir, recursive=True)
        names = sorted(os.path.relpath(p, self.dir).replace("\\", "/") for p in found)
        self.assertEqual(names, ["mouse1/day1/bl72_fp.csv", "mouse2/FiberData470.csv"])

    def test_cli_discovery_and_loading(self):
        from cli import discover_raw_files, load_raw_file

        nested = os.path.join(self.dir, "a", "b")
        os.makedirs(nested)
        self.write_fp3002(folder=nested, name="bl72_fp.csv", events=True)
        p415, p470 = self.write_split(self.dir)
        found = discover_raw_files([self.dir])
        names = sorted(p.name for p in found)
        self.assertEqual(names, ["FiberData470.csv", "bl72_fp.csv"])
        # Naming the companion explicitly does not create a duplicate recording.
        self.assertEqual(discover_raw_files([p415]), [])
        loaded = load_raw_file(Path(p470))
        self.assertEqual(loaded.channels, ["Region0G"])


class NonNpmFilesTests(NpmFixtureMixin, unittest.TestCase):
    def test_other_csv_layouts_are_left_to_their_readers(self):
        rwd = _write(os.path.join(self.dir, "Fluorescence.csv"), [
            "TimeStamp,Events,CH1-410,CH1-470",
            "0,,10,20", "25,,11,21", "50,,12,22",
        ])
        generic = _write(os.path.join(self.dir, "manual.csv"), ["time,raw,ref", "0,1,2", "1,1,2"])
        headerless = _write(os.path.join(self.dir, "bonsai_plain.csv"), [
            "40263510.4768,6558.5,4429.4,7018.7",
            "40263535.68,4822.6,3898.8,6989.9",
        ])
        for path in (rwd, generic, headerless):
            self.assertFalse(is_npm_photometry_csv(path), path)
            self.assertIsNone(load_npm_csv(path))
            self.assertEqual(npm_csv_role(path), "")  # no NPM photometry in this folder
        self.assertIsNotNone(load_rwd_csv(rwd))

    def test_region_colour_names(self):
        self.assertEqual(region_color("Region0G"), "G")
        self.assertEqual(region_color("Region12R"), "R")
        self.assertEqual(region_color("G3"), "G")
        self.assertEqual(region_color("r0"), "R")
        self.assertEqual(region_color("Channel1G"), "G")
        self.assertEqual(region_color("NAc green"), "G")
        self.assertEqual(region_color("ROI"), "")


if __name__ == "__main__":
    unittest.main()
