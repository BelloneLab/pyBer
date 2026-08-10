"""Tests for data-driven preprocessing recommendations."""
import csv
import os
import sys
import unittest

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "pyBer"))

import analysis_core as ac  # noqa: E402


def _corr(x, y):
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    m = np.isfinite(x) & np.isfinite(y)
    if np.sum(m) < 10:
        return np.nan
    return float(np.corrcoef(x[m], y[m])[0, 1])


def _mixed_drift_trial(fs=5.0, duration=900.0):
    t = np.arange(0.0, duration, 1.0 / fs)
    slow = 35.0 * np.sin(2.0 * np.pi * t / 700.0) + 18.0 * np.cos(2.0 * np.pi * t / 280.0)
    motion = 5.0 * np.sin(2.0 * np.pi * 0.18 * t)
    transients = 1.5 * np.exp(-0.5 * ((t - 610.0) / 3.5) ** 2)
    sig = 290.0 + slow + motion + transients
    ref = 75.0 + 0.28 * slow - 0.8 * motion
    return ac.LoadedTrial(
        path=r"C:\data\mixed_drift.csv",
        channel_id="CH1",
        time=t,
        signal_465=sig,
        reference_405=ref,
        sampling_rate=fs,
    )


class PreprocessingRecommendationTests(unittest.TestCase):
    def test_mixed_drift_recommends_band_limited_inverted_output(self):
        rec = ac.recommend_preprocessing_settings(_mixed_drift_trial())

        self.assertEqual(rec.params.output_mode, ac.BAND_LIMITED_INVERTED_ISO_MODE)
        self.assertFalse(rec.params.invert_polarity)
        self.assertGreater(rec.metrics["raw_corr_405_465"], 0.2)
        self.assertLess(rec.metrics["detrended_corr_405_465"], -0.25)
        self.assertIn("band-limited", rec.sections["output"])
        self.assertIn("polarity looks normal", rec.sections["filtering"])

    def test_band_limited_inverted_output_reduces_highpass_reference_correlation(self):
        trial = _mixed_drift_trial()
        params = ac.ProcessingParams()
        params.artifact_detection_enabled = False
        params.target_fs_hz = trial.sampling_rate
        params.lowpass_hz = 2.0
        params.baseline_max_iter = 20

        non_params = ac.ProcessingParams.from_dict(params.to_dict())
        non_params.output_mode = "dFF (non motion corrected)"
        band_params = ac.ProcessingParams.from_dict(params.to_dict())
        band_params.output_mode = ac.BAND_LIMITED_INVERTED_ISO_MODE
        band_params.band_limited_reference_window_s = 60.0

        processor = ac.PhotometryProcessor()
        non = processor.process_trial(trial, non_params, preview_mode=False)
        band = processor.process_trial(trial, band_params, preview_mode=False)
        dff_ref = ac.safe_divide(band.ref_f - band.baseline_ref, band.baseline_ref)
        ref_hp, _, _ = ac._rolling_median_highpass(dff_ref, band.fs_used, 60.0)
        non_hp, _, _ = ac._rolling_median_highpass(non.output, non.fs_used, 60.0)
        band_hp, _, _ = ac._rolling_median_highpass(band.output, band.fs_used, 60.0)

        self.assertLess(abs(_corr(band_hp, ref_hp)), abs(_corr(non_hp, ref_hp)))
        self.assertIn("Band-limited inverted 405", band.output_context)

    def test_advice_carries_headline_settings_and_why_per_panel(self):
        rec = ac.recommend_preprocessing_settings(_mixed_drift_trial())

        for key in ("artifacts", "filtering", "baseline", "output"):
            advice = rec.advice[key]
            self.assertTrue(advice.headline.strip(), f"{key} headline is empty")
            self.assertTrue(advice.settings, f"{key} has no settings to apply")
            self.assertTrue(advice.why, f"{key} has no why entries")
            for name, value in advice.settings:
                self.assertTrue(str(name).strip())
                self.assertTrue(str(value).strip())
            # The flat section text stays in sync for anything reading it.
            self.assertEqual(rec.sections[key], advice.as_text())
            self.assertIn(advice.headline.strip(), rec.sections[key])

    def test_no_reference_recommends_signal_only_dff(self):
        t = np.arange(0.0, 120.0, 0.1)
        sig = 100.0 + np.sin(2.0 * np.pi * 0.2 * t)
        ref = np.full_like(sig, np.nan)
        trial = ac.LoadedTrial(
            path=r"C:\data\no_ref.csv",
            channel_id="CH1",
            time=t,
            signal_465=sig,
            reference_405=ref,
            sampling_rate=10.0,
        )
        rec = ac.recommend_preprocessing_settings(trial)
        self.assertEqual(rec.params.output_mode, "dFF (non motion corrected)")
        self.assertFalse(rec.metrics["has_reference"])

    @unittest.skipUnless(
        os.path.isfile(r"I:\fiber_photometry\souris_57_5fps_160725\Fluorescence_souris_57_5fps_160725.csv"),
        "local I: fiber photometry sample is not available",
    )
    def test_local_souris57_sample_recommends_band_limited_inverted_output(self):
        path = r"I:\fiber_photometry\souris_57_5fps_160725\Fluorescence_souris_57_5fps_160725.csv"
        rows = []
        with open(path, newline="") as f:
            next(f)
            for row in csv.DictReader(f):
                rows.append((
                    float(row["TimeStamp"]) / 1000.0,
                    float(row["CH1-410"]),
                    float(row["CH1-470"]),
                ))
        arr = np.asarray(rows, float)
        trial = ac.LoadedTrial(
            path=path,
            channel_id="CH1",
            time=arr[:, 0],
            signal_465=arr[:, 2],
            reference_405=arr[:, 1],
            sampling_rate=1.0 / float(np.nanmedian(np.diff(arr[:, 0]))),
        )
        rec = ac.recommend_preprocessing_settings(trial)
        self.assertEqual(rec.params.output_mode, ac.BAND_LIMITED_INVERTED_ISO_MODE)


if __name__ == "__main__":
    unittest.main()
