"""Tests for the inverted isobestic fitted-reference output mode."""
import os
import sys
import unittest

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "pyBer"))

import analysis_core as ac  # noqa: E402


INVERTED_FIT_MODE = "dFF (motion corrected with inverted isobestic fit)"


class InvertedIsobesticFitTests(unittest.TestCase):
    def test_helper_fits_negative_reference_before_dff(self):
        params = ac.ProcessingParams()
        params.reference_fit = "OLS (recommended)"

        t = np.linspace(0.0, 12.0, 300)
        artifact = np.sin(2.0 * np.pi * 0.35 * t)
        neural = 0.25 * np.sin(2.0 * np.pi * 3.0 * t)
        sig = 200.0 + 6.0 * artifact + neural
        ref = 80.0 - 3.0 * artifact

        normal_out, normal_fit, normal_slope, _ = ac._compute_fitted_reference_dff(
            ref,
            sig,
            params,
        )
        out, fitted_ref, slope, intercept = ac._compute_fitted_reference_dff(
            ref,
            sig,
            params,
            invert_reference=True,
        )
        manual_slope, manual_intercept = ac.fit_reference_to_signal(
            -ref,
            sig,
            params,
            nonnegative_slope=True,
        )
        manual_fit = manual_slope * (-ref) + manual_intercept
        manual_out = ac.safe_divide(sig - manual_fit, manual_fit)

        self.assertEqual(normal_slope, 0.0)
        self.assertGreater(np.nanstd(fitted_ref), np.nanstd(normal_fit))
        self.assertGreater(slope, 0.0)
        self.assertAlmostEqual(intercept, manual_intercept)
        self.assertGreater(np.nanmean(np.abs(out - normal_out)), 1e-4)
        np.testing.assert_allclose(fitted_ref, manual_fit)
        np.testing.assert_allclose(out, manual_out)

    def test_process_trial_uses_inverted_mode_as_real_correction(self):
        fs = 40.0
        t = np.arange(0.0, 12.0, 1.0 / fs)
        artifact = np.sin(2.0 * np.pi * 0.4 * t)
        neural = 0.35 * np.exp(-0.5 * ((t - 6.0) / 0.35) ** 2)
        sig = 200.0 + 5.0 * artifact + neural
        ref = 80.0 - 2.5 * artifact
        trigger = (t > 5.8).astype(float)

        trial = ac.LoadedTrial(
            path=r"C:\data\inverted_iso.csv",
            channel_id="CH1",
            time=t,
            signal_465=sig,
            reference_405=ref,
            sampling_rate=fs,
            trigger_time=t,
            trigger=trigger,
            trigger_name="Event",
            triggers={"Event": trigger},
            trigger_times={"Event": t},
        )
        params = ac.ProcessingParams()
        params.artifact_detection_enabled = False
        params.target_fs_hz = fs
        params.lowpass_hz = 10.0
        params.baseline_max_iter = 15

        normal_params = ac.ProcessingParams.from_dict(params.to_dict())
        normal_params.output_mode = "dFF (motion corrected with fitted ref)"
        normal_processed = ac.PhotometryProcessor().process_trial(
            trial,
            normal_params,
            preview_mode=False,
        )

        inverted_params = ac.ProcessingParams.from_dict(params.to_dict())
        inverted_params.output_mode = INVERTED_FIT_MODE
        processed = ac.PhotometryProcessor().process_trial(
            trial,
            inverted_params,
            preview_mode=False,
        )

        self.assertEqual(processed.output_label, INVERTED_FIT_MODE)
        self.assertEqual(processed.output.shape, processed.time.shape)
        self.assertGreater(np.mean(np.isfinite(processed.output)), 0.95)
        self.assertIn("slope=", processed.output_context)
        self.assertIn("intercept=", processed.output_context)
        self.assertIn("Iso polarity: inverted before fit", processed.output_context)
        self.assertIn(INVERTED_FIT_MODE, processed.outputs)
        self.assertIn("slope=0", normal_processed.output_context)
        self.assertGreater(np.nanmean(np.abs(processed.output - normal_processed.output)), 1e-5)


if __name__ == "__main__":
    unittest.main()
