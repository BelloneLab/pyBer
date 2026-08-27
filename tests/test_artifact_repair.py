"""Tests for artifact repair quality: bridging, gap merging, resampler fidelity.

Regression coverage for the sawtooth defect where resample_poly's default
kernel painted a |fs - target_fs| beat tone onto interpolated artifact
bridges, and for endpoint-noise sensitivity of the linear repair.
"""
import os
import sys
import unittest

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "pyBer"))

import analysis_core as ac  # noqa: E402


class ResampleFidelityTests(unittest.TestCase):
    def test_fractional_ratio_preserves_linear_ramp(self):
        # 120.48 -> 100 Hz is the Doric case that exposed the beat-tone bug.
        fs, target = 120.48, 100.0
        t = np.arange(0.0, 30.0, 1.0 / fs)
        ramp = 0.10 + 1e-3 * t  # slope comparable to a real artifact bridge
        t2, y2, _, fs_used = ac._resample_pair_to_target_fs(t, ramp, ramp, fs, target)
        self.assertAlmostEqual(fs_used, 100.0, places=6)

        # Interior samples (away from edge transients) must stay on the line.
        core = (t2 > 2.0) & (t2 < t2[-1] - 2.0)
        coef = np.polyfit(t2[core], y2[core], 1)
        residual = y2[core] - np.polyval(coef, t2[core])
        rms = float(np.sqrt(np.mean(residual ** 2)))
        # Default beta=5 kernel leaves ~3e-5 rms here; beta=14 is ~100x cleaner.
        self.assertLess(rms, 3e-7)


class BridgeMaskedRegionsTests(unittest.TestCase):
    def test_bridge_ignores_noisy_edge_samples(self):
        fs = 100.0
        n = 1000
        y = np.full(n, 1.0)
        mask = np.zeros(n, dtype=bool)
        mask[400:500] = True
        # Corrupt the single samples hugging the masked run: a naive
        # sample-to-sample interpolation would tilt the whole bridge.
        y[399] = 3.0
        y[500] = -1.0
        out = ac._bridge_masked_regions(y, mask, fs)
        self.assertTrue(np.all(np.isfinite(out)))
        # Median anchors over ~8 samples keep the bridge near the true level.
        self.assertLess(float(np.max(np.abs(out[mask] - 1.0))), 0.35)

    def test_bridge_is_straight_inside_run(self):
        fs = 100.0
        rng = np.random.default_rng(7)
        t = np.arange(0.0, 20.0, 1.0 / fs)
        y = 1.0 + 0.01 * t + 0.002 * rng.standard_normal(t.size)
        mask = np.zeros(t.size, dtype=bool)
        mask[800:1000] = True
        out = ac._bridge_masked_regions(y, mask, fs)
        inside = out[801:999]
        second_diff = np.diff(inside, 2)
        self.assertLess(float(np.max(np.abs(second_diff))), 1e-9)

    def test_bridge_handles_edge_of_record(self):
        fs = 50.0
        y = np.linspace(2.0, 3.0, 200)
        mask = np.zeros(200, dtype=bool)
        mask[:20] = True
        mask[-15:] = True
        out = ac._bridge_masked_regions(y, mask, fs)
        self.assertTrue(np.all(np.isfinite(out)))
        # No left anchor: leading run holds the first clean level.
        self.assertLess(abs(float(out[0]) - float(np.median(y[20:28]))), 0.05)


class CloseMaskGapsTests(unittest.TestCase):
    def test_short_gap_is_merged_and_long_gap_is_kept(self):
        fs = 100.0
        t = np.arange(0.0, 10.0, 1.0 / fs)
        m = np.zeros(t.size, dtype=bool)
        m[100:150] = True
        m[160:200] = True   # 0.10 s gap -> merged
        m[400:450] = True   # 2.0 s gap -> kept separate
        merged = ac._close_mask_gaps(t, m, max_gap_s=0.25)
        regions = ac.regions_from_mask(t, merged)
        self.assertEqual(len(regions), 2)
        self.assertTrue(np.all(merged[100:200]))
        self.assertFalse(np.any(merged[200:400]))

    def test_smart_detector_merges_twin_dips(self):
        fs = 100.0
        t = np.arange(0.0, 60.0, 1.0 / fs)
        rng = np.random.default_rng(3)
        signal = 250.0 + 0.05 * rng.standard_normal(t.size)
        reference = 80.0 + 0.02 * rng.standard_normal(t.size)
        # Two sharp shared dips 0.30 s apart (like a double head-bump).
        for center in (30.0, 30.3):
            dip = np.exp(-0.5 * ((t - center) / 0.02) ** 2)
            signal -= 12.0 * dip
            reference -= 4.0 * dip
        result = ac.detect_artifacts_smart(t, signal, reference, k=7.0,
                                           window_s=5.0, pad_s=0.25, fs=fs)
        covering = [(a, b) for a, b in result.regions if a <= 30.0 and b >= 30.3]
        self.assertEqual(len(covering), 1, msg=f"regions={result.regions}")


def _quiet_pair(duration_s=120.0, fs=100.0, seed=11):
    t = np.arange(0.0, duration_s, 1.0 / fs)
    rng = np.random.default_rng(seed)
    signal = 250.0 + 0.05 * rng.standard_normal(t.size)
    reference = 80.0 + 0.02 * rng.standard_normal(t.size)
    return t, signal, reference


def _detect(t, signal, reference, fs=100.0):
    return ac.detect_artifacts_smart(t, signal, reference, k=8.0,
                                     window_s=5.0, pad_s=0.25, fs=fs)


def _interior_regions(regions, lo=2.0, hi=118.0):
    """Regions overlapping the interior of the 120 s synthetic record.

    The detector has always flagged up to ~1 s at each record boundary
    (windowed statistics are boundary-biased there; on real data those
    samples are LED/detector settling anyway), so tests assert on the
    interior only.
    """
    return [(a, b) for a, b in regions if b > lo and a < hi]


class SharedDipTierTests(unittest.TestCase):
    """The looser shared tier flags concurrent dips that neither channel
    could report on its own, and only concurrent NEGATIVE ones.

    Borderline shared dips are corroboration-gated: they count only when the
    recording also carries at least one strong interior artifact. On real
    clean-but-spiky recordings, isolated moderate co-dips are noise
    coincidences (user ground truth, 2026-08-27)."""

    SIG_DEPTH = 0.22   # ~4.4 noise sigma: below the k=8 single-channel gates
    REF_DEPTH = 0.12   # ~6 noise sigma, still below the 405's own core bar
    DIP_SIGMA = 0.04   # seconds

    def _dip(self, t, center):
        return np.exp(-0.5 * ((t - center) / self.DIP_SIGMA) ** 2)

    def _strong_hit(self, t, signal, reference, center=30.0):
        """A gross bilateral artifact (>> SMART_STRONG_SCORE) as corroborator."""
        hit = np.exp(-0.5 * ((t - center) / 0.05) ** 2)
        return signal - 3.0 * hit, reference - 1.0 * hit

    def test_concurrent_moderate_dips_are_flagged_when_corroborated(self):
        t, signal, reference = _quiet_pair()
        dip = self._dip(t, 60.0)
        signal, reference = self._strong_hit(t, signal - self.SIG_DEPTH * dip,
                                             reference - self.REF_DEPTH * dip)
        res = _detect(t, signal, reference)
        covering = [(a, b) for a, b in res.regions if a <= 60.0 <= b]
        self.assertEqual(len(covering), 1, msg=f"regions={res.regions}")

    def test_isolated_moderate_dip_is_suppressed(self):
        # No strong artifact anywhere in the record: the moderate co-dip is
        # indistinguishable from heavy-tailed noise coincidence and must not
        # be flagged.
        t, signal, reference = _quiet_pair()
        dip = self._dip(t, 60.0)
        res = _detect(t, signal - self.SIG_DEPTH * dip,
                      reference - self.REF_DEPTH * dip)
        covering = [(a, b) for a, b in res.regions if a <= 60.0 <= b]
        self.assertEqual(len(covering), 0, msg=f"regions={res.regions}")
        self.assertIn("suppressed", res.summary)

    def test_same_dip_in_one_channel_stays_clean(self):
        t, signal, reference = _quiet_pair()
        sig_only = signal - self.SIG_DEPTH * self._dip(t, 60.0)
        sig_only, reference = self._strong_hit(t, sig_only, reference)
        res = _detect(t, sig_only, reference)
        covering = [(a, b) for a, b in res.regions if a <= 60.0 <= b]
        self.assertEqual(len(covering), 0, msg=f"regions={res.regions}")

    def test_concurrent_moderate_bumps_stay_clean(self):
        # Positive-going concurrence is NOT artifact-certain (bleed-through,
        # hemodynamics), so the loose tier must not fire on bumps, even in a
        # recording corroborated by a strong artifact elsewhere.
        t, signal, reference = _quiet_pair()
        bump = self._dip(t, 60.0)
        signal, reference = self._strong_hit(t, signal + self.SIG_DEPTH * bump,
                                             reference + self.REF_DEPTH * bump)
        res = _detect(t, signal, reference)
        covering = [(a, b) for a, b in res.regions if a <= 60.0 <= b]
        self.assertEqual(len(covering), 0, msg=f"regions={res.regions}")

    def test_quiet_noise_yields_no_interior_regions(self):
        for seed in (5, 11, 23):
            t, signal, reference = _quiet_pair(seed=seed)
            res = _detect(t, signal, reference)
            interior = _interior_regions(res.regions)
            self.assertEqual(len(interior), 0,
                             msg=f"seed={seed} regions={res.regions}")


class SessionCorroborationTests(unittest.TestCase):
    """Heavy-tailed channel noise crosses fixed z thresholds many times per
    minute; without a strong interior artifact those crossings are false
    alarms and the whole recording must come back clean."""

    def _heavy_tailed_pair(self, duration_s=300.0, fs=100.0, seed=10):
        t = np.arange(0.0, duration_s, 1.0 / fs)
        rng = np.random.default_rng(seed)
        signal = 250.0 + 0.05 * rng.standard_t(3, t.size)
        reference = 80.0 + 0.02 * rng.standard_t(3, t.size)
        return t, signal, reference

    def test_heavy_tailed_noise_yields_no_regions(self):
        t, signal, reference = self._heavy_tailed_pair()
        res = _detect(t, signal, reference)
        self.assertEqual(res.regions, [], msg=f"regions={res.regions}")

    def test_strong_hit_in_heavy_tailed_noise_is_still_caught(self):
        t, signal, reference = self._heavy_tailed_pair()
        hit = np.exp(-0.5 * ((t - 150.0) / 0.05) ** 2)
        res = _detect(t, signal - 3.0 * hit, reference - 1.0 * hit)
        covering = [(a, b) for a, b in res.regions if a <= 150.0 <= b]
        self.assertEqual(len(covering), 1, msg=f"regions={res.regions}")


class TailExtensionTests(unittest.TestCase):
    """Regions grow over SHARED recovery ramps so bridge anchors land on
    settled signal, but never over single-channel structure (which may be
    real physiology next to the artifact)."""

    def _setup(self):
        fs = 100.0
        t = np.arange(0.0, 120.0, 1.0 / fs)
        rng = np.random.default_rng(7)
        ch_a = 1.0 + 0.05 * rng.standard_normal(t.size)
        ch_b = 2.0 + 0.02 * rng.standard_normal(t.size)
        core = np.zeros(t.size, dtype=bool)
        core[(t >= 60.0) & (t <= 60.1)] = True
        # Linear recovery ramp from 8 to 0 noise sigma over 0.8 s.
        ramp = np.where((t > 60.1) & (t <= 60.9), (60.9 - t) / 0.8, 0.0)
        return fs, t, ch_a, ch_b, core, ramp

    def test_shared_recovery_ramp_is_covered(self):
        fs, t, ch_a, ch_b, core, ramp = self._setup()
        ext = ac._extend_regions_to_settled(
            t, [ch_a - 0.4 * ramp, ch_b - 0.16 * ramp], core, fs)
        regions = ac.regions_from_mask(t, ext)
        self.assertEqual(len(regions), 1, msg=f"regions={regions}")
        self.assertGreater(regions[0][1], 60.45, msg=f"regions={regions}")

    def test_single_channel_ramp_is_not_swallowed(self):
        fs, t, ch_a, ch_b, core, ramp = self._setup()
        ext = ac._extend_regions_to_settled(
            t, [ch_a - 0.4 * ramp, ch_b], core, fs)
        regions = ac.regions_from_mask(t, ext)
        self.assertEqual(len(regions), 1, msg=f"regions={regions}")
        self.assertLess(regions[0][1], 60.15, msg=f"regions={regions}")


if __name__ == "__main__":
    unittest.main()
