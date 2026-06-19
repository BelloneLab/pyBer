import os
import sys
import unittest

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "pyBer"))

import led_extract as le  # noqa: E402


class LedExtractTests(unittest.TestCase):
    def test_reduce_modes_contrast(self):
        # A bright pixel in a dark ROI: max/bright preserve it, mean dilutes it.
        plane = np.zeros((4, 4), float)
        plane[0, 0] = 250.0
        self.assertEqual(le._reduce_plane(plane, "max"), 250.0)
        self.assertLess(le._reduce_plane(plane, "mean"), 20.0)
        self.assertGreaterEqual(le._reduce_plane(plane, "bright"), le._reduce_plane(plane, "mean"))

    def test_binarize_is_clean_square_wave(self):
        t = np.linspace(0, 10, 1000)
        raw = (np.sin(2 * np.pi * t) > 0) * 200.0 + 30.0
        b = le.binarize_signal(raw)
        self.assertEqual(sorted(set(b.tolist())), [0.0, 1.0])
        # 10 full cycles -> 20 transitions (allow a small boundary tolerance).
        edges = int(np.sum(np.abs(np.diff(b)) > 0))
        self.assertTrue(18 <= edges <= 22, edges)

    def test_debounce_removes_single_frame_chatter(self):
        b = np.array([0, 0, 1, 0, 0, 0, 1, 1, 1, 0, 0], float)  # lone 1 at idx 2
        out = le.debounce_binary(b, min_run=2)
        self.assertEqual(out[2], 0.0)  # chatter absorbed
        self.assertTrue(np.all(out[6:9] == 1.0))  # genuine 3-frame run kept

    def test_apply_signal_format_binary(self):
        raw = np.r_[np.zeros(50), np.full(50, 100.0)] + np.random.RandomState(0).randn(100) * 2
        out = le.apply_signal_format(raw, "binary")
        self.assertEqual(sorted(set(out.tolist())), [0.0, 1.0])
        self.assertEqual(out[0], 0.0)
        self.assertEqual(out[-1], 1.0)

    def test_plan_chunks_min_size_and_coverage(self):
        chunks = le.plan_chunks(0, 10000, 8)
        # Covers the whole range contiguously.
        self.assertEqual(chunks[0][0], 0)
        self.assertEqual(chunks[-1][1], 10000)
        for (a, b), (c, _d) in zip(chunks, chunks[1:]):
            self.assertEqual(b, c)
        # Min chunk size amortizes the lead-in.
        self.assertTrue(all((b - a) >= 800 or b == 10000 for a, b in chunks))

    def test_plan_chunks_single(self):
        self.assertEqual(le.plan_chunks(0, 500, 1), [(0, 500)])
        self.assertEqual(le.plan_chunks(5, 5, 4), [])

    def test_auto_threshold_picks_triangle_for_sparse(self):
        # Low-duty-cycle LED barcode: auto must switch off Otsu to Triangle.
        rng = np.random.RandomState(0)
        v = 85.0 + rng.randn(4000) * 2.0
        on = np.zeros(4000, bool)
        for c in rng.choice(np.arange(50, 3950), size=12, replace=False):
            on[c:c + rng.randint(2, 6)] = True
        v[on] = 215.0
        thr = le.compute_threshold(v, "auto")
        # Threshold sits in the gap (above baseline noise, below the flash).
        self.assertTrue(90.0 <= thr <= 214.0, thr)
        b = le.binarize_signal(v)
        self.assertGreater(float(np.mean((b > 0) == on)), 0.99)

    def test_binarize_robust_to_specular_outlier(self):
        # A single bright outlier stretches the min-max range; the old
        # "0.5 of range" rule jumps above a faint LED and drops every flash.
        rng = np.random.RandomState(1)
        v = 80.0 + rng.randn(4000) * 2.0
        on = np.zeros(4000, bool)
        for c in rng.choice(np.arange(50, 3950), size=12, replace=False):
            on[c:c + rng.randint(3, 7)] = True
        v[on] = 120.0           # faint "on" level
        v[1000] = 250.0         # specular reflection frame
        old_fixed = 80.0 + 0.5 * (v.max() - v.min())
        self.assertGreater(old_fixed, 120.0)  # the rule we replaced would miss flashes
        b = le.binarize_signal(v)
        recovered = float(np.mean(b[on] > 0))
        self.assertGreater(recovered, 0.95, recovered)


if __name__ == "__main__":
    unittest.main()
