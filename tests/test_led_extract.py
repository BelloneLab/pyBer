import os
import sys
import unittest

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "pyBer"))

import led_extract  # noqa: E402


class LedExtractThresholdTests(unittest.TestCase):
    def test_auto_threshold_matches_skimage_otsu_for_balanced_signal(self):
        try:
            from skimage.filters import threshold_otsu
        except Exception as exc:  # pragma: no cover - optional dependency
            self.skipTest(f"skimage unavailable: {exc}")
        values = np.r_[np.full(100, 84.0), np.full(100, 218.0)]
        threshold, method = led_extract.auto_threshold(values)
        self.assertEqual(method, "Otsu")
        self.assertAlmostEqual(threshold, float(threshold_otsu(values)))

    def test_auto_threshold_matches_skimage_triangle_for_sparse_signal(self):
        try:
            from skimage.filters import threshold_triangle
        except Exception as exc:  # pragma: no cover - optional dependency
            self.skipTest(f"skimage unavailable: {exc}")
        values = np.r_[np.full(1000, 84.0), np.full(60, 218.0)]
        threshold, method = led_extract.auto_threshold(values)
        self.assertEqual(method, "Triangle")
        self.assertAlmostEqual(threshold, float(threshold_triangle(values)))


if __name__ == "__main__":
    unittest.main()
