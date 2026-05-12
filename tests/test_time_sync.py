import os
import sys
import unittest

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "pyBer"))

from time_sync import align_timebase, extract_sync_events  # noqa: E402


class TimeSyncTests(unittest.TestCase):
    def test_extract_ttl_rising_edges(self):
        t = np.arange(0, 10, 0.1)
        x = np.zeros_like(t)
        x[(t >= 1.0) & (t < 1.2)] = 1.0
        x[(t >= 4.0) & (t < 4.2)] = 1.0
        edges = extract_sync_events(t, x, mode="ttl_rising", min_interval_s=0.2)
        self.assertEqual(edges.size, 2)
        np.testing.assert_allclose(edges, [1.0, 4.0], atol=0.11)

    def test_linear_alignment_recovers_camera_time(self):
        fiber_events = np.arange(10.0, 70.0, 10.0)
        camera_events = 1.002 * fiber_events + 0.5
        fiber_time = np.linspace(0, 80, 1000)
        result = align_timebase(fiber_time, camera_events, fiber_events, method="linear")
        self.assertEqual(result.status, "ok")
        self.assertGreaterEqual(result.matched_fiber_events.size, 6)
        np.testing.assert_allclose(result.aligned_time, 1.002 * fiber_time + 0.5, atol=1e-9)

    def test_interpolation_alignment_uses_pulse_pairs(self):
        fiber_events = np.array([0.0, 10.0, 20.0, 30.0])
        camera_events = np.array([0.2, 10.1, 20.4, 30.8])
        fiber_time = np.array([0.0, 5.0, 10.0, 30.0])
        result = align_timebase(fiber_time, camera_events, fiber_events, method="interpolation")
        np.testing.assert_allclose(result.aligned_time[[0, 2, 3]], [0.2, 10.1, 30.8], atol=1e-9)
        self.assertGreater(float(np.nanmax(np.abs(result.residuals))), 0.0)


if __name__ == "__main__":
    unittest.main()
