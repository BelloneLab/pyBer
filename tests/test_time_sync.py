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

    def test_overlap_time_infers_unmatched_leading_camera_pulses(self):
        camera_lead = np.array([4.0, 8.0, 8.5, 12.0, 12.5, 16.0, 16.5])
        shared = np.array([21.0, 21.5, 25.0, 25.5, 29.0, 29.5, 34.0, 34.5])
        camera_events = np.r_[camera_lead, shared]
        fiber_events = shared + 0.12
        fiber_time = np.linspace(20.0, 36.0, 50)
        result = align_timebase(fiber_time, camera_events, fiber_events, method="linear", max_offset=0)
        self.assertEqual(result.pair_offset, -7)
        self.assertEqual(result.status, "ok")
        self.assertLess(result.rms_error_s, 1e-9)
        np.testing.assert_allclose(result.aligned_time, fiber_time - 0.12, atol=1e-9)


if __name__ == "__main__":
    unittest.main()
