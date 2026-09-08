"""Repeated independent interval checks and a legacy histogram timing reference."""
from __future__ import annotations

# Reproducible validation and presentation parameters.
SEEDS = 20
FIRST_SEED = 5200
DURATION_S = 600.0
EVENT_COUNT = 80
BIN_WIDTHS_S = (1.0, 7.0, 30.0)
OUTPUT_DIRECTORY = "_test/behavior_summary_validation"

import csv
from pathlib import Path
import sys
from time import perf_counter

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "pyBer"))
from behavior_summary import summarize_behavior


def exact_observed_duration(onsets, offsets, observed):
    """Integrate occupancy on elementary endpoint partitions independently."""
    knots = np.unique(np.r_[onsets, offsets, np.asarray(observed).reshape(-1)])
    midpoint = (knots[:-1] + knots[1:]) / 2
    occupied = np.zeros(len(midpoint), bool)
    available = np.zeros(len(midpoint), bool)
    for start, stop in zip(onsets, offsets):
        occupied |= (midpoint >= start) & (midpoint < stop)
    for start, stop in observed:
        available |= (midpoint >= start) & (midpoint < stop)
    return float(np.sum(np.diff(knots)[occupied & available]))


def main():
    """Check every seed and bin width; retain timings and exact target values."""
    output = ROOT / OUTPUT_DIRECTORY
    output.mkdir(parents=True, exist_ok=True)
    rows = []
    for seed in range(FIRST_SEED, FIRST_SEED + SEEDS):
        rng = np.random.default_rng(seed)
        onsets = np.sort(rng.uniform(1, DURATION_S - 20, EVENT_COUNT))
        offsets = onsets + rng.uniform(.2, 15, EVENT_COUNT)
        observed = np.array([[0, 170], [195, 410], [450, DURATION_S]])
        recording = dict(file_id="synthetic", start=0, end=DURATION_S,
                         observed_intervals=observed, onsets=onsets, offsets=offsets)
        target = exact_observed_duration(onsets, offsets, observed)
        start = perf_counter()
        np.histogram(offsets - onsets, bins=min(20, max(5, int(np.sqrt(EVENT_COUNT)))))
        legacy_ms = (perf_counter() - start) * 1000
        for width in BIN_WIDTHS_S:
            start = perf_counter()
            cumulative = summarize_behavior([recording], "cumulative", bin_s=width)
            current_ms = (perf_counter() - start) * 1000
            np.testing.assert_allclose(cumulative["values"][-1], target, atol=1e-10)
            assert np.all(np.diff(cumulative["values"]) >= -1e-10)
            rate = summarize_behavior([recording], "frequency", bin_s=width)
            estimated_count = np.nansum(rate["values"] * rate["observed_seconds"][0] / 60)
            expected_count = sum(any(a <= event <= b for a, b in observed) for event in onsets)
            np.testing.assert_allclose(estimated_count, expected_count, atol=1e-10)
            rows.append(dict(seed=seed, bin_s=width, observed_duration_s=target,
                             exported_duration_s=float(cumulative["values"][-1]),
                             recovered_onsets=float(estimated_count), expected_onsets=expected_count,
                             cumulative_ms=current_ms, legacy_duration_histogram_ms=legacy_ms))
    with (output / "checks.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Passed {len(rows)} seeded/bin combinations; exact occupancy and event-count conservation.")
    print(f"Median cumulative summary: {np.median([row['cumulative_ms'] for row in rows]):.3f} ms.")
    print(f"Results: {output / 'checks.csv'}")


if __name__ == "__main__":
    main()
