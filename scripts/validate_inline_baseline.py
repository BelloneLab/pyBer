"""Reproducible speed, invariance and real-recording checks for inline choices.

Run in the pyBer environment. All generated data and figures go to _test;
the optional supplied recording is opened read-only and fingerprinted twice.
The old strict advisor and the inline ranker answer different questions, so
timings compare workflow cost rather than equivalent statistical guarantees.
"""
from __future__ import annotations

# Adjustable validation parameters, including all figure presentation choices.
SEEDS = (5200, 5201, 5202)
PRE_WINDOW_S = 5.0
OUTPUT_DIRECTORY = "_test/inline_baseline_validation"
FIGURE_DPI = 170
COLORS = {"old": "#8390a6", "new": "#4caca8", "score": "#8466d8"}

import argparse
import csv
from dataclasses import replace
import hashlib
from pathlib import Path
import sys
from time import perf_counter

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "pyBer"))
from baseline_advisor import BaselineRecording, recommend_baseline
from baseline_suggestions import BaselineSuggestionConfig, suggest_baselines
from validate_baseline_advisor import SCENARIOS, SCENARIO_LABELS, simulate, array_digest


def validate_recording(recording, config):
    """Independently check bounded scores, pre-event geometry and invariance."""
    before = array_digest(recording)
    start = perf_counter()
    current = suggest_baselines([recording], config)
    duration_ms = 1000 * (perf_counter() - start)
    assert array_digest(recording) == before, "Input arrays were modified"
    scores = []
    for choice in current["choices"]:
        a, b = choice["window"]
        assert -config.pre_window_s <= a < b < 0
        assert 0 <= choice["score"] <= 100
        scores.append(choice["score"])
    assert scores == sorted(scores, reverse=True)
    assert len(scores) <= 3
    # Unit changes must not select a different reference or alter its score.
    scaled = suggest_baselines([replace(recording, signal=recording.signal * 1000)], config)
    assert [c["window"] for c in scaled["choices"]] == [c["window"] for c in current["choices"]]
    np.testing.assert_allclose([c["score"] for c in scaled["choices"]], scores, atol=1e-6)
    # Known behavior responses are excluded from ranking. Increasing their
    # amplitude must not make a different pre-event reference look preferable.
    changed = recording.signal.copy()
    for onset, offset in recording.exclusion_intervals:
        changed[(recording.time >= onset) & (recording.time <= offset)] += 10000
    response = suggest_baselines([replace(recording, signal=changed)], config)
    assert [c["window"] for c in response["choices"]] == [c["window"] for c in current["choices"]]
    np.testing.assert_allclose([c["score"] for c in response["choices"]], scores, atol=1e-6)
    return current, duration_ms


def optional_real_recording(path, behavior_path):
    """Read known processed H5 and binary social-contact events without edits."""
    import h5py
    import pandas as pd
    paths = (Path(path), Path(behavior_path))
    hashes = [hashlib.sha256(p.read_bytes()).hexdigest() for p in paths]
    with h5py.File(paths[0], "r") as data:
        time, signal = data["data/time"][:], data["data/output"][:]
    frame = pd.read_csv(paths[1])
    t = frame["time"].to_numpy(float)
    active = frame["social_contacts"].to_numpy(float) > .5
    edges = np.diff(np.r_[False, active, False].astype(int))
    starts, stops = np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)
    offsets = t[np.minimum(stops, len(t) - 1)]
    result = BaselineRecording("Provided social contacts", time, signal, t[starts],
                               np.column_stack((t[starts], offsets)))
    assert hashes == [hashlib.sha256(p.read_bytes()).hexdigest() for p in paths]
    return result, hashes


def main():
    """Compare both implementations and save auditable tables and figures."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--processed-h5")
    parser.add_argument("--behavior-csv")
    args = parser.parse_args()
    output = ROOT / OUTPUT_DIRECTORY
    output.mkdir(parents=True, exist_ok=True)
    config = BaselineSuggestionConfig(pre_window_s=PRE_WINDOW_S)
    rows = []
    for scenario in SCENARIOS:
        for seed in SEEDS:
            recording = simulate(scenario, seed)
            current, elapsed = validate_recording(recording, config)
            start = perf_counter()
            previous = recommend_baseline([recording])
            old_ms = 1000 * (perf_counter() - start)
            if scenario == "flat_signal":
                assert not current["choices"], "Flat signals cannot support normalization"
            rows.append({"scenario": scenario, "seed": seed, "inline_ms": elapsed,
                         "strict_ms": old_ms, "choices": len(current["choices"]),
                         "best_score": current["choices"][0]["score"] if current["choices"] else 0,
                         "best_quality": current["choices"][0]["quality"] if current["choices"] else "Unavailable",
                         "strict_status": previous["status"]})
    with (output / "benchmark.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    if args.processed_h5 and args.behavior_csv:
        import json
        recording, hashes = optional_real_recording(args.processed_h5, args.behavior_csv)
        result, elapsed = validate_recording(recording, config)
        (output / "provided_recording.json").write_text(
            json.dumps({"elapsed_ms": elapsed, "source_sha256": hashes, "report": result},
                       indent=2, default=lambda value: value.tolist() if hasattr(value, "tolist") else str(value)),
            encoding="utf-8")
        print("Provided recording:", elapsed, "ms;", len(result["choices"]), "choices")
    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                         "svg.fonttype": "none", "pdf.fonttype": 42})
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.3), layout="constrained")
    x = np.arange(len(SCENARIOS))
    for key, offset, color, label in (("strict_ms", -.15, COLORS["old"], "Previous strict advisor"),
                                      ("inline_ms", .15, COLORS["new"], "Inline suggestions")):
        medians = [np.median([row[key] for row in rows if row["scenario"] == name]) for name in SCENARIOS]
        axes[0].bar(x + offset, medians, width=.26, color=color, label=label)
    axes[0].set(yscale="log", ylabel="Median computation time (ms)", title="Workflow speed (3 seeds per case)")
    axes[0].legend(frameon=False, fontsize=8)
    scores = [np.median([r["best_score"] for r in rows if r["scenario"] == name]) for name in SCENARIOS]
    axes[1].bar(x, scores, width=.46, color=COLORS["score"])
    axes[1].set(ylim=(0, 105), ylabel="Top suitability score (%)", title="Scores describe suitability, not confidence")
    for axis in axes:
        axis.set_xticks(x, SCENARIO_LABELS, rotation=45, ha="right")
        axis.grid(axis="y", alpha=.13)
        axis.set_axisbelow(True)
    for suffix in ("png", "pdf", "svg"):
        fig.savefig(output / f"baseline_comparison.{suffix}", dpi=FIGURE_DPI)
    plt.close(fig)
    print(f"Passed {len(rows)} cases, pre-event geometry, bounded scores, unit invariance and input integrity.")
    print("Median milliseconds:", {key: float(np.median([r[key] for r in rows])) for key in ("strict_ms", "inline_ms")})


if __name__ == "__main__":
    main()
