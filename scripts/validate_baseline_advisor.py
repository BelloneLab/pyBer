"""Reproduce baseline-advisor validation without reading or changing user data.

Run from an IDE, or use ``python scripts/validate_baseline_advisor.py --seeds 10``.
All simulations, policies, and figure settings are explicit below. The fixed
reference is the former one-second pre-event choice, not an optimization method.
The comparisons describe reference reliability, not neural-response accuracy or
the false-positive rate of a statistical test.
"""
from __future__ import annotations

# User-editable experiment parameters and visual settings.
DEFAULT_SEEDS = 10
FIRST_SEED = 5100
SAMPLE_RATE_HZ = 40.0
RECORDING_SECONDS = 840.0
EVENT_SPACING_SECONDS = 22.0
FIRST_EVENT_SECONDS = 40.0
FIXED_BASELINE = (-1.0, 0.0)
PSTH_WINDOW = (-4.0, 4.0)
AR1_CORRELATION_SECONDS = 0.30
OUTPUT_DIRECTORY = "_test/baseline_advisor_validation"
FIGURE_DPI = 170
FIGURE_COLORS = {"advisor": "#357FA8", "fixed": "#D68B56", "muted": "#8F9BA8"}
SCENARIOS = (
    "white_noise", "correlated_noise", "sparse_transients", "dense_events",
    "long_prior_bouts", "cut_gaps", "strong_drift", "flat_signal", "late_scale_change",
)
SCENARIO_LABELS = (
    "White noise", "Correlated noise", "Sparse transients", "Dense events",
    "Long prior bouts", "Cut gaps", "Strong drift", "Flat signal", "Later scale change",
)

import argparse
import csv
from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path
import sys
from time import perf_counter

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import lfilter

REPOSITORY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY / "pyBer"))
from baseline_advisor import BaselineAdvisorConfig, BaselineRecording, recommend_baseline
from postprocessing_core import compute_psth_matrix


def simulate(scenario: str, seed: int) -> BaselineRecording:
    """Build measured samples, unrelated event times, and known event intervals.

    Events are independent of random signal innovations in every stochastic
    scenario. Structured drift, missing data, and later variance changes are
    deliberate failure cases. No effect amplitude is used to choose a window.
    """
    rng = np.random.default_rng(seed)
    time = np.arange(int(RECORDING_SECONDS * SAMPLE_RATE_HZ)) / SAMPLE_RATE_HZ
    signal = rng.normal(size=len(time))
    events = np.arange(FIRST_EVENT_SECONDS, RECORDING_SECONDS - 8, EVENT_SPACING_SECONDS)
    events = events + rng.uniform(-0.4, 0.4, len(events))
    intervals = np.column_stack((events, events + 0.5))
    if scenario in ("correlated_noise", "sparse_transients"):
        rho = np.exp(-1 / (SAMPLE_RATE_HZ * AR1_CORRELATION_SECONDS))
        signal = lfilter([np.sqrt(1 - rho * rho)], [1, -rho], signal)
    if scenario == "sparse_transients":
        impulses = (rng.random(len(time)) < 0.08 / SAMPLE_RATE_HZ) * rng.exponential(3, len(time))
        signal = 0.35 * signal + lfilter([1], [1, -np.exp(-1 / (0.5 * SAMPLE_RATE_HZ))], impulses)
    elif scenario == "dense_events":
        events = np.arange(FIRST_EVENT_SECONDS, RECORDING_SECONDS - 8, 1.2)
        intervals = np.column_stack((events, events + 0.5))
    elif scenario == "long_prior_bouts":
        intervals = np.column_stack((events, events + 12))
    elif scenario == "cut_gaps":
        # Every one-second interval contains a documented cut. Alternate NaN
        # masks and genuinely absent timestamps to exercise both representations.
        cut = np.mod(time, 0.8) < 0.10
        if seed % 2:
            time, signal = time[~cut], signal[~cut]
        else:
            signal[cut] = np.nan
    elif scenario == "strong_drift":
        signal = 0.15 * signal + 0.3 * time
    elif scenario == "flat_signal":
        signal = np.zeros_like(signal)
    elif scenario == "late_scale_change":
        # The later block is not consulted to rank candidates. A change of
        # scale here should cause the one selected candidate to fail validation.
        signal[time > RECORDING_SECONDS * 0.69] *= 8
    return BaselineRecording(scenario, time, signal, events, intervals)


def array_digest(recording: BaselineRecording) -> str:
    """Fingerprint all simulation inputs so unintended mutation is detectable."""
    digest = hashlib.sha256()
    for array in (recording.time, recording.signal, recording.events, recording.exclusion_intervals):
        digest.update(np.asarray(array).tobytes())
    return digest.hexdigest()


def finite_quantile(values, quantile: float) -> float | None:
    """Keep reports JSON-compliant when no diagnostic is available."""
    array = np.asarray(values, float)
    array = array[np.isfinite(array)]
    return float(np.quantile(array, quantile)) if len(array) else None


def window_diagnostics(recording, window, information_s, config):
    """Describe fixed or advised windows on the same observed native samples.

    Information is an ACF-based approximation inherited from the advisor's
    training diagnostics for this particular window. Values here are descriptive across all events, including
    the validation events, and never feed back into the recommendation.
    """
    scales, information = [], []
    overlaps, missing = 0, 0
    time, signal = recording.time, recording.signal
    dt = float(np.median(np.diff(time)))
    excluded = recording.exclusion_intervals + np.array([-config.event_guard_s, config.recovery_s])
    for event in recording.events:
        start, stop = event + window[0], event + window[1]
        overlap = np.any((excluded[:, 0] <= stop) & (excluded[:, 1] >= start))
        overlaps += int(overlap)
        left, right = np.searchsorted(time, (start, stop))
        part = signal[left:right]
        if (len(part) < 2 or not np.all(np.isfinite(part))
                or np.any(np.diff(time[left:right]) > 3 * dt)):
            missing += 1
            continue
        sd = float(np.std(part))
        if sd > np.finfo(float).eps:
            scales.append(sd)
            information.append(min(len(part), len(part) * dt / information_s))
    p10, p90 = finite_quantile(scales, 0.1), finite_quantile(scales, 0.9)
    return {
        "duration_s": float(window[1] - window[0]),
        "effective_samples_p10": finite_quantile(information, 0.1),
        "between_trial_sd_ratio": p90 / p10 if p10 else None,
        "overlap_fraction": overlaps / len(recording.events),
        "missing_fraction": missing / len(recording.events),
    }


def null_curve(recording, baseline):
    """Run the actual whole-trial normalizer and summarize an unrelated-event curve.

    Absolute curve size is reported, not thresholded as a statistical discovery.
    Dense or long-bout events need not be independent sampling units.
    """
    time, matrix = compute_psth_matrix(
        recording.time, recording.signal, recording.events, PSTH_WINDOW,
        baseline, SAMPLE_RATE_HZ, normalization="zscore",
    )
    valid = np.isfinite(matrix)
    count = valid.sum(axis=0)
    mean = np.divide(np.where(valid, matrix, 0).sum(axis=0), count,
                     out=np.full(len(time), np.nan), where=count > 0)
    pre = mean[(time >= -3) & (time <= -1.5)]
    post = mean[(time >= 1) & (time <= 2.5)]
    contrast = float(np.nanmean(post) - np.nanmean(pre)) if np.any(np.isfinite(mean)) else None
    return time, mean, {"pre_post_curve_difference": contrast,
                        "absolute_curve_p95": finite_quantile(np.abs(mean), 0.95)}


def selected_candidate(report):
    """Locate the training winner, including a winner later rejected on holdout."""
    return next((row for row in report["candidates"] if row["eligible"]), None)


def validate_case(scenario, seed, config):
    """Run one independent seed and retain every failure, rather than retuning."""
    recording = simulate(scenario, seed)
    original = array_digest(recording)
    start = perf_counter()
    report = recommend_baseline([recording], config, current_window=FIXED_BASELINE)
    elapsed = perf_counter() - start
    assert array_digest(recording) == original, "Advisor mutated source arrays."
    context = report["recordings"][0]["training_context"]
    start = perf_counter()
    fixed_information = report["current"]["per_recording"][0].get("information_s", context["information_s"])
    fixed = window_diagnostics(recording, FIXED_BASELINE, fixed_information, config)
    fixed_time = perf_counter() - start
    relative, fixed_curve, fixed_null = null_curve(recording, FIXED_BASELINE)
    winner = selected_candidate(report)
    row = {"scenario": scenario, "seed": seed, "status": report["status"],
           "advisor_time_ms": elapsed * 1000, "fixed_diagnostic_time_ms": fixed_time * 1000,
           "event_count": len(recording.events), "source_sha256": original,
           "information_time_s": context["information_s"], "context_skewness": context["skewness"],
           "reasons": "; ".join(report["reasons"]),
           **{"fixed_" + key: value for key, value in {**fixed, **fixed_null}.items()}}
    curves = {"time": relative, "fixed": fixed_curve}
    if report["status"] == "recommended":
        assert winner["diagnostics"]["validation"]["eligible"], "Accepted a failed validation."
        assert winner["diagnostics"]["training"]["eligible"], "Accepted a failed training candidate."
        assert report["window"][1] < 0, "Advised a post-event reference."
        candidate_information = winner["diagnostics"]["training"]["per_recording"][0].get(
            "information_s", context["information_s"])
        diagnostics = window_diagnostics(recording, report["window"], candidate_information, config)
        _, curves["advisor"], null = null_curve(recording, report["window"])
        row.update({"window_start_s": report["window"][0], "window_end_s": report["window"][1],
                    **{"advisor_" + key: value for key, value in {**diagnostics, **null}.items()}})
    row["holdout_rejected"] = bool(winner and "validation" in winner["diagnostics"]
                                    and not winner["diagnostics"]["validation"]["eligible"])
    return row, report, curves


def invariance_checks(config):
    """Test response exclusion, physical units, determinism, and group safeguards."""
    recording = simulate("white_noise", FIRST_SEED)
    original = recommend_baseline([recording], config)
    changed = recording.signal.copy()
    for start, end in recording.exclusion_intervals:
        mask = (recording.time >= start) & (recording.time <= end + config.recovery_s)
        changed[mask] = 100000 * np.sin(recording.time[mask] * 113)
    mutated = recommend_baseline([replace(recording, signal=changed)], config)
    repeat = recommend_baseline([recording], config)
    scaled = recommend_baseline([replace(recording, signal=recording.signal * 0.001)], config)
    grouped = recommend_baseline([recording, simulate("flat_signal", FIRST_SEED)], config)
    checks = {
        "excluded_post_event_values_do_not_change_report": original == mutated,
        "repeated_identical_inputs_are_deterministic": original == repeat,
        "unit_conversion_preserves_status_and_window":
            (original["status"], original["window"]) == (scaled["status"], scaled["window"]),
        "flat_recording_cannot_be_hidden_in_group": grouped["status"] != "recommended",
    }
    for name, passed in checks.items():
        assert passed, name
    return checks


def summarize(rows):
    """Aggregate seed repeats without presenting them as biological replicates."""
    result = []
    for scenario in SCENARIOS:
        cases = [row for row in rows if row["scenario"] == scenario]
        accepted = [row for row in cases if row["status"] == "recommended"]
        item = {"scenario": scenario, "runs": len(cases), "recommended": len(accepted),
                "recommended_fraction": len(accepted) / len(cases),
                "holdout_rejected": sum(row["holdout_rejected"] for row in cases)}
        keys = ("advisor_time_ms", "fixed_diagnostic_time_ms", "fixed_effective_samples_p10",
                "advisor_effective_samples_p10", "advisor_duration_s", "fixed_absolute_curve_p95",
                "advisor_absolute_curve_p95")
        for key in keys:
            item["median_" + key] = finite_quantile(
                [row[key] for row in cases if row.get(key) is not None], 0.5)
        result.append(item)
    return result


def write_csv(path, rows):
    """Write union-schema diagnostic rows using portable plain-text values."""
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def format_axis(axis):
    """Keep the validation figures compact, readable, and free from chart frames."""
    axis.spines[["top", "right"]].set_visible(False)
    axis.grid(axis="x", color="#E5E9ED", linewidth=0.6, zorder=0)
    axis.tick_params(length=2, color="#8F9BA8")


def save_figure(figure, prefix):
    """Export the same verified layout as vector PDF/SVG and raster PNG."""
    for extension in ("png", "pdf", "svg"):
        figure.savefig(str(prefix) + "." + extension, dpi=FIGURE_DPI, facecolor="white")
    plt.close(figure)


def plot_validation(rows, summary, curves, directory):
    """Show abstention and computational cost as clearly as successful cases."""
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9,
                         "axes.titlesize": 11, "axes.labelsize": 9, "svg.fonttype": "none"})
    y = np.arange(len(summary))
    figure, axes = plt.subplots(1, 3, figsize=(13.4, 4.8), layout="constrained", sharey=True)
    axes[0].barh(y, [100 * item["recommended_fraction"] for item in summary],
                 height=0.42, color=FIGURE_COLORS["advisor"], zorder=3)
    axes[0].set_yticks(y, SCENARIO_LABELS)
    axes[0].invert_yaxis()
    axes[0].set(xlim=(0, 110), xlabel="Seeds with a recommendation (%)", title="Recommendation or abstention")
    for position, item in enumerate(summary):
        axes[0].text(102, position, f'{item["recommended"]}/{item["runs"]}', va="center", fontsize=8)
    for offset, method in ((-0.13, "fixed"), (0.13, "advisor")):
        values = [item.get("median_" + method + "_effective_samples_p10") for item in summary]
        for index, value in enumerate(values):
            if value is not None and value > 0:
                axes[1].plot(value, index + offset, "o", ms=4, color=FIGURE_COLORS[method],
                             label="Fixed 1 s" if method == "fixed" and index == 0 else
                             "Advised" if method == "advisor" and index == 0 else None)
    axes[1].axvline(20, color="#98A1AB", lw=0.8, ls="--")
    axes[1].set(xscale="log", xlabel="Approximate effective samples (10th percentile)",
                title="Information supporting each reference")
    axes[1].legend(frameon=False, loc="lower right", fontsize=8)
    for index, item in enumerate(summary):
        times = [row["advisor_time_ms"] for row in rows if row["scenario"] == item["scenario"]]
        axes[2].scatter(times, np.full(len(times), index), s=11, alpha=0.35,
                        color=FIGURE_COLORS["advisor"], edgecolors="none")
        axes[2].plot(item["median_advisor_time_ms"], index, "|", color="#203247", ms=12, mew=1.4)
    axes[2].set(xlabel="Advisor runtime per recording (ms)", title="All seeds, with median")
    for axis in axes:
        format_axis(axis)
    figure.suptitle("Baseline advisor: deterministic simulations with unchanged default policy", fontsize=13)
    save_figure(figure, directory / "validation_overview")

    figure, axes = plt.subplots(1, 3, figsize=(12.0, 3.2), layout="constrained")
    for axis, scenario, title in zip(axes, SCENARIOS[:3], SCENARIO_LABELS[:3]):
        values = curves[scenario]
        for method in ("fixed", "advisor"):
            if method in values:
                axis.plot(values["time"], values[method], lw=1.0, color=FIGURE_COLORS[method],
                          label="Fixed 1 s" if method == "fixed" else "Advised")
        axis.axhline(0, color="#BCC4CD", lw=0.7)
        axis.axvline(0, color="#BCC4CD", lw=0.7, ls="--")
        axis.set(title=title, xlabel="Time from unrelated event (s)", ylabel="Mean trial baseline z-score")
        axis.legend(frameon=False, fontsize=8)
        format_axis(axis)
    figure.suptitle(f"Unrelated-event diagnostic examples, seed {FIRST_SEED}; descriptive curves, no significance test", fontsize=11)
    save_figure(figure, directory / "unrelated_event_examples")


def main(argv=None):
    """Run validation, preserve complete reports, and fail on explicit invariants."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, default=DEFAULT_SEEDS)
    parser.add_argument("--output", type=Path, default=REPOSITORY / OUTPUT_DIRECTORY)
    args = parser.parse_args(argv)
    if args.seeds < 1:
        parser.error("--seeds must be positive")
    args.output.mkdir(parents=True, exist_ok=True)
    config = BaselineAdvisorConfig()
    rows, reports, curves = [], [], {}
    start = perf_counter()
    for scenario in SCENARIOS:
        for seed in range(FIRST_SEED, FIRST_SEED + args.seeds):
            row, report, example = validate_case(scenario, seed, config)
            rows.append(row)
            reports.append({"scenario": scenario, "seed": seed, "report": report})
            if seed == FIRST_SEED:
                curves[scenario] = example
        count = sum(row["status"] == "recommended" for row in rows if row["scenario"] == scenario)
        print(f"{scenario}: {count}/{args.seeds} recommended", flush=True)
    checks = invariance_checks(config)
    for scenario in ("dense_events", "cut_gaps", "strong_drift", "flat_signal", "late_scale_change"):
        checks[scenario + "_abstains"] = all(
            row["status"] != "recommended" for row in rows if row["scenario"] == scenario)
    summary = summarize(rows)
    metadata = {"method": "baseline_advisor_v1", "first_seed": FIRST_SEED,
                "seeds_per_scenario": args.seeds, "sample_rate_hz": SAMPLE_RATE_HZ,
                "recording_seconds": RECORDING_SECONDS, "policy": asdict(config),
                "checks": checks, "summary": summary, "total_runtime_s": perf_counter() - start,
                "interpretation": "Engineering reliability diagnostics; not a neural effect accuracy or false-positive-rate benchmark."}
    write_csv(args.output / "cases.csv", rows)
    write_csv(args.output / "summary.csv", summary)
    (args.output / "summary.json").write_text(json.dumps(metadata, indent=2, allow_nan=False), encoding="utf-8")
    (args.output / "complete_reports.json").write_text(json.dumps(reports, indent=2, allow_nan=False), encoding="utf-8")
    plot_validation(rows, summary, curves, args.output)
    (args.output / "README.md").write_text(
        "# Baseline advisor validation\n\n"
        "Run `python scripts/validate_baseline_advisor.py --seeds 10` to reproduce. "
        "All recordings are generated from fixed, documented random seeds. Raw user data are never accessed.\n\n"
        "`cases.csv` retains individual runs, timings, hashes, rejection reasons, and descriptive null curves. "
        "`summary.csv` and `summary.json` aggregate repeats; `complete_reports.json` preserves every ranked candidate. "
        "Figures are provided as PNG, PDF, and SVG.\n\n"
        "The fixed [-1, 0] s baseline reproduces the previous one-second choice and includes the alignment boundary. "
        "It has no event-protection guard. The direct descriptive diagnostics deliberately retain that distinction. "
        "Runtime for its simple diagnostics is recorded separately from the advisor's search and validation.\n\n"
        "Effective sample counts are approximate ACF diagnostics, not counts of independent animals or confidence intervals. "
        "Reported pre/post differences are not hypothesis tests. Repeated simulations cannot establish an optimal "
        "baseline for arbitrary recordings, and independent biological validation remains necessary.\n\n"
        "Recommendations are ranked using earlier event-free reference samples and tested once on later events, "
        "with a full lookback purge. Rejection on that held-out block causes abstention. "
        "Thresholds are unchanged across scenarios and seeds.\n",
        encoding="utf-8",
    )
    print(json.dumps({"checks": checks, "runtime_s": metadata["total_runtime_s"],
                      "output": str(args.output)}, indent=2), flush=True)
    assert all(checks.values()), "A declared validation check failed; inspect summary.json."
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
