"""Validate postprocessing against real recordings without changing source data.

Run in an IDE or with ``conda run -n pyBer python scripts/validate_postprocessing.py``.
The default data folder is the user-requested ``_test/_data``. If absent, the
known sibling ``D:/Apps/pyBer_test_data`` is used and explicitly recorded.
Use identical arguments with ``--label before`` and ``--label after`` to compare.
All generated files live below --output-dir; application settings and autosaves
are redirected to a temporary directory for the lifetime of the Qt session.
"""

from __future__ import annotations

import argparse
from contextlib import ExitStack
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
import time
from unittest.mock import patch

# Analysis and figure controls are intentionally collected here for easy editing.
ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA = ROOT / "_test" / "_data"
FALLBACK_DATA = Path("D:/Apps/pyBer_test_data")
DEFAULT_OUTPUT = ROOT / "_test" / "postprocessing_validation"
BEHAVIOR = "social_contacts"
PRE_SECONDS, POST_SECONDS = 3.0, 6.0
BASELINE_START, BASELINE_END = -2.0, 0.0
RESAMPLE_HZ, SMOOTH_SECONDS = 50.0, 0.08
FIGURE_SIZE = (8.0, 5.8)
FIGURE_DPI = 180
FIGURE_COLOR = "#167d9a"
HEATMAP_CMAP = "magma"
WINDOW_SIZE = (1600, 1000)

os.environ.setdefault("QT_QPA_PLATFORM", "windows" if os.name == "nt" else "offscreen")
os.environ.setdefault("PYBER_SMOKE_TEST", "1")
os.environ.setdefault("PYTHONNOUSERSITE", "1")
sys.path.insert(0, str(ROOT / "pyBer"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PySide6 import QtCore, QtWidgets
from gui_postprocessing import PostProcessingPanel
from main import MainWindow
from styles import apply_app_palette


def pump(app: QtWidgets.QApplication, seconds: float = 0.3) -> None:
    """Process pending Qt events so layout and plot captures are complete."""
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        app.processEvents()
        time.sleep(0.01)


def set_combo(combo: QtWidgets.QComboBox, value: str) -> None:
    """Select an exact visible choice and fail clearly when the API changed."""
    index = combo.findText(value)
    if index < 0:
        raise ValueError(f"Missing choice {value!r}: {[combo.itemText(i) for i in range(combo.count())]}")
    combo.setCurrentIndex(index)


def digest(path: Path) -> str:
    """Hash inputs before and after validation to prove raw files were untouched."""
    checksum = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            checksum.update(chunk)
    return checksum.hexdigest()


def find_recordings(data_dir: Path) -> list[tuple[Path, Path]]:
    """Find genuine processed CSV/H5 pairs with matching binary behavior tables."""
    pairs = []
    for processed in sorted(data_dir.rglob("*_AIN01.csv")):
        trial = processed.stem.removesuffix("_AIN01")
        candidates = [processed.parent / f"{trial}_behavior_with_time.csv",
                      processed.parent / f"{trial}_with_time.csv"]
        behavior = next((path for path in candidates if path.is_file()), None)
        if behavior is not None:
            pairs.append((processed, behavior))
    if not pairs:
        raise FileNotFoundError(f"No processed *_AIN01.csv with matching behavior table in {data_dir}")
    return pairs


def save_figures(matrix: np.ndarray, tvec: np.ndarray, target: Path, title: str) -> None:
    """Render descriptive event summaries, with pointwise finite observation counts.

    Shading is event SEM, not an animal-level inferential confidence interval.
    Repeated events are not treated as independent biological replicates.
    """
    finite = np.isfinite(matrix)
    counts = finite.sum(axis=0)
    mean = np.divide(np.nansum(matrix, axis=0), counts,
                     out=np.full(tvec.shape, np.nan), where=counts > 0)
    sumsquares = np.nansum(np.square(matrix - mean), axis=0)
    variance = np.divide(sumsquares, counts - 1,
                         out=np.full(tvec.shape, np.nan), where=counts > 1)
    sem = np.sqrt(np.divide(variance, counts,
                            out=np.full(tvec.shape, np.nan), where=counts > 1))
    pd.DataFrame({"time_s": tvec, "mean": mean, "event_sem": sem,
                  "finite_events": counts}).to_csv(target.with_name(target.name + "_average.csv"), index=False)
    plt.rcParams.update({"font.size": 9, "axes.spines.top": False,
                         "axes.spines.right": False, "savefig.facecolor": "white"})
    fig, axes = plt.subplots(2, 1, figsize=FIGURE_SIZE, constrained_layout=True,
                             gridspec_kw={"height_ratios": [1, 1.5]})
    axes[0].plot(tvec, mean, color=FIGURE_COLOR, linewidth=1.4)
    axes[0].fill_between(tvec, mean - sem, mean + sem, color=FIGURE_COLOR, alpha=0.2, linewidth=0)
    axes[0].set(title=title, ylabel="Processed signal", xlabel="Time from event (s)")
    image = axes[1].imshow(matrix, aspect="auto", origin="lower", cmap=HEATMAP_CMAP,
                            extent=(tvec[0], tvec[-1], 0.5, matrix.shape[0] + 0.5))
    axes[1].set(xlabel="Time from event (s)", ylabel="Event")
    fig.colorbar(image, ax=axes[1], label="Processed signal", fraction=0.035, pad=0.02)
    for axis in axes:
        axis.axvline(0, color="#596574", linewidth=0.8, linestyle=":")
    for extension in ("png", "pdf", "svg"):
        fig.savefig(target.with_suffix(f".{extension}"), dpi=FIGURE_DPI)
    plt.close(fig)


def validate_interactions(app, window, pairs, target_dir: Path) -> bool:
    """Check real GUI transitions and capture each coordinated plot palette.

    These checks exercise signal wiring rather than calling the compute method
    after every control change. Failed checks are written before raising, so a
    broken workflow remains reviewable even when validation does not complete.
    """
    post = window.post_tab
    checks = []

    def record(name, passed, detail):
        """Append one explicit assertion result and preserve it immediately."""
        checks.append({"check": name, "passed": bool(passed), "detail": str(detail)})
        pd.DataFrame(checks).to_csv(target_dir / "workflow_checks.csv", index=False)
        print(f"  workflow {name}: {'PASS' if passed else 'FAIL'} ({detail})", flush=True)

    def settle():
        """Allow the debounced GUI computation to run without manual compute."""
        pump(app, 0.65)
        deadline = time.monotonic() + 10
        while (post._psth_timer.isActive() or post._psth_computing) and time.monotonic() < deadline:
            pump(app, 0.1)

    if not hasattr(post, "combo_psth_normalization"):
        record("interactive_controls_available", False, "Normalization control missing")
        return False
    old_columns = post._last_mat.shape[1]
    post.spin_post.setValue(POST_SECONDS + 1)
    settle()
    record("window_change_recomputes", post._last_mat is not None and post._last_mat.shape[1] > old_columns,
           f"columns {old_columns} -> {None if post._last_mat is None else post._last_mat.shape[1]}")
    post.spin_post.setValue(POST_SECONDS)
    settle()
    normalization_results = {}
    for mode in ("none", "subtract", "zscore"):
        post.combo_psth_normalization.setCurrentIndex(post.combo_psth_normalization.findData(mode))
        settle()
        matrix = np.asarray(post._last_mat) if post._last_mat is not None else np.array([])
        valid = matrix.ndim == 2 and matrix.size > 0 and np.isfinite(matrix).any()
        record(f"normalization_{mode}", valid, f"matrix shape {matrix.shape}")
        if valid:
            normalization_results[mode] = matrix.copy()
    if len(normalization_results) == 3:
        record("normalizations_change_values", not np.allclose(normalization_results["none"], normalization_results["subtract"], equal_nan=True)
               and not np.allclose(normalization_results["subtract"], normalization_results["zscore"], equal_nan=True),
               "Original, subtract and z-score yield distinct values")

    # Save the same real trace, window and event matrix for every color preset.
    for preset in ("Midnight", "Paper", "Sand"):
        set_combo(post.combo_plot_preset, preset)
        settle()
        path = target_dir / f"preset_{preset.lower()}_gui.png"
        record(f"preset_{preset.lower()}_capture", window.grab().save(str(path), "PNG"), path.name)

    # Layout and contrast are display choices and must preserve numerical data.
    reference_matrix = np.array(post._last_mat, copy=True)
    set_combo(post.combo_plot_preset, "Midnight")
    for index in range(post.combo_view_layout.count()):
        post.combo_view_layout.setCurrentIndex(index)
        settle()
        layout = post.combo_view_layout.currentText()
        unchanged = post._last_mat is not None and np.array_equal(post._last_mat, reference_matrix, equal_nan=True)
        path = target_dir / f"layout_{layout.lower().replace(' ', '_')}_gui.png"
        record(f"layout_{layout}", unchanged and window.grab().save(str(path), "PNG"), "Numerical matrix preserved; screenshot saved")
        if layout == "All" and hasattr(post, "_results_scroll"):
            scrollbar = post._results_scroll.verticalScrollBar()
            scrollbar.setValue(scrollbar.maximum())
            settle()
            window.grab().save(str(target_dir / "layout_all_lower_gui.png"), "PNG")
            scrollbar.setValue(0)
    set_combo(post.combo_view_layout, "Standard")
    for index in range(post.combo_heat_scale.count()):
        post.combo_heat_scale.setCurrentIndex(index)
        settle()
        unchanged = post._last_mat is not None and np.array_equal(post._last_mat, reference_matrix, equal_nan=True)
        record(f"contrast_{post.combo_heat_scale.currentText()}", unchanged, "Display change preserves numerical matrix")
    post.combo_heat_scale.setCurrentIndex(0)
    post.btn_fit_psth.click()
    settle()
    record("fit_plots_preserves_data", np.array_equal(post._last_mat, reference_matrix, equal_nan=True), "Fit plots button clicked")

    # Two recordings have distinct event counts, making aggregation testable.
    trials = [post._load_processed_csv(str(path)) for path, _ in pairs]
    post.receive_current_processed(trials)
    post._load_behavior_paths([str(path) for _, path in pairs], replace=True)
    post._refresh_behavior_list()
    set_combo(post.combo_behavior_name, BEHAVIOR)
    post.tab_sources.setCurrentIndex(1)
    post.tab_visual_mode.setCurrentIndex(1)
    post.cb_exclude_low_event_animals.setChecked(False)
    post.cb_group_keep_trials.setChecked(False)
    settle()
    animal_rows = 0 if post._last_mat is None else post._last_mat.shape[0]
    record("group_animal_rows", animal_rows == len(trials), f"{animal_rows} rows for {len(trials)} recordings")
    post.cb_group_keep_trials.setChecked(True)
    settle()
    trial_rows = 0 if post._last_mat is None else post._last_mat.shape[0]
    expected_trials = sum(item[1].shape[0] for item in post._per_file_mats.values())
    record("group_trial_rows", trial_rows == expected_trials and trial_rows > animal_rows,
           f"{trial_rows} displayed, {expected_trials} per-file total")
    post.cb_exclude_low_event_animals.setChecked(True)
    post.spin_min_events_per_animal.setValue(1000000)
    settle()
    record("all_excluded_group_clears", post._last_mat is None or post._last_mat.size == 0,
           f"excluded files: {len(post._psth_excluded_files)}")
    post.tab_visual_mode.setCurrentIndex(0)
    settle()
    record("individual_recovers_excluded_file", post._last_mat is not None and post._last_mat.size > 0,
           f"selected {post.combo_individual_file.currentText()}")
    counts = []
    for index in range(post.combo_individual_file.count()):
        post.combo_individual_file.setCurrentIndex(index)
        settle()
        fid = post.combo_individual_file.currentText()
        expected = post._per_file_mats.get(fid)
        actual = post._last_mat
        passed = expected is not None and actual is not None and np.allclose(actual, expected[1], equal_nan=True)
        record(f"individual_selection_{fid}", passed, f"rows {None if actual is None else actual.shape[0]}")
        if actual is not None:
            counts.append(actual.shape[0])
    record("selection_exercised_multiple_files", len(counts) == len(trials), f"row counts: {counts}")
    # Exercise actual publication rendering, including single-event missing SEM.
    # Both cases use real events selected through the normal GUI filter.
    post.tab_sources.setCurrentIndex(0)
    post.cb_exclude_low_event_animals.setChecked(False)
    settle()
    for name, event_end in (("all_events", 0), ("single_event", 1)):
        post.spin_event_end.setValue(event_end)
        settle()
        prefix = f"smoke_{name}"
        try:
            post._export_publication_figure(str(target_dir), prefix, "Heatmap + Avg PSTH + Metrics")
            paths = [target_dir / f"{prefix}_publication_figure.{extension}" for extension in ("png", "pdf", "svg")]
            passed = all(path.is_file() and path.stat().st_size > 0 for path in paths)
            record(f"publication_{name}", passed, "All behaviors exported to PNG/PDF/SVG")
        except Exception as error:
            record(f"publication_{name}", False, f"{type(error).__name__}: {error}")
    post.spin_event_end.setValue(0)
    settle()
    return all(row["passed"] for row in checks)


def validate(data_dir: Path, output_dir: Path, label: str) -> None:
    """Exercise actual loaders and onset/offset GUI analysis for available files."""
    requested = data_dir.resolve()
    if not data_dir.is_dir():
        if requested == DEFAULT_DATA.resolve() and FALLBACK_DATA.is_dir():
            print(f"Requested data folder absent: {requested}. Using real fallback: {FALLBACK_DATA}", flush=True)
            data_dir = FALLBACK_DATA
        else:
            raise FileNotFoundError(data_dir)
    pairs = find_recordings(data_dir)
    target_dir = output_dir / label
    target_dir.mkdir(parents=True, exist_ok=True)
    inputs = sorted({path for pair in pairs for path in pair} |
                    {csv.with_suffix(".h5") for csv, _ in pairs if csv.with_suffix(".h5").is_file()})
    hashes_before = {str(path): digest(path) for path in inputs}
    code_hashes = {str(path.relative_to(ROOT)): digest(path) for path in
                   (ROOT / "pyBer" / "gui_postprocessing.py", ROOT / "pyBer" / "analysis_core.py",
                    ROOT / "pyBer" / "postprocessing_core.py") if path.is_file()}
    metrics, loader_checks = [], []
    workflows_passed = True
    # Temporary paths prevent validation from changing normal preferences/layout.
    with tempfile.TemporaryDirectory(prefix="pyber-post-validation-") as temporary, ExitStack() as stack:
        QtCore.QSettings.setDefaultFormat(QtCore.QSettings.Format.IniFormat)
        QtCore.QSettings.setPath(QtCore.QSettings.Format.IniFormat, QtCore.QSettings.Scope.UserScope, temporary)
        stack.enter_context(patch.object(MainWindow, "_panel_config_json_path", return_value=str(Path(temporary) / "panel_layout.json")))
        stack.enter_context(patch.object(PostProcessingPanel, "_autosave_project_cache_path", return_value=str(Path(temporary) / "autosave.h5")))
        stack.enter_context(patch.object(PostProcessingPanel, "_restore_project_autosave_if_needed", return_value=None))
        app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
        app.setApplicationName("pyBer postprocessing validation")
        apply_app_palette(app, "dark")
        window = MainWindow()
        window.resize(*WINDOW_SIZE)
        window.show()
        pump(app)
        post = window.post_tab
        window.tabs.setCurrentWidget(post)
        # Toasts are temporary UI state and would obscure reproducible captures.
        toaster = getattr(window, "_toaster", None)
        if toaster is not None:
            for toast in list(getattr(toaster, "_toasts", [])):
                toast.close()
        window._toaster = None
        for csv_path, behavior_path in pairs:
            print(f"Validating {csv_path.name}", flush=True)
            csv_trial = post._load_processed_csv(str(csv_path))
            if csv_trial is None:
                raise RuntimeError(f"CSV loader failed: {csv_path}")
            h5_path = csv_path.with_suffix(".h5")
            if h5_path.is_file():
                h5_trial = post._load_processed_h5(str(h5_path))
                if h5_trial is None:
                    raise RuntimeError(f"H5 loader failed: {h5_path}")
                for field in ("time", "output"):
                    left, right = np.asarray(getattr(csv_trial, field)), np.asarray(getattr(h5_trial, field))
                    equal_shape = left.shape == right.shape
                    finite = np.isfinite(left) & np.isfinite(right) if equal_shape else np.array([], bool)
                    max_difference = float(np.max(np.abs(left[finite] - right[finite]))) if finite.any() else np.nan
                    loader_checks.append({"file": csv_path.stem, "field": field,
                                          "csv_samples": left.size, "h5_samples": right.size,
                                          "same_shape": equal_shape, "max_absolute_difference": max_difference,
                                          "same_nan_mask": bool(np.array_equal(np.isnan(left), np.isnan(right))) if equal_shape else False})
            post.receive_current_processed([csv_trial])
            post._load_behavior_paths([str(behavior_path)], replace=True)
            post._refresh_behavior_list()
            set_combo(post.combo_align, "Behavior (CSV/XLSX)")
            set_combo(post.combo_behavior_name, BEHAVIOR)
            for widget, value in ((post.spin_pre, PRE_SECONDS), (post.spin_post, POST_SECONDS),
                                  (post.spin_b0, BASELINE_START), (post.spin_b1, BASELINE_END),
                                  (post.spin_resample, RESAMPLE_HZ), (post.spin_smooth, SMOOTH_SECONDS)):
                widget.setValue(value)
            for alignment in ("onset", "offset"):
                set_combo(post.combo_behavior_align, f"Align to {alignment}")
                started = time.perf_counter()
                post._compute_psth()
                elapsed = time.perf_counter() - started
                if post._last_mat is None or post._last_tvec is None:
                    raise RuntimeError(f"No PSTH for {csv_path.name}, {alignment}")
                matrix, tvec = np.array(post._last_mat, copy=True), np.array(post._last_tvec, copy=True)
                events = np.asarray(post._last_events)
                stem = target_dir / f"{csv_path.stem}_{BEHAVIOR}_{alignment}"
                pd.DataFrame(matrix, columns=[f"{t:.9g}" for t in tvec]).to_csv(stem.with_name(stem.name + "_matrix.csv"), index=False)
                pd.DataFrame(post._last_event_rows).to_csv(stem.with_name(stem.name + "_events.csv"), index=False)
                save_figures(matrix, tvec, stem, f"{csv_path.stem}: {BEHAVIOR}, {alignment}")
                metrics.append({"file": csv_path.stem, "alignment": alignment,
                                "events": events.size, "rows": matrix.shape[0], "columns": matrix.shape[1],
                                "finite_fraction": float(np.isfinite(matrix).mean()),
                                "mean": float(np.nanmean(matrix)), "std": float(np.nanstd(matrix)),
                                "compute_seconds": elapsed, "input_samples": np.asarray(csv_trial.time).size})
                set_combo(post.combo_view_layout, "Standard")
                post.plot_trace.autoRange()
                post._force_hide_post_drawer_initially()
                pump(app, 0.5)
                screenshot = stem.with_name(stem.name + "_gui.png")
                if not window.grab().save(str(screenshot), "PNG"):
                    raise RuntimeError(f"Screenshot failed: {screenshot}")
                print(f"  {alignment}: {matrix.shape}, finite={np.isfinite(matrix).mean():.4f}, {elapsed:.3f}s", flush=True)
        if label != "before":
            workflows_passed = validate_interactions(app, window, pairs, target_dir)
        # Closing happens while all persistence redirections remain active.
        post._project_dirty = False
        window.close()
        pump(app)
    hashes_after = {str(path): digest(path) for path in inputs}
    if hashes_after != hashes_before:
        raise RuntimeError("Input integrity failed: an input changed during validation")
    pd.DataFrame(metrics).to_csv(target_dir / "metrics.csv", index=False)
    pd.DataFrame(loader_checks).to_csv(target_dir / "loader_comparison.csv", index=False)
    (target_dir / "manifest.json").write_text(json.dumps({
        "label": label, "requested_data_dir": str(requested), "actual_data_dir": str(data_dir.resolve()),
        "code_sha256_at_start": code_hashes,
        "input_sha256": hashes_before, "inputs_unchanged": True,
        "analysis": {"behavior": BEHAVIOR, "pre_s": PRE_SECONDS, "post_s": POST_SECONDS,
                     "baseline_start_s": BASELINE_START, "baseline_end_s": BASELINE_END,
                     "resample_hz": RESAMPLE_HZ, "smooth_s": SMOOTH_SECONDS},
        "statistics": "Descriptive event mean and pointwise SEM only; events are repeated observations, not independent animals."
    }, indent=2), encoding="utf-8")
    # A paired file comparison makes any numerical change visible after edits.
    baseline_dir = output_dir / "before"
    if label != "before" and (baseline_dir / "metrics.csv").is_file():
        baseline = pd.read_csv(baseline_dir / "metrics.csv")
        comparison = baseline.merge(pd.DataFrame(metrics), on=["file", "alignment"], suffixes=("_before", "_after"))
        for field in ("events", "rows", "columns", "finite_fraction", "mean", "std", "compute_seconds"):
            comparison[field + "_change"] = comparison[field + "_after"] - comparison[field + "_before"]
        comparison.to_csv(target_dir / "before_after_comparison.csv", index=False)
        compare_matrices(baseline_dir, target_dir)
    if not workflows_passed:
        raise AssertionError(f"GUI workflow checks failed; inspect {target_dir / 'workflow_checks.csv'}")
    print(f"Validation complete: {target_dir.resolve()}", flush=True)


def compare_matrices(baseline_dir: Path, target_dir: Path) -> None:
    """Compare every exported number and plot mean overlays for direct inspection.

    Aggregate means alone can hide cancelling errors. This check therefore records
    shape, time-grid, finite-mask and maximum pointwise differences separately.
    Runtime values are single descriptive timings, not a performance claim.
    """
    checks = []
    candidates = sorted(target_dir.glob("*_matrix.csv"))
    fig, axes = plt.subplots(len(candidates), 1, figsize=(8, 2.1 * len(candidates)),
                             constrained_layout=True, squeeze=False)
    for axis, current in zip(axes[:, 0], candidates):
        previous = baseline_dir / current.name
        if not previous.is_file():
            axis.set_visible(False)
            continue
        old_frame, new_frame = pd.read_csv(previous), pd.read_csv(current)
        old, new = old_frame.to_numpy(float), new_frame.to_numpy(float)
        same_shape = old.shape == new.shape
        old_time, new_time = np.asarray(old_frame.columns, float), np.asarray(new_frame.columns, float)
        same_grid = old_time.shape == new_time.shape and np.allclose(old_time, new_time, atol=1e-10, rtol=0)
        comparable = same_shape and same_grid
        common = np.isfinite(old) & np.isfinite(new) if comparable else np.array([], bool)
        difference = np.abs(old[common] - new[common]) if common.any() else np.array([])
        checks.append({"matrix": current.name, "same_shape": same_shape, "same_time_grid": same_grid,
                       "max_time_difference_s": float(np.max(np.abs(old_time - new_time))) if old_time.shape == new_time.shape else np.nan,
                       "same_finite_mask": bool(np.array_equal(np.isfinite(old), np.isfinite(new))) if same_shape else False,
                       "max_absolute_difference": float(difference.max()) if difference.size else np.nan,
                       "rms_difference": float(np.sqrt(np.mean(difference ** 2))) if difference.size else np.nan})
        for frame, color, name, style in ((old_frame, "#a55f22", "Before", "-"),
                                          (new_frame, FIGURE_COLOR, "After", "--")):
            axis.plot(np.asarray(frame.columns, float), frame.mean(axis=0).to_numpy(),
                      color=color, linewidth=1.2, linestyle=style, label=name)
        axis.set(title=current.stem.removesuffix("_matrix"), ylabel="Mean signal")
        axis.axvline(0, color="#596574", linewidth=0.8, linestyle=":")
        axis.legend(frameon=False, fontsize=8, loc="upper right")
    axes[-1, 0].set_xlabel("Time from event (s)")
    pd.DataFrame(checks).to_csv(target_dir / "matrix_comparison.csv", index=False)
    for extension in ("png", "pdf", "svg"):
        fig.savefig(target_dir / f"before_after_means.{extension}", dpi=FIGURE_DPI)
    plt.close(fig)


def main() -> None:
    """Parse reproducible input/output locations and run the real-data checks."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--label", default="after")
    args = parser.parse_args()
    if Path(args.label).name != args.label or args.label in {".", ".."}:
        parser.error("--label must be a simple folder name")
    validate(args.data_dir, args.output_dir, args.label)


if __name__ == "__main__":
    main()
