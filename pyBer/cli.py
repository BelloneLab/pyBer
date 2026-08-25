"""Headless batch preprocessing command line interface for pyBer."""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
import traceback
from dataclasses import fields
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from analysis_core import (
    ExportSelection,
    LoadedDoricFile,
    PhotometryProcessor,
    ProcessingParams,
    export_processed_csv,
    export_processed_h5,
    is_rwd_events_csv,
    load_rwd_csv,
    recommend_preprocessing_settings,
)
from sensor_registry import SENSOR_UNKNOWN, all_sensors, get_sensor
from version import __version__


RAW_EXTENSIONS = frozenset({".doric", ".h5", ".hdf5", ".csv"})
SKIP_DIRS = frozenset({".git", ".pytest_cache", "__pycache__", "build", "dist", "release", "pyber_processed"})
RWD_STEMS = ("fluorescence", "fluorescence-unaligned")


def _is_rwd_name(path: Path) -> bool:
    stem = path.stem.lower()
    return any(stem == base or stem.startswith(base + "_") for base in RWD_STEMS)


def discover_raw_files(inputs: Sequence[str], recursive: bool = True) -> List[Path]:
    """Resolve files and recursively find supported raw recordings in folders."""
    found: List[Path] = []
    seen: set[str] = set()

    def add(path: Path, *, explicit: bool = False, root_level: bool = False) -> None:
        if not path.is_file() or path.suffix.lower() not in RAW_EXTENSIONS:
            return
        if path.suffix.lower() == ".csv":
            if is_rwd_events_csv(str(path)):
                return
            sidecar = path.with_suffix(".pyber.json")
            if sidecar.is_file():
                return
            if not (explicit or root_level or _is_rwd_name(path)):
                return
        if path.suffix.lower() in {".h5", ".hdf5"}:
            try:
                import h5py

                with h5py.File(path, "r") as handle:
                    if str(handle.attrs.get("pyber_format", "")) == "processed_trace":
                        return
            except OSError:
                pass
        key = os.path.normcase(str(path.resolve()))
        if key not in seen:
            seen.add(key)
            found.append(path.resolve())

    for raw in inputs:
        path = Path(raw).expanduser().resolve()
        if path.is_file():
            add(path, explicit=True)
            continue
        if not path.is_dir():
            raise FileNotFoundError(f"Input does not exist: {raw}")
        if recursive:
            for dirpath, dirnames, filenames in os.walk(path):
                dirnames[:] = [
                    name for name in sorted(dirnames)
                    if name.lower() not in SKIP_DIRS and not name.startswith(".")
                ]
                current = Path(dirpath)
                for name in sorted(filenames):
                    add(current / name, root_level=(current == path))
        else:
            for child in sorted(path.iterdir()):
                add(child, root_level=True)
    return sorted(found, key=lambda item: str(item).lower())


def resolve_sensor(value: str) -> str:
    """Resolve a sensor ID or display name, case-insensitively."""
    text = str(value or SENSOR_UNKNOWN).strip()
    by_id = {sensor.sensor_id.lower(): sensor.sensor_id for sensor in all_sensors()}
    by_name = {sensor.name.lower(): sensor.sensor_id for sensor in all_sensors()}
    if text.lower() in by_id:
        return by_id[text.lower()]
    if text.lower() in by_name:
        return by_name[text.lower()]
    choices = ", ".join(sensor.sensor_id for sensor in all_sensors())
    raise ValueError(f"Unknown sensor '{value}'. Valid sensor IDs include: {choices}")


def load_raw_file(path: Path, processor: Optional[PhotometryProcessor] = None) -> LoadedDoricFile:
    processor = processor or PhotometryProcessor()
    if path.suffix.lower() == ".csv":
        loaded = load_rwd_csv(str(path))
        if loaded is None:
            raise ValueError(
                "CSV is not a recognized RWD fluorescence export. Use an aligned "
                "CH*-410/CH*-470 table or an unaligned TimeStamp/Lights table."
            )
        return loaded
    return processor.load_file(str(path))


def _parse_bool(value: str) -> bool:
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "on"}:
        return True
    if text in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"Expected a boolean, got '{value}'")


def _coerce_param(name: str, value: str, template: ProcessingParams) -> Any:
    current = getattr(template, name)
    if isinstance(current, bool):
        return _parse_bool(value)
    if isinstance(current, int) and not isinstance(current, bool):
        return int(value)
    if isinstance(current, float):
        return float(value)
    return str(value)


def apply_overrides(params: ProcessingParams, overrides: Sequence[str]) -> ProcessingParams:
    """Apply repeatable NAME=VALUE overrides to recommended parameters."""
    valid = {item.name for item in fields(ProcessingParams)}
    for item in overrides:
        if "=" not in item:
            raise ValueError(f"Parameter override must be NAME=VALUE, got '{item}'")
        name, raw_value = item.split("=", 1)
        name = name.strip().replace("-", "_")
        if name not in valid:
            raise ValueError(f"Unknown processing parameter '{name}'")
        setattr(params, name, _coerce_param(name, raw_value.strip(), params))
    return params


def load_override_file(path: Optional[str]) -> List[str]:
    if not path:
        return []
    source = Path(path).expanduser()
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except Exception as exc:
        raise ValueError(f"Could not read parameter JSON '{source}': {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError("Parameter JSON must contain one object of ProcessingParams fields")
    return [f"{key}={value}" for key, value in payload.items()]


def _requested_channels(values: Sequence[str]) -> List[str]:
    out: List[str] = []
    for value in values:
        out.extend(part.strip() for part in value.split(",") if part.strip())
    return out


def select_channels(available: Sequence[str], requested: Sequence[str]) -> List[str]:
    if not requested:
        return list(available)
    lookup = {name.lower(): name for name in available}
    missing = [name for name in requested if name.lower() not in lookup]
    if missing:
        raise ValueError(
            f"Requested channel(s) {', '.join(missing)} not found. Available: {', '.join(available)}"
        )
    return [lookup[name.lower()] for name in requested]


def _safe_component(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("._")
    return cleaned or "recording"


def _output_base(path: Path, channel: str, output_dir: Path, common_root: Path) -> Path:
    try:
        relative_parent = path.parent.relative_to(common_root)
    except ValueError:
        relative_parent = Path()
    target = output_dir / relative_parent
    target.mkdir(parents=True, exist_ok=True)
    return target / f"{_safe_component(path.stem)}_{_safe_component(channel)}"


def evaluate_qc(processed: Any, recommendation: Any) -> Dict[str, Any]:
    """Return a conservative, transparent CLI quality verdict."""
    reasons: List[str] = []
    warnings: List[str] = []
    metrics = dict(getattr(recommendation, "metrics", {}) or {})
    artifact_fraction = float(metrics.get("artifact_fraction", 0.0) or 0.0)
    output = np.asarray(getattr(processed, "output", []), float)
    finite_fraction = float(np.mean(np.isfinite(output))) if output.size else 0.0
    raw = np.asarray(getattr(processed, "raw_signal", []), float)
    raw_finite = raw[np.isfinite(raw)]
    raw_variation = float(np.nanstd(raw_finite)) if raw_finite.size else 0.0

    if artifact_fraction >= 0.08:
        reasons.append(f"artifact burden is {artifact_fraction:.1%} (fail threshold 8%)")
    elif artifact_fraction >= 0.02:
        warnings.append(f"artifact burden is {artifact_fraction:.1%} (review threshold 2%)")
    if finite_fraction < 0.95:
        reasons.append(f"only {finite_fraction:.1%} of processed samples are finite")
    if raw_variation <= 1e-12:
        reasons.append("raw signal is flat or has no measurable variance")
    confidence = float(getattr(recommendation, "confidence", 0.0) or 0.0)
    if confidence < 0.50:
        warnings.append(f"automatic recommendation confidence is low ({confidence:.0%})")
    sensor_check = dict(getattr(processed, "sensor_check", {}) or {})
    if str(sensor_check.get("status", "")).lower() == "warn":
        warnings.append(str(sensor_check.get("message", "sensor/trace mismatch")))
    for warning in getattr(recommendation, "warnings", []) or []:
        if warning and warning not in warnings:
            warnings.append(str(warning))

    flagged = bool(reasons or warnings)
    tier = "FAIL" if reasons else ("REVIEW" if warnings else "PASS")
    return {
        "tier": tier,
        "flagged": flagged,
        "reasons": reasons,
        "warnings": warnings,
        "artifact_fraction": artifact_fraction,
        "finite_fraction": finite_fraction,
        "recommendation_confidence": confidence,
    }


def save_preprocessing_figure(
    path: Path,
    trial: Any,
    processed: Any,
    recommendation: Any,
    qc: Dict[str, Any],
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    raw_t = np.asarray(getattr(processed, "raw_display_time", None), float)
    raw_sig = np.asarray(getattr(processed, "raw_display_signal", None), float)
    raw_ref = np.asarray(getattr(processed, "raw_display_reference", None), float)
    if raw_t.ndim == 0 or raw_t.size == 0:
        raw_t = np.asarray(trial.time, float)
        raw_sig = np.asarray(trial.signal_465, float)
        raw_ref = np.asarray(trial.reference_405, float)
    t = np.asarray(processed.time, float)
    output = np.asarray(processed.output, float)

    fig, axes = plt.subplots(3, 1, figsize=(13, 9), constrained_layout=True)
    axes[0].plot(raw_t, raw_sig, color="#35a7ff", lw=0.7, label="signal")
    if raw_ref.size == raw_t.size and np.isfinite(raw_ref).any():
        axes[0].plot(raw_t, raw_ref, color="#f5b942", lw=0.65, alpha=0.8, label="reference")
    for start, end in getattr(processed, "artifact_regions_sec", []) or []:
        axes[0].axvspan(start, end, color="#ef476f", alpha=0.18)
    axes[0].set(title="Raw traces and detected artifacts", ylabel="fluorescence")
    axes[0].legend(loc="upper right", frameon=False)

    axes[1].plot(t, output, color="#06d6a0", lw=0.8)
    axes[1].axhline(0.0, color="#777777", lw=0.6, alpha=0.6)
    axes[1].set(title=str(processed.output_label), ylabel="processed output")

    finite = output[np.isfinite(output)]
    if finite.size:
        axes[2].hist(finite, bins=80, color="#118ab2", alpha=0.85)
    axes[2].set(title="Processed output distribution", xlabel="value", ylabel="count")
    for axis in axes[:2]:
        axis.set_xlabel("time (s)")
        axis.grid(alpha=0.18)
    axes[2].grid(alpha=0.18)

    sensor_name = get_sensor(recommendation.params.sensor_id).name
    fig.suptitle(
        f"pyBer {__version__} preprocessing report | {Path(trial.path).name} | {trial.channel_id}\n"
        f"{sensor_name} | QC {qc['tier']} | {recommendation.summary}",
        fontsize=11,
    )
    fig.savefig(path, dpi=160, facecolor="white")
    plt.close(fig)


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _write_rows(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    fieldnames = [
        "input", "channel", "status", "qc_tier", "flagged", "sensor", "output_mode",
        "artifact_fraction", "finite_fraction", "confidence", "csv", "h5", "figure",
        "reasons", "warnings",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def run_batch(args: argparse.Namespace) -> int:
    inputs = discover_raw_files(args.inputs, recursive=not args.no_recursive)
    if not inputs:
        raise ValueError("No supported raw recordings were found")
    sensor_id = resolve_sensor(args.sensor)
    requested = _requested_channels(args.channel)
    parameter_overrides = load_override_file(args.params_file) + list(args.set)
    processor = PhotometryProcessor()

    explicit_output = Path(args.output_dir).expanduser().resolve() if args.output_dir else None
    if explicit_output is not None:
        output_dir = explicit_output
    elif len(args.inputs) == 1 and Path(args.inputs[0]).expanduser().is_dir():
        output_dir = Path(args.inputs[0]).expanduser().resolve() / "pyber_processed"
    else:
        output_dir = inputs[0].parent / "pyber_processed"
    output_dir.mkdir(parents=True, exist_ok=True)
    common_root = Path(os.path.commonpath([str(path.parent) for path in inputs]))

    rows: List[Dict[str, Any]] = []
    for source in inputs:
        try:
            loaded = load_raw_file(source, processor)
            channels = select_channels(loaded.channels, requested)
        except Exception as exc:
            rows.append({
                "input": str(source), "channel": "", "status": "ERROR", "qc_tier": "FAIL",
                "flagged": True, "sensor": sensor_id, "output_mode": "", "artifact_fraction": "",
                "finite_fraction": "", "confidence": "", "csv": "", "h5": "", "figure": "",
                "reasons": str(exc), "warnings": "",
            })
            if args.verbose:
                traceback.print_exc()
            continue

        for channel in channels:
            row: Dict[str, Any] = {"input": str(source), "channel": channel, "sensor": sensor_id}
            try:
                if args.trigger and args.trigger not in loaded.trigger_by_name:
                    available_triggers = ", ".join(sorted(loaded.trigger_by_name)) or "none"
                    raise ValueError(
                        f"Requested trigger '{args.trigger}' was not found. Available: {available_triggers}"
                    )
                trigger = args.trigger or None
                trial = loaded.make_trial(channel, trigger_name=trigger)
                base_params = ProcessingParams(sensor_id=sensor_id)
                recommendation = recommend_preprocessing_settings(trial, base_params)
                recommended_params = recommendation.params.to_dict()
                params = apply_overrides(ProcessingParams.from_dict(recommended_params), parameter_overrides)
                params.sensor_id = sensor_id
                processed = processor.process_trial(trial, params, preview_mode=False)
                qc = evaluate_qc(processed, recommendation)
                base = _output_base(source, channel, output_dir, common_root)
                metadata = {str(k): str(v) for k, v in dict(getattr(loaded, "metadata", {}) or {}).items()}
                metadata.update({"source_file": str(source), "channel": channel, "sensor_id": sensor_id})
                selection = ExportSelection()
                csv_path = base.with_suffix(".csv")
                h5_path = base.with_suffix(".h5")
                figure_path = base.with_name(base.name + "_preprocessing_report.png")
                report_path = base.with_name(base.name + "_preprocessing_report.json")
                if args.format in {"both", "csv"}:
                    export_processed_csv(str(csv_path), processed, metadata=metadata, selection=selection, params=params)
                if args.format in {"both", "h5"}:
                    export_processed_h5(str(h5_path), processed, metadata=metadata, selection=selection, params=params)
                save_preprocessing_figure(figure_path, trial, processed, recommendation, qc)
                report = {
                    "pyber_version": __version__, "input": str(source), "channel": channel,
                    "sensor": get_sensor(sensor_id).__dict__, "acquisition_metadata": metadata,
                    "recommended_parameters": recommended_params,
                    "effective_parameters": params.to_dict(), "recommendation_summary": recommendation.summary,
                    "recommendation_metrics": recommendation.metrics, "qc": qc,
                    "outputs": {"csv": str(csv_path) if args.format in {"both", "csv"} else None,
                                "h5": str(h5_path) if args.format in {"both", "h5"} else None,
                                "figure": str(figure_path)},
                }
                report_path.write_text(json.dumps(_json_safe(report), indent=2), encoding="utf-8")
                row.update({
                    "status": "OK", "qc_tier": qc["tier"], "flagged": qc["flagged"],
                    "output_mode": params.output_mode, "artifact_fraction": qc["artifact_fraction"],
                    "finite_fraction": qc["finite_fraction"], "confidence": qc["recommendation_confidence"],
                    "csv": str(csv_path) if args.format in {"both", "csv"} else "",
                    "h5": str(h5_path) if args.format in {"both", "h5"} else "", "figure": str(figure_path),
                    "reasons": "; ".join(qc["reasons"]), "warnings": "; ".join(qc["warnings"]),
                })
                print(f"[{qc['tier']}] {source} [{channel}] -> {base}")
            except Exception as exc:
                row.update({
                    "status": "ERROR", "qc_tier": "FAIL", "flagged": True, "output_mode": "",
                    "artifact_fraction": "", "finite_fraction": "", "confidence": "", "csv": "",
                    "h5": "", "figure": "", "reasons": str(exc), "warnings": "",
                })
                if args.verbose:
                    traceback.print_exc()
            rows.append(row)

    _write_rows(output_dir / "batch_summary.csv", rows)
    flagged = [row for row in rows if bool(row.get("flagged")) or row.get("status") != "OK"]
    _write_rows(output_dir / "flagged_recordings.csv", flagged)
    (output_dir / "batch_summary.json").write_text(
        json.dumps(_json_safe({"pyber_version": __version__, "records": rows}), indent=2), encoding="utf-8"
    )
    print(f"Processed {sum(row.get('status') == 'OK' for row in rows)}/{len(rows)} recording-channel pairs")
    print(f"Flagged for review: {len(flagged)}. Reports: {output_dir}")
    return 2 if any(row.get("status") != "OK" for row in rows) else 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="pyber-cli",
        description="Recursively preprocess Doric and RWD fiber-photometry recordings.",
    )
    parser.add_argument("inputs", nargs="+", help="One or more raw files or folders")
    parser.add_argument("-o", "--output-dir", help="Output folder (default: pyber_processed)")
    parser.add_argument("--sensor", default=SENSOR_UNKNOWN, help="Sensor ID or exact sensor name")
    parser.add_argument("--channel", action="append", default=[], help="Channel name; repeat or comma-separate")
    parser.add_argument("--trigger", default="", help="DIO/AOUT trigger channel to include")
    parser.add_argument("--set", action="append", default=[], metavar="NAME=VALUE",
                        help="Override any ProcessingParams field after recommendations; repeatable")
    parser.add_argument("--params-file", help="JSON object of ProcessingParams overrides; --set wins")
    parser.add_argument("--format", choices=("both", "csv", "h5"), default="both")
    parser.add_argument("--no-recursive", action="store_true", help="Do not descend into input folders")
    parser.add_argument("--verbose", action="store_true", help="Print tracebacks for failed recordings")
    parser.add_argument("--version", action="version", version=f"pyBer {__version__}")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    try:
        return run_batch(parser.parse_args(argv))
    except (ValueError, FileNotFoundError) as exc:
        parser.error(str(exc))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
