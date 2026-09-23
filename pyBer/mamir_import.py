"""Read MAMIR behavior and zone exports without changing source files.

MAMIR writes one row per animal and frame.  The generic pyBer table loader
cannot treat those interleaved rows as one animal's binary time series, so this
adapter selects an identity before exposing events to postprocessing.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


_FRAME_METADATA = frozenset({"frame", "time_s", "identity", "behavior_top"})


def legacy_import_warning(sources) -> str:
    """Warn about pre-fix snapshots without rewriting embedded user data."""
    for source in sources.values():
        report = source.get("import_report") or {}
        if not isinstance(report, dict):
            continue
        reports = [report, *(report.get("mamir_imports") or [])]
        for item in reports:
            if (isinstance(item, dict) and item.get("format") == "mamir"
                    and item.get("category") == "behavior"):
                version = item.get("adapter_version", 0)
                if not isinstance(version, int) or version < 2:
                    return ("Older MAMIR import: re-import original frame files to verify behavior labels. "
                            "The early frame importer had a column-order bug. Saved data have not been rewritten.")
    return ""


def _identity(value: object) -> str:
    """Use one stable display key for CSV integers and strings."""
    if pd.isna(value):
        return ""
    text = str(value).strip()
    try:
        number = float(text)
        if np.isfinite(number) and number.is_integer():
            return str(int(number))
    except (TypeError, ValueError):
        pass
    return text


def _root(path: str | Path, category: str) -> Path:
    candidate = Path(path).expanduser().resolve()
    folder = candidate if candidate.is_dir() else candidate.parent
    if category == "behavior":
        if (folder / "behavior_bouts.csv").is_file() or (folder / "behavior_frames.csv").is_file() \
                or (folder / "behavior_frames_long.csv").is_file():
            return folder
        if (folder / "behavior" / "behavior_bouts.csv").is_file():
            return folder / "behavior"
        if candidate.is_file() and candidate.suffix.lower() == ".csv":
            columns = set(pd.read_csv(candidate, nrows=0).columns)
            if {"frame", "time_s", "identity"}.issubset(columns) or \
                    {"behavior", "identity", "start_s", "stop_s"}.issubset(columns):
                return folder
    elif (folder / "zone_bouts.csv").is_file():
        return folder
    elif (folder / "assay_metrics" / "zone_bouts.csv").is_file():
        return folder / "assay_metrics"
    elif candidate.is_file() and candidate.suffix.lower() == ".csv":
        columns = set(pd.read_csv(candidate, nrows=0).columns)
        if {"animal_id", "zone", "start_frame", "end_frame"}.issubset(columns):
            return folder
    raise ValueError(f"No MAMIR {category} export found beside {candidate.name}.")


def _unique_values(path: Path, column: str) -> list[str]:
    values: set[str] = set()
    for chunk in pd.read_csv(path, usecols=[column], chunksize=100_000):
        values.update(_identity(value) for value in chunk[column].dropna().unique())
    return sorted(value for value in values if value)


def inspect_mamir(path: str | Path, category: str) -> dict[str, Any]:
    """Inspect labels and animal IDs without loading a large frame matrix."""
    if category not in {"behavior", "zone"}:
        raise ValueError("Choose Behavior or Zone.")
    folder = _root(path, category)
    if category == "behavior":
        bouts: Path | None = folder / "behavior_bouts.csv"
        frames: Path | None = folder / "behavior_frames.csv"
        long_frames: Path | None = folder / "behavior_frames_long.csv"
        chosen = Path(path).expanduser().resolve()
        if chosen.is_file() and chosen.suffix.lower() == ".csv":
            columns = set(pd.read_csv(chosen, nrows=0).columns)
            if {"behavior", "identity", "start_s", "stop_s"}.issubset(columns):
                bouts = chosen
                frames = None
                long_frames = None
            elif {"frame", "time_s", "identity", "behavior"}.issubset(columns):
                long_frames = chosen
                bouts = None
                frames = None
            elif {"frame", "time_s", "identity"}.issubset(columns):
                frames = chosen
                bouts = None
                long_frames = None
            else:
                raise ValueError("Select a time-resolved MAMIR bout or frame export, not an aggregate summary.")
        if bouts is not None and bouts.is_file():
            table = pd.read_csv(bouts, usecols=lambda c: c in {"behavior", "identity", "partner"})
            if not {"behavior", "identity"}.issubset(table):
                raise ValueError("MAMIR bout table needs behavior and identity columns.")
            names = set(table["behavior"].dropna().astype(str))
            identities = sorted({_identity(value) for value in table["identity"].dropna()})
            partners = sorted({_identity(value) for value in table.get("partner", pd.Series(dtype=object)).dropna()})
        else:
            names, identities, partners = set(), [], []
        if frames is not None and frames.is_file():
            names.update(set(pd.read_csv(frames, nrows=0).columns) - _FRAME_METADATA)
            identities = sorted(set(identities) | set(_unique_values(frames, "identity")))
        if long_frames is not None and long_frames.is_file():
            names.update(_unique_values(long_frames, "behavior"))
            identities = sorted(set(identities) | set(_unique_values(long_frames, "identity")))
        return {"root": folder, "category": category, "labels": sorted(names),
                "identities": identities, "partners": partners,
                "bouts_path": bouts if bouts is not None and bouts.is_file() else None,
                "frames_path": frames if frames is not None and frames.is_file() else None,
                "long_frames_path": long_frames if long_frames is not None and long_frames.is_file() else None}

    bouts = folder / "zone_bouts.csv"
    chosen = Path(path).expanduser().resolve()
    if chosen.is_file() and chosen.suffix.lower() == ".csv":
        bouts = chosen
    table = pd.read_csv(bouts, usecols=lambda c: c in {"animal_id", "zone"})
    if not {"animal_id", "zone"}.issubset(table):
        raise ValueError("MAMIR zone table needs animal_id and zone columns.")
    names = set(table["zone"].dropna().astype(str))
    identities = {_identity(value) for value in table["animal_id"].dropna()}
    summary_path = folder / "summary.json"
    if summary_path.is_file():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        names.update(str(zone["name"]) for zone in summary.get("zones", [])
                     if isinstance(zone, dict) and zone.get("name"))
        identities.update(_identity(value) for value in summary.get("animals", []))
    return {"root": folder, "category": category,
            "labels": sorted(names), "identities": sorted(identities),
            "partners": [], "bouts_path": bouts}


def _event_arrays(groups: dict[str, list[tuple[float, float]]]) -> dict[str, dict[str, np.ndarray]]:
    result = {}
    for label, spans in groups.items():
        ordered = sorted(spans)
        on = np.asarray([start for start, _ in ordered], dtype=float)
        off = np.asarray([stop for _, stop in ordered], dtype=float)
        result[label] = {"on": on, "off": off, "dur": off - on,
                         "source": "MAMIR"}
    return result


def _frame_events(path: Path, identity: str, labels: list[str],
                  default_step: float = 0.0) -> dict[str, list[tuple[float, float]]]:
    """Build bouts from a selected animal only, retaining missing-frame gaps."""
    groups: dict[str, list[tuple[float, float]]] = {name: [] for name in labels}
    active: dict[str, tuple[float, int]] = {}
    last_time: float | None = None
    last_frame: int | None = None
    step = default_step
    usecols = ["frame", "time_s", "identity", *labels]
    for chunk in pd.read_csv(path, usecols=usecols, chunksize=100_000):
        # pandas usecols filters columns but retains CSV order, not usecols order.
        # Explicit reindexing is essential before pairing states with labels.
        selected = chunk.loc[chunk["identity"].map(_identity) == identity, usecols]
        for row in selected.itertuples(index=False, name=None):
            frame, time_s, _animal, *states = row
            if not np.isfinite(time_s) or not np.isfinite(frame) or frame < 0 or frame != int(frame):
                raise ValueError("MAMIR frame/time values must be finite with nonnegative integer frames.")
            frame, time_s = int(frame), float(time_s)
            previous_step = step
            if last_time is not None and last_frame is not None:
                if frame <= last_frame or time_s <= last_time:
                    raise ValueError("MAMIR frame rows must have strictly increasing frame and time per animal.")
                step = (time_s - last_time) / (frame - last_frame)
                if previous_step <= 0:
                    previous_step = step
            contiguous = last_frame is not None and frame == last_frame + 1
            for name, state in zip(labels, states):
                on = pd.notna(state) and float(state) > 0.5
                if name in active and (not on or not contiguous):
                    start, _ = active.pop(name)
                    stop = time_s if contiguous else last_time + previous_step
                    if stop > start:
                        groups[name].append((start, stop))
                if on and name not in active:
                    active[name] = (time_s, frame)
            last_time, last_frame = time_s, frame
    for name, (start, _) in active.items():
        if step <= 0:
            raise ValueError("Cannot infer frame duration; keep behavior_summary.json beside this export.")
        stop = (last_time if last_time is not None else start) + step
        if stop > start:
            groups[name].append((start, stop))
    return groups


def _long_frame_events(path: Path, identity: str, labels: list[str],
                       default_step: float = 0.0) -> dict[str, list[tuple[float, float]]]:
    """Rebuild bouts from sparse long rows, preserving independent labels."""
    groups: dict[str, list[tuple[float, float]]] = {name: [] for name in labels}
    active: dict[str, tuple[int, float, int, float]] = {}
    frame_step = default_step
    latest_frame = -1
    latest_time = None
    columns = ["frame", "time_s", "identity", "behavior"]
    for chunk in pd.read_csv(path, usecols=columns, chunksize=100_000):
        selected = chunk.loc[chunk["identity"].map(_identity) == identity, columns]
        for frame, time_s, _animal, behavior in selected.itertuples(index=False, name=None):
            if pd.isna(behavior):
                continue
            if not np.isfinite(time_s) or not np.isfinite(frame) or frame < 0 or frame != int(frame):
                raise ValueError("MAMIR frame/time values must be finite with nonnegative integer frames.")
            frame, time_s, label = int(frame), float(time_s), str(behavior)
            if frame < latest_frame:
                raise ValueError("MAMIR long frame rows must be ordered by frame.")
            if latest_time is not None:
                if frame == latest_frame and time_s != latest_time:
                    raise ValueError("MAMIR rows at the same frame must have the same time.")
                if frame > latest_frame:
                    if time_s <= latest_time:
                        raise ValueError("MAMIR times must increase with frame number.")
                    frame_step = (time_s - latest_time) / (frame - latest_frame)
            latest_frame = frame
            latest_time = time_s
            previous = active.get(label)
            if previous is not None and frame == previous[2]:
                continue  # A repeated long row must not create a duplicate bout.
            if previous is not None and frame != previous[2] + 1:
                stop = previous[3] + frame_step
                if stop > previous[1]:
                    groups[label].append((previous[1], stop))
                active.pop(label)
            if label not in active:
                active[label] = (frame, time_s, frame, time_s)
            else:
                start_frame, start_time, _last_frame, _last_time = active[label]
                active[label] = (start_frame, start_time, frame, time_s)
    if active and frame_step <= 0:
        raise ValueError("Cannot infer frame duration from this long export; keep behavior_summary.json beside it.")
    for label, (_start_frame, start, _last_frame, last_time) in active.items():
        stop = last_time + frame_step
        if stop > start:
            groups[label].append((start, stop))
    return groups


def load_mamir(path: str | Path, category: str, identity: str,
               partner: str | None = None) -> dict[str, Any]:
    """Return the existing pyBer behavior-source shape for one animal.

    Bout times are used directly. Zone frame indices use the FPS recorded in
    MAMIR's summary.json; its end_frame is inclusive.
    """
    overview = inspect_mamir(path, category)
    identity = _identity(identity)
    if identity not in overview["identities"]:
        raise ValueError(f"Animal {identity!r} is absent from this MAMIR export.")
    folder = overview["root"]
    groups: dict[str, list[tuple[float, float]]] = {name: [] for name in overview["labels"]}
    if category == "behavior":
        bouts = overview["bouts_path"]
        if bouts is not None:
            table = pd.read_csv(bouts)
            required = {"behavior", "identity", "start_s", "stop_s"}
            if not required.issubset(table):
                raise ValueError(f"MAMIR bout table is missing {sorted(required - set(table))}.")
            table = table.loc[table["identity"].map(_identity) == identity]
            if partner is not None:
                table = table.loc[table.get("partner", pd.Series(index=table.index, dtype=object)).map(_identity)
                                  == _identity(partner)]
            for row in table.itertuples(index=False):
                name = str(getattr(row, "behavior"))
                start, stop = float(getattr(row, "start_s")), float(getattr(row, "stop_s"))
                if np.isfinite(start) and np.isfinite(stop) and stop > start:
                    groups.setdefault(name, []).append((start, stop))
        else:
            if partner is not None:
                raise ValueError("Partner filtering needs behavior_bouts.csv; frame states have no partner column.")
            summary_path = folder / "behavior_summary.json"
            default_step = 0.0
            if summary_path.is_file():
                fps = float(json.loads(summary_path.read_text(encoding="utf-8")).get("fps", 0))
                if np.isfinite(fps) and fps > 0:
                    default_step = 1.0 / fps
            if overview["frames_path"] is not None:
                groups = _frame_events(overview["frames_path"], identity, overview["labels"], default_step)
            elif overview["long_frames_path"] is not None:
                groups = _long_frame_events(overview["long_frames_path"], identity,
                                            overview["labels"], default_step)
            else:
                raise ValueError("This MAMIR export has no time-resolved behavior rows.")
    else:
        summary_path = folder / "summary.json"
        if not summary_path.is_file():
            raise ValueError("MAMIR zone_bouts.csv needs its adjacent summary.json for FPS.")
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        fps = float(summary.get("fps", 0))
        if not np.isfinite(fps) or fps <= 0:
            raise ValueError("MAMIR zone summary has no valid FPS.")
        table = pd.read_csv(overview["bouts_path"])
        required = {"animal_id", "zone", "start_frame", "end_frame"}
        if not required.issubset(table):
            raise ValueError(f"MAMIR zone table is missing {sorted(required - set(table))}.")
        table = table.loc[table["animal_id"].map(_identity) == identity]
        for row in table.itertuples(index=False):
            start = float(getattr(row, "start_frame")) / fps
            stop = (float(getattr(row, "end_frame")) + 1) / fps
            if np.isfinite(start) and np.isfinite(stop) and stop > start:
                groups.setdefault(str(getattr(row, "zone")), []).append((start, stop))

    return {"kind": "binary_columns", "time": np.array([], dtype=float),
            "behaviors": {}, "event_behaviors": _event_arrays(groups), "trajectory": {},
            "trajectory_time": np.array([], dtype=float), "trajectory_time_col": "",
            "time_candidates": {}, "row_count": sum(len(v) for v in groups.values()),
            "needs_generated_time": False, "source_path": str(Path(path).resolve()),
            "import_report": {"format": "mamir", "category": category,
                              "adapter_version": 2,
                              "identity": identity, "partner": partner or "",
                              "labels": overview["labels"]}}
