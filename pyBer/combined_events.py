"""Union (OR) of labels within one recording; never pool different animals."""
import numpy as np


def union_states(arrays):
    """True if any observed state is true; false only when all are known false."""
    values = [np.asarray(array, float) for array in arrays]
    if len(values) < 2 or any(a.ndim != 1 or a.shape != values[0].shape for a in values):
        raise ValueError("Combined states need at least two columns on the same frame/time axis.")
    if any(np.any(np.isfinite(a) & ~np.isin(a, [0., 1.])) for a in values):
        raise ValueError("Only binary state columns can be combined as sampled states.")
    active = np.zeros(values[0].shape, bool)
    known = np.ones(values[0].shape, bool)
    for array in values:
        active |= array == 1
        known &= np.isfinite(array)
    return np.where(active, 1., np.where(known, 0., np.nan))


def union_intervals(events):
    """Merge overlapping/touching bouts; unique point events remain points."""
    intervals = []
    for on, off, _duration in events:
        on, off = np.asarray(on, float), np.asarray(off, float)
        if on.ndim != 1 or off.shape != on.shape:
            raise ValueError("Combined event onsets and offsets must match.")
        if np.any(~np.isfinite(on) | ~np.isfinite(off) | (off < on)):
            raise ValueError("Cannot combine events with unknown or invalid boundaries.")
        intervals.extend(zip(on.tolist(), off.tolist()))
    merged = []
    for start, stop in sorted(intervals):
        if merged and start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], stop)
        else:
            merged.append([start, stop])
    on = np.array([row[0] for row in merged], float)
    off = np.array([row[1] for row in merged], float)
    return {"on": on, "off": off, "dur": np.where(off > on, off - on, np.nan),
            "rule": "OR: union of selected events", "source": "combined"}


def combined_value(info, members, extract):
    """Return the destination and derived values, without modifying a source."""
    states = info.get("behaviors") or {}
    events = info.get("event_behaviors") or {}
    if len(set(members)) < 2:
        raise ValueError("Select at least two different behaviors or zones.")
    if not set(members).issubset(set(states) | set(events)):
        raise ValueError("This recording does not contain all selected labels.")
    sampled = [name in states and name not in events and info.get("kind") != "timestamps"
               for name in members]
    # Legacy timestamp-column mode uses this serialized name.
    if info.get("kind") == "timestamp_columns":
        sampled = [False] * len(members)
    if all(sampled):
        return "behaviors", union_states([states[name] for name in members])
    if any(sampled):
        raise ValueError("Combine sampled state columns separately from event-list labels; "
                         "their missing-observation semantics differ.")
    return "event_behaviors", union_intervals([extract(info, name) for name in members])


def refresh_combined_labels(info, extract):
    """Rebuild derived columns after an arena/source change; stale unions disappear."""
    definitions = (info.get("import_report") or {}).get("combined_labels") or {}
    errors = []
    for name, definition in definitions.items():
        info.get("behaviors", {}).pop(name, None)
        info.get("event_behaviors", {}).pop(name, None)
        try:
            destination, value = combined_value(info, definition["members"], extract)
            info.setdefault(destination, {})[name] = value
        except (ValueError, KeyError) as exc:
            errors.append(f"{name}: {exc}")
    return errors
