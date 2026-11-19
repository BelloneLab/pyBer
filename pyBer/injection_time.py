"""Per-recording injection time conversion and before/after behavior summaries."""
import numpy as np


def injection_times(proc):
    """Return raw photometry seconds, synchronized camera seconds and any issue."""
    elapsed = proc.injection_seconds
    if elapsed is None:
        return None, None, "None"
    base = np.asarray(proc.time, float)
    finite = base[np.isfinite(base)]
    if not finite.size:
        return None, None, "Empty trace"
    aligned = np.asarray(proc.sync_aligned_time, float)
    camera = fiber = np.empty(0)
    if aligned.shape == base.shape:
        valid = np.isfinite(aligned) & np.isfinite(base)
        camera, fiber = aligned[valid], base[valid]
    synced = (camera.size >= 2 and np.all(np.diff(camera) > 0)
              and np.all(np.diff(fiber) > 0))
    if proc.injection_clock == "video":
        if not synced:
            return None, None, "Video sync required"
        if not camera[0] <= elapsed <= camera[-1]:
            return None, None, "Outside synced trace"
        return float(np.interp(elapsed, camera, fiber)), elapsed, ""
    raw = finite[0] + elapsed
    if not finite[0] <= raw <= finite[-1]:
        return None, None, "Outside recording"
    mapped = float(np.interp(raw, fiber, camera)) if synced and fiber[0] <= raw <= fiber[-1] else None
    return float(raw), mapped, ""


def before_injection(times, injection):
    # Interpolating between two clocks can move an exact boundary by one ULP.
    return (times < injection) & ~np.isclose(times, injection, rtol=0., atol=1e-9)


def period_summary(onsets, durations, intervals, injection):
    """Count observed onsets and split occupied time at gaps and the injection."""
    onsets = np.asarray(onsets, float)
    durations = np.asarray(durations, float)
    before = before_injection(onsets, injection)
    bouts = sorted(zip(onsets, durations))
    rows = []
    for name in ("Before", "After"):
        observed = occupied = 0.0
        count = 0
        for start, end in intervals:
            visible = (onsets >= start) & (onsets <= end)
            if name == "Before":
                count += np.count_nonzero(visible & before)
                end = min(end, injection)
            else:
                count += np.count_nonzero(visible & ~before)
                start = max(start, injection)
            if end <= start:
                continue
            observed += end - start
            previous_end = start
            for onset, duration in bouts:
                if np.isfinite(onset) and np.isfinite(duration) and duration > 0:
                    stop = min(end, onset + duration)
                    occupied += max(0., stop - max(start, previous_end, onset))
                    previous_end = max(previous_end, stop)
        rows.append((name, int(count), observed, count * 60 / observed if observed else None,
                     occupied if np.all(np.isfinite(durations)) else None))
    return rows
