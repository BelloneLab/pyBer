"""LED / barcode sync-signal extraction from videos.

GUI-free (no PySide6) so it can be the body of a ``ProcessPoolExecutor`` worker:
child processes only need NumPy + OpenCV, keeping spawn startup cheap on Windows.

Design notes (matching BelloneLab/video_barcode_extractor for signal quality):

* **Frame-accurate seeking.** Long-GOP codecs (H.264 MP4) snap ``POS_FRAMES``
  to the nearest keyframe, which drifts the extracted trace and smears the
  barcode. We seek ``_LEAD_IN_FRAMES`` before the target and ``grab()`` forward
  to land exactly on ``start_frame``.
* **Chunk parallelism.** A single long video is split into frame chunks that are
  decoded in parallel, then concatenated in order. This is the main speed win
  for barcode extraction of long recordings (per-video parallelism does nothing
  for one big file).
* **Reduction modes.** ``mean`` (low contrast for a small LED in a wide ROI),
  ``max`` and ``bright`` (mean of the brightest pixels) give a much sharper
  on/off swing for a small LED.
* **Clean output.** ``binarize_signal`` thresholds with ``compute_threshold``
  (Otsu when balanced, Triangle when the duty cycle is skewed, matching the
  sync GUI preview) so fast barcode transitions are preserved; ``smooth_signal``
  is available only when the user explicitly asks for it.
"""
from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# OpenCV channel index into a BGR frame; None means grayscale luminance.
_CH_MAP = {"grayscale": None, "gray": None, "red": 2, "green": 1, "blue": 0}
# BT.601 luminance weights in OpenCV BGR order.
_LUM_W = np.array([0.114, 0.587, 0.299], dtype=np.float32)
# Frames to decode ahead of a seek target to defeat keyframe snapping.
_LEAD_IN_FRAMES = 60


def gpu_available() -> bool:
    """True when OpenCV was built with CUDA and a CUDA device is present."""
    try:
        import cv2

        getter = getattr(getattr(cv2, "cuda", None), "getCudaEnabledDeviceCount", None)
        return bool(getter) and int(getter()) > 0
    except Exception:
        return False


def _channel_index(channel: str) -> Optional[int]:
    return _CH_MAP.get(str(channel or "grayscale").strip().lower(), None)


def _reduce_plane(plane: np.ndarray, reduce: str) -> float:
    """Reduce a 2-D ROI plane to one scalar by the chosen method."""
    reduce = str(reduce or "mean").strip().lower()
    if plane.size == 0:
        return float("nan")
    if reduce == "max":
        return float(plane.max())
    if reduce in ("bright", "bright10", "top10"):
        flat = plane.reshape(-1)
        k = max(1, int(flat.size * 0.10))
        if k >= flat.size:
            return float(flat.mean())
        idx = np.argpartition(flat, flat.size - k)[-k:]
        return float(flat[idx].mean())
    if reduce.startswith("p"):
        try:
            q = float(reduce[1:])
            return float(np.percentile(plane, q))
        except Exception:
            pass
    return float(plane.mean())


def _crop_plane(frame: np.ndarray, roi: Tuple[int, int, int, int], ch_idx: Optional[int]) -> np.ndarray:
    x, y, w, h = roi
    crop = frame[max(0, y): y + h, max(0, x): x + w]
    if crop.size == 0:
        return np.empty((0,), dtype=np.float32)
    if ch_idx is None:
        return crop[:, :, :3].astype(np.float32) @ _LUM_W
    return crop[:, :, ch_idx].astype(np.float32)


def extract_chunk(
    video_path: str,
    roi: Tuple[int, int, int, int],
    start_frame: int,
    end_frame: int,
    channel: str = "grayscale",
    reduce: str = "mean",
) -> Tuple[int, np.ndarray]:
    """Decode frames ``[start_frame, end_frame)`` and reduce the ROI per frame.

    Returns ``(start_frame, values)``. Uses a lead-in seek so the first returned
    frame is exactly ``start_frame`` even on long-GOP codecs.
    """
    import cv2

    ch_idx = _channel_index(channel)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return int(start_frame), np.empty((0,), dtype=np.float32)
    seek_to = max(0, int(start_frame) - _LEAD_IN_FRAMES)
    cap.set(cv2.CAP_PROP_POS_FRAMES, seek_to)
    pos = int(cap.get(cv2.CAP_PROP_POS_FRAMES) or seek_to)
    while pos < start_frame:
        if not cap.grab():
            break
        pos += 1
    n = max(0, int(end_frame) - int(start_frame))
    values = np.empty(n, dtype=np.float32)
    count = 0
    try:
        for _ in range(n):
            ok, frame = cap.read()
            if not ok or frame is None:
                break
            plane = _crop_plane(frame, roi, ch_idx)
            values[count] = _reduce_plane(plane, reduce) if plane.size else np.nan
            count += 1
    finally:
        cap.release()
    return int(start_frame), values[:count]


def extract_chunk_task(task: Dict[str, Any]) -> Dict[str, Any]:
    """Picklable ProcessPoolExecutor worker. ``task`` carries an ``id`` (video
    key) so results can be regrouped per video after parallel decode."""
    out: Dict[str, Any] = {"id": task.get("id", ""), "start": int(task.get("start_frame", 0)),
                           "values": np.empty((0,), np.float32), "error": ""}
    try:
        _start, values = extract_chunk(
            str(task.get("video_path", "")),
            tuple(int(v) for v in (task.get("roi") or (0, 0, 1, 1))),
            int(task.get("start_frame", 0)),
            int(task.get("end_frame", 0)),
            str(task.get("channel", "grayscale")),
            str(task.get("reduce", "mean")),
        )
        out["values"] = values
    except Exception as exc:  # pragma: no cover - decode/runtime dependent
        out["error"] = str(exc)
    return out


def smooth_signal(values: np.ndarray, window: int = 5) -> np.ndarray:
    """Moving-average denoise that preserves NaNs lightly (odd window)."""
    v = np.asarray(values, dtype=float)
    w = max(1, int(window))
    if w <= 1 or v.size < 3:
        return v
    if w % 2 == 0:
        w += 1
    kernel = np.ones(w, dtype=float) / float(w)
    filled = np.nan_to_num(v, nan=float(np.nanmedian(v)) if np.any(np.isfinite(v)) else 0.0)
    pad = w // 2
    padded = np.pad(filled, (pad, pad), mode="edge")
    return np.convolve(padded, kernel, mode="valid")


def _finite_values(x: np.ndarray) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float64).reshape(-1)
    return arr[np.isfinite(arr)]


def _otsu_threshold(x: np.ndarray, bins: int = 256) -> float:
    """Match video_barcode_extractor's Otsu threshold."""
    finite = _finite_values(x)
    if finite.size < 2:
        return 0.0
    try:
        from skimage.filters import threshold_otsu as _otsu

        return float(_otsu(finite))
    except Exception:
        pass
    lo, hi = float(finite.min()), float(finite.max())
    if hi <= lo:
        return lo
    hist, edges = np.histogram(finite, bins=bins, range=(lo, hi))
    hist = hist.astype(np.float64)
    total = float(hist.sum())
    if total <= 0:
        return 0.5 * (lo + hi)
    centers = 0.5 * (edges[:-1] + edges[1:])
    w0 = np.cumsum(hist)
    w1 = total - w0
    weighted = np.cumsum(hist * centers)
    grand = float(weighted[-1])
    with np.errstate(invalid="ignore", divide="ignore"):
        m0 = weighted / w0
        m1 = (grand - weighted) / w1
        between = w0 * w1 * (m0 - m1) ** 2
    between[~np.isfinite(between)] = -1.0
    return float(centers[int(np.argmax(between))])


def _triangle_threshold(x: np.ndarray, bins: int = 256) -> float:
    """Match video_barcode_extractor's Triangle threshold."""
    finite = _finite_values(x)
    if finite.size < 2:
        return 0.0
    try:
        from skimage.filters import threshold_triangle as _tri

        return float(_tri(finite))
    except Exception:
        pass
    lo, hi = float(finite.min()), float(finite.max())
    if hi <= lo:
        return lo
    hist, edges = np.histogram(finite, bins=bins, range=(lo, hi))
    centers = 0.5 * (edges[:-1] + edges[1:])
    nz = np.flatnonzero(hist)
    if nz.size == 0:
        return float(0.5 * (lo + hi))
    first, last = int(nz[0]), int(nz[-1])
    peak = int(np.argmax(hist))
    if first == last:
        return float(centers[first])
    flip = (peak - first) < (last - peak)
    h = hist[::-1].astype(np.float64) if flip else hist.astype(np.float64)
    if flip:
        peak, first = bins - 1 - peak, bins - 1 - last
    width = max(peak - first, 1)
    pk = float(h[peak])
    x1 = np.arange(width)
    y1 = h[first:first + width]
    norm = float(np.hypot(pk, width))
    dist = (pk * x1 - width * y1) / max(norm, 1e-12)
    arg = int(np.argmax(dist)) + first
    if flip:
        arg = bins - 1 - arg
    return float(centers[int(np.clip(arg, 0, bins - 1))])


def _mad_threshold(x: np.ndarray, k: float = 3.0) -> float:
    """Robust off-state baseline + k.sigma, with sigma from the MAD of the
    samples at or below the median (the assumed off-state). Exactly the right
    model for sparse pulses; bounded noise keeps it usable when balanced too."""
    arr = _finite_values(x)
    if arr.size < 2:
        return 0.0
    med = float(np.median(arr))
    off = arr[arr <= med]
    if off.size < 2:
        return med
    med_off = float(np.median(off))
    mad_off = float(np.median(np.abs(off - med_off)))
    return med_off + k * mad_off * 1.4826


def auto_threshold(values: np.ndarray) -> Tuple[float, str]:
    """Return the standalone app's automatic threshold and method name."""
    arr = _finite_values(values)
    if arr.size < 2:
        return 0.0, "Otsu"
    mid = 0.5 * (float(arr.min()) + float(arr.max()))
    frac_high = float(np.mean(arr > mid))
    if frac_high < 0.15 or frac_high > 0.85:
        return _triangle_threshold(arr), "Triangle"
    return _otsu_threshold(arr), "Otsu"


def compute_threshold(x: np.ndarray, method: str = "auto") -> float:
    """Absolute threshold for an extracted ROI trace.

    Single source of truth shared by ``binarize_signal`` (export path) and the
    sync GUI's live preview/alignment, so what you see is what you export.

    ``auto`` inspects class balance and picks Otsu for roughly balanced signals,
    Triangle for skewed/sparse ones (the LED barcode case). ``otsu``,
    ``triangle``, ``mad`` and ``p<NN>`` (percentile, e.g. ``p95``) force a method.
    """
    arr = _finite_values(x)
    if arr.size < 2:
        return 0.0
    m = str(method or "auto").strip().lower()
    if m == "otsu":
        return _otsu_threshold(arr)
    if m == "triangle":
        return _triangle_threshold(arr)
    if m == "mad":
        return _mad_threshold(arr)
    if m.startswith("p"):
        try:
            return float(np.percentile(arr, float(m[1:])))
        except Exception:
            pass
    threshold, _method = auto_threshold(arr)
    return threshold


def debounce_binary(binary: np.ndarray, min_run: int = 2) -> np.ndarray:
    """Drop runs shorter than ``min_run`` samples to remove single-frame chatter
    from a low-contrast LED while preserving genuine (multi-frame) barcode bits."""
    b = np.asarray(binary, dtype=float)
    if b.size < 3 or min_run <= 1:
        return b
    out = b.copy()
    n = out.size
    i = 0
    while i < n:
        j = i
        while j + 1 < n and out[j + 1] == out[i]:
            j += 1
        run = j - i + 1
        if run < min_run and 0 < i:  # absorb short run into the previous level
            out[i:j + 1] = out[i - 1]
        i = j + 1
    return out


def binarize_signal(values: np.ndarray, method: str = "auto") -> np.ndarray:
    """Per-frame binarization at the adaptive threshold from ``compute_threshold``.

    Replaces the old fixed "0.5 of the min-max range" rule, which mis-thresholds
    a low-duty-cycle LED barcode (the midpoint sits far above the brief flashes)
    and disagreed with the sync GUI's Otsu preview. ``auto`` now matches the
    preview exactly. No smoothing, hysteresis, or debounce: every fast barcode
    transition is preserved.
    """
    v = np.asarray(values, dtype=float)
    thr = compute_threshold(v, method)
    if not np.isfinite(thr):
        return np.zeros_like(v)
    out = (v >= thr).astype(float)
    out[~np.isfinite(v)] = 0.0
    return out


def apply_signal_format(values: np.ndarray, signal_format: str, smooth_window: int = 5) -> np.ndarray:
    """Post-process a raw reduced trace into raw / smoothed / binary form.

    Binary uses the plain per-frame threshold (no smoothing/debounce) so fast
    barcode bits stay sharp. "Smoothed" is offered only for genuinely noisy,
    slow LEDs; it is never applied before binarizing.
    """
    fmt = str(signal_format or "raw").strip().lower()
    if fmt.startswith("bin"):
        return binarize_signal(values)
    if fmt.startswith("smooth"):
        return smooth_signal(values, smooth_window)
    return np.asarray(values, dtype=float)


def extract_led_signal(config: Dict[str, Any], use_gpu: Optional[bool] = None) -> Dict[str, Any]:
    """Single-process extraction of one video's LED signal (raw reduced trace).

    ``config`` keys: ``video_path``, ``fps``, ``n_frames``, ``roi`` (x,y,w,h),
    ``channel``, ``reduce``, ``start_frame``, ``end_frame``. An optional ``id``
    is echoed back. Returns ``id``, ``time``, ``values``, ``count``, ``error``.
    """
    out: Dict[str, Any] = {
        "id": config.get("id", ""), "time": np.array([], float),
        "values": np.array([], float), "count": 0, "used_gpu": False, "error": "",
    }
    path = str(config.get("video_path") or "")
    if not path or not os.path.isfile(path):
        out["error"] = "Choose a video file."
        return out
    fps = float(config.get("fps") or 30.0)
    if not np.isfinite(fps) or fps <= 0:
        fps = 30.0
    n_frames = max(0, int(config.get("n_frames") or 0))
    start = max(0, int(config.get("start_frame") or 0))
    end = int(config.get("end_frame") or n_frames or 0)
    if n_frames:
        end = min(max(start + 1, end), n_frames)
    if end <= start:
        out["error"] = "End frame must be greater than start frame."
        return out
    roi = tuple(int(v) for v in (config.get("roi") or (0, 0, 1, 1)))
    if roi[2] <= 0 or roi[3] <= 0:
        out["error"] = "Choose a non-empty LED ROI."
        return out
    try:
        _start, values = extract_chunk(
            path, roi, start, end,
            str(config.get("channel", "grayscale")),
            str(config.get("reduce", "mean")),
        )
    except Exception as exc:
        out["error"] = str(exc)
        return out
    count = int(values.size)
    if count < 2:
        out["error"] = "No usable LED samples were extracted."
        return out
    out["time"] = (np.arange(count, dtype=float) + start) / fps
    out["values"] = np.asarray(values, float)
    out["count"] = count
    return out


def plan_chunks(start: int, end: int, n_chunks: int) -> List[Tuple[int, int]]:
    """Split ``[start, end)`` into about ``n_chunks`` balanced chunks.

    Splitting a single video helps only when there are spare cores: each chunk
    pays a fixed lead-in seek cost, and several processes decoding the *same*
    file contend on I/O. The caller therefore passes a small ``n_chunks`` (often
    1) when many independent videos already saturate the worker pool. A minimum
    chunk size amortizes the lead-in.
    """
    total = max(0, int(end) - int(start))
    if total <= 0:
        return []
    n_chunks = max(1, int(n_chunks))
    chunk = max(800, -(-total // n_chunks))  # ceil division, min 800 frames
    chunks: List[Tuple[int, int]] = []
    f = int(start)
    while f < int(end):
        e = min(f + chunk, int(end))
        chunks.append((f, e))
        f = e
    return chunks
