# main.py
"""
Fiber Photometry Processor (Doric .doric) - PySide6 + pyqtgraph

Run:
    python main.py

Dependencies:
    pip install PySide6 pyqtgraph h5py numpy scipy scikit-learn pybaselines
"""

from __future__ import annotations

import os
import json
import logging
import sys
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple


_DLL_DIR_HANDLES = []


def _bootstrap_windows_conda_runtime() -> None:
    if os.name != "nt":
        return

    os.environ.setdefault("PYTHONNOUSERSITE", "1")

    try:
        import site
        user_site = os.path.normcase(os.path.abspath(site.getusersitepackages()))
    except Exception:
        user_site = ""

    appdata_python = ""
    appdata = os.environ.get("APPDATA", "")
    if appdata:
        appdata_python = os.path.normcase(os.path.abspath(os.path.join(appdata, "Python")))

    def _is_user_site_path(path: str) -> bool:
        if not path:
            return False
        try:
            norm = os.path.normcase(os.path.abspath(path))
        except Exception:
            return False
        if user_site and (norm == user_site or norm.startswith(user_site + os.sep)):
            return True
        return bool(appdata_python and (norm == appdata_python or norm.startswith(appdata_python + os.sep)))

    sys.path[:] = [path for path in sys.path if not _is_user_site_path(path)]
    script_dir = os.path.dirname(os.path.abspath(__file__))
    if script_dir and script_dir not in sys.path:
        sys.path.insert(0, script_dir)

    prefix = os.environ.get("CONDA_PREFIX") or sys.prefix
    dll_dirs = [
        prefix,
        os.path.join(prefix, "Library", "mingw-w64", "bin"),
        os.path.join(prefix, "Library", "usr", "bin"),
        os.path.join(prefix, "Library", "bin"),
        os.path.join(prefix, "Scripts"),
    ]
    existing = [path for path in dll_dirs if path and os.path.isdir(path)]

    if hasattr(os, "add_dll_directory"):
        for path in existing:
            try:
                _DLL_DIR_HANDLES.append(os.add_dll_directory(path))
            except Exception:
                pass

    old_path = os.environ.get("PATH", "")
    old_parts = [os.path.normcase(os.path.abspath(p)) for p in old_path.split(os.pathsep) if p]
    prepend = [p for p in existing if os.path.normcase(os.path.abspath(p)) not in old_parts]
    if prepend:
        os.environ["PATH"] = os.pathsep.join(prepend + [old_path])


_bootstrap_windows_conda_runtime()


def _crash_log_path() -> str:
    base = os.environ.get("LOCALAPPDATA") or os.environ.get("TEMP") or os.path.expanduser("~")
    folder = os.path.join(base, "pyBer")
    try:
        os.makedirs(folder, exist_ok=True)
    except Exception:
        folder = base
    return os.path.join(folder, "pyber_crash.log")


# Dump a native C-level traceback (e.g. a NumPy-ABI / OpenCV segfault) to a log
# file so a hard crash leaves evidence instead of vanishing silently.
try:
    import faulthandler

    _CRASH_LOG_FILE = open(_crash_log_path(), "a", buffering=1, encoding="utf-8")
    faulthandler.enable(file=_CRASH_LOG_FILE, all_threads=True)
except Exception:
    _CRASH_LOG_FILE = None

from PySide6 import QtCore, QtGui, QtWidgets
import pyqtgraph as pg
from pyqtgraph.dockarea import DockArea, Dock
import h5py

from analysis_core import (
    ExportSelection,
    OUTPUT_MODES,
    PhotometryProcessor,
    ProcessingParams,
    LoadedDoricFile,
    LoadedTrial,
    ProcessedTrial,
    export_processed_csv,
    export_processed_h5,
    load_processed_csv,
    load_processed_h5,
    load_rwd_csv,
    is_rwd_events_csv,
    recommend_preprocessing_settings,
    safe_stem_from_metadata,
    detect_artifacts_adaptive,
    interpolate_nans,
    zscore_median_std,
    safe_divide,
    _lowpass_sos,
    coerce_time_value,
)
from gui_preprocessing import (
    FileQueuePanel,
    ParameterPanel,
    PlotDashboard,
    MetadataDialog,
    ArtifactPanel,
    AdvancedOptionsDialog,
)
from gui_postprocessing import PostProcessingPanel
from numeric_controls import install_spinbox_scrubbers
from onboarding import (
    ToastManager,
    TutorialOverlay,
    PreferencesDialog,
    PanelHeader,
    register_global_shortcuts,
    attach_dirty_title,
    install_close_confirmation,
    reset_focused_plot_view,
    build_default_tutorial,
    add_empty_state_hint,
)
from styles import (
    apply_app_palette,
    app_qss,
    _make_icon,
    _paint_database,
    _paint_sliders,
    _paint_filter,
    _paint_wave,
    _paint_chart,
    _paint_badge,
    _paint_export,
    _paint_gear,
)
import numpy as np


# Icon painters now live in styles.py and are imported above.


def _dock_area_to_int(value: object, fallback: int = 2) -> int:
    """
    Convert Qt DockWidgetArea enum/flag objects (or stored values) to int safely.
    Some PySide6 builds do not allow int(Qt enum) directly.
    """
    try:
        enum_value = getattr(value, "value", None)
        if enum_value is not None:
            return int(enum_value)
    except Exception:
        pass
    try:
        if isinstance(value, str):
            v = value.strip().lower()
            if "left" in v:
                return 1
            if "right" in v:
                return 2
            if "top" in v:
                return 4
            if "bottom" in v:
                return 8
    except Exception:
        pass
    try:
        return int(value)
    except Exception:
        return int(fallback)


def _to_bool(value: object, default: bool = False) -> bool:
    """
    Convert mixed QSettings bool payloads (bool/int/str) safely.
    """
    if isinstance(value, bool):
        return value
    if value is None:
        return bool(default)
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        v = value.strip().lower()
        if v in {"1", "true", "yes", "y", "on"}:
            return True
        if v in {"0", "false", "no", "n", "off", ""}:
            return False
    return bool(default)


_DOCK_STATE_VERSION = 3
# Dock snapshot keys are versioned because object names changed from pre_/post_
# to pre./post. and old blobs are incompatible with restoreState.
_PRE_DOCK_STATE_KEY = "pre_main_dock_state_v4"
_POST_DOCK_STATE_KEY = "post_main_dock_state_v4"
_PRE_TAB_GROUPS_KEY = "pre_tab_groups_v1"
_PRE_DOCKAREA_STATE_KEY = "pre_dockarea_state_v1"
_PRE_DOCKAREA_VISIBLE_KEY = "pre_dockarea_visible_v1"
_PRE_DOCKAREA_ACTIVE_KEY = "pre_dockarea_active_v1"
_PRE_DOCK_PREFIX = "pre."
_POST_DOCK_PREFIX = "post."
_FORCE_FIXED_DOCK_LAYOUTS = False
_USE_PG_DOCKAREA_PRE_LAYOUT = True
_PRE_DOCKAREA_PRIMARY_ORDER = ("artifacts", "filtering", "baseline", "output", "export")
_PRE_DOCKAREA_OPTIONAL_ORDER = ("qc", "config")
_PRE_DOCKAREA_DEFAULT_VISIBLE = frozenset(_PRE_DOCKAREA_PRIMARY_ORDER)
_CSV_NONE_LABEL = "(none)"
_PRE_PROJECT_TYPE = "pyber_preprocessing_project"
_PRE_PROJECT_VERSION = 1
_SUPPORTED_DATA_EXTS = (".doric", ".h5", ".hdf5", ".csv")
_RWD_FLUORESCENCE_CSV_STEMS = ("fluorescence", "fluorescence-unaligned")
_FOLDER_DISCOVERY_SKIP_DIRS = frozenset({
    ".git",
    ".pytest_cache",
    "__pycache__",
    "build",
    "dist",
    "release",
})

_LOG = logging.getLogger(__name__)


def _is_rwd_fluorescence_csv_name(path: str) -> bool:
    name = os.path.basename(str(path or "")).strip().lower()
    stem, ext = os.path.splitext(name)
    if ext != ".csv":
        return False
    return any(stem == base or stem.startswith(base + "_") for base in _RWD_FLUORESCENCE_CSV_STEMS)


def _is_supported_preprocessing_folder_file(path: str, *, include_generic_csv: bool = False) -> bool:
    ext = os.path.splitext(str(path or ""))[1].lower()
    if ext in (".doric", ".h5", ".hdf5"):
        return True
    if ext != ".csv":
        return False
    if is_rwd_events_csv(path):
        return False
    if _is_rwd_fluorescence_csv_name(path):
        return True
    return bool(include_generic_csv)


def _discover_preprocessing_data_files(folder: str, *, recursive: bool = True) -> List[str]:
    """Find raw preprocessing files under a folder.

    Direct CSV files are kept for backward compatibility with the old one-level
    folder import. In nested folders, CSV discovery is deliberately narrower:
    RWD fluorescence files are included, while sibling Events.csv files are
    ignored because the RWD loader attaches them automatically.
    """
    root = os.path.abspath(str(folder or ""))
    if not root or not os.path.isdir(root):
        return []
    root_norm = os.path.normcase(root)
    out: List[str] = []
    seen: set[str] = set()

    for dirpath, dirnames, filenames in os.walk(root):
        if recursive:
            dirnames[:] = [
                name for name in sorted(dirnames)
                if name not in _FOLDER_DISCOVERY_SKIP_DIRS and not name.startswith(".")
            ]
        else:
            dirnames[:] = []

        is_root = os.path.normcase(os.path.abspath(dirpath)) == root_norm
        for filename in sorted(filenames):
            full = os.path.join(dirpath, filename)
            if not _is_supported_preprocessing_folder_file(full, include_generic_csv=is_root):
                continue
            key = os.path.normcase(os.path.abspath(full))
            if key in seen:
                continue
            seen.add(key)
            out.append(full)

    return out


def _asset_candidates(filename: str) -> List[str]:
    if getattr(sys, "frozen", False):
        base_dir = str(getattr(sys, "_MEIPASS", "")) or os.path.dirname(sys.executable)
        return [
            os.path.join(base_dir, "assets", filename),
            os.path.join(os.path.dirname(sys.executable), "assets", filename),
        ]
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return [os.path.join(base_dir, "assets", filename)]


def _first_existing_asset(filename: str) -> str:
    candidates = _asset_candidates(filename)
    for path in candidates:
        if os.path.isfile(path):
            return path
    return candidates[0] if candidates else filename


def _pyber_icon_path() -> str:
    for filename in ("pyBer.ico", "pyBer_logo_big.png"):
        path = _first_existing_asset(filename)
        if os.path.isfile(path):
            return path
    return _first_existing_asset("pyBer.ico")


def _pyber_splash_path() -> str:
    return _first_existing_asset("pyBer_logo_big.png")


def _set_windows_app_user_model_id() -> None:
    """Give Windows a stable app identity so Qt's app icon is used on the taskbar."""
    if os.name != "nt":
        return
    try:
        import ctypes
        from ctypes import wintypes
        app_id = "BelloneLab.pyBer.FiberPhotometry"
        func = ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID
        func.argtypes = [wintypes.LPCWSTR]
        func.restype = ctypes.HRESULT
        hr = func(app_id)
        if hr != 0:
            logging.warning("SetCurrentProcessExplicitAppUserModelID returned HRESULT 0x%08x", hr & 0xFFFFFFFF)
        else:
            logging.info("Windows AppUserModelID set to %s", app_id)
    except Exception as exc:
        logging.warning("Could not set Windows AppUserModelID: %s", exc)


def _build_pyber_icon() -> Optional["QtGui.QIcon"]:
    """Build a QIcon backed by every embedded size of pyBer.ico.

    On some Windows configurations Qt fails to render small taskbar icons
    when only the multi-image .ico file is given to QIcon. Explicitly pulling
    each pixmap out and re-adding it guarantees the 16/24/32/.../256 set is
    available to Windows shell.
    """
    icon_path = _pyber_icon_path()
    if not os.path.isfile(icon_path):
        logging.warning("App icon not found at %s", icon_path)
        return None
    try:
        from_file = QtGui.QIcon(icon_path)
        if from_file.isNull():
            logging.warning("QIcon failed to load %s", icon_path)
            return None
        icon = QtGui.QIcon()
        sizes = from_file.availableSizes()
        if not sizes:
            sizes = [QtCore.QSize(s, s) for s in (16, 24, 32, 48, 64, 128, 256)]
        for size in sizes:
            pixmap = from_file.pixmap(size)
            if not pixmap.isNull():
                icon.addPixmap(pixmap)
        if icon.isNull():
            # Last-resort fallback: use whatever QIcon parsed from the file.
            icon = from_file
        logging.info("pyBer icon built from %s (sizes: %s)",
                     icon_path, [s.width() for s in sizes])
        return icon
    except Exception as exc:
        logging.warning("Failed to build pyBer icon: %s", exc)
        return None


def _set_qt_application_icon(app: QtWidgets.QApplication) -> None:
    icon = _build_pyber_icon()
    if icon is None:
        return
    try:
        app.setWindowIcon(icon)
    except Exception as exc:
        logging.warning("setWindowIcon (app) failed: %s", exc)


def _set_qt_window_icon(window: QtWidgets.QWidget) -> None:
    icon = _build_pyber_icon()
    if icon is None:
        return
    try:
        window.setWindowIcon(icon)
    except Exception:
        pass


def _force_windows_taskbar_icon(window: QtWidgets.QWidget) -> None:
    """Send WM_SETICON to the HWND so the Windows taskbar picks up the icon.

    Required when running under python.exe in dev mode: Qt's setWindowIcon
    updates the title bar but Windows often keeps the taskbar entry's icon
    pointing at the python.exe resource. WM_SETICON tells the OS to use a
    specific HICON for this window's taskbar entry and alt-tab thumbnail.
    """
    if os.name != "nt":
        return
    try:
        import ctypes
        from ctypes import wintypes
        hwnd = int(window.winId())
        if not hwnd:
            return
        ico_path = _pyber_icon_path()
        if not os.path.isfile(ico_path):
            return
        IMAGE_ICON = 1
        LR_LOADFROMFILE = 0x00000010
        ICON_SMALL = 0
        ICON_BIG = 1
        WM_SETICON = 0x0080
        LoadImageW = ctypes.windll.user32.LoadImageW
        LoadImageW.argtypes = [
            wintypes.HINSTANCE, wintypes.LPCWSTR, wintypes.UINT,
            ctypes.c_int, ctypes.c_int, wintypes.UINT,
        ]
        LoadImageW.restype = wintypes.HANDLE
        SendMessageW = ctypes.windll.user32.SendMessageW
        SendMessageW.argtypes = [
            wintypes.HWND, wintypes.UINT, wintypes.WPARAM, wintypes.LPARAM,
        ]
        SendMessageW.restype = wintypes.LPARAM
        h_big = LoadImageW(
            None, ico_path, IMAGE_ICON, 32, 32, LR_LOADFROMFILE,
        )
        h_small = LoadImageW(
            None, ico_path, IMAGE_ICON, 16, 16, LR_LOADFROMFILE,
        )
        if h_big:
            SendMessageW(hwnd, WM_SETICON, ICON_BIG, h_big)
        if h_small:
            SendMessageW(hwnd, WM_SETICON, ICON_SMALL, h_small)
        logging.info(
            "WM_SETICON applied to HWND 0x%X (big=%s small=%s)",
            hwnd, bool(h_big), bool(h_small),
        )
    except Exception as exc:
        logging.warning("WM_SETICON path failed: %s", exc)


def _rolling_corr(x: np.ndarray, y: np.ndarray, win: int) -> Tuple[np.ndarray, np.ndarray]:
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    if win <= 10 or x.size < win or y.size < win:
        return np.array([], float), np.array([], int)
    step = max(10, win // 2)
    rs = []
    centers = []
    for i in range(0, x.size - win + 1, step):
        xx = x[i:i + win]
        yy = y[i:i + win]
        m = np.isfinite(xx) & np.isfinite(yy)
        if np.sum(m) < 10:
            r = np.nan
        else:
            r = float(np.corrcoef(xx[m], yy[m])[0, 1])
        rs.append(r)
        centers.append(i + win // 2)
    return np.asarray(rs, float), np.asarray(centers, int)


# ----------------------------------------------------------------------------
# Quality-check verdict: tiered overall score + per-metric breakdown.
# ----------------------------------------------------------------------------

# Tier table: (min_score_inclusive, label, color)
_QC_TIERS: List[Tuple[float, str, str]] = [
    (85.0, "EXCELLENT", "#5dd39e"),
    (70.0, "GOOD",      "#9ce0a3"),
    (55.0, "FAIR",      "#f5c542"),
    (40.0, "MARGINAL",  "#f0915e"),
    (0.0,  "POOR",      "#ee6471"),
]


def _qc_tier_for(score: float) -> Tuple[str, str]:
    for thr, name, color in _QC_TIERS:
        if score >= thr:
            return name, color
    return _QC_TIERS[-1][1], _QC_TIERS[-1][2]


def _qc_float(qc: Dict[str, object], key: str, default: float = float("nan")) -> float:
    try:
        value = float(qc.get(key, default))
    except Exception:
        return default
    return value if np.isfinite(value) else default


def _qc_robust_sigma(values: np.ndarray) -> float:
    arr = np.asarray(values, float)
    arr = arr[np.isfinite(arr)]
    if arr.size < 3:
        return float("nan")
    med = float(np.nanmedian(arr))
    mad = float(np.nanmedian(np.abs(arr - med)))
    if np.isfinite(mad) and mad > 1e-12:
        return float(1.4826 * mad)
    std = float(np.nanstd(arr))
    return std if np.isfinite(std) else float("nan")


# ----------------------------------------------------------------------------
# Strict per-metric thresholds: (WARN boundary, FAIL boundary).
#
# Every metric is graded PASS / WARN / FAIL against these absolute boundaries;
# there are no user-tunable weights anywhere in the verdict. The values are
# deliberately tighter than the permissive set used before, because that set let
# visibly compromised recordings land on GOOD. Each boundary now sits at the
# conservative end of the fiber-photometry literature and of common lab
# practice (Patel et al. 2020 on motion correction, De Jong et al. 2019 on
# artifact rates, Sherathiya et al. 2021 and Bruno et al. 2021 on SNR,
# bleaching and reference stability).
# ----------------------------------------------------------------------------
_QC_T_ARTIFACT = (2.0, 8.0)      # % of samples flagged as artifact
_QC_T_MOTION_R = (0.40, 0.80)    # |r| between signal dF/F and isobestic dF/F
_QC_T_SNR = (6.0, 3.0)           # event amplitude / noise floor (higher is better)
_QC_T_SIG_NOISE = (0.30, 0.60)   # signal high-frequency noise, % dF/F
_QC_T_REF_NOISE = (0.30, 0.60)   # isobestic high-frequency noise, % dF/F
_QC_T_ROLL_STD = (0.12, 0.25)    # std of the rolling reference correlation
_QC_T_BLEACH = (8.0, 20.0)       # |baseline drift| across the session, %
_QC_T_CUT_FRAC = (5.0, 20.0)     # % of the session that should be cut out

# Corrected-output distribution boundaries (heavy tails / off-centre median).
_QC_T_Z5_FAIL = 1.00             # % of corrected samples beyond |z| > 5
_QC_T_Z5NEG_FAIL = 0.25          # % of corrected samples below z = -5 (artifact)
_QC_T_Z3_WARN = 3.0              # % of corrected samples beyond |z| > 3
_QC_T_MEDIAN_WARN = 0.20         # |median| of the corrected z distribution
_QC_T_KURT_WARN = 6.0            # |excess kurtosis| of the corrected distribution
_QC_T_KURT_FAIL = 20.0

_QC_PASS_SCORE = 88.0
_QC_WARN_SCORE = 55.0
_QC_FAIL_SCORE = 22.0


def _qc_decide_tier(value: float, warn_thr: float, fail_thr: float,
                    *, higher_is_worse: bool = True) -> Tuple[str, float]:
    """Classify a value against two thresholds. Returns (tier_label, score)."""
    if not np.isfinite(value):
        # Missing data is treated as WARN, not FAIL: we don't know it's bad,
        # but we can't confirm it's good either.
        return "WARN", _QC_WARN_SCORE
    if higher_is_worse:
        if value < warn_thr:
            return "PASS", _QC_PASS_SCORE
        if value < fail_thr:
            return "WARN", _QC_WARN_SCORE
        return "FAIL", _QC_FAIL_SCORE
    else:
        if value >= warn_thr:
            return "PASS", _QC_PASS_SCORE
        if value >= fail_thr:
            return "WARN", _QC_WARN_SCORE
        return "FAIL", _QC_FAIL_SCORE


# ----------------------------------------------------------------------------
# Time-resolved problem detection.
#
# The tiered metrics above summarise the whole session with a single number, so
# they cannot tell the user that "the recording is fine except for 90 seconds in
# the middle". The helpers below scan the session in short bins and return the
# concrete time spans that should be cut out (Advanced Options -> Cut out
# regions), which is what the recommendation panel shows.
# ----------------------------------------------------------------------------

_QC_SEG_MIN_S = 2.0          # ignore flagged spans shorter than this
_QC_SEG_MAX_REPORTED = 8     # keep the recommendation list readable

# Human-readable label for each detector.
_QC_SEG_LABELS = {
    "bleach_in": "Bleach-in",
    "artifact": "Artifact burst",
    "decoupled": "Reference decoupled",
    "noise": "Noise burst",
    "flat": "Flat / dead signal",
}


def _qc_fmt_span(a: float, b: float) -> str:
    """Format a time span in seconds, the unit the cut-out table expects."""
    if (b - a) >= 10.0:
        return f"{a:.0f}-{b:.0f} s"
    return f"{a:.1f}-{b:.1f} s"


def _qc_merge_spans(spans: List[Tuple[float, float]], gap_s: float) -> List[Tuple[float, float]]:
    """Merge overlapping spans, and spans separated by less than ``gap_s``."""
    ordered = sorted(
        (float(a), float(b))
        for a, b in spans
        if np.isfinite(a) and np.isfinite(b) and b > a
    )
    merged: List[Tuple[float, float]] = []
    for a, b in ordered:
        if merged and (a - merged[-1][1]) <= gap_s:
            merged[-1] = (merged[-1][0], max(merged[-1][1], b))
        else:
            merged.append((a, b))
    return merged


def _qc_span_total(spans: List[Tuple[float, float]]) -> float:
    """Total covered time of a set of spans, counting overlaps only once."""
    return float(sum(b - a for a, b in _qc_merge_spans(spans, 0.0)))


def _qc_bins(t: np.ndarray, bin_s: float) -> List[Tuple[int, int, float, float]]:
    """Split a time vector into consecutive bins.

    Returns a list of ``(index_lo, index_hi, t_start, t_end)`` tuples. Bins with
    fewer than four samples are dropped so that per-bin robust statistics stay
    meaningful.
    """
    arr = np.asarray(t, float)
    if arr.size < 8 or not np.isfinite(bin_s) or bin_s <= 0:
        return []
    t0, t1 = float(arr[0]), float(arr[-1])
    if not (np.isfinite(t0) and np.isfinite(t1)) or t1 <= t0:
        return []
    count = int(min(4000, max(1, np.ceil((t1 - t0) / bin_s))))
    edges = t0 + bin_s * np.arange(count + 1, dtype=float)
    edges[-1] = max(float(edges[-1]), t1)
    idx = np.searchsorted(arr, edges, side="left")
    out: List[Tuple[int, int, float, float]] = []
    for i in range(count):
        lo, hi = int(idx[i]), int(idx[i + 1])
        if hi - lo >= 4:
            out.append((lo, hi, float(edges[i]), float(min(edges[i + 1], t1))))
    return out


def _qc_seg(start: float, end: float, kind: str, severity: float, detail: str) -> Dict[str, object]:
    """Build one flagged-span record."""
    return {
        "start": float(start),
        "end": float(end),
        "kind": str(kind),
        "label": _QC_SEG_LABELS.get(kind, kind),
        "severity": float(severity),
        "detail": str(detail),
    }


def _qc_detect_bad_segments(qc: Dict[str, object]) -> List[Dict[str, object]]:
    """Find the stretches of a recording that are not worth analysing.

    Five independent detectors run over the session:

    1. ``bleach_in``  - the steep fluorescence drop at the very start, which no
       baseline model handles gracefully.
    2. ``artifact``   - bins where the smart artifact detector flags a quarter
       or more of the samples.
    3. ``decoupled``  - bins where the isobestic reference stops tracking the
       signal (or flips sign), so motion correction would *inject* artifact.
    4. ``noise``      - bins whose high-frequency noise is several times the
       typical noise floor of the same recording.
    5. ``flat``       - bins with almost no variance at all (fibre unplugged,
       LED off, or a saturated detector).

    Returns the flagged spans sorted by start time.
    """
    t = np.asarray(qc.get("t", []), float)
    if t.size < 16:
        return []
    duration = float(t[-1] - t[0])
    if not np.isfinite(duration) or duration <= 0:
        return []

    # Bin width: ~1/60th of the session, clamped to a sane 5-30 s range.
    bin_s = float(np.clip(duration / 60.0, 5.0, 30.0))
    bins = _qc_bins(t, bin_s)
    has_reference = bool(qc.get("has_reference", True))
    segments: List[Dict[str, object]] = []

    # --- 1) Bleach-in transient at the start of the session -----------------
    base = np.asarray(qc.get("sig_base", []), float)
    fs = _qc_float(qc, "fs", float("nan"))
    if base.size == t.size and base.size > 32 and np.isfinite(fs) and fs > 0:
        step = max(1, int(round(fs)))                      # coarse ~1 s grid
        tb = t[::step]
        bb = base[::step]
        finite = np.isfinite(bb)
        if int(np.sum(finite)) > 8:
            tb, bb = tb[finite], bb[finite]
            start_level = float(bb[0])
            end_level = float(np.nanmedian(bb[max(1, bb.size // 2):]))
            drop = start_level - end_level
            # Only meaningful when the session actually loses >=3% of its
            # fluorescence and the trace starts high (i.e. a real bleach-in).
            if start_level > 1e-9 and drop > 0 and (drop / start_level) >= 0.03:
                frac_done = (start_level - bb) / drop      # share of the total drop
                reached50 = np.flatnonzero(frac_done >= 0.5)
                if reached50.size:
                    i50 = int(reached50[0])
                    # A bleach-in is "front-loaded": half the whole session drop
                    # happens inside the first 15% of the recording.
                    if (tb[i50] - tb[0]) <= 0.15 * duration:
                        reached80 = np.flatnonzero(frac_done >= 0.8)
                        i80 = int(reached80[0]) if reached80.size else i50
                        cut_to = min(float(tb[i80]), float(tb[0] + 0.15 * duration))
                        cut_to = max(cut_to, float(tb[i50]))
                        span = cut_to - float(tb[0])
                        if span >= _QC_SEG_MIN_S:
                            lost = (start_level - float(np.interp(cut_to, tb, bb))) / start_level * 100.0
                            share = float(np.interp(cut_to, tb, frac_done)) * 100.0
                            segments.append(_qc_seg(
                                float(tb[0]), cut_to, "bleach_in",
                                severity=min(1.0, share / 100.0),
                                detail=(f"fluorescence falls {lost:.0f}% here, which is "
                                        f"{share:.0f}% of the whole session's drop"),
                            ))

    # --- 2) Artifact-dense stretches ----------------------------------------
    art_mask = np.asarray(qc.get("art_mask", []), bool)
    if bins and art_mask.size == t.size:
        hot = [(a, b) for lo, hi, a, b in bins if float(np.mean(art_mask[lo:hi])) >= 0.25]
        for a, b in _qc_merge_spans(hot, bin_s * 0.6):
            lo = int(np.searchsorted(t, a, side="left"))
            hi = int(np.searchsorted(t, b, side="right"))
            frac = float(np.mean(art_mask[lo:hi])) if hi > lo else 0.0
            if (b - a) >= _QC_SEG_MIN_S:
                segments.append(_qc_seg(
                    a, b, "artifact", severity=frac,
                    detail=f"{frac * 100:.0f}% of the samples here are flagged as artifact",
                ))

    # --- 3) Reference decoupling / sign flips -------------------------------
    r_roll = np.asarray(qc.get("r_roll", []), float)
    centers = np.asarray(qc.get("r_centers", []), int)
    r_win_s = _qc_float(qc, "r_win_s", 10.0)
    if has_reference and r_roll.size and centers.size == r_roll.size:
        tc = t[np.clip(centers, 0, t.size - 1)]
        finite = np.isfinite(r_roll)
        if int(np.sum(finite)) > 4:
            typical = float(np.nanmedian(r_roll[finite]))
            strong_branch = abs(typical) < 0.30
            if not strong_branch:
                # The reference normally tracks the signal, so the broken windows
                # are the ones where the correlation actually flips sign: there,
                # regressing the reference out *adds* variance instead of
                # removing movement.
                bad = finite & (np.sign(typical) * r_roll < 0.0)
                reason = "the isobestic reference stops tracking the signal (correlation flips sign)"
            else:
                # The reference normally carries nothing, so windows of strong
                # coupling are movement bursts.
                bad = finite & (np.abs(r_roll) > 0.70)
                reason = "a burst of strong signal/reference coupling (movement)"
            half = max(0.5, float(r_win_s) * 0.5)
            spans = [(float(tc[i] - half), float(tc[i] + half)) for i in np.flatnonzero(bad)]
            # Require more than a single rolling window so that one noisy
            # estimate does not turn into a recommendation to cut.
            for a, b in _qc_merge_spans(spans, float(r_win_s)):
                if (b - a) < max(_QC_SEG_MIN_S, 1.5 * r_win_s):
                    continue
                sel = (tc >= a) & (tc <= b) & finite
                detail = reason
                if np.any(sel):
                    vals = r_roll[sel]
                    # Quote the most telling rolling-r value inside the span: the
                    # strongest coupling for a movement burst, the deepest
                    # sign flip for a decoupling.
                    worst = (float(vals[int(np.argmax(np.abs(vals)))]) if strong_branch
                             else float(vals[int(np.argmin(np.sign(typical) * vals))]))
                    if np.isfinite(worst):
                        detail += f" (rolling r reaches {worst:+.2f})"
                segments.append(_qc_seg(
                    max(a, float(t[0])), min(b, float(t[-1])), "decoupled",
                    severity=0.6, detail=detail,
                ))

    # --- 4) Noise bursts and 5) flat / dead stretches ------------------------
    hf = np.asarray(qc.get("hf_sig_pct", []), float)
    if len(bins) >= 6 and hf.size == t.size:
        sigmas = np.array([_qc_robust_sigma(hf[lo:hi]) for lo, hi, _a, _b in bins], float)
        good = np.isfinite(sigmas)
        med = float(np.nanmedian(sigmas[good])) if np.any(good) else float("nan")
        if np.isfinite(med) and med > 1e-6:
            noisy = [(a, b) for (lo, hi, a, b), s in zip(bins, sigmas)
                     if np.isfinite(s) and s >= 3.0 * med]
            for a, b in _qc_merge_spans(noisy, bin_s * 0.6):
                sel = [(s, lo, hi) for (lo, hi, ba, bb_), s in zip(bins, sigmas)
                       if np.isfinite(s) and not (bb_ <= a or ba >= b)]
                worst = max((s for s, _lo, _hi in sel), default=float("nan"))
                if (b - a) >= _QC_SEG_MIN_S and np.isfinite(worst):
                    segments.append(_qc_seg(
                        a, b, "noise", severity=min(1.0, worst / (10.0 * med)),
                        detail=(f"noise here is {worst / med:.1f}x the rest of the "
                                f"recording ({worst:.2f}% dF/F)"),
                    ))
            dead = [(a, b) for (lo, hi, a, b), s in zip(bins, sigmas)
                    if np.isfinite(s) and s <= 0.12 * med]
            for a, b in _qc_merge_spans(dead, bin_s * 0.6):
                if (b - a) >= max(_QC_SEG_MIN_S, bin_s):
                    segments.append(_qc_seg(
                        a, b, "flat", severity=0.8,
                        detail="almost no signal variance - check for an unplugged "
                               "patch cord, LED dropout or detector saturation",
                    ))

    # A bleach-in span already swallows whatever else happens at the very start
    # of the session, so drop anything mostly hidden inside it: showing the same
    # seconds three times only makes the recommendation list harder to read.
    bleach_spans = [(float(s["start"]), float(s["end"])) for s in segments if s["kind"] == "bleach_in"]
    if bleach_spans:
        def _mostly_inside(seg: Dict[str, object]) -> bool:
            if seg["kind"] == "bleach_in":
                return False
            a, b = float(seg["start"]), float(seg["end"])
            span = max(b - a, 1e-9)
            covered = sum(max(0.0, min(b, hi) - max(a, lo)) for lo, hi in bleach_spans)
            return (covered / span) >= 0.8

        segments = [s for s in segments if not _mostly_inside(s)]

    segments.sort(key=lambda s: float(s["start"]))
    return segments


# Map overall verdict tier to a representative score for the headline card.
_QC_TIER_SCORE = {
    "EXCELLENT": 92.0,
    "GOOD": 78.0,
    "FAIR": 62.0,
    "MARGINAL": 42.0,
    "POOR": 22.0,
}


# Short, jargon-free description of what a failing metric actually means. Used
# to build the "why this verdict" sentence out of the metrics that went wrong.
_QC_PLAIN_PROBLEM = {
    "Artifact load": "too much of the trace is artifact",
    "Motion bleed": "the signal mostly repeats what the 405 reference does",
    "Usable SNR": "transients barely rise above the noise",
    "Isobestic noise": "the 405 reference is too noisy to correct with",
    "Usable coverage": "a large part of the session has to be thrown away",
    "Signal noise floor": "the signal itself is noisy",
    "Temporal stability": "the reference coupling changes during the session",
    "Corrected output": "outliers survive into the corrected trace",
    "Photobleach": "the baseline drifts a lot across the session",
}

# What the user should physically do with the file, per overall tier.
# (chip label, chip colour, one-sentence instruction)
_QC_ACTION_BY_TIER = {
    "EXCELLENT": ("KEEP", "#5dd39e",
                  "Keep this recording as it is. Every check passed."),
    "GOOD": ("KEEP", "#5dd39e",
             "Keep this recording. Anything below is optional polish."),
    "FAIR": ("USE WITH CARE", "#f5c542",
             "Usable, but clean it up before pooling it with the rest of the dataset, "
             "and say in the methods what you did."),
    "MARGINAL": ("REPAIR FIRST", "#f0915e",
                 "Do not analyse this file as it stands. Apply the fixes below and run "
                 "the check again; if the verdict does not improve, leave the file out."),
    "POOR": ("EXCLUDE", "#ee6471",
             "Leave this recording out of the analysis. Too many core checks failed for "
             "the result to mean anything."),
}


@dataclass
class QcVerdict:
    """Everything the quality panel needs to show, in one object."""

    score: float                                   # representative 0-100 score
    tier: str                                      # EXCELLENT ... POOR
    color: str                                     # colour for the tier
    metrics: List[Tuple[str, float, float, str, str]]  # (name, score, criticality, why, tier)
    why: str = ""                                  # friendly "why this verdict"
    counts: str = ""                               # dim one-liner with the tallies
    action_kind: str = "KEEP"                      # KEEP / USE WITH CARE / ...
    action_color: str = "#5dd39e"
    headline: str = ""                             # what to do with the file
    actions: List[str] = field(default_factory=list)     # concrete fixes
    segments: List[Dict[str, object]] = field(default_factory=list)  # spans to cut
    cut_seconds: float = 0.0
    cut_fraction: float = 0.0
    duration_s: float = float("nan")


def _qc_build_recommendations(
    qc: Dict[str, object],
    tiers: Dict[str, str],
    tier_name: str,
    segments: List[Dict[str, object]],
    cut_seconds: float,
    cut_fraction: float,
    duration_s: float,
) -> Tuple[str, str, str, List[str]]:
    """Turn the graded metrics into plain-language, actionable advice.

    Returns ``(action_kind, action_color, headline, actions)``. Each action is a
    complete sentence the user can act on without knowing the metric names.
    """
    kind, color, headline = _QC_ACTION_BY_TIER.get(tier_name, _QC_ACTION_BY_TIER["FAIR"])
    actions: List[str] = []

    def bad(name: str) -> bool:
        return tiers.get(name) in ("WARN", "FAIL")

    def failed(name: str) -> bool:
        return tiers.get(name) == "FAIL"

    r_abs = abs(_qc_float(qc, "r", float("nan")))
    snr = _qc_float(qc, "usable_snr", float("nan"))
    art_pct = _qc_float(qc, "art_frac", 0.0) * 100.0
    ref_noise = _qc_float(qc, "ref_hf_noise_pct", float("nan"))
    sig_noise = _qc_float(qc, "hf_noise_pct", float("nan"))
    roll_std = _qc_float(qc, "r_roll_std", float("nan"))
    bleach = _qc_float(qc, "bleach_pct", float("nan"))
    frac_gt3 = _qc_float(qc, "frac_gt3", 0.0)
    frac_gt5 = _qc_float(qc, "frac_gt5", 0.0)
    frac_neg5 = _qc_float(qc, "frac_neg5", 0.0)
    kurt = _qc_float(qc, "kurt", 0.0)
    kept_s = duration_s - cut_seconds if np.isfinite(duration_s) else float("nan")

    # --- Escalations that override the tier-based headline -------------------
    # If most of the session has to be cut, or almost nothing survives, the file
    # is not salvageable no matter how good the surviving part looks.
    if cut_fraction >= 0.35:
        kind, color = "EXCLUDE", "#ee6471"
        headline = (f"Leave this recording out: about {cut_fraction * 100:.0f}% of the session "
                    f"has to be cut, so what is left is no longer a fair sample of it.")
    elif np.isfinite(kept_s) and kept_s < 60.0 and np.isfinite(duration_s) and duration_s > 60.0:
        kind, color = "EXCLUDE", "#ee6471"
        headline = (f"Leave this recording out: only about {max(kept_s, 0.0):.0f} s survive the "
                    f"cuts below, which is too short to analyse.")

    # --- Cutting advice ------------------------------------------------------
    if segments:
        n_seg = len(segments)
        actions.append(
            f"Cut the {n_seg} {'span' if n_seg == 1 else 'spans'} listed below "
            f"({cut_seconds:.0f} s in total, {cut_fraction * 100:.0f}% of the session) in "
            f"Advanced Options -> Cut out regions, then run this check again."
        )
        kinds = {str(s.get("kind", "")) for s in segments}
        if "bleach_in" in kinds:
            actions.append(
                "The start of the session is a steep bleach-in. No baseline model handles "
                "that cleanly, so trimming it is almost always better than fitting through it."
            )
        if "flat" in kinds:
            actions.append(
                "At least one stretch has essentially no signal variance. Check the patch cord, "
                "the LED and the detector gain for that period before trusting anything around it."
            )

    # --- Per-metric advice ---------------------------------------------------
    if bad("Artifact load"):
        actions.append(
            f"Artifacts cover {art_pct:.1f}% of the samples. Keep artifact handling on "
            f"'Interpolate' for short hits, and cut the long ones instead of interpolating "
            f"across them."
        )
    if failed("Motion bleed"):
        actions.append(
            f"The 465 and 405 traces move together almost perfectly (|r|={r_abs:.2f}, "
            f"{r_abs * r_abs * 100:.0f}% shared variance). Use a fitted-reference dF/F output, "
            f"then look at the corrected trace: if the transients disappear with the reference, "
            f"this recording is measuring movement rather than your sensor and should be dropped."
        )
    elif bad("Motion bleed"):
        actions.append(
            f"There is real movement bleed (|r|={r_abs:.2f}). Keep the fitted-reference dF/F "
            f"output so the shared component is regressed out before you score events."
        )
    if failed("Usable SNR"):
        actions.append(
            f"Transients are only {snr:.1f}x the noise floor, so event amplitudes here are "
            f"mostly noise. Exclude the file, and for the next session raise LED power, check "
            f"the fibre coupling, and confirm the implant is on target."
        )
    elif bad("Usable SNR"):
        actions.append(
            f"Signal-to-noise is modest ({snr:.1f}x). Lower the low-pass cutoff or widen the "
            f"smoothing window before scoring events, and avoid reading anything into small peaks."
        )
    if failed("Isobestic noise"):
        actions.append(
            f"The 405 reference is too noisy to correct with ({ref_noise:.2f}% dF/F). Switch the "
            f"output to a signal-only dF/F: subtracting this reference would add noise rather "
            f"than remove movement. Ignore the motion-bleed number above as well."
        )
    elif bad("Isobestic noise"):
        actions.append(
            f"The 405 reference is noisy ({ref_noise:.2f}% dF/F). Compare corrected and "
            f"uncorrected traces side by side before you commit to motion correction."
        )
    if bad("Signal noise floor"):
        actions.append(
            f"The signal noise floor is high ({sig_noise:.2f}% dF/F). Check the fibre connection "
            f"and ambient light, and consider a lower low-pass cutoff."
        )
    if bad("Temporal stability"):
        actions.append(
            f"Movement coupling is not stable over the session (rolling-r std={roll_std:.2f}), "
            f"which usually means the fibre or the animal moved. Either cut the unstable spans, "
            f"or split the file into sections and process them separately."
        )
    if bad("Corrected output"):
        tail = (f" {frac_neg5:.2f}% of it dips below z=-5, which no sensor produces."
                if frac_neg5 > 0.05 else "")
        actions.append(
            f"The corrected trace still has heavy tails ({frac_gt3:.2f}% beyond |z|>3, "
            f"{frac_gt5:.2f}% beyond |z|>5, kurtosis {kurt:+.0f}).{tail} Re-run artifact detection "
            f"at a higher sensitivity, otherwise these spikes will be counted as events."
        )
    if failed("Photobleach"):
        actions.append(
            f"Fluorescence changes {bleach:+.0f}% across the session. Cut the bleach-in, keep an "
            f"adaptive baseline (airPLS), and never compare raw amplitudes between the start and "
            f"the end of this file. Lower the LED power next time."
        )
    elif bad("Photobleach"):
        actions.append(
            f"Fluorescence drifts {bleach:+.0f}% across the session. Keep the adaptive baseline on "
            f"and be careful comparing early and late events."
        )

    if not actions:
        actions.append("Nothing to fix. Process it with your standard settings.")

    return kind, color, headline, actions


def _evaluate_qc(qc: Dict[str, object]) -> QcVerdict:
    """Grade a fiber-photometry recording and say what to do with it.

    Every metric is classified PASS / WARN / FAIL against the strict absolute
    thresholds defined above (no user-chosen weights anywhere). The overall
    verdict is set by the worst *critical* metric; advisory metrics can only
    pull the verdict down, never lift it.

    On top of the grade, the returned :class:`QcVerdict` carries a
    plain-language explanation, a list of concrete fixes, and the exact time
    spans that should be cut out of the session.
    """
    art_frac_pct = _qc_float(qc, "art_frac", 0.0) * 100.0
    has_reference = bool(qc.get("has_reference", True))
    hf_noise_pct = _qc_float(qc, "hf_noise_pct", float("nan"))
    ref_hf_noise_pct = _qc_float(qc, "ref_hf_noise_pct", float("nan"))
    usable_snr = _qc_float(qc, "usable_snr", float("nan"))
    frac_gt3 = _qc_float(qc, "frac_gt3", 0.0)
    frac_gt5 = _qc_float(qc, "frac_gt5", 0.0)
    frac_neg5 = _qc_float(qc, "frac_neg5", 0.0)
    r_abs = abs(_qc_float(qc, "r", float("nan")))
    r_roll_std = _qc_float(qc, "r_roll_std", 0.0)
    bleach_pct = _qc_float(qc, "bleach_pct", 0.0)
    bleach_abs = abs(bleach_pct)
    median = abs(_qc_float(qc, "q50", 0.0))
    kurt = _qc_float(qc, "kurt", 0.0)
    skew = _qc_float(qc, "skew", 0.0)

    # Time-resolved scan: which stretches of the session are unusable.
    t_arr = np.asarray(qc.get("t", []), float)
    duration_s = float(t_arr[-1] - t_arr[0]) if t_arr.size >= 2 else float("nan")
    segments = _qc_detect_bad_segments(qc)
    cut_seconds = _qc_span_total([(float(s["start"]), float(s["end"])) for s in segments])
    cut_fraction = (cut_seconds / duration_s) if (np.isfinite(duration_s) and duration_s > 0) else 0.0
    cut_pct = cut_fraction * 100.0

    # 5-tuple: (name, score, criticality, why, tier)
    metrics: List[Tuple[str, float, float, str, str]] = []

    # ---- Critical metrics (gate the verdict) -------------------------------

    # 1) Artifact load. Tightened from 5/15% to 2/8%: above ~8% flagged samples
    # the interpolated trace is mostly invention, not measurement.
    warn, fail = _QC_T_ARTIFACT
    tier_art, score_art = _qc_decide_tier(art_frac_pct, warn_thr=warn, fail_thr=fail)
    why_art = {
        "PASS": f"Only {art_frac_pct:.2f}% of samples are artifact - clean trace (pass below {warn:.0f}%).",
        "WARN": f"{art_frac_pct:.2f}% of samples are artifact - the trace is being patched in places "
                f"(warn {warn:.0f}-{fail:.0f}%).",
        "FAIL": f"{art_frac_pct:.2f}% of samples are artifact - too much of this trace is "
                f"reconstructed rather than measured (fail above {fail:.0f}%).",
    }[tier_art]
    metrics.append(("Artifact load", score_art, 1.0, why_art, tier_art))

    # 2) Motion bleed |r|. Tightened from 0.45/0.90 to 0.40/0.80. At |r|=0.8 the
    # reference already explains ~64% of the signal variance, which is the point
    # where "correctable movement" turns into "the signal is the movement".
    # When the isobestic channel is itself too noisy the |r| reading stops being
    # interpretable, so a caveat is appended instead of silently retiering it.
    if has_reference:
        warn, fail = _QC_T_MOTION_R
        ref_unreliable = np.isfinite(ref_hf_noise_pct) and ref_hf_noise_pct >= _QC_T_REF_NOISE[1]
        if not np.isfinite(r_abs):
            tier_mot, score_mot = "WARN", _QC_WARN_SCORE
            why_mot = "Signal/reference correlation could not be computed - check the rolling-r plot by eye."
        else:
            tier_mot, score_mot = _qc_decide_tier(r_abs, warn_thr=warn, fail_thr=fail)
            shared = r_abs * r_abs * 100.0
            why_mot = {
                "PASS": f"|r|={r_abs:.2f} - the signal moves on its own, independently of the "
                        f"reference (pass below {warn:.2f}).",
                "WARN": f"|r|={r_abs:.2f} - {shared:.0f}% of the signal is shared with the reference; "
                        f"correctable movement (warn {warn:.2f}-{fail:.2f}).",
                "FAIL": f"|r|={r_abs:.2f} - {shared:.0f}% of the signal is shared with the reference, "
                        f"so it is mostly movement, not biology (fail above {fail:.2f}).",
            }[tier_mot]
        if ref_unreliable:
            why_mot += (" Treat this number with suspicion: the reference is too noisy for "
                        "|r| to mean much.")
        metrics.append(("Motion bleed", score_mot, 1.0, why_mot, tier_mot))

    # 3) Usable SNR. Tightened from 4/1.5 to 6/3: below ~3x the noise floor,
    # peak amplitudes and event counts are dominated by noise.
    warn, fail = _QC_T_SNR
    if not np.isfinite(usable_snr):
        tier_snr, score_snr = "WARN", _QC_WARN_SCORE
        why_snr = "Signal-to-noise could not be computed - the trace may be too short or invalid."
    else:
        tier_snr, score_snr = _qc_decide_tier(usable_snr, warn_thr=warn, fail_thr=fail,
                                              higher_is_worse=False)
        why_snr = {
            "PASS": f"Transients are ~{usable_snr:.1f}x the noise floor - clearly readable "
                    f"(pass above {warn:.0f}x).",
            "WARN": f"Transients are ~{usable_snr:.1f}x the noise floor - readable but not robust "
                    f"(warn {fail:.0f}-{warn:.0f}x).",
            "FAIL": f"Transients are only ~{usable_snr:.1f}x the noise floor - amplitudes here are "
                    f"mostly noise (fail below {fail:.0f}x).",
        }[tier_snr]
    metrics.append(("Usable SNR", score_snr, 1.0, why_snr, tier_snr))

    # 4) Isobestic HF noise (critical). If the 405 reference is itself noisy,
    # motion correction adds noise instead of removing movement, and the |r|
    # reading above becomes meaningless. Tightened from 0.45/0.90 to 0.30/0.60.
    if has_reference:
        warn, fail = _QC_T_REF_NOISE
        if not np.isfinite(ref_hf_noise_pct):
            tier_ref, score_ref = "WARN", _QC_WARN_SCORE
            why_ref = "Reference noise could not be measured - inspect the 405 trace by eye."
        else:
            tier_ref, score_ref = _qc_decide_tier(ref_hf_noise_pct, warn_thr=warn, fail_thr=fail)
            why_ref = {
                "PASS": f"The 405 reference is quiet ({ref_hf_noise_pct:.2f}% dF/F), so motion "
                        f"correction will work (pass below {warn:.2f}%).",
                "WARN": f"The 405 reference is noisy ({ref_hf_noise_pct:.2f}% dF/F); correcting with "
                        f"it also injects some of that noise (warn {warn:.2f}-{fail:.2f}%).",
                "FAIL": f"The 405 reference is too noisy to correct with ({ref_hf_noise_pct:.2f}% dF/F) "
                        f"- use a signal-only output instead (fail above {fail:.2f}%).",
            }[tier_ref]
        metrics.append(("Isobestic noise", score_ref, 1.0, why_ref, tier_ref))

    # 5) Usable coverage (critical, new). Summarises the time-resolved scan: how
    # much of the session has to be thrown away before analysis. This is what
    # turns "some metric is bad" into "cut these spans / drop this file".
    warn, fail = _QC_T_CUT_FRAC
    tier_cov, score_cov = _qc_decide_tier(cut_pct, warn_thr=warn, fail_thr=fail)
    kept_s = duration_s - cut_seconds if np.isfinite(duration_s) else float("nan")
    kept_txt = f"{max(kept_s, 0.0):.0f} s usable" if np.isfinite(kept_s) else "usable length unknown"
    why_cov = {
        "PASS": f"Nothing worth cutting ({cut_pct:.1f}% flagged, {kept_txt}).",
        "WARN": f"{cut_pct:.0f}% of the session should be cut ({cut_seconds:.0f} s, {kept_txt}) "
                f"- see the spans below (warn {warn:.0f}-{fail:.0f}%).",
        "FAIL": f"{cut_pct:.0f}% of the session should be cut ({cut_seconds:.0f} s, {kept_txt}) "
                f"- what is left may no longer represent the session (fail above {fail:.0f}%).",
    }[tier_cov]
    metrics.append(("Usable coverage", score_cov, 1.0, why_cov, tier_cov))

    # ---- Advisory metrics (can only pull the verdict down) ------------------

    # 6) Signal HF noise floor. Tightened from 0.45/0.90 to 0.30/0.60.
    warn, fail = _QC_T_SIG_NOISE
    if not np.isfinite(hf_noise_pct):
        tier_nse, score_nse = "WARN", _QC_WARN_SCORE
        why_nse = "Signal noise could not be measured - inspect the raw trace by eye."
    else:
        tier_nse, score_nse = _qc_decide_tier(hf_noise_pct, warn_thr=warn, fail_thr=fail)
        why_nse = {
            "PASS": f"Signal noise is low ({hf_noise_pct:.2f}% dF/F, pass below {warn:.2f}%).",
            "WARN": f"Signal noise is high ({hf_noise_pct:.2f}% dF/F) - small events will be hard "
                    f"to trust (warn {warn:.2f}-{fail:.2f}%).",
            "FAIL": f"Signal noise dominates the trace ({hf_noise_pct:.2f}% dF/F, "
                    f"fail above {fail:.2f}%).",
        }[tier_nse]
    metrics.append(("Signal noise floor", score_nse, 0.5, why_nse, tier_nse))

    # 7) Temporal stability of the movement coupling. Tightened from 0.20/0.35
    # to 0.12/0.25: a coupling that wanders means one global correction cannot
    # be right everywhere in the session.
    if has_reference:
        warn, fail = _QC_T_ROLL_STD
        tier_roll, score_roll = _qc_decide_tier(r_roll_std, warn_thr=warn, fail_thr=fail)
        why_roll = {
            "PASS": f"Movement coupling stays the same all session (rolling-r std={r_roll_std:.2f}, "
                    f"pass below {warn:.2f}).",
            "WARN": f"Movement coupling drifts during the session (rolling-r std={r_roll_std:.2f}) - "
                    f"one global correction fits some parts better than others "
                    f"(warn {warn:.2f}-{fail:.2f}).",
            "FAIL": f"Movement coupling jumps around (rolling-r std={r_roll_std:.2f}) - the fibre or "
                    f"the animal moved; process in sections or cut the unstable spans "
                    f"(fail above {fail:.2f}).",
        }[tier_roll]
        metrics.append(("Temporal stability", score_roll, 0.5, why_roll, tier_roll))

    # 8) Corrected output distribution (heavy tails / off-centre median).
    # Tightened: WARN at 3% beyond |z|>3 (was 10%), median 0.20 (was 0.35),
    # kurtosis 6 (was 10). The FAIL rule leans on the *negative* tail, because a
    # large upward excursion can be a genuine transient while a fast five-sigma
    # drop below baseline is artifact almost by definition. A heavy but clearly
    # one-sided positive tail is therefore treated as strong signal, not as a
    # failure - otherwise the best recordings would be penalised for having
    # large events.
    neg_heavy = frac_neg5 > _QC_T_Z5NEG_FAIL
    pos_heavy = frac_gt5 > _QC_T_Z5_FAIL
    transient_like = (skew >= 1.0) and (frac_neg5 <= 0.05)
    if neg_heavy or abs(kurt) > _QC_T_KURT_FAIL or (pos_heavy and not transient_like):
        tier_dist, score_dist = "FAIL", _QC_FAIL_SCORE
        if neg_heavy:
            why_dist = (f"{frac_neg5:.2f}% of the corrected trace drops below z=-5. Sensors do "
                        f"not do that, so this is leftover artifact and it will be scored as "
                        f"events.")
        else:
            why_dist = (f"Extreme values survive the correction: {frac_gt5:.2f}% beyond |z|>5, "
                        f"kurtosis {kurt:+.0f}, and the tails are symmetric rather than "
                        f"transient-shaped.")
    elif frac_gt3 > _QC_T_Z3_WARN or median > _QC_T_MEDIAN_WARN or abs(kurt) > _QC_T_KURT_WARN:
        tier_dist, score_dist = "WARN", _QC_WARN_SCORE
        why_dist = (f"The corrected trace is lopsided: {frac_gt3:.2f}% beyond |z|>3, "
                    f"median {median:.2f}, kurtosis {kurt:+.0f} - check for leftover artifact "
                    f"before you trust event counts.")
    else:
        tier_dist, score_dist = "PASS", _QC_PASS_SCORE
        why_dist = (f"The corrected trace is well behaved ({frac_gt3:.2f}% beyond |z|>3, "
                    f"median {median:.2f}).")
    metrics.append(("Corrected output", score_dist, 0.5, why_dist, tier_dist))

    # 9) Photobleach across the session. Tightened from 10/30% to 8/20%.
    warn, fail = _QC_T_BLEACH
    tier_bl, score_bl = _qc_decide_tier(bleach_abs, warn_thr=warn, fail_thr=fail)
    why_bl = {
        "PASS": f"Fluorescence is stable ({bleach_pct:+.1f}% across the session, "
                f"pass below {warn:.0f}%).",
        "WARN": f"Fluorescence drifts {bleach_pct:+.1f}% across the session - early and late events "
                f"are not directly comparable (warn {warn:.0f}-{fail:.0f}%).",
        "FAIL": f"Fluorescence changes {bleach_pct:+.1f}% across the session - heavy bleaching or "
                f"saturation (fail above {fail:.0f}%).",
    }[tier_bl]
    metrics.append(("Photobleach", score_bl, 0.5, why_bl, tier_bl))

    # Keep critical rows above advisory rows so the panel shows one clean split.
    metrics.sort(key=lambda m: -float(m[2]))

    # ---- Aggregation: worst critical metric wins; advisory rows can only
    # ---- pull the verdict down. Stricter than before at every step.
    tiers_by_name = {name: tier for name, _s, _c, _w, tier in metrics}
    critical_tiers = [tier for _n, _s, crit, _w, tier in metrics if crit >= 1.0]
    advisory_tiers = [tier for _n, _s, crit, _w, tier in metrics if 0.0 < crit < 1.0]

    n_crit_fail = critical_tiers.count("FAIL")
    n_crit_warn = critical_tiers.count("WARN")
    n_sec_fail = advisory_tiers.count("FAIL")
    n_sec_warn = advisory_tiers.count("WARN")

    if n_crit_fail >= 2:
        tier_name = "POOR"
    elif n_crit_fail == 1:
        # One critical failure already disqualifies the file as it stands, but
        # it is often repairable (cut the bad spans, change the output mode), so
        # it lands on MARGINAL. It only drops to POOR when the rest of the
        # recording is also coming apart.
        tier_name = "POOR" if (n_sec_fail >= 2 or n_crit_warn >= 2) else "MARGINAL"
    elif n_crit_warn >= 3:
        tier_name = "MARGINAL"
    elif n_crit_warn == 2:
        tier_name = "MARGINAL" if n_sec_fail >= 1 else "FAIR"
    elif n_crit_warn == 1:
        tier_name = "FAIR" if (n_sec_fail >= 1 or n_sec_warn >= 2) else "GOOD"
    else:
        # All critical checks pass - only advisory rows are left to weigh.
        if n_sec_fail >= 2:
            tier_name = "FAIR"
        elif n_sec_fail == 1 or n_sec_warn >= 2:
            tier_name = "GOOD"
        elif n_sec_warn == 1:
            tier_name = "GOOD"
        else:
            tier_name = "EXCELLENT"

    # ---- Plain-language explanation ----------------------------------------
    failing = [name for name, _s, _c, _w, tier in metrics if tier == "FAIL"]
    warning = [name for name, _s, _c, _w, tier in metrics if tier == "WARN"]

    def _plain(names: List[str], limit: int = 3) -> str:
        """Join the plain-language problems, keeping the sentence readable."""
        parts = [_QC_PLAIN_PROBLEM.get(n, n.lower()) for n in names]
        extra = max(0, len(parts) - limit)
        parts = parts[:limit]
        if len(parts) == 1:
            text = parts[0]
        else:
            text = ", ".join(parts[:-1]) + " and " + parts[-1]
        if extra:
            text += f" (+{extra} more)"
        return text

    if failing:
        why = f"What is wrong: {_plain(failing)}."
        if warning:
            why += f" Also worth watching: {_plain(warning)}."
    elif warning:
        why = f"Nothing failed outright. Worth watching: {_plain(warning)}."
    else:
        why = "Every check passed: clean signal, quiet reference, stable baseline."

    if n_crit_fail:
        why += " The grade is set by the worst critical check."

    n_total = len(metrics)
    counts = (f"{n_crit_fail} critical failed, {n_crit_warn} critical warned, "
              f"{n_sec_fail} advisory failed, {n_sec_warn} advisory warned, "
              f"out of {n_total} checks.")

    action_kind, action_color, headline, actions = _qc_build_recommendations(
        qc, tiers_by_name, tier_name, segments, cut_seconds, cut_fraction, duration_s
    )

    overall_score = _QC_TIER_SCORE.get(tier_name, 0.0)
    _tier_name_dbg, tier_color = _qc_tier_for(overall_score)
    return QcVerdict(
        score=overall_score,
        tier=tier_name,
        color=tier_color,
        metrics=metrics,
        why=why,
        counts=counts,
        action_kind=action_kind,
        action_color=action_color,
        headline=headline,
        actions=actions,
        segments=segments,
        cut_seconds=cut_seconds,
        cut_fraction=cut_fraction,
        duration_s=duration_s,
    )


_QC_TIER_BADGE_COLORS = {
    "PASS": "#5dd39e",
    "WARN": "#f5c542",
    "FAIL": "#ee6471",
    "": "#6f7a8e",
}


# Colour of the recommendation panel. Green reads as "here is what to do",
# independently of how bad the recording turned out to be; the severity itself
# is carried by the KEEP / USE WITH CARE / REPAIR FIRST / EXCLUDE chip inside.
_QC_REC_GREEN = "#5dd39e"


def _qc_tint(color: str, alpha_pct: int) -> str:
    """Return a Qt stylesheet ``rgba()`` string for ``color`` at ``alpha_pct``.

    Qt parses eight-digit hex as ``#AARRGGBB`` (alpha first), so appending an
    alpha suffix to a ``#rrggbb`` value silently produces a completely different
    colour - a faint green tint comes out olive, for instance. Building the
    rgba() text explicitly keeps the tint predictable.
    """
    text = str(color or "").lstrip("#")
    if len(text) != 6:
        return str(color)
    try:
        r, g, b = int(text[0:2], 16), int(text[2:4], 16), int(text[4:6], 16)
    except ValueError:
        return str(color)
    return f"rgba({r}, {g}, {b}, {max(0, min(100, int(alpha_pct)))}%)"


class QualityVerdictCard(QtWidgets.QFrame):
    """Left-hand card: overall verdict, why, per-metric tiers, recommendations."""

    def __init__(self, verdict: QcVerdict, parent: Optional[QtWidgets.QWidget] = None):
        super().__init__(parent)
        tier_color = verdict.color
        self.setObjectName("qcVerdictCard")
        self.setStyleSheet(
            "QFrame#qcVerdictCard { background: #1a1d26; border: 1px solid #2c3240;"
            " border-radius: 12px; }"
            "QFrame#qcVerdictCard QLabel { background: transparent; color: #e9ecf3; }"
        )
        outer = QtWidgets.QVBoxLayout(self)
        outer.setContentsMargins(16, 14, 16, 14)
        outer.setSpacing(10)

        # ---- Header strip: the one-word verdict ----------------------------
        # QLabel derives from QFrame, so a bare "QFrame { border: ... }" rule
        # would draw a box around every label inside as well. Scope the rules by
        # object name and clear the border on the children.
        head = QtWidgets.QFrame()
        head.setObjectName("qcVerdictHead")
        head.setStyleSheet(
            f"QFrame#qcVerdictHead {{ background: {_qc_tint(tier_color, 13)};"
            f" border: 1px solid {tier_color}; border-radius: 10px; }}"
            "QFrame#qcVerdictHead QLabel { background: transparent; border: 0; }"
        )
        head_lay = QtWidgets.QHBoxLayout(head)
        head_lay.setContentsMargins(14, 10, 14, 10)
        head_lay.setSpacing(10)

        col = QtWidgets.QVBoxLayout()
        col.setContentsMargins(0, 0, 0, 0)
        col.setSpacing(0)
        verdict_lbl = QtWidgets.QLabel(verdict.tier)
        verdict_lbl.setStyleSheet(
            f"color: {tier_color}; font-size: 26pt; font-weight: 800; letter-spacing: 1px;"
        )
        col.addWidget(verdict_lbl)
        sub = QtWidgets.QLabel("Overall quality")
        sub.setStyleSheet(
            "color: #aab4c5; font-size: 8.8pt; letter-spacing: 0.5px; text-transform: uppercase;"
        )
        col.addWidget(sub)
        head_lay.addLayout(col, 1)
        outer.addWidget(head)

        # ---- Why this verdict ----------------------------------------------
        note = QtWidgets.QFrame()
        note.setObjectName("qcVerdictWhy")
        note.setStyleSheet(
            f"QFrame#qcVerdictWhy {{ background: {_qc_tint(tier_color, 9)};"
            f" border: 1px solid {tier_color}; border-radius: 8px; }}"
            "QFrame#qcVerdictWhy QLabel { background: transparent; border: 0; }"
        )
        note_lay = QtWidgets.QVBoxLayout(note)
        note_lay.setContentsMargins(10, 8, 10, 8)
        note_lay.setSpacing(4)
        note_title = QtWidgets.QLabel("Why this verdict")
        note_title.setStyleSheet(
            f"color: {tier_color}; font-size: 9.5pt; font-weight: 800;"
            " letter-spacing: 0.4px; text-transform: uppercase;"
        )
        note_lay.addWidget(note_title)
        note_why = QtWidgets.QLabel(verdict.why)
        note_why.setWordWrap(True)
        note_why.setStyleSheet("color: #eef1f7; font-size: 9.0pt;")
        note_lay.addWidget(note_why)
        note_counts = QtWidgets.QLabel(verdict.counts)
        note_counts.setWordWrap(True)
        note_counts.setStyleSheet("color: #98a3b6; font-size: 8.1pt;")
        note_lay.addWidget(note_counts)
        outer.addWidget(note)

        # ---- Per-metric rows -------------------------------------------------
        rows = [m for m in verdict.metrics if m[2] > 0.0]
        if rows:
            crit_header = QtWidgets.QLabel("The checks, one by one")
            crit_header.setStyleSheet(
                "color: #aab4c5; font-size: 8.7pt; letter-spacing: 0.5px;"
                " text-transform: uppercase; padding-top: 2px;"
            )
            outer.addWidget(crit_header)

        last_was_critical: Optional[bool] = None
        for name, sub_score, criticality, why, tier in rows:
            is_critical = criticality >= 1.0
            # Single divider between the gating checks and the advisory ones.
            if last_was_critical is True and not is_critical:
                sep_label = QtWidgets.QLabel("Advisory (does not set the grade)")
                sep_label.setStyleSheet(
                    "color: #6f7a8e; font-size: 8.0pt; letter-spacing: 0.4px;"
                    " text-transform: uppercase; padding-top: 6px;"
                )
                outer.addWidget(sep_label)
            last_was_critical = is_critical

            metric_row = QtWidgets.QFrame()
            metric_row.setStyleSheet("QFrame { background: transparent; border: 0; }")
            mlay = QtWidgets.QVBoxLayout(metric_row)
            mlay.setContentsMargins(0, 2, 0, 2)
            mlay.setSpacing(3)

            head_row = QtWidgets.QHBoxLayout()
            head_row.setContentsMargins(0, 0, 0, 0)
            head_row.setSpacing(6)

            crit_chip_html = (
                "  <span style='color:#aab4c5; font-size:7.5pt;'>critical</span>"
                if is_critical else
                "  <span style='color:#6f7a8e; font-size:7.5pt;'>advisory</span>"
            )
            name_lbl = QtWidgets.QLabel(f"{name}{crit_chip_html}")
            name_lbl.setTextFormat(QtCore.Qt.TextFormat.RichText)
            name_lbl.setStyleSheet("color: #e9ecf3; font-size: 9.2pt; font-weight: 700;")
            head_row.addWidget(name_lbl, 1)

            tier_color_chip = _QC_TIER_BADGE_COLORS.get(tier or "", "#6f7a8e")
            tier_chip = QtWidgets.QLabel(tier or "-")
            tier_chip.setStyleSheet(
                f"color: {tier_color_chip}; background: {_qc_tint(tier_color_chip, 13)};"
                f" border: 1px solid {tier_color_chip}; border-radius: 6px;"
                " padding: 1px 8px; font-weight: 800; font-size: 8.4pt;"
                " letter-spacing: 0.5px;"
            )
            head_row.addWidget(tier_chip, 0, QtCore.Qt.AlignmentFlag.AlignRight)
            mlay.addLayout(head_row)

            bar = QtWidgets.QProgressBar()
            bar.setRange(0, 100)
            bar.setValue(int(round(sub_score)))
            bar.setTextVisible(False)
            bar.setFixedHeight(5)
            bar.setStyleSheet(
                "QProgressBar { border: 0; background: #2c3240; border-radius: 2px; margin: 0; }"
                f"QProgressBar::chunk {{ background: {tier_color_chip}; border-radius: 2px; }}"
            )
            mlay.addWidget(bar)

            why_lbl = QtWidgets.QLabel(why)
            why_lbl.setWordWrap(True)
            why_lbl.setStyleSheet("color: #aab4c5; font-size: 8.4pt;")
            mlay.addWidget(why_lbl)
            outer.addWidget(metric_row)

        outer.addStretch(1)


class QcRecommendationCard(QtWidgets.QFrame):
    """Green 'what to do with this file' panel, shown under the checks.

    Kept separate from :class:`QualityVerdictCard` so that it can sit in its own
    pane at the bottom of the left column: the advice stays on screen even when
    the list of checks above it has to be scrolled.
    """

    def __init__(self, verdict: QcVerdict, parent: Optional[QtWidgets.QWidget] = None):
        super().__init__(parent)
        self.setObjectName("qcRecommendations")
        self.setStyleSheet(
            f"QFrame#qcRecommendations {{ background: {_qc_tint(_QC_REC_GREEN, 8)};"
            f" border: 1px solid {_QC_REC_GREEN}; border-radius: 10px; }}"
            "QFrame#qcRecommendations QLabel { background: transparent; border: 0; }"
        )
        lay = QtWidgets.QVBoxLayout(self)
        lay.setContentsMargins(12, 10, 12, 12)
        lay.setSpacing(6)

        title = QtWidgets.QLabel("Recommendations")
        title.setStyleSheet(
            f"color: {_QC_REC_GREEN}; font-size: 9.8pt; font-weight: 800;"
            " letter-spacing: 0.6px; text-transform: uppercase;"
        )
        lay.addWidget(title)

        # Verdict chip: what to physically do with the file.
        chip = QtWidgets.QLabel(verdict.action_kind)
        chip.setStyleSheet(
            f"color: {verdict.action_color}; background: {_qc_tint(verdict.action_color, 15)};"
            f" border: 1px solid {verdict.action_color}; border-radius: 6px;"
            " padding: 2px 10px; font-weight: 800; font-size: 8.6pt; letter-spacing: 0.6px;"
        )
        chip_row = QtWidgets.QHBoxLayout()
        chip_row.setContentsMargins(0, 0, 0, 0)
        chip_row.addWidget(chip, 0, QtCore.Qt.AlignmentFlag.AlignLeft)
        chip_row.addStretch(1)
        lay.addLayout(chip_row)

        headline = QtWidgets.QLabel(verdict.headline)
        headline.setWordWrap(True)
        headline.setStyleSheet("color: #eef1f7; font-size: 9.2pt; font-weight: 600;")
        lay.addWidget(headline)

        for text in verdict.actions:
            lay.addWidget(self._bullet_row("•", text, "#cfd7e5", lead_width=12))

        # Concrete spans to cut, with the exact seconds to type into
        # Advanced Options -> Cut out regions.
        if verdict.segments:
            shown = sorted(
                verdict.segments,
                key=lambda s: (-float(s.get("severity", 0.0)), float(s.get("start", 0.0))),
            )[:_QC_SEG_MAX_REPORTED]
            shown.sort(key=lambda s: float(s.get("start", 0.0)))

            sub = QtWidgets.QLabel("Parts of the recording to cut")
            sub.setStyleSheet(
                "color: #aab4c5; font-size: 8.3pt; letter-spacing: 0.5px;"
                " text-transform: uppercase; padding-top: 4px;"
            )
            lay.addWidget(sub)

            for seg in shown:
                span = _qc_fmt_span(float(seg["start"]), float(seg["end"]))
                lay.addWidget(self._bullet_row(
                    span,
                    f"{seg['label']} - {seg['detail']}",
                    "#cfd7e5",
                    lead_width=92,
                    lead_color=_QC_REC_GREEN,
                    lead_bold=True,
                ))

            hidden = len(verdict.segments) - len(shown)
            if hidden > 0:
                more = QtWidgets.QLabel(f"... and {hidden} shorter span(s) not listed.")
                more.setWordWrap(True)
                more.setStyleSheet("color: #8d97a9; font-size: 8.1pt;")
                lay.addWidget(more)

        lay.addStretch(1)

    @staticmethod
    def _bullet_row(
        lead: str,
        text: str,
        color: str,
        *,
        lead_width: int = 12,
        lead_color: str = "#8d97a9",
        lead_bold: bool = False,
    ) -> QtWidgets.QWidget:
        """One hanging-indent line: fixed-width lead (bullet or time span) + text."""
        row = QtWidgets.QWidget()
        row.setStyleSheet("background: transparent;")
        lay = QtWidgets.QHBoxLayout(row)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(6)
        lead_lbl = QtWidgets.QLabel(lead)
        lead_lbl.setFixedWidth(int(lead_width))
        lead_lbl.setAlignment(
            QtCore.Qt.AlignmentFlag.AlignLeft | QtCore.Qt.AlignmentFlag.AlignTop
        )
        lead_lbl.setStyleSheet(
            f"color: {lead_color}; font-size: 8.5pt;"
            f" font-weight: {'800' if lead_bold else '400'};"
        )
        lay.addWidget(lead_lbl, 0)
        body = QtWidgets.QLabel(text)
        body.setWordWrap(True)
        body.setStyleSheet(f"color: {color}; font-size: 8.6pt;")
        lay.addWidget(body, 1)
        return row


class QcDialog(QtWidgets.QDialog):
    def __init__(self, qc: Dict[str, object], parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Quality Check (strict, noise-aware)")
        self.resize(1280, 900)
        self._qc = qc
        self._navigation_delta = 0
        layout = QtWidgets.QVBoxLayout(self)

        self.plot_z = pg.PlotWidget(title="dF/F signal and reference")
        self.plot_noise = pg.PlotWidget(title="Signal and isobestic high-frequency residuals")
        self.plot_corr = pg.PlotWidget(title="Motion/reference coupling (dF/F)")
        self.plot_zdist = pg.PlotWidget(title="Corrected output distribution (z-score, secondary)")
        self.plot_roll = pg.PlotWidget(title="Rolling motion coupling (dF/F r)")
        for w in (self.plot_z, self.plot_noise, self.plot_corr, self.plot_zdist, self.plot_roll):
            w.showGrid(x=True, y=True, alpha=0.25)
            w.showAxis("top", False)
            w.showAxis("right", False)
            for axis_name in ("left", "bottom"):
                axis = w.getAxis(axis_name)
                try:
                    axis.enableAutoSIPrefix(False)
                except Exception:
                    pass
        self.plot_z.setMinimumHeight(220)
        for w in (self.plot_noise, self.plot_corr, self.plot_zdist, self.plot_roll):
            w.setMinimumHeight(155)

        has_reference = bool(qc.get("has_reference", True))
        t = np.asarray(qc["t"], float)
        dff_sig = np.asarray(qc.get("dff_sig_pct", qc["z_sig"]), float)
        dff_ref = np.asarray(qc.get("dff_ref_pct", qc["z_ref"]), float)
        dff_env = np.asarray(qc.get("dff_envelope_pct", dff_sig), float)
        noise_sigma = float(qc.get("hf_noise_pct", np.nan))
        ref_noise_sigma = float(qc.get("ref_hf_noise_pct", np.nan))
        self.plot_z.addLegend(offset=(8, 8))
        if np.isfinite(noise_sigma):
            n = min(t.size, dff_env.size)
            if n >= 2:
                center = dff_env[:n]
                self._add_filled_band(
                    self.plot_z,
                    t[:n],
                    center - noise_sigma,
                    center + noise_sigma,
                    fill_rgba=(150, 150, 150, 45),
                    line_rgba=(175, 175, 175, 80),
                    z=-8.0,
                )
        self.plot_z.plot(t, dff_sig, pen=pg.mkPen((90, 190, 255), width=1.0), name="signal")
        if has_reference:
            self.plot_z.plot(t, dff_ref, pen=pg.mkPen((240, 180, 80, 150), width=0.9), name="isobestic")
        self.plot_z.plot(t, dff_env, pen=pg.mkPen((120, 245, 210), width=1.5), name="slow envelope")
        self.plot_z.setLabel("left", "dF/F (%)")
        self.plot_z.setLabel("bottom", "Time (s)")

        hf_sig = np.asarray(qc.get("hf_sig_pct", np.zeros_like(t)), float)
        hf_ref = np.asarray(qc.get("hf_ref_pct", np.zeros_like(t)), float)
        if np.isfinite(noise_sigma):
            n = min(t.size, hf_sig.size)
            if n >= 2:
                self._add_filled_band(
                    self.plot_noise,
                    t[:n],
                    np.full(n, -noise_sigma, dtype=float),
                    np.full(n, noise_sigma, dtype=float),
                    fill_rgba=(150, 150, 150, 45),
                    line_rgba=(190, 190, 190, 90),
                    z=-8.0,
                )
        self.plot_noise.plot(t, hf_sig, pen=pg.mkPen((90, 190, 255), width=0.9), name="signal residual")
        if has_reference:
            self.plot_noise.plot(t, hf_ref, pen=pg.mkPen((240, 180, 80, 110), width=0.7), name="isobestic residual")
        if np.isfinite(noise_sigma):
            if has_reference and np.isfinite(ref_noise_sigma):
                ref_pen = pg.mkPen((240, 180, 80), width=0.7, style=QtCore.Qt.PenStyle.DotLine)
                self.plot_noise.addItem(pg.InfiniteLine(pos=ref_noise_sigma, angle=0, pen=ref_pen))
                self.plot_noise.addItem(pg.InfiniteLine(pos=-ref_noise_sigma, angle=0, pen=ref_pen))
            note = f"signal noise={noise_sigma:.3g}% dF/F"
            if has_reference and np.isfinite(ref_noise_sigma):
                note += f"  isobestic={ref_noise_sigma:.3g}%"
            elif not has_reference:
                note += "  no isobestic channel"
            self._add_plot_text_topleft(self.plot_noise, note)
        self.plot_noise.setLabel("left", "Residual (% dF/F)")
        self.plot_noise.setLabel("bottom", "Time (s)")

        # Z distribution
        Zf = qc["Zf"]
        if Zf.size:
            hist, edges = np.histogram(Zf, bins=80)
            bg = pg.BarGraphItem(x=edges[:-1], height=hist, width=np.diff(edges), brush=pg.mkBrush(90, 143, 214, 80))
            self.plot_zdist.addItem(bg)
            q25 = float(qc.get("q25", np.nan))
            q50 = float(qc.get("q50", np.nan))
            q75 = float(qc.get("q75", np.nan))
            if np.isfinite(q25) and np.isfinite(q75):
                region = pg.LinearRegionItem(values=(q25, q75), brush=(90, 143, 214, 50), movable=False)
                self.plot_zdist.addItem(region)
            if np.isfinite(q50):
                self.plot_zdist.addItem(pg.InfiniteLine(pos=q50, angle=90, pen=pg.mkPen((220, 220, 220), width=1.0)))
            iqr = float(qc.get("iqr", np.nan))
            if np.isfinite(q50) and np.isfinite(iqr):
                self._add_plot_text_topleft(self.plot_zdist, f"median={q50:.3g}  IQR={iqr:.3g}")
        self.plot_zdist.setLabel("left", "count")
        self.plot_zdist.setLabel("bottom", "Corrected output (z)")

        # Correlation scatter + fit, displayed in dF/F units so amplitude and
        # reference coupling are interpretable without z-score scaling.
        if has_reference:
            z_ref = dff_ref
            z_sig = dff_sig
            m = np.isfinite(z_ref) & np.isfinite(z_sig)
            if np.sum(m) >= 10:
                idx = np.flatnonzero(m)
                if idx.size > 12000:
                    step = int(np.ceil(idx.size / 12000))
                    idx = idx[::step]
                self.plot_corr.plot(z_ref[idx], z_sig[idx], pen=None, symbol="o", symbolSize=3, symbolBrush=(120, 180, 220, 70))
                a, b = np.polyfit(z_ref[m], z_sig[m], 1)
                xs = np.linspace(np.nanmin(z_ref[m]), np.nanmax(z_ref[m]), 200)
                self.plot_corr.plot(xs, a * xs + b, pen=pg.mkPen((220, 120, 120), width=1.2))
                r = float(qc.get("r", np.nan))
                r2 = r * r if np.isfinite(r) else np.nan
                if np.isfinite(r):
                    self._add_plot_text_topleft(
                        self.plot_corr,
                        f"r={r:.3g}  r2={r2:.3g}",
                        color=(255, 213, 95),
                        corner="topright",
                        fill=(12, 16, 24, 205),
                        border=(255, 213, 95, 150),
                    )
            self.plot_corr.setLabel("left", "Signal dF/F (%)")
            self.plot_corr.setLabel("bottom", "Isobestic dF/F (%)")
        else:
            self.plot_corr.setTitle("Motion/reference coupling unavailable")
            self.plot_corr.setLabel("left", "")
            self.plot_corr.setLabel("bottom", "")
            self.plot_corr.setXRange(0.0, 1.0)
            self.plot_corr.setYRange(0.0, 1.0)
            item = pg.TextItem("No isobestic/reference channel.\nQC uses signal-only metrics.", color=(220, 220, 220), anchor=(0.5, 0.5))
            item.setPos(0.5, 0.5)
            self.plot_corr.addItem(item)

        # Rolling corr
        if has_reference and qc["r_roll"].size:
            t_cent = qc["t"][qc["r_centers"]]
            self.plot_roll.plot(t_cent, qc["r_roll"], pen=pg.mkPen((180, 200, 120), width=1.0))
            self.plot_roll.addItem(pg.InfiniteLine(pos=0.0, angle=0, pen=pg.mkPen((200, 200, 200), width=0.8, style=QtCore.Qt.PenStyle.DashLine)))
            self.plot_roll.addItem(pg.InfiniteLine(pos=0.5, angle=0, pen=pg.mkPen((180, 180, 180), width=0.6, style=QtCore.Qt.PenStyle.DotLine)))
            self.plot_roll.addItem(pg.InfiniteLine(pos=-0.5, angle=0, pen=pg.mkPen((180, 180, 180), width=0.6, style=QtCore.Qt.PenStyle.DotLine)))
            r_avg = float(np.nanmean(qc["r_roll"])) if qc["r_roll"].size else np.nan
            if np.isfinite(r_avg):
                self._add_plot_text_topleft(self.plot_roll, f"avg r={r_avg:.3g}")
        self.plot_roll.setLabel("left", "r")
        self.plot_roll.setLabel("bottom", "Time (s)")
        self.plot_roll.setYRange(-1.0, 1.0, padding=0.04)
        if not has_reference:
            self.plot_roll.setTitle("Rolling motion coupling unavailable")

        stats = qc["stats"]
        self.lbl_stats = QtWidgets.QLabel(stats)
        self.lbl_stats.setProperty("class", "hint")
        self.lbl_stats.setWordWrap(True)
        self.lbl_method = QtWidgets.QLabel(
            "Strict grading: every check is PASS / WARN / FAIL against fixed thresholds, with no "
            "user-chosen weights. Five critical checks set the grade - artifact load (fail above 8%), "
            "motion bleed |r| (fail above 0.80), usable SNR (fail below 3x the noise floor), "
            "isobestic noise (fail above 0.60% dF/F, at which point the 405 reference adds more noise "
            "than it removes), and usable coverage (fail when more than 20% of the session has to be "
            "cut). Four advisory checks (signal noise, coupling stability, corrected-output shape, "
            "photobleach) can pull the grade down but never lift it. The grade equals the worst "
            "critical check; the green panel on the left says what to do with the file and which "
            "time spans to cut."
        )
        self.lbl_method.setProperty("class", "hint")
        self.lbl_method.setWordWrap(True)

        # Quality verdict (left column, scrollable because the recommendation
        # panel grows with the number of problems found).
        verdict = _evaluate_qc(qc)
        self._verdict = verdict
        self._qc_score = verdict.score
        self._qc_tier = verdict.tier
        self._qc_sub_metrics = verdict.metrics
        self.verdict_card = QualityVerdictCard(verdict)
        self.recommendation_card = QcRecommendationCard(verdict)

        def _scrolled(widget: QtWidgets.QWidget) -> QtWidgets.QScrollArea:
            """Wrap a card so long content scrolls instead of overflowing."""
            area = QtWidgets.QScrollArea()
            area.setWidget(widget)
            area.setWidgetResizable(True)
            area.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
            area.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
            area.setStyleSheet("QScrollArea { background: transparent; border: 0; }")
            return area

        # Left column: the graded checks on top, the green recommendation panel
        # pinned lower down so the advice is always visible without scrolling.
        self.verdict_scroll = _scrolled(self.verdict_card)
        self.recommendation_scroll = _scrolled(self.recommendation_card)
        self.verdict_column = QtWidgets.QWidget()
        column = QtWidgets.QVBoxLayout(self.verdict_column)
        column.setContentsMargins(0, 0, 0, 0)
        column.setSpacing(8)
        column.addWidget(self.verdict_scroll, 5)
        column.addWidget(self.recommendation_scroll, 4)
        self.verdict_column.setMinimumWidth(455)
        self.verdict_column.setMaximumWidth(500)

        btn_row = QtWidgets.QHBoxLayout()
        self.btn_prev_file = QtWidgets.QPushButton("Previous file")
        self.btn_next_file = QtWidgets.QPushButton("Next file")
        self.lbl_nav = QtWidgets.QLabel("")
        self.lbl_nav.setProperty("class", "hint")
        btn_row.addWidget(self.btn_prev_file)
        btn_row.addWidget(self.btn_next_file)
        btn_row.addWidget(self.lbl_nav)
        btn_row.addStretch(1)
        self.btn_save = QtWidgets.QPushButton("Save report images")
        self.btn_close = QtWidgets.QPushButton("Close")
        btn_row.addWidget(self.btn_save)
        btn_row.addWidget(self.btn_close)

        content = QtWidgets.QHBoxLayout()
        content.setSpacing(10)
        content.addWidget(self.verdict_column)

        plots_col = QtWidgets.QVBoxLayout()
        plots_col.setSpacing(8)
        plots_col.addWidget(self.lbl_method)
        plots_col.addWidget(self.plot_z, stretch=3)
        middle = QtWidgets.QHBoxLayout()
        middle.setSpacing(8)
        middle.addWidget(self.plot_noise, stretch=1)
        middle.addWidget(self.plot_corr, stretch=1)
        plots_col.addLayout(middle, stretch=2)
        bottom = QtWidgets.QHBoxLayout()
        bottom.setSpacing(8)
        bottom.addWidget(self.plot_zdist, stretch=1)
        bottom.addWidget(self.plot_roll, stretch=1)
        plots_col.addLayout(bottom, stretch=2)
        content.addLayout(plots_col, stretch=1)
        layout.addLayout(content, stretch=1)

        layout.addWidget(self.lbl_stats)
        layout.addLayout(btn_row)

        self.btn_close.clicked.connect(self.close)
        self.btn_save.clicked.connect(self._save_images)
        self.btn_prev_file.clicked.connect(lambda: self._request_navigation(-1))
        self.btn_next_file.clicked.connect(lambda: self._request_navigation(1))
        self.set_navigation_state("", False, False)

    @property
    def navigation_delta(self) -> int:
        return self._navigation_delta

    def set_navigation_state(self, label: str, can_prev: bool, can_next: bool) -> None:
        has_navigation = can_prev or can_next or bool(label)
        self.btn_prev_file.setVisible(has_navigation)
        self.btn_next_file.setVisible(has_navigation)
        self.lbl_nav.setVisible(has_navigation)
        self.btn_prev_file.setEnabled(can_prev)
        self.btn_next_file.setEnabled(can_next)
        self.lbl_nav.setText(label)

    def _request_navigation(self, delta: int) -> None:
        self._navigation_delta = int(delta)
        self.accept()

    def _add_filled_band(
        self,
        plot: pg.PlotWidget,
        x: np.ndarray,
        lower: np.ndarray,
        upper: np.ndarray,
        fill_rgba: Tuple[int, int, int, int],
        line_rgba: Tuple[int, int, int, int],
        z: float = -5.0,
    ) -> None:
        xx = np.asarray(x, float)
        lo = np.asarray(lower, float)
        hi = np.asarray(upper, float)
        n = min(xx.size, lo.size, hi.size)
        if n < 2:
            return
        xx, lo, hi = xx[:n], lo[:n], hi[:n]
        finite = np.isfinite(xx) & np.isfinite(lo) & np.isfinite(hi)
        if int(np.sum(finite)) < 2:
            return
        pen = pg.mkPen(line_rgba, width=0.6)
        upper_curve = pg.PlotCurveItem(xx[finite], hi[finite], pen=pen)
        lower_curve = pg.PlotCurveItem(xx[finite], lo[finite], pen=pen)
        band = pg.FillBetweenItem(upper_curve, lower_curve, brush=pg.mkBrush(fill_rgba))
        band.setZValue(z)
        upper_curve.setZValue(z + 0.1)
        lower_curve.setZValue(z + 0.1)
        plot.addItem(band)
        plot.addItem(upper_curve)
        plot.addItem(lower_curve)

    def _add_plot_text_topleft(
        self,
        plot: pg.PlotWidget,
        text: str,
        *,
        color: Tuple[int, int, int] = (220, 220, 220),
        corner: str = "topleft",
        fill: Optional[Tuple[int, int, int, int]] = None,
        border: Optional[Tuple[int, int, int, int]] = None,
    ) -> None:
        if not text:
            return
        vb = plot.getViewBox()
        if not vb:
            return
        (x0, x1), (y0, y1) = vb.viewRange()
        if not all(np.isfinite(v) for v in (x0, x1, y0, y1)):
            return
        pad_x = (x1 - x0) * 0.03
        pad_y = (y1 - y0) * 0.08
        corner_norm = str(corner or "topleft").strip().lower()
        if corner_norm == "topright":
            anchor = (1, 1)
            pos = (x1 - pad_x, y1 - pad_y)
        else:
            anchor = (0, 1)
            pos = (x0 + pad_x, y1 - pad_y)
        item = pg.TextItem(
            text,
            color=color,
            anchor=anchor,
            fill=pg.mkBrush(fill) if fill is not None else None,
            border=pg.mkPen(border, width=1.0) if border is not None else None,
        )
        item.setZValue(50)
        item.setPos(*pos)
        plot.addItem(item)

    def _save_images(self) -> None:
        self.save_report()

    def save_report(self, out_dir: Optional[str] = None) -> None:
        path = self._qc.get("path", "")
        channel = self._qc.get("channel", "")
        stem = os.path.splitext(os.path.basename(path))[0] if path else "quality"
        if channel:
            stem = f"{stem}_{channel}"
        out_dir = out_dir or (os.path.dirname(path) if path else os.getcwd())
        img_path = os.path.join(out_dir, f"{stem}_quality.png")
        txt_path = os.path.join(out_dir, f"{stem}_quality.txt")
        try:
            pix = self.grab()
            pix.save(img_path)
        except Exception:
            pass
        try:
            with open(txt_path, "w") as f:
                verdict = getattr(self, "_verdict", None) or _evaluate_qc(self._qc)
                f.write(f"Overall quality: {verdict.tier} "
                        f"({verdict.score:.0f}/100 representative)\n")
                f.write(f"Decision: {verdict.action_kind} - {verdict.headline}\n\n")
                f.write(f"Why: {verdict.why}\n")
                f.write(f"Tally: {verdict.counts}\n\n")

                f.write("Checks:\n")
                for entry in verdict.metrics:
                    # 5-tuple (name, score, criticality, why, tier)
                    name = entry[0]
                    sub_score = float(entry[1])
                    criticality = float(entry[2])
                    why = str(entry[3])
                    tier = str(entry[4]) if len(entry) > 4 else ""
                    kind = "critical" if criticality >= 1.0 else "advisory"
                    tier_tag = f" [{tier}]" if tier else ""
                    f.write(f"- {name} ({kind}){tier_tag}: {sub_score:.0f}/100 - {why}\n")

                f.write("\nRecommendations:\n")
                for action in verdict.actions:
                    f.write(f"- {action}\n")

                if verdict.segments:
                    f.write(
                        f"\nParts to cut ({verdict.cut_seconds:.0f} s total, "
                        f"{verdict.cut_fraction * 100:.0f}% of the session):\n"
                    )
                    for seg in verdict.segments:
                        f.write(
                            f"- {_qc_fmt_span(float(seg['start']), float(seg['end']))}: "
                            f"{seg['label']} - {seg['detail']}\n"
                        )

                f.write("\n")
                f.write(str(self._qc.get("stats", "")))
        except Exception:
            pass


class CsvChannelMappingDialog(QtWidgets.QDialog):
    def __init__(
        self,
        headers: List[str],
        numeric_headers: List[str],
        defaults: Optional[Dict[str, object]] = None,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("CSV channel mapping")
        self.setModal(True)
        self.resize(520, 260)
        self._headers = list(headers or [])
        self._numeric_headers = list(numeric_headers or [])
        self._defaults = defaults or {}

        layout = QtWidgets.QVBoxLayout(self)
        hint = QtWidgets.QLabel(
            "Choose how this CSV maps to preprocessing inputs. The same column names will be reused for CSV files in this session."
        )
        hint.setWordWrap(True)
        hint.setProperty("class", "hint")
        layout.addWidget(hint)

        form = QtWidgets.QFormLayout()
        self.combo_time = QtWidgets.QComboBox()
        self.combo_time_unit = QtWidgets.QComboBox()
        self.combo_raw1 = QtWidgets.QComboBox()
        self.combo_raw2 = QtWidgets.QComboBox()
        self.combo_ref = QtWidgets.QComboBox()
        self.combo_trigger = QtWidgets.QComboBox()

        self.combo_time.addItems(self._headers)
        self.combo_time_unit.addItems(["Auto", "Seconds", "Milliseconds"])
        self.combo_raw1.addItems(self._numeric_headers)
        self.combo_raw2.addItem(_CSV_NONE_LABEL)
        self.combo_raw2.addItems(self._numeric_headers)
        self.combo_ref.addItems(self._numeric_headers)
        self.combo_trigger.addItem(_CSV_NONE_LABEL)
        self.combo_trigger.addItems(self._numeric_headers)

        form.addRow("Time column", self.combo_time)
        form.addRow("Time unit", self.combo_time_unit)
        form.addRow("Raw signal 1", self.combo_raw1)
        form.addRow("Raw signal 2 (optional)", self.combo_raw2)
        form.addRow("Isobestic / reference", self.combo_ref)
        form.addRow("Event / DIO (optional)", self.combo_trigger)
        layout.addLayout(form)

        btn_row = QtWidgets.QHBoxLayout()
        btn_row.addStretch(1)
        self.btn_ok = QtWidgets.QPushButton("OK")
        self.btn_cancel = QtWidgets.QPushButton("Cancel")
        self.btn_ok.setDefault(True)
        btn_row.addWidget(self.btn_ok)
        btn_row.addWidget(self.btn_cancel)
        layout.addLayout(btn_row)

        self.btn_ok.clicked.connect(self._accept_if_valid)
        self.btn_cancel.clicked.connect(self.reject)
        self._apply_defaults()

    def _set_combo_text(self, combo: QtWidgets.QComboBox, value: object) -> None:
        text = str(value or "").strip()
        if not text:
            return
        idx = combo.findText(text, QtCore.Qt.MatchFlag.MatchFixedString)
        if idx >= 0:
            combo.setCurrentIndex(idx)

    def _apply_defaults(self) -> None:
        self._set_combo_text(self.combo_time, self._defaults.get("time"))
        self._set_combo_text(self.combo_time_unit, self._defaults.get("time_unit") or "Auto")
        self._set_combo_text(self.combo_raw1, self._defaults.get("raw1"))
        self._set_combo_text(self.combo_raw2, self._defaults.get("raw2") or _CSV_NONE_LABEL)
        self._set_combo_text(self.combo_ref, self._defaults.get("reference"))
        self._set_combo_text(self.combo_trigger, self._defaults.get("trigger") or _CSV_NONE_LABEL)

    def mapping(self) -> Dict[str, str]:
        raw2 = self.combo_raw2.currentText().strip()
        trigger = self.combo_trigger.currentText().strip()
        return {
            "time": self.combo_time.currentText().strip(),
            "time_unit": self.combo_time_unit.currentText().strip() or "Auto",
            "raw1": self.combo_raw1.currentText().strip(),
            "raw2": "" if raw2 == _CSV_NONE_LABEL else raw2,
            "reference": self.combo_ref.currentText().strip(),
            "trigger": "" if trigger == _CSV_NONE_LABEL else trigger,
        }

    def _accept_if_valid(self) -> None:
        m = self.mapping()
        raw1 = m.get("raw1", "")
        raw2 = m.get("raw2", "")
        ref = m.get("reference", "")
        if not m.get("time") or not raw1 or not ref:
            QtWidgets.QMessageBox.warning(self, "CSV mapping", "Choose a time column, raw signal 1, and isobestic/reference column.")
            return
        if raw1 == ref:
            QtWidgets.QMessageBox.warning(self, "CSV mapping", "Raw signal 1 and isobestic/reference must use different columns.")
            return
        if raw2 and raw2 in {raw1, ref}:
            QtWidgets.QMessageBox.warning(self, "CSV mapping", "Raw signal 2 must use a different column.")
            return
        self.accept()

class MainWindow(QtWidgets.QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Pyber - Fiber Photometry Analysis")
        _set_qt_window_icon(self)
        self.setAcceptDrops(True)
        self._set_initial_window_size()
        self.setDockOptions(
            QtWidgets.QMainWindow.DockOption.AllowNestedDocks
            | QtWidgets.QMainWindow.DockOption.AllowTabbedDocks
            | QtWidgets.QMainWindow.DockOption.AnimatedDocks
        )
        self.setDockNestingEnabled(True)

        # Core
        self.processor = PhotometryProcessor()

        # State
        self._loaded_files: Dict[str, LoadedDoricFile] = {}
        self._current_path: Optional[str] = None
        self._current_channel: Optional[str] = None
        self._current_trigger: Optional[str] = None
        self._pre_project_path: Optional[str] = None
        self._csv_channel_mapping_session: Optional[Dict[str, str]] = None
        self._csv_mappings_by_path: Dict[str, Dict[str, str]] = {}

        self._manual_regions_by_key: Dict[Tuple[str, str], List[Tuple[float, float]]] = {}
        self._manual_exclude_by_key: Dict[Tuple[str, str], List[Tuple[float, float]]] = {}
        self._auto_regions_by_key: Dict[Tuple[str, str], List[Tuple[float, float]]] = {}
        self._metadata_by_key: Dict[Tuple[str, str], Dict[str, str]] = {}
        self._cutout_regions_by_key: Dict[Tuple[str, str], List[Tuple[float, float]]] = {}
        self._sections_by_key: Dict[Tuple[str, str], List[Dict[str, object]]] = {}
        self._pending_box_region_by_key: Dict[Tuple[str, str], Tuple[float, float]] = {}

        self._last_processed: Dict[Tuple[str, str], ProcessedTrial] = {}
        self._advanced_dialog: Optional[AdvancedOptionsDialog] = None
        self._box_select_callback: Optional[Callable[[float, float], None]] = None
        self._last_artifact_params: Optional[Tuple[object, ...]] = None
        self._section_docks: Dict[str, QtWidgets.QDockWidget] = {}
        self._use_pg_dockarea_pre_layout: bool = bool(_USE_PG_DOCKAREA_PRE_LAYOUT)
        self._pre_dockarea: Optional[DockArea] = None
        self._pre_drawer_splitter: Optional[QtWidgets.QSplitter] = None
        self._pre_dockarea_docks: Dict[str, Dock] = {}
        self._pre_section_scroll_hosts: Dict[str, QtWidgets.QScrollArea] = {}
        self._pre_dockarea_fixed_layout_applied: bool = False
        self._shortcuts: List[QtGui.QShortcut] = []
        self._last_opened_section: Optional[str] = None
        self._section_popup_initialized: set[str] = set()
        self._is_restoring_panel_layout: bool = False
        # Prevent startup widget initialization from overwriting previously saved panel layout.
        self._panel_layout_persistence_ready: bool = False
        # Prevent temporary tab-switch popup visibility changes from overwriting persisted layout.
        self._suspend_panel_layout_persistence: bool = False
        self._pre_popups_hidden_by_tab_switch: bool = False
        self._pre_section_visibility_before_tab_switch: Dict[str, bool] = {}
        self._pre_section_state_before_tab_switch: Dict[str, Dict[str, object]] = {}
        self._pre_artifact_visible_before_tab_switch: bool = False
        self._pre_artifact_state_before_tab_switch: Dict[str, object] = {}
        self._pre_advanced_visible_before_tab_switch: bool = False
        self._pre_main_dock_state_before_tab_switch: Optional[QtCore.QByteArray] = None
        self._pre_tab_groups_before_tab_switch: List[Dict[str, object]] = []
        self._pre_last_interacted_dock_name: Optional[str] = None
        self._pre_snapshot_applied: bool = False
        self._pre_snapshot_retry_attempts: int = 0
        self._pre_snapshot_retry_scheduled: bool = False
        self._pre_snapshot_max_retries: int = 6
        self._post_docks_ready: bool = False
        self._handling_main_tab_change: bool = False
        self._pending_main_tab_index: Optional[int] = None
        self._force_fixed_dock_layouts: bool = bool(_FORCE_FIXED_DOCK_LAYOUTS)
        self._app_theme_mode: str = "dark"
        self._pre_history_undo: List[Dict[str, Any]] = []
        self._pre_history_redo: List[Dict[str, Any]] = []
        self._pre_history_current: Optional[Dict[str, Any]] = None
        self._pre_history_key: str = ""
        self._pre_history_restoring: bool = False
        self._pre_history_limit: int = 60
        self._export_progress_generation: int = 0

        # Worker infra (stable)
        self._pool = QtCore.QThreadPool.globalInstance()
        self._job_counter = 0
        self._latest_job_id = 0
        self._preview_preserve_view_pending: bool = False
        self._preview_preserve_view_by_job: Dict[int, bool] = {}

        # Debounce
        self._preview_timer = QtCore.QTimer(self)
        self._preview_timer.setSingleShot(True)
        self._preview_timer.setInterval(180)
        self._preview_timer.timeout.connect(self._start_preview_processing)

        # Settings (persist folder + params)
        self.settings = QtCore.QSettings("FiberPhotometryApp", "DoricProcessor")
        self._migrate_legacy_dock_state_settings()
        # Load panel layout JSON into QSettings before UI is built.
        self._load_panel_config_json_into_settings()

        self._build_ui()
        self._restore_settings()
        self._panel_layout_persistence_ready = True
        self._reset_pre_history_snapshot()
        # Enforce: preprocessing drawer is hidden until the user
        # explicitly clicks a rail section button (overrides any saved state).
        self._force_hide_pre_drawer_initially()

    # ---------------- UI ----------------

    def _set_initial_window_size(self) -> None:
        """Choose a sensible non-fullscreen default size relative to the active screen."""
        screen = QtGui.QGuiApplication.primaryScreen()
        if screen is None:
            self.resize(1280, 780)
            return
        rect = screen.availableGeometry()
        width = max(1024, min(1500, int(rect.width() * 0.86)))
        height = max(680, min(900, int(rect.height() * 0.84)))
        min_w = max(860, min(980, int(rect.width() * 0.65)))
        min_h = max(560, min(640, int(rect.height() * 0.60)))
        self.setMinimumSize(min_w, min_h)
        self.resize(width, height)

    def _build_ui(self) -> None:
        self.setStyleSheet(app_qss(self._app_theme_mode))

        self.tabs = QtWidgets.QTabWidget()
        self.setCentralWidget(self.tabs)
        self._status_bar = QtWidgets.QStatusBar(self)
        self.setStatusBar(self._status_bar)
        self.btn_app_theme = QtWidgets.QPushButton("Theme")
        self.btn_app_theme.setProperty("class", "blueSecondarySmall")
        self.btn_app_theme.setSizePolicy(QtWidgets.QSizePolicy.Policy.Minimum, QtWidgets.QSizePolicy.Policy.Fixed)
        self.btn_app_theme.setFocusPolicy(QtCore.Qt.FocusPolicy.NoFocus)
        self.menu_app_theme = QtWidgets.QMenu(self.btn_app_theme)
        self._app_theme_group = QtGui.QActionGroup(self)
        self._app_theme_group.setExclusive(True)
        self.act_app_theme_dark = self.menu_app_theme.addAction("Dark mode")
        self.act_app_theme_dark.setCheckable(True)
        self.act_app_theme_light = self.menu_app_theme.addAction("Light mode")
        self.act_app_theme_light.setCheckable(True)
        self._app_theme_group.addAction(self.act_app_theme_dark)
        self._app_theme_group.addAction(self.act_app_theme_light)
        self.act_app_theme_dark.setChecked(True)
        self.btn_app_theme.setMenu(self.menu_app_theme)

        self._export_progress_widget = QtWidgets.QFrame()
        self._export_progress_widget.setObjectName("pyberExportProgressWidget")
        export_progress_layout = QtWidgets.QHBoxLayout(self._export_progress_widget)
        export_progress_layout.setContentsMargins(8, 0, 8, 0)
        export_progress_layout.setSpacing(0)
        self._export_progress_bar = QtWidgets.QProgressBar()
        self._export_progress_bar.setObjectName("pyberExportProgressBar")
        self._export_progress_bar.setRange(0, 100)
        self._export_progress_bar.setValue(0)
        self._export_progress_bar.setTextVisible(False)
        self._export_progress_bar.setFixedSize(180, 6)
        self._export_progress_bar.setToolTip("Export progress")
        export_progress_layout.addWidget(self._export_progress_bar)
        self._export_progress_widget.setVisible(False)
        self._status_bar.addPermanentWidget(self._export_progress_widget)

        self._status_bar.addPermanentWidget(QtWidgets.QLabel("App theme"))
        self._status_bar.addPermanentWidget(self.btn_app_theme)

        # Busy / cancel indicator (left of theme widgets). Hidden until something runs.
        self._busy_widget = QtWidgets.QFrame()
        self._busy_widget.setObjectName("pyberBusyWidget")
        bl = QtWidgets.QHBoxLayout(self._busy_widget)
        bl.setContentsMargins(6, 1, 6, 1)
        bl.setSpacing(8)
        self._busy_label = QtWidgets.QLabel("Busy...")
        bl.addWidget(self._busy_label)
        self._busy_cancel = QtWidgets.QPushButton("Cancel")
        self._busy_cancel.setToolTip("Cancel the running batch operation (Esc).")
        self._busy_cancel.clicked.connect(self._cancel_current_operation)
        bl.addWidget(self._busy_cancel)
        self._busy_widget.setVisible(False)
        self._status_bar.addPermanentWidget(self._busy_widget)

        # Poll the temporal panel's progress bar to surface batch state in status bar.
        self._busy_poll = QtCore.QTimer(self)
        self._busy_poll.setInterval(250)
        self._busy_poll.timeout.connect(self._update_busy_indicator)
        self._busy_poll.start()

        # Preprocessing tab
        self.pre_tab = QtWidgets.QWidget()
        self.tabs.addTab(self.pre_tab, "Preprocessing")

        self.file_panel = FileQueuePanel(self.pre_tab)
        self.param_panel = ParameterPanel(self.pre_tab)
        self.param_panel.setVisible(False)
        self.plots = PlotDashboard(self.pre_tab)
        self.artifact_panel = ArtifactPanel(self.pre_tab)

        self.art_dock: Optional[QtWidgets.QDockWidget] = None
        # Legacy artifact list dock (kept for non-DockArea preprocessing mode).
        if not self._use_pg_dockarea_pre_layout:
            self.art_dock = QtWidgets.QDockWidget("Artifact list", self)
            self.art_dock.setObjectName("pre.artifact.dock")
            self.art_dock.setWidget(self.artifact_panel)
            self.art_dock.setAllowedAreas(QtCore.Qt.DockWidgetArea.LeftDockWidgetArea)
            self.art_dock.setVisible(False)
            self.art_dock.visibilityChanged.connect(lambda *_: self._save_panel_layout_state())
            self.art_dock.topLevelChanged.connect(lambda *_: self._save_panel_layout_state())
            self.art_dock.dockLocationChanged.connect(lambda *_: self._save_panel_layout_state())
            self.art_dock.installEventFilter(self)
            self.artifact_panel.installEventFilter(self)
            self.addDockWidget(QtCore.Qt.DockWidgetArea.LeftDockWidgetArea, self.art_dock)

        # Data browser: mounted immediately to the right of the toolbar rail.
        self.file_panel.setMinimumWidth(260)
        self.file_panel.setMaximumWidth(340)
        self.file_panel.setSizePolicy(QtWidgets.QSizePolicy.Policy.Fixed, QtWidgets.QSizePolicy.Policy.Expanding)

        # Center pane: workflow toolbar + plots
        self.btn_workflow_load = QtWidgets.QPushButton("File")
        self.btn_workflow_load.setProperty("class", "blueSecondarySmall")
        self.menu_workflow_load = QtWidgets.QMenu(self.btn_workflow_load)
        self.act_pre_new_project = self.menu_workflow_load.addAction("New Project")
        self.act_pre_open_project = self.menu_workflow_load.addAction("Open Project...")
        self.act_pre_save_project = self.menu_workflow_load.addAction("Save Project...")
        self.menu_workflow_load.addSeparator()
        self.act_open_file = self.menu_workflow_load.addAction("Open File...")
        self.act_add_folder = self.menu_workflow_load.addAction("Add Folder...")
        self.menu_workflow_load_recent = self.menu_workflow_load.addMenu("Recent Files")
        self.menu_workflow_load_recent.aboutToShow.connect(self._refresh_recent_preprocessing_menu)
        self.menu_workflow_load.addSeparator()
        self.act_focus_data = self.menu_workflow_load.addAction("Focus Data Browser")
        self.btn_workflow_load.setMenu(self.menu_workflow_load)

        self.btn_workflow_artifacts = QtWidgets.QPushButton("Detected artifacts")
        self.btn_workflow_qc = QtWidgets.QPushButton("QC")
        self.btn_workflow_export = QtWidgets.QPushButton("Export")
        self.btn_plot_style = QtWidgets.QPushButton("Plot style")
        self.btn_toggle_data = QtWidgets.QToolButton(); self.btn_toggle_data.setText("Data")
        self.btn_toggle_data.setCheckable(True)
        self.btn_toggle_data.setChecked(True)
        self.btn_toggle_data.setProperty("class", "blueSecondarySmall")
        self.btn_workflow_export.setProperty("class", "bluePrimarySmall")
        for b in (
            self.btn_toggle_data,
            self.btn_workflow_load,
            self.btn_workflow_artifacts,
            self.btn_workflow_qc,
            self.btn_workflow_export,
            self.btn_plot_style,
        ):
            b.setSizePolicy(QtWidgets.QSizePolicy.Policy.Minimum, QtWidgets.QSizePolicy.Policy.Fixed)
            b.setFocusPolicy(QtCore.Qt.FocusPolicy.NoFocus)
        for b in (self.btn_workflow_artifacts, self.btn_workflow_qc, self.btn_plot_style):
            b.setProperty("class", "blueSecondarySmall")

        self.menu_plot_style = QtWidgets.QMenu(self.btn_plot_style)
        self._plot_bg_group = QtGui.QActionGroup(self)
        self._plot_bg_group.setExclusive(True)
        self.act_plot_bg_dark = self.menu_plot_style.addAction("Dark background")
        self.act_plot_bg_dark.setCheckable(True)
        self.act_plot_bg_white = self.menu_plot_style.addAction("White background")
        self.act_plot_bg_white.setCheckable(True)
        self._plot_bg_group.addAction(self.act_plot_bg_dark)
        self._plot_bg_group.addAction(self.act_plot_bg_white)
        self.menu_plot_style.addSeparator()
        self.act_plot_grid = self.menu_plot_style.addAction("Show grid")
        self.act_plot_grid.setCheckable(True)
        self.act_plot_bg_dark.setChecked(True)
        self.act_plot_grid.setChecked(True)
        self.btn_plot_style.setMenu(self.menu_plot_style)

        # Inline parameter section buttons (same row as workflow actions).
        self.btn_section_artifacts = QtWidgets.QToolButton(); self.btn_section_artifacts.setText("Artifacts")
        self.btn_section_filtering = QtWidgets.QToolButton(); self.btn_section_filtering.setText("Filtering")
        self.btn_section_baseline = QtWidgets.QToolButton(); self.btn_section_baseline.setText("Baseline")
        self.btn_section_output = QtWidgets.QToolButton(); self.btn_section_output.setText("Output")
        self.btn_section_qc = QtWidgets.QToolButton(); self.btn_section_qc.setText("QC")
        self.btn_section_export = QtWidgets.QToolButton(); self.btn_section_export.setText("Export")
        self.btn_section_config = QtWidgets.QToolButton(); self.btn_section_config.setText("Configuration")
        self._section_buttons: Dict[str, QtWidgets.QPushButton] = {
            "artifacts": self.btn_section_artifacts,
            "filtering": self.btn_section_filtering,
            "baseline": self.btn_section_baseline,
            "output": self.btn_section_output,
            "export": self.btn_section_export,
            "qc": self.btn_section_qc,
            "config": self.btn_section_config,
        }
        for btn in self._section_buttons.values():
            btn.setCheckable(True)
            btn.setProperty("class", "blueSecondarySmall")
            btn.setSizePolicy(QtWidgets.QSizePolicy.Policy.Minimum, QtWidgets.QSizePolicy.Policy.Fixed)
            btn.setFocusPolicy(QtCore.Qt.FocusPolicy.NoFocus)

        # ----- Modern shell: vertical icon rail + thin transport bar ------
        # Configure section buttons as icon-only rail buttons.
        _rail_section_meta = {
            "artifacts":      ("Artifacts",  "Detection thresholds and artifact list", _paint_sliders),
            "filtering":      ("Filtering",  "Low-pass and smoothing options",     _paint_filter),
            "baseline":       ("Baseline",   "Baseline estimation across recording", _paint_wave),
            "output":         ("Output",     "Choose dFF / dF / z-score formula",  _paint_chart),
            "qc":             ("QC",         "Per-recording diagnostic checks",    _paint_badge),
            "export":         ("Export",     "Export processed traces",             _paint_export),
            "config":         ("Config",     "Save / load preprocessing parameter sets", _paint_gear),
        }
        self._pre_rail_icon_painters = {key: meta[2] for key, meta in _rail_section_meta.items()}
        self._pre_toggle_data_icon_painter = _paint_database
        for key, btn in self._section_buttons.items():
            label, hint, painter = _rail_section_meta[key]
            btn.setObjectName("railButton")
            btn.setProperty("class", "")
            btn.setText(label)
            btn.setToolTip(f"{label} - {hint}")
            btn.setStatusTip(f"{label} - {hint}")
            btn.setIcon(_make_icon(painter))
            btn.setIconSize(QtCore.QSize(22, 22))
            btn.setToolButtonStyle(QtCore.Qt.ToolButtonStyle.ToolButtonTextUnderIcon)
            btn.setFixedSize(76, 60)
            btn.setCheckable(True)

        # Data-browser toggle as a rail toggle button.
        self.btn_toggle_data.setObjectName("railToggleButton")
        self.btn_toggle_data.setProperty("class", "")
        self.btn_toggle_data.setText("Data")
        self.btn_toggle_data.setToolTip("Data - Show or hide data browser")
        self.btn_toggle_data.setStatusTip("Data - Show or hide data browser")
        self.btn_toggle_data.setIcon(_make_icon(_paint_database))
        self.btn_toggle_data.setIconSize(QtCore.QSize(22, 22))
        self.btn_toggle_data.setToolButtonStyle(QtCore.Qt.ToolButtonStyle.ToolButtonTextUnderIcon)
        self.btn_toggle_data.setFixedSize(76, 60)
        self.btn_toggle_data.setCheckable(True)

        side_rail = QtWidgets.QFrame()
        side_rail.setObjectName("sideRail")
        rail_layout = QtWidgets.QVBoxLayout(side_rail)
        rail_layout.setContentsMargins(8, 10, 8, 10)
        rail_layout.setSpacing(6)
        rail_layout.addWidget(self.btn_toggle_data, 0, QtCore.Qt.AlignmentFlag.AlignHCenter)
        sep = QtWidgets.QFrame()
        sep.setObjectName("railSeparator")
        sep.setFrameShape(QtWidgets.QFrame.Shape.HLine)
        rail_layout.addWidget(sep)
        for key in ("artifacts", "filtering", "baseline",
                    "output", "qc", "export", "config"):
            rail_layout.addWidget(self._section_buttons[key], 0,
                                  QtCore.Qt.AlignmentFlag.AlignHCenter)
        rail_layout.addStretch(1)
        side_rail.setFixedWidth(96)

        # Transport bar: workflow actions + status meta. Compact, single row.
        transport_bar = QtWidgets.QFrame()
        transport_bar.setObjectName("transportBar")
        transport_layout = QtWidgets.QHBoxLayout(transport_bar)
        transport_layout.setContentsMargins(12, 8, 12, 8)
        transport_layout.setSpacing(8)
        # Rename action buttons to clearer verbs to avoid confusion with rail.
        self.btn_workflow_load.setText("File")
        self.btn_workflow_qc.setText("Run QC")
        self.btn_workflow_export.setText("Run Export")
        transport_layout.addWidget(self.btn_workflow_load)
        transport_layout.addWidget(self.btn_workflow_qc)
        transport_layout.addWidget(self.btn_workflow_export)
        transport_layout.addSpacing(8)
        transport_layout.addWidget(self.btn_plot_style)
        transport_layout.addStretch(1)
        # Redundant duplicate: 'Detected artifacts' workflow button is covered
        # by the artifact-list rail button. Hide from layout but keep instance
        # so existing wiring (signals, references) remains intact.
        self.btn_workflow_artifacts.setVisible(False)

        center_panel = QtWidgets.QFrame()
        center_panel.setObjectName("centerPanel")
        center_panel_layout = QtWidgets.QVBoxLayout(center_panel)
        center_panel_layout.setContentsMargins(10, 10, 10, 10)
        center_panel_layout.setSpacing(8)
        center_panel_layout.addWidget(transport_bar)
        center_panel_layout.addWidget(self.plots, stretch=1)

        center_widget = QtWidgets.QWidget()
        center_h = QtWidgets.QHBoxLayout(center_widget)
        center_h.setContentsMargins(0, 0, 0, 0)
        center_h.setSpacing(8)
        center_h.addWidget(side_rail)
        if self._use_pg_dockarea_pre_layout:
            self._pre_drawer_splitter = QtWidgets.QSplitter(QtCore.Qt.Orientation.Horizontal)
            self._pre_drawer_splitter.setChildrenCollapsible(False)
            self._pre_dockarea = DockArea()
            # Wrap the DockArea in a rounded drawer frame so it matches the
            # modern shell, and start hidden. It sits beside the left rail so
            # tool panels open on the same side as the data browser.
            self._pre_drawer = QtWidgets.QFrame()
            self._pre_drawer.setObjectName("drawerPanel")
            _drawer_l = QtWidgets.QVBoxLayout(self._pre_drawer)
            _drawer_l.setContentsMargins(12, 10, 12, 10)
            _drawer_l.setSpacing(8)
            # Rich panel header (badge + title + subtitle); set per active section.
            self._pre_drawer_header = PanelHeader()
            _drawer_l.addWidget(self._pre_drawer_header)
            # Hidden compat label so legacy lookups don't crash.
            self._pre_drawer_title = QtWidgets.QLabel("")
            self._pre_drawer_title.setVisible(False)
            _drawer_l.addWidget(self._pre_dockarea, stretch=1)
            self._pre_drawer.setVisible(False)
            self._pre_drawer_splitter.addWidget(self._pre_drawer)
            self._pre_drawer_splitter.addWidget(center_panel)
            self._pre_drawer_splitter.setStretchFactor(0, 0)
            self._pre_drawer_splitter.setStretchFactor(1, 1)
            self._pre_drawer_splitter.setSizes([0, 1400])
            content_widget = self._pre_drawer_splitter
        else:
            content_widget = center_panel

        # Main splitter: data browser + visuals, both to the right of the toolbar rail.
        self.pre_splitter = QtWidgets.QSplitter(QtCore.Qt.Orientation.Horizontal)
        self.pre_splitter.setObjectName("preprocessing_splitter")
        self.pre_splitter.addWidget(self.file_panel)
        self.pre_splitter.addWidget(content_widget)
        self.pre_splitter.setChildrenCollapsible(False)
        self.pre_splitter.setStretchFactor(0, 0)
        self.pre_splitter.setStretchFactor(1, 1)
        self.pre_splitter.setSizes([350, 1350])
        self.pre_splitter.splitterMoved.connect(self._save_splitter_sizes)
        center_h.addWidget(self.pre_splitter, stretch=1)

        pre_layout = QtWidgets.QVBoxLayout(self.pre_tab)
        pre_layout.setContentsMargins(10, 10, 10, 10)
        pre_layout.addWidget(center_widget)

        # Postprocessing tab
        self.post_tab = PostProcessingPanel()
        if hasattr(self.post_tab, "set_app_theme_mode"):
            try:
                self.post_tab.set_app_theme_mode(self._app_theme_mode)
            except Exception:
                pass
        if hasattr(self.post_tab, "set_force_fixed_default_layout"):
            try:
                self.post_tab.set_force_fixed_default_layout(self._force_fixed_dock_layouts)
            except Exception:
                pass
        self.tabs.addTab(self.post_tab, "Postprocessing")
        self.post_tab.statusUpdate.connect(self._show_status_message)
        if hasattr(self.post_tab, "exportProgress"):
            try:
                self.post_tab.exportProgress.connect(self._on_post_export_progress)
            except Exception:
                pass
        if hasattr(self.post_tab, "helpRequested"):
            try:
                self.post_tab.helpRequested.connect(lambda: self._show_tutorial_again(automatic=False))
            except Exception:
                pass

        # Wiring - file panel
        self.file_panel.openFileRequested.connect(self._open_files_dialog)
        self.file_panel.openFolderRequested.connect(self._open_folder_dialog)
        self.file_panel.selectionChanged.connect(self._on_file_selection_changed)
        self.file_panel.channelChanged.connect(self._on_channel_changed)
        self.file_panel.triggerChanged.connect(self._on_trigger_changed)
        self.file_panel.timeWindowChanged.connect(self._on_time_window_changed)

        self.file_panel.updatePreviewRequested.connect(self._trigger_preview)
        self.file_panel.metadataRequested.connect(self._edit_metadata_for_current)
        self.file_panel.exportRequested.connect(self._export_selected_or_all)
        self.file_panel.toggleArtifactsRequested.connect(self._toggle_artifacts_panel)
        self.file_panel.advancedOptionsRequested.connect(self._open_advanced_options)
        self.file_panel.qcRequested.connect(self._run_qc_dialog)
        self.file_panel.batchQcRequested.connect(self._run_batch_qc)
        self.file_panel.sendToPostprocessingRequested.connect(self._send_preprocessing_paths_to_postprocessing)

        # Parameters: changes and actions
        self.param_panel.paramsChanged.connect(self._on_params_changed)
        self.param_panel.paramsChanged.connect(self._update_export_summary_label)
        self.param_panel.previewRequested.connect(self._trigger_preview)
        self.param_panel.metadataRequested.connect(self._edit_metadata_for_current)
        self.param_panel.exportRequested.connect(self._export_selected_or_all)
        self.param_panel.artifactsRequested.connect(self._toggle_artifacts_panel)
        self.param_panel.artifactOverlayToggled.connect(self._on_artifact_overlay_toggled)
        self.param_panel.advancedOptionsRequested.connect(self._open_advanced_options)
        self.param_panel.qcRequested.connect(self._run_qc_dialog)
        self.param_panel.batchQcRequested.connect(self._run_batch_qc)
        self.param_panel.set_config_state_hooks(
            self._export_preprocessing_ui_state_for_config,
            self._import_preprocessing_ui_state_from_config,
        )

        # Workflow toolbar
        self.act_pre_new_project.triggered.connect(self._new_preprocessing_project)
        self.act_pre_open_project.triggered.connect(self._open_preprocessing_project_file)
        self.act_pre_save_project.triggered.connect(self._save_preprocessing_project_file)
        self.act_open_file.triggered.connect(self._open_files_dialog)
        self.act_add_folder.triggered.connect(self._open_folder_dialog)
        self.act_focus_data.triggered.connect(self._focus_data_browser)
        self.act_plot_bg_dark.triggered.connect(self._on_pre_plot_style_changed)
        self.act_plot_bg_white.triggered.connect(self._on_pre_plot_style_changed)
        self.act_plot_grid.toggled.connect(self._on_pre_plot_style_changed)
        self.act_app_theme_dark.triggered.connect(lambda _checked=False: self._apply_app_theme("dark", persist=True))
        self.act_app_theme_light.triggered.connect(lambda _checked=False: self._apply_app_theme("light", persist=True))
        self.btn_toggle_data.toggled.connect(self._set_data_panel_visible)
        self.btn_workflow_artifacts.clicked.connect(self._toggle_artifacts_panel)
        self.btn_workflow_qc.clicked.connect(self._run_qc_dialog)
        self.btn_workflow_export.clicked.connect(self._export_selected_or_all)

        # Section popup controls
        self._setup_section_popups()
        for key, btn in self._section_buttons.items():
            btn.toggled.connect(lambda checked, section_key=key: self._toggle_section_popup(section_key, checked))

        # Plot sync — range propagation is handled inside the plots widget
        # (sigXRangeChanged → _emit_xrange_from_any). The self-connection here
        # would double-call set_xrange_all and corrupt PanMode drag state.

        # Manual artifacts
        self.plots.manualRegionFromSelectorRequested.connect(self._add_manual_region_from_selector)
        self.plots.manualRegionFromDragRequested.connect(self._add_manual_region_from_drag)
        self.plots.clearManualRegionsRequested.connect(self._clear_manual_regions_current)
        self.plots.undoRequested.connect(self._undo_pre_action)
        self.plots.redoRequested.connect(self._redo_pre_action)
        self.plots.showArtifactsRequested.connect(self._toggle_artifacts_panel)
        self.plots.boxSelectionCleared.connect(self._cancel_box_select_request)
        self.plots.boxSelectionContextRequested.connect(self._show_box_selection_context_menu)
        self.plots.artifactThresholdsToggled.connect(self._on_artifact_thresholds_toggled)

        self.artifact_panel.regionsChanged.connect(self._artifact_regions_changed)
        self.artifact_panel.selectionChanged.connect(self.plots.highlight_artifact_regions)

        # Postprocessing needs access to "current processed"
        self.post_tab.requestCurrentProcessed.connect(self._post_get_current_processed)
        self.post_tab.requestDioList.connect(self._post_get_current_dio_list)
        self.post_tab.requestDioData.connect(self._post_get_dio_data_for_path)
        self.tabs.currentChanged.connect(self._on_main_tab_changed)

        self._init_shortcuts()
        self.plots.set_artifact_overlay_visible(self.param_panel.artifact_overlay_visible())
        self.plots.set_artifact_thresholds_visible(True)
        self._update_plot_status()
        self.setAcceptDrops(True)

        # ----- UX polish: toasts, dirty-title, shortcuts, tutorial -----
        try:
            self._toaster = ToastManager(self, max_visible=4)
        except Exception:
            self._toaster = None

        # Mirror status-bar messages to toasts (longer-lived, easier to spot).
        try:
            self.post_tab.statusUpdate.connect(self._toast_from_status)
        except Exception:
            pass

        # Dirty-title indicator: '*' suffix while postprocessing has unsaved changes.
        def _is_dirty() -> bool:
            try:
                checker = getattr(self.post_tab, "is_project_dirty", None)
                if callable(checker):
                    return bool(checker())
                return bool(getattr(self.post_tab, "_project_dirty", False))
            except Exception:
                return False

        self._refresh_dirty_title = attach_dirty_title(
            self, "Pyber - Fiber Photometry", _is_dirty,
        )
        self._dirty_poll = QtCore.QTimer(self)
        self._dirty_poll.setInterval(800)
        self._dirty_poll.timeout.connect(self._refresh_dirty_title)
        self._dirty_poll.start()

        install_close_confirmation(
            self,
            _is_dirty,
            save_callback=self._save_post_project_for_close,
            discard_callback=self._discard_post_project_for_close,
        )

        # Register the global shortcut bundle. Methods that don't exist become no-ops.
        register_global_shortcuts(self)

        # First-run tutorial.
        QtCore.QTimer.singleShot(450, self._maybe_show_first_run_tutorial)

    def _setup_section_popups(self) -> None:
        """Create preprocessing section panels using DockArea or legacy floating docks."""
        if self._use_pg_dockarea_pre_layout and self._pre_dockarea_docks:
            return
        if (not self._use_pg_dockarea_pre_layout) and self._section_docks:
            return

        # Move section cards out of the hidden ParameterPanel container and into docks.
        root_layout = self.param_panel.layout()
        section_cards = [
            self.param_panel.card_artifacts,
            self.param_panel.card_filtering,
            self.param_panel.card_baseline,
            self.param_panel.card_output,
            self.param_panel.card_actions,
        ]
        if root_layout is not None:
            for w in section_cards:
                root_layout.removeWidget(w)
                w.setParent(None)
        self.param_panel.card_actions.setVisible(False)

        section_widgets: Dict[str, QtWidgets.QWidget] = {
            "artifacts": self._build_artifacts_section_widget(),
            "filtering": self.param_panel.card_filtering,
            "baseline": self.param_panel.card_baseline,
            "output": self.param_panel.card_output,
            "export": self._build_export_actions_widget(),
            "qc": self._build_qc_actions_widget(),
            "config": self._build_config_actions_widget(),
        }
        section_titles: Dict[str, str] = {
            "artifacts": "Artifacts",
            "filtering": "Filtering",
            "baseline": "Baseline",
            "output": "Output",
            "export": "Export",
            "qc": "QC",
            "config": "Configuration",
        }

        if self._use_pg_dockarea_pre_layout:
            if self._pre_dockarea is None:
                return
            for key, title in section_titles.items():
                widget = section_widgets[key]
                scroll = QtWidgets.QScrollArea()
                scroll.setWidgetResizable(True)
                scroll.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
                scroll.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
                scroll.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAsNeeded)
                widget.setMinimumSize(0, 0)
                widget.setSizePolicy(QtWidgets.QSizePolicy.Policy.Preferred, QtWidgets.QSizePolicy.Policy.Preferred)
                scroll.setWidget(widget)
                self._pre_section_scroll_hosts[key] = scroll

                dock = Dock(title, area=self._pre_dockarea, closable=False)
                dock.setObjectName(f"pre.da.{key}.dock")
                dock.addWidget(scroll)
                # Collapse the per-dock label/tab to 0px without deleting it
                # (pyqtgraph still references dock.label when restacking).
                try:
                    dock.label.setMaximumHeight(0)
                    dock.label.setMinimumHeight(0)
                    dock.label.setFixedHeight(0)
                    dock.label.setVisible(False)
                except Exception:
                    pass
                self._lock_pre_pg_dock_interactions(dock)
                try:
                    dock.sigClosed.connect(lambda *_, section_key=key: self._on_pre_dockarea_dock_closed(section_key))
                except Exception:
                    pass
                self._pre_dockarea_docks[key] = dock

            self._restore_pre_dockarea_layout_state()
            self._pre_dockarea_fixed_layout_applied = False
            return

        for key, title in section_titles.items():
            widget = section_widgets[key]
            dock = QtWidgets.QDockWidget(title, self)
            dock.setObjectName(f"pre.{key}.dock")
            dock.setAllowedAreas(QtCore.Qt.DockWidgetArea.AllDockWidgetAreas)
            dock.setFeatures(
                QtWidgets.QDockWidget.DockWidgetFeature.DockWidgetClosable
                | QtWidgets.QDockWidget.DockWidgetFeature.DockWidgetMovable
                | QtWidgets.QDockWidget.DockWidgetFeature.DockWidgetFloatable
            )
            dock.setWidget(widget)
            dock.visibilityChanged.connect(lambda visible, section_key=key: self._on_section_dock_visibility(section_key, visible))
            dock.topLevelChanged.connect(lambda *_: self._save_panel_layout_state())
            dock.dockLocationChanged.connect(lambda *_: self._save_panel_layout_state())
            # Register with main window once; each popup opens floating by default.
            self.addDockWidget(QtCore.Qt.DockWidgetArea.LeftDockWidgetArea, dock)
            dock.setFloating(True)
            dock.hide()
            dock.installEventFilter(self)
            widget.installEventFilter(self)
            self._section_docks[key] = dock

    def _build_artifacts_section_widget(self) -> QtWidgets.QWidget:
        panel = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(panel)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(10)

        self.param_panel.card_artifacts.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Expanding,
            QtWidgets.QSizePolicy.Policy.Fixed,
        )
        layout.addWidget(self.param_panel.card_artifacts)

        try:
            self.artifact_panel.btn_close.setVisible(False)
        except Exception:
            pass
        table_min_heights = {
            "table_auto": 260,
            "table": 260,
        }
        for table_name, min_height in table_min_heights.items():
            try:
                table = getattr(self.artifact_panel, table_name)
                table.setMinimumHeight(min_height)
                table.setSizePolicy(
                    QtWidgets.QSizePolicy.Policy.Expanding,
                    QtWidgets.QSizePolicy.Policy.Expanding,
                )
            except Exception:
                pass
        self.artifact_panel.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Expanding,
            QtWidgets.QSizePolicy.Policy.Expanding,
        )
        self.artifact_panel.show()
        layout.addWidget(self.artifact_panel, 1)

        return panel

    def _pre_dockarea_dock(self, key: str) -> Optional[Dock]:
        return self._pre_dockarea_docks.get(key)

    def _pre_dockarea_ordered_keys(self) -> List[str]:
        ordered = list(_PRE_DOCKAREA_PRIMARY_ORDER) + list(_PRE_DOCKAREA_OPTIONAL_ORDER)
        return [key for key in ordered if self._pre_dockarea_dock(key) is not None]

    def _pre_dockarea_default_visible_map(self) -> Dict[str, bool]:
        return {key: (key in _PRE_DOCKAREA_DEFAULT_VISIBLE) for key in self._pre_dockarea_docks.keys()}

    def _lock_pre_pg_dock_interactions(self, dock: Dock) -> None:
        label = getattr(dock, "label", None)
        if label is None:
            return
        if not self._force_fixed_dock_layouts:
            self._style_pg_dock_label_buttons(dock, label)
            return
        if bool(getattr(label, "_pyber_fixed_interaction_lock", False)):
            return

        def _ignore_drag(event: QtGui.QMouseEvent) -> None:
            event.ignore()

        def _ignore_double_click(event: QtGui.QMouseEvent) -> None:
            event.accept()

        try:
            label.mouseMoveEvent = _ignore_drag
            label.mouseDoubleClickEvent = _ignore_double_click
            label.setCursor(QtCore.Qt.CursorShape.ArrowCursor)
            label._pyber_fixed_interaction_lock = True
        except Exception:
            pass
        self._style_pg_dock_label_buttons(dock, label)

    def _style_pg_dock_label_buttons(self, dock: Dock, label: object) -> None:
        if label is None:
            return
        try:
            buttons = label.findChildren(QtWidgets.QToolButton)
        except Exception:
            buttons = []
        for btn in buttons:
            try:
                btn.setText("x")
                btn.setIcon(QtGui.QIcon())
                btn.setAutoRaise(True)
                btn.setFixedSize(13, 13)
                btn.setToolTip("Close")
                if not bool(btn.property("_pyber_hide_wired")):
                    try:
                        btn.clicked.disconnect()
                    except Exception:
                        pass
                    btn.clicked.connect(lambda _checked=False, section_dock=dock: self._hide_pre_dockarea_dock(section_dock))
                    btn.setProperty("_pyber_hide_wired", True)
                light = str(getattr(self, "_app_theme_mode", "dark")).lower() == "light"
                fg = "#4a5568" if light else "#f3f5f8"
                fg_hover = "#172033" if light else "#ffffff"
                btn.setStyleSheet(
                    "QToolButton {"
                    " background: transparent;"
                    f" color: {fg};"
                    " border: none;"
                    " padding: 0px;"
                    " margin: 0px;"
                    " font-size: 8pt;"
                    " font-weight: 700;"
                    " }"
                    "QToolButton:hover {"
                    " background: transparent;"
                    f" color: {fg_hover};"
                    " border: none;"
                    " }"
                )
            except Exception:
                continue

    def _hide_pre_dockarea_dock(self, dock: Dock) -> None:
        if dock is None:
            return
        try:
            dock.hide()
        except Exception:
            return
        for key, candidate in self._pre_dockarea_docks.items():
            if candidate is dock:
                self._set_section_button_checked(key, False)
                if self._last_opened_section == key:
                    self._last_opened_section = None
                break
        self._update_pre_drawer_visibility()
        self._save_panel_layout_state()

    def _arrange_pre_dockarea_default(self) -> None:
        if self._pre_dockarea is None:
            return
        ordered = self._pre_dockarea_ordered_keys()
        root = self._pre_dockarea_dock("artifacts")
        if root is None and ordered:
            root = self._pre_dockarea_dock(ordered[0])
        if root is None:
            return
        self._pre_dockarea.addDock(root, "left")
        for key in ordered:
            dock = self._pre_dockarea_dock(key)
            if dock is not None and dock is not root:
                self._pre_dockarea.addDock(dock, "above", root)

    def _pre_dockarea_active_key(self) -> Optional[str]:
        active = self._last_opened_section
        if isinstance(active, str) and active in self._pre_dockarea_docks:
            return active
        for key in self._pre_dockarea_ordered_keys():
            dock = self._pre_dockarea_dock(key)
            if dock is not None and dock.isVisible():
                return key
        return None

    def _set_pre_dockarea_visible(self, key: str, visible: bool) -> None:
        dock = self._pre_dockarea_dock(key)
        if dock is None:
            return
        if visible:
            if key == "artifacts":
                self.artifact_panel.show()
            self._arrange_pre_dockarea_default()
            dock.show()
            try:
                dock.raiseDock()
            except Exception:
                pass
        else:
            dock.hide()

    def _save_pre_dockarea_layout_state(self) -> None:
        if self._pre_dockarea is None:
            return
        try:
            state = dict(self._pre_dockarea.saveState() or {})
        except Exception:
            state = {}
        visible = {key: bool(dock.isVisible()) for key, dock in self._pre_dockarea_docks.items()}
        active = self._pre_dockarea_active_key() or ""
        try:
            self.settings.setValue(_PRE_DOCKAREA_STATE_KEY, json.dumps(state))
            self.settings.setValue(_PRE_DOCKAREA_VISIBLE_KEY, json.dumps(visible))
            self.settings.setValue(_PRE_DOCKAREA_ACTIVE_KEY, active)
            self.settings.remove(_PRE_DOCK_STATE_KEY)
            self.settings.remove(_PRE_TAB_GROUPS_KEY)
        except Exception:
            pass

        left_i = _dock_area_to_int(QtCore.Qt.DockWidgetArea.LeftDockWidgetArea, 1)
        for key, dock in self._pre_dockarea_docks.items():
            try:
                base = f"pre_section_docks/{key}"
                self.settings.setValue(f"{base}/visible", bool(dock.isVisible()))
                self.settings.setValue(f"{base}/floating", False)
                self.settings.setValue(f"{base}/area", left_i)
            except Exception:
                continue
        try:
            art_base = "pre_artifact_dock_state"
            art_vis = bool(visible.get("artifacts", False))
            self.settings.setValue(f"{art_base}/visible", art_vis)
            self.settings.setValue(f"{art_base}/floating", False)
            self.settings.setValue(f"{art_base}/area", left_i)
        except Exception:
            pass

    def _restore_pre_dockarea_layout_state(self) -> None:
        if self._pre_dockarea is None or not self._pre_dockarea_docks:
            return
        self._pre_dockarea_fixed_layout_applied = False
        self._arrange_pre_dockarea_default()

        # The left rail drawer behaves as a single-section stack. Restoring old
        # DockArea splitter topology can strand the active dock in a zero-height slot.
        visible_map: Dict[str, bool] = {}
        raw_vis = self.settings.value(_PRE_DOCKAREA_VISIBLE_KEY, "")
        try:
            if isinstance(raw_vis, str) and raw_vis.strip():
                parsed = json.loads(raw_vis)
                if isinstance(parsed, dict):
                    visible_map = {str(k): bool(v) for k, v in parsed.items()}
        except Exception:
            visible_map = {}

        legacy_artifacts_visible = visible_map.pop("artifacts_list", None)
        if legacy_artifacts_visible is not None and "artifacts" in self._pre_dockarea_docks:
            visible_map["artifacts"] = bool(visible_map.get("artifacts", False) or legacy_artifacts_visible)

        if not visible_map:
            for key in self._pre_dockarea_docks.keys():
                raw = self.settings.value(f"pre_section_docks/{key}/visible", None)
                if raw is not None:
                    visible_map[key] = _to_bool(raw, False)
                if key == "artifacts":
                    legacy_raw = self.settings.value("pre_artifact_dock_state/visible", None)
                    if legacy_raw is not None:
                        visible_map[key] = bool(visible_map.get(key, False) or _to_bool(legacy_raw, False))
        if not visible_map:
            visible_map = self._pre_dockarea_default_visible_map()

        active = str(self.settings.value(_PRE_DOCKAREA_ACTIVE_KEY, "artifacts") or "artifacts")
        if active == "artifacts_list":
            active = "artifacts"
        if not bool(visible_map.get(active, False)):
            active = next((key for key in self._pre_dockarea_ordered_keys() if bool(visible_map.get(key, False))), "")

        for key in self._pre_dockarea_docks.keys():
            self._set_pre_dockarea_visible(key, bool(active and key == active))

        active_dock = self._pre_dockarea_dock(active)
        if active_dock is not None and active_dock.isVisible():
            try:
                active_dock.raiseDock()
            except Exception:
                pass
            if active in self._section_buttons:
                self._last_opened_section = active
        else:
            self._last_opened_section = None
            for key in self._pre_dockarea_ordered_keys():
                dock = self._pre_dockarea_dock(key)
                if dock is not None and dock.isVisible():
                    try:
                        dock.raiseDock()
                    except Exception:
                        pass
                    self._last_opened_section = key
                    break

        self._sync_section_button_states_from_docks()
        self._update_pre_drawer_visibility()

    def _apply_pre_fixed_dockarea_layout(self) -> None:
        if self._pre_dockarea is None or not self._pre_dockarea_docks:
            return
        visible_map = {key: bool(dock.isVisible()) for key, dock in self._pre_dockarea_docks.items()}
        if not any(visible_map.values()):
            visible_map = self._pre_dockarea_default_visible_map()
        self._arrange_pre_dockarea_default()
        for key in self._pre_dockarea_docks.keys():
            self._set_pre_dockarea_visible(key, bool(visible_map.get(key, False)))

        active = self._last_opened_section if bool(visible_map.get(self._last_opened_section or "", False)) else None
        if active is None:
            for key in self._pre_dockarea_ordered_keys():
                if bool(visible_map.get(key, False)):
                    active = key
                    break
        dock = self._pre_dockarea_dock(active) if active else None
        if dock is not None and dock.isVisible():
            try:
                dock.raiseDock()
            except Exception:
                pass
        self._last_opened_section = active
        self._sync_section_button_states_from_docks()
        self._update_pre_drawer_visibility()
        self._save_pre_dockarea_layout_state()
        self._pre_dockarea_fixed_layout_applied = True

    def _on_pre_dockarea_dock_closed(self, key: str) -> None:
        if key in self._section_buttons:
            self._set_section_button_checked(key, False)
            if self._last_opened_section == key:
                self._last_opened_section = None
        self._save_panel_layout_state()

    def _build_qc_actions_widget(self) -> QtWidgets.QWidget:
        panel = QtWidgets.QWidget()
        v = QtWidgets.QVBoxLayout(panel)
        v.setContentsMargins(8, 8, 8, 8)
        v.setSpacing(6)
        self.param_panel.btn_qc.setProperty("class", "blueSecondarySmall")
        self.param_panel.btn_qc_batch.setProperty("class", "blueSecondarySmall")
        v.addWidget(self.param_panel.btn_qc)
        v.addWidget(self.param_panel.btn_qc_batch)
        v.addWidget(self.param_panel.lbl_fs)
        v.addStretch(1)
        return panel

    def _build_export_actions_widget(self) -> QtWidgets.QWidget:
        panel = QtWidgets.QWidget()
        v = QtWidgets.QVBoxLayout(panel)
        v.setContentsMargins(8, 8, 8, 8)
        v.setSpacing(6)
        self.lbl_export_summary = QtWidgets.QLabel("")
        self.lbl_export_summary.setWordWrap(True)
        self.lbl_export_summary.setProperty("class", "hint")
        v.addWidget(self.lbl_export_summary)
        self.param_panel.btn_export.setProperty("class", "bluePrimarySmall")
        v.addWidget(self.param_panel.btn_export)
        v.addStretch(1)
        self._update_export_summary_label()
        return panel

    def _build_config_actions_widget(self) -> QtWidgets.QWidget:
        panel = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(panel)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)
        row = QtWidgets.QHBoxLayout()
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(6)
        self.param_panel.btn_metadata.setProperty("class", "blueSecondarySmall")
        self.param_panel.btn_save_config.setProperty("class", "blueSecondarySmall")
        self.param_panel.btn_load_config.setProperty("class", "blueSecondarySmall")
        self.param_panel.btn_reset_defaults.setProperty("class", "blueSecondarySmall")
        for btn in (
            self.param_panel.btn_metadata,
            self.param_panel.btn_save_config,
            self.param_panel.btn_load_config,
            self.param_panel.btn_reset_defaults,
        ):
            btn.setSizePolicy(QtWidgets.QSizePolicy.Policy.Ignored, QtWidgets.QSizePolicy.Policy.Fixed)
            btn.setMinimumWidth(90)
            btn.setFocusPolicy(QtCore.Qt.FocusPolicy.NoFocus)
            row.addWidget(btn)
        layout.addLayout(row)
        if hasattr(self.param_panel, "export_options_group"):
            self.param_panel.export_options_group.setSizePolicy(
                QtWidgets.QSizePolicy.Policy.Expanding,
                QtWidgets.QSizePolicy.Policy.Fixed,
            )
            layout.addWidget(self.param_panel.export_options_group)
        layout.addStretch(1)
        return panel

    def _update_export_summary_label(self) -> None:
        if hasattr(self, "lbl_export_summary") and self.lbl_export_summary is not None:
            self.lbl_export_summary.setText(self.param_panel.export_selection_summary())

    def _set_section_button_checked(self, key: str, checked: bool) -> None:
        btn = self._section_buttons.get(key)
        if btn is None:
            return
        btn.blockSignals(True)
        btn.setChecked(bool(checked))
        btn.blockSignals(False)

    def _force_hide_pre_drawer_initially(self) -> None:
        """Hide every preprocessing section dock and the left drawer at startup."""
        for key, btn in self._section_buttons.items():
            if btn.isChecked():
                blocked = btn.blockSignals(True)
                try:
                    btn.setChecked(False)
                finally:
                    btn.blockSignals(blocked)
            try:
                dock = self._pre_dockarea_dock(key) if self._use_pg_dockarea_pre_layout else self._section_docks.get(key)
                if dock is not None:
                    dock.hide()
            except Exception:
                pass
        drawer = getattr(self, "_pre_drawer", None)
        if drawer is not None:
            drawer.setVisible(False)
        splitter = self._pre_drawer_splitter
        if splitter is not None:
            try:
                sizes = splitter.sizes()
                if len(sizes) >= 2:
                    sizes[1] += sizes[0]
                    sizes[0] = 0
                    splitter.setSizes(sizes)
            except Exception:
                pass

    _PRE_SECTION_TITLES = {
        "artifacts": "Artifacts",
        "filtering": "Filtering",
        "baseline": "Baseline",
        "output": "Output",
        "qc": "Quality control",
        "export": "Export",
        "config": "Configuration",
    }

    def _update_pre_drawer_visibility(self) -> None:
        """Show the left preprocessing drawer iff at least one section is active."""
        drawer = getattr(self, "_pre_drawer", None)
        if drawer is None:
            return
        any_checked = any(btn.isChecked() for btn in self._section_buttons.values())
        active_key = next((k for k, b in self._section_buttons.items() if b.isChecked()), None)
        # Compat title (kept hidden, used by tests / legacy code).
        title_lbl = getattr(self, "_pre_drawer_title", None)
        if title_lbl is not None:
            title_lbl.setText(self._PRE_SECTION_TITLES.get(active_key or "", ""))
        # Rich header (badge + title + subtitle).
        header = getattr(self, "_pre_drawer_header", None)
        if header is not None:
            try:
                header.set_preprocess_section(active_key or "")
            except Exception:
                pass
        drawer.setVisible(any_checked)
        splitter = self._pre_drawer_splitter
        if splitter is None:
            return
        try:
            sizes = splitter.sizes()
            if len(sizes) >= 2:
                if any_checked:
                    total = sum(sizes) or 1
                    if active_key == "artifacts":
                        target_w = max(520, int(total * 0.34))
                    else:
                        target_w = max(420, int(total * 0.28))
                    drawer_w = min(target_w, max(420, total - 640))
                    if sizes[0] < 60 or (active_key == "artifacts" and sizes[0] < drawer_w - 20):
                        delta_w = max(0, drawer_w - max(0, sizes[0]))
                        sizes[0] = drawer_w
                        sizes[1] = max(400, sizes[1] - delta_w)
                        splitter.setSizes(sizes)
                else:
                    if sizes[0] > 0:
                        sizes[1] += sizes[0]
                        sizes[0] = 0
                        splitter.setSizes(sizes)
        except Exception:
            pass

    def _toggle_section_popup(self, key: str, checked: bool) -> None:
        if self._use_pg_dockarea_pre_layout:
            dock = self._pre_dockarea_dock(key)
            if dock is None:
                return
            if checked:
                self._arrange_pre_dockarea_default()
                # Radio behavior: hide all other section docks and uncheck
                # their rail buttons so only one drawer section is visible.
                for other_key, other_btn in self._section_buttons.items():
                    if other_key == key:
                        continue
                    if other_btn.isChecked():
                        blocked = other_btn.blockSignals(True)
                        try:
                            other_btn.setChecked(False)
                        finally:
                            other_btn.blockSignals(blocked)
                    other_dock = self._pre_dockarea_dock(other_key)
                    if other_dock is not None:
                        try:
                            other_dock.hide()
                        except Exception:
                            pass
                dock.show()
                if key == "artifacts":
                    self.artifact_panel.show()
                try:
                    dock.raiseDock()
                except Exception:
                    pass
                scroll = self._pre_section_scroll_hosts.get(key)
                self._focus_first_editable(scroll.widget() if scroll is not None else None)
                self._last_opened_section = key
            else:
                dock.hide()
            self._update_pre_drawer_visibility()
            self._save_panel_layout_state()
            return
        dock = self._section_docks.get(key)
        if dock is None:
            return
        if checked:
            if key not in self._section_popup_initialized or not self._is_popup_on_screen(dock):
                dock.setFloating(True)
                self._position_section_popup(dock)
                self._section_popup_initialized.add(key)
            dock.show()
            dock.raise_()
            dock.activateWindow()
            self._focus_first_editable(dock.widget())
            self._last_opened_section = key
        else:
            dock.hide()

    def _on_section_dock_visibility(self, key: str, visible: bool) -> None:
        if self._use_pg_dockarea_pre_layout:
            if key in self._section_buttons:
                self._set_section_button_checked(key, visible)
                if not visible and self._last_opened_section == key:
                    self._last_opened_section = None
                if visible:
                    self._last_opened_section = key
            self._update_pre_drawer_visibility()
            self._save_panel_layout_state()
            return
        self._set_section_button_checked(key, visible)
        if not visible and self._last_opened_section == key:
            self._last_opened_section = None
        if visible:
            self._last_opened_section = key
        self._save_panel_layout_state()

    def _position_section_popup(self, dock: QtWidgets.QDockWidget) -> None:
        """Place floating popups near the window while keeping them inside visible screen bounds."""
        geom = self.frameGeometry()
        screen_rect = self._active_screen_geometry()

        pref_w, pref_h = self._default_popup_size(dock)
        max_w = max(320, screen_rect.width() - 40)
        max_h = max(260, screen_rect.height() - 40)
        width = min(pref_w, max_w)
        height = min(pref_h, max_h)

        # Prefer the left side of the main window, then fall back to right, then clamp.
        x_right = geom.x() + geom.width() + 12
        x_left = geom.x() - width - 12
        y_pref = geom.y() + 60

        x_min = screen_rect.x() + 10
        y_min = screen_rect.y() + 10
        x_max = screen_rect.x() + max(10, screen_rect.width() - width - 10)
        y_max = screen_rect.y() + max(10, screen_rect.height() - height - 10)

        if x_left >= x_min:
            x = x_left
        elif x_right <= x_max:
            x = x_right
        else:
            x = x_max
        y = min(max(y_pref, y_min), y_max)

        dock.resize(width, height)
        dock.move(int(x), int(y))

    def _default_popup_size(self, dock: QtWidgets.QDockWidget) -> Tuple[int, int]:
        """Compact default popup sizes, with smaller heights per section."""
        geom = self.frameGeometry()
        name = str(dock.objectName() or "")
        if name.startswith(f"{_PRE_DOCK_PREFIX}") and name.endswith(".dock"):
            key = name[len(_PRE_DOCK_PREFIX):-len(".dock")]
        else:
            key = name
        height_by_section = {
            "artifacts": 380,
            "filtering": 340,
            "baseline": 380,
            "output": 410,
            "qc": 300,
            "export": 270,
            "config": 280,
        }
        pref_h = int(height_by_section.get(key, 340))
        pref_w = max(360, int(geom.width() * 0.24))
        return pref_w, pref_h

    def _active_screen_geometry(self) -> QtCore.QRect:
        handle = self.windowHandle()
        screen = handle.screen() if handle else None
        if screen is None:
            screen = QtGui.QGuiApplication.screenAt(self.frameGeometry().center())
        if screen is None:
            screen = QtGui.QGuiApplication.primaryScreen()
        if screen is None:
            return QtCore.QRect(0, 0, 1920, 1080)
        return screen.availableGeometry()

    def _is_popup_on_screen(self, dock: QtWidgets.QDockWidget) -> bool:
        rect = dock.frameGeometry()
        if rect.width() <= 0 or rect.height() <= 0:
            return False
        for screen in QtGui.QGuiApplication.screens():
            if screen.availableGeometry().intersects(rect):
                return True
        return False

    def eventFilter(self, obj: QtCore.QObject, event: QtCore.QEvent) -> bool:
        try:
            if event.type() in (QtCore.QEvent.Type.MouseButtonPress, QtCore.QEvent.Type.FocusIn):
                dock: Optional[QtWidgets.QDockWidget] = None
                if isinstance(obj, QtWidgets.QDockWidget):
                    dock = obj
                elif isinstance(obj, QtWidgets.QWidget):
                    parent = obj
                    while parent is not None and not isinstance(parent, QtWidgets.QDockWidget):
                        parent = parent.parentWidget()
                    if isinstance(parent, QtWidgets.QDockWidget):
                        dock = parent
                if dock is not None:
                    name = str(dock.objectName() or "")
                    if name.startswith(_PRE_DOCK_PREFIX):
                        self._pre_last_interacted_dock_name = name
        except Exception:
            pass
        return super().eventFilter(obj, event)

    def getPreDockWidgets(self) -> List[QtWidgets.QDockWidget]:
        if self._use_pg_dockarea_pre_layout:
            docks: List[QtWidgets.QDockWidget] = []
            if isinstance(self.art_dock, QtWidgets.QDockWidget):
                docks.append(self.art_dock)
        else:
            docks = list(self._section_docks.values())
            if isinstance(self.art_dock, QtWidgets.QDockWidget):
                docks.append(self.art_dock)
        seen: set[int] = set()
        out: List[QtWidgets.QDockWidget] = []
        for dock in docks:
            did = id(dock)
            if did in seen:
                continue
            seen.add(did)
            out.append(dock)
        return out

    def getPostDockWidgets(self) -> List[QtWidgets.QDockWidget]:
        docks: List[QtWidgets.QDockWidget] = []
        try:
            self.post_tab.ensure_section_popups_initialized()
            docks = list(self.post_tab.get_section_dock_widgets())
        except Exception:
            docks = []
        if docks:
            return docks
        # Fallback for legacy sessions where post docks may already exist but not registered.
        return [
            d for d in self.findChildren(QtWidgets.QDockWidget)
            if str(d.objectName() or "").startswith(_POST_DOCK_PREFIX)
        ]

    def _hide_dock_widgets(self, docks: List[QtWidgets.QDockWidget], *, remove: bool = True) -> None:
        for dock in docks:
            if dock is None:
                continue
            try:
                dock.hide()
            except Exception:
                pass
            if remove:
                try:
                    self.removeDockWidget(dock)
                except Exception:
                    pass

    def hideOtherTabDocks(self, tab_name: str) -> None:
        if tab_name == "pre":
            remove_post = not self._force_fixed_dock_layouts
            self._hide_dock_widgets(self.getPostDockWidgets(), remove=remove_post)
        elif tab_name == "post":
            self._hide_dock_widgets(self.getPreDockWidgets(), remove=True)
            # Final guard: keep post dock registry initialized before post restore paths run.
            try:
                self.post_tab.ensure_section_popups_initialized()
            except Exception:
                pass

    def _enforce_only_tab_docks_visible(self, tab_name: str) -> None:
        self.hideOtherTabDocks(tab_name)

    def captureDockSnapshotForTab(self, tab_name: str) -> Optional[QtCore.QByteArray]:
        """
        Capture a tab-scoped dock snapshot. Other-tab docks are hidden first so
        QMainWindow.saveState() cannot serialize mixed-tab layouts.
        """
        if tab_name not in {"pre", "post"}:
            return None
        try:
            self.hideOtherTabDocks(tab_name)
            QtWidgets.QApplication.processEvents(QtCore.QEventLoop.ProcessEventsFlag.ExcludeUserInputEvents)
            state = self.saveState(_DOCK_STATE_VERSION)
            self.hideOtherTabDocks(tab_name)
            if state is None or state.isEmpty():
                _LOG.warning("Dock snapshot capture returned empty state for tab=%s", tab_name)
                return None
            if not self._is_tab_scoped_dock_state(tab_name, state):
                _LOG.warning("Discarding contaminated dock snapshot for tab=%s", tab_name)
                return None
            return state
        except Exception:
            _LOG.exception("Dock snapshot capture failed for tab=%s", tab_name)
            return None

    def restoreDockSnapshotForTab(self, tab_name: str, state: QtCore.QByteArray) -> bool:
        """
        Restore a tab-scoped dock snapshot with hard post-restore enforcement so
        foreign tab docks cannot leak back into the active tab.
        """
        if tab_name not in {"pre", "post"}:
            return False
        if state is None or state.isEmpty():
            return False
        if not self._is_tab_scoped_dock_state(tab_name, state):
            _LOG.warning("Rejecting invalid/contaminated dock snapshot for tab=%s", tab_name)
            try:
                if tab_name == "pre":
                    self.settings.remove(_PRE_DOCK_STATE_KEY)
                else:
                    self.settings.remove(_POST_DOCK_STATE_KEY)
            except Exception:
                pass
            return False
        try:
            self.hideOtherTabDocks(tab_name)
            ok = bool(self.restoreState(state, _DOCK_STATE_VERSION))
            self.hideOtherTabDocks(tab_name)
            self._enforce_only_tab_docks_visible(tab_name)
            if not ok:
                _LOG.warning("Dock snapshot restore failed for tab=%s", tab_name)
            else:
                _LOG.info("Dock snapshot restore succeeded for tab=%s", tab_name)
            return ok
        except Exception:
            _LOG.exception("Dock snapshot restore crashed for tab=%s", tab_name)
            return False

    def _capture_pre_tab_groups_state(self) -> List[Dict[str, object]]:
        """
        Capture tabified pre-dock groups + active tab candidate for fallback restore.
        """
        docks = [d for d in self.getPreDockWidgets() if not d.isFloating()]
        by_name: Dict[str, QtWidgets.QDockWidget] = {
            str(d.objectName()): d for d in docks if str(d.objectName() or "")
        }
        groups: List[Dict[str, object]] = []
        visited: set[str] = set()
        for dock in docks:
            name = str(dock.objectName() or "")
            if not name or name in visited:
                continue
            members = [dock] + [d for d in self.tabifiedDockWidgets(dock) if d in docks]
            member_names = sorted({str(d.objectName() or "") for d in members if str(d.objectName() or "")})
            if len(member_names) < 2:
                continue
            visited.update(member_names)
            active = ""
            if self._pre_last_interacted_dock_name and self._pre_last_interacted_dock_name in member_names:
                active = self._pre_last_interacted_dock_name
            if not active:
                for n in member_names:
                    d = by_name.get(n)
                    if d is not None and d.isVisible():
                        active = n
                        break
            if not active:
                active = member_names[0]
            groups.append({"members": member_names, "active": active})
        return groups

    def _save_pre_tab_groups_to_settings(self, groups: List[Dict[str, object]]) -> None:
        try:
            self.settings.setValue(_PRE_TAB_GROUPS_KEY, json.dumps(groups))
        except Exception:
            pass

    def _load_pre_tab_groups_from_settings(self) -> List[Dict[str, object]]:
        try:
            raw = self.settings.value(_PRE_TAB_GROUPS_KEY, "", type=str)
            if not raw:
                return []
            data = json.loads(raw)
            if isinstance(data, list):
                return [g for g in data if isinstance(g, dict)]
        except Exception:
            pass
        return []

    def _restore_pre_tab_groups_fallback(self, groups: List[Dict[str, object]]) -> None:
        if not groups:
            return
        by_name = {
            str(d.objectName() or ""): d for d in self.getPreDockWidgets() if str(d.objectName() or "")
        }
        for group in groups:
            members_raw = group.get("members", [])
            if not isinstance(members_raw, list):
                continue
            members = [by_name.get(str(n)) for n in members_raw]
            members = [d for d in members if isinstance(d, QtWidgets.QDockWidget)]
            members = [d for d in members if not d.isFloating()]
            if len(members) < 2:
                continue
            root = members[0]
            for d in members[1:]:
                try:
                    self.tabifyDockWidget(root, d)
                except Exception:
                    continue
            active_name = str(group.get("active", ""))
            active = by_name.get(active_name)
            if isinstance(active, QtWidgets.QDockWidget) and active in members:
                try:
                    active.show()
                    active.raise_()
                    active.setFocus(QtCore.Qt.FocusReason.OtherFocusReason)
                    self._pre_last_interacted_dock_name = active_name
                except Exception:
                    pass

    def _schedule_pre_snapshot_retry(self, delay_ms: int) -> None:
        if self._pre_snapshot_retry_scheduled:
            return
        self._pre_snapshot_retry_scheduled = True
        QtCore.QTimer.singleShot(max(0, int(delay_ms)), self._retry_pre_snapshot_restore)

    def _retry_pre_snapshot_restore(self) -> None:
        self._pre_snapshot_retry_scheduled = False
        self._apply_pre_main_dock_snapshot_if_needed()

    def onPostDocksReady(self) -> None:
        self.on_post_docks_ready()

    def on_post_docks_ready(self) -> None:
        self._post_docks_ready = True
        if self._force_fixed_dock_layouts:
            self._pre_snapshot_applied = True
            return
        if not self._pre_snapshot_applied:
            self._schedule_pre_snapshot_retry(0)

    def _set_data_panel_visible(self, visible: bool, persist: bool = True) -> None:
        vis = bool(visible)
        self.file_panel.setVisible(vis)
        self.btn_toggle_data.blockSignals(True)
        self.btn_toggle_data.setChecked(vis)
        self.btn_toggle_data.blockSignals(False)
        self._save_splitter_sizes()
        if persist:
            self._save_settings()

    def _toggle_data_panel_shortcut(self) -> None:
        self._set_data_panel_visible(not self.file_panel.isVisible())

    def _toggle_all_parameter_popups_shortcut(self) -> None:
        if self._use_pg_dockarea_pre_layout:
            any_open = any(
                bool(dock.isVisible())
                for key, dock in self._pre_dockarea_docks.items()
                if key in self._section_buttons
            )
            if any_open:
                for key in self._section_buttons.keys():
                    self._set_pre_dockarea_visible(key, False)
                    self._set_section_button_checked(key, False)
                self._last_opened_section = None
                self._update_pre_drawer_visibility()
                self._save_panel_layout_state()
                return
            self._toggle_section_shortcut("output")
            return
        any_open = any(d.isVisible() for d in self._section_docks.values())
        if any_open:
            for key, dock in self._section_docks.items():
                dock.hide()
                self._set_section_button_checked(key, False)
            self._last_opened_section = None
            return
        self._toggle_section_shortcut("output")

    def _toggle_section_shortcut(self, key: str) -> None:
        btn = self._section_buttons.get(key)
        if btn is None:
            return
        next_state = not btn.isChecked()
        self._set_section_button_checked(key, next_state)
        self._toggle_section_popup(key, next_state)

    def _close_focused_popup(self) -> None:
        if self._use_pg_dockarea_pre_layout:
            if self._last_opened_section:
                dock = self._pre_dockarea_dock(self._last_opened_section)
                if dock is not None and dock.isVisible():
                    dock.hide()
                    self._set_section_button_checked(self._last_opened_section, False)
                    self._last_opened_section = None
                    self._update_pre_drawer_visibility()
                    self._save_panel_layout_state()
            return
        fw = QtWidgets.QApplication.focusWidget()
        while fw is not None and not isinstance(fw, QtWidgets.QDockWidget):
            fw = fw.parentWidget()
        if isinstance(fw, QtWidgets.QDockWidget):
            fw.close()
            return
        if self._last_opened_section:
            dock = self._section_docks.get(self._last_opened_section)
            if dock is not None and dock.isVisible():
                dock.close()

    def _is_text_entry_focused(self) -> bool:
        fw = QtWidgets.QApplication.focusWidget()
        if fw is None:
            return False
        if isinstance(fw, QtWidgets.QAbstractButton):
            return False
        if isinstance(fw, (QtWidgets.QLineEdit, QtWidgets.QPlainTextEdit, QtWidgets.QTextEdit)):
            return True
        if isinstance(fw, (QtWidgets.QSpinBox, QtWidgets.QDoubleSpinBox, QtWidgets.QAbstractSpinBox)):
            return True
        if isinstance(fw, QtWidgets.QComboBox) and fw.isEditable():
            return True
        parent = fw.parentWidget()
        while parent is not None:
            if isinstance(parent, (QtWidgets.QSpinBox, QtWidgets.QDoubleSpinBox, QtWidgets.QAbstractSpinBox)):
                return True
            if isinstance(parent, QtWidgets.QComboBox) and parent.isEditable():
                return True
            parent = parent.parentWidget()
        return False

    def _focus_first_editable(self, root: Optional[QtWidgets.QWidget]) -> None:
        if root is None:
            return
        editable_types = (
            QtWidgets.QLineEdit,
            QtWidgets.QPlainTextEdit,
            QtWidgets.QTextEdit,
            QtWidgets.QSpinBox,
            QtWidgets.QDoubleSpinBox,
            QtWidgets.QAbstractSpinBox,
            QtWidgets.QComboBox,
        )
        for w in root.findChildren(QtWidgets.QWidget):
            if not isinstance(w, editable_types):
                continue
            if not w.isVisible() or not w.isEnabled():
                continue
            if isinstance(w, QtWidgets.QComboBox) and not w.isEditable():
                continue
            try:
                w.setFocus(QtCore.Qt.FocusReason.TabFocusReason)
            except Exception:
                continue
            if isinstance(w, QtWidgets.QAbstractSpinBox):
                le = w.lineEdit()
                if le is not None:
                    le.selectAll()
            elif isinstance(w, QtWidgets.QLineEdit):
                w.selectAll()
            return

    def _bind_shortcut(
        self,
        sequence: str,
        callback: Callable[[], None],
        *,
        require_non_text_focus: bool = False,
    ) -> None:
        shortcut = QtGui.QShortcut(QtGui.QKeySequence(sequence), self)
        shortcut.setContext(QtCore.Qt.ShortcutContext.WidgetWithChildrenShortcut)

        def _on_activated() -> None:
            if self.tabs.currentWidget() is not self.pre_tab:
                return
            if self._is_text_entry_focused():
                return
            callback()

        shortcut.activated.connect(_on_activated)
        self._shortcuts.append(shortcut)

    def _init_shortcuts(self) -> None:
        # Keyboard-first workflow for preprocessing actions.
        self._bind_shortcut("Ctrl+O", self._open_files_dialog)
        self._bind_shortcut("Ctrl+Shift+O", self._open_folder_dialog)
        self._bind_shortcut("Delete", self.file_panel._remove_selected_files, require_non_text_focus=True)
        self._bind_shortcut("Ctrl+Return", self._trigger_preview, require_non_text_focus=True)
        self._bind_shortcut("Ctrl+Enter", self._trigger_preview, require_non_text_focus=True)
        self._bind_shortcut("Ctrl+E", self._export_selected_or_all, require_non_text_focus=True)
        self._bind_shortcut("Ctrl+K", lambda: self._toggle_section_shortcut("artifacts"), require_non_text_focus=True)
        self._bind_shortcut("Ctrl+F", lambda: self._toggle_section_shortcut("filtering"), require_non_text_focus=True)
        self._bind_shortcut("Ctrl+B", lambda: self._toggle_section_shortcut("baseline"), require_non_text_focus=True)
        self._bind_shortcut("Ctrl+M", lambda: self._toggle_section_shortcut("output"), require_non_text_focus=True)
        self._bind_shortcut("Ctrl+Q", self._run_qc_dialog, require_non_text_focus=True)
        self._bind_shortcut("Ctrl+Shift+Q", self._run_batch_qc, require_non_text_focus=True)
        self._bind_shortcut("Ctrl+L", self.param_panel._load_config, require_non_text_focus=True)
        self._bind_shortcut("Ctrl+S", self.param_panel._save_config, require_non_text_focus=True)
        self._bind_shortcut("Ctrl+D", self._toggle_data_panel_shortcut, require_non_text_focus=True)
        self._bind_shortcut("Ctrl+P", self._toggle_all_parameter_popups_shortcut, require_non_text_focus=True)
        self._bind_shortcut("A", self._assign_pending_box_to_artifact, require_non_text_focus=True)
        self._bind_shortcut("C", self._assign_pending_box_to_cut, require_non_text_focus=True)
        self._bind_shortcut("S", self._assign_pending_box_to_section, require_non_text_focus=True)
        self._bind_shortcut("Escape", self._close_focused_popup, require_non_text_focus=True)

    # ----------------------------------------------------------------------
    # UX polish: tutorial / toasts / global shortcut callbacks
    # ----------------------------------------------------------------------

    def _toast_from_status(self, message: str, timeout_ms: int = 0) -> None:
        if not getattr(self, "_toaster", None) or not message:
            return
        text = str(message)
        lower = text.lower()
        sev = "info"
        if "fail" in lower or "error" in lower or "could not" in lower:
            sev = "error"
        elif "warn" in lower or "dropped" in lower or "skipped" in lower:
            sev = "warn"
        elif "complete" in lower or "saved" in lower or "loaded" in lower or " ok" in lower:
            sev = "ok"
        timeout = int(timeout_ms) if timeout_ms and timeout_ms > 0 else int(self.settings.value("ui/toast_timeout_ms", 5000) or 5000)
        self._toaster.post(text, sev, timeout)

    def _maybe_show_first_run_tutorial(self) -> None:
        try:
            seen = self.settings.value("onboarding/first_run_completed", False)
            replay_next_launch = self.settings.value("onboarding/replay_next_launch", False)
            from onboarding import _to_bool
            if _to_bool(seen, False) and not _to_bool(replay_next_launch, False):
                return
        except Exception:
            pass
        self._show_tutorial_again(automatic=True)

    def _show_tutorial_again(self, automatic: bool = False) -> None:
        try:
            steps = build_default_tutorial(self)
            overlay = TutorialOverlay(self, steps)
            def _finish_tutorial() -> None:
                self.settings.setValue("onboarding/first_run_completed", True)
                if automatic or overlay.dont_show_again():
                    self.settings.setValue("onboarding/replay_next_launch", False)
                    self.settings.setValue("onboarding/show_on_startup", False)
            overlay.finished.connect(_finish_tutorial)
            overlay.start()
        except Exception:
            pass

    def _show_keyboard_cheatsheet(self) -> None:
        # Open the Preferences dialog directly on the Keyboard tab.
        try:
            dlg = PreferencesDialog(self, self.settings)
            for i in range(dlg.findChild(QtWidgets.QTabWidget).count()):
                tabs = dlg.findChild(QtWidgets.QTabWidget)
                if tabs.tabText(i).lower().startswith("keyboard"):
                    tabs.setCurrentIndex(i)
                    break
            dlg.exec()
        except Exception:
            pass

    def _open_preferences(self) -> None:
        try:
            dlg = PreferencesDialog(self, self.settings)
            if dlg.exec() == QtWidgets.QDialog.DialogCode.Accepted:
                # Apply theme right away if it changed.
                desired = str(self.settings.value("app/theme", "dark") or "dark").lower()
                current = getattr(self, "_app_theme_mode", "dark")
                if desired != current:
                    if desired == "light":
                        self.act_app_theme_light.setChecked(True)
                    else:
                        self.act_app_theme_dark.setChecked(True)
                    self._on_app_theme_changed()
                if getattr(self, "_toaster", None):
                    self._toaster.ok("Preferences saved.")
        except Exception:
            pass

    # --- tab navigation ---

    def _focus_pre_tab(self) -> None:
        try:
            self.tabs.setCurrentIndex(0)
        except Exception:
            pass

    def _focus_post_tab(self) -> None:
        try:
            self.tabs.setCurrentIndex(1)
        except Exception:
            pass

    def _cycle_main_tab(self) -> None:
        try:
            n = self.tabs.count()
            if n <= 1:
                return
            self.tabs.setCurrentIndex((self.tabs.currentIndex() + 1) % n)
        except Exception:
            pass

    # --- file navigation in postprocessing/temporal ---

    def _step_active_file_next(self) -> None:
        self._step_active_file(+1)

    def _step_active_file_prev(self) -> None:
        self._step_active_file(-1)

    def _step_active_file(self, delta: int) -> None:
        # Try the temporal panel's combo (covers the GLM scope strip).
        try:
            section = getattr(self.post_tab, "section_temporal", None)
            if section is not None and hasattr(section, "_step_active_file"):
                section._step_active_file(delta)
                return
        except Exception:
            pass
        # Fall back to postprocessing's own file combo, if any.
        try:
            combo = getattr(self.post_tab, "combo_individual_file", None)
            if combo is not None:
                idx = max(0, min(combo.count() - 1, combo.currentIndex() + int(delta)))
                combo.setCurrentIndex(idx)
        except Exception:
            pass

    def _toggle_individual_group(self) -> None:
        try:
            bar = getattr(self.post_tab, "tab_visual_mode", None)
            if bar is None:
                return
            cur = bar.currentIndex()
            bar.setCurrentIndex((cur + 1) % bar.count())
        except Exception:
            pass

    # --- temporal modeling ---

    def _fit_temporal_model(self) -> None:
        try:
            section = getattr(self.post_tab, "section_temporal", None)
            if section is None:
                return
            self.tabs.setCurrentIndex(1)
            section._on_fit_clicked()
        except Exception:
            pass

    def _fit_temporal_all_files(self) -> None:
        try:
            section = getattr(self.post_tab, "section_temporal", None)
            if section is None:
                return
            self.tabs.setCurrentIndex(1)
            section._on_fit_all_files_clicked()
        except Exception:
            pass

    def _recompute_psth(self) -> None:
        try:
            self.tabs.setCurrentIndex(1)
            fn = getattr(self.post_tab, "_compute_psth", None)
            if callable(fn):
                fn()
        except Exception:
            pass

    def _run_postprocess_export(self) -> None:
        try:
            self.tabs.setCurrentIndex(1)
            for name in ("_run_export", "_export_current", "_on_run_export_clicked", "run_export"):
                fn = getattr(self.post_tab, name, None)
                if callable(fn):
                    fn()
                    return
        except Exception:
            pass

    def _cancel_current_operation(self) -> None:
        # Temporal modeling batch
        try:
            section = getattr(self.post_tab, "section_temporal", None)
            if section is not None:
                section._batch_cancel_requested = True
        except Exception:
            pass
        # Preprocessing has its own Esc handling for popups; let it through too.
        try:
            self._close_focused_popup()
        except Exception:
            pass

    def _reset_focused_plot_view(self) -> None:
        try:
            reset_focused_plot_view(self)
            if getattr(self, "_toaster", None):
                self._toaster.info("Reset plot view.", timeout_ms=1800)
        except Exception:
            pass

    def _update_busy_indicator(self) -> None:
        """Reflect any running batch op (currently: Temporal Modeling) in the status bar."""
        try:
            section = getattr(self.post_tab, "section_temporal", None)
            if section is None or not hasattr(section, "progress_model"):
                self._busy_widget.setVisible(False)
                return
            progress = section.progress_model
            if progress.isVisible():
                fmt = progress.format() or "Running..."
                # Strip the trailing "%p%" placeholder for our compact label.
                label = fmt.replace("%p%", "").strip(" :")
                self._busy_label.setText(label or "Running...")
                self._busy_widget.setVisible(True)
            else:
                self._busy_widget.setVisible(False)
        except Exception:
            self._busy_widget.setVisible(False)

    def _save_post_project_for_close(self) -> bool:
        """
        Used by the close-confirmation handler. Returns True on success.
        """
        try:
            fn = (
                getattr(self.post_tab, "_save_project_dialog", None)
                or getattr(self.post_tab, "_save_project_file", None)
                or getattr(self.post_tab, "_save_project", None)
            )
            if callable(fn):
                fn()
                checker = getattr(self.post_tab, "is_project_dirty", None)
                if callable(checker):
                    return not bool(checker())
                return not bool(getattr(self.post_tab, "_project_dirty", False))
        except Exception:
            pass
        # No save handler available: let the user decide via Discard/Cancel.
        return False

    def _discard_post_project_for_close(self) -> bool:
        """
        Used by the close-confirmation handler after the user chooses Discard.
        Prevents the discarded state from being autosaved and reopened next launch.
        """
        try:
            fn = getattr(self.post_tab, "discard_unsaved_project_for_close", None)
            if callable(fn):
                return bool(fn())
        except Exception:
            pass
        try:
            self.post_tab._project_dirty = False
            self.post_tab._project_recovered_from_autosave = False
            self.post_tab._clear_project_autosave_cache(delete_file=True)
            self.post_tab._mark_project_clean()
            return True
        except Exception:
            return False

    def _dock_area_from_settings(
        self,
        value: object,
        default: QtCore.Qt.DockWidgetArea = QtCore.Qt.DockWidgetArea.LeftDockWidgetArea,
    ) -> QtCore.Qt.DockWidgetArea:
        left_i = _dock_area_to_int(QtCore.Qt.DockWidgetArea.LeftDockWidgetArea, 1)
        right_i = _dock_area_to_int(QtCore.Qt.DockWidgetArea.RightDockWidgetArea, 2)
        top_i = _dock_area_to_int(QtCore.Qt.DockWidgetArea.TopDockWidgetArea, 4)
        bottom_i = _dock_area_to_int(QtCore.Qt.DockWidgetArea.BottomDockWidgetArea, 8)
        area_int = _dock_area_to_int(value, _dock_area_to_int(default, right_i))
        area_map: Dict[int, QtCore.Qt.DockWidgetArea] = {
            left_i: QtCore.Qt.DockWidgetArea.LeftDockWidgetArea,
            right_i: QtCore.Qt.DockWidgetArea.RightDockWidgetArea,
            top_i: QtCore.Qt.DockWidgetArea.TopDockWidgetArea,
            bottom_i: QtCore.Qt.DockWidgetArea.BottomDockWidgetArea,
        }
        if area_int in area_map:
            return area_map[area_int]
        return default

    def _to_qbytearray(self, value: object) -> Optional[QtCore.QByteArray]:
        if isinstance(value, QtCore.QByteArray):
            return value
        if isinstance(value, (bytes, bytearray)):
            return QtCore.QByteArray(bytes(value))
        if isinstance(value, str):
            # QSettings may return serialized bytearrays as text with some backends.
            try:
                return QtCore.QByteArray.fromBase64(value.encode("utf-8"))
            except Exception:
                return None
        return None

    def _qbytearray_to_b64(self, value: Optional[QtCore.QByteArray]) -> str:
        if value is None:
            return ""
        try:
            if value.isEmpty():
                return ""
            return bytes(value.toBase64()).decode("ascii")
        except Exception:
            return ""

    def _b64_to_qbytearray(self, value: object) -> Optional[QtCore.QByteArray]:
        if not value:
            return None
        if isinstance(value, QtCore.QByteArray):
            return value
        if isinstance(value, (bytes, bytearray)):
            try:
                return QtCore.QByteArray.fromBase64(bytes(value))
            except Exception:
                return None
        if isinstance(value, str):
            try:
                return QtCore.QByteArray.fromBase64(value.encode("ascii"))
            except Exception:
                return None
        return None

    def _dock_state_prefix_presence(self, state: QtCore.QByteArray) -> Tuple[bool, bool]:
        """
        Return (has_pre_prefix, has_post_prefix) for a Qt dock-state blob.
        Object names may be serialized as ASCII or UTF-16LE.
        """
        try:
            raw = bytes(state)
        except Exception:
            return False, False
        pre_ascii = b"pre."
        post_ascii = b"post."
        pre_utf16 = "pre.".encode("utf-16-le")
        post_utf16 = "post.".encode("utf-16-le")
        has_pre = (pre_ascii in raw) or (pre_utf16 in raw)
        has_post = (post_ascii in raw) or (post_utf16 in raw)
        return has_pre, has_post

    def _is_tab_scoped_dock_state(self, tab_name: str, state: QtCore.QByteArray) -> bool:
        if tab_name not in {"pre", "post"}:
            return False
        if state is None or state.isEmpty():
            return False
        has_pre, has_post = self._dock_state_prefix_presence(state)
        if tab_name == "pre" and has_post:
            return False
        if tab_name == "post" and has_pre:
            return False
        return True

    def _migrate_legacy_dock_state_settings(self) -> None:
        """
        Drop legacy full-window dock blobs that reference old object names.
        New snapshots are tab-scoped and use pre./post. dock prefixes.
        """
        try:
            if self.settings.contains("pre_main_dock_state_v3") and not self.settings.contains(_PRE_DOCK_STATE_KEY):
                self.settings.remove("pre_main_dock_state_v3")
            if self.settings.contains("post_main_dock_state_v3") and not self.settings.contains(_POST_DOCK_STATE_KEY):
                self.settings.remove("post_main_dock_state_v3")
            pre_state = self._to_qbytearray(self.settings.value(_PRE_DOCK_STATE_KEY, None))
            if pre_state is not None and not pre_state.isEmpty():
                if not self._is_tab_scoped_dock_state("pre", pre_state):
                    self.settings.remove(_PRE_DOCK_STATE_KEY)
            post_state = self._to_qbytearray(self.settings.value(_POST_DOCK_STATE_KEY, None))
            if post_state is not None and not post_state.isEmpty():
                if not self._is_tab_scoped_dock_state("post", post_state):
                    self.settings.remove(_POST_DOCK_STATE_KEY)
        except Exception:
            pass

    def _panel_config_json_path(self) -> str:
        base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        return os.path.join(base_dir, "panel_layout.json")

    def _load_panel_config_json_into_settings(self) -> None:
        """Load panel layout JSON into QSettings so existing restore logic can use it."""
        path = self._panel_config_json_path()
        if not os.path.isfile(path):
            return
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            return
        self._load_panel_config_payload_into_settings(data)

    def _load_panel_config_payload_into_settings(self, data: object) -> None:
        """Load a panel layout payload (same schema as panel_layout.json) into QSettings."""
        if not isinstance(data, dict):
            return

        try:
            layout_version = int(data.get("version", 1))
        except Exception:
            layout_version = 1
        # Snapshot blobs are considered stable starting from layout schema v3.
        # v2 files may contain mixed pre/post docks from legacy full-window capture.
        allow_snapshot_blobs = layout_version >= 3

        pre = data.get("pre", {}) if isinstance(data.get("pre"), dict) else {}
        post = data.get("post", {}) if isinstance(data.get("post"), dict) else {}

        try:
            if "pre_data_panel_visible" in pre:
                self.settings.setValue("pre_data_panel_visible", bool(pre["pre_data_panel_visible"]))
            if "pre_splitter_sizes" in pre and isinstance(pre["pre_splitter_sizes"], list):
                self.settings.setValue("pre_splitter_sizes", [int(x) for x in pre["pre_splitter_sizes"]])
            if allow_snapshot_blobs and "pre_main_dock_state" in pre:
                ba = self._b64_to_qbytearray(pre.get("pre_main_dock_state"))
                if ba is not None and self._is_tab_scoped_dock_state("pre", ba):
                    self.settings.setValue(_PRE_DOCK_STATE_KEY, ba)
                else:
                    self.settings.remove(_PRE_DOCK_STATE_KEY)
            if allow_snapshot_blobs and "post_main_dock_state" in post:
                ba = self._b64_to_qbytearray(post.get("post_main_dock_state"))
                if ba is not None and self._is_tab_scoped_dock_state("post", ba):
                    self.settings.setValue(_POST_DOCK_STATE_KEY, ba)
                else:
                    self.settings.remove(_POST_DOCK_STATE_KEY)
            if "tab_groups" in pre and isinstance(pre.get("tab_groups"), list):
                self.settings.setValue(_PRE_TAB_GROUPS_KEY, json.dumps(pre.get("tab_groups")))
        except Exception:
            pass

        def _apply_section_settings(prefix: str, section_map: object) -> None:
            if not isinstance(section_map, dict):
                return
            for key, sec in section_map.items():
                if not isinstance(sec, dict):
                    continue
                base = f"{prefix}/{key}"
                if "visible" in sec:
                    self.settings.setValue(f"{base}/visible", bool(sec["visible"]))
                if "floating" in sec:
                    self.settings.setValue(f"{base}/floating", bool(sec["floating"]))
                if "area" in sec:
                    try:
                        self.settings.setValue(f"{base}/area", int(sec["area"]))
                    except Exception:
                        pass
                if "geometry" in sec:
                    ba = self._b64_to_qbytearray(sec.get("geometry"))
                    if ba is not None:
                        self.settings.setValue(f"{base}/geometry", ba)

        _apply_section_settings("pre_section_docks", pre.get("sections"))
        _apply_section_settings("post_section_docks", post.get("sections"))

        art = pre.get("artifact", {}) if isinstance(pre.get("artifact"), dict) else {}
        if art:
            base = "pre_artifact_dock_state"
            if "visible" in art:
                self.settings.setValue(f"{base}/visible", bool(art["visible"]))
            if "floating" in art:
                self.settings.setValue(f"{base}/floating", bool(art["floating"]))
            if "area" in art:
                try:
                    self.settings.setValue(f"{base}/area", int(art["area"]))
                except Exception:
                    pass
            if "geometry" in art:
                ba = self._b64_to_qbytearray(art.get("geometry"))
                if ba is not None:
                    self.settings.setValue(f"{base}/geometry", ba)

        try:
            self.settings.sync()
        except Exception:
            pass

    def _collect_panel_layout_payload(self) -> Dict[str, object]:
        """Build the panel layout payload used for both JSON persistence and config export."""
        # Ensure QSettings has the latest dock values
        try:
            self._save_panel_layout_state()
        except Exception:
            pass
        try:
            self.post_tab.flush_post_section_state_to_settings()
            self.post_tab._save_panel_layout_state()
        except Exception:
            pass

        splitter_sizes: Optional[List[int]] = None
        try:
            raw_sizes = self.settings.value("pre_splitter_sizes", None)
            if raw_sizes is not None and hasattr(raw_sizes, "__len__"):
                splitter_sizes = [int(x) for x in raw_sizes]
        except Exception:
            splitter_sizes = None

        def _read_section_settings(prefix: str, keys: List[str]) -> Dict[str, Dict[str, object]]:
            out: Dict[str, Dict[str, object]] = {}
            for key in keys:
                base = f"{prefix}/{key}"
                visible = self.settings.value(f"{base}/visible", None)
                floating = self.settings.value(f"{base}/floating", None)
                area = self.settings.value(f"{base}/area", None)
                geom = self._to_qbytearray(self.settings.value(f"{base}/geometry", None))
                left_i = _dock_area_to_int(QtCore.Qt.DockWidgetArea.LeftDockWidgetArea, 1)
                out[key] = {
                    "visible": _to_bool(visible, False) if visible is not None else False,
                    "floating": _to_bool(floating, True) if floating is not None else True,
                    "area": _dock_area_to_int(area, left_i) if area is not None else left_i,
                    "geometry": self._qbytearray_to_b64(geom),
                }
            return out

        if self._use_pg_dockarea_pre_layout and self._pre_dockarea_docks:
            pre_sections = list(self._pre_dockarea_docks.keys())
        else:
            pre_sections = list(self._section_docks.keys())
        post_sections = []
        try:
            self.post_tab.ensure_section_popups_initialized()
            post_sections = list(self.post_tab.get_section_popup_keys())
        except Exception:
            post_sections = []

        pre_main = self._to_qbytearray(self.settings.value(_PRE_DOCK_STATE_KEY, None))
        post_main = self._to_qbytearray(self.settings.value(_POST_DOCK_STATE_KEY, None))
        pre_tab_groups = self._load_pre_tab_groups_from_settings()

        art_geom = self._to_qbytearray(self.settings.value("pre_artifact_dock_state/geometry", None))
        art_visible = self.settings.value("pre_artifact_dock_state/visible", None)
        art_floating = self.settings.value("pre_artifact_dock_state/floating", None)
        art_area = self.settings.value("pre_artifact_dock_state/area", None)

        data = {
            "version": 3,
            "pre": {
                "pre_data_panel_visible": _to_bool(self.settings.value("pre_data_panel_visible", True), True),
                "pre_splitter_sizes": splitter_sizes,
                "pre_main_dock_state": self._qbytearray_to_b64(pre_main),
                "tab_groups": pre_tab_groups,
                "sections": _read_section_settings("pre_section_docks", pre_sections),
                "artifact": {
                    "visible": _to_bool(art_visible, False) if art_visible is not None else False,
                    "floating": _to_bool(art_floating, False) if art_floating is not None else False,
                    "area": int(art_area) if art_area is not None else _dock_area_to_int(QtCore.Qt.DockWidgetArea.LeftDockWidgetArea, 1),
                    "geometry": self._qbytearray_to_b64(art_geom),
                },
            },
            "post": {
                "post_main_dock_state": self._qbytearray_to_b64(post_main),
                "sections": _read_section_settings("post_section_docks", post_sections),
            },
        }
        return data

    def _save_panel_config_json(self) -> None:
        """Persist current panel layout into a JSON file."""
        try:
            data = self._collect_panel_layout_payload()
            path = self._panel_config_json_path()
            with open(path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
        except Exception:
            _LOG.exception("Failed to write panel layout JSON")

    def _export_preprocessing_ui_state_for_config(self) -> Dict[str, object]:
        """Extra UI payload stored in preprocessing_config.json."""
        return {
            "artifact_overlay_visible": bool(self.param_panel.artifact_overlay_visible()),
            "artifact_thresholds_visible": bool(self.plots.artifact_thresholds_visible()),
            "export_selection": self.param_panel.export_selection().to_dict(),
            "export_channel_names": self.param_panel.export_channel_names(),
            "export_trigger_names": self.param_panel.export_trigger_names(),
            "auto_export_to_source_dir": bool(self.param_panel.auto_export_enabled()),
            "panel_layout": self._collect_panel_layout_payload(),
        }

    def _apply_panel_layout_from_settings(self) -> None:
        """
        Apply dock disposition from current QSettings values.
        Used after importing panel layout from preprocessing config files.
        """
        if self._force_fixed_dock_layouts:
            self._apply_pre_fixed_layout()
            try:
                self.post_tab.ensure_section_popups_initialized()
                if hasattr(self.post_tab, "apply_fixed_default_layout"):
                    self.post_tab.apply_fixed_default_layout()
            except Exception:
                pass
            if self.tabs.currentWidget() is self.pre_tab:
                self._enforce_only_tab_docks_visible("pre")
            else:
                self._enforce_only_tab_docks_visible("post")
            return

        self._restore_panel_layout_state()
        self._pre_snapshot_applied = False
        self._pre_snapshot_retry_attempts = 0
        self._pre_snapshot_retry_scheduled = False
        self._apply_pre_main_dock_snapshot_if_needed()
        try:
            self.post_tab.ensure_section_popups_initialized()
            self.post_tab._restore_panel_layout_state()
            self.post_tab._post_snapshot_applied = False
            self.post_tab._apply_post_main_dock_snapshot_if_needed()
        except Exception:
            pass
        if self.tabs.currentWidget() is self.pre_tab:
            self._enforce_only_tab_docks_visible("pre")
        else:
            self._enforce_only_tab_docks_visible("post")

    def _import_preprocessing_ui_state_from_config(self, ui_state: Dict[str, object]) -> None:
        if not isinstance(ui_state, dict):
            return
        if "artifact_overlay_visible" in ui_state:
            visible = bool(ui_state.get("artifact_overlay_visible"))
            self.param_panel.set_artifact_overlay_visible(visible)
            self.plots.set_artifact_overlay_visible(visible)
        if "artifact_thresholds_visible" in ui_state:
            self.plots.set_artifact_thresholds_visible(bool(ui_state.get("artifact_thresholds_visible")))
        if "export_selection" in ui_state:
            self.param_panel.set_export_selection(ExportSelection.from_dict(ui_state.get("export_selection")))
        if "export_channel_names" in ui_state:
            self.param_panel.set_export_channel_names(list(ui_state.get("export_channel_names") or []))
        if "export_trigger_names" in ui_state:
            self.param_panel.set_export_trigger_names(list(ui_state.get("export_trigger_names") or []))
        if "auto_export_to_source_dir" in ui_state:
            self.param_panel.set_auto_export_enabled(_to_bool(ui_state.get("auto_export_to_source_dir"), False))
        self._update_export_summary_label()
        panel_layout = ui_state.get("panel_layout")
        if isinstance(panel_layout, dict):
            self._load_panel_config_payload_into_settings(panel_layout)
            self._apply_panel_layout_from_settings()
            self._save_panel_config_json()
        self._save_settings()

    def _pre_history_clone(self, state: Dict[str, Any]) -> Dict[str, Any]:
        try:
            return json.loads(json.dumps(state))
        except Exception:
            return dict(state)

    def _pre_history_state_key(self, state: Dict[str, Any]) -> str:
        try:
            return json.dumps(state, sort_keys=True, separators=(",", ":"))
        except Exception:
            return repr(state)

    def _pre_history_snapshot(self) -> Dict[str, Any]:
        try:
            params = self.param_panel.get_params().to_dict()
        except Exception:
            params = {}
        start_s, end_s = self._time_window_bounds()
        return {
            "params": params,
            "artifact_overlay_visible": bool(self.param_panel.artifact_overlay_visible()),
            "artifact_thresholds_visible": bool(self.plots.artifact_thresholds_visible()),
            "plot_background": self.plots.plot_background_mode(),
            "plot_grid": bool(self.plots.plot_grid_visible()),
            "export_selection": self.param_panel.export_selection().to_dict(),
            "export_channel_names": self.param_panel.export_channel_names(),
            "export_trigger_names": self.param_panel.export_trigger_names(),
            "export_output_modes": self.param_panel.export_output_modes(),
            "auto_export_to_source_dir": bool(self.param_panel.auto_export_enabled()),
            "time_window": {"start_s": start_s, "end_s": end_s},
            "manual_regions": self._keyed_regions_to_project(self._manual_regions_by_key),
            "manual_exclude_regions": self._keyed_regions_to_project(self._manual_exclude_by_key),
            "cutout_regions": self._keyed_regions_to_project(self._cutout_regions_by_key),
            "sections": self._sections_to_project(),
        }

    def _update_pre_history_buttons(self) -> None:
        try:
            self.plots.set_history_available(bool(self._pre_history_undo), bool(self._pre_history_redo))
        except Exception:
            pass

    def _reset_pre_history_snapshot(self) -> None:
        self._pre_history_undo.clear()
        self._pre_history_redo.clear()
        self._pre_history_current = self._pre_history_snapshot()
        self._pre_history_key = self._pre_history_state_key(self._pre_history_current)
        self._update_pre_history_buttons()

    def _record_pre_history_change(self) -> None:
        if self._pre_history_restoring:
            return
        state = self._pre_history_snapshot()
        key = self._pre_history_state_key(state)
        if self._pre_history_current is None:
            self._pre_history_current = self._pre_history_clone(state)
            self._pre_history_key = key
            self._update_pre_history_buttons()
            return
        if key == self._pre_history_key:
            return
        self._pre_history_undo.append(self._pre_history_clone(self._pre_history_current))
        if len(self._pre_history_undo) > self._pre_history_limit:
            self._pre_history_undo = self._pre_history_undo[-self._pre_history_limit:]
        self._pre_history_redo.clear()
        self._pre_history_current = self._pre_history_clone(state)
        self._pre_history_key = key
        self._update_pre_history_buttons()

    def _refresh_artifact_panel_for_current(self) -> None:
        key = self._current_key()
        if not key:
            self.artifact_panel.set_auto_regions([])
            self.artifact_panel.set_regions([])
            return
        start_s, end_s = self._time_window_bounds()
        manual_win = self._clip_regions_to_window(self._manual_regions_by_key.get(key, []), start_s, end_s)
        ignore_win = self._clip_regions_to_window(self._manual_exclude_by_key.get(key, []), start_s, end_s)
        auto_win = self._clip_regions_to_window(self._auto_regions_by_key.get(key, []), start_s, end_s)
        checked_auto = [r for r in auto_win if not any(self._regions_match(r, ig) for ig in ignore_win)]
        self.artifact_panel.set_auto_regions(auto_win, checked_regions=checked_auto)
        self.artifact_panel.set_regions(manual_win)

    def _restore_pre_history_state(self, state: Dict[str, Any]) -> None:
        self._pre_history_restoring = True
        try:
            params = state.get("params")
            if isinstance(params, dict):
                self.param_panel.set_params(ProcessingParams.from_dict(params))
            if "artifact_overlay_visible" in state:
                visible = bool(state.get("artifact_overlay_visible"))
                self.param_panel.set_artifact_overlay_visible(visible)
                self.plots.set_artifact_overlay_visible(visible)
            if "artifact_thresholds_visible" in state:
                self.plots.set_artifact_thresholds_visible(bool(state.get("artifact_thresholds_visible")))
            if "export_selection" in state:
                self.param_panel.set_export_selection(ExportSelection.from_dict(state.get("export_selection")))
            if "export_output_modes" in state:
                self.param_panel.set_export_output_modes(list(state.get("export_output_modes") or []), follow_current=False)
            if "export_channel_names" in state:
                self.param_panel.set_export_channel_names(list(state.get("export_channel_names") or []))
            if "export_trigger_names" in state:
                self.param_panel.set_export_trigger_names(list(state.get("export_trigger_names") or []))
            if "auto_export_to_source_dir" in state:
                self.param_panel.set_auto_export_enabled(_to_bool(state.get("auto_export_to_source_dir"), False))
            self._apply_pre_plot_style(
                state.get("plot_background", self.plots.plot_background_mode()),
                state.get("plot_grid", self.plots.plot_grid_visible()),
                persist=False,
            )
            self._manual_regions_by_key = self._project_to_keyed_regions(state.get("manual_regions"))
            self._manual_exclude_by_key = self._project_to_keyed_regions(state.get("manual_exclude_regions"))
            self._cutout_regions_by_key = self._project_to_keyed_regions(state.get("cutout_regions"))
            self._sections_by_key = self._project_to_sections(state.get("sections"))
            self._pending_box_region_by_key.clear()
            tw = state.get("time_window") if isinstance(state.get("time_window"), dict) else {}
            for ed, value in (
                (self.file_panel.edit_time_start, tw.get("start_s")),
                (self.file_panel.edit_time_end, tw.get("end_s")),
            ):
                ed.blockSignals(True)
                try:
                    ed.setText("" if value is None else f"{float(value):.6g}")
                except Exception:
                    ed.setText("")
                finally:
                    ed.blockSignals(False)
            self._last_processed.clear()
            self._refresh_artifact_panel_for_current()
            self._update_export_summary_label()
            self._update_raw_plot(preserve_view=True)
            self._trigger_preview(preserve_view=True)
            self._save_settings()
        finally:
            self._pre_history_restoring = False

    def _undo_pre_action(self) -> None:
        if not self._pre_history_undo:
            return
        current = self._pre_history_snapshot()
        previous = self._pre_history_undo.pop()
        self._pre_history_redo.append(self._pre_history_clone(current))
        self._restore_pre_history_state(previous)
        self._pre_history_current = self._pre_history_clone(previous)
        self._pre_history_key = self._pre_history_state_key(previous)
        self._update_pre_history_buttons()
        self._show_status_message("Undid preprocessing action.", 2500)

    def _redo_pre_action(self) -> None:
        if not self._pre_history_redo:
            return
        current = self._pre_history_snapshot()
        next_state = self._pre_history_redo.pop()
        self._pre_history_undo.append(self._pre_history_clone(current))
        self._restore_pre_history_state(next_state)
        self._pre_history_current = self._pre_history_clone(next_state)
        self._pre_history_key = self._pre_history_state_key(next_state)
        self._update_pre_history_buttons()
        self._show_status_message("Redid preprocessing action.", 2500)

    def _sync_section_button_states_from_docks(self) -> None:
        if self._use_pg_dockarea_pre_layout:
            self._last_opened_section = None
            for key in self._section_buttons.keys():
                dock = self._pre_dockarea_dock(key)
                visible = bool(dock.isVisible()) if dock is not None else False
                self._set_section_button_checked(key, visible)
                if visible and self._last_opened_section is None:
                    self._last_opened_section = key
            return
        self._last_opened_section = None
        for key, dock in self._section_docks.items():
            vis = bool(dock.isVisible())
            self._set_section_button_checked(key, vis)
            if vis:
                self._last_opened_section = key

    def _save_panel_layout_state(self) -> None:
        """Persist popup/artifact panel visibility, docking mode, area, and geometry."""
        if not self._panel_layout_persistence_ready:
            return
        if self._is_restoring_panel_layout:
            return
        if self._suspend_panel_layout_persistence:
            return
        # Do not overwrite stored layout while preprocessing panels are hidden for tab switching.
        if self._pre_popups_hidden_by_tab_switch:
            return

        if self._use_pg_dockarea_pre_layout:
            self._save_pre_dockarea_layout_state()
            try:
                self.settings.sync()
            except Exception:
                pass
            return

        # Per-dock persistence is isolated so one faulty dock payload cannot drop all others.
        for key, dock in self._section_docks.items():
            try:
                base = f"pre_section_docks/{key}"
                cached = (
                    self._pre_section_state_before_tab_switch.get(key, {})
                    if self._pre_popups_hidden_by_tab_switch
                    else {}
                )
                visible = bool(cached.get("visible", dock.isVisible()))
                floating = bool(cached.get("floating", dock.isFloating()))
                left_i = _dock_area_to_int(QtCore.Qt.DockWidgetArea.LeftDockWidgetArea, 1)
                area_val = _dock_area_to_int(cached.get("area", self.dockWidgetArea(dock)), left_i)
                geom = cached.get("geometry", dock.saveGeometry())
                self.settings.setValue(f"{base}/visible", visible)
                self.settings.setValue(f"{base}/floating", floating)
                self.settings.setValue(f"{base}/area", area_val)
                self.settings.setValue(f"{base}/geometry", geom)
            except Exception:
                continue

        if isinstance(self.art_dock, QtWidgets.QDockWidget):
            try:
                base = "pre_artifact_dock_state"
                cached = self._pre_artifact_state_before_tab_switch if self._pre_popups_hidden_by_tab_switch else {}
                visible = bool(cached.get("visible", self.art_dock.isVisible()))
                floating = bool(cached.get("floating", self.art_dock.isFloating()))
                left_i = _dock_area_to_int(QtCore.Qt.DockWidgetArea.LeftDockWidgetArea, 1)
                area_val = _dock_area_to_int(cached.get("area", self.dockWidgetArea(self.art_dock)), left_i)
                geom = cached.get("geometry", self.art_dock.saveGeometry())
                self.settings.setValue(f"{base}/visible", visible)
                self.settings.setValue(f"{base}/floating", floating)
                self.settings.setValue(f"{base}/area", area_val)
                self.settings.setValue(f"{base}/geometry", geom)
            except Exception:
                pass
        try:
            self._save_pre_tab_groups_to_settings(self._capture_pre_tab_groups_state())
        except Exception:
            pass
        try:
            self.settings.sync()
        except Exception:
            pass

    def _save_full_main_dock_state(self) -> None:
        """
        Save full main-window dock disposition (tabified/split relationships).
        This complements per-dock visibility settings.
        """
        try:
            self.settings.setValue("main_dock_state_v2", self.saveState(_DOCK_STATE_VERSION))
            self.settings.sync()
        except Exception:
            pass

    def _restore_full_main_dock_state(self) -> None:
        """
        Restore full main-window dock disposition after all docks are registered.
        """
        try:
            raw = self.settings.value("main_dock_state_v2", None)
            state = self._to_qbytearray(raw)
            if state is None or state.isEmpty():
                return
            ok = self.restoreState(state, _DOCK_STATE_VERSION)
            if not ok:
                # Drop invalid payload and fall back to per-dock restore.
                self.settings.remove("main_dock_state_v2")
                return
            try:
                self.post_tab.mark_dock_layout_restored()
            except Exception:
                pass
        except Exception:
            pass

    def _restore_panel_layout_state(self) -> None:
        """Restore popup/artifact panel layout from the previous app session."""
        if self._use_pg_dockarea_pre_layout:
            self._is_restoring_panel_layout = True
            try:
                self._setup_section_popups()
                self._restore_pre_dockarea_layout_state()
            finally:
                self._is_restoring_panel_layout = False
            return

        self._is_restoring_panel_layout = True
        for key, dock in self._section_docks.items():
            base = f"pre_section_docks/{key}"
            try:
                visible = _to_bool(self.settings.value(f"{base}/visible", False), False)
                floating = _to_bool(self.settings.value(f"{base}/floating", True), True)
                area_val = self.settings.value(
                    f"{base}/area",
                    _dock_area_to_int(QtCore.Qt.DockWidgetArea.LeftDockWidgetArea, 1),
                )
                area = self._dock_area_from_settings(area_val, QtCore.Qt.DockWidgetArea.LeftDockWidgetArea)
                geom = self._to_qbytearray(self.settings.value(f"{base}/geometry", None))

                dock.blockSignals(True)
                if bool(floating):
                    dock.setFloating(True)
                else:
                    self.addDockWidget(area, dock)
                    dock.setFloating(False)

                if geom is not None and not geom.isEmpty():
                    dock.restoreGeometry(geom)
                    self._section_popup_initialized.add(key)

                if visible:
                    dock.show()
                    if dock.isFloating() and not self._is_popup_on_screen(dock):
                        self._position_section_popup(dock)
                    self._set_section_button_checked(key, True)
                    self._last_opened_section = key
                else:
                    dock.hide()
                    self._set_section_button_checked(key, False)
            except Exception:
                continue
            finally:
                try:
                    dock.blockSignals(False)
                except Exception:
                    pass

        if isinstance(self.art_dock, QtWidgets.QDockWidget):
            try:
                base = "pre_artifact_dock_state"
                visible = _to_bool(self.settings.value(f"{base}/visible", False), False)
                floating = _to_bool(self.settings.value(f"{base}/floating", False), False)
                area_val = self.settings.value(
                    f"{base}/area",
                    _dock_area_to_int(QtCore.Qt.DockWidgetArea.LeftDockWidgetArea, 1),
                )
                area = self._dock_area_from_settings(area_val, QtCore.Qt.DockWidgetArea.LeftDockWidgetArea)
                geom = self._to_qbytearray(self.settings.value(f"{base}/geometry", None))

                if bool(floating):
                    self.art_dock.setFloating(True)
                else:
                    self.addDockWidget(area, self.art_dock)
                    self.art_dock.setFloating(False)

                if geom is not None and not geom.isEmpty():
                    self.art_dock.restoreGeometry(geom)
                self.art_dock.setVisible(bool(visible))
            except Exception:
                pass

        self._sync_section_button_states_from_docks()
        self._restore_pre_tab_groups_fallback(self._load_pre_tab_groups_from_settings())
        self._is_restoring_panel_layout = False

    def _has_saved_pre_layout_state(self) -> bool:
        try:
            if self._use_pg_dockarea_pre_layout:
                if self.settings.contains(_PRE_DOCKAREA_STATE_KEY) or self.settings.contains(_PRE_DOCKAREA_VISIBLE_KEY):
                    return True
            if self.settings.contains(_PRE_DOCK_STATE_KEY):
                return True
            if self.settings.contains("pre_artifact_dock_state/visible"):
                return True
            keys = list(self._section_docks.keys())
            if self._use_pg_dockarea_pre_layout and self._pre_dockarea_docks:
                keys = list(self._pre_dockarea_docks.keys())
            for key in keys:
                if self.settings.contains(f"pre_section_docks/{key}/visible"):
                    return True
        except Exception:
            pass
        return False

    def _has_saved_post_layout_state(self) -> bool:
        try:
            if self.settings.contains(_POST_DOCK_STATE_KEY):
                return True
            self.post_tab.ensure_section_popups_initialized()
            for key in self.post_tab.get_section_popup_keys():
                if self.settings.contains(f"post_section_docks/{key}/visible"):
                    return True
        except Exception:
            pass
        return False

    # ---------------- Settings persistence ----------------

    def _restore_settings(self) -> None:
        last_dir = self.settings.value("last_open_dir", "", type=str)
        if last_dir and os.path.isdir(last_dir):
            self.file_panel.set_path_hint(last_dir)

        try:
            app_theme = self.settings.value("app_theme_mode", "dark", type=str)
        except Exception:
            app_theme = "dark"
        self._apply_app_theme(app_theme, persist=False)

        # restore params
        try:
            raw = self.settings.value("params_json", "", type=str)
            if raw:
                d = json.loads(raw)
                # One-time migration: ensure invert polarity defaults to off
                migrated = self.settings.value("invert_polarity_migrated", False, type=bool)
                if not migrated:
                    d["invert_polarity"] = False
                    self.settings.setValue("invert_polarity_migrated", True)
                    self.settings.setValue("params_json", json.dumps(d))
                p = ProcessingParams.from_dict(d)
                self.param_panel.set_params(p)
                self._update_plot_status(fs_target=float(p.target_fs_hz))
        except Exception:
            pass

        try:
            show_overlay = self.settings.value("artifact_overlay_visible", True, type=bool)
            self.param_panel.set_artifact_overlay_visible(bool(show_overlay))
            self.plots.set_artifact_overlay_visible(bool(show_overlay))
        except Exception:
            pass
        try:
            show_thresholds = self.settings.value("artifact_thresholds_visible", True, type=bool)
            self.plots.set_artifact_thresholds_visible(bool(show_thresholds))
        except Exception:
            pass
        try:
            auto_export = _to_bool(self.settings.value("auto_export_to_source_dir", False), False)
            self.param_panel.set_auto_export_enabled(auto_export)
            self._update_export_summary_label()
        except Exception:
            pass
        try:
            raw_export_selection = self.settings.value("pre_export_selection_json", "", type=str)
            if raw_export_selection:
                self.param_panel.set_export_selection(ExportSelection.from_dict(json.loads(raw_export_selection)))
                self._update_export_summary_label()
        except Exception:
            pass
        try:
            default_bg = "white" if self._app_theme_mode == "light" else "dark"
            plot_bg = self.settings.value("pre_plot_background", default_bg, type=str)
            if self._app_theme_mode == "dark" and self._normalize_pre_plot_background(plot_bg) == "white":
                plot_bg = "dark"
                self.settings.setValue("pre_plot_background", "dark")
        except Exception:
            plot_bg = "dark"
        try:
            plot_grid = _to_bool(self.settings.value("pre_plot_grid", True), True)
        except Exception:
            plot_grid = True
        self._apply_pre_plot_style(plot_bg, plot_grid, persist=False)

        if self._force_fixed_dock_layouts:
            # Fixed mode: always enforce deterministic defaults.
            try:
                self._set_pre_splitter_sizes(data_width=300, center_width=1200)
            except Exception:
                pass
            try:
                splitter_sizes = self.settings.value("pre_splitter_sizes", None)
                if splitter_sizes is None:
                    splitter_sizes = self.settings.value("splitter_sizes", None)
                if splitter_sizes and hasattr(splitter_sizes, "__len__"):
                    vals = [int(x) for x in splitter_sizes]
                    if self._use_pg_dockarea_pre_layout:
                        if len(vals) >= 3:
                            self._set_pre_splitter_sizes(vals[0], max(640, vals[1] + vals[2]))
                        elif len(vals) == 2:
                            self._set_pre_splitter_sizes(vals[0], vals[1])
                    elif len(vals) >= 3:
                        left = max(260, vals[0])
                        center = max(640, vals[1] + vals[2])
                        self._set_pre_splitter_sizes(left, center)
                    elif len(vals) == 2:
                        self._set_pre_splitter_sizes(vals[0], vals[1])
            except Exception:
                pass
            try:
                show_data = self.settings.value("pre_data_panel_visible", False, type=bool)
            except Exception:
                show_data = False
            self._set_data_panel_visible(bool(show_data), persist=False)

            self._apply_pre_fixed_layout()

            try:
                self.post_tab.ensure_section_popups_initialized()
                if hasattr(self.post_tab, "apply_fixed_default_layout"):
                    self.post_tab.apply_fixed_default_layout()
                # Keep post docks detached while Preprocessing is active at startup.
                self._hide_dock_widgets(self.getPostDockWidgets(), remove=True)
            except Exception:
                pass
        else:
            # restore splitter sizes (2-pane layout; migrate older layouts)
            try:
                splitter_sizes = self.settings.value("pre_splitter_sizes", None)
                if splitter_sizes is None:
                    splitter_sizes = self.settings.value("splitter_sizes", None)
                if splitter_sizes and hasattr(splitter_sizes, "__len__"):
                    vals = [int(x) for x in splitter_sizes]
                    if self._use_pg_dockarea_pre_layout:
                        if len(vals) >= 3:
                            self._set_pre_splitter_sizes(vals[0], max(640, vals[1] + vals[2]))
                        elif len(vals) == 2:
                            self._set_pre_splitter_sizes(vals[0], vals[1])
                    elif len(vals) >= 3:
                        # Migrate old 3-pane [left, center, right] into [left, center+right].
                        left = max(260, vals[0])
                        center = max(640, vals[1] + vals[2])
                        self._set_pre_splitter_sizes(left, center)
                    elif len(vals) == 2:
                        self._set_pre_splitter_sizes(vals[0], vals[1])
            except Exception:
                pass

            # restore data panel visibility
            try:
                show_data = self.settings.value("pre_data_panel_visible", True, type=bool)
                self._set_data_panel_visible(bool(show_data), persist=False)
            except Exception:
                pass

            # restore panel layout/disposition (floating popups + artifacts dock).
            self._restore_panel_layout_state()
            # Apply default preprocessing dock layout if no saved layout exists.
            self._apply_pre_default_layout_if_missing()
            # Apply saved preprocessing dock snapshot at startup.
            self._apply_pre_main_dock_snapshot_if_needed()

        # restore last selected main tab
        try:
            idx = self.settings.value("main_current_tab", 0, type=int)
            if isinstance(idx, int) and 0 <= idx < self.tabs.count():
                self.tabs.setCurrentIndex(idx)
        except Exception:
            pass

    def _save_settings(self) -> None:
        try:
            last_dir = self.file_panel.current_dir_hint()
            if last_dir:
                self.settings.setValue("last_open_dir", last_dir)
        except Exception:
            pass

        try:
            p = self.param_panel.get_params()
            self.settings.setValue("params_json", json.dumps(p.to_dict()))
        except Exception:
            pass

        try:
            self.settings.setValue("artifact_overlay_visible", bool(self.param_panel.artifact_overlay_visible()))
        except Exception:
            pass
        try:
            self.settings.setValue("artifact_thresholds_visible", bool(self.plots.artifact_thresholds_visible()))
        except Exception:
            pass
        try:
            self.settings.setValue("auto_export_to_source_dir", bool(self.param_panel.auto_export_enabled()))
        except Exception:
            pass
        try:
            self.settings.setValue(
                "pre_export_selection_json",
                json.dumps(self.param_panel.export_selection().to_dict()),
            )
        except Exception:
            pass
        try:
            self.settings.setValue("pre_plot_background", str(self.plots.plot_background_mode()))
            self.settings.setValue("pre_plot_grid", bool(self.plots.plot_grid_visible()))
        except Exception:
            pass
        try:
            self.settings.setValue("app_theme_mode", str(self._app_theme_mode))
        except Exception:
            pass

        try:
            self.settings.setValue("pre_data_panel_visible", bool(self.file_panel.isVisible()))
        except Exception:
            pass
        try:
            self.settings.setValue("main_current_tab", int(self.tabs.currentIndex()))
        except Exception:
            pass
        try:
            self.settings.sync()
        except Exception:
            pass

    def _set_pre_splitter_sizes(self, data_width: int, center_width: int) -> None:
        """Apply logical [data, center] sizes to the preprocessing splitter."""
        try:
            data = max(0, int(data_width))
            center = max(640, int(center_width))
            self.pre_splitter.setSizes([data, center])
        except Exception:
            pass

    def _save_splitter_sizes(self, *_args) -> None:
        """Save the current splitter sizes to settings."""
        try:
            if hasattr(self, "pre_splitter") and self.pre_splitter:
                sizes = self.pre_splitter.sizes()
                self.settings.setValue("pre_splitter_sizes", sizes)
                self.settings.setValue("splitter_sizes", sizes)
        except Exception:
            pass

    # ---------------- File loading ----------------

    def _load_recent_preprocessing_files(self) -> List[str]:
        raw = self.settings.value("recent_pre_files", "[]", type=str)
        try:
            data = json.loads(raw) if raw else []
        except Exception:
            data = []
        out: List[str] = []
        if isinstance(data, list):
            for item in data:
                p = str(item or "").strip()
                if p:
                    out.append(p)
        return out

    def _save_recent_preprocessing_files(self, paths: List[str]) -> None:
        try:
            self.settings.setValue("recent_pre_files", json.dumps(paths))
        except Exception:
            pass

    def _push_recent_preprocessing_files(self, paths: List[str], max_items: int = 15) -> None:
        if not paths:
            return
        existing = self._load_recent_preprocessing_files()
        merged: List[str] = []
        for p in paths:
            sp = str(p or "").strip()
            if not sp:
                continue
            if sp in merged:
                continue
            merged.append(sp)
        for p in existing:
            if p not in merged:
                merged.append(p)
        self._save_recent_preprocessing_files(merged[:max_items])

    def _refresh_recent_preprocessing_menu(self) -> None:
        if not hasattr(self, "menu_workflow_load_recent"):
            return
        menu = self.menu_workflow_load_recent
        menu.clear()
        recent = self._load_recent_preprocessing_files()
        if not recent:
            act_empty = menu.addAction("(No recent files)")
            act_empty.setEnabled(False)
            return

        missing: List[str] = []
        for path in recent:
            label = os.path.basename(path) or path
            if not os.path.isfile(path):
                label = f"{label} (missing)"
            act = menu.addAction(label)
            act.setToolTip(path)
            act.setEnabled(os.path.isfile(path))
            if os.path.isfile(path):
                act.triggered.connect(lambda _checked=False, p=path: self._add_files([p]))
            else:
                missing.append(path)
        menu.addSeparator()
        act_clear = menu.addAction("Clear recent")
        act_clear.triggered.connect(lambda: self._save_recent_preprocessing_files([]))
        if missing:
            act_prune = menu.addAction("Remove missing")
            act_prune.triggered.connect(self._prune_recent_preprocessing_files)

    def _prune_recent_preprocessing_files(self) -> None:
        recent = self._load_recent_preprocessing_files()
        kept = [p for p in recent if os.path.isfile(p)]
        self._save_recent_preprocessing_files(kept)

    # ---------------- Preprocessing projects ----------------

    def _preprocessing_project_state_exists(self) -> bool:
        return bool(
            self._loaded_files
            or self._manual_regions_by_key
            or self._manual_exclude_by_key
            or self._metadata_by_key
            or self._cutout_regions_by_key
            or self._sections_by_key
        )

    def _confirm_discard_preprocessing_project(self, title: str) -> bool:
        if not self._preprocessing_project_state_exists():
            return True
        reply = QtWidgets.QMessageBox.question(
            self,
            title,
            "Discard the current preprocessing project state?",
            QtWidgets.QMessageBox.StandardButton.Yes | QtWidgets.QMessageBox.StandardButton.No,
            QtWidgets.QMessageBox.StandardButton.No,
        )
        return reply == QtWidgets.QMessageBox.StandardButton.Yes

    def _clear_preprocessing_project_state(self) -> None:
        try:
            self._preview_timer.stop()
        except Exception:
            pass
        self._job_counter += 1
        self._latest_job_id = self._job_counter
        self._preview_preserve_view_by_job.clear()
        self._loaded_files.clear()
        self._current_path = None
        self._current_channel = None
        self._current_trigger = None
        self._manual_regions_by_key.clear()
        self._manual_exclude_by_key.clear()
        self._auto_regions_by_key.clear()
        self._metadata_by_key.clear()
        self._cutout_regions_by_key.clear()
        self._sections_by_key.clear()
        self._pending_box_region_by_key.clear()
        self._last_processed.clear()
        self._csv_channel_mapping_session = None
        self._csv_mappings_by_path.clear()

        self.file_panel.list_files.clear()
        self.file_panel.set_available_channels([])
        self.file_panel.set_available_triggers([])
        self.param_panel.set_available_export_channels([])
        self.param_panel.set_available_export_triggers([])
        for ed in (self.file_panel.edit_time_start, self.file_panel.edit_time_end):
            ed.blockSignals(True)
            try:
                ed.clear()
            finally:
                ed.blockSignals(False)
        self.artifact_panel.set_regions([])
        self.artifact_panel.set_auto_regions([])
        self.plots.set_title("No file loaded")
        self.plots.set_log("")
        self.plots.show_raw()
        self._update_plot_status()
        self.post_tab.set_current_source_label("", "")

    def _new_preprocessing_project(self) -> None:
        if not self._confirm_discard_preprocessing_project("New preprocessing project"):
            return
        self._clear_preprocessing_project_state()
        self._pre_project_path = None
        try:
            self.post_tab.reset_for_new_preprocessing_project()
        except Exception:
            pass
        self._reset_pre_history_snapshot()
        self._show_status_message("Started a new preprocessing project.", 5000)

    def _keyed_regions_to_project(
        self,
        mapping: Dict[Tuple[str, str], List[Tuple[float, float]]],
    ) -> List[Dict[str, object]]:
        out: List[Dict[str, object]] = []
        for (path, channel), regions in mapping.items():
            clean_regions = []
            for a, b in regions or []:
                try:
                    clean_regions.append([float(a), float(b)])
                except Exception:
                    continue
            if clean_regions:
                out.append({"path": path, "channel": channel, "regions": clean_regions})
        return out

    def _project_to_keyed_regions(self, data: object) -> Dict[Tuple[str, str], List[Tuple[float, float]]]:
        out: Dict[Tuple[str, str], List[Tuple[float, float]]] = {}
        if not isinstance(data, list):
            return out
        for entry in data:
            if not isinstance(entry, dict):
                continue
            path = str(entry.get("path") or "").strip()
            channel = str(entry.get("channel") or "").strip()
            if not path or not channel:
                continue
            regions: List[Tuple[float, float]] = []
            for item in entry.get("regions") or []:
                try:
                    a, b = item
                    regions.append((float(a), float(b)))
                except Exception:
                    continue
            if regions:
                regions.sort(key=lambda x: x[0])
                out[(path, channel)] = regions
        return out

    def _keyed_dict_to_project(self, mapping: Dict[Tuple[str, str], Dict[str, str]]) -> List[Dict[str, object]]:
        out: List[Dict[str, object]] = []
        for (path, channel), value in mapping.items():
            if isinstance(value, dict) and value:
                out.append({"path": path, "channel": channel, "value": dict(value)})
        return out

    def _project_to_keyed_dict(self, data: object) -> Dict[Tuple[str, str], Dict[str, str]]:
        out: Dict[Tuple[str, str], Dict[str, str]] = {}
        if not isinstance(data, list):
            return out
        for entry in data:
            if not isinstance(entry, dict):
                continue
            path = str(entry.get("path") or "").strip()
            channel = str(entry.get("channel") or "").strip()
            value = entry.get("value")
            if path and channel and isinstance(value, dict):
                out[(path, channel)] = {str(k): str(v) for k, v in value.items()}
        return out

    def _sections_to_project(self) -> List[Dict[str, object]]:
        out: List[Dict[str, object]] = []
        for (path, channel), sections in self._sections_by_key.items():
            if not sections:
                continue
            try:
                clean_sections = json.loads(json.dumps(sections))
            except Exception:
                clean_sections = []
            if clean_sections:
                out.append({"path": path, "channel": channel, "sections": clean_sections})
        return out

    def _project_to_sections(self, data: object) -> Dict[Tuple[str, str], List[Dict[str, object]]]:
        out: Dict[Tuple[str, str], List[Dict[str, object]]] = {}
        if not isinstance(data, list):
            return out
        for entry in data:
            if not isinstance(entry, dict):
                continue
            path = str(entry.get("path") or "").strip()
            channel = str(entry.get("channel") or "").strip()
            sections = entry.get("sections")
            if path and channel and isinstance(sections, list):
                out[(path, channel)] = [s for s in sections if isinstance(s, dict)]
        return out

    def _preprocessing_config_payload(self) -> Dict[str, object]:
        params = self.param_panel.get_params()
        return {
            "artifact_detection_enabled": bool(self.param_panel.cb_artifact.isChecked()),
            "artifact_overlay_visible": bool(self.param_panel.cb_show_artifact_overlay.isChecked()),
            "filtering_enabled": bool(self.param_panel.cb_filtering.isChecked()),
            "parameters": params.to_dict(),
            "ui_state": self._export_preprocessing_ui_state_for_config(),
        }

    def _apply_preprocessing_config_payload(self, config: object) -> None:
        if not isinstance(config, dict):
            return
        try:
            params = config.get("parameters")
            if isinstance(params, dict):
                self.param_panel.set_params(ProcessingParams.from_dict(params))
            if "artifact_detection_enabled" in config:
                self.param_panel.cb_artifact.setChecked(bool(config.get("artifact_detection_enabled")))
            if "artifact_overlay_visible" in config:
                visible = bool(config.get("artifact_overlay_visible"))
                self.param_panel.cb_show_artifact_overlay.setChecked(visible)
                self.param_panel.set_artifact_overlay_visible(visible)
                self.plots.set_artifact_overlay_visible(visible)
            if "filtering_enabled" in config:
                self.param_panel.cb_filtering.setChecked(bool(config.get("filtering_enabled")))
            ui_state = config.get("ui_state")
            if isinstance(ui_state, dict):
                self._import_preprocessing_ui_state_from_config(ui_state)
        except Exception:
            _LOG.exception("Failed to apply preprocessing project config")

    def _collect_preprocessing_project_payload(self) -> Dict[str, object]:
        selected_paths = self._selected_paths()
        start_s, end_s = self._time_window_bounds()
        return {
            "project_type": _PRE_PROJECT_TYPE,
            "project_version": _PRE_PROJECT_VERSION,
            "source_paths": self.file_panel.all_paths(),
            "selected_paths": selected_paths,
            "current_path": self._current_path or "",
            "current_channel": self._current_channel or "",
            "current_trigger": self._current_trigger or "",
            "time_window": {"start_s": start_s, "end_s": end_s},
            "preprocessing_config": self._preprocessing_config_payload(),
            "manual_regions": self._keyed_regions_to_project(self._manual_regions_by_key),
            "manual_exclude_regions": self._keyed_regions_to_project(self._manual_exclude_by_key),
            "auto_regions": self._keyed_regions_to_project(self._auto_regions_by_key),
            "metadata": self._keyed_dict_to_project(self._metadata_by_key),
            "cutout_regions": self._keyed_regions_to_project(self._cutout_regions_by_key),
            "sections": self._sections_to_project(),
            "csv_mapping_session": dict(self._csv_channel_mapping_session or {}),
            "csv_mappings_by_path": [
                {"path": path, "mapping": dict(mapping)}
                for path, mapping in self._csv_mappings_by_path.items()
                if path and isinstance(mapping, dict)
            ],
        }

    def _save_preprocessing_project_file(self) -> None:
        start_dir = (
            os.path.dirname(self._pre_project_path)
            if self._pre_project_path
            else (self.file_panel.current_dir_hint() or self.settings.value("last_open_dir", "", type=str) or os.getcwd())
        )
        default_name = os.path.basename(self._pre_project_path) if self._pre_project_path else "pyber_preprocessing_project.json"
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self,
            "Save preprocessing project",
            os.path.join(start_dir, default_name),
            "pyBer preprocessing project (*.json)",
        )
        if not path:
            return
        if not path.lower().endswith(".json"):
            path = f"{path}.json"
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(self._collect_preprocessing_project_payload(), f, indent=2)
        except Exception as exc:
            QtWidgets.QMessageBox.warning(self, "Save project", f"Could not save preprocessing project:\n{exc}")
            return
        self._pre_project_path = path
        self._show_status_message(f"Preprocessing project saved: {os.path.basename(path)}", 5000)

    def _open_preprocessing_project_file(self) -> None:
        start_dir = self.file_panel.current_dir_hint() or self.settings.value("last_open_dir", "", type=str) or os.getcwd()
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Open preprocessing project",
            start_dir,
            "pyBer preprocessing project (*.json);;All files (*.*)",
        )
        if not path:
            return
        self._load_preprocessing_project_from_path(path)

    def _load_preprocessing_project_from_path(self, path: str) -> None:
        if not self._confirm_discard_preprocessing_project("Open preprocessing project"):
            return
        try:
            with open(path, "r", encoding="utf-8") as f:
                payload = json.load(f)
        except Exception as exc:
            QtWidgets.QMessageBox.warning(self, "Open project", f"Could not read preprocessing project:\n{exc}")
            return
        if not isinstance(payload, dict) or payload.get("project_type") != _PRE_PROJECT_TYPE:
            QtWidgets.QMessageBox.warning(self, "Open project", "This file is not a pyBer preprocessing project.")
            return

        self._clear_preprocessing_project_state()
        self._pre_project_path = path
        try:
            self.post_tab.reset_for_new_preprocessing_project()
        except Exception:
            pass
        self._apply_preprocessing_config_payload(payload.get("preprocessing_config"))

        session_mapping = payload.get("csv_mapping_session")
        if isinstance(session_mapping, dict):
            self._csv_channel_mapping_session = {str(k): str(v) for k, v in session_mapping.items()}
        for entry in payload.get("csv_mappings_by_path") or []:
            if not isinstance(entry, dict):
                continue
            src_path = str(entry.get("path") or "").strip()
            mapping = entry.get("mapping")
            if src_path and isinstance(mapping, dict):
                self._csv_mappings_by_path[src_path] = {str(k): str(v) for k, v in mapping.items()}

        source_paths = [str(p) for p in payload.get("source_paths") or [] if str(p or "").strip()]
        existing_paths = [p for p in source_paths if os.path.isfile(p)]
        missing_paths = [p for p in source_paths if p not in existing_paths]
        if existing_paths:
            self._add_files(existing_paths, select_after=False)

        self._manual_regions_by_key = self._project_to_keyed_regions(payload.get("manual_regions"))
        self._manual_exclude_by_key = self._project_to_keyed_regions(payload.get("manual_exclude_regions"))
        self._auto_regions_by_key = self._project_to_keyed_regions(payload.get("auto_regions"))
        self._metadata_by_key = self._project_to_keyed_dict(payload.get("metadata"))
        self._cutout_regions_by_key = self._project_to_keyed_regions(payload.get("cutout_regions"))
        self._sections_by_key = self._project_to_sections(payload.get("sections"))

        tw = payload.get("time_window") if isinstance(payload.get("time_window"), dict) else {}
        for ed, value in (
            (self.file_panel.edit_time_start, tw.get("start_s")),
            (self.file_panel.edit_time_end, tw.get("end_s")),
        ):
            ed.blockSignals(True)
            try:
                ed.setText("" if value is None else f"{float(value):.6g}")
            except Exception:
                ed.setText("")
            finally:
                ed.blockSignals(False)

        self._current_path = str(payload.get("current_path") or "") or None
        self._current_channel = str(payload.get("current_channel") or "") or None
        self._current_trigger = str(payload.get("current_trigger") or "") or None
        selected_paths = [str(p) for p in payload.get("selected_paths") or [] if str(p or "").strip()]
        self._restore_file_selection(selected_paths, self._current_path)
        self._push_recent_preprocessing_files(existing_paths)
        self._on_file_selection_changed()

        if missing_paths:
            QtWidgets.QMessageBox.warning(
                self,
                "Open project",
                "Some linked input files are missing and were skipped:\n" + "\n".join(missing_paths[:12]),
            )
        self._reset_pre_history_snapshot()
        self._show_status_message(f"Preprocessing project loaded: {os.path.basename(path)}", 5000)

    def _restore_file_selection(self, selected_paths: List[str], current_path: Optional[str]) -> None:
        selected = set(selected_paths or [])
        if current_path:
            selected.add(current_path)
        list_widget = self.file_panel.list_files
        list_widget.blockSignals(True)
        try:
            target_row = -1
            for i in range(list_widget.count()):
                item = list_widget.item(i)
                if item is None:
                    continue
                path = str(item.data(QtCore.Qt.ItemDataRole.UserRole) or "")
                item.setSelected(path in selected)
                if current_path and path == current_path:
                    target_row = i
            if target_row >= 0:
                list_widget.setCurrentRow(target_row)
            elif list_widget.count() and not selected:
                list_widget.setCurrentRow(0)
                item0 = list_widget.item(0)
                if item0 is not None:
                    item0.setSelected(True)
        finally:
            list_widget.blockSignals(False)

    # ---------------- Raw CSV preprocessing import ----------------

    def _normalize_csv_column_name(self, value: object) -> str:
        return "".join(ch.lower() for ch in str(value or "") if ch.isalnum())

    def _is_csv_time_column(self, value: object) -> bool:
        norm = self._normalize_csv_column_name(value)
        return norm in {"time", "t", "timestamp", "times", "timesec", "times", "timems"} or "timestamp" in norm

    def _parse_csv_float(self, value: object) -> float:
        text = str(value or "").strip()
        if not text or text.lower() in {"nan", "none", "null", "na"}:
            return np.nan
        try:
            return float(text)
        except Exception:
            pass
        try:
            return float(text.replace(" ", "").replace(",", "."))
        except Exception:
            pass
        return coerce_time_value(text)

    def _clean_csv_row(self, row: List[str]) -> List[str]:
        out = [str(cell or "").strip() for cell in row]
        while out and not out[-1]:
            out.pop()
        return out

    def _read_csv_rows(self, path: str) -> List[List[str]]:
        import csv

        last_error: Optional[Exception] = None
        for encoding in ("utf-8-sig", "utf-8", "cp1252"):
            try:
                with open(path, "r", newline="", encoding=encoding) as f:
                    return [self._clean_csv_row(row) for row in csv.reader(f)]
            except UnicodeDecodeError as exc:
                last_error = exc
                continue
        if last_error is not None:
            raise last_error
        return []

    def _find_raw_csv_table(self, rows: List[List[str]]) -> Tuple[List[str], List[List[str]]]:
        cleaned = [row for row in (self._clean_csv_row(r) for r in rows) if row and any(cell for cell in row)]
        for idx, row in enumerate(cleaned):
            if len(row) < 2:
                continue
            if any(self._is_csv_time_column(cell) for cell in row):
                headers = [h.strip() or f"Column {i + 1}" for i, h in enumerate(row)]
                return headers, cleaned[idx + 1 :]

        # Fallback for CSVs without a canonical time header: find the first row whose
        # following line looks numeric in at least two columns.
        for idx, row in enumerate(cleaned[:-1]):
            if len(row) < 2:
                continue
            next_row = cleaned[idx + 1]
            numeric_count = 0
            for col_idx in range(min(len(row), len(next_row))):
                if np.isfinite(self._parse_csv_float(next_row[col_idx])):
                    numeric_count += 1
            if numeric_count >= 2:
                headers = [h.strip() or f"Column {i + 1}" for i, h in enumerate(row)]
                return headers, cleaned[idx + 1 :]

        raise ValueError("Could not find a CSV header row with a time column.")

    def _csv_numeric_headers(self, headers: List[str], rows: List[List[str]]) -> List[str]:
        out: List[str] = []
        sample = rows[: min(len(rows), 1000)]
        min_count = 1 if len(sample) < 10 else 3
        for idx, name in enumerate(headers):
            if self._is_csv_time_column(name):
                continue
            count = 0
            for row in sample:
                if idx < len(row) and np.isfinite(self._parse_csv_float(row[idx])):
                    count += 1
            if count >= min_count:
                out.append(name)
        return out

    def _resolve_csv_column_name(self, headers: List[str], wanted: object) -> str:
        text = str(wanted or "").strip()
        if not text:
            return ""
        for h in headers:
            if h == text:
                return h
        for h in headers:
            if h.lower() == text.lower():
                return h
        norm = self._normalize_csv_column_name(text)
        for h in headers:
            if self._normalize_csv_column_name(h) == norm:
                return h
        return ""

    def _sanitize_csv_mapping_for_headers(
        self,
        mapping: object,
        headers: List[str],
        *,
        require_all: bool = True,
    ) -> Optional[Dict[str, str]]:
        if not isinstance(mapping, dict):
            return None

        def _resolve(name_key: str, index_key: str) -> str:
            col = self._resolve_csv_column_name(headers, mapping.get(name_key))
            if col:
                return col
            try:
                idx = int(mapping.get(index_key))
            except Exception:
                idx = -1
            if 0 <= idx < len(headers):
                return headers[idx]
            return ""

        time_col = _resolve("time", "time_index")
        raw1 = _resolve("raw1", "raw1_index")
        ref = _resolve("reference", "reference_index")
        if require_all and (not time_col or not raw1 or not ref):
            return None
        raw2 = _resolve("raw2", "raw2_index")
        trigger = _resolve("trigger", "trigger_index")
        unit = str(mapping.get("time_unit") or "Auto").strip()
        if unit.lower().startswith("milli"):
            unit = "Milliseconds"
        elif unit.lower().startswith("sec"):
            unit = "Seconds"
        else:
            unit = "Auto"
        return {
            "time": time_col,
            "time_unit": unit,
            "raw1": raw1,
            "raw2": raw2,
            "reference": ref,
            "trigger": trigger,
            "time_index": str(self._csv_column_index(headers, time_col)),
            "raw1_index": str(self._csv_column_index(headers, raw1)),
            "raw2_index": str(self._csv_column_index(headers, raw2)),
            "reference_index": str(self._csv_column_index(headers, ref)),
            "trigger_index": str(self._csv_column_index(headers, trigger)),
        }

    def _infer_csv_mapping_defaults(self, headers: List[str], numeric_headers: List[str]) -> Dict[str, str]:
        time_col = next((h for h in headers if self._is_csv_time_column(h)), headers[0] if headers else "")

        def _has_any(name: str, terms: Tuple[str, ...]) -> bool:
            norm = self._normalize_csv_column_name(name)
            return any(term in norm for term in terms)

        candidates = [h for h in numeric_headers if h != time_col]
        ref = next((h for h in candidates if _has_any(h, ("410", "405", "isob", "isos", "ref"))), "")
        raw_priority = [h for h in candidates if h != ref and _has_any(h, ("470", "465", "signal", "sig"))]
        raw_rest = [h for h in candidates if h != ref and h not in raw_priority and not _has_any(h, ("event", "dio", "ttl", "digital"))]
        raw_candidates = raw_priority + raw_rest
        if not ref and len(candidates) >= 2:
            ref = candidates[1] if raw_candidates and candidates[1] != raw_candidates[0] else candidates[0]
        raw1 = raw_candidates[0] if raw_candidates else next((h for h in candidates if h != ref), "")
        raw2 = raw_candidates[1] if len(raw_candidates) > 1 else ""
        trigger = next((h for h in candidates if _has_any(h, ("event", "dio", "ttl", "digital"))), "")
        return {
            "time": time_col,
            "time_unit": "Auto",
            "raw1": raw1,
            "raw2": raw2,
            "reference": ref,
            "trigger": trigger,
        }

    def _csv_mapping_for_file(
        self,
        path: str,
        headers: List[str],
        numeric_headers: List[str],
    ) -> Optional[Dict[str, str]]:
        for candidate in (
            self._csv_mappings_by_path.get(path),
            self._csv_channel_mapping_session,
        ):
            resolved = self._sanitize_csv_mapping_for_headers(candidate, headers)
            if resolved is not None:
                self._csv_mappings_by_path[path] = dict(resolved)
                if self._csv_channel_mapping_session is None:
                    self._csv_channel_mapping_session = dict(resolved)
                return resolved

        defaults = self._infer_csv_mapping_defaults(headers, numeric_headers)
        partial = self._sanitize_csv_mapping_for_headers(self._csv_channel_mapping_session, headers, require_all=False)
        if partial:
            for key, value in partial.items():
                if value:
                    defaults[key] = value
        if len(numeric_headers) < 2:
            raise ValueError("CSV must contain at least two numeric columns for raw signal and isobestic/reference.")

        dlg = CsvChannelMappingDialog(headers, numeric_headers, defaults, self)
        if dlg.exec() != QtWidgets.QDialog.DialogCode.Accepted:
            return None
        mapping = self._sanitize_csv_mapping_for_headers(dlg.mapping(), headers)
        if mapping is None:
            raise ValueError("Invalid CSV channel mapping.")
        self._csv_channel_mapping_session = dict(mapping)
        self._csv_mappings_by_path[path] = dict(mapping)
        return mapping

    def _csv_column_index(self, headers: List[str], column: str) -> int:
        try:
            return headers.index(column)
        except ValueError:
            return -1

    def _csv_time_seconds(self, time: np.ndarray, time_col: str, unit: str) -> np.ndarray:
        t = np.asarray(time, float)
        unit_l = str(unit or "Auto").strip().lower()
        if unit_l.startswith("milli"):
            return t / 1000.0
        if unit_l.startswith("sec"):
            return t
        finite = t[np.isfinite(t)]
        if finite.size > 2:
            dt = float(np.nanmedian(np.abs(np.diff(finite))))
        else:
            dt = np.nan
        norm = self._normalize_csv_column_name(time_col)
        if "ms" in norm or "millisecond" in norm or (np.isfinite(dt) and dt >= 10.0):
            return t / 1000.0
        return t

    def _load_raw_csv_as_pre_file(self, path: str) -> Optional[LoadedDoricFile]:
        rwd_loaded = load_rwd_csv(path)
        if rwd_loaded is not None:
            return rwd_loaded

        rows = self._read_csv_rows(path)
        if not rows:
            raise ValueError("CSV file is empty.")
        headers, data_rows = self._find_raw_csv_table(rows)
        if not data_rows:
            raise ValueError("CSV file has no data rows.")
        numeric_headers = self._csv_numeric_headers(headers, data_rows)
        mapping = self._csv_mapping_for_file(path, headers, numeric_headers)
        if mapping is None:
            return None

        idx_time = self._csv_column_index(headers, mapping["time"])
        idx_raw1 = self._csv_column_index(headers, mapping["raw1"])
        idx_raw2 = self._csv_column_index(headers, mapping.get("raw2", ""))
        idx_ref = self._csv_column_index(headers, mapping["reference"])
        idx_trig = self._csv_column_index(headers, mapping.get("trigger", ""))
        if min(idx_time, idx_raw1, idx_ref) < 0:
            raise ValueError("CSV channel mapping refers to a missing column.")

        time_vals: List[float] = []
        raw1_vals: List[float] = []
        raw2_vals: List[float] = []
        ref_vals: List[float] = []
        trig_vals: List[float] = []
        has_raw2 = idx_raw2 >= 0
        has_trig = idx_trig >= 0

        for row in data_rows:
            tval = self._parse_csv_float(row[idx_time] if idx_time < len(row) else "")
            if not np.isfinite(tval):
                continue
            time_vals.append(tval)
            raw1_vals.append(self._parse_csv_float(row[idx_raw1] if idx_raw1 < len(row) else ""))
            ref_vals.append(self._parse_csv_float(row[idx_ref] if idx_ref < len(row) else ""))
            if has_raw2:
                raw2_vals.append(self._parse_csv_float(row[idx_raw2] if idx_raw2 < len(row) else ""))
            if has_trig:
                trig_vals.append(self._parse_csv_float(row[idx_trig] if idx_trig < len(row) else ""))

        if len(time_vals) < 2:
            raise ValueError("CSV file has fewer than two valid time samples.")

        t = self._csv_time_seconds(np.asarray(time_vals, float), mapping["time"], mapping.get("time_unit", "Auto"))
        raw1 = np.asarray(raw1_vals, float)
        ref = np.asarray(ref_vals, float)
        if not np.isfinite(raw1).any():
            raise ValueError(f"Raw signal column '{mapping['raw1']}' has no numeric values.")
        if not np.isfinite(ref).any():
            raise ValueError(f"Isobestic/reference column '{mapping['reference']}' has no numeric values.")

        order = np.argsort(t)
        if not np.all(order == np.arange(t.size)):
            t = t[order]
            raw1 = raw1[order]
            ref = ref[order]
            if has_raw2:
                raw2_vals = list(np.asarray(raw2_vals, float)[order])
            if has_trig:
                trig_vals = list(np.asarray(trig_vals, float)[order])

        channels = [mapping["raw1"]]
        time_by = {mapping["raw1"]: t.copy()}
        signal_by = {mapping["raw1"]: raw1.copy()}
        reference_by = {mapping["raw1"]: ref.copy()}

        if has_raw2:
            raw2 = np.asarray(raw2_vals, float)
            if np.isfinite(raw2).any():
                channels.append(mapping["raw2"])
                time_by[mapping["raw2"]] = t.copy()
                signal_by[mapping["raw2"]] = raw2.copy()
                reference_by[mapping["raw2"]] = ref.copy()

        trigger_by: Dict[str, np.ndarray] = {}
        trigger_time_by: Dict[str, np.ndarray] = {}
        digital_time: Optional[np.ndarray] = None
        if has_trig:
            trig = np.asarray(trig_vals, float)
            if trig.size == t.size and np.isfinite(trig).any():
                trig_name = mapping.get("trigger", "") or "Events"
                digital_time = t.copy()
                trigger_by[trig_name] = trig.copy()
                trigger_time_by[trig_name] = t.copy()

        return LoadedDoricFile(
            path=path,
            channels=channels,
            time_by_channel=time_by,
            signal_by_channel=signal_by,
            reference_by_channel=reference_by,
            digital_time=digital_time,
            digital_by_name={k: v.copy() for k, v in trigger_by.items()},
            trigger_time_by_name=trigger_time_by,
            trigger_by_name=trigger_by,
        )

    def _open_files_dialog(self) -> None:
        start_dir = self.file_panel.current_dir_hint() or self.settings.value("last_open_dir", "", type=str) or os.getcwd()
        paths, _ = QtWidgets.QFileDialog.getOpenFileNames(
            self,
            "Open files",
            start_dir,
            "Data files (*.doric *.h5 *.hdf5 *.csv);;Doric/HDF5 files (*.doric *.h5 *.hdf5);;CSV files (*.csv);;All files (*.*)",
        )
        if not paths:
            return

        self.settings.setValue("last_open_dir", os.path.dirname(paths[0]))
        self._push_recent_preprocessing_files(paths)
        self._add_files(paths)

    def _open_folder_dialog(self) -> None:
        start_dir = self.file_panel.current_dir_hint() or self.settings.value("last_open_dir", "", type=str) or os.getcwd()
        folder = QtWidgets.QFileDialog.getExistingDirectory(self, "Add folder with data files", start_dir)
        if not folder:
            return
        self.settings.setValue("last_open_dir", folder)

        paths = _discover_preprocessing_data_files(folder, recursive=True)
        if not paths:
            self._show_status_message("No raw preprocessing files found in that folder or its subfolders.", 6000)
            return
        self._push_recent_preprocessing_files(paths)
        self._add_files(paths)

    def _expand_dropped_url_paths(self, urls: List[QtCore.QUrl]) -> List[str]:
        """Resolve a list of dropped URLs into supported file paths.
        Folders are scanned recursively, matching the Add-folder dialog. Files
        are passed through if their extension is supported. Duplicates are
        dropped, order is preserved."""
        paths: List[str] = []
        seen: set[str] = set()
        for url in urls:
            if not url.isLocalFile():
                continue
            local = url.toLocalFile()
            if not local:
                continue
            if os.path.isdir(local):
                discovered = _discover_preprocessing_data_files(local, recursive=True)
                for full in discovered:
                    key = os.path.normcase(os.path.abspath(full))
                    if key in seen:
                        continue
                    seen.add(key)
                    paths.append(full)
            elif os.path.isfile(local):
                ext = os.path.splitext(local)[1].lower()
                include_generic_csv = ext == ".csv" and not is_rwd_events_csv(local)
                if not _is_supported_preprocessing_folder_file(local, include_generic_csv=include_generic_csv):
                    continue
                key = os.path.normcase(os.path.abspath(local))
                if key in seen:
                    continue
                seen.add(key)
                paths.append(local)
        return paths

    def _add_files(self, paths: List[str], select_after: bool = True) -> None:
        for p in paths:
            if p in self._loaded_files:
                continue
            ext = os.path.splitext(p)[1].lower()
            if ext == ".csv":
                if is_rwd_events_csv(p):
                    self._show_status_message(
                        f"Skipped RWD Events.csv: load the matching Fluorescence.csv file instead.",
                        6000,
                    )
                    continue
                try:
                    loaded_from_csv = self._load_raw_csv_as_pre_file(p)
                except Exception as e:
                    QtWidgets.QMessageBox.critical(self, "Load error", f"Failed to load CSV:\n{p}\n\n{e}")
                    continue
                if loaded_from_csv is None:
                    continue
                self._loaded_files[p] = loaded_from_csv
                self.file_panel.add_file(p)
                self._show_status_message(f"Loaded CSV: {os.path.basename(p)}", 5000)
                continue
            try:
                doric = self.processor.load_file(p)
                self._loaded_files[p] = doric
                self.file_panel.add_file(p)
                self._show_status_message(f"Loaded: {os.path.basename(p)}", 5000)
            except Exception as e:
                loaded_from_processed: Optional[LoadedDoricFile] = None
                if ext in (".h5", ".hdf5"):
                    loaded_from_processed = self._load_processed_h5_as_pre_file(p)
                if loaded_from_processed is not None:
                    self._loaded_files[p] = loaded_from_processed
                    self.file_panel.add_file(p)
                    self._show_status_message(
                        f"Loaded processed H5 as preprocessing source: {os.path.basename(p)}",
                        6000,
                    )
                    continue
                QtWidgets.QMessageBox.critical(self, "Load error", f"Failed to load:\n{p}\n\n{e}")

        self._push_recent_preprocessing_files(paths)

        # set current selection -> triggers preview
        if select_after:
            self._on_file_selection_changed()

    # ---------------- Current selection ----------------

    def _selected_paths(self) -> List[str]:
        return self.file_panel.selected_paths()

    def _current_key(self) -> Optional[Tuple[str, str]]:
        if not self._current_path or not self._current_channel:
            return None
        return (self._current_path, self._current_channel)

    def _focus_data_browser(self) -> None:
        if not self.file_panel.isVisible():
            self._set_data_panel_visible(True)
        self.file_panel.setFocus()
        self.file_panel.list_files.setFocus()

    def _hide_preprocessing_popups_for_tab_switch(self) -> None:
        if self._use_pg_dockarea_pre_layout:
            # DockArea lives inside the Preprocessing tab widget; avoid costly hide/remove
            # churn during main-tab switches for smoother transitions.
            return
        if self._pre_popups_hidden_by_tab_switch:
            # Re-apply hide in case late dock events re-show a preprocessing dock.
            self._enforce_preprocessing_popups_hidden()
            return
        host = self
        self._pre_section_visibility_before_tab_switch = {
            key: bool(dock.isVisible()) for key, dock in self._section_docks.items()
        }
        self._pre_section_state_before_tab_switch = {}
        for key, dock in self._section_docks.items():
            area = _dock_area_to_int(host.dockWidgetArea(dock), _dock_area_to_int(QtCore.Qt.DockWidgetArea.LeftDockWidgetArea, 1))
            self._pre_section_state_before_tab_switch[key] = {
                "visible": bool(dock.isVisible()),
                "floating": bool(dock.isFloating()),
                "area": area,
                "geometry": dock.saveGeometry(),
            }
        self._pre_artifact_visible_before_tab_switch = bool(self.art_dock.isVisible())
        self._pre_artifact_state_before_tab_switch = {
            "visible": bool(self.art_dock.isVisible()),
            "floating": bool(self.art_dock.isFloating()),
            "area": _dock_area_to_int(host.dockWidgetArea(self.art_dock), _dock_area_to_int(QtCore.Qt.DockWidgetArea.LeftDockWidgetArea, 1)),
            "geometry": self.art_dock.saveGeometry(),
        }
        self._pre_advanced_visible_before_tab_switch = bool(
            self._advanced_dialog is not None and self._advanced_dialog.isVisible()
        )
        self._pre_tab_groups_before_tab_switch = self._capture_pre_tab_groups_state()
        self._pre_main_dock_state_before_tab_switch = self.captureDockSnapshotForTab("pre")
        self._store_pre_main_dock_snapshot()
        # Mark switch-hide state before any dock visibility changes so asynchronous
        # visibility signals cannot persist temporary hidden defaults.
        self._pre_popups_hidden_by_tab_switch = True
        # Persist cached state now; dock hide/remove operations below are temporary.
        self._persist_hidden_preprocessing_layout_state()

        self._suspend_panel_layout_persistence = True
        try:
            for key in self._section_docks.keys():
                self._set_section_button_checked(key, False)
            self._hide_dock_widgets(self.getPreDockWidgets(), remove=True)
            if self._advanced_dialog is not None:
                self._advanced_dialog.hide()
        finally:
            self._suspend_panel_layout_persistence = False

    def _enforce_preprocessing_popups_hidden(self) -> None:
        """
        Hard-hide preprocessing docks/dialogs while Post Processing is active.
        This protects against late Qt dock re-show events when dock tab stacks are rebuilt.
        """
        if self._use_pg_dockarea_pre_layout:
            if isinstance(self.art_dock, QtWidgets.QDockWidget):
                try:
                    self.art_dock.hide()
                    self.removeDockWidget(self.art_dock)
                except Exception:
                    pass
            return
        if hasattr(self, "tabs") and self.tabs.currentWidget() is self.pre_tab:
            return
        self._suspend_panel_layout_persistence = True
        try:
            for key in self._section_docks.keys():
                self._set_section_button_checked(key, False)
            self._hide_dock_widgets(self.getPreDockWidgets(), remove=True)
            if self._advanced_dialog is not None:
                self._advanced_dialog.hide()
            # Extra safety: hide any dock that belongs to preprocessing by object name prefix.
            for dock in self.findChildren(QtWidgets.QDockWidget):
                name = str(dock.objectName() or "")
                if name.startswith(_PRE_DOCK_PREFIX):
                    dock.hide()
                    try:
                        self.removeDockWidget(dock)
                    except Exception:
                        pass
        finally:
            self._suspend_panel_layout_persistence = False

    def _enforce_postprocessing_popups_hidden(self) -> None:
        """Hide post-processing docks while Preprocessing is active."""
        if hasattr(self, "tabs") and self.tabs.currentWidget() is not self.pre_tab:
            return
        remove_post = not self._force_fixed_dock_layouts
        self._hide_dock_widgets(self.getPostDockWidgets(), remove=remove_post)

    def _store_pre_main_dock_snapshot(self) -> None:
        """Persist the current preprocessing dock arrangement."""
        if self._use_pg_dockarea_pre_layout:
            self._save_panel_layout_state()
            return
        try:
            state = self.captureDockSnapshotForTab("pre")
            if state is not None and not state.isEmpty():
                self.settings.setValue(_PRE_DOCK_STATE_KEY, state)
            self._save_pre_tab_groups_to_settings(self._capture_pre_tab_groups_state())
            self.settings.sync()
        except Exception:
            pass

    def _persist_hidden_preprocessing_layout_state(self) -> None:
        """
        Persist cached preprocessing layout while preprocessing docks are hidden
        during a main-tab switch.
        """
        if self._use_pg_dockarea_pre_layout:
            self._save_panel_layout_state()
            return
        if not self._pre_popups_hidden_by_tab_switch:
            return
        left_i = _dock_area_to_int(QtCore.Qt.DockWidgetArea.LeftDockWidgetArea, 1)
        try:
            for key in self._section_docks.keys():
                state = self._pre_section_state_before_tab_switch.get(key, {})
                base = f"pre_section_docks/{key}"
                self.settings.setValue(f"{base}/visible", bool(state.get("visible", False)))
                self.settings.setValue(f"{base}/floating", bool(state.get("floating", True)))
                self.settings.setValue(f"{base}/area", _dock_area_to_int(state.get("area", left_i), left_i))
                geom = state.get("geometry")
                if isinstance(geom, QtCore.QByteArray) and not geom.isEmpty():
                    self.settings.setValue(f"{base}/geometry", geom)
        except Exception:
            pass

        try:
            art_state = self._pre_artifact_state_before_tab_switch or {}
            base = "pre_artifact_dock_state"
            self.settings.setValue(f"{base}/visible", bool(art_state.get("visible", False)))
            self.settings.setValue(f"{base}/floating", bool(art_state.get("floating", False)))
            self.settings.setValue(f"{base}/area", _dock_area_to_int(art_state.get("area", left_i), left_i))
            art_geom = art_state.get("geometry")
            if isinstance(art_geom, QtCore.QByteArray) and not art_geom.isEmpty():
                self.settings.setValue(f"{base}/geometry", art_geom)
        except Exception:
            pass

        try:
            state = self._pre_main_dock_state_before_tab_switch
            if isinstance(state, QtCore.QByteArray) and not state.isEmpty():
                self.settings.setValue(_PRE_DOCK_STATE_KEY, state)
        except Exception:
            pass
        self._save_pre_tab_groups_to_settings(self._pre_tab_groups_before_tab_switch)

        try:
            self.settings.sync()
        except Exception:
            pass

    def _apply_pre_main_dock_snapshot_if_needed(self) -> None:
        if self._use_pg_dockarea_pre_layout:
            self._pre_snapshot_applied = True
            return
        if self._force_fixed_dock_layouts:
            self._pre_snapshot_applied = True
            return
        if self._pre_snapshot_applied:
            return
        try:
            raw = self.settings.value(_PRE_DOCK_STATE_KEY, None)
            state = self._to_qbytearray(raw)
            if state is None or state.isEmpty():
                self._pre_snapshot_applied = True
                return

            ok = self.restoreDockSnapshotForTab("pre", state)
            if ok:
                self._pre_snapshot_applied = True
                self._pre_snapshot_retry_attempts = 0
                self._sync_section_button_states_from_docks()
                _LOG.info("Pre dock snapshot applied successfully")
                return

            self._pre_snapshot_retry_attempts += 1
            _LOG.warning(
                "Pre dock snapshot restore failed (attempt %s/%s)",
                self._pre_snapshot_retry_attempts,
                self._pre_snapshot_max_retries,
            )
            if self._pre_snapshot_retry_attempts >= self._pre_snapshot_max_retries:
                # Incompatible payload (old object names or stale version): drop and continue
                # with per-dock fallback settings.
                self.settings.remove(_PRE_DOCK_STATE_KEY)
                self._pre_snapshot_applied = True
                return

            delay = 0 if self._post_docks_ready else 120
            self._schedule_pre_snapshot_retry(delay)
        except Exception:
            _LOG.exception("Pre dock snapshot restore raised unexpectedly")

    def _apply_pre_default_layout_if_missing(self) -> None:
        """Set a sensible preprocessing dock layout when no saved layout exists."""
        if self._use_pg_dockarea_pre_layout:
            self._setup_section_popups()
            self._restore_pre_dockarea_layout_state()
            self._save_panel_layout_state()
            return
        try:
            if self.settings.contains(_PRE_DOCK_STATE_KEY):
                return
            has_any = False
            for key in self._section_docks.keys():
                if self.settings.contains(f"pre_section_docks/{key}/visible"):
                    has_any = True
                    break
            if self.settings.contains("pre_artifact_dock_state/visible"):
                has_any = True
            if has_any:
                return
        except Exception:
            return

        if not self._section_docks:
            return

        left = QtCore.Qt.DockWidgetArea.LeftDockWidgetArea
        self._suspend_panel_layout_persistence = True
        try:
            self._enforce_postprocessing_popups_hidden()
            # Default preprocessing layout:
            # - top tab group: Artifacts list / Artifacts / Filtering / Baseline / Output
            # - middle: QC
            # - bottom: Export
            # - left tab: Configuration
            artifacts = self._section_docks.get("artifacts")
            filtering = self._section_docks.get("filtering")
            baseline = self._section_docks.get("baseline")
            output = self._section_docks.get("output")
            qc = self._section_docks.get("qc")
            export = self._section_docks.get("export")
            config = self._section_docks.get("config")

            for dock in (self.art_dock, artifacts, filtering, baseline, output, qc, export, config):
                if dock is None:
                    continue
                dock.setFloating(False)
                dock.show()

            self.addDockWidget(left, self.art_dock)
            for dock in (artifacts, filtering, baseline, output, qc, export):
                if dock is not None:
                    self.addDockWidget(left, dock)

            if qc is not None:
                self.splitDockWidget(self.art_dock, qc, QtCore.Qt.Orientation.Vertical)
            if export is not None:
                if qc is not None:
                    self.splitDockWidget(qc, export, QtCore.Qt.Orientation.Vertical)
                else:
                    self.splitDockWidget(self.art_dock, export, QtCore.Qt.Orientation.Vertical)

            if config is not None:
                self.addDockWidget(left, config)
                config.raise_()

            for dock in (artifacts, filtering, baseline, output, config):
                if dock is not None:
                    self.tabifyDockWidget(self.art_dock, dock)
            self.art_dock.raise_()
            if qc is not None:
                qc.raise_()
            if export is not None:
                export.raise_()

            self._sync_section_button_states_from_docks()
        finally:
            self._suspend_panel_layout_persistence = False

        self._save_panel_layout_state()
        self._store_pre_main_dock_snapshot()

    def _apply_pre_fixed_layout(self) -> None:
        """
        Force a deterministic preprocessing dock layout matching the project default:
        - Left column top: Artifacts list tab group
          (Artifacts list / Artifacts / Filtering / Baseline / Output)
        - Left column middle: QC
        - Left column bottom: Export
        - Left tab: Configuration
        """
        if self._use_pg_dockarea_pre_layout:
            self._setup_section_popups()
            if not self._pre_dockarea_fixed_layout_applied:
                self._apply_pre_fixed_dockarea_layout()
            else:
                self._sync_section_button_states_from_docks()
            return
        if not self._section_docks:
            return

        host = self
        left = QtCore.Qt.DockWidgetArea.LeftDockWidgetArea
        artifacts = self._section_docks.get("artifacts")
        filtering = self._section_docks.get("filtering")
        baseline = self._section_docks.get("baseline")
        output = self._section_docks.get("output")
        qc = self._section_docks.get("qc")
        export = self._section_docks.get("export")
        config = self._section_docks.get("config")

        self._suspend_panel_layout_persistence = True
        try:
            self._hide_dock_widgets(self.getPostDockWidgets(), remove=True)
            # Attach all preprocessing docks in a deterministic non-floating state first.
            ordered_left: List[QtWidgets.QDockWidget] = []
            if isinstance(self.art_dock, QtWidgets.QDockWidget):
                ordered_left.append(self.art_dock)
            for dock in (artifacts, filtering, baseline, output, qc, export, config):
                if isinstance(dock, QtWidgets.QDockWidget):
                    ordered_left.append(dock)

            for dock in ordered_left:
                dock.blockSignals(True)
                try:
                    dock.setFloating(False)
                    host.addDockWidget(left, dock)
                    dock.show()
                finally:
                    dock.blockSignals(False)

            # Vertical stack in left area: top tab group -> QC -> Export.
            if qc is not None:
                host.splitDockWidget(self.art_dock, qc, QtCore.Qt.Orientation.Vertical)
            if export is not None:
                if qc is not None:
                    host.splitDockWidget(qc, export, QtCore.Qt.Orientation.Vertical)
                else:
                    host.splitDockWidget(self.art_dock, export, QtCore.Qt.Orientation.Vertical)

            # Top tab group: Artifacts list + Artifacts + Filtering + Baseline + Output.
            if artifacts is not None:
                host.tabifyDockWidget(self.art_dock, artifacts)
            if filtering is not None:
                host.tabifyDockWidget(self.art_dock, filtering)
            if baseline is not None:
                host.tabifyDockWidget(self.art_dock, baseline)
            if output is not None:
                host.tabifyDockWidget(self.art_dock, output)
            if config is not None:
                host.tabifyDockWidget(self.art_dock, config)

            # Keep active tabs consistent with the default arrangement.
            try:
                self.art_dock.raise_()
            except Exception:
                pass
            if qc is not None:
                qc.raise_()
            if export is not None:
                export.raise_()
            if config is not None:
                config.raise_()

            # Approximate default height proportions for left-column groups.
            try:
                vdocks: List[QtWidgets.QDockWidget] = []
                sizes: List[int] = []
                if isinstance(self.art_dock, QtWidgets.QDockWidget):
                    vdocks.append(self.art_dock)
                    sizes.append(560)
                if qc is not None:
                    vdocks.append(qc)
                    sizes.append(220)
                if export is not None:
                    vdocks.append(export)
                    sizes.append(120)
                if vdocks and sizes:
                    host.resizeDocks(vdocks, sizes, QtCore.Qt.Orientation.Vertical)
            except Exception:
                pass

            self._sync_section_button_states_from_docks()
        finally:
            self._suspend_panel_layout_persistence = False

    def _restore_preprocessing_popups_after_tab_switch(self) -> None:
        if self._use_pg_dockarea_pre_layout:
            return
        if not self._pre_popups_hidden_by_tab_switch:
            return

        host = self
        restored_from_snapshot = False
        self._suspend_panel_layout_persistence = True
        try:
            snapshot = self._pre_main_dock_state_before_tab_switch
            if (
                not self._force_fixed_dock_layouts
                and isinstance(snapshot, QtCore.QByteArray)
                and not snapshot.isEmpty()
            ):
                try:
                    restored_from_snapshot = self.restoreDockSnapshotForTab("pre", snapshot)
                except Exception:
                    restored_from_snapshot = False
            if restored_from_snapshot:
                self._sync_section_button_states_from_docks()
                if self._pre_advanced_visible_before_tab_switch and self._advanced_dialog is not None:
                    self._advanced_dialog.show()
                    self._advanced_dialog.raise_()
                    self._advanced_dialog.activateWindow()
            else:
                for key, dock in self._section_docks.items():
                    state = self._pre_section_state_before_tab_switch.get(key, {})
                    visible = bool(state.get("visible", self._pre_section_visibility_before_tab_switch.get(key, False)))
                    floating = bool(state.get("floating", dock.isFloating()))
                    area = self._dock_area_from_settings(
                        state.get("area", _dock_area_to_int(QtCore.Qt.DockWidgetArea.LeftDockWidgetArea, 1)),
                        QtCore.Qt.DockWidgetArea.LeftDockWidgetArea,
                    )
                    geom = state.get("geometry", None)

                    dock.blockSignals(True)
                    try:
                        if floating:
                            dock.setFloating(True)
                        else:
                            host.addDockWidget(area, dock)
                            dock.setFloating(False)
                        if isinstance(geom, QtCore.QByteArray) and not geom.isEmpty():
                            dock.restoreGeometry(geom)
                            self._section_popup_initialized.add(key)
                        if visible:
                            if dock.isFloating() and not self._is_popup_on_screen(dock):
                                self._position_section_popup(dock)
                            dock.show()
                            self._set_section_button_checked(key, True)
                            self._last_opened_section = key
                        else:
                            dock.hide()
                            self._set_section_button_checked(key, False)
                    finally:
                        dock.blockSignals(False)

                art_state = self._pre_artifact_state_before_tab_switch or {}
                art_visible = bool(art_state.get("visible", self._pre_artifact_visible_before_tab_switch))
                art_floating = bool(art_state.get("floating", self.art_dock.isFloating()))
                art_area = self._dock_area_from_settings(
                    art_state.get("area", _dock_area_to_int(QtCore.Qt.DockWidgetArea.LeftDockWidgetArea, 1)),
                    QtCore.Qt.DockWidgetArea.LeftDockWidgetArea,
                )
                art_geom = art_state.get("geometry", None)
                if art_floating:
                    self.art_dock.setFloating(True)
                else:
                    host.addDockWidget(art_area, self.art_dock)
                    self.art_dock.setFloating(False)
                if isinstance(art_geom, QtCore.QByteArray) and not art_geom.isEmpty():
                    self.art_dock.restoreGeometry(art_geom)
                self.art_dock.setVisible(art_visible)

                if self._pre_advanced_visible_before_tab_switch and self._advanced_dialog is not None:
                    self._advanced_dialog.show()
                    self._advanced_dialog.raise_()
                    self._advanced_dialog.activateWindow()
                self._restore_pre_tab_groups_fallback(self._pre_tab_groups_before_tab_switch)
        finally:
            self._suspend_panel_layout_persistence = False

        self._pre_popups_hidden_by_tab_switch = False
        self._pre_section_visibility_before_tab_switch.clear()
        self._pre_section_state_before_tab_switch.clear()
        self._pre_artifact_visible_before_tab_switch = False
        self._pre_artifact_state_before_tab_switch.clear()
        self._pre_advanced_visible_before_tab_switch = False
        self._pre_main_dock_state_before_tab_switch = None
        self._pre_tab_groups_before_tab_switch = []
        self._enforce_postprocessing_popups_hidden()
        self._save_panel_layout_state()

    def _restore_window_state_after_tab_switch(self, was_fullscreen: bool, was_maximized: bool) -> None:
        """
        Keep the top-level window mode stable across heavy dock add/remove/tabify
        operations triggered by main-tab switches.
        """
        if was_fullscreen:
            if self.isFullScreen():
                return

            def _ensure_fullscreen() -> None:
                if not self.isFullScreen():
                    self.showFullScreen()

            try:
                _ensure_fullscreen()
                QtCore.QTimer.singleShot(0, _ensure_fullscreen)
                QtCore.QTimer.singleShot(120, _ensure_fullscreen)
            except Exception:
                pass
            return

        if was_maximized and not self.isMaximized() and not self.isFullScreen():
            try:
                self.showMaximized()
            except Exception:
                pass

    def _apply_fixed_post_layout_deferred(self) -> None:
        if not hasattr(self, "tabs") or self.tabs.currentWidget() is not self.post_tab:
            return
        try:
            self.post_tab.ensure_section_popups_initialized()
            if hasattr(self.post_tab, "apply_fixed_default_layout"):
                self.post_tab.apply_fixed_default_layout()
        except Exception:
            _LOG.exception("Failed to apply fixed post layout on tab switch")
            return
        try:
            self._enforce_only_tab_docks_visible("post")
        except Exception:
            pass

    def _enforce_fixed_layout_for_active_tab(self) -> None:
        if not self._force_fixed_dock_layouts or not hasattr(self, "tabs"):
            return
        try:
            current = self.tabs.currentWidget()
        except Exception:
            return
        try:
            if current is self.pre_tab:
                self._apply_pre_fixed_layout()
                self._enforce_only_tab_docks_visible("pre")
            elif current is self.post_tab:
                self._apply_fixed_post_layout_deferred()
                self._enforce_only_tab_docks_visible("post")
        except Exception:
            _LOG.exception("Failed to enforce fixed layout for active tab")

    def _on_main_tab_changed(self, index: int) -> None:
        if self._handling_main_tab_change:
            self._pending_main_tab_index = int(index)
            return
        self._handling_main_tab_change = True
        was_fullscreen = bool(self.isFullScreen())
        was_maximized = bool(self.isMaximized())
        try:
            current = self.tabs.widget(index)
            if self._force_fixed_dock_layouts:
                try:
                    if current is self.pre_tab:
                        try:
                            self.post_tab.hide_section_popups_for_tab_switch()
                        except Exception:
                            pass
                        self._enforce_postprocessing_popups_hidden()
                        self._restore_preprocessing_popups_after_tab_switch()
                        self._apply_pre_fixed_layout()
                        self._enforce_only_tab_docks_visible("pre")
                    else:
                        self._hide_preprocessing_popups_for_tab_switch()
                        self._enforce_preprocessing_popups_hidden()
                        QtCore.QTimer.singleShot(0, self._enforce_preprocessing_popups_hidden)
                        self._apply_fixed_post_layout_deferred()
                        QtCore.QTimer.singleShot(0, self._apply_fixed_post_layout_deferred)
                        QtCore.QTimer.singleShot(120, self._apply_fixed_post_layout_deferred)
                        self._enforce_only_tab_docks_visible("post")
                    try:
                        self._save_panel_layout_state()
                    except Exception:
                        pass
                    if current is self.pre_tab:
                        try:
                            self._store_pre_main_dock_snapshot()
                        except Exception:
                            pass
                    else:
                        try:
                            self._persist_hidden_preprocessing_layout_state()
                        except Exception:
                            pass
                    if current is self.pre_tab:
                        try:
                            self.post_tab.persist_layout_state_snapshot()
                        except Exception:
                            pass
                    self._save_settings()
                    self._save_panel_config_json()
                except Exception:
                    _LOG.exception("Failed to handle fixed-layout tab switch")
                return
            try:
                if current is self.pre_tab:
                    try:
                        self.post_tab.hide_section_popups_for_tab_switch()
                    except Exception:
                        pass
                    self._enforce_postprocessing_popups_hidden()
                    self._restore_preprocessing_popups_after_tab_switch()
                    self._apply_pre_main_dock_snapshot_if_needed()
                    self._enforce_only_tab_docks_visible("pre")
                else:
                    self._hide_preprocessing_popups_for_tab_switch()
                    # Run once now and once after queued dock events so preprocessing panels
                    # cannot bleed into Post Processing.
                    self._enforce_preprocessing_popups_hidden()
                    QtCore.QTimer.singleShot(0, self._enforce_preprocessing_popups_hidden)
                    self._enforce_only_tab_docks_visible("post")
                # Persist active main tab immediately.
                self._save_settings()
                # Persist panel layout JSON on each tab switch.
                self._save_panel_config_json()
            except Exception:
                _LOG.exception("Failed to handle main tab switch")
        finally:
            self._restore_window_state_after_tab_switch(was_fullscreen, was_maximized)
            try:
                if self.tabs.currentWidget() is self.post_tab:
                    QtCore.QTimer.singleShot(0, self._post_get_current_dio_list)
            except Exception:
                pass
            self._handling_main_tab_change = False
            if self._force_fixed_dock_layouts:
                QtCore.QTimer.singleShot(0, self._enforce_fixed_layout_for_active_tab)
                QtCore.QTimer.singleShot(80, self._enforce_fixed_layout_for_active_tab)
            if self._pending_main_tab_index is not None:
                pending = int(self._pending_main_tab_index)
                self._pending_main_tab_index = None
                QtCore.QTimer.singleShot(0, lambda idx=pending: self._on_main_tab_changed(idx))

    def _on_artifact_overlay_toggled(self, visible: bool) -> None:
        self.plots.set_artifact_overlay_visible(bool(visible))
        self._record_pre_history_change()
        self._save_settings()

    def _on_artifact_thresholds_toggled(self, visible: bool) -> None:
        self.plots.set_artifact_thresholds_visible(bool(visible))
        self._record_pre_history_change()
        self._save_settings()

    def _normalize_app_theme_mode(self, value: object) -> str:
        mode = str(value or "").strip().lower()
        if mode in {"light", "white", "l", "w"}:
            return "light"
        return "dark"

    def _selected_app_theme_mode(self) -> str:
        if hasattr(self, "act_app_theme_dark") and self.act_app_theme_dark.isChecked():
            return "dark"
        if hasattr(self, "act_app_theme_light") and self.act_app_theme_light.isChecked():
            return "light"
        return self._normalize_app_theme_mode(getattr(self, "_app_theme_mode", "dark"))

    def _apply_app_theme(self, theme_mode: object, persist: bool = True) -> None:
        mode = self._normalize_app_theme_mode(theme_mode)
        self._app_theme_mode = mode

        if hasattr(self, "act_app_theme_dark"):
            self.act_app_theme_dark.blockSignals(True)
            self.act_app_theme_dark.setChecked(mode == "dark")
            self.act_app_theme_dark.blockSignals(False)
        if hasattr(self, "act_app_theme_light"):
            self.act_app_theme_light.blockSignals(True)
            self.act_app_theme_light.setChecked(mode == "light")
            self.act_app_theme_light.blockSignals(False)
        if hasattr(self, "btn_app_theme"):
            try:
                self.btn_app_theme.setToolTip(f"Current app theme: {mode.title()}. Open to switch.")
            except Exception:
                pass

        try:
            apply_app_palette(QtWidgets.QApplication.instance(), mode)
            self.setStyleSheet(app_qss(mode))
        except Exception:
            pass
        self._refresh_pre_rail_icons()

        pre_bg = "white" if mode == "light" else "dark"
        pre_grid = self.act_plot_grid.isChecked() if hasattr(self, "act_plot_grid") else True
        self._apply_pre_plot_style(pre_bg, pre_grid, persist=False)

        try:
            if hasattr(self.post_tab, "set_app_theme_mode"):
                self.post_tab.set_app_theme_mode(mode)
        except Exception:
            pass

        if persist:
            self._save_settings()

    def _refresh_pre_rail_icons(self) -> None:
        try:
            from styles import _make_icon
        except Exception:
            return
        icon_color = "#334155" if self._app_theme_mode == "light" else "#cdd6f4"
        painters = getattr(self, "_pre_rail_icon_painters", {}) or {}
        for key, painter in painters.items():
            btn = getattr(self, "_section_buttons", {}).get(key)
            if btn is None:
                continue
            try:
                btn.setIcon(_make_icon(painter, color=icon_color))
            except Exception:
                continue
        toggle_painter = getattr(self, "_pre_toggle_data_icon_painter", None)
        if toggle_painter is not None and hasattr(self, "btn_toggle_data"):
            try:
                self.btn_toggle_data.setIcon(_make_icon(toggle_painter, color=icon_color))
            except Exception:
                pass

    def _on_app_theme_changed(self, *_args) -> None:
        sender = self.sender()
        if sender is getattr(self, "act_app_theme_dark", None):
            self._apply_app_theme("dark", persist=True)
            return
        if sender is getattr(self, "act_app_theme_light", None):
            self._apply_app_theme("light", persist=True)
            return
        self._apply_app_theme(self._selected_app_theme_mode(), persist=True)

    def _normalize_pre_plot_background(self, value: object) -> str:
        mode = str(value or "").strip().lower()
        if mode in {"white", "light", "w"}:
            return "white"
        return "dark"

    def _selected_pre_plot_background(self) -> str:
        if hasattr(self, "act_plot_bg_white") and self.act_plot_bg_white.isChecked():
            return "white"
        return "dark"

    def _apply_pre_plot_style(self, background: object, show_grid: object, persist: bool = True) -> None:
        mode = self._normalize_pre_plot_background(background)
        grid = bool(show_grid)
        if hasattr(self, "act_plot_bg_dark"):
            self.act_plot_bg_dark.blockSignals(True)
            self.act_plot_bg_dark.setChecked(mode == "dark")
            self.act_plot_bg_dark.blockSignals(False)
        if hasattr(self, "act_plot_bg_white"):
            self.act_plot_bg_white.blockSignals(True)
            self.act_plot_bg_white.setChecked(mode == "white")
            self.act_plot_bg_white.blockSignals(False)
        if hasattr(self, "act_plot_grid"):
            self.act_plot_grid.blockSignals(True)
            self.act_plot_grid.setChecked(grid)
            self.act_plot_grid.blockSignals(False)
        try:
            self.plots.set_plot_appearance(mode, grid)
        except Exception:
            pass
        if persist:
            self._save_settings()

    def _on_pre_plot_style_changed(self, *_args) -> None:
        self._apply_pre_plot_style(
            self._selected_pre_plot_background(),
            self.act_plot_grid.isChecked() if hasattr(self, "act_plot_grid") else True,
            persist=True,
        )
        self._record_pre_history_change()

    def _auto_range_for_processed(self, processed: ProcessedTrial) -> None:
        try:
            start_s, end_s = self._time_window_bounds()
            t = np.asarray(processed.time, float)
            if t.size > 1:
                x0 = float(np.nanmin(t)) if start_s is None else float(start_s)
                x1 = float(np.nanmax(t)) if end_s is None else float(end_s)
                if np.isfinite(x0) and np.isfinite(x1) and x1 > x0:
                    self.plots.auto_range_all(x0=x0, x1=x1)
                else:
                    self.plots.auto_range_all()
            else:
                self.plots.auto_range_all()
        except Exception:
            self.plots.auto_range_all()

    def _fmt_fs(self, fs: Optional[float]) -> str:
        if fs is None or not np.isfinite(float(fs)):
            return "-"
        return f"{float(fs):.2f}"

    def _current_fs_actual(self) -> Optional[float]:
        key = self._current_key()
        if key:
            proc = self._last_processed.get(key)
            if proc is not None and np.isfinite(float(getattr(proc, "fs_actual", np.nan))):
                return float(proc.fs_actual)
        if not self._current_path or not self._current_channel:
            return None
        doric = self._loaded_files.get(self._current_path)
        if doric is None:
            return None
        try:
            trial = doric.make_trial(self._current_channel, trigger_name=self._current_trigger)
            trial = self._apply_time_window(trial)
            fs = float(trial.sampling_rate)
            return fs if np.isfinite(fs) else None
        except Exception:
            return None

    def _refresh_preprocessing_recommendation(self) -> None:
        panel = getattr(self, "param_panel", None)
        if panel is None or not hasattr(panel, "set_recommendation"):
            return
        if not self._current_path or not self._current_channel:
            try:
                panel.set_recommendation(None)
            except Exception:
                pass
            return
        doric = self._loaded_files.get(self._current_path)
        if doric is None:
            try:
                panel.set_recommendation(None)
            except Exception:
                pass
            return
        try:
            trial = doric.make_trial(self._current_channel, trigger_name=self._current_trigger)
            trial = self._apply_time_window(trial)
            key = (self._current_path, self._current_channel)
            cutouts = self._cutout_regions_by_key.get(key, [])
            trial = self._apply_cutouts(trial, cutouts)
            recommendation = recommend_preprocessing_settings(trial, panel.get_params())
            panel.set_recommendation(recommendation)
        except Exception as exc:
            try:
                panel.set_recommendation(None)
            except Exception:
                pass
            self._show_status_message(f"Recommendation unavailable: {exc}", 6000)

    def _pump_export_progress_events(self) -> None:
        app = QtWidgets.QApplication.instance()
        if app is None:
            return
        try:
            app.processEvents(QtCore.QEventLoop.ProcessEventsFlag.AllEvents, 20)
        except TypeError:
            app.processEvents()

    def _begin_export_progress(self, total_steps: int, message: str = "Exporting data...") -> None:
        self._export_progress_generation += 1
        self._set_export_progress(0, total_steps, message)

    def _on_post_export_progress(self, value: int, total_steps: int, message: str) -> None:
        total = max(1, int(total_steps))
        current = max(0, min(int(value), total))
        if current <= 0:
            self._begin_export_progress(total, message or "Exporting data...")
        elif current >= total:
            self._finish_export_progress(total, message or "Export complete")
        else:
            self._set_export_progress(current, total, message or "Exporting data...")

    def _set_export_progress(self, value: int, total_steps: int, message: str = "") -> None:
        widget = getattr(self, "_export_progress_widget", None)
        bar = getattr(self, "_export_progress_bar", None)
        if not isinstance(widget, QtWidgets.QWidget) or not isinstance(bar, QtWidgets.QProgressBar):
            return
        total = max(1, int(total_steps))
        current = max(0, min(int(value), total))
        try:
            bar.setRange(0, total)
            bar.setValue(current)
            if message:
                text = str(message)
                bar.setToolTip(text)
                self._show_status_message(text, 0)
            widget.setVisible(True)
            self._pump_export_progress_events()
        except Exception:
            pass

    def _finish_export_progress(self, total_steps: int, message: str = "") -> None:
        generation = int(getattr(self, "_export_progress_generation", 0))
        self._set_export_progress(total_steps, total_steps, message or "Export complete")
        if message:
            self._show_status_message(message, 5000)
        QtCore.QTimer.singleShot(900, lambda gen=generation: self._hide_export_progress(gen))

    def _hide_export_progress(self, generation: Optional[int] = None) -> None:
        if generation is not None and int(generation) != int(getattr(self, "_export_progress_generation", 0)):
            return
        widget = getattr(self, "_export_progress_widget", None)
        bar = getattr(self, "_export_progress_bar", None)
        try:
            if isinstance(widget, QtWidgets.QWidget):
                widget.setVisible(False)
            if isinstance(bar, QtWidgets.QProgressBar):
                bar.setValue(0)
        except Exception:
            pass

    def _show_status_message(self, message: str, timeout_ms: int = 0) -> None:
        sb = getattr(self, "_status_bar", None)
        if not isinstance(sb, QtWidgets.QStatusBar):
            attr = getattr(self, "statusBar", None)
            if callable(attr):
                try:
                    sb = attr()
                except Exception:
                    sb = None
            elif isinstance(attr, QtWidgets.QStatusBar):
                sb = attr
        if not isinstance(sb, QtWidgets.QStatusBar):
            return
        try:
            sb.showMessage(str(message), int(timeout_ms))
        except Exception:
            pass

    def _update_plot_status(self, fs_actual: Optional[float] = None, fs_target: Optional[float] = None) -> None:
        channel = self._current_channel or "-"
        trig = self._current_trigger or "None"
        mode = "-"
        target = fs_target
        try:
            p = self.param_panel.get_params()
            mode = str(p.output_mode)
            if target is None:
                target = float(p.target_fs_hz)
        except Exception:
            pass
        if fs_actual is None:
            fs_actual = self._current_fs_actual()

        status = (
            f"Channel: {channel} | A/D: {trig} | Fs: {self._fmt_fs(fs_actual)} -> "
            f"{self._fmt_fs(target)} Hz | Mode: {mode}"
        )
        self._show_status_message(status, 30000)

    def _on_file_selection_changed(self) -> None:
        sel = self._selected_paths()
        if not sel:
            all_paths = self.file_panel.all_paths()
            if all_paths:
                self.file_panel.list_files.setCurrentRow(0)
                item0 = self.file_panel.list_files.item(0)
                if item0 is not None:
                    item0.setSelected(True)
                sel = self._selected_paths()
            if not sel:
                self._current_path = None
                self._current_channel = None
                self._current_trigger = None
                self.plots.set_title("No file loaded")
                self._refresh_preprocessing_recommendation()
                self._post_get_current_dio_list()
                self._update_plot_status()
            return

        # preview shows first selected
        path = sel[0]
        self._current_path = path

        doric = self._loaded_files.get(path)
        if not doric:
            return

        self.file_panel.set_available_channels(doric.channels)
        self.file_panel.set_available_triggers(sorted(doric.trigger_by_name.keys()))
        self.param_panel.set_available_export_channels(doric.channels)
        self.param_panel.set_available_export_triggers(sorted(doric.trigger_by_name.keys()))
        self._update_export_summary_label()

        # keep channel if still valid
        if self._current_channel in doric.channels:
            self.file_panel.set_channel(self._current_channel)
        else:
            self._current_channel = doric.channels[0] if doric.channels else None
            if self._current_channel:
                self.file_panel.set_channel(self._current_channel)

        # keep trigger if still valid
        if self._current_trigger and self._current_trigger not in doric.trigger_by_name:
            self._current_trigger = None
            self.file_panel.set_trigger("")
        self._update_export_summary_label()

        self._refresh_preprocessing_recommendation()
        self._update_raw_plot()
        self._trigger_preview()

        # update post tab selection context
        self.post_tab.set_current_source_label(os.path.basename(path), self._current_channel or "")
        self._post_get_current_dio_list()
        self._update_plot_status()

    def _on_channel_changed(self, ch: str) -> None:
        self._current_channel = ch
        if self._current_path:
            doric = self._loaded_files.get(self._current_path)
            if doric is not None:
                self.param_panel.set_available_export_channels(
                    doric.channels,
                    preferred=self.param_panel.export_channel_names(),
                )
                self._update_export_summary_label()
        self._refresh_preprocessing_recommendation()
        self._update_raw_plot()
        self._trigger_preview()
        self.post_tab.set_current_source_label(os.path.basename(self._current_path or ""), self._current_channel or "")
        self._update_plot_status()

    def _on_trigger_changed(self, trig: str) -> None:
        self._current_trigger = trig if trig else None
        if self._current_path:
            doric = self._loaded_files.get(self._current_path)
            if doric is not None:
                self.param_panel.set_available_export_triggers(
                    sorted(doric.trigger_by_name.keys()),
                    preferred=self.param_panel.export_trigger_names(),
                )
                self._update_export_summary_label()
        self._update_raw_plot()
        self._update_plot_status()

    def _on_time_window_changed(self) -> None:
        self._last_processed.clear()
        key = self._current_key()
        if key:
            start_s, end_s = self._time_window_bounds()
            manual_win = self._clip_regions_to_window(self._manual_regions_by_key.get(key, []), start_s, end_s)
            ignore_win = self._clip_regions_to_window(self._manual_exclude_by_key.get(key, []), start_s, end_s)
            auto_win = self._clip_regions_to_window(self._auto_regions_by_key.get(key, []), start_s, end_s)
            checked_auto = [r for r in auto_win if not any(self._regions_match(r, ig) for ig in ignore_win)]
            self.artifact_panel.set_auto_regions(auto_win, checked_regions=checked_auto)
            self.artifact_panel.set_regions(manual_win)
        self._record_pre_history_change()
        self._refresh_preprocessing_recommendation()
        self._update_raw_plot()
        self._trigger_preview()
        self._update_plot_status()

    def _open_advanced_options(self) -> None:
        key = self._current_key()
        if not key:
            return
        if self._advanced_dialog and self._advanced_dialog.isVisible():
            self._advanced_dialog.raise_()
            self._advanced_dialog.activateWindow()
            return
        cutouts = self._cutout_regions_by_key.get(key, [])
        sections = self._sections_by_key.get(key, [])
        dlg = AdvancedOptionsDialog(
            cutouts,
            sections,
            self.param_panel.get_params(),
            request_box_select=self._request_box_select,
            parent=self,
        )
        self._advanced_dialog = dlg

        def _cleanup() -> None:
            if self._advanced_dialog is dlg:
                self._advanced_dialog = None
            self._cancel_box_select_request()

        def _apply() -> None:
            if self._advanced_dialog is not dlg:
                return
            self._cutout_regions_by_key[key] = dlg.get_cutouts()
            self._sections_by_key[key] = dlg.get_sections()
            self._last_processed.clear()
            self._update_raw_plot()
            self._trigger_preview()
            _cleanup()

        dlg.accepted.connect(_apply)
        dlg.rejected.connect(_cleanup)
        dlg.finished.connect(_cleanup)
        dlg.show()

    def _qc_file_paths(self) -> List[str]:
        try:
            paths = self.file_panel.all_paths()
        except Exception:
            paths = []
        return [p for p in paths if p in self._loaded_files]

    def _select_file_for_qc(self, path: str) -> None:
        paths = self.file_panel.all_paths()
        if path in paths:
            row = paths.index(path)
            lw = self.file_panel.list_files
            old_block = lw.blockSignals(True)
            try:
                lw.clearSelection()
                lw.setCurrentRow(row)
                item = lw.item(row)
                if item is not None:
                    item.setSelected(True)
            finally:
                lw.blockSignals(old_block)
        self._on_file_selection_changed()
        QtWidgets.QApplication.processEvents(QtCore.QEventLoop.ProcessEventsFlag.ExcludeUserInputEvents)

    def _compute_current_qc(self) -> Optional[Dict[str, object]]:
        if not self._current_path or not self._current_channel:
            return None
        doric = self._loaded_files.get(self._current_path)
        if not doric:
            return None
        if self._current_channel not in doric.channels:
            self._current_channel = doric.channels[0] if doric.channels else None
            if self._current_channel:
                self.file_panel.set_channel(self._current_channel)
        if not self._current_channel:
            return None
        trial = doric.make_trial(self._current_channel, trigger_name=self._current_trigger)
        trial = self._apply_time_window(trial)
        key = (self._current_path, self._current_channel)
        cutouts = self._cutout_regions_by_key.get(key, [])
        trial = self._apply_cutouts(trial, cutouts)
        return self._compute_qc(trial)

    def _run_qc_dialog(self) -> None:
        paths = self._qc_file_paths()
        if not paths:
            return
        if self._current_path not in paths:
            self._select_file_for_qc(paths[0])

        while True:
            qc = self._compute_current_qc()
            if qc is None:
                return
            paths = self._qc_file_paths()
            try:
                index = paths.index(self._current_path) if self._current_path in paths else 0
            except Exception:
                index = 0
            dlg = QcDialog(qc, self)
            if len(paths) > 1:
                label = f"File {index + 1}/{len(paths)}: {os.path.basename(self._current_path or '')}"
                dlg.set_navigation_state(label, True, True)
            dlg.exec()
            delta = int(getattr(dlg, "navigation_delta", 0))
            if not delta:
                break
            paths = self._qc_file_paths()
            if not paths:
                break
            try:
                index = paths.index(self._current_path) if self._current_path in paths else 0
            except Exception:
                index = 0
            next_index = (index + delta) % len(paths)
            self._select_file_for_qc(paths[next_index])

    def _run_batch_qc(self) -> None:
        paths = self._selected_paths()
        if not paths:
            return
        for p in paths:
            doric = self._loaded_files.get(p)
            if not doric:
                continue
            if self._current_channel and self._current_channel in doric.channels:
                ch = self._current_channel
            else:
                ch = doric.channels[0] if doric.channels else None
            if not ch:
                continue
            trial = doric.make_trial(ch, trigger_name=self._current_trigger)
            trial = self._apply_time_window(trial)
            key = (p, ch)
            cutouts = self._cutout_regions_by_key.get(key, [])
            trial = self._apply_cutouts(trial, cutouts)
            qc = self._compute_qc(trial)
            if qc is None:
                continue
            dlg = QcDialog(qc, self)
            dlg.save_report()
            dlg.close()

    def _compute_qc(self, trial: LoadedTrial) -> Optional[Dict[str, object]]:
        t = np.asarray(trial.time, float)
        sig = np.asarray(trial.signal_465, float)
        ref = np.asarray(trial.reference_405, float)
        if t.size < 10:
            return None
        fs = float(trial.sampling_rate) if np.isfinite(trial.sampling_rate) else (
            1.0 / float(np.nanmedian(np.diff(t))) if t.size > 2 else np.nan
        )
        if ref.size != sig.size:
            ref = np.full_like(sig, np.nan, dtype=float)
        has_reference = bool(ref.size == sig.size and np.sum(np.isfinite(ref)) >= max(10, int(0.05 * sig.size)))
        m = np.isfinite(t) & np.isfinite(sig)
        if has_reference:
            m = m & np.isfinite(ref)
        t = t[m]; sig = sig[m]; ref = ref[m] if ref.size == m.size else np.full(int(np.sum(m)), np.nan, dtype=float)
        if t.size < 10:
            return None

        # Artifact removal (adaptive MAD)
        duration_s = float(np.nanmax(t) - np.nanmin(t)) if t.size else 0.0
        mask_sig = detect_artifacts_adaptive(t, sig, k=6.0, window_s=1.0, pad_s=0.2)
        mask_ref = detect_artifacts_adaptive(t, ref, k=6.0, window_s=1.0, pad_s=0.2) if has_reference else np.zeros_like(mask_sig, dtype=bool)
        mask = mask_sig | mask_ref
        sig_clean = sig.copy()
        ref_clean = ref.copy()
        sig_clean[mask] = np.nan
        if has_reference:
            ref_clean[mask] = np.nan
        sig_clean = interpolate_nans(sig_clean)
        ref_clean = interpolate_nans(ref_clean) if has_reference else np.full_like(sig_clean, np.nan, dtype=float)
        art_frac = float(np.mean(mask)) if mask.size else 0.0

        # Baseline + dff
        cutoff = 0.01
        sig_base = _lowpass_sos(sig_clean, fs, cutoff, 3)
        ref_base = _lowpass_sos(ref_clean, fs, cutoff, 3) if has_reference else np.full_like(sig_base, np.nan, dtype=float)
        dff_sig = safe_divide(sig_clean - sig_base, sig_base)
        dff_ref = safe_divide(ref_clean - ref_base, ref_base) if has_reference else np.full_like(dff_sig, np.nan, dtype=float)

        # Noise-aware metrics are computed on raw dF/F, before z-scoring can
        # hide the absolute noise floor. A low-pass envelope captures slower
        # transients, and the high-frequency residual estimates the local
        # noise floor in interpretable dF/F units.
        def _metric_envelope(values: np.ndarray) -> np.ndarray:
            vals = interpolate_nans(np.asarray(values, float))
            if vals.size < 20:
                return vals
            if np.isfinite(fs) and fs > 1.0:
                cutoff_hz = min(2.0, max(0.05, fs * 0.20))
                if cutoff_hz < fs * 0.45:
                    try:
                        return _lowpass_sos(vals, fs, cutoff_hz, 3)
                    except Exception:
                        return vals
            return vals

        dff_sig_metric = interpolate_nans(np.asarray(dff_sig, float))
        dff_ref_metric = interpolate_nans(np.asarray(dff_ref, float)) if has_reference else np.full_like(dff_sig_metric, np.nan, dtype=float)
        sig_envelope = _metric_envelope(dff_sig_metric)
        ref_envelope = _metric_envelope(dff_ref_metric) if has_reference else np.full_like(sig_envelope, np.nan, dtype=float)
        sig_hf = dff_sig_metric - sig_envelope
        ref_hf = dff_ref_metric - ref_envelope if has_reference else np.full_like(sig_hf, np.nan, dtype=float)
        hf_noise_pct = _qc_robust_sigma(sig_hf) * 100.0
        ref_hf_noise_pct = _qc_robust_sigma(ref_hf) * 100.0 if has_reference else float("nan")
        env_centered = sig_envelope - float(np.nanmedian(sig_envelope)) if sig_envelope.size else sig_envelope
        env_f = np.asarray(env_centered, float)
        env_f = env_f[np.isfinite(env_f)]
        event_amp_pct = float(np.nanpercentile(np.abs(env_f), 95) * 100.0) if env_f.size else float("nan")
        dff_centered = dff_sig_metric - float(np.nanmedian(dff_sig_metric)) if dff_sig_metric.size else dff_sig_metric
        dff_scale_pct = _qc_robust_sigma(dff_centered) * 100.0
        jitter_pct = _qc_robust_sigma(np.diff(dff_sig_metric)) * 100.0 if dff_sig_metric.size > 2 else float("nan")
        usable_snr = event_amp_pct / max(hf_noise_pct, 1e-12) if np.isfinite(event_amp_pct) and np.isfinite(hf_noise_pct) else float("nan")
        jitter_ratio = jitter_pct / max(dff_scale_pct, 1e-12) if np.isfinite(jitter_pct) and np.isfinite(dff_scale_pct) else float("nan")

        # z-score
        z_sig = zscore_median_std(dff_sig)
        z_ref = zscore_median_std(dff_ref) if has_reference else np.full_like(z_sig, np.nan, dtype=float)
        Z = z_sig - z_ref if has_reference else z_sig.copy()
        Zf = Z[np.isfinite(Z)]

        # Reference coupling is measured on dF/F, not on z-score, so the user
        # can interpret the coupling in the same units as the raw noise metrics.
        m2 = np.isfinite(dff_sig_metric) & np.isfinite(dff_ref_metric) if has_reference else np.zeros_like(dff_sig_metric, dtype=bool)
        r = float(np.corrcoef(dff_ref_metric[m2], dff_sig_metric[m2])[0, 1]) if has_reference and np.sum(m2) >= 10 else np.nan
        win = int(max(10, round(fs * 10.0))) if np.isfinite(fs) and fs > 0 else 5000
        # Rolling-window length in seconds: needed to turn a flagged rolling-r
        # sample back into a time span the user can cut out.
        r_win_s = float(win / fs) if np.isfinite(fs) and fs > 0 else float("nan")
        if has_reference:
            r_roll, centers = _rolling_corr(dff_ref_metric, dff_sig_metric, win)
        else:
            r_roll, centers = np.array([], float), np.array([], int)

        # Distribution stats + new shape / stability metrics. The negative tail
        # is tracked separately: sensors do not produce large fast *downward*
        # excursions, so deep negative outliers are the fingerprint of leftover
        # artifact rather than of real transients.
        if Zf.size:
            q25, q50, q75 = np.quantile(Zf, [0.25, 0.5, 0.75])
            frac_gt3 = float(np.mean(np.abs(Zf) > 3.0) * 100.0)
            frac_gt5 = float(np.mean(np.abs(Zf) > 5.0) * 100.0)
            frac_neg5 = float(np.mean(Zf < -5.0) * 100.0)
            iqr = float(q75 - q25)
        else:
            q25 = q50 = q75 = frac_gt3 = frac_gt5 = frac_neg5 = iqr = np.nan

        # Skewness + excess kurtosis of Zf (no scipy dependency).
        if Zf.size > 10:
            zmean = float(np.nanmean(Zf))
            zstd = float(np.nanstd(Zf))
            if zstd > 1e-9:
                zn = (Zf - zmean) / zstd
                skew = float(np.nanmean(zn ** 3))
                kurt = float(np.nanmean(zn ** 4) - 3.0)
            else:
                skew = kurt = 0.0
        else:
            skew = kurt = float("nan")

        # Rolling-correlation stability: std of the rolling r series.
        if r_roll.size:
            r_roll_finite = r_roll[np.isfinite(r_roll)]
            r_roll_std = float(np.nanstd(r_roll_finite)) if r_roll_finite.size else float("nan")
            r_roll_mean = float(np.nanmean(r_roll_finite)) if r_roll_finite.size else float("nan")
        else:
            r_roll_std = r_roll_mean = float("nan")

        # Photobleach: percent baseline change from the first ~10 s to the
        # last ~10 s of the low-passed signal envelope.
        if np.isfinite(fs) and fs > 0 and sig_base.size > int(fs * 20):
            n10 = max(1, int(round(fs * 10.0)))
            first = float(np.nanmedian(sig_base[:n10]))
            last = float(np.nanmedian(sig_base[-n10:]))
            if first > 1e-9:
                bleach_pct = ((first - last) / first) * 100.0
            else:
                bleach_pct = float("nan")
        else:
            bleach_pct = float("nan")

        stats = (
            f"artifact_frac={art_frac*100:.2f}% | HF noise={hf_noise_pct:.3g}% dF/F "
            f"({'ref ' + format(ref_hf_noise_pct, '.3g') + '%' if has_reference else 'no isobestic'}) | event_amp={event_amp_pct:.3g}% | "
            f"SNR~{usable_snr:.2f} | "
            f"r={r:.3f} (avg roll r={r_roll_mean:.2f}, std={r_roll_std:.2f}) | "
            f"Z median={q50:.3g} IQR=({q25:.3g},{q75:.3g}) | "
            f"|Z|>3: {frac_gt3:.2f}% | |Z|>5: {frac_gt5:.2f}% | "
            f"skew={skew:.2f} kurt={kurt:.2f} | bleach={bleach_pct:.1f}%"
        )

        return {
            "path": trial.path,
            "channel": trial.channel_id,
            "has_reference": has_reference,
            "t": t,
            "dff_sig_pct": dff_sig_metric * 100.0,
            "dff_ref_pct": dff_ref_metric * 100.0,
            "dff_envelope_pct": sig_envelope * 100.0,
            "hf_sig_pct": sig_hf * 100.0,
            "hf_ref_pct": ref_hf * 100.0,
            "z_sig": z_sig,
            "z_ref": z_ref,
            "Z": Z,
            "Zf": Zf,
            "r_roll": r_roll,
            "r_centers": centers,
            "r_win_s": r_win_s,
            "r": r,
            # Time-resolved inputs for the "which parts should I cut" scan.
            "art_mask": mask,
            "sig_base": sig_base,
            "duration_s": duration_s,
            "q25": q25,
            "q50": q50,
            "q75": q75,
            "iqr": iqr,
            "frac_gt3": frac_gt3,
            "frac_gt5": frac_gt5,
            "frac_neg5": frac_neg5,
            "art_frac": art_frac,
            "hf_noise_pct": hf_noise_pct,
            "ref_hf_noise_pct": ref_hf_noise_pct,
            "event_amp_pct": event_amp_pct,
            "dff_scale_pct": dff_scale_pct,
            "jitter_pct": jitter_pct,
            "usable_snr": usable_snr,
            "jitter_ratio": jitter_ratio,
            "r_roll_std": r_roll_std,
            "r_roll_mean": r_roll_mean,
            "skew": skew,
            "kurt": kurt,
            "bleach_pct": bleach_pct,
            "fs": fs,
            "stats": stats,
        }

    # ---------------- Raw plot update ----------------

    def _apply_time_window(self, trial: LoadedTrial) -> LoadedTrial:
        start_s, end_s = self.file_panel.time_window()
        if start_s is None and end_s is None:
            return trial

        t = np.asarray(trial.time, float)
        if start_s is None:
            mask = t <= float(end_s)
        elif end_s is None:
            mask = t >= float(start_s)
        else:
            if end_s <= start_s:
                return trial
            mask = (t >= float(start_s)) & (t <= float(end_s))
        if np.sum(mask) < 2:
            return trial

        def _mask_arr(arr: Optional[np.ndarray], use_time_mask: bool) -> Optional[np.ndarray]:
            if arr is None:
                return None
            if use_time_mask and arr.size == t.size:
                return np.asarray(arr, float)[mask]
            return np.asarray(arr, float)

        time = t[mask]
        sig = np.asarray(trial.signal_465, float)[mask]
        ref = np.asarray(trial.reference_405, float)[mask]

        trig_time = trial.trigger_time
        trig = trial.trigger
        if trig_time is not None and trig is not None:
            if trig_time.size == t.size:
                trig_time = _mask_arr(trig_time, True)
                trig = _mask_arr(trig, True)
            else:
                if start_s is None:
                    tmask = np.asarray(trig_time, float) <= float(end_s)
                elif end_s is None:
                    tmask = np.asarray(trig_time, float) >= float(start_s)
                else:
                    tmask = (trig_time >= float(start_s)) & (trig_time <= float(end_s))
                trig_time = np.asarray(trig_time, float)[tmask]
                trig = np.asarray(trig, float)[tmask]

        fs = 1.0 / float(np.nanmedian(np.diff(time))) if time.size > 2 else np.nan

        new_triggers = {}
        new_trigger_times = {}
        if hasattr(trial, "triggers") and trial.triggers:
            for name, val in trial.triggers.items():
                vt = trial.trigger_times.get(name)
                if vt is not None:
                    if vt.size == t.size:
                        new_triggers[name] = np.asarray(val, float)[mask]
                        new_trigger_times[name] = np.asarray(vt, float)[mask]
                    else:
                        tmask = (vt >= float(start_s)) & (vt <= float(end_s))
                        new_triggers[name] = np.asarray(val, float)[tmask]
                        new_trigger_times[name] = np.asarray(vt, float)[tmask]

        return LoadedTrial(
            path=trial.path,
            channel_id=trial.channel_id,
            time=time,
            signal_465=sig,
            reference_405=ref,
            sampling_rate=float(fs) if np.isfinite(fs) else np.nan,
            trigger_time=trig_time,
            trigger=trig,
            trigger_name=trial.trigger_name,
            triggers=new_triggers,
            trigger_times=new_trigger_times,
        )

    def _apply_cutouts(self, trial: LoadedTrial, cutouts: List[Tuple[float, float]]) -> LoadedTrial:
        if not cutouts:
            return trial
        t = np.asarray(trial.time, float)
        sig = np.asarray(trial.signal_465, float).copy()
        ref = np.asarray(trial.reference_405, float).copy()
        for (a, b) in cutouts:
            mask = (t >= float(a)) & (t <= float(b))
            sig[mask] = np.nan
            ref[mask] = np.nan
        return LoadedTrial(
            path=trial.path,
            channel_id=trial.channel_id,
            time=t,
            signal_465=sig,
            reference_405=ref,
            sampling_rate=trial.sampling_rate,
            trigger_time=trial.trigger_time,
            trigger=trial.trigger,
            trigger_name=trial.trigger_name,
            triggers=dict(trial.triggers) if hasattr(trial, "triggers") else {},
            trigger_times=dict(trial.trigger_times) if hasattr(trial, "trigger_times") else {},
        )

    def _apply_cutouts_to_processed(self, processed: ProcessedTrial, cutouts: List[Tuple[float, float]]) -> ProcessedTrial:
        if not cutouts or processed.time is None:
            return processed
        t = np.asarray(processed.time, float)
        mask = np.zeros_like(t, dtype=bool)
        for (a, b) in cutouts:
            mask |= (t >= float(a)) & (t <= float(b))
        if not np.any(mask):
            return processed

        def _mask_arr(arr: Optional[np.ndarray]) -> Optional[np.ndarray]:
            if arr is None:
                return None
            y = np.asarray(arr, float).copy()
            if y.size == t.size:
                y[mask] = np.nan
            return y

        processed.raw_signal = _mask_arr(processed.raw_signal)
        processed.raw_reference = _mask_arr(processed.raw_reference)
        processed.raw_thr_hi = _mask_arr(processed.raw_thr_hi)
        processed.raw_thr_lo = _mask_arr(processed.raw_thr_lo)
        processed.sig_f = _mask_arr(processed.sig_f)
        processed.ref_f = _mask_arr(processed.ref_f)
        processed.baseline_sig = _mask_arr(processed.baseline_sig)
        processed.baseline_ref = _mask_arr(processed.baseline_ref)
        processed.output = _mask_arr(processed.output)

        raw_t = getattr(processed, "raw_display_time", None)
        if raw_t is not None:
            raw_t = np.asarray(raw_t, float)
            raw_mask = np.zeros_like(raw_t, dtype=bool)
            for (a, b) in cutouts:
                raw_mask |= (raw_t >= float(a)) & (raw_t <= float(b))

            def _mask_raw_arr(arr: Optional[np.ndarray]) -> Optional[np.ndarray]:
                if arr is None:
                    return None
                y = np.asarray(arr, float).copy()
                if y.size == raw_t.size:
                    y[raw_mask] = np.nan
                return y

            processed.raw_display_signal = _mask_raw_arr(getattr(processed, "raw_display_signal", None))
            processed.raw_display_reference = _mask_raw_arr(getattr(processed, "raw_display_reference", None))
            processed.raw_display_thr_hi = _mask_raw_arr(getattr(processed, "raw_display_thr_hi", None))
            processed.raw_display_thr_lo = _mask_raw_arr(getattr(processed, "raw_display_thr_lo", None))
            processed.raw_display_ref_thr_hi = _mask_raw_arr(getattr(processed, "raw_display_ref_thr_hi", None))
            processed.raw_display_ref_thr_lo = _mask_raw_arr(getattr(processed, "raw_display_ref_thr_lo", None))
            if getattr(processed, "raw_display_dio_time", None) is not None:
                dio_t = np.asarray(getattr(processed, "raw_display_dio_time"), float)
                dio_mask = np.zeros_like(dio_t, dtype=bool)
                for (a, b) in cutouts:
                    dio_mask |= (dio_t >= float(a)) & (dio_t <= float(b))
                raw_dio = getattr(processed, "raw_display_dio", None)
                if raw_dio is not None:
                    y = np.asarray(raw_dio, float).copy()
                    if y.size == dio_t.size:
                        y[dio_mask] = np.nan
                    processed.raw_display_dio = y
        if hasattr(processed, "outputs") and processed.outputs:
            masked_outputs = {}
            for label, values in processed.outputs.items():
                masked = _mask_arr(values)
                if masked is not None:
                    masked_outputs[str(label)] = masked
            processed.outputs = masked_outputs
        
        # Mask triggers too if requested by convention, but here we keep them as-is or NaN them
        if hasattr(processed, "triggers") and processed.triggers:
            new_triggers = {}
            for name, val in processed.triggers.items():
                new_triggers[name] = _mask_arr(val)
            processed.triggers = new_triggers

        return processed

    def _slice_trial(self, trial: LoadedTrial, start_s: float, end_s: float) -> Optional[LoadedTrial]:
        t = np.asarray(trial.time, float)
        mask = (t >= float(start_s)) & (t <= float(end_s))
        if np.sum(mask) < 2:
            return None
        time = t[mask]
        sig = np.asarray(trial.signal_465, float)[mask]
        ref = np.asarray(trial.reference_405, float)[mask]
        trig_time = trial.trigger_time
        trig = trial.trigger
        if trig_time is not None and trig is not None:
            if trig_time.size == t.size:
                trig_time = np.asarray(trig_time, float)[mask]
                trig = np.asarray(trig, float)[mask]
            else:
                tmask = (trig_time >= float(start_s)) & (trig_time <= float(end_s))
                trig_time = np.asarray(trig_time, float)[tmask]
                trig = np.asarray(trig, float)[tmask]
        
        new_triggers = {}
        new_trigger_times = {}
        if hasattr(trial, "triggers") and trial.triggers:
            for name, val in trial.triggers.items():
                vt = trial.trigger_times.get(name)
                if vt is not None:
                    if vt.size == t.size:
                        new_triggers[name] = np.asarray(val, float)[mask]
                        new_trigger_times[name] = np.asarray(vt, float)[mask]
                    else:
                        tmask = (vt >= float(start_s)) & (vt <= float(end_s))
                        new_triggers[name] = np.asarray(val, float)[tmask]
                        new_trigger_times[name] = np.asarray(vt, float)[tmask]

        fs = 1.0 / float(np.nanmedian(np.diff(time))) if time.size > 2 else np.nan
        return LoadedTrial(
            path=trial.path,
            channel_id=trial.channel_id,
            time=time,
            signal_465=sig,
            reference_405=ref,
            sampling_rate=float(fs) if np.isfinite(fs) else np.nan,
            trigger_time=trig_time,
            trigger=trig,
            trigger_name=trial.trigger_name,
            triggers=new_triggers,
            trigger_times=new_trigger_times,
        )

    def _update_raw_plot(self, preserve_view: bool = False) -> None:
        if not self._current_path or not self._current_channel:
            return
        doric = self._loaded_files.get(self._current_path)
        if not doric:
            return

        trial = doric.make_trial(self._current_channel, trigger_name=self._current_trigger)
        trial = self._apply_time_window(trial)
        key = (self._current_path, self._current_channel)
        cutouts = self._cutout_regions_by_key.get(key, [])
        trial = self._apply_cutouts(trial, cutouts)
        start_s, end_s = self._time_window_bounds()
        manual = self._clip_regions_to_window(self._manual_regions_by_key.get(key, []), start_s, end_s)
        params = self.param_panel.get_params()

        raw465 = trial.signal_465
        raw405 = trial.reference_405
        if bool(getattr(params, "invert_polarity", False)):
            raw465 = -np.asarray(raw465, float)
            raw405 = -np.asarray(raw405, float)

        self.plots.set_title("raw signal")
        self.plots.show_raw(
            time=trial.time,
            raw465=raw465,
            raw405=raw405,
            trig_time=trial.trigger_time,
            trig=trial.trigger,
            trig_label=self._current_trigger or "",
            manual_regions=manual,
            preserve_view=preserve_view,
        )
        self._update_plot_status(fs_actual=float(trial.sampling_rate), fs_target=float(params.target_fs_hz))

    # ---------------- Preview processing (worker) ----------------

    def _artifact_param_signature(self, params: ProcessingParams) -> Tuple[object, ...]:
        return (
            bool(getattr(params, "artifact_detection_enabled", True)),
            str(params.artifact_mode),
            str(getattr(params, "artifact_handling", "Interpolate")),
            float(params.mad_k),
            float(params.adaptive_window_s),
            float(params.artifact_pad_s),
        )

    def _on_params_changed(self) -> None:
        if self._pre_history_restoring:
            return
        try:
            params = self.param_panel.get_params()
        except Exception:
            self._trigger_preview(preserve_view=True)
            return
        self._update_plot_status(fs_target=float(params.target_fs_hz))
        sig = self._artifact_param_signature(params)
        if self._last_artifact_params is None:
            self._last_artifact_params = sig
        elif sig != self._last_artifact_params:
            self._last_artifact_params = sig
            # Reset auto artifact selections when detection params change
            self._manual_exclude_by_key.clear()
            key = self._current_key()
            if key:
                auto = self._auto_regions_by_key.get(key, [])
                if auto:
                    self.artifact_panel.set_auto_regions(auto, checked_regions=auto)
        # Update raw display for toggles like polarity inversion
        try:
            self._update_raw_plot(preserve_view=True)
        except Exception:
            pass
        self._record_pre_history_change()
        self._trigger_preview(preserve_view=True)

    def _trigger_preview(self, preserve_view: bool = False) -> None:
        # persist params quickly
        self._save_settings()
        self._preview_preserve_view_pending = bool(preserve_view)
        self._preview_timer.start()

    def _start_preview_processing(self) -> None:
        if not self._current_path or not self._current_channel:
            return
        doric = self._loaded_files.get(self._current_path)
        if not doric:
            return

        params = self.param_panel.get_params()
        trial = doric.make_trial(self._current_channel, trigger_name=self._current_trigger)
        trial = self._apply_time_window(trial)
        key = (self._current_path, self._current_channel)
        cutouts = self._cutout_regions_by_key.get(key, [])
        trial = self._apply_cutouts(trial, cutouts)

        start_s, end_s = self._time_window_bounds()
        manual = self._clip_regions_to_window(self._manual_regions_by_key.get(key, []), start_s, end_s)
        manual_exclude = self._clip_regions_to_window(self._manual_exclude_by_key.get(key, []), start_s, end_s)
        self._job_counter += 1
        job_id = self._job_counter
        self._latest_job_id = job_id
        preserve_view = bool(self._preview_preserve_view_pending)
        self._preview_preserve_view_pending = False
        self._preview_preserve_view_by_job[job_id] = preserve_view

        self._show_status_message(
            f"Processing preview... (fs={trial.sampling_rate:.2f} Hz -> target {params.target_fs_hz:.1f} Hz, "
            f"baseline={params.baseline_method})"
        )
        self._update_plot_status(fs_actual=float(trial.sampling_rate), fs_target=float(params.target_fs_hz))

        task = self.processor.make_preview_task(
            trial=trial,
            params=params,
            manual_regions_sec=manual,
            manual_exclude_regions_sec=manual_exclude,
            job_id=job_id,
        )
        task.signals.finished.connect(self._on_preview_finished)
        task.signals.failed.connect(self._on_preview_failed)
        self._pool.start(task)

    @QtCore.Slot(object, int, float)
    def _on_preview_finished(self, processed: ProcessedTrial, job_id: int, elapsed_s: float) -> None:
        preserve_view = bool(self._preview_preserve_view_by_job.pop(job_id, False))
        if job_id != self._latest_job_id:
            return  # ignore stale jobs

        key = (processed.path, processed.channel_id)
        cutouts = self._cutout_regions_by_key.get(key, [])
        processed = self._apply_cutouts_to_processed(processed, cutouts)
        self._last_processed[key] = processed

        # Update artifact panel regions list
        start_s, end_s = self._time_window_bounds()
        auto_regs_raw = processed.artifact_regions_auto_sec or []
        auto_core_raw = list(getattr(processed, "artifact_regions_auto_core_sec", None) or [])
        auto_sources_raw = list(getattr(processed, "artifact_regions_auto_source", None) or [])
        auto_regs = self._clip_regions_to_window(auto_regs_raw, start_s, end_s)
        auto_sources = []
        auto_cores = []
        for a, b in auto_regs:
            source = ""
            core = (float(a), float(b))
            for idx, (ra, rb) in enumerate(auto_regs_raw):
                if float(rb) >= float(a) and float(ra) <= float(b):
                    if idx < len(auto_sources_raw):
                        source = str(auto_sources_raw[idx] or "")
                    if idx < len(auto_core_raw):
                        ca, cb = auto_core_raw[idx]
                        core = (max(float(ca), float(a)), min(float(cb), float(b)))
                    break
            auto_sources.append(source)
            auto_cores.append(core)
        self._auto_regions_by_key[key] = auto_regs
        ignore = self._clip_regions_to_window(self._manual_exclude_by_key.get(key, []), start_s, end_s)
        # build checked list by excluding ignored
        checked_auto = [r for r in auto_regs if not any(self._regions_match(r, ig) for ig in ignore)]
        self.artifact_panel.set_auto_regions(
            auto_regs, checked_regions=checked_auto, sources=auto_sources, core_regions=auto_cores
        )
        manual_regs = self._clip_regions_to_window(self._manual_regions_by_key.get(key, []), start_s, end_s)
        self.artifact_panel.set_regions(manual_regs)

        # Update plots (decimated signals)
        self.plots.update_plots(processed, preserve_view=preserve_view)
        if not preserve_view:
            # Auto-range on each update so file/time-window changes do not require manual reset.
            self._auto_range_for_processed(processed)

        log_msg = (
            f"Preview updated: {processed.output_label} | fs={processed.fs_actual:.2f}->{processed.fs_used:.2f} Hz "
            f"(target {processed.fs_target:.2f}) | n={processed.time.size} | {elapsed_s*1000:.0f} ms"
        )
        self._show_status_message(log_msg, 10000)
        self.param_panel.set_fs_info(processed.fs_actual, processed.fs_target, processed.fs_used)
        self._update_plot_status(fs_actual=float(processed.fs_actual), fs_target=float(processed.fs_target))

        # Inform post tab that current processed changed
        self.post_tab.notify_preprocessing_updated(processed)

    @QtCore.Slot(str, int)
    def _on_preview_failed(self, err: str, job_id: int) -> None:
        self._preview_preserve_view_by_job.pop(job_id, None)
        if job_id != self._latest_job_id:
            return
        self._show_status_message(f"Preview error: {err}")

    # ---------------- Manual artifacts ----------------

    def _regions_match(self, a: Tuple[float, float], b: Tuple[float, float], tol: float = 1e-3) -> bool:
        return (abs(a[0] - b[0]) <= tol) and (abs(a[1] - b[1]) <= tol)

    def _time_window_bounds(self) -> Tuple[Optional[float], Optional[float]]:
        start_s, end_s = self.file_panel.time_window()
        if start_s is not None and end_s is not None and end_s <= start_s:
            return None, None
        return start_s, end_s

    def _clip_regions_to_window(
        self,
        regions: List[Tuple[float, float]],
        start_s: Optional[float],
        end_s: Optional[float],
    ) -> List[Tuple[float, float]]:
        if start_s is None and end_s is None:
            return list(regions)
        lo = -np.inf if start_s is None else float(start_s)
        hi = np.inf if end_s is None else float(end_s)
        out: List[Tuple[float, float]] = []
        for a, b in regions or []:
            t0, t1 = (min(a, b), max(a, b))
            if t1 < lo or t0 > hi:
                continue
            out.append((max(t0, lo), min(t1, hi)))
        return out

    def _merge_regions_with_window(
        self,
        original: List[Tuple[float, float]],
        windowed: List[Tuple[float, float]],
        start_s: Optional[float],
        end_s: Optional[float],
    ) -> List[Tuple[float, float]]:
        if start_s is None and end_s is None:
            out = list(windowed)
            out.sort(key=lambda x: x[0])
            return out
        lo = -np.inf if start_s is None else float(start_s)
        hi = np.inf if end_s is None else float(end_s)
        kept: List[Tuple[float, float]] = []
        for a, b in original or []:
            t0, t1 = (min(a, b), max(a, b))
            if t1 < lo or t0 > hi:
                kept.append((t0, t1))
        out = kept + list(windowed)
        out.sort(key=lambda x: x[0])
        return out

    def _add_manual_region_from_selector(self) -> None:
        key = self._current_key()
        if not key:
            return
        t0, t1 = self.plots.selector_region()
        regs = self._manual_regions_by_key.get(key, [])
        regs.append((min(t0, t1), max(t0, t1)))
        self._manual_regions_by_key[key] = regs
        start_s, end_s = self._time_window_bounds()
        self.artifact_panel.set_regions(self._clip_regions_to_window(regs, start_s, end_s))
        self._record_pre_history_change()
        self._trigger_preview(preserve_view=True)

    def _add_manual_region_from_drag(self, t0: float, t1: float) -> None:
        if self._box_select_callback:
            cb = self._box_select_callback
            self._box_select_callback = None
            self.plots.btn_box_select.setChecked(False)
            cb(float(min(t0, t1)), float(max(t0, t1)))
            return
        key = self._current_key()
        if not key:
            return
        if not np.isfinite(t0) or not np.isfinite(t1) or t0 == t1:
            return
        region = (float(min(t0, t1)), float(max(t0, t1)))
        self._pending_box_region_by_key[key] = region
        self.plots.set_selector_region(*region, visible=True)
        self._show_status_message("Selection ready: press A=artifact, C=cut, S=section, or right-click for actions.")

    def _clear_manual_regions_current(self) -> None:
        key = self._current_key()
        if not key:
            return
        self._manual_regions_by_key[key] = []
        self._manual_exclude_by_key[key] = []
        self._pending_box_region_by_key.pop(key, None)
        self.artifact_panel.set_regions([])
        self._record_pre_history_change()
        self._trigger_preview(preserve_view=True)

    def _request_box_select(self, callback: Callable[[float, float], None]) -> None:
        self._box_select_callback = callback
        self.plots.btn_box_select.setChecked(True)
        self._show_status_message("Box select: drag on the raw plot to set the time window; right-click to cancel.")

    def _cancel_box_select_request(self) -> None:
        key = self._current_key()
        self._box_select_callback = None
        if key:
            self._pending_box_region_by_key.pop(key, None)
        self.plots.set_selector_region(0.0, 1.0, visible=False)
        self.plots.btn_box_select.setChecked(False)

    def _pending_box_region(self) -> Optional[Tuple[float, float]]:
        key = self._current_key()
        if not key:
            return None
        region = self._pending_box_region_by_key.get(key)
        if not region:
            if not self.plots.selector_visible():
                return None
            t0, t1 = self.plots.selector_region()
            return (float(min(t0, t1)), float(max(t0, t1)))
        return (float(min(region)), float(max(region)))

    def _consume_pending_box_region(self) -> Optional[Tuple[float, float]]:
        key = self._current_key()
        if not key:
            return None
        
        is_tool_active = self.plots.btn_box_select.isChecked()
        region = self._pending_box_region_by_key.pop(key, None)
        
        if not region:
            if not self.plots.selector_visible():
                return None
            t0, t1 = self.plots.selector_region()
            region = (float(min(t0, t1)), float(max(t0, t1)))
            # If not using the box-select tool, we do NOT hide the persistent selector.
            if not is_tool_active:
                return region

        # Cleanup if we were in tool mode or had a temporary drag selection.
        self.plots.set_selector_region(0.0, 1.0, visible=False)
        self.plots.btn_box_select.setChecked(False)
        return (float(min(region)), float(max(region)))

    def _assign_pending_box_to_artifact(self) -> None:
        region = self._consume_pending_box_region()
        key = self._current_key()
        if not region or not key:
            return
        regs = self._manual_regions_by_key.get(key, [])
        regs.append(region)
        self._manual_regions_by_key[key] = regs
        start_s, end_s = self._time_window_bounds()
        self.artifact_panel.set_regions(self._clip_regions_to_window(regs, start_s, end_s))
        self._record_pre_history_change()
        self._trigger_preview(preserve_view=True)

    def _assign_pending_box_to_cut(self) -> None:
        region = self._consume_pending_box_region()
        key = self._current_key()
        if not region or not key:
            return
        regs = self._cutout_regions_by_key.get(key, [])
        regs.append(region)
        regs.sort(key=lambda x: x[0])
        self._cutout_regions_by_key[key] = regs
        self._last_processed.clear()
        self._record_pre_history_change()
        self._update_raw_plot()
        self._trigger_preview()

    def _assign_pending_box_to_section(self) -> None:
        region = self._consume_pending_box_region()
        key = self._current_key()
        if not region or not key:
            return
        sections = self._sections_by_key.get(key, [])
        sections.append({
            "start": float(region[0]),
            "end": float(region[1]),
            "params": self.param_panel.get_params().to_dict(),
        })
        sections.sort(key=lambda sec: float(sec.get("start", 0.0)))
        self._sections_by_key[key] = sections
        self._record_pre_history_change()
        self._show_status_message(f"Section added: {region[0]:.3f}s to {region[1]:.3f}s")

    def _show_box_selection_context_menu(self) -> None:
        region = self._pending_box_region()
        if region is None:
            self._cancel_box_select_request()
            return
        menu = QtWidgets.QMenu(self)
        act_art = menu.addAction("Set as artifact")
        act_cut = menu.addAction("Set as cut")
        act_sec = menu.addAction("Set as section")
        menu.addSeparator()
        act_cancel = menu.addAction("Cancel selection")
        chosen = menu.exec(QtGui.QCursor.pos())
        if chosen is act_art:
            self._assign_pending_box_to_artifact()
        elif chosen is act_cut:
            self._assign_pending_box_to_cut()
        elif chosen is act_sec:
            self._assign_pending_box_to_section()
        elif chosen is act_cancel:
            self._cancel_box_select_request()

    def _artifact_regions_changed(self, regions: List[Tuple[float, float]]) -> None:
        key = self._current_key()
        if not key:
            return
        start_s, end_s = self._time_window_bounds()
        auto = self._clip_regions_to_window(self._auto_regions_by_key.get(key, []), start_s, end_s)

        def _contains(target: Tuple[float, float], arr: List[Tuple[float, float]]) -> bool:
            return any(self._regions_match(target, other) for other in arr)

        manual_add = [r for r in regions if not _contains(r, auto)]
        manual_ignore = [r for r in auto if not _contains(r, regions)]

        prev_manual = self._manual_regions_by_key.get(key, [])
        prev_ignore = self._manual_exclude_by_key.get(key, [])
        self._manual_regions_by_key[key] = self._merge_regions_with_window(prev_manual, manual_add, start_s, end_s)
        self._manual_exclude_by_key[key] = self._merge_regions_with_window(prev_ignore, manual_ignore, start_s, end_s)
        self._record_pre_history_change()
        self._trigger_preview(preserve_view=True)

    def _toggle_artifacts_panel(self) -> None:
        if self._use_pg_dockarea_pre_layout:
            self._setup_section_popups()
            dock = self._pre_dockarea_dock("artifacts")
            if dock is None:
                return
            if dock.isVisible():
                dock.hide()
            else:
                self.artifact_panel.show()
                dock.show()
                try:
                    dock.raiseDock()
                except Exception:
                    pass
                self._last_opened_section = "artifacts"
            self._sync_section_button_states_from_docks()
            self._save_panel_layout_state()
            return

        section_dock = self._section_docks.get("artifacts")
        if isinstance(section_dock, QtWidgets.QDockWidget):
            if section_dock.isVisible():
                section_dock.setVisible(False)
            else:
                self.artifact_panel.show()
                section_dock.setVisible(True)
                section_dock.raise_()
            self._save_panel_layout_state()
            return

        if isinstance(self.art_dock, QtWidgets.QDockWidget):
            if self.art_dock.isVisible():
                self.art_dock.setVisible(False)
                self._save_panel_layout_state()
                return
            self.artifact_panel.show()
            self.art_dock.setVisible(True)
        self._save_panel_layout_state()

    # ---------------- Metadata ----------------

    def _edit_metadata_for_current(self) -> None:
        if not self._current_path:
            QtWidgets.QMessageBox.information(self, "Metadata", "Select a file first.")
            return
        doric = self._loaded_files.get(self._current_path)
        if not doric:
            QtWidgets.QMessageBox.warning(self, "Metadata", "Current file is not loaded.")
            return
        if not doric.channels:
            QtWidgets.QMessageBox.warning(self, "Metadata", "No channels available for metadata editing.")
            return

        # existing per channel
        existing: Dict[str, Dict[str, str]] = {}
        for ch in doric.channels:
            existing[ch] = self._metadata_by_key.get((self._current_path, ch), {})

        defaults: Dict[str, str] = {}
        try:
            raw = self.settings.value("last_metadata_template", "", type=str)
            if raw:
                defaults = json.loads(raw)
        except Exception:
            defaults = {}

        dlg = MetadataDialog(channels=doric.channels, existing=existing, defaults=defaults, parent=self)
        if dlg.exec() != QtWidgets.QDialog.DialogCode.Accepted:
            return
        meta = dlg.get_metadata()
        for ch, md in meta.items():
            self._metadata_by_key[(self._current_path, ch)] = md
        try:
            if self._current_channel and self._current_channel in meta:
                self.settings.setValue("last_metadata_template", json.dumps(meta[self._current_channel]))
            elif meta:
                first = next(iter(meta.values()))
                self.settings.setValue("last_metadata_template", json.dumps(first))
        except Exception:
            pass

    # ---------------- Export (multi-file) ----------------

    def _export_origin_dir(self, selected_paths: List[str]) -> str:
        if selected_paths:
            d = os.path.dirname(selected_paths[0])
            if d and os.path.isdir(d):
                return d
        hint = self.file_panel.current_dir_hint()
        if hint and os.path.isdir(hint):
            return hint
        return ""

    def _export_start_dir(self, selected_paths: List[str]) -> str:
        origin_dir = self._export_origin_dir(selected_paths)
        last_dir = self.settings.value("last_save_dir", "", type=str)
        override = self.settings.value("last_save_dir_override", False, type=bool)

        def _valid(p: str) -> bool:
            return bool(p) and os.path.isdir(p)

        if override and _valid(last_dir):
            return last_dir
        if _valid(origin_dir):
            return origin_dir
        if _valid(last_dir):
            return last_dir
        return os.getcwd()

    def _remember_export_dir(self, out_dir: str, origin_dir: str) -> None:
        try:
            self.settings.setValue("last_save_dir", out_dir)
            out_norm = os.path.normcase(os.path.abspath(out_dir)) if out_dir else ""
            origin_norm = os.path.normcase(os.path.abspath(origin_dir)) if origin_dir else ""
            override = bool(out_norm) and (not origin_norm or out_norm != origin_norm)
            self.settings.setValue("last_save_dir_override", override)
        except Exception:
            pass

    def _process_trial_for_export(
        self,
        trial: LoadedTrial,
        params: ProcessingParams,
        export_selection: ExportSelection,
        manual_regions_sec: List[Tuple[float, float]],
        manual_exclude_regions_sec: List[Tuple[float, float]],
    ) -> ProcessedTrial:
        # "What you select is what you get": the previewed/primary output
        # (params.output_mode, from combo_output) is ALWAYS exported and is
        # always written first, followed by any additional selected outputs.
        modes: List[str] = []
        if export_selection.output:
            if params.output_mode in OUTPUT_MODES:
                modes.append(params.output_mode)
            for mode in export_selection.output_modes or []:
                mode = str(mode or "").strip()
                if mode in OUTPUT_MODES and mode not in modes:
                    modes.append(mode)
        if not modes:
            modes = [params.output_mode if params.output_mode in OUTPUT_MODES else OUTPUT_MODES[0]]

        ordered_modes = modes
        base_processed: Optional[ProcessedTrial] = None
        outputs: Dict[str, np.ndarray] = {}

        for mode in ordered_modes:
            mode_params = ProcessingParams.from_dict(params.to_dict())
            mode_params.output_mode = mode
            processed = self.processor.process_trial(
                trial=trial,
                params=mode_params,
                manual_regions_sec=manual_regions_sec,
                manual_exclude_regions_sec=manual_exclude_regions_sec,
                preview_mode=False,
            )
            if base_processed is None:
                base_processed = processed
            if processed.output is not None:
                outputs[str(processed.output_label or mode)] = np.asarray(processed.output, float)

        if base_processed is None:
            fallback_params = ProcessingParams.from_dict(params.to_dict())
            base_processed = self.processor.process_trial(
                trial=trial,
                params=fallback_params,
                manual_regions_sec=manual_regions_sec,
                manual_exclude_regions_sec=manual_exclude_regions_sec,
                preview_mode=False,
            )
        if export_selection.output:
            base_processed.outputs = outputs
        return base_processed

    def _export_selected_or_all(self) -> None:
        selected = self._selected_paths()
        if not selected:
            selected = self.file_panel.all_paths()
        if not selected:
            return

        auto_export = bool(self.param_panel.auto_export_enabled())
        origin_dir = self._export_origin_dir(selected)
        out_dir = ""
        if not auto_export:
            start_dir = self._export_start_dir(selected)
            out_dir = QtWidgets.QFileDialog.getExistingDirectory(self, "Select export folder", start_dir)
            if not out_dir:
                return
            self._remember_export_dir(out_dir, origin_dir)

        params = self.param_panel.get_params()
        export_selection = self.param_panel.export_selection()
        export_channel_names = [] if auto_export else self.param_panel.export_channel_names()
        export_trigger_names = self.param_panel.export_trigger_names()

        def _channels_for_export(doric: LoadedDoricFile) -> List[str]:
            if auto_export:
                return list(doric.channels)
            channels = [name for name in export_channel_names if name in doric.channels]
            if not channels:
                fallback = self._current_channel if (self._current_channel in doric.channels) else (doric.channels[0] if doric.channels else None)
                channels = [fallback] if fallback else []
            return [ch for ch in channels if ch]

        estimated_jobs = 0
        for path in selected:
            doric = self._loaded_files.get(path)
            if not doric:
                continue
            for ch in _channels_for_export(doric):
                sections = self._sections_by_key.get((path, ch), [])
                estimated_jobs += max(1, len(sections))
        progress_total = max(1, estimated_jobs * 3)
        progress_step = 0
        if estimated_jobs:
            self._begin_export_progress(progress_total, f"Exporting 0/{estimated_jobs} recording(s)...")

        # Process/export each selected file. Auto export writes beside each source
        # file and intentionally exports every analog channel with the same params.
        n_total = 0
        exported_dirs = set()
        for path in selected:
            doric = self._loaded_files.get(path)
            if not doric:
                continue
            channels = _channels_for_export(doric)
            path_out_dir = out_dir
            if auto_export:
                path_out_dir = os.path.dirname(path)
                if not path_out_dir or not os.path.isdir(path_out_dir):
                    path_out_dir = origin_dir if origin_dir and os.path.isdir(origin_dir) else os.getcwd()
            dio_names = [name for name in export_trigger_names if name in doric.trigger_by_name]
            if not export_selection.dio:
                dio_names = [None]
            elif not dio_names:
                dio_names = [self._current_trigger] if self._current_trigger else [None]

            for ch in channels:
                if not ch:
                    continue
                key = (path, ch)
                cutouts = self._cutout_regions_by_key.get(key, [])
                start_s, end_s = self._time_window_bounds()
                manual = self._clip_regions_to_window(self._manual_regions_by_key.get(key, []), start_s, end_s)
                manual_exclude = self._clip_regions_to_window(self._manual_exclude_by_key.get(key, []), start_s, end_s)
                meta = self._metadata_by_key.get(key, {})
                sections = self._sections_by_key.get(key, [])

                # Use all selected triggers for one export per channel
                # If current trigger is in dio_names, use it as primary for alignment.
                # Otherwise, pick first available as primary.
                primary_trigger = None
                if export_selection.dio and dio_names:
                    primary_trigger = (self._current_trigger if self._current_trigger in dio_names else (dio_names[0] if dio_names[0] else None))

                trial = doric.make_trial(ch, trigger_name=primary_trigger, trigger_names=(dio_names if export_selection.dio else None))
                trial = self._apply_time_window(trial)
                trial = self._apply_cutouts(trial, cutouts)

                def _export_one(proc: ProcessedTrial, suffix: str = "", params_used: Optional[ProcessingParams] = None) -> None:
                    nonlocal n_total, progress_step
                    proc = self._apply_cutouts_to_processed(proc, cutouts)
                    stem = safe_stem_from_metadata(path, ch, meta)
                    if suffix:
                        stem = f"{stem}_{suffix}"
                    csv_path = os.path.join(path_out_dir, f"{stem}.csv")
                    h5_path = os.path.join(path_out_dir, f"{stem}.h5")
                    export_params = params_used if params_used is not None else params
                    if estimated_jobs:
                        self._set_export_progress(progress_step, progress_total, f"Writing CSV: {stem}")
                    export_processed_csv(csv_path, proc, metadata=meta, selection=export_selection, params=export_params)
                    progress_step += 1
                    if estimated_jobs:
                        self._set_export_progress(progress_step, progress_total, f"Writing HDF5: {stem}")
                    export_processed_h5(h5_path, proc, metadata=meta, selection=export_selection, params=export_params)
                    progress_step += 1
                    exported_dirs.add(path_out_dir)
                    n_total += 1
                    if estimated_jobs:
                        self._set_export_progress(
                            progress_step,
                            progress_total,
                            f"Exported {n_total}/{estimated_jobs} recording(s)",
                        )

                try:
                    if sections:
                        for i, sec in enumerate(sections, start=1):
                            s0 = float(sec.get("start", 0.0))
                            s1 = float(sec.get("end", 0.0))
                            sec_trial = self._slice_trial(trial, s0, s1)
                            if sec_trial is None:
                                continue
                            sec_params = ProcessingParams.from_dict(sec.get("params", {})) if isinstance(sec.get("params"), dict) else params
                            if estimated_jobs:
                                self._set_export_progress(
                                    progress_step,
                                    progress_total,
                                    f"Processing {os.path.basename(path)} [{ch}] section {i}",
                                )
                            processed = self._process_trial_for_export(
                                trial=sec_trial,
                                params=sec_params,
                                export_selection=export_selection,
                                manual_regions_sec=manual,
                                manual_exclude_regions_sec=manual_exclude,
                            )
                            progress_step += 1
                            _export_one(processed, suffix=f"sec{i}_{s0:.2f}_{s1:.2f}", params_used=sec_params)
                    else:
                        if estimated_jobs:
                            self._set_export_progress(
                                progress_step,
                                progress_total,
                                f"Processing {os.path.basename(path)} [{ch}]",
                            )
                        processed = self._process_trial_for_export(
                            trial=trial,
                            params=params,
                            export_selection=export_selection,
                            manual_regions_sec=manual,
                            manual_exclude_regions_sec=manual_exclude,
                        )
                        progress_step += 1
                        _export_one(processed)
                except Exception as e:
                    QtWidgets.QMessageBox.warning(
                        self,
                        "Export error",
                        f"Failed export:\n{path} [{ch}] [{primary_trigger or 'no DIO'}]\n\n{e}",
                    )
        if auto_export:
            if len(exported_dirs) == 1:
                target = next(iter(exported_dirs))
            elif exported_dirs:
                target = f"{len(exported_dirs)} source folders"
            else:
                target = "source folders"
        else:
            target = out_dir
        message = f"Export complete: {n_total} recording(s) written to {target}"
        if estimated_jobs:
            self._finish_export_progress(progress_total, message)
        else:
            self._show_status_message(message)

        # optional: update post tab list by loading exported results? (user can load later)

    # ---------------- Postprocessing bridge ----------------

    def _postprocessing_bridge_paths(self, paths: Optional[List[str]] = None) -> List[str]:
        raw_paths = [str(p or "") for p in (paths or []) if str(p or "")]
        if not raw_paths:
            raw_paths = self._selected_paths()
        if not raw_paths:
            raw_paths = [self._current_path] if self._current_path else []
        out: List[str] = []
        seen: set[str] = set()
        for path in raw_paths:
            if not path or path in seen:
                continue
            seen.add(path)
            out.append(path)
        return out

    def _processed_trials_for_postprocessing_paths(self, paths: List[str]) -> List[ProcessedTrial]:
        out: List[ProcessedTrial] = []
        try:
            params = self.param_panel.get_params()
        except Exception:
            params = ProcessingParams()
        start_s, end_s = self._time_window_bounds()
        for p in paths:
            doric = self._loaded_files.get(p)
            if not doric:
                continue
            if self._current_channel and self._current_channel in doric.channels:
                ch = self._current_channel
            else:
                ch = doric.channels[0] if doric.channels else "AIN01"
            key = (p, ch)
            if key in self._last_processed:
                out.append(self._last_processed[key])
                continue
            try:
                trial = doric.make_trial(ch, trigger_name=self._current_trigger)
                trial = self._apply_time_window(trial)
                manual = self._clip_regions_to_window(self._manual_regions_by_key.get(key, []), start_s, end_s)
                manual_exclude = self._clip_regions_to_window(self._manual_exclude_by_key.get(key, []), start_s, end_s)
                proc = self.processor.process_trial(
                    trial,
                    params,
                    manual_regions_sec=manual,
                    manual_exclude_regions_sec=manual_exclude,
                    preview_mode=False,
                )
                cutouts = self._cutout_regions_by_key.get(key, [])
                proc = self._apply_cutouts_to_processed(proc, cutouts)
                self._last_processed[key] = proc
                out.append(proc)
            except Exception:
                pass
        return out

    def _send_dio_list_for_paths_to_postprocessing(self, paths: List[str]) -> None:
        dio: set[str] = set()
        for p in paths:
            f = self._loaded_files.get(p)
            if f:
                dio |= set(f.trigger_by_name.keys())
        self.post_tab.receive_dio_list(sorted(dio))

    @QtCore.Slot(list)
    def _send_preprocessing_paths_to_postprocessing(self, paths: List[str]) -> None:
        source_paths = self._postprocessing_bridge_paths(paths)
        processed = self._processed_trials_for_postprocessing_paths(source_paths)
        if not processed:
            self._show_status_message("No selected preprocessing file could be loaded into postprocessing.", 6000)
            return
        self.post_tab.receive_current_processed(processed)
        self._send_dio_list_for_paths_to_postprocessing(source_paths)
        first = processed[0]
        self.post_tab.set_current_source_label(
            os.path.basename(getattr(first, "path", "") or ""),
            str(getattr(first, "channel_id", "") or ""),
        )
        try:
            idx = self.tabs.indexOf(self.post_tab)
            if idx >= 0:
                self.tabs.setCurrentIndex(idx)
        except Exception:
            pass
        self._show_status_message(f"Loaded {len(processed)} preprocessing file(s) into postprocessing.", 6000)

    @QtCore.Slot()
    def _post_get_current_processed(self):
        paths = self._postprocessing_bridge_paths()
        out = self._processed_trials_for_postprocessing_paths(paths)
        self.post_tab.receive_current_processed(out)
        self._send_dio_list_for_paths_to_postprocessing(paths)

    @QtCore.Slot()
    def _post_get_current_dio_list(self):
        paths = self._postprocessing_bridge_paths()
        self._send_dio_list_for_paths_to_postprocessing(paths)

    @QtCore.Slot(str, str)
    def _post_get_dio_data_for_path(self, path: str, dio_name: str):
        """
        Returns (t_dio, y_dio) for the requested dio_name for a given *raw* path
        currently loaded/parsed in the cache.

        Fixes numpy array truth-value ambiguity by checking None/len explicitly.
        """
        f = self._loaded_files.get(path, None)

        if f is None:
            return

        trigger_map = getattr(f, "trigger_by_name", None)
        if not isinstance(trigger_map, dict) or dio_name not in trigger_map:
            return

        y_dio = np.asarray(trigger_map[dio_name], float)
        if y_dio.size == 0:
            return

        t_map = getattr(f, "trigger_time_by_name", None)
        t_dio = None
        if isinstance(t_map, dict):
            t_dio = t_map.get(dio_name)
        if t_dio is None and getattr(f, "digital_time", None) is not None and dio_name in getattr(f, "digital_by_name", {}):
            t_dio = f.digital_time
        if t_dio is None:
            # Fallback: use any analog channel timebase with matching length.
            for t_candidate in getattr(f, "time_by_channel", {}).values():
                arr = np.asarray(t_candidate, float)
                if arr.size == y_dio.size:
                    t_dio = arr
                    break
        if t_dio is None:
            return
        t_dio = np.asarray(t_dio, float)
        if t_dio.size == 0:
            return

        # Ensure same length.
        n = min(t_dio.size, y_dio.size)
        t_dio = t_dio[:n]
        y_dio = y_dio[:n]

        self.post_tab.receive_dio_data(path, dio_name, t_dio, y_dio)
        return

    # ---------------- Drag and drop ----------------

    def dragEnterEvent(self, event) -> None:
        mime = event.mimeData()
        if mime and mime.hasUrls() and any(u.isLocalFile() for u in mime.urls()):
            event.acceptProposedAction()
        else:
            event.ignore()

    def dragMoveEvent(self, event) -> None:
        mime = event.mimeData()
        if mime and mime.hasUrls() and any(u.isLocalFile() for u in mime.urls()):
            event.acceptProposedAction()
        else:
            event.ignore()

    def dropEvent(self, event) -> None:
        mime = event.mimeData()
        if not mime or not mime.hasUrls():
            event.ignore()
            return
        paths = self._expand_dropped_url_paths(list(mime.urls()))
        if not paths:
            event.ignore()
            self._show_status_message(
                "Drop ignored: no .doric / .h5 / .hdf5 / .csv files found in the dropped item(s).",
                6000,
            )
            return
        event.acceptProposedAction()
        first_dir = next(
            (os.path.dirname(p) for p in paths if os.path.isfile(p)), ""
        )
        if first_dir:
            self.settings.setValue("last_open_dir", first_dir)
        self._push_recent_preprocessing_files(paths)
        self._handle_drop(paths)
        self._show_status_message(
            f"Loaded {len(paths)} file(s) via drag-and-drop.", 5000,
        )

    def _handle_drop(self, paths: List[str]) -> None:
        doric_paths: List[str] = []
        processed: List[ProcessedTrial] = []
        pre_active = bool(hasattr(self, "tabs") and self.tabs.currentWidget() is self.pre_tab)

        for p in paths:
            if not p:
                continue
            ext = os.path.splitext(p)[1].lower()
            if ext == ".doric":
                doric_paths.append(p)
                continue
            if ext == ".csv":
                if pre_active:
                    doric_paths.append(p)
                    continue
                trial = self._load_processed_csv(p)
                if trial is not None:
                    processed.append(trial)
                continue
            if ext in (".h5", ".hdf5"):
                if pre_active:
                    # On preprocessing tab, treat dropped H5 as a raw-input candidate.
                    # _add_files() will load Doric H5 natively and falls back to processed H5 import.
                    doric_paths.append(p)
                    continue
                trial = self._load_processed_h5(p)
                if trial is not None:
                    processed.append(trial)
                else:
                    doric_paths.append(p)

        if doric_paths:
            self._add_files(doric_paths)
        if processed:
            self.post_tab.append_processed(processed)

    def _load_processed_csv(self, path: str) -> Optional[ProcessedTrial]:
        return load_processed_csv(path)

    def _load_processed_h5(self, path: str) -> Optional[ProcessedTrial]:
        return load_processed_h5(path)

    def _processed_trial_to_loaded_doric(self, processed: ProcessedTrial) -> Optional[LoadedDoricFile]:
        t = np.asarray(processed.time if processed.time is not None else np.array([], float), float)
        if t.size < 2:
            return None

        raw_sig = np.asarray(
            processed.raw_signal if processed.raw_signal is not None else np.array([], float),
            float,
        )
        raw_ref = np.asarray(
            processed.raw_reference if processed.raw_reference is not None else np.array([], float),
            float,
        )
        out = np.asarray(processed.output if processed.output is not None else np.array([], float), float)

        if raw_sig.size != t.size:
            if out.size == t.size:
                raw_sig = out.copy()
            else:
                return None
        if raw_ref.size != t.size or not np.isfinite(raw_ref).any():
            # Keep preprocessing numerically stable even if original H5 has no raw_405.
            raw_ref = raw_sig.copy()

        if not np.isfinite(raw_sig).any():
            return None

        channel = str(processed.channel_id or "AIN01").strip() or "AIN01"
        dio_map: Dict[str, np.ndarray] = {}
        dio_time_map: Dict[str, np.ndarray] = {}
        digital_time: Optional[np.ndarray] = None
        dio = np.asarray(processed.dio, float) if processed.dio is not None else np.array([], float)
        if dio.size == t.size:
            dio_name = str(processed.dio_name or "DIO_import").strip() or "DIO_import"
            digital_time = t.copy()
            dio_map[dio_name] = dio.copy()
            dio_time_map[dio_name] = t.copy()

        return LoadedDoricFile(
            path=str(processed.path or ""),
            channels=[channel],
            time_by_channel={channel: t.copy()},
            signal_by_channel={channel: raw_sig.copy()},
            reference_by_channel={channel: raw_ref.copy()},
            digital_time=digital_time,
            digital_by_name={k: v.copy() for k, v in dio_map.items()},
            trigger_time_by_name={k: v.copy() for k, v in dio_time_map.items()},
            trigger_by_name={k: v.copy() for k, v in dio_map.items()},
        )

    def _load_processed_h5_as_pre_file(self, path: str) -> Optional[LoadedDoricFile]:
        processed = self._load_processed_h5(path)
        if processed is None:
            return None
        loaded = self._processed_trial_to_loaded_doric(processed)
        if loaded is None:
            return None
        loaded.path = path
        return loaded

    def closeEvent(self, event):
        try:
            self.post_tab.mark_app_closing()
        except Exception:
            pass
        try:
            current = self.tabs.currentWidget() if hasattr(self, "tabs") else None
            if current is self.pre_tab:
                # Closing on preprocessing: capture the live preprocessing dock topology.
                self._store_pre_main_dock_snapshot()
            else:
                # Closing on postprocessing: preprocessing docks are hidden by tab switch.
                self._persist_hidden_preprocessing_layout_state()
        except Exception:
            pass
        try:
            # Persist post layout from live state or cached tab-switch state without
            # overwriting it with preprocessing topology.
            self.post_tab.persist_layout_state_snapshot()
        except Exception:
            pass
        try:
            self.post_tab._on_about_to_quit()
        except Exception:
            pass
        self._save_panel_layout_state()
        self._save_panel_config_json()
        self._save_settings()
        try:
            self.settings.sync()
        except Exception:
            pass
        super().closeEvent(event)


def _install_global_excepthook() -> None:
    """Log unhandled Python exceptions and show a dialog instead of letting
    them abort the process. PySide6 terminates the app on an uncaught exception
    raised inside a Qt slot; this keeps the app alive and surfaces the cause."""
    import traceback

    def _hook(exc_type, exc_value, exc_tb):
        if issubclass(exc_type, KeyboardInterrupt):
            sys.__excepthook__(exc_type, exc_value, exc_tb)
            return
        text = "".join(traceback.format_exception(exc_type, exc_value, exc_tb))
        try:
            logging.getLogger("pyber").error("Unhandled exception:\n%s", text)
        except Exception:
            pass
        try:
            with open(_crash_log_path(), "a", encoding="utf-8") as fh:
                fh.write("\n=== Unhandled exception ===\n")
                fh.write(text)
        except Exception:
            pass
        try:
            sys.stderr.write(text)
        except Exception:
            pass
        try:
            if QtWidgets.QApplication.instance() is not None:
                box = QtWidgets.QMessageBox()
                box.setIcon(QtWidgets.QMessageBox.Icon.Critical)
                box.setWindowTitle("pyBer - unexpected error")
                box.setText("An unexpected error occurred but pyBer kept running.")
                box.setInformativeText(f"{exc_type.__name__}: {exc_value}")
                box.setDetailedText(text)
                box.exec()
        except Exception:
            pass

    sys.excepthook = _hook


def main() -> None:
    pg.setConfigOptions(antialias=False)
    smoke_test = str(os.environ.get("PYBER_SMOKE_TEST", "")).strip().lower() in {"1", "true", "yes", "on"}
    _set_windows_app_user_model_id()
    app = QtWidgets.QApplication([])
    _install_global_excepthook()
    apply_app_palette(app, "dark")
    spinbox_scrubber = install_spinbox_scrubbers(app)
    _set_qt_application_icon(app)
    splash = None
    if not smoke_test:
        try:
            icon_path = _pyber_splash_path()
            if os.path.isfile(icon_path):
                pix = QtGui.QPixmap(icon_path)
                if not pix.isNull():
                    splash = QtWidgets.QSplashScreen(pix, QtCore.Qt.WindowType.WindowStaysOnTopHint)
                    _set_qt_window_icon(splash)
                    splash.show()
                    app.processEvents(QtCore.QEventLoop.ProcessEventsFlag.AllEvents)
        except Exception:
            splash = None
    w = MainWindow()
    spinbox_scrubber.scan(w)

    if smoke_test:
        try:
            w.show()
            app.processEvents(QtCore.QEventLoop.ProcessEventsFlag.AllEvents)
            if hasattr(w, "tabs") and hasattr(w, "post_tab"):
                idx = w.tabs.indexOf(w.post_tab)
                if idx >= 0:
                    w.tabs.setCurrentIndex(idx)
                for _ in range(8):
                    app.processEvents(QtCore.QEventLoop.ProcessEventsFlag.AllEvents)
            try:
                if hasattr(w, "post_tab") and hasattr(w.post_tab, "ensure_section_popups_initialized"):
                    w.post_tab.ensure_section_popups_initialized()
            except Exception:
                pass
            for _ in range(8):
                app.processEvents(QtCore.QEventLoop.ProcessEventsFlag.AllEvents)
        finally:
            try:
                w.close()
            except Exception:
                pass
            app.processEvents(QtCore.QEventLoop.ProcessEventsFlag.AllEvents)
        return

    w.show()
    # On Windows, re-applying the icon after show() forces the taskbar entry
    # to refresh - the OS otherwise caches whatever icon was active at the
    # moment the window was first realized.
    _set_qt_window_icon(w)
    # Belt-and-suspenders: send WM_SETICON directly to the HWND so the Windows
    # taskbar and alt-tab thumbnail definitely pick up pyBer.ico even when
    # running under python.exe in dev mode.
    _force_windows_taskbar_icon(w)
    if splash is not None:
        splash.finish(w)
    app.exec()


if __name__ == "__main__":
    main()
