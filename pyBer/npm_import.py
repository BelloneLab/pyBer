"""Neurophotometrics (NPM) raw CSV import for pyBer preprocessing.

What an NPM recording looks like
--------------------------------
Neurophotometrics FP3001 / FP3002 systems acquire through Bonsai and write one
CSV per session. Each row is one camera frame. The excitation LEDs are strobed
in turn (415 nm isosbestic, 470 nm for green sensors, 560 nm for red sensors),
so the channels are interleaved down the rows instead of sitting in separate
columns. A state column says which LED lit each frame:

* ``LedState`` (FP3002) or ``Flags`` (older FP3001 workflows). The three
  lowest bits carry one flag per LED: 1 = 415 nm, 2 = 470 nm, 4 = 560 nm.
  Higher bits are digital lines. Legacy ``Flags`` values 17/18 carry a constant
  bit 16, and 273/274 add a TTL on bit 256.
* The first frame is often an initialisation frame with all three LED bits set
  (``7``, or ``23`` with a TTL high) or with none (``0``, ``16``). It is not a
  measurement and is dropped.
* Every other numeric column is a region of interest drawn on the camera
  image: ``Region0G`` (green emission), ``Region1R`` (red emission), or
  ``G0`` / ``R0`` in newer workflows.
* Timestamps: ``SystemTimestamp`` (FP3002 hardware clock, seconds),
  ``Timestamp`` (seconds), ``ComputerTimestamp`` / ``msTimestamp`` (computer
  clock, milliseconds, with Windows timer jitter of about 16 ms).

Two layouts are merged automatically:

* split files, one per LED, whose names differ only by the wavelength
  (``FiberData415.csv`` + ``FiberData470.csv``);
* header-less event files written next to the photometry file
  (``*_stimuli.csv``, ``ttls.csv``, ``*_ts0.csv``). Their first column is a
  time on one of the photometry clocks; the optional second column is a label
  (event name, numeric code, or True/False line state).

What pyBer builds from it
-------------------------
One pyBer channel per region column:

* green or unlabelled region: signal = 470 nm frames, reference = 415 nm frames
* red region: signal = 560 nm frames, reference = 415 nm frames

A frame lit by the reference LED together with a signal LED is excluded from
both, because its emission mixes the two excitations. Frames lit by 470 and
560 nm together (``LedState`` 6) remain valid: green regions read the 470 nm
response and red regions the 560 nm response.

The signal frames define the channel's time axis, measured from the first LED
frame (the absolute origin is kept in ``metadata["npm_time_origin_s"]``). The
415 nm frames are taken at other instants (half an LED cycle away with two
LEDs, a third with three), so the reference is linearly interpolated onto the
signal frame times instead of being paired by index. With two LEDs at 20 Hz
each, pairing shifts the reference by 25 ms, which leaves about 31 % of a 2 Hz
motion component after subtraction; linear interpolation removes the shift at
the cost of a mild low-pass (gain cos(pi f / fs), residual 5 % at 2 Hz). On six
real NPM recordings the band-passed signal/reference R^2 was higher with
interpolation in 9 of 11 channels. Signal samples outside the reference time
span are trimmed instead of extrapolated.

Fully header-less photometry files (stock Bonsai CsvWriter, no LED state
column) are not recognised: which row belongs to which LED cannot be known from
the file, so they still go through the manual CSV column mapping.
"""
from __future__ import annotations

import csv
import functools
import os
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from analysis_core import LoadedDoricFile


# =============================================================================
# Configuration (format conventions and import choices)
# =============================================================================

#: LED wavelength (nm) -> bit in the Flags / LedState word.
NPM_LED_BITS: Dict[int, int] = {415: 0b001, 470: 0b010, 560: 0b100}
#: Mask of the three excitation bits. Everything above is a digital line.
NPM_EXCITATION_MASK = 0b111
#: Isosbestic reference LED.
NPM_REFERENCE_NM = 415
#: Signal LED for each region emission colour ("" = colour not encoded in the name).
NPM_SIGNAL_NM_BY_COLOR: Dict[str, int] = {"G": 470, "R": 560, "": 470}
#: Highest state bit scanned for TTL lines (bits 3..15 = values 8..32768).
NPM_MAX_TTL_BIT = 15

#: Names of the LED state column, normalised (lower case, alphanumerics only).
NPM_STATE_COLUMNS: Tuple[str, ...] = ("ledstate", "flags")
#: Time columns in order of preference for the pyBer time axis. The FP3002
#: hardware clock has <5 us precision; the computer clock jitters by ~16 ms.
NPM_TIME_PREFERENCE: Tuple[str, ...] = (
    "systemtimestamp",
    "timestamp",
    "computertimestamp",
    "mstimestamp",
)
#: Time columns stored in milliseconds.
NPM_MS_TIME_COLUMNS = frozenset({"computertimestamp", "mstimestamp"})
#: Columns written around the regions. This is a closed set in Bonsai's
#: photometry writers (corroborated by NeuroConv's NPM interface); anything
#: else that is numeric is a region.
NPM_NON_REGION_COLUMNS = frozenset({
    "framecounter",
    "timestamp",
    "systemtimestamp",
    "computertimestamp",
    "mstimestamp",
    "triggerevents",
    "flags",
    "ledstate",
    "stimulation",
    "output0",
    "output1",
    "input0",
    "input1",
})
#: Any column whose name ends like a clock or a frame counter is not a region.
NPM_NON_REGION_SUFFIXES: Tuple[str, ...] = ("timestamp", "counter")
#: Digital line columns kept as pyBer triggers when they change state.
NPM_DIGITAL_COLUMNS = frozenset({"input0", "input1", "output0", "output1", "stimulation", "triggerevents"})

#: Wavelength tokens that tie per-LED split files together (FiberData415.csv ...).
NPM_SPLIT_TOKEN_RE = re.compile(r"(?<!\d)(415|470|560)(?!\d)")
#: Frames read from the top of a candidate split file to confirm it holds one LED.
NPM_SPLIT_HEAD_ROWS = 12

#: Width of the trigger pulse drawn for an instantaneous event (seconds). Onsets
#: (rising edges) are exact to one frame; the width only keeps the pulse visible
#: after optional decimation down to 10 Hz. It is not an event duration.
NPM_EVENT_PULSE_S = 0.1
#: Event files bigger than this are not treated as sidecar event lists.
NPM_EVENT_FILE_MAX_BYTES = 5 * 1024 * 1024
#: Maximum number of columns for a header-less event file.
NPM_EVENT_MAX_COLUMNS = 4
#: More distinct labels than this means the file is a data table, not events.
NPM_EVENT_MAX_LABELS = 64
#: Minimum fraction of an event file's times that must fall within the
#: recording span on one clock for the file to be attached to that recording.
NPM_EVENT_MIN_FRACTION_IN_RANGE = 0.5
#: Relative tolerance on the recording span when matching event clocks.
NPM_EVENT_RANGE_TOLERANCE = 0.01

#: Label sets read as the on/off state of a single line.
_STATE_LABEL_SETS: Tuple[frozenset, ...] = (
    frozenset({"true", "false"}),
    frozenset({"on", "off"}),
    frozenset({"high", "low"}),
    frozenset({"1", "0"}),
)
_STATE_HIGH_LABELS = frozenset({"true", "on", "high", "1"})

_ENCODINGS: Tuple[str, ...] = ("utf-8-sig", "utf-8", "cp1252")

#: Region name patterns that encode the emission colour.
_REGION_COLOR_PATTERNS: Tuple[re.Pattern, ...] = (
    re.compile(r"^(?:region|channel|roi|fiber|fibre)?[\s_\-]*\d+[\s_\-]*([gr])$", re.IGNORECASE),
    re.compile(r"^([gr])[\s_\-]*\d+$", re.IGNORECASE),
)


# =============================================================================
# Small helpers
# =============================================================================

def _norm(value: object) -> str:
    """Lower-case a header and keep only letters and digits ("Region 0-G" -> "region0g")."""
    return "".join(ch.lower() for ch in str(value or "") if ch.isalnum())


def _parse_float(text: object) -> float:
    """Parse one CSV cell as float, returning NaN for blanks and text."""
    s = str(text or "").strip()
    if not s:
        return np.nan
    try:
        return float(s)
    except ValueError:
        return np.nan


def _file_signature(path: str) -> Optional[Tuple[str, int, int]]:
    """Return (normalised path, mtime_ns, size) used as a cache key, or None."""
    try:
        st = os.stat(path)
    except OSError:
        return None
    return os.path.normcase(os.path.abspath(path)), int(st.st_mtime_ns), int(st.st_size)


@functools.lru_cache(maxsize=512)
def _read_first_rows_cached(signature: Tuple[str, int, int], n_rows: int) -> Tuple[Tuple[str, ...], ...]:
    """Read the first non-empty CSV rows of a file (cached by path, mtime, size)."""
    path = signature[0]
    for encoding in _ENCODINGS:
        try:
            rows: List[Tuple[str, ...]] = []
            with open(path, "r", newline="", encoding=encoding) as handle:
                for row in csv.reader(handle):
                    cells = tuple(str(cell or "").strip() for cell in row)
                    if not any(cells):
                        continue
                    rows.append(cells)
                    if len(rows) >= n_rows:
                        break
            return tuple(rows)
        except UnicodeDecodeError:
            continue
        except (OSError, csv.Error):
            return tuple()
    return tuple()


def _first_rows(path: str, n_rows: int = 3) -> List[Tuple[str, ...]]:
    """First non-empty rows of a CSV file; empty list when unreadable."""
    if not path or os.path.splitext(str(path))[1].lower() != ".csv":
        return []
    signature = _file_signature(path)
    if signature is None:
        return []
    return list(_read_first_rows_cached(signature, int(n_rows)))


def _state_column_index(headers: Sequence[str]) -> int:
    """Index of the Flags / LedState column, or -1."""
    norms = [_norm(h) for h in headers]
    for wanted in NPM_STATE_COLUMNS:
        if wanted in norms:
            return norms.index(wanted)
    return -1


def _is_time_column_name(norm: str) -> bool:
    return norm in NPM_TIME_PREFERENCE or norm.endswith("timestamp")


def _region_candidate_indices(headers: Sequence[str]) -> List[int]:
    """Indices of columns that may hold region data (named, not a known non-region column)."""
    out: List[int] = []
    for idx, header in enumerate(headers):
        norm = _norm(header)
        if not norm:
            continue  # blank header cells hold counters / copies of the state
        if norm in NPM_NON_REGION_COLUMNS or norm.endswith(NPM_NON_REGION_SUFFIXES):
            continue
        out.append(idx)
    return out


def region_color(name: str) -> str:
    """Return "G", "R" or "" from a region column name.

    ``Region0G`` / ``G0`` -> "G"; ``Region1R`` / ``R1`` -> "R"; names that spell
    out green / red are recognised too. Anything else returns "" and is read as
    a green-channel region (470 nm signal), the most common configuration.
    """
    text = str(name or "").strip()
    for pattern in _REGION_COLOR_PATTERNS:
        match = pattern.match(text)
        if match:
            return match.group(1).upper()
    lowered = text.lower()
    if "green" in lowered:
        return "G"
    if "red" in lowered:
        return "R"
    return ""


def _unique_name(name: str, used: set) -> str:
    base = str(name or "Event").strip() or "Event"
    out = base
    i = 2
    while out in used:
        out = f"{base}_{i}"
        i += 1
    used.add(out)
    return out


# =============================================================================
# Detection and companion roles (cheap: header only)
# =============================================================================

def is_npm_photometry_csv(path: str) -> bool:
    """True when a CSV header looks like a Neurophotometrics photometry table.

    Required: a ``Flags`` or ``LedState`` column, at least one timestamp
    column, and at least one candidate region column.
    """
    rows = _first_rows(path, 1)
    if not rows:
        return False
    headers = rows[0]
    if _state_column_index(headers) < 0:
        return False
    if not any(_is_time_column_name(_norm(h)) for h in headers):
        return False
    return bool(_region_candidate_indices(headers))


def _single_led_in_head(path: str) -> Optional[int]:
    """Wavelength of the only LED lighting the first frames, or None.

    Per-LED split files hold one LED each (after the startup frame). A file
    whose first frames alternate between LEDs is a complete recording.
    """
    rows = _first_rows(path, NPM_SPLIT_HEAD_ROWS + 2)
    if len(rows) < 3:
        return None
    state_idx = _state_column_index(rows[0])
    if state_idx < 0:
        return None
    leds = set()
    for row in rows[2:]:  # skip header and the startup frame
        value = _parse_float(row[state_idx]) if state_idx < len(row) else np.nan
        if np.isfinite(value) and (int(value) & NPM_EXCITATION_MASK):
            leds.add(int(value) & NPM_EXCITATION_MASK)
    if len(leds) != 1:
        return None
    bits = leds.pop()
    return next((nm for nm, bit in NPM_LED_BITS.items() if bit == bits), None)


def _split_group(path: str) -> List[Tuple[int, str]]:
    """Return [(wavelength_nm, path)] for per-LED split files of one recording.

    Files belong together when their names differ only by the wavelength token
    (415 / 470 / 560), all of them are NPM photometry tables, and each one is
    lit only by the LED its name states. The last condition keeps complete
    recordings whose names merely contain such a number (an animal ID such as
    ``mouse415.csv`` next to ``mouse470.csv``) apart. A lone file returns [].
    """
    folder, name = os.path.split(os.path.abspath(path))
    stem, ext = os.path.splitext(name)
    matches = list(NPM_SPLIT_TOKEN_RE.finditer(stem))
    if not matches:
        return []
    match = matches[-1]
    members: List[Tuple[int, str]] = []
    for nm in sorted(NPM_LED_BITS):
        candidate_name = stem[: match.start()] + str(nm) + stem[match.end():] + ext
        candidate = os.path.join(folder, candidate_name)
        if os.path.normcase(candidate) == os.path.normcase(os.path.abspath(path)):
            members.append((nm, os.path.abspath(path)))
        elif os.path.isfile(candidate) and is_npm_photometry_csv(candidate):
            members.append((nm, candidate))
    if len(members) < 2:
        return []
    if any(_single_led_in_head(member) != nm for nm, member in members):
        return []
    return members


def _split_primary(members: List[Tuple[int, str]]) -> str:
    """The file that represents a split group: the lowest signal wavelength."""
    for nm, member in members:
        if nm != NPM_REFERENCE_NM:
            return member
    return members[0][1]


def _event_file_kind(path: str) -> str:
    """Classify a CSV as an NPM sidecar event list: "headerless", "headed" or ""."""
    try:
        if os.path.getsize(path) > NPM_EVENT_FILE_MAX_BYTES:
            return ""
    except OSError:
        return ""
    rows = _first_rows(path, 2)
    if not rows:
        return ""
    first = rows[0]
    if np.isfinite(_parse_float(first[0])):
        return "headerless" if len(first) <= NPM_EVENT_MAX_COLUMNS else ""
    norms = [_norm(h) for h in first]
    if _state_column_index(first) >= 0:
        return ""
    if len(first) <= 8 and any(_is_time_column_name(n) for n in norms) and len(rows) >= 2:
        return "headed"
    return ""


@functools.lru_cache(maxsize=256)
def _folder_npm_photometry_cached(folder_key: Tuple[str, int]) -> Tuple[str, ...]:
    folder = folder_key[0]
    try:
        names = sorted(os.listdir(folder))
    except OSError:
        return tuple()
    out = []
    for name in names:
        full = os.path.join(folder, name)
        if name.lower().endswith(".csv") and os.path.isfile(full) and is_npm_photometry_csv(full):
            out.append(full)
    return tuple(out)


def _folder_npm_photometry(folder: str) -> List[str]:
    """NPM photometry CSVs in a folder (cached until the folder changes)."""
    try:
        mtime = int(os.stat(folder).st_mtime_ns)
    except OSError:
        return []
    return list(_folder_npm_photometry_cached((os.path.normcase(os.path.abspath(folder)), mtime)))


def npm_csv_role(path: str) -> str:
    """Role of a CSV in a Neurophotometrics session.

    Returns
    -------
    "photometry"       a photometry table to list and load
    "split_companion"  a per-LED split file merged into its group's primary file
    "event"            a sidecar event list attached to the photometry file
    ""                 not an NPM file
    """
    if not path or os.path.splitext(str(path))[1].lower() != ".csv" or not os.path.isfile(path):
        return ""
    if is_npm_photometry_csv(path):
        members = _split_group(path)
        if members and os.path.normcase(_split_primary(members)) != os.path.normcase(os.path.abspath(path)):
            return "split_companion"
        return "photometry"
    if _event_file_kind(path) and _folder_npm_photometry(os.path.dirname(os.path.abspath(path))):
        return "event"
    return ""


def npm_primary_for(path: str) -> str:
    """For a split companion, the primary file it is merged into ("" otherwise)."""
    members = _split_group(path)
    return _split_primary(members) if members else ""


# =============================================================================
# Reading frames
# =============================================================================

@dataclass
class _NpmFrames:
    """Raw frame table of one NPM file (values untouched, as read)."""

    path: str
    state_name: str
    state: np.ndarray                      # int64 LED/TTL word per frame
    times: Dict[str, np.ndarray]           # time column name -> raw values
    regions: Dict[str, np.ndarray]         # region column name -> values
    digital: Dict[str, np.ndarray] = field(default_factory=dict)


def _read_frames(path: str) -> _NpmFrames:
    """Read one NPM photometry CSV into numeric arrays."""
    import pandas as pd

    rows = _first_rows(path, 1)
    if not rows:
        raise ValueError("The file has no header row.")
    headers = list(rows[0])
    state_idx = _state_column_index(headers)
    if state_idx < 0:
        raise ValueError("No Flags or LedState column was found.")

    last_error: Optional[Exception] = None
    df = None
    for encoding in _ENCODINGS:
        try:
            df = pd.read_csv(
                path,
                header=None,
                skiprows=1,
                names=list(range(len(headers))),
                index_col=False,
                encoding=encoding,
                skip_blank_lines=True,
                on_bad_lines="skip",
                low_memory=False,
            )
            break
        except UnicodeDecodeError as exc:
            last_error = exc
            continue
    if df is None:
        raise ValueError(f"Could not decode the CSV ({last_error}).")

    def numeric(idx: int) -> np.ndarray:
        return pd.to_numeric(df[idx], errors="coerce").to_numpy(dtype=float)

    state_f = numeric(state_idx)
    times: Dict[str, np.ndarray] = {}
    regions: Dict[str, np.ndarray] = {}
    digital: Dict[str, np.ndarray] = {}
    used_regions: set = set()
    for idx, header in enumerate(headers):
        norm = _norm(header)
        if not norm or idx == state_idx:
            continue
        if _is_time_column_name(norm):
            values = numeric(idx)
            if np.sum(np.isfinite(values)) >= 2:
                times[header] = values
        elif norm in NPM_DIGITAL_COLUMNS:
            digital[header] = numeric(idx)
    for idx in _region_candidate_indices(headers):
        values = numeric(idx)
        if values.size == 0 or np.mean(np.isfinite(values)) < 0.5:
            continue  # text or mostly empty column: not a region
        regions[_unique_name(headers[idx], used_regions)] = values

    if not times:
        raise ValueError("No usable timestamp column was found.")
    if not regions:
        raise ValueError("No numeric region column was found.")
    finite_state = np.isfinite(state_f)
    state = np.zeros(state_f.size, dtype=np.int64)
    state[finite_state] = np.rint(state_f[finite_state]).astype(np.int64)
    state[~finite_state] = -1  # marks unreadable rows, dropped later
    return _NpmFrames(
        path=os.path.abspath(path),
        state_name=headers[state_idx],
        state=state,
        times=times,
        regions=regions,
        digital=digital,
    )


def _primary_time_name(times: Dict[str, np.ndarray]) -> str:
    """Pick the most precise clock present (hardware first, computer clock last)."""
    by_norm = {_norm(name): name for name in times}
    for wanted in NPM_TIME_PREFERENCE:
        if wanted in by_norm:
            return by_norm[wanted]
    return next(iter(times))


def _seconds_scale(name: str, values: np.ndarray) -> float:
    """Factor converting a time column to seconds.

    Known millisecond columns are scaled by 1e-3. The median frame interval is
    then checked so that a mislabelled column is still read correctly: frame
    intervals above 2 "seconds" can only be milliseconds.
    """
    scale = 1e-3 if _norm(name) in NPM_MS_TIME_COLUMNS else 1.0
    finite = values[np.isfinite(values)]
    if finite.size >= 3:
        diffs = np.diff(np.sort(finite))
        diffs = diffs[diffs > 0]
        if diffs.size:
            dt = float(np.median(diffs)) * scale
            if scale == 1.0 and dt > 2.0:
                scale = 1e-3
            elif scale == 1e-3 and dt < 1e-4:
                scale = 1.0
    return scale


def _merge_frames(parts: List[_NpmFrames]) -> _NpmFrames:
    """Merge per-LED split files into one frame table sorted by time."""
    if len(parts) == 1:
        return parts[0]
    first = parts[0]
    time_names = [name for name in first.times if all(name in p.times for p in parts)]
    region_names = [name for name in first.regions if all(name in p.regions for p in parts)]
    digital_names = [name for name in first.digital if all(name in p.digital for p in parts)]
    if not time_names or not region_names:
        raise ValueError("Split NPM files do not share the same timestamp and region columns.")
    primary = _primary_time_name({n: first.times[n] for n in time_names})

    def cat(getter) -> np.ndarray:
        return np.concatenate([np.asarray(getter(p), float) for p in parts])

    t = cat(lambda p: p.times[primary])
    order = np.argsort(t, kind="stable")
    t_sorted = t[order]
    # The startup frame is written into every split file: keep one copy.
    keep = np.ones(t_sorted.size, dtype=bool)
    keep[1:] = t_sorted[1:] != t_sorted[:-1]
    sel = order[keep]
    return _NpmFrames(
        path=first.path,
        state_name=first.state_name,
        state=np.concatenate([p.state for p in parts])[sel],
        times={name: cat(lambda p, n=name: p.times[n])[sel] for name in time_names},
        regions={name: cat(lambda p, n=name: p.regions[n])[sel] for name in region_names},
        digital={name: cat(lambda p, n=name: p.digital[n])[sel] for name in digital_names},
    )


# =============================================================================
# Event sidecar files
# =============================================================================

@dataclass
class _EventFile:
    path: str
    times_raw: np.ndarray
    labels: Optional[List[str]]
    time_column: str = ""     # headed files: name of their time column


def _read_event_file(path: str, kind: str) -> Optional[_EventFile]:
    """Read a sidecar event list (header-less or headed)."""
    rows: List[List[str]] = []
    for encoding in _ENCODINGS:
        try:
            with open(path, "r", newline="", encoding=encoding) as handle:
                rows = [[str(c or "").strip() for c in row] for row in csv.reader(handle) if any(row)]
            break
        except UnicodeDecodeError:
            continue
        except (OSError, csv.Error):
            return None
    if not rows:
        return None

    time_idx = 0
    label_idx = 1
    time_column = ""
    if kind == "headed":
        headers = rows[0]
        rows = rows[1:]
        norms = [_norm(h) for h in headers]
        time_candidates = [i for i, n in enumerate(norms) if _is_time_column_name(n)]
        if not time_candidates:
            return None
        ranked = sorted(
            time_candidates,
            key=lambda i: NPM_TIME_PREFERENCE.index(norms[i]) if norms[i] in NPM_TIME_PREFERENCE else 99,
        )
        time_idx = ranked[0]
        time_column = headers[time_idx]
        label_idx = -1
        for i in range(len(headers)):
            if i in time_candidates:
                continue
            values = {r[i] for r in rows if i < len(r)}
            if len(values) > 1:
                label_idx = i
                break

    times: List[float] = []
    labels: List[str] = []
    for row in rows:
        if time_idx >= len(row):
            continue
        value = _parse_float(row[time_idx])
        if not np.isfinite(value):
            continue
        times.append(value)
        labels.append(row[label_idx] if 0 <= label_idx < len(row) else "")
    if not times:
        return None
    has_labels = any(labels)
    if has_labels and len(set(labels)) > NPM_EVENT_MAX_LABELS:
        return None  # a continuous second column: a data table, not an event list
    return _EventFile(
        path=os.path.abspath(path),
        times_raw=np.asarray(times, float),
        labels=labels if has_labels else None,
        time_column=time_column,
    )


def _match_event_clock(
    event: _EventFile,
    frames: _NpmFrames,
    kept: np.ndarray,
) -> Tuple[str, float]:
    """Find which photometry clock an event file uses.

    A headed file naming one of the photometry time columns uses that clock.
    Otherwise the clock is the one whose span over the recording contains the
    largest fraction of the event times (raw units, before any conversion).
    Returns (time column name, fraction of events inside), or ("", 0.0).
    """
    if event.time_column:
        by_norm = {_norm(n): n for n in frames.times}
        name = by_norm.get(_norm(event.time_column), "")
        if name:
            col = frames.times[name][kept]
            lo, hi = float(np.nanmin(col)), float(np.nanmax(col))
            inside = (event.times_raw >= lo) & (event.times_raw <= hi)
            return name, float(np.mean(inside))
    best_name, best_frac = "", 0.0
    for name, values in frames.times.items():
        col = values[kept]
        col = col[np.isfinite(col)]
        if col.size < 2:
            continue
        lo, hi = float(col.min()), float(col.max())
        tol = NPM_EVENT_RANGE_TOLERANCE * max(hi - lo, 0.0)
        inside = (event.times_raw >= lo - tol) & (event.times_raw <= hi + tol)
        frac = float(np.mean(inside))
        if frac > best_frac:
            best_name, best_frac = name, frac
    return best_name, best_frac


def _clock_map(
    frames: _NpmFrames,
    kept: np.ndarray,
    source: str,
    primary: str,
) -> Tuple[float, float, float]:
    """Linear map from the source clock (s) to the primary clock (s).

    Returns (slope, intercept, residual SD in seconds). The least-squares line
    over every frame averages out the ~16 ms jitter of the computer clock and
    absorbs any linear drift between the two clocks.
    """
    src = frames.times[source][kept] * _seconds_scale(source, frames.times[source])
    dst = frames.times[primary][kept] * _seconds_scale(primary, frames.times[primary])
    ok = np.isfinite(src) & np.isfinite(dst)
    if source == primary or np.sum(ok) < 2:
        return 1.0, 0.0, 0.0
    slope, intercept = np.polyfit(src[ok], dst[ok], 1)
    resid = dst[ok] - (slope * src[ok] + intercept)
    return float(slope), float(intercept), float(np.std(resid))


def _pulse_trace(timebase: np.ndarray, event_times: np.ndarray) -> np.ndarray:
    """Dense 0/1 trace with a short pulse starting at each event.

    Each pulse starts at the first sample at or after the event and lasts
    ``NPM_EVENT_PULSE_S`` (at least one sample). A pulse is cut one sample
    before the next one starts, so consecutive events keep separate rising edges.
    """
    t = np.asarray(timebase, float)
    trace = np.zeros(t.size, dtype=float)
    if t.size == 0:
        return trace
    starts = np.searchsorted(t, np.sort(np.asarray(event_times, float)), side="left")
    starts = starts[starts < t.size]
    starts = np.unique(starts)
    for k, start in enumerate(starts):
        end = int(np.searchsorted(t, t[start] + NPM_EVENT_PULSE_S, side="left"))
        end = max(end, start + 1)
        if k + 1 < starts.size:
            end = min(end, max(int(starts[k + 1]) - 1, start + 1))
        trace[start:end] = 1.0
    return trace


def _state_trace(timebase: np.ndarray, times: np.ndarray, high: np.ndarray) -> np.ndarray:
    """Dense 0/1 trace from on/off transitions (sample-and-hold between them)."""
    t = np.asarray(timebase, float)
    order = np.argsort(times, kind="stable")
    times = np.asarray(times, float)[order]
    high = np.asarray(high, bool)[order]
    trace = np.zeros(t.size, dtype=float)
    if t.size == 0 or times.size == 0:
        return trace
    current = 0.0 if bool(high[0]) else 1.0  # the level before the first transition
    cursor = 0
    for event_t, is_high in zip(times, high):
        idx = int(np.clip(np.searchsorted(t, event_t, side="left"), 0, t.size))
        if idx > cursor:
            trace[cursor:idx] = current
            cursor = idx
        current = 1.0 if is_high else 0.0
    trace[cursor:] = current
    return trace


def _event_traces(
    event: _EventFile,
    times_rel: np.ndarray,
    timebase: np.ndarray,
    used: set,
) -> Dict[str, np.ndarray]:
    """Turn one event file into named dense traces on the channel timebase."""
    stem = os.path.splitext(os.path.basename(event.path))[0]
    out: Dict[str, np.ndarray] = {}
    if event.labels is None:
        out[_unique_name(stem, used)] = _pulse_trace(timebase, times_rel)
        return out
    labels = [str(lbl).strip() for lbl in event.labels]
    distinct = {lbl.lower() for lbl in labels}
    if len(distinct) == 2 and any(distinct == s for s in _STATE_LABEL_SETS):
        high = np.array([lbl.lower() in _STATE_HIGH_LABELS for lbl in labels], dtype=bool)
        out[_unique_name(stem, used)] = _state_trace(timebase, times_rel, high)
        return out
    if len(distinct) == 1:
        out[_unique_name(stem, used)] = _pulse_trace(timebase, times_rel)
        return out
    def label_order(text: str) -> Tuple[int, float, str]:
        value = _parse_float(text)
        return (0, value, text) if np.isfinite(value) else (1, 0.0, text)

    for label in sorted(set(labels), key=label_order):
        mask = np.array([lbl == label for lbl in labels], dtype=bool)
        numeric_label = np.isfinite(_parse_float(label))
        if numeric_label:
            value = _parse_float(label)
            label_text = str(int(value)) if float(value).is_integer() else label
            name = f"{stem}_{label_text}"
        else:
            name = label or stem
        out[_unique_name(name, used)] = _pulse_trace(timebase, times_rel[mask])
    return out


# =============================================================================
# Loader
# =============================================================================

def _sample_and_hold(frame_t: np.ndarray, values: np.ndarray, timebase: np.ndarray) -> np.ndarray:
    """Value of a per-frame digital line at each timebase instant (last frame at or before it)."""
    idx = np.searchsorted(frame_t, timebase, side="right") - 1
    idx = np.clip(idx, 0, max(frame_t.size - 1, 0))
    v = np.asarray(values, float)[idx]
    return (np.nan_to_num(v, nan=0.0) > 0.5).astype(float)


def load_npm_csv(path: str) -> Optional[LoadedDoricFile]:
    """Load a Neurophotometrics photometry CSV as a pyBer raw recording.

    Returns None when the file is not an NPM photometry table, so callers can
    fall through to other CSV readers. Raises ValueError when the file is an
    NPM table that cannot be used (for example no region has a signal LED).
    """
    if not path or not os.path.isfile(path) or not is_npm_photometry_csv(path):
        return None

    # 1) Read the file, or the whole per-LED split group it belongs to.
    members = _split_group(path)
    source_paths = [m[1] for m in members] if members else [os.path.abspath(path)]
    frames = _merge_frames([_read_frames(p) for p in source_paths])

    # 2) Pick the time axis and drop rows that are not LED measurements.
    primary = _primary_time_name(frames.times)
    scale = _seconds_scale(primary, frames.times[primary])
    t_abs_all = frames.times[primary] * scale
    state = frames.state
    valid = np.isfinite(t_abs_all) & (state >= 0)
    order = np.argsort(np.where(valid, t_abs_all, np.inf), kind="stable")
    order = order[valid[order]]
    if order.size < 3:
        raise ValueError("The Neurophotometrics file has fewer than three valid frames.")
    startup_dropped = bool((int(state[order[0]]) & NPM_EXCITATION_MASK) == NPM_EXCITATION_MASK)
    if startup_dropped:
        order = order[1:]
    lit = (state[order] & NPM_EXCITATION_MASK) != 0
    unlit_dropped = int(np.sum(~lit))
    kept = order[lit]
    if kept.size < 3:
        raise ValueError("The Neurophotometrics file has no frames lit by an excitation LED.")
    t_abs = t_abs_all[kept]
    st = state[kept]
    t0 = float(t_abs[0])
    t_rel = t_abs - t0

    bit_ref = NPM_LED_BITS[NPM_REFERENCE_NM]
    signal_bits = NPM_EXCITATION_MASK & ~bit_ref
    ref_mask = ((st & bit_ref) != 0) & ((st & signal_bits) == 0)
    frames_by_led = {str(nm): int(np.sum((st & bit) != 0)) for nm, bit in NPM_LED_BITS.items()}
    present = {nm for nm, bit in NPM_LED_BITS.items() if np.any((st & bit) != 0)}

    # 3) One pyBer channel per region: signal frames + reference interpolated onto them.
    channels: List[str] = []
    time_by: Dict[str, np.ndarray] = {}
    signal_by: Dict[str, np.ndarray] = {}
    reference_by: Dict[str, np.ndarray] = {}
    channel_info: Dict[str, Dict[str, Any]] = {}
    skipped: Dict[str, str] = {}
    for name, values_all in frames.regions.items():
        values = values_all[kept]
        color = region_color(name)
        signal_nm = NPM_SIGNAL_NM_BY_COLOR.get(color, 470)
        if signal_nm not in present and color == "":
            # Unlabelled region in a recording without 470 nm: use the other signal LED.
            alternatives = sorted(present - {NPM_REFERENCE_NM, signal_nm})
            if alternatives:
                signal_nm = alternatives[0]
        if signal_nm not in present:
            skipped[name] = f"no {signal_nm} nm frames for a {'red' if color == 'R' else 'green'} region"
            continue
        sig_mask = ((st & NPM_LED_BITS[signal_nm]) != 0) & ((st & bit_ref) == 0)
        t_sig = t_rel[sig_mask]
        y_sig = values[sig_mask]
        if t_sig.size < 2:
            skipped[name] = f"fewer than two {signal_nm} nm frames"
            continue

        t_ref = t_rel[ref_mask]
        y_ref = values[ref_mask]
        ok_ref = np.isfinite(y_ref)
        t_ref, y_ref = t_ref[ok_ref], y_ref[ok_ref]
        trimmed = 0
        if t_ref.size >= 2:
            inside = (t_sig >= t_ref[0]) & (t_sig <= t_ref[-1])
            trimmed = int(t_sig.size - np.sum(inside))
            t_sig, y_sig = t_sig[inside], y_sig[inside]
            reference = np.interp(t_sig, t_ref, y_ref)
            reference_nm: Optional[int] = NPM_REFERENCE_NM
        else:
            reference = np.full(t_sig.size, np.nan, dtype=float)
            reference_nm = None
        if t_sig.size < 2:
            skipped[name] = "signal frames do not overlap the reference frames"
            continue

        channels.append(name)
        time_by[name] = np.asarray(t_sig, float)
        signal_by[name] = np.asarray(y_sig, float)
        reference_by[name] = np.asarray(reference, float)
        dt = np.diff(t_sig)
        channel_info[name] = {
            "color": color or "unlabelled (read as green)",
            "signal_nm": int(signal_nm),
            "reference_nm": reference_nm,
            "n_samples": int(t_sig.size),
            "fs_hz": float(1.0 / np.median(dt)) if dt.size and np.median(dt) > 0 else float("nan"),
            "trimmed_signal_samples": trimmed,
        }

    if not channels:
        reasons = "; ".join(f"{k}: {v}" for k, v in skipped.items()) or "no region columns"
        raise ValueError(f"No usable Neurophotometrics region ({reasons}).")

    # 4) Triggers on the first channel's timebase: digital columns, TTL bits, event files.
    timebase = time_by[channels[0]]
    used: set = set()
    trigger_by: Dict[str, np.ndarray] = {}

    for name, values_all in frames.digital.items():
        values = values_all[kept]
        finite = values[np.isfinite(values)]
        if finite.size and np.unique(finite > 0.5).size > 1:
            trigger_by[_unique_name(name, used)] = _sample_and_hold(t_rel, values, timebase)

    for bit in range(3, NPM_MAX_TTL_BIT + 1):
        line = (st & (1 << bit)) != 0
        if line.any() and not line.all():
            label = f"{frames.state_name} bit {bit}"
            trigger_by[_unique_name(label, used)] = _sample_and_hold(t_rel, line.astype(float), timebase)

    event_reports: List[Dict[str, Any]] = []
    unmatched: List[str] = []
    folder = os.path.dirname(os.path.abspath(path))
    photometry_in_folder = {os.path.normcase(p) for p in _folder_npm_photometry(folder)}
    try:
        folder_csvs = sorted(n for n in os.listdir(folder) if n.lower().endswith(".csv"))
    except OSError:
        folder_csvs = []
    for name in folder_csvs:
        candidate = os.path.join(folder, name)
        if os.path.normcase(candidate) in photometry_in_folder:
            continue
        kind = _event_file_kind(candidate)
        if not kind:
            continue
        event = _read_event_file(candidate, kind)
        if event is None:
            continue
        clock, frac = _match_event_clock(event, frames, kept)
        if not clock or frac < NPM_EVENT_MIN_FRACTION_IN_RANGE:
            unmatched.append(candidate)
            continue
        slope, intercept, resid_sd = _clock_map(frames, kept, clock, primary)
        ev_s = event.times_raw * _seconds_scale(clock, frames.times[clock])
        ev_rel = slope * ev_s + intercept - t0
        in_range = (ev_rel >= timebase[0]) & (ev_rel <= timebase[-1])
        kept_event = _EventFile(
            path=event.path,
            times_raw=event.times_raw[in_range],
            labels=[lbl for lbl, keep in zip(event.labels, in_range) if keep] if event.labels else None,
        )
        traces = _event_traces(kept_event, ev_rel[in_range], timebase, used)
        trigger_by.update(traces)
        event_reports.append({
            "path": candidate,
            "clock": clock,
            "n_events": int(event.times_raw.size),
            "n_in_recording": int(np.sum(in_range)),
            "traces": sorted(traces),
            "clock_fit_residual_sd_s": resid_sd,
        })

    trigger_time_by = {name: timebase.copy() for name in trigger_by}
    metadata: Dict[str, Any] = {
        "acquisition_system": "Neurophotometrics",
        "npm_source_files": source_paths,
        "npm_state_column": frames.state_name,
        "npm_time_column": primary,
        "npm_time_unit": "ms" if scale == 1e-3 else "s",
        "npm_time_origin_s": t0,
        "npm_frames_used": int(kept.size),
        "npm_frames_by_led": frames_by_led,
        "npm_startup_frame_dropped": startup_dropped,
        "npm_unlit_frames_dropped": unlit_dropped,
        "npm_reference_alignment": "415 nm linearly interpolated onto signal frame times",
        "npm_channels": channel_info,
        "npm_skipped_regions": skipped,
        "npm_event_files": event_reports,
        "npm_unmatched_event_files": unmatched,
        "npm_event_pulse_s": NPM_EVENT_PULSE_S,
    }
    return LoadedDoricFile(
        path=path,
        channels=channels,
        time_by_channel=time_by,
        signal_by_channel=signal_by,
        reference_by_channel=reference_by,
        digital_time=timebase.copy() if trigger_by else None,
        digital_by_name={k: v.copy() for k, v in trigger_by.items()},
        trigger_time_by_name=trigger_time_by,
        trigger_by_name=trigger_by,
        metadata=metadata,
    )


def npm_load_summary(loaded: LoadedDoricFile) -> str:
    """One-line description of an NPM load for the status bar."""
    md = getattr(loaded, "metadata", {}) or {}
    if md.get("acquisition_system") != "Neurophotometrics":
        return ""
    info = md.get("npm_channels", {})
    parts = []
    for name in loaded.channels:
        ch = info.get(name, {})
        ref = ch.get("reference_nm")
        leds = f"{ch.get('signal_nm', '?')}/{ref}" if ref else f"{ch.get('signal_nm', '?')}, no reference"
        parts.append(f"{name} ({leds} nm)")
    n_events = len(loaded.trigger_by_name or {})
    text = f"{len(loaded.channels)} region(s): " + ", ".join(parts)
    if n_events:
        text += f"; {n_events} trigger(s)"
    if md.get("npm_skipped_regions"):
        text += f"; skipped {', '.join(md['npm_skipped_regions'])}"
    return text
