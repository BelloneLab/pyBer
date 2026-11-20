"""PCA state-space views of event-aligned photometry responses.

Dashboard panels ("Change to > State-space (PCA)") share one analysis:

* PCA trajectories : the state-space picture itself;
* PCA loadings     : what each component measures;
* PCA separation   : a permutation test of how far two conditions are apart;
* Cluster separation: silhouette scores for trial-response Gaussian mixture clusters.

Their options (behaviors, groups, method, axes) are chosen from the right-click
menu of any of the PCA panels.

Methods (single fiber, many trials)
-----------------------------------
A fiber gives one signal, so a state space has to be built from that signal's
own time course or from its trials. Four constructions are offered.

1. Onset-to-offset trajectories (default). Each bout is taken from its onset
   to its offset, resampled to a common phase axis (0 = onset, 1 = offset) so
   bouts of different length can be compared, then lag-embedded: every point
   is described by the stretch of trace around it. One PCA is fitted on all
   bouts pooled. Thin lines are single bouts, thick lines condition means.
   It needs bouts with a start and an end; without durations the next method
   is used and the panel says so.
2. Lag-embedded trajectories. The same embedding applied to each trial's
   event-aligned response over the PSTH window (real time, no resampling of
   bout length): the temporal response pattern of every trial.
3. Trial-response PCA. Each trial's whole response is one sample; the plot has
   one point per event, with a diamond per recording. It shows trial-to-trial
   variability and whether conditions occupy different regions.
4. Recordings as dimensions (pseudo-population). Each recording is one
   dimension and each behavior one trajectory; the selected groups are pooled
   because the recordings are the dimensions.

In every method one PCA is fitted on all conditions together, so they share
their axes and can be compared. Traces are block-averaged to at most
``MAX_TRACE_POINTS`` points before embedding (display resolution; the
recordings are not modified).

Loadings
--------
For the lag-embedded methods a loading is a weight per lag inside the delay
window: a flat PC1 means "local signal level", a PC2 that changes sign means
"local slope". For event-aligned lag trajectories the panel defaults to the
mean PC scores at successive window centers in each condition. The fixed
lag weights are available from PCA options. When comparing groups, the weights
view can fit each group's selected trials separately. PC rank and explained
variance then refer to that group. Trajectories always use the shared fit.
For trial-response PCA a loading is a time course over the
response window. For recordings as dimensions it is a weight per recording.

Separation
----------
Statistic: mean Euclidean distance between the two condition mean
trajectories (between the two centroids for trial-response PCA), in the
shared PCA space. Null distribution: the condition labels are permuted and
the statistic recomputed in the same, label-blind PCA space. When each
condition has at least ``MIN_RECORDINGS_PER_CONDITION`` different recordings
the recordings are permuted (one mean trajectory per recording: animal
level). Otherwise single trials are permuted and the result is marked
exploratory, because trials of one animal are not independent. All
labelings are enumerated when there are few of them; the p-value is
(1 + number of permuted statistics >= observed) / (1 + number of
permutations). With more than two conditions the first two are tested.
"""
from __future__ import annotations

import itertools
from html import escape
import json
import math
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pyqtgraph as pg
from PySide6 import QtCore, QtGui, QtWidgets

from postprocessing_core import compute_psth_matrix
from postprocessing_style import style_plot
from recording_groups import UNGROUPED_COLOR, UNGROUPED_LABEL, color_dot_icon
from trial_response_clusters import cluster_ellipse, fit_trial_clusters, trial_cluster_silhouettes


# =============================================================================
# Configuration
# =============================================================================

FAMILY = "trajectory"
CODE = "pca"                       # the trajectories / points panel
CODE_LOADINGS = "loadings"
CODE_SEPARATION = "separation"
CODE_CLUSTER_SEPARATION = "cluster_separation"
SPEC = f"{FAMILY}:{CODE}"
SPEC_LOADINGS = f"{FAMILY}:{CODE_LOADINGS}"
SPEC_SEPARATION = f"{FAMILY}:{CODE_SEPARATION}"
SPEC_CLUSTER_SEPARATION = f"{FAMILY}:{CODE_CLUSTER_SEPARATION}"
PANEL_LABEL = "PCA trajectories"
PANELS: Tuple[Tuple[str, str], ...] = (
    (CODE, PANEL_LABEL), (CODE_LOADINGS, "PCA loadings"), (CODE_SEPARATION, "PCA separation test"),
    (CODE_CLUSTER_SEPARATION, "Cluster separation"),
)

METHOD_BOUT = "bout"
METHOD_LAG = "lag"
METHOD_TRIALS = "trials"
METHOD_RECORDINGS = "recordings"
METHOD_DELAY = METHOD_LAG          # name used by earlier versions and saved options
METHODS: Tuple[Tuple[str, str], ...] = (
    (METHOD_BOUT, "Onset-to-offset trajectories (each bout, start to end)"),
    (METHOD_LAG, "Lag-embedded trajectories (temporal pattern of each trial)"),
    (METHOD_TRIALS, "Trial-response PCA (one point per event)"),
    (METHOD_RECORDINGS, "Recordings as dimensions (pseudo-population)"),
)
METHOD_TITLES = {METHOD_BOUT: "onset-to-offset trajectories", METHOD_LAG: "lag-embedded trajectories",
                 METHOD_TRIALS: "trial responses", METHOD_RECORDINGS: "recordings as dimensions"}
DEFAULT_METHOD = METHOD_BOUT
DISPLAY_OPTIONS = {"axes", "units", "trial_clusters", "show_lag_weights", "group_loadings"}

#: Length of the delay window, in seconds, offered for lag-embedded trajectories.
DELAY_SPANS_S: Tuple[float, ...] = (0.25, 0.5, 1.0, 2.0)
DEFAULT_DELAY_SPAN_S = 1.0
MIN_DELAY_SAMPLES = 3
MAX_DELAY_FRACTION = 0.5           # the delay window never exceeds this fraction of a trace
#: Onset-to-offset: points on the phase axis, delay window as a fraction of the
#: bout, longest bout read (seconds; longer bouts are cut there) and the fewest
#: samples a bout must contain.
BOUT_PHASE_POINTS = 60
BOUT_DELAY_FRACTION = 0.2
BOUT_MAX_S = 60.0
MIN_BOUT_SAMPLES = 4
#: Traces are block-averaged to at most this many points before embedding.
MAX_TRACE_POINTS = 200
#: At most this many embedded samples are used to fit the PCA (evenly spaced).
FIT_MAX_SAMPLES = 60000
#: Component pairs offered for the two plot axes (0-based).
AXES_CHOICES: Tuple[Tuple[int, int], ...] = ((0, 1), (0, 2), (1, 2))
N_COMPONENTS = 3
MIN_RECORDINGS_AS_DIMENSIONS = 2

# Separation test.
N_PERMUTATIONS = 2000
PERMUTATION_SEED = 0               # fixed: the same data always give the same p-value
MIN_RECORDINGS_PER_CONDITION = 3   # below this, trials are permuted (exploratory)
HISTOGRAM_BINS = 30

#: Group keys used in the options: pooled recordings and ungrouped recordings.
POOLED = "*"
UNGROUPED = ""
SETTINGS_KEY = "pca_trajectories_options"

# Figure styling.
MEAN_LINE_WIDTH = 2.6
UNIT_LINE_WIDTH = 0.8
UNIT_LINE_ALPHA = 55               # 0..255 opacity of single-trial / single-bout lines
MAX_UNIT_LINES = 60                # single lines drawn per condition (evenly chosen)
MARKER_SIZE = 10
POINT_SIZE = 5
POINT_ALPHA = 110
RECORDING_MARKER_SIZE = 11
CLUSTER_OUTLINE_WIDTH = 1.6
CLUSTER_POINT_SIZE = 7
TRIAL_SYMBOLS = ("o", "s", "t", "t1", "p", "h", "star", "+", "x")
LEGEND_COLUMN_WIDTH = 135          # width allowed per legend column (pixels)
LEGEND_LAYOUT_ROW = 4              # plot layout row under the bottom axis (row 3)
#: Categorical colors (fixed order) for behaviors when one group is shown.
COLORS_ON_LIGHT = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948")
COLORS_ON_DARK = ("#3987e5", "#d95926", "#199e70", "#c98500", "#d55181", "#0ca30c", "#9085e9", "#e66767")
#: Line styles (fixed order) for behaviors when several groups are shown.
LINE_STYLES = (QtCore.Qt.PenStyle.SolidLine, QtCore.Qt.PenStyle.DashLine,
               QtCore.Qt.PenStyle.DotLine, QtCore.Qt.PenStyle.DashDotLine)
PC_ALPHA = (255, 210, 165)         # PC1, PC2, PC3 in an overlay of groups
BEHAVIOR_SYMBOLS = ("o", "s", "t", "d", "+", "x", "star", "p", "h")


# =============================================================================
# Numerics (no GUI)
# =============================================================================

def fit_pca(samples: np.ndarray, n_components: int = N_COMPONENTS) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """PCA by SVD of the centered samples (rows = samples, columns = dimensions).

    Returns ``(mean, components, explained_ratio)``. Components are rows, at
    most ``n_components`` of them. Each component's sign is fixed so that its
    largest loading is positive, which makes results reproducible.
    """
    x = np.asarray(samples, float)
    if x.ndim != 2 or x.shape[0] < 2 or x.shape[1] < 1 or not np.all(np.isfinite(x)):
        raise ValueError("PCA needs at least two finite samples.")
    mean = x.mean(axis=0)
    _u, singular, vt = np.linalg.svd(x - mean, full_matrices=False)
    variance = singular ** 2
    total = float(variance.sum())
    ratio = variance / total if total > 0 else np.zeros_like(variance)
    count = min(int(n_components), vt.shape[0])
    components = vt[:count].copy()
    for row in components:
        if row[np.argmax(np.abs(row))] < 0:
            row *= -1.0
    return mean, components, ratio[:count]


def delay_embed(trace: np.ndarray, n_delays: int) -> Tuple[np.ndarray, np.ndarray]:
    """Delay vectors of a trace and the index of each vector's center sample.

    Row i is ``[y[i], y[i+1], ..., y[i+n_delays-1]]``. Rows that contain a
    missing sample are dropped, so nothing is interpolated.
    """
    y = np.asarray(trace, float).reshape(-1)
    m = int(n_delays)
    if m < 1 or y.size < m:
        return np.empty((0, max(m, 1))), np.empty(0, dtype=int)
    windows = np.lib.stride_tricks.sliding_window_view(y, m)
    keep = np.all(np.isfinite(windows), axis=1)
    centers = np.arange(windows.shape[0]) + (m - 1) // 2
    return np.array(windows[keep]), centers[keep]


def delay_sample_count(span_s: float, time: np.ndarray) -> int:
    """Number of samples in a delay window of ``span_s`` on this time axis."""
    t = np.asarray(time, float)
    if t.size < 2:
        return MIN_DELAY_SAMPLES
    step = float(np.nanmedian(np.diff(t)))
    wanted = int(round(float(span_s) / step)) + 1 if step > 0 else MIN_DELAY_SAMPLES
    return int(max(MIN_DELAY_SAMPLES, min(wanted, int(t.size * MAX_DELAY_FRACTION))))


def delay_trajectories(traces: Dict[object, Tuple[np.ndarray, np.ndarray]], span_s: float,
                       n_components: int = N_COMPONENTS) -> Tuple[Dict[object, Tuple[np.ndarray, np.ndarray]], np.ndarray]:
    """Trajectories of several mean traces in one shared delay-embedding PCA space.

    ``traces``: key -> (time, trace). Returns key -> (time of each point,
    scores with one column per component) and the explained variance ratios.
    Keys whose trace is too short or entirely missing are left out.
    """
    embedded = {}
    for key, (time, trace) in traces.items():
        t = np.asarray(time, float)
        vectors, centers = delay_embed(trace, delay_sample_count(span_s, t))
        if vectors.shape[0] >= 2:
            embedded[key] = (t[centers], vectors)
    if not embedded:
        return {}, np.empty(0)
    widths = {vectors.shape[1] for _t, vectors in embedded.values()}
    if len(widths) != 1:
        raise ValueError("Traces must share one time step to be embedded together.")
    mean, components, ratio = fit_pca(np.vstack([vectors for _t, vectors in embedded.values()]), n_components)
    return {key: (t, (vectors - mean) @ components.T) for key, (t, vectors) in embedded.items()}, ratio


def population_trajectories(responses: Dict[object, Tuple[np.ndarray, np.ndarray]],
                            n_components: int = N_COMPONENTS, return_model: bool = False):
    """Trajectories of several conditions with recordings as dimensions.

    ``responses``: key -> (time, matrix with one row per time point and one
    column per recording; the same recordings, in the same order, for every
    key). Time points with a missing value in any recording are dropped.
    With ``return_model`` the components (loadings per recording) are
    returned as a third value.
    """
    kept = {}
    for key, (time, matrix) in responses.items():
        mat = np.asarray(matrix, float)
        rows = np.all(np.isfinite(mat), axis=1) if mat.ndim == 2 else np.zeros(0, bool)
        if mat.ndim == 2 and rows.sum() >= 2:
            kept[key] = (np.asarray(time, float)[rows], mat[rows])
    if not kept:
        return ({}, np.empty(0), np.empty((0, 0))) if return_model else ({}, np.empty(0))
    if len({mat.shape[1] for _t, mat in kept.values()}) != 1:
        raise ValueError("Every condition must use the same recordings.")
    mean, components, ratio = fit_pca(np.vstack([mat for _t, mat in kept.values()]), n_components)
    out = {key: (t, (mat - mean) @ components.T) for key, (t, mat) in kept.items()}
    return (out, ratio, components) if return_model else (out, ratio)


def decimate_rows(matrix: np.ndarray, time: np.ndarray, max_points: int = MAX_TRACE_POINTS):
    """Block-average traces (rows) so that at most ``max_points`` samples remain.

    Averaging blocks, instead of taking every n-th sample, keeps fast noise
    from aliasing into the trajectory. A block with no finite sample stays
    missing. Returns the reduced matrix and time axis (block centers).
    """
    mat = np.asarray(matrix, float)
    t = np.asarray(time, float)
    if mat.ndim != 2 or mat.shape[1] <= max_points:
        return mat, t
    factor = int(math.ceil(mat.shape[1] / float(max_points)))
    usable = (mat.shape[1] // factor) * factor
    blocks = mat[:, :usable].reshape(mat.shape[0], -1, factor)
    counts = np.isfinite(blocks).sum(axis=2)
    sums = np.where(np.isfinite(blocks), blocks, 0.0).sum(axis=2)
    reduced = np.where(counts > 0, sums / np.maximum(counts, 1), np.nan)
    return reduced, t[:usable].reshape(-1, factor).mean(axis=1)


def _fit_delay_samples(samples, n_components, fit_max):
    """Fit from evenly spaced complete windows, using the same cap for each PCA."""
    if len(samples) > fit_max:
        samples = samples[np.linspace(0, len(samples) - 1, fit_max).astype(int)]
    return fit_pca(samples, n_components)


def embed_units(traces: np.ndarray, n_delays: int, n_components: int = N_COMPONENTS,
                fit_max: int = FIT_MAX_SAMPLES):
    """Lag-embed many traces and project them in one pooled PCA space.

    ``traces``: one row per unit (trial or bout), all on the same axis.
    Returns ``(scores, centers, components, ratio)``: scores has shape
    (units, points, components) with NaN where a delay window contained a
    missing sample; ``centers`` is the axis index of each point.
    """
    mat = np.asarray(traces, float)
    m = int(n_delays)
    if mat.ndim != 2 or mat.shape[0] < 1 or mat.shape[1] < m or m < 1:
        raise ValueError("Traces are too short for the delay window.")
    windows = np.lib.stride_tricks.sliding_window_view(mat, m, axis=1)        # (units, points, m)
    valid = np.isfinite(windows).all(axis=2)
    samples = windows[valid]
    if samples.shape[0] < 2:
        raise ValueError("Not enough complete samples to fit the PCA.")
    mean, components, ratio = _fit_delay_samples(samples, n_components, fit_max)
    scores = np.full(windows.shape[:2] + (components.shape[0],), np.nan)
    scores[valid] = (samples - mean) @ components.T
    centers = np.arange(windows.shape[1]) + (m - 1) // 2
    return scores, centers, components, ratio


def bouts_to_phase(matrix: np.ndarray, time: np.ndarray, durations: np.ndarray,
                   n_points: int = BOUT_PHASE_POINTS) -> Tuple[np.ndarray, np.ndarray, int]:
    """Resample each onset-aligned row from its onset to its offset onto a phase axis.

    ``matrix`` rows are aligned to bout onset (time 0). A bout longer than the
    available window is cut at the window end. Returns the resampled rows
    (bouts x n_points), the index of each kept bout, and how many were cut.
    Bouts without a duration or with too few finite samples are left out.
    """
    mat = np.asarray(matrix, float)
    t = np.asarray(time, float)
    phase = np.linspace(0.0, 1.0, int(n_points))
    rows, kept, truncated = [], [], 0
    window_end = float(t[-1]) if t.size else 0.0
    # Half a sample of tolerance: the sample at the onset (or offset) must not be
    # lost to rounding of the time axis.
    slack = 0.5 * float(np.median(np.diff(t))) if t.size > 1 else 0.0
    for index, duration in enumerate(np.asarray(durations, float)):
        if not np.isfinite(duration) or duration <= 0:
            continue
        end = min(float(duration), window_end)
        mask = (t >= -slack) & (t <= end + slack) & np.isfinite(mat[index])
        if int(mask.sum()) < MIN_BOUT_SAMPLES or end <= 0:
            continue
        truncated += int(duration > window_end + slack)
        rows.append(np.interp(phase, np.clip(t[mask] / end, 0.0, 1.0), mat[index][mask]))
        kept.append(index)
    if not rows:
        return np.empty((0, int(n_points))), np.empty(0, dtype=int), 0
    return np.vstack(rows), np.asarray(kept, dtype=int), truncated


def separation_statistic(group_a: np.ndarray, group_b: np.ndarray) -> float:
    """Mean distance between two mean trajectories (or two centroids).

    Inputs: units x points x components (trajectories) or units x components
    (points). The means are taken over units, the distance over components,
    and the average over points.
    """
    difference = np.asarray(group_a, float).mean(axis=0) - np.asarray(group_b, float).mean(axis=0)
    return float(np.mean(np.sqrt(np.sum(difference ** 2, axis=-1))))


def permutation_separation(group_a: np.ndarray, group_b: np.ndarray, n_permutations: int = N_PERMUTATIONS,
                           seed: int = PERMUTATION_SEED) -> Dict[str, object]:
    """Permutation test of the separation between two sets of units.

    Units are exchanged between the two conditions; the PCA space is not
    refitted (it was fitted without the labels). All labelings are enumerated
    when there are at most ``n_permutations`` of them (exact test). The
    p-value is (1 + #{permuted >= observed}) / (1 + number of permutations);
    for an exact test the observed labeling is one of the labelings and the
    p-value is simply the fraction of labelings at least as extreme.
    """
    a, b = np.asarray(group_a, float), np.asarray(group_b, float)
    n_a, n_b = a.shape[0], b.shape[0]
    pooled = np.concatenate([a, b], axis=0)
    observed = separation_statistic(a, b)
    total = pooled.sum(axis=0)
    n = n_a + n_b

    def statistic(index_a) -> float:
        sum_a = pooled[list(index_a)].sum(axis=0)
        difference = sum_a / n_a - (total - sum_a) / n_b
        return float(np.mean(np.sqrt(np.sum(difference ** 2, axis=-1))))

    exact = math.comb(n, n_a) <= int(n_permutations)
    if exact:
        null = np.asarray([statistic(combo) for combo in itertools.combinations(range(n), n_a)])
        p_value = float(np.mean(null >= observed - 1e-12))
    else:
        rng = np.random.default_rng(seed)
        null = np.asarray([statistic(rng.permutation(n)[:n_a]) for _ in range(int(n_permutations))])
        p_value = float((1 + np.count_nonzero(null >= observed - 1e-12)) / (1 + null.size))
    return {"observed": observed, "null": null, "p": p_value, "exact": bool(exact),
            "n_permutations": int(null.size), "n_a": int(n_a), "n_b": int(n_b)}


# =============================================================================
# Panel
# =============================================================================

class PcaTrajectoryView(QtCore.QObject):
    """Options, data collection and drawing of the PCA panels."""

    def __init__(self, panel) -> None:
        super().__init__(panel)
        self.panel = panel
        # behaviors: [] = follow the PSTH behavior. groups: None = automatic
        # (the groups shown by Hide/show groups, or all recordings pooled).
        self.options: Dict[str, object] = {"behaviors": [], "groups": None, "method": DEFAULT_METHOD,
                                           "span_s": DEFAULT_DELAY_SPAN_S, "axes": [0, 1], "units": True,
                                           "show_lag_weights": False, "group_loadings": True,
                                           "trial_clusters": True}
        self._legends: Dict[int, pg.LegendItem] = {}
        self.last_report: Dict[str, object] = {}
        self._cache: Optional[Tuple[str, Dict[str, object]]] = None
        self._data_revision = 0
        self._load_options()

    # ---------------------------------------------------------------- options
    def _load_options(self) -> None:
        try:
            stored = json.loads(str(self.panel._settings.value(SETTINGS_KEY, "") or "{}"))
        except Exception:
            stored = {}
        if isinstance(stored, dict):
            for key in self.options:
                if key in stored:
                    self.options[key] = stored[key]
        if self.options["method"] == "delay":          # saved by the first version
            self.options["method"] = METHOD_LAG
        if self.options["method"] not in dict(METHODS):
            self.options["method"] = DEFAULT_METHOD
        if tuple(self.options.get("axes") or ()) not in AXES_CHOICES:
            self.options["axes"] = [0, 1]
        # Earlier versions allowed pooled and named groups simultaneously,
        # which included those groups' trials twice in the fitted PCA.
        if POOLED in (self.options.get("groups") or []):
            self.options["groups"] = [POOLED]

    def set_option(self, key: str, value) -> None:
        self.options[key] = value
        try:
            self.panel._settings.setValue(SETTINGS_KEY, json.dumps(self.options))
        except Exception:
            pass
        self.refresh(reuse_fit=key in DISPLAY_OPTIONS)

    def new_pass(self) -> None:
        """Invalidate shared and per-event fits after an external dashboard refresh."""
        self._data_revision += 1
        self._cache = None

    def refresh(self, *, reuse_fit: bool = False) -> None:
        """Redraw PCA panels. Only explicit presentation options may reuse the fit."""
        panels = getattr(self.panel, "dashboard_panels", None)
        if panels is not None:
            panels.render_substitutes((FAMILY,), reuse_pca_fit=reuse_fit)
        elif not reuse_fit:
            self.new_pass()

    # ------------------------------------------------------- what can be shown
    def behaviors_apply(self) -> bool:
        """Behaviors can be chosen only when the PSTH is aligned to one scored behavior."""
        panel = self.panel
        return (panel.combo_align.currentText().startswith("Behavior")
                and not panel.combo_behavior_align.currentText().startswith("Transition")
                and bool(self.available_behaviors()))

    def available_behaviors(self) -> List[str]:
        combo = self.panel.combo_behavior_name
        return [combo.itemText(index) for index in range(combo.count()) if combo.itemText(index).strip()]

    def selected_behaviors(self) -> List[str]:
        """Behaviors drawn: the chosen ones still loaded, else the PSTH behavior."""
        if not self.behaviors_apply():
            return [self._current_event_label()]
        available = self.available_behaviors()
        chosen = [name for name in (self.options.get("behaviors") or []) if name in available]
        if chosen:
            return chosen
        current = self.panel.combo_behavior_name.currentText().strip()
        return [current] if current else available[:1]

    def _current_event_label(self) -> str:
        panel = self.panel
        align = panel.combo_align.currentText()
        if align == "Event marker":
            return str(panel._marker_label()) if hasattr(panel, "_marker_label") else "Event marker"
        if align.startswith("Behavior"):
            if panel.combo_behavior_align.currentText().startswith("Transition"):
                return f"{panel.combo_behavior_from.currentText()} → {panel.combo_behavior_to.currentText()}"
            return panel.combo_behavior_name.currentText().strip() or "Behavior"
        return panel.combo_dio.currentText().strip() or "Events"

    def individual_recording(self) -> Optional[str]:
        """Return the active recording ID in Individual view, or None in Group view."""
        panel = self.panel
        if panel.tab_visual_mode.currentIndex() != 0:
            return None
        selected = panel.combo_individual_file.currentText().strip()
        if selected or not panel._processed:
            return selected
        # Before the PSTH populates the picker, its trace previews the first file.
        return panel._file_id_for_proc(panel._processed[0])

    def cache_signature(self) -> str:
        """Identify a fit by its data revision, analytical options and recording."""
        analysis = {key: value for key, value in self.options.items() if key not in DISPLAY_OPTIONS}
        return json.dumps((self._data_revision, analysis, self.individual_recording()), sort_keys=True)

    def group_entries(self) -> List[Tuple[str, str, str, List[str]]]:
        """(key, name, color, recording ids) for each group that has loaded recordings."""
        panel = self.panel
        file_ids = [panel._file_id_for_proc(proc) for proc in panel._processed]
        entries = []
        for group in panel.recording_groups.groups():
            members = [fid for fid in file_ids if panel.recording_groups.group_id_of(fid) == group["id"]]
            if members:
                entries.append((group["id"], group["name"], group["color"], members))
        known = {group["id"] for group in panel.recording_groups.groups()}
        loose = [fid for fid in file_ids if panel.recording_groups.group_id_of(fid) not in known]
        if loose and entries:
            entries.append((UNGROUPED, UNGROUPED_LABEL, UNGROUPED_COLOR, loose))
        return entries

    def selected_groups(self) -> List[Tuple[str, str, Optional[str], List[str]]]:
        """Groups drawn as separate conditions (color None = use the behavior colors)."""
        panel = self.panel
        recording = self.individual_recording()
        if recording is not None:
            return [(recording, recording, None, [recording])] if recording else []
        all_ids = [panel._file_id_for_proc(proc) for proc in panel._processed]
        pooled = (POOLED, "All recordings", None, all_ids)
        entries = self.group_entries()
        chosen = self.options.get("groups")
        if chosen is None:   # automatic: the groups that Hide/show groups leaves visible
            hidden = panel._hidden_groups() if hasattr(panel, "_hidden_groups") else set()
            visible = [entry for entry in entries if entry[0] not in hidden]
            return visible if entries else [pooled]
        if POOLED in chosen:
            return [pooled]
        out = [entry for entry in entries if entry[0] in chosen]
        return out or [pooled]

    # ------------------------------------------------------------------ data
    def _trial_responses(self, behaviors: Sequence[str], to_offset: bool):
        """Trial matrices for the recordings and behaviors in the active view.

        Uses the panel's own event extraction (clock, alignment edge, event
        filters, hidden events) and the current PSTH baseline, normalization
        and smoothing, so the PCA is built from the same responses as the
        PSTH. The behavior picker is switched silently and put back.

        ``to_offset`` False: rows are the PSTH window around each event.
        ``to_offset`` True : rows run from bout onset to bout offset on the
        phase axis (see ``bouts_to_phase``).

        Returns ``(responses, axis, info)`` with responses[behavior][file_id]
        = matrix (trials x points).
        """
        panel = self.panel
        pre, post = float(panel.spin_pre.value()), float(panel.spin_post.value())
        baseline = (float(panel.spin_b0.value()), float(panel.spin_b1.value()))
        rate = float(panel.spin_resample.value())
        smooth = float(panel.spin_smooth.value())
        normalization = panel._psth_normalization()
        marker_alignment = panel.combo_align.currentText() == "Event marker"
        combo = panel.combo_behavior_name
        switch = self.behaviors_apply()
        previous = combo.currentIndex()
        responses: Dict[str, Dict[str, np.ndarray]] = {}
        axis: Optional[np.ndarray] = None
        info = {"truncated": 0, "without_duration": 0}
        recordings = panel._processed
        recording = self.individual_recording()
        if recording is not None:
            # Restrict the input before event extraction, normalization and fitting.
            active = next((proc for proc in recordings if panel._file_id_for_proc(proc) == recording), None)
            recordings = (active,) if active is not None else ()
        blocker = QtCore.QSignalBlocker(combo)
        try:
            for behavior in behaviors:
                if switch:
                    index = combo.findText(behavior)
                    if index < 0:
                        continue
                    combo.setCurrentIndex(index)
                per_file: Dict[str, np.ndarray] = {}
                for proc in recordings:
                    _label, signal = panel._psth_input(proc)
                    if signal is None:
                        continue
                    events, durations = panel._get_events_for_proc(proc)
                    if not marker_alignment:
                        events, durations = panel._filter_events(events, durations)
                    events = np.asarray(events, float)
                    durations = np.asarray(durations, float)
                    if events.size == 0:
                        continue
                    if durations.size != events.size:
                        durations = np.full(events.shape, np.nan)
                    window = (-pre, post)
                    if to_offset:
                        usable = np.isfinite(durations) & (durations > 0)
                        info["without_duration"] += int(np.count_nonzero(~usable))
                        if not usable.any():
                            continue
                        # Bouts are read from their onset, whichever edge the PSTH uses.
                        if not marker_alignment and switch:
                            events = panel._behavior_onsets(events, durations)
                        events, durations = events[usable], durations[usable]
                        longest = max(float(durations.max()), MIN_BOUT_SAMPLES / rate)
                        window = (-pre, float(min(longest, BOUT_MAX_S)))
                    try:
                        time, matrix = compute_psth_matrix(panel._proc_time(proc), signal, events, window, baseline,
                                                           rate, smooth_sigma_s=smooth, normalization=normalization)
                    except ValueError:
                        continue
                    if matrix.ndim != 2 or matrix.size == 0:
                        continue
                    if to_offset:
                        matrix, _kept, truncated = bouts_to_phase(matrix, time, durations)
                        info["truncated"] += truncated
                        time = np.linspace(0.0, 1.0, BOUT_PHASE_POINTS)
                    else:
                        matrix = matrix[np.any(np.isfinite(matrix), axis=1)]
                    if matrix.shape[0] == 0:
                        continue
                    per_file[panel._file_id_for_proc(proc)] = matrix
                    axis = time
                responses[behavior] = per_file
        finally:
            if switch:
                combo.setCurrentIndex(previous)
            del blocker
        return responses, axis, info

    def compute(self) -> Dict[str, object]:
        """Share the same fit until the dashboard starts its next render pass."""
        signature = self.cache_signature()
        if self._cache is not None and self._cache[0] == signature:
            return self._cache[1]
        try:
            report = self._compute()
        except Exception as exc:        # a display problem must not break the analysis
            report = self._empty_report(str(self.options.get("method")), f"Could not compute the PCA: {exc}")
        self._cache = (signature, report)
        self.last_report = report
        return report

    def _empty_report(self, method: str, note: str) -> Dict[str, object]:
        return {"method": method, "requested_method": method, "behaviors": [], "curves": [], "ratio": np.empty(0),
                "components": np.empty((0, 0)), "loading_axis": np.empty(0), "loading_label": "",
                "loading_ticks": None, "note": note, "skipped": [], "fallback": "", "info": {},
                "individual_recording": self.individual_recording()}

    def _compute(self, behaviors: Optional[Sequence[str]] = None) -> Dict[str, object]:
        """Fit the selected behaviors, or those represented by a dashboard column."""
        if behaviors is None:
            behaviors = self.selected_behaviors()
        groups = self.selected_groups()
        method = str(self.options.get("method") or DEFAULT_METHOD)
        if not self.panel._processed or not behaviors:
            return self._empty_report(method, "Load recordings and events to draw the PCA.")
        if method == METHOD_RECORDINGS:
            if self.individual_recording() is not None:
                report = self._compute_units(METHOD_TRIALS, behaviors, groups)
                report["requested_method"] = method
                report["fallback"] = ("Individual view uses trial-response PCA for the selected recording. "
                                      "Group view uses the selected recordings-as-dimensions method.")
                return report
            return self._compute_recordings(behaviors, groups)
        report = self._compute_units(method, behaviors, groups)
        if method == METHOD_BOUT and not report["curves"] and not report.get("hard_error"):
            # No bout has a start and an end: show the trial pattern instead.
            fallback = self._compute_units(METHOD_LAG, behaviors, groups)
            fallback["requested_method"] = METHOD_BOUT
            if fallback["curves"]:
                fallback["fallback"] = ("No bout with a start and an end was found, so lag-embedded "
                                        "trajectories around each event are shown instead.")
            return fallback
        return report

    def _compute_units(self, method: str, behaviors: Sequence[str], groups) -> Dict[str, object]:
        """Onset-to-offset, lag-embedded or trial-response PCA: units are trials or bouts."""
        report = self._empty_report(method, "")
        report["behaviors"] = list(behaviors)
        responses, axis, info = self._trial_responses(behaviors, to_offset=(method == METHOD_BOUT))
        report["info"] = info
        if axis is None:
            report["note"] = ("No bout with a start and an end for the selected behaviors."
                              if method == METHOD_BOUT else "No event-aligned response for the selected behaviors.")
            return report

        # One block of unit traces per condition (group x behavior), all on the same axis.
        blocks = []
        for group_index, (_key, name, color, members) in enumerate(groups):
            for behavior_index, behavior in enumerate(behaviors):
                rows, files = [], []
                for fid in members:
                    matrix = responses.get(behavior, {}).get(fid)
                    if matrix is not None and matrix.shape[0]:
                        rows.append(matrix)
                        files.extend([fid] * matrix.shape[0])
                if not rows:
                    report["skipped"].append(f"{name} / {behavior}")
                    continue
                blocks.append({"group": name, "group_color": color, "behavior": behavior,
                               "behavior_index": behavior_index, "group_index": group_index,
                               "rows": np.vstack(rows), "files": np.asarray(files, dtype=object)})
        if not blocks:
            report["note"] = "The selected groups have no response for the selected behaviors."
            return report
        stacked, axis = decimate_rows(np.vstack([block["rows"] for block in blocks]), axis)
        step = float(np.nanmedian(np.diff(axis))) if axis.size > 1 else 1.0

        if method == METHOD_TRIALS:
            complete = np.all(np.isfinite(stacked), axis=1)
            if complete.sum() < 3:
                report["note"] = "Trial-response PCA needs at least three complete trials."
                return report
            mean, components, ratio = fit_pca(stacked[complete], N_COMPONENTS)
            scores = np.full((stacked.shape[0], components.shape[0]), np.nan)
            scores[complete] = (stacked[complete] - mean) @ components.T
            point_time = None
            report.update(loading_axis=axis, loading_label="Time from event (s)")
        else:
            if method == METHOD_BOUT:
                n_delays = max(MIN_DELAY_SAMPLES, int(round(BOUT_DELAY_FRACTION * axis.size)))
            else:
                n_delays = delay_sample_count(float(self.options.get("span_s") or DEFAULT_DELAY_SPAN_S), axis)
            try:
                scores, centers, components, ratio = embed_units(stacked, n_delays)
            except ValueError as exc:
                report["note"] = str(exc)
                report["hard_error"] = True
                return report
            point_time = axis[centers]
            lags = (np.arange(n_delays) - (n_delays - 1) // 2) * step
            if method == METHOD_BOUT:
                report.update(loading_axis=100.0 * lags, loading_label="Lag inside the delay window (% of bout)")
            else:
                report.update(loading_axis=lags, loading_label="Lag inside the delay window (s)")
        report.update(ratio=ratio, components=components)

        # Blocks are contiguous by group. Keep views into the same decimated
        # traces so fitting local weights costs nothing until that panel opens.
        if len(groups) > 1:
            group_rows = []
            start = 0
            for gi, group_blocks in itertools.groupby(blocks, key=lambda block: block["group_index"]):
                stop = start + sum(len(block["rows"]) for block in group_blocks)
                _key, name, color, _members = groups[gi]
                group_rows.append((name, color or UNGROUPED_COLOR, stacked[start:stop]))
                start = stop
            report["loading_groups"] = group_rows
            report["loading_delays"] = None if method == METHOD_TRIALS else n_delays

        several_groups = len(groups) > 1
        several_behaviors = len(behaviors) > 1
        unit_word = "bouts" if method == METHOD_BOUT else "trials"
        start = 0
        for block in blocks:
            count = block["rows"].shape[0]
            unit_scores = scores[start:start + count]
            start += count
            files = block["files"]
            with np.errstate(all="ignore"):
                import warnings
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", category=RuntimeWarning)
                    mean_scores = np.nanmean(unit_scores, axis=0)
            n_recordings = len(set(files.tolist()))
            if report["individual_recording"]:
                label = f"{block['behavior']} · {block['group']} (n={count} {unit_word})"
            else:
                parts = ([block["behavior"]] if several_behaviors or not several_groups else []) + \
                        ([block["group"]] if several_groups or not several_behaviors else [])
                label = f"{' · '.join(parts)} (n={count} {unit_word}, {n_recordings} rec.)"
            report["curves"].append({
                "label": label,
                "behavior": block["behavior"], "behavior_index": block["behavior_index"],
                "group": block["group"], "group_color": block["group_color"] if several_groups else None,
                "time": point_time, "scores": mean_scores, "units": unit_scores, "unit_files": files,
                "n": count, "n_recordings": n_recordings})
        return report

    def _compute_recordings(self, behaviors: Sequence[str], groups) -> Dict[str, object]:
        """Recordings as dimensions: one trajectory per behavior, selected groups pooled."""
        report = self._empty_report(METHOD_RECORDINGS, "")
        report["behaviors"] = list(behaviors)
        responses, tvec, _info = self._trial_responses(behaviors, to_offset=False)
        if tvec is None:
            report["note"] = "No event-aligned response for the selected behaviors."
            return report
        wanted: List[str] = []
        for _key, _name, _color, members in groups:
            wanted.extend(fid for fid in members if fid not in wanted)
        usable = [fid for fid in wanted if all(fid in responses.get(b, {}) for b in behaviors)]
        report["skipped"] = [fid for fid in wanted if fid not in usable]
        report["recordings"] = usable
        if len(usable) < MIN_RECORDINGS_AS_DIMENSIONS:
            report["note"] = (f"Recordings as dimensions needs at least {MIN_RECORDINGS_AS_DIMENSIONS} recordings "
                              f"with every selected behavior ({len(usable)} available). Choose another method.")
            return report
        stacked = {}
        for behavior in behaviors:
            with np.errstate(all="ignore"):
                columns = [np.nanmean(responses[behavior][fid], axis=0) for fid in usable]
            matrix, axis = decimate_rows(np.vstack(columns), tvec)
            stacked[behavior] = (axis, matrix.T)
        trajectories, ratio, components = population_trajectories(stacked, return_model=True)
        for index, behavior in enumerate(behaviors):
            if behavior in trajectories:
                time, scores = trajectories[behavior]
                report["curves"].append({"label": f"{behavior} (n={len(usable)} rec.)", "behavior": behavior,
                                         "behavior_index": index, "group": None, "group_color": None,
                                         "time": time, "scores": scores, "units": None, "unit_files": None,
                                         "n": len(usable), "n_recordings": len(usable)})
        report.update(ratio=ratio, components=components, loading_axis=np.arange(len(usable), dtype=float),
                      loading_label="Recording", loading_ticks=list(usable))
        return report

    # ------------------------------------------------------------ separation
    def separation(self, report: Dict[str, object]) -> Dict[str, object]:
        """Permutation test between the first two conditions of a report (cached on it)."""
        if "separation" in report:
            return report["separation"]
        result: Dict[str, object] = {"available": False, "note": ""}
        report["separation"] = result
        curves = [curve for curve in report.get("curves", []) if curve.get("units") is not None]
        if report.get("method") == METHOD_RECORDINGS:
            result["note"] = ("With recordings as dimensions each behavior has a single trajectory, "
                              "so there is nothing to permute. Choose another method.")
            return result
        if len(curves) < 2:
            if report.get("individual_recording"):
                result["note"] = "The separation test needs two behaviors from the active recording."
            else:
                result["note"] = "The separation test needs two conditions: select two groups or two behaviors."
            return result
        first, second = curves[0], curves[1]
        result.update(label_a=first["label"], label_b=second["label"], extra_conditions=len(curves) - 2)
        files_a, files_b = set(first["unit_files"].tolist()), set(second["unit_files"].tolist())
        animal_level = (not files_a & files_b and min(len(files_a), len(files_b)) >= MIN_RECORDINGS_PER_CONDITION)
        after_event = report.get("method") == METHOD_LAG and first["time"] is not None

        def units_of(curve) -> np.ndarray:
            units = np.asarray(curve["units"], float)
            if animal_level:      # one mean trajectory per recording
                import warnings
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", category=RuntimeWarning)
                    units = np.stack([np.nanmean(units[curve["unit_files"] == fid], axis=0)
                                      for fid in sorted(set(curve["unit_files"].tolist()))])
            return units

        a, b = units_of(first), units_of(second)
        if a.ndim == 3:
            result["window"] = "after the event" if after_event else "from onset to offset"
            # Keep the points present in every unit (for event-aligned trajectories: after the event).
            keep = np.isfinite(a).all(axis=(0, 2)) & np.isfinite(b).all(axis=(0, 2))
            if after_event:
                keep &= np.asarray(first["time"]) >= 0.0
            if int(keep.sum()) < 2:
                # A few incomplete units would remove every point: drop those units instead.
                a = a[np.isfinite(a).all(axis=(1, 2))]
                b = b[np.isfinite(b).all(axis=(1, 2))]
                keep = np.ones(first["scores"].shape[0], bool)
                if after_event:
                    keep &= np.asarray(first["time"]) >= 0.0
            a, b = a[:, keep], b[:, keep]
        else:
            a, b = a[np.isfinite(a).all(axis=1)], b[np.isfinite(b).all(axis=1)]
            result["window"] = "between centroids"
        if a.shape[0] < 1 or b.shape[0] < 1 or a.shape[0] + b.shape[0] < 3 or a[0].size == 0:
            result["note"] = "Too few complete units to permute."
            return result
        result.update(permutation_separation(a, b))
        result.update(available=True, level="recordings" if animal_level else "trials")
        return result

    # --------------------------------------------------------------- drawing
    def _behavior_colors(self, palette: Dict[str, object]) -> Sequence[str]:
        text = QtGui.QColor(str(palette.get("text", "#000000")))
        return COLORS_ON_DARK if text.lightness() > 128 else COLORS_ON_LIGHT

    def _curve_color(self, curve, colors) -> str:
        return curve["group_color"] or colors[curve["behavior_index"] % len(colors)]

    def _prepare(self, widget: pg.PlotWidget) -> Tuple[Dict[str, object], str]:
        """Empty the plot and give it the current plot style (background, axes, text)."""
        palette = self.panel._current_palette()
        widget.clear()
        plot_item = widget.getPlotItem()
        legend = self._legends.pop(id(widget), None)
        if legend is not None:
            try:
                plot_item.layout.removeItem(legend)
                legend.setParentItem(None)
                if legend.scene() is not None:
                    legend.scene().removeItem(legend)
            except Exception:
                pass
        style_plot(widget, palette)
        plot_item.setAspectLocked(False)
        plot_item.getAxis("bottom").setTicks(None)
        for side in ("bottom", "left"):
            axis = plot_item.getAxis(side)
            axis.setStyle(showValues=True)
            # Reset any previous SI multiplier before the renderer sets its
            # label. A visible label can keep a stale x1000 tick multiplier.
            axis.showLabel(False)
            axis.enableAutoSIPrefix(False)
        # These are not time series: the shortcuts of the other plots (peak
        # downsampling, clipping to the view) would redraw curves as staircases.
        plot_item.setDownsampling(ds=1, auto=False)
        plot_item.setClipToView(False)
        widget.setProperty("hasPlotData", True)
        plot_item.setVisible(True)             # stays reachable by right-click even when empty
        return palette, str(palette.get("text", "#000000"))

    def _message(self, widget: pg.PlotWidget, text: str, color: str) -> None:
        """Show a wrapped message in an otherwise empty panel."""
        item = pg.TextItem(color=color, anchor=(0.5, 0.5))
        width = max(120, widget.width() - 70)
        item.setHtml(f'<div style="text-align:center; color:{color};">{text}</div>')
        item.setTextWidth(width)
        widget.addItem(item)
        item.setPos(0.5, 0.5)
        widget.setXRange(0, 1, padding=0)
        widget.setYRange(0, 1, padding=0)
        for axis in ("bottom", "left"):
            widget.setLabel(axis, "")
            widget.getPlotItem().getAxis(axis).setStyle(showValues=False)
        widget.setToolTip(text)

    def _legend(self, widget: pg.PlotWidget, text_color: str, entries: int, columns=None) -> pg.LegendItem:
        """Legend under the x axis, outside the data area, in as many columns as fit."""
        plot_item = widget.getPlotItem()
        if columns is None:
            columns = max(1, min(entries, widget.width() // LEGEND_COLUMN_WIDTH))
        legend = pg.LegendItem(labelTextColor=text_color, labelTextSize="8pt", colCount=columns)
        legend.setParentItem(plot_item)
        plot_item.layout.addItem(legend, LEGEND_LAYOUT_ROW, 1)
        self._legends[id(widget)] = legend
        return legend

    def _title(self, report: Dict[str, object], what: str = PANEL_LABEL) -> str:
        return f"{what} · {METHOD_TITLES.get(str(report.get('method')), '')}"

    def render(self, widget: pg.PlotWidget, code: str = CODE, report=None) -> None:
        """Draw the selected PCA panel in a dashboard plot."""
        palette, text_color = self._prepare(widget)
        if report is None:
            report = self.compute()
        if code == CODE_LOADINGS:
            self._render_loadings(widget, report, palette, text_color)
        elif code == CODE_SEPARATION:
            self._render_separation(widget, report, palette, text_color)
        elif code == CODE_CLUSTER_SEPARATION:
            self._render_cluster_separation(widget, report, palette, text_color)
        else:
            self._render_trajectories(widget, report, palette, text_color)
        recording = report.get("individual_recording")
        context = (f"Recording: {recording}" if recording else "", report.get("fallback"), widget.toolTip())
        widget.setToolTip("\n".join(filter(None, context)))

    def _resolve_axes(self, n_components: int) -> Tuple[int, Optional[int]]:
        """Chosen component pair, or the first pair that exists for this analysis."""
        ax, ay = (int(value) for value in (self.options.get("axes") or [0, 1]))
        if max(ax, ay) < n_components:
            return ax, ay
        return (0, 1) if n_components >= 2 else (0, None)

    @staticmethod
    def trial_clusters(report):
        """Fit once per report so scatter plots and silhouettes share the same assignments."""
        if "trial_clusters" not in report:
            scores = np.vstack([curve["units"] for curve in report["curves"]])
            result = fit_trial_clusters(scores)
            result["scores"] = scores  # Keep the exact row order for the optional silhouette calculation.
            report["trial_clusters"] = result
        return report["trial_clusters"]

    def _draw_trial_points(self, widget, report, colors, text_color, ax, ay, clusters):
        """Color trials by component and preserve their condition in symbols and recording centroids."""
        start = 0
        n_clusters = clusters.get("n_clusters", 0)
        cluster_brushes = [pg.mkBrush(colors[k % len(colors)]) for k in range(n_clusters)]
        condition_labels = [curve["label"] for curve in report["curves"]]
        cluster_labels = [f"Cluster {k + 1} (n={np.count_nonzero(clusters['labels'] == k)})"
                          for k in range(n_clusters)]
        if n_clusters:
            # A full condition name plus cluster labels can force the whole graphics
            # layout wider than the dashboard slot. Reserve the axis and legend swatches.
            condition_labels = [name.rsplit(" (n=", 1)[0] for name in condition_labels]
            if len(condition_labels) == 1 and report.get("individual_recording"):
                condition_labels[0] = str(report["individual_recording"])
        font = QtGui.QFont(widget.font())
        font.setPointSize(8)
        metrics = QtGui.QFontMetrics(font)
        available = max(100, widget.width() - 85)
        condition_labels = [metrics.elidedText(name, QtCore.Qt.TextElideMode.ElideRight, max(60, available - 55))
                            for name in condition_labels]
        all_labels = condition_labels + cluster_labels
        cell_width = max(metrics.horizontalAdvance(name) for name in all_labels) + 55
        columns = max(1, min(len(all_labels), available // cell_width))
        legend = self._legend(widget, text_color, len(all_labels), columns)
        for index, curve in enumerate(report["curves"]):
            color = self._curve_color(curve, colors)
            points = curve["units"]
            valid = np.all(np.isfinite(points), axis=1)
            finite = points[valid]
            files = curve["unit_files"][valid]
            symbol = TRIAL_SYMBOLS[index % len(TRIAL_SYMBOLS)] if n_clusters else "o"
            point_pen = pg.mkPen(None)
            if n_clusters:
                labels = clusters["labels"][start:start + len(points)][valid]
                brushes = [cluster_brushes[k] for k in labels]
                if len(condition_labels) > 1:
                    point_pen = pg.mkPen(color, width=1)
            else:
                ink = QtGui.QColor(color)
                ink.setAlpha(POINT_ALPHA)
                brushes = pg.mkBrush(ink)
            start += len(points)
            if len(finite):
                size = CLUSTER_POINT_SIZE if n_clusters else POINT_SIZE
                dots = pg.ScatterPlotItem(finite[:, ax], finite[:, ay] if ay is not None else np.zeros(len(finite)),
                                          symbol=symbol, size=size, pen=point_pen, brush=brushes)
                dots.setZValue(4)
                widget.addItem(dots)
            for fid in sorted(set(files.tolist())):
                centre = finite[files == fid].mean(axis=0)
                marker = pg.ScatterPlotItem(
                    [centre[ax]], [centre[ay] if ay is not None else 0.0], symbol="d",
                    size=RECORDING_MARKER_SIZE, pen=pg.mkPen(text_color, width=1), brush=pg.mkBrush(color))
                marker.setZValue(6)
                widget.addItem(marker)
            if n_clusters and len(condition_labels) == 1:
                sample = pg.PlotDataItem([0], [0], pen=None, symbol="d", symbolSize=RECORDING_MARKER_SIZE,
                                         symbolBrush=pg.mkBrush(color), symbolPen=pg.mkPen(text_color))
            else:
                sample = pg.PlotDataItem([0], [0], pen=None, symbol=symbol, symbolSize=POINT_SIZE + 2,
                                         symbolBrush=pg.mkBrush(None if n_clusters else color), symbolPen=point_pen)
            legend.addItem(sample, escape(condition_labels[index]))

        # Take the two-dimensional marginal of the same fitted component for every axis choice.
        for k, cluster_label in enumerate(cluster_labels):
            pen = pg.mkPen(colors[k % len(colors)], width=CLUSTER_OUTLINE_WIDTH)
            if ay is not None:
                contour = cluster_ellipse(clusters["means"][k], clusters["covariances"][k], ax, ay)
                sample = widget.plot(contour[:, 0], contour[:, 1], pen=pen, antialias=True, name=cluster_label)
            else:
                sample = pg.PlotDataItem(pen=pen)
            legend.addItem(sample, cluster_label)

    def _render_trajectories(self, widget, report, palette, text_color) -> None:
        widget.setTitle(self._title(report), color=text_color, size="10pt")
        curves = report["curves"]
        ratio = np.asarray(report.get("ratio", np.empty(0)), float)
        if not curves or ratio.size == 0:
            self._message(widget, str(report.get("note") or "Nothing to show for the current selection."), text_color)
            return
        ax, ay = self._resolve_axes(ratio.size)
        colors = self._behavior_colors(palette)
        method = report["method"]
        clusters = self.trial_clusters(report) if method == METHOD_TRIALS and self.options["trial_clusters"] else {}
        if method == METHOD_TRIALS:
            self._draw_trial_points(widget, report, colors, text_color, ax, ay, clusters)
        else:
            legend = self._legend(widget, text_color, len(curves))
            self._draw_time_trajectories(widget, curves, method, legend, colors, palette, ax, ay)
        plot_item = widget.getPlotItem()
        if ay is None and method == METHOD_TRIALS:
            widget.setLabel("bottom", f"PC1 ({100 * ratio[0]:.0f}% of variance)")
            widget.setLabel("left", "")
            plot_item.getAxis("left").setStyle(showValues=False)
        elif ay is None:
            widget.setLabel("bottom", "Phase of the bout (0 = onset, 1 = offset)" if method == METHOD_BOUT
                            else "Time from event (s)")
            widget.setLabel("left", f"PC1 ({100 * ratio[0]:.0f}% of variance)")
        else:
            widget.setLabel("bottom", f"PC{ax + 1} ({100 * ratio[ax]:.0f}% of variance)")
            widget.setLabel("left", f"PC{ay + 1} ({100 * ratio[ay]:.0f}% of variance)")
            plot_item.setAspectLocked(True)        # both axes share the signal's units
        plot_item.enableAutoRange()
        widget.setToolTip(self._tooltip(report))

    def _draw_time_trajectories(self, widget, curves, method, legend, colors, palette, ax, ay):
        """Draw the trial or bout paths and each condition's mean in the chosen plane."""
        show_units = bool(self.options.get("units", True))
        surface = str(palette.get("surface", "#000000"))
        for curve in curves:
            color = self._curve_color(curve, colors)
            style = LINE_STYLES[curve["behavior_index"] % len(LINE_STYLES)] if curve["group_color"] else LINE_STYLES[0]
            faint = QtGui.QColor(color)
            units = curve.get("units")
            mean = np.asarray(curve["scores"], float)
            time = np.asarray(curve["time"], float)
            if show_units and units is not None and units.shape[0] > 1:
                # Single trials / bouts: one faint line each, drawn as one item.
                chosen = units if units.shape[0] <= MAX_UNIT_LINES else units[
                    np.linspace(0, units.shape[0] - 1, MAX_UNIT_LINES).astype(int)]
                gap = np.full((chosen.shape[0], 1), np.nan)
                if ay is not None:
                    xs = np.concatenate([chosen[..., ax], gap], axis=1).ravel()
                    ys = np.concatenate([chosen[..., ay], gap], axis=1).ravel()
                else:
                    xs = np.concatenate([np.tile(time, (chosen.shape[0], 1)), gap], axis=1).ravel()
                    ys = np.concatenate([chosen[..., ax], gap], axis=1).ravel()
                faint.setAlpha(UNIT_LINE_ALPHA)
                widget.plot(xs, ys, pen=pg.mkPen(faint, width=UNIT_LINE_WIDTH), connect="finite", antialias=True)
            good = np.isfinite(mean[:, ax]) & (np.isfinite(mean[:, ay]) if ay is not None else True)
            if good.sum() < 2:
                continue
            x = mean[good, ax] if ay is not None else time[good]
            y = mean[good, ay] if ay is not None else mean[good, ax]
            line = widget.plot(x, y, pen=pg.mkPen(color, width=MEAN_LINE_WIDTH, style=style), antialias=True)
            line.setZValue(5)
            legend.addItem(line, curve["label"])
            t_good = time[good]
            if method == METHOD_BOUT:   # onset: hollow circle; offset: square
                marks = [(0, "o", None), (x.size - 1, "s", color)]
            else:                       # start: hollow circle; event: diamond; end: triangle
                marks = [(0, "o", None), (int(np.argmin(np.abs(t_good))), "d", color), (x.size - 1, "t1", color)]
                if not (t_good[0] <= 0.0 <= t_good[-1]):
                    marks.pop(1)
            for index, symbol, fill in marks:
                marker = pg.ScatterPlotItem([x[index]], [y[index]], symbol=symbol, size=MARKER_SIZE,
                                            pen=pg.mkPen(color, width=1.6), brush=pg.mkBrush(fill or surface))
                marker.setZValue(6)
                widget.addItem(marker)

    def _tooltip(self, report: Dict[str, object]) -> str:
        method = report.get("method")
        tips = []
        if method == METHOD_BOUT:
            tips.append("Each bout from its onset (hollow circle) to its offset (square), on a common phase axis. "
                        "Thin lines: single bouts. Thick lines: condition means.")
        elif method == METHOD_LAG:
            tips.append("Each trial around its event. Hollow circle: start of the window. Diamond: event time. "
                        "Triangle: end. Thin lines: single trials. Thick lines: condition means.")
        elif method == METHOD_TRIALS:
            tips.append("One point per event. Diamonds: mean of each recording.")
            tips.extend(curve["label"] for curve in report["curves"])
            if self.options["trial_clusters"] and "trial_clusters" in report:
                clusters = report["trial_clusters"]
                if clusters["n_clusters"]:
                    tips.append("Point colors give Gaussian mixture assignments. Shapes identify conditions. "
                                "Outlines enclose 95% of each fitted Gaussian in the displayed plane. "
                                "The lowest BIC among eligible models selects the component count.")
                if clusters["note"]:
                    tips.append(clusters["note"])
        else:
            tips.append(f"Dimensions: {len(report.get('recordings', []))} recordings (selected groups pooled).")
        info = report.get("info") or {}
        if info.get("truncated"):
            tips.append(f"{info['truncated']} bout(s) longer than {BOUT_MAX_S:g} s were cut there.")
        if method == METHOD_BOUT and info.get("without_duration"):
            tips.append(f"{info['without_duration']} event(s) without a duration were left out.")
        skipped = report.get("skipped") or []
        if skipped:
            tips.append("Left out (no response): " + ", ".join(str(item) for item in skipped[:6])
                        + (" ..." if len(skipped) > 6 else ""))
        tips.append("Descriptive view. 'PCA separation test' gives a permutation test, "
                    "'PCA loadings' the meaning of the axes (Change to > State-space).")
        return "\n".join(tips)

    @staticmethod
    def _pc_pen(color, pc):
        """Preserve the group hue and encode the component twice for legibility."""
        ink = QtGui.QColor(color)
        ink.setAlpha(PC_ALPHA[pc])
        return pg.mkPen(ink, width=2, style=LINE_STYLES[pc])

    @staticmethod
    def _group_weights(report):
        """Fit each group's feature weights on demand, pooling its selected behaviors."""
        if "group_models" in report:
            return report["group_models"]
        models = []
        delays = report["loading_delays"]
        for name, color, traces in report["loading_groups"]:
            samples = (traces if delays is None else
                       np.lib.stride_tricks.sliding_window_view(traces, delays, axis=1))
            samples = samples[np.isfinite(samples).all(axis=-1)]
            if len(samples) < 2:
                continue
            if delays is None:
                _, weights, ratio = fit_pca(samples)
            else:
                _, weights, ratio = _fit_delay_samples(samples, N_COMPONENTS, FIT_MAX_SAMPLES)
            # A zero-variance direction has no identifiable loading. Orient
            # independent fits toward the shared PC of the same variance rank.
            count = np.count_nonzero(ratio > ratio[0] * np.finfo(float).eps)
            if count:
                weights = weights[:count]
                shared = report["components"][:count]
                weights[(weights * shared).sum(axis=1) < 0] *= -1
                models.append((name, color, weights, ratio[:count]))
        report["group_models"] = models
        return models

    def _render_loadings(self, widget, report, palette, text_color) -> None:
        if report["method"] == METHOD_LAG and not self.options["show_lag_weights"]:
            self._render_event_scores(widget, report, palette, text_color)
            return
        widget.setTitle(self._title(report, "PCA loadings"), color=text_color, size="10pt")
        components = np.asarray(report.get("components", np.empty((0, 0))), float)
        ratio = np.asarray(report.get("ratio", np.empty(0)), float)
        if components.size == 0:
            self._message(widget, str(report.get("note") or "Nothing to show for the current selection."), text_color)
            return
        x = np.asarray(report.get("loading_axis"), float)
        colors = self._behavior_colors(palette)
        ticks = report.get("loading_ticks")
        grouped = bool(report.get("loading_groups")) and self.options["group_loadings"]
        models = self._group_weights(report) if grouped else [(None, None, components, ratio)]
        if grouped:
            widget.setTitle("PC loadings by group", color=text_color, size="10pt")
        if not models:
            self._message(widget, "No group has two complete samples with measurable variation.", text_color)
            return
        legend = self._legend(widget, text_color, sum(len(weights) for _, _, weights, _ in models))
        for name, group_color, weights, fractions in models:
            for index, row in enumerate(weights):
                color = group_color or colors[index % len(colors)]
                pen = self._pc_pen(color, index) if grouped else pg.mkPen(color, width=2)
                if ticks is not None:      # a weight per recording
                    line = widget.plot(x, row, pen=pg.mkPen(color, width=1.2), symbol="o", symbolSize=7,
                                       symbolBrush=pg.mkBrush(color), symbolPen=pg.mkPen(None), antialias=True)
                else:
                    line = widget.plot(x, row, pen=pen, antialias=True)
                prefix = f"{escape(name)} · " if grouped else ""
                legend.addItem(line, f"{prefix}PC{index + 1} ({100 * fractions[index]:.0f}%)")
        zero = pg.InfiniteLine(pos=0.0, angle=0, pen=pg.mkPen(text_color, width=0.6, style=QtCore.Qt.PenStyle.DotLine))
        zero.setZValue(-1)
        widget.addItem(zero)
        if ticks is not None:
            widget.getPlotItem().getAxis("bottom").setTicks([[(float(i), str(name)) for i, name in enumerate(ticks)]])
        widget.setLabel("bottom", str(report.get("loading_label", "")))
        widget.setLabel("left", "Loading (weight)")
        widget.getPlotItem().enableAutoRange()
        embedding = ("A flat component follows the local signal level; one that changes sign across "
                     "the window follows the local slope.")
        meaning = {METHOD_BOUT: embedding, METHOD_LAG: embedding,
                   METHOD_TRIALS: "Each component is a time course: a trial scores high when its response looks like it.",
                   METHOD_RECORDINGS: "Weight of each recording in the component."}
        tip = meaning.get(str(report.get("method")), "")
        if grouped:
            tip += (" Each group has its own PCA of the selected behaviors. PC rank and variance percentages "
                    "are within group. Signs follow the shared component of the same rank where possible.")
            shown = {name for name, _, _, _ in models}
            skipped = [name for name, _, _ in report["loading_groups"] if name not in shown]
            if skipped:
                tip += " Insufficient variation or complete data: " + ", ".join(skipped) + "."
        widget.setToolTip(tip)

    def _render_event_scores(self, widget, report, palette, text_color) -> None:
        """Draw the coordinates of the shared lag trajectories on the PSTH clock."""
        widget.setTitle("Event-aligned PC scores", color=text_color, size="10pt")
        curves = report["curves"]
        if not curves:
            self._message(widget, str(report["note"] or "No event-aligned trajectories."), text_color)
            return

        # The clock is the one used to extract the trials, including offset,
        # transition and marker alignment. An event name can contain HTML characters.
        behaviors = report["behaviors"]
        event = behaviors[0] if len(behaviors) == 1 else "behavior"
        source = self.panel.combo_align.currentText()
        alignment = (self.panel.combo_behavior_align if source.startswith("Behavior")
                     else self.panel.combo_dio_align).currentText()
        if source != "Event marker" and alignment.startswith("Align to "):
            event += " " + alignment.removeprefix("Align to ")
        widget.setLabel("bottom", f"Time from {escape(event)} (s)")
        widget.setLabel("left", "Mean PC score")

        ratios = report["ratio"]
        colors = self._behavior_colors(palette)
        many_behaviors = len(behaviors) > 1
        many_groups = bool(report.get("loading_groups"))
        several = many_groups or len(curves) > 1
        legend = self._legend(widget, text_color, len(ratios) + (len(curves) if several else 0))
        for ci, curve in enumerate(curves):
            color = self._curve_color(curve, colors)
            if many_groups:
                color = curve["group_color"] or UNGROUPED_COLOR
            # Sparse symbols distinguish behaviors when hue and dash pattern
            # are already assigned to group and PC respectively.
            symbol = (BEHAVIOR_SYMBOLS[curve["behavior_index"] % len(BEHAVIOR_SYMBOLS)]
                      if many_behaviors and many_groups else None)
            for pc, fraction in enumerate(ratios):
                pen = self._pc_pen(color, pc) if several else pg.mkPen(colors[pc], width=2)
                line = widget.plot(curve["time"], curve["scores"][:, pc], pen=pen,
                                   connect="finite", antialias=True)
                if symbol:
                    stride = max(1, len(curve["time"]) // 8)
                    times = curve["time"][::stride]
                    values = curve["scores"][::stride, pc]
                    ok = np.isfinite(values)
                    widget.addItem(pg.ScatterPlotItem(times[ok], values[ok], symbol=symbol, size=5,
                                                      brush=pen.color(), pen=None))
                if ci == 0:
                    key = pg.PlotDataItem(pen=self._pc_pen(text_color, pc)) if several else line
                    legend.addItem(key, f"PC{pc + 1} ({100 * fraction:.0f}%)")
            if several:
                name = " · ".join(part for part in
                                  (curve["behavior"] if many_behaviors else "", curve["group"] if many_groups else "")
                                  if part)
                legend.addItem(pg.PlotDataItem(pen=pg.mkPen(color, width=2), symbol=symbol, symbolSize=5,
                                               symbolBrush=color, symbolPen=None), escape(name))

        event_line = pg.InfiniteLine(pos=0, angle=90, pen=pg.mkPen(
            text_color, width=1, style=QtCore.Qt.PenStyle.DashLine))
        event_line.setZValue(-1)
        widget.addItem(event_line, ignoreBounds=True)
        widget.getPlotItem().enableAutoRange()
        widget.setXRange(-self.panel.spin_pre.value(), self.panel.spin_post.value(), padding=0.01)
        tip = ("Each line shows a condition's trial-mean PC score from the trajectory fit. "
               "The vertical line marks the selected event. Scores are shown at delay-window "
               "centers with valid data. Choose ‘Show delay-window weights’ to inspect the loadings.")
        widget.setToolTip(tip)

    def _render_cluster_separation(self, widget, report, palette, text_color):
        """Draw each trial's silhouette, sorted within its assigned mixture component."""
        widget.setTitle("Cluster separation", color=text_color, size="10pt")
        if report["method"] != METHOD_TRIALS:
            self._message(widget, "Choose Trial-response PCA under PCA options to see cluster separation.", text_color)
            return
        if not report["curves"]:
            self._message(widget, report.get("note") or "No trial responses are available.", text_color)
            return
        result = self.trial_clusters(report)
        if result["n_clusters"] < 2:
            self._message(widget, result["note"], text_color)
            return

        silhouettes = trial_cluster_silhouettes(result["scores"], result)
        mean = float(np.nanmean(silhouettes))
        widget.setTitle(f"Cluster separation · mean {mean:.2f}", color=text_color, size="10pt")
        colors = self._behavior_colors(palette)
        ticks, counts = [], []
        position = 0
        gap = max(1, round(np.count_nonzero(np.isfinite(silhouettes)) * 0.035))
        for k in range(result["n_clusters"]):
            values = np.sort(silhouettes[result["labels"] == k])
            xs = position + np.arange(len(values))
            widget.addItem(pg.BarGraphItem(x=xs, height=values, width=0.8, pen=pg.mkPen(None),
                                           brush=pg.mkBrush(colors[k % len(colors)])))
            ticks.append((position + (len(values) - 1) / 2, f"C{k + 1}"))
            counts.append(f"C{k + 1}: {len(values)} trials")
            position += len(values) + gap
        widget.addItem(pg.InfiniteLine(pos=0, angle=0, pen=pg.mkPen(text_color, width=0.7)))
        widget.addItem(pg.InfiniteLine(pos=mean, angle=0,
                                       pen=pg.mkPen(text_color, width=1.5, style=QtCore.Qt.PenStyle.DashLine)))
        widget.setLabel("bottom", "Trials by cluster")
        widget.setLabel("left", "Silhouette score")
        widget.getAxis("bottom").setTicks([ticks])
        widget.setXRange(-0.5, position - gap - 0.5, padding=0.015)
        widget.setYRange(-1, 1, padding=0.025)
        tip = [result["note"], ", ".join(counts), f"Dashed line: mean silhouette {mean:.3f}.",
               "Each bar compares one trial's mean distance to its own cluster with its mean distance "
               "to the nearest other cluster, across the retained PC scores. Values near 1 indicate "
               "separation, near 0 indicate overlap, and negative values favor a different cluster."]
        if result["silhouette_sampled"]:
            tip.append("Distances use a reproducible reference sample within each large cluster.")
        widget.setToolTip("\n".join(tip))

    def _render_separation(self, widget, report, palette, text_color) -> None:
        widget.setTitle(self._title(report, "PCA separation"), color=text_color, size="10pt")
        if not report["curves"]:
            self._message(widget, str(report.get("note") or "Nothing to show for the current selection."), text_color)
            return
        result = self.separation(report)
        if not result.get("available"):
            self._message(widget, str(result.get("note")), text_color)
            return
        level = "animal-level" if result["level"] == "recordings" else "trial-level"
        widget.setTitle(f"Separation · {level} permutation (p = {result['p']:.3g})", color=text_color, size="10pt")
        null = np.asarray(result["null"], float)
        observed = float(result["observed"])
        low, high = float(min(null.min(), observed)), float(max(null.max(), observed))
        span = (high - low) or max(abs(high), 1e-9)
        edges = np.linspace(low - 0.02 * span, high + 0.04 * span, HISTOGRAM_BINS + 1)
        counts, _ = np.histogram(null, bins=edges)
        fill = QtGui.QColor(self._behavior_colors(palette)[0])
        fill.setAlpha(170)
        widget.addItem(pg.BarGraphItem(x0=edges[:-1], x1=edges[1:], height=counts, pen=pg.mkPen(None),
                                       brush=pg.mkBrush(fill)))
        widget.addItem(pg.InfiniteLine(pos=observed, angle=90,
                                       pen=pg.mkPen(text_color, width=1.8, style=QtCore.Qt.PenStyle.DashLine)))
        widget.setLabel("bottom", f"Mean separation, {result.get('window', '')} (dashed: observed)")
        widget.setLabel("left", "Labelings" if result["exact"] else "Permutations")
        widget.setXRange(float(edges[0]), float(edges[-1]), padding=0.02)
        widget.setYRange(0, float(max(counts.max(), 1)) * 1.08, padding=0)
        unit = "recordings" if result["level"] == "recordings" else "trials"
        lines = [f"{result['label_a']}  vs  {result['label_b']}",
                 f"Observed separation {observed:.4g} (dashed line); {result['n_a']} vs {result['n_b']} {unit}.",
                 ("Exact test over all " if result["exact"] else "Random permutations: ")
                 + f"{result['n_permutations']} labelings; p = {result['p']:.4g}."]
        if result["level"] != "recordings":
            lines.append(f"Trials were permuted because a condition has fewer than {MIN_RECORDINGS_PER_CONDITION} "
                         "recordings (or the conditions share recordings). Trials of one animal are not "
                         "independent, so this p-value is exploratory.")
        if result.get("extra_conditions"):
            lines.append(f"{result['extra_conditions']} further condition(s) are shown but not tested.")
        widget.setToolTip("\n".join(lines))

    # ------------------------------------------------------------------ menu
    def build_menu(self, parent: QtWidgets.QMenu, column_report=None) -> QtWidgets.QMenu:
        """Right-click options shared by the PCA panels: method, behaviors, groups, axes."""
        menu = QtWidgets.QMenu("PCA options", parent)
        report = self.last_report if column_report is None else column_report

        methods = menu.addMenu("Method")
        group_only_method = self.options["method"] == METHOD_RECORDINGS and self.individual_recording() is not None
        effective_method = METHOD_TRIALS if group_only_method else self.options["method"]
        for value, label in METHODS:
            if group_only_method and value == METHOD_TRIALS:
                label += " (Individual view)"
            if group_only_method and value == METHOD_RECORDINGS:
                label += " (selected for Group view)"
            action = methods.addAction(label)
            action.setCheckable(True)
            action.setChecked(effective_method == value)
            if value == METHOD_RECORDINGS and self.individual_recording() is not None:
                action.setEnabled(False)
                action.setToolTip("Switch to Group view to use recordings as dimensions.")
            action.triggered.connect(lambda _=False, v=value: self.set_option("method", v))

        behaviors = menu.addMenu("Behaviors")
        if column_report is not None:
            behaviors.addAction(", ".join(report["behaviors"])).setEnabled(False)
        elif self.behaviors_apply():
            chosen = [name for name in (self.options.get("behaviors") or []) if name in self.available_behaviors()]
            follow = behaviors.addAction("Follow the PSTH behavior")
            follow.setCheckable(True)
            follow.setChecked(not chosen)
            follow.triggered.connect(lambda: self.set_option("behaviors", []))
            behaviors.addSeparator()
            for name in self.available_behaviors():
                action = behaviors.addAction(name)
                action.setCheckable(True)
                action.setChecked(name in chosen)
                action.triggered.connect(lambda checked, n=name: self._toggle_behavior(n, checked))
        else:
            note = behaviors.addAction("Align the PSTH to a behavior to choose several")
            note.setEnabled(False)

        self._add_scope_menu(menu)

        units = menu.addAction("Show single trials / bouts")
        units.setCheckable(True)
        units.setChecked(bool(self.options.get("units", True)))
        units.triggered.connect(lambda checked: self.set_option("units", bool(checked)))

        spans = menu.addMenu("Delay window")
        spans.setEnabled(self.options.get("method") == METHOD_LAG)
        for span in DELAY_SPANS_S:
            action = spans.addAction(f"{span:g} s")
            action.setCheckable(True)
            action.setChecked(abs(float(self.options.get("span_s") or 0) - span) < 1e-9)
            action.triggered.connect(lambda _=False, s=span: self.set_option("span_s", float(s)))

        if report.get("method", self.options["method"]) == METHOD_LAG:
            weights = menu.addAction("Show delay-window weights")
            weights.setCheckable(True)
            weights.setChecked(bool(self.options["show_lag_weights"]))
            weights.triggered.connect(lambda checked: self.set_option("show_lag_weights", checked))

        if report.get("method", self.options["method"]) == METHOD_TRIALS:
            clusters = menu.addAction("Show GMM clusters")
            clusters.setToolTip("Color trials by cluster and show the fitted 95% outlines.")
            clusters.setCheckable(True)
            clusters.setChecked(bool(self.options["trial_clusters"]))
            clusters.triggered.connect(lambda checked: self.set_option("trial_clusters", bool(checked)))

        if report.get("loading_groups"):
            by_group = menu.addAction("Fit weights separately by group")
            by_group.setCheckable(True)
            by_group.setEnabled(report.get("method") != METHOD_LAG or self.options["show_lag_weights"])
            by_group.setChecked(bool(self.options["group_loadings"]))
            by_group.triggered.connect(lambda checked: self.set_option("group_loadings", checked))

        axes = menu.addMenu("Axes")
        for first, second in AXES_CHOICES:
            action = axes.addAction(f"PC{first + 1} vs PC{second + 1}")
            action.setCheckable(True)
            action.setChecked(list(self.options.get("axes") or [0, 1]) == [first, second])
            action.triggered.connect(lambda _=False, a=(first, second): self.set_option("axes", list(a)))
        return menu

    def _toggle_behavior(self, name: str, checked: bool) -> None:
        available = self.available_behaviors()
        chosen = [item for item in (self.options.get("behaviors") or []) if item in available]
        if checked and name not in chosen:
            chosen.append(name)
        elif not checked and name in chosen:
            chosen.remove(name)
        self.set_option("behaviors", [item for item in available if item in chosen])

    def _add_scope_menu(self, menu: QtWidgets.QMenu) -> None:
        """Expose group filters only where the toolbar permits multiple recordings."""
        recording = self.individual_recording()
        if recording is not None:
            menu.addAction(f"Recording · {recording}" if recording else "No recording selected").setEnabled(False)
            return
        groups = menu.addMenu("Groups")
        chosen_groups = self.options.get("groups")
        automatic = groups.addAction("Automatic (groups shown by Hide/show groups)")
        automatic.setCheckable(True)
        automatic.setChecked(chosen_groups is None)
        automatic.triggered.connect(lambda: self.set_option("groups", None))
        groups.addSeparator()
        pooled = groups.addAction("All recordings pooled")
        pooled.setCheckable(True)
        pooled.setChecked(chosen_groups is not None and POOLED in chosen_groups)
        pooled.triggered.connect(lambda checked: self._toggle_group(POOLED, checked))
        for key, name, color, members in self.group_entries():
            action = groups.addAction(color_dot_icon(color), f"{name} ({len(members)})")
            action.setCheckable(True)
            action.setChecked(chosen_groups is not None and key in chosen_groups)
            action.triggered.connect(lambda checked, k=key: self._toggle_group(k, checked))

    def _toggle_group(self, key: str, checked: bool) -> None:
        """Select either one pooled sample or a set of distinct group samples."""
        chosen = list(self.options.get("groups") or [])
        if checked and key == POOLED:
            self.set_option("groups", [POOLED])
            return
        if checked and POOLED in chosen:
            chosen.remove(POOLED)
        if checked and key not in chosen:
            chosen.append(key)
        elif not checked and key in chosen:
            chosen.remove(key)
        self.set_option("groups", chosen if chosen else None)
