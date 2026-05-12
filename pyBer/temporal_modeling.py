# temporal_modeling.py
"""
Temporal Modeling module for pyBer post-processing.

Provides two backends:
  1. ContinuousGLM – design-matrix approach with temporal basis functions
     (raised-cosine, B-spline, FIR) and ridge/lasso regression.
  2. TrialFLMM – Functional Linear Mixed Model via the R *fastFMM* package
     (Loewinger et al., 2024), called through rpy2.

The TemporalModelingWidget is a PySide6 panel that is embedded in the
PostProcessingPanel side-rail/dock system.
"""
from __future__ import annotations

import concurrent.futures
import csv
import hashlib
import json
import logging
import math
import os
import re
import sys
import traceback
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PySide6 import QtCore, QtWidgets, QtGui
import pyqtgraph as pg

_LOG = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional dependency detection
# ---------------------------------------------------------------------------

def _check_rpy2() -> bool:
    """Return True if rpy2 can talk to R and fastFMM is loadable."""
    try:
        _init_r()
        return True
    except Exception:
        return False


def _init_r():
    """Initialise rpy2 + R + fastFMM.  Idempotent."""
    global _R_READY
    if _R_READY:
        return

    def _candidate_r_homes() -> List[str]:
        roots: List[str] = []
        for prefix in (os.environ.get("CONDA_PREFIX", ""), sys.prefix, os.path.dirname(sys.executable)):
            if not prefix:
                continue
            roots.extend([
                os.path.join(prefix, "Lib", "R"),
                os.path.join(prefix, "lib", "R"),
                os.path.join(prefix, "R"),
            ])
        program_files = "C:/Program Files/R"
        if os.path.isdir(program_files):
            subs = sorted(os.listdir(program_files), reverse=True)
            roots.extend(os.path.join(program_files, sub) for sub in subs)
        return roots

    # On Windows, R is often not on PATH. Try active envs first, then Program Files.
    r_home = os.environ.get("R_HOME", "").strip()
    if not r_home or not os.path.isdir(r_home):
        for candidate in _candidate_r_homes():
            if os.path.isdir(candidate):
                r_home = candidate
                os.environ["R_HOME"] = r_home
                break
    if r_home:
        r_bins = [os.path.join(r_home, "bin", "x64"), os.path.join(r_home, "bin")]
        for bin_dir in r_bins:
            if not os.path.isdir(bin_dir):
                continue
            try:
                os.add_dll_directory(bin_dir)
            except (OSError, AttributeError):
                pass
        # Ensure PATH includes R so child processes find R.dll.
        cur_path = os.environ.get("PATH", "")
        for bin_dir in reversed(r_bins):
            if os.path.isdir(bin_dir) and bin_dir not in cur_path:
                cur_path = bin_dir + os.pathsep + cur_path
        os.environ["PATH"] = cur_path

    # rpy2 may call R's config.sh on Windows; make sh/make visible if the env has them.
    tool_bins: List[str] = []
    for prefix in (os.environ.get("CONDA_PREFIX", ""), sys.prefix):
        if prefix:
            tool_bins.append(os.path.join(prefix, "Library", "usr", "bin"))
    cur_path = os.environ.get("PATH", "")
    for tool_bin in reversed(tool_bins):
        if os.path.isdir(tool_bin) and tool_bin not in cur_path:
            cur_path = tool_bin + os.pathsep + cur_path
    os.environ["PATH"] = cur_path

    r_libs_user = os.environ.get("R_LIBS_USER", "")
    if not r_libs_user:
        candidate_roots = [
            os.path.join(os.environ.get("LOCALAPPDATA", ""), "R", "win-library"),
            os.path.expanduser("~/R/win-library"),
        ]
        for candidate_lib in candidate_roots:
            if os.path.isdir(candidate_lib):
                subs = sorted(os.listdir(candidate_lib), reverse=True)
                if subs:
                    os.environ["R_LIBS_USER"] = os.path.join(candidate_lib, subs[0])
                    break

    import rpy2.robjects as ro  # noqa: F811
    from rpy2.robjects import r as R  # noqa: F811

    # Set .libPaths so R can find user-installed packages
    user_lib = os.environ.get("R_LIBS_USER", "")
    if user_lib and os.path.isdir(user_lib):
        sys_lib = os.path.join(r_home, "library") if r_home else ""
        paths = [p for p in (user_lib, sys_lib) if p and os.path.isdir(p)]
        R(f'.libPaths(c({", ".join(repr(p.replace(chr(92), "/")) for p in paths)}))')

    R("library(fastFMM)")
    _R_READY = True


_R_READY = False
_BEHAVIOR_PARSE_BINARY = "binary_columns"
_BEHAVIOR_PARSE_TIMESTAMPS = "timestamp_columns"


# ============================================================================
# 1. Continuous GLM backend
# ============================================================================

def _raised_cosine_basis(n_basis: int, n_samples: int,
                         peak_range: Tuple[float, float] = (0.0, 1.0)) -> np.ndarray:
    """Create a raised-cosine basis set (n_samples x n_basis)."""
    peaks = np.linspace(peak_range[0], peak_range[1], n_basis)
    width = (peak_range[1] - peak_range[0]) / max(n_basis - 1, 1) * 1.5
    t = np.linspace(0, 1, n_samples)
    B = np.zeros((n_samples, n_basis))
    for i, pk in enumerate(peaks):
        phi = np.clip((t - pk) / width * np.pi, -np.pi, np.pi)
        B[:, i] = 0.5 * (1 + np.cos(phi))
    return B


def _bspline_basis(n_basis: int, n_samples: int, degree: int = 3) -> np.ndarray:
    """Create a B-spline basis (n_samples x n_basis) using scipy."""
    from scipy.interpolate import BSpline
    t_eval = np.linspace(0, 1, n_samples)
    n_internal = n_basis - degree + 1
    internal_knots = np.linspace(0, 1, n_internal + 2)[1:-1]
    knots = np.concatenate([np.zeros(degree + 1), internal_knots, np.ones(degree + 1)])
    B = np.zeros((n_samples, n_basis))
    for i in range(n_basis):
        coeffs = np.zeros(n_basis)
        coeffs[i] = 1.0
        spl = BSpline(knots, coeffs, degree, extrapolate=False)
        vals = spl(t_eval)
        vals[np.isnan(vals)] = 0.0
        B[:, i] = vals
    return B


def _fir_basis(n_basis: int, n_samples: int) -> np.ndarray:
    """FIR (identity / boxcar) basis — one column per time bin."""
    step = max(1, n_samples // n_basis)
    B = np.zeros((n_samples, n_basis))
    for i in range(n_basis):
        lo = i * step
        hi = min(lo + step, n_samples)
        B[lo:hi, i] = 1.0
    return B


def _glm_kernel_geometry(
    time: np.ndarray,
    kernel_window: Tuple[float, float],
) -> Tuple[float, int, int, np.ndarray]:
    """Return dt, pre samples, post samples, and an inclusive kernel time axis."""
    t = np.asarray(time, float)
    if t.size < 3:
        raise ValueError("Need at least 3 time samples for GLM fitting.")
    diffs = np.diff(t)
    diffs = diffs[np.isfinite(diffs)]
    dt = float(np.nanmedian(diffs)) if diffs.size else float("nan")
    if not np.isfinite(dt) or dt <= 0:
        raise ValueError("GLM time vector must be strictly increasing.")
    pre_samp = max(0, int(round(abs(float(kernel_window[0])) / dt)))
    post_samp = max(0, int(round(abs(float(kernel_window[1])) / dt)))
    kernel_tvec = np.arange(-pre_samp, post_samp + 1, dtype=float) * dt
    if kernel_tvec.size < 2:
        kernel_tvec = np.array([0.0, dt], float)
        post_samp = max(post_samp, 1)
    return dt, pre_samp, post_samp, kernel_tvec


def _glm_basis_matrix(n_basis: int, kernel_len: int, basis_type: str) -> np.ndarray:
    """Construct a temporal basis on the already-decided kernel sample grid."""
    n_basis = max(1, int(n_basis))
    kernel_len = max(2, int(kernel_len))
    if basis_type == "bspline":
        return _bspline_basis(n_basis, kernel_len)
    if basis_type == "fir":
        return _fir_basis(n_basis, kernel_len)
    return _raised_cosine_basis(n_basis, kernel_len)


def _glm_convolved_columns(input_vec: np.ndarray, basis: np.ndarray, pre_samp: int, n_time: int) -> np.ndarray:
    """Convolve without circular wrap so end-of-recording events cannot leak to the start."""
    v = np.asarray(input_vec, float)
    v = np.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0)
    n_basis = int(basis.shape[1])
    cols = np.zeros((int(n_time), n_basis), float)
    for b in range(n_basis):
        conv = np.convolve(v, np.asarray(basis[:, b], float), mode="full")
        cols[:, b] = conv[int(pre_samp):int(pre_samp) + int(n_time)]
    return cols


def _bh_fdr(p_values: List[float]) -> List[float]:
    """Benjamini-Hochberg FDR-adjusted q-values, preserving input order."""
    p = np.asarray(p_values, float)
    q = np.full(p.shape, np.nan, float)
    finite = np.isfinite(p)
    if not np.any(finite):
        return q.tolist()
    idx = np.where(finite)[0]
    order = idx[np.argsort(p[idx])]
    ranked = p[order]
    m = float(ranked.size)
    raw = ranked * m / np.arange(1, ranked.size + 1, dtype=float)
    adj = np.minimum.accumulate(raw[::-1])[::-1]
    q[order] = np.clip(adj, 0.0, 1.0)
    return q.tolist()


def _finite_curve_stats(values: np.ndarray) -> Tuple[float, float, float]:
    """Return mean, mean absolute value, and peak absolute value without NaN warnings."""
    vals = np.asarray(values, float)
    finite = vals[np.isfinite(vals)]
    if finite.size == 0:
        return float("nan"), float("nan"), float("nan")
    abs_vals = np.abs(finite)
    return float(np.mean(finite)), float(np.mean(abs_vals)), float(np.max(abs_vals))


@dataclass
class GLMResult:
    """Result container for a continuous GLM fit."""
    predictor_names: List[str]
    kernels: Dict[str, np.ndarray]        # predictor -> (n_kernel_samples,)
    kernel_tvec: np.ndarray               # time vector for kernel x-axis
    time: np.ndarray                      # time vector used for the fitted trace
    y_pred: np.ndarray                    # predicted trace
    y_actual: np.ndarray                  # actual trace
    residuals: np.ndarray
    r2: float
    coefficients: np.ndarray              # raw beta vector
    design_matrix: np.ndarray
    stats: Dict[str, Any] = field(default_factory=dict)
    feature_importance: List[Dict[str, Any]] = field(default_factory=list)
    kernel_ci_lower: Dict[str, np.ndarray] = field(default_factory=dict)
    kernel_ci_upper: Dict[str, np.ndarray] = field(default_factory=dict)


class ContinuousGLM:
    """Build a design matrix from event times and fit a linear model."""

    BASIS_TYPES = ("raised_cosine", "bspline", "fir")
    REGULARIZATION = ("ridge", "lasso", "ols")

    def __init__(self):
        self._result: Optional[GLMResult] = None

    @staticmethod
    def _predictor_vector(time: np.ndarray, predictor: Any) -> np.ndarray:
        """Convert an event or continuous predictor spec into a model-time vector."""
        t = np.asarray(time, float)
        T = int(t.size)
        vec = np.zeros(T, float)
        if T == 0:
            return vec

        if isinstance(predictor, dict):
            kind = str(predictor.get("kind", "events")).strip().lower()
            if kind in {"vector", "sampled"}:
                values = np.asarray(predictor.get("values", []), float)
                if values.size != T:
                    return vec
                vec = values.astype(float, copy=True)
                vec[~np.isfinite(vec)] = 0.0
                return vec
            if kind == "continuous":
                pt = np.asarray(predictor.get("time", []), float)
                values = np.asarray(predictor.get("values", []), float)
                if pt.size != values.size:
                    n = min(pt.size, values.size)
                    pt = pt[:n]
                    values = values[:n]
                m = np.isfinite(pt) & np.isfinite(values)
                pt = pt[m]
                values = values[m]
                if pt.size < 2 or values.size < 2:
                    return vec
                order = np.argsort(pt)
                pt = pt[order]
                values = values[order]
                keep = np.concatenate([[True], np.diff(pt) > 0])
                pt = pt[keep]
                values = values[keep]
                if pt.size < 2:
                    return vec
                interp = np.interp(t, pt, values, left=np.nan, right=np.nan)
                finite = np.isfinite(interp)
                if not np.any(finite):
                    return vec
                centered = interp.astype(float)
                mean = float(np.nanmean(centered[finite]))
                std = float(np.nanstd(centered[finite]))
                if np.isfinite(std) and std > 1e-12:
                    centered = (centered - mean) / std
                else:
                    centered = centered - mean
                centered[~np.isfinite(centered)] = 0.0
                return centered
            ev_times = predictor.get("events", predictor.get("times", []))
        else:
            ev_times = predictor

        ev_times = np.asarray(ev_times, float)
        ev_times = ev_times[np.isfinite(ev_times)]
        if ev_times.size == 0:
            return vec
        ev_idx = np.searchsorted(t, ev_times)
        ev_idx = ev_idx[(ev_idx >= 0) & (ev_idx < T)]
        for idx in ev_idx:
            vec[int(idx)] += 1.0
        return vec

    @staticmethod
    def build_design_matrix(
        time: np.ndarray,
        predictors: Dict[str, Any],
        kernel_window: Tuple[float, float],
        n_basis: int = 8,
        basis_type: str = "raised_cosine",
    ) -> Tuple[np.ndarray, List[str], int, List[str]]:
        """
        Build a (T x P) design matrix from event times.

        Parameters
        ----------
        time : (T,) array — continuous time vector
        predictors : dict mapping predictor name -> 1-D array of event times
        kernel_window : (pre, post) in seconds relative to event
        n_basis : number of temporal basis functions
        basis_type : 'raised_cosine', 'bspline', or 'fir'

        Returns
        -------
        X : (T, n_predictors * n_basis) design matrix
        col_names : column labels
        n_basis : basis count (for later kernel extraction)
        """
        time = np.asarray(time, float)
        if time.size < 3:
            raise ValueError("Need at least 3 time samples for GLM fitting.")
        _, pre_samp, _, kernel_tvec = _glm_kernel_geometry(time, kernel_window)
        kernel_len = int(kernel_tvec.size)
        B = _glm_basis_matrix(n_basis, kernel_len, basis_type)

        T = len(time)
        col_names: List[str] = []
        X_parts: List[np.ndarray] = []
        used_predictors: List[str] = []

        for pred_name, pred_spec in predictors.items():
            input_vec = ContinuousGLM._predictor_vector(time, pred_spec)
            if input_vec.size != T or not np.any(np.isfinite(input_vec)):
                continue
            input_vec = np.asarray(input_vec, float)
            input_vec[~np.isfinite(input_vec)] = 0.0
            if not np.any(np.abs(input_vec) > 1e-12):
                continue

            part = _glm_convolved_columns(input_vec, B, pre_samp, T)

            X_parts.append(part)
            used_predictors.append(pred_name)
            for b in range(n_basis):
                col_names.append(f"{pred_name}_b{b}")

        X = np.hstack(X_parts) if X_parts else np.zeros((T, 0))
        # Add intercept
        X = np.column_stack([np.ones(T), X])
        col_names.insert(0, "intercept")
        return X, col_names, n_basis, used_predictors

    @staticmethod
    def _scaled_design(X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Scale non-intercept columns to unit L2 norm for comparable regularization."""
        X = np.asarray(X, float)
        scales = np.ones(X.shape[1], float)
        for j in range(1, X.shape[1]):
            norm = float(np.linalg.norm(X[:, j]))
            if np.isfinite(norm) and norm > 1e-12:
                scales[j] = norm
        return X / scales, scales

    @staticmethod
    def fit_coefficients(
        X: np.ndarray,
        y: np.ndarray,
        regularization: str = "ridge",
        alpha: float = 1.0,
    ) -> np.ndarray:
        """Fit coefficients with scale-aware ridge/lasso/OLS and return betas on the original scale."""
        X = np.asarray(X, float)
        y = np.asarray(y, float)
        Xs, scales = ContinuousGLM._scaled_design(X)
        regularization = str(regularization or "ridge").lower()
        if regularization == "lasso":
            try:
                from sklearn.linear_model import Lasso as _Lasso
                model = _Lasso(alpha=float(alpha), max_iter=10000, fit_intercept=False)
                model.fit(Xs, y)
                beta_scaled = np.asarray(model.coef_, float)
            except ImportError:
                _LOG.warning("sklearn not available; falling back to ridge")
                beta_scaled = ContinuousGLM._ridge_fit(Xs, y, alpha)
        elif regularization == "ols":
            beta_scaled, *_ = np.linalg.lstsq(Xs, y, rcond=None)
        else:
            beta_scaled = ContinuousGLM._ridge_fit(Xs, y, alpha)
        return beta_scaled / scales

    @staticmethod
    def kernels_from_coefficients(
        beta: np.ndarray,
        predictor_names: List[str],
        n_basis: int,
        kernel_tvec: np.ndarray,
        basis_type: str,
    ) -> Dict[str, np.ndarray]:
        B = _glm_basis_matrix(int(n_basis), int(np.asarray(kernel_tvec).size), basis_type)
        kernels: Dict[str, np.ndarray] = {}
        idx = 1
        beta = np.asarray(beta, float)
        for pred_name in predictor_names:
            w = beta[idx:idx + int(n_basis)]
            if w.size == int(n_basis):
                kernels[pred_name] = B @ w
            idx += int(n_basis)
        return kernels

    def fit(
        self,
        time: np.ndarray,
        signal: np.ndarray,
        predictors: Dict[str, Any],
        kernel_window: Tuple[float, float] = (-1.0, 3.0),
        n_basis: int = 8,
        basis_type: str = "raised_cosine",
        regularization: str = "ridge",
        alpha: float = 1.0,
    ) -> GLMResult:
        """Fit the GLM and return the result."""
        time = np.asarray(time, float)
        signal = np.asarray(signal, float)

        X, col_names, n_b, used_predictors = self.build_design_matrix(
            time, predictors, kernel_window, n_basis, basis_type,
        )
        if not used_predictors:
            raise ValueError("No usable predictors after alignment to the fitted trace.")

        # Mask NaN samples
        valid = np.isfinite(signal) & np.all(np.isfinite(X), axis=1)
        Xv = X[valid]
        yv = signal[valid]

        # Fit. Non-intercept columns are L2-normalized internally so ridge/lasso
        # alpha has comparable meaning for event and continuous predictors.
        beta = self.fit_coefficients(Xv, yv, regularization=regularization, alpha=alpha)

        y_pred = X @ beta
        residuals = signal - y_pred
        ss_res = np.nansum((signal[valid] - y_pred[valid]) ** 2)
        ss_tot = np.nansum((signal[valid] - np.nanmean(signal[valid])) ** 2)
        r2 = 1.0 - ss_res / max(ss_tot, 1e-12)
        valid_fit = np.isfinite(signal) & np.isfinite(y_pred)
        res_fit = residuals[valid_fit]
        y_fit = signal[valid_fit]
        pred_fit = y_pred[valid_fit]
        mse = float(np.nanmean(res_fit ** 2)) if res_fit.size else float("nan")
        rmse = float(np.sqrt(mse)) if np.isfinite(mse) else float("nan")
        mae = float(np.nanmean(np.abs(res_fit))) if res_fit.size else float("nan")
        resid_std = float(np.nanstd(res_fit)) if res_fit.size else float("nan")
        corr = float("nan")
        if y_fit.size > 2 and np.nanstd(y_fit) > 1e-12 and np.nanstd(pred_fit) > 1e-12:
            corr = float(np.corrcoef(y_fit, pred_fit)[0, 1])
        stats = {
            "n_samples": float(np.sum(valid_fit)),
            "mse": mse,
            "rmse": rmse,
            "mae": mae,
            "residual_std": resid_std,
            "corr": corr,
        }

        # Extract kernels on the same inclusive dt-spaced axis used to build X.
        _, _, _, kernel_tvec = _glm_kernel_geometry(time, kernel_window)
        kernels = self.kernels_from_coefficients(beta, list(used_predictors), n_b, kernel_tvec, basis_type)

        self._result = GLMResult(
            predictor_names=list(used_predictors),
            kernels=kernels,
            kernel_tvec=kernel_tvec,
            time=time,
            y_pred=y_pred,
            y_actual=signal,
            residuals=residuals,
            r2=r2,
            coefficients=beta,
            design_matrix=X,
            stats=stats,
        )
        return self._result

    @staticmethod
    def _ridge_fit(X: np.ndarray, y: np.ndarray, alpha: float) -> np.ndarray:
        n = X.shape[1]
        I = np.eye(n)
        I[0, 0] = 0  # don't regularize intercept
        return np.linalg.solve(X.T @ X + alpha * I, X.T @ y)


# ============================================================================
# 2. Trial-level FLMM backend (via R fastFMM)
# ============================================================================

@dataclass
class FLMMResult:
    """Result container for a trial-level FLMM fit."""
    tvec: np.ndarray                      # peri-event time vector
    coefficients: Dict[str, np.ndarray]   # term -> (n_time,) coefficient curve
    ci_lower: Dict[str, np.ndarray]       # term -> lower 95 % CI
    ci_upper: Dict[str, np.ndarray]       # term -> upper 95 % CI
    joint_ci_lower: Dict[str, np.ndarray]
    joint_ci_upper: Dict[str, np.ndarray]
    residuals: Optional[np.ndarray] = None
    aic: Optional[float] = None
    summary_text: str = ""
    stats: Dict[str, float] = field(default_factory=dict)
    feature_importance: List[Dict[str, Any]] = field(default_factory=list)


class TrialFLMM:
    """
    Functional Linear Mixed Model using R's fastFMM package.

    Wraps fastFMM::fui() via rpy2.  The user provides a trial-level data
    matrix (n_trials x n_timepoints) plus a design dataframe (n_trials rows)
    with fixed/random predictors.  The backend passes the trace matrix as the
    functional response expected by fastFMM.
    """

    def __init__(self):
        self._result: Optional[FLMMResult] = None
        self._available: Optional[bool] = None

    @property
    def available(self) -> bool:
        if self._available is None:
            self._available = _check_rpy2()
        return self._available

    @staticmethod
    def _fixed_terms_from_formula(
        formula_fixed: str,
        design: Dict[str, np.ndarray],
        group_var: str,
    ) -> List[str]:
        """Extract scalar fixed-effect terms that are present in the design dict."""
        formula_text = str(formula_fixed or "Y.obs ~ 1")
        if "~" not in formula_text:
            return [k for k in design.keys() if k != group_var]
        rhs = formula_text.split("~", 1)[1].replace("\n", " ")
        terms: List[str] = []
        for raw in rhs.split("+"):
            term = raw.strip().strip("`")
            if not term or term in {"0", "1", "-1"}:
                continue
            if "|" in term or "(" in term or ")" in term:
                continue
            if term in design and term != group_var and term not in terms:
                terms.append(term)
        return terms

    def _fit_fixed_effect_functional(
        self,
        mat: np.ndarray,
        tvec: np.ndarray,
        design: Dict[str, np.ndarray],
        formula_fixed: str,
        group_var: str,
        reason: str,
    ) -> FLMMResult:
        """
        Functional fixed-effect fallback for single-subject / non-repeated-group data.

        This is intentionally not a mixed model. It fits one scalar-design linear
        model across trials at every timepoint, which is the useful single-animal
        test case while avoiding fake random-effect groups.
        """
        mat = np.asarray(mat, float)
        tvec = np.asarray(tvec, float)
        if mat.ndim != 2 or mat.shape[0] < 2:
            raise ValueError("Functional fixed-effect modeling needs at least two trial/row curves.")
        n_trials, n_time = mat.shape
        terms = self._fixed_terms_from_formula(formula_fixed, design, group_var)

        cols = [np.ones(n_trials, float)]
        term_names = ["(Intercept)"]
        dropped: List[str] = []
        for term in terms:
            vals = np.asarray(design.get(term, np.array([], float)), float).reshape(-1)
            if vals.size != n_trials:
                dropped.append(f"{term} (wrong row count)")
                continue
            finite = np.isfinite(vals)
            if np.sum(finite) < 2:
                dropped.append(f"{term} (too few finite values)")
                continue
            filled = vals.astype(float, copy=True)
            mean = float(np.nanmean(filled[finite]))
            filled[~finite] = mean
            if float(np.nanstd(filled)) <= 1e-12:
                dropped.append(f"{term} (no across-row variation)")
                continue
            cols.append(filled)
            term_names.append(term)

        X = np.column_stack(cols)
        rank = int(np.linalg.matrix_rank(X))
        if rank <= 0:
            raise ValueError("Functional fixed-effect model could not build a valid design matrix.")
        if rank < X.shape[1]:
            # Keep the fit numerically stable and make the limitation visible.
            keep = [0]
            X_keep = X[:, [0]]
            for j in range(1, X.shape[1]):
                cand = np.column_stack([X_keep, X[:, j]])
                if int(np.linalg.matrix_rank(cand)) > int(np.linalg.matrix_rank(X_keep)):
                    keep.append(j)
                    X_keep = cand
                else:
                    dropped.append(f"{term_names[j]} (collinear)")
            X = X[:, keep]
            term_names = [term_names[j] for j in keep]
            rank = int(np.linalg.matrix_rank(X))

        n_params = int(X.shape[1])
        beta = np.full((n_params, n_time), np.nan, float)
        y_pred = np.full_like(mat, np.nan, dtype=float)
        residuals = np.full_like(mat, np.nan, dtype=float)
        se_beta = np.full((n_params, n_time), np.nan, float)
        complete_design = np.all(np.isfinite(X), axis=1)
        rank = int(np.linalg.matrix_rank(X[complete_design])) if np.any(complete_design) else rank
        valid_timepoints = 0
        skipped_timepoints = 0
        for ti in range(n_time):
            y = mat[:, ti]
            ok = complete_design & np.isfinite(y)
            if int(np.sum(ok)) < max(rank, 2):
                skipped_timepoints += 1
                continue
            X_ok = X[ok]
            y_ok = y[ok]
            rank_ok = int(np.linalg.matrix_rank(X_ok))
            if rank_ok < rank:
                skipped_timepoints += 1
                continue
            try:
                b, *_ = np.linalg.lstsq(X_ok, y_ok, rcond=None)
            except np.linalg.LinAlgError:
                b = np.linalg.pinv(X_ok) @ y_ok
            beta[:, ti] = b
            pred = X_ok @ b
            y_pred[ok, ti] = pred
            residuals[ok, ti] = y_ok - pred
            df_t = int(np.sum(ok)) - rank_ok
            if df_t > 0:
                rss_t = float(np.nansum((y_ok - pred) ** 2))
                sigma2 = rss_t / max(df_t, 1)
                xtx_inv = np.linalg.pinv(X_ok.T @ X_ok)
                diag = np.maximum(np.diag(xtx_inv), 0.0)
                se_beta[:, ti] = np.sqrt(diag * max(sigma2, 0.0))
            valid_timepoints += 1
        n_obs = max(1, int(np.isfinite(residuals).sum()))
        rss = float(np.nansum(residuals ** 2))
        df = max(0, int(np.nanmax(np.sum(np.isfinite(residuals), axis=0)) - rank)) if valid_timepoints else 0
        aic = float(n_obs * np.log(max(rss / n_obs, 1e-12)) + 2 * rank * max(valid_timepoints, 1))

        coefficients = {name: np.asarray(beta[i, :], float) for i, name in enumerate(term_names)}
        ci_lower: Dict[str, np.ndarray] = {}
        ci_upper: Dict[str, np.ndarray] = {}
        if np.any(np.isfinite(se_beta)):
            for i, name in enumerate(term_names):
                ci_lower[name] = coefficients[name] - 1.96 * se_beta[i, :]
                ci_upper[name] = coefficients[name] + 1.96 * se_beta[i, :]

        coeff_abs_peaks: List[float] = []
        coeff_abs_means: List[float] = []
        summary_parts = [
            f"Functional fixed-effect model: {len(term_names)} terms, {n_trials} rows, {n_time} timepoints",
            f"Note: {reason}",
            "This fallback estimates coefficient curves across trials/rows without random effects.",
        ]
        if valid_timepoints == 0:
            summary_parts.append("No timepoints were estimable; check predictor variation and missing PSTH rows.")
        elif skipped_timepoints:
            summary_parts.append(
                f"Skipped {skipped_timepoints}/{n_time} timepoints with too many missing rows or rank-deficient design."
            )
        if not np.any(np.isfinite(se_beta)):
            summary_parts.append("CIs hidden: not enough residual degrees of freedom.")
        else:
            summary_parts.append("Pointwise 95% CIs use classical row-level residual variance.")
        summary_parts.append(f"AIC = {aic:.1f}")
        for name in term_names:
            coeff = coefficients[name]
            mean_coef, mean_abs, peak_abs = _finite_curve_stats(coeff)
            coeff_abs_peaks.append(peak_abs)
            coeff_abs_means.append(mean_abs)
            if np.isfinite(mean_abs) or np.isfinite(peak_abs):
                summary_parts.append(
                    f"  {name}: mean coef = {mean_coef:.4f}, "
                    f"mean abs = {mean_abs:.4f}, peak abs = {peak_abs:.4f}"
                )
            else:
                summary_parts.append(f"  {name}: not estimable (no finite coefficient bins)")
        if dropped:
            summary_parts.append("Dropped terms: " + ", ".join(dropped))
        summary_text = "\n".join(summary_parts)
        stats = {
            "n_trials": float(n_trials),
            "n_timepoints": float(n_time),
            "n_terms": float(len(term_names)),
            "aic": aic,
            "mean_abs_coefficient": _finite_curve_stats(np.asarray(coeff_abs_means, float))[0],
            "peak_abs_coefficient": _finite_curve_stats(np.asarray(coeff_abs_peaks, float))[2],
            "formula": str(formula_fixed or "Y.obs ~ 1"),
            "group_var": "none (fixed-effect fallback)",
            "requested_group_var": group_var,
            "fallback_grouping": "",
            "fit_notes": [reason],
            "backend": "python_functional_fixed_effect",
            "variance_unavailable": bool(not np.any(np.isfinite(se_beta))),
            "dropped_terms": dropped,
            "valid_timepoints": float(valid_timepoints),
            "skipped_timepoints": float(skipped_timepoints),
        }
        self._result = FLMMResult(
            tvec=tvec,
            coefficients=coefficients,
            ci_lower=ci_lower,
            ci_upper=ci_upper,
            joint_ci_lower={},
            joint_ci_upper={},
            residuals=residuals,
            aic=aic,
            summary_text=summary_text,
            stats=stats,
        )
        return self._result

    def fit(
        self,
        mat: np.ndarray,
        tvec: np.ndarray,
        design: Dict[str, np.ndarray],
        formula_fixed: str = "Y.obs ~ 1",
        random_effects: str = "~1",
        group_var: str = "subject",
        parallel: bool = False,
        nknots_min: Optional[int] = None,
        num_boots: int = 0,
    ) -> FLMMResult:
        """
        Fit a functional LMM via fastFMM::fui().

        Parameters
        ----------
        mat : (n_trials, n_timepoints) — the Y matrix (z-scored PSTH rows)
        tvec : (n_timepoints,) — peri-event time
        design : dict of predictor_name -> (n_trials,) arrays
            Must include the grouping variable.
        formula_fixed : R formula string for fixed effects
        random_effects : R formula string for random effects
        group_var : column name for the grouping/random-effects variable
        parallel : whether to use parallelisation in fui()
        nknots_min : minimum number of knots for penalised splines
        num_boots : number of bootstrap iterations (0 = analytic only)

        Returns
        -------
        FLMMResult
        """
        n_trials, n_time = mat.shape

        if group_var not in design:
            raise ValueError(f"FLMM design is missing group variable '{group_var}'.")

        design = dict(design)
        r_df_vars = {}
        group_var_model = group_var
        fallback_grouping = ""
        group_vals = np.asarray(design[group_var]).astype(str)
        unique_groups = np.unique(group_vals)
        has_repeated_groups = 1 < unique_groups.size < n_trials
        if not has_repeated_groups:
            if unique_groups.size <= 1:
                reason = (
                    "Single-animal functional regression: one subject in the design, "
                    "so a random-effects FLMM is not applicable. Coefficients are fit "
                    "with pointwise CIs from the within-recording fit."
                )
            else:
                reason = (
                    "Single-trial-per-subject functional regression: each subject "
                    "contributes one row, so within-subject random effects are not "
                    "identifiable. Coefficients are fit with pointwise CIs."
                )
            return self._fit_fixed_effect_functional(
                mat,
                tvec,
                design,
                formula_fixed=formula_fixed,
                group_var=group_var,
                reason=reason,
            )

        _init_r()
        import rpy2.robjects as ro
        from rpy2.robjects import r as R
        from rpy2.robjects.packages import importr

        for col_name, col_vals in design.items():
            col_vals = np.asarray(col_vals)
            if col_vals.size != n_trials:
                raise ValueError(f"FLMM design column '{col_name}' has {col_vals.size} values for {n_trials} rows.")
            if np.issubdtype(col_vals.dtype, np.floating):
                r_df_vars[col_name] = ro.FloatVector(col_vals.astype(float))
            elif np.issubdtype(col_vals.dtype, np.integer):
                r_df_vars[col_name] = ro.IntVector(col_vals.astype(int))
            else:
                r_df_vars[col_name] = ro.FactorVector(ro.StrVector(col_vals.astype(str)))

        r_df = ro.DataFrame(r_df_vars)
        y_matrix = R.matrix(ro.FloatVector(np.asarray(mat, float).ravel(order="F")), nrow=n_trials, ncol=n_time)
        ro.globalenv[".__pyber_flmm_df"] = r_df
        ro.globalenv[".__pyber_flmm_y"] = y_matrix
        r_df = R(".__pyber_flmm_df$Y.obs <- I(.__pyber_flmm_y); .__pyber_flmm_df")

        formula_text = str(formula_fixed or "Y.obs ~ 1").strip() or "Y.obs ~ 1"
        if group_var_model != group_var:
            formula_text = re.sub(
                rf"\|\s*`?{re.escape(group_var)}`?\s*\)",
                f"| {group_var_model})",
                formula_text,
            )
        if "|" not in formula_text:
            rand = str(random_effects or "~1").strip()
            rand_rhs = rand.split("~", 1)[1].strip() if "~" in rand else rand
            if rand_rhs.lower() in {"", "0", "none", "fixed"}:
                rand_rhs = ""
            if rand_rhs:
                formula_text = f"{formula_text} + ({rand_rhs} | {group_var_model})"

        # Call fui()
        base = importr("base")
        fastFMM = importr("fastFMM")
        kwargs = {
            "formula": ro.Formula(formula_text),
            "data": r_df,
            "var": ro.BoolVector([True]),
            "parallel": ro.BoolVector([parallel]),
            "silent": ro.BoolVector([True]),
            "subj_id": ro.StrVector([group_var_model]),
            "override_zero_var": ro.BoolVector([True]),
        }
        if nknots_min is not None:
            kwargs["nknots_min"] = ro.IntVector([nknots_min])
        if n_time < 35:
            kwargs["nknots_min_cov"] = ro.IntVector([max(4, min(35, max(4, n_time // 2)))])
        if num_boots > 0:
            kwargs["analytic"] = ro.BoolVector([False])
            kwargs["n_boots"] = ro.IntVector([num_boots])
            kwargs["argvals"] = ro.FloatVector(np.asarray(tvec, float))
        else:
            kwargs["analytic"] = ro.BoolVector([True])
            kwargs["n_boots"] = ro.IntVector([0])

        _LOG.info("Calling fastFMM::fui() with formula=%s, %d trials, %d timepoints",
                  formula_text, n_trials, n_time)

        fit_notes: List[str] = []
        variance_unavailable = False
        old_warn = R("getOption('warn')")
        base.options(warn=ro.IntVector([-1]))
        try:
            try:
                fui_result = fastFMM.fui(**kwargs)
            except Exception as exc:
                msg = str(exc)
                recoverable = any(
                    token in msg.lower()
                    for token in ("downdated vtv", "not positive definite", "singular", "rank deficient")
                )
                if not recoverable:
                    raise
                retry_kwargs = dict(kwargs)
                retry_kwargs["var"] = ro.BoolVector([False])
                retry_kwargs["analytic"] = ro.BoolVector([True])
                retry_kwargs["n_boots"] = ro.IntVector([0])
                variance_unavailable = True
                fit_notes.append(
                    "fastFMM variance inference failed because the mixed-model design was near-singular; "
                    "refit with variance/CIs disabled."
                )
                _LOG.warning("Retrying fastFMM without variance inference after: %s", msg)
                fui_result = fastFMM.fui(**retry_kwargs)
        finally:
            try:
                base.options(warn=old_warn)
            except Exception:
                base.options(warn=ro.IntVector([0]))

        # Parse the result.
        # fui() returns a list with elements:
        #   $betaHat  — matrix (n_terms x n_time) of coefficient estimates
        #   $betaHat.LB / $betaHat.UB — pointwise 95% CI
        #   $betaHat.LB.joint / $betaHat.UB.joint — joint 95% CI
        #   $AIC
        coefficients: Dict[str, np.ndarray] = {}
        ci_lower: Dict[str, np.ndarray] = {}
        ci_upper: Dict[str, np.ndarray] = {}
        joint_ci_lower: Dict[str, np.ndarray] = {}
        joint_ci_upper: Dict[str, np.ndarray] = {}

        try:
            try:
                result_names = set(str(name) for name in R('names')(fui_result))
            except Exception:
                result_names = set()
            beta_hat = np.atleast_2d(np.array(R('as.matrix')(fui_result.rx2("betaHat")), dtype=float))

            # Term names from rownames
            try:
                term_names = list(R('rownames')(fui_result.rx2("betaHat")))
            except Exception:
                term_names = [f"term_{i}" for i in range(beta_hat.shape[0])]

            se_mat = None
            try:
                if result_names and "se_mat" not in result_names:
                    raise KeyError("se_mat")
                se_mat = np.atleast_2d(np.array(R('as.matrix')(fui_result.rx2("se_mat")), dtype=float))
                if se_mat.shape != beta_hat.shape:
                    se_mat = None
            except Exception:
                se_mat = None

            beta_lb = None
            beta_ub = None
            if se_mat is not None and not variance_unavailable:
                beta_lb = beta_hat - 1.96 * se_mat
                beta_ub = beta_hat + 1.96 * se_mat
            elif not variance_unavailable:
                try:
                    if result_names and ("betaHat.LB" not in result_names or "betaHat.UB" not in result_names):
                        raise KeyError("betaHat CI")
                    beta_lb = np.atleast_2d(np.array(R('as.matrix')(fui_result.rx2("betaHat.LB")), dtype=float))
                    beta_ub = np.atleast_2d(np.array(R('as.matrix')(fui_result.rx2("betaHat.UB")), dtype=float))
                except Exception:
                    beta_lb = None
                    beta_ub = None

            try:
                qn = np.asarray(fui_result.rx2("qn"), float).ravel()
            except Exception:
                qn = np.array([], float)

            for i, name in enumerate(term_names):
                coefficients[name] = beta_hat[i, :]
                if beta_lb is not None and beta_ub is not None and beta_lb.shape == beta_hat.shape and beta_ub.shape == beta_hat.shape:
                    ci_lower[name] = beta_lb[i, :]
                    ci_upper[name] = beta_ub[i, :]

            # Joint CIs (may not always be present)
            if not variance_unavailable:
                try:
                    if result_names and ("betaHat.LB.joint" not in result_names or "betaHat.UB.joint" not in result_names):
                        raise KeyError("joint CI")
                    jlb = np.array(R('as.matrix')(fui_result.rx2("betaHat.LB.joint")))
                    jub = np.array(R('as.matrix')(fui_result.rx2("betaHat.UB.joint")))
                    for i, name in enumerate(term_names):
                        joint_ci_lower[name] = jlb[i, :]
                        joint_ci_upper[name] = jub[i, :]
                except Exception:
                    if se_mat is not None and qn.size:
                        for i, name in enumerate(term_names):
                            qcrit = float(qn[min(i, qn.size - 1)])
                            joint_ci_lower[name] = beta_hat[i, :] - qcrit * se_mat[i, :]
                            joint_ci_upper[name] = beta_hat[i, :] + qcrit * se_mat[i, :]
                    elif ci_lower and ci_upper:
                        joint_ci_lower = {k: v.copy() for k, v in ci_lower.items()}
                        joint_ci_upper = {k: v.copy() for k, v in ci_upper.items()}

            aic_val = None
            for key in ("aic", "AIC"):
                try:
                    aic_arr = np.array(R('as.matrix')(fui_result.rx2(key)), dtype=float)
                    if aic_arr.size:
                        if aic_arr.ndim == 2 and aic_arr.shape[1] >= 1:
                            aic_val = float(np.nanmean(aic_arr[:, 0]))
                        else:
                            aic_val = float(np.nanmean(aic_arr))
                        break
                except Exception:
                    continue

            summary_parts = [f"FLMM fit: {len(term_names)} terms, {n_trials} trials, {n_time} timepoints"]
            if fallback_grouping:
                summary_parts.append(f"Note: {fallback_grouping}")
            for note in fit_notes:
                summary_parts.append(f"Note: {note}")
            if variance_unavailable:
                summary_parts.append("CIs hidden: point estimates only because variance inference was unavailable.")
            if aic_val is not None:
                summary_parts.append(f"AIC = {aic_val:.1f}")
            coeff_abs_peaks: List[float] = []
            coeff_abs_means: List[float] = []
            for name in term_names:
                coeff = np.asarray(coefficients[name], float)
                mean_coef, mean_abs, peak_abs = _finite_curve_stats(coeff)
                coeff_abs_peaks.append(peak_abs)
                coeff_abs_means.append(mean_abs)
                if np.isfinite(mean_abs) or np.isfinite(peak_abs):
                    summary_parts.append(
                        f"  {name}: mean coef = {mean_coef:.4f}, "
                        f"mean abs = {mean_abs:.4f}, peak abs = {peak_abs:.4f}"
                    )
                else:
                    summary_parts.append(f"  {name}: not estimable (no finite coefficient bins)")
            summary_text = "\n".join(summary_parts)
            stats = {
                "n_trials": float(n_trials),
                "n_timepoints": float(n_time),
                "n_terms": float(len(term_names)),
                "aic": float(aic_val) if aic_val is not None else float("nan"),
                "mean_abs_coefficient": _finite_curve_stats(np.asarray(coeff_abs_means, float))[0],
                "peak_abs_coefficient": _finite_curve_stats(np.asarray(coeff_abs_peaks, float))[2],
                "formula": formula_text,
                "group_var": group_var_model,
                "requested_group_var": group_var,
                "fallback_grouping": fallback_grouping,
                "fit_notes": list(fit_notes),
                "variance_unavailable": bool(variance_unavailable),
            }

        except Exception as exc:
            _LOG.error("Failed to parse fui() result: %s", exc)
            raise RuntimeError(f"fastFMM::fui() result parsing failed: {exc}") from exc

        self._result = FLMMResult(
            tvec=tvec,
            coefficients=coefficients,
            ci_lower=ci_lower,
            ci_upper=ci_upper,
            joint_ci_lower=joint_ci_lower,
            joint_ci_upper=joint_ci_upper,
            aic=aic_val,
            summary_text=summary_text,
            stats=stats,
        )
        return self._result


# ============================================================================
# 3. PySide6 Widget
# ============================================================================

_TEMPORAL_QSS = """
TemporalModelingWidget {
    background: #111821;
    color: #d7e0ee;
}
QFrame#temporalHeader {
    background: #101b2b;
    border: 1px solid #263a52;
    border-radius: 8px;
}
QFrame#temporalNav {
    background: #0f1a28;
    border: 1px solid #263a52;
    border-radius: 8px;
}
QFrame#temporalControls {
    background: #111d2c;
    border: 1px solid #263a52;
    border-radius: 8px;
}
QFrame#temporalWorkspace {
    background: #111821;
    border: 1px solid #263a52;
    border-radius: 8px;
}
QFrame#temporalScopeBar {
    background: #101b2b;
    border: 1px solid #263a52;
    border-radius: 8px;
}
QLabel {
    color: #d7e0ee;
}
QLabel[class="muted"] {
    color: #9bacc3;
}
QLabel[class="title"] {
    color: #eef4ff;
    font-size: 15pt;
    font-weight: 800;
}
QGroupBox {
    color: #d7e0ee;
    font-weight: 700;
    border: 1px solid #29405c;
    border-radius: 7px;
    margin-top: 10px;
    padding: 14px 10px 10px 10px;
    background: #142033;
}
QGroupBox::title {
    subcontrol-origin: margin;
    subcontrol-position: top left;
    left: 10px;
    padding: 0 6px;
    color: #dce8f8;
}
QComboBox, QSpinBox, QDoubleSpinBox, QLineEdit {
    color: #e9f0fb;
    background: #0f1724;
    border: 1px solid #314963;
    border-radius: 6px;
    padding: 5px 8px;
    min-height: 24px;
}
QComboBox::drop-down {
    border: 0;
    width: 24px;
}
QProgressBar {
    color: #e9f0fb;
    background: #0f1724;
    border: 1px solid #314963;
    border-radius: 6px;
    min-height: 18px;
    text-align: center;
}
QProgressBar::chunk {
    background: #2d8cff;
    border-radius: 5px;
}
QListWidget, QTextEdit {
    color: #e6edf8;
    background: #0d1420;
    border: 1px solid #314963;
    border-radius: 6px;
    selection-background-color: #2d78c4;
}
QPushButton {
    color: #eef4ff;
    background: #17263a;
    border: 1px solid #34506c;
    border-radius: 7px;
    padding: 6px 12px;
    font-weight: 700;
}
QPushButton:hover {
    background: #203450;
}
QPushButton[class="primary"] {
    background: #2d8cff;
    border: 1px solid #4ba0ff;
}
QToolButton {
    color: #cdd8e8;
    background: #101b2b;
    border: 1px solid #29405c;
    border-radius: 8px;
    padding: 9px 8px;
    font-weight: 700;
}
QToolButton:checked {
    color: #ffffff;
    background: #1f6db1;
    border: 1px solid #35a4e8;
}
QTabWidget::pane {
    border: 1px solid #263a52;
    border-radius: 6px;
    top: -1px;
}
QTabBar::tab {
    color: #b9c8dc;
    background: #152237;
    border: 1px solid #263a52;
    padding: 7px 14px;
    border-top-left-radius: 6px;
    border-top-right-radius: 6px;
}
QTabBar::tab:selected {
    color: #ffffff;
    background: #1f6db1;
    border-color: #35a4e8;
}
"""
_SECTION_QSS = _TEMPORAL_QSS

_TEMPORAL_QSS_LIGHT = _TEMPORAL_QSS + """
TemporalModelingWidget {
    background: #f4f6fb;
    color: #1f2a37;
}
QFrame#temporalHeader {
    background: #eef4ff;
    border: 1px solid #bfd7ff;
}
QFrame#temporalNav,
QFrame#temporalControls,
QFrame#temporalWorkspace,
QFrame#temporalScopeBar {
    background: #ffffff;
    border: 1px solid #d6dde9;
}
QLabel {
    color: #1f2a37;
}
QLabel[class="muted"] {
    color: #4c5a6f;
}
QLabel[class="title"] {
    color: #172033;
}
QGroupBox {
    color: #1f2a37;
    border: 1px solid #d6dde9;
    background: #ffffff;
}
QGroupBox::title {
    background: #f4f6fb;
    color: #334155;
}
QComboBox, QSpinBox, QDoubleSpinBox, QLineEdit {
    color: #1f2a37;
    background: #ffffff;
    border: 1px solid #c2ccda;
    selection-background-color: #2563eb;
    selection-color: #ffffff;
}
QComboBox:focus, QSpinBox:focus, QDoubleSpinBox:focus, QLineEdit:focus {
    border: 1px solid #2563eb;
}
QComboBox QAbstractItemView {
    background: #ffffff;
    color: #1f2a37;
    border: 1px solid #c2ccda;
    selection-background-color: #2563eb;
    selection-color: #ffffff;
}
QProgressBar {
    color: #1f2a37;
    background: #ffffff;
    border: 1px solid #c2ccda;
}
QProgressBar::chunk {
    background: #2563eb;
}
QListWidget, QTextEdit {
    color: #1f2a37;
    background: #ffffff;
    border: 1px solid #c2ccda;
    selection-background-color: #2563eb;
    selection-color: #ffffff;
}
QListWidget::item:hover {
    background: #dbeafe;
    color: #172033;
}
QListWidget::item:selected {
    background: #2563eb;
    color: #ffffff;
}
QPushButton {
    color: #1f2a37;
    background: #ffffff;
    border: 1px solid #c2ccda;
}
QPushButton:hover {
    background: #eff6ff;
    border: 1px solid #93c5fd;
}
QPushButton[class="primary"] {
    color: #ffffff;
    background: #2563eb;
    border: 1px solid #1d4ed8;
}
QPushButton[class="primary"]:hover {
    background: #1d4ed8;
}
QToolButton {
    color: #334155;
    background: #ffffff;
    border: 1px solid #c2ccda;
}
QToolButton:hover {
    background: #dbeafe;
    border: 1px solid #93c5fd;
}
QToolButton:checked {
    color: #ffffff;
    background: #2563eb;
    border: 1px solid #1d4ed8;
}
QTabWidget::pane {
    border: 1px solid #d6dde9;
    background: #ffffff;
}
QTabBar::tab {
    color: #4a5568;
    background: #eef2f8;
    border: 1px solid #d6dde9;
}
QTabBar::tab:hover:!selected {
    color: #172033;
    background: #dbeafe;
    border-color: #93c5fd;
}
QTabBar::tab:selected {
    color: #ffffff;
    background: #2563eb;
    border-color: #1d4ed8;
}
QCheckBox::indicator {
    border: 1px solid #c2ccda;
    background: #ffffff;
}
QCheckBox::indicator:checked {
    background: #2563eb;
    border: 1px solid #1d4ed8;
}
"""


class TemporalModelingWidget(QtWidgets.QWidget):
    """
    PySide6 panel for Temporal Modeling (GLM / FLMM).
    Embeddable in the PostProcessingPanel dock system.
    """

    statusMessage = QtCore.Signal(str, int)

    def __init__(self, parent: Optional[QtWidgets.QWidget] = None):
        super().__init__(parent)
        self._settings = QtCore.QSettings("BelloneLab", "pyBer")
        self._app_theme_mode: str = "dark"
        self._loading_settings = True
        self._saved_predictor_keys: List[str] = []
        self._predictor_keys_by_file: Dict[str, List[str]] = {}
        self._restoring_predictors: bool = False
        self._kernel_visible: Dict[str, bool] = {}
        self._kernel_filter_guard = False
        self._illustration_vb: Optional[pg.ViewBox] = None
        self._kernel_grid_plots: List[pg.PlotWidget] = []
        # FLMM coefficient-filter state (mirrors the kernel-filter UX).
        self._flmm_coeff_visible: Dict[str, bool] = {}
        self._flmm_coeff_filter_guard = False
        self._flmm_coeff_grid_plots: List[pg.PlotWidget] = []
        self._glm = ContinuousGLM()
        self._flmm = TrialFLMM()
        self._glm_cache_version = 2
        self._glm_result: Optional[GLMResult] = None
        self._flmm_result: Optional[FLMMResult] = None

        # Per-file fits (filled by batch / group runs).
        self._glm_results_by_file: Dict[str, GLMResult] = {}
        self._flmm_results_by_file: Dict[str, FLMMResult] = {}
        self._fit_summary_by_file: Dict[str, str] = {}
        self._group_glm_summary: Dict[str, Any] = {}
        self._flmm_trace_mat: Optional[np.ndarray] = None
        self._flmm_trace_tvec: Optional[np.ndarray] = None
        self._flmm_trace_labels: List[str] = []
        self._flmm_trace_design: Dict[str, np.ndarray] = {}
        self._flmm_trace_term_labels: Dict[str, str] = {}
        self._flmm_trace_scope: str = ""
        # Batch state
        self._batch_cancel_requested: bool = False
        self._fit_mode: str = "all"  # one of "active" | "all" | "batch"
        self._active_file_id: str = ""

        # Data references (set by host panel)
        self._processed_trials = []
        self._psth_mat: Optional[np.ndarray] = None
        self._psth_tvec: Optional[np.ndarray] = None
        self._event_times: Optional[np.ndarray] = None
        self._file_ids: List[str] = []
        self._per_file_mats: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
        self._behavior_sources: Dict[str, Dict[str, Any]] = {}
        self._event_rows: List[Dict[str, object]] = []
        self._predictor_catalog: Dict[str, Dict[str, Any]] = {}
        self._group_mat: Optional[np.ndarray] = None
        self._group_tvec: Optional[np.ndarray] = None
        self._group_labels: List[str] = []
        self._flmm_row_meta: List[Dict[str, Any]] = []
        self._visual_mode: int = 0
        self._group_mode: bool = False

        self.setMinimumSize(980, 640)
        self.resize(1320, 880)  # preferred default whenever shown as a standalone window
        self._first_show_done = False
        self._build_compact_ui()
        self._load_temporal_settings()
        self._connect_signals()
        self._on_model_type_changed(self.combo_model_type.currentIndex())

    def sizeHint(self) -> QtCore.QSize:  # noqa: N802 - Qt API name
        return QtCore.QSize(1320, 880)

    def minimumSizeHint(self) -> QtCore.QSize:  # noqa: N802 - Qt API name
        return QtCore.QSize(980, 640)

    def showEvent(self, event: QtGui.QShowEvent) -> None:  # noqa: N802
        """Force a sensible default size the first time the widget appears as a
        top-level window. Subsequent shows (e.g. re-docking) keep whatever
        size the user has set."""
        super().showEvent(event)
        if self._first_show_done:
            return
        self._first_show_done = True
        # Resize only when we're standing alone (a floating dock-content window)
        # so embedded dock-area placement isn't disturbed.
        win = self.window()
        if win is self:
            self.resize(1320, 880)
        else:
            # If our top-level window is brand-new and smaller than our
            # preferred size, grow it so all controls + plots are visible.
            try:
                cur = win.size()
                target_w = max(cur.width(), 1320)
                target_h = max(cur.height(), 880)
                if target_w != cur.width() or target_h != cur.height():
                    win.resize(target_w, target_h)
            except Exception:
                pass

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self):
        root = QtWidgets.QVBoxLayout(self)
        root.setContentsMargins(6, 6, 6, 6)
        root.setSpacing(8)

        # ---- Model selector ----
        grp_model = QtWidgets.QGroupBox("Model type")
        grp_model.setStyleSheet(_SECTION_QSS)
        ml = QtWidgets.QVBoxLayout(grp_model)
        ml.setSpacing(4)

        row_type = QtWidgets.QHBoxLayout()
        self.combo_model_type = QtWidgets.QComboBox()
        self.combo_model_type.addItems(["Continuous GLM", "Trial-level FLMM (fastFMM)"])
        self.combo_model_type.setToolTip(
            "Continuous GLM models how predictor time courses shape one continuous signal.\n"
            "Trial-level FLMM models how per-trial scalar covariates modulate aligned response curves "
            "and needs repeated trials/subjects."
        )
        row_type.addWidget(QtWidgets.QLabel("Type:"))
        row_type.addWidget(self.combo_model_type, 1)
        ml.addLayout(row_type)

        self.lbl_flmm_status = QtWidgets.QLabel("")
        self.lbl_flmm_status.setProperty("class", "hint")
        self.lbl_flmm_status.setWordWrap(True)
        ml.addWidget(self.lbl_flmm_status)
        root.addWidget(grp_model)

        # ---- GLM settings ----
        self.grp_glm = QtWidgets.QGroupBox("GLM settings")
        self.grp_glm.setStyleSheet(_SECTION_QSS)
        gl = QtWidgets.QFormLayout(self.grp_glm)
        gl.setSpacing(4)

        self.combo_basis = QtWidgets.QComboBox()
        self.combo_basis.addItems(["Raised cosine", "B-spline", "FIR"])
        gl.addRow("Basis:", self.combo_basis)

        self.spin_n_basis = QtWidgets.QSpinBox()
        self.spin_n_basis.setRange(2, 50)
        self.spin_n_basis.setValue(8)
        gl.addRow("# basis:", self.spin_n_basis)

        self.combo_reg = QtWidgets.QComboBox()
        self.combo_reg.addItems(["Ridge", "Lasso", "OLS"])
        gl.addRow("Regularization:", self.combo_reg)

        self.spin_alpha = QtWidgets.QDoubleSpinBox()
        self.spin_alpha.setRange(0.001, 1000.0)
        self.spin_alpha.setValue(1.0)
        self.spin_alpha.setDecimals(3)
        gl.addRow("Alpha (λ):", self.spin_alpha)

        self.spin_kernel_pre = QtWidgets.QDoubleSpinBox()
        self.spin_kernel_pre.setRange(-30.0, 0.0)
        self.spin_kernel_pre.setValue(-1.0)
        self.spin_kernel_pre.setDecimals(1)
        self.spin_kernel_pre.setSuffix(" s")
        gl.addRow("Kernel pre:", self.spin_kernel_pre)

        self.spin_kernel_post = QtWidgets.QDoubleSpinBox()
        self.spin_kernel_post.setRange(0.1, 60.0)
        self.spin_kernel_post.setValue(3.0)
        self.spin_kernel_post.setDecimals(1)
        self.spin_kernel_post.setSuffix(" s")
        gl.addRow("Kernel post:", self.spin_kernel_post)

        self.spin_glm_bootstrap = QtWidgets.QSpinBox()
        self.spin_glm_bootstrap.setRange(0, 2000)
        self.spin_glm_bootstrap.setValue(100)
        self.spin_glm_bootstrap.setSpecialValueText("off")
        self.spin_glm_bootstrap.setToolTip("Circular-shift bootstraps for leave-one-out contribution p-values.")
        gl.addRow("Shift bootstraps:", self.spin_glm_bootstrap)

        self.spin_glm_jobs = QtWidgets.QSpinBox()
        max_jobs = max(1, os.cpu_count() or 1)
        self.spin_glm_jobs.setRange(1, max_jobs)
        self.spin_glm_jobs.setValue(min(4, max_jobs))
        self.spin_glm_jobs.setToolTip("Parallel jobs used for circular-shift bootstrap fits.")
        gl.addRow("Bootstrap jobs:", self.spin_glm_jobs)

        root.addWidget(self.grp_glm)

        # ---- FLMM settings ----
        self.grp_flmm = QtWidgets.QGroupBox("FLMM settings")
        self.grp_flmm.setStyleSheet(_SECTION_QSS)
        fl = QtWidgets.QFormLayout(self.grp_flmm)
        fl.setSpacing(4)

        self.edit_formula = QtWidgets.QLineEdit("Y.obs ~ 1")
        self.edit_formula.setPlaceholderText("Leave as Y.obs ~ 1 to auto-use selected predictors")
        fl.addRow("Fixed formula:", self.edit_formula)

        self.edit_random = QtWidgets.QLineEdit("~1")
        self.edit_random.setPlaceholderText("e.g. ~1 or ~time")
        fl.addRow("Random:", self.edit_random)

        self.edit_group_var = QtWidgets.QLineEdit("subject")
        fl.addRow("Group var:", self.edit_group_var)

        self.spin_nknots = QtWidgets.QSpinBox()
        self.spin_nknots.setRange(0, 100)
        self.spin_nknots.setValue(0)
        self.spin_nknots.setSpecialValueText("auto")
        fl.addRow("Min knots:", self.spin_nknots)

        self.spin_boots = QtWidgets.QSpinBox()
        self.spin_boots.setRange(0, 5000)
        self.spin_boots.setValue(0)
        self.spin_boots.setSpecialValueText("analytic")
        fl.addRow("Bootstrap iter:", self.spin_boots)
        self.combo_flmm_importance = QtWidgets.QComboBox()
        self.combo_flmm_importance.addItem("Fast coefficient ranking", "fast")
        self.combo_flmm_importance.addItem("Leave-one-out AIC (slow)", "loo")
        self.combo_flmm_importance.addItem("Off", "off")
        self.combo_flmm_importance.setToolTip(
            "Leave-one-out refits fastFMM once per predictor and can be very slow."
        )
        fl.addRow("Contribution:", self.combo_flmm_importance)

        root.addWidget(self.grp_flmm)

        # ---- Predictor builder ----
        self.grp_predictors = QtWidgets.QGroupBox("Predictors")
        self.grp_predictors.setStyleSheet(_SECTION_QSS)
        pl = QtWidgets.QVBoxLayout(self.grp_predictors)
        pl.setSpacing(4)

        self.combo_available_predictors = QtWidgets.QComboBox()
        self.combo_available_predictors.setToolTip("Choose from loaded event, behavior, and numeric behavior columns.")
        pl.addWidget(self.combo_available_predictors)

        self.list_predictors = QtWidgets.QListWidget()
        self.list_predictors.setMaximumHeight(100)
        pl.addWidget(self.list_predictors)

        pred_btn_row = QtWidgets.QHBoxLayout()
        self.btn_add_predictor = QtWidgets.QPushButton("+ Add")
        self.btn_add_predictor.setProperty("class", "compactSmall")
        self.btn_remove_predictor = QtWidgets.QPushButton("- Remove")
        self.btn_remove_predictor.setProperty("class", "compactSmall")
        pred_btn_row.addWidget(self.btn_add_predictor)
        pred_btn_row.addWidget(self.btn_remove_predictor)
        pred_btn_row.addStretch(1)
        pl.addLayout(pred_btn_row)

        self.lbl_predictor_hint = QtWidgets.QLabel(
            "Choose predictors from loaded DIO events, behavior states, behavior onsets, or numeric behavior columns."
        )
        self.lbl_predictor_hint.setProperty("class", "hint")
        self.lbl_predictor_hint.setWordWrap(True)
        pl.addWidget(self.lbl_predictor_hint)
        root.addWidget(self.grp_predictors)

        # ---- Fit button ----
        btn_row = QtWidgets.QHBoxLayout()
        self.btn_fit = QtWidgets.QPushButton("Fit model")
        self.btn_fit.setProperty("class", "compactPrimarySmall")
        btn_row.addWidget(self.btn_fit)
        btn_row.addStretch(1)
        root.addLayout(btn_row)

        # ---- Results / summary ----
        self.grp_results = QtWidgets.QGroupBox("Model summary")
        self.grp_results.setStyleSheet(_SECTION_QSS)
        rl = QtWidgets.QVBoxLayout(self.grp_results)
        rl.setSpacing(2)
        self.txt_summary = QtWidgets.QTextEdit()
        self.txt_summary.setReadOnly(True)
        self.txt_summary.setMaximumHeight(140)
        self.txt_summary.setStyleSheet("background: #1b2029; border: 1px solid #3a4050; border-radius: 4px;")
        rl.addWidget(self.txt_summary)
        root.addWidget(self.grp_results)

        # ---- Plot area ----
        self.grp_plots = QtWidgets.QGroupBox("Plots")
        self.grp_plots.setStyleSheet(_SECTION_QSS)
        plot_lay = QtWidgets.QVBoxLayout(self.grp_plots)
        plot_lay.setSpacing(4)

        self.plot_kernel = pg.PlotWidget(title="Estimated kernels")
        self.plot_kernel.setMinimumHeight(160)
        self.plot_kernel.showGrid(x=True, y=True, alpha=0.25)
        self.plot_kernel.addLegend(offset=(10, 10))
        plot_lay.addWidget(self.plot_kernel)

        self.plot_coeff = pg.PlotWidget(title="FLMM coefficient curves")
        self.plot_coeff.setMinimumHeight(160)
        self.plot_coeff.showGrid(x=True, y=True, alpha=0.25)
        self.plot_coeff.addLegend(offset=(10, 10))
        plot_lay.addWidget(self.plot_coeff)

        root.addWidget(self.grp_plots)
        root.addStretch(1)

        # Initial visibility
        self._on_model_type_changed(0)

    def _build_compact_ui(self):
        self._apply_theme_styles(restyle_plots=False)

        root = QtWidgets.QVBoxLayout(self)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(10)

        header = QtWidgets.QFrame()
        header.setObjectName("temporalHeader")
        h = QtWidgets.QHBoxLayout(header)
        h.setContentsMargins(14, 10, 14, 10)
        h.setSpacing(10)

        badge = QtWidgets.QLabel("T")
        badge.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        badge.setFixedSize(34, 34)
        badge.setStyleSheet(
            "background: #2d8cff; color: white; border-radius: 17px; "
            "font-weight: 800; font-size: 14pt;"
        )
        h.addWidget(badge)

        title_col = QtWidgets.QVBoxLayout()
        title_col.setContentsMargins(0, 0, 0, 0)
        title_col.setSpacing(1)
        title = QtWidgets.QLabel("Temporal Modeling")
        title.setProperty("class", "title")
        subtitle = QtWidgets.QLabel("Continuous GLM and trial-level FLMM analysis")
        subtitle.setProperty("class", "muted")
        title_col.addWidget(title)
        title_col.addWidget(subtitle)
        h.addLayout(title_col, 1)

        self.combo_model_type = QtWidgets.QComboBox()
        self.combo_model_type.addItems(["Continuous GLM", "Trial-level FLMM (fastFMM)"])
        self.combo_model_type.setMinimumWidth(220)
        self.combo_model_type.setToolTip(
            "Continuous GLM models how predictor time courses shape one continuous signal.\n"
            "Trial-level FLMM models how per-trial scalar covariates modulate aligned response curves "
            "and needs repeated trials/subjects."
        )
        h.addWidget(self.combo_model_type)

        self.btn_fit = QtWidgets.QPushButton("Fit")
        self.btn_fit.setProperty("class", "primary")
        self.btn_fit.setMinimumWidth(86)
        self.btn_fit.setToolTip("Fit the model using the current scope (Active file / All / Per-file batch).")
        h.addWidget(self.btn_fit)

        self.btn_cancel = QtWidgets.QPushButton("Cancel")
        self.btn_cancel.setMinimumWidth(78)
        self.btn_cancel.setEnabled(False)
        self.btn_cancel.setToolTip("Stop the current batch fit (single fits run synchronously).")
        h.addWidget(self.btn_cancel)

        # Export menu - data CSV / plot PNG / full HDF5 bundle.
        self.btn_export = QtWidgets.QToolButton()
        self.btn_export.setText("Export")
        self.btn_export.setToolTip("Export the active temporal model report, tables, and plots.")
        self.btn_export.setPopupMode(QtWidgets.QToolButton.ToolButtonPopupMode.MenuButtonPopup)
        self.menu_export = QtWidgets.QMenu(self.btn_export)
        self.act_export_all = self.menu_export.addAction("Full report - results + plots...")
        self.menu_export.addSeparator()
        self.act_export_results_bundle = self.menu_export.addAction("Results folder - summary + tables...")
        self.act_export_plots_bundle = self.menu_export.addAction("Plots folder - all visible plots...")
        self.menu_export.addSeparator()
        self.act_export_kernels_csv = self.menu_export.addAction("Active GLM kernels CSV...")
        self.act_export_importance_csv = self.menu_export.addAction("Active importance CSV...")
        self.act_export_summary_txt = self.menu_export.addAction("Summary text...")
        self.act_export_group_kernels_csv = self.menu_export.addAction("Group kernels CSV...")
        self.act_export_group_importance_csv = self.menu_export.addAction("Group importance CSV...")
        self.menu_export.addSeparator()
        self.act_export_state_json = self.menu_export.addAction("JSON snapshot...")
        self.menu_export.addSeparator()
        self.act_export_current_plot = self.menu_export.addAction("Current tab plot PNG...")
        self.act_export_current_plot_svg = self.menu_export.addAction("Current tab plot SVG...")
        self.menu_export.addSeparator()
        self.act_export_publication_figures = self.menu_export.addAction(
            "Publication figures (PDF + 300 DPI PNG)..."
        )
        self.btn_export.setMenu(self.menu_export)
        h.addWidget(self.btn_export)

        self.progress_model = QtWidgets.QProgressBar()
        self.progress_model.setMinimumWidth(220)
        self.progress_model.setMaximumWidth(320)
        self.progress_model.setVisible(False)
        h.addWidget(self.progress_model)
        root.addWidget(header)

        # ---- Scope strip: Active / All / Per-file (batch) + file selector ----
        scope_bar = QtWidgets.QFrame()
        scope_bar.setObjectName("temporalScopeBar")
        sb = QtWidgets.QHBoxLayout(scope_bar)
        sb.setContentsMargins(14, 4, 14, 6)
        sb.setSpacing(10)

        scope_lbl = QtWidgets.QLabel("Scope")
        scope_lbl.setProperty("class", "muted")
        sb.addWidget(scope_lbl)

        self.combo_fit_scope = QtWidgets.QComboBox()
        self.combo_fit_scope.addItem("Active file (single)", "active")
        self.combo_fit_scope.addItem("All loaded (concatenated)", "all")
        self.combo_fit_scope.addItem("Per-file batch + group", "batch")
        self.combo_fit_scope.setMinimumWidth(220)
        self.combo_fit_scope.setToolTip(
            "Active: fit the selected animal only.\n"
            "All: concatenate every loaded recording into one GLM.\n"
            "Per-file batch: fit each animal independently, then aggregate for the Group tab."
        )
        sb.addWidget(self.combo_fit_scope)

        file_lbl = QtWidgets.QLabel("File")
        file_lbl.setProperty("class", "muted")
        sb.addWidget(file_lbl)

        self.combo_active_file = QtWidgets.QComboBox()
        self.combo_active_file.setMinimumWidth(280)
        self.combo_active_file.setToolTip("Active animal/recording (used when Scope = Active file).")
        sb.addWidget(self.combo_active_file, 1)

        self.btn_prev_file = QtWidgets.QToolButton()
        self.btn_prev_file.setText("◀")
        self.btn_prev_file.setToolTip("Previous file")
        self.btn_next_file = QtWidgets.QToolButton()
        self.btn_next_file.setText("▶")
        self.btn_next_file.setToolTip("Next file")
        sb.addWidget(self.btn_prev_file)
        sb.addWidget(self.btn_next_file)

        self.lbl_fit_state = QtWidgets.QLabel("No fit yet")
        self.lbl_fit_state.setProperty("class", "muted")
        sb.addWidget(self.lbl_fit_state)
        root.addWidget(scope_bar)

        split = QtWidgets.QSplitter(QtCore.Qt.Orientation.Horizontal)
        split.setChildrenCollapsible(False)
        split.setHandleWidth(6)
        root.addWidget(split, 1)

        left = QtWidgets.QFrame()
        left.setObjectName("temporalControls")
        left_lay = QtWidgets.QHBoxLayout(left)
        left_lay.setContentsMargins(8, 8, 8, 8)
        left_lay.setSpacing(10)

        nav = QtWidgets.QFrame()
        nav.setObjectName("temporalNav")
        nav_lay = QtWidgets.QVBoxLayout(nav)
        nav_lay.setContentsMargins(8, 8, 8, 8)
        nav_lay.setSpacing(8)
        self.btn_nav_model = self._make_nav_button("Model")
        self.btn_nav_predictors = self._make_nav_button("Predictors")
        self.btn_nav_files = self._make_nav_button("Files\n& Group")
        self.btn_nav_fit = self._make_nav_button("Fit")
        for btn in (self.btn_nav_model, self.btn_nav_predictors, self.btn_nav_files, self.btn_nav_fit):
            nav_lay.addWidget(btn)
        nav_lay.addStretch(1)
        left_lay.addWidget(nav)

        self.stack_controls = QtWidgets.QStackedWidget()
        left_lay.addWidget(self.stack_controls, 1)
        split.addWidget(left)

        workspace = QtWidgets.QFrame()
        workspace.setObjectName("temporalWorkspace")
        workspace_lay = QtWidgets.QVBoxLayout(workspace)
        workspace_lay.setContentsMargins(10, 10, 10, 10)
        workspace_lay.setSpacing(8)
        self.tabs_workspace = QtWidgets.QTabWidget()
        self.tabs_workspace.setDocumentMode(True)
        workspace_lay.addWidget(self.tabs_workspace, 1)
        split.addWidget(workspace)
        split.setStretchFactor(0, 0)
        split.setStretchFactor(1, 1)
        split.setSizes([430, 1100])

        self._build_model_page()
        self._build_predictor_page()
        self._build_files_page()
        self._build_fit_page()
        self._build_workspace_pages()
        self._install_empty_state_hints()

        self.btn_nav_model.setChecked(True)
        self.btn_nav_model.clicked.connect(lambda: self._select_control_page(0))
        self.btn_nav_predictors.clicked.connect(lambda: self._select_control_page(1))
        self.btn_nav_files.clicked.connect(lambda: self._select_control_page(2))
        self.btn_nav_fit.clicked.connect(lambda: self._select_control_page(3))
        self.btn_fit_side.clicked.connect(self._on_fit_clicked)
        self.btn_fit_all_files.clicked.connect(self._on_fit_all_files_clicked)
        self.btn_cancel.clicked.connect(self._on_cancel_clicked)
        # Export menu actions.
        self.btn_export.clicked.connect(self._export_temporal_full_report)
        self.act_export_all.triggered.connect(self._export_temporal_full_report)
        self.act_export_results_bundle.triggered.connect(self._export_temporal_results)
        self.act_export_plots_bundle.triggered.connect(self._export_temporal_plots)
        self.act_export_kernels_csv.triggered.connect(self._export_kernels_csv)
        self.act_export_importance_csv.triggered.connect(self._export_importance_csv)
        self.act_export_summary_txt.triggered.connect(self._export_summary_txt)
        self.act_export_group_kernels_csv.triggered.connect(self._export_group_kernels_csv)
        self.act_export_group_importance_csv.triggered.connect(self._export_group_importance_csv)
        self.act_export_state_json.triggered.connect(self._export_state_json)
        self.act_export_current_plot.triggered.connect(lambda: self._export_current_plot("png"))
        self.act_export_current_plot_svg.triggered.connect(lambda: self._export_current_plot("svg"))
        self.act_export_publication_figures.triggered.connect(self._export_publication_figures)
        self.combo_fit_scope.currentIndexChanged.connect(self._on_fit_scope_changed)
        self.combo_active_file.currentIndexChanged.connect(self._on_active_file_changed)
        self.btn_prev_file.clicked.connect(lambda: self._step_active_file(-1))
        self.btn_next_file.clicked.connect(lambda: self._step_active_file(1))
        self._on_model_type_changed(0)

    def _build_model_page(self):
        page = QtWidgets.QWidget()
        lay = QtWidgets.QVBoxLayout(page)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(8)

        self.grp_glm = QtWidgets.QGroupBox("GLM Settings")
        gl = QtWidgets.QFormLayout(self.grp_glm)
        gl.setContentsMargins(12, 18, 12, 12)
        gl.setHorizontalSpacing(10)
        gl.setVerticalSpacing(8)

        self.combo_basis = QtWidgets.QComboBox()
        self.combo_basis.addItems(["Raised cosine", "B-spline", "FIR"])
        gl.addRow("Basis", self.combo_basis)

        self.spin_n_basis = QtWidgets.QSpinBox()
        self.spin_n_basis.setRange(2, 50)
        self.spin_n_basis.setValue(8)
        gl.addRow("Basis count", self.spin_n_basis)

        self.combo_reg = QtWidgets.QComboBox()
        self.combo_reg.addItems(["Ridge", "Lasso", "OLS"])
        gl.addRow("Regularization", self.combo_reg)

        self.spin_alpha = QtWidgets.QDoubleSpinBox()
        self.spin_alpha.setRange(0.001, 1000.0)
        self.spin_alpha.setValue(1.0)
        self.spin_alpha.setDecimals(3)
        gl.addRow("Alpha", self.spin_alpha)

        self.spin_kernel_pre = QtWidgets.QDoubleSpinBox()
        self.spin_kernel_pre.setRange(-30.0, 0.0)
        self.spin_kernel_pre.setValue(-1.0)
        self.spin_kernel_pre.setDecimals(1)
        self.spin_kernel_pre.setSuffix(" s")
        gl.addRow("Kernel pre", self.spin_kernel_pre)

        self.spin_kernel_post = QtWidgets.QDoubleSpinBox()
        self.spin_kernel_post.setRange(0.1, 60.0)
        self.spin_kernel_post.setValue(3.0)
        self.spin_kernel_post.setDecimals(1)
        self.spin_kernel_post.setSuffix(" s")
        gl.addRow("Kernel post", self.spin_kernel_post)

        self.spin_glm_bootstrap = QtWidgets.QSpinBox()
        self.spin_glm_bootstrap.setRange(0, 2000)
        self.spin_glm_bootstrap.setValue(100)
        self.spin_glm_bootstrap.setSpecialValueText("off")
        self.spin_glm_bootstrap.setToolTip("Circular-shift bootstraps for leave-one-out contribution p-values.")
        gl.addRow("Shift bootstraps", self.spin_glm_bootstrap)

        self.spin_glm_jobs = QtWidgets.QSpinBox()
        max_jobs = max(1, os.cpu_count() or 1)
        self.spin_glm_jobs.setRange(1, max_jobs)
        self.spin_glm_jobs.setValue(min(4, max_jobs))
        self.spin_glm_jobs.setToolTip("Parallel jobs used for circular-shift bootstrap fits.")
        gl.addRow("Bootstrap jobs", self.spin_glm_jobs)

        self.spin_glm_cv_folds = QtWidgets.QSpinBox()
        self.spin_glm_cv_folds.setRange(0, 20)
        self.spin_glm_cv_folds.setValue(5)
        self.spin_glm_cv_folds.setSpecialValueText("auto")
        self.spin_glm_cv_folds.setToolTip(
            "Number of cross-validation folds for out-of-sample R^2. "
            "Set 0/auto to let pyBer pick: one fold per file when multiple files "
            "are loaded, otherwise 5 contiguous time blocks for a single file."
        )
        gl.addRow("CV folds", self.spin_glm_cv_folds)
        lay.addWidget(self.grp_glm)

        self.grp_flmm = QtWidgets.QGroupBox("FLMM Settings")
        fl = QtWidgets.QFormLayout(self.grp_flmm)
        fl.setContentsMargins(12, 18, 12, 12)
        fl.setHorizontalSpacing(10)
        fl.setVerticalSpacing(8)

        self.lbl_flmm_status = QtWidgets.QLabel("")
        self.lbl_flmm_status.setProperty("class", "muted")
        self.lbl_flmm_status.setWordWrap(True)
        fl.addRow("Backend", self.lbl_flmm_status)

        self.edit_formula = QtWidgets.QLineEdit("Y.obs ~ 1")
        self.edit_formula.setPlaceholderText("Leave as Y.obs ~ 1 to auto-use selected predictors")
        fl.addRow("Fixed formula", self.edit_formula)
        self.edit_random = QtWidgets.QLineEdit("~1")
        self.edit_random.setPlaceholderText("e.g. ~1 or ~time")
        fl.addRow("Random", self.edit_random)
        self.edit_group_var = QtWidgets.QLineEdit("subject")
        fl.addRow("Group var", self.edit_group_var)
        self.spin_nknots = QtWidgets.QSpinBox()
        self.spin_nknots.setRange(0, 100)
        self.spin_nknots.setValue(0)
        self.spin_nknots.setSpecialValueText("auto")
        fl.addRow("Min knots", self.spin_nknots)
        self.spin_boots = QtWidgets.QSpinBox()
        self.spin_boots.setRange(0, 5000)
        self.spin_boots.setValue(0)
        self.spin_boots.setSpecialValueText("analytic")
        fl.addRow("Bootstrap iter", self.spin_boots)
        self.combo_flmm_importance = QtWidgets.QComboBox()
        self.combo_flmm_importance.addItem("Fast coefficient ranking", "fast")
        self.combo_flmm_importance.addItem("Permutation contribution test", "perm")
        self.combo_flmm_importance.addItem("Leave-one-out AIC (slow)", "loo")
        self.combo_flmm_importance.addItem("Off", "off")
        self.combo_flmm_importance.setToolTip(
            "Fast: descriptive ranking from the existing fit.\n"
            "Permutation: shuffle each predictor's trial values N times, "
            "refit, derive a p-value for variable contribution (slower but rigorous).\n"
            "Leave-one-out: refit FLMM without each predictor and compare AICs (very slow)."
        )
        fl.addRow("Contribution", self.combo_flmm_importance)

        self.spin_flmm_perm = QtWidgets.QSpinBox()
        self.spin_flmm_perm.setRange(50, 5000)
        self.spin_flmm_perm.setValue(200)
        self.spin_flmm_perm.setSingleStep(50)
        self.spin_flmm_perm.setToolTip(
            "Number of permutations for the variable-contribution test "
            "(only used when Contribution = Permutation). The smallest "
            "achievable p-value is 1/(N+1)."
        )
        fl.addRow("Permutations", self.spin_flmm_perm)
        lay.addWidget(self.grp_flmm)
        lay.addStretch(1)
        self.stack_controls.addWidget(page)

    def _build_predictor_page(self):
        page = QtWidgets.QWidget()
        lay = QtWidgets.QVBoxLayout(page)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(8)

        self.grp_predictors = QtWidgets.QGroupBox("Predictors")
        pl = QtWidgets.QVBoxLayout(self.grp_predictors)
        pl.setContentsMargins(12, 18, 12, 12)
        pl.setSpacing(8)

        self.combo_available_predictors = QtWidgets.QComboBox()
        self.combo_available_predictors.setToolTip("Choose from loaded event, behavior, and numeric behavior columns.")
        pl.addWidget(self.combo_available_predictors)

        self.list_predictors = QtWidgets.QListWidget()
        self.list_predictors.setMinimumHeight(220)
        pl.addWidget(self.list_predictors)

        row = QtWidgets.QHBoxLayout()
        self.btn_add_predictor = QtWidgets.QPushButton("+ Add")
        self.btn_remove_predictor = QtWidgets.QPushButton("- Remove")
        row.addWidget(self.btn_add_predictor)
        row.addWidget(self.btn_remove_predictor)
        row.addStretch(1)
        pl.addLayout(row)

        self.lbl_predictor_hint = QtWidgets.QLabel(
            "Predictors are populated from loaded DIO events, behavior states, behavior onsets, and numeric behavior columns."
        )
        self.lbl_predictor_hint.setProperty("class", "muted")
        self.lbl_predictor_hint.setWordWrap(True)
        pl.addWidget(self.lbl_predictor_hint)
        lay.addWidget(self.grp_predictors, 1)
        self.stack_controls.addWidget(page)

    def _build_files_page(self):
        page = QtWidgets.QWidget()
        lay = QtWidgets.QVBoxLayout(page)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(8)

        grp = QtWidgets.QGroupBox("Loaded recordings")
        gl = QtWidgets.QVBoxLayout(grp)
        gl.setContentsMargins(12, 18, 12, 12)
        gl.setSpacing(8)

        hint = QtWidgets.QLabel(
            "Select an animal to make it the active recording. The Group tab aggregates "
            "per-file fits once you run a Per-file batch."
        )
        hint.setProperty("class", "muted")
        hint.setWordWrap(True)
        gl.addWidget(hint)

        self.list_files = QtWidgets.QListWidget()
        self.list_files.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.SingleSelection)
        self.list_files.setMinimumHeight(220)
        gl.addWidget(self.list_files, 1)

        self.btn_fit_all_files = QtWidgets.QPushButton("Fit each file (per-file batch)")
        self.btn_fit_all_files.setProperty("class", "primary")
        gl.addWidget(self.btn_fit_all_files)

        self.lbl_batch_status = QtWidgets.QLabel("")
        self.lbl_batch_status.setProperty("class", "muted")
        self.lbl_batch_status.setWordWrap(True)
        gl.addWidget(self.lbl_batch_status)

        self.btn_clear_fits = QtWidgets.QPushButton("Clear cached fits")
        self.btn_clear_fits.setToolTip("Discard all cached per-file results and group aggregates.")
        gl.addWidget(self.btn_clear_fits)
        try:
            self.btn_clear_fits.clicked.connect(self._on_clear_cached_fits)
        except Exception:
            pass
        lay.addWidget(grp, 1)
        self.stack_controls.addWidget(page)
        self.list_files.itemSelectionChanged.connect(self._on_file_list_selection_changed)

    def _build_fit_page(self):
        page = QtWidgets.QWidget()
        lay = QtWidgets.QVBoxLayout(page)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(8)

        grp = QtWidgets.QGroupBox("Fit control")
        gl = QtWidgets.QVBoxLayout(grp)
        gl.setContentsMargins(12, 18, 12, 12)
        gl.setSpacing(10)
        self.lbl_data_status = QtWidgets.QLabel("No PSTH data has been pushed yet.")
        self.lbl_data_status.setProperty("class", "muted")
        self.lbl_data_status.setWordWrap(True)
        gl.addWidget(self.lbl_data_status)

        # Mode-aware Fit button (the same handler as the top-bar button).
        self.btn_fit_side = QtWidgets.QPushButton("Fit current scope")
        self.btn_fit_side.setProperty("class", "primary")
        self.btn_fit_side.setToolTip(
            "Run the model on the current scope. Equivalent to the Fit button in the top bar."
        )
        gl.addWidget(self.btn_fit_side)

        export_row = QtWidgets.QHBoxLayout()
        export_row.setContentsMargins(0, 0, 0, 0)
        export_row.setSpacing(8)
        self.btn_export_temporal_all = QtWidgets.QPushButton("Export full report")
        self.btn_export_temporal_all.setProperty("class", "primary")
        self.btn_export_temporal_all.setToolTip("Export summary, model tables, input traces, and all temporal plots.")
        self.btn_export_temporal_results = QtWidgets.QPushButton("Tables")
        self.btn_export_temporal_results.setToolTip("Export model summary, predictions, kernels, coefficients, traces, and importance tables.")
        self.btn_export_temporal_plots = QtWidgets.QPushButton("Plots")
        self.btn_export_temporal_plots.setToolTip("Export the temporal modeling plots as PNG and PDF.")
        export_row.addWidget(self.btn_export_temporal_all)
        export_row.addWidget(self.btn_export_temporal_results)
        export_row.addWidget(self.btn_export_temporal_plots)
        export_row.addStretch(1)
        gl.addLayout(export_row)

        contrib_lay = QtWidgets.QHBoxLayout()
        self.chk_run_contrib = QtWidgets.QCheckBox("Run leave-one-predictor-out contribution")
        self.chk_run_contrib.setChecked(True)
        self.chk_run_contrib.setToolTip(
            "Disable to skip the per-predictor reduced-fit comparison (saves time on large designs)."
        )
        contrib_lay.addWidget(self.chk_run_contrib)
        contrib_lay.addStretch(1)
        gl.addLayout(contrib_lay)

        gl.addStretch(1)
        lay.addWidget(grp, 1)
        self.stack_controls.addWidget(page)

    def _build_workspace_pages(self):
        summary_page = QtWidgets.QWidget()
        summary_lay = QtWidgets.QVBoxLayout(summary_page)
        summary_lay.setContentsMargins(10, 10, 10, 10)
        self.txt_summary = QtWidgets.QTextEdit()
        self.txt_summary.setReadOnly(True)
        summary_lay.addWidget(self.txt_summary, 1)
        self.tabs_workspace.addTab(summary_page, "Summary")

        kernel_page = QtWidgets.QWidget()
        kernel_lay = QtWidgets.QVBoxLayout(kernel_page)
        kernel_lay.setContentsMargins(10, 10, 10, 10)
        filter_row = QtWidgets.QHBoxLayout()
        filter_row.setSpacing(8)
        lbl_filter = QtWidgets.QLabel("Show kernels")
        lbl_filter.setProperty("class", "muted")
        filter_row.addWidget(lbl_filter)
        self.btn_kernel_all = QtWidgets.QPushButton("All")
        self.btn_kernel_none = QtWidgets.QPushButton("None")
        filter_row.addWidget(self.btn_kernel_all)
        filter_row.addWidget(self.btn_kernel_none)
        filter_row.addSpacing(8)
        filter_row.addWidget(QtWidgets.QLabel("Layout"))
        self.combo_kernel_layout = QtWidgets.QComboBox()
        self.combo_kernel_layout.addItem("Overlay", "overlay")
        self.combo_kernel_layout.addItem("Small panels", "grid")
        self.combo_kernel_layout.setToolTip(
            "Overlay compares every selected kernel in one plot. Small panels give each feature its own axis."
        )
        filter_row.addWidget(self.combo_kernel_layout)
        self.list_kernel_filter = QtWidgets.QListWidget()
        self.list_kernel_filter.setMaximumHeight(86)
        self.list_kernel_filter.setFlow(QtWidgets.QListView.Flow.LeftToRight)
        self.list_kernel_filter.setWrapping(True)
        self.list_kernel_filter.setResizeMode(QtWidgets.QListView.ResizeMode.Adjust)
        self.list_kernel_filter.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.list_kernel_filter.setVerticalScrollMode(QtWidgets.QAbstractItemView.ScrollMode.ScrollPerPixel)
        filter_row.addWidget(self.list_kernel_filter, 1)
        kernel_lay.addLayout(filter_row)

        self.lbl_kernel_hint = QtWidgets.QLabel(
            "Event predictors show impulse-response kernels. Continuous predictors are z-scored and show signal gain per +1 SD."
        )
        self.lbl_kernel_hint.setProperty("class", "muted")
        self.lbl_kernel_hint.setWordWrap(True)
        kernel_lay.addWidget(self.lbl_kernel_hint)

        self.stack_kernel_plots = QtWidgets.QStackedWidget()
        self.plot_kernel = pg.PlotWidget(title="Estimated kernels")
        self._style_plot(self.plot_kernel)
        self.stack_kernel_plots.addWidget(self.plot_kernel)

        self.kernel_grid_container = QtWidgets.QWidget()
        self.kernel_grid_layout = QtWidgets.QGridLayout(self.kernel_grid_container)
        self.kernel_grid_layout.setContentsMargins(0, 0, 0, 0)
        self.kernel_grid_layout.setHorizontalSpacing(10)
        self.kernel_grid_layout.setVerticalSpacing(10)
        self.kernel_grid_scroll = QtWidgets.QScrollArea()
        self.kernel_grid_scroll.setWidgetResizable(True)
        self.kernel_grid_scroll.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
        self.kernel_grid_scroll.setWidget(self.kernel_grid_container)
        self.stack_kernel_plots.addWidget(self.kernel_grid_scroll)
        kernel_lay.addWidget(self.stack_kernel_plots, 1)
        self.tabs_workspace.addTab(kernel_page, "Kernels")

        prediction_page = QtWidgets.QWidget()
        prediction_lay = QtWidgets.QVBoxLayout(prediction_page)
        prediction_lay.setContentsMargins(10, 10, 10, 10)
        self.plot_prediction = pg.PlotWidget(title="Actual vs predicted")
        self._style_plot(self.plot_prediction)
        prediction_lay.addWidget(self.plot_prediction, 1)
        self.tabs_workspace.addTab(prediction_page, "Prediction")

        traces_page = QtWidgets.QWidget()
        traces_lay = QtWidgets.QVBoxLayout(traces_page)
        traces_lay.setContentsMargins(10, 10, 10, 10)
        trace_controls = QtWidgets.QHBoxLayout()
        trace_controls.setSpacing(8)
        trace_controls.addWidget(QtWidgets.QLabel("Color"))
        self.combo_flmm_trace_color = QtWidgets.QComboBox()
        self.combo_flmm_trace_color.addItem("Single", "single")
        self.combo_flmm_trace_color.addItem("File / group", "group")
        self.combo_flmm_trace_color.addItem("Predictor value", "predictor")
        self.combo_flmm_trace_color.setToolTip("Choose how individual FLMM input traces are colored.")
        trace_controls.addWidget(self.combo_flmm_trace_color)
        trace_controls.addWidget(QtWidgets.QLabel("Predictor"))
        self.combo_flmm_trace_feature = QtWidgets.QComboBox()
        self.combo_flmm_trace_feature.setMinimumWidth(220)
        self.combo_flmm_trace_feature.setToolTip("Predictor used when coloring traces by predictor value.")
        trace_controls.addWidget(self.combo_flmm_trace_feature)
        trace_controls.addWidget(QtWidgets.QLabel("Max traces"))
        self.spin_flmm_trace_max = QtWidgets.QSpinBox()
        self.spin_flmm_trace_max.setRange(5, 2000)
        self.spin_flmm_trace_max.setValue(250)
        self.spin_flmm_trace_max.setToolTip("Limit visible individual traces for speed and readability.")
        trace_controls.addWidget(self.spin_flmm_trace_max)
        self.chk_flmm_trace_mean = QtWidgets.QCheckBox("Mean +/- SEM")
        self.chk_flmm_trace_mean.setChecked(True)
        trace_controls.addWidget(self.chk_flmm_trace_mean)
        self.lbl_flmm_trace_summary = QtWidgets.QLabel("")
        self.lbl_flmm_trace_summary.setProperty("class", "muted")
        trace_controls.addWidget(self.lbl_flmm_trace_summary, 1)
        traces_lay.addLayout(trace_controls)
        self.plot_flmm_traces = pg.PlotWidget(title="FLMM input traces")
        self._style_plot(self.plot_flmm_traces)
        traces_lay.addWidget(self.plot_flmm_traces, 1)
        self.tabs_workspace.addTab(traces_page, "Traces")

        illustration_page = QtWidgets.QWidget()
        illustration_lay = QtWidgets.QVBoxLayout(illustration_page)
        illustration_lay.setContentsMargins(10, 10, 10, 10)
        controls = QtWidgets.QHBoxLayout()
        controls.setSpacing(8)
        lbl_feature = QtWidgets.QLabel("Overlay feature")
        lbl_feature.setProperty("class", "muted")
        controls.addWidget(lbl_feature)
        self.combo_illustration_feature = QtWidgets.QComboBox()
        self.combo_illustration_feature.setMinimumWidth(220)
        controls.addWidget(self.combo_illustration_feature)

        self.chk_show_signal = QtWidgets.QCheckBox("Signal")
        self.chk_show_signal.setChecked(True)
        self.chk_show_predicted = QtWidgets.QCheckBox("Predicted")
        self.chk_show_predicted.setChecked(False)
        self.chk_show_contribution = QtWidgets.QCheckBox("Contribution")
        self.chk_show_contribution.setChecked(True)
        self.chk_show_raw_predictor = QtWidgets.QCheckBox("Raw predictor")
        self.chk_show_raw_predictor.setChecked(False)
        for chk in (
            self.chk_show_signal,
            self.chk_show_predicted,
            self.chk_show_contribution,
            self.chk_show_raw_predictor,
        ):
            controls.addWidget(chk)
            chk.toggled.connect(self._on_illustration_overlay_toggled)
        self.lbl_illustration_stats = QtWidgets.QLabel("")
        self.lbl_illustration_stats.setProperty("class", "muted")
        controls.addWidget(self.lbl_illustration_stats, 1)
        illustration_lay.addLayout(controls)
        self.plot_illustration = pg.PlotWidget(title="Signal and selected feature contribution")
        self._style_plot(self.plot_illustration)
        pi = self.plot_illustration.getPlotItem()
        pi.showAxis("right")
        self._illustration_vb = pg.ViewBox()
        pi.scene().addItem(self._illustration_vb)
        pi.getAxis("right").linkToView(self._illustration_vb)
        self._illustration_vb.setXLink(pi)
        pi.vb.sigResized.connect(self._update_illustration_view)
        illustration_lay.addWidget(self.plot_illustration, 1)
        self.tabs_workspace.addTab(illustration_page, "Illustration")

        residual_page = QtWidgets.QWidget()
        residual_lay = QtWidgets.QVBoxLayout(residual_page)
        residual_lay.setContentsMargins(10, 10, 10, 10)
        self.plot_residuals = pg.PlotWidget(title="Residuals")
        self._style_plot(self.plot_residuals)
        residual_lay.addWidget(self.plot_residuals, 1)
        self.tabs_workspace.addTab(residual_page, "Residuals")

        importance_page = QtWidgets.QWidget()
        importance_lay = QtWidgets.QVBoxLayout(importance_page)
        importance_lay.setContentsMargins(10, 10, 10, 10)
        self.plot_importance = pg.PlotWidget(title="Feature contribution")
        self._style_plot(self.plot_importance)
        importance_lay.addWidget(self.plot_importance, 1)
        self.tabs_workspace.addTab(importance_page, "Importance")

        flmm_page = QtWidgets.QWidget()
        flmm_lay = QtWidgets.QVBoxLayout(flmm_page)
        flmm_lay.setContentsMargins(10, 10, 10, 10)
        flmm_lay.setSpacing(6)

        # Filter row (mirrors the Kernels-tab UX so the controls feel familiar).
        flmm_filter_row = QtWidgets.QHBoxLayout()
        flmm_filter_row.setSpacing(8)
        lbl_flmm_filter = QtWidgets.QLabel("Show coefficients")
        lbl_flmm_filter.setProperty("class", "muted")
        flmm_filter_row.addWidget(lbl_flmm_filter)
        self.btn_flmm_all = QtWidgets.QPushButton("All")
        self.btn_flmm_none = QtWidgets.QPushButton("None")
        flmm_filter_row.addWidget(self.btn_flmm_all)
        flmm_filter_row.addWidget(self.btn_flmm_none)
        flmm_filter_row.addSpacing(8)
        flmm_filter_row.addWidget(QtWidgets.QLabel("Layout"))
        self.combo_flmm_layout = QtWidgets.QComboBox()
        self.combo_flmm_layout.addItem("Overlay", "overlay")
        self.combo_flmm_layout.addItem("Single focus", "single")
        self.combo_flmm_layout.addItem("Small panels", "grid")
        self.combo_flmm_layout.setToolTip(
            "Overlay shows every selected coefficient curve on one axis.\n"
            "Single focus shows just one predictor large with CIs prominent.\n"
            "Small panels gives every predictor its own axis."
        )
        flmm_filter_row.addWidget(self.combo_flmm_layout)
        self.list_flmm_filter = QtWidgets.QListWidget()
        self.list_flmm_filter.setMaximumHeight(86)
        self.list_flmm_filter.setFlow(QtWidgets.QListView.Flow.LeftToRight)
        self.list_flmm_filter.setWrapping(True)
        self.list_flmm_filter.setResizeMode(QtWidgets.QListView.ResizeMode.Adjust)
        self.list_flmm_filter.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.list_flmm_filter.setVerticalScrollMode(QtWidgets.QAbstractItemView.ScrollMode.ScrollPerPixel)
        flmm_filter_row.addWidget(self.list_flmm_filter, 1)
        flmm_lay.addLayout(flmm_filter_row)

        self.lbl_flmm_hint = QtWidgets.QLabel(
            "Each curve is the time-varying coefficient of one predictor: "
            "how a +1-SD change in that scalar covariate shifts the aligned "
            "response at each time t. Solid = estimate. Filled band = joint 95% CI "
            "(multiple-comparison corrected across time). Dashed = pointwise 95% CI. "
            "Curves whose 95% CI does not include zero indicate a time window where "
            "the predictor significantly modulates the response."
        )
        self.lbl_flmm_hint.setProperty("class", "muted")
        self.lbl_flmm_hint.setWordWrap(True)
        flmm_lay.addWidget(self.lbl_flmm_hint)

        # Stacked: overlay plot vs grid scroll area, selected by Layout combo.
        self.stack_flmm_plots = QtWidgets.QStackedWidget()
        self.plot_coeff = pg.PlotWidget(title="FLMM coefficient curves")
        self._style_plot(self.plot_coeff)
        self.stack_flmm_plots.addWidget(self.plot_coeff)

        self.scroll_flmm_grid = QtWidgets.QScrollArea()
        self.scroll_flmm_grid.setWidgetResizable(True)
        self.scroll_flmm_grid.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
        self._flmm_grid_host = QtWidgets.QWidget()
        self._flmm_grid_layout = QtWidgets.QGridLayout(self._flmm_grid_host)
        self._flmm_grid_layout.setSpacing(10)
        self._flmm_grid_layout.setContentsMargins(0, 0, 0, 0)
        self.scroll_flmm_grid.setWidget(self._flmm_grid_host)
        self.stack_flmm_plots.addWidget(self.scroll_flmm_grid)

        flmm_lay.addWidget(self.stack_flmm_plots, 1)
        self.tabs_workspace.addTab(flmm_page, "FLMM")

        # Group tab — aggregates per-file fits.
        group_page = QtWidgets.QWidget()
        group_lay = QtWidgets.QVBoxLayout(group_page)
        group_lay.setContentsMargins(10, 10, 10, 10)
        group_top = QtWidgets.QHBoxLayout()
        group_top.setSpacing(8)
        self.lbl_group_summary = QtWidgets.QLabel(
            "Run a Per-file batch fit to populate the Group view."
        )
        self.lbl_group_summary.setProperty("class", "muted")
        self.lbl_group_summary.setWordWrap(True)
        group_top.addWidget(self.lbl_group_summary, 1)
        group_lay.addLayout(group_top)
        self.plot_group_kernels = pg.PlotWidget(title="Group kernels (precision weighted when CIs exist)")
        self._style_plot(self.plot_group_kernels)
        group_lay.addWidget(self.plot_group_kernels, 1)
        self.plot_group_importance = pg.PlotWidget(title="Group leave-one-out contribution")
        self._style_plot(self.plot_group_importance)
        group_lay.addWidget(self.plot_group_importance, 1)
        self.tabs_workspace.addTab(group_page, "Group")
        idx = self.tabs_workspace.indexOf(summary_page)
        if idx >= 0:
            self.tabs_workspace.setTabToolTip(
                idx,
                "GLM: how predictor time courses shape one continuous signal. "
                "FLMM: how trial-level scalar covariates modulate aligned response curves."
            )
        idx = self.tabs_workspace.indexOf(flmm_page)
        if idx >= 0:
            self.tabs_workspace.setTabToolTip(
                idx,
                "FLMM requires repeated trials/subjects; use Continuous GLM for one animal or one continuous trace."
            )

    def _make_nav_button(self, text: str) -> QtWidgets.QToolButton:
        btn = QtWidgets.QToolButton()
        btn.setText(text)
        btn.setCheckable(True)
        btn.setMinimumWidth(86)
        btn.setMinimumHeight(54)
        btn.setToolButtonStyle(QtCore.Qt.ToolButtonStyle.ToolButtonTextOnly)
        return btn

    def _select_control_page(self, index: int) -> None:
        self.stack_controls.setCurrentIndex(index)
        buttons = (self.btn_nav_model, self.btn_nav_predictors, self.btn_nav_files, self.btn_nav_fit)
        for i, btn in enumerate(buttons):
            btn.setChecked(i == index)

    def _install_empty_state_hints(self) -> None:
        """Empty-state hints removed by design - keep plots visually clean."""
        self._empty_state_items = []

    def _clear_empty_state_hints(self) -> None:
        self._empty_state_items = []

    def _style_plot(self, plot: pg.PlotWidget, min_height: int = 360) -> None:
        plot.setMinimumHeight(int(min_height))
        light = self._app_theme_mode == "light"
        bg = "#fbfcfe" if light else "#05080d"
        axis_color = "#334155" if light else "#516179"
        text_color = "#1f2a37" if light else "#c5d2e3"
        title_color = "#172033" if light else "#d7e0ee"
        plot.setBackground(bg)
        plot.showGrid(x=True, y=True, alpha=0.26 if light else 0.22)
        pi = plot.getPlotItem()
        if getattr(pi, "legend", None) is None:
            plot.addLegend(offset=(12, 12))
        for axis_name in ("left", "right", "bottom", "top"):
            axis = pi.getAxis(axis_name)
            if axis is None:
                continue
            axis.setPen(pg.mkPen(axis_color))
            axis.setTextPen(pg.mkPen(text_color))
        pi.titleLabel.item.setDefaultTextColor(QtGui.QColor(title_color))

    def _plot_widgets(self) -> List[pg.PlotWidget]:
        attrs = (
            "plot_kernel",
            "plot_prediction",
            "plot_flmm_traces",
            "plot_illustration",
            "plot_residuals",
            "plot_importance",
            "plot_coeff",
            "plot_group_kernels",
            "plot_group_importance",
        )
        plots: List[pg.PlotWidget] = []
        for attr in attrs:
            plot = getattr(self, attr, None)
            if isinstance(plot, pg.PlotWidget):
                plots.append(plot)
        plots.extend([plot for plot in getattr(self, "_kernel_grid_plots", []) if isinstance(plot, pg.PlotWidget)])
        return plots

    def _normalize_app_theme_mode(self, value: object) -> str:
        mode = str(value or "").strip().lower()
        return "light" if mode in {"light", "white", "l", "w"} else "dark"

    def _apply_theme_styles(self, restyle_plots: bool = True) -> None:
        self.setStyleSheet(_TEMPORAL_QSS_LIGHT if self._app_theme_mode == "light" else _TEMPORAL_QSS)
        if restyle_plots:
            for plot in self._plot_widgets():
                self._style_plot(plot)

    def set_app_theme_mode(self, theme_mode: object) -> None:
        self._app_theme_mode = self._normalize_app_theme_mode(theme_mode)
        self._apply_theme_styles(restyle_plots=True)

    def _progress_start(self, label: str, maximum: int = 0) -> None:
        if not hasattr(self, "progress_model"):
            return
        self.progress_model.setVisible(True)
        if maximum <= 0:
            self.progress_model.setRange(0, 0)
            self.progress_model.setFormat(label)
        else:
            self.progress_model.setRange(0, maximum)
            self.progress_model.setValue(0)
            self.progress_model.setFormat(f"{label} %p%")
        QtWidgets.QApplication.processEvents()

    def _progress_update(self, value: int, label: Optional[str] = None) -> None:
        if not hasattr(self, "progress_model") or not self.progress_model.isVisible():
            return
        if label:
            self.progress_model.setFormat(f"{label} %p%" if self.progress_model.maximum() > 0 else label)
        if self.progress_model.maximum() > 0:
            self.progress_model.setValue(max(0, min(int(value), self.progress_model.maximum())))
        QtWidgets.QApplication.processEvents()

    def _progress_finish(self) -> None:
        if not hasattr(self, "progress_model"):
            return
        self.progress_model.setVisible(False)
        self.progress_model.setRange(0, 100)
        self.progress_model.setValue(0)
        QtWidgets.QApplication.processEvents()

    def _set_fit_enabled(self, enabled: bool) -> None:
        for attr in ("btn_fit", "btn_fit_side"):
            btn = getattr(self, attr, None)
            if btn is not None:
                btn.setEnabled(enabled)

    # ------------------------------------------------------------------
    # Signal wiring
    # ------------------------------------------------------------------

    def _connect_signals(self):
        self.combo_model_type.currentIndexChanged.connect(self._on_model_type_changed)
        self.btn_fit.clicked.connect(self._on_fit_clicked)
        self.btn_add_predictor.clicked.connect(self._on_add_predictor)
        self.btn_remove_predictor.clicked.connect(self._on_remove_predictor)
        for widget in (
            self.combo_basis,
            self.combo_reg,
            self.combo_flmm_importance,
            self.spin_n_basis,
            self.spin_alpha,
            self.spin_kernel_pre,
            self.spin_kernel_post,
            self.spin_glm_bootstrap,
            self.spin_glm_jobs,
            self.spin_nknots,
            self.spin_boots,
        ):
            signal = getattr(widget, "currentIndexChanged", None) or getattr(widget, "valueChanged", None)
            if signal is not None:
                signal.connect(lambda *_: self._save_temporal_settings())
        for edit in (self.edit_formula, self.edit_random, self.edit_group_var):
            edit.editingFinished.connect(self._save_temporal_settings)
        if hasattr(self, "list_kernel_filter"):
            self.list_kernel_filter.itemChanged.connect(self._on_kernel_filter_changed)
            self.btn_kernel_all.clicked.connect(lambda: self._set_all_kernels_visible(True))
            self.btn_kernel_none.clicked.connect(lambda: self._set_all_kernels_visible(False))
        if hasattr(self, "combo_kernel_layout"):
            self.combo_kernel_layout.currentIndexChanged.connect(self._on_kernel_layout_changed)
        if hasattr(self, "list_flmm_filter"):
            self.list_flmm_filter.itemChanged.connect(self._on_flmm_filter_changed)
            self.btn_flmm_all.clicked.connect(lambda: self._set_all_flmm_visible(True))
            self.btn_flmm_none.clicked.connect(lambda: self._set_all_flmm_visible(False))
        if hasattr(self, "combo_flmm_layout"):
            self.combo_flmm_layout.currentIndexChanged.connect(self._on_flmm_layout_changed)
        if hasattr(self, "combo_illustration_feature"):
            self.combo_illustration_feature.currentIndexChanged.connect(self._on_illustration_feature_changed)
        if hasattr(self, "combo_flmm_trace_color"):
            self.combo_flmm_trace_color.currentIndexChanged.connect(lambda *_: self._refresh_flmm_trace_plot())
        if hasattr(self, "combo_flmm_trace_feature"):
            self.combo_flmm_trace_feature.currentIndexChanged.connect(lambda *_: self._refresh_flmm_trace_plot())
        if hasattr(self, "spin_flmm_trace_max"):
            self.spin_flmm_trace_max.valueChanged.connect(lambda *_: self._refresh_flmm_trace_plot())
        if hasattr(self, "chk_flmm_trace_mean"):
            self.chk_flmm_trace_mean.toggled.connect(lambda *_: self._refresh_flmm_trace_plot())
        if hasattr(self, "btn_export_temporal_results"):
            self.btn_export_temporal_results.clicked.connect(self._export_temporal_results)
        if hasattr(self, "btn_export_temporal_plots"):
            self.btn_export_temporal_plots.clicked.connect(self._export_temporal_plots)
        if hasattr(self, "btn_export_temporal_all"):
            self.btn_export_temporal_all.clicked.connect(self._export_temporal_full_report)

    def _load_temporal_settings(self) -> None:
        self._loading_settings = True
        try:
            prefix = "temporal_modeling/"
            self.combo_model_type.setCurrentIndex(int(self._settings.value(prefix + "model_type", 0)))
            for combo, key in ((self.combo_basis, "basis"), (self.combo_reg, "regularization")):
                text = str(self._settings.value(prefix + key, "") or "")
                idx = combo.findText(text)
                if idx >= 0:
                    combo.setCurrentIndex(idx)
            self.spin_n_basis.setValue(int(self._settings.value(prefix + "n_basis", self.spin_n_basis.value())))
            self.spin_alpha.setValue(float(self._settings.value(prefix + "alpha", self.spin_alpha.value())))
            self.spin_kernel_pre.setValue(float(self._settings.value(prefix + "kernel_pre", self.spin_kernel_pre.value())))
            self.spin_kernel_post.setValue(float(self._settings.value(prefix + "kernel_post", self.spin_kernel_post.value())))
            self.spin_glm_bootstrap.setValue(int(self._settings.value(prefix + "glm_shift_bootstraps", self.spin_glm_bootstrap.value())))
            self.spin_glm_jobs.setValue(int(self._settings.value(prefix + "glm_bootstrap_jobs", self.spin_glm_jobs.value())))
            self.edit_formula.setText(str(self._settings.value(prefix + "flmm_formula", self.edit_formula.text()) or "Y.obs ~ 1"))
            self.edit_random.setText(str(self._settings.value(prefix + "flmm_random", self.edit_random.text()) or "~1"))
            self.edit_group_var.setText(str(self._settings.value(prefix + "flmm_group_var", self.edit_group_var.text()) or "subject"))
            self.spin_nknots.setValue(int(self._settings.value(prefix + "flmm_nknots", self.spin_nknots.value())))
            self.spin_boots.setValue(int(self._settings.value(prefix + "flmm_boots", self.spin_boots.value())))
            mode = str(self._settings.value(prefix + "flmm_importance_mode", "fast") or "fast")
            idx = self.combo_flmm_importance.findData(mode, QtCore.Qt.ItemDataRole.UserRole)
            if idx < 0:
                idx = self.combo_flmm_importance.findText(mode)
            if idx >= 0:
                self.combo_flmm_importance.setCurrentIndex(idx)
            raw_predictors = str(self._settings.value(prefix + "predictor_keys", "") or "")
            self._saved_predictor_keys = [key for key in raw_predictors.split("\n") if key.strip()]
            raw_by_file = self._settings.value(prefix + "predictor_keys_by_file_json", "")
            if isinstance(raw_by_file, (bytes, bytearray)):
                raw_by_file = raw_by_file.decode("utf-8", errors="replace")
            try:
                parsed = json.loads(str(raw_by_file or "{}"))
            except Exception:
                parsed = {}
            if isinstance(parsed, dict):
                self._predictor_keys_by_file = {
                    str(fid): [str(key) for key in keys if str(key).strip()]
                    for fid, keys in parsed.items()
                    if isinstance(keys, list)
                }
            layout = str(self._settings.value(prefix + "kernel_layout", "overlay") or "overlay")
            if hasattr(self, "combo_kernel_layout"):
                idx = self.combo_kernel_layout.findData(layout, QtCore.Qt.ItemDataRole.UserRole)
                if idx >= 0:
                    self.combo_kernel_layout.setCurrentIndex(idx)
        finally:
            self._loading_settings = False

    def _save_temporal_settings(self) -> None:
        if getattr(self, "_loading_settings", False):
            return
        prefix = "temporal_modeling/"
        self._settings.setValue(prefix + "model_type", self.combo_model_type.currentIndex())
        self._settings.setValue(prefix + "basis", self.combo_basis.currentText())
        self._settings.setValue(prefix + "regularization", self.combo_reg.currentText())
        self._settings.setValue(prefix + "n_basis", self.spin_n_basis.value())
        self._settings.setValue(prefix + "alpha", self.spin_alpha.value())
        self._settings.setValue(prefix + "kernel_pre", self.spin_kernel_pre.value())
        self._settings.setValue(prefix + "kernel_post", self.spin_kernel_post.value())
        self._settings.setValue(prefix + "glm_shift_bootstraps", self.spin_glm_bootstrap.value())
        self._settings.setValue(prefix + "glm_bootstrap_jobs", self.spin_glm_jobs.value())
        self._settings.setValue(prefix + "flmm_formula", self.edit_formula.text().strip())
        self._settings.setValue(prefix + "flmm_random", self.edit_random.text().strip())
        self._settings.setValue(prefix + "flmm_group_var", self.edit_group_var.text().strip())
        self._settings.setValue(prefix + "flmm_nknots", self.spin_nknots.value())
        self._settings.setValue(prefix + "flmm_boots", self.spin_boots.value())
        self._settings.setValue(prefix + "flmm_importance_mode", self.combo_flmm_importance.currentData(QtCore.Qt.ItemDataRole.UserRole) or "fast")
        predictors = self._selected_predictor_keys() if hasattr(self, "list_predictors") else self._saved_predictor_keys
        predictors = list(predictors)
        self._saved_predictor_keys = list(predictors)
        if getattr(self, "_active_file_id", ""):
            self._predictor_keys_by_file[str(self._active_file_id)] = list(predictors)
        self._settings.setValue(prefix + "predictor_keys", "\n".join(predictors))
        self._settings.setValue(prefix + "predictor_keys_by_file_json", json.dumps(self._predictor_keys_by_file, sort_keys=True))
        if hasattr(self, "combo_kernel_layout"):
            self._settings.setValue(prefix + "kernel_layout", self.combo_kernel_layout.currentData(QtCore.Qt.ItemDataRole.UserRole) or "overlay")

    # ------------------------------------------------------------------
    # Public API — called by PostProcessingPanel
    # ------------------------------------------------------------------

    def set_data(
        self,
        processed_trials,
        psth_mat: Optional[np.ndarray] = None,
        psth_tvec: Optional[np.ndarray] = None,
        event_times: Optional[np.ndarray] = None,
        file_ids: Optional[List[str]] = None,
        per_file_mats: Optional[Dict[str, Tuple[np.ndarray, np.ndarray]]] = None,
        behavior_sources: Optional[Dict[str, Dict[str, Any]]] = None,
        event_rows: Optional[List[Dict[str, object]]] = None,
        group_mat: Optional[np.ndarray] = None,
        group_tvec: Optional[np.ndarray] = None,
        group_labels: Optional[List[str]] = None,
        visual_mode: int = 0,
        group_mode: bool = False,
    ):
        """Push data from the host panel into this widget."""
        self._processed_trials = processed_trials or []
        self._psth_mat = psth_mat
        self._psth_tvec = psth_tvec
        self._event_times = event_times
        self._file_ids = file_ids or []
        self._per_file_mats = per_file_mats or {}
        self._behavior_sources = dict(behavior_sources or {})
        self._event_rows = list(event_rows or [])
        self._group_mat = np.asarray(group_mat, float) if group_mat is not None else None
        self._group_tvec = np.asarray(group_tvec, float) if group_tvec is not None else None
        self._group_labels = list(group_labels or [])
        self._visual_mode = int(visual_mode) if isinstance(visual_mode, (int, np.integer)) else 0
        self._group_mode = bool(group_mode)
        self._refresh_file_widgets()
        self._refresh_predictor_catalog()
        # Drop cached fits for files that disappeared.
        live_ids = {self._proc_file_id(p, fallback=f"file_{i + 1}") for i, p in enumerate(self._processed_trials)}
        for fid in list(self._glm_results_by_file):
            if fid not in live_ids:
                self._glm_results_by_file.pop(fid, None)
                self._fit_summary_by_file.pop(fid, None)

        n_trials = len(self._processed_trials)
        psth_shape = tuple(np.shape(psth_mat)) if psth_mat is not None else None
        bits = [f"Processed recordings: {n_trials}"]
        if psth_shape:
            bits.append(f"PSTH matrix: {psth_shape[0]} x {psth_shape[1]}")
        if event_times is not None:
            bits.append(f"Events: {len(event_times)}")
        if self._behavior_sources:
            bits.append(f"Behavior files: {len(self._behavior_sources)}")
        if self._group_mat is not None and self._group_labels:
            bits.append(f"Group animals: {len(self._group_labels)}")
        if self._predictor_catalog:
            bits.append(f"Available predictors: {len(self._predictor_catalog)}")
        self.lbl_data_status.setText("\n".join(bits))

    # ------------------------------------------------------------------
    # State serialization (used by project save/load)
    # ------------------------------------------------------------------

    @staticmethod
    def _serialize_glm_result(r: GLMResult, include_design_matrix: bool = False) -> Dict[str, Any]:
        def _json_safe(value: Any) -> Any:
            if isinstance(value, (bool, np.bool_)):
                return bool(value)
            if isinstance(value, (int, float, np.floating, np.integer)):
                return float(value)
            if isinstance(value, dict):
                return {str(k): _json_safe(v) for k, v in value.items()}
            if isinstance(value, (list, tuple)):
                return [_json_safe(v) for v in value]
            return value

        payload = {
            "predictor_names": list(r.predictor_names or []),
            "kernels": {k: np.asarray(v, float).tolist() for k, v in (r.kernels or {}).items()},
            "kernel_ci_lower": {k: np.asarray(v, float).tolist() for k, v in (r.kernel_ci_lower or {}).items()},
            "kernel_ci_upper": {k: np.asarray(v, float).tolist() for k, v in (r.kernel_ci_upper or {}).items()},
            "kernel_tvec": np.asarray(r.kernel_tvec, float).tolist(),
            "time": np.asarray(r.time, float).tolist() if r.time is not None else [],
            "y_pred": np.asarray(r.y_pred, float).tolist() if r.y_pred is not None else [],
            "y_actual": np.asarray(r.y_actual, float).tolist() if r.y_actual is not None else [],
            "residuals": np.asarray(r.residuals, float).tolist() if r.residuals is not None else [],
            "r2": float(r.r2),
            "coefficients": np.asarray(r.coefficients, float).tolist() if r.coefficients is not None else [],
            "stats": {str(k): _json_safe(v) for k, v in (r.stats or {}).items()},
            "feature_importance": [
                {str(k): _json_safe(v) for k, v in (row or {}).items()}
                for row in (r.feature_importance or [])
            ],
        }
        if include_design_matrix:
            payload["design_matrix"] = np.asarray(r.design_matrix, float).tolist() if r.design_matrix is not None else []
        return payload

    @staticmethod
    def _deserialize_glm_result(payload: Dict[str, Any]) -> Optional[GLMResult]:
        if not isinstance(payload, dict):
            return None
        try:
            kernels = {k: np.asarray(v, float) for k, v in (payload.get("kernels", {}) or {}).items()}
            design_matrix = np.asarray(payload.get("design_matrix", []), float)
            if design_matrix.ndim != 2:
                design_matrix = np.zeros((0, 0), float)
            return GLMResult(
                predictor_names=list(payload.get("predictor_names", []) or []),
                kernels=kernels,
                kernel_ci_lower={k: np.asarray(v, float) for k, v in (payload.get("kernel_ci_lower", {}) or {}).items()},
                kernel_ci_upper={k: np.asarray(v, float) for k, v in (payload.get("kernel_ci_upper", {}) or {}).items()},
                kernel_tvec=np.asarray(payload.get("kernel_tvec", []), float),
                time=np.asarray(payload.get("time", []), float),
                y_pred=np.asarray(payload.get("y_pred", []), float),
                y_actual=np.asarray(payload.get("y_actual", []), float),
                residuals=np.asarray(payload.get("residuals", []), float),
                r2=float(payload.get("r2", 0.0)),
                coefficients=np.asarray(payload.get("coefficients", []), float),
                design_matrix=design_matrix,
                stats=dict(payload.get("stats", {}) or {}),
                feature_importance=list(payload.get("feature_importance", []) or []),
            )
        except Exception as exc:
            _LOG.warning("Could not deserialize GLM result: %s", exc)
            return None

    def serialize_state(self) -> Dict[str, Any]:
        """Return a JSON-serializable snapshot of the latest GLM/FLMM fits."""
        state: Dict[str, Any] = {
            "version": 1,
            "fit_mode": self._fit_mode,
            "active_file_id": self._active_file_id,
            "selected_predictors": self._selected_predictor_keys() if hasattr(self, "list_predictors") else [],
            "predictors_by_file": dict(self._predictor_keys_by_file),
            "settings": {
                "model_type": int(self.combo_model_type.currentIndex()) if hasattr(self, "combo_model_type") else 0,
                "basis": self.combo_basis.currentText() if hasattr(self, "combo_basis") else "",
                "n_basis": int(self.spin_n_basis.value()) if hasattr(self, "spin_n_basis") else 8,
                "regularization": self.combo_reg.currentText() if hasattr(self, "combo_reg") else "",
                "alpha": float(self.spin_alpha.value()) if hasattr(self, "spin_alpha") else 1.0,
                "kernel_pre": float(self.spin_kernel_pre.value()) if hasattr(self, "spin_kernel_pre") else -1.0,
                "kernel_post": float(self.spin_kernel_post.value()) if hasattr(self, "spin_kernel_post") else 3.0,
            },
            "glm_results_by_file": {
                fid: self._serialize_glm_result(res)
                for fid, res in self._glm_results_by_file.items()
            },
            "fit_summary_by_file": dict(self._fit_summary_by_file),
            "group_summary": dict(self._group_glm_summary or {}),
        }
        if self._glm_result is not None:
            state["glm_result_active"] = self._serialize_glm_result(self._glm_result)
        return state

    def deserialize_state(self, state: Dict[str, Any]) -> None:
        """Restore a snapshot produced by serialize_state()."""
        if not isinstance(state, dict):
            return
        try:
            self._fit_mode = str(state.get("fit_mode", "all"))
            by_file_predictors = state.get("predictors_by_file", {}) or {}
            if isinstance(by_file_predictors, dict):
                self._predictor_keys_by_file.update({
                    str(fid): [str(key) for key in keys if str(key).strip()]
                    for fid, keys in by_file_predictors.items()
                    if isinstance(keys, list)
                })
            if hasattr(self, "combo_fit_scope"):
                idx = self.combo_fit_scope.findData(self._fit_mode)
                if idx >= 0:
                    self.combo_fit_scope.blockSignals(True)
                    self.combo_fit_scope.setCurrentIndex(idx)
                    self.combo_fit_scope.blockSignals(False)
            self._active_file_id = str(state.get("active_file_id", "") or "")

            by_file = state.get("glm_results_by_file", {}) or {}
            self._glm_results_by_file = {}
            for fid, payload in by_file.items():
                res = self._deserialize_glm_result(payload)
                if res is not None:
                    self._glm_results_by_file[str(fid)] = res
            self._fit_summary_by_file = dict(state.get("fit_summary_by_file", {}) or {})
            self._group_glm_summary = dict(state.get("group_summary", {}) or {})

            active_payload = state.get("glm_result_active")
            if active_payload:
                res = self._deserialize_glm_result(active_payload)
                if res is not None:
                    self._glm_result = res
                    self._render_glm_summary_html(res, file_filter=self._active_file_id or None)
                    self._plot_glm_kernels(res)
                    self._plot_glm_fit(res)
                    self._plot_glm_illustration(res)
                    self._plot_feature_importance(
                        res.feature_importance or [],
                        value_key="delta_r2",
                        title="GLM leave-one-predictor-out contribution",
                        y_label="Drop in R^2",
                    )
            if self._glm_results_by_file:
                self._aggregate_group_results()
            self._refresh_file_widgets()
            self._update_fit_state_label()
        except Exception as exc:
            _LOG.warning("Could not restore temporal modeling state: %s", exc)

    # ------------------------------------------------------------------
    # Disk cache for expensive GLM intermediates
    # ------------------------------------------------------------------

    def _glm_cache_dir(self) -> str:
        base = QtCore.QStandardPaths.writableLocation(QtCore.QStandardPaths.StandardLocation.CacheLocation)
        if not base:
            base = os.path.join(os.path.expanduser("~"), ".cache", "pyBer")
        path = os.path.join(base, "temporal_glm_v2")
        os.makedirs(path, exist_ok=True)
        return path

    @staticmethod
    def _array_digest(values: np.ndarray) -> str:
        arr = np.asarray(values, float)
        arr = np.nan_to_num(arr, nan=1.23456789e300, posinf=9.87654321e299, neginf=-9.87654321e299)
        arr = np.ascontiguousarray(arr)
        return hashlib.blake2b(arr.view(np.uint8), digest_size=16).hexdigest()

    def _glm_cache_key(
        self,
        dataset: Dict[str, Any],
        kernel_window: Tuple[float, float],
        basis_type: str,
        regularization: str,
        alpha: float,
        n_basis: int,
        n_boot: int,
        run_contrib: bool,
    ) -> str:
        time = np.asarray(dataset.get("time", []), float)
        predictors = dict(dataset.get("predictors", {}) or {})
        predictor_digests = []
        for name in predictors:
            try:
                vec = ContinuousGLM._predictor_vector(time, predictors[name])
                digest = self._array_digest(vec)
            except Exception:
                digest = "unreadable"
            predictor_digests.append([str(name), digest])
        meta = {
            "version": int(getattr(self, "_glm_cache_version", 2)),
            "time": self._array_digest(time),
            "signal": self._array_digest(np.asarray(dataset.get("signal", []), float)),
            "predictors": predictor_digests,
            "used_records": [str(v) for v in (dataset.get("used_records", []) or [])],
            "segment_slices": [[int(a), int(b)] for a, b in (dataset.get("segment_slices", []) or [])],
            "kernel_window": [float(kernel_window[0]), float(kernel_window[1])],
            "basis_type": str(basis_type),
            "n_basis": int(n_basis),
            "regularization": str(regularization),
            "alpha": float(alpha),
            "n_boot": int(n_boot),
            "run_contrib": bool(run_contrib),
        }
        text = json.dumps(meta, sort_keys=True, separators=(",", ":"))
        return hashlib.blake2b(text.encode("utf-8"), digest_size=20).hexdigest()

    def _load_glm_cache(self, cache_key: str) -> Tuple[Optional[GLMResult], str]:
        path = os.path.join(self._glm_cache_dir(), f"{cache_key}.json")
        if not os.path.exists(path):
            return None, ""
        try:
            with open(path, "r", encoding="utf-8") as fh:
                payload = json.load(fh)
            if int(payload.get("cache_version", -1)) != int(getattr(self, "_glm_cache_version", 2)):
                return None, ""
            result = self._deserialize_glm_result(payload.get("result", {}) or {})
            summary = str(payload.get("summary", "") or "")
            if result is not None:
                result.stats = dict(result.stats or {})
                result.stats["cache_hit"] = True
            return result, summary
        except Exception as exc:
            _LOG.debug("Could not load GLM cache %s: %s", cache_key, exc)
            return None, ""

    def _save_glm_cache(self, cache_key: str, result: GLMResult, summary: str) -> None:
        path = os.path.join(self._glm_cache_dir(), f"{cache_key}.json")
        tmp = path + ".tmp"
        try:
            payload = {
                "cache_version": int(getattr(self, "_glm_cache_version", 2)),
                "summary": str(summary or ""),
                "result": self._serialize_glm_result(result, include_design_matrix=True),
            }
            with open(tmp, "w", encoding="utf-8") as fh:
                json.dump(payload, fh, separators=(",", ":"), allow_nan=True)
            os.replace(tmp, path)
        except Exception as exc:
            _LOG.debug("Could not save GLM cache %s: %s", cache_key, exc)
            try:
                if os.path.exists(tmp):
                    os.remove(tmp)
            except Exception:
                pass

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------

    @staticmethod
    def _safe_filename(text: object, fallback: str = "temporal_modeling") -> str:
        name = re.sub(r"[^0-9A-Za-z._-]+", "_", str(text or "").strip()).strip("._-")
        return name or fallback

    def _export_dir_from_user(self, title: str) -> str:
        start_dir = str(self._settings.value("temporal_modeling/export_dir", "") or "")
        if not start_dir or not os.path.isdir(start_dir):
            start_dir = os.getcwd()
        out_dir = QtWidgets.QFileDialog.getExistingDirectory(self, title, start_dir)
        if out_dir:
            self._settings.setValue("temporal_modeling/export_dir", out_dir)
        return out_dir

    def _export_prefix(self) -> str:
        model = "glm" if self.combo_model_type.currentIndex() == 0 else "flmm"
        scope = self._fit_mode or "all"
        fid = self._active_file_id if scope == "active" else scope
        return self._safe_filename(f"temporal_{model}_{fid}")

    @staticmethod
    def _write_dict_rows_csv(path: str, rows: List[Dict[str, Any]]) -> None:
        if not rows:
            with open(path, "w", newline="", encoding="utf-8") as fh:
                fh.write("")
            return
        fieldnames: List[str] = []
        for row in rows:
            for key in row.keys():
                if key not in fieldnames:
                    fieldnames.append(str(key))
        with open(path, "w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=fieldnames)
            writer.writeheader()
            for row in rows:
                writer.writerow({key: row.get(key, "") for key in fieldnames})

    def _export_glm_tables(self, out_dir: str, prefix: str) -> List[str]:
        result = self._glm_result
        if result is None:
            return []
        written: List[str] = []
        t = np.asarray(result.kernel_tvec, float)
        kernel_path = os.path.join(out_dir, f"{prefix}_glm_kernels.csv")
        with open(kernel_path, "w", newline="", encoding="utf-8") as fh:
            writer = csv.writer(fh)
            writer.writerow(["time_s"] + [self._predictor_label(name) for name in result.predictor_names])
            for idx in range(int(t.size)):
                row = [t[idx]]
                for name in result.predictor_names:
                    vals = np.asarray(result.kernels.get(name, np.array([], float)), float)
                    row.append(vals[idx] if idx < vals.size else "")
                writer.writerow(row)
        written.append(kernel_path)

        n = min(
            len(np.asarray(result.time)),
            len(np.asarray(result.y_actual)),
            len(np.asarray(result.y_pred)),
            len(np.asarray(result.residuals)),
        )
        pred_path = os.path.join(out_dir, f"{prefix}_glm_prediction.csv")
        with open(pred_path, "w", newline="", encoding="utf-8") as fh:
            writer = csv.writer(fh)
            writer.writerow(["time_s", "actual", "predicted", "residual"])
            for i in range(n):
                writer.writerow([result.time[i], result.y_actual[i], result.y_pred[i], result.residuals[i]])
        written.append(pred_path)

        imp_path = os.path.join(out_dir, f"{prefix}_glm_importance.csv")
        self._write_dict_rows_csv(imp_path, list(result.feature_importance or []))
        written.append(imp_path)
        return written

    def _export_flmm_tables(self, out_dir: str, prefix: str) -> List[str]:
        result = self._flmm_result
        if result is None:
            return []
        written: List[str] = []
        t = np.asarray(result.tvec, float)
        coeff_path = os.path.join(out_dir, f"{prefix}_flmm_coefficients.csv")
        with open(coeff_path, "w", newline="", encoding="utf-8") as fh:
            writer = csv.writer(fh)
            writer.writerow(["term", "time_s", "coefficient", "ci_lower", "ci_upper", "joint_ci_lower", "joint_ci_upper"])
            for term, coeff in (result.coefficients or {}).items():
                coeff = np.asarray(coeff, float)
                lo = np.asarray((result.ci_lower or {}).get(term, np.full_like(coeff, np.nan)), float)
                hi = np.asarray((result.ci_upper or {}).get(term, np.full_like(coeff, np.nan)), float)
                jlo = np.asarray((result.joint_ci_lower or {}).get(term, np.full_like(coeff, np.nan)), float)
                jhi = np.asarray((result.joint_ci_upper or {}).get(term, np.full_like(coeff, np.nan)), float)
                for i in range(min(t.size, coeff.size)):
                    writer.writerow([
                        term,
                        t[i],
                        coeff[i],
                        lo[i] if i < lo.size else "",
                        hi[i] if i < hi.size else "",
                        jlo[i] if i < jlo.size else "",
                        jhi[i] if i < jhi.size else "",
                    ])
        written.append(coeff_path)

        imp_path = os.path.join(out_dir, f"{prefix}_flmm_importance.csv")
        self._write_dict_rows_csv(imp_path, list(result.feature_importance or []))
        written.append(imp_path)

        mat = np.asarray(self._flmm_trace_mat, float) if self._flmm_trace_mat is not None else np.array([], float)
        tvec = np.asarray(self._flmm_trace_tvec, float) if self._flmm_trace_tvec is not None else np.array([], float)
        if mat.ndim == 2 and mat.shape[0] and tvec.size == mat.shape[1]:
            labels = self._flmm_trace_labels if len(self._flmm_trace_labels) == mat.shape[0] else [f"row_{i + 1}" for i in range(mat.shape[0])]
            trace_path = os.path.join(out_dir, f"{prefix}_flmm_input_traces.csv")
            with open(trace_path, "w", newline="", encoding="utf-8") as fh:
                writer = csv.writer(fh)
                writer.writerow(["row_index", "row_label", "time_s", "response"])
                for row_idx, label in enumerate(labels):
                    row = mat[row_idx, :]
                    for i in range(min(tvec.size, row.size)):
                        writer.writerow([row_idx + 1, label, tvec[i], row[i]])
            written.append(trace_path)
        return written

    def _export_temporal_results_to_dir(self, out_dir: str, prefix: str) -> List[str]:
        summary_path = os.path.join(out_dir, f"{prefix}_summary.txt")
        with open(summary_path, "w", encoding="utf-8") as fh:
            fh.write(self.txt_summary.toPlainText() if hasattr(self, "txt_summary") else "")

        state_path = os.path.join(out_dir, f"{prefix}_state.json")
        with open(state_path, "w", encoding="utf-8") as fh:
            json.dump(self.serialize_state(), fh, indent=2, allow_nan=True)

        written = [summary_path, state_path]
        written.extend(self._export_glm_tables(out_dir, prefix))
        written.extend(self._export_flmm_tables(out_dir, prefix))
        return written

    def _export_temporal_results(self) -> None:
        out_dir = self._export_dir_from_user("Export temporal modeling results")
        if not out_dir:
            return
        prefix = self._export_prefix()
        written = self._export_temporal_results_to_dir(out_dir, prefix)
        self.statusMessage.emit(f"Exported temporal results: {len(written)} file(s).", 5000)

    @staticmethod
    def _render_widget_png_pdf(widget: QtWidgets.QWidget, base_path: str) -> List[str]:
        written: List[str] = []
        if widget is None:
            return written
        png_path = f"{base_path}.png"
        pix = widget.grab()
        if not pix.isNull() and pix.save(png_path, "PNG"):
            written.append(png_path)
        pdf_path = f"{base_path}.pdf"
        size = widget.size()
        if size.width() > 0 and size.height() > 0:
            writer = QtGui.QPdfWriter(pdf_path)
            writer.setResolution(300)
            page_size = QtGui.QPageSize(
                QtCore.QSizeF(size.width() / 96.0 * 25.4, size.height() / 96.0 * 25.4),
                QtGui.QPageSize.Unit.Millimeter,
            )
            writer.setPageSize(page_size)
            painter = QtGui.QPainter(writer)
            try:
                widget.render(painter)
                written.append(pdf_path)
            finally:
                painter.end()
        return written

    def _temporal_plot_targets(self) -> List[Tuple[str, QtWidgets.QWidget]]:
        targets: List[Tuple[str, QtWidgets.QWidget]] = []
        if self._kernel_layout_mode() == "grid" and self._kernel_grid_plots:
            for idx, plot in enumerate(self._kernel_grid_plots, 1):
                title_label = getattr(plot.getPlotItem(), "titleLabel", None)
                title_text = getattr(title_label, "text", "") if title_label is not None else ""
                title = self._safe_filename(title_text or f"kernel_{idx}", f"kernel_{idx}")
                targets.append((f"kernel_{idx:02d}_{title}", plot))
        elif hasattr(self, "plot_kernel"):
            targets.append(("kernels", self.plot_kernel))
        for suffix, attr in (
            ("prediction", "plot_prediction"),
            ("flmm_traces", "plot_flmm_traces"),
            ("illustration", "plot_illustration"),
            ("residuals", "plot_residuals"),
            ("importance", "plot_importance"),
            ("flmm_coefficients", "plot_coeff"),
            ("group_kernels", "plot_group_kernels"),
            ("group_importance", "plot_group_importance"),
        ):
            widget = getattr(self, attr, None)
            if isinstance(widget, QtWidgets.QWidget):
                targets.append((suffix, widget))
        return targets

    def _export_temporal_plots_to_dir(self, out_dir: str, prefix: str) -> List[str]:
        written: List[str] = []
        for suffix, widget in self._temporal_plot_targets():
            base = os.path.join(out_dir, self._safe_filename(f"{prefix}_{suffix}"))
            try:
                written.extend(self._render_widget_png_pdf(widget, base))
            except Exception as exc:
                _LOG.warning("Temporal plot export failed for %s: %s", suffix, exc)
        return written

    def _export_temporal_plots(self) -> None:
        out_dir = self._export_dir_from_user("Export temporal modeling plots")
        if not out_dir:
            return
        prefix = self._export_prefix()
        written = self._export_temporal_plots_to_dir(out_dir, prefix)
        if written:
            self.statusMessage.emit(f"Exported temporal plots: {len(written)} file(s).", 5000)
        else:
            QtWidgets.QMessageBox.warning(self, "Export plots", "No temporal plots could be exported.")

    def _export_temporal_full_report(self) -> None:
        out_dir = self._export_dir_from_user("Export full temporal modeling report")
        if not out_dir:
            return
        prefix = self._export_prefix()
        written = []
        written.extend(self._export_temporal_results_to_dir(out_dir, prefix))
        written.extend(self._export_temporal_plots_to_dir(out_dir, prefix))
        if written:
            self.statusMessage.emit(f"Exported full temporal report: {len(written)} file(s).", 6000)
        else:
            QtWidgets.QMessageBox.warning(self, "Export report", "No temporal model outputs could be exported.")

    # ------------------------------------------------------------------
    # Predictor catalog and extraction
    # ------------------------------------------------------------------

    @staticmethod
    def _proc_file_id(proc: Any, fallback: str = "import") -> str:
        path = str(getattr(proc, "path", "") or "").strip()
        if not path:
            return fallback
        stem = os.path.splitext(os.path.basename(path))[0]
        return stem or fallback

    @staticmethod
    def _clean_id(value: object) -> str:
        text = str(value or "").strip().lower()
        text = re.sub(r"_ain0*[0-9]+$", "", text)
        text = re.sub(r"[^a-z0-9]+", "", text)
        return text

    def _ids_match(self, a: object, b: object) -> bool:
        aa = self._clean_id(a)
        bb = self._clean_id(b)
        return bool(aa and bb and aa == bb)

    def _predictor_label(self, key: str) -> str:
        entry = self._predictor_catalog.get(str(key), {})
        label = str(entry.get("label", "") or "").strip()
        if label:
            if label.startswith("Numeric column:"):
                return label.split(":", 1)[1].strip()
            return label
        if key == "events":
            return "PSTH alignment events"
        if key == "dio":
            return "DIO rising edges"
        if key.startswith("trigger::"):
            return f"Trigger: {key.split('::', 1)[1]}"
        if key.startswith("behavior_event::"):
            return f"Behavior onset: {key.split('::', 1)[1]}"
        if key.startswith("behavior_state::"):
            return f"Behavior state: {key.split('::', 1)[1]}"
        if key.startswith("behavior_cont::"):
            return key.split("::", 1)[1]
        return str(key)

    @staticmethod
    def _compact_feature_label(label: object, max_len: int = 42) -> str:
        text = str(label or "").strip()
        if text.startswith("Numeric column:"):
            text = text.split(":", 1)[1].strip()
        text = re.sub(r"\s+", " ", text)
        if len(text) > max_len:
            return text[:max_len - 3] + "..."
        return text

    @staticmethod
    def _kernel_color(key: object) -> str:
        palette = [
            "#4b9df8", "#f5a97f", "#6bdb74", "#ee99a0", "#c6a0f6",
            "#89dceb", "#f5c542", "#5fd0c5", "#ff7ab2", "#a6e3a1",
            "#fab387", "#74c7ec", "#b4befe", "#f38ba8", "#94e2d5",
            "#eba0ac", "#8bd5ca", "#eed49f", "#91d7e3", "#f5bde6",
        ]
        digest = hashlib.blake2s(str(key).encode("utf-8", errors="replace"), digest_size=2).digest()
        return palette[int.from_bytes(digest, "little") % len(palette)]

    def _sync_kernel_filter(self, result: GLMResult) -> None:
        if not hasattr(self, "list_kernel_filter"):
            return
        names = list(result.predictor_names or result.kernels.keys())
        self._kernel_filter_guard = True
        try:
            self.list_kernel_filter.clear()
            for name in names:
                visible = bool(self._kernel_visible.get(name, True))
                self._kernel_visible[name] = visible
                label = self._compact_feature_label(self._predictor_label(name), 34)
                item = QtWidgets.QListWidgetItem(label)
                item.setData(QtCore.Qt.ItemDataRole.UserRole, name)
                item.setFlags(item.flags() | QtCore.Qt.ItemFlag.ItemIsUserCheckable)
                item.setCheckState(QtCore.Qt.CheckState.Checked if visible else QtCore.Qt.CheckState.Unchecked)
                item.setForeground(QtGui.QColor(self._kernel_color(name)))
                self.list_kernel_filter.addItem(item)
        finally:
            self._kernel_filter_guard = False

    def _kernel_layout_mode(self) -> str:
        if hasattr(self, "combo_kernel_layout"):
            mode = self.combo_kernel_layout.currentData(QtCore.Qt.ItemDataRole.UserRole)
            if str(mode) == "grid":
                return "grid"
        return "overlay"

    def _selected_kernel_names(self, result: GLMResult) -> List[str]:
        names: List[str] = []
        for name in result.predictor_names:
            if not self._kernel_visible.get(name, True):
                continue
            if result.kernels.get(name) is not None:
                names.append(name)
        return names

    def _kernel_y_label(self, key: str) -> str:
        kind = str(self._predictor_catalog.get(str(key), {}).get("kind", "") or "")
        if kind == "continuous":
            return "Signal gain / +1 SD"
        return "Event kernel weight"

    def _kernel_title(self, key: str) -> str:
        label = self._compact_feature_label(self._predictor_label(key), 54)
        kind = str(self._predictor_catalog.get(str(key), {}).get("kind", "") or "")
        if kind == "continuous":
            return f"{label} (continuous, z-scored)"
        return f"{label} (event)"

    def _clear_kernel_grid(self) -> None:
        self._kernel_grid_plots = []
        layout = getattr(self, "kernel_grid_layout", None)
        if layout is None:
            return
        while layout.count():
            item = layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.setParent(None)
                widget.deleteLater()

    def _plot_single_kernel_panel(self, plot: pg.PlotWidget, result: GLMResult, name: str) -> None:
        kernel = result.kernels.get(name)
        plot.clear()
        try:
            plot.getPlotItem().legend.clear()
        except Exception:
            pass
        color = self._kernel_color(name)
        if kernel is not None:
            lo = (result.kernel_ci_lower or {}).get(name)
            hi = (result.kernel_ci_upper or {}).get(name)
            if lo is not None and hi is not None:
                lo_arr = np.asarray(lo, float)
                hi_arr = np.asarray(hi, float)
                if lo_arr.size == result.kernel_tvec.size and hi_arr.size == result.kernel_tvec.size:
                    qcol = QtGui.QColor(color)
                    qcol.setAlpha(42)
                    plot.addItem(pg.FillBetweenItem(
                        pg.PlotDataItem(result.kernel_tvec, lo_arr),
                        pg.PlotDataItem(result.kernel_tvec, hi_arr),
                        brush=qcol,
                    ))
            plot.plot(result.kernel_tvec, kernel, pen=pg.mkPen(color, width=2.0), name=self._predictor_label(name))
        plot.setTitle(self._kernel_title(name))
        plot.setLabel("bottom", "Time", units="s")
        plot.setLabel("left", self._kernel_y_label(name))
        plot.addLine(y=0, pen=pg.mkPen("#5a6274", width=1, style=QtCore.Qt.PenStyle.DashLine))
        plot.addLine(x=0, pen=pg.mkPen("#5a6274", width=1, style=QtCore.Qt.PenStyle.DashLine))

    def _plot_kernel_grid(self, result: GLMResult, names: List[str]) -> None:
        self._clear_kernel_grid()
        if hasattr(self, "stack_kernel_plots"):
            self.stack_kernel_plots.setCurrentIndex(1)
        layout = getattr(self, "kernel_grid_layout", None)
        if layout is None:
            return
        if not names:
            msg = QtWidgets.QLabel("No kernels selected.")
            msg.setProperty("class", "muted")
            msg.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
            layout.addWidget(msg, 0, 0)
            return
        cols = 2 if max(self.width(), self.sizeHint().width()) >= 1100 else 1
        for idx, name in enumerate(names):
            plot = pg.PlotWidget()
            self._style_plot(plot, min_height=230)
            self._plot_single_kernel_panel(plot, result, name)
            row = idx // cols
            col = idx % cols
            layout.addWidget(plot, row, col)
            self._kernel_grid_plots.append(plot)
        layout.setRowStretch((len(names) + cols - 1) // cols, 1)

    def _plot_kernel_overlay(self, result: GLMResult, names: List[str]) -> None:
        pw = self.plot_kernel
        if hasattr(self, "stack_kernel_plots"):
            self.stack_kernel_plots.setCurrentIndex(0)
        pw.clear()
        try:
            pw.getPlotItem().legend.clear()
        except Exception:
            pass
        for name in names:
            kernel = result.kernels.get(name)
            if kernel is None:
                continue
            color = self._kernel_color(name)
            lo = (result.kernel_ci_lower or {}).get(name)
            hi = (result.kernel_ci_upper or {}).get(name)
            if lo is not None and hi is not None:
                lo_arr = np.asarray(lo, float)
                hi_arr = np.asarray(hi, float)
                if lo_arr.size == result.kernel_tvec.size and hi_arr.size == result.kernel_tvec.size:
                    qcol = QtGui.QColor(color)
                    qcol.setAlpha(34)
                    pw.addItem(pg.FillBetweenItem(
                        pg.PlotDataItem(result.kernel_tvec, lo_arr),
                        pg.PlotDataItem(result.kernel_tvec, hi_arr),
                        brush=qcol,
                    ))
            pw.plot(result.kernel_tvec, kernel, pen=pg.mkPen(color, width=2), name=self._predictor_label(name))
        if not names:
            txt = pg.TextItem("No kernels selected.", color="#c5d2e3")
            pw.addItem(txt)
            txt.setPos(0, 0)
        pw.setTitle("Estimated kernels")
        pw.setLabel("bottom", "Time", units="s")
        pw.setLabel("left", "Kernel weight")
        pw.addLine(y=0, pen=pg.mkPen("#5a6274", width=1, style=QtCore.Qt.PenStyle.DashLine))
        pw.addLine(x=0, pen=pg.mkPen("#5a6274", width=1, style=QtCore.Qt.PenStyle.DashLine))

    def _on_kernel_layout_changed(self, *_args) -> None:
        self._save_temporal_settings()
        if self._glm_result is not None:
            self._plot_glm_kernels(self._glm_result, refresh_filter=False)

    def _on_kernel_filter_changed(self, item: QtWidgets.QListWidgetItem) -> None:
        if self._kernel_filter_guard:
            return
        key = item.data(QtCore.Qt.ItemDataRole.UserRole)
        if isinstance(key, str) and key:
            self._kernel_visible[key] = item.checkState() == QtCore.Qt.CheckState.Checked
        if self._glm_result is not None:
            self._plot_glm_kernels(self._glm_result, refresh_filter=False)

    def _set_all_kernels_visible(self, visible: bool) -> None:
        if not hasattr(self, "list_kernel_filter"):
            return
        self._kernel_filter_guard = True
        try:
            for i in range(self.list_kernel_filter.count()):
                item = self.list_kernel_filter.item(i)
                key = item.data(QtCore.Qt.ItemDataRole.UserRole)
                if isinstance(key, str) and key:
                    self._kernel_visible[key] = bool(visible)
                item.setCheckState(QtCore.Qt.CheckState.Checked if visible else QtCore.Qt.CheckState.Unchecked)
        finally:
            self._kernel_filter_guard = False
        if self._glm_result is not None:
            self._plot_glm_kernels(self._glm_result, refresh_filter=False)

    def _update_illustration_view(self) -> None:
        if not hasattr(self, "plot_illustration") or self._illustration_vb is None:
            return
        plot_item = self.plot_illustration.getPlotItem()
        self._illustration_vb.setGeometry(plot_item.vb.sceneBoundingRect())
        self._illustration_vb.linkedViewChanged(plot_item.vb, self._illustration_vb.XAxis)

    def _glm_feature_order(self, result: GLMResult) -> List[str]:
        ordered: List[str] = []
        for row in result.feature_importance or []:
            key = str(row.get("feature", "") or "")
            if key in result.predictor_names and key not in ordered:
                ordered.append(key)
        for key in result.predictor_names:
            if key not in ordered:
                ordered.append(key)
        return ordered

    def _sync_illustration_features(self, result: GLMResult) -> None:
        if not hasattr(self, "combo_illustration_feature"):
            return
        current = self.combo_illustration_feature.currentData(QtCore.Qt.ItemDataRole.UserRole)
        self.combo_illustration_feature.blockSignals(True)
        try:
            self.combo_illustration_feature.clear()
            for key in self._glm_feature_order(result):
                self.combo_illustration_feature.addItem(self._predictor_label(key), key)
            if isinstance(current, str) and current:
                idx = self.combo_illustration_feature.findData(current, QtCore.Qt.ItemDataRole.UserRole)
                if idx >= 0:
                    self.combo_illustration_feature.setCurrentIndex(idx)
        finally:
            self.combo_illustration_feature.blockSignals(False)

    @staticmethod
    def _pearson_stats(x: np.ndarray, y: np.ndarray) -> Tuple[float, float, int]:
        xv = np.asarray(x, float)
        yv = np.asarray(y, float)
        m = np.isfinite(xv) & np.isfinite(yv)
        xv = xv[m]
        yv = yv[m]
        if xv.size < 3 or np.nanstd(xv) <= 1e-12 or np.nanstd(yv) <= 1e-12:
            return float("nan"), float("nan"), int(xv.size)
        try:
            from scipy import stats as _stats
            res = _stats.pearsonr(xv, yv)
            return float(res.statistic), float(res.pvalue), int(xv.size)
        except Exception:
            return float(np.corrcoef(xv, yv)[0, 1]), float("nan"), int(xv.size)

    @staticmethod
    def _p_label(p_value: float) -> str:
        if not np.isfinite(p_value):
            return "p = n/a"
        if p_value < 1e-4:
            return "p < 1e-4"
        return f"p = {p_value:.3g}"

    @staticmethod
    def _p_stars(p_value: float) -> str:
        if not np.isfinite(p_value):
            return ""
        if p_value < 0.001:
            return "***"
        if p_value < 0.01:
            return "**"
        if p_value < 0.05:
            return "*"
        return "n.s."

    def _glm_feature_contribution(self, result: GLMResult, key: str) -> Optional[np.ndarray]:
        if key not in result.predictor_names:
            return None
        n_pred = len(result.predictor_names)
        if n_pred <= 0 or result.design_matrix is None or result.coefficients is None:
            return None
        n_basis = (int(np.asarray(result.coefficients).size) - 1) // n_pred
        if n_basis <= 0:
            return None
        pred_idx = result.predictor_names.index(key)
        lo = 1 + pred_idx * n_basis
        hi = lo + n_basis
        X = np.asarray(result.design_matrix, float)
        beta = np.asarray(result.coefficients, float)
        if X.ndim != 2 or hi > X.shape[1] or hi > beta.size:
            return None
        return X[:, lo:hi] @ beta[lo:hi]

    def _on_illustration_feature_changed(self, *_args) -> None:
        if self._glm_result is not None:
            self._plot_glm_illustration(self._glm_result)

    def _selected_predictor_keys(self) -> List[str]:
        keys: List[str] = []
        for i in range(self.list_predictors.count()):
            item = self.list_predictors.item(i)
            key = item.data(QtCore.Qt.ItemDataRole.UserRole)
            if not isinstance(key, str) or not key.strip():
                key = item.text().strip()
                for cat_key, entry in self._predictor_catalog.items():
                    if key == cat_key or key == str(entry.get("label", "")):
                        key = cat_key
                        break
            key = str(key).strip()
            if key and key not in keys:
                keys.append(key)
        return keys

    def _add_predictor_item(self, key: str) -> bool:
        key = str(key or "").strip()
        if not key:
            return False
        for existing in self._selected_predictor_keys():
            if existing == key:
                return False
        item = QtWidgets.QListWidgetItem(self._predictor_label(key))
        item.setData(QtCore.Qt.ItemDataRole.UserRole, key)
        self.list_predictors.addItem(item)
        return True

    def _remember_current_predictors_for_file(self, file_id: Optional[str] = None) -> None:
        if getattr(self, "_restoring_predictors", False) or not hasattr(self, "list_predictors"):
            return
        fid = str(file_id or self._active_file_id or "").strip()
        keys = self._selected_predictor_keys()
        self._saved_predictor_keys = list(keys)
        if fid:
            self._predictor_keys_by_file[fid] = list(keys)

    def _restore_predictors_for_active_file(self, allow_default: bool = True) -> None:
        if not hasattr(self, "list_predictors"):
            return
        fid = str(self._active_file_id or "").strip()
        keys: List[str] = []
        if fid and fid in self._predictor_keys_by_file:
            keys = [key for key in self._predictor_keys_by_file.get(fid, []) if key in self._predictor_catalog]
        if not keys:
            keys = [key for key in self._saved_predictor_keys if key in self._predictor_catalog]
        self._restoring_predictors = True
        try:
            self.list_predictors.clear()
            for key in keys:
                self._add_predictor_item(key)
            if allow_default and self.list_predictors.count() == 0 and "events" in self._predictor_catalog:
                self._add_predictor_item("events")
        finally:
            self._restoring_predictors = False

    def _refresh_predictor_combo(self) -> None:
        if not hasattr(self, "combo_available_predictors"):
            return
        selected = self.combo_available_predictors.currentData(QtCore.Qt.ItemDataRole.UserRole)
        self.combo_available_predictors.blockSignals(True)
        self.combo_available_predictors.clear()
        for key, entry in self._predictor_catalog.items():
            self.combo_available_predictors.addItem(str(entry.get("label", key)), key)
        if not self._predictor_catalog:
            self.combo_available_predictors.addItem("No predictors available yet", "")
        if isinstance(selected, str) and selected:
            idx = self.combo_available_predictors.findData(selected, QtCore.Qt.ItemDataRole.UserRole)
            if idx >= 0:
                self.combo_available_predictors.setCurrentIndex(idx)
        self.combo_available_predictors.blockSignals(False)

    def _refresh_predictor_catalog(self) -> None:
        catalog: Dict[str, Dict[str, Any]] = {}
        if self._event_times is not None and len(np.asarray(self._event_times, float)):
            catalog["events"] = {"kind": "event", "label": "PSTH alignment events"}
        if self._event_rows:
            catalog["events"] = {"kind": "event", "label": "PSTH alignment events"}

        trigger_names: set[str] = set()
        has_dio = False
        for proc in self._processed_trials:
            if getattr(proc, "dio", None) is not None:
                has_dio = True
            triggers = getattr(proc, "triggers", None) or {}
            if isinstance(triggers, dict):
                trigger_names.update(str(k) for k in triggers.keys() if str(k).strip())
        if has_dio:
            catalog["dio"] = {"kind": "event", "label": "DIO rising edges"}
        for name in sorted(trigger_names):
            catalog[f"trigger::{name}"] = {"kind": "event", "name": name, "label": f"Trigger: {name}"}

        behavior_names: set[str] = set()
        state_names: set[str] = set()
        continuous_names: set[str] = set()
        for info in self._behavior_sources.values():
            if not isinstance(info, dict):
                continue
            kind = str(info.get("kind", _BEHAVIOR_PARSE_BINARY))
            behaviors = info.get("behaviors") or {}
            for name in behaviors.keys():
                clean = str(name).strip()
                if not clean:
                    continue
                behavior_names.add(clean)
                if kind != _BEHAVIOR_PARSE_TIMESTAMPS:
                    state_names.add(clean)
            trajectory = info.get("trajectory") or {}
            for name in trajectory.keys():
                clean = str(name).strip()
                if clean:
                    continuous_names.add(clean)
        for name in sorted(behavior_names):
            catalog[f"behavior_event::{name}"] = {
                "kind": "behavior_event",
                "name": name,
                "label": f"Behavior onset: {name}",
            }
        for name in sorted(state_names):
            catalog[f"behavior_state::{name}"] = {
                "kind": "continuous",
                "name": name,
                "label": f"Behavior state: {name}",
            }
        for name in sorted(continuous_names):
            catalog[f"behavior_cont::{name}"] = {
                "kind": "continuous",
                "name": name,
                "label": name,
            }

        previous_keys = self._selected_predictor_keys() if hasattr(self, "list_predictors") else []
        self._predictor_catalog = catalog
        self._refresh_predictor_combo()
        if hasattr(self, "list_predictors"):
            if str(self._active_file_id or "") in self._predictor_keys_by_file:
                self._restore_predictors_for_active_file(allow_default=True)
            else:
                restore_keys = previous_keys or [key for key in self._saved_predictor_keys if key in catalog]
                self._restoring_predictors = True
                try:
                    self.list_predictors.clear()
                    for key in restore_keys:
                        if key in catalog:
                            self._add_predictor_item(key)
                    if self.list_predictors.count() == 0 and "events" in catalog:
                        self._add_predictor_item("events")
                finally:
                    self._restoring_predictors = False
            if self.list_predictors.count() > 0 or previous_keys:
                self._save_temporal_settings()

    def _behavior_source_for_proc(self, proc: Any) -> Optional[Dict[str, Any]]:
        if not self._behavior_sources:
            return None
        file_id = self._proc_file_id(proc)
        info = self._behavior_sources.get(file_id)
        if info is not None:
            return info
        for key, val in self._behavior_sources.items():
            if self._ids_match(key, file_id):
                return val
        try:
            idx = next(i for i, p in enumerate(self._processed_trials) if (p is proc) or (getattr(p, "path", "") == getattr(proc, "path", "")))
        except StopIteration:
            idx = None
        if idx is not None:
            keys = list(self._behavior_sources.keys())
            if 0 <= idx < len(keys):
                return self._behavior_sources.get(keys[idx])
        if len(self._behavior_sources) == 1:
            return next(iter(self._behavior_sources.values()))
        return None

    @staticmethod
    def _event_vector(time: np.ndarray, events: np.ndarray) -> np.ndarray:
        t = np.asarray(time, float)
        out = np.zeros(t.size, float)
        ev = np.asarray(events, float)
        ev = ev[np.isfinite(ev)]
        if t.size == 0 or ev.size == 0:
            return out
        idx = np.searchsorted(t, ev)
        idx = idx[(idx >= 0) & (idx < t.size)]
        for i in idx:
            out[int(i)] += 1.0
        return out

    def _events_for_proc(self, proc: Any, time: np.ndarray) -> np.ndarray:
        file_id = self._proc_file_id(proc)
        if self._event_rows:
            vals = []
            for row in self._event_rows:
                if self._ids_match(row.get("file_id", ""), file_id):
                    try:
                        vals.append(float(row.get("event_time_sec", np.nan)))
                    except Exception:
                        pass
            return np.asarray(vals, float)
        if len(self._processed_trials) == 1 and self._event_times is not None:
            return np.asarray(self._event_times, float)
        return np.array([], float)

    @staticmethod
    def _rising_edges_from_signal(time: np.ndarray, signal: np.ndarray) -> np.ndarray:
        t = np.asarray(time, float)
        x = np.asarray(signal, float)
        if t.size < 2 or x.size != t.size:
            return np.array([], float)
        b = x > 0.5
        idx = np.where((~b[:-1]) & (b[1:]))[0] + 1
        return t[idx]

    @staticmethod
    def _behavior_onsets(info: Dict[str, Any], name: str) -> np.ndarray:
        behaviors = info.get("behaviors") or {}
        if name not in behaviors:
            return np.array([], float)
        kind = str(info.get("kind", _BEHAVIOR_PARSE_BINARY))
        if kind == _BEHAVIOR_PARSE_TIMESTAMPS:
            ev = np.asarray(behaviors[name], float)
            ev = ev[np.isfinite(ev)]
            return np.sort(np.unique(ev))
        t = np.asarray(info.get("time", np.array([], float)), float)
        x = np.asarray(behaviors[name], float)
        if t.size < 1 or x.size != t.size:
            return np.array([], float)
        b = x > 0.5
        on = np.where((~b[:-1]) & (b[1:]))[0] + 1 if b.size > 1 else np.array([], int)
        if b.size and bool(b[0]):
            on = np.concatenate([[0], on])
        return t[on]

    @staticmethod
    def _interp_to_time(target_time: np.ndarray, source_time: np.ndarray, values: np.ndarray) -> np.ndarray:
        target_time = np.asarray(target_time, float)
        source_time = np.asarray(source_time, float)
        values = np.asarray(values, float)
        out = np.zeros(target_time.size, float)
        if source_time.size != values.size:
            n = min(source_time.size, values.size)
            source_time = source_time[:n]
            values = values[:n]
        m = np.isfinite(source_time) & np.isfinite(values)
        source_time = source_time[m]
        values = values[m]
        if source_time.size == target_time.size and np.allclose(source_time, target_time, equal_nan=False):
            out = values.astype(float, copy=True)
            out[~np.isfinite(out)] = 0.0
            return out
        if source_time.size < 2:
            if values.size == target_time.size:
                out = values.astype(float, copy=True)
                out[~np.isfinite(out)] = 0.0
            return out
        order = np.argsort(source_time)
        source_time = source_time[order]
        values = values[order]
        keep = np.concatenate([[True], np.diff(source_time) > 0])
        source_time = source_time[keep]
        values = values[keep]
        if source_time.size < 2:
            return out
        interp = np.interp(target_time, source_time, values, left=np.nan, right=np.nan)
        interp[~np.isfinite(interp)] = 0.0
        return interp

    def _predictor_vector_for_proc(self, key: str, proc: Any, time: np.ndarray) -> Tuple[np.ndarray, str]:
        entry = self._predictor_catalog.get(key, {})
        if key == "events":
            return self._event_vector(time, self._events_for_proc(proc, time)), "event"
        if key == "dio":
            dio = getattr(proc, "dio", None)
            if dio is None:
                return np.zeros(time.size, float), "event"
            return self._event_vector(time, self._rising_edges_from_signal(time, dio)), "event"
        if key.startswith("trigger::"):
            name = key.split("::", 1)[1]
            triggers = getattr(proc, "triggers", None) or {}
            sig = triggers.get(name) if isinstance(triggers, dict) else None
            if sig is None:
                return np.zeros(time.size, float), "event"
            return self._event_vector(time, self._rising_edges_from_signal(time, sig)), "event"

        info = self._behavior_source_for_proc(proc)
        if not isinstance(info, dict):
            return np.zeros(time.size, float), str(entry.get("kind", "event"))
        name = str(entry.get("name", "") or key.split("::")[-1])
        if key.startswith("behavior_event::"):
            return self._event_vector(time, self._behavior_onsets(info, name)), "event"
        if key.startswith("behavior_state::"):
            behaviors = info.get("behaviors") or {}
            values = behaviors.get(name)
            source_time = np.asarray(info.get("time", np.array([], float)), float)
            if values is None or source_time.size == 0:
                return np.zeros(time.size, float), "continuous"
            return self._interp_to_time(time, source_time, values), "continuous"
        if key.startswith("behavior_cont::"):
            trajectory = info.get("trajectory") or {}
            values = trajectory.get(name)
            source_time = np.asarray(info.get("trajectory_time", np.array([], float)), float)
            if values is None or source_time.size == 0:
                return np.zeros(time.size, float), "continuous"
            return self._interp_to_time(time, source_time, values), "continuous"
        return np.zeros(time.size, float), str(entry.get("kind", "event"))

    def _build_glm_dataset_from_selected_predictors(
        self, file_filter: Optional[str] = None
    ) -> Dict[str, Any]:
        selected = self._selected_predictor_keys()
        if not selected:
            return {"error": "Choose at least one predictor before fitting."}
        if not self._processed_trials:
            return {"error": "No processed recordings are loaded."}

        kernel_span = abs(float(self.spin_kernel_pre.value())) + abs(float(self.spin_kernel_post.value()))
        segments: List[Tuple[str, np.ndarray, np.ndarray, Any]] = []
        dropped_records: List[str] = []
        for idx, proc in enumerate(self._processed_trials):
            t = np.asarray(getattr(proc, "time", np.array([], float)), float)
            y_raw = getattr(proc, "output", None)
            y = np.asarray(y_raw, float) if y_raw is not None else np.array([], float)
            file_id = self._proc_file_id(proc, fallback=f"file_{idx + 1}")
            if file_filter is not None and not self._ids_match(file_id, file_filter) and file_id != file_filter:
                continue
            if t.size < 3 or y.size != t.size:
                dropped_records.append(file_id)
                continue
            m = np.isfinite(t)
            t = t[m]
            y = y[m]
            if t.size < 3:
                dropped_records.append(file_id)
                continue
            order = np.argsort(t)
            t = t[order]
            y = y[order]
            keep = np.concatenate([[True], np.diff(t) > 0])
            t = t[keep]
            y = y[keep]
            if t.size < 3:
                dropped_records.append(file_id)
                continue
            segments.append((file_id, t, y, proc))

        if not segments:
            return {"error": "No recordings have usable time and output traces."}

        time_parts: List[np.ndarray] = []
        signal_parts: List[np.ndarray] = []
        vec_parts: Dict[str, List[np.ndarray]] = {key: [] for key in selected}
        pred_types: Dict[str, str] = {}
        used_records: List[str] = []
        segment_slices: List[Tuple[int, int]] = []
        offset = 0.0
        cursor = 0
        for seg_idx, (file_id, t, y, proc) in enumerate(segments):
            dt = float(np.nanmedian(np.diff(t)))
            if not np.isfinite(dt) or dt <= 0:
                dropped_records.append(file_id)
                continue
            t_shift = (t - float(t[0])) + offset
            start = cursor
            time_parts.append(t_shift)
            signal_parts.append(y.astype(float, copy=True))
            used_records.append(file_id)
            cursor += int(t.size)
            segment_slices.append((start, cursor))
            for key in selected:
                vec, ptype = self._predictor_vector_for_proc(key, proc, t)
                vec = np.asarray(vec, float)
                if vec.size != t.size:
                    vec = np.zeros(t.size, float)
                vec[~np.isfinite(vec)] = 0.0
                vec_parts[key].append(vec)
                pred_types[key] = ptype

            if seg_idx < len(segments) - 1:
                pad_n = max(2, int(np.ceil((kernel_span + dt) / dt)))
                pad_t = t_shift[-1] + dt * np.arange(1, pad_n + 1, dtype=float)
                time_parts.append(pad_t)
                signal_parts.append(np.full(pad_n, np.nan, float))
                for key in selected:
                    vec_parts[key].append(np.zeros(pad_n, float))
                cursor += int(pad_n)
                offset = float(pad_t[-1] + dt)

        if not time_parts:
            return {"error": "No recordings could be aligned for GLM fitting."}
        time = np.concatenate(time_parts)
        signal = np.concatenate(signal_parts)
        valid_signal = np.isfinite(signal)
        predictors: Dict[str, Dict[str, Any]] = {}
        dropped_predictors: List[str] = []
        for key in selected:
            vec = np.concatenate(vec_parts.get(key, [np.zeros(time.size, float)]))
            vec = vec.astype(float, copy=True)
            vec[~np.isfinite(vec)] = 0.0
            if pred_types.get(key) == "continuous":
                finite = valid_signal & np.isfinite(vec)
                vals = vec[finite]
                if vals.size:
                    mean = float(np.nanmean(vals))
                    std = float(np.nanstd(vals))
                    if np.isfinite(std) and std > 1e-12:
                        vec[finite] = (vec[finite] - mean) / std
                    else:
                        vec[finite] = vec[finite] - mean
                vec[~valid_signal] = 0.0
            if not np.any(np.abs(vec[valid_signal]) > 1e-12):
                dropped_predictors.append(self._predictor_label(key))
                continue
            predictors[key] = {"kind": "vector", "values": vec}

        if not predictors:
            return {
                "error": "The selected predictors contain no usable events or variation for the loaded recordings.",
                "dropped_predictors": dropped_predictors,
            }
        return {
            "time": time,
            "signal": signal,
            "predictors": predictors,
            "used_records": used_records,
            "dropped_records": dropped_records,
            "dropped_predictors": dropped_predictors,
            "valid_samples": int(np.sum(valid_signal)),
            "segment_slices": segment_slices,
        }

    def _proc_for_file_id(self, file_id: str) -> Optional[Any]:
        for proc in self._processed_trials:
            if self._ids_match(self._proc_file_id(proc), file_id):
                return proc
        try:
            idx = self._group_labels.index(file_id)
        except ValueError:
            idx = -1
        if 0 <= idx < len(self._processed_trials):
            return self._processed_trials[idx]
        return None

    @staticmethod
    def _safe_design_name(label: str, used: set[str]) -> str:
        base = re.sub(r"[^0-9A-Za-z_]+", "_", str(label or "").strip())
        base = re.sub(r"_+", "_", base).strip("_") or "predictor"
        if base[0].isdigit():
            base = f"pred_{base}"
        name = base
        i = 2
        while name in used:
            name = f"{base}_{i}"
            i += 1
        used.add(name)
        return name

    @staticmethod
    def _flmm_term_name(index: int, used: set[str]) -> str:
        base = f"pyber_x{max(1, int(index)):06d}z"
        name = base
        i = 2
        while name in used:
            name = f"{base}_{i}"
            i += 1
        used.add(name)
        return name

    def _flmm_matrix_and_labels(self) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], List[str], str]:
        self._flmm_row_meta = []
        if self._per_file_mats:
            mats: List[np.ndarray] = []
            labels: List[str] = []
            meta_rows: List[Dict[str, Any]] = []
            ref_tvec: Optional[np.ndarray] = None
            event_by_file: Dict[str, List[Dict[str, object]]] = {}
            for row in self._event_rows:
                fid = str(row.get("file_id", "") or "")
                if fid:
                    event_by_file.setdefault(fid, []).append(row)
            ordered_ids = list(self._file_ids) if self._file_ids else list(self._per_file_mats.keys())
            for file_id in ordered_ids:
                if file_id not in self._per_file_mats:
                    continue
                tvec_f, mat_f = self._per_file_mats[file_id]
                tvec_f = np.asarray(tvec_f, float)
                mat_f = np.asarray(mat_f, float)
                if mat_f.ndim != 2 or mat_f.shape[0] == 0 or tvec_f.size != mat_f.shape[1]:
                    continue
                if ref_tvec is None:
                    ref_tvec = tvec_f
                elif ref_tvec.size != tvec_f.size or not np.allclose(ref_tvec, tvec_f, equal_nan=True):
                    continue
                mats.append(mat_f)
                rows = event_by_file.get(file_id, [])
                for row_idx in range(mat_f.shape[0]):
                    labels.append(file_id)
                    ev_row = rows[row_idx] if row_idx < len(rows) else {}
                    meta_rows.append({
                        "file_id": file_id,
                        "trial_index": row_idx,
                        "event_time_sec": ev_row.get("event_time_sec", np.nan),
                        "duration_sec": ev_row.get("duration_sec", np.nan),
                    })
            if mats and ref_tvec is not None:
                mat = np.vstack(mats)
                if mat.shape[0] >= 2 and len(labels) == mat.shape[0]:
                    self._flmm_row_meta = meta_rows
                    return mat, ref_tvec, labels, "trial"

        group_mat = np.asarray(self._group_mat, float) if self._group_mat is not None else None
        group_tvec = np.asarray(self._group_tvec, float) if self._group_tvec is not None else None
        if (
            group_mat is not None
            and group_tvec is not None
            and group_mat.ndim == 2
            and group_mat.shape[0] >= 2
            and len(self._group_labels) == group_mat.shape[0]
        ):
            self._flmm_row_meta = [{"file_id": label, "trial_index": i} for i, label in enumerate(self._group_labels)]
            return group_mat, group_tvec, list(self._group_labels), "animal"

        mat = np.asarray(self._psth_mat, float) if self._psth_mat is not None else None
        tvec = np.asarray(self._psth_tvec, float) if self._psth_tvec is not None else None
        if mat is None or tvec is None or mat.ndim != 2:
            return None, None, [], "none"
        if len(self._file_ids) == mat.shape[0]:
            labels = list(self._file_ids)
            scope = "trial"
        else:
            labels = [f"trial_{i + 1}" for i in range(mat.shape[0])]
            scope = "trial"
        self._flmm_row_meta = [{"file_id": labels[i] if i < len(labels) else "", "trial_index": i} for i in range(mat.shape[0])]
        return mat, tvec, labels, scope

    def _animal_covariate_value(self, key: str, file_id: str) -> float:
        proc = self._proc_for_file_id(file_id)
        if proc is None:
            return np.nan
        t = np.asarray(getattr(proc, "time", np.array([], float)), float)
        if t.size < 2:
            return np.nan

        if key in {"events", "dio"} or key.startswith("trigger::") or key.startswith("behavior_event::"):
            vec, _ = self._predictor_vector_for_proc(key, proc, t)
            return float(np.nansum(np.asarray(vec, float)))

        entry = self._predictor_catalog.get(key, {})
        name = str(entry.get("name", "") or key.split("::")[-1])
        info = self._behavior_source_for_proc(proc)
        if not isinstance(info, dict):
            return np.nan
        if key.startswith("behavior_state::"):
            behaviors = info.get("behaviors") or {}
            values = np.asarray(behaviors.get(name, np.array([], float)), float)
            finite = values[np.isfinite(values)]
            if finite.size == 0:
                return np.nan
            return float(np.nanmean(finite > 0.5))
        if key.startswith("behavior_cont::"):
            trajectory = info.get("trajectory") or {}
            values = np.asarray(trajectory.get(name, np.array([], float)), float)
            finite = values[np.isfinite(values)]
            if finite.size == 0:
                return np.nan
            return float(np.nanmean(finite))
        return np.nan

    def _trial_covariate_value(self, key: str, meta: Dict[str, Any]) -> float:
        file_id = str(meta.get("file_id", "") or "")
        event_time = meta.get("event_time_sec", np.nan)
        try:
            event_time = float(event_time)
        except (TypeError, ValueError):
            event_time = np.nan
        proc = self._proc_for_file_id(file_id)
        if proc is None or not np.isfinite(event_time):
            return self._animal_covariate_value(key, file_id)
        t = np.asarray(getattr(proc, "time", np.array([], float)), float)
        if t.size < 2:
            return self._animal_covariate_value(key, file_id)

        if key in {"events", "dio"} or key.startswith("trigger::") or key.startswith("behavior_event::"):
            vec, _ = self._predictor_vector_for_proc(key, proc, t)
            edges = t[np.asarray(vec, float) > 0.5]
            if edges.size == 0:
                return 0.0
            dt = float(np.nanmedian(np.diff(np.sort(t))))
            tol = max(dt if np.isfinite(dt) and dt > 0 else 1e-3, 1e-3)
            return float(np.any(np.abs(edges - event_time) <= tol))

        entry = self._predictor_catalog.get(key, {})
        name = str(entry.get("name", "") or key.split("::")[-1])
        info = self._behavior_source_for_proc(proc)
        if not isinstance(info, dict):
            return self._animal_covariate_value(key, file_id)
        if key.startswith("behavior_state::"):
            behaviors = info.get("behaviors") or {}
            values = behaviors.get(name)
            source_time = np.asarray(info.get("time", np.array([], float)), float)
            if values is None or source_time.size == 0:
                return np.nan
            return float(self._interp_to_time(np.array([event_time], float), source_time, values)[0])
        if key.startswith("behavior_cont::"):
            trajectory = info.get("trajectory") or {}
            values = trajectory.get(name)
            source_time = np.asarray(info.get("trajectory_time", np.array([], float)), float)
            if values is None or source_time.size == 0:
                return np.nan
            return float(self._interp_to_time(np.array([event_time], float), source_time, values)[0])
        return self._animal_covariate_value(key, file_id)

    def _build_flmm_design(
        self,
        labels: List[str],
        group_var: str,
    ) -> Tuple[Dict[str, np.ndarray], List[str], List[str], Dict[str, str]]:
        design: Dict[str, np.ndarray] = {group_var: np.asarray(labels, dtype=object)}
        terms: List[str] = []
        dropped: List[str] = []
        term_labels: Dict[str, str] = {}
        used_names = {group_var, "Y.obs", ".index", ".obs"}

        for key in self._selected_predictor_keys():
            if len(self._flmm_row_meta) == len(labels):
                values = np.asarray([
                    self._trial_covariate_value(key, self._flmm_row_meta[i])
                    for i, _label in enumerate(labels)
                ], float)
            else:
                values = np.asarray([self._animal_covariate_value(key, label) for label in labels], float)
            finite = np.isfinite(values)
            if np.sum(finite) < 2:
                dropped.append(self._predictor_label(key))
                continue
            mean = float(np.nanmean(values[finite]))
            std = float(np.nanstd(values[finite]))
            if not np.isfinite(std) or std <= 1e-12:
                dropped.append(self._predictor_label(key))
                continue
            values[~finite] = mean
            values = (values - mean) / std
            # fastFMM detects functional covariates by shared name prefixes.
            # Human labels such as "Distance" and "Distance to point" collide,
            # so use neutral non-prefixing IDs and keep labels separately.
            term = self._flmm_term_name(len(terms) + 1, used_names)
            design[term] = values
            terms.append(term)
            term_labels[term] = self._predictor_label(key)
        return design, terms, dropped, term_labels

    def _prune_flmm_terms(
        self,
        design: Dict[str, np.ndarray],
        terms: List[str],
        term_labels: Dict[str, str],
        n_rows: int,
    ) -> Tuple[List[str], List[str]]:
        if not terms:
            return [], []
        max_terms = max(0, min(len(terms), int(n_rows) - 2))
        if max_terms <= 0:
            return [], [f"{term_labels.get(term, term)} (not enough rows)" for term in terms]

        X = np.ones((int(n_rows), 1), float)
        rank = int(np.linalg.matrix_rank(X))
        kept: List[str] = []
        dropped: List[str] = []
        for term in terms:
            if len(kept) >= max_terms:
                dropped.append(f"{term_labels.get(term, term)} (too many predictors for {n_rows} rows)")
                continue
            col = np.asarray(design.get(term, np.array([], float)), float).reshape(-1)
            if col.size != n_rows or not np.all(np.isfinite(col)):
                dropped.append(f"{term_labels.get(term, term)} (invalid values)")
                continue
            candidate = np.column_stack([X, col])
            new_rank = int(np.linalg.matrix_rank(candidate))
            if new_rank > rank:
                kept.append(term)
                X = candidate
                rank = new_rank
            else:
                dropped.append(f"{term_labels.get(term, term)} (collinear with existing predictors)")
        return kept, dropped

    @staticmethod
    def _intercept_only_fit_stats(signal: np.ndarray) -> Dict[str, float]:
        signal = np.asarray(signal, float)
        valid = np.isfinite(signal)
        if not np.any(valid):
            return {"r2": float("nan"), "mse": float("nan")}
        y = signal[valid]
        mean = float(np.nanmean(y))
        residuals = y - mean
        ss_res = float(np.nansum(residuals ** 2))
        ss_tot = float(np.nansum((y - np.nanmean(y)) ** 2))
        mse = float(np.nanmean(residuals ** 2)) if y.size else float("nan")
        return {"r2": 1.0 - ss_res / max(ss_tot, 1e-12), "mse": mse}

    @staticmethod
    def _shift_vector_by_segment(
        values: np.ndarray,
        segment_slices: List[Tuple[int, int]],
        rng: np.random.Generator,
    ) -> np.ndarray:
        vec = np.asarray(values, float)
        shifted = np.zeros_like(vec)
        slices = segment_slices or [(0, int(vec.size))]
        for lo, hi in slices:
            lo = max(0, int(lo))
            hi = min(int(hi), int(vec.size))
            n = hi - lo
            if n <= 1:
                continue
            shift = int(rng.integers(1, n))
            shifted[lo:hi] = np.roll(vec[lo:hi], shift)
        shifted[~np.isfinite(shifted)] = 0.0
        return shifted

    def _compute_glm_shift_bootstrap_significance(
        self,
        dataset: Dict[str, Any],
        rows: List[Dict[str, Any]],
        kernel_window: Tuple[float, float],
        basis_type: str,
        regularization: str,
        alpha: float,
        n_boot: int,
    ) -> None:
        if n_boot <= 0 or not rows:
            for row in rows:
                row["p_value"] = float("nan")
                row["q_value"] = float("nan")
                row["significant"] = False
                row["bootstrap_n"] = 0
            return

        predictors = dict(dataset.get("predictors", {}) or {})
        time = np.asarray(dataset["time"], float)
        signal = np.asarray(dataset["signal"], float)
        segment_slices = list(dataset.get("segment_slices", []) or [])
        rng = np.random.default_rng()
        total = max(1, int(n_boot) * len(rows))
        max_jobs = max(1, min(int(getattr(self, "spin_glm_jobs", None).value() if hasattr(self, "spin_glm_jobs") else 1), os.cpu_count() or 1))
        n_basis = int(self.spin_n_basis.value())
        label = f"GLM circular-shift test ({max_jobs} job{'s' if max_jobs != 1 else ''})"
        self._progress_start(label, total)
        step = 0
        work_items: List[Tuple[int, str, np.ndarray, float, int]] = []

        # Pre-compute the basis matrix and per-predictor design columns ONCE,
        # so each bootstrap only rebuilds the columns of the shifted predictor.
        try:
            _, pre_samp, _, kernel_tvec = _glm_kernel_geometry(time, kernel_window)
        except Exception:
            pre_samp = 0
            kernel_tvec = np.arange(2, dtype=float)
        kernel_len = int(kernel_tvec.size)
        basis_mat = _glm_basis_matrix(n_basis, kernel_len, basis_type)
        segment_lengths = [max(0, int(hi) - int(lo)) for lo, hi in (segment_slices or [(0, int(time.size))])]
        length_warning = ""
        if segment_lengths and min(segment_lengths) < 10 * max(kernel_len, 1):
            length_warning = (
                f"Recording/block shorter than 10x kernel length "
                f"({min(segment_lengths)} samples vs {10 * kernel_len}); circular-shift p-values are approximate."
            )
        count_warning = ""
        if int(n_boot) < 200:
            count_warning = "Use >=200 circular-shift bootstraps for stable q-values."

        T_full = int(time.size)

        def _columns_for_vector(input_vec: np.ndarray) -> Optional[np.ndarray]:
            v = np.asarray(input_vec, float)
            if v.size != T_full or not np.any(np.abs(v) > 1e-12):
                return None
            return _glm_convolved_columns(v, basis_mat, pre_samp, T_full)

        # Cache columns for every predictor in its un-shifted form.
        static_cols_cache: Dict[str, np.ndarray] = {}
        for name, spec in predictors.items():
            v = ContinuousGLM._predictor_vector(time, spec)
            cols = _columns_for_vector(v)
            if cols is not None:
                static_cols_cache[name] = cols
        used_predictors = list(static_cols_cache.keys())
        intercept_col = np.ones((T_full, 1), float)
        valid_mask = np.isfinite(signal)
        signal_v = signal[valid_mask]
        ss_tot_full = float(np.nansum((signal - np.nanmean(signal)) ** 2))
        row_meta: Dict[int, Tuple[Dict[str, Any], float]] = {}
        for row in rows:
            warnings = [text for text in (count_warning, length_warning) if text]
            if warnings:
                row["bootstrap_warning"] = " ".join(warnings)
            feature = str(row.get("feature", ""))
            obs_delta = float(row.get("delta_r2", float("nan")))
            reduced_r2 = float(row.get("reduced_r2", float("nan")))
            spec = predictors.get(feature)
            if (
                not feature
                or spec is None
                or not np.isfinite(obs_delta)
                or obs_delta <= 0
                or not np.isfinite(reduced_r2)
            ):
                row["p_value"] = 1.0
                row["q_value"] = 1.0
                row["significant"] = False
                row["bootstrap_n"] = 0
                step += int(n_boot)
                self._progress_update(step, label)
                continue

            base_vec = ContinuousGLM._predictor_vector(time, spec)
            row_idx = id(row)
            row_meta[row_idx] = (row, obs_delta)
            for seed in rng.integers(0, np.iinfo(np.uint32).max, size=int(n_boot), dtype=np.uint32):
                work_items.append((row_idx, feature, base_vec, reduced_r2, int(seed)))

        def _one_shift_fit(job: Tuple[int, str, np.ndarray, float, int]) -> Tuple[int, float]:
            """
            Fast circular-shift refit. Only the shifted predictor's columns
            are recomputed; columns for the other predictors come from the
            shared static cache. Falls back to the full GLM path if the cache
            for this feature is unavailable.
            """
            row_idx, feature, base_vec, reduced_r2, seed = job
            local_rng = np.random.default_rng(seed)
            shifted_vec = self._shift_vector_by_segment(base_vec, segment_slices, local_rng)
            if feature not in static_cols_cache or regularization not in ("ridge", "ols"):
                shifted_predictors = dict(predictors)
                shifted_predictors[feature] = {"kind": "vector", "values": shifted_vec}
                shifted = ContinuousGLM().fit(
                    time, signal, shifted_predictors,
                    kernel_window=kernel_window, n_basis=n_basis,
                    basis_type=basis_type, regularization=regularization,
                    alpha=alpha,
                )
                return row_idx, float(shifted.r2 - reduced_r2)

            shifted_cols = _columns_for_vector(shifted_vec)
            if shifted_cols is None:
                return row_idx, float("nan")
            blocks = [intercept_col]
            for name in used_predictors:
                if name == feature:
                    blocks.append(shifted_cols)
                else:
                    blocks.append(static_cols_cache[name])
            X = np.hstack(blocks)
            Xv = X[valid_mask]
            try:
                beta = ContinuousGLM.fit_coefficients(
                    Xv, signal_v, regularization=regularization, alpha=alpha
                )
            except np.linalg.LinAlgError:
                return row_idx, float("nan")
            y_pred = X @ beta
            residuals = signal - y_pred
            ss_res = float(np.nansum(residuals ** 2))
            r2 = 1.0 - ss_res / max(ss_tot_full, 1e-12)
            return row_idx, float(r2 - reduced_r2)

        null_by_row: Dict[int, List[float]] = {row_idx: [] for row_idx in row_meta}
        if work_items:
            if max_jobs == 1:
                for job in work_items:
                    try:
                        row_idx, null_delta = _one_shift_fit(job)
                        if np.isfinite(null_delta):
                            null_by_row[row_idx].append(null_delta)
                    except Exception as exc:
                        _LOG.debug("Circular-shift GLM bootstrap failed: %s", exc)
                    step += 1
                    if step == total or step % 5 == 0:
                        self._progress_update(step, label)
            else:
                with concurrent.futures.ThreadPoolExecutor(max_workers=max_jobs) as executor:
                    futures = [executor.submit(_one_shift_fit, job) for job in work_items]
                    for future in concurrent.futures.as_completed(futures):
                        try:
                            row_idx, null_delta = future.result()
                            if np.isfinite(null_delta):
                                null_by_row[row_idx].append(null_delta)
                        except Exception as exc:
                            _LOG.debug("Circular-shift GLM bootstrap failed: %s", exc)
                        step += 1
                        if step == total or step % 5 == 0:
                            self._progress_update(step, label)

        for row_idx, (row, obs_delta) in row_meta.items():
            null_arr = np.asarray(null_by_row.get(row_idx, []), float)
            if null_arr.size:
                exceed = int(np.sum(null_arr >= obs_delta))
                p_value = float((1 + exceed) / (null_arr.size + 1))
                row["p_value"] = p_value
                row["p_value_upper_bound"] = bool(exceed == 0)
                row["bootstrap_n"] = int(null_arr.size)
                row["null_delta_mean"] = float(np.nanmean(null_arr))
                row["null_delta_q95"] = float(np.nanpercentile(null_arr, 95))
            else:
                row["p_value"] = float("nan")
                row["bootstrap_n"] = 0
        q_values = _bh_fdr([float(row.get("p_value", float("nan"))) for row in rows])
        for row, q_value in zip(rows, q_values):
            row["q_value"] = float(q_value) if np.isfinite(q_value) else float("nan")
            row["significant"] = bool(
                np.isfinite(row["q_value"])
                and row["q_value"] < 0.05
                and float(row.get("delta_r2", float("nan"))) > 0
            )
        rows.sort(key=lambda item: (
            bool(item.get("significant", False)),
            np.isfinite(item.get("delta_r2", np.nan)),
            float(item.get("delta_r2", -np.inf)) if np.isfinite(item.get("delta_r2", np.nan)) else -np.inf,
        ), reverse=True)

    def _compute_glm_leave_one_out(
        self,
        dataset: Dict[str, Any],
        result: GLMResult,
        kernel_window: Tuple[float, float],
        basis_type: str,
        regularization: str,
        alpha: float,
    ) -> List[Dict[str, Any]]:
        predictors = dict(dataset.get("predictors", {}) or {})
        if not predictors or not result.predictor_names:
            return []
        full_mse = float(result.stats.get("mse", float("nan")))
        rows: List[Dict[str, Any]] = []
        time = np.asarray(dataset["time"], float)
        signal = np.asarray(dataset["signal"], float)
        self._progress_start("GLM leave-one-out", len(result.predictor_names))
        for pred_name in result.predictor_names:
            row: Dict[str, Any] = {
                "feature": pred_name,
                "label": self._predictor_label(pred_name),
                "full_r2": float(result.r2),
                "full_mse": full_mse,
                "reduced_r2": float("nan"),
                "reduced_mse": float("nan"),
                "delta_r2": float("nan"),
                "delta_mse": float("nan"),
                "contribution_pct": float("nan"),
                "status": "ok",
            }
            try:
                reduced_predictors = {k: v for k, v in predictors.items() if k != pred_name}
                if reduced_predictors:
                    reduced = ContinuousGLM().fit(
                        time,
                        signal,
                        reduced_predictors,
                        kernel_window=kernel_window,
                        n_basis=self.spin_n_basis.value(),
                        basis_type=basis_type,
                        regularization=regularization,
                        alpha=alpha,
                    )
                    reduced_r2 = float(reduced.r2)
                    reduced_mse = float(reduced.stats.get("mse", float("nan")))
                else:
                    stats = self._intercept_only_fit_stats(signal)
                    reduced_r2 = float(stats["r2"])
                    reduced_mse = float(stats["mse"])
                delta_r2 = float(result.r2 - reduced_r2)
                delta_mse = float(reduced_mse - full_mse) if np.isfinite(reduced_mse) and np.isfinite(full_mse) else float("nan")
                denom = float(result.r2) if np.isfinite(result.r2) and abs(result.r2) > 1e-12 else float("nan")
                row.update({
                    "reduced_r2": reduced_r2,
                    "reduced_mse": reduced_mse,
                    "delta_r2": delta_r2,
                    "delta_mse": delta_mse,
                    "contribution_pct": 100.0 * delta_r2 / denom if np.isfinite(denom) else float("nan"),
                })
            except Exception as exc:
                row["status"] = f"failed: {exc}"
            rows.append(row)
            self._progress_update(len(rows), "GLM leave-one-out")
        rows.sort(key=lambda item: (
            np.isfinite(item.get("delta_r2", np.nan)),
            float(item.get("delta_r2", -np.inf)) if np.isfinite(item.get("delta_r2", np.nan)) else -np.inf,
        ), reverse=True)
        return rows

    def _compute_glm_cross_validated_r2(
        self,
        dataset: Dict[str, Any],
        kernel_window: Tuple[float, float],
        basis_type: str,
        regularization: str,
        alpha: float,
    ) -> Dict[str, Any]:
        """Block-CV by file segment, or contiguous time blocks for a single file."""
        try:
            time = np.asarray(dataset["time"], float)
            signal = np.asarray(dataset["signal"], float)
            X, _cols, _n_b, _used = ContinuousGLM.build_design_matrix(
                time,
                dict(dataset.get("predictors", {}) or {}),
                kernel_window,
                int(self.spin_n_basis.value()),
                basis_type,
            )
        except Exception as exc:
            return {"cv_r2": float("nan"), "cv_folds": 0, "cv_note": f"CV unavailable: {exc}"}

        valid = np.isfinite(signal) & np.all(np.isfinite(X), axis=1)
        valid_idx = np.where(valid)[0]
        if valid_idx.size < max(20, X.shape[1] + 2):
            return {"cv_r2": float("nan"), "cv_folds": 0, "cv_note": "CV unavailable: too few valid samples."}

        user_n_folds = 0
        if hasattr(self, "spin_glm_cv_folds"):
            try:
                user_n_folds = int(self.spin_glm_cv_folds.value())
            except Exception:
                user_n_folds = 0

        folds: List[np.ndarray] = []
        segments = list(dataset.get("segment_slices", []) or [])
        if len(segments) >= 2 and user_n_folds <= 0:
            for lo, hi in segments:
                idx = np.arange(max(0, int(lo)), min(int(hi), signal.size))
                idx = idx[valid[idx]] if idx.size else idx
                if idx.size >= 3:
                    folds.append(idx)
        else:
            # User-driven (or no per-file blocks). Cap at samples//(min_fold_n).
            min_fold_size = max(50, X.shape[1] + 2)
            auto_n = min(5, max(2, valid_idx.size // min_fold_size))
            n_folds = user_n_folds if user_n_folds >= 2 else auto_n
            n_folds = max(2, min(n_folds, max(2, valid_idx.size // 3)))
            for part in np.array_split(valid_idx, n_folds):
                if part.size >= 3:
                    folds.append(part)
        if len(folds) < 2:
            return {"cv_r2": float("nan"), "cv_folds": 0, "cv_note": "CV unavailable: need at least two blocks."}

        held_true: List[np.ndarray] = []
        held_pred: List[np.ndarray] = []
        for test_idx in folds:
            train = valid.copy()
            train[test_idx] = False
            if np.sum(train) < max(5, X.shape[1] + 1):
                continue
            try:
                beta = ContinuousGLM.fit_coefficients(
                    X[train], signal[train], regularization=regularization, alpha=alpha
                )
            except Exception as exc:
                _LOG.debug("GLM CV fold failed: %s", exc)
                continue
            held_true.append(signal[test_idx])
            held_pred.append(X[test_idx] @ beta)
        if not held_true:
            return {"cv_r2": float("nan"), "cv_folds": 0, "cv_note": "CV unavailable: all folds failed."}
        y = np.concatenate(held_true)
        yp = np.concatenate(held_pred)
        ok = np.isfinite(y) & np.isfinite(yp)
        if np.sum(ok) < 3:
            return {"cv_r2": float("nan"), "cv_folds": len(held_true), "cv_note": "CV unavailable: invalid predictions."}
        ss_res = float(np.nansum((y[ok] - yp[ok]) ** 2))
        ss_tot = float(np.nansum((y[ok] - np.nanmean(y[ok])) ** 2))
        return {
            "cv_r2": 1.0 - ss_res / max(ss_tot, 1e-12),
            "cv_folds": int(len(held_true)),
            "cv_samples": int(np.sum(ok)),
        }

    def _compute_glm_diagnostics(self, dataset: Dict[str, Any], result: GLMResult) -> Dict[str, Any]:
        diagnostics: Dict[str, Any] = {}
        try:
            X = np.asarray(result.design_matrix, float)
            y = np.asarray(result.y_actual, float)
            valid = np.isfinite(y) & np.all(np.isfinite(X), axis=1)
            Xv = X[valid]
            if Xv.shape[0] > 2 and Xv.shape[1] > 1:
                Xs, _scales = ContinuousGLM._scaled_design(Xv)
                diagnostics["condition_number"] = float(np.linalg.cond(Xs))
        except Exception as exc:
            diagnostics["condition_note"] = f"condition number unavailable: {exc}"

        try:
            time = np.asarray(dataset["time"], float)
            signal = np.asarray(dataset["signal"], float)
            valid = np.isfinite(signal)
            raw: Dict[str, np.ndarray] = {}
            for name in result.predictor_names:
                spec = (dataset.get("predictors", {}) or {}).get(name)
                if spec is None:
                    continue
                vec = ContinuousGLM._predictor_vector(time, spec)
                vec = np.asarray(vec, float)
                if vec.size == signal.size:
                    raw[name] = vec
            high: List[Dict[str, Any]] = []
            names = list(raw)
            for i, a in enumerate(names):
                va = raw[a][valid]
                if va.size < 3 or np.nanstd(va) <= 1e-12:
                    continue
                for b in names[i + 1:]:
                    vb = raw[b][valid]
                    if vb.size != va.size or np.nanstd(vb) <= 1e-12:
                        continue
                    r = float(np.corrcoef(va, vb)[0, 1])
                    if np.isfinite(r) and abs(r) >= 0.8:
                        high.append({
                            "a": a,
                            "b": b,
                            "label_a": self._predictor_label(a),
                            "label_b": self._predictor_label(b),
                            "r": r,
                        })
            diagnostics["high_predictor_correlations"] = high[:20]
        except Exception as exc:
            diagnostics["correlation_note"] = f"predictor correlations unavailable: {exc}"
        return diagnostics

    def _sample_residual_blocks(
        self,
        residuals: np.ndarray,
        segment_slices: List[Tuple[int, int]],
        block_len: int,
        rng: np.random.Generator,
    ) -> np.ndarray:
        resid = np.asarray(residuals, float)
        sampled = np.zeros_like(resid)
        slices = segment_slices or [(0, int(resid.size))]
        block_len = max(2, int(block_len))
        for lo, hi in slices:
            lo = max(0, int(lo))
            hi = min(int(hi), int(resid.size))
            n = hi - lo
            if n <= 1:
                continue
            seg = resid[lo:hi].copy()
            finite = np.isfinite(seg)
            if not np.any(finite):
                continue
            seg[~finite] = 0.0
            starts = list(range(0, n, block_len))
            blocks = [seg[s:min(s + block_len, n)] for s in starts if s < n]
            out: List[np.ndarray] = []
            while sum(block.size for block in out) < n:
                out.append(blocks[int(rng.integers(0, len(blocks)))])
            sampled[lo:hi] = np.concatenate(out)[:n]
        sampled[~np.isfinite(sampled)] = 0.0
        return sampled

    def _compute_glm_kernel_bootstrap_ci(
        self,
        dataset: Dict[str, Any],
        result: GLMResult,
        kernel_window: Tuple[float, float],
        basis_type: str,
        regularization: str,
        alpha: float,
        n_boot: int,
    ) -> None:
        """Residual block-bootstrap pointwise 95% CIs for fitted kernels."""
        if n_boot <= 0 or not result.predictor_names:
            result.kernel_ci_lower = {}
            result.kernel_ci_upper = {}
            return
        requested_n_boot = int(n_boot)
        n_boot = min(requested_n_boot, 200)
        if requested_n_boot > n_boot:
            self.statusMessage.emit(
                f"Kernel-CI bootstrap capped at {n_boot} iterations (requested {requested_n_boot}). "
                f"The 95% percentile band has ~+/-1% width at N=200 - that's usually sufficient.",
                4000,
            )
        time = np.asarray(dataset["time"], float)
        signal = np.asarray(dataset["signal"], float)
        X = np.asarray(result.design_matrix, float)
        if X.ndim != 2 or X.shape[0] != signal.size:
            return
        try:
            _dt, _pre, _post, kernel_tvec = _glm_kernel_geometry(time, kernel_window)
        except Exception:
            kernel_tvec = np.asarray(result.kernel_tvec, float)
        block_len = max(2, int(np.asarray(kernel_tvec).size))
        valid = np.isfinite(signal) & np.isfinite(result.y_pred) & np.all(np.isfinite(X), axis=1)
        if np.sum(valid) < max(10, X.shape[1] + 2):
            return
        residuals = np.asarray(result.residuals, float)
        y_pred = np.asarray(result.y_pred, float)
        n_basis = max(1, (int(np.asarray(result.coefficients).size) - 1) // max(1, len(result.predictor_names)))
        segment_slices = list(dataset.get("segment_slices", []) or [])
        rng = np.random.default_rng()
        seeds = [int(v) for v in rng.integers(0, np.iinfo(np.uint32).max, size=n_boot, dtype=np.uint32)]
        max_jobs = max(1, min(int(self.spin_glm_jobs.value()) if hasattr(self, "spin_glm_jobs") else 1, os.cpu_count() or 1))
        self._progress_start(f"GLM kernel CI bootstrap ({max_jobs} job{'s' if max_jobs != 1 else ''})", n_boot)

        def _one(seed: int) -> Optional[Dict[str, np.ndarray]]:
            local_rng = np.random.default_rng(seed)
            boot_resid = self._sample_residual_blocks(residuals, segment_slices, block_len, local_rng)
            y_boot = y_pred + boot_resid
            try:
                beta = ContinuousGLM.fit_coefficients(
                    X[valid], y_boot[valid], regularization=regularization, alpha=alpha
                )
                return ContinuousGLM.kernels_from_coefficients(
                    beta,
                    list(result.predictor_names),
                    n_basis,
                    np.asarray(result.kernel_tvec, float),
                    basis_type,
                )
            except Exception as exc:
                _LOG.debug("GLM kernel CI bootstrap failed: %s", exc)
                return None

        stacks: Dict[str, List[np.ndarray]] = {name: [] for name in result.predictor_names}
        done = 0
        if max_jobs == 1:
            for seed in seeds:
                out = _one(seed)
                if out:
                    for name, vals in out.items():
                        stacks.setdefault(name, []).append(np.asarray(vals, float))
                done += 1
                if done == n_boot or done % 5 == 0:
                    self._progress_update(done, "GLM kernel CI bootstrap")
        else:
            with concurrent.futures.ThreadPoolExecutor(max_workers=max_jobs) as executor:
                futures = [executor.submit(_one, seed) for seed in seeds]
                for future in concurrent.futures.as_completed(futures):
                    try:
                        out = future.result()
                    except Exception as exc:
                        _LOG.debug("GLM kernel CI bootstrap worker failed: %s", exc)
                        out = None
                    if out:
                        for name, vals in out.items():
                            stacks.setdefault(name, []).append(np.asarray(vals, float))
                    done += 1
                    if done == n_boot or done % 5 == 0:
                        self._progress_update(done, "GLM kernel CI bootstrap")

        lower: Dict[str, np.ndarray] = {}
        upper: Dict[str, np.ndarray] = {}
        se_mean: Dict[str, float] = {}
        for name, curves in stacks.items():
            if len(curves) < 5:
                continue
            arr = np.vstack(curves)
            lower[name] = np.nanpercentile(arr, 2.5, axis=0)
            upper[name] = np.nanpercentile(arr, 97.5, axis=0)
            se = np.nanstd(arr, axis=0, ddof=1)
            se_mean[name] = float(np.nanmean(se)) if se.size else float("nan")
        result.kernel_ci_lower = lower
        result.kernel_ci_upper = upper
        result.stats = dict(result.stats or {})
        result.stats["kernel_ci_bootstrap_n"] = int(min(len(v) for v in stacks.values()) if stacks else 0)
        result.stats["kernel_ci_se_mean"] = se_mean

    @staticmethod
    def _simple_formula_terms(formula: str) -> List[str]:
        if "~" not in str(formula):
            return []
        rhs = str(formula).split("~", 1)[1].replace("\n", " ")
        terms: List[str] = []
        for raw in rhs.split("+"):
            term = raw.strip().strip("`")
            if not term or term in {"0", "1", "-1"}:
                continue
            terms.append(term)
        return terms

    @staticmethod
    def _term_mean_abs_coefficient(result: FLMMResult, term: str) -> float:
        coeff = TemporalModelingWidget._term_coefficient_curve(result, term)
        if coeff is None:
            return float("nan")
        vals = np.asarray(coeff, float)
        return _finite_curve_stats(vals)[1]

    @staticmethod
    def _term_coefficient_curve(result: FLMMResult, term: str) -> Optional[np.ndarray]:
        name = TemporalModelingWidget._term_coefficient_name(result, term)
        if name is None:
            return None
        return np.asarray(result.coefficients.get(name, np.array([], float)), float)

    @staticmethod
    def _term_coefficient_name(result: FLMMResult, term: str) -> Optional[str]:
        if not result.coefficients:
            return None
        clean = re.sub(r"[^0-9A-Za-z_]+", "", str(term).lower())
        for name, coeff in result.coefficients.items():
            if str(name) == str(term):
                return str(name)
        for name, coeff in result.coefficients.items():
            name_clean = re.sub(r"[^0-9A-Za-z_]+", "", str(name).lower())
            if clean and (clean in name_clean or name_clean in clean):
                return str(name)
        return None

    @staticmethod
    def _normal_two_sided_p(z_values: np.ndarray) -> np.ndarray:
        z = np.asarray(z_values, float)
        out = np.full(z.shape, np.nan, float)
        finite = np.isfinite(z)
        if not np.any(finite):
            return out
        vals = np.abs(z[finite]) / math.sqrt(2.0)
        out[finite] = np.array([math.erfc(float(v)) for v in vals], float)
        return np.clip(out, 0.0, 1.0)

    @staticmethod
    def _coefficient_ci_for_name(result: FLMMResult, name: str) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], str]:
        lo = (result.ci_lower or {}).get(name)
        hi = (result.ci_upper or {}).get(name)
        if lo is not None and hi is not None:
            return np.asarray(lo, float), np.asarray(hi, float), "pointwise_95ci"
        lo = (result.joint_ci_lower or {}).get(name)
        hi = (result.joint_ci_upper or {}).get(name)
        if lo is not None and hi is not None:
            return np.asarray(lo, float), np.asarray(hi, float), "joint_95ci"
        return None, None, ""

    def _compute_flmm_curve_significance(
        self,
        result: FLMMResult,
        terms: List[str],
        term_labels: Dict[str, str],
    ) -> Dict[str, Dict[str, Any]]:
        rows: List[Dict[str, Any]] = []
        for term in terms:
            name = self._term_coefficient_name(result, term)
            if name is None:
                continue
            coeff = np.asarray(result.coefficients.get(name, np.array([], float)), float)
            lo, hi, ci_source = self._coefficient_ci_for_name(result, name)
            row: Dict[str, Any] = {
                "feature": term,
                "coefficient_name": name,
                "label": term_labels.get(term, term),
                "ci_source": ci_source or "none",
                "p_value": float("nan"),
                "q_value": float("nan"),
                "significant": False,
                "significant_bins": 0,
                "significant_fraction": 0.0,
                "max_abs_z": float("nan"),
                "status": "no coefficient CI available" if not ci_source else "ok",
            }
            if coeff.size and lo is not None and hi is not None:
                n = min(int(coeff.size), int(lo.size), int(hi.size))
                c = coeff[:n]
                l = lo[:n]
                h = hi[:n]
                valid = np.isfinite(c) & np.isfinite(l) & np.isfinite(h) & (h > l)
                sig_mask = valid & (((l > 0.0) & (h > 0.0)) | ((l < 0.0) & (h < 0.0)))
                row["significant_bins"] = int(np.sum(sig_mask))
                row["significant_fraction"] = float(np.sum(sig_mask) / max(1, int(np.sum(valid))))
                if np.any(valid):
                    se = (h[valid] - l[valid]) / (2.0 * 1.96)
                    z = np.divide(c[valid], se, out=np.full_like(se, np.nan), where=se > 1e-12)
                    p_point = self._normal_two_sided_p(z)
                    finite_p = p_point[np.isfinite(p_point)]
                    finite_z = np.abs(z[np.isfinite(z)])
                    if finite_z.size:
                        row["max_abs_z"] = float(np.nanmax(finite_z))
                    if finite_p.size:
                        # Conservative curve-level p-value: Bonferroni over fitted time bins.
                        row["p_value"] = float(min(1.0, np.nanmin(finite_p) * finite_p.size))
                    row["status"] = "ok"
                elif coeff.size:
                    row["status"] = "no finite CI bins"
            rows.append(row)

        q_values = _bh_fdr([float(row.get("p_value", float("nan"))) for row in rows])
        for row, q_value in zip(rows, q_values):
            row["q_value"] = float(q_value) if np.isfinite(q_value) else float("nan")
            row["significant"] = bool(
                int(row.get("significant_bins", 0)) > 0
                and np.isfinite(row["q_value"])
                and row["q_value"] < 0.05
            )
        return {str(row["feature"]): row for row in rows}

    def _compute_flmm_coefficient_importance(
        self,
        result: FLMMResult,
        terms: List[str],
        term_labels: Dict[str, str],
    ) -> List[Dict[str, Any]]:
        rows: List[Dict[str, Any]] = []
        for term in terms:
            coeff = self._term_coefficient_curve(result, term)
            if coeff is None:
                continue
            vals = np.asarray(coeff, float)
            if vals.size == 0:
                continue
            _, mean_abs, peak_abs = _finite_curve_stats(vals)
            if not np.isfinite(mean_abs) and not np.isfinite(peak_abs):
                rows.append({
                    "feature": term,
                    "label": term_labels.get(term, term),
                    "mean_abs_coefficient": float("nan"),
                    "peak_abs_coefficient": float("nan"),
                    "delta_aic": float("nan"),
                    "status": "no finite coefficient bins",
                })
                continue
            rows.append({
                "feature": term,
                "label": term_labels.get(term, term),
                "mean_abs_coefficient": mean_abs,
                "peak_abs_coefficient": peak_abs,
                "delta_aic": float("nan"),
                "status": "ok",
            })
        rows.sort(key=lambda item: (
            np.isfinite(item.get("mean_abs_coefficient", np.nan)),
            float(item.get("mean_abs_coefficient", -np.inf)) if np.isfinite(item.get("mean_abs_coefficient", np.nan)) else -np.inf,
        ), reverse=True)
        return rows

    @staticmethod
    def _merge_flmm_significance_rows(
        rows: List[Dict[str, Any]],
        significance: Dict[str, Dict[str, Any]],
    ) -> None:
        for row in rows:
            feature = str(row.get("feature", ""))
            sig = dict(significance.get(feature, {}) or {})
            if not sig:
                row.setdefault("p_value", float("nan"))
                row.setdefault("q_value", float("nan"))
                row.setdefault("significant", False)
                row.setdefault("significance_status", "not tested")
                continue
            row["p_value"] = float(sig.get("p_value", float("nan")))
            row["q_value"] = float(sig.get("q_value", float("nan")))
            row["significant"] = bool(sig.get("significant", False))
            row["significant_bins"] = int(sig.get("significant_bins", 0) or 0)
            row["significant_fraction"] = float(sig.get("significant_fraction", 0.0) or 0.0)
            row["max_abs_z"] = float(sig.get("max_abs_z", float("nan")))
            row["ci_source"] = str(sig.get("ci_source", "none") or "none")
            row["significance_status"] = str(sig.get("status", "not tested") or "not tested")

    @staticmethod
    def _fmt_stat(value: object, digits: int = 5) -> str:
        try:
            val = float(value)
        except (TypeError, ValueError):
            return "n/a"
        if not np.isfinite(val):
            return "n/a"
        return f"{val:.{digits}g}"

    @staticmethod
    def _flmm_readable_label(term: str, term_labels: Dict[str, str]) -> str:
        text = str(term or "")
        if text in {"(Intercept)", "Intercept", "intercept"}:
            return "(Intercept)"
        return str(term_labels.get(text, text) or text)

    def _readable_flmm_formula(self, formula_terms: List[str], term_labels: Dict[str, str]) -> str:
        labels = [self._flmm_readable_label(term, term_labels) for term in formula_terms]
        return "Aligned response ~ " + (" + ".join(labels) if labels else "1")

    # ------------------------------------------------------------------
    # FLMM rich-HTML summary
    # ------------------------------------------------------------------

    @staticmethod
    def _html_p_text(value: float) -> str:
        if not np.isfinite(value):
            return "<span style='color:#6f7a8e;'>n/a</span>"
        if value < 0.0001:
            return "<b>&lt; 1e-4</b>"
        return f"{value:.3g}"

    @staticmethod
    def _html_q_text(value: float, significant: bool) -> str:
        if not np.isfinite(value):
            return "<span style='color:#6f7a8e;'>n/a</span>"
        color = "#5dd39e" if significant else "#cfd8e6"
        weight = "bold" if significant else "normal"
        return f"<span style='color:{color}; font-weight:{weight};'>{value:.3g}</span>"

    @staticmethod
    def _html_badge(text: str, color: str, bg: str) -> str:
        return (f"<span style='background:{bg}; color:{color}; border:1px solid {color};"
                f" padding:1px 6px; border-radius:4px; font-size:8pt; font-weight:700;'>"
                f"{text}</span>")

    @staticmethod
    def _html_escape(text: object) -> str:
        import html
        return html.escape(str(text or ""))

    def _build_flmm_summary_html(
        self,
        *,
        result: FLMMResult,
        importance_rows: List[Dict[str, Any]],
        importance_mode: str,
        term_labels: Dict[str, str],
        formula_terms: List[str],
        backend_name: str,
        scope: str,
        n_rows: int,
        effective_group_var: str,
        readable_formula: str,
        fallback_grouping: str,
        missing_terms: List[str],
        fit_terms: List[str],
        dropped_terms: List[str],
        use_fixed_fallback: bool,
        num_boots: int,
    ) -> str:
        esc = self._html_escape
        stats = result.stats or {}

        # ---- Verdict card -------------------------------------------------
        tested_rows = [r for r in importance_rows if r.get("significance_status") == "ok"]
        sig_rows = [r for r in tested_rows if r.get("significant", False)]
        verdict_color = "#5dd39e" if sig_rows else "#aab4c5"
        verdict_bg = "#1c2e22" if sig_rows else "#1f242e"
        verdict_border = "#2f7a4a" if sig_rows else "#3a4050"
        if tested_rows:
            verdict_main = (
                f"<b style='font-size:13pt; color:{verdict_color};'>"
                f"{len(sig_rows)}/{len(tested_rows)}</b>"
                f" <span style='color:#cfd8e6;'>predictors significant after FDR (q &lt; 0.05)</span>"
            )
        else:
            verdict_main = "<b style='color:#aab4c5;'>Significance untested</b> "\
                           "<span style='color:#aab4c5;'>(no coefficient CIs available)</span>"
        verdict_html = (
            f"<table width='100%' cellpadding='0' cellspacing='0'><tr><td "
            f"style='background:{verdict_bg}; border:1px solid {verdict_border};"
            f" padding:10px 14px;'>"
            f"<div>{verdict_main}</div>"
            f"<div style='color:#aab4c5; font-size:8.6pt; margin-top:4px;'>"
            f"Method: conservative z-test on coefficient CI widths &middot; "
            f"Bonferroni across time bins &middot; Benjamini-Hochberg FDR across predictors."
            f"</div>"
            f"</td></tr></table>"
        )

        # ---- Fit meta line ------------------------------------------------
        meta_bits = [
            esc(backend_name),
            f"scope = {esc('animal/group rows' if scope == 'animal' else 'trial rows')}",
            f"rows = {int(n_rows)}",
            f"ID = {esc(effective_group_var)}",
        ]
        n_terms = int(float(stats.get("n_terms", len(result.coefficients or {})) or 0))
        n_tp = int(float(stats.get("n_timepoints", np.asarray(result.tvec).size) or 0))
        if n_terms or n_tp:
            meta_bits.append(f"{n_terms} term(s), {n_tp} timepoint(s)")
        aic_val = float(stats.get("aic", float("nan")))
        if not np.isfinite(aic_val) and getattr(result, "aic", None) is not None:
            try:
                aic_val = float(result.aic)
            except (TypeError, ValueError):
                aic_val = float("nan")
        if np.isfinite(aic_val):
            meta_bits.append(f"AIC = {aic_val:.5g}")
        meta_html = (
            f"<div style='color:#aab4c5; font-size:8.7pt; padding:6px 0 2px 0;'>"
            f"{' &middot; '.join(meta_bits)}"
            f"</div>"
            f"<div style='color:#cfd8e6; font-size:8.8pt; padding:0 0 6px 0;'>"
            f"<span style='color:#6f7a8e;'>formula:</span> {esc(readable_formula)}"
            f"</div>"
        )
        warning_bits: List[str] = []
        if use_fixed_fallback or fallback_grouping:
            warning_bits.append(self._html_badge("SINGLE-ANIMAL MODE", "#f5c542", "#2e2918"))
            warning_bits.append(
                "<span style='color:#f5c542;'>Functional regression fitted at the trial level "
                "(no between-subject random effects). Coefficients describe within-recording "
                "associations; do not generalize to population-level inference without a "
                "multi-animal cohort.</span>"
            )
        if stats.get("variance_unavailable"):
            warning_bits.append(self._html_badge("NO CI", "#f5c542", "#2e2918"))
            warning_bits.append(
                "<span style='color:#f5c542;'>Variance inference unavailable for this fit; "
                "significance cannot be assessed.</span>"
            )
        if warning_bits:
            meta_html += (
                f"<div style='padding:4px 0 6px 0;'>{' '.join(warning_bits)}</div>"
            )

        # ---- Significance table (CI-based p/q and optional permutation p/q) -
        sig_html = ""
        if importance_rows:
            has_perm = any("p_perm" in row for row in importance_rows)
            sorted_sig = sorted(
                importance_rows,
                key=lambda r: (
                    not bool(r.get("significant", False)),
                    not bool(r.get("significant_perm", False)),
                    float(r.get("q_value", float("inf")))
                    if np.isfinite(float(r.get("q_value", float("inf"))))
                    else float("inf"),
                    -(float(r.get("mean_abs_coefficient", 0.0))
                      if np.isfinite(float(r.get("mean_abs_coefficient", 0.0))) else 0.0),
                ),
            )
            rows_html: List[str] = []
            for row in sorted_sig:
                label = self._html_escape(row.get("label", row.get("feature", "")))
                p_val = float(row.get("p_value", float("nan")))
                q_val = float(row.get("q_value", float("nan")))
                is_sig = bool(row.get("significant", False))
                is_sig_perm = bool(row.get("significant_perm", False))
                status = str(row.get("significance_status", "not tested") or "not tested")
                if is_sig:
                    badge = self._html_badge("SIG", "#5dd39e", "#1c2e22")
                elif status != "ok":
                    badge = self._html_badge("UNTESTED", "#aab4c5", "#1f242e")
                else:
                    badge = self._html_badge("n.s.", "#aab4c5", "#1f242e")
                frac = float(row.get("significant_fraction", 0.0) or 0.0)
                frac_text = f"{frac * 100:.1f}%" if frac else "0.0%"

                # Permutation columns (only present when perm test was run).
                if has_perm:
                    p_perm_val = float(row.get("p_perm", float("nan")))
                    q_perm_val = float(row.get("q_perm", float("nan")))
                    if row.get("p_perm_upper_bound"):
                        n_perm = int(row.get("perm_n", 0))
                        p_perm_html = (
                            f"<b>&le; {1.0/max(n_perm + 1, 1):.3g}</b>"
                            if n_perm > 0 else
                            "<span style='color:#6f7a8e;'>n/a</span>"
                        )
                    else:
                        p_perm_html = self._html_p_text(p_perm_val)
                    q_perm_html = self._html_q_text(q_perm_val, is_sig_perm)
                    perm_badge = (self._html_badge("SIG", "#5dd39e", "#1c2e22")
                                  if is_sig_perm else
                                  self._html_badge("n.s.", "#aab4c5", "#1f242e"))
                    perm_cells = (
                        f"<td style='padding:4px 8px; border-bottom:1px solid #2c3240;"
                        f" font-family:Consolas,monospace;'>{p_perm_html}</td>"
                        f"<td style='padding:4px 8px; border-bottom:1px solid #2c3240;"
                        f" font-family:Consolas,monospace;'>{q_perm_html}</td>"
                        f"<td style='padding:4px 8px; border-bottom:1px solid #2c3240;'>{perm_badge}</td>"
                    )
                else:
                    perm_cells = ""

                rows_html.append(
                    "<tr>"
                    f"<td style='padding:4px 8px; border-bottom:1px solid #2c3240;"
                    f" color:{'#5dd39e' if (is_sig or is_sig_perm) else '#e9ecf3'};'>"
                    f"{'<b>' if (is_sig or is_sig_perm) else ''}{label}"
                    f"{'</b>' if (is_sig or is_sig_perm) else ''}"
                    f"</td>"
                    f"<td style='padding:4px 8px; border-bottom:1px solid #2c3240;"
                    f" font-family:Consolas,monospace;'>{self._html_p_text(p_val)}</td>"
                    f"<td style='padding:4px 8px; border-bottom:1px solid #2c3240;"
                    f" font-family:Consolas,monospace;'>{self._html_q_text(q_val, is_sig)}</td>"
                    f"<td style='padding:4px 8px; border-bottom:1px solid #2c3240;"
                    f" color:#cfd8e6;'>{frac_text}</td>"
                    f"<td style='padding:4px 8px; border-bottom:1px solid #2c3240;'>{badge}</td>"
                    + perm_cells
                    + "</tr>"
                )
            header_style = ("padding:6px 8px; background:#1f242e; color:#aab4c5;"
                            " text-align:left; font-weight:700; font-size:8.5pt;"
                            " letter-spacing:0.4px; text-transform:uppercase;"
                            " border-bottom:1px solid #2c3240;")
            perm_hdr = (
                f"<th style='{header_style}'>p (perm)</th>"
                f"<th style='{header_style}'>q (perm FDR)</th>"
                f"<th style='{header_style}'>Result (perm)</th>"
            ) if has_perm else ""
            footer_extra = (
                "<br>"
                "<b>p (perm) / q (perm FDR):</b> per-predictor permutation test of "
                "variable contribution. Each predictor's trial values are shuffled, the "
                "model is refit, and the L<sup>2</sup> norm of the resulting coefficient "
                "curve is compared to the observed one. Robust to autocorrelation and to "
                "the CI assumption of the conservative z-test."
            ) if has_perm else ""
            sig_html = (
                "<h3 style='color:#aab4c5; font-size:9pt; font-weight:600;"
                " letter-spacing:0.4px; text-transform:uppercase; margin:14px 0 6px 0;'>"
                "Variable significance (CI-based and permutation)</h3>"
                "<table width='100%' cellpadding='0' cellspacing='0' "
                "style='border:1px solid #2c3240; border-radius:6px;'>"
                f"<tr>"
                f"<th style='{header_style}'>Predictor</th>"
                f"<th style='{header_style}'>p (raw)</th>"
                f"<th style='{header_style}'>q (FDR)</th>"
                f"<th style='{header_style}'>Sig bins</th>"
                f"<th style='{header_style}'>Result</th>"
                f"{perm_hdr}"
                f"</tr>"
                + "".join(rows_html)
                + "</table>"
                "<div style='color:#6f7a8e; font-size:8.4pt; margin:4px 0 0 0;'>"
                "<b>p (raw) / q (FDR):</b> CI-based z-test on coefficient widths, "
                "Bonferroni across time bins, Benjamini-Hochberg FDR across predictors."
                f"{footer_extra}"
                "</div>"
            )

        # ---- Coefficient magnitudes table --------------------------------
        coeff_html = ""
        coefficients = result.coefficients or {}
        if coefficients:
            preferred = ["(Intercept)"] + list(formula_terms)
            seen = set()
            ordered_names: List[str] = []
            for name in preferred:
                cn = self._term_coefficient_name(result, name) if name != "(Intercept)" else "(Intercept)"
                if cn in coefficients and cn not in seen:
                    ordered_names.append(cn)
                    seen.add(cn)
            for name in coefficients:
                if name not in seen:
                    ordered_names.append(name)
                    seen.add(name)

            rows_html = []
            for name in ordered_names[:12]:
                coeff = np.asarray(coefficients.get(name, np.array([], float)), float)
                mean_coef, mean_abs, peak_abs = _finite_curve_stats(coeff)
                label = self._html_escape(self._flmm_readable_label(name, term_labels))
                if not np.isfinite(mean_abs) and not np.isfinite(peak_abs):
                    rows_html.append(
                        f"<tr><td style='padding:4px 8px; border-bottom:1px solid #2c3240;'>{label}</td>"
                        f"<td colspan='3' style='padding:4px 8px; border-bottom:1px solid #2c3240;"
                        f" color:#aab4c5;'>not estimable</td></tr>"
                    )
                    continue
                rows_html.append(
                    "<tr>"
                    f"<td style='padding:4px 8px; border-bottom:1px solid #2c3240; color:#e9ecf3;'>{label}</td>"
                    f"<td style='padding:4px 8px; border-bottom:1px solid #2c3240;"
                    f" font-family:Consolas,monospace; color:#cfd8e6;'>"
                    f"{self._fmt_stat(mean_coef, 4)}</td>"
                    f"<td style='padding:4px 8px; border-bottom:1px solid #2c3240;"
                    f" font-family:Consolas,monospace; color:#cfd8e6;'>"
                    f"{self._fmt_stat(mean_abs, 4)}</td>"
                    f"<td style='padding:4px 8px; border-bottom:1px solid #2c3240;"
                    f" font-family:Consolas,monospace; color:#cfd8e6;'>"
                    f"{self._fmt_stat(peak_abs, 4)}</td>"
                    "</tr>"
                )
            header_style = ("padding:6px 8px; background:#1f242e; color:#aab4c5;"
                            " text-align:left; font-weight:700; font-size:8.5pt;"
                            " letter-spacing:0.4px; text-transform:uppercase;"
                            " border-bottom:1px solid #2c3240;")
            extra_rows = ""
            if len(ordered_names) > 12:
                extra_rows = (
                    f"<tr><td colspan='4' style='padding:4px 8px;"
                    f" color:#6f7a8e; font-style:italic;'>"
                    f"... {len(ordered_names) - 12} more coefficient curve(s)</td></tr>"
                )
            coeff_html = (
                "<h3 style='color:#aab4c5; font-size:9pt; font-weight:600;"
                " letter-spacing:0.4px; text-transform:uppercase; margin:14px 0 6px 0;'>"
                "Coefficient magnitudes</h3>"
                "<table width='100%' cellpadding='0' cellspacing='0' "
                "style='border:1px solid #2c3240; border-radius:6px;'>"
                f"<tr>"
                f"<th style='{header_style}'>Predictor</th>"
                f"<th style='{header_style}'>Mean coef</th>"
                f"<th style='{header_style}'>Mean abs</th>"
                f"<th style='{header_style}'>Peak abs</th>"
                f"</tr>"
                + "".join(rows_html)
                + extra_rows
                + "</table>"
            )

        # ---- Interpretation line -----------------------------------------
        interp_html = (
            "<h3 style='color:#aab4c5; font-size:9pt; font-weight:600;"
            " letter-spacing:0.4px; text-transform:uppercase; margin:14px 0 6px 0;'>"
            "How to read this</h3>"
            "<div style='color:#cfd8e6; font-size:9pt;'>"
            "Each coefficient curve is the expected change in the aligned response "
            "for a +1 SD increase in that predictor. Positive coefficients mean higher "
            "predictor values are associated with a higher response at that time; "
            "negative coefficients mean a lower response."
            "</div>"
        )

        # ---- Diagnostics / dropped covariates ----------------------------
        notes_bits: List[str] = []
        if missing_terms:
            readable_missing = [self._html_escape(term_labels.get(t, t)) for t in missing_terms]
            notes_bits.append(
                f"<div style='color:#cfd8e6; font-size:8.6pt; padding:3px 0;'>"
                f"<b style='color:#aab4c5;'>Auto formula used:</b> {', '.join(readable_missing)}"
                f"</div>"
            )
        if dropped_terms:
            notes_bits.append(
                f"<div style='color:#cfd8e6; font-size:8.6pt; padding:3px 0;'>"
                f"<b style='color:#aab4c5;'>Dropped covariates:</b> "
                f"{self._html_escape(', '.join(dropped_terms))}"
                f"</div>"
            )
        for note in stats.get("fit_notes", []) or []:
            notes_bits.append(
                f"<div style='color:#cfd8e6; font-size:8.6pt; padding:3px 0;'>"
                f"<b style='color:#aab4c5;'>Fit note:</b> {self._html_escape(note)}"
                f"</div>"
            )
        if importance_mode == "loo" and num_boots > 0:
            notes_bits.append(
                f"<div style='color:#6f7a8e; font-size:8.4pt; padding:3px 0;'>"
                f"Leave-one-out comparison uses analytic fits; bootstrap CIs are not repeated."
                f"</div>"
            )
        notes_html = ""
        if notes_bits:
            notes_html = (
                "<h3 style='color:#aab4c5; font-size:9pt; font-weight:600;"
                " letter-spacing:0.4px; text-transform:uppercase; margin:14px 0 6px 0;'>"
                "Notes</h3>"
                + "".join(notes_bits)
            )

        # ---- Assemble ----------------------------------------------------
        title_html = (
            "<div style='color:#e9ecf3; font-size:11pt; font-weight:700;"
            " padding-bottom:4px;'>FLMM / functional model summary</div>"
        )
        return (
            "<div style='font-family:\"Segoe UI\", sans-serif; color:#e9ecf3;"
            " line-height:1.4;'>"
            + title_html
            + verdict_html
            + meta_html
            + sig_html
            + coeff_html
            + interp_html
            + notes_html
            + "</div>"
        )

    def _build_glm_summary_html(
        self,
        *,
        result: GLMResult,
        importance_rows: List[Dict[str, Any]],
        basis_label: str,
        regularization_label: str,
        n_basis: int,
        alpha: float,
        n_boot: int,
        n_jobs: int,
        used_records: List[str],
        used_labels: List[str],
        dropped_predictors: List[str],
        dropped_records: List[str],
        valid_samples: int,
        scope_label: str,
        from_cache: bool = False,
    ) -> str:
        esc = self._html_escape
        stats = result.stats or {}

        # ---- Verdict card ------------------------------------------------
        tested_rows = [
            row for row in importance_rows
            if np.isfinite(float(row.get("p_value", float("nan"))))
        ]
        sig_rows = [row for row in tested_rows if row.get("significant", False)]
        r2_val = float(result.r2) if np.isfinite(result.r2) else float("nan")
        cv_r2 = float(stats.get("cv_r2", float("nan")))

        verdict_color = "#5dd39e" if sig_rows else "#aab4c5"
        verdict_bg = "#1c2e22" if sig_rows else "#1f242e"
        verdict_border = "#2f7a4a" if sig_rows else "#3a4050"
        r2_text = f"{r2_val:.3f}" if np.isfinite(r2_val) else "n/a"
        cv_text = (
            f"<span style='color:#cfd8e6;'> &middot; out-of-sample (block-CV) "
            f"R<sup>2</sup> = <b style='color:#e9ecf3;'>{cv_r2:.3f}</b></span>"
            if np.isfinite(cv_r2) else ""
        )
        if tested_rows:
            sig_phrase = (
                f"<b style='font-size:13pt; color:{verdict_color};'>"
                f"{len(sig_rows)}/{len(tested_rows)}</b>"
                f" <span style='color:#cfd8e6;'>predictors significant "
                f"(BH-FDR q &lt; 0.05)</span>"
            )
        else:
            sig_phrase = (
                "<span style='color:#aab4c5;'>"
                "Significance untested (no shift-bootstrap)</span>"
            )
        verdict_html = (
            f"<table width='100%' cellpadding='0' cellspacing='0'><tr><td "
            f"style='background:{verdict_bg}; border:1px solid {verdict_border};"
            f" padding:10px 14px;'>"
            f"<div style='font-size:11pt; color:#e9ecf3;'>"
            f"In-sample R<sup>2</sup> = "
            f"<b style='color:#e9ecf3;'>{r2_text}</b>{cv_text}"
            f"</div>"
            f"<div style='margin-top:4px;'>{sig_phrase}</div>"
            f"<div style='color:#aab4c5; font-size:8.6pt; margin-top:4px;'>"
            f"Method: leave-one-predictor-out &Delta;R<sup>2</sup> with "
            f"circular-shift bootstrap null distribution &middot; "
            f"Benjamini-Hochberg FDR across predictors."
            f"</div>"
            f"</td></tr></table>"
        )

        # ---- Meta line ---------------------------------------------------
        record_preview = ", ".join(esc(v) for v in used_records[:6])
        if len(used_records) > 6:
            record_preview += "..."
        scope_text = (
            f"{esc(scope_label)} &middot; {len(used_records)} recording"
            f"{'s' if len(used_records) != 1 else ''}"
        )
        meta_bits = [
            scope_text,
            f"{int(valid_samples)} samples",
            f"{len(result.predictor_names)} predictor(s)",
            f"{esc(basis_label)} (n={int(n_basis)})",
            f"{esc(regularization_label)} (&alpha;={float(alpha):.3g})",
        ]
        if n_boot > 0:
            meta_bits.append(
                f"shift-bootstrap N={int(n_boot)}, jobs={int(n_jobs)}"
            )
        meta_html = (
            f"<div style='color:#aab4c5; font-size:8.7pt; padding:6px 0 2px 0;'>"
            f"{' &middot; '.join(meta_bits)}"
            f"</div>"
        )
        if record_preview:
            meta_html += (
                f"<div style='color:#cfd8e6; font-size:8.8pt; padding:0 0 6px 0;'>"
                f"<span style='color:#6f7a8e;'>recordings:</span> {record_preview}"
                f"</div>"
            )
        if used_labels:
            meta_html += (
                f"<div style='color:#cfd8e6; font-size:8.8pt; padding:0 0 6px 0;'>"
                f"<span style='color:#6f7a8e;'>predictors:</span> "
                f"{esc(', '.join(used_labels))}"
                f"</div>"
            )
        warning_bits: List[str] = []
        if from_cache:
            warning_bits.append(self._html_badge("CACHED", "#9ad6f5", "#1a242e"))
            warning_bits.append(
                "<span style='color:#9ad6f5;'>Loaded from disk cache; "
                "no refit was needed.</span>"
            )
        if n_boot == 0 and tested_rows == []:
            warning_bits.append(self._html_badge("NO BOOT", "#f5c542", "#2e2918"))
            warning_bits.append(
                "<span style='color:#f5c542;'>Shift-bootstrap is off; "
                "predictor significance cannot be assessed.</span>"
            )
        if warning_bits:
            meta_html += (
                f"<div style='padding:4px 0 6px 0;'>{' '.join(warning_bits)}</div>"
            )

        header_style = (
            "padding:6px 8px; background:#1f242e; color:#aab4c5;"
            " text-align:left; font-weight:700; font-size:8.5pt;"
            " letter-spacing:0.4px; text-transform:uppercase;"
            " border-bottom:1px solid #2c3240;"
        )
        cell_style = "padding:4px 8px; border-bottom:1px solid #2c3240;"
        mono_style = cell_style + " font-family:Consolas,monospace;"

        # ---- Fit statistics table ----------------------------------------
        fit_stats_rows = [
            ("In-sample R<sup>2</sup>", r2_val, 4),
            ("Block-CV R<sup>2</sup>", cv_r2, 4),
            ("RMSE", float(stats.get("rmse", float("nan"))), 5),
            ("MAE", float(stats.get("mae", float("nan"))), 5),
            ("MSE", float(stats.get("mse", float("nan"))), 5),
            ("Residual SD", float(stats.get("residual_std", float("nan"))), 5),
            ("Actual/predicted corr", float(stats.get("corr", float("nan"))), 4),
        ]
        fit_rows_html: List[str] = []
        for label, value, digits in fit_stats_rows:
            fit_rows_html.append(
                f"<tr>"
                f"<td style='{cell_style} color:#e9ecf3;'>{label}</td>"
                f"<td style='{mono_style} color:#cfd8e6;'>"
                f"{self._fmt_stat(value, digits)}</td>"
                f"</tr>"
            )
        cv_folds = int(stats.get("cv_folds", 0) or 0)
        cv_note = str(stats.get("cv_note", "") or "")
        cv_extra = f" ({esc(cv_note)})" if cv_note else ""
        fit_rows_html.append(
            f"<tr>"
            f"<td style='{cell_style} color:#e9ecf3;'>Block-CV folds</td>"
            f"<td style='{mono_style} color:#cfd8e6;'>"
            f"{cv_folds}{cv_extra}</td>"
            f"</tr>"
        )
        fit_html = (
            "<h3 style='color:#aab4c5; font-size:9pt; font-weight:600;"
            " letter-spacing:0.4px; text-transform:uppercase; margin:14px 0 6px 0;'>"
            "Fit statistics</h3>"
            "<table width='100%' cellpadding='0' cellspacing='0' "
            "style='border:1px solid #2c3240; border-radius:6px;'>"
            f"<tr>"
            f"<th style='{header_style}'>Metric</th>"
            f"<th style='{header_style}'>Value</th>"
            f"</tr>"
            + "".join(fit_rows_html)
            + "</table>"
        )

        # ---- Diagnostics table ------------------------------------------
        diag_rows_html: List[str] = []
        condition = float(stats.get("condition_number", float("nan")))
        cond_color = "#e9ecf3" if (np.isfinite(condition) and condition < 30) else (
            "#f5c542" if (np.isfinite(condition) and condition < 100) else "#f5755a"
        )
        cond_badge = ""
        if np.isfinite(condition):
            if condition < 30:
                cond_badge = self._html_badge("OK", "#5dd39e", "#1c2e22")
            elif condition < 100:
                cond_badge = self._html_badge("WATCH", "#f5c542", "#2e2918")
            else:
                cond_badge = self._html_badge("UNSTABLE", "#f5755a", "#2e1c18")
        diag_rows_html.append(
            f"<tr>"
            f"<td style='{cell_style} color:#e9ecf3;'>Design condition number</td>"
            f"<td style='{mono_style} color:{cond_color};'>"
            f"{self._fmt_stat(condition, 3)}</td>"
            f"<td style='{cell_style}'>{cond_badge}</td>"
            f"</tr>"
        )
        high_corr = stats.get("high_predictor_correlations", []) or []
        if high_corr:
            for item in high_corr[:5]:
                lbl_a = esc(item.get("label_a", item.get("a", "")))
                lbl_b = esc(item.get("label_b", item.get("b", "")))
                r_val = float(item.get("r", float("nan")))
                diag_rows_html.append(
                    f"<tr>"
                    f"<td style='{cell_style} color:#e9ecf3;'>"
                    f"{lbl_a} &harr; {lbl_b}</td>"
                    f"<td style='{mono_style} color:#f5c542;'>"
                    f"r = {self._fmt_stat(r_val, 3)}</td>"
                    f"<td style='{cell_style}'>"
                    f"{self._html_badge('|r| &ge; 0.8', '#f5c542', '#2e2918')}</td>"
                    f"</tr>"
                )
            if len(high_corr) > 5:
                diag_rows_html.append(
                    f"<tr><td colspan='3' style='{cell_style} color:#6f7a8e;"
                    f" font-style:italic;'>... {len(high_corr) - 5} more "
                    f"correlated pair(s)</td></tr>"
                )
        else:
            diag_rows_html.append(
                f"<tr>"
                f"<td colspan='3' style='{cell_style} color:#cfd8e6;'>"
                f"No predictor pairs with |r| &ge; 0.8."
                f"</td>"
                f"</tr>"
            )
        diag_html = (
            "<h3 style='color:#aab4c5; font-size:9pt; font-weight:600;"
            " letter-spacing:0.4px; text-transform:uppercase; margin:14px 0 6px 0;'>"
            "Diagnostics</h3>"
            "<table width='100%' cellpadding='0' cellspacing='0' "
            "style='border:1px solid #2c3240; border-radius:6px;'>"
            f"<tr>"
            f"<th style='{header_style}'>Check</th>"
            f"<th style='{header_style}'>Value</th>"
            f"<th style='{header_style}'>Verdict</th>"
            f"</tr>"
            + "".join(diag_rows_html)
            + "</table>"
        )

        # ---- Variable contribution / significance table -----------------
        contrib_html = ""
        if importance_rows:
            sorted_rows = sorted(
                importance_rows,
                key=lambda r: (
                    not bool(r.get("significant", False)),
                    float(r.get("q_value", float("inf")))
                    if np.isfinite(float(r.get("q_value", float("inf"))))
                    else float("inf"),
                    -(float(r.get("delta_r2", 0.0))
                      if np.isfinite(float(r.get("delta_r2", 0.0))) else 0.0),
                ),
            )
            rows_html = []
            for row in sorted_rows:
                label = esc(row.get("label", row.get("feature", "")))
                delta_r2 = float(row.get("delta_r2", float("nan")))
                delta_mse = float(row.get("delta_mse", float("nan")))
                reduced_r2 = float(row.get("reduced_r2", float("nan")))
                p_val = float(row.get("p_value", float("nan")))
                q_val = float(row.get("q_value", float("nan")))
                is_sig = bool(row.get("significant", False))
                status = str(row.get("status", "ok") or "ok")
                if row.get("p_value_upper_bound"):
                    p_html = (
                        f"<b>&le; {1.0/max(int(n_boot) + 1, 1):.3g}</b>"
                        if n_boot > 0 else
                        "<span style='color:#6f7a8e;'>n/a</span>"
                    )
                else:
                    p_html = self._html_p_text(p_val)
                q_html = self._html_q_text(q_val, is_sig)
                if status != "ok" and "failed" in status:
                    badge = self._html_badge("FAILED", "#f5755a", "#2e1c18")
                elif is_sig:
                    badge = self._html_badge("SIG", "#5dd39e", "#1c2e22")
                elif not np.isfinite(p_val):
                    badge = self._html_badge("UNTESTED", "#aab4c5", "#1f242e")
                else:
                    badge = self._html_badge("n.s.", "#aab4c5", "#1f242e")
                name_color = "#5dd39e" if is_sig else "#e9ecf3"
                bold_open = "<b>" if is_sig else ""
                bold_close = "</b>" if is_sig else ""
                rows_html.append(
                    "<tr>"
                    f"<td style='{cell_style} color:{name_color};'>"
                    f"{bold_open}{label}{bold_close}</td>"
                    f"<td style='{mono_style} color:#cfd8e6;'>"
                    f"{self._fmt_stat(delta_r2, 4)}</td>"
                    f"<td style='{mono_style} color:#cfd8e6;'>"
                    f"{self._fmt_stat(delta_mse, 4)}</td>"
                    f"<td style='{mono_style} color:#cfd8e6;'>"
                    f"{self._fmt_stat(reduced_r2, 4)}</td>"
                    f"<td style='{mono_style}'>{p_html}</td>"
                    f"<td style='{mono_style}'>{q_html}</td>"
                    f"<td style='{cell_style}'>{badge}</td>"
                    "</tr>"
                )
            warnings = sorted({
                str(row.get("bootstrap_warning", ""))
                for row in importance_rows if row.get("bootstrap_warning")
            })
            warn_html = ""
            if warnings:
                warn_lines = "<br>".join(esc(w) for w in warnings)
                warn_html = (
                    f"<div style='color:#f5c542; font-size:8.5pt; margin:6px 0 0 0;'>"
                    f"<b>Bootstrap warnings:</b> {warn_lines}</div>"
                )
            contrib_html = (
                "<h3 style='color:#aab4c5; font-size:9pt; font-weight:600;"
                " letter-spacing:0.4px; text-transform:uppercase; margin:14px 0 6px 0;'>"
                "Variable contribution (leave-one-predictor-out)</h3>"
                "<table width='100%' cellpadding='0' cellspacing='0' "
                "style='border:1px solid #2c3240; border-radius:6px;'>"
                f"<tr>"
                f"<th style='{header_style}'>Predictor</th>"
                f"<th style='{header_style}'>&Delta;R<sup>2</sup></th>"
                f"<th style='{header_style}'>&Delta;MSE</th>"
                f"<th style='{header_style}'>Reduced R<sup>2</sup></th>"
                f"<th style='{header_style}'>p (raw)</th>"
                f"<th style='{header_style}'>q (FDR)</th>"
                f"<th style='{header_style}'>Result</th>"
                f"</tr>"
                + "".join(rows_html)
                + "</table>"
                "<div style='color:#6f7a8e; font-size:8.4pt; margin:4px 0 0 0;'>"
                "<b>&Delta;R<sup>2</sup>:</b> drop in fit when a predictor is "
                "removed. <b>p (raw):</b> circular-shift bootstrap "
                "(predictor's contribution under a time-shifted null). "
                "<b>q (FDR):</b> Benjamini-Hochberg correction across predictors."
                + warn_html
                + "</div>"
            )

        # ---- Interpretation ---------------------------------------------
        interp_html = (
            "<h3 style='color:#aab4c5; font-size:9pt; font-weight:600;"
            " letter-spacing:0.4px; text-transform:uppercase; margin:14px 0 6px 0;'>"
            "How to read this</h3>"
            "<div style='color:#cfd8e6; font-size:9pt;'>"
            "Each kernel shows the expected change in the signal locked to a "
            "predictor (event or behavior trace). &Delta;R<sup>2</sup> is the "
            "drop in fit when that predictor is left out &mdash; how much the "
            "model relies on it. The circular-shift bootstrap asks whether "
            "&Delta;R<sup>2</sup> is bigger than what a chance-aligned version "
            "of the same predictor produces; the q-value is the FDR-corrected "
            "verdict across all tested predictors."
            "</div>"
        )

        # ---- Notes ------------------------------------------------------
        notes_bits: List[str] = []
        if dropped_predictors:
            notes_bits.append(
                f"<div style='color:#cfd8e6; font-size:8.6pt; padding:3px 0;'>"
                f"<b style='color:#aab4c5;'>Dropped predictors:</b> "
                f"{esc(', '.join(str(v) for v in dropped_predictors))}"
                f"</div>"
            )
        if dropped_records:
            notes_bits.append(
                f"<div style='color:#cfd8e6; font-size:8.6pt; padding:3px 0;'>"
                f"<b style='color:#aab4c5;'>Dropped recordings:</b> "
                f"{esc(', '.join(str(v) for v in dropped_records))}"
                f"</div>"
            )
        failed = [
            row for row in importance_rows
            if str(row.get("status", "ok") or "ok") != "ok"
        ]
        if failed:
            notes_bits.append(
                f"<div style='color:#cfd8e6; font-size:8.6pt; padding:3px 0;'>"
                f"<b style='color:#f5755a;'>{len(failed)} reduced fit(s) "
                f"failed;</b> see log for details."
                f"</div>"
            )
        notes_html = ""
        if notes_bits:
            notes_html = (
                "<h3 style='color:#aab4c5; font-size:9pt; font-weight:600;"
                " letter-spacing:0.4px; text-transform:uppercase; margin:14px 0 6px 0;'>"
                "Notes</h3>"
                + "".join(notes_bits)
            )

        title_html = (
            "<div style='color:#e9ecf3; font-size:11pt; font-weight:700;"
            " padding-bottom:4px;'>Continuous GLM summary</div>"
        )
        return (
            "<div style='font-family:\"Segoe UI\", sans-serif; color:#e9ecf3;"
            " line-height:1.4;'>"
            + title_html
            + verdict_html
            + meta_html
            + fit_html
            + diag_html
            + contrib_html
            + interp_html
            + notes_html
            + "</div>"
        )

    def _glm_summary_html_from_result(
        self,
        result: GLMResult,
        *,
        file_filter: Optional[str] = None,
        from_cache: Optional[bool] = None,
    ) -> str:
        """Regenerate the rich GLM summary from the result payload.

        Disk/project caches intentionally keep the plain-text summary for
        backwards compatibility, but the GUI regenerates HTML from structured
        result fields so old cache entries and fresh fits render consistently.
        """
        stats = result.stats or {}

        def _combo_text(widget_name: str, fallback: str = "") -> str:
            widget = getattr(self, widget_name, None)
            try:
                return str(widget.currentText())
            except Exception:
                return fallback

        def _spin_value(widget_name: str, fallback: float) -> float:
            widget = getattr(self, widget_name, None)
            try:
                return float(widget.value())
            except Exception:
                return float(fallback)

        used_records = stats.get("used_records", []) or []
        if not isinstance(used_records, list):
            used_records = [str(used_records)]
        used_records = [str(v) for v in used_records if str(v).strip()]
        if not used_records and file_filter:
            used_records = [str(file_filter)]

        used_labels = stats.get("used_labels", []) or []
        if not isinstance(used_labels, list):
            used_labels = [str(used_labels)]
        used_labels = [str(v) for v in used_labels if str(v).strip()]
        if not used_labels:
            used_labels = [self._predictor_label(k) for k in (result.predictor_names or [])]

        dropped_predictors = stats.get("dropped_predictors", []) or []
        if not isinstance(dropped_predictors, list):
            dropped_predictors = [str(dropped_predictors)]
        dropped_records = stats.get("dropped_records", []) or []
        if not isinstance(dropped_records, list):
            dropped_records = [str(dropped_records)]

        valid_samples = stats.get("valid_samples", stats.get("n_samples", 0))
        try:
            valid_samples_i = int(float(valid_samples))
        except Exception:
            valid_samples_i = 0

        cache_flag = bool(stats.get("cache_hit", False)) if from_cache is None else bool(from_cache)
        return self._build_glm_summary_html(
            result=result,
            importance_rows=list(result.feature_importance or []),
            basis_label=str(stats.get("basis_label") or _combo_text("combo_basis", "Raised cosine")),
            regularization_label=str(stats.get("regularization_label") or _combo_text("combo_reg", "Ridge")),
            n_basis=int(float(stats.get("n_basis", _spin_value("spin_n_basis", 8)) or 8)),
            alpha=float(stats.get("alpha", _spin_value("spin_alpha", 1.0)) or 1.0),
            n_boot=int(float(stats.get("n_boot", _spin_value("spin_glm_bootstrap", 0)) or 0)),
            n_jobs=int(float(stats.get("n_jobs", _spin_value("spin_glm_jobs", 1)) or 1)),
            used_records=used_records,
            used_labels=used_labels,
            dropped_predictors=[str(v) for v in dropped_predictors],
            dropped_records=[str(v) for v in dropped_records],
            valid_samples=valid_samples_i,
            scope_label=str(stats.get("scope_label") or file_filter or "all files"),
            from_cache=cache_flag,
        )

    def _render_glm_summary_html(
        self,
        result: GLMResult,
        *,
        file_filter: Optional[str] = None,
        from_cache: Optional[bool] = None,
    ) -> str:
        html = self._glm_summary_html_from_result(result, file_filter=file_filter, from_cache=from_cache)
        self.txt_summary.setHtml(html)
        return html

    def _flmm_curve_summary_lines(
        self,
        result: FLMMResult,
        formula_terms: List[str],
        term_labels: Dict[str, str],
        max_rows: int = 12,
    ) -> List[str]:
        coefficients = result.coefficients or {}
        preferred = ["(Intercept)"] + list(formula_terms)
        seen = set()
        ordered_names: List[str] = []
        for name in preferred:
            coeff_name = self._term_coefficient_name(result, name) if name != "(Intercept)" else "(Intercept)"
            if coeff_name in coefficients and coeff_name not in seen:
                ordered_names.append(coeff_name)
                seen.add(coeff_name)
        for name in coefficients:
            if name not in seen:
                ordered_names.append(name)
                seen.add(name)

        lines = ["Coefficient curve summary:"]
        for name in ordered_names[:max_rows]:
            coeff = np.asarray(coefficients.get(name, np.array([], float)), float)
            mean_coef, mean_abs, peak_abs = _finite_curve_stats(coeff)
            label = self._flmm_readable_label(name, term_labels)
            if not np.isfinite(mean_abs) and not np.isfinite(peak_abs):
                lines.append(f"  {label}: not estimable")
                continue
            lines.append(
                f"  {label}: mean coefficient = {self._fmt_stat(mean_coef, 4)}, "
                f"mean abs = {self._fmt_stat(mean_abs, 4)}, peak abs = {self._fmt_stat(peak_abs, 4)}"
            )
        if len(ordered_names) > max_rows:
            lines.append(f"  ... {len(ordered_names) - max_rows} more coefficient curve(s)")
        return lines

    def _flmm_interpretation_lines(
        self,
        result: FLMMResult,
        rows: List[Dict[str, Any]],
        use_fixed_fallback: bool,
        max_rows: int = 5,
    ) -> List[str]:
        lines = [
            "Plain-language interpretation:",
            "  Predictors are standardized before fitting; each coefficient curve is the expected change in the aligned response for a +1 SD increase in that predictor.",
            "  Positive coefficients mean higher predictor values are associated with a higher response at that time; negative coefficients mean a lower response.",
        ]
        if use_fixed_fallback:
            lines.append(
                "  This is a single-file functional fixed-effect fallback, so interpret effects as within-recording trial associations, not population mixed-effect inference."
            )

        usable = [
            row for row in rows
            if np.isfinite(float(row.get("mean_abs_coefficient", float("nan"))))
        ]
        if not usable:
            lines.append("  No interpretable coefficient contribution was available for the selected variables.")
            return lines

        tvec = np.asarray(result.tvec, float)
        for row in usable[:max_rows]:
            feature = str(row.get("feature", ""))
            label = str(row.get("label", feature) or feature)
            coeff = self._term_coefficient_curve(result, feature)
            if coeff is None:
                continue
            vals = np.asarray(coeff, float)
            finite = np.isfinite(vals)
            if not np.any(finite):
                continue
            finite_idx = np.where(finite)[0]
            peak_local = int(np.nanargmax(np.abs(vals[finite])))
            peak_idx = int(finite_idx[peak_local])
            peak_val = float(vals[peak_idx])
            direction = "higher" if peak_val > 0 else "lower"
            peak_time = float(tvec[peak_idx]) if peak_idx < tvec.size and np.isfinite(tvec[peak_idx]) else float("nan")
            q_value = float(row.get("q_value", float("nan")))
            sig_text = (
                f" This coefficient curve is significant after FDR correction (q = {q_value:.3g})."
                if bool(row.get("significant", False)) and np.isfinite(q_value)
                else " This coefficient curve was not significant after FDR correction."
                if row.get("significance_status") == "ok"
                else ""
            )
            time_text = f" at {peak_time:+.2f} s" if np.isfinite(peak_time) else ""
            lines.append(
                f"  {label}: strongest association is {direction} response{time_text} "
                f"(peak coefficient {peak_val:.3g}, mean abs {float(row.get('mean_abs_coefficient', float('nan'))):.3g}).{sig_text}"
            )
        return lines

    def _compute_flmm_permutation_contribution(
        self,
        mat: np.ndarray,
        tvec: np.ndarray,
        design: Dict[str, np.ndarray],
        formula: str,
        random_eff: str,
        group_var: str,
        nknots: Optional[int],
        full_result: FLMMResult,
        formula_terms: List[str],
        term_labels: Dict[str, str],
        n_perm: int = 200,
        use_fastfmm: bool = True,
    ) -> List[Dict[str, Any]]:
        """Permutation test for variable contribution.

        For each predictor independently, permute that column's trial values,
        refit the functional model, and record the L2 norm of its coefficient
        curve. Compare to the observed L2 norm; p-value = (1 + n_exceed) / (N+1).
        Apply Benjamini-Hochberg FDR across predictors.

        Permuting a single column at a time (rather than shuffling the response
        matrix) gives a per-predictor null that respects the joint design, so
        we get correct decoupled p-values even when predictors are correlated.
        """
        if not formula_terms or n_perm <= 0:
            return []
        # Pre-compute observed L2 norms.
        obs_stats: Dict[str, float] = {}
        for term in formula_terms:
            coeff = self._term_coefficient_curve(full_result, term)
            obs_stats[term] = (
                float(np.nansum(np.asarray(coeff, float) ** 2))
                if coeff is not None and np.asarray(coeff).size else float("nan")
            )

        rng = np.random.default_rng()
        n_trials = int(mat.shape[0])
        null_stats: Dict[str, List[float]] = {t: [] for t in formula_terms}
        failed_terms: List[str] = []

        # Cancel/progress plumbing.
        total = max(1, int(n_perm) * len(formula_terms))
        label = f"FLMM permutation test ({len(formula_terms)} pred. x {n_perm} iter)"
        self._progress_start(label, total)
        step = 0
        self._batch_cancel_requested = False

        def _fit_perm(perm_design: Dict[str, np.ndarray]) -> Optional[FLMMResult]:
            try:
                if use_fastfmm and self._flmm.available:
                    return self._flmm.fit(
                        mat, tvec, perm_design,
                        formula_fixed=formula,
                        random_effects=random_eff,
                        group_var=group_var,
                        nknots_min=nknots,
                        num_boots=0,
                    )
                return self._flmm._fit_fixed_effect_functional(
                    mat, tvec, perm_design,
                    formula_fixed=formula,
                    group_var=group_var,
                    reason="permutation",
                )
            except Exception as exc:
                _LOG.debug("FLMM perm fit failed: %s", exc)
                return None

        for term in formula_terms:
            if self._batch_cancel_requested:
                break
            base_col = np.asarray(design.get(term, np.array([], float)), float).reshape(-1)
            if base_col.size != n_trials:
                failed_terms.append(term)
                step += n_perm
                self._progress_update(step, label)
                continue
            for k in range(int(n_perm)):
                if self._batch_cancel_requested:
                    break
                perm = rng.permutation(n_trials)
                perm_design = dict(design)
                perm_design[term] = base_col[perm]
                perm_res = _fit_perm(perm_design)
                if perm_res is not None:
                    coeff = self._term_coefficient_curve(perm_res, term)
                    if coeff is not None and np.asarray(coeff).size:
                        stat = float(np.nansum(np.asarray(coeff, float) ** 2))
                        if np.isfinite(stat):
                            null_stats[term].append(stat)
                step += 1
                if step == total or step % max(1, len(formula_terms)) == 0:
                    self._progress_update(step, label)

        p_values: List[float] = []
        rows: List[Dict[str, Any]] = []
        for term in formula_terms:
            obs = obs_stats.get(term, float("nan"))
            nulls = np.asarray(null_stats.get(term, []), float)
            n_null = int(nulls.size)
            if not np.isfinite(obs) or n_null == 0:
                p = float("nan")
                exceed = 0
            else:
                exceed = int(np.sum(nulls >= obs))
                p = float((1 + exceed) / (n_null + 1))
            p_values.append(p)
            rows.append({
                "feature": term,
                "label": term_labels.get(term, term),
                "obs_l2sq": obs,
                "perm_n": n_null,
                "perm_exceed": exceed,
                "p_perm": p,
                "p_perm_upper_bound": bool(np.isfinite(p) and exceed == 0),
                "perm_failed": term in failed_terms,
            })
        q_values = _bh_fdr(p_values)
        for row, q in zip(rows, q_values):
            row["q_perm"] = float(q) if np.isfinite(q) else float("nan")
            row["significant_perm"] = bool(
                np.isfinite(row["q_perm"]) and row["q_perm"] < 0.05
            )
        return rows

    @staticmethod
    def _merge_flmm_permutation_rows(
        importance_rows: List[Dict[str, Any]],
        perm_rows: List[Dict[str, Any]],
    ) -> None:
        perm_by_feature = {str(r.get("feature", "")): r for r in perm_rows}
        for row in importance_rows:
            feature = str(row.get("feature", ""))
            perm = perm_by_feature.get(feature)
            if not perm:
                continue
            for key in ("p_perm", "q_perm", "obs_l2sq", "perm_n",
                        "perm_exceed", "p_perm_upper_bound",
                        "significant_perm", "perm_failed"):
                if key in perm:
                    row[key] = perm[key]

    def _compute_flmm_leave_one_out(
        self,
        mat: np.ndarray,
        tvec: np.ndarray,
        design: Dict[str, np.ndarray],
        formula: str,
        random_eff: str,
        group_var: str,
        nknots: Optional[int],
        full_result: FLMMResult,
        term_labels: Dict[str, str],
    ) -> List[Dict[str, Any]]:
        terms = [term for term in self._simple_formula_terms(formula) if term in design and term != group_var]
        if not terms:
            return []
        full_aic = float(full_result.aic) if full_result.aic is not None else float("nan")
        rows: List[Dict[str, Any]] = []
        self._progress_start("FLMM leave-one-out", len(terms))
        for term in terms:
            reduced_terms = [name for name in terms if name != term]
            reduced_formula = "Y.obs ~ " + " + ".join(reduced_terms) if reduced_terms else "Y.obs ~ 1"
            row: Dict[str, Any] = {
                "feature": term,
                "label": term_labels.get(term, term),
                "full_aic": full_aic,
                "reduced_aic": float("nan"),
                "delta_aic": float("nan"),
                "mean_abs_coefficient": self._term_mean_abs_coefficient(full_result, term),
                "status": "ok",
            }
            try:
                reduced = TrialFLMM().fit(
                    mat,
                    tvec,
                    design,
                    formula_fixed=reduced_formula,
                    random_effects=random_eff,
                    group_var=group_var,
                    nknots_min=nknots,
                    num_boots=0,
                )
                reduced_aic = float(reduced.aic) if reduced.aic is not None else float("nan")
                row["reduced_aic"] = reduced_aic
                if np.isfinite(full_aic) and np.isfinite(reduced_aic):
                    row["delta_aic"] = reduced_aic - full_aic
            except Exception as exc:
                row["status"] = f"failed: {exc}"
            rows.append(row)
            self._progress_update(len(rows), "FLMM leave-one-out")
        rows.sort(key=lambda item: (
            np.isfinite(item.get("delta_aic", np.nan)),
            float(item.get("delta_aic", -np.inf)) if np.isfinite(item.get("delta_aic", np.nan)) else -np.inf,
            float(item.get("mean_abs_coefficient", -np.inf)) if np.isfinite(item.get("mean_abs_coefficient", np.nan)) else -np.inf,
        ), reverse=True)
        return rows

    # ------------------------------------------------------------------
    # Slots
    # ------------------------------------------------------------------

    def _on_model_type_changed(self, index: int):
        is_glm = (index == 0)
        self.grp_glm.setVisible(is_glm)
        self.grp_flmm.setVisible(not is_glm)
        self.plot_kernel.setVisible(is_glm)
        self.plot_coeff.setVisible(not is_glm)
        if is_glm:
            self.lbl_flmm_status.setText("")
            if hasattr(self, "tabs_workspace"):
                self.tabs_workspace.setCurrentWidget(self.plot_kernel.parentWidget())
        elif hasattr(self, "tabs_workspace"):
            self.tabs_workspace.setCurrentWidget(self.plot_coeff.parentWidget())

        # Check FLMM availability
        if not is_glm:
            if self._flmm.available:
                self.lbl_flmm_status.setText("R + fastFMM detected.")
                self.lbl_flmm_status.setStyleSheet(
                    "color: #15803d;" if self._app_theme_mode == "light" else "color: #6bdb74;"
                )
            else:
                self.lbl_flmm_status.setText(
                    "R or fastFMM not found. Install R and the fastFMM package, "
                    "then install rpy2 (pip install rpy2)."
                )
                self.lbl_flmm_status.setStyleSheet(
                    "color: #b45309;" if self._app_theme_mode == "light" else "color: #f5a97f;"
                )
        self._save_temporal_settings()

    def _on_add_predictor(self):
        key = ""
        if hasattr(self, "combo_available_predictors"):
            data = self.combo_available_predictors.currentData(QtCore.Qt.ItemDataRole.UserRole)
            key = str(data or "").strip()
        if not key:
            self.statusMessage.emit("No predictor is available yet. Load or compute behavior/events first.", 5000)
            return
        if self._add_predictor_item(key):
            self._save_temporal_settings()
            self.statusMessage.emit(f"Added predictor: {self._predictor_label(key)}", 3000)

    def _on_remove_predictor(self):
        sel = self.list_predictors.currentRow()
        if sel >= 0:
            self.list_predictors.takeItem(sel)
            self._save_temporal_settings()

    # ------------------------------------------------------------------
    # Scope (Active / All / Per-file batch) handling
    # ------------------------------------------------------------------

    def _refresh_file_widgets(self) -> None:
        """Populate combo_active_file and list_files from currently-loaded recordings."""
        if not hasattr(self, "combo_active_file"):
            return
        ids: List[Tuple[str, str]] = []
        for idx, proc in enumerate(self._processed_trials):
            file_id = self._proc_file_id(proc, fallback=f"file_{idx + 1}")
            label = file_id
            ids.append((file_id, label))

        # Combo
        self.combo_active_file.blockSignals(True)
        try:
            self.combo_active_file.clear()
            for fid, label in ids:
                self.combo_active_file.addItem(label, fid)
            if not ids:
                self.combo_active_file.addItem("No files loaded", "")
            target = self._active_file_id
            if target:
                idx = self.combo_active_file.findData(target)
                if idx >= 0:
                    self.combo_active_file.setCurrentIndex(idx)
                elif ids:
                    self.combo_active_file.setCurrentIndex(0)
                    self._active_file_id = ids[0][0]
            elif ids:
                self.combo_active_file.setCurrentIndex(0)
                self._active_file_id = ids[0][0]
        finally:
            self.combo_active_file.blockSignals(False)

        # List
        if hasattr(self, "list_files"):
            self.list_files.blockSignals(True)
            try:
                self.list_files.clear()
                for fid, label in ids:
                    item = QtWidgets.QListWidgetItem(label)
                    item.setData(QtCore.Qt.ItemDataRole.UserRole, fid)
                    if fid in self._glm_results_by_file:
                        item.setText(f"{label}    [fit cached]")
                        item.setForeground(QtGui.QColor("#6bdb74"))
                    self.list_files.addItem(item)
                # Sync selection
                for i in range(self.list_files.count()):
                    if self.list_files.item(i).data(QtCore.Qt.ItemDataRole.UserRole) == self._active_file_id:
                        self.list_files.setCurrentRow(i)
                        break
            finally:
                self.list_files.blockSignals(False)
        self._update_fit_state_label()

    def _step_active_file(self, delta: int) -> None:
        if not hasattr(self, "combo_active_file"):
            return
        n = self.combo_active_file.count()
        if n == 0:
            return
        idx = max(0, min(n - 1, self.combo_active_file.currentIndex() + int(delta)))
        self.combo_active_file.setCurrentIndex(idx)

    def _on_active_file_changed(self, *_):
        if not hasattr(self, "combo_active_file"):
            return
        data = self.combo_active_file.currentData()
        if isinstance(data, str) and data:
            old_file_id = str(self._active_file_id or "")
            if old_file_id and old_file_id != data:
                self._remember_current_predictors_for_file(old_file_id)
            self._active_file_id = data
            if old_file_id != data:
                self._restore_predictors_for_active_file(allow_default=True)
                self._save_temporal_settings()
            # Mirror selection in the list
            if hasattr(self, "list_files"):
                for i in range(self.list_files.count()):
                    if self.list_files.item(i).data(QtCore.Qt.ItemDataRole.UserRole) == data:
                        if self.list_files.currentRow() != i:
                            self.list_files.blockSignals(True)
                            self.list_files.setCurrentRow(i)
                            self.list_files.blockSignals(False)
                        break
            # Push selection back into the host postprocessing panel so the
            # PSTH file picker and the Temporal scope file always agree.
            try:
                host = self.parent()
                while host is not None and not hasattr(host, "combo_individual_file"):
                    host = host.parent() if hasattr(host, "parent") else None
                combo = getattr(host, "combo_individual_file", None) if host is not None else None
                if combo is not None:
                    idx = combo.findText(data)
                    if idx >= 0 and combo.currentIndex() != idx:
                        combo.blockSignals(True)
                        combo.setCurrentIndex(idx)
                        combo.blockSignals(False)
                        if hasattr(host, "_rerender_visual_from_cache"):
                            host._rerender_visual_from_cache()
            except Exception:
                pass
            # If we are in Active scope and a fit is cached for this file, render it.
            if self._fit_mode == "active":
                cached = self._glm_results_by_file.get(self._active_file_id)
                if cached is not None:
                    self._glm_result = cached
                    self._render_glm_summary_html(cached, file_filter=self._active_file_id)
                    self._plot_glm_kernels(cached)
                    self._plot_glm_fit(cached)
                    self._plot_glm_illustration(cached)
                    self._plot_feature_importance(
                        cached.feature_importance or [],
                        value_key="delta_r2",
                        title=f"GLM contribution - {self._active_file_id}",
                        y_label="Drop in R^2",
                    )
        self._update_fit_state_label()

    def _on_file_list_selection_changed(self) -> None:
        if not hasattr(self, "list_files"):
            return
        item = self.list_files.currentItem()
        if item is None:
            return
        fid = item.data(QtCore.Qt.ItemDataRole.UserRole)
        if isinstance(fid, str) and fid and fid != self._active_file_id:
            idx = self.combo_active_file.findData(fid)
            if idx >= 0:
                self.combo_active_file.setCurrentIndex(idx)

    def _on_fit_scope_changed(self, *_):
        data = self.combo_fit_scope.currentData()
        if isinstance(data, str):
            self._fit_mode = data
        self._update_fit_state_label()

    def _update_fit_state_label(self) -> None:
        if not hasattr(self, "lbl_fit_state"):
            return
        n_cached = len(self._glm_results_by_file)
        n_files = len(self._processed_trials)
        bits = []
        if self._glm_result is not None:
            bits.append(f"R^2 = {self._glm_result.r2:.3f}")
        if n_cached > 0:
            bits.append(f"{n_cached}/{n_files} files fit")
        if not bits:
            bits.append("No fit yet")
        self.lbl_fit_state.setText(" | ".join(bits))

    def _on_clear_cached_fits(self) -> None:
        self._glm_results_by_file.clear()
        self._flmm_results_by_file.clear()
        self._fit_summary_by_file.clear()
        self._group_glm_summary = {}
        removed_disk = 0
        try:
            cache_dir = self._glm_cache_dir()
            for name in os.listdir(cache_dir):
                if name.endswith(".json") or name.endswith(".json.tmp"):
                    os.remove(os.path.join(cache_dir, name))
                    removed_disk += 1
        except Exception as exc:
            _LOG.debug("Could not clear GLM disk cache: %s", exc)
        self._refresh_file_widgets()
        if hasattr(self, "lbl_batch_status"):
            self.lbl_batch_status.setText(f"Cleared cached fits ({removed_disk} disk cache file(s)).")
        if hasattr(self, "lbl_group_summary"):
            self.lbl_group_summary.setText(
                "Run a Per-file batch fit to populate the Group view."
            )
        if hasattr(self, "plot_group_kernels"):
            self.plot_group_kernels.clear()
        if hasattr(self, "plot_group_importance"):
            self.plot_group_importance.clear()

    def _on_cancel_clicked(self):
        self._batch_cancel_requested = True
        self.btn_cancel.setEnabled(False)
        self.statusMessage.emit("Cancelling current batch...", 3000)

    # ------------------------------------------------------------------
    # Export menu handlers
    # ------------------------------------------------------------------

    def _pick_save_path(self, default_name: str, filters: str) -> str:
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Save", default_name, filters,
        )
        return path or ""

    def _csv_quote(self, value: object) -> str:
        text = "" if value is None else str(value)
        if any(ch in text for ch in (",", "\"", "\n")):
            text = '"' + text.replace('"', '""') + '"'
        return text

    def _export_kernels_csv(self) -> None:
        result = self._glm_result
        if result is None:
            self.statusMessage.emit("No active GLM fit to export.", 5000)
            return
        path = self._pick_save_path("temporal_kernels.csv", "CSV files (*.csv)")
        if not path:
            return
        try:
            tvec = np.asarray(result.kernel_tvec, float)
            cols = [("time_s", tvec)]
            for name in result.predictor_names:
                kern = np.asarray(result.kernels.get(name, np.zeros_like(tvec)), float)
                cols.append((self._predictor_label(name), kern))
            n_rows = max((c[1].size for c in cols), default=0)
            with open(path, "w", encoding="utf-8", newline="") as fh:
                fh.write(",".join(self._csv_quote(name) for name, _ in cols) + "\n")
                for i in range(n_rows):
                    fh.write(",".join(
                        self._csv_quote(f"{c[1][i]:.6g}" if i < c[1].size else "")
                        for c in cols
                    ) + "\n")
            self.statusMessage.emit(f"Saved kernels to {os.path.basename(path)}", 5000)
        except Exception as exc:
            _LOG.error("Export kernels CSV failed: %s", exc)
            self.statusMessage.emit(f"Export failed: {exc}", 7000)

    def _export_importance_csv(self) -> None:
        if self.combo_model_type.currentIndex() == 1 and self._flmm_result is not None:
            rows = list(self._flmm_result.feature_importance or [])
            default_name = "temporal_flmm_importance.csv"
        else:
            result = self._glm_result
            rows = list(result.feature_importance or []) if result is not None else []
            default_name = "temporal_glm_importance.csv"
        if not rows:
            self.statusMessage.emit("No feature importance table to export.", 5000)
            return
        path = self._pick_save_path(default_name, "CSV files (*.csv)")
        if not path:
            return
        try:
            # Stable column order; pull all keys present across rows.
            preferred = ["feature", "label", "full_r2", "reduced_r2", "delta_r2",
                         "delta_mse", "contribution_pct", "p_value", "significant",
                         "q_value", "significant_bins", "significant_fraction",
                         "bootstrap_n", "full_aic", "reduced_aic", "delta_aic",
                         "mean_abs_coefficient", "peak_abs_coefficient", "status"]
            seen = list(preferred)
            for row in rows:
                for k in row.keys():
                    if k not in seen:
                        seen.append(k)
            with open(path, "w", encoding="utf-8", newline="") as fh:
                fh.write(",".join(self._csv_quote(k) for k in seen) + "\n")
                for row in rows:
                    fh.write(",".join(self._csv_quote(row.get(k, "")) for k in seen) + "\n")
            self.statusMessage.emit(f"Saved importance to {os.path.basename(path)}", 5000)
        except Exception as exc:
            _LOG.error("Export importance CSV failed: %s", exc)
            self.statusMessage.emit(f"Export failed: {exc}", 7000)

    def _export_summary_txt(self) -> None:
        text = self.txt_summary.toPlainText() if hasattr(self, "txt_summary") else ""
        if not text:
            self.statusMessage.emit("No summary text available.", 5000)
            return
        path = self._pick_save_path("temporal_summary.txt", "Text files (*.txt)")
        if not path:
            return
        try:
            with open(path, "w", encoding="utf-8") as fh:
                fh.write(text)
            self.statusMessage.emit(f"Saved summary to {os.path.basename(path)}", 5000)
        except Exception as exc:
            self.statusMessage.emit(f"Export failed: {exc}", 7000)

    def _export_group_kernels_csv(self) -> None:
        results = self._glm_results_by_file
        if not results:
            self.statusMessage.emit("Run a Per-file batch fit first.", 5000)
            return
        path = self._pick_save_path("temporal_group_kernels.csv", "CSV files (*.csv)")
        if not path:
            return
        try:
            # Long-format: file_id, time_s, predictor, kernel_value
            with open(path, "w", encoding="utf-8", newline="") as fh:
                fh.write("file_id,time_s,predictor,kernel_value\n")
                for file_id, res in results.items():
                    tvec = np.asarray(res.kernel_tvec, float)
                    for name in res.predictor_names:
                        kern = np.asarray(res.kernels.get(name, np.zeros_like(tvec)), float)
                        label = self._predictor_label(name)
                        n = min(tvec.size, kern.size)
                        for i in range(n):
                            fh.write(
                                f"{self._csv_quote(file_id)},"
                                f"{tvec[i]:.6g},"
                                f"{self._csv_quote(label)},"
                                f"{kern[i]:.6g}\n"
                            )
            self.statusMessage.emit(f"Saved group kernels to {os.path.basename(path)}", 5000)
        except Exception as exc:
            self.statusMessage.emit(f"Export failed: {exc}", 7000)

    def _export_group_importance_csv(self) -> None:
        rows = list((self._group_glm_summary or {}).get("importance", []) or [])
        if not rows:
            self.statusMessage.emit("No group importance summary available.", 5000)
            return
        path = self._pick_save_path("temporal_group_importance.csv", "CSV files (*.csv)")
        if not path:
            return
        try:
            cols = ["feature", "label", "delta_r2", "delta_r2_sem", "n_animals"]
            with open(path, "w", encoding="utf-8", newline="") as fh:
                fh.write(",".join(self._csv_quote(c) for c in cols) + "\n")
                for row in rows:
                    fh.write(",".join(self._csv_quote(row.get(c, "")) for c in cols) + "\n")
            self.statusMessage.emit(f"Saved group importance to {os.path.basename(path)}", 5000)
        except Exception as exc:
            self.statusMessage.emit(f"Export failed: {exc}", 7000)

    def _export_state_json(self) -> None:
        import json
        try:
            state = self.serialize_state()
        except Exception as exc:
            self.statusMessage.emit(f"Could not snapshot state: {exc}", 7000)
            return
        path = self._pick_save_path("temporal_state.json", "JSON files (*.json)")
        if not path:
            return
        try:
            with open(path, "w", encoding="utf-8") as fh:
                json.dump(state, fh, indent=2)
            n = len(state.get("glm_results_by_file", {}))
            self.statusMessage.emit(f"Saved {n} cached fit(s) to {os.path.basename(path)}", 5000)
        except Exception as exc:
            self.statusMessage.emit(f"Export failed: {exc}", 7000)

    def _current_workspace_plot(self) -> Optional[pg.PlotWidget]:
        if not hasattr(self, "tabs_workspace"):
            return None
        page = self.tabs_workspace.currentWidget()
        if page is None:
            return None
        for pw in page.findChildren(pg.PlotWidget):
            return pw
        return None

    def _export_current_plot(self, fmt: str = "png") -> None:
        pw = self._current_workspace_plot()
        if pw is None:
            self.statusMessage.emit("No plot is visible on the current tab.", 5000)
            return
        tab_name = self.tabs_workspace.tabText(self.tabs_workspace.currentIndex()).lower().replace(" ", "_")
        ext = "svg" if fmt == "svg" else "png"
        path = self._pick_save_path(f"temporal_{tab_name}.{ext}",
                                    f"{ext.upper()} files (*.{ext})")
        if not path:
            return
        try:
            from pyqtgraph.exporters import ImageExporter, SVGExporter
            scene = pw.getPlotItem()
            if fmt == "svg":
                exp = SVGExporter(scene)
            else:
                exp = ImageExporter(scene)
                # Pixel-doubled output for crispness.
                try:
                    exp.parameters()["width"] = max(int(pw.width() * 2), 1200)
                except Exception:
                    pass
            exp.export(path)
            self.statusMessage.emit(f"Saved plot to {os.path.basename(path)}", 5000)
        except Exception as exc:
            _LOG.error("Plot export failed: %s", exc)
            self.statusMessage.emit(f"Plot export failed: {exc}", 7000)

    # ------------------------------------------------------------------
    # Publication-quality figure export (matplotlib, PDF + 300 DPI PNG)
    # ------------------------------------------------------------------

    @staticmethod
    def _pub_rcparams() -> Dict[str, Any]:
        """Nature-style matplotlib defaults: clean, minimal, readable at small size."""
        return {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 7.0,
            "axes.titlesize": 8.0,
            "axes.titleweight": "bold",
            "axes.labelsize": 7.0,
            "axes.labelweight": "regular",
            "axes.linewidth": 0.6,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "xtick.labelsize": 6.5,
            "ytick.labelsize": 6.5,
            "xtick.major.width": 0.6,
            "ytick.major.width": 0.6,
            "xtick.major.size": 3.0,
            "ytick.major.size": 3.0,
            "xtick.direction": "out",
            "ytick.direction": "out",
            "legend.fontsize": 6.5,
            "legend.frameon": False,
            "lines.linewidth": 1.1,
            "lines.solid_capstyle": "round",
            "figure.dpi": 100,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.05,
            "pdf.fonttype": 42,  # TrueType embed - editable in Illustrator
            "ps.fonttype": 42,
        }

    def _export_publication_figures(self) -> None:
        try:
            import matplotlib  # noqa: F401
        except Exception:
            self.statusMessage.emit(
                "matplotlib is required for publication figure export. Install it with: pip install matplotlib",
                8000,
            )
            return

        out_dir = QtWidgets.QFileDialog.getExistingDirectory(
            self, "Choose folder for publication figures", "",
        )
        if not out_dir:
            return
        if not os.path.isdir(out_dir):
            self.statusMessage.emit("Selected folder is not writable.", 5000)
            return

        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except Exception as exc:
            self.statusMessage.emit(f"matplotlib import failed: {exc}", 7000)
            return

        saved: List[str] = []
        with plt.rc_context(self._pub_rcparams()):
            # GLM kernels
            if self._glm_result is not None:
                try:
                    p_kern = self._pub_save_glm_kernels(plt, out_dir)
                    if p_kern:
                        saved.append(p_kern)
                except Exception as exc:
                    _LOG.error("publication kernels export failed: %s", exc)

                try:
                    p_imp = self._pub_save_importance(plt, out_dir)
                    if p_imp:
                        saved.append(p_imp)
                except Exception as exc:
                    _LOG.error("publication importance export failed: %s", exc)

            # FLMM coefficients
            if self._flmm_result is not None:
                try:
                    p_coef = self._pub_save_flmm_coefficients(plt, out_dir)
                    if p_coef:
                        saved.append(p_coef)
                except Exception as exc:
                    _LOG.error("publication coefficients export failed: %s", exc)

            # Group kernels
            if self._glm_results_by_file:
                try:
                    p_group = self._pub_save_group_kernels(plt, out_dir)
                    if p_group:
                        saved.append(p_group)
                except Exception as exc:
                    _LOG.error("publication group export failed: %s", exc)

        if saved:
            self.statusMessage.emit(
                f"Saved {len(saved)} publication figure(s) to {os.path.basename(out_dir)}.",
                5000,
            )
        else:
            self.statusMessage.emit(
                "No figures available to export. Fit a model first.", 5000,
            )

    def _pub_save_glm_kernels(self, plt, out_dir: str) -> str:
        result = self._glm_result
        if result is None or not result.predictor_names:
            return ""
        fig, ax = plt.subplots(figsize=(3.5, 2.6))
        t = np.asarray(result.kernel_tvec, float)
        importance = {row.get("feature"): row for row in (result.feature_importance or [])}
        for name in result.predictor_names:
            kern = np.asarray(result.kernels.get(name, []), float)
            if kern.size == 0:
                continue
            color = self._kernel_color(name)
            row = importance.get(name) or {}
            is_sig = bool(row.get("significant", False))
            label = self._predictor_label(name)
            if is_sig:
                q = float(row.get("q_value", float("nan")))
                label = f"{label}* (q={q:.2g})" if np.isfinite(q) else f"{label}*"
            ax.plot(t, kern, color=color, lw=1.2 if is_sig else 0.9, label=label,
                    alpha=1.0 if is_sig else 0.85)
            lo = (result.kernel_ci_lower or {}).get(name)
            hi = (result.kernel_ci_upper or {}).get(name)
            if lo is not None and hi is not None:
                lo_arr = np.asarray(lo, float)
                hi_arr = np.asarray(hi, float)
                if lo_arr.size == t.size and hi_arr.size == t.size:
                    ax.fill_between(t, lo_arr, hi_arr, color=color, alpha=0.20,
                                    linewidth=0)
        ax.axhline(0, color="0.6", lw=0.5, ls="--")
        ax.axvline(0, color="0.6", lw=0.5, ls="--")
        ax.set_xlabel("Time from event (s)")
        ax.set_ylabel("Kernel weight (signal units)")
        ax.set_title("GLM kernels")
        if len(result.predictor_names) <= 8:
            ax.legend(loc="best", fontsize=5.5)
        base = os.path.join(out_dir, "glm_kernels")
        fig.savefig(base + ".pdf"); fig.savefig(base + ".png")
        plt.close(fig)
        return base + ".pdf"

    def _pub_save_importance(self, plt, out_dir: str) -> str:
        result = self._glm_result
        if result is None or not result.feature_importance:
            return ""
        rows = [r for r in result.feature_importance
                if np.isfinite(float(r.get("delta_r2", float("nan"))))]
        if not rows:
            return ""
        rows = sorted(rows, key=lambda r: float(r.get("delta_r2", 0.0)))[-25:]
        names = [self._predictor_label(r.get("feature", "")) for r in rows]
        vals = np.asarray([float(r.get("delta_r2", 0.0)) for r in rows], float)
        sig = np.asarray([bool(r.get("significant", False)) for r in rows], bool)
        fig, ax = plt.subplots(figsize=(3.5, max(1.5, 0.18 * len(rows) + 0.6)))
        y = np.arange(len(rows))
        colors = ["#5dd39e" if s else "#a0aec0" for s in sig]
        ax.barh(y, vals, color=colors, edgecolor="0.2", linewidth=0.3)
        for i, row in enumerate(rows):
            q = float(row.get("q_value", float("nan")))
            if np.isfinite(q) and row.get("significant"):
                ax.text(float(vals[i]) + max(vals.max(), 1e-9) * 0.02, i,
                        f"q={q:.2g}", va="center", fontsize=5.5, color="#1c2e22")
        ax.set_yticks(y)
        ax.set_yticklabels([self._compact_feature_label(n, 30) for n in names])
        ax.axvline(0, color="0.6", lw=0.5)
        ax.set_xlabel("Delta R^2 (predictor contribution)")
        ax.set_title("Variable contribution (leave-one-predictor-out)")
        base = os.path.join(out_dir, "glm_importance")
        fig.savefig(base + ".pdf"); fig.savefig(base + ".png")
        plt.close(fig)
        return base + ".pdf"

    def _pub_save_flmm_coefficients(self, plt, out_dir: str) -> str:
        result = self._flmm_result
        if result is None or not result.coefficients:
            return ""
        t = np.asarray(result.tvec, float)
        term_labels = self._flmm_term_labels(result)
        sig_by_coeff = self._flmm_significance_by_coeff(result)
        items = [(k, v) for k, v in result.coefficients.items()
                 if "(Intercept)" not in str(k)]
        if not items:
            return ""
        fig, ax = plt.subplots(figsize=(3.5, 2.6))
        for name, coeff in items:
            color = self._kernel_color(name)
            row = sig_by_coeff.get(str(name), {})
            is_sig = bool(row.get("significant", False))
            is_sig_perm = bool(row.get("significant_perm", False))
            sig_any = is_sig or is_sig_perm
            label = term_labels.get(name, name)
            if sig_any:
                pieces = []
                if is_sig and np.isfinite(float(row.get("q_value", float("nan")))):
                    pieces.append(f"q={float(row['q_value']):.2g}")
                if is_sig_perm and np.isfinite(float(row.get("q_perm", float("nan")))):
                    pieces.append(f"q_perm={float(row['q_perm']):.2g}")
                tag = ", ".join(pieces)
                label = f"{label}* ({tag})" if tag else f"{label}*"
            ax.plot(t, np.asarray(coeff, float), color=color,
                    lw=1.2 if sig_any else 0.9, label=label,
                    alpha=1.0 if sig_any else 0.85)
            jlo = result.joint_ci_lower.get(name)
            jhi = result.joint_ci_upper.get(name)
            if jlo is not None and jhi is not None:
                lo_arr = np.asarray(jlo, float); hi_arr = np.asarray(jhi, float)
                if lo_arr.size == t.size and hi_arr.size == t.size:
                    ax.fill_between(t, lo_arr, hi_arr, color=color, alpha=0.18,
                                    linewidth=0)
        ax.axhline(0, color="0.6", lw=0.5, ls="--")
        ax.axvline(0, color="0.6", lw=0.5, ls="--")
        ax.set_xlabel("Time from event (s)")
        ax.set_ylabel("Coefficient (per +1 SD of predictor)")
        ax.set_title("FLMM coefficient curves")
        if len(items) <= 8:
            ax.legend(loc="best", fontsize=5.5)
        base = os.path.join(out_dir, "flmm_coefficients")
        fig.savefig(base + ".pdf"); fig.savefig(base + ".png")
        plt.close(fig)
        return base + ".pdf"

    def _pub_save_group_kernels(self, plt, out_dir: str) -> str:
        results = self._glm_results_by_file
        if not results:
            return ""
        predictor_lists = [list(r.predictor_names) for r in results.values()]
        common = set(predictor_lists[0])
        for lst in predictor_lists[1:]:
            common &= set(lst)
        if not common:
            return ""
        ref = next(iter(results.values()))
        ref_t = np.asarray(ref.kernel_tvec, float)
        fig, ax = plt.subplots(figsize=(3.5, 2.6))
        n_animals = len(results)
        for key in sorted(common):
            stack = []
            for r in results.values():
                kern = np.asarray((r.kernels or {}).get(key, []), float)
                t = np.asarray(r.kernel_tvec, float)
                if kern.size == ref_t.size:
                    stack.append(kern)
                elif t.size and kern.size == t.size:
                    stack.append(np.interp(ref_t, t, kern, left=np.nan, right=np.nan))
            if not stack:
                continue
            arr = np.vstack(stack)
            mean = np.nanmean(arr, axis=0)
            sem = np.nanstd(arr, axis=0, ddof=1) / np.sqrt(max(arr.shape[0], 1))
            color = self._kernel_color(key)
            ax.plot(ref_t, mean, color=color, lw=1.1,
                    label=f"{self._predictor_label(key)} (n={arr.shape[0]})")
            ax.fill_between(ref_t, mean - sem, mean + sem, color=color, alpha=0.20,
                            linewidth=0)
        ax.axhline(0, color="0.6", lw=0.5, ls="--")
        ax.axvline(0, color="0.6", lw=0.5, ls="--")
        ax.set_xlabel("Time from event (s)")
        ax.set_ylabel("Kernel weight (mean +/- SEM)")
        ax.set_title(f"Group GLM kernels (N = {n_animals})")
        ax.legend(loc="best", fontsize=5.5)
        base = os.path.join(out_dir, "glm_group_kernels")
        fig.savefig(base + ".pdf"); fig.savefig(base + ".png")
        plt.close(fig)
        return base + ".pdf"

    def _on_fit_all_files_clicked(self):
        if not self._processed_trials:
            self.statusMessage.emit("No recordings loaded.", 5000)
            return
        # Force scope to per-file batch and run.
        idx = self.combo_fit_scope.findData("batch")
        if idx >= 0:
            self.combo_fit_scope.setCurrentIndex(idx)
        self._fit_mode = "batch"
        self._on_fit_clicked()

    def _on_illustration_overlay_toggled(self, *_):
        if self._glm_result is not None:
            self._plot_glm_illustration(self._glm_result)

    def _on_fit_clicked(self):
        model_idx = self.combo_model_type.currentIndex()
        self._batch_cancel_requested = False
        self._set_fit_enabled(False)
        if hasattr(self, "btn_cancel"):
            self.btn_cancel.setEnabled(self._fit_mode == "batch")
        try:
            if model_idx == 0:
                if self._fit_mode == "active":
                    if not self._active_file_id:
                        self.statusMessage.emit("No active file selected.", 5000)
                        return
                    self._fit_glm_catalog(file_filter=self._active_file_id)
                elif self._fit_mode == "batch":
                    self._fit_glm_per_file_batch()
                else:
                    self._fit_glm_catalog(file_filter=None)
            else:
                self._fit_flmm()
        except Exception as exc:
            _LOG.error("Temporal modeling fit failed: %s\n%s", exc, traceback.format_exc())
            self.txt_summary.setPlainText(f"Error: {exc}")
            self.statusMessage.emit(f"Temporal model fit failed: {exc}", 8000)
        finally:
            self._set_fit_enabled(True)
            if hasattr(self, "btn_cancel"):
                self.btn_cancel.setEnabled(False)
            self._progress_finish()
            self._update_fit_state_label()

    # ------------------------------------------------------------------
    # GLM fit
    # ------------------------------------------------------------------

    def _fit_glm_catalog(self, file_filter: Optional[str] = None) -> Optional[GLMResult]:
        self._save_temporal_settings()
        dataset = self._build_glm_dataset_from_selected_predictors(file_filter=file_filter)
        if "error" in dataset:
            msg = str(dataset.get("error", "Could not build GLM dataset."))
            dropped = dataset.get("dropped_predictors", []) or []
            if dropped:
                msg = f"{msg}\nDropped: {', '.join(str(v) for v in dropped)}"
            self.txt_summary.setPlainText(msg)
            self.statusMessage.emit(msg.splitlines()[0], 7000)
            self._select_control_page(1)
            return None

        basis_map = {"Raised cosine": "raised_cosine", "B-spline": "bspline", "FIR": "fir"}
        reg_map = {"Ridge": "ridge", "Lasso": "lasso", "OLS": "ols"}
        kernel_win = (self.spin_kernel_pre.value(), self.spin_kernel_post.value())
        basis_type = basis_map.get(self.combo_basis.currentText(), "raised_cosine")
        regularization = reg_map.get(self.combo_reg.currentText(), "ridge")
        run_contrib = bool(getattr(self, "chk_run_contrib", None) is None or self.chk_run_contrib.isChecked())
        n_boot = int(self.spin_glm_bootstrap.value())
        n_basis = int(self.spin_n_basis.value())
        alpha = float(self.spin_alpha.value())
        cache_key = self._glm_cache_key(
            dataset,
            kernel_win,
            basis_type,
            regularization,
            alpha,
            n_basis,
            n_boot,
            run_contrib,
        )
        cached_result, cached_summary = self._load_glm_cache(cache_key)
        if cached_result is not None:
            self._glm_result = cached_result
            summary_text = cached_summary or (
                f"Continuous GLM - R^2 = {cached_result.r2:.4f}\nLoaded from GLM disk cache."
            )
            if "Loaded from GLM disk cache" not in summary_text:
                summary_text = summary_text + "\n\nLoaded from GLM disk cache; no refit was needed."
            self._render_glm_summary_html(cached_result, file_filter=file_filter, from_cache=True)
            if file_filter:
                self._glm_results_by_file[file_filter] = cached_result
                self._fit_summary_by_file[file_filter] = summary_text
                self._refresh_file_widgets()
            self._plot_glm_kernels(cached_result)
            self._plot_glm_fit(cached_result)
            self._plot_glm_illustration(cached_result)
            self._plot_feature_importance(
                cached_result.feature_importance or [],
                value_key="delta_r2",
                title="GLM leave-one-predictor-out contribution",
                y_label="Drop in R^2",
            )
            self.statusMessage.emit(f"Loaded cached GLM fit - R^2 = {cached_result.r2:.4f}", 5000)
            return cached_result
        scope_label = file_filter if file_filter else "all files"
        self._progress_start(f"Fitting GLM ({scope_label})", 0)
        result = self._glm.fit(
            np.asarray(dataset["time"], float),
            np.asarray(dataset["signal"], float),
            dataset["predictors"],
            kernel_window=kernel_win,
            n_basis=n_basis,
            basis_type=basis_type,
            regularization=regularization,
            alpha=alpha,
        )
        self._glm_result = result
        cv_stats = self._compute_glm_cross_validated_r2(
            dataset, kernel_win, basis_type, regularization, alpha
        )
        diagnostics = self._compute_glm_diagnostics(dataset, result)
        result.stats.update(cv_stats)
        result.stats.update(diagnostics)

        if run_contrib:
            self.statusMessage.emit("Calculating GLM leave-one-predictor-out contribution...", 0)
            QtWidgets.QApplication.processEvents()
            importance_rows = self._compute_glm_leave_one_out(
                dataset,
                result,
                kernel_win,
                basis_type,
                regularization,
                alpha,
            )
            self._compute_glm_shift_bootstrap_significance(
                dataset,
                importance_rows,
                kernel_win,
                basis_type,
                regularization,
                alpha,
                n_boot,
            )
        else:
            importance_rows = []
        result.feature_importance = importance_rows
        if n_boot > 0:
            self._compute_glm_kernel_bootstrap_ci(
                dataset,
                result,
                kernel_win,
                basis_type,
                regularization,
                alpha,
                n_boot,
            )

        used_labels = [self._predictor_label(k) for k in result.predictor_names]
        n_jobs = int(self.spin_glm_jobs.value()) if hasattr(self, "spin_glm_jobs") else 1
        dropped_predictors = dataset.get("dropped_predictors", []) or []
        used_records = dataset.get("used_records", []) or []
        dropped_records = dataset.get("dropped_records", []) or []
        record_preview = ", ".join(str(v) for v in used_records[:6])
        if len(used_records) > 6:
            record_preview += "..."
        cv_r2 = float(result.stats.get("cv_r2", float("nan")))
        cv_text = f", block-CV R^2 = {cv_r2:.4f}" if np.isfinite(cv_r2) else ""
        lines = [
            f"Continuous GLM - in-sample R^2 = {result.r2:.4f}{cv_text}",
            f"Recordings used: {len(used_records)} ({record_preview})",
            f"Samples fit: {int(dataset.get('valid_samples', 0))}",
            f"Predictors: {', '.join(used_labels)}",
            f"Basis: {self.combo_basis.currentText()}, n={n_basis}, kernel samples={len(result.kernel_tvec)}",
            f"Regularization: {self.combo_reg.currentText()}, alpha={alpha:.3f} (scale-aware columns)",
            f"Circular-shift bootstraps: {n_boot if n_boot > 0 else 'off'} ({n_jobs} job{'s' if n_jobs != 1 else ''})",
        ]
        stats = result.stats or {}
        lines.extend([
            "",
            "Fit statistics:",
            f"  RMSE = {stats.get('rmse', float('nan')):.5g}",
            f"  MAE = {stats.get('mae', float('nan')):.5g}",
            f"  MSE = {stats.get('mse', float('nan')):.5g}",
            f"  residual SD = {stats.get('residual_std', float('nan')):.5g}",
            f"  actual/predicted corr = {stats.get('corr', float('nan')):.5g}",
            f"  block-CV folds = {int(stats.get('cv_folds', 0) or 0)}",
        ])
        if stats.get("cv_note"):
            lines.append(f"  {stats.get('cv_note')}")
        condition = float(stats.get("condition_number", float("nan")))
        high_corr = stats.get("high_predictor_correlations", []) or []
        lines.extend(["", "Diagnostics:"])
        lines.append(f"  design condition number = {condition:.3g}" if np.isfinite(condition) else "  design condition number = n/a")
        if high_corr:
            lines.append("  High predictor correlations (|r| >= 0.8):")
            for item in high_corr[:5]:
                lines.append(
                    f"    {item.get('label_a', item.get('a'))} vs {item.get('label_b', item.get('b'))}: "
                    f"r = {float(item.get('r', float('nan'))):.3f}"
                )
        else:
            lines.append("  No predictor input correlations above |r| = 0.8.")
        if importance_rows:
            lines.extend(["", "Leave-one-predictor-out contribution (full - reduced):"])
            for row in importance_rows[:10]:
                p_value = float(row.get("p_value", float("nan")))
                q_value = float(row.get("q_value", float("nan")))
                op = "<= " if row.get("p_value_upper_bound", False) else "= "
                p_text = f", p {op}{p_value:.4g}" if np.isfinite(p_value) else ""
                q_text = f", q = {q_value:.4g}" if np.isfinite(q_value) else ""
                sig_text = " [significant]" if row.get("significant", False) else ""
                lines.append(
                    f"  {row['label']}: delta R^2 = {row['delta_r2']:.5g}, "
                    f"delta MSE = {row['delta_mse']:.5g}, reduced R^2 = {row['reduced_r2']:.5g}"
                    f"{p_text}{q_text}{sig_text}"
                )
            if n_boot > 0:
                significant = [row for row in importance_rows if row.get("significant", False)]
                lines.append(f"  Significant predictors at BH-FDR q < 0.05: {len(significant)}")
                warnings = sorted({str(row.get("bootstrap_warning", "")) for row in importance_rows if row.get("bootstrap_warning")})
                for warning in warnings:
                    lines.append(f"  Bootstrap warning: {warning}")
            failed = [row for row in importance_rows if row.get("status") != "ok"]
            if failed:
                lines.append(f"  {len(failed)} reduced fits failed; see log for details.")
        if dropped_predictors:
            lines.append(f"Dropped predictors: {', '.join(str(v) for v in dropped_predictors)}")
        if dropped_records:
            lines.append(f"Dropped recordings: {', '.join(str(v) for v in dropped_records)}")
        summary_text = "\n".join(lines)
        result.stats.update({
            "basis_label": self.combo_basis.currentText(),
            "regularization_label": self.combo_reg.currentText(),
            "n_basis": int(n_basis),
            "alpha": float(alpha),
            "n_boot": int(n_boot),
            "n_jobs": int(n_jobs),
            "used_records": [str(v) for v in used_records],
            "used_labels": [str(v) for v in used_labels],
            "dropped_predictors": [str(v) for v in dropped_predictors],
            "dropped_records": [str(v) for v in dropped_records],
            "valid_samples": int(dataset.get("valid_samples", 0)),
            "scope_label": str(scope_label),
        })
        self._render_glm_summary_html(result, file_filter=file_filter, from_cache=False)
        self._save_glm_cache(cache_key, result, summary_text)

        # Cache per-file fit if scope was a single file.
        if file_filter:
            self._glm_results_by_file[file_filter] = result
            self._fit_summary_by_file[file_filter] = summary_text
            self._refresh_file_widgets()

        self._plot_glm_kernels(result)
        self._plot_glm_fit(result)
        self._plot_glm_illustration(result)
        self._plot_feature_importance(
            importance_rows,
            value_key="delta_r2",
            title="GLM leave-one-predictor-out contribution",
            y_label="Drop in R^2",
        )
        if hasattr(self, "tabs_workspace"):
            self.tabs_workspace.setCurrentWidget(self.plot_importance.parentWidget() if importance_rows else self.plot_kernel.parentWidget())
        self.statusMessage.emit(f"GLM fit complete - R^2 = {result.r2:.4f}", 5000)
        return result

    def _fit_glm_per_file_batch(self) -> None:
        """Fit each loaded file independently and populate the Group tab."""
        if not self._processed_trials:
            self.statusMessage.emit("No recordings loaded.", 5000)
            return
        file_ids = [
            self._proc_file_id(p, fallback=f"file_{i + 1}")
            for i, p in enumerate(self._processed_trials)
        ]
        n = len(file_ids)
        self._progress_start(f"Per-file batch ({n} files)", n)
        ok_results: Dict[str, GLMResult] = {}
        ok_summaries: Dict[str, str] = {}
        for idx, fid in enumerate(file_ids, 1):
            if self._batch_cancel_requested:
                self.statusMessage.emit(f"Batch cancelled after {idx - 1}/{n}.", 6000)
                break
            self._progress_update(idx - 1, f"Per-file batch ({idx}/{n})")
            if hasattr(self, "lbl_batch_status"):
                self.lbl_batch_status.setText(f"Fitting {idx}/{n}: {fid}")
                QtWidgets.QApplication.processEvents()
            try:
                result = self._fit_glm_catalog(file_filter=fid)
            except Exception as exc:
                _LOG.warning("Per-file fit failed for %s: %s", fid, exc)
                result = None
            if result is not None:
                ok_results[fid] = result
                ok_summaries[fid] = self._fit_summary_by_file.get(fid, "")
            self._progress_update(idx, f"Per-file batch ({idx}/{n})")
        if hasattr(self, "lbl_batch_status"):
            self.lbl_batch_status.setText(
                f"Batch complete: {len(ok_results)}/{n} files fit successfully."
            )
        self._aggregate_group_results()
        if hasattr(self, "tabs_workspace") and ok_results:
            self.tabs_workspace.setCurrentWidget(self.plot_group_kernels.parentWidget())

    def _aggregate_group_results(self) -> None:
        """Average per-file kernels and importance for the Group tab."""
        if not hasattr(self, "plot_group_kernels"):
            return
        results = self._glm_results_by_file
        self.plot_group_kernels.clear()
        self.plot_group_importance.clear()
        try:
            self.plot_group_kernels.getPlotItem().legend.clear()
        except Exception:
            pass
        try:
            self.plot_group_importance.getPlotItem().legend.clear()
        except Exception:
            pass
        if not results:
            self.lbl_group_summary.setText(
                "Run a Per-file batch fit to populate the Group view."
            )
            return

        # Find common predictors and a shared kernel time vector.
        predictor_lists = [list(r.predictor_names) for r in results.values()]
        common_keys = set(predictor_lists[0])
        for lst in predictor_lists[1:]:
            common_keys &= set(lst)
        if not common_keys:
            self.lbl_group_summary.setText(
                "Per-file fits do not share any common predictor; cannot aggregate."
            )
            return

        # Use the first result's kernel_tvec; resample others to it if shapes differ.
        ref = next(iter(results.values()))
        ref_t = np.asarray(ref.kernel_tvec, float)
        kernel_stack: Dict[str, List[np.ndarray]] = {k: [] for k in common_keys}
        kernel_se_stack: Dict[str, List[np.ndarray]] = {k: [] for k in common_keys}
        for fid, r in results.items():
            t = np.asarray(r.kernel_tvec, float)
            for key in common_keys:
                kern = r.kernels.get(key)
                if kern is None:
                    continue
                kern = np.asarray(kern, float)
                if kern.size != ref_t.size:
                    if t.size and kern.size == t.size:
                        kern = np.interp(ref_t, t, kern, left=np.nan, right=np.nan)
                    else:
                        continue
                kernel_stack[key].append(kern)
                lo = (r.kernel_ci_lower or {}).get(key)
                hi = (r.kernel_ci_upper or {}).get(key)
                if lo is not None and hi is not None:
                    lo_arr = np.asarray(lo, float)
                    hi_arr = np.asarray(hi, float)
                    if lo_arr.size != ref_t.size and t.size and lo_arr.size == t.size:
                        lo_arr = np.interp(ref_t, t, lo_arr, left=np.nan, right=np.nan)
                    if hi_arr.size != ref_t.size and t.size and hi_arr.size == t.size:
                        hi_arr = np.interp(ref_t, t, hi_arr, left=np.nan, right=np.nan)
                    if lo_arr.size == ref_t.size and hi_arr.size == ref_t.size:
                        se = np.abs(hi_arr - lo_arr) / (2.0 * 1.96)
                        se[~np.isfinite(se)] = np.nan
                        kernel_se_stack[key].append(se)

        # Plot precision-weighted mean +/- uncertainty when at least one per-file
        # kernel has CIs; animals without CIs get the median weight of the
        # animals that do (so we don't silently demote a no-CI fit to zero
        # contribution, nor over-weight them when they're noisy).
        plotted = 0
        for key, stack in kernel_stack.items():
            if not stack:
                continue
            arr = np.vstack(stack)
            se_stack = kernel_se_stack.get(key, [])
            n_files = arr.shape[0]
            if se_stack and len(se_stack) >= 1:
                # Build a weight row per animal. Animals with CIs get
                # 1/SE^2 across time; animals without CIs get the median
                # weight across the CI-bearing animals (per-time), so they
                # are pulled toward the typical precision of the cohort.
                weight_rows: List[np.ndarray] = []
                ci_rows: List[np.ndarray] = []
                for i in range(n_files):
                    if i < len(se_stack):
                        se = np.asarray(se_stack[i], float)
                    else:
                        se = np.array([], float)
                    if se.size == arr.shape[1] and np.any(np.isfinite(se)):
                        w_row = 1.0 / np.maximum(se ** 2, 1e-12)
                        w_row[~np.isfinite(w_row)] = 0.0
                        weight_rows.append(w_row)
                        ci_rows.append(w_row)
                    else:
                        weight_rows.append(None)  # type: ignore[arg-type]
                if ci_rows:
                    median_weight = np.median(np.vstack(ci_rows), axis=0)
                else:
                    median_weight = None
                # Replace None rows with median (or fall back to unweighted later).
                if median_weight is not None and np.all(np.isfinite(median_weight)):
                    finalized = np.vstack([
                        w if w is not None else median_weight for w in weight_rows
                    ])
                    denom = np.sum(finalized, axis=0)
                    mean = np.where(denom > 0,
                                    np.sum(arr * finalized, axis=0) / np.where(denom > 0, denom, 1.0),
                                    np.nanmean(arr, axis=0))
                    sem = np.where(denom > 0,
                                   1.0 / np.sqrt(np.where(denom > 0, denom, 1.0)),
                                   np.nanstd(arr, axis=0, ddof=1) / np.sqrt(max(n_files, 1)))
                else:
                    mean = np.nanmean(arr, axis=0)
                    sem = np.nanstd(arr, axis=0, ddof=1) / np.sqrt(max(n_files, 1))
            else:
                mean = np.nanmean(arr, axis=0)
                sem = np.nanstd(arr, axis=0, ddof=1) / np.sqrt(max(n_files, 1))
            color = self._kernel_color(key)
            qcol = QtGui.QColor(color)
            qcol.setAlpha(60)
            lo = mean - sem
            hi = mean + sem
            self.plot_group_kernels.plot(
                ref_t, mean, pen=pg.mkPen(color, width=2),
                name=f"{self._predictor_label(key)} (n={arr.shape[0]})",
            )
            fill = pg.FillBetweenItem(
                pg.PlotDataItem(ref_t, lo),
                pg.PlotDataItem(ref_t, hi),
                brush=qcol,
            )
            self.plot_group_kernels.addItem(fill)
            plotted += 1
        self.plot_group_kernels.setLabel("bottom", "Time", units="s")
        self.plot_group_kernels.setLabel("left", "Kernel weight")
        self.plot_group_kernels.addLine(y=0, pen=pg.mkPen("#5a6274", width=1, style=QtCore.Qt.PenStyle.DashLine))
        self.plot_group_kernels.addLine(x=0, pen=pg.mkPen("#5a6274", width=1, style=QtCore.Qt.PenStyle.DashLine))

        # Aggregate leave-one-out importance: mean delta_r2 across animals.
        importance_acc: Dict[str, List[Tuple[float, float]]] = {}
        importance_labels: Dict[str, str] = {}
        for r in results.values():
            for row in r.feature_importance or []:
                feat = str(row.get("feature", ""))
                if not feat:
                    continue
                val = float(row.get("delta_r2", float("nan")))
                if not np.isfinite(val):
                    continue
                null_scale = float("nan")
                if np.isfinite(float(row.get("null_delta_q95", float("nan")))) and np.isfinite(float(row.get("null_delta_mean", float("nan")))):
                    null_scale = abs(float(row.get("null_delta_q95")) - float(row.get("null_delta_mean"))) / 1.645
                weight = 1.0 / max(null_scale ** 2, 1e-12) if np.isfinite(null_scale) and null_scale > 0 else 1.0
                importance_acc.setdefault(feat, []).append((val, weight))
                importance_labels[feat] = str(row.get("label", feat) or feat)
        agg_rows: List[Dict[str, Any]] = []
        for feat, vals_weights in importance_acc.items():
            arr = np.asarray([vw[0] for vw in vals_weights], float)
            weights = np.asarray([vw[1] for vw in vals_weights], float)
            if np.any(np.isfinite(weights)) and float(np.nansum(weights)) > 0:
                mean_val = float(np.nansum(arr * weights) / np.nansum(weights))
                sem_val = float(1.0 / np.sqrt(np.nansum(weights))) if arr.size > 1 else 0.0
            else:
                mean_val = float(np.nanmean(arr))
                sem_val = float(np.nanstd(arr, ddof=1) / np.sqrt(max(arr.size, 1))) if arr.size > 1 else 0.0
            agg_rows.append({
                "feature": feat,
                "label": importance_labels.get(feat, feat),
                "delta_r2": mean_val,
                "delta_r2_sem": sem_val,
                "n_animals": int(arr.size),
                "significant": False,
            })
        agg_rows.sort(key=lambda r: r.get("delta_r2", -np.inf), reverse=True)
        self._group_glm_summary = {
            "n_files": len(results),
            "common_predictors": len(common_keys),
            "importance": agg_rows,
            "kernel_tvec": ref_t.tolist(),
        }
        self._render_group_importance(agg_rows)
        self.lbl_group_summary.setText(
            f"Group GLM aggregate: {len(results)} animals, {len(common_keys)} common predictors. "
            f"Kernels use precision weighting when per-file CIs are available."
        )

    def _render_group_importance(self, rows: List[Dict[str, Any]]) -> None:
        if not hasattr(self, "plot_group_importance"):
            return
        pw = self.plot_group_importance
        pw.clear()
        try:
            pw.getPlotItem().legend.clear()
        except Exception:
            pass
        usable = [r for r in rows if np.isfinite(float(r.get("delta_r2", float("nan"))))]
        if not usable:
            txt = pg.TextItem("No group importance available.", color="#c5d2e3")
            pw.addItem(txt)
            txt.setPos(0, 0)
            return
        usable = usable[:25]
        vals = np.asarray([float(r.get("delta_r2", 0.0)) for r in usable], float)
        sems = np.asarray([float(r.get("delta_r2_sem", 0.0)) for r in usable], float)
        y_pos = np.arange(len(usable), dtype=float)[::-1]
        brushes = [pg.mkBrush("#4b9df8" if v >= 0 else "#ee99a0") for v in vals]
        bar = pg.BarGraphItem(x0=np.zeros_like(vals), x1=vals, y=y_pos, height=0.62, brushes=brushes)
        pw.addItem(bar)
        # Error bars
        err = pg.ErrorBarItem(x=vals, y=y_pos, left=sems, right=sems, beam=0.18, pen=pg.mkPen("#c5d2e3"))
        pw.addItem(err)
        pw.addLine(x=0, pen=pg.mkPen("#5a6274", width=1, style=QtCore.Qt.PenStyle.DashLine))
        labels = []
        for pos, row in zip(y_pos, usable):
            label = self._compact_feature_label(row.get("label", row.get("feature", "")), 46)
            n_an = int(row.get("n_animals", 0))
            labels.append((float(pos), f"{label}  (n={n_an})"))
        pw.getAxis("left").setTicks([labels])
        pw.setLabel("bottom", "Mean drop in R^2 +/- SEM")
        pw.setLabel("left", "Feature")

    def _fit_glm(self):
        self._fit_glm_catalog()
        return  # legacy path below kept for reference
        if not self._processed_trials:
            self.statusMessage.emit("No processed data — run preprocessing first.", 5000)
            return

        proc = self._processed_trials[0]
        time = np.asarray(proc.time, float)
        signal = np.asarray(proc.output, float) if proc.output is not None else None
        if signal is None or signal.size == 0:
            self.statusMessage.emit("No output signal available.", 5000)
            return

        # Build predictors from the list widget
        predictors: Dict[str, np.ndarray] = {}
        for i in range(self.list_predictors.count()):
            pred_name = self.list_predictors.item(i).text()
            if pred_name == "events" and self._event_times is not None:
                predictors[pred_name] = self._event_times
            elif proc.dio is not None and pred_name.lower() in ("dio", "digital"):
                # Derive event times from DIO rising edges
                dio = np.asarray(proc.dio, float)
                edges = np.where(np.diff(dio > 0.5) == True)[0]  # noqa: E712
                if edges.size > 0:
                    predictors[pred_name] = time[edges]
            else:
                # Try to find among triggers
                if pred_name in (proc.triggers or {}):
                    trig = np.asarray(proc.triggers[pred_name], float)
                    edges = np.where(np.diff(trig > 0.5) == True)[0]  # noqa: E712
                    if edges.size > 0:
                        predictors[pred_name] = time[edges]

        if not predictors:
            self.statusMessage.emit("No valid predictors found. Add event-based predictors.", 5000)
            return

        basis_map = {"Raised cosine": "raised_cosine", "B-spline": "bspline", "FIR": "fir"}
        reg_map = {"Ridge": "ridge", "Lasso": "lasso", "OLS": "ols"}

        kernel_win = (self.spin_kernel_pre.value(), self.spin_kernel_post.value())
        result = self._glm.fit(
            time, signal, predictors,
            kernel_window=kernel_win,
            n_basis=self.spin_n_basis.value(),
            basis_type=basis_map.get(self.combo_basis.currentText(), "raised_cosine"),
            regularization=reg_map.get(self.combo_reg.currentText(), "ridge"),
            alpha=self.spin_alpha.value(),
        )
        self._glm_result = result

        # Summary
        lines = [
            f"Continuous GLM — R² = {result.r2:.4f}",
            f"Predictors: {', '.join(result.predictor_names)}",
            f"Basis: {self.combo_basis.currentText()}, n={self.spin_n_basis.value()}",
            f"Regularization: {self.combo_reg.currentText()}, α={self.spin_alpha.value():.3f}",
        ]
        self.txt_summary.setPlainText("\n".join(lines))

        # Plot kernels
        self._plot_glm_kernels(result)
        self._plot_glm_fit(result)
        self._plot_glm_illustration(result)
        if hasattr(self, "tabs_workspace"):
            self.tabs_workspace.setCurrentWidget(self.plot_kernel.parentWidget())
        self.statusMessage.emit(f"GLM fit complete — R² = {result.r2:.4f}", 5000)

    def _plot_glm_kernels(self, result: GLMResult, refresh_filter: bool = True):
        if refresh_filter:
            self._sync_kernel_filter(result)
        names = self._selected_kernel_names(result)
        if self._kernel_layout_mode() == "grid":
            self._plot_kernel_grid(result, names)
        else:
            self._plot_kernel_overlay(result, names)

    def _plot_glm_fit(self, result: GLMResult):
        pw = self.plot_prediction
        pw.clear()
        try:
            pw.getPlotItem().legend.clear()
        except Exception:
            pass
        x = np.asarray(result.time, float) if result.time is not None else np.arange(result.y_actual.size)
        pw.plot(x, result.y_actual, pen=pg.mkPen("#4b9df8", width=1.2), name="actual")
        pw.plot(x, result.y_pred, pen=pg.mkPen("#f5a97f", width=1.4), name="predicted")
        pw.setLabel("bottom", "Time", units="s")
        pw.setLabel("left", "Signal")

        rw = self.plot_residuals
        rw.clear()
        rw.plot(x, result.residuals, pen=pg.mkPen("#ee99a0", width=1.1), name="residual")
        rw.addLine(y=0, pen=pg.mkPen("#5a6274", width=1, style=QtCore.Qt.PenStyle.DashLine))
        rw.setLabel("bottom", "Time", units="s")
        rw.setLabel("left", "Residual")

    def _plot_glm_illustration(self, result: GLMResult) -> None:
        if not hasattr(self, "plot_illustration") or self._illustration_vb is None:
            return
        self._sync_illustration_features(result)
        key = self.combo_illustration_feature.currentData(QtCore.Qt.ItemDataRole.UserRole)
        if not isinstance(key, str) or not key:
            ordered = self._glm_feature_order(result)
            key = ordered[0] if ordered else ""

        show_signal = bool(getattr(self, "chk_show_signal", None) is None or self.chk_show_signal.isChecked())
        show_pred = bool(getattr(self, "chk_show_predicted", None) is not None and self.chk_show_predicted.isChecked())
        show_contrib = bool(getattr(self, "chk_show_contribution", None) is None or self.chk_show_contribution.isChecked())
        show_raw = bool(getattr(self, "chk_show_raw_predictor", None) is not None and self.chk_show_raw_predictor.isChecked())

        pw = self.plot_illustration
        vb = self._illustration_vb
        pw.clear()
        vb.clear()
        try:
            pw.getPlotItem().legend.clear()
        except Exception:
            pass
        pw.setTitle("Signal and selected feature contribution")
        if not key:
            self.lbl_illustration_stats.setText("No fitted GLM feature is available.")
            return

        x = np.asarray(result.time, float)
        signal = np.asarray(result.y_actual, float)
        predicted = np.asarray(result.y_pred, float) if result.y_pred is not None else None
        contribution = self._glm_feature_contribution(result, key)
        if contribution is None:
            self.lbl_illustration_stats.setText("No contribution trace is available for the selected feature.")
            return
        contribution = np.asarray(contribution, float)

        # Raw predictor input (model-time vector) if requested.
        raw_predictor: Optional[np.ndarray] = None
        if show_raw and result.design_matrix is not None and result.predictor_names:
            try:
                pred_idx = result.predictor_names.index(key)
                n_pred = len(result.predictor_names)
                n_basis = max(1, (int(np.asarray(result.coefficients).size) - 1) // n_pred)
                lo = 1 + pred_idx * n_basis
                # Best proxy for the raw input: the un-convolved indicator (sum across basis cols).
                raw_predictor = np.asarray(result.design_matrix[:, lo:lo + n_basis], float).sum(axis=1)
            except Exception:
                raw_predictor = None

        n = min(
            int(x.size),
            int(signal.size),
            int(contribution.size),
            int(predicted.size) if predicted is not None else int(signal.size),
            int(raw_predictor.size) if raw_predictor is not None else int(signal.size),
        )
        x = x[:n]
        signal = signal[:n]
        contribution = contribution[:n]
        if predicted is not None:
            predicted = predicted[:n]
        if raw_predictor is not None:
            raw_predictor = raw_predictor[:n]

        valid = np.isfinite(x) & np.isfinite(signal) & np.isfinite(contribution)
        r_value, p_value, n_corr = self._pearson_stats(signal[valid], contribution[valid])
        p_text = self._p_label(p_value)
        stars = self._p_stars(p_value)
        stats_text = f"Pearson r = {r_value:.3f}, {p_text}, n = {n_corr} (descriptive; autocorrelated samples)"
        if stars:
            stats_text += f" ({stars})"
        self.lbl_illustration_stats.setText(stats_text)

        signal_color = "#4b9df8"
        predicted_color = "#f5a97f"
        feature_color = self._kernel_color(key)
        raw_color = "#94e2d5"

        if show_signal:
            pw.plot(x, signal, pen=pg.mkPen(signal_color, width=1.25), name="signal")
        if show_pred and predicted is not None:
            pw.plot(x, predicted, pen=pg.mkPen(predicted_color, width=1.0, style=QtCore.Qt.PenStyle.DashLine), name="predicted")
        if show_contrib:
            feat_curve = pg.PlotDataItem(
                x, contribution, pen=pg.mkPen(feature_color, width=1.8),
                name=self._predictor_label(key),
            )
            vb.addItem(feat_curve)
        if show_raw and raw_predictor is not None:
            raw_curve = pg.PlotDataItem(
                x, raw_predictor, pen=pg.mkPen(raw_color, width=1.0, style=QtCore.Qt.PenStyle.DotLine),
                name=f"{self._predictor_label(key)} (raw)",
            )
            vb.addItem(raw_curve)

        plot_item = pw.getPlotItem()
        plot_item.getAxis("right").setPen(pg.mkPen(feature_color))
        plot_item.getAxis("right").setTextPen(pg.mkPen(feature_color))
        plot_item.getAxis("left").setPen(pg.mkPen(signal_color))
        plot_item.getAxis("left").setTextPen(pg.mkPen(signal_color))
        plot_item.setLabel("left", "Signal")
        plot_item.setLabel("right", "Feature contribution / raw")
        plot_item.setLabel("bottom", "Time", units="s")
        pw.addLine(y=0, pen=pg.mkPen("#5a6274", width=1, style=QtCore.Qt.PenStyle.DashLine))

        finite_signal = signal[np.isfinite(signal)]
        if finite_signal.size and (show_signal or show_pred):
            y0 = float(np.nanmin(finite_signal))
            y1 = float(np.nanmax(finite_signal))
            pad = max((y1 - y0) * 0.08, 1e-9)
            pw.setYRange(y0 - pad, y1 + pad, padding=0.0)
        # Right viewbox range from whichever overlay is shown there.
        right_arrays: List[np.ndarray] = []
        if show_contrib:
            right_arrays.append(contribution[np.isfinite(contribution)])
        if show_raw and raw_predictor is not None:
            right_arrays.append(raw_predictor[np.isfinite(raw_predictor)])
        if right_arrays:
            stacked = np.concatenate([np.asarray(a, float) for a in right_arrays if a.size])
            if stacked.size:
                f0 = float(np.nanmin(stacked))
                f1 = float(np.nanmax(stacked))
                pad = max((f1 - f0) * 0.08, 1e-9)
                vb.setYRange(f0 - pad, f1 + pad, padding=0.0)
        self._update_illustration_view()

        finite_x = x[np.isfinite(x)]
        if finite_x.size and finite_signal.size:
            xr0 = float(np.nanmin(finite_x))
            xr1 = float(np.nanmax(finite_x))
            yr0 = float(np.nanmin(finite_signal))
            yr1 = float(np.nanmax(finite_signal))
            txt = pg.TextItem(stats_text, color="#e9f0fb", anchor=(0.0, 0.0))
            pw.addItem(txt)
            txt.setPos(xr0 + 0.02 * max(xr1 - xr0, 1e-9), yr1 - 0.08 * max(yr1 - yr0, 1e-9))

    @staticmethod
    def _nanmean_sem_axis0(mat: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        arr = np.asarray(mat, float)
        if arr.ndim != 2 or arr.shape[1] == 0:
            return np.array([], float), np.array([], float)
        finite = np.isfinite(arr)
        count = np.sum(finite, axis=0).astype(float)
        sums = np.nansum(np.where(finite, arr, 0.0), axis=0)
        mean = np.divide(sums, count, out=np.full(arr.shape[1], np.nan, float), where=count > 0)
        centered = np.where(finite, arr - mean[None, :], 0.0)
        var = np.divide(
            np.sum(centered ** 2, axis=0),
            np.maximum(count - 1.0, 1.0),
            out=np.full(arr.shape[1], np.nan, float),
            where=count > 1,
        )
        sem = np.divide(
            np.sqrt(var),
            np.sqrt(count),
            out=np.full(arr.shape[1], np.nan, float),
            where=count > 1,
        )
        return mean, sem

    def _store_flmm_trace_input(
        self,
        mat: np.ndarray,
        tvec: np.ndarray,
        row_labels: List[str],
        design: Dict[str, np.ndarray],
        term_labels: Dict[str, str],
        formula_terms: List[str],
        scope: str,
    ) -> None:
        self._flmm_trace_mat = np.asarray(mat, float)
        self._flmm_trace_tvec = np.asarray(tvec, float)
        self._flmm_trace_labels = list(row_labels or [])
        self._flmm_trace_design = {str(k): np.asarray(v) for k, v in (design or {}).items()}
        self._flmm_trace_term_labels = dict(term_labels or {})
        self._flmm_trace_scope = str(scope or "")
        combo = getattr(self, "combo_flmm_trace_feature", None)
        if combo is not None:
            combo.blockSignals(True)
            try:
                current = combo.currentData(QtCore.Qt.ItemDataRole.UserRole)
                combo.clear()
                combo.addItem("None", "")
                for term in formula_terms:
                    if term in self._flmm_trace_design:
                        combo.addItem(self._flmm_trace_term_labels.get(term, term), term)
                idx = combo.findData(current, QtCore.Qt.ItemDataRole.UserRole)
                combo.setCurrentIndex(idx if idx >= 0 else 0)
            finally:
                combo.blockSignals(False)
        self._refresh_flmm_trace_plot()

    def _trace_color_for_row(
        self,
        row_index: int,
        row_label: str,
        mode: str,
        feature_values: Optional[np.ndarray],
        group_colors: Dict[str, QtGui.QColor],
    ) -> QtGui.QColor:
        color = QtGui.QColor("#7aa2f7")
        color.setAlpha(58)
        if mode == "group":
            key = str(row_label or "")
            if key not in group_colors:
                qcol = QtGui.QColor(pg.intColor(len(group_colors), hues=16, values=1, maxValue=255))
                qcol.setAlpha(78)
                group_colors[key] = qcol
            return QtGui.QColor(group_colors[key])
        if mode == "predictor" and feature_values is not None and row_index < feature_values.size:
            val = float(feature_values[row_index])
            finite = feature_values[np.isfinite(feature_values)]
            if np.isfinite(val) and finite.size:
                scale = float(np.nanpercentile(np.abs(finite), 95))
                scale = max(scale, 1e-9)
                strength = min(1.0, abs(val) / scale)
                color = QtGui.QColor("#f5a97f" if val >= 0 else "#4b9df8")
                color.setAlpha(int(45 + 105 * strength))
                return color
        return color

    def _refresh_flmm_trace_plot(self) -> None:
        pw = getattr(self, "plot_flmm_traces", None)
        if pw is None:
            return
        pw.clear()
        try:
            pw.getPlotItem().legend.clear()
        except Exception:
            pass
        mat = np.asarray(self._flmm_trace_mat, float) if self._flmm_trace_mat is not None else np.array([], float)
        tvec = np.asarray(self._flmm_trace_tvec, float) if self._flmm_trace_tvec is not None else np.array([], float)
        if mat.ndim != 2 or mat.shape[0] == 0 or tvec.size != mat.shape[1]:
            txt = pg.TextItem("Fit an FLMM to inspect the individual input traces.", color="#c5d2e3")
            pw.addItem(txt)
            txt.setPos(0, 0)
            if hasattr(self, "lbl_flmm_trace_summary"):
                self.lbl_flmm_trace_summary.setText("")
            return

        max_traces = int(self.spin_flmm_trace_max.value()) if hasattr(self, "spin_flmm_trace_max") else 250
        n_rows = int(mat.shape[0])
        if n_rows > max_traces:
            indices = np.unique(np.linspace(0, n_rows - 1, max_traces).astype(int))
        else:
            indices = np.arange(n_rows, dtype=int)

        mode = str(self.combo_flmm_trace_color.currentData(QtCore.Qt.ItemDataRole.UserRole) or "single") if hasattr(self, "combo_flmm_trace_color") else "single"
        term = str(self.combo_flmm_trace_feature.currentData(QtCore.Qt.ItemDataRole.UserRole) or "") if hasattr(self, "combo_flmm_trace_feature") else ""
        feature_values = None
        if mode == "predictor" and term:
            values = np.asarray(self._flmm_trace_design.get(term, np.array([], float)), float).reshape(-1)
            if values.size == n_rows:
                feature_values = values
        labels = self._flmm_trace_labels if len(self._flmm_trace_labels) == n_rows else [f"row_{i + 1}" for i in range(n_rows)]
        group_colors: Dict[str, QtGui.QColor] = {}
        for row_index in indices:
            row = mat[int(row_index), :]
            if not np.any(np.isfinite(row)):
                continue
            qcol = self._trace_color_for_row(int(row_index), labels[int(row_index)], mode, feature_values, group_colors)
            pw.plot(tvec, row, pen=pg.mkPen(qcol, width=1))

        show_mean = bool(self.chk_flmm_trace_mean.isChecked()) if hasattr(self, "chk_flmm_trace_mean") else True
        if show_mean:
            mean, sem = self._nanmean_sem_axis0(mat)
            if mean.size == tvec.size and np.any(np.isfinite(mean)):
                fill_col = QtGui.QColor("#8bd5ca")
                fill_col.setAlpha(42)
                fill = pg.FillBetweenItem(
                    pg.PlotDataItem(tvec, mean - sem),
                    pg.PlotDataItem(tvec, mean + sem),
                    brush=fill_col,
                )
                pw.addItem(fill)
                pw.plot(tvec, mean, pen=pg.mkPen("#8bd5ca", width=3), name="mean")

        pw.addLine(x=0, pen=pg.mkPen("#5a6274", width=1, style=QtCore.Qt.PenStyle.DashLine))
        pw.setLabel("bottom", "Time", units="s")
        pw.setLabel("left", "Response")
        if hasattr(self, "lbl_flmm_trace_summary"):
            suffix = ""
            if mode == "predictor" and term:
                suffix = f"; colored by {self._flmm_trace_term_labels.get(term, term)}"
            elif mode == "group":
                suffix = f"; {len(set(labels))} file/group labels"
            self.lbl_flmm_trace_summary.setText(
                f"{n_rows} rows, {mat.shape[1]} timepoints; showing {len(indices)} traces{suffix}"
            )

    def _plot_feature_importance(
        self,
        rows: List[Dict[str, Any]],
        value_key: str,
        title: str,
        y_label: str,
    ) -> None:
        if not hasattr(self, "plot_importance"):
            return
        pw = self.plot_importance
        pw.clear()
        try:
            pw.getPlotItem().legend.clear()
        except Exception:
            pass
        pw.setTitle(title)
        usable = [
            row for row in rows
            if np.isfinite(float(row.get(value_key, float("nan"))))
        ]
        if not usable:
            txt = pg.TextItem("No feature contribution is available.", color="#c5d2e3")
            pw.addItem(txt)
            txt.setPos(0, 0)
            pw.setLabel("bottom", y_label)
            pw.setLabel("left", "Feature")
            return

        usable = usable[:25]
        vals = np.asarray([float(row.get(value_key, 0.0)) for row in usable], float)
        y_pos = np.arange(len(usable), dtype=float)[::-1]
        brushes = []
        for row, val in zip(usable, vals):
            if row.get("significant", False):
                brushes.append(pg.mkBrush("#f5c542"))
            else:
                brushes.append(pg.mkBrush("#4b9df8" if val >= 0 else "#ee99a0"))
        bar = pg.BarGraphItem(x0=np.zeros_like(vals), x1=vals, y=y_pos, height=0.62, brushes=brushes)
        pw.addItem(bar)
        pw.addLine(x=0, pen=pg.mkPen("#5a6274", width=1, style=QtCore.Qt.PenStyle.DashLine))
        labels = []
        max_abs = max(float(np.nanmax(np.abs(vals))) if vals.size else 1.0, 1e-9)
        label_offset = max_abs * 0.025
        for pos, row, val in zip(y_pos, usable, vals):
            label = self._compact_feature_label(row.get("label", row.get("feature", "")), 46)
            labels.append((float(pos), label))
            p_value = float(row.get("p_value", float("nan")))
            q_value = float(row.get("q_value", float("nan")))
            if row.get("significant", False) and (np.isfinite(q_value) or np.isfinite(p_value)):
                if np.isfinite(q_value):
                    label_text = f"q={q_value:.3g}"
                else:
                    op = "<=" if row.get("p_value_upper_bound", False) else "="
                    label_text = f"p{op}{p_value:.3g}"
                p_txt = pg.TextItem(label_text, color="#f5c542", anchor=(0.0, 0.5))
                pw.addItem(p_txt)
                p_txt.setPos(float(val) + label_offset, float(pos))
        pw.getAxis("left").setTicks([labels])
        pw.getAxis("bottom").setTicks(None)
        pw.setLabel("bottom", y_label)
        pw.setLabel("left", "Feature")
        lo = min(0.0, float(np.nanmin(vals)) if vals.size else 0.0)
        hi = max(0.0, float(np.nanmax(vals)) if vals.size else 1.0)
        pad = max((hi - lo) * 0.15, max_abs * 0.12, 1e-6)
        pw.setXRange(lo - pad, hi + pad, padding=0.0)
        pw.setYRange(-1, len(usable), padding=0.02)

    # ------------------------------------------------------------------
    # FLMM fit
    # ------------------------------------------------------------------

    def _fit_flmm_group(self) -> None:
        self._save_temporal_settings()
        mat, tvec, row_labels, scope = self._flmm_matrix_and_labels()
        if mat is None or tvec is None:
            self.statusMessage.emit("No PSTH matrix - compute PSTH first.", 5000)
            return
        if mat.ndim != 2 or mat.shape[0] < 2:
            self.statusMessage.emit("Need at least 2 animals/trials for FLMM.", 5000)
            return

        n_rows = int(mat.shape[0])
        group_var = self.edit_group_var.text().strip() or "subject"
        if not re.match(r"^[A-Za-z_][0-9A-Za-z_]*$", group_var):
            group_var = "subject"
            self.edit_group_var.setText(group_var)
        if len(row_labels) != n_rows:
            row_labels = [f"{scope}_{i + 1}" for i in range(n_rows)]

        design, auto_terms, dropped_terms, term_labels = self._build_flmm_design(row_labels, group_var)
        fit_terms, pruned_terms = self._prune_flmm_terms(design, auto_terms, term_labels, n_rows)
        if pruned_terms:
            dropped_terms.extend(pruned_terms)

        requested_formula = self.edit_formula.text().strip()
        requested_terms = [
            term for term in self._simple_formula_terms(requested_formula)
            if "|" not in term and "(" not in term and ")" not in term
        ]
        missing_terms = [term for term in requested_terms if term not in design]
        use_auto = (
            requested_formula in {"", "Y.obs ~ 1", "Y.obs ~ group"}
            or "~" not in requested_formula
            or not requested_formula.lstrip().startswith("Y.obs")
            or bool(missing_terms)
        )
        if use_auto:
            formula_terms = list(fit_terms)
        else:
            formula_terms = [term for term in requested_terms if term in fit_terms]
            removed_manual = [term for term in requested_terms if term in design and term not in fit_terms]
            if removed_manual:
                missing_terms.extend(removed_manual)
        formula = "Y.obs ~ " + " + ".join(formula_terms) if formula_terms else "Y.obs ~ 1"
        if use_auto:
            if requested_formula != "Y.obs ~ 1":
                self.edit_formula.setText("Y.obs ~ 1")
        elif requested_formula != formula:
            self.edit_formula.setText(formula)
        random_eff = self.edit_random.text().strip() or "~1"
        nknots = self.spin_nknots.value() if self.spin_nknots.value() > 0 else None
        num_boots = self.spin_boots.value()

        group_vals = np.asarray(design.get(group_var, np.array([], object))).astype(str)
        unique_groups = np.unique(group_vals) if group_vals.size else np.array([], dtype=str)
        has_repeated_groups = 1 < unique_groups.size < n_rows
        use_fixed_fallback = not has_repeated_groups
        if not use_fixed_fallback and not self._flmm.available:
            self.statusMessage.emit(
                "R + fastFMM not available. Single-file fallback is available, but true repeated-subject FLMM needs R + fastFMM.",
                8000,
            )
            return

        if use_fixed_fallback:
            status = "Fitting single-file functional fixed-effect model..."
        else:
            status = "Fitting FLMM via fastFMM - this may take a while..."
        self.statusMessage.emit(status, 0)
        self._progress_start(status, 0)
        QtWidgets.QApplication.processEvents()

        result = self._flmm.fit(
            mat, tvec, design,
            formula_fixed=formula,
            random_effects=random_eff,
            group_var=group_var,
            nknots_min=nknots,
            num_boots=num_boots,
        )
        self._flmm_result = result
        result.stats = dict(result.stats or {})
        result.stats["term_labels"] = dict(term_labels or {})
        self._store_flmm_trace_input(mat, tvec, row_labels, design, term_labels, formula_terms, scope)
        significance_by_term = self._compute_flmm_curve_significance(result, formula_terms, term_labels)

        importance_mode = str(self.combo_flmm_importance.currentData(QtCore.Qt.ItemDataRole.UserRole) or "fast")
        if importance_mode == "loo":
            self.statusMessage.emit("Calculating FLMM leave-one-feature-out AIC contribution...", 0)
            QtWidgets.QApplication.processEvents()
            importance_rows = self._compute_flmm_leave_one_out(
                mat,
                tvec,
                design,
                formula,
                random_eff,
                group_var,
                nknots,
                result,
                term_labels,
            )
        elif importance_mode == "perm":
            importance_rows = self._compute_flmm_coefficient_importance(result, formula_terms, term_labels)
            n_perm = int(self.spin_flmm_perm.value()) if hasattr(self, "spin_flmm_perm") else 200
            self.statusMessage.emit(
                f"Running FLMM permutation contribution test ({n_perm} iter)...", 0,
            )
            QtWidgets.QApplication.processEvents()
            perm_results = self._compute_flmm_permutation_contribution(
                mat, tvec, design, formula, random_eff, group_var, nknots,
                result, formula_terms, term_labels, n_perm=n_perm,
                use_fastfmm=(not use_fixed_fallback),
            )
            self._merge_flmm_permutation_rows(importance_rows, perm_results)
        elif importance_mode == "fast":
            importance_rows = self._compute_flmm_coefficient_importance(result, formula_terms, term_labels)
        else:
            importance_rows = []
        self._merge_flmm_significance_rows(importance_rows, significance_by_term)
        result.feature_importance = importance_rows
        importance_value_key = (
            "delta_aic"
            if importance_mode == "loo" and any(np.isfinite(float(row.get("delta_aic", float("nan")))) for row in importance_rows)
            else "mean_abs_coefficient"
        )
        importance_rows.sort(key=lambda item: (
            bool(item.get("significant", False)),
            np.isfinite(item.get(importance_value_key, np.nan)),
            float(item.get(importance_value_key, -np.inf)) if np.isfinite(item.get(importance_value_key, np.nan)) else -np.inf,
            float(item.get("mean_abs_coefficient", -np.inf)) if np.isfinite(item.get("mean_abs_coefficient", np.nan)) else -np.inf,
        ), reverse=True)

        effective_group_var = str(result.stats.get("group_var", group_var)) if result.stats else group_var
        fallback_grouping = str(result.stats.get("fallback_grouping", "")) if result.stats else ""
        readable_formula = self._readable_flmm_formula(formula_terms, term_labels)
        backend_name = str(result.stats.get("backend", "fastFMM mixed model")) if result.stats else "fastFMM mixed model"
        # Build a structured HTML summary instead of a wall of plain text.
        self.txt_summary.setHtml(self._build_flmm_summary_html(
            result=result,
            importance_rows=importance_rows,
            importance_mode=importance_mode,
            term_labels=term_labels,
            formula_terms=formula_terms,
            backend_name=backend_name,
            scope=scope,
            n_rows=n_rows,
            effective_group_var=effective_group_var,
            readable_formula=readable_formula,
            fallback_grouping=fallback_grouping,
            missing_terms=missing_terms,
            fit_terms=fit_terms,
            dropped_terms=dropped_terms,
            use_fixed_fallback=use_fixed_fallback,
            num_boots=num_boots,
        ))
        # Below is the legacy plain-text builder; left intact so the GLM
        # summary path (and any external callers reading txt_summary) keep
        # working if HTML rendering ever needs to be disabled.
        summary_lines = [
            "FLMM / functional model summary",
            f"Backend: {backend_name}",
            f"Scope: {'animal/group rows' if scope == 'animal' else 'trial rows'}",
            f"Rows: {n_rows}",
            f"ID variable: {effective_group_var}",
            f"Model formula: {readable_formula}",
        ]
        if result.stats:
            n_timepoints = int(float(result.stats.get("n_timepoints", np.asarray(tvec).size) or 0))
            n_terms = int(float(result.stats.get("n_terms", len(result.coefficients or {})) or 0))
            if n_timepoints or n_terms:
                summary_lines.append(f"Curves fit: {n_terms} term(s), {n_timepoints} timepoint(s)")
            valid_timepoints = float(result.stats.get("valid_timepoints", float("nan")))
            skipped_timepoints = float(result.stats.get("skipped_timepoints", float("nan")))
            if np.isfinite(valid_timepoints) and np.isfinite(skipped_timepoints) and skipped_timepoints > 0:
                summary_lines.append(
                    f"Estimable timepoints: {int(valid_timepoints)}; skipped because of missing/rank-deficient rows: {int(skipped_timepoints)}"
                )
            for note in result.stats.get("fit_notes", []) or []:
                summary_lines.append(f"Note: {note}")
            if result.stats.get("variance_unavailable"):
                summary_lines.append("Note: coefficient uncertainty is unavailable for this fit.")
        if missing_terms:
            readable_missing = [term_labels.get(term, term) for term in missing_terms]
            summary_lines.append(f"Auto formula used because saved terms were unavailable: {', '.join(readable_missing)}")
        if fallback_grouping:
            summary_lines.append(f"Grouping note: {fallback_grouping}")
        if result.stats:
            mean_abs_stat = float(result.stats.get("mean_abs_coefficient", float("nan")))
            peak_abs_stat = float(result.stats.get("peak_abs_coefficient", float("nan")))
            summary_lines.extend([
                "",
                "Fit statistics:",
                f"  AIC = {result.stats.get('aic', float('nan')):.5g}",
                "  mean abs coefficient = "
                + (f"{mean_abs_stat:.5g}" if np.isfinite(mean_abs_stat) else "n/a"),
                "  peak abs coefficient = "
                + (f"{peak_abs_stat:.5g}" if np.isfinite(peak_abs_stat) else "n/a"),
            ])
        summary_lines.extend([""] + self._flmm_curve_summary_lines(result, formula_terms, term_labels))
        tested_sig = [row for row in importance_rows if row.get("significance_status") == "ok"]
        significant_rows = [row for row in tested_sig if row.get("significant", False)]
        if importance_rows:
            summary_lines.extend([
                "",
                "Coefficient-curve significance:",
                "  Conservative test: coefficient CI widths -> z-test, Bonferroni over time bins, BH-FDR across variables.",
                f"  Tested variables: {len(tested_sig)}; significant at q < 0.05: {len(significant_rows)}",
            ])
            if significant_rows:
                for row in significant_rows[:10]:
                    q_value = float(row.get("q_value", float("nan")))
                    p_value = float(row.get("p_value", float("nan")))
                    frac = float(row.get("significant_fraction", 0.0))
                    q_text = f"q = {q_value:.3g}" if np.isfinite(q_value) else "q = n/a"
                    p_text = f"p = {p_value:.3g}" if np.isfinite(p_value) else "p = n/a"
                    summary_lines.append(
                        f"  {row['label']}: {q_text}, {p_text}, significant bins = {frac:.1%}"
                    )
            else:
                unavailable = [row for row in importance_rows if row.get("significance_status") != "ok"]
                if unavailable and not tested_sig:
                    summary_lines.append("  No coefficient CIs were available, so significance could not be assessed.")
        if importance_rows and importance_mode == "loo":
            summary_lines.extend([
                "",
                "Leave-one-feature-out contribution (reduced AIC - full AIC):",
            ])
            for row in importance_rows[:10]:
                delta = float(row.get("delta_aic", float("nan")))
                delta_text = f"{delta:.5g}" if np.isfinite(delta) else "n/a"
                reduced = float(row.get("reduced_aic", float("nan")))
                reduced_text = f"{reduced:.5g}" if np.isfinite(reduced) else "n/a"
                mean_abs = float(row.get("mean_abs_coefficient", float("nan")))
                mean_text = f"{mean_abs:.5g}" if np.isfinite(mean_abs) else "not estimable"
                q_value = float(row.get("q_value", float("nan")))
                q_text = f", q = {q_value:.3g}" if np.isfinite(q_value) else ""
                sig_text = " [significant]" if row.get("significant", False) else ""
                summary_lines.append(
                    f"  {row['label']}: delta AIC = {delta_text}, "
                    f"reduced AIC = {reduced_text}, mean abs coef = {mean_text}{q_text}{sig_text}"
                )
            failed = [row for row in importance_rows if row.get("status") != "ok"]
            if failed:
                summary_lines.append(f"  {len(failed)} reduced FLMM fits failed; see log for details.")
            if num_boots > 0:
                summary_lines.append("  Leave-one-out comparison uses analytic fits; bootstrap CIs are not repeated.")
        elif importance_rows and importance_mode == "fast":
            summary_lines.extend([
                "",
                "Fast coefficient contribution (single fitted model):",
            ])
            for row in importance_rows[:10]:
                mean_abs = float(row.get("mean_abs_coefficient", float("nan")))
                peak_abs = float(row.get("peak_abs_coefficient", float("nan")))
                if not np.isfinite(mean_abs) and not np.isfinite(peak_abs):
                    summary_lines.append(
                        f"  {row['label']}: not estimable ({row.get('status', 'no finite coefficient bins')})"
                    )
                    continue
                q_value = float(row.get("q_value", float("nan")))
                q_text = f", q = {q_value:.3g}" if np.isfinite(q_value) else ""
                sig_text = " [significant]" if row.get("significant", False) else ""
                summary_lines.append(
                    f"  {row['label']}: mean abs coef = {mean_abs:.5g}, "
                    f"peak abs coef = {peak_abs:.5g}{q_text}{sig_text}"
                )
            if str(result.stats.get("backend", "")) == "python_functional_fixed_effect":
                summary_lines.append("  Leave-one-out AIC is disabled by default because it refits the functional model once per predictor.")
            else:
                summary_lines.append("  Leave-one-out AIC is disabled by default because it refits fastFMM once per predictor.")
        elif importance_mode == "off":
            summary_lines.extend(["", "Feature contribution: off."])
        if importance_rows:
            summary_lines.extend([""] + self._flmm_interpretation_lines(result, importance_rows, use_fixed_fallback))
        if fit_terms:
            readable = [term_labels.get(term, term) for term in fit_terms]
            summary_lines.append("Model variables: " + "; ".join(readable))
        if dropped_terms:
            summary_lines.append("Dropped covariates: " + ", ".join(dropped_terms))
        # The HTML summary was already installed above; keep summary_lines as a
        # text snapshot only (used by exports / save_report and as a fallback
        # text representation for tests). Do NOT overwrite the rich HTML.
        self._last_flmm_summary_text = "\n".join(summary_lines)
        self._plot_flmm_coefficients(result)
        self._plot_feature_importance(
            importance_rows,
            value_key=importance_value_key,
            title="FLMM leave-one-feature-out contribution" if importance_mode == "loo" else "FLMM coefficient contribution",
            y_label="Delta AIC" if importance_value_key == "delta_aic" else "Mean abs coefficient",
        )
        if hasattr(self, "tabs_workspace") and importance_rows:
            self.tabs_workspace.setCurrentWidget(self.plot_importance.parentWidget())
        if backend_name == "python_functional_fixed_effect":
            self.statusMessage.emit("Functional fixed-effect model fit complete.", 5000)
        else:
            self.statusMessage.emit("FLMM fit complete.", 5000)

    def _fit_flmm(self):
        self._fit_flmm_group()
        return
        if not self._flmm.available:
            self.statusMessage.emit(
                "R + fastFMM not available. Please install R, rpy2, and the fastFMM R package.", 8000
            )
            return

        if self._psth_mat is None or self._psth_tvec is None:
            self.statusMessage.emit("No PSTH matrix — compute PSTH first.", 5000)
            return

        mat = self._psth_mat
        tvec = self._psth_tvec
        if mat.ndim != 2 or mat.shape[0] < 2:
            self.statusMessage.emit("Need at least 2 trials for FLMM.", 5000)
            return

        n_trials = mat.shape[0]

        # Build the design dict from the predictor list and file_ids
        design: Dict[str, np.ndarray] = {}
        group_var = self.edit_group_var.text().strip() or "subject"

        # Default: use file IDs as subject labels if available
        if self._file_ids and len(self._file_ids) > 0:
            # In group mode, each row = one animal; in individual, each row = one trial
            if len(self._file_ids) == n_trials:
                design[group_var] = np.array(self._file_ids)
            else:
                # Per-trial: assign subject based on which file the trial came from
                design[group_var] = np.array([f"subj_{i}" for i in range(n_trials)])
        else:
            design[group_var] = np.array([f"subj_{i}" for i in range(n_trials)])

        # Add any custom predictors from the list
        for i in range(self.list_predictors.count()):
            pred_name = self.list_predictors.item(i).text()
            if pred_name == group_var or pred_name == "events":
                continue
            # Placeholder: user must supply these via design extensions
            if pred_name not in design:
                design[pred_name] = np.zeros(n_trials, float)

        formula = self.edit_formula.text().strip() or "Y.obs ~ 1"
        random_eff = self.edit_random.text().strip() or "~1"
        nknots = self.spin_nknots.value() if self.spin_nknots.value() > 0 else None
        num_boots = self.spin_boots.value()

        self.statusMessage.emit("Fitting FLMM via fastFMM — this may take a while...", 0)
        QtWidgets.QApplication.processEvents()

        result = self._flmm.fit(
            mat, tvec, design,
            formula_fixed=formula,
            random_effects=random_eff,
            group_var=group_var,
            nknots_min=nknots,
            num_boots=num_boots,
        )
        self._flmm_result = result

        self.txt_summary.setPlainText(result.summary_text)
        self._plot_flmm_coefficients(result)
        self.statusMessage.emit("FLMM fit complete.", 5000)

    # ------------------------------------------------------------------
    # FLMM coefficient filter / layout helpers
    # ------------------------------------------------------------------

    def _flmm_term_labels(self, result: FLMMResult) -> Dict[str, str]:
        try:
            return dict((result.stats or {}).get("term_labels", {}) or {})
        except Exception:
            return {}

    def _flmm_significance_by_coeff(self, result: FLMMResult) -> Dict[str, Dict[str, Any]]:
        sig_by_coeff: Dict[str, Dict[str, Any]] = {}
        for row in (result.feature_importance or []):
            feature = str(row.get("feature", ""))
            coeff_name = self._term_coefficient_name(result, feature)
            if coeff_name:
                sig_by_coeff[coeff_name] = row
        return sig_by_coeff

    def _sync_flmm_filter(self, result: FLMMResult) -> None:
        if not hasattr(self, "list_flmm_filter"):
            return
        names = list(result.coefficients.keys())
        term_labels = self._flmm_term_labels(result)
        self._flmm_coeff_filter_guard = True
        try:
            self.list_flmm_filter.clear()
            for name in names:
                # Default visibility: keep prior preference; hide the intercept
                # by default on first sync (it sits at a different y-scale and
                # dominates the auto-range).
                if name in self._flmm_coeff_visible:
                    visible = bool(self._flmm_coeff_visible[name])
                else:
                    visible = "(Intercept)" not in str(name)
                self._flmm_coeff_visible[name] = visible
                label = self._compact_feature_label(term_labels.get(name, name), 34)
                item = QtWidgets.QListWidgetItem(label)
                item.setData(QtCore.Qt.ItemDataRole.UserRole, name)
                item.setFlags(item.flags() | QtCore.Qt.ItemFlag.ItemIsUserCheckable)
                item.setCheckState(QtCore.Qt.CheckState.Checked if visible else QtCore.Qt.CheckState.Unchecked)
                item.setForeground(QtGui.QColor(self._kernel_color(name)))
                self.list_flmm_filter.addItem(item)
        finally:
            self._flmm_coeff_filter_guard = False

    def _on_flmm_filter_changed(self, item: QtWidgets.QListWidgetItem) -> None:
        if self._flmm_coeff_filter_guard:
            return
        key = item.data(QtCore.Qt.ItemDataRole.UserRole)
        if isinstance(key, str) and key:
            self._flmm_coeff_visible[key] = item.checkState() == QtCore.Qt.CheckState.Checked
        if self._flmm_result is not None:
            self._plot_flmm_coefficients(self._flmm_result, refresh_filter=False)

    def _set_all_flmm_visible(self, visible: bool) -> None:
        if not hasattr(self, "list_flmm_filter"):
            return
        self._flmm_coeff_filter_guard = True
        try:
            for i in range(self.list_flmm_filter.count()):
                it = self.list_flmm_filter.item(i)
                key = it.data(QtCore.Qt.ItemDataRole.UserRole)
                if isinstance(key, str) and key:
                    self._flmm_coeff_visible[key] = bool(visible)
                it.setCheckState(QtCore.Qt.CheckState.Checked if visible else QtCore.Qt.CheckState.Unchecked)
        finally:
            self._flmm_coeff_filter_guard = False
        if self._flmm_result is not None:
            self._plot_flmm_coefficients(self._flmm_result, refresh_filter=False)

    def _on_flmm_layout_changed(self, *_) -> None:
        if self._flmm_result is not None:
            self._plot_flmm_coefficients(self._flmm_result, refresh_filter=False)

    def _flmm_clear_grid(self) -> None:
        for pw in self._flmm_coeff_grid_plots:
            try:
                self._flmm_grid_layout.removeWidget(pw)
                pw.setParent(None)
                pw.deleteLater()
            except Exception:
                pass
        self._flmm_coeff_grid_plots = []

    def _plot_flmm_coefficients(self, result: FLMMResult, refresh_filter: bool = True):
        if refresh_filter:
            self._sync_flmm_filter(result)

        layout_mode = "overlay"
        if hasattr(self, "combo_flmm_layout"):
            layout_mode = str(self.combo_flmm_layout.currentData() or "overlay")

        tvec = np.asarray(result.tvec, float)
        term_labels = self._flmm_term_labels(result)
        sig_by_coeff = self._flmm_significance_by_coeff(result)
        items_visible = [
            (name, coeff)
            for name, coeff in result.coefficients.items()
            if self._flmm_coeff_visible.get(name, True)
        ]

        # If "Single focus" mode is selected and 0 or >1 items are visible,
        # fall back to overlay so something reasonable is drawn.
        if layout_mode == "single" and len(items_visible) != 1:
            if len(items_visible) == 0:
                layout_mode = "overlay"
            else:
                # Auto-pick: the first significant coefficient, else the first.
                pick = next((nm for nm, _c in items_visible
                             if sig_by_coeff.get(nm, {}).get("significant", False)), None)
                if pick is None:
                    pick = items_visible[0][0]
                items_visible = [(pick, result.coefficients[pick])]

        if layout_mode == "grid":
            self.stack_flmm_plots.setCurrentWidget(self.scroll_flmm_grid)
            self._flmm_clear_grid()
            cols = 2 if len(items_visible) > 4 else 1
            for idx, (name, coeff) in enumerate(items_visible):
                pw_i = pg.PlotWidget()
                self._style_plot(pw_i)
                pw_i.setMinimumHeight(200)
                self._draw_flmm_coeff_curve(pw_i, name, np.asarray(coeff, float), tvec,
                                            result, term_labels, sig_by_coeff,
                                            show_legend=False, big=False)
                self._flmm_coeff_grid_plots.append(pw_i)
                self._flmm_grid_layout.addWidget(pw_i, idx // cols, idx % cols)
            return

        # Overlay or single focus.
        self.stack_flmm_plots.setCurrentWidget(self.plot_coeff)
        pw = self.plot_coeff
        pw.clear()
        try:
            pw.getPlotItem().legend.clear()
        except Exception:
            pass

        big = (layout_mode == "single")
        for name, coeff in items_visible:
            self._draw_flmm_coeff_curve(pw, name, np.asarray(coeff, float), tvec,
                                        result, term_labels, sig_by_coeff,
                                        show_legend=True, big=big)

        if result.coefficients and not (result.ci_lower or result.joint_ci_lower):
            txt = pg.TextItem("Point estimates only; variance inference unavailable.",
                              color="#f5a97f", anchor=(0, 0))
            pw.addItem(txt)
            finite_t = np.asarray(tvec, float)
            if finite_t.size:
                txt.setPos(float(np.nanmin(finite_t)), 0.0)

        title = ("FLMM coefficient (single-focus)"
                 if big and items_visible else "FLMM coefficient curves")
        pw.setTitle(title)
        pw.setLabel("bottom", "Time", units="s")
        pw.setLabel("left", "Coefficient")
        pw.addLine(y=0, pen=pg.mkPen("#5a6274", width=1, style=QtCore.Qt.PenStyle.DashLine))
        pw.addLine(x=0, pen=pg.mkPen("#5a6274", width=1, style=QtCore.Qt.PenStyle.DashLine))

    def _draw_flmm_coeff_curve(
        self,
        pw: pg.PlotWidget,
        name: str,
        coeff: np.ndarray,
        tvec: np.ndarray,
        result: FLMMResult,
        term_labels: Dict[str, str],
        sig_by_coeff: Dict[str, Dict[str, Any]],
        *,
        show_legend: bool,
        big: bool,
    ) -> None:
        color = self._kernel_color(name)
        sig_row = sig_by_coeff.get(str(name), {})
        is_sig = bool(sig_row.get("significant", False))
        line_w = (3.5 if big else 2.4) if is_sig else (2.4 if big else 1.5)
        pen = pg.mkPen(color, width=line_w)
        label = term_labels.get(name, name)
        if is_sig:
            q_value = float(sig_row.get("q_value", float("nan")))
            label = f"{label} *" if not np.isfinite(q_value) else f"{label} q={q_value:.2g}"
        plot_kwargs = {"pen": pen}
        if show_legend:
            plot_kwargs["name"] = label
        pw.plot(tvec, coeff, **plot_kwargs)

        # Joint CI band (filled).
        if name in result.joint_ci_lower and name in result.joint_ci_upper:
            ci_lo = np.asarray(result.joint_ci_lower[name], float)
            ci_hi = np.asarray(result.joint_ci_upper[name], float)
            fill_color = QtGui.QColor(color)
            fill_color.setAlpha(70 if big else 40)
            fill = pg.FillBetweenItem(
                pg.PlotDataItem(tvec, ci_lo),
                pg.PlotDataItem(tvec, ci_hi),
                brush=fill_color,
            )
            pw.addItem(fill)

        # Pointwise CI dashed lines.
        if name in result.ci_lower and name in result.ci_upper:
            dash_pen = pg.mkPen(color, width=1.2 if big else 1,
                                style=QtCore.Qt.PenStyle.DashLine)
            pw.plot(tvec, result.ci_lower[name], pen=dash_pen)
            pw.plot(tvec, result.ci_upper[name], pen=dash_pen)

        # In grid / single mode add the title so each panel is self-explanatory.
        if not show_legend or big:
            human = term_labels.get(name, name)
            pw.setTitle(human if not is_sig else f"{human}  (significant)")
            pw.setLabel("bottom", "Time", units="s")
            pw.setLabel("left", "Coefficient")
            pw.addLine(y=0, pen=pg.mkPen("#5a6274", width=1, style=QtCore.Qt.PenStyle.DashLine))
            pw.addLine(x=0, pen=pg.mkPen("#5a6274", width=1, style=QtCore.Qt.PenStyle.DashLine))
