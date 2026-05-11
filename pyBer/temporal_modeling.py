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
        dt = float(np.nanmedian(np.diff(time)))
        if not np.isfinite(dt) or dt <= 0:
            raise ValueError("GLM time vector must be strictly increasing.")
        pre_samp = int(round(abs(kernel_window[0]) / dt))
        post_samp = int(round(abs(kernel_window[1]) / dt))
        kernel_len = max(2, pre_samp + post_samp)

        if basis_type == "bspline":
            B = _bspline_basis(n_basis, kernel_len)
        elif basis_type == "fir":
            B = _fir_basis(n_basis, kernel_len)
        else:
            B = _raised_cosine_basis(n_basis, kernel_len)

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

            # Convolve impulse with each basis function
            part = np.zeros((T, n_basis), float)
            for b in range(n_basis):
                # Pad the basis to align with pre_samp offset
                kernel = np.zeros(kernel_len)
                kernel[:] = B[:, b]
                conv = np.convolve(input_vec, kernel, mode="full")[:T]
                # Shift so that the kernel starts at -pre_samp
                if pre_samp > 0:
                    part[:, b] = np.roll(conv, -pre_samp)
                    part[-pre_samp:, b] = 0.0
                else:
                    part[:, b] = conv

            X_parts.append(part)
            used_predictors.append(pred_name)
            for b in range(n_basis):
                col_names.append(f"{pred_name}_b{b}")

        X = np.hstack(X_parts) if X_parts else np.zeros((T, 0))
        # Add intercept
        X = np.column_stack([np.ones(T), X])
        col_names.insert(0, "intercept")
        return X, col_names, n_basis, used_predictors

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
        valid = np.isfinite(signal)
        Xv = X[valid]
        yv = signal[valid]

        # Fit
        if regularization == "lasso":
            try:
                from sklearn.linear_model import Lasso as _Lasso
                model = _Lasso(alpha=alpha, max_iter=5000, fit_intercept=False)
                model.fit(Xv, yv)
                beta = model.coef_
            except ImportError:
                _LOG.warning("sklearn not available; falling back to ridge")
                beta = self._ridge_fit(Xv, yv, alpha)
        elif regularization == "ridge":
            beta = self._ridge_fit(Xv, yv, alpha)
        else:  # ols
            beta, *_ = np.linalg.lstsq(Xv, yv, rcond=None)

        y_pred = X @ beta
        residuals = signal - y_pred
        ss_res = np.nansum(residuals ** 2)
        ss_tot = np.nansum((signal - np.nanmean(signal)) ** 2)
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

        # Extract kernels
        dt = np.median(np.diff(time))
        pre_samp = int(round(abs(kernel_window[0]) / dt))
        post_samp = int(round(abs(kernel_window[1]) / dt))
        kernel_len = max(2, pre_samp + post_samp)

        if basis_type == "bspline":
            B = _bspline_basis(n_b, kernel_len)
        elif basis_type == "fir":
            B = _fir_basis(n_b, kernel_len)
        else:
            B = _raised_cosine_basis(n_b, kernel_len)

        kernel_tvec = np.linspace(kernel_window[0], kernel_window[1], kernel_len)
        kernels: Dict[str, np.ndarray] = {}
        idx = 1  # skip intercept
        for pred_name in used_predictors:
            w = beta[idx:idx + n_b]
            kernels[pred_name] = B @ w
            idx += n_b

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
        _init_r()
        import rpy2.robjects as ro
        from rpy2.robjects import r as R
        from rpy2.robjects.packages import importr

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
            if n_trials < 4:
                raise ValueError(
                    "FLMM needs at least 4 rows when the selected grouping variable has fewer than two repeated levels. "
                    "Compute per-file/per-trial PSTH rows or choose a grouping variable with repeated samples."
                )
            block_name = "pyber_block"
            i = 2
            while block_name in design:
                block_name = f"pyber_block_{i}"
                i += 1
            n_blocks = min(4, max(2, n_trials // 2))
            design[block_name] = np.asarray([f"block_{(idx % n_blocks) + 1}" for idx in range(n_trials)], dtype=object)
            group_var_model = block_name
            fallback_grouping = (
                f"Grouping variable '{group_var}' had {unique_groups.size} level(s) across {n_trials} rows. "
                f"Used exploratory block grouping '{group_var_model}' with {n_blocks} levels so fastFMM can fit."
            )

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

            if se_mat is not None:
                beta_lb = beta_hat - 1.96 * se_mat
                beta_ub = beta_hat + 1.96 * se_mat
            else:
                try:
                    if result_names and ("betaHat.LB" not in result_names or "betaHat.UB" not in result_names):
                        raise KeyError("betaHat CI")
                    beta_lb = np.atleast_2d(np.array(R('as.matrix')(fui_result.rx2("betaHat.LB")), dtype=float))
                    beta_ub = np.atleast_2d(np.array(R('as.matrix')(fui_result.rx2("betaHat.UB")), dtype=float))
                except Exception:
                    beta_lb = beta_hat.copy()
                    beta_ub = beta_hat.copy()

            try:
                qn = np.asarray(fui_result.rx2("qn"), float).ravel()
            except Exception:
                qn = np.array([], float)

            for i, name in enumerate(term_names):
                coefficients[name] = beta_hat[i, :]
                ci_lower[name] = beta_lb[i, :]
                ci_upper[name] = beta_ub[i, :]

            # Joint CIs (may not always be present)
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
                else:
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
            if aic_val is not None:
                summary_parts.append(f"AIC = {aic_val:.1f}")
            coeff_abs_peaks: List[float] = []
            coeff_abs_means: List[float] = []
            for name in term_names:
                coeff = np.asarray(coefficients[name], float)
                coeff_abs_peaks.append(float(np.nanmax(np.abs(coeff))) if coeff.size else float("nan"))
                coeff_abs_means.append(float(np.nanmean(np.abs(coeff))) if coeff.size else float("nan"))
                summary_parts.append(
                    f"  {name}: mean coef = {np.nanmean(coeff):.4f}, "
                    f"mean abs = {np.nanmean(np.abs(coeff)):.4f}, peak abs = {np.nanmax(np.abs(coeff)):.4f}"
                )
            summary_text = "\n".join(summary_parts)
            stats = {
                "n_trials": float(n_trials),
                "n_timepoints": float(n_time),
                "n_terms": float(len(term_names)),
                "aic": float(aic_val) if aic_val is not None else float("nan"),
                "mean_abs_coefficient": float(np.nanmean(coeff_abs_means)) if coeff_abs_means else float("nan"),
                "peak_abs_coefficient": float(np.nanmax(coeff_abs_peaks)) if coeff_abs_peaks else float("nan"),
                "formula": formula_text,
                "group_var": group_var_model,
                "requested_group_var": group_var,
                "fallback_grouping": fallback_grouping,
                "fit_notes": list(fit_notes),
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
        self._glm = ContinuousGLM()
        self._flmm = TrialFLMM()
        self._glm_result: Optional[GLMResult] = None
        self._flmm_result: Optional[FLMMResult] = None

        # Per-file fits (filled by batch / group runs).
        self._glm_results_by_file: Dict[str, GLMResult] = {}
        self._flmm_results_by_file: Dict[str, FLMMResult] = {}
        self._fit_summary_by_file: Dict[str, str] = {}
        self._group_glm_summary: Dict[str, Any] = {}
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
        self.btn_export.setText("Export  v")
        self.btn_export.setToolTip("Export fit results (CSV / JSON / HDF5) or save plots as images.")
        self.btn_export.setPopupMode(QtWidgets.QToolButton.ToolButtonPopupMode.InstantPopup)
        self.menu_export = QtWidgets.QMenu(self.btn_export)
        self.act_export_kernels_csv = self.menu_export.addAction("Active fit - kernels CSV...")
        self.act_export_importance_csv = self.menu_export.addAction("Active fit - importance CSV...")
        self.act_export_summary_txt = self.menu_export.addAction("Active fit - summary text...")
        self.menu_export.addSeparator()
        self.act_export_group_kernels_csv = self.menu_export.addAction("Group - kernels CSV...")
        self.act_export_group_importance_csv = self.menu_export.addAction("Group - importance CSV...")
        self.menu_export.addSeparator()
        self.act_export_state_json = self.menu_export.addAction("All cached fits - JSON snapshot...")
        self.menu_export.addSeparator()
        self.act_export_current_plot = self.menu_export.addAction("Current plot - PNG...")
        self.act_export_current_plot_svg = self.menu_export.addAction("Current plot - SVG...")
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
        self.act_export_kernels_csv.triggered.connect(self._export_kernels_csv)
        self.act_export_importance_csv.triggered.connect(self._export_importance_csv)
        self.act_export_summary_txt.triggered.connect(self._export_summary_txt)
        self.act_export_group_kernels_csv.triggered.connect(self._export_group_kernels_csv)
        self.act_export_group_importance_csv.triggered.connect(self._export_group_importance_csv)
        self.act_export_state_json.triggered.connect(self._export_state_json)
        self.act_export_current_plot.triggered.connect(lambda: self._export_current_plot("png"))
        self.act_export_current_plot_svg.triggered.connect(lambda: self._export_current_plot("svg"))
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
        self.combo_flmm_importance.addItem("Leave-one-out AIC (slow)", "loo")
        self.combo_flmm_importance.addItem("Off", "off")
        self.combo_flmm_importance.setToolTip(
            "Leave-one-out refits fastFMM once per predictor and can be very slow."
        )
        fl.addRow("Contribution", self.combo_flmm_importance)
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
        self.btn_export_temporal_results = QtWidgets.QPushButton("Export results")
        self.btn_export_temporal_results.setToolTip("Export model summary, predictions, kernels, and importance tables.")
        self.btn_export_temporal_plots = QtWidgets.QPushButton("Export plots")
        self.btn_export_temporal_plots.setToolTip("Export the temporal modeling plots as PNG and PDF.")
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
        self.plot_coeff = pg.PlotWidget(title="FLMM coefficient curves")
        self._style_plot(self.plot_coeff)
        flmm_lay.addWidget(self.plot_coeff, 1)
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
        self.plot_group_kernels = pg.PlotWidget(title="Group kernels (mean +/- SEM across animals)")
        self._style_plot(self.plot_group_kernels)
        group_lay.addWidget(self.plot_group_kernels, 1)
        self.plot_group_importance = pg.PlotWidget(title="Group leave-one-out contribution (mean across animals)")
        self._style_plot(self.plot_group_importance)
        group_lay.addWidget(self.plot_group_importance, 1)
        self.tabs_workspace.addTab(group_page, "Group")

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
        if hasattr(self, "combo_illustration_feature"):
            self.combo_illustration_feature.currentIndexChanged.connect(self._on_illustration_feature_changed)
        if hasattr(self, "btn_export_temporal_results"):
            self.btn_export_temporal_results.clicked.connect(self._export_temporal_results)
        if hasattr(self, "btn_export_temporal_plots"):
            self.btn_export_temporal_plots.clicked.connect(self._export_temporal_plots)

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
    def _serialize_glm_result(r: GLMResult) -> Dict[str, Any]:
        return {
            "predictor_names": list(r.predictor_names or []),
            "kernels": {k: np.asarray(v, float).tolist() for k, v in (r.kernels or {}).items()},
            "kernel_tvec": np.asarray(r.kernel_tvec, float).tolist(),
            "time": np.asarray(r.time, float).tolist() if r.time is not None else [],
            "y_pred": np.asarray(r.y_pred, float).tolist() if r.y_pred is not None else [],
            "y_actual": np.asarray(r.y_actual, float).tolist() if r.y_actual is not None else [],
            "residuals": np.asarray(r.residuals, float).tolist() if r.residuals is not None else [],
            "r2": float(r.r2),
            "coefficients": np.asarray(r.coefficients, float).tolist() if r.coefficients is not None else [],
            "stats": {k: (float(v) if isinstance(v, (int, float, np.floating, np.integer)) else v) for k, v in (r.stats or {}).items()},
            "feature_importance": [
                {k: (float(v) if isinstance(v, (int, float, np.floating, np.integer)) else v)
                 for k, v in (row or {}).items()}
                for row in (r.feature_importance or [])
            ],
        }

    @staticmethod
    def _deserialize_glm_result(payload: Dict[str, Any]) -> Optional[GLMResult]:
        if not isinstance(payload, dict):
            return None
        try:
            kernels = {k: np.asarray(v, float) for k, v in (payload.get("kernels", {}) or {}).items()}
            return GLMResult(
                predictor_names=list(payload.get("predictor_names", []) or []),
                kernels=kernels,
                kernel_tvec=np.asarray(payload.get("kernel_tvec", []), float),
                time=np.asarray(payload.get("time", []), float),
                y_pred=np.asarray(payload.get("y_pred", []), float),
                y_actual=np.asarray(payload.get("y_actual", []), float),
                residuals=np.asarray(payload.get("residuals", []), float),
                r2=float(payload.get("r2", 0.0)),
                coefficients=np.asarray(payload.get("coefficients", []), float),
                # design_matrix is intentionally not persisted (large, redundant);
                # contribution overlays will simply be unavailable until refit.
                design_matrix=np.zeros((0, 0), float),
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
                    self.txt_summary.setPlainText(self._fit_summary_by_file.get(self._active_file_id, ""))
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
        return written

    def _export_temporal_results(self) -> None:
        out_dir = self._export_dir_from_user("Export temporal modeling results")
        if not out_dir:
            return
        prefix = self._export_prefix()
        summary_path = os.path.join(out_dir, f"{prefix}_summary.txt")
        with open(summary_path, "w", encoding="utf-8") as fh:
            fh.write(self.txt_summary.toPlainText() if hasattr(self, "txt_summary") else "")

        state_path = os.path.join(out_dir, f"{prefix}_state.json")
        with open(state_path, "w", encoding="utf-8") as fh:
            json.dump(self.serialize_state(), fh, indent=2, allow_nan=True)

        written = [summary_path, state_path]
        written.extend(self._export_glm_tables(out_dir, prefix))
        written.extend(self._export_flmm_tables(out_dir, prefix))
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

    def _export_temporal_plots(self) -> None:
        out_dir = self._export_dir_from_user("Export temporal modeling plots")
        if not out_dir:
            return
        prefix = self._export_prefix()
        written: List[str] = []
        for suffix, widget in self._temporal_plot_targets():
            base = os.path.join(out_dir, self._safe_filename(f"{prefix}_{suffix}"))
            try:
                written.extend(self._render_widget_png_pdf(widget, base))
            except Exception as exc:
                _LOG.warning("Temporal plot export failed for %s: %s", suffix, exc)
        if written:
            self.statusMessage.emit(f"Exported temporal plots: {len(written)} file(s).", 5000)
        else:
            QtWidgets.QMessageBox.warning(self, "Export plots", "No temporal plots could be exported.")

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
            dt = float(np.nanmedian(np.diff(time)))
        except Exception:
            dt = 0.0
        if not np.isfinite(dt) or dt <= 0:
            dt = 1.0
        pre_samp = int(round(abs(kernel_window[0]) / dt))
        post_samp = int(round(abs(kernel_window[1]) / dt))
        kernel_len = max(2, pre_samp + post_samp)
        if basis_type == "bspline":
            basis_mat = _bspline_basis(n_basis, kernel_len)
        elif basis_type == "fir":
            basis_mat = _fir_basis(n_basis, kernel_len)
        else:
            basis_mat = _raised_cosine_basis(n_basis, kernel_len)

        T_full = int(time.size)

        def _columns_for_vector(input_vec: np.ndarray) -> Optional[np.ndarray]:
            v = np.asarray(input_vec, float)
            if v.size != T_full or not np.any(np.abs(v) > 1e-12):
                return None
            cols = np.zeros((T_full, n_basis), float)
            for b in range(n_basis):
                kernel = basis_mat[:, b]
                conv = np.convolve(v, kernel, mode="full")[:T_full]
                if pre_samp > 0:
                    conv = np.roll(conv, -pre_samp)
                    conv[-pre_samp:] = 0.0
                cols[:, b] = conv
            return cols

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
        alpha_val = float(alpha)

        def _ridge_solve(X: np.ndarray, y: np.ndarray) -> np.ndarray:
            n_cols = X.shape[1]
            I = np.eye(n_cols)
            I[0, 0] = 0.0
            return np.linalg.solve(X.T @ X + alpha_val * I, X.T @ y)
        row_meta: Dict[int, Tuple[Dict[str, Any], float]] = {}
        for row in rows:
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
                if regularization == "ols":
                    beta, *_ = np.linalg.lstsq(Xv, signal_v, rcond=None)
                else:
                    beta = _ridge_solve(Xv, signal_v)
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
                p_value = float((1 + np.sum(null_arr >= obs_delta)) / (null_arr.size + 1))
                row["p_value"] = p_value
                row["significant"] = bool(p_value < 0.05 and obs_delta > 0)
                row["bootstrap_n"] = int(null_arr.size)
                row["null_delta_mean"] = float(np.nanmean(null_arr))
                row["null_delta_q95"] = float(np.nanpercentile(null_arr, 95))
            else:
                row["p_value"] = float("nan")
                row["significant"] = False
                row["bootstrap_n"] = 0
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
        return float(np.nanmean(np.abs(vals))) if vals.size else float("nan")

    @staticmethod
    def _term_coefficient_curve(result: FLMMResult, term: str) -> Optional[np.ndarray]:
        if not result.coefficients:
            return None
        clean = re.sub(r"[^0-9A-Za-z_]+", "", str(term).lower())
        for name, coeff in result.coefficients.items():
            if str(name) == str(term):
                return np.asarray(coeff, float)
        for name, coeff in result.coefficients.items():
            name_clean = re.sub(r"[^0-9A-Za-z_]+", "", str(name).lower())
            if clean and (clean in name_clean or name_clean in clean):
                return np.asarray(coeff, float)
        return None

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
            rows.append({
                "feature": term,
                "label": term_labels.get(term, term),
                "mean_abs_coefficient": float(np.nanmean(np.abs(vals))),
                "peak_abs_coefficient": float(np.nanmax(np.abs(vals))),
                "delta_aic": float("nan"),
                "status": "ok",
            })
        rows.sort(key=lambda item: (
            np.isfinite(item.get("mean_abs_coefficient", np.nan)),
            float(item.get("mean_abs_coefficient", -np.inf)) if np.isfinite(item.get("mean_abs_coefficient", np.nan)) else -np.inf,
        ), reverse=True)
        return rows

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
                    self.txt_summary.setPlainText(self._fit_summary_by_file.get(self._active_file_id, ""))
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
        self._refresh_file_widgets()
        if hasattr(self, "lbl_batch_status"):
            self.lbl_batch_status.setText("Cleared cached per-file fits.")
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
        result = self._glm_result
        if result is None or not result.feature_importance:
            self.statusMessage.emit("No leave-one-out importance to export.", 5000)
            return
        path = self._pick_save_path("temporal_importance.csv", "CSV files (*.csv)")
        if not path:
            return
        try:
            rows = result.feature_importance
            # Stable column order; pull all keys present across rows.
            preferred = ["feature", "label", "full_r2", "reduced_r2", "delta_r2",
                         "delta_mse", "contribution_pct", "p_value", "significant",
                         "bootstrap_n", "status"]
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
        scope_label = file_filter if file_filter else "all files"
        self._progress_start(f"Fitting GLM ({scope_label})", 0)
        result = self._glm.fit(
            np.asarray(dataset["time"], float),
            np.asarray(dataset["signal"], float),
            dataset["predictors"],
            kernel_window=kernel_win,
            n_basis=self.spin_n_basis.value(),
            basis_type=basis_type,
            regularization=regularization,
            alpha=self.spin_alpha.value(),
        )
        self._glm_result = result

        run_contrib = bool(getattr(self, "chk_run_contrib", None) is None or self.chk_run_contrib.isChecked())
        if run_contrib:
            self.statusMessage.emit("Calculating GLM leave-one-predictor-out contribution...", 0)
            QtWidgets.QApplication.processEvents()
            importance_rows = self._compute_glm_leave_one_out(
                dataset,
                result,
                kernel_win,
                basis_type,
                regularization,
                self.spin_alpha.value(),
            )
            n_boot = int(self.spin_glm_bootstrap.value())
            self._compute_glm_shift_bootstrap_significance(
                dataset,
                importance_rows,
                kernel_win,
                basis_type,
                regularization,
                self.spin_alpha.value(),
                n_boot,
            )
        else:
            importance_rows = []
            n_boot = 0
        result.feature_importance = importance_rows

        used_labels = [self._predictor_label(k) for k in result.predictor_names]
        n_jobs = int(self.spin_glm_jobs.value()) if hasattr(self, "spin_glm_jobs") else 1
        dropped_predictors = dataset.get("dropped_predictors", []) or []
        used_records = dataset.get("used_records", []) or []
        dropped_records = dataset.get("dropped_records", []) or []
        record_preview = ", ".join(str(v) for v in used_records[:6])
        if len(used_records) > 6:
            record_preview += "..."
        lines = [
            f"Continuous GLM - R^2 = {result.r2:.4f}",
            f"Recordings used: {len(used_records)} ({record_preview})",
            f"Samples fit: {int(dataset.get('valid_samples', 0))}",
            f"Predictors: {', '.join(used_labels)}",
            f"Basis: {self.combo_basis.currentText()}, n={self.spin_n_basis.value()}",
            f"Regularization: {self.combo_reg.currentText()}, alpha={self.spin_alpha.value():.3f}",
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
        ])
        if importance_rows:
            lines.extend(["", "Leave-one-predictor-out contribution (full - reduced):"])
            for row in importance_rows[:10]:
                p_value = float(row.get("p_value", float("nan")))
                p_text = f", p = {p_value:.4g}" if np.isfinite(p_value) else ""
                sig_text = " [significant]" if row.get("significant", False) else ""
                lines.append(
                    f"  {row['label']}: delta R^2 = {row['delta_r2']:.5g}, "
                    f"delta MSE = {row['delta_mse']:.5g}, reduced R^2 = {row['reduced_r2']:.5g}"
                    f"{p_text}{sig_text}"
                )
            if n_boot > 0:
                significant = [row for row in importance_rows if row.get("significant", False)]
                lines.append(f"  Significant predictors at p < 0.05: {len(significant)}")
            failed = [row for row in importance_rows if row.get("status") != "ok"]
            if failed:
                lines.append(f"  {len(failed)} reduced fits failed; see log for details.")
        if dropped_predictors:
            lines.append(f"Dropped predictors: {', '.join(str(v) for v in dropped_predictors)}")
        if dropped_records:
            lines.append(f"Dropped recordings: {', '.join(str(v) for v in dropped_records)}")
        summary_text = "\n".join(lines)
        self.txt_summary.setPlainText(summary_text)

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

        # Plot mean +/- SEM per predictor.
        plotted = 0
        for key, stack in kernel_stack.items():
            if not stack:
                continue
            arr = np.vstack(stack)
            mean = np.nanmean(arr, axis=0)
            sem = np.nanstd(arr, axis=0, ddof=1) / np.sqrt(max(arr.shape[0], 1))
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
        importance_acc: Dict[str, List[float]] = {}
        importance_labels: Dict[str, str] = {}
        for r in results.values():
            for row in r.feature_importance or []:
                feat = str(row.get("feature", ""))
                if not feat:
                    continue
                val = float(row.get("delta_r2", float("nan")))
                if not np.isfinite(val):
                    continue
                importance_acc.setdefault(feat, []).append(val)
                importance_labels[feat] = str(row.get("label", feat) or feat)
        agg_rows: List[Dict[str, Any]] = []
        for feat, vals in importance_acc.items():
            arr = np.asarray(vals, float)
            agg_rows.append({
                "feature": feat,
                "label": importance_labels.get(feat, feat),
                "delta_r2": float(np.nanmean(arr)),
                "delta_r2_sem": float(np.nanstd(arr, ddof=1) / np.sqrt(max(arr.size, 1))) if arr.size > 1 else 0.0,
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
            f"Kernels show mean +/- SEM across animals."
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
        stats_text = f"Pearson r = {r_value:.3f}, {p_text}, n = {n_corr}"
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
            if row.get("significant", False) and np.isfinite(p_value):
                p_txt = pg.TextItem(f"p={p_value:.3g}", color="#f5c542", anchor=(0.0, 0.5))
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
        if not self._flmm.available:
            self.statusMessage.emit(
                "R + fastFMM not available. Please install R, rpy2, and the fastFMM R package.", 8000
            )
            return

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
        if requested_formula != formula:
            self.edit_formula.setText(formula)
        random_eff = self.edit_random.text().strip() or "~1"
        nknots = self.spin_nknots.value() if self.spin_nknots.value() > 0 else None
        num_boots = self.spin_boots.value()

        self.statusMessage.emit("Fitting FLMM via fastFMM - this may take a while...", 0)
        self._progress_start("Fitting FLMM via fastFMM", 0)
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
        elif importance_mode == "fast":
            importance_rows = self._compute_flmm_coefficient_importance(result, formula_terms, term_labels)
        else:
            importance_rows = []
        result.feature_importance = importance_rows
        importance_value_key = (
            "delta_aic"
            if importance_mode == "loo" and any(np.isfinite(float(row.get("delta_aic", float("nan")))) for row in importance_rows)
            else "mean_abs_coefficient"
        )

        effective_group_var = str(result.stats.get("group_var", group_var)) if result.stats else group_var
        effective_formula = str(result.stats.get("formula", formula)) if result.stats else formula
        fallback_grouping = str(result.stats.get("fallback_grouping", "")) if result.stats else ""
        summary_lines = [
            result.summary_text,
            "",
            f"Scope: {'animal/group rows' if scope == 'animal' else 'trial rows'}",
            f"Rows: {n_rows}",
            f"ID variable: {effective_group_var}",
            f"Formula: {effective_formula}",
        ]
        if missing_terms:
            summary_lines.append(f"Auto formula used because saved terms were unavailable: {', '.join(missing_terms)}")
        if fallback_grouping:
            summary_lines.append(f"Grouping note: {fallback_grouping}")
        if result.stats:
            summary_lines.extend([
                "",
                "Fit statistics:",
                f"  AIC = {result.stats.get('aic', float('nan')):.5g}",
                f"  mean abs coefficient = {result.stats.get('mean_abs_coefficient', float('nan')):.5g}",
                f"  peak abs coefficient = {result.stats.get('peak_abs_coefficient', float('nan')):.5g}",
            ])
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
                summary_lines.append(
                    f"  {row['label']}: delta AIC = {delta_text}, "
                    f"reduced AIC = {reduced_text}, mean abs coef = {row['mean_abs_coefficient']:.5g}"
                )
            failed = [row for row in importance_rows if row.get("status") != "ok"]
            if failed:
                summary_lines.append(f"  {len(failed)} reduced FLMM fits failed; see log for details.")
            if num_boots > 0:
                summary_lines.append("  Leave-one-out comparison uses analytic fits; bootstrap CIs are not repeated.")
        elif importance_rows and importance_mode == "fast":
            summary_lines.extend([
                "",
                "Fast coefficient contribution (single FLMM fit):",
            ])
            for row in importance_rows[:10]:
                summary_lines.append(
                    f"  {row['label']}: mean abs coef = {row['mean_abs_coefficient']:.5g}, "
                    f"peak abs coef = {row['peak_abs_coefficient']:.5g}"
                )
            summary_lines.append("  Leave-one-out AIC is disabled by default because it refits fastFMM once per predictor.")
        elif importance_mode == "off":
            summary_lines.extend(["", "Feature contribution: off."])
        if fit_terms:
            readable = [f"{term} = {term_labels.get(term, term)}" for term in fit_terms]
            summary_lines.append("Covariates: " + "; ".join(readable))
        if dropped_terms:
            summary_lines.append("Dropped covariates: " + ", ".join(dropped_terms))
        self.txt_summary.setPlainText("\n".join(summary_lines))
        self._plot_flmm_coefficients(result)
        self._plot_feature_importance(
            importance_rows,
            value_key=importance_value_key,
            title="FLMM leave-one-feature-out contribution" if importance_mode == "loo" else "FLMM coefficient contribution",
            y_label="Delta AIC" if importance_value_key == "delta_aic" else "Mean abs coefficient",
        )
        if hasattr(self, "tabs_workspace") and importance_rows:
            self.tabs_workspace.setCurrentWidget(self.plot_importance.parentWidget())
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

    def _plot_flmm_coefficients(self, result: FLMMResult):
        pw = self.plot_coeff
        pw.clear()
        try:
            pw.getPlotItem().legend.clear()
        except Exception:
            pass

        colors = ["#4b9df8", "#f5a97f", "#6bdb74", "#ee99a0", "#c6a0f6",
                  "#f5e0dc", "#89dceb", "#fab387"]
        tvec = result.tvec

        for i, (name, coeff) in enumerate(result.coefficients.items()):
            color = colors[i % len(colors)]
            pen = pg.mkPen(color, width=2)
            pw.plot(tvec, coeff, pen=pen, name=name)

            # Joint CI as filled region
            if name in result.joint_ci_lower and name in result.joint_ci_upper:
                ci_lo = result.joint_ci_lower[name]
                ci_hi = result.joint_ci_upper[name]
                fill_color = QtGui.QColor(color)
                fill_color.setAlpha(40)
                fill = pg.FillBetweenItem(
                    pg.PlotDataItem(tvec, ci_lo),
                    pg.PlotDataItem(tvec, ci_hi),
                    brush=fill_color,
                )
                pw.addItem(fill)

            # Pointwise CI as dashed lines
            if name in result.ci_lower and name in result.ci_upper:
                dash_pen = pg.mkPen(color, width=1, style=QtCore.Qt.PenStyle.DashLine)
                pw.plot(tvec, result.ci_lower[name], pen=dash_pen)
                pw.plot(tvec, result.ci_upper[name], pen=dash_pen)

        pw.setLabel("bottom", "Time", units="s")
        pw.setLabel("left", "Coefficient")
        pw.addLine(y=0, pen=pg.mkPen("#5a6274", width=1, style=QtCore.Qt.PenStyle.DashLine))
        pw.addLine(x=0, pen=pg.mkPen("#5a6274", width=1, style=QtCore.Qt.PenStyle.DashLine))
