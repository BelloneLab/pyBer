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

import logging
import os
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
    # On Windows, R is often not on PATH.  Try the standard install location.
    r_home = os.environ.get("R_HOME", "")
    if not r_home:
        candidate = "C:/Program Files/R"
        if os.path.isdir(candidate):
            subs = sorted(os.listdir(candidate), reverse=True)
            if subs:
                r_home = os.path.join(candidate, subs[0])
                os.environ["R_HOME"] = r_home
    if r_home:
        bin_x64 = os.path.join(r_home, "bin", "x64")
        if os.path.isdir(bin_x64):
            try:
                os.add_dll_directory(bin_x64)
            except (OSError, AttributeError):
                pass
        # Ensure PATH includes R so child processes find R.dll
        cur_path = os.environ.get("PATH", "")
        if bin_x64 not in cur_path:
            os.environ["PATH"] = bin_x64 + os.pathsep + cur_path

    r_libs_user = os.environ.get("R_LIBS_USER", "")
    if not r_libs_user:
        candidate_lib = os.path.expanduser("~/R/win-library")
        if os.path.isdir(candidate_lib):
            subs = sorted(os.listdir(candidate_lib), reverse=True)
            if subs:
                os.environ["R_LIBS_USER"] = os.path.join(candidate_lib, subs[0])

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
    y_pred: np.ndarray                    # predicted trace
    y_actual: np.ndarray                  # actual trace
    residuals: np.ndarray
    r2: float
    coefficients: np.ndarray              # raw beta vector
    design_matrix: np.ndarray


class ContinuousGLM:
    """Build a design matrix from event times and fit a linear model."""

    BASIS_TYPES = ("raised_cosine", "bspline", "fir")
    REGULARIZATION = ("ridge", "lasso", "ols")

    def __init__(self):
        self._result: Optional[GLMResult] = None

    @staticmethod
    def build_design_matrix(
        time: np.ndarray,
        predictors: Dict[str, np.ndarray],
        kernel_window: Tuple[float, float],
        n_basis: int = 8,
        basis_type: str = "raised_cosine",
    ) -> Tuple[np.ndarray, List[str], int]:
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
        dt = np.median(np.diff(time))
        pre_samp = int(round(abs(kernel_window[0]) / dt))
        post_samp = int(round(abs(kernel_window[1]) / dt))
        kernel_len = pre_samp + post_samp

        if basis_type == "bspline":
            B = _bspline_basis(n_basis, kernel_len)
        elif basis_type == "fir":
            B = _fir_basis(n_basis, kernel_len)
        else:
            B = _raised_cosine_basis(n_basis, kernel_len)

        T = len(time)
        col_names: List[str] = []
        X_parts: List[np.ndarray] = []

        for pred_name, ev_times in predictors.items():
            ev_times = np.asarray(ev_times, float)
            ev_times = ev_times[np.isfinite(ev_times)]

            # Convert event times to sample indices
            ev_idx = np.searchsorted(time, ev_times)
            ev_idx = ev_idx[(ev_idx >= 0) & (ev_idx < T)]

            # Build impulse vector
            impulse = np.zeros(T, float)
            for idx in ev_idx:
                impulse[idx] = 1.0

            # Convolve impulse with each basis function
            part = np.zeros((T, n_basis), float)
            for b in range(n_basis):
                # Pad the basis to align with pre_samp offset
                kernel = np.zeros(kernel_len)
                kernel[:] = B[:, b]
                conv = np.convolve(impulse, kernel, mode="full")[:T]
                # Shift so that the kernel starts at -pre_samp
                if pre_samp > 0:
                    part[:, b] = np.roll(conv, -pre_samp)
                    part[-pre_samp:, b] = 0.0
                else:
                    part[:, b] = conv

            X_parts.append(part)
            for b in range(n_basis):
                col_names.append(f"{pred_name}_b{b}")

        X = np.hstack(X_parts) if X_parts else np.zeros((T, 0))
        # Add intercept
        X = np.column_stack([np.ones(T), X])
        col_names.insert(0, "intercept")
        return X, col_names, n_basis

    def fit(
        self,
        time: np.ndarray,
        signal: np.ndarray,
        predictors: Dict[str, np.ndarray],
        kernel_window: Tuple[float, float] = (-1.0, 3.0),
        n_basis: int = 8,
        basis_type: str = "raised_cosine",
        regularization: str = "ridge",
        alpha: float = 1.0,
    ) -> GLMResult:
        """Fit the GLM and return the result."""
        time = np.asarray(time, float)
        signal = np.asarray(signal, float)

        X, col_names, n_b = self.build_design_matrix(
            time, predictors, kernel_window, n_basis, basis_type,
        )

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

        # Extract kernels
        dt = np.median(np.diff(time))
        pre_samp = int(round(abs(kernel_window[0]) / dt))
        post_samp = int(round(abs(kernel_window[1]) / dt))
        kernel_len = pre_samp + post_samp

        if basis_type == "bspline":
            B = _bspline_basis(n_b, kernel_len)
        elif basis_type == "fir":
            B = _fir_basis(n_b, kernel_len)
        else:
            B = _raised_cosine_basis(n_b, kernel_len)

        kernel_tvec = np.linspace(kernel_window[0], kernel_window[1], kernel_len)
        kernels: Dict[str, np.ndarray] = {}
        idx = 1  # skip intercept
        for pred_name in predictors:
            w = beta[idx:idx + n_b]
            kernels[pred_name] = B @ w
            idx += n_b

        self._result = GLMResult(
            predictor_names=list(predictors.keys()),
            kernels=kernels,
            kernel_tvec=kernel_tvec,
            y_pred=y_pred,
            y_actual=signal,
            residuals=residuals,
            r2=r2,
            coefficients=beta,
            design_matrix=X,
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


class TrialFLMM:
    """
    Functional Linear Mixed Model using R's fastFMM package.

    Wraps fastFMM::fui() via rpy2.  The user provides a trial-level data
    matrix (n_trials x n_timepoints) plus a design dataframe (n_trials rows)
    with fixed/random predictors.  The backend constructs the long-form
    data and calls fui().
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
        formula_fixed: str = "Y.obs ~ group",
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
        from rpy2.robjects import r as R, pandas2ri, numpy2ri
        from rpy2.robjects.packages import importr

        n_trials, n_time = mat.shape

        # Build long-form dataframe in R
        # Columns: Y.obs, .index (timepoint), .obs (trial id), + design vars
        Y_long = mat.ravel(order="C")  # trial-major
        index_long = np.tile(np.arange(n_time), n_trials)
        obs_long = np.repeat(np.arange(n_trials), n_time)

        r_df_vars = {
            "Y.obs": ro.FloatVector(Y_long),
            ".index": ro.IntVector(index_long.astype(int)),
            ".obs": ro.IntVector(obs_long.astype(int)),
        }

        for col_name, col_vals in design.items():
            col_vals = np.asarray(col_vals)
            repeated = np.repeat(col_vals, n_time)
            if np.issubdtype(col_vals.dtype, np.floating):
                r_df_vars[col_name] = ro.FloatVector(repeated)
            elif np.issubdtype(col_vals.dtype, np.integer):
                r_df_vars[col_name] = ro.IntVector(repeated.astype(int))
            else:
                r_df_vars[col_name] = ro.StrVector(repeated.astype(str))

        r_df = ro.DataFrame(r_df_vars)

        # Call fui()
        fastFMM = importr("fastFMM")
        kwargs = {
            "formula": ro.Formula(formula_fixed),
            "data": r_df,
            "id": ro.StrVector([group_var]),
            "G": ro.Formula(random_effects),
            "parallel": ro.BoolVector([parallel]),
        }
        if nknots_min is not None:
            kwargs["nknots_min"] = ro.IntVector([nknots_min])
        if num_boots > 0:
            kwargs["num_boots"] = ro.IntVector([num_boots])

        _LOG.info("Calling fastFMM::fui() with formula=%s, %d trials, %d timepoints",
                  formula_fixed, n_trials, n_time)

        fui_result = fastFMM.fui(**kwargs)

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
            beta_hat = np.array(R('as.matrix')(fui_result.rx2("betaHat")))
            beta_lb = np.array(R('as.matrix')(fui_result.rx2("betaHat.LB")))
            beta_ub = np.array(R('as.matrix')(fui_result.rx2("betaHat.UB")))

            # Term names from rownames
            try:
                term_names = list(R('rownames')(fui_result.rx2("betaHat")))
            except Exception:
                term_names = [f"term_{i}" for i in range(beta_hat.shape[0])]

            for i, name in enumerate(term_names):
                coefficients[name] = beta_hat[i, :]
                ci_lower[name] = beta_lb[i, :]
                ci_upper[name] = beta_ub[i, :]

            # Joint CIs (may not always be present)
            try:
                jlb = np.array(R('as.matrix')(fui_result.rx2("betaHat.LB.joint")))
                jub = np.array(R('as.matrix')(fui_result.rx2("betaHat.UB.joint")))
                for i, name in enumerate(term_names):
                    joint_ci_lower[name] = jlb[i, :]
                    joint_ci_upper[name] = jub[i, :]
            except Exception:
                joint_ci_lower = {k: v.copy() for k, v in ci_lower.items()}
                joint_ci_upper = {k: v.copy() for k, v in ci_upper.items()}

            aic_val = None
            try:
                aic_val = float(np.array(fui_result.rx2("AIC"))[0])
            except Exception:
                pass

            summary_parts = [f"FLMM fit: {len(term_names)} terms, {n_trials} trials, {n_time} timepoints"]
            if aic_val is not None:
                summary_parts.append(f"AIC = {aic_val:.1f}")
            for name in term_names:
                summary_parts.append(f"  {name}: mean coef = {np.nanmean(coefficients[name]):.4f}")
            summary_text = "\n".join(summary_parts)

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


class TemporalModelingWidget(QtWidgets.QWidget):
    """
    PySide6 panel for Temporal Modeling (GLM / FLMM).
    Embeddable in the PostProcessingPanel dock system.
    """

    statusMessage = QtCore.Signal(str, int)

    def __init__(self, parent: Optional[QtWidgets.QWidget] = None):
        super().__init__(parent)
        self._glm = ContinuousGLM()
        self._flmm = TrialFLMM()
        self._glm_result: Optional[GLMResult] = None
        self._flmm_result: Optional[FLMMResult] = None

        # Data references (set by host panel)
        self._processed_trials = []
        self._psth_mat: Optional[np.ndarray] = None
        self._psth_tvec: Optional[np.ndarray] = None
        self._event_times: Optional[np.ndarray] = None
        self._file_ids: List[str] = []

        self._build_compact_ui()
        self._connect_signals()

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

        root.addWidget(self.grp_glm)

        # ---- FLMM settings ----
        self.grp_flmm = QtWidgets.QGroupBox("FLMM settings")
        self.grp_flmm.setStyleSheet(_SECTION_QSS)
        fl = QtWidgets.QFormLayout(self.grp_flmm)
        fl.setSpacing(4)

        self.edit_formula = QtWidgets.QLineEdit("Y.obs ~ group")
        self.edit_formula.setPlaceholderText("e.g. Y.obs ~ group + condition")
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

        root.addWidget(self.grp_flmm)

        # ---- Predictor builder ----
        self.grp_predictors = QtWidgets.QGroupBox("Predictors")
        self.grp_predictors.setStyleSheet(_SECTION_QSS)
        pl = QtWidgets.QVBoxLayout(self.grp_predictors)
        pl.setSpacing(4)

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
            "Predictors are auto-populated from DIO / behavior events when PSTH is computed."
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
        self.setStyleSheet(_TEMPORAL_QSS)

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
        self.combo_model_type.setMinimumWidth(230)
        h.addWidget(self.combo_model_type)

        self.btn_fit = QtWidgets.QPushButton("Fit model")
        self.btn_fit.setProperty("class", "primary")
        self.btn_fit.setMinimumWidth(120)
        h.addWidget(self.btn_fit)
        root.addWidget(header)

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
        self.btn_nav_fit = self._make_nav_button("Fit")
        for btn in (self.btn_nav_model, self.btn_nav_predictors, self.btn_nav_fit):
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
        self._build_fit_page()
        self._build_workspace_pages()

        self.btn_nav_model.setChecked(True)
        self.btn_nav_model.clicked.connect(lambda: self._select_control_page(0))
        self.btn_nav_predictors.clicked.connect(lambda: self._select_control_page(1))
        self.btn_nav_fit.clicked.connect(lambda: self._select_control_page(2))
        self.btn_fit_side.clicked.connect(self._on_fit_clicked)
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

        self.edit_formula = QtWidgets.QLineEdit("Y.obs ~ group")
        self.edit_formula.setPlaceholderText("e.g. Y.obs ~ group + condition")
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
            "Predictors are populated from DIO or behavior events when PSTH is computed."
        )
        self.lbl_predictor_hint.setProperty("class", "muted")
        self.lbl_predictor_hint.setWordWrap(True)
        pl.addWidget(self.lbl_predictor_hint)
        lay.addWidget(self.grp_predictors, 1)
        self.stack_controls.addWidget(page)

    def _build_fit_page(self):
        page = QtWidgets.QWidget()
        lay = QtWidgets.QVBoxLayout(page)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(8)

        grp = QtWidgets.QGroupBox("Fit Control")
        gl = QtWidgets.QVBoxLayout(grp)
        gl.setContentsMargins(12, 18, 12, 12)
        gl.setSpacing(10)
        self.lbl_data_status = QtWidgets.QLabel("No PSTH data has been pushed yet.")
        self.lbl_data_status.setProperty("class", "muted")
        self.lbl_data_status.setWordWrap(True)
        gl.addWidget(self.lbl_data_status)
        self.btn_fit_side = QtWidgets.QPushButton("Fit model")
        self.btn_fit_side.setProperty("class", "primary")
        gl.addWidget(self.btn_fit_side)
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
        self.plot_kernel = pg.PlotWidget(title="Estimated kernels")
        self._style_plot(self.plot_kernel)
        kernel_lay.addWidget(self.plot_kernel, 1)
        self.tabs_workspace.addTab(kernel_page, "Kernels")

        prediction_page = QtWidgets.QWidget()
        prediction_lay = QtWidgets.QVBoxLayout(prediction_page)
        prediction_lay.setContentsMargins(10, 10, 10, 10)
        self.plot_prediction = pg.PlotWidget(title="Actual vs predicted")
        self._style_plot(self.plot_prediction)
        prediction_lay.addWidget(self.plot_prediction, 1)
        self.tabs_workspace.addTab(prediction_page, "Prediction")

        residual_page = QtWidgets.QWidget()
        residual_lay = QtWidgets.QVBoxLayout(residual_page)
        residual_lay.setContentsMargins(10, 10, 10, 10)
        self.plot_residuals = pg.PlotWidget(title="Residuals")
        self._style_plot(self.plot_residuals)
        residual_lay.addWidget(self.plot_residuals, 1)
        self.tabs_workspace.addTab(residual_page, "Residuals")

        flmm_page = QtWidgets.QWidget()
        flmm_lay = QtWidgets.QVBoxLayout(flmm_page)
        flmm_lay.setContentsMargins(10, 10, 10, 10)
        self.plot_coeff = pg.PlotWidget(title="FLMM coefficient curves")
        self._style_plot(self.plot_coeff)
        flmm_lay.addWidget(self.plot_coeff, 1)
        self.tabs_workspace.addTab(flmm_page, "FLMM")

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
        buttons = (self.btn_nav_model, self.btn_nav_predictors, self.btn_nav_fit)
        for i, btn in enumerate(buttons):
            btn.setChecked(i == index)

    def _style_plot(self, plot: pg.PlotWidget) -> None:
        plot.setMinimumHeight(360)
        plot.setBackground("#05080d")
        plot.showGrid(x=True, y=True, alpha=0.22)
        plot.addLegend(offset=(12, 12))
        pi = plot.getPlotItem()
        pi.getAxis("bottom").setPen(pg.mkPen("#516179"))
        pi.getAxis("left").setPen(pg.mkPen("#516179"))
        pi.getAxis("bottom").setTextPen(pg.mkPen("#c5d2e3"))
        pi.getAxis("left").setTextPen(pg.mkPen("#c5d2e3"))
        pi.titleLabel.item.setDefaultTextColor(QtGui.QColor("#d7e0ee"))

    # ------------------------------------------------------------------
    # Signal wiring
    # ------------------------------------------------------------------

    def _connect_signals(self):
        self.combo_model_type.currentIndexChanged.connect(self._on_model_type_changed)
        self.btn_fit.clicked.connect(self._on_fit_clicked)
        self.btn_add_predictor.clicked.connect(self._on_add_predictor)
        self.btn_remove_predictor.clicked.connect(self._on_remove_predictor)

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
    ):
        """Push data from the host panel into this widget."""
        self._processed_trials = processed_trials or []
        self._psth_mat = psth_mat
        self._psth_tvec = psth_tvec
        self._event_times = event_times
        self._file_ids = file_ids or []
        self._per_file_mats = per_file_mats or {}

        # Auto-populate predictors for GLM
        if self.list_predictors.count() == 0 and event_times is not None and len(event_times):
            self.list_predictors.addItem("events")
        n_trials = len(self._processed_trials)
        psth_shape = tuple(np.shape(psth_mat)) if psth_mat is not None else None
        bits = [f"Processed recordings: {n_trials}"]
        if psth_shape:
            bits.append(f"PSTH matrix: {psth_shape[0]} x {psth_shape[1]}")
        if event_times is not None:
            bits.append(f"Events: {len(event_times)}")
        self.lbl_data_status.setText("\n".join(bits))

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
                self.lbl_flmm_status.setStyleSheet("color: #6bdb74;")
            else:
                self.lbl_flmm_status.setText(
                    "R or fastFMM not found. Install R and the fastFMM package, "
                    "then install rpy2 (pip install rpy2)."
                )
                self.lbl_flmm_status.setStyleSheet("color: #f5a97f;")

    def _on_add_predictor(self):
        name, ok = QtWidgets.QInputDialog.getText(
            self, "Add predictor", "Predictor name (must match a column in design):"
        )
        if ok and name.strip():
            self.list_predictors.addItem(name.strip())

    def _on_remove_predictor(self):
        sel = self.list_predictors.currentRow()
        if sel >= 0:
            self.list_predictors.takeItem(sel)

    def _on_fit_clicked(self):
        model_idx = self.combo_model_type.currentIndex()
        try:
            if model_idx == 0:
                self._fit_glm()
            else:
                self._fit_flmm()
        except Exception as exc:
            _LOG.error("Temporal modeling fit failed: %s\n%s", exc, traceback.format_exc())
            self.txt_summary.setPlainText(f"Error: {exc}")
            self.statusMessage.emit(f"Temporal model fit failed: {exc}", 8000)

    # ------------------------------------------------------------------
    # GLM fit
    # ------------------------------------------------------------------

    def _fit_glm(self):
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
        if hasattr(self, "tabs_workspace"):
            self.tabs_workspace.setCurrentWidget(self.plot_kernel.parentWidget())
        self.statusMessage.emit(f"GLM fit complete — R² = {result.r2:.4f}", 5000)

    def _plot_glm_kernels(self, result: GLMResult):
        pw = self.plot_kernel
        pw.clear()
        try:
            pw.getPlotItem().legend.clear()
        except Exception:
            pass
        colors = ["#4b9df8", "#f5a97f", "#6bdb74", "#ee99a0", "#c6a0f6",
                  "#f5e0dc", "#89dceb", "#fab387"]
        for i, (name, kernel) in enumerate(result.kernels.items()):
            color = colors[i % len(colors)]
            pw.plot(result.kernel_tvec, kernel, pen=pg.mkPen(color, width=2), name=name)
        pw.setLabel("bottom", "Time", units="s")
        pw.setLabel("left", "Kernel weight")
        # Zero line
        pw.addLine(y=0, pen=pg.mkPen("#5a6274", width=1, style=QtCore.Qt.PenStyle.DashLine))
        pw.addLine(x=0, pen=pg.mkPen("#5a6274", width=1, style=QtCore.Qt.PenStyle.DashLine))

    def _plot_glm_fit(self, result: GLMResult):
        pw = self.plot_prediction
        pw.clear()
        try:
            pw.getPlotItem().legend.clear()
        except Exception:
            pass
        x = np.arange(result.y_actual.size)
        pw.plot(x, result.y_actual, pen=pg.mkPen("#4b9df8", width=1.2), name="actual")
        pw.plot(x, result.y_pred, pen=pg.mkPen("#f5a97f", width=1.4), name="predicted")
        pw.setLabel("bottom", "Sample")
        pw.setLabel("left", "Signal")

        rw = self.plot_residuals
        rw.clear()
        rw.plot(x, result.residuals, pen=pg.mkPen("#ee99a0", width=1.1), name="residual")
        rw.addLine(y=0, pen=pg.mkPen("#5a6274", width=1, style=QtCore.Qt.PenStyle.DashLine))
        rw.setLabel("bottom", "Sample")
        rw.setLabel("left", "Residual")

    # ------------------------------------------------------------------
    # FLMM fit
    # ------------------------------------------------------------------

    def _fit_flmm(self):
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
