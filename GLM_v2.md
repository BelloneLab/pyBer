# GLM / FLMM v2 — statistical-soundness overhaul

Tracking the fixes that take the pyBer temporal-modeling pipeline from
"looks rigorous" to actually rigorous. Each item lists the file & symbol
touched, the change required, and a test that proves the fix.

Conventions used below:
- `tm.*` = `pyBer/temporal_modeling.py`
- Severity tags follow the audits: **CRIT**, **HIGH**, **MED**, **MIN**.

## Phase 1 — Foundational correctness

### F1. Non-circular kernel shifting (CRIT)
`tm.build_design_matrix` (line ~335) uses `np.roll(conv, -pre_samp)` then
zeros the trailing `pre_samp` samples. `np.roll` is **circular**, so the
last `pre_samp` samples of the recording leak back to the start before
the zero-out. Same pattern in the bootstrap fast path. Replace with
non-circular slicing:
```python
conv = np.convolve(input_vec, kernel, mode='full')[pre_samp : pre_samp + T]
```
Test: feed a single impulse at the very end of the trace; the leading
samples of the resulting design column must be exactly zero.

### F2. Kernel time axis off-by-one (HIGH)
`kernel_len = max(2, pre_samp + post_samp)` plus
`np.linspace(kernel_window[0], kernel_window[1], kernel_len)` yields
`kernel_len` samples spanning `[pre, post]` with step
`(post - pre) / (kernel_len - 1)`. For an inclusive window the step
should equal the recording dt. Use `kernel_len = pre_samp + post_samp + 1`
and verify `np.diff(kernel_tvec)[0] ≈ dt`.

### F3. Basis layout / time-axis refactor (MIN)
Pull `basis_at_time(t, kernel_window, n_basis, basis_type)` out so the
basis matrix is defined directly on the kernel time vector rather than
on `[0, 1]` followed by a roll. Drops one layer of entanglement.

## Phase 2 — Significance reporting

### F4. Bootstrap p-value reporting (MED)
Currently `(1 + sum(null >= obs)) / (n_boot + 1)`. When no null exceeds
the observation report `p <= 1/(n_boot+1)` instead of the misleading
`p = 0.0099`-style number. Warn when `n_boot < 200`.

### F5. Multiple-comparison correction across predictors (CRIT)
Add Benjamini-Hochberg FDR to the `feature_importance` rows produced by
`_compute_glm_shift_bootstrap_significance`. Add `q_value` field; flag
significant only when `q_value < 0.05`. Update the summary text and the
Importance plot so `significant` reflects the q-value.

### F6. Length sanity for circular-shift bootstrap (MIN)
For recordings shorter than `10 × kernel_length` the cyclic boundary
contributes to many shifts. Emit a warning when this holds (or, in
extreme cases, skip bootstrap entirely with an explanation).

## Phase 3 — Regularization

### F7. Scale-aware ridge / OLS / lasso (CRIT)
Standardize every non-intercept design column to unit L2 norm before
solving, then transform coefficients back. Event-impulse columns and
z-scored continuous-predictor columns then receive comparable
regularization for a given `alpha`. The user-set `alpha` stays as the
overall strength knob.

## Phase 4 — Uncertainty

### F8. Kernel uncertainty bands (CRIT)
Block bootstrap (resample non-overlapping segments of width >= kernel
length) to build a per-predictor null distribution of kernel curves.
Compute 95% pointwise CIs and overlay them on the Kernels tab. Persist
them in `GLMResult.kernel_ci_lower / kernel_ci_upper`.

### F9. Cross-validated R² (HIGH)
Block-CV: split by file (or by time-block when only one file is loaded).
Refit on K-1 folds, predict on the held-out fold, compute out-of-sample
R². Report next to in-sample R² in the summary.

## Phase 5 — Group analysis

### F10. Precision-weighted group aggregation (MED)
Replace the equal-weight mean kernels with an inverse-variance weighted
average. Use the per-animal kernel-bootstrap SE from F8 as weight
denominators. Same for importance ΔR² aggregation.

### F11. Multicollinearity diagnostic (MED)
Compute and surface:
- design-matrix condition number;
- pairwise correlations between predictor inputs; flag |r| > 0.8.
Add a `Diagnostics` section to the Summary tab.

## Phase 6 — FLMM rigor

### F12. Refuse synthetic FLMM grouping (HIGH)
Remove the `pyber_block` synthetic-grouping fallback. If the random-
effect variable has < 2 levels or = n_rows, abort the fit with a clear
message: "FLMM needs >= 2 subjects with repeated trials. With 1 row per
animal use the GLM tab; with one animal use trial-level rows."

### F13. Hide FLMM CIs when variance inference fails (HIGH)
The retry path sets `var=False`, then later falls back to
`beta_hat.copy()` for missing CI fields. That paints "zero-width" CIs
which read as "very precise" in the plot. When variance inference
failed, drop the CI dictionaries entirely and label the curves
"point estimates; variance inference unavailable".

## Phase 7 — UI / labeling

### F14. GLM vs FLMM tooltips (MED)
Tooltips on the model-type combo and on the Summary tab making explicit
that:
- GLM models how a predictor's *time course* shapes the signal;
- FLMM models how a per-trial *scalar covariate* modulates the average
  aligned response.

### F15. Illustration Pearson label (MIN)
The Pearson r / p shown on the Illustration tab treats autocorrelated
samples as independent. Append "(descriptive — autocorrelated samples)"
to the readout so it can't be mistaken for an inferential statistic.

## Smoke-test strategy

After each fix the smoke test must:
1. Build a synthetic clean photometry trace with one known event-response.
2. Run the GLM fit and verify the recovered kernel matches the injected
   one within a tolerance.
3. Run the bootstrap and check that the p-value of the true predictor is
   < 0.05 and that a sham predictor's q-value > 0.5.
4. For multi-file scenarios, run per-file batch and check the group
   summary uses precision weighting.
