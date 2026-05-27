# GLM v2 — post-implementation audit

Fresh read of `pyBer/temporal_modeling.py` against the GLM_v2.md task list.
Each item is marked **DONE**, **PARTIAL**, **NOT DONE**, or **NEW** (new issue
introduced by the fixes). Severity tags follow the original audit.

---

## Phase 1 — foundational correctness

### F1. Non-circular kernel shifting — **DONE**

`build_design_matrix` (line 394) now routes through `_glm_convolved_columns`
(line 216):

```python
conv = np.convolve(v, basis[:, b], mode="full")
cols[:, b] = conv[pre_samp : pre_samp + n_time]
```

No `np.roll` anywhere in the design-matrix path. The only remaining `np.roll`
is in `_shift_vector_by_segment` (line 4228) where it's a per-segment
**circular shift of the predictor itself** for the null-hypothesis bootstrap —
that's the textbook circular-shift null and is correct.

**Verified clean.**

### F2. Kernel time axis off-by-one — **DONE**

`_glm_kernel_geometry` (line 198):

```python
kernel_tvec = np.arange(-pre_samp, post_samp + 1, dtype=float) * dt
```

Inclusive of both endpoints; `np.diff(kernel_tvec)[0]` equals `dt` exactly.
`build_design_matrix` uses this as the single source of truth.

### F3. Basis / time-axis refactor — **DONE**

`_glm_basis_matrix` is a free function that maps `(n_basis, kernel_len,
basis_type)` to a basis matrix; `_glm_convolved_columns` does the time-shift
slicing. Both `build_design_matrix` and the bootstrap fast path consume the
same helpers, so a future edit can't desynchronize them. 

---

## Phase 2 — significance reporting

### F4. Bootstrap p-value reporting — **DONE**

`_compute_glm_shift_bootstrap_significance` (line 4232) computes `exceed =
sum(null >= obs)` and stores `row["p_value_upper_bound"] = bool(exceed == 0)`
alongside the raw `p_value`. Downstream rendering can show
"p ≤ 1/(N+1)" when this flag is true. **Verified.**

Bootstrap-count warning is also wired: when `n_boot < 200` the function sets
`row["bootstrap_warning"] = "Use >=200 circular-shift bootstraps for stable
q-values."` (line 4281).

### F5. Multiple-comparison correction — **DONE**

`_bh_fdr` (line 228) implements Benjamini-Hochberg with NaN-preserving order.
Called on every batch of bootstrap p-values (line 4417). The `significant`
field is set from `q_value < 0.05`, not the raw `p_value` (line 4420).

The FLMM summary also displays both `p (raw)` and `q (FDR)` columns
side-by-side with `SIG / n.s. / UNTESTED` badges — see HTML summary at
`_build_flmm_summary_html`. **Done end-to-end.**

### F6. Length sanity for circular-shift — **DONE**

Line 4274-4278 emits `length_warning` when any segment is shorter than
`10 × kernel_len`. Persisted on each importance row as
`bootstrap_warning`. The warning surfaces in exports + the summary table.

---

## Phase 3 — regularization

### F7. Scale-aware ridge / lasso / OLS — **DONE**

`ContinuousGLM._scaled_design` (line 407) normalizes each non-intercept
column to unit L2; `fit_coefficients` (line 418) fits on the scaled design
then divides coefficients by the per-column scale so beta values stay on
the original scale. Used in the main fit and in the kernel-CI bootstrap.

User-set α now acts as a *uniform* regularization strength rather than
arbitrarily favoring high-magnitude basis columns. **Verified.**

---

## Phase 4 — uncertainty

### F8. Kernel uncertainty bands — **DONE**

`_compute_glm_kernel_bootstrap_ci` (line 4655) implements residual
block-bootstrap CIs:

1. Block length = full kernel length (the timescale of autocorrelation).
2. Refit on `y_pred + bootstrap_residuals` per bootstrap iteration.
3. `ci_lower = np.nanpercentile(..., 2.5)`, `ci_upper = np.nanpercentile(...,
   97.5)`; both stored on `GLMResult.kernel_ci_lower / _upper`.

Rendered in both the overlay (`plot_glm_kernels`, line 3309) and the
small-panels layout. Per-file caps at 200 iterations for tractable compute.
**Done.**

### F9. Cross-validated R² — **DONE**

`_compute_glm_cross_validated_r2` (line 4500) does file-block CV (one fold
per file when multiple files are loaded; contiguous time-blocks otherwise).
Returns `{"cv_r2", "cv_folds", "cv_note"}`. The full-fit `result.stats`
exposes `cv_r2` alongside in-sample `r2`. Note rendered in the summary as
"CV unavailable: …" when too few blocks/samples.

The legacy in-sample R² is preserved (still shown), which matches the
original audit's concession that the in-sample number is useful for
exploration. **Done.**

---

## Phase 5 — group analysis

### F10. Precision-weighted group aggregation — **DONE**

`_aggregate_group_results` (line 6176):

- Per-file kernel CIs are read from each `GLMResult.kernel_ci_lower / _upper`
  to derive an approximate per-file SE = `|hi - lo| / (2 × 1.96)` (line 6236).
- When all per-file SEs are available, the group mean is **inverse-variance
  weighted** with weights `1 / max(SE², ε)` (line 6249); the group SE is
  `1 / sqrt(sum_weights)` (line 6253).
- When no per-file SEs exist, the code falls back to an unweighted mean +
  classical SEM (line 6258).

Caveat: a **bad recording with very loose CIs gets little weight**, but the
weighting only fires when **all** per-file fits had CIs (the code does
`if len(se_stack) == arr.shape[0]`). Mixed cohorts (some fits with CIs,
some without) fall through to the unweighted path. That's a defensible
default — partial precision-weighting would bias toward the CI-bearing
animals.

**Not yet done (NEW):** one-sample / sign / permutation tests against zero at
the group level. The Group tab still renders mean ± SEM without per-time-bin
significance. This was item 6 in the user's recommended priority list and is
*not* claimed by GLM_v2.md, but it's the obvious next step.

### F11. Multicollinearity diagnostic — **DONE**

`diagnostics` block at line 4575-4622:

1. **Condition number** of the standardized design matrix (`np.linalg.cond`).
2. **Pairwise predictor correlations**: any pair with `|r| >= 0.8` is added
   to `diagnostics["high_predictor_correlations"]` and flagged in the
   summary.

These appear in the GLM Summary tab. **Done.**

---

## Phase 6 — FLMM rigor

### F12. Refuse synthetic FLMM grouping — **PARTIAL / FIXED-EFFECT FALLBACK**

The audit recommended *refusing* to fit. The actual change instead
**replaces the synthetic-block fastFMM call with a purely functional
fixed-effect fit** (`_fit_fixed_effect_functional`, called at line 841 and
above at 6886). The fit happens in pure Python, no R/fastFMM involved when
grouping is invalid.

The result:
- `result.stats["backend"] = "python_functional_fixed_effect"`,
- `fallback_grouping = "<reason>"` carried through,
- Summary HTML shows a yellow **`SINGLE-FILE FALLBACK`** badge with the
  message: *"Functional fixed-effect fallback used (no repeated-subject
  grouping). Interpret as within-recording associations, not population
  mixed-effects inference."*

This is **better than refusing** — the user still gets a defensible model
(linear functional regression with pointwise CIs) and is loudly told that
random-effects inference was not performed. The original `pyber_block`
synthetic grouping is gone. **Acceptable resolution.**

### F13. Hide FLMM CIs when variance inference fails — **DONE**

Line 983: CI bands are computed **only if `not variance_unavailable`**.
When fastFMM's variance retry sets `variance_unavailable = True`, the CI
dicts stay empty. The plot path checks `if name in result.ci_lower` before
drawing dashed lines and joint bands (lines 6824, 6837); when the dicts are
empty, an orange *"Point estimates only; variance inference unavailable"*
note is drawn instead of fabricated zero-width bands. Summary HTML shows a
yellow **`NO CI`** badge.

The previous `beta_hat.copy()` fallback (the bug in the audit) is gone.
**Done.**

---

## Phase 7 — UI / labeling

### F14. GLM vs FLMM tooltips — **DONE**

Lines 2378-2387 set tooltips on the Summary and FLMM workspace tabs:

> *"GLM: how predictor time courses shape one continuous signal. FLMM: how
> trial-level scalar covariates modulate aligned response curves."*

> *"FLMM requires repeated trials/subjects; use Continuous GLM for one
> animal or one continuous trace."*

Also explicit in the FLMM-tab hint line above the coefficient plot.

### F15. Pearson p-value descriptive label — **NOT DONE**

The Illustration tab still computes and displays Pearson r/p without an
"autocorrelated samples → descriptive only" caveat. The user can still be
misled into treating the displayed p-value as inferential.

Severity: **MIN**. Single line of text to fix.

---

## New issues introduced by v2

### N1. Kernel CI bootstrap is hard-capped at 200 iterations — **MIN**

Line 4670: `n_boot = min(int(n_boot), 200)`. This cap is sensible for
interactive use (200 iter × ~25 ms per fit ≈ 5 s) but is **silent** —
a user who explicitly requests `n_boot = 1000` for tighter CIs gets 200
without warning. The kernel-CI percentile precision at N=200 is
`±sqrt(0.025·0.975/200) ≈ ±1%`, which is fine for the 95% percentile but
not advertised.

Fix: surface this in the status message ("kernel-CI bootstrap capped at
200; for tighter bands set n_boot in advanced and rerun"), or remove the
cap and let the user pay the time cost.

### N2. Group aggregation precision weighting is all-or-nothing — **MIN**

`_aggregate_group_results` uses precision weighting only when every
per-file fit has CIs (`len(se_stack) == arr.shape[0]`). When even one
animal's CI bootstrap failed (e.g. too few samples), the code reverts to
the **unweighted** path for the whole group — discarding the precision
info from animals that DID produce CIs.

Fix: weight the animals that have CIs by `1/SE²` and weight CI-less
animals by the median weight across the others. Or just drop CI-less
animals from the precision-weighted mean and report them separately.

### N3. Cross-validated R² block size when no segment_slices — **MIN**

For single-file GLM fits, the CV path creates contiguous time blocks. The
number of blocks is decided internally (need to confirm). If too few
blocks are created, CV reports "need at least two blocks" and bails. The
user has no UI control over the CV fold count.

Severity: **MIN**. A spinbox or default-3-folds would suffice.

### N4. FLMM grouping fallback is generous — **MED**

When `unique_groups.size > 1` but `< n_trials` is False (i.e. one row per
animal), the code still falls back to functional fixed-effect even though
a *true* group-level inference is possible by aggregating per-animal fits
into a one-sample test. The current code calls this a "fixed-effect
fallback" but it's actually closer to fitting a regression on animal
means.

Fix: when the user has one row per animal AND the row count is small,
prefer reporting it as a *single-animal-level fixed-effect functional
regression* (which is what it is) rather than calling it a "fallback" —
the wording suggests something failed when nothing did.

---

## Untouched items from the prior audits

- **Continuous predictors interpreted as temporal kernels.** Still the same
  treatment; the Kernels-tab hint line spells out that continuous predictors
  show "signal gain per +1 SD" rather than impulse responses. Documentation
  resolved, no algorithmic change needed.
- **One-sample / permutation tests at group level.** Not implemented; the
  Group tab is descriptive. This was deferred from F10.
- **Pearson autocorrelation caveat on Illustration tab.** F15, still open.

---

## Summary

| Item | Severity | Status |
|---|---|---|
| F1 non-circular shift | CRIT | DONE |
| F2 kernel axis off-by-one | HIGH | DONE |
| F3 basis refactor | MIN | DONE |
| F4 p-upper-bound + N<200 warn | MED | DONE |
| F5 BH-FDR + q-values | CRIT | DONE |
| F6 length sanity | MIN | DONE |
| F7 scale-aware ridge | CRIT | DONE |
| F8 kernel CIs | CRIT | DONE |
| F9 cross-validated R² | HIGH | DONE |
| F10 precision-weighted group | MED | DONE (with N2 caveat) |
| F11 multicollinearity diagnostic | MED | DONE |
| F12 refuse / fallback FLMM grouping | HIGH | PARTIAL (clean fallback) |
| F13 hide FLMM CIs when var failed | HIGH | DONE |
| F14 GLM-vs-FLMM tooltips | MED | DONE |
| F15 Pearson autocorr caveat | MIN | NOT DONE |
| N1 silent kernel-CI cap | MIN | NEW |
| N2 all-or-nothing precision weighting | MIN | NEW |
| N3 CV fold count not user-controllable | MIN | NEW |
| N4 fallback wording | MED | NEW (cosmetic) |

**13 / 15 of the original task list are clean, 1 partial-but-acceptable, 1
trivial item left open. Four small new issues to track.**
