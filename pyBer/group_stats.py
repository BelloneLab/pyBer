"""Between-group statistics for PSTH metrics and global signal metrics.

The statistical unit is the animal / recording, never the trial: trials of one
animal are not independent, so trial-level rows are first averaged per file.

Test choice (documented in every result, so it can be reported verbatim):

* 2 groups: Welch's t-test when every group passes Shapiro-Wilk (p > alpha),
  otherwise Mann-Whitney U (two-sided). Effect size: Hedges' g (Welch) or the
  rank-biserial correlation (Mann-Whitney).
* 3+ groups: one-way ANOVA when every group is normal and Levene's test does
  not reject equal variances, otherwise Kruskal-Wallis. Effect size: eta
  squared (ANOVA) or epsilon squared (Kruskal-Wallis). Pairwise follow-ups use
  the matching two-group test with Holm correction.
* A group with fewer than 3 animals cannot be checked for normality and gives
  no reliable inference: the result is then descriptive only.

Recordings without a group ("Ungrouped") are summarized but never tested.
No value is modified, imputed or excluded except non-finite entries.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

MIN_TEST_N = 3


def describe(values: Sequence[float]) -> Dict[str, float]:
    """n, mean, SEM (ddof=1), median and SD of the finite values."""
    v = np.asarray(values, float)
    v = v[np.isfinite(v)]
    n = int(v.size)
    if n == 0:
        return {"n": 0, "mean": np.nan, "sem": np.nan, "median": np.nan, "sd": np.nan}
    sd = float(np.std(v, ddof=1)) if n > 1 else np.nan
    return {"n": n, "mean": float(np.mean(v)), "median": float(np.median(v)), "sd": sd,
            "sem": sd / np.sqrt(n) if n > 1 else np.nan}


def per_file_means(values: Sequence[float], file_ids: Sequence[str]) -> Tuple[List[str], np.ndarray]:
    """Average rows of the same file (trial rows -> one value per animal), in first-seen order."""
    order: List[str] = []
    sums: Dict[str, List[float]] = {}
    for value, fid in zip(np.asarray(values, float), file_ids):
        if fid not in sums:
            order.append(fid)
            sums[fid] = []
        if np.isfinite(value):
            sums[fid].append(float(value))
    return order, np.asarray([np.mean(sums[f]) if sums[f] else np.nan for f in order], float)


def _holm(pvalues: Sequence[float]) -> List[float]:
    """Holm step-down adjustment; non-finite p-values stay NaN."""
    p = np.asarray(pvalues, float)
    adjusted = np.full(p.shape, np.nan)
    finite = np.flatnonzero(np.isfinite(p))
    running = 0.0
    for rank, index in enumerate(finite[np.argsort(p[finite])]):
        running = min(1.0, max(running, p[index] * (finite.size - rank)))
        adjusted[index] = running
    return adjusted.tolist()


def _hedges_g(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = a.size, b.size
    pooled = np.sqrt(((na - 1) * np.var(a, ddof=1) + (nb - 1) * np.var(b, ddof=1)) / (na + nb - 2))
    if not np.isfinite(pooled) or pooled == 0:
        return np.nan
    correction = 1 - 3 / (4 * (na + nb) - 9)
    return float((np.mean(b) - np.mean(a)) / pooled * correction)


def _two_group(a: np.ndarray, b: np.ndarray, parametric: bool) -> Dict[str, object]:
    from scipy import stats
    if parametric:
        result = stats.ttest_ind(a, b, equal_var=False)
        return {"test": "Welch t-test", "statistic": float(result.statistic), "p": float(result.pvalue),
                "effect_name": "Hedges g (B - A)", "effect": _hedges_g(a, b)}
    result = stats.mannwhitneyu(a, b, alternative="two-sided")
    u = float(result.statistic)
    return {"test": "Mann-Whitney U", "statistic": u, "p": float(result.pvalue),
            "effect_name": "rank-biserial r", "effect": float(1 - 2 * u / (a.size * b.size))}


def compare_groups(groups: Sequence[Dict[str, object]], alpha: float = 0.05) -> Dict[str, object]:
    """Describe every group and test the grouped ones against each other.

    ``groups``: dicts with ``name``, ``color``, ``values`` (one per animal),
    optional ``ids`` (file ids) and ``tested`` (False for "Ungrouped").
    """
    from scipy import stats
    described = []
    for group in groups:
        values = np.asarray(group.get("values", []), float)
        described.append({"name": str(group["name"]), "color": str(group.get("color", "#888888")),
                          "values": values, "ids": list(group.get("ids", [])),
                          "tested": bool(group.get("tested", True)), **describe(values)})
    result: Dict[str, object] = {"groups": described, "test": "Descriptive only", "statistic": np.nan,
                                 "p": np.nan, "effect_name": "", "effect": np.nan,
                                 "normality": "", "variance": "", "pairwise": [], "note": ""}
    tested = [g for g in described if g["tested"] and g["n"] > 0]
    if len(tested) < 2:
        result["note"] = "Fewer than two groups with data: no between-group test."
        return result
    samples = [g["values"][np.isfinite(g["values"])] for g in tested]
    if min(s.size for s in samples) < MIN_TEST_N:
        result["note"] = (f"A group has fewer than {MIN_TEST_N} animals: normality cannot be "
                          "checked and no reliable test is possible. Descriptive only.")
        return result
    shapiro = [float(stats.shapiro(s).pvalue) if np.ptp(s) > 0 else 0.0 for s in samples]
    normal = all(p > alpha for p in shapiro)
    result["normality"] = "Shapiro-Wilk p: " + ", ".join(
        f"{g['name']}={p:.3g}" for g, p in zip(tested, shapiro))
    if len(tested) == 2:
        result.update(_two_group(samples[0], samples[1], normal))
        result["note"] = ("Normal in both groups: Welch t-test (no equal-variance assumption)."
                          if normal else "Normality rejected or untestable: Mann-Whitney U.")
        return result
    levene_p = float(stats.levene(*samples).pvalue)
    result["variance"] = f"Levene p={levene_p:.3g}"
    parametric = normal and levene_p > alpha
    if parametric:
        omnibus = stats.f_oneway(*samples)
        values = np.concatenate(samples)
        grand = np.mean(values)
        between = sum(s.size * (np.mean(s) - grand) ** 2 for s in samples)
        total = np.sum((values - grand) ** 2)
        result.update(test="One-way ANOVA", statistic=float(omnibus.statistic), p=float(omnibus.pvalue),
                      effect_name="eta squared", effect=float(between / total) if total > 0 else np.nan)
    else:
        omnibus = stats.kruskal(*samples)
        n_total = sum(s.size for s in samples)
        result.update(test="Kruskal-Wallis", statistic=float(omnibus.statistic), p=float(omnibus.pvalue),
                      effect_name="epsilon squared",
                      effect=float(omnibus.statistic / (n_total - 1)) if n_total > 1 else np.nan)
    pairs = []
    for i in range(len(tested)):
        for j in range(i + 1, len(tested)):
            pairs.append({"a": tested[i]["name"], "b": tested[j]["name"],
                          **_two_group(samples[i], samples[j], parametric)})
    for pair, adjusted in zip(pairs, _holm([pair["p"] for pair in pairs])):
        pair["p_holm"] = adjusted
    result["pairwise"] = pairs
    result["note"] = ("Normal groups with equal variances: ANOVA, pairwise Welch t-tests (Holm)."
                      if parametric else "Normality or equal variances rejected: Kruskal-Wallis, "
                      "pairwise Mann-Whitney U (Holm).")
    return result


def stats_label(comparison: Optional[Dict[str, object]]) -> str:
    """Short on-plot annotation, e.g. 'Welch t-test p=0.031 · g=1.2'."""
    if not comparison:
        return ""
    if not np.isfinite(float(comparison.get("p", np.nan))):
        return "Descriptive only"
    effect = float(comparison.get("effect", np.nan))
    short = {"Hedges g (B - A)": "g", "rank-biserial r": "r", "eta squared": "η²",
             "epsilon squared": "ε²"}.get(str(comparison.get("effect_name", "")), "")
    text = f"{comparison['test']} p={float(comparison['p']):.3g}"
    return text + (f" · {short}={effect:.2g}" if short and np.isfinite(effect) else "")


def comparison_rows(comparison: Dict[str, object], **fixed) -> List[Dict[str, object]]:
    """Flat CSV rows: one per group with the shared test result repeated."""
    rows = []
    for group in comparison.get("groups", []):
        rows.append({**fixed, "group": group["name"], "tested": group["tested"], "n": group["n"],
                     "mean": group["mean"], "sem": group["sem"], "median": group["median"],
                     "sd": group["sd"], "test": comparison["test"], "statistic": comparison["statistic"],
                     "p": comparison["p"], "effect_name": comparison["effect_name"],
                     "effect": comparison["effect"], "normality": comparison["normality"],
                     "variance": comparison["variance"], "note": comparison["note"]})
    return rows


def window_interaction(groups: Sequence[np.ndarray], alpha: float = 0.05) -> Dict[str, object]:
    """Group x window interaction of a mixed (split-plot) ANOVA.

    Question: do the groups follow different courses across the windows (Pre,
    Post, Late, ...)? ``groups``: one array per tested group, shape
    (animals, windows); an animal with a missing window is left out.

    Each animal's own mean is removed, then the remaining window profiles are
    compared between groups: F = MS(group x window) / MS(animal x window within
    group), with (G-1)(k-1) and (N-G)(k-1) degrees of freedom. With more than
    two windows the test assumes sphericity, so both degrees of freedom are
    multiplied by the Greenhouse-Geisser epsilon of the pooled within-group
    covariance (epsilon = 1 for two windows, where the test is the pooled
    t-test of the Post - Pre change). Residual normality is reported
    (Shapiro-Wilk), not enforced: there is no rank-based equivalent here.

    Descriptive only (p = NaN) with fewer than two groups or two windows, or a
    group with fewer than ``MIN_TEST_N`` complete animals.
    """
    from scipy import stats
    arrays = []
    for values in groups:
        data = np.atleast_2d(np.asarray(values, float))
        arrays.append(data[np.all(np.isfinite(data), axis=1)] if data.size else data.reshape(0, 0))
    windows = arrays[0].shape[1] if arrays and arrays[0].ndim == 2 else 0
    total = int(sum(len(data) for data in arrays))
    result: Dict[str, object] = {"test": "Descriptive only", "statistic": np.nan, "p": np.nan, "df1": np.nan,
                                 "df2": np.nan, "epsilon": np.nan, "effect_name": "", "effect": np.nan,
                                 "n": total, "normality": "", "note": ""}
    if len(arrays) < 2 or windows < 2:
        result["note"] = "An interaction needs at least two groups and two windows."
        return result
    if any(len(data) < MIN_TEST_N for data in arrays):
        result["note"] = f"A group has fewer than {MIN_TEST_N} animals with every window: descriptive only."
        return result
    # Remove each animal's mean: what is left is its profile across the windows.
    centered = [data - data.mean(axis=1, keepdims=True) for data in arrays]
    pooled = np.vstack(centered)
    overall = pooled.mean(axis=0)
    ss_effect = float(sum(len(data) * np.sum((data.mean(axis=0) - overall) ** 2) for data in centered))
    residuals = np.vstack([data - data.mean(axis=0) for data in centered])
    ss_error = float(np.sum(residuals ** 2))
    df1, df2 = (len(arrays) - 1) * (windows - 1), (total - len(arrays)) * (windows - 1)
    if ss_error <= 0 or df2 <= 0:
        result["note"] = "No variability between animals within the groups: no test."
        return result
    # Greenhouse-Geisser epsilon from the pooled within-group covariance (double-centered).
    covariance = sum((data - data.mean(axis=0)).T @ (data - data.mean(axis=0)) for data in arrays)
    covariance = covariance / (total - len(arrays))
    center = np.eye(windows) - 1.0 / windows
    contrast = center @ covariance @ center
    epsilon = float(np.trace(contrast) ** 2 / ((windows - 1) * np.sum(contrast * contrast)))
    epsilon = min(1.0, max(1.0 / (windows - 1), epsilon))
    statistic = (ss_effect / df1) / (ss_error / df2)
    shapiro_p = float(stats.shapiro(residuals.ravel()).pvalue) if residuals.size >= 3 else np.nan
    corrected = windows > 2
    result.update(
        test="Mixed ANOVA interaction" + (" (Greenhouse-Geisser)" if corrected else ""),
        statistic=float(statistic), df1=df1 * epsilon, df2=df2 * epsilon, epsilon=epsilon,
        p=float(stats.f.sf(statistic, df1 * epsilon, df2 * epsilon)),
        effect_name="partial eta squared", effect=ss_effect / (ss_effect + ss_error),
        normality=f"Shapiro-Wilk p (residuals)={shapiro_p:.3g}",
        note=("Group x window interaction: animals with every window only. Assumes normal residuals"
              + ("" if not np.isfinite(shapiro_p) or shapiro_p > alpha else " (rejected here: interpret with care)")
              + (" and sphericity, corrected by Greenhouse-Geisser." if corrected else ".")))
    return result
