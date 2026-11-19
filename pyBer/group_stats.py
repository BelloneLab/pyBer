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
