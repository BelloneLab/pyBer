"""Between-group statistics: test choice, effect sizes, Holm, descriptive fallback."""
import unittest

import numpy as np
from scipy import stats

from pyBer.group_stats import compare_groups, per_file_means, stats_label


def _groups(*samples, tested=None):
    tested = tested or [True] * len(samples)
    return [{"name": f"G{i}", "color": "#000000", "values": s, "tested": t}
            for i, (s, t) in enumerate(zip(samples, tested))]


class GroupStatsTests(unittest.TestCase):
    def test_two_normal_groups_use_welch_and_match_scipy(self):
        a = np.array([1.0, 1.2, 0.9, 1.1, 1.05])
        b = np.array([2.0, 2.4, 1.9, 2.2, 2.1])
        result = compare_groups(_groups(a, b))
        self.assertEqual(result["test"], "Welch t-test")
        self.assertAlmostEqual(result["p"], stats.ttest_ind(a, b, equal_var=False).pvalue)
        self.assertGreater(result["effect"], 0)  # B above A
        self.assertIn("Welch t-test p=", stats_label(result))

    def test_non_normal_groups_use_mann_whitney(self):
        a = np.array([0, 0, 0, 0, 0, 0, 10.0])
        b = np.array([1, 1, 1, 1, 1, 1, 30.0])
        result = compare_groups(_groups(a, b))
        self.assertEqual(result["test"], "Mann-Whitney U")
        self.assertAlmostEqual(result["p"], stats.mannwhitneyu(a, b, alternative="two-sided").pvalue)

    def test_three_groups_omnibus_and_holm_pairwise(self):
        rng = np.random.default_rng(0)
        samples = [rng.normal(mu, 1, 8) for mu in (0, 0.5, 3)]
        result = compare_groups(_groups(*samples))
        self.assertIn(result["test"], {"One-way ANOVA", "Kruskal-Wallis"})
        self.assertEqual(len(result["pairwise"]), 3)
        raw = sorted(pair["p"] for pair in result["pairwise"])
        adjusted = sorted(pair["p_holm"] for pair in result["pairwise"])
        self.assertAlmostEqual(adjusted[0], min(1, raw[0] * 3))

    def test_small_groups_and_ungrouped_are_descriptive(self):
        result = compare_groups(_groups([1.0, 2.0], [3.0, 4.0, 5.0]))
        self.assertEqual(result["test"], "Descriptive only")
        self.assertEqual(result["groups"][0]["n"], 2)
        result = compare_groups(_groups([1, 2, 3.0], [4, 5, 6.0], tested=[True, False]))
        self.assertEqual(result["test"], "Descriptive only")

    def test_trial_rows_average_per_animal(self):
        order, means = per_file_means([1, 3, np.nan, 10], ["a", "a", "b", "b"])
        self.assertEqual(order, ["a", "b"])
        np.testing.assert_allclose(means, [2, 10])


if __name__ == "__main__":
    unittest.main()
