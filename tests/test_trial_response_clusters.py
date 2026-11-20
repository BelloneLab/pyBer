"""Model selection, probability contours and separation on seeded trial scores."""

from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import numpy as np
from sklearn.metrics import silhouette_samples

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "pyBer"))
import trial_response_clusters as clusters


class TrialResponseClustersTests(unittest.TestCase):
    def test_known_clusters_row_alignment_projection_and_silhouettes(self):
        rng = np.random.default_rng(173)
        points = np.vstack((rng.normal((-5, 0, 0), 0.6, (65, 3)),
                            rng.normal((5, 1, 0), 0.6, (65, 3))))
        points = np.insert(points, 42, (np.nan, 0, 0), axis=0)
        result = clusters.fit_trial_clusters(points)
        self.assertEqual(result["n_clusters"], 2)
        self.assertEqual(min(result["bic"], key=result["bic"].get), 2)
        runner_up = sorted(result["bic"], key=result["bic"].get)[1]
        gap = result["bic"][runner_up] - result["bic"][2]
        self.assertIn(f"{runner_up}-component model", result["note"])
        self.assertIn(f"BIC {gap:.2f} higher", result["note"])
        complete = np.isfinite(points).all(axis=1)
        np.testing.assert_array_equal(result["labels"][complete], np.repeat([0, 1], 65))
        self.assertEqual(result["labels"][42], -1)
        for k in range(2):
            members = points[result["labels"] == k]
            np.testing.assert_allclose(result["means"][k], members.mean(axis=0), atol=1e-8)
            np.testing.assert_allclose(result["covariances"][k], np.cov(members.T, bias=True), atol=4e-4)
        self.assertIsNone(result["silhouettes"])
        actual = clusters.trial_cluster_silhouettes(points, result)
        np.testing.assert_allclose(actual[complete], silhouette_samples(points[complete], result["labels"][complete]))
        self.assertTrue(np.isnan(actual[42]))
        self.assertIs(actual, clusters.trial_cluster_silhouettes(points, result))

    def test_unimodal_cloud_can_win_against_multiple_components(self):
        points = np.random.default_rng(37).normal(size=(120, 3))
        result = clusters.fit_trial_clusters(points)
        self.assertGreater(len(result["bic"]), 1)
        self.assertEqual(result["n_clusters"], 1)
        np.testing.assert_array_equal(result["labels"], 0)
        self.assertTrue(np.isnan(clusters.trial_cluster_silhouettes(points, result)).all())

    def test_too_few_complete_or_constant_points(self):
        for points in [np.empty((0, 3)), np.eye(3), np.full((20, 3), 0.1),
                       np.full((10, 3), np.nan)]:
            result = clusters.fit_trial_clusters(points)
            self.assertEqual(result["n_clusters"], 0)
            self.assertTrue(result["note"])
            np.testing.assert_array_equal(result["labels"], -1)
            self.assertEqual(result["covariances"].shape, (0, 3, 3))

    def test_outlier_cannot_form_a_singleton_component(self):
        rng = np.random.default_rng(111)
        points = np.vstack((rng.normal(size=(50, 2)), [[100.0, 100.0]]))
        result = clusters.fit_trial_clusters(points)
        self.assertGreaterEqual(result["n_clusters"], 1)
        self.assertGreaterEqual(np.bincount(result["labels"]).min(), clusters.MIN_CLUSTER_TRIALS)

    def test_supported_restart_is_compared_even_when_another_collapses(self):
        # With this outlier some two-component starts assign only one or
        # three trials to a component. A different start splits 11/16.
        points = np.random.default_rng(1).normal(size=(27, 3))
        points[-1] = [7, -1, 5]
        result = clusters.fit_trial_clusters(points)
        self.assertIn(2, result["bic"])
        self.assertEqual(result["n_clusters"], min(result["bic"], key=result["bic"].get))

        # With too few trials to support a second covariance, explain why
        # the panel reports one component instead of implying evidence for it.
        small = clusters.fit_trial_clusters(points[:7])
        self.assertEqual(set(small["bic"]), {1})
        self.assertIn("Only the one-component model", small["note"])

    def test_ellipse_matches_marginal_mahalanobis_distance(self):
        center = np.array([5.0, -3.0, 2.0])
        covariance = np.array([[4, 1, 1.8], [1, 2, 0.2], [1.8, 0.2, 3]])
        for plane in ([0, 1], [2, 0], [1, 2]):
            outline = clusters.cluster_ellipse(center, covariance, *plane)
            centered = outline - center[plane]
            inverse = np.linalg.inv(covariance[np.ix_(plane, plane)])
            squared_radius = np.einsum("ij,jk,ik->i", centered, inverse, centered)
            np.testing.assert_allclose(squared_radius, 5.991464547)
            np.testing.assert_allclose(outline[0], outline[-1])

    def test_rank_deficiency_and_stable_memberships_in_rescaled_units(self):
        scores = np.r_[np.arange(-20, -10), np.arange(10, 20)]
        points = np.stack([scores, 2 * scores, -scores], axis=1)
        result = clusters.fit_trial_clusters(points)
        self.assertEqual((result["rank"], result["n_clusters"]), (1, 2))
        outline = clusters.cluster_ellipse(result["means"][0], result["covariances"][0], 0, 1)
        self.assertEqual(outline.shape, (2, 2))
        np.testing.assert_allclose(outline[:, 1], 2 * outline[:, 0])
        shifted = clusters.fit_trial_clusters(points * 0.001 + 100)
        np.testing.assert_array_equal(result["labels"], shifted["labels"])

    def test_capped_model_fit_and_sampled_silhouettes_include_every_trial(self):
        rng = np.random.default_rng(42)
        points = np.vstack([rng.normal(loc, 0.25, (80, 2)) for loc in [-4, 4]])
        with patch.object(clusters, "MAX_FIT_TRIALS", 100), patch.object(clusters, "SILHOUETTE_REFERENCES", 12):
            first, second = clusters.fit_trial_clusters(points), clusters.fit_trial_clusters(points)
            scores = clusters.trial_cluster_silhouettes(points, first)
            np.testing.assert_array_equal(scores, clusters.trial_cluster_silhouettes(points, second))
            self.assertTrue(first["silhouette_sampled"])
            self.assertEqual((first["fit_count"], first["n_clusters"]), (100, 2))
            self.assertTrue(np.all((scores > 0.85) & (scores <= 1)))


if __name__ == "__main__":
    unittest.main()
