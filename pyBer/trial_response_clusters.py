"""Gaussian mixture selection and descriptive separation for trial-response PCA."""

import warnings

import numpy as np


# The dashboard supplies at most three PCA scores per trial. A component needs
# enough assigned trials to estimate its own covariance without a singleton fit.
MAX_CLUSTERS = 6
MIN_CLUSTER_TRIALS = 5
MAX_FIT_TRIALS = 2000
GMM_STARTS = 4
GMM_SEED = 0
GMM_REGULARIZATION = 1e-5  # relative to the variance along each fitted direction
ELLIPSE_PROBABILITY = 0.95
ELLIPSE_POINTS = 121
SILHOUETTE_REFERENCES = 256  # per cluster; all trials still receive a score
DISTANCE_BATCH = 512


def fit_trial_clusters(scores):
    """Fit full-covariance mixtures in the common PCA space, retaining the lowest eligible BIC.

    Rows are trials. Incomplete rows keep label -1. Centers and covariance
    matrices use the original PC coordinates so changing the displayed plane
    never changes cluster membership. Larger inputs use a seeded uniform sample
    for model selection and estimate labels for every complete trial.
    """
    points = np.asarray(scores, dtype=float)
    if points.ndim != 2:
        raise ValueError("Trial scores must have shape (trials, PCs).")
    n_rows, n_pcs = points.shape
    valid = np.isfinite(points).all(axis=1)
    complete = points[valid]
    result = {
        "n_clusters": 0, "fit_count": 0, "rank": 0, "bic": {}, "note": "",
        "labels": np.full(n_rows, -1, dtype=int),
        "means": np.empty((0, n_pcs)), "covariances": np.empty((0, n_pcs, n_pcs)),
        "silhouettes": None, "silhouette_sampled": False,
    }
    if n_pcs == 0 or len(complete) < MIN_CLUSTER_TRIALS:
        result["note"] = f"Clustering needs at least {MIN_CLUSTER_TRIALS} complete trials with PC scores."
        return result

    sample = complete
    if len(sample) > MAX_FIT_TRIALS:
        indices = np.random.default_rng(GMM_SEED).choice(len(sample), MAX_FIT_TRIALS, replace=False)
        sample = sample[np.sort(indices)]

    # Work in the nonconstant subspace. Whitening makes regularization relative
    # to each score direction. Transform parameters back before plotting.
    origin = sample[0]
    offset = (sample - origin).mean(axis=0)
    centered = (sample - origin) - offset
    _, singular_values, axes = np.linalg.svd(centered, full_matrices=False)
    # Translation can leave rounding noise in a mathematically flat direction.
    # Include the precision of the input coordinates before whitening it.
    input_precision = np.spacing(np.max(np.abs(sample))) * np.sqrt(sample.size)
    tolerance = max(input_precision, singular_values[0] * max(centered.shape) * np.finfo(float).eps)
    rank = int(np.count_nonzero(singular_values > tolerance))
    result["rank"] = rank
    if rank == 0:
        result["note"] = "The complete trials have identical PC scores."
        return result
    axes = axes[:rank]
    scales = singular_values[:rank] / np.sqrt(len(sample))
    training = (centered @ axes.T) / scales
    min_members = max(MIN_CLUSTER_TRIALS, 2 * rank + 1)
    limit = min(MAX_CLUSTERS, max(1, len(sample) // min_members))

    # Keep the scientific stack off the application's startup path. The local
    # thread limit avoids oversubscribing BLAS for these one-to-three-PC fits.
    try:
        from sklearn.exceptions import ConvergenceWarning
        from sklearn.mixture import GaussianMixture
        from threadpoolctl import threadpool_limits
    except ImportError:
        result["note"] = "Trial clustering requires scikit-learn."
        return result

    best, best_bic = None, np.inf
    # Express BIC in the original, nonconstant subspace for every candidate.
    bic_offset = 2 * len(training) * np.log(scales).sum()
    result["fit_count"] = len(training)
    with threadpool_limits(limits=1), warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        for k in range(1, limit + 1):
            # Check each restart before comparing likelihoods. The highest
            # unconstrained likelihood can belong to an unsupported tiny cluster.
            starts = 1 if k == 1 else GMM_STARTS
            for restart in range(starts):
                mixture = GaussianMixture(
                    n_components=k, covariance_type="full", n_init=1,
                    init_params="random" if restart == GMM_STARTS - 1 else "kmeans",
                    reg_covar=GMM_REGULARIZATION, random_state=GMM_SEED + restart,
                )
                try:
                    assignments = mixture.fit_predict(training)
                except (ValueError, np.linalg.LinAlgError):
                    continue
                if not mixture.converged_:
                    continue
                if k > 1 and not _supported_components(training, assignments, k, min_members):
                    continue
                bic = float(mixture.bic(training) + bic_offset)
                result["bic"][k] = min(bic, result["bic"].get(k, np.inf))
                if bic < best_bic:
                    best, best_bic = mixture, bic

        if best is None:
            result["note"] = "No eligible Gaussian mixture converged."
            return result
        assigned = best.predict((((complete - origin) - offset) @ axes.T) / scales)

    # Number clusters by their original PC coordinates (PC1 first). The
    # arbitrary ordering chosen by EM then has no effect on the legend.
    transform = scales[:, None] * axes
    means = (best.means_ @ transform + offset) + origin
    order = np.lexsort(means.T[::-1])
    result["labels"][valid] = np.argsort(order)[assigned]
    result["n_clusters"] = best.n_components
    result["means"] = means[order]
    result["covariances"] = (transform.T @ best.covariances_ @ transform)[order]
    if len(result["bic"]) == 1:
        count = "one" if best.n_components == 1 else str(best.n_components)
        result["note"] = f"Only the {count}-component model met the fit criteria."
    else:
        noun = "component" if best.n_components == 1 else "components"
        alternative = min((k for k in result["bic"] if k != best.n_components), key=result["bic"].get)
        gap = result["bic"][alternative] - best_bic
        result["note"] = (f"BIC selected {best.n_components} {noun}. "
                          f"The {alternative}-component model was next, with BIC {gap:.2f} higher.")
    if len(training) < len(complete):
        result["note"] += f" Fitted on a sample of {len(training)} of {len(complete)} complete trials."
    return result


def _supported_components(points, labels, k, min_members):
    """Require enough distinct, nondegenerate observations for every covariance."""
    for cluster in range(k):
        members = points[labels == cluster]
        if len(members) < min_members:
            return False
        if np.linalg.matrix_rank(members - members[0]) != points.shape[1]:
            return False
    return True


def cluster_ellipse(mean, covariance, ax, ay):
    """Return the 95% Gaussian probability outline in the requested marginal PC plane."""
    from scipy.special import ndtri

    plane = [ax, ay]
    center = np.asarray(mean)[plane]
    marginal = np.asarray(covariance)[np.ix_(plane, plane)]
    variances, vectors = np.linalg.eigh((marginal + marginal.T) / 2)
    variances = np.maximum(variances, 0.0)
    # A rank-one projection has a 1-D Gaussian probability interval.
    if variances[0] <= variances[1] * np.finfo(float).eps * 2:
        radius = ndtri((1 + ELLIPSE_PROBABILITY) / 2) * np.sqrt(variances[1])
        return np.stack((center - radius * vectors[:, 1], center + radius * vectors[:, 1]))
    angles = np.linspace(0.0, 2 * np.pi, ELLIPSE_POINTS)
    circle = np.column_stack((np.cos(angles), np.sin(angles)))
    chi2_radius_squared = -2 * np.log1p(-ELLIPSE_PROBABILITY)  # two degrees of freedom
    return center + (circle * np.sqrt(chi2_radius_squared * variances)) @ vectors.T


def trial_cluster_silhouettes(scores, result):
    """Cache Euclidean silhouettes in PC score units, keeping each trial's row index.

    For large clusters, compare every trial with a fixed random reference
    sample per cluster. Where the trial itself is in its reference sample,
    exclude it from the average. Batch distances to bound peak memory use.
    """
    from scipy.spatial.distance import cdist

    if result["silhouettes"] is not None:
        return result["silhouettes"]
    labels = result["labels"]
    silhouettes = np.full(len(labels), np.nan)
    k = result["n_clusters"]
    if k >= 2:
        valid = labels >= 0
        points = np.asarray(scores, dtype=float)[valid]
        assigned = labels[valid]
        rng = np.random.default_rng(GMM_SEED)
        references = []
        for cluster in range(k):
            indices = np.flatnonzero(assigned == cluster)
            if len(indices) > SILHOUETTE_REFERENCES:
                indices = rng.choice(indices, SILHOUETTE_REFERENCES, replace=False)
                result["silhouette_sampled"] = True
            references.append(indices)
        values = np.empty(len(points))
        for start in range(0, len(points), DISTANCE_BATCH):
            rows = np.arange(start, min(start + DISTANCE_BATCH, len(points)))
            distances = np.empty((len(rows), k))
            for cluster, reference in enumerate(references):
                totals = cdist(points[rows], points[reference]).sum(axis=1)
                denominators = len(reference) - np.isin(rows, reference)
                distances[:, cluster] = totals / denominators
            entry = (np.arange(len(rows)), assigned[rows])
            within = distances[entry]
            distances[entry] = np.inf
            nearest = distances.min(axis=1)
            scale = np.maximum(within, nearest)
            values[rows] = np.divide(nearest - within, scale, out=np.zeros_like(scale), where=scale > 0)
        silhouettes[valid] = values
    result["silhouettes"] = silhouettes
    return silhouettes
