"""Clustering algorithm runners: K-Means, DBSCAN, GMM."""
from __future__ import annotations

import logging

import numpy as np
from sklearn.cluster import DBSCAN, KMeans
from sklearn.metrics import silhouette_score
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import NearestNeighbors

from .types import ClusteringConfig, ClusterRunResult

log = logging.getLogger("segplus.clustering")


def run_kmeans(X: np.ndarray, config: ClusteringConfig) -> ClusterRunResult:
    """Run K-Means clustering."""
    model = KMeans(
        n_clusters=config.k,
        init=config.kmeans_init,
        n_init=config.kmeans_n_init,
        max_iter=config.kmeans_max_iter,
        random_state=config.random_state,
    )
    labels = model.fit_predict(X)
    return ClusterRunResult(
        algorithm="kmeans",
        labels=labels,
        n_clusters=len(set(labels)),
        model=model,
        extra={"inertia": float(model.inertia_)},
    )


def run_dbscan(X: np.ndarray, config: ClusteringConfig) -> ClusterRunResult:
    """Run DBSCAN density-based clustering."""
    model = DBSCAN(eps=config.dbscan_eps, min_samples=config.dbscan_min_samples)
    labels = model.fit_predict(X)
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    noise_count = int((labels == -1).sum())
    return ClusterRunResult(
        algorithm="dbscan",
        labels=labels,
        n_clusters=max(n_clusters, 1),
        model=model,
        extra={"noise_count": noise_count, "noise_pct": noise_count / len(labels)},
    )


def run_gmm(X: np.ndarray, config: ClusteringConfig) -> ClusterRunResult:
    """Run Gaussian Mixture Model clustering."""
    model = GaussianMixture(
        n_components=config.k,
        covariance_type=config.gmm_covariance_type,
        n_init=config.gmm_n_init,
        random_state=config.random_state,
    )
    labels = model.fit_predict(X)
    probs = model.predict_proba(X)
    return ClusterRunResult(
        algorithm="gmm",
        labels=labels,
        n_clusters=len(set(labels)),
        model=model,
        probabilities=probs,
        extra={"bic": float(model.bic(X)), "aic": float(model.aic(X))},
    )


def run_all_algorithms(X: np.ndarray, config: ClusteringConfig) -> dict[str, ClusterRunResult]:
    """Run all three clustering algorithms, applying feature subset if configured."""
    X_work = X
    if config.feature_subset_indices is not None:
        X_work = X[:, config.feature_subset_indices]

    results = {}
    for name, runner in [("kmeans", run_kmeans), ("dbscan", run_dbscan), ("gmm", run_gmm)]:
        try:
            results[name] = runner(X_work, config)
            log.info(
                "  [%s] clusters=%d",
                name, results[name].n_clusters,
            )
        except Exception as e:
            log.warning("  [%s] failed: %s", name, e)
    return results


def find_optimal_k(
    X: np.ndarray,
    k_range: tuple[int, int],
    random_state: int = 42,
) -> tuple[int, dict[int, float]]:
    """Silhouette sweep to find optimal K for K-Means."""
    best_k, best_score = k_range[0], -1.0
    scores: dict[int, float] = {}

    for k in range(k_range[0], k_range[1] + 1):
        km = KMeans(n_clusters=k, init="k-means++", n_init=5, random_state=random_state)
        labels = km.fit_predict(X)
        if len(set(labels)) < 2:
            continue
        s = silhouette_score(X, labels, sample_size=min(2000, len(X)))
        scores[k] = round(s, 4)
        if s > best_score:
            best_score, best_k = s, k

    log.info("K search: scores=%s | best k=%d (sil=%.4f)", scores, best_k, best_score)
    return best_k, scores


def estimate_dbscan_eps(X: np.ndarray, min_samples: int = 5) -> float:
    """Estimate DBSCAN eps using k-distance knee detection."""
    nn = NearestNeighbors(n_neighbors=min_samples)
    nn.fit(X)
    distances, _ = nn.kneighbors(X)
    k_dist = np.sort(distances[:, -1])

    # Simple knee detection: max second derivative
    if len(k_dist) < 10:
        return float(np.median(k_dist))

    diffs = np.diff(k_dist)
    diffs2 = np.diff(diffs)
    knee_idx = int(np.argmax(diffs2)) + 2
    eps = float(k_dist[min(knee_idx, len(k_dist) - 1)])
    eps = max(eps, 0.1)  # floor

    log.info("DBSCAN eps estimated: %.3f (knee at index %d)", eps, knee_idx)
    return round(eps, 3)
