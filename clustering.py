"""
Segmentation Plus — Clustering Engine
======================================
K-Means, DBSCAN, and GMM clustering with hyperparameter search
and model comparison.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN, KMeans
from sklearn.metrics import (
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)
from sklearn.mixture import GaussianMixture

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────
# Clustering Result
# ──────────────────────────────────────────────

@dataclass
class ClusteringResult:
    """Container for a single clustering run's outputs."""

    algorithm: str
    labels: np.ndarray
    n_clusters: int
    metrics: Dict[str, float] = field(default_factory=dict)
    model: object = None  # The fitted model object
    probabilities: Optional[np.ndarray] = None  # GMM soft assignments
    extra: Dict = field(default_factory=dict)

    @property
    def is_valid(self) -> bool:
        """Check if clustering produced at least 2 distinct clusters."""
        unique = len(set(self.labels) - {-1})  # Exclude DBSCAN noise
        return unique >= 2


# ──────────────────────────────────────────────
# Clustering Engine
# ──────────────────────────────────────────────

class ClusteringEngine:
    """
    Runs multiple clustering algorithms, evaluates them,
    and selects the best model.

    Parameters
    ----------
    X : np.ndarray
        Scaled/PCA-transformed feature matrix (n_samples, n_features).
    k_range : tuple
        Min and max K to evaluate (default: 2 to 10).
    random_state : int
        Random seed for reproducibility.
    """

    def __init__(
        self,
        X: np.ndarray,
        k_range: Tuple[int, int] = (2, 10),
        random_state: int = 42,
    ) -> None:
        self.X = X
        self.k_min, self.k_max = k_range
        self.random_state = random_state
        self.results: Dict[str, ClusteringResult] = {}

    # ── K-Means ──

    def run_kmeans(self, n_clusters: Optional[int] = None) -> ClusteringResult:
        """
        Run K-Means clustering.

        If n_clusters is None, runs elbow analysis and picks the optimal K.
        """
        logger.info("Running K-Means clustering...")

        if n_clusters is None:
            n_clusters = self._find_optimal_k_kmeans()

        model = KMeans(
            n_clusters=n_clusters,
            n_init=10,
            max_iter=300,
            random_state=self.random_state,
        )
        labels = model.fit_predict(self.X)

        result = ClusteringResult(
            algorithm="kmeans",
            labels=labels,
            n_clusters=n_clusters,
            model=model,
            extra={"inertia": model.inertia_},
        )
        result.metrics = self._compute_metrics(labels)

        self.results["kmeans"] = result
        logger.info(
            "K-Means: K=%d, Silhouette=%.3f, Inertia=%.1f",
            n_clusters, result.metrics.get("silhouette", 0), model.inertia_,
        )
        return result

    def _find_optimal_k_kmeans(self) -> int:
        """Find optimal K using silhouette score sweep."""
        best_k, best_score = 2, -1
        scores = {}

        for k in range(self.k_min, self.k_max + 1):
            km = KMeans(n_clusters=k, n_init=10, random_state=self.random_state)
            labs = km.fit_predict(self.X)
            score = silhouette_score(self.X, labs)
            scores[k] = score
            if score > best_score:
                best_k, best_score = k, score

        logger.info("K-Means silhouette sweep: %s → optimal K=%d (%.3f)", scores, best_k, best_score)
        return best_k

    # ── DBSCAN ──

    def run_dbscan(
        self,
        eps: Optional[float] = None,
        min_samples: int = 5,
    ) -> ClusteringResult:
        """
        Run DBSCAN clustering.

        If eps is None, auto-selects using k-distance heuristic.
        """
        logger.info("Running DBSCAN clustering...")

        if eps is None:
            eps = self._estimate_eps(min_samples)

        model = DBSCAN(eps=eps, min_samples=min_samples)
        labels = model.fit_predict(self.X)

        n_clusters = len(set(labels) - {-1})
        n_noise = int((labels == -1).sum())

        result = ClusteringResult(
            algorithm="dbscan",
            labels=labels,
            n_clusters=n_clusters,
            model=model,
            extra={"eps": eps, "min_samples": min_samples, "n_noise": n_noise},
        )

        if result.is_valid:
            result.metrics = self._compute_metrics(labels[labels != -1], exclude_noise=True)
        else:
            logger.warning("DBSCAN found < 2 clusters (n_clusters=%d, noise=%d)", n_clusters, n_noise)

        self.results["dbscan"] = result
        logger.info("DBSCAN: clusters=%d, noise=%d, eps=%.3f", n_clusters, n_noise, eps)
        return result

    def _estimate_eps(self, min_samples: int) -> float:
        """Estimate eps using k-distance plot knee detection."""
        from sklearn.neighbors import NearestNeighbors

        nn = NearestNeighbors(n_neighbors=min_samples)
        nn.fit(self.X)
        distances, _ = nn.kneighbors(self.X)
        k_distances = np.sort(distances[:, -1])

        # Simple knee detection: point of max curvature
        diffs = np.diff(k_distances)
        knee_idx = np.argmax(diffs) + 1
        eps = float(k_distances[knee_idx])

        logger.info("Auto-estimated DBSCAN eps=%.3f (knee at index %d)", eps, knee_idx)
        return eps

    # ── GMM ──

    def run_gmm(self, n_components: Optional[int] = None) -> ClusteringResult:
        """
        Run Gaussian Mixture Model clustering.

        If n_components is None, selects using BIC minimization.
        """
        logger.info("Running GMM clustering...")

        if n_components is None:
            n_components = self._find_optimal_k_gmm()

        model = GaussianMixture(
            n_components=n_components,
            covariance_type="full",
            n_init=3,
            random_state=self.random_state,
        )
        labels = model.fit_predict(self.X)
        probabilities = model.predict_proba(self.X)

        result = ClusteringResult(
            algorithm="gmm",
            labels=labels,
            n_clusters=n_components,
            model=model,
            probabilities=probabilities,
            extra={
                "bic": model.bic(self.X),
                "aic": model.aic(self.X),
                "converged": model.converged_,
            },
        )
        result.metrics = self._compute_metrics(labels)

        self.results["gmm"] = result
        logger.info(
            "GMM: K=%d, Silhouette=%.3f, BIC=%.1f, AIC=%.1f",
            n_components, result.metrics.get("silhouette", 0),
            result.extra["bic"], result.extra["aic"],
        )
        return result

    def _find_optimal_k_gmm(self) -> int:
        """Find optimal K for GMM using BIC minimization."""
        best_k, best_bic = 2, np.inf

        for k in range(self.k_min, self.k_max + 1):
            gm = GaussianMixture(
                n_components=k, covariance_type="full",
                n_init=3, random_state=self.random_state,
            )
            gm.fit(self.X)
            bic = gm.bic(self.X)
            if bic < best_bic:
                best_k, best_bic = k, bic

        logger.info("GMM BIC sweep → optimal K=%d (BIC=%.1f)", best_k, best_bic)
        return best_k

    # ── Metrics ──

    def _compute_metrics(
        self, labels: np.ndarray, exclude_noise: bool = False,
    ) -> Dict[str, float]:
        """Compute clustering quality metrics."""
        X = self.X
        if exclude_noise:
            mask = labels != -1
            X = self.X[mask]
            labels = labels[mask]

        if len(set(labels)) < 2:
            return {}

        return {
            "silhouette": float(silhouette_score(X, labels)),
            "davies_bouldin": float(davies_bouldin_score(X, labels)),
            "calinski_harabasz": float(calinski_harabasz_score(X, labels)),
        }

    # ── Run All & Compare ──

    def run_all(self) -> Dict[str, ClusteringResult]:
        """Run all three algorithms and return results."""
        self.run_kmeans()
        self.run_dbscan()
        self.run_gmm()
        return self.results

    def get_best_model(self, metric: str = "silhouette") -> ClusteringResult:
        """
        Select best model by a given metric.

        Parameters
        ----------
        metric : str
            Metric to compare (default: 'silhouette', higher is better).

        Returns
        -------
        ClusteringResult
            Best performing clustering result.
        """
        valid = {
            name: r for name, r in self.results.items()
            if r.is_valid and metric in r.metrics
        }

        if not valid:
            raise ValueError("No valid clustering results found.")

        # Higher is better for silhouette & calinski_harabasz; lower for davies_bouldin
        reverse = metric != "davies_bouldin"
        best_name = max(valid, key=lambda n: valid[n].metrics[metric] * (1 if reverse else -1))

        logger.info(
            "Best model by %s: %s (%.4f)",
            metric, best_name, valid[best_name].metrics[metric],
        )
        return valid[best_name]
