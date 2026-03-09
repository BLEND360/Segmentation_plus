"""Cluster evaluation: scoring, pass/fail gate, stability testing."""
from __future__ import annotations

import logging

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import (
    adjusted_rand_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)

from .config import PipelineConfig
from .types import ClusterRunResult, EvaluationResult, StabilityResult

log = logging.getLogger("segplus.evaluation")


class ClusterEvaluator:
    """Evaluates clustering results and applies the pass/fail gate."""

    def __init__(self, config: PipelineConfig):
        self.config = config

    def evaluate(
        self,
        X: np.ndarray,
        results: dict[str, ClusterRunResult],
    ) -> EvaluationResult:
        """Score all algorithm results, pick the best via composite ranking, apply pass/fail gate."""
        # Filter to valid results first (>= 2 non-noise clusters)
        valid_results = {name: res for name, res in results.items() if res.is_valid}
        if not valid_results:
            log.warning("No algorithm produced >= 2 valid clusters; scoring all results as fallback.")
            valid_results = results

        scored: dict[str, dict[str, float]] = {}

        for name, res in valid_results.items():
            s = self._score_one(X, res)
            s["n_clusters"] = float(res.n_clusters)
            # Coverage: fraction of data points actually clustered (penalises heavy noise)
            n_clustered = int((res.labels != -1).sum())
            s["coverage"] = n_clustered / max(len(res.labels), 1)
            scored[name] = s
            log.info(
                "  [%s] k=%d | sil=%.4f | DB=%.4f | CH=%.1f | cov=%.2f",
                name, res.n_clusters,
                s["silhouette"], s["davies_bouldin"], s["calinski_harabasz"],
                s["coverage"],
            )

        # Pick best using composite rank across all three metrics + coverage
        best_name = self._pick_best_composite(scored)
        best_s = scored[best_name]
        best_res = valid_results[best_name]

        passes = self._check_pass(best_s["silhouette"], best_s["davies_bouldin"])

        return EvaluationResult(
            algorithm=best_name,
            labels=best_res.labels,
            n_clusters=best_res.n_clusters,
            silhouette=best_s["silhouette"],
            davies_bouldin=best_s["davies_bouldin"],
            calinski_harabasz=best_s["calinski_harabasz"],
            passes=passes,
            all_scores=scored,
            model=best_res.model,
        )

    def _pick_best_composite(self, scored: dict[str, dict[str, float]]) -> str:
        """
        Rank-based composite selection using all computed metrics.

        Each algorithm is ranked per metric (1=best), ranks are normalised to [0,1],
        then blended:
            composite = 0.40 * silhouette_rank    (higher is better)
                      + 0.25 * db_rank             (lower is better → inverted)
                      + 0.20 * ch_rank             (higher is better)
                      + 0.15 * coverage_rank       (higher is better, penalises DBSCAN noise)

        If there is only one algorithm, it wins by default.
        """
        names = list(scored.keys())
        if len(names) == 1:
            return names[0]

        n = len(names)

        def _rank_higher_better(metric: str) -> dict[str, float]:
            """Rank so that higher value → rank 1 (best)."""
            ordered = sorted(names, key=lambda nm: scored[nm][metric], reverse=True)
            return {nm: (i + 1) for i, nm in enumerate(ordered)}

        def _rank_lower_better(metric: str) -> dict[str, float]:
            """Rank so that lower value → rank 1 (best)."""
            ordered = sorted(names, key=lambda nm: scored[nm][metric])
            return {nm: (i + 1) for i, nm in enumerate(ordered)}

        sil_ranks = _rank_higher_better("silhouette")
        db_ranks = _rank_lower_better("davies_bouldin")
        ch_ranks = _rank_higher_better("calinski_harabasz")
        cov_ranks = _rank_higher_better("coverage")

        # Normalise ranks to [0,1] where 1=best
        def _norm(rank: float) -> float:
            return 1.0 - (rank - 1.0) / max(n - 1, 1)

        composite: dict[str, float] = {}
        for nm in names:
            composite[nm] = (
                0.40 * _norm(sil_ranks[nm])
                + 0.25 * _norm(db_ranks[nm])
                + 0.20 * _norm(ch_ranks[nm])
                + 0.15 * _norm(cov_ranks[nm])
            )

        best = max(composite, key=lambda nm: composite[nm])
        log.info(
            "Composite ranking: %s",
            " | ".join(f"{nm}={composite[nm]:.3f}" for nm in names),
        )
        log.info("Winner: %s (composite=%.3f)", best, composite[best])
        return best

    def _score_one(self, X: np.ndarray, result: ClusterRunResult) -> dict[str, float]:
        """Compute metrics for a single clustering result."""
        labels = result.labels
        valid_mask = labels != -1
        X_v = X[valid_mask]
        labels_v = labels[valid_mask]

        if len(set(labels_v)) < 2 or len(X_v) < 10:
            return {"silhouette": -1.0, "davies_bouldin": 99.0, "calinski_harabasz": 0.0}

        sample_size = min(2000, len(X_v))
        sil = silhouette_score(X_v, labels_v, sample_size=sample_size)
        db = davies_bouldin_score(X_v, labels_v)
        ch = calinski_harabasz_score(X_v, labels_v)

        return {
            "silhouette": round(sil, 4),
            "davies_bouldin": round(db, 4),
            "calinski_harabasz": round(ch, 2),
        }

    def _check_pass(self, silhouette: float, davies_bouldin: float) -> bool:
        """Pure pass/fail gate."""
        return (
            silhouette >= self.config.silhouette_threshold
            and davies_bouldin <= self.config.davies_bouldin_threshold
        )


def run_stability_test(
    X: np.ndarray,
    labels: np.ndarray,
    n_clusters: int,
    config: PipelineConfig,
    algorithm: str = "kmeans",
    model: object = None,
) -> StabilityResult:
    """Bootstrap ARI stability test using the winning algorithm."""
    from sklearn.mixture import GaussianMixture

    rng = np.random.default_rng(config.random_state)
    n = len(X)
    ari_scores: list[float] = []

    for _ in range(config.stability_n_bootstraps):
        idx = rng.choice(n, size=n, replace=True)
        X_boot = X[idx]

        if algorithm == "gmm":
            cov_type = (
                model.covariance_type
                if model is not None and hasattr(model, "covariance_type")
                else "full"
            )
            boot_model = GaussianMixture(
                n_components=n_clusters, covariance_type=cov_type,
                n_init=3, random_state=config.random_state,
            )
        else:
            # KMeans for kmeans winner; also used as proxy for DBSCAN
            # (DBSCAN is density-based so bootstrap changes density structure)
            boot_model = KMeans(
                n_clusters=n_clusters, n_init=5,
                random_state=config.random_state,
            )

        boot_labels = boot_model.fit_predict(X_boot)
        ari = adjusted_rand_score(labels[idx], boot_labels)
        ari_scores.append(ari)

    ari_mean = float(np.mean(ari_scores))
    ari_std = float(np.std(ari_scores))
    stable = ari_mean >= config.stability_ari_threshold

    log.info(
        "Stability (%s): ARI=%.3f +/- %.3f (threshold=%.2f, stable=%s)",
        algorithm, ari_mean, ari_std, config.stability_ari_threshold, stable,
    )
    return StabilityResult(
        ari_mean=round(ari_mean, 4),
        ari_std=round(ari_std, 4),
        n_bootstraps=config.stability_n_bootstraps,
        stable=stable,
    )
