"""Core modeling loop: configure -> cluster -> evaluate -> reconfigure retry cycle."""
from __future__ import annotations

import logging
from datetime import datetime
from enum import Enum

import numpy as np
from sklearn.metrics import silhouette_score

from .clustering import (
    estimate_dbscan_eps,
    find_optimal_k,
    run_all_algorithms,
)
from .config import PipelineConfig
from .evaluation import ClusterEvaluator
from .experiment_log import ExperimentLog
from .types import ClusteringConfig, EvaluationResult, ExperimentRecord

log = logging.getLogger("segplus.modeling_loop")


class ReconfigStrategy(str, Enum):
    ADJUST_K = "adjust_k"
    ADJUST_EPS = "adjust_eps"
    FEATURE_SUBSET = "feature_subset"
    CHANGE_GMM_COV = "change_gmm_covariance"
    WIDEN_K = "widen_k_range"


class ClusteringConfigurator:
    """Initial configuration and smart reconfiguration on evaluation failure."""

    def __init__(self, config: PipelineConfig, feature_names: list[str]):
        self.config = config
        self.feature_names = feature_names
        self._k_scores: dict[int, float] = {}

    def initial_configure(self, X: np.ndarray) -> ClusteringConfig:
        """Silhouette-guided K selection + DBSCAN eps estimation."""
        best_k, self._k_scores = find_optimal_k(
            X, self.config.k_range, self.config.random_state
        )
        eps = estimate_dbscan_eps(X, min_samples=5)

        return ClusteringConfig(
            k=best_k,
            dbscan_eps=eps,
            dbscan_min_samples=5,
            random_state=self.config.random_state,
        )

    def reconfigure(
        self,
        X: np.ndarray,
        current: ClusteringConfig,
        eval_result: EvaluationResult,
        iteration: int,
        history: list[ExperimentRecord],
    ) -> tuple[ClusteringConfig, str]:
        """Smart reconfiguration. Returns (new_config, strategy_name)."""
        strategy = self._select_strategy(iteration, eval_result, history)
        new = ClusteringConfig(
            k=current.k,
            dbscan_eps=current.dbscan_eps,
            dbscan_min_samples=current.dbscan_min_samples,
            gmm_covariance_type=current.gmm_covariance_type,
            random_state=current.random_state,
            feature_subset_indices=current.feature_subset_indices,
        )

        if strategy == ReconfigStrategy.ADJUST_K:
            new = self._adjust_k(new, eval_result)
        elif strategy == ReconfigStrategy.ADJUST_EPS:
            new = self._adjust_eps(new, eval_result)
        elif strategy == ReconfigStrategy.FEATURE_SUBSET:
            new = self._try_feature_subset(X, eval_result, new)
        elif strategy == ReconfigStrategy.CHANGE_GMM_COV:
            new = self._change_gmm_cov(new)
        elif strategy == ReconfigStrategy.WIDEN_K:
            new = self._widen_k(X, new)

        log.info(
            "Reconfigured (strategy=%s): k=%d, eps=%.3f, gmm_cov=%s, subset=%s",
            strategy.value, new.k, new.dbscan_eps, new.gmm_covariance_type,
            new.feature_subset_indices is not None,
        )
        return new, strategy.value

    def _select_strategy(
        self, iteration: int, eval_result: EvaluationResult, history: list[ExperimentRecord]
    ) -> ReconfigStrategy:
        """Deterministic strategy selection based on iteration and history."""
        strategies = [
            ReconfigStrategy.ADJUST_K,
            ReconfigStrategy.ADJUST_EPS,
            ReconfigStrategy.FEATURE_SUBSET,
            ReconfigStrategy.CHANGE_GMM_COV,
            ReconfigStrategy.WIDEN_K,
        ]
        idx = min(iteration - 1, len(strategies) - 1)
        return strategies[idx]

    def _adjust_k(self, cfg: ClusteringConfig, eval_result: EvaluationResult) -> ClusteringConfig:
        """Adjust K based on silhouette trend."""
        # If we have k_scores from initial scan, use them to decide direction
        current_k = cfg.k
        k_min, k_max = self.config.k_range

        # Check if higher K had better scores during initial scan
        higher_better = any(
            self._k_scores.get(k, -1) > self._k_scores.get(current_k, -1)
            for k in range(current_k + 1, k_max + 1)
        )
        lower_better = any(
            self._k_scores.get(k, -1) > self._k_scores.get(current_k, -1)
            for k in range(k_min, current_k)
        )

        if higher_better and current_k < k_max:
            cfg.k = current_k + 1
        elif lower_better and current_k > k_min:
            cfg.k = current_k - 1
        elif current_k < k_max:
            cfg.k = current_k + 1
        else:
            cfg.k = max(k_min, current_k - 1)

        return cfg

    def _adjust_eps(self, cfg: ClusteringConfig, eval_result: EvaluationResult) -> ClusteringConfig:
        """Adjust DBSCAN eps based on noise ratio."""
        dbscan_scores = eval_result.all_scores.get("dbscan", {})
        # If DBSCAN silhouette was very low or negative, try larger eps
        dbscan_sil = dbscan_scores.get("silhouette", -1)
        if dbscan_sil < 0:
            cfg.dbscan_eps = round(cfg.dbscan_eps * 1.5, 3)
        else:
            cfg.dbscan_eps = round(cfg.dbscan_eps * 1.2, 3)
        cfg.dbscan_min_samples = max(3, cfg.dbscan_min_samples - 1)
        return cfg

    def _try_feature_subset(
        self, X: np.ndarray, eval_result: EvaluationResult, cfg: ClusteringConfig
    ) -> ClusteringConfig:
        """Drop lowest-importance features using quick permutation check."""
        labels = eval_result.labels
        valid_mask = labels != -1
        Xv, lv = X[valid_mask], labels[valid_mask]

        if len(set(lv)) < 2 or Xv.shape[1] <= 3:
            return cfg

        rng = np.random.default_rng(0)
        baseline = silhouette_score(Xv, lv, sample_size=min(1000, len(Xv)))

        drops = []
        for i in range(Xv.shape[1]):
            Xp = Xv.copy()
            rng.shuffle(Xp[:, i])
            s = silhouette_score(Xp, lv, sample_size=min(1000, len(Xv)))
            drops.append(baseline - s)

        # Keep features whose removal hurts silhouette the most
        n_keep = max(3, int(len(drops) * 0.7))
        top_indices = sorted(range(len(drops)), key=lambda i: -drops[i])[:n_keep]
        cfg.feature_subset_indices = sorted(top_indices)
        return cfg

    def _change_gmm_cov(self, cfg: ClusteringConfig) -> ClusteringConfig:
        """Cycle GMM covariance type."""
        cov_types = ["full", "tied", "diag", "spherical"]
        current_idx = cov_types.index(cfg.gmm_covariance_type) if cfg.gmm_covariance_type in cov_types else 0
        cfg.gmm_covariance_type = cov_types[(current_idx + 1) % len(cov_types)]
        return cfg

    def _widen_k(self, X: np.ndarray, cfg: ClusteringConfig) -> ClusteringConfig:
        """Widen K range and re-search."""
        wider_min = max(2, self.config.k_range[0] - 1)
        wider_max = min(self.config.k_range[1] + 2, len(X) // 10)
        best_k, scores = find_optimal_k(X, (wider_min, wider_max), self.config.random_state)
        self._k_scores.update(scores)
        cfg.k = best_k
        return cfg


def modeling_loop(
    X: np.ndarray,
    feature_names: list[str],
    config: PipelineConfig,
    experiment_log: ExperimentLog,
) -> tuple[EvaluationResult, ClusteringConfig]:
    """
    Core retry loop:
      1. Configure initial clustering params
      2. Run all 3 algorithms (KMeans, DBSCAN, GMM)
      3. Evaluate -> pass/fail gate
      4. If fail -> reconfigure (smart strategy) -> goto 2
      5. If pass or max_iterations -> return best result
    """
    configurator = ClusteringConfigurator(config, feature_names)
    evaluator = ClusterEvaluator(config)

    cluster_config = configurator.initial_configure(X)
    best_eval: EvaluationResult | None = None
    best_config: ClusteringConfig | None = None

    for iteration in range(1, config.max_iterations + 1):
        log.info("=" * 60)
        log.info("Modeling Loop - Iteration %d/%d | k=%d", iteration, config.max_iterations, cluster_config.k)
        log.info("=" * 60)

        # Run all algorithms
        results = run_all_algorithms(X, cluster_config)
        if not results:
            log.warning("No clustering algorithms succeeded in iteration %d", iteration)
            continue

        # Evaluate
        eval_result = evaluator.evaluate(X, results)

        # Log experiment
        record = ExperimentRecord(
            iteration=iteration,
            timestamp=datetime.now().isoformat(),
            config_k=cluster_config.k,
            config_eps=cluster_config.dbscan_eps,
            config_gmm_cov=cluster_config.gmm_covariance_type,
            best_algorithm=eval_result.algorithm,
            n_clusters=eval_result.n_clusters,
            silhouette=eval_result.silhouette,
            davies_bouldin=eval_result.davies_bouldin,
            calinski_harabasz=eval_result.calinski_harabasz,
            passed=eval_result.passes,
        )
        experiment_log.add(record)

        # Track global best
        if best_eval is None or eval_result.silhouette > best_eval.silhouette:
            best_eval = eval_result
            best_config = cluster_config

        log.info(
            "Best: %s | sil=%.4f | DB=%.4f | PASS=%s",
            eval_result.algorithm, eval_result.silhouette,
            eval_result.davies_bouldin, eval_result.passes,
        )

        # Gate check
        if eval_result.passes:
            log.info("Evaluation PASSED on iteration %d.", iteration)
            break

        # Reconfigure for next iteration
        if iteration < config.max_iterations:
            log.info("Evaluation FAILED. Reconfiguring...")
            cluster_config, strategy = configurator.reconfigure(
                X, cluster_config, eval_result, iteration, experiment_log.records
            )
            # Update the record with reconfiguration strategy
            record.reconfiguration_strategy = strategy

    if best_eval is None:
        raise RuntimeError("No valid clustering result produced across all iterations.")

    if not best_eval.passes:
        log.warning(
            "Max iterations reached without passing gate. Using best result "
            "(sil=%.4f, algorithm=%s).", best_eval.silhouette, best_eval.algorithm,
        )

    return best_eval, best_config
