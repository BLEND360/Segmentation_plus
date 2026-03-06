"""
Segmentation Plus — Cluster Evaluation Framework
=================================================
4-step evaluation: Optimal K, Quality Metrics, Interpretability
(PCA loadings + SHAP), and Stability testing.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, adjusted_rand_score
from sklearn.mixture import GaussianMixture

from clustering import ClusteringResult

logger = logging.getLogger(__name__)


class ClusterEvaluator:
    """
    Comprehensive cluster evaluation framework.

    Parameters
    ----------
    X : np.ndarray
        Feature matrix used for clustering.
    labels : np.ndarray
        Cluster assignments.
    feature_names : list[str]
        Names of the features (columns).
    output_dir : Path
        Directory to save evaluation plots and reports.
    """

    def __init__(
        self,
        X: np.ndarray,
        labels: np.ndarray,
        feature_names: List[str],
        output_dir: Path,
    ) -> None:
        self.X = X
        self.labels = labels
        self.feature_names = feature_names
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.report: Dict = {}

    def run_full_evaluation(
        self,
        pca: Optional[PCA] = None,
        k_range: Tuple[int, int] = (2, 10),
        random_state: int = 42,
    ) -> Dict:
        """
        Run the complete 4-step evaluation pipeline.

        Returns
        -------
        dict
            Full evaluation report.
        """
        logger.info("Starting cluster evaluation (4 steps)...")

        # Step 1: Optimal K analysis
        self.report["optimal_k"] = self._step1_optimal_k(k_range, random_state)

        # Step 2: Quality metrics
        self.report["quality_metrics"] = self._step2_quality_metrics()

        # Step 3: Interpretability
        self.report["pca_loadings"] = self._step3a_pca_loadings(pca)
        self.report["shap_importance"] = self._step3b_shap_analysis(random_state)
        self._step3c_cluster_profiles()

        # Step 4: Stability
        self.report["stability"] = self._step4_stability(random_state)

        # Save report
        report_path = self.output_dir / "evaluation_report.json"
        with open(report_path, "w") as f:
            json.dump(self._serialize_report(self.report), f, indent=2)
        logger.info("Evaluation report saved to %s", report_path)

        return self.report

    # ── Step 1: Optimal K ──

    def _step1_optimal_k(self, k_range: Tuple[int, int], random_state: int) -> Dict:
        """Elbow plot (inertia), Silhouette vs K, BIC/AIC curves."""
        k_min, k_max = k_range
        ks = list(range(k_min, k_max + 1))
        inertias, silhouettes, bics, aics = [], [], [], []

        for k in ks:
            # K-Means
            km = KMeans(n_clusters=k, n_init=10, random_state=random_state)
            labs = km.fit_predict(self.X)
            inertias.append(float(km.inertia_))
            silhouettes.append(float(silhouette_score(self.X, labs)))

            # GMM
            gm = GaussianMixture(n_components=k, n_init=3, random_state=random_state)
            gm.fit(self.X)
            bics.append(float(gm.bic(self.X)))
            aics.append(float(gm.aic(self.X)))

        # Plot
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        axes[0].plot(ks, inertias, "bo-", linewidth=2)
        axes[0].set_title("Elbow Method (Inertia)", fontweight="bold")
        axes[0].set_xlabel("K"); axes[0].set_ylabel("Inertia")

        axes[1].plot(ks, silhouettes, "go-", linewidth=2)
        axes[1].set_title("Silhouette Score vs K", fontweight="bold")
        axes[1].set_xlabel("K"); axes[1].set_ylabel("Silhouette")

        axes[2].plot(ks, bics, "ro-", linewidth=2, label="BIC")
        axes[2].plot(ks, aics, "mo-", linewidth=2, label="AIC")
        axes[2].set_title("GMM: BIC / AIC", fontweight="bold")
        axes[2].set_xlabel("K"); axes[2].legend()

        plt.tight_layout()
        fig.savefig(self.output_dir / "optimal_k_analysis.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

        return {"k_range": ks, "inertias": inertias, "silhouettes": silhouettes, "bics": bics, "aics": aics}

    # ── Step 2: Quality Metrics ──

    def _step2_quality_metrics(self) -> Dict[str, float]:
        """Compute silhouette, Davies-Bouldin, Calinski-Harabasz."""
        from sklearn.metrics import davies_bouldin_score, calinski_harabasz_score

        mask = self.labels != -1
        X, labels = self.X[mask], self.labels[mask]

        if len(set(labels)) < 2:
            logger.warning("Cannot compute quality metrics: < 2 clusters")
            return {}

        metrics = {
            "silhouette": float(silhouette_score(X, labels)),
            "davies_bouldin": float(davies_bouldin_score(X, labels)),
            "calinski_harabasz": float(calinski_harabasz_score(X, labels)),
            "n_clusters": int(len(set(labels))),
        }

        logger.info(
            "Quality: Silhouette=%.3f, DB=%.3f, CH=%.1f",
            metrics["silhouette"], metrics["davies_bouldin"], metrics["calinski_harabasz"],
        )
        return metrics

    # ── Step 3a: PCA Loadings ──

    def _step3a_pca_loadings(self, pca: Optional[PCA]) -> Optional[Dict]:
        """Generate PCA loadings heatmap."""
        if pca is None:
            logger.info("No PCA object provided — fitting new PCA for loadings analysis")
            pca = PCA(n_components=min(5, self.X.shape[1])).fit(self.X)

        n_components = pca.n_components_
        feature_names = self.feature_names[:pca.components_.shape[1]]

        loadings = pd.DataFrame(
            pca.components_.T,
            columns=[f"PC{i+1}" for i in range(n_components)],
            index=feature_names,
        )

        # Heatmap
        fig, ax = plt.subplots(figsize=(max(8, n_components * 1.5), max(6, len(feature_names) * 0.4)))
        sns.heatmap(
            loadings, annot=True, fmt=".2f", cmap="RdBu_r",
            center=0, ax=ax, linewidths=0.5
        )
        ax.set_title("PCA Loadings Heatmap", fontsize=14, fontweight="bold")
        plt.tight_layout()
        fig.savefig(self.output_dir / "pca_loadings_heatmap.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

        logger.info("PCA loadings heatmap saved (%d components × %d features)", n_components, len(feature_names))

        return {
            "explained_variance": pca.explained_variance_ratio_.tolist(),
            "top_features_per_pc": {
                f"PC{i+1}": loadings.iloc[:, i].abs().nlargest(5).index.tolist()
                for i in range(n_components)
            }
        }

    # ── Step 3b: SHAP Analysis ──

    def _step3b_shap_analysis(self, random_state: int) -> Optional[Dict]:
        """Train RF on cluster labels → SHAP feature importance."""
        mask = self.labels != -1
        X, labels = self.X[mask], self.labels[mask]

        if len(set(labels)) < 2:
            return None

        # Train Random Forest
        rf = RandomForestClassifier(
            n_estimators=100, max_depth=10, random_state=random_state, n_jobs=-1,
        )
        rf.fit(X, labels)
        accuracy = rf.score(X, labels)
        logger.info("RF classifier accuracy on cluster labels: %.3f", accuracy)

        try:
            import shap
            explainer = shap.TreeExplainer(rf)
            shap_values = explainer.shap_values(X)

            # Summary plot
            fig, ax = plt.subplots(figsize=(10, max(6, len(self.feature_names) * 0.35)))
            plt.sca(ax)
            shap.summary_plot(
                shap_values, X,
                feature_names=self.feature_names[:X.shape[1]],
                show=False, max_display=15,
            )
            plt.tight_layout()
            fig.savefig(self.output_dir / "shap_summary.png", dpi=150, bbox_inches="tight")
            plt.close(fig)

            # Feature importance bar plot
            fig, ax = plt.subplots(figsize=(10, max(6, len(self.feature_names) * 0.35)))
            plt.sca(ax)
            shap.summary_plot(
                shap_values, X,
                feature_names=self.feature_names[:X.shape[1]],
                plot_type="bar", show=False, max_display=15,
            )
            plt.tight_layout()
            fig.savefig(self.output_dir / "shap_feature_importance.png", dpi=150, bbox_inches="tight")
            plt.close(fig)

            logger.info("SHAP plots saved")

            # Extract top features
            if isinstance(shap_values, list):
                mean_abs = np.mean([np.abs(sv).mean(axis=0) for sv in shap_values], axis=0)
            else:
                mean_abs = np.abs(shap_values).mean(axis=0)

            feature_importance = dict(zip(
                self.feature_names[:X.shape[1]],
                mean_abs.tolist() if hasattr(mean_abs, 'tolist') else mean_abs,
            ))

            return {
                "rf_accuracy": accuracy,
                "feature_importance": feature_importance,
            }

        except ImportError:
            logger.warning("shap not installed — using RF feature_importances_ as fallback")
            importance = dict(zip(
                self.feature_names[:X.shape[1]],
                rf.feature_importances_.tolist(),
            ))
            # Fallback bar plot
            sorted_imp = sorted(importance.items(), key=lambda x: x[1], reverse=True)[:15]
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.barh([x[0] for x in sorted_imp], [x[1] for x in sorted_imp], color="#5b8def")
            ax.set_title("Feature Importance (RF)", fontweight="bold")
            ax.invert_yaxis()
            plt.tight_layout()
            fig.savefig(self.output_dir / "rf_feature_importance.png", dpi=150, bbox_inches="tight")
            plt.close(fig)

            return {"rf_accuracy": accuracy, "feature_importance": importance}

    # ── Step 3c: Cluster Profiles ──

    def _step3c_cluster_profiles(self) -> None:
        """Generate radar charts and box plots per cluster."""
        mask = self.labels != -1
        n_features = min(len(self.feature_names), self.X.shape[1])
        df = pd.DataFrame(self.X[mask, :n_features], columns=self.feature_names[:n_features])
        df["cluster"] = self.labels[mask]

        n_clusters = df["cluster"].nunique()

        # Box plots for top features
        top_features = self.feature_names[:min(8, n_features)]
        if top_features:
            fig, axes = plt.subplots(2, 4, figsize=(20, 10))
            axes = axes.flatten()
            for idx, feat in enumerate(top_features):
                if idx >= len(axes):
                    break
                df.boxplot(column=feat, by="cluster", ax=axes[idx])
                axes[idx].set_title(feat, fontweight="bold")
            for idx in range(len(top_features), len(axes)):
                axes[idx].set_visible(False)
            plt.suptitle("Feature Distributions by Cluster", fontsize=14, fontweight="bold", y=1.02)
            plt.tight_layout()
            fig.savefig(self.output_dir / "cluster_boxplots.png", dpi=150, bbox_inches="tight")
            plt.close(fig)

        # Cluster size distribution
        fig, ax = plt.subplots(figsize=(8, 5))
        cluster_sizes = df["cluster"].value_counts().sort_index()
        cluster_sizes.plot(kind="bar", ax=ax, color="#5b8def", edgecolor="white")
        ax.set_title("Cluster Size Distribution", fontweight="bold")
        ax.set_xlabel("Cluster"); ax.set_ylabel("Count")
        for i, v in enumerate(cluster_sizes):
            ax.text(i, v + 5, str(v), ha="center", fontweight="bold")
        plt.tight_layout()
        fig.savefig(self.output_dir / "cluster_sizes.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

        logger.info("Cluster profile plots saved")

    # ── Step 4: Stability ──

    def _step4_stability(self, random_state: int, n_bootstraps: int = 30) -> Dict:
        """Bootstrap stability test using Adjusted Rand Index."""
        ari_scores = []
        n_samples = len(self.X)
        n_clusters = len(set(self.labels) - {-1})

        if n_clusters < 2:
            return {"ari_mean": 0, "ari_std": 0, "n_bootstraps": 0}

        for i in range(n_bootstraps):
            rng = np.random.RandomState(random_state + i)
            idx = rng.choice(n_samples, size=n_samples, replace=True)

            km = KMeans(n_clusters=n_clusters, n_init=5, random_state=random_state + i)
            boot_labels = km.fit_predict(self.X[idx])

            # Compare with original labels (reindexed)
            ari = adjusted_rand_score(self.labels[idx], boot_labels)
            ari_scores.append(float(ari))

        result = {
            "ari_mean": float(np.mean(ari_scores)),
            "ari_std": float(np.std(ari_scores)),
            "n_bootstraps": n_bootstraps,
            "stable": float(np.mean(ari_scores)) >= 0.7,
        }

        logger.info("Bootstrap stability: ARI=%.3f ± %.3f (stable=%s)", result["ari_mean"], result["ari_std"], result["stable"])
        return result

    # ── Utility ──

    @staticmethod
    def _serialize_report(report: Dict) -> Dict:
        """Make report JSON-serializable."""
        def _convert(obj):
            if isinstance(obj, (np.integer,)):
                return int(obj)
            if isinstance(obj, (np.floating,)):
                return float(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            if isinstance(obj, dict):
                return {k: _convert(v) for k, v in obj.items()}
            if isinstance(obj, list):
                return [_convert(v) for v in obj]
            return obj
        return _convert(report)
