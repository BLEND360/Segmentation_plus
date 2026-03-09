"""Shared dataclasses and type definitions for the segplus pipeline."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Optional

import numpy as np
import pandas as pd


# ── Data Layer ───────────────────────────────────────────────────────────────

@dataclass
class ColumnMeta:
    name: str
    dtype: Literal["numeric", "categorical", "datetime", "text", "boolean"]
    null_rate: float
    n_unique: int


@dataclass
class DataSchema:
    n_rows: int
    n_cols: int
    columns: list[ColumnMeta] = field(default_factory=list)

    @property
    def numeric_cols(self) -> list[str]:
        return [c.name for c in self.columns if c.dtype == "numeric"]

    @property
    def categorical_cols(self) -> list[str]:
        return [c.name for c in self.columns if c.dtype == "categorical"]


@dataclass
class DataQualityReport:
    total_rows: int
    total_columns: int
    missing_values: dict[str, int] = field(default_factory=dict)
    missing_pct: dict[str, float] = field(default_factory=dict)
    duplicate_rows: int = 0
    column_types: dict[str, str] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    passed: bool = True

    def summary(self) -> str:
        lines = [
            f"Data Quality Report",
            f"  Rows: {self.total_rows:,}  |  Columns: {self.total_columns}",
            f"  Duplicates: {self.duplicate_rows:,}",
        ]
        if self.missing_values:
            top = sorted(self.missing_pct.items(), key=lambda x: -x[1])[:5]
            lines.append("  Top missing:")
            for col, pct in top:
                if pct > 0:
                    lines.append(f"    {col}: {pct:.1%}")
        if self.warnings:
            lines.append(f"  Warnings ({len(self.warnings)}):")
            for w in self.warnings[:5]:
                lines.append(f"    - {w}")
        if self.errors:
            lines.append(f"  Errors ({len(self.errors)}):")
            for e in self.errors[:5]:
                lines.append(f"    - {e}")
        lines.append(f"  Status: {'PASSED' if self.passed else 'FAILED'}")
        return "\n".join(lines)


# ── Feature Engineering Layer ────────────────────────────────────────────────

@dataclass
class FeatureEngineeringResult:
    df_original: pd.DataFrame
    df_engineered: pd.DataFrame
    X_scaled: np.ndarray
    feature_names: list[str]
    pca: object | None = None  # Optional[PCA]
    scaler: object | None = None  # Optional[StandardScaler]
    label_encoders: dict = field(default_factory=dict)


# ── Clustering Layer ─────────────────────────────────────────────────────────

@dataclass
class ClusteringConfig:
    k: int = 3
    kmeans_init: str = "k-means++"
    kmeans_n_init: int = 10
    kmeans_max_iter: int = 300
    dbscan_eps: float = 0.5
    dbscan_min_samples: int = 5
    gmm_covariance_type: str = "full"
    gmm_n_init: int = 5
    random_state: int = 42
    feature_subset_indices: list[int] | None = None


@dataclass
class ClusterRunResult:
    algorithm: str
    labels: np.ndarray
    n_clusters: int
    model: object
    probabilities: np.ndarray | None = None
    extra: dict = field(default_factory=dict)

    @property
    def is_valid(self) -> bool:
        unique = set(self.labels)
        unique.discard(-1)
        return len(unique) >= 2


# ── Evaluation Layer ─────────────────────────────────────────────────────────

@dataclass
class EvaluationResult:
    algorithm: str
    labels: np.ndarray
    n_clusters: int
    silhouette: float
    davies_bouldin: float
    calinski_harabasz: float
    passes: bool
    all_scores: dict[str, dict[str, float]] = field(default_factory=dict)
    model: object = None


@dataclass
class StabilityResult:
    ari_mean: float
    ari_std: float
    n_bootstraps: int
    stable: bool


# ── Explainability Layer ─────────────────────────────────────────────────────

@dataclass
class ExplainabilityReport:
    top_features: list[str]
    feature_importances: dict[str, float]
    pca_loadings: pd.DataFrame
    cluster_profiles: pd.DataFrame
    pca_variance_ratio: list[float]
    inertia_curve: dict[int, float] | None = None
    feature_importance_pct: dict[str, float] = field(default_factory=dict)
    ordered_feature_drivers: pd.DataFrame | None = None
    pc_feature_map: dict[str, list[str]] = field(default_factory=dict)


# ── Persona Layer ────────────────────────────────────────────────────────────

@dataclass
class PersonaResult:
    cluster_id: int | str
    persona_name: str
    archetype: str
    description: str
    key_traits: list[str]
    business_recommendations: list[str]
    cluster_size: int
    cluster_pct: float
    top_features: dict[str, float] = field(default_factory=dict)
    naming_rationale: str = ""
    categorization_basis: list[str] = field(default_factory=list)
    profile_descriptor: str = ""  # Data-driven descriptor (e.g. "High Credit Score & Low Risk")


@dataclass
class BusinessGrounding:
    executive_summary: str
    cluster_priorities: list[dict] = field(default_factory=list)
    quick_wins: list[str] = field(default_factory=list)
    cluster_actions: list[dict] = field(default_factory=list)


# ── Experiment Tracking ──────────────────────────────────────────────────────

@dataclass
class ExperimentRecord:
    iteration: int
    timestamp: str
    config_k: int
    config_eps: float
    config_gmm_cov: str
    best_algorithm: str
    n_clusters: int
    silhouette: float
    davies_bouldin: float
    calinski_harabasz: float
    passed: bool
    reconfiguration_strategy: str | None = None


# ── Pipeline Result ──────────────────────────────────────────────────────────

@dataclass
class PipelineResult:
    data_quality: DataQualityReport
    feature_engineering: FeatureEngineeringResult
    best_evaluation: EvaluationResult
    stability: StabilityResult
    explainability: ExplainabilityReport
    personas: list[PersonaResult]
    grounding: BusinessGrounding
    experiment_log: list[ExperimentRecord]
