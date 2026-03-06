"""Feature engineering: domain rules, imputation, encoding, scaling, PCA."""
from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, StandardScaler

from .config import DomainConfig, PipelineConfig
from .types import DataSchema, FeatureEngineeringResult

log = logging.getLogger("segplus.feature_engineering")


class FeatureEngineer:
    """Full feature engineering pipeline: FE rules -> impute -> encode -> winsorize -> scale -> PCA."""

    def __init__(self, domain_config: DomainConfig, pipeline_config: PipelineConfig):
        self.domain = domain_config
        self.config = pipeline_config
        self._scaler: StandardScaler | None = None
        self._pca: PCA | None = None
        self._imputer: SimpleImputer | None = None
        self._label_encoders: dict[str, LabelEncoder] = {}
        self._feature_cols: list[str] = []

    def run(self, df: pd.DataFrame) -> FeatureEngineeringResult:
        """Execute the full FE pipeline and return results."""
        df_original = df.copy()

        # Step 1: Apply domain-specific FE rules
        df = self._apply_domain_rules(df)

        # Step 2: Drop excluded columns
        drop_cols = [c for c in self.config.exclude_cols if c in df.columns]
        df_work = df.drop(columns=drop_cols, errors="ignore")

        # Step 3: Identify column types
        num_cols = [c for c in df_work.columns if pd.api.types.is_numeric_dtype(df_work[c])]
        cat_cols = [c for c in self.domain.categorical_columns if c in df_work.columns]

        # Step 4: Encode categoricals
        for col in cat_cols:
            le = LabelEncoder()
            df_work[col] = le.fit_transform(df_work[col].astype(str))
            self._label_encoders[col] = le
            if col not in num_cols:
                num_cols.append(col)

        # Step 5: Select feature columns (numeric + encoded categoricals)
        self._feature_cols = [c for c in num_cols if c in df_work.columns]
        X = df_work[self._feature_cols].copy()

        # Step 6: Impute missing values
        self._imputer = SimpleImputer(strategy=self.config.imputation_strategy)
        X_arr = self._imputer.fit_transform(X)
        X = pd.DataFrame(X_arr, columns=self._feature_cols, index=df_work.index)

        # Step 7: Winsorize outliers
        X = self._winsorize(X, self._feature_cols)

        df_engineered = X.copy()

        # Step 8: Scale
        self._scaler = StandardScaler()
        X_scaled = self._scaler.fit_transform(X)

        # Step 9: PCA (optional)
        pca = None
        if self.config.pca_variance_threshold < 1.0:
            pca, X_scaled = self._apply_pca(X_scaled)
            pca_names = [f"PC{i+1}" for i in range(X_scaled.shape[1])]
            log.info(
                "PCA: %d components explain %.1f%% variance",
                X_scaled.shape[1],
                sum(pca.explained_variance_ratio_[:X_scaled.shape[1]]) * 100,
            )
        else:
            pca_names = None

        feature_names = pca_names if pca_names else self._feature_cols

        log.info("Feature engineering complete: %d features -> %d-dim output", len(self._feature_cols), X_scaled.shape[1])

        return FeatureEngineeringResult(
            df_original=df_original,
            df_engineered=df_engineered,
            X_scaled=X_scaled,
            feature_names=feature_names if pca_names else list(self._feature_cols),
            pca=pca,
            scaler=self._scaler,
            label_encoders=self._label_encoders,
        )

    def _apply_domain_rules(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply feature engineering rules from domain config using df.eval()."""
        df = df.copy()
        for rule in self.domain.feature_engineering:
            try:
                result = df.eval(rule.formula)
                if rule.bins:
                    n_bins = len(rule.bins)
                    df[rule.name] = pd.qcut(result, q=n_bins, labels=rule.bins, duplicates="drop")
                else:
                    df[rule.name] = result
                log.info("  FE rule applied: %s", rule.name)
            except Exception as e:
                log.warning("  FE rule '%s' failed: %s", rule.name, e)
        return df

    def _winsorize(self, X: pd.DataFrame, num_cols: list[str]) -> pd.DataFrame:
        """Clip outliers to [lower, upper] percentiles."""
        for col in num_cols:
            if col in X.columns:
                lo = X[col].quantile(self.config.winsorize_lower)
                hi = X[col].quantile(self.config.winsorize_upper)
                X[col] = X[col].clip(lo, hi)
        return X

    def _apply_pca(self, X_scaled: np.ndarray) -> tuple[PCA, np.ndarray]:
        """Apply PCA with automatic component selection."""
        pca = PCA(random_state=self.config.random_state)
        pca.fit(X_scaled)

        cumvar = np.cumsum(pca.explained_variance_ratio_)
        n_components = int(np.searchsorted(cumvar, self.config.pca_variance_threshold) + 1)
        n_components = max(2, min(n_components, X_scaled.shape[1]))

        pca_final = PCA(n_components=n_components, random_state=self.config.random_state)
        X_pca = pca_final.fit_transform(X_scaled)
        self._pca = pca_final
        return pca_final, X_pca

    @property
    def original_feature_names(self) -> list[str]:
        """Original feature column names before PCA."""
        return list(self._feature_cols)
