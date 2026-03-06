"""
Segmentation Plus — Domain-Aware EDA & Feature Engineering
==========================================================
Generic + domain-specific exploratory analysis, data cleaning,
feature engineering, and preprocessing for clustering.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Tuple

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, LabelEncoder

from config import DomainConfig

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────
# EDA Engine
# ──────────────────────────────────────────────

class EDAEngine:
    """
    Runs generic and domain-specific exploratory data analysis.

    Parameters
    ----------
    df : pd.DataFrame
        Input data (raw or lightly cleaned).
    domain_config : DomainConfig
        Domain configuration driving domain-specific analyses.
    output_dir : Path
        Directory to save EDA charts.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        domain_config: DomainConfig,
        output_dir: Path,
    ) -> None:
        self.df = df.copy()
        self.config = domain_config
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def run_all(self) -> dict:
        """Run all EDA analyses and return summary statistics."""
        logger.info("Running EDA for domain: %s", self.config.display_name)
        results = {}

        results["summary_stats"] = self._summary_statistics()
        results["missing_analysis"] = self._missing_value_analysis()
        self._distribution_plots()
        self._correlation_heatmap()
        self._outlier_detection()
        self._run_domain_analyses()

        logger.info("EDA complete. Charts saved to %s", self.output_dir)
        return results

    def _summary_statistics(self) -> dict:
        """Compute and log summary statistics."""
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()
        stats = self.df[numeric_cols].describe().to_dict()
        logger.info("Summary stats computed for %d numeric columns", len(numeric_cols))
        return stats

    def _missing_value_analysis(self) -> dict:
        """Analyze missing values."""
        missing = self.df.isnull().sum()
        missing_pct = (missing / len(self.df) * 100).round(2)
        missing_report = {
            col: {"count": int(missing[col]), "pct": float(missing_pct[col])}
            for col in self.df.columns if missing[col] > 0
        }
        if missing_report:
            logger.warning("Missing values found in %d columns", len(missing_report))
        else:
            logger.info("No missing values detected")
        return missing_report

    def _distribution_plots(self) -> None:
        """Plot distributions for all numeric features."""
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()
        # Exclude ID-like columns
        plot_cols = [c for c in numeric_cols if "id" not in c.lower()]

        if not plot_cols:
            return

        n_cols = min(4, len(plot_cols))
        n_rows = (len(plot_cols) + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
        axes = np.atleast_2d(axes)

        for idx, col in enumerate(plot_cols):
            ax = axes[idx // n_cols, idx % n_cols]
            self.df[col].hist(bins=30, ax=ax, color="#5b8def", edgecolor="white", alpha=0.8)
            ax.set_title(col, fontsize=10, fontweight="bold")
            ax.tick_params(labelsize=8)

        # Hide unused axes
        for idx in range(len(plot_cols), n_rows * n_cols):
            axes[idx // n_cols, idx % n_cols].set_visible(False)

        plt.tight_layout()
        fig.savefig(self.output_dir / "distributions.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        logger.info("Distribution plots saved")

    def _correlation_heatmap(self) -> None:
        """Plot correlation heatmap for numeric features."""
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()
        numeric_cols = [c for c in numeric_cols if "id" not in c.lower()]

        if len(numeric_cols) < 2:
            return

        corr = self.df[numeric_cols].corr()
        fig, ax = plt.subplots(figsize=(max(10, len(numeric_cols) * 0.6), max(8, len(numeric_cols) * 0.5)))
        mask = np.triu(np.ones_like(corr, dtype=bool))
        sns.heatmap(
            corr, mask=mask, annot=len(numeric_cols) <= 15, fmt=".2f",
            cmap="RdBu_r", center=0, vmin=-1, vmax=1,
            square=True, linewidths=0.5, ax=ax,
            cbar_kws={"shrink": 0.8}
        )
        ax.set_title("Feature Correlation Matrix", fontsize=14, fontweight="bold", pad=15)
        plt.tight_layout()
        fig.savefig(self.output_dir / "correlation_heatmap.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        logger.info("Correlation heatmap saved")

    def _outlier_detection(self) -> None:
        """Box plots for outlier detection."""
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()
        plot_cols = [c for c in numeric_cols if "id" not in c.lower()]

        if not plot_cols:
            return

        n_cols = min(4, len(plot_cols))
        n_rows = (len(plot_cols) + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 3 * n_rows))
        axes = np.atleast_2d(axes)

        for idx, col in enumerate(plot_cols):
            ax = axes[idx // n_cols, idx % n_cols]
            self.df.boxplot(column=col, ax=ax)
            ax.set_title(col, fontsize=10, fontweight="bold")
            ax.tick_params(labelsize=8)

        for idx in range(len(plot_cols), n_rows * n_cols):
            axes[idx // n_cols, idx % n_cols].set_visible(False)

        plt.tight_layout()
        fig.savefig(self.output_dir / "outlier_boxplots.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        logger.info("Outlier box plots saved")

    def _run_domain_analyses(self) -> None:
        """Run domain-specific EDA analyses defined in config."""
        for analysis_name in self.config.eda_analyses:
            method_name = f"_eda_{analysis_name}"
            if hasattr(self, method_name):
                logger.info("Running domain analysis: %s", analysis_name)
                getattr(self, method_name)()
            else:
                logger.debug(
                    "Domain analysis '%s' not implemented — skipping. "
                    "Implement as method '%s' in EDAEngine.",
                    analysis_name, method_name,
                )

    # ── Domain-specific EDA methods (add more as needed) ──

    def _eda_product_cross_holding_matrix(self) -> None:
        """Cross-holding heatmap for product columns."""
        product_cols = self.config.features.get("products", [])
        bool_cols = [c for c in product_cols if c in self.df.columns and self.df[c].dtype in ["bool", "int64", "float64"]]
        if len(bool_cols) < 2:
            return

        cross = self.df[bool_cols].corr()
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(cross, annot=True, fmt=".2f", cmap="YlOrRd", ax=ax, square=True)
        ax.set_title("Product Cross-Holding Matrix", fontsize=13, fontweight="bold")
        plt.tight_layout()
        fig.savefig(self.output_dir / "product_cross_holdings.png", dpi=150, bbox_inches="tight")
        plt.close(fig)


# ──────────────────────────────────────────────
# Feature Engineering
# ──────────────────────────────────────────────

class FeatureEngineer:
    """
    Applies domain-specific and generic feature engineering.

    Parameters
    ----------
    domain_config : DomainConfig
        Domain configuration with FE rules.
    """

    def __init__(self, domain_config: DomainConfig) -> None:
        self.config = domain_config

    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply all feature engineering rules from domain config.

        Parameters
        ----------
        df : pd.DataFrame
            Cleaned DataFrame.

        Returns
        -------
        pd.DataFrame
            DataFrame with new engineered columns added.
        """
        df = df.copy()

        for rule in self.config.feature_engineering:
            try:
                # Evaluate the formula safely using DataFrame columns
                df[rule.name] = df.eval(rule.formula)
                logger.info("Engineered feature: '%s'", rule.name)

                # Apply binning if specified
                if rule.bins:
                    df[f"{rule.name}_bin"] = pd.qcut(
                        df[rule.name], q=len(rule.bins), labels=rule.bins, duplicates="drop"
                    )
                    logger.info("  → Binned into %d categories", len(rule.bins))

            except Exception as e:
                logger.warning(
                    "Failed to engineer feature '%s': %s. "
                    "Check that all columns in formula '%s' exist.",
                    rule.name, e, rule.formula,
                )

        return df


# ──────────────────────────────────────────────
# Preprocessor
# ──────────────────────────────────────────────

class Preprocessor:
    """
    Data cleaning, encoding, scaling, and PCA.

    Parameters
    ----------
    domain_config : DomainConfig
        Domain configuration.
    """

    def __init__(self, domain_config: DomainConfig) -> None:
        self.config = domain_config
        self.scaler: Optional[StandardScaler] = None
        self.pca: Optional[PCA] = None
        self.label_encoders: dict[str, LabelEncoder] = {}
        self.feature_columns: list[str] = []

    def fit_transform(
        self,
        df: pd.DataFrame,
        n_pca_components: Optional[int] = None,
        pca_variance_threshold: float = 0.85,
    ) -> Tuple[pd.DataFrame, np.ndarray, Optional[PCA]]:
        """
        Full preprocessing pipeline: clean → encode → scale → PCA.

        Parameters
        ----------
        df : pd.DataFrame
            Input DataFrame (after feature engineering).
        n_pca_components : int, optional
            Fixed number of PCA components. If None, auto-selects based on
            variance threshold.
        pca_variance_threshold : float
            Minimum cumulative variance to retain (default 85%).

        Returns
        -------
        tuple[pd.DataFrame, np.ndarray, PCA | None]
            - Processed DataFrame (with encodings, scaled features)
            - Feature matrix for clustering (scaled, optionally PCA-reduced)
            - Fitted PCA object (or None if not applied)
        """
        df = df.copy()

        # 1. Handle missing values
        df = self._handle_missing(df)

        # 2. Encode categoricals
        df = self._encode_categoricals(df)

        # 3. Select numeric features for clustering
        exclude = set(self.config.scaling_exclude)
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        self.feature_columns = [c for c in numeric_cols if c not in exclude]

        if not self.feature_columns:
            raise ValueError("No numeric feature columns found for clustering after exclusions.")

        logger.info("Features for clustering: %d columns", len(self.feature_columns))

        # 4. Handle outliers (winsorize)
        df = self._winsorize(df)

        # 5. Scale
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(df[self.feature_columns])

        logger.info("Scaling applied: StandardScaler (mean=0, std=1)")

        # 6. PCA (optional)
        if n_pca_components is not None or pca_variance_threshold < 1.0:
            X_scaled, self.pca = self._apply_pca(
                X_scaled, n_pca_components, pca_variance_threshold
            )

        return df, X_scaled, self.pca

    def _handle_missing(self, df: pd.DataFrame) -> pd.DataFrame:
        """Impute missing values: median for numeric, mode for categorical."""
        for col in df.columns:
            if df[col].isnull().sum() == 0:
                continue
            if df[col].dtype in [np.float64, np.int64, float, int]:
                median_val = df[col].median()
                df[col] = df[col].fillna(median_val)
                logger.info("Imputed '%s' with median: %.2f", col, median_val)
            else:
                mode_val = df[col].mode()
                if len(mode_val) > 0:
                    df[col] = df[col].fillna(mode_val.iloc[0])
                    logger.info("Imputed '%s' with mode: %s", col, mode_val.iloc[0])
        return df

    def _encode_categoricals(self, df: pd.DataFrame) -> pd.DataFrame:
        """Label-encode categorical columns."""
        cat_cols = [
            c for c in self.config.categorical_columns
            if c in df.columns and df[c].dtype == "object"
        ]
        for col in cat_cols:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            self.label_encoders[col] = le
            logger.info("Label-encoded '%s' (%d classes)", col, len(le.classes_))
        return df

    def _winsorize(self, df: pd.DataFrame, lower: float = 0.01, upper: float = 0.99) -> pd.DataFrame:
        """Winsorize numeric features at given percentiles."""
        for col in self.feature_columns:
            lo = df[col].quantile(lower)
            hi = df[col].quantile(upper)
            clipped = df[col].clip(lo, hi)
            n_clipped = (df[col] != clipped).sum()
            if n_clipped > 0:
                df[col] = clipped
                logger.debug("Winsorized '%s': %d values clipped", col, n_clipped)
        return df

    def _apply_pca(
        self,
        X: np.ndarray,
        n_components: Optional[int],
        variance_threshold: float,
    ) -> Tuple[np.ndarray, PCA]:
        """Apply PCA for dimensionality reduction."""
        if n_components is None:
            # Auto-select: find n_components that explain ≥ threshold variance
            pca_full = PCA().fit(X)
            cumvar = np.cumsum(pca_full.explained_variance_ratio_)
            n_components = int(np.searchsorted(cumvar, variance_threshold) + 1)
            n_components = max(2, min(n_components, X.shape[1]))
            logger.info(
                "Auto-selected %d PCA components (%.1f%% variance)",
                n_components, cumvar[n_components - 1] * 100,
            )

        pca = PCA(n_components=n_components)
        X_pca = pca.fit_transform(X)

        logger.info(
            "PCA: %d → %d components (%.1f%% variance explained)",
            X.shape[1], n_components,
            sum(pca.explained_variance_ratio_) * 100,
        )

        return X_pca, pca
