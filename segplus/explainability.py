"""Explainability: SHAP importance, PCA loadings, inertia curve, cluster profiles."""
from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import silhouette_score

from .types import ExplainabilityReport

log = logging.getLogger("segplus.explainability")


def compute_shap_importance(
    X: np.ndarray,
    labels: np.ndarray,
    feature_names: list[str],
    random_state: int = 42,
) -> dict[str, float]:
    """
    Feature importance using three-tier fallback:
    1. TreeSHAP (if shap is installed)
    2. RF feature_importances_
    3. Permutation importance
    """
    valid_mask = labels != -1
    Xv, lv = X[valid_mask], labels[valid_mask]

    if len(set(lv)) < 2:
        return {f: 0.0 for f in feature_names}

    # Train RF classifier on cluster labels
    rf = RandomForestClassifier(
        n_estimators=100, max_depth=10, random_state=random_state, n_jobs=-1
    )
    rf.fit(Xv, lv)

    # Tier 1: Try TreeSHAP
    try:
        import shap
        explainer = shap.TreeExplainer(rf)
        sample_size = min(500, len(Xv))
        rng = np.random.default_rng(random_state)
        idx = rng.choice(len(Xv), size=sample_size, replace=False)
        shap_values = explainer.shap_values(Xv[idx])

        if isinstance(shap_values, list):
            mean_abs = np.mean([np.abs(sv).mean(axis=0) for sv in shap_values], axis=0)
        else:
            mean_abs = np.abs(shap_values).mean(axis=0)

        importances = {feature_names[i]: round(float(mean_abs[i]), 5) for i in range(len(feature_names))}
        log.info("SHAP importance computed via TreeSHAP")
        return dict(sorted(importances.items(), key=lambda x: -x[1]))

    except ImportError:
        log.info("shap not installed, falling back to RF feature_importances_")
    except Exception as e:
        log.warning("TreeSHAP failed (%s), falling back to RF feature_importances_", e)

    # Tier 2: RF feature importances
    try:
        imp = rf.feature_importances_
        importances = {feature_names[i]: round(float(imp[i]), 5) for i in range(len(feature_names))}
        log.info("Feature importance computed via RF feature_importances_")
        return dict(sorted(importances.items(), key=lambda x: -x[1]))
    except Exception as e:
        log.warning("RF importances failed (%s), falling back to permutation", e)

    # Tier 3: Permutation importance
    return _compute_permutation_importance(Xv, lv, feature_names, random_state=random_state)


def to_importance_percentages(importances: dict[str, float]) -> dict[str, float]:
    """Convert absolute importance scores to percentage contribution."""
    total = float(sum(max(v, 0.0) for v in importances.values()))
    if total <= 0:
        return {k: 0.0 for k in importances.keys()}
    return {k: round((max(v, 0.0) / total) * 100.0, 4) for k, v in importances.items()}


def _compute_permutation_importance(
    X: np.ndarray,
    labels: np.ndarray,
    feature_names: list[str],
    n_repeats: int = 10,
    random_state: int = 0,
) -> dict[str, float]:
    """Permutation importance: silhouette drop when feature is shuffled."""
    rng = np.random.default_rng(random_state)
    baseline = silhouette_score(X, labels, sample_size=min(1000, len(X)))
    importances: dict[str, float] = {}

    for i, fname in enumerate(feature_names):
        drops = []
        for _ in range(n_repeats):
            Xp = X.copy()
            rng.shuffle(Xp[:, i])
            s = silhouette_score(Xp, labels, sample_size=min(1000, len(X)))
            drops.append(baseline - s)
        importances[fname] = round(float(np.mean(drops)), 5)

    log.info("Feature importance computed via permutation importance")
    return dict(sorted(importances.items(), key=lambda x: -x[1]))


def compute_pca_loadings(
    pca: PCA,
    feature_names: list[str],
) -> pd.DataFrame:
    """PCA component loadings as a DataFrame."""
    n_components = pca.n_components_
    return pd.DataFrame(
        np.abs(pca.components_),
        columns=feature_names,
        index=[f"PC{i+1}" for i in range(n_components)],
    )


def compute_inertia_curve(
    X: np.ndarray,
    k_range: tuple[int, int],
    random_state: int = 42,
) -> dict[int, float]:
    """Elbow curve: k -> inertia for K-Means."""
    curve: dict[int, float] = {}
    for k in range(k_range[0], k_range[1] + 1):
        km = KMeans(n_clusters=k, n_init=5, random_state=random_state)
        km.fit(X)
        curve[k] = float(km.inertia_)
    return curve


def build_pc_feature_map(
    pca_loadings: pd.DataFrame,
    top_n: int = 5,
) -> dict[str, list[str]]:
    """Map each principal component to its top contributing original features."""
    pc_map: dict[str, list[str]] = {}
    for pc in pca_loadings.index:
        top_features = (
            pca_loadings.loc[pc]
            .sort_values(ascending=False)
            .head(top_n)
            .index
            .tolist()
        )
        pc_map[str(pc)] = top_features
    return pc_map


def build_ordered_feature_drivers(
    feature_importances: dict[str, float],
    pca_loadings: pd.DataFrame,
    pca_variance_ratio: list[float],
    inertia_curve: dict[int, float] | None,
) -> pd.DataFrame:
    """
    Build ordered feature drivers of convergence by combining:
      - SHAP/RF feature importance
      - Weighted PCA loading strength
      - Global inertia elbow strength (reported as context)
    """
    shap_series = pd.Series(feature_importances, dtype=float)

    # Weighted PCA contribution score per original feature
    if len(pca_variance_ratio) > 0:
        comp_weights = np.array(pca_variance_ratio[: len(pca_loadings.index)], dtype=float)
    else:
        comp_weights = np.ones(len(pca_loadings.index), dtype=float)
    comp_weights = comp_weights / max(comp_weights.sum(), 1e-12)
    pca_score = (pca_loadings.T * comp_weights).sum(axis=1)

    df = pd.DataFrame({
        "feature": sorted(set(shap_series.index).union(set(pca_score.index))),
    })
    df["shap_importance"] = df["feature"].map(shap_series).fillna(0.0).astype(float)
    df["pca_weighted_loading"] = df["feature"].map(pca_score).fillna(0.0).astype(float)

    # Rank normalize to [0,1], then blend.
    df["shap_rank"] = df["shap_importance"].rank(ascending=False, method="average")
    df["pca_rank"] = df["pca_weighted_loading"].rank(ascending=False, method="average")
    n = max(len(df), 1)
    df["shap_rank_norm"] = 1.0 - (df["shap_rank"] - 1.0) / max(n - 1, 1)
    df["pca_rank_norm"] = 1.0 - (df["pca_rank"] - 1.0) / max(n - 1, 1)

    # Inertia is cluster-level evidence, not feature-specific. Keep as metadata/context column.
    elbow_strength = 0.0
    if inertia_curve and len(inertia_curve) >= 3:
        ks = sorted(inertia_curve.keys())
        vals = np.array([inertia_curve[k] for k in ks], dtype=float)
        second_diff = np.diff(vals, n=2)
        if len(second_diff) > 0:
            elbow_strength = float(np.max(np.abs(second_diff)))
    df["inertia_elbow_strength"] = elbow_strength

    df["convergence_score"] = (
        0.6 * df["shap_rank_norm"] +
        0.4 * df["pca_rank_norm"]
    )
    df = df.sort_values("convergence_score", ascending=False).reset_index(drop=True)
    return df


def build_cluster_profiles(
    df: pd.DataFrame,
    labels: np.ndarray,
    feature_names: list[str],
) -> pd.DataFrame:
    """Cluster x feature mean table on raw/unscaled data."""
    profiled = df.copy()
    profiled["_cluster"] = labels
    # Use only numeric columns that exist in the dataframe
    num_cols = [c for c in feature_names if c in profiled.columns and pd.api.types.is_numeric_dtype(profiled[c])]
    if not num_cols:
        num_cols = [c for c in profiled.select_dtypes(include="number").columns if c != "_cluster"]
    profiles = profiled[profiled["_cluster"] != -1].groupby("_cluster")[num_cols].mean().round(2)
    return profiles


def build_explainability_report(
    X: np.ndarray,
    labels: np.ndarray,
    feature_names: list[str],
    df_raw: pd.DataFrame,
    pca: Optional[PCA],
    k_range: tuple[int, int],
    random_state: int = 42,
    n_top: int = 10,
    X_original: np.ndarray | None = None,
    original_feature_names: list[str] | None = None,
) -> ExplainabilityReport:
    """Orchestrate all explainability analyses into one report."""
    log.info("Computing feature importances...")
    shap_X = X_original if X_original is not None else X
    shap_feature_names = (
        original_feature_names
        if original_feature_names is not None and len(original_feature_names) == shap_X.shape[1]
        else feature_names
    )
    importances = compute_shap_importance(shap_X, labels, shap_feature_names, random_state)
    importance_pct = to_importance_percentages(importances)
    top_features = list(importances.keys())[:n_top]

    # PCA loadings are defined on the original pre-PCA feature space.
    # When clustering uses PCA output, `feature_names` may be ["PC1", ...],
    # so derive a compatible name list for loadings to avoid shape mismatch.
    n_pca_input_features = int(pca.components_.shape[1]) if pca is not None else len(feature_names)
    numeric_raw_cols = [c for c in df_raw.columns if pd.api.types.is_numeric_dtype(df_raw[c])]
    if len(feature_names) == n_pca_input_features:
        loading_feature_names = feature_names
    elif len(numeric_raw_cols) == n_pca_input_features:
        loading_feature_names = numeric_raw_cols
    else:
        loading_feature_names = [f"feature_{i+1}" for i in range(n_pca_input_features)]

    log.info("Computing PCA loadings...")
    if pca is not None:
        pca_loadings = compute_pca_loadings(pca, loading_feature_names)
        pca_var = pca.explained_variance_ratio_.tolist()
    else:
        # Fit a quick PCA for loadings analysis
        n_comp = min(5, len(feature_names))
        pca_temp = PCA(n_components=n_comp, random_state=random_state)
        pca_temp.fit(X)
        pca_loadings = compute_pca_loadings(pca_temp, loading_feature_names)
        pca_var = pca_temp.explained_variance_ratio_.tolist()

    log.info("Computing inertia curve...")
    inertia = compute_inertia_curve(X, k_range, random_state)

    log.info("Computing cluster profiles...")
    profiles = build_cluster_profiles(df_raw, labels, feature_names)

    log.info("Building ordered convergence drivers...")
    ordered_drivers = build_ordered_feature_drivers(
        feature_importances=importances,
        pca_loadings=pca_loadings,
        pca_variance_ratio=pca_var,
        inertia_curve=inertia,
    )
    pc_map = build_pc_feature_map(pca_loadings, top_n=5)
    top_features = ordered_drivers["feature"].head(n_top).tolist()

    return ExplainabilityReport(
        top_features=top_features,
        feature_importances=importances,
        feature_importance_pct=importance_pct,
        pca_loadings=pca_loadings,
        cluster_profiles=profiles,
        pca_variance_ratio=pca_var,
        inertia_curve=inertia,
        ordered_feature_drivers=ordered_drivers,
        pc_feature_map=pc_map,
    )
