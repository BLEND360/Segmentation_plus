"""Visualization: all matplotlib/seaborn charts for the pipeline."""
from __future__ import annotations

import logging
from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier

from .experiment_log import ExperimentLog
from .types import ExplainabilityReport, EvaluationResult, PersonaResult

log = logging.getLogger("segplus.visualization")

PALETTE = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B2", "#937860", "#DA8BC3"]


def _setup_style() -> None:
    sns.set_theme(style="whitegrid", palette=PALETTE)
    plt.rcParams.update({"figure.dpi": 130, "axes.titlesize": 13, "axes.labelsize": 11})


def plot_cluster_scatter_2d(
    X: np.ndarray,
    labels: np.ndarray,
    output_dir: Path,
    eval_result: EvaluationResult | None = None,
) -> None:
    """PCA 2D scatter plot of clusters with optional silhouette comparison."""
    _setup_style()
    pca2 = PCA(n_components=2, random_state=42)
    X_2d = pca2.fit_transform(X)
    var_explained = pca2.explained_variance_ratio_

    n_subplots = 2 if eval_result else 1
    fig, axes = plt.subplots(1, n_subplots, figsize=(8 * n_subplots, 6))
    if n_subplots == 1:
        axes = [axes]

    # Cluster scatter
    unique_labels = sorted(set(labels))
    colors = PALETTE[:len(unique_labels)]
    for lbl, col in zip(unique_labels, colors):
        mask = labels == lbl
        name = f"Cluster {lbl}" if lbl != -1 else "Noise"
        axes[0].scatter(X_2d[mask, 0], X_2d[mask, 1], c=col, alpha=0.55, s=18,
                        label=name, edgecolors="none")

    algo = eval_result.algorithm.upper() if eval_result else "Clustering"
    sil = f" (sil={eval_result.silhouette:.3f})" if eval_result else ""
    axes[0].set_title(f"Clusters - {algo}{sil}")
    axes[0].set_xlabel(f"PC1 ({var_explained[0] * 100:.1f}% var)")
    axes[0].set_ylabel(f"PC2 ({var_explained[1] * 100:.1f}% var)")
    axes[0].legend(framealpha=0.8)

    # Silhouette comparison bar chart
    if eval_result and eval_result.all_scores:
        algos = list(eval_result.all_scores.keys())
        sil_scores = [eval_result.all_scores[a].get("silhouette", 0) for a in algos]
        bar_colors = [PALETTE[2] if a == eval_result.algorithm else PALETTE[0] for a in algos]
        bars = axes[1].bar(algos, sil_scores, color=bar_colors, width=0.4, edgecolor="white")
        axes[1].set_title("Silhouette Score by Algorithm")
        axes[1].set_ylabel("Silhouette Score")
        for bar, val in zip(bars, sil_scores):
            axes[1].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                         f"{val:.3f}", ha="center", va="bottom", fontsize=10)

    plt.tight_layout()
    fig.savefig(output_dir / "cluster_scatter.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved: cluster_scatter.png")


def plot_shap_importance(
    importances: dict[str, float],
    output_dir: Path,
    top_n: int = 15,
    importance_pct: dict[str, float] | None = None,
) -> None:
    """Horizontal bar chart of feature importances with percentage labels."""
    _setup_style()
    names = list(importances.keys())[:top_n]
    scores = list(importances.values())[:top_n]

    # Compute percentages from raw scores if not provided
    if importance_pct:
        pcts = [importance_pct.get(n, 0.0) for n in names]
    else:
        total = sum(abs(s) for s in scores) or 1.0
        pcts = [(abs(s) / total) * 100 for s in scores]

    fig, ax = plt.subplots(figsize=(10, max(4, len(names) * 0.45)))
    colors = [PALETTE[2] if s > 0 else PALETTE[3] for s in scores]
    bars = ax.barh(names[::-1], scores[::-1], color=colors[::-1], edgecolor="white")

    # Add percentage labels on each bar
    for bar, pct in zip(bars, pcts[::-1]):
        width = bar.get_width()
        label_x = width + (ax.get_xlim()[1] - ax.get_xlim()[0]) * 0.01
        ax.text(
            label_x, bar.get_y() + bar.get_height() / 2,
            f"{pct:.1f}%",
            va="center", ha="left", fontsize=9, fontweight="bold", color="#333333",
        )

    ax.set_title("Feature Importance (Cluster Separation Drivers)")
    ax.set_xlabel("Importance Score")
    ax.axvline(0, color="gray", linewidth=0.8)
    # Extend x-axis slightly for percentage labels
    xlim = ax.get_xlim()
    ax.set_xlim(xlim[0], xlim[1] * 1.15)

    plt.tight_layout()
    fig.savefig(output_dir / "feature_importance.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved: feature_importance.png")


def plot_shap_summary(
    X: np.ndarray,
    labels: np.ndarray,
    feature_names: list[str],
    output_dir: Path,
    random_state: int = 42,
    max_display: int = 15,
) -> None:
    """SHAP summary plots (beeswarm + bar) with robust multiclass handling."""
    valid_mask = labels != -1
    Xv, lv = X[valid_mask], labels[valid_mask]
    if len(set(lv)) < 2:
        log.warning("Skipping SHAP summary: need at least 2 clusters after excluding noise.")
        return

    try:
        import shap
    except ImportError:
        log.warning("Skipping SHAP summary: shap is not installed.")
        return

    try:
        model = RandomForestClassifier(
            n_estimators=120, max_depth=12, random_state=random_state, n_jobs=-1
        )
        model.fit(Xv, lv)

        sample_n = min(1000, len(Xv))
        rng = np.random.default_rng(random_state)
        idx = rng.choice(len(Xv), size=sample_n, replace=False)
        X_sample = Xv[idx]

        explainer = shap.TreeExplainer(model)
        raw_shap = explainer.shap_values(X_sample)

        # Normalize shap outputs to 2D (n_samples, n_features) for clean beeswarm plots.
        major_class = pd.Series(lv).value_counts().index[0]
        if isinstance(raw_shap, list):
            class_idx = int(major_class) if int(major_class) < len(raw_shap) else 0
            shap_2d = np.array(raw_shap[class_idx], dtype=float)
        else:
            arr = np.array(raw_shap)
            if arr.ndim == 2:
                shap_2d = arr
            elif arr.ndim == 3:
                # Common shape: (n_samples, n_features, n_classes)
                if arr.shape[0] == X_sample.shape[0] and arr.shape[1] == X_sample.shape[1]:
                    class_idx = int(major_class) if int(major_class) < arr.shape[2] else 0
                    shap_2d = arr[:, :, class_idx]
                # Alternate shape: (n_classes, n_samples, n_features)
                elif arr.shape[1] == X_sample.shape[0] and arr.shape[2] == X_sample.shape[1]:
                    class_idx = int(major_class) if int(major_class) < arr.shape[0] else 0
                    shap_2d = arr[class_idx, :, :]
                else:
                    # Last-resort aggregation across class axis.
                    shap_2d = np.mean(arr, axis=-1)
                    if shap_2d.ndim != 2:
                        shap_2d = shap_2d.reshape(X_sample.shape[0], X_sample.shape[1])
            else:
                log.warning("Unexpected SHAP output shape %s; skipping SHAP summary.", arr.shape)
                return

        # Human-readable labels (no hardcoding; generated from column names)
        pretty_feature_names = [f.replace("_", " ").title() for f in feature_names]
        display_n = min(max_display, len(pretty_feature_names))
        plot_h = max(6, int(display_n * 0.45))

        # Build interpretation table: importance % + directionality hint.
        mean_abs = np.mean(np.abs(shap_2d), axis=0)
        total_abs = float(np.sum(mean_abs)) if np.sum(mean_abs) > 0 else 1.0
        rows = []
        for i, fname in enumerate(feature_names):
            xi = X_sample[:, i]
            si = shap_2d[:, i]
            if np.std(xi) > 1e-12 and np.std(si) > 1e-12:
                corr = float(np.corrcoef(xi, si)[0, 1])
            else:
                corr = 0.0
            if corr > 0.1:
                effect = "Higher value tends to increase model score"
            elif corr < -0.1:
                effect = "Higher value tends to decrease model score"
            else:
                effect = "Mixed / nonlinear effect"
            rows.append({
                "feature": fname,
                "feature_pretty": pretty_feature_names[i],
                "mean_abs_shap": float(mean_abs[i]),
                "importance_pct": float((mean_abs[i] / total_abs) * 100.0),
                "value_shap_corr": corr,
                "effect_hint": effect,
            })
        interp_df = pd.DataFrame(rows).sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)
        interp_df.to_csv(output_dir / "shap_interpretation.csv", index=False)

        # Beeswarm summary
        plt.figure(figsize=(12, plot_h))
        shap.summary_plot(
            shap_2d,
            features=X_sample,
            feature_names=pretty_feature_names,
            max_display=display_n,
            show=False,
            plot_size=(12, plot_h),
        )
        plt.tight_layout()
        plt.savefig(output_dir / "shap_summary.png", dpi=180, bbox_inches="tight")
        plt.close()

        # Custom bar summary with percentage labels (replaces shap's default bar chart)
        top_interp = interp_df.head(display_n).copy()
        fig_bar, ax_bar = plt.subplots(figsize=(11, max(5, int(display_n * 0.45))))
        bar_names = top_interp["feature_pretty"].tolist()[::-1]
        bar_pcts = top_interp["importance_pct"].tolist()[::-1]
        bar_colors = [PALETTE[2]] * len(bar_names)
        bars = ax_bar.barh(bar_names, bar_pcts, color=bar_colors, edgecolor="white", height=0.6)

        for bar, pct in zip(bars, bar_pcts):
            label_x = bar.get_width() + 0.3
            ax_bar.text(
                label_x, bar.get_y() + bar.get_height() / 2,
                f"{pct:.1f}%",
                va="center", ha="left", fontsize=10, fontweight="bold", color="#333333",
            )

        ax_bar.set_xlabel("Importance (%)", fontsize=11)
        ax_bar.set_title("SHAP Feature Importance (%)", fontsize=13, fontweight="bold")
        xlim = ax_bar.get_xlim()
        ax_bar.set_xlim(xlim[0], xlim[1] * 1.12)
        plt.tight_layout()
        fig_bar.savefig(output_dir / "shap_summary_bar.png", dpi=180, bbox_inches="tight")
        plt.close(fig_bar)

        log.info("Saved: shap_summary.png, shap_summary_bar.png, shap_interpretation.csv")
    except Exception as e:
        log.warning("SHAP summary plot failed: %s", e)


def plot_pca_loadings_heatmap(
    loadings: pd.DataFrame,
    output_dir: Path,
) -> None:
    """Heatmap of PCA component loadings."""
    _setup_style()
    fig, ax = plt.subplots(figsize=(max(10, len(loadings.columns) * 0.6), max(4, len(loadings) * 0.8)))
    sns.heatmap(loadings, ax=ax, cmap="YlOrRd", annot=True, fmt=".2f",
                linewidths=0.5, cbar_kws={"label": "|Loading|"})
    ax.set_title("PCA Component Loadings")
    ax.set_ylabel("Component")

    plt.tight_layout()
    fig.savefig(output_dir / "pca_loadings.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved: pca_loadings.png")


def plot_cluster_profiles_heatmap(
    profiles: pd.DataFrame,
    top_features: list[str],
    output_dir: Path,
) -> None:
    """Z-scored cluster profile heatmap for top features."""
    _setup_style()
    cols = [c for c in top_features if c in profiles.columns][:10]
    if not cols:
        return

    sub = profiles[cols]
    z_scored = (sub - sub.mean()) / sub.std().replace(0, 1)

    fig, ax = plt.subplots(figsize=(max(10, len(cols) * 0.8), max(4, len(z_scored) * 0.8)))
    sns.heatmap(z_scored.T, ax=ax, cmap="RdBu_r", center=0, annot=True, fmt=".2f",
                linewidths=0.5, cbar_kws={"label": "Z-score vs. mean"})
    ax.set_title("Cluster Profiles - Top Driving Features")
    ax.set_xlabel("Cluster ID")

    plt.tight_layout()
    fig.savefig(output_dir / "cluster_profiles.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved: cluster_profiles.png")


def plot_cluster_sizes(
    labels: np.ndarray,
    personas: list[PersonaResult] | None,
    output_dir: Path,
) -> None:
    """Pie chart of cluster size distribution."""
    _setup_style()
    unique = sorted(c for c in set(labels) if c != -1)

    fig, ax = plt.subplots(figsize=(8, 6))
    sizes = [int((labels == c).sum()) for c in unique]

    if personas and len(personas) == len(unique):
        pie_labels = [f"{p.persona_name}\n{p.archetype[:20]}" for p in personas]
    else:
        pie_labels = [f"Cluster {c}" for c in unique]

    colors = PALETTE[:len(unique)]
    wedges, texts, autotexts = ax.pie(
        sizes, labels=pie_labels, colors=colors,
        autopct="%1.1f%%", startangle=140,
        wedgeprops={"edgecolor": "white", "linewidth": 1.5},
    )
    for at in autotexts:
        at.set_fontsize(10)
    ax.set_title("Cluster Size Distribution", fontsize=13, pad=15)

    plt.tight_layout()
    fig.savefig(output_dir / "cluster_sizes.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved: cluster_sizes.png")


def plot_radar_charts(
    df_raw: pd.DataFrame,
    labels: np.ndarray,
    personas: list[PersonaResult],
    top_features: list[str],
    output_dir: Path,
) -> None:
    """Radar chart per cluster for top features."""
    _setup_style()
    num_feats = [f for f in top_features if f in df_raw.columns
                 and pd.api.types.is_numeric_dtype(df_raw[f])][:6]
    n = len(num_feats)
    if n < 3:
        log.warning("Not enough numeric features for radar chart (need >= 3, got %d)", n)
        return

    n_personas = min(len(personas), 3)
    fig = plt.figure(figsize=(6 * n_personas, 6))
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False).tolist()
    angles += angles[:1]

    global_min = df_raw[num_feats].min()
    global_max = df_raw[num_feats].max()

    for idx, p in enumerate(personas[:n_personas]):
        ax = fig.add_subplot(1, n_personas, idx + 1, polar=True)
        mask = labels == p.cluster_id
        means = df_raw[mask][num_feats].mean()
        normed = ((means - global_min) / (global_max - global_min + 1e-9)).values.tolist()
        normed += normed[:1]

        ax.plot(angles, normed, color=PALETTE[idx], linewidth=2)
        ax.fill(angles, normed, color=PALETTE[idx], alpha=0.25)
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(num_feats, size=8)
        ax.set_ylim(0, 1)
        ax.set_title(f"{p.persona_name}\n{p.archetype[:18]}", size=10, pad=14,
                     color=PALETTE[idx], fontweight="bold")

    plt.suptitle("Customer Persona Profiles", fontsize=15, fontweight="bold", y=1.01)
    plt.tight_layout()
    fig.savefig(output_dir / "persona_radar.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved: persona_radar.png")


def plot_experiment_history(
    exp_log: ExperimentLog,
    output_dir: Path,
) -> None:
    """Line chart of silhouette score across modeling loop iterations."""
    _setup_style()
    df = exp_log.to_dataframe()
    if df.empty:
        return

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(df["iteration"], df["silhouette"], "o-", color=PALETTE[0], linewidth=2, markersize=8)

    for _, row in df.iterrows():
        marker = "o" if not row["passed"] else "s"
        color = PALETTE[3] if not row["passed"] else PALETTE[2]
        ax.scatter(row["iteration"], row["silhouette"], c=color, s=100, zorder=5, marker=marker)
        if row.get("reconfig_strategy"):
            ax.annotate(row["reconfig_strategy"], (row["iteration"], row["silhouette"]),
                        textcoords="offset points", xytext=(5, 10), fontsize=8, alpha=0.7)

    ax.set_xlabel("Iteration")
    ax.set_ylabel("Silhouette Score")
    ax.set_title("Modeling Loop - Silhouette Score per Iteration")
    ax.legend(handles=[
        mpatches.Patch(color=PALETTE[2], label="Passed"),
        mpatches.Patch(color=PALETTE[3], label="Failed"),
    ])

    plt.tight_layout()
    fig.savefig(output_dir / "experiment_history.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved: experiment_history.png")


def plot_elbow_curve(
    inertia_curve: dict[int, float],
    output_dir: Path,
) -> None:
    """Elbow curve (K vs inertia)."""
    _setup_style()
    ks = sorted(inertia_curve.keys())
    inertias = [inertia_curve[k] for k in ks]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(ks, inertias, "o-", color=PALETTE[0], linewidth=2, markersize=8)
    ax.set_xlabel("Number of Clusters (K)")
    ax.set_ylabel("Inertia")
    ax.set_title("Elbow Curve - K-Means Inertia")
    ax.set_xticks(ks)

    plt.tight_layout()
    fig.savefig(output_dir / "elbow_curve.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved: elbow_curve.png")
