"""
Segmentation Plus — Interactive Dashboard
==========================================
Plotly-based interactive dashboard combining cluster visualizations,
evaluation metrics, and persona cards into a single HTML file.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

logger = logging.getLogger(__name__)


class DashboardBuilder:
    """
    Builds an interactive HTML dashboard.

    Parameters
    ----------
    output_dir : Path
        Directory to save the dashboard HTML.
    """

    def __init__(self, output_dir: Path) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.sections: List[str] = []

    def build(
        self,
        df: pd.DataFrame,
        labels: np.ndarray,
        X_for_viz: np.ndarray,
        feature_names: List[str],
        personas: Optional[List[Dict]] = None,
        evaluation_report: Optional[Dict] = None,
    ) -> Path:
        """
        Build and save the full dashboard.

        Parameters
        ----------
        df : pd.DataFrame
            Original DataFrame.
        labels : np.ndarray
            Cluster assignments.
        X_for_viz : np.ndarray
            2D or 3D array for scatter plots (PCA-reduced).
        feature_names : list[str]
            Feature column names.
        personas : list[dict], optional
            Generated personas.
        evaluation_report : dict, optional
            Evaluation metrics.

        Returns
        -------
        Path
            Path to the saved dashboard HTML file.
        """
        logger.info("Building interactive dashboard...")

        self.sections = []

        # Cluster scatter plot
        self._add_cluster_scatter(X_for_viz, labels)

        # Cluster size distribution
        self._add_cluster_sizes(labels)

        # Feature distributions by cluster
        self._add_feature_distributions(df, labels, feature_names)

        # Cluster radar chart
        self._add_radar_chart(df, labels, feature_names)

        # Evaluation metrics summary
        if evaluation_report:
            self._add_metrics_summary(evaluation_report)

        # Persona cards
        if personas:
            self._add_persona_cards(personas)

        # Combine into HTML
        html = self._render_html()
        output_path = self.output_dir / "dashboard.html"
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(html)

        logger.info("Dashboard saved to %s", output_path)
        return output_path

    def _add_cluster_scatter(self, X: np.ndarray, labels: np.ndarray) -> None:
        """PCA scatter plot colored by cluster."""
        df_plot = pd.DataFrame({
            "PC1": X[:, 0],
            "PC2": X[:, 1] if X.shape[1] > 1 else np.zeros(len(X)),
            "Cluster": labels.astype(str),
        })

        fig = px.scatter(
            df_plot, x="PC1", y="PC2", color="Cluster",
            title="Customer Segments (PCA Projection)",
            opacity=0.6, width=900, height=500,
            color_discrete_sequence=px.colors.qualitative.Set2,
        )
        fig.update_layout(
            template="plotly_dark",
            plot_bgcolor="#1e2233",
            paper_bgcolor="#0f1117",
            font=dict(family="Inter"),
        )
        self.sections.append(("Cluster Scatter Plot", fig.to_html(include_plotlyjs=False, full_html=False)))

    def _add_cluster_sizes(self, labels: np.ndarray) -> None:
        """Cluster size bar chart."""
        unique, counts = np.unique(labels[labels != -1], return_counts=True)

        fig = go.Figure(data=[go.Bar(
            x=[f"Segment {c}" for c in unique],
            y=counts,
            marker_color=px.colors.qualitative.Set2[:len(unique)],
            text=counts, textposition="auto",
        )])
        fig.update_layout(
            title="Segment Size Distribution",
            template="plotly_dark",
            plot_bgcolor="#1e2233",
            paper_bgcolor="#0f1117",
            width=900, height=400,
            font=dict(family="Inter"),
        )
        self.sections.append(("Segment Sizes", fig.to_html(include_plotlyjs=False, full_html=False)))

    def _add_feature_distributions(
        self, df: pd.DataFrame, labels: np.ndarray, feature_names: List[str],
    ) -> None:
        """Violin plots for key features per cluster."""
        plot_df = df.copy()
        plot_df["Cluster"] = labels.astype(str)

        numeric_features = [
            c for c in feature_names
            if c in plot_df.columns and plot_df[c].dtype in [np.float64, np.int64]
        ][:6]  # Top 6 features

        if not numeric_features:
            return

        fig = make_subplots(rows=2, cols=3, subplot_titles=numeric_features)
        for idx, feat in enumerate(numeric_features):
            row, col = idx // 3 + 1, idx % 3 + 1
            for cluster_val in sorted(plot_df["Cluster"].unique()):
                subset = plot_df[plot_df["Cluster"] == cluster_val]
                fig.add_trace(
                    go.Violin(y=subset[feat], name=f"Seg {cluster_val}", legendgroup=cluster_val,
                              scalemode="count", showlegend=(idx == 0)),
                    row=row, col=col,
                )

        fig.update_layout(
            title="Feature Distributions by Segment",
            template="plotly_dark", plot_bgcolor="#1e2233", paper_bgcolor="#0f1117",
            width=1000, height=600, font=dict(family="Inter"),
        )
        self.sections.append(("Feature Distributions", fig.to_html(include_plotlyjs=False, full_html=False)))

    def _add_radar_chart(
        self, df: pd.DataFrame, labels: np.ndarray, feature_names: List[str],
    ) -> None:
        """Radar chart comparing cluster profiles."""
        plot_df = df.copy()
        plot_df["cluster"] = labels

        numeric_features = [
            c for c in feature_names
            if c in plot_df.columns and plot_df[c].dtype in [np.float64, np.int64]
        ][:8]

        if len(numeric_features) < 3:
            return

        # Normalize to 0-1 for radar
        means = plot_df.groupby("cluster")[numeric_features].mean()
        for col in numeric_features:
            col_range = means[col].max() - means[col].min()
            if col_range > 0:
                means[col] = (means[col] - means[col].min()) / col_range

        fig = go.Figure()
        colors = px.colors.qualitative.Set2
        for idx, (cluster_id, row) in enumerate(means.iterrows()):
            if cluster_id == -1:
                continue
            fig.add_trace(go.Scatterpolar(
                r=row.values.tolist() + [row.values[0]],
                theta=numeric_features + [numeric_features[0]],
                fill="toself", name=f"Segment {cluster_id}",
                opacity=0.6, line=dict(color=colors[idx % len(colors)]),
            ))

        fig.update_layout(
            title="Segment Profile Comparison",
            template="plotly_dark", paper_bgcolor="#0f1117",
            width=800, height=500, font=dict(family="Inter"),
            polar=dict(bgcolor="#1e2233"),
        )
        self.sections.append(("Radar Comparison", fig.to_html(include_plotlyjs=False, full_html=False)))

    def _add_metrics_summary(self, report: Dict) -> None:
        """Render evaluation metrics as an HTML card."""
        quality = report.get("quality_metrics", {})
        stability = report.get("stability", {})

        html = '<div style="display:grid;grid-template-columns:repeat(4,1fr);gap:16px;margin:20px 0;">'
        metrics = [
            ("Silhouette", f"{quality.get('silhouette', 0):.3f}", "#5b8def"),
            ("Davies-Bouldin", f"{quality.get('davies_bouldin', 0):.3f}", "#fb923c"),
            ("Calinski-Harabasz", f"{quality.get('calinski_harabasz', 0):.0f}", "#34d399"),
            ("Stability (ARI)", f"{stability.get('ari_mean', 0):.3f}", "#a78bfa"),
        ]
        for name, val, color in metrics:
            html += f'''
            <div style="background:#1e2233;border:1px solid #2d3348;border-radius:12px;
                        padding:20px;text-align:center;">
                <div style="font-size:.75rem;color:#6b7189;text-transform:uppercase;
                            letter-spacing:1px;margin-bottom:6px;">{name}</div>
                <div style="font-size:1.8rem;font-weight:700;color:{color};">{val}</div>
            </div>'''
        html += '</div>'
        self.sections.append(("Evaluation Metrics", html))

    def _add_persona_cards(self, personas: List[Dict]) -> None:
        """Render persona cards as styled HTML."""
        cards_html = '<div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(400px,1fr));gap:20px;margin:20px 0;">'

        for p in personas:
            name = p.get("name", f"Segment {p.get('segment_id', '?')}")
            tagline = p.get("tagline", "")
            size = p.get("segment_size", "?")
            pct = p.get("segment_pct", "?")

            goals = p.get("goals", [])
            if isinstance(goals, str):
                goals = [goals]
            goals_html = "".join(f"<li>{g}</li>" for g in goals[:4])

            pain_points = p.get("pain_points", [])
            if isinstance(pain_points, str):
                pain_points = [pain_points]
            pp_html = "".join(f"<li>{pp}</li>" for pp in pain_points[:4])

            cards_html += f'''
            <div style="background:#1e2233;border:1px solid #2d3348;border-radius:14px;padding:28px;">
                <div style="font-size:1.2rem;font-weight:700;color:#e8eaf0;margin-bottom:4px;">
                    🧑 {name}
                </div>
                <div style="font-size:.85rem;color:#a78bfa;margin-bottom:12px;font-style:italic;">
                    {tagline}
                </div>
                <div style="font-size:.78rem;color:#6b7189;margin-bottom:16px;">
                    {size} customers ({pct}% of total)
                </div>
                <div style="font-size:.82rem;color:#9ca3b8;">
                    <strong style="color:#34d399;">Goals:</strong>
                    <ul style="padding-left:16px;margin:4px 0 10px;">{goals_html}</ul>
                    <strong style="color:#f87171;">Pain Points:</strong>
                    <ul style="padding-left:16px;margin:4px 0;">{pp_html}</ul>
                </div>
            </div>'''
        cards_html += '</div>'
        self.sections.append(("Customer Personas", cards_html))

    def _render_html(self) -> str:
        """Combine all sections into a final HTML page."""
        sections_html = ""
        for title, content in self.sections:
            sections_html += f'''
            <div style="margin-bottom:48px;">
                <h2 style="font-size:1.3rem;font-weight:700;color:#e8eaf0;
                           margin-bottom:16px;padding-bottom:8px;
                           border-bottom:1px solid #2d3348;">{title}</h2>
                {content}
            </div>'''

        return f'''<!DOCTYPE html>
<html lang="en"><head>
<meta charset="UTF-8"><meta name="viewport" content="width=device-width,initial-scale=1.0">
<title>Segmentation Plus — Dashboard</title>
<script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700;800&display=swap" rel="stylesheet">
<style>
body{{font-family:'Inter',sans-serif;background:#0f1117;color:#e8eaf0;margin:0;padding:0;}}
.header{{background:linear-gradient(135deg,#1a1040,#0f1117 50%,#0a1628);padding:48px 40px;
         border-bottom:1px solid #2d3348;}}
.header h1{{font-size:2rem;font-weight:800;background:linear-gradient(135deg,#5b8def,#a78bfa);
            -webkit-background-clip:text;-webkit-text-fill-color:transparent;margin-bottom:8px;}}
.header p{{color:#9ca3b8;font-size:.95rem;}}
.container{{max-width:1100px;margin:0 auto;padding:36px 40px;}}
</style></head><body>
<div class="header">
<h1>Segmentation Plus — Dashboard</h1>
<p>Interactive customer segmentation results and AI-generated personas</p>
</div>
<div class="container">{sections_html}</div>
</body></html>'''
