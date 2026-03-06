"""
Segmentation Plus — GenAI Persona Generation
=============================================
Generates rich marketing personas from cluster profiles
using Google Gemini API, with domain-aware prompt engineering.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from config import DomainConfig

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────
# Cluster Profile Builder
# ──────────────────────────────────────────────

def build_cluster_profiles(
    df: pd.DataFrame,
    labels: np.ndarray,
    feature_columns: List[str],
    shap_importance: Optional[Dict] = None,
) -> List[Dict[str, Any]]:
    """
    Build statistical profiles for each cluster.

    Parameters
    ----------
    df : pd.DataFrame
        Original DataFrame (pre-scaling, for interpretable stats).
    labels : np.ndarray
        Cluster assignments.
    feature_columns : list[str]
        Feature columns to compute stats on.
    shap_importance : dict, optional
        Feature importance from SHAP (feature_name → importance).

    Returns
    -------
    list[dict]
        Per-cluster profile dictionaries.
    """
    df = df.copy()
    df["cluster"] = labels

    profiles = []
    available_cols = [c for c in feature_columns if c in df.columns]

    for cluster_id in sorted(df["cluster"].unique()):
        if cluster_id == -1:
            continue  # Skip DBSCAN noise

        cluster_df = df[df["cluster"] == cluster_id]
        profile = {
            "segment_id": int(cluster_id),
            "size": len(cluster_df),
            "pct_of_total": round(len(cluster_df) / len(df) * 100, 1),
        }

        # Compute mean for numeric columns
        stats = {}
        for col in available_cols:
            if df[col].dtype in [np.float64, np.int64, float, int]:
                stats[col] = {
                    "mean": round(float(cluster_df[col].mean()), 2),
                    "median": round(float(cluster_df[col].median()), 2),
                    "std": round(float(cluster_df[col].std()), 2),
                }
        profile["stats"] = stats

        # Top SHAP drivers for this cluster
        if shap_importance:
            sorted_feats = sorted(
                shap_importance.items(), key=lambda x: abs(x[1]), reverse=True
            )
            profile["shap_drivers"] = [
                {"feature": f, "importance": round(v, 4)}
                for f, v in sorted_feats[:7]
            ]

        profiles.append(profile)

    logger.info("Built profiles for %d clusters", len(profiles))
    return profiles


# ──────────────────────────────────────────────
# Prompt Builder
# ──────────────────────────────────────────────

def build_persona_prompt(
    profile: Dict,
    domain_config: DomainConfig,
) -> str:
    """
    Build the GenAI prompt for a single cluster.

    Parameters
    ----------
    profile : dict
        Cluster profile from build_cluster_profiles().
    domain_config : DomainConfig
        Domain configuration with prompt template.

    Returns
    -------
    str
        Formatted prompt ready for the LLM.
    """
    # Format stats as readable text
    stats_lines = []
    for feature, vals in profile.get("stats", {}).items():
        stats_lines.append(f"  - {feature}: mean={vals['mean']}, median={vals['median']}")
    segment_stats = "\n".join(stats_lines) if stats_lines else "  - No stats available"

    # Format SHAP drivers
    shap_lines = []
    for driver in profile.get("shap_drivers", []):
        shap_lines.append(f"  - {driver['feature']}: {driver['importance']}")
    shap_drivers = "\n".join(shap_lines) if shap_lines else "  - No SHAP data available"

    # Fill template
    template = domain_config.persona_prompt_template
    if not template:
        template = _DEFAULT_PROMPT_TEMPLATE

    prompt = template.format(
        segment_id=profile["segment_id"],
        segment_stats=segment_stats,
        shap_drivers=shap_drivers,
        segment_size=profile["size"],
        segment_pct=profile["pct_of_total"],
    )

    return prompt


_DEFAULT_PROMPT_TEMPLATE = """
You are a marketing strategist advising a company on their customer segments.

Segment {segment_id} ({segment_size} customers, {segment_pct}% of total):
{segment_stats}

Top Feature Drivers (SHAP importance):
{shap_drivers}

Generate a detailed customer persona as structured JSON with keys:
name, tagline, demographics, goals, pain_points, product_recommendations,
channels, strategy
"""


# ──────────────────────────────────────────────
# Persona Generator
# ──────────────────────────────────────────────

class PersonaGenerator:
    """
    Generates marketing personas using Google Gemini API.

    Parameters
    ----------
    domain_config : DomainConfig
        Domain configuration.
    output_dir : Path
        Output directory for personas.
    api_key : str, optional
        Gemini API key. If None, reads from GOOGLE_API_KEY env var.
    model_name : str
        Gemini model name (default: 'gemini-pro').
    """

    def __init__(
        self,
        domain_config: DomainConfig,
        output_dir: Path,
        api_key: Optional[str] = None,
        model_name: str = "gemini-pro",
    ) -> None:
        self.config = domain_config
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.api_key = api_key or os.environ.get("GOOGLE_API_KEY", "")
        self.model_name = model_name
        self._model = None

    def _init_model(self):
        """Lazily initialize the Gemini model."""
        if self._model is not None:
            return

        if not self.api_key:
            raise ValueError(
                "Google API key not found. Set GOOGLE_API_KEY environment variable "
                "or pass api_key parameter."
            )

        try:
            import google.generativeai as genai
            genai.configure(api_key=self.api_key)
            self._model = genai.GenerativeModel(self.model_name)
            logger.info("Gemini model '%s' initialized", self.model_name)
        except ImportError:
            raise ImportError(
                "google-generativeai package not installed. "
                "Run: pip install google-generativeai"
            )

    def generate_personas(
        self,
        profiles: List[Dict],
    ) -> List[Dict]:
        """
        Generate personas for all cluster profiles.

        Parameters
        ----------
        profiles : list[dict]
            Cluster profiles from build_cluster_profiles().

        Returns
        -------
        list[dict]
            Generated persona dictionaries.
        """
        self._init_model()
        personas = []

        for profile in profiles:
            try:
                prompt = build_persona_prompt(profile, self.config)
                logger.info("Generating persona for Segment %d...", profile["segment_id"])

                response = self._model.generate_content(prompt)
                persona_text = response.text

                # Try to parse as JSON
                persona = self._parse_persona(persona_text, profile["segment_id"])
                persona["segment_id"] = profile["segment_id"]
                persona["segment_size"] = profile["size"]
                persona["segment_pct"] = profile["pct_of_total"]
                personas.append(persona)

                logger.info("  → Generated: %s", persona.get("name", "Unknown"))

            except Exception as e:
                logger.error("Failed to generate persona for Segment %d: %s", profile["segment_id"], e)
                personas.append({
                    "segment_id": profile["segment_id"],
                    "error": str(e),
                    "raw_profile": profile,
                })

        # Save outputs
        self._save_personas(personas)
        return personas

    def _parse_persona(self, text: str, segment_id: int) -> Dict:
        """Parse LLM response text into a structured persona dict."""
        # Try JSON extraction
        try:
            # Look for JSON block in the response
            start = text.find("{")
            end = text.rfind("}") + 1
            if start != -1 and end > start:
                return json.loads(text[start:end])
        except json.JSONDecodeError:
            pass

        # Fallback: return as raw text
        logger.warning("Could not parse JSON for Segment %d — storing as raw text", segment_id)
        return {"raw_text": text}

    def _save_personas(self, personas: List[Dict]) -> None:
        """Save personas as JSON and Markdown."""
        # JSON
        json_path = self.output_dir / "personas.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(personas, f, indent=2, ensure_ascii=False)
        logger.info("Personas saved to %s", json_path)

        # Markdown
        md_path = self.output_dir / "personas.md"
        md_lines = ["# Customer Personas\n"]
        for p in personas:
            md_lines.append(f"## Segment {p.get('segment_id', '?')}: {p.get('name', 'Unknown')}\n")
            md_lines.append(f"**{p.get('tagline', '')}**\n")
            md_lines.append(f"- Size: {p.get('segment_size', '?')} customers ({p.get('segment_pct', '?')}%)\n")

            for key in ["demographics", "goals", "pain_points", "product_recommendations", "channels", "strategy"]:
                if key in p:
                    md_lines.append(f"\n### {key.replace('_', ' ').title()}\n")
                    val = p[key]
                    if isinstance(val, list):
                        for item in val:
                            md_lines.append(f"- {item}")
                    elif isinstance(val, str):
                        md_lines.append(val)
                    md_lines.append("")

            if "raw_text" in p:
                md_lines.append(f"\n{p['raw_text']}\n")

            md_lines.append("\n---\n")

        with open(md_path, "w", encoding="utf-8") as f:
            f.write("\n".join(md_lines))
        logger.info("Personas markdown saved to %s", md_path)
