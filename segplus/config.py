"""Pipeline and domain configuration management."""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import yaml

log = logging.getLogger("segplus.config")

_DOMAINS_DIR = Path(__file__).resolve().parent.parent / "domains"


# ── Domain Config (parsed from YAML) ────────────────────────────────────────

@dataclass
class FeatureEngineeringRule:
    name: str
    formula: str
    bins: list[str] | None = None
    description: str = ""


@dataclass
class DomainConfig:
    domain_key: str
    display_name: str
    features: dict[str, list[str]] = field(default_factory=dict)
    required_columns: list[str] = field(default_factory=list)
    feature_engineering: list[FeatureEngineeringRule] = field(default_factory=list)
    eda_analyses: list[str] = field(default_factory=list)
    persona_prompt_template: str = ""
    scaling_exclude: list[str] = field(default_factory=list)
    categorical_columns: list[str] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)

    @property
    def all_feature_columns(self) -> list[str]:
        cols: list[str] = []
        for group_cols in self.features.values():
            cols.extend(group_cols)
        return cols

    @property
    def numerical_columns(self) -> list[str]:
        return [c for c in self.all_feature_columns if c not in self.categorical_columns]


def load_domain_config(domain_key: str, domains_dir: Path | None = None) -> DomainConfig:
    """Load a domain configuration from YAML."""
    d = domains_dir or _DOMAINS_DIR
    yaml_path = d / f"{domain_key}.yaml"
    if not yaml_path.exists():
        available = [f.stem for f in d.glob("*.yaml") if not f.stem.startswith("_")]
        raise FileNotFoundError(
            f"Domain '{domain_key}' not found at {yaml_path}. "
            f"Available: {available}"
        )

    with open(yaml_path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    fe_rules = []
    for rule in raw.get("feature_engineering", []):
        fe_rules.append(FeatureEngineeringRule(
            name=rule["name"],
            formula=rule["formula"],
            bins=rule.get("bins"),
            description=rule.get("description", ""),
        ))

    eda_analyses = []
    eda_block = raw.get("eda", {})
    if isinstance(eda_block, dict):
        eda_analyses = eda_block.get("domain_analyses", [])

    return DomainConfig(
        domain_key=raw.get("domain", domain_key),
        display_name=raw.get("display_name", domain_key),
        features=raw.get("features", {}),
        required_columns=raw.get("required_columns", []),
        feature_engineering=fe_rules,
        eda_analyses=eda_analyses,
        persona_prompt_template=raw.get("persona_prompt_template", ""),
        scaling_exclude=raw.get("scaling_exclude", []),
        categorical_columns=raw.get("categorical_columns", []),
        metadata=raw.get("metadata", {}),
    )


def list_available_domains(domains_dir: Path | None = None) -> list[str]:
    d = domains_dir or _DOMAINS_DIR
    return sorted(f.stem for f in d.glob("*.yaml") if not f.stem.startswith("_"))


# ── Pipeline Config ──────────────────────────────────────────────────────────

@dataclass
class PipelineConfig:
    """All tunable pipeline parameters with sensible defaults."""

    # Data
    data_path: str = "final_enterprise_clustering_dataset.xlsx"
    domain_key: str = "financial_services"
    sheet_name: Optional[str] = None
    exclude_cols: list[str] = field(default_factory=lambda: ["customer_id"])

    # Feature Engineering
    pca_variance_threshold: float = 0.85
    winsorize_lower: float = 0.01
    winsorize_upper: float = 0.99
    imputation_strategy: str = "median"

    # Clustering
    k_range: tuple[int, int] = (2, 8)
    random_state: int = 42

    # Evaluation Gate
    silhouette_threshold: float = 0.15
    davies_bouldin_threshold: float = 2.5
    stability_ari_threshold: float = 0.7
    stability_n_bootstraps: int = 30

    # Modeling Loop
    max_iterations: int = 5

    # Explainability
    n_top_features: int = 10
    shap_n_repeats: int = 10

    # Ollama
    ollama_host: str = "http://localhost:11434"
    ollama_model: str = "llama3"
    ollama_timeout: int = 120

    # Business Objective
    business_objective: str = (
        "Identify distinct customer segments to personalise marketing campaigns, "
        "improve retention for high-value customers, and convert mid-tier customers "
        "to premium products."
    )

    # Output
    output_dir: str = "segplus_output"
