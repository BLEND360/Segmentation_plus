"""
Segmentation Plus — Domain Configuration System
=================================================
Loads domain-specific YAML configs that drive the entire pipeline:
EDA, feature engineering, clustering, and persona generation.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────
# Domain Config Dataclass
# ──────────────────────────────────────────────

@dataclass
class FeatureEngineeringRule:
    """Single feature engineering rule from the domain config."""
    name: str
    formula: str
    bins: Optional[List[str]] = None
    description: str = ""


@dataclass
class DomainConfig:
    """
    Complete configuration for a single domain.

    Attributes
    ----------
    domain_key : str
        Machine-readable domain identifier (e.g., 'financial_services').
    display_name : str
        Human-readable name (e.g., 'Financial Services').
    features : dict
        Feature groups → list of column names.
    required_columns : list
        Columns that MUST exist in any input dataset for this domain.
    feature_engineering : list
        Derived feature rules (formulas, bins, etc.).
    eda_analyses : list
        Domain-specific EDA analyses to run.
    persona_prompt_template : str
        Jinja-style prompt template for GenAI persona generation.
    scaling_exclude : list
        Columns to exclude from StandardScaler (e.g., IDs, booleans).
    categorical_columns : list
        Columns to one-hot encode.
    target_column : str | None
        Optional target column (not used in unsupervised, but kept for flexibility).
    metadata : dict
        Any extra domain-specific metadata.
    """

    domain_key: str
    display_name: str
    features: Dict[str, List[str]] = field(default_factory=dict)
    required_columns: List[str] = field(default_factory=list)
    feature_engineering: List[FeatureEngineeringRule] = field(default_factory=list)
    eda_analyses: List[str] = field(default_factory=list)
    persona_prompt_template: str = ""
    scaling_exclude: List[str] = field(default_factory=list)
    categorical_columns: List[str] = field(default_factory=list)
    target_column: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    # ── Computed properties ──

    @property
    def all_feature_columns(self) -> List[str]:
        """Flatten all feature groups into a single list."""
        cols = []
        for group_cols in self.features.values():
            cols.extend(group_cols)
        return cols

    @property
    def numerical_columns(self) -> List[str]:
        """All feature columns minus categorical ones."""
        return [c for c in self.all_feature_columns if c not in self.categorical_columns]


# ──────────────────────────────────────────────
# Config Loader
# ──────────────────────────────────────────────

_DOMAINS_DIR = Path(__file__).parent / "domains"


def list_available_domains() -> List[str]:
    """Return names of all available domain configs (without .yaml extension)."""
    if not _DOMAINS_DIR.exists():
        return []
    return [
        p.stem for p in _DOMAINS_DIR.glob("*.yaml")
        if not p.stem.startswith("_")
    ]


def load_domain_config(domain_key: str, config_path: Optional[Path] = None) -> DomainConfig:
    """
    Load and parse a domain YAML config into a DomainConfig dataclass.

    Parameters
    ----------
    domain_key : str
        Name of the domain (matches the YAML filename without extension).
    config_path : Path, optional
        Override path. If None, looks in the default `domains/` directory.

    Returns
    -------
    DomainConfig
        Fully parsed domain configuration.

    Raises
    ------
    FileNotFoundError
        If the YAML file does not exist.
    ValueError
        If the YAML is malformed or missing required fields.
    """
    if config_path is None:
        config_path = _DOMAINS_DIR / f"{domain_key}.yaml"

    if not config_path.exists():
        available = list_available_domains()
        raise FileNotFoundError(
            f"Domain config '{domain_key}' not found at {config_path}. "
            f"Available domains: {available}"
        )

    logger.info("Loading domain config: %s from %s", domain_key, config_path)

    with open(config_path, "r", encoding="utf-8") as f:
        raw: Dict[str, Any] = yaml.safe_load(f)

    if not raw or not isinstance(raw, dict):
        raise ValueError(f"Domain config '{config_path}' is empty or malformed.")

    # Parse feature engineering rules
    fe_rules = []
    for rule_dict in raw.get("feature_engineering", []):
        fe_rules.append(FeatureEngineeringRule(
            name=rule_dict["name"],
            formula=rule_dict.get("formula", ""),
            bins=rule_dict.get("bins"),
            description=rule_dict.get("description", ""),
        ))

    config = DomainConfig(
        domain_key=raw.get("domain", domain_key),
        display_name=raw.get("display_name", domain_key.replace("_", " ").title()),
        features=raw.get("features", {}),
        required_columns=raw.get("required_columns", []),
        feature_engineering=fe_rules,
        eda_analyses=raw.get("eda", {}).get("domain_analyses", []),
        persona_prompt_template=raw.get("persona_prompt_template", ""),
        scaling_exclude=raw.get("scaling_exclude", []),
        categorical_columns=raw.get("categorical_columns", []),
        target_column=raw.get("target_column"),
        metadata=raw.get("metadata", {}),
    )

    logger.info(
        "Loaded domain '%s': %d feature groups, %d FE rules, %d EDA analyses",
        config.display_name,
        len(config.features),
        len(config.feature_engineering),
        len(config.eda_analyses),
    )

    return config
