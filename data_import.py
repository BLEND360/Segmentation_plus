"""
Segmentation Plus — Data Import & Validation
=============================================
Production-grade data ingestion with schema validation,
column mapping, and data quality checks.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd

from config import DomainConfig

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────
# Data Quality Report
# ──────────────────────────────────────────────

class DataQualityReport:
    """Stores data quality check results."""

    def __init__(self) -> None:
        self.total_rows: int = 0
        self.total_columns: int = 0
        self.missing_values: dict[str, int] = {}
        self.missing_pct: dict[str, float] = {}
        self.duplicate_rows: int = 0
        self.column_types: dict[str, str] = {}
        self.warnings: list[str] = []
        self.errors: list[str] = []
        self.passed: bool = True

    def summary(self) -> str:
        status = "✅ PASSED" if self.passed else "❌ FAILED"
        lines = [
            f"Data Quality Report — {status}",
            f"  Rows: {self.total_rows:,}  |  Columns: {self.total_columns}",
            f"  Duplicates: {self.duplicate_rows}",
        ]
        if self.warnings:
            lines.append(f"  Warnings ({len(self.warnings)}):")
            for w in self.warnings:
                lines.append(f"    ⚠ {w}")
        if self.errors:
            lines.append(f"  Errors ({len(self.errors)}):")
            for e in self.errors:
                lines.append(f"    ✗ {e}")
        return "\n".join(lines)


# ──────────────────────────────────────────────
# Data Loader
# ──────────────────────────────────────────────

def load_data(
    file_path: str | Path,
    sheet_name: Optional[str] = None,
) -> pd.DataFrame:
    """
    Load data from CSV or Excel file.

    Parameters
    ----------
    file_path : str or Path
        Path to the data file (.csv, .xlsx, .xls).
    sheet_name : str, optional
        Sheet name for Excel files.

    Returns
    -------
    pd.DataFrame
        Raw loaded DataFrame.

    Raises
    ------
    FileNotFoundError
        If the file does not exist.
    ValueError
        If the file format is not supported.
    """
    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"Data file not found: {path}")

    suffix = path.suffix.lower()
    logger.info("Loading data from %s (%s)", path.name, suffix)

    if suffix == ".csv":
        df = pd.read_csv(path)
    elif suffix in (".xlsx", ".xls"):
        df = pd.read_excel(path, sheet_name=sheet_name)
    elif suffix == ".parquet":
        df = pd.read_parquet(path)
    else:
        raise ValueError(
            f"Unsupported file format: '{suffix}'. "
            f"Supported: .csv, .xlsx, .xls, .parquet"
        )

    logger.info("Loaded %d rows × %d columns", len(df), len(df.columns))
    return df


# ──────────────────────────────────────────────
# Schema Validation
# ──────────────────────────────────────────────

def validate_schema(
    df: pd.DataFrame,
    domain_config: DomainConfig,
    strict: bool = False,
) -> DataQualityReport:
    """
    Validate a DataFrame against the domain schema.

    Parameters
    ----------
    df : pd.DataFrame
        Input data to validate.
    domain_config : DomainConfig
        Domain configuration with required columns and feature schema.
    strict : bool
        If True, fail on warnings (missing optional columns). Default False.

    Returns
    -------
    DataQualityReport
        Detailed quality report with pass/fail status.
    """
    report = DataQualityReport()
    report.total_rows = len(df)
    report.total_columns = len(df.columns)
    report.column_types = {col: str(df[col].dtype) for col in df.columns}

    # 1. Check required columns
    df_cols_lower = {c.lower(): c for c in df.columns}
    for req_col in domain_config.required_columns:
        if req_col.lower() not in df_cols_lower:
            report.errors.append(f"Required column missing: '{req_col}'")
            report.passed = False

    # 2. Check optional domain feature columns
    all_domain_cols = domain_config.all_feature_columns
    for col in all_domain_cols:
        if col.lower() not in df_cols_lower and col not in df.columns:
            report.warnings.append(f"Domain feature column missing: '{col}'")
            if strict:
                report.passed = False

    # 3. Missing values
    missing = df.isnull().sum()
    for col in df.columns:
        if missing[col] > 0:
            report.missing_values[col] = int(missing[col])
            report.missing_pct[col] = round(missing[col] / len(df) * 100, 2)
            if report.missing_pct[col] > 50:
                report.warnings.append(
                    f"Column '{col}' has {report.missing_pct[col]:.1f}% missing values"
                )

    # 4. Duplicate rows
    report.duplicate_rows = int(df.duplicated().sum())
    if report.duplicate_rows > 0:
        report.warnings.append(
            f"{report.duplicate_rows} duplicate rows found ({report.duplicate_rows/len(df)*100:.1f}%)"
        )

    # 5. Row count check
    if len(df) < 50:
        report.warnings.append(
            f"Very small dataset ({len(df)} rows) — clustering may not be reliable"
        )

    logger.info("Schema validation: %s", "PASSED" if report.passed else "FAILED")
    return report


# ──────────────────────────────────────────────
# Column Mapping (fuzzy matching)
# ──────────────────────────────────────────────

def auto_map_columns(
    df: pd.DataFrame,
    domain_config: DomainConfig,
) -> Tuple[pd.DataFrame, dict[str, str]]:
    """
    Attempt to auto-map DataFrame columns to domain schema columns
    using case-insensitive and partial matching.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame with potentially different column names.
    domain_config : DomainConfig
        Domain config with expected column names.

    Returns
    -------
    tuple[pd.DataFrame, dict]
        Renamed DataFrame and the mapping used (original → domain).
    """
    mapping: dict[str, str] = {}
    domain_cols = (
        domain_config.required_columns + domain_config.all_feature_columns
    )

    df_cols_lower = {c.lower().replace(" ", "_").replace("-", "_"): c for c in df.columns}

    for domain_col in domain_cols:
        normalized = domain_col.lower().replace(" ", "_").replace("-", "_")

        # Exact match (case-insensitive)
        if normalized in df_cols_lower:
            original = df_cols_lower[normalized]
            if original != domain_col:
                mapping[original] = domain_col

    if mapping:
        logger.info("Auto-mapped %d columns: %s", len(mapping), mapping)
        df = df.rename(columns=mapping)

    return df, mapping


# ──────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────

def import_and_validate(
    file_path: str | Path,
    domain_config: DomainConfig,
    auto_map: bool = True,
    sheet_name: Optional[str] = None,
) -> Tuple[pd.DataFrame, DataQualityReport]:
    """
    Full import pipeline: load → auto-map → validate → report.

    Parameters
    ----------
    file_path : str or Path
        Path to data file.
    domain_config : DomainConfig
        Domain configuration to validate against.
    auto_map : bool
        Whether to attempt automatic column name mapping.
    sheet_name : str, optional
        Sheet name for Excel files.

    Returns
    -------
    tuple[pd.DataFrame, DataQualityReport]
        Processed DataFrame and its quality report.
    """
    # Load
    df = load_data(file_path, sheet_name=sheet_name)

    # Auto-map columns
    if auto_map:
        df, col_mapping = auto_map_columns(df, domain_config)

    # Validate
    report = validate_schema(df, domain_config)

    # Log summary
    logger.info("\n%s", report.summary())

    return df, report
