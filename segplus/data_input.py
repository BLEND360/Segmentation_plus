"""Data loading, schema inference, and validation."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import pandas as pd

from .config import DomainConfig
from .types import ColumnMeta, DataQualityReport, DataSchema

log = logging.getLogger("segplus.data_input")


def load_data(file_path: str | Path, sheet_name: Optional[str] = None) -> pd.DataFrame:
    """Load data from CSV, Excel, or Parquet."""
    p = Path(file_path)
    if not p.exists():
        raise FileNotFoundError(f"Data file not found: {p}")

    suffix = p.suffix.lower()
    if suffix == ".csv":
        df = pd.read_csv(p)
    elif suffix in (".xlsx", ".xls"):
        df = pd.read_excel(p, sheet_name=sheet_name or 0, engine="openpyxl")
    elif suffix == ".parquet":
        df = pd.read_parquet(p)
    else:
        raise ValueError(f"Unsupported file format: {suffix}")

    log.info("Loaded %s: %d rows x %d cols", p.name, len(df), len(df.columns))
    return df


def infer_schema(df: pd.DataFrame) -> DataSchema:
    """Infer column types and build a DataSchema."""
    columns: list[ColumnMeta] = []
    for col in df.columns:
        s = df[col]
        n_unique = s.nunique()
        if pd.api.types.is_bool_dtype(s):
            dtype = "boolean"
        elif pd.api.types.is_datetime64_any_dtype(s):
            dtype = "datetime"
        elif pd.api.types.is_numeric_dtype(s):
            dtype = "numeric"
        elif n_unique / max(len(s), 1) < 0.05 or n_unique <= 20:
            dtype = "categorical"
        else:
            dtype = "text"
        columns.append(ColumnMeta(col, dtype, round(s.isna().mean(), 4), n_unique))
    return DataSchema(len(df), len(df.columns), columns)


def validate_schema(
    df: pd.DataFrame,
    domain_config: DomainConfig,
    schema: DataSchema,
) -> DataQualityReport:
    """Validate the dataframe against domain requirements."""
    report = DataQualityReport(
        total_rows=len(df),
        total_columns=len(df.columns),
    )

    # Column type map
    report.column_types = {c.name: c.dtype for c in schema.columns}

    # Missing values
    missing = df.isnull().sum()
    report.missing_values = {c: int(v) for c, v in missing.items() if v > 0}
    report.missing_pct = {c: round(v / len(df), 4) for c, v in report.missing_values.items()}

    # Duplicates
    report.duplicate_rows = int(df.duplicated().sum())
    if report.duplicate_rows > 0:
        report.warnings.append(f"{report.duplicate_rows} duplicate rows found")

    # Required columns check
    df_cols_lower = {c.lower(): c for c in df.columns}
    for req in domain_config.required_columns:
        if req.lower() not in df_cols_lower:
            report.errors.append(f"Required column missing: '{req}'")
            report.passed = False

    # Row count check
    if len(df) < 50:
        report.warnings.append(f"Very few rows ({len(df)}). Results may be unreliable.")

    # High-missing columns
    for col, pct in report.missing_pct.items():
        if pct > 0.5:
            report.warnings.append(f"Column '{col}' is {pct:.0%} missing")

    if report.errors:
        report.passed = False

    return report


def auto_map_columns(
    df: pd.DataFrame,
    domain_config: DomainConfig,
) -> tuple[pd.DataFrame, dict[str, str]]:
    """Case-insensitive column mapping to domain config names."""
    df_cols_lower = {c.lower().replace(" ", "_").replace("-", "_"): c for c in df.columns}
    mapping: dict[str, str] = {}

    all_expected = set(domain_config.all_feature_columns + domain_config.required_columns)

    for expected in all_expected:
        norm = expected.lower().replace(" ", "_").replace("-", "_")
        if norm in df_cols_lower and df_cols_lower[norm] != expected:
            mapping[df_cols_lower[norm]] = expected

    if mapping:
        df = df.rename(columns=mapping)
        log.info("Auto-mapped %d columns: %s", len(mapping), mapping)

    return df, mapping


def import_and_validate(
    file_path: str | Path,
    domain_config: DomainConfig,
    auto_map: bool = True,
    sheet_name: Optional[str] = None,
) -> tuple[pd.DataFrame, DataQualityReport, DataSchema]:
    """Full import pipeline: load -> auto-map -> infer schema -> validate."""
    df = load_data(file_path, sheet_name)

    if auto_map:
        df, _ = auto_map_columns(df, domain_config)

    schema = infer_schema(df)
    report = validate_schema(df, domain_config, schema)

    log.info("Data quality: %s", "PASSED" if report.passed else "FAILED")
    return df, report, schema
