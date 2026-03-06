"""Experiment tracking: logs every modeling loop iteration."""
from __future__ import annotations

import json
import logging
from pathlib import Path

import pandas as pd

from .types import ExperimentRecord

log = logging.getLogger("segplus.experiment_log")


def _to_native(value):
    """Recursively convert numpy/pandas scalars and containers to JSON-safe Python types."""
    # Dict
    if isinstance(value, dict):
        return {str(k): _to_native(v) for k, v in value.items()}
    # List / tuple
    if isinstance(value, (list, tuple)):
        return [_to_native(v) for v in value]
    # Numpy scalars / arrays (without importing numpy directly)
    if hasattr(value, "item") and callable(getattr(value, "item")):
        try:
            return value.item()
        except Exception:
            pass
    if hasattr(value, "tolist") and callable(getattr(value, "tolist")):
        try:
            return value.tolist()
        except Exception:
            pass
    # Path
    if isinstance(value, Path):
        return str(value)
    return value


def _json_default(obj):
    """Fallback serializer for json.dump to handle numpy/pandas scalar objects."""
    if hasattr(obj, "item") and callable(getattr(obj, "item")):
        try:
            return obj.item()
        except Exception:
            pass
    if hasattr(obj, "tolist") and callable(getattr(obj, "tolist")):
        try:
            return obj.tolist()
        except Exception:
            pass
    return str(obj)


class ExperimentLog:
    """Tracks config, metrics, and results for every modeling loop iteration."""

    def __init__(self) -> None:
        self._records: list[ExperimentRecord] = []

    @property
    def records(self) -> list[ExperimentRecord]:
        return list(self._records)

    def add(self, record: ExperimentRecord) -> None:
        self._records.append(record)
        log.info(
            "Experiment %d: %s k=%d sil=%.4f pass=%s",
            record.iteration, record.best_algorithm,
            record.n_clusters, record.silhouette, record.passed,
        )

    def to_dataframe(self) -> pd.DataFrame:
        if not self._records:
            return pd.DataFrame()
        rows = []
        for r in self._records:
            rows.append({
                "iteration": r.iteration,
                "timestamp": r.timestamp,
                "k": r.config_k,
                "eps": r.config_eps,
                "gmm_cov": r.config_gmm_cov,
                "best_algorithm": r.best_algorithm,
                "n_clusters": r.n_clusters,
                "silhouette": r.silhouette,
                "davies_bouldin": r.davies_bouldin,
                "calinski_harabasz": r.calinski_harabasz,
                "passed": r.passed,
                "reconfig_strategy": r.reconfiguration_strategy,
            })
        return pd.DataFrame(rows)

    def to_json(self, path: Path) -> None:
        data = [
            {
                "iteration": int(r.iteration),
                "timestamp": r.timestamp,
                "config": {
                    "k": int(r.config_k),
                    "eps": float(r.config_eps),
                    "gmm_cov": str(r.config_gmm_cov),
                },
                "best_algorithm": r.best_algorithm,
                "n_clusters": int(r.n_clusters),
                "silhouette": float(r.silhouette),
                "davies_bouldin": float(r.davies_bouldin),
                "calinski_harabasz": float(r.calinski_harabasz),
                "passed": bool(r.passed),
                "reconfiguration_strategy": r.reconfiguration_strategy,
            }
            for r in self._records
        ]
        data = _to_native(data)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, default=_json_default)
        log.info("Experiment log saved: %s", path)

    def summary(self) -> str:
        if not self._records:
            return "No experiments recorded."
        lines = ["Experiment Log Summary:", f"  Total iterations: {len(self._records)}"]
        for r in self._records:
            status = "PASS" if r.passed else "FAIL"
            strategy = f" (reconfig: {r.reconfiguration_strategy})" if r.reconfiguration_strategy else ""
            lines.append(
                f"  [{r.iteration}] {r.best_algorithm:8s} k={r.n_clusters} "
                f"sil={r.silhouette:.4f} DB={r.davies_bouldin:.4f} "
                f"{status}{strategy}"
            )
        best = max(self._records, key=lambda r: r.silhouette)
        lines.append(
            f"  Best: iter {best.iteration} ({best.best_algorithm}, sil={best.silhouette:.4f})"
        )
        return "\n".join(lines)
