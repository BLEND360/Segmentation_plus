"""
Segmentation Plus — Pipeline Orchestrator
==========================================
End-to-end pipeline: Data Import → EDA → Cluster → Evaluate → Personas → Dashboard.

Usage:
    python main.py --data path/to/data.csv --domain financial_services
    python main.py --data path/to/data.csv --domain retail --skip-personas
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

# ──────────────────────────────────────────────
# Logging Setup
# ──────────────────────────────────────────────

def setup_logging(verbose: bool = False) -> None:
    """Configure logging for the pipeline."""
    level = logging.DEBUG if verbose else logging.INFO
    fmt = "%(asctime)s | %(levelname)-8s | %(name)-25s | %(message)s"
    logging.basicConfig(level=level, format=fmt, datefmt="%H:%M:%S")
    # Suppress noisy libraries
    logging.getLogger("matplotlib").setLevel(logging.WARNING)
    logging.getLogger("PIL").setLevel(logging.WARNING)

logger = logging.getLogger("segmentation_plus")


# ──────────────────────────────────────────────
# Pipeline
# ──────────────────────────────────────────────

def run_pipeline(
    data_path: str,
    domain_key: str,
    output_base: str = "output",
    skip_personas: bool = False,
    k_range: tuple = (2, 10),
    pca_variance: float = 0.85,
    verbose: bool = False,
) -> dict:
    """
    Run the full segmentation pipeline.

    Parameters
    ----------
    data_path : str
        Path to the input data file (CSV/Excel/Parquet).
    domain_key : str
        Domain config to use (e.g., 'financial_services', 'retail').
    output_base : str
        Base output directory (default: 'output').
    skip_personas : bool
        If True, skip GenAI persona generation (useful if no API key).
    k_range : tuple
        Min and max K for clustering (default: 2 to 10).
    pca_variance : float
        Cumulative variance threshold for PCA (default: 0.85).
    verbose : bool
        Enable debug logging.

    Returns
    -------
    dict
        Pipeline results including cluster labels, evaluation, and personas.
    """
    setup_logging(verbose)
    results = {}

    # ── Setup directories ──
    output_dir = Path(output_base)
    eda_dir = output_dir / "eda"
    eval_dir = output_dir / "evaluation"
    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)

    logger.info("=" * 60)
    logger.info("SEGMENTATION PLUS — Pipeline Start")
    logger.info("  Data: %s", data_path)
    logger.info("  Domain: %s", domain_key)
    logger.info("  Output: %s", output_dir)
    logger.info("=" * 60)

    # ────────────────────────────────────────
    # STEP 1: Load Domain Config
    # ────────────────────────────────────────
    logger.info("\n📋 Step 1: Loading domain configuration...")
    from config import load_domain_config
    domain_config = load_domain_config(domain_key)
    logger.info("  Domain: %s", domain_config.display_name)

    # ────────────────────────────────────────
    # STEP 2: Import & Validate Data
    # ────────────────────────────────────────
    logger.info("\n📥 Step 2: Importing and validating data...")
    from data_import import import_and_validate
    df, quality_report = import_and_validate(data_path, domain_config)
    logger.info("\n%s", quality_report.summary())

    if not quality_report.passed:
        logger.error("Data validation FAILED. Fix the errors above and retry.")
        sys.exit(1)

    results["data_shape"] = df.shape
    results["quality_report"] = quality_report

    # ────────────────────────────────────────
    # STEP 3: EDA & Feature Engineering
    # ────────────────────────────────────────
    logger.info("\n🔍 Step 3: Running EDA & feature engineering...")
    from eda_preprocessing import EDAEngine, FeatureEngineer, Preprocessor

    # EDA
    eda = EDAEngine(df, domain_config, eda_dir)
    eda_results = eda.run_all()

    # Feature Engineering
    fe = FeatureEngineer(domain_config)
    df_engineered = fe.engineer_features(df)

    # Preprocessing
    preprocessor = Preprocessor(domain_config)
    df_processed, X_scaled, pca = preprocessor.fit_transform(
        df_engineered,
        pca_variance_threshold=pca_variance,
    )

    feature_names = preprocessor.feature_columns
    results["n_features"] = len(feature_names)
    results["pca_components"] = pca.n_components_ if pca else None

    # Save processed data
    df_processed.to_csv(data_dir / "processed_customers.csv", index=False)
    logger.info("  Processed data saved: %d rows × %d features", len(df_processed), len(feature_names))

    # ────────────────────────────────────────
    # STEP 4: Clustering
    # ────────────────────────────────────────
    logger.info("\n🎯 Step 4: Running clustering algorithms...")
    from clustering import ClusteringEngine

    engine = ClusteringEngine(X_scaled, k_range=k_range)
    all_results = engine.run_all()

    # Select best model
    best = engine.get_best_model(metric="silhouette")
    labels = best.labels
    logger.info("  Best model: %s (K=%d, Silhouette=%.3f)",
                best.algorithm, best.n_clusters, best.metrics.get("silhouette", 0))

    results["best_algorithm"] = best.algorithm
    results["n_clusters"] = best.n_clusters
    results["cluster_metrics"] = best.metrics

    # Save clustered data
    df_processed["cluster"] = labels
    df_processed.to_csv(data_dir / "clustered_customers.csv", index=False)
    logger.info("  Clustered data saved")

    # ────────────────────────────────────────
    # STEP 5: Cluster Evaluation
    # ────────────────────────────────────────
    logger.info("\n🔬 Step 5: Evaluating clusters...")
    from cluster_evaluation import ClusterEvaluator

    evaluator = ClusterEvaluator(X_scaled, labels, feature_names, eval_dir)
    eval_report = evaluator.run_full_evaluation(pca=pca, k_range=k_range)
    results["evaluation"] = eval_report

    # ────────────────────────────────────────
    # STEP 6: GenAI Persona Generation
    # ────────────────────────────────────────
    personas = []
    if not skip_personas:
        logger.info("\n🤖 Step 6: Generating personas with GenAI...")
        from persona_generation import PersonaGenerator, build_cluster_profiles

        profiles = build_cluster_profiles(
            df_processed, labels, feature_names,
            shap_importance=eval_report.get("shap_importance", {}).get("feature_importance"),
        )

        try:
            generator = PersonaGenerator(domain_config, output_dir)
            personas = generator.generate_personas(profiles)
            results["personas"] = personas
            logger.info("  Generated %d personas", len(personas))
        except Exception as e:
            logger.warning("Persona generation failed: %s (continuing without personas)", e)
    else:
        logger.info("\n⏭️ Step 6: Skipping persona generation (--skip-personas)")

    # ────────────────────────────────────────
    # STEP 7: Dashboard
    # ────────────────────────────────────────
    logger.info("\n📊 Step 7: Building dashboard...")
    from dashboard import DashboardBuilder

    # Prepare 2D projection for scatter plot
    X_viz = X_scaled[:, :2] if X_scaled.shape[1] >= 2 else X_scaled

    builder = DashboardBuilder(output_dir)
    dashboard_path = builder.build(
        df=df_processed,
        labels=labels,
        X_for_viz=X_viz,
        feature_names=feature_names,
        personas=personas,
        evaluation_report=eval_report,
    )
    results["dashboard_path"] = str(dashboard_path)

    # ────────────────────────────────────────
    # Done
    # ────────────────────────────────────────
    logger.info("\n" + "=" * 60)
    logger.info("✅ PIPELINE COMPLETE")
    logger.info("  Clusters: %d (%s)", best.n_clusters, best.algorithm)
    logger.info("  Silhouette: %.3f", best.metrics.get("silhouette", 0))
    logger.info("  Dashboard: %s", dashboard_path)
    logger.info("=" * 60)

    return results


# ──────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────

def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Segmentation Plus — Customer Segmentation & Persona Generation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --data data/customers.csv --domain financial_services
  python main.py --data data/sales.xlsx --domain retail --skip-personas
  python main.py --data data/patients.csv --domain healthcare --k-min 3 --k-max 8
        """,
    )

    parser.add_argument("--data", required=True, help="Path to input data file (CSV/Excel/Parquet)")
    parser.add_argument("--domain", required=True, help="Domain config name (e.g., financial_services, retail)")
    parser.add_argument("--output", default="output", help="Output directory (default: output)")
    parser.add_argument("--skip-personas", action="store_true", help="Skip GenAI persona generation")
    parser.add_argument("--k-min", type=int, default=2, help="Minimum K for clustering (default: 2)")
    parser.add_argument("--k-max", type=int, default=10, help="Maximum K for clustering (default: 10)")
    parser.add_argument("--pca-variance", type=float, default=0.85, help="PCA variance threshold (default: 0.85)")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable debug logging")

    args = parser.parse_args()

    run_pipeline(
        data_path=args.data,
        domain_key=args.domain,
        output_base=args.output,
        skip_personas=args.skip_personas,
        k_range=(args.k_min, args.k_max),
        pca_variance=args.pca_variance,
        verbose=args.verbose,
    )


if __name__ == "__main__":
    main()
