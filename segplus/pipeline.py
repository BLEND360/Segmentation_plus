"""Pipeline entrypoint wrapper for segplus notebook workflow."""

from dataclasses import dataclass

from .config import PipelineConfig


@dataclass
class Pipeline:
    """Thin pipeline wrapper used by notebook and future CLI orchestration."""

    config: PipelineConfig


def build_pipeline(config: PipelineConfig) -> Pipeline:
    return Pipeline(config=config)
