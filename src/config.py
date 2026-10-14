"""Configuration management for the Freight Tiger Anomaly Detection system."""

from pathlib import Path
from typing import Literal, Optional
import yaml
from pydantic import BaseModel, Field


class DataConfig(BaseModel):
    shipment_records_path: str = "data/shipment_records.csv"
    context_notes_path: str = "data/context_notes.csv"
    sample_output_path: str = "data/sample_output_format_v2.csv"


class OutputConfig(BaseModel):
    output_csv_path: str = "outputs/anomaly_report.csv"
    export_only_candidates: bool = True


class AnomalyDetectionConfig(BaseModel):
    history_window_weeks: int = Field(default=8, ge=1)
    threshold_pct_vs_own: float = Field(default=9.0, ge=0.0)
    threshold_pct_vs_peers: float = Field(default=20.0, ge=0.0)
    combination_logic: Literal["OR", "AND"] = "OR"


class RetrievalConfig(BaseModel):
    model_name: str = "all-MiniLM-L6-v2"
    top_k: int = Field(default=3, ge=1)
    similarity_threshold: float = Field(default=0.15, ge=0.0, le=1.0)


class ApplicabilityConfig(BaseModel):
    default_start_only_window_weeks: int = Field(default=2, ge=1)
    near_miss_similarity_threshold: float = Field(default=0.20, ge=0.0, le=1.0)
    max_days_for_near_miss: int = Field(default=14, ge=1)


class ExplanationConfig(BaseModel):
    mode: Literal["template", "llm"] = "template"
    llm_model: str = "gpt-4o-mini"
    llm_temperature: float = Field(default=0.0, ge=0.0, le=1.0)


class AppConfig(BaseModel):
    data: DataConfig = Field(default_factory=DataConfig)
    output: OutputConfig = Field(default_factory=OutputConfig)
    anomaly_detection: AnomalyDetectionConfig = Field(default_factory=AnomalyDetectionConfig)
    retrieval: RetrievalConfig = Field(default_factory=RetrievalConfig)
    applicability: ApplicabilityConfig = Field(default_factory=ApplicabilityConfig)
    explanation: ExplanationConfig = Field(default_factory=ExplanationConfig)


def load_config(config_path: Optional[str | Path] = None) -> AppConfig:
    """Load configuration from a YAML file, falling back to defaults if not found."""
    if config_path is None:
        config_path = Path("config.yaml")
    else:
        config_path = Path(config_path)

    if not config_path.exists():
        return AppConfig()

    with open(config_path, "r", encoding="utf-8") as f:
        raw_dict = yaml.safe_load(f) or {}

    return AppConfig(**raw_dict)
