"""Unit tests for output contract validator."""

from pathlib import Path
import pandas as pd
import pytest

from src.validator import validate_output_dataframe, validate_output_file, OutputValidationError


def test_validator_on_official_sample_file():
    """Verify that the official sample output file passes all validator checks."""
    sample_path = Path("data/sample_output_format_v2.csv")
    assert validate_output_file(sample_path) is True


def test_validator_on_generated_anomaly_report():
    """Verify that our pipeline's output passes validator checks."""
    report_path = Path("outputs/anomaly_report.csv")
    if report_path.exists():
        assert validate_output_file(report_path) is True


def test_validator_catches_wrong_columns():
    """Verify error on wrong column names or order."""
    df = pd.DataFrame([{
        "week_of": "2024-01-15",  # Swapped order!
        "route": "Mumbai-Pune",
        "cost_per_tonne_km": "3.50",
        "vs_own_history": "+10.0% vs this route's past average",
        "vs_similar_routes": "+5.0% vs similar-length routes this week",
        "flagged": "Yes",
        "matched_note_id": "",
        "reason": "Unexplained",
    }])
    is_valid, errors = validate_output_dataframe(df)
    assert is_valid is False
    assert any("Header mismatch" in e for e in errors)


def test_validator_catches_invalid_flagged_domain():
    """Verify error when flagged is not 'Yes' or 'No (justified)'."""
    df = pd.DataFrame([{
        "route": "Mumbai-Pune",
        "week_of": "2024-01-15",
        "cost_per_tonne_km": "3.50",
        "vs_own_history": "+10.0% vs this route's past average",
        "vs_similar_routes": "+5.0% vs similar-length routes this week",
        "flagged": "TRUE",  # Invalid!
        "matched_note_id": "",
        "reason": "Unexplained",
    }])
    is_valid, errors = validate_output_dataframe(df)
    assert is_valid is False
    assert any("flagged value" in e for e in errors)


def test_validator_catches_non_blank_note_id_on_unexplained():
    """When flagged='Yes', matched_note_id MUST be blank."""
    df = pd.DataFrame([{
        "route": "Mumbai-Pune",
        "week_of": "2024-01-15",
        "cost_per_tonne_km": "3.50",
        "vs_own_history": "+10.0% vs this route's past average",
        "vs_similar_routes": "+5.0% vs similar-length routes this week",
        "flagged": "Yes",
        "matched_note_id": "N001",  # Invalid! Must be blank for 'Yes'
        "reason": "Unexplained",
    }])
    is_valid, errors = validate_output_dataframe(df)
    assert is_valid is False
    assert any("matched_note_id must be blank when flagged='Yes'" in e for e in errors)


def test_validator_catches_sunday_week_of():
    """week_of must be a Monday."""
    df = pd.DataFrame([{
        "route": "Mumbai-Pune",
        "week_of": "2024-01-14",  # Sunday!
        "cost_per_tonne_km": "3.50",
        "vs_own_history": "+10.0% vs this route's past average",
        "vs_similar_routes": "+5.0% vs similar-length routes this week",
        "flagged": "Yes",
        "matched_note_id": "",
        "reason": "Unexplained",
    }])
    is_valid, errors = validate_output_dataframe(df)
    assert is_valid is False
    assert any("is not a Monday" in e for e in errors)
