"""Unit tests for cost-per-tonne-km calculation and data validation edge cases."""

import tempfile
from pathlib import Path
import pandas as pd
import pytest

from src.data_loader import load_shipment_records, DataValidationError
from src.metrics import compute_weekly_metrics


def create_temp_csv(df: pd.DataFrame) -> Path:
    tmp = tempfile.NamedTemporaryFile(suffix=".csv", delete=False, mode="w", encoding="utf-8")
    df.to_csv(tmp.name, index=False)
    return Path(tmp.name)


def test_cost_per_tonne_km_weighted_aggregation():
    """
    Verify that weekly cost_per_tonne_km is computed as:
    sum(cost) / sum(quantity * distance), NOT the unweighted average of trip rates.
    """
    # Trip 1: 10 tonnes, 100 km (1000 tonne-km), cost 2000 INR -> rate 2.00
    # Trip 2: 2 tonnes, 50 km (100 tonne-km), cost 500 INR -> rate 5.00
    # Total cost = 2500, Total tonne-km = 1100 -> weighted rate = 2500 / 1100 = 2.2727...
    # (Unweighted average would have been (2.0 + 5.0) / 2 = 3.50, which is heavily distorted!)
    data = pd.DataFrame([{
        "shipment_id": "S1",
        "origin": "Mumbai",
        "destination": "Pune",
        "route_type": "Short",
        "material": "Steel",
        "quantity_tonnes": 10.0,
        "distance_km": 100.0,
        "freight_cost_inr": 2000,
        "shipment_date": "2024-01-15",
        "transporter": "T1",
    }, {
        "shipment_id": "S2",
        "origin": "Mumbai",
        "destination": "Pune",
        "route_type": "Short",
        "material": "Steel",
        "quantity_tonnes": 2.0,
        "distance_km": 50.0,
        "freight_cost_inr": 500,
        "shipment_date": "2024-01-16",
        "transporter": "T2",
    }])

    tmp_path = create_temp_csv(data)
    loaded_df = load_shipment_records(tmp_path)
    weekly = compute_weekly_metrics(loaded_df)

    expected_cost = 2500.0 / 1100.0
    actual_cost = weekly.iloc[0]["cost_per_tonne_km"]
    assert abs(actual_cost - expected_cost) < 1e-6


def test_data_validation_negative_or_zero_quantity():
    """Verify DataValidationError when quantity_tonnes <= 0."""
    data = pd.DataFrame([{
        "shipment_id": "S1",
        "origin": "Mumbai",
        "destination": "Pune",
        "route_type": "Short",
        "material": "Steel",
        "quantity_tonnes": 0.0,  # Invalid zero quantity
        "distance_km": 100.0,
        "freight_cost_inr": 2000,
        "shipment_date": "2024-01-15",
        "transporter": "T1",
    }])
    tmp_path = create_temp_csv(data)
    with pytest.raises(DataValidationError, match="quantity_tonnes <= 0"):
        load_shipment_records(tmp_path)


def test_data_validation_negative_or_zero_distance():
    """Verify DataValidationError when distance_km <= 0."""
    data = pd.DataFrame([{
        "shipment_id": "S1",
        "origin": "Mumbai",
        "destination": "Pune",
        "route_type": "Short",
        "material": "Steel",
        "quantity_tonnes": 10.0,
        "distance_km": -50.0,  # Invalid negative distance
        "freight_cost_inr": 2000,
        "shipment_date": "2024-01-15",
        "transporter": "T1",
    }])
    tmp_path = create_temp_csv(data)
    with pytest.raises(DataValidationError, match="distance_km <= 0"):
        load_shipment_records(tmp_path)


def test_data_validation_negative_cost():
    """Verify DataValidationError when freight_cost_inr < 0."""
    data = pd.DataFrame([{
        "shipment_id": "S1",
        "origin": "Mumbai",
        "destination": "Pune",
        "route_type": "Short",
        "material": "Steel",
        "quantity_tonnes": 10.0,
        "distance_km": 100.0,
        "freight_cost_inr": -100,  # Invalid negative freight cost
        "shipment_date": "2024-01-15",
        "transporter": "T1",
    }])
    tmp_path = create_temp_csv(data)
    with pytest.raises(DataValidationError, match="freight_cost_inr < 0"):
        load_shipment_records(tmp_path)


def test_data_validation_missing_required_column():
    """Verify DataValidationError when required columns are absent."""
    data = pd.DataFrame([{
        "shipment_id": "S1",
        "origin": "Mumbai",
        # Missing destination!
        "route_type": "Short",
        "material": "Steel",
        "quantity_tonnes": 10.0,
        "distance_km": 100.0,
        "freight_cost_inr": 2000,
        "shipment_date": "2024-01-15",
        "transporter": "T1",
    }])
    tmp_path = create_temp_csv(data)
    with pytest.raises(DataValidationError, match="missing required columns"):
        load_shipment_records(tmp_path)


def test_data_validation_duplicate_rows():
    """Verify duplicate rows are dropped with warning."""
    row = {
        "shipment_id": "S1",
        "origin": "Mumbai",
        "destination": "Pune",
        "route_type": "Short",
        "material": "Steel",
        "quantity_tonnes": 10.0,
        "distance_km": 100.0,
        "freight_cost_inr": 2000,
        "shipment_date": "2024-01-15",
        "transporter": "T1",
    }
    data = pd.DataFrame([row, row])  # Exact duplicate
    tmp_path = create_temp_csv(data)
    loaded = load_shipment_records(tmp_path)
    assert len(loaded) == 1
