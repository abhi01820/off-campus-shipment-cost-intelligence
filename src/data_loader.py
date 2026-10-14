"""Data loader and schema validator for Freight Tiger shipment records and context notes."""

import logging
from pathlib import Path
from typing import Tuple
import pandas as pd

logger = logging.getLogger(__name__)

REQUIRED_SHIPMENT_COLUMNS = [
    "shipment_id",
    "origin",
    "destination",
    "route_type",
    "material",
    "quantity_tonnes",
    "distance_km",
    "freight_cost_inr",
    "shipment_date",
    "transporter",
]

REQUIRED_NOTE_COLUMNS = [
    "note_id",
    "date",
    "applies_to",
    "note",
]

VALID_ROUTE_TYPES = {"Short", "Medium", "Long"}


class DataValidationError(Exception):
    """Raised when input data violates schema or business domain constraints."""
    pass


def load_shipment_records(file_path: str | Path) -> pd.DataFrame:
    """
    Load, validate, and clean shipment records CSV.
    
    Checks:
    - File existence
    - Missing required columns
    - Valid date formats
    - Positive quantities and distances
    - Non-negative freight costs
    - Valid route_type values
    - Drops exact duplicate shipment rows with warning
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Shipment records file not found at: {path}")

    # Read CSV (handles CRLF and UTF-8 cleanly)
    df = pd.read_csv(path, encoding="utf-8")

    # 1. Column presence check
    missing_cols = [c for c in REQUIRED_SHIPMENT_COLUMNS if c not in df.columns]
    if missing_cols:
        raise DataValidationError(f"Shipment records missing required columns: {missing_cols}")

    if df.empty:
        raise DataValidationError("Shipment records file is empty.")

    initial_len = len(df)

    # 2. Check and remove exact duplicates
    df = df.drop_duplicates()
    if len(df) < initial_len:
        logger.warning("Dropped %d exact duplicate shipment rows.", initial_len - len(df))

    # 3. Clean string columns
    for col in ["origin", "destination", "route_type", "material", "transporter"]:
        df[col] = df[col].astype(str).str.strip()

    # 4. Validate route types
    invalid_rtypes = set(df["route_type"].unique()) - VALID_ROUTE_TYPES
    if invalid_rtypes:
        raise DataValidationError(f"Invalid route_type values found: {invalid_rtypes}")

    # 5. Parse and validate dates
    try:
        df["shipment_date"] = pd.to_datetime(df["shipment_date"], format="%Y-%m-%d", errors="coerce")
    except Exception as e:
        raise DataValidationError(f"Error parsing shipment dates: {e}") from e

    bad_dates = df[df["shipment_date"].isna()]
    if not bad_dates.empty:
        raise DataValidationError(f"Found {len(bad_dates)} rows with unparseable shipment_date.")

    # 6. Validate numeric columns
    numeric_cols = ["quantity_tonnes", "distance_km", "freight_cost_inr"]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
        bad_num = df[df[col].isna()]
        if not bad_num.empty:
            raise DataValidationError(f"Found {len(bad_num)} non-numeric values in column {col}.")

    # Positive domain checks
    if (df["quantity_tonnes"] <= 0).any():
        bad_count = (df["quantity_tonnes"] <= 0).sum()
        raise DataValidationError(f"Found {bad_count} records with quantity_tonnes <= 0.")

    if (df["distance_km"] <= 0).any():
        bad_count = (df["distance_km"] <= 0).sum()
        raise DataValidationError(f"Found {bad_count} records with distance_km <= 0.")

    if (df["freight_cost_inr"] < 0).any():
        bad_count = (df["freight_cost_inr"] < 0).sum()
        raise DataValidationError(f"Found {bad_count} records with freight_cost_inr < 0.")

    # 7. Construct standard canonical route column: <origin>-<destination>
    df["route"] = df["origin"] + "-" + df["destination"]

    logger.info("Successfully loaded and validated %d shipment records across %d routes.",
                len(df), df["route"].nunique())
    return df


def load_context_notes(file_path: str | Path) -> pd.DataFrame:
    """
    Load, validate, and clean context notes CSV.
    
    Checks:
    - File existence
    - Missing required columns
    - Parseable dates
    - Non-empty note texts
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Context notes file not found at: {path}")

    df = pd.read_csv(path, encoding="utf-8")

    missing_cols = [c for c in REQUIRED_NOTE_COLUMNS if c not in df.columns]
    if missing_cols:
        raise DataValidationError(f"Context notes missing required columns: {missing_cols}")

    if df.empty:
        raise DataValidationError("Context notes file is empty.")

    # Clean text columns
    for col in ["note_id", "applies_to", "note"]:
        df[col] = df[col].astype(str).str.strip()

    # Parse note dates
    try:
        df["date"] = pd.to_datetime(df["date"], format="%Y-%m-%d", errors="coerce")
    except Exception as e:
        raise DataValidationError(f"Error parsing context note dates: {e}") from e

    bad_dates = df[df["date"].isna()]
    if not bad_dates.empty:
        raise DataValidationError(f"Found {len(bad_dates)} context notes with unparseable date.")

    # Unique note_id check
    if df["note_id"].duplicated().any():
        dups = df[df["note_id"].duplicated()]["note_id"].tolist()
        raise DataValidationError(f"Duplicate note_id values found: {dups}")

    logger.info("Successfully loaded and validated %d context notes.", len(df))
    return df


def load_all_data(shipment_path: str | Path, notes_path: str | Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Convenience function to load both datasets."""
    shipments = load_shipment_records(shipment_path)
    notes = load_context_notes(notes_path)
    return shipments, notes
