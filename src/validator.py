"""Validator verifying output CSV format, headers, regexes, and value domains."""

import logging
import re
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple
import pandas as pd

logger = logging.getLogger(__name__)

EXPECTED_COLUMNS = [
    "route",
    "week_of",
    "cost_per_tonne_km",
    "vs_own_history",
    "vs_similar_routes",
    "flagged",
    "matched_note_id",
    "reason",
]

VALID_FLAGGED_DOMAINS = {"Yes", "No (justified)"}

OWN_HIST_PATTERN = re.compile(r"^[+-]\d+\.\d% vs this route's past average$")
PEER_HIST_PATTERN = re.compile(r"^[+-]\d+\.\d% vs similar-length routes this week$")
NOTE_ID_PATTERN = re.compile(r"^N\d{3}$")
TWO_DECIMAL_PATTERN = re.compile(r"^\d+\.\d{2}$")


class OutputValidationError(Exception):
    """Raised when output CSV fails contract specifications."""
    pass


def validate_output_dataframe(df: pd.DataFrame) -> Tuple[bool, List[str]]:
    """
    Validate a candidate output DataFrame against the strict output contract.
    Returns (is_valid, list_of_errors).
    """
    errors: List[str] = []

    # 1. Column presence and exact order
    actual_cols = list(df.columns)
    if actual_cols != EXPECTED_COLUMNS:
        errors.append(f"Header mismatch. Expected {EXPECTED_COLUMNS}, but got {actual_cols}")
        return False, errors

    if df.empty:
        errors.append("Output DataFrame is empty.")
        return False, errors

    for idx, row in df.iterrows():
        row_num = idx + 2  # 1-indexed header + row

        # 2. Route format: <Origin>-<Destination>
        route_str = str(row["route"]).strip()
        if "-" not in route_str or len(route_str.split("-")) != 2:
            errors.append(f"Row {row_num}: Invalid route format '{route_str}'. Expected '<Origin>-<Destination>'.")

        # 3. week_of format and Monday check
        week_str = str(row["week_of"]).strip()
        try:
            w_dt = datetime.strptime(week_str, "%Y-%m-%d")
            if w_dt.weekday() != 0:
                errors.append(f"Row {row_num}: week_of '{week_str}' is not a Monday (weekday {w_dt.weekday()}).")
        except ValueError:
            errors.append(f"Row {row_num}: week_of '{week_str}' is not valid ISO 'YYYY-MM-DD'.")

        # 4. cost_per_tonne_km: 2 decimal representation
        cost_val = row["cost_per_tonne_km"]
        cost_str = str(cost_val).strip()
        if not TWO_DECIMAL_PATTERN.match(cost_str):
            # Check if float rounded to 2 decimals
            try:
                f_val = float(cost_val)
                if f_val <= 0:
                    errors.append(f"Row {row_num}: cost_per_tonne_km must be positive, got {cost_val}")
            except ValueError:
                errors.append(f"Row {row_num}: cost_per_tonne_km '{cost_val}' is not a valid 2-decimal number.")

        # 5. vs_own_history string format
        own_str = str(row["vs_own_history"]).strip()
        if not (OWN_HIST_PATTERN.match(own_str) or own_str.startswith("N/A")):
            errors.append(f"Row {row_num}: vs_own_history '{own_str}' does not match expected format '+XX.X% vs this route\'s past average'")

        # 6. vs_similar_routes string format
        peer_str = str(row["vs_similar_routes"]).strip()
        if not (PEER_HIST_PATTERN.match(peer_str) or peer_str.startswith("N/A")):
            errors.append(f"Row {row_num}: vs_similar_routes '{peer_str}' does not match expected format '+XX.X% vs similar-length routes this week'")

        # 7. flagged domain
        flag_val = str(row["flagged"]).strip()
        if flag_val not in VALID_FLAGGED_DOMAINS:
            errors.append(f"Row {row_num}: flagged value '{flag_val}' must be strictly in {VALID_FLAGGED_DOMAINS}")

        # 8. matched_note_id rules
        raw_note_id = row["matched_note_id"]
        note_id_str = "" if pd.isna(raw_note_id) else str(raw_note_id).strip()

        if flag_val == "Yes":
            if note_id_str != "":
                errors.append(f"Row {row_num}: matched_note_id must be blank when flagged='Yes', got '{note_id_str}'")
        elif flag_val == "No (justified)":
            if not NOTE_ID_PATTERN.match(note_id_str):
                errors.append(f"Row {row_num}: matched_note_id must be a valid note ID (e.g. N001) when flagged='No (justified)', got '{note_id_str}'")

        # 9. reason string
        reason_str = str(row["reason"]).strip()
        if not reason_str:
            errors.append(f"Row {row_num}: reason cannot be empty.")
        elif flag_val == "No (justified)" and note_id_str not in reason_str:
            errors.append(f"Row {row_num}: reason must cite matched_note_id '{note_id_str}', got '{reason_str}'")

    is_valid = len(errors) == 0
    return is_valid, errors


def load_csv_flexibly(path: Path) -> pd.DataFrame:
    """Load CSV supporting both standard RFC 4180 quotes and unquoted legacy sample files."""
    try:
        return pd.read_csv(path, dtype=str)
    except Exception:
        with open(path, "r", encoding="utf-8") as f:
            lines = [l.strip() for l in f if l.strip()]
        if not lines:
            return pd.DataFrame()
        header = [c.strip() for c in lines[0].split(",")]
        rows = []
        for line in lines[1:]:
            parts = line.split(",", len(header) - 1)
            row_dict = {header[i]: parts[i].strip().strip('"') if i < len(parts) else "" for i in range(len(header))}
            rows.append(row_dict)
        return pd.DataFrame(rows)


def validate_output_file(file_path: str | Path) -> bool:
    """Validate a written CSV file on disk against contract requirements."""
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"File to validate not found: {path}")

    # Read flexibly
    df = load_csv_flexibly(path)
    is_valid, errors = validate_output_dataframe(df)

    if not is_valid:
        logger.error("Validation failed with %d errors:", len(errors))
        for err in errors[:10]:
            logger.error("  - %s", err)
        if len(errors) > 10:
            logger.error("  ... and %d more errors", len(errors) - 10)
        raise OutputValidationError(f"CSV contract validation failed with {len(errors)} errors.")

    logger.info("Output file '%s' passed all contract schema and domain validations.", path)
    return True
