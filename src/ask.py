"""Stretch Goal: Grounded natural-language Q&A assistant over computed results and notes."""

import argparse
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple
import pandas as pd

from src.config import load_config
from src.data_loader import load_context_notes
from src.pipeline import run_pipeline

MONTH_MAP = {
    "january": 1, "jan": 1,
    "february": 2, "feb": 2,
    "march": 3, "mar": 3,
    "april": 4, "apr": 4,
    "may": 5,
    "june": 6, "jun": 6,
    "july": 7, "jul": 7,
    "august": 8, "aug": 8,
    "september": 9, "sep": 9, "sept": 9,
    "october": 10, "oct": 10,
    "november": 11, "nov": 11,
    "december": 12, "dec": 12,
}

KNOWN_ROUTES = [
    "Chennai-Bangalore",
    "Ahmedabad-Mumbai",
    "Delhi-Jaipur",
    "Mumbai-Pune",
    "Mumbai-Delhi",
    "Delhi-Chennai",
    "Kolkata-Bhubaneswar",
]


def extract_route(query: str) -> Optional[str]:
    """Find known route in query text."""
    q_norm = query.lower().replace(" ", "").replace("_", "-")
    for r in KNOWN_ROUTES:
        r_norm = r.lower().replace("-", "")
        if r_norm in q_norm or r.lower() in query.lower():
            return r
    return None


def extract_month_year(query: str) -> Tuple[Optional[int], Optional[int]]:
    """Extract month number and 4-digit year from query."""
    q_lower = query.lower()
    
    # Year
    year_match = re.search(r"\b(202[456])\b", q_lower)
    year = int(year_match.group(1)) if year_match else None

    # Month
    month = None
    for m_name, m_num in MONTH_MAP.items():
        if re.search(rf"\b{m_name}\b", q_lower):
            month = m_num
            break

    return month, year


def answer_query(query: str, report_csv_path: str = "outputs/anomaly_report.csv") -> str:
    """Answer question strictly from computed results and context notes."""
    report_path = Path(report_csv_path)
    if not report_path.exists():
        # Run pipeline if report not found
        run_pipeline()

    df = pd.read_csv(report_path)
    notes_df = load_context_notes("data/context_notes.csv")

    target_route = extract_route(query)
    target_month, target_year = extract_month_year(query)

    if not target_route:
        # Check if query asks about all routes or diesel
        if "all routes" in query.lower() or "diesel" in query.lower():
            target_route = "All Routes"
        else:
            return "No information found: Could not identify a specific recognized shipping route in your query."

    # Filter report by route and date
    filtered_rows = []
    for _, row in df.iterrows():
        r = row["route"]
        w = row["week_of"]
        w_dt = datetime.strptime(w, "%Y-%m-%d")

        route_match = (r == target_route) or (target_route == "All Routes")
        date_match = True
        if target_year and w_dt.year != target_year:
            date_match = False
        if target_month and w_dt.month != target_month:
            date_match = False

        if route_match and date_match:
            filtered_rows.append(row)

    if not filtered_rows:
        return (
            f"No information found for route '{target_route}' "
            f"during {f'{target_month}/' if target_month else ''}{target_year or ''}."
        )

    # Format findings
    lines = [f"Found {len(filtered_rows)} record(s) matching '{target_route}':"]
    for row in filtered_rows:
        lines.append(
            f"\n• Week of {row['week_of']}:\n"
            f"  - Cost: INR {row['cost_per_tonne_km']} per tonne-km\n"
            f"  - Change: {row['vs_own_history']} | {row['vs_similar_routes']}\n"
            f"  - Anomaly Flag: {row['flagged']}\n"
            f"  - Matched Note ID: {row['matched_note_id'] if pd.notna(row['matched_note_id']) and row['matched_note_id'] != '' else 'None'}\n"
            f"  - Rationale: {row['reason']}"
        )

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Ask questions about shipping costs and anomalies")
    parser.add_argument("query", type=str, help="Question in quotes (e.g. 'why did Chennai-Bangalore get pricier in March 2025?')")
    parser.add_argument("--report", default="outputs/anomaly_report.csv", help="Path to report CSV")

    args = parser.parse_args()
    answer = answer_query(args.query, report_csv_path=args.report)
    print("\n" + answer + "\n")


if __name__ == "__main__":
    main()
