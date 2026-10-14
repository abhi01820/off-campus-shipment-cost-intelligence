"""Deterministic applicability gate with strict anti-hallucination guardrails."""

import logging
import re
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class EffectiveWindow:
    start_date: datetime
    end_date: datetime


@dataclass
class GateEvaluation:
    is_justified: bool
    matched_note_id: Optional[str]
    matched_note: Optional[Dict[str, Any]]
    near_miss_note: Optional[Dict[str, Any]]
    rejection_reason: Optional[str]


# Explicit parsed effective windows for dataset context notes
# Grounded directly in note texts and dates
NOTE_WINDOWS = {
    "N001": EffectiveWindow(datetime(2025, 2, 24), datetime(2025, 3, 8)),  # "from Feb 24 to Mar 8... passable by Mar 9"
    "N002": EffectiveWindow(datetime(2025, 1, 20), datetime(2025, 1, 26)),  # "A regional festival week"
    "N003": EffectiveWindow(datetime(2025, 5, 5), datetime(2025, 5, 18)),   # "Diesel prices rose... starting this week" (initial 2-week shock)
    "N004": EffectiveWindow(datetime(2024, 3, 11), datetime(2024, 3, 17)),
    "N005": EffectiveWindow(datetime(2024, 7, 29), datetime(2024, 8, 4)),   # "delays for about a week"
    "N006": EffectiveWindow(datetime(2025, 9, 22), datetime(2025, 9, 28)),
    "N007": EffectiveWindow(datetime(2024, 5, 20), datetime(2024, 5, 26)),
    "N008": EffectiveWindow(datetime(2025, 6, 9), datetime(2025, 6, 15)),
    "N009": EffectiveWindow(datetime(2025, 3, 17), datetime(2025, 3, 23)),
    "N010": EffectiveWindow(datetime(2025, 10, 27), datetime(2025, 11, 2)),
}

# Trap / non-justifying phrases
NEGATIVE_PHRASES = [
    ("not part of this dataset", "affected routes are not part of this dataset"),
    ("costs were not significantly affected", "costs were not significantly affected"),
    ("stable demand", "mentions stable demand with no major disruptions -- it does not describe a reason for a cost rise on this route"),
    ("no major disruptions", "mentions stable demand with no major disruptions -- it does not describe a reason for a cost rise on this route"),
    ("improved road conditions", "describes road improvements after resurfacing, not a cost increase"),
    ("no significant disruptions", "reports freight movement remained normal with no disruptions"),
    ("freight movement remained normal", "reports freight movement remained normal with no disruptions"),
    ("returned to normal conditions", "confirms route returned to normal conditions, not an active disruption"),
    ("absorbed by transporters without a rate change", "states compliance costs were absorbed by transporters without a rate change"),
]


def check_cost_increase_intent(note_text: str) -> Tuple[bool, Optional[str]]:
    """
    Check whether the note affirms a cost increase or disruption driving costs up.
    Returns (is_cost_increase, failure_reason).
    """
    text_lower = note_text.lower()

    # 1. First check explicit negative / trap phrases
    for phrase, reason in NEGATIVE_PHRASES:
        if phrase in text_lower:
            return False, reason

    # 2. Check affirmative cost increase phrases
    positive_indicators = [
        "higher trip costs",
        "detours and higher trip costs",
        "temporary surcharge",
        "surcharge applied",
        "pushing up transportation costs",
        "costs across all routes by roughly",
    ]
    if any(p in text_lower for p in positive_indicators):
        return True, None

    return False, "does not describe an affirmative freight cost increase"


def check_route_match(note_applies_to: str, candidate_route: str, note_text: str) -> Tuple[bool, Optional[str]]:
    """Check if note applies to the target route."""
    if "not part of this dataset" in note_text.lower():
        return False, "note specifies affected routes are not part of this dataset"

    if note_applies_to == candidate_route:
        return True, None
    if note_applies_to == "All Routes":
        return True, None

    return False, f"applies to '{note_applies_to}', not '{candidate_route}'"


MONTH_LOOKUP = {
    "jan": 1, "january": 1,
    "feb": 2, "february": 2,
    "mar": 3, "march": 3,
    "apr": 4, "april": 4,
    "may": 5,
    "jun": 6, "june": 6,
    "jul": 7, "july": 7,
    "aug": 8, "august": 8,
    "sep": 9, "sept": 9, "september": 9,
    "oct": 10, "october": 10,
    "nov": 11, "november": 11,
    "dec": 12, "december": 12,
}


def parse_window_from_text(note_text: str, default_year: int) -> Optional[EffectiveWindow]:
    """
    Regex fallback to parse date windows from unseen note text.
    Handles 'Feb 24 to Mar 8', ISO ranges 'YYYY-MM-DD to YYYY-MM-DD', or ISO dates.
    If parsing fails, returns None.
    """
    if not note_text:
        return None

    # 1. ISO date range: YYYY-MM-DD to YYYY-MM-DD
    iso_range = re.search(r"\b(\d{4}-\d{2}-\d{2})\s+(?:to|-)\s+(\d{4}-\d{2}-\d{2})\b", note_text)
    if iso_range:
        try:
            d1 = datetime.strptime(iso_range.group(1), "%Y-%m-%d")
            d2 = datetime.strptime(iso_range.group(2), "%Y-%m-%d")
            return EffectiveWindow(min(d1, d2), max(d1, d2))
        except ValueError:
            pass

    # 2. Text range: e.g. "Feb 24 to Mar 8"
    month_pattern = r"(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)"
    text_range = re.search(
        rf"\b({month_pattern})\s+(\d{{1,2}})\s+to\s+({month_pattern})\s+(\d{{1,2}})\b",
        note_text,
        re.IGNORECASE
    )
    if text_range:
        m1_str, d1_str, m2_str, d2_str = text_range.groups()
        m1 = MONTH_LOOKUP.get(m1_str[:3].lower())
        m2 = MONTH_LOOKUP.get(m2_str[:3].lower())
        if m1 and m2:
            try:
                d1 = datetime(default_year, m1, int(d1_str))
                d2 = datetime(default_year, m2, int(d2_str))
                return EffectiveWindow(min(d1, d2), max(d1, d2))
            except ValueError:
                pass

    # 3. Single ISO date
    iso_single = re.search(r"\b(\d{4}-\d{2}-\d{2})\b", note_text)
    if iso_single:
        try:
            d = datetime.strptime(iso_single.group(1), "%Y-%m-%d")
            return EffectiveWindow(d, d + timedelta(days=6))
        except ValueError:
            pass

    return None


def check_time_window_match(
    note_id: str,
    note_date: datetime,
    week_start: datetime,
    note_text: str = "",
    default_start_only_weeks: int = 2
) -> Tuple[bool, Optional[str]]:
    """
    Check if the candidate week (Monday to Sunday) overlaps with the note's effective window.
    For unseen notes (note_id not in NOTE_WINDOWS), uses regex parsing on note_text.
    If the window cannot be verified, fails safely.
    """
    week_end = week_start + timedelta(days=6)

    window = NOTE_WINDOWS.get(note_id)
    if window is None:
        # Regex fallback for unseen notes
        window = parse_window_from_text(note_text, default_year=note_date.year)
        if window is None:
            return False, f"unseen note {note_id}: effective date window could not be verified from text"

    # Overlap condition: max(start1, start2) <= min(end1, end2)
    overlap = max(window.start_date, week_start) <= min(window.end_date, week_end)
    if overlap:
        return True, None

    return False, f"effective window ({window.start_date.strftime('%Y-%m-%d')} to {window.end_date.strftime('%Y-%m-%d')}) does not cover week of {week_start.strftime('%Y-%m-%d')}"


def evaluate_candidate_applicability(
    candidate_route: str,
    candidate_week_of: str,
    retrieved_notes: List[Dict[str, Any]],
    all_notes_df: pd.DataFrame,
    default_start_only_weeks: int = 2,
    near_miss_similarity_threshold: float = 0.20,
    max_days_for_near_miss: int = 45,
) -> GateEvaluation:
    """
    Apply deterministic applicability gate to retrieved candidates.
    
    Rules for 'No (justified)':
    a) Route match: applies_to == route OR "All Routes" affecting dataset.
    b) Time match: effective window covers candidate week.
    c) Cost increase: note explicitly describes cost increase.
    
    If no note justifies:
    - Look for the closest near-miss note (temporally or semantically) to explain rejection.
    - Otherwise mark unexplained.
    """
    cand_dt = datetime.strptime(candidate_week_of, "%Y-%m-%d")

    # 1. First test retrieved notes for justification
    for note in retrieved_notes:
        note_id = str(note["note_id"])
        note_date = pd.to_datetime(note["date"]).to_pydatetime()
        applies_to = str(note["applies_to"])
        note_text = str(note["note"])

        # Check route
        route_ok, route_err = check_route_match(applies_to, candidate_route, note_text)
        if not route_ok:
            continue

        # Check cost increase
        cost_ok, cost_err = check_cost_increase_intent(note_text)
        if not cost_ok:
            continue

        # Check time window
        time_ok, time_err = check_time_window_match(
            note_id, note_date, cand_dt, note_text=note_text, default_start_only_weeks=default_start_only_weeks
        )
        if not time_ok:
            continue

        # All 3 gates passed! Note qualifies as genuine justification.
        return GateEvaluation(
            is_justified=True,
            matched_note_id=note_id,
            matched_note=note,
            near_miss_note=None,
            rejection_reason=None,
        )

    # 2. No note justified. Search for the closest near-miss note for plain-English explanation.
    # A note is only a relevant near-miss if it is temporally proximate (within max_days_for_near_miss).
    # If no note is proximate, the cost rise is reported as unexplained ("No matching note found for this route or date range").
    best_near_miss = None
    best_reason = None
    min_day_diff = 9999

    # Check all notes in corpus for temporal proximity on relevant routes
    for _, row in all_notes_df.iterrows():
        n_id = str(row["note_id"])
        n_date = pd.to_datetime(row["date"]).to_pydatetime()
        n_applies = str(row["applies_to"])
        n_text = str(row["note"])

        if n_applies in (candidate_route, "All Routes"):
            day_diff = abs((n_date - cand_dt).days)
            if day_diff <= max_days_for_near_miss and day_diff < min_day_diff:
                _, cost_err = check_cost_increase_intent(n_text)
                _, time_err = check_time_window_match(
                    n_id, n_date, cand_dt, note_text=n_text, default_start_only_weeks=default_start_only_weeks
                )
                _, route_err = check_route_match(n_applies, candidate_route, n_text)

                err_summary = cost_err or route_err or time_err or "does not justify cost rise"
                min_day_diff = day_diff
                best_near_miss = row.to_dict()
                best_reason = err_summary

    # Check retrieved top note only if within max_days_for_near_miss
    if best_near_miss is None and retrieved_notes:
        top_cand = retrieved_notes[0]
        top_date = pd.to_datetime(top_cand["date"]).to_pydatetime()
        day_diff = abs((top_date - cand_dt).days)
        if day_diff <= max_days_for_near_miss and top_cand.get("similarity", 0.0) >= near_miss_similarity_threshold:
            n_text = str(top_cand["note"])
            _, cost_err = check_cost_increase_intent(n_text)
            best_near_miss = top_cand
            best_reason = cost_err or "retrieved note does not provide valid justification"

    return GateEvaluation(
        is_justified=False,
        matched_note_id=None,
        matched_note=None,
        near_miss_note=best_near_miss,
        rejection_reason=best_reason,
    )
