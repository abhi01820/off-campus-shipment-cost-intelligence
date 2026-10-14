"""Unit tests for deterministic applicability gate and anti-hallucination guardrails."""

import pandas as pd
import pytest

from src.data_loader import load_context_notes
from src.applicability_gate import (
    evaluate_candidate_applicability,
    check_cost_increase_intent,
    check_route_match,
    check_time_window_match,
)


@pytest.fixture
def notes_df():
    return load_context_notes("data/context_notes.csv")


def test_positive_justification_n001_chennai_bangalore(notes_df):
    """N001 must justify Chennai-Bangalore for weeks within Feb 24 - Mar 8, 2025."""
    n001 = notes_df[notes_df["note_id"] == "N001"].iloc[0].to_dict()

    # Week 1: 2025-02-24 (Monday) -> in window
    eval1 = evaluate_candidate_applicability(
        candidate_route="Chennai-Bangalore",
        candidate_week_of="2025-02-24",
        retrieved_notes=[n001],
        all_notes_df=notes_df,
    )
    assert eval1.is_justified is True
    assert eval1.matched_note_id == "N001"

    # Week 2: 2025-03-03 (Monday) -> covers Mar 3 to Mar 9 -> in window (disruption till Mar 8)
    eval2 = evaluate_candidate_applicability(
        candidate_route="Chennai-Bangalore",
        candidate_week_of="2025-03-03",
        retrieved_notes=[n001],
        all_notes_df=notes_df,
    )
    assert eval2.is_justified is True
    assert eval2.matched_note_id == "N001"


def test_positive_justification_n001_window_edge(notes_df):
    """N001 must NOT justify after Mar 8 (e.g. week of 2025-03-10 onwards)."""
    n001 = notes_df[notes_df["note_id"] == "N001"].iloc[0].to_dict()

    eval_after = evaluate_candidate_applicability(
        candidate_route="Chennai-Bangalore",
        candidate_week_of="2025-03-10",
        retrieved_notes=[n001],
        all_notes_df=notes_df,
    )
    assert eval_after.is_justified is False
    assert eval_after.matched_note_id is None


def test_positive_justification_n002_ahmedabad_mumbai(notes_df):
    """N002 must justify Ahmedabad-Mumbai on week of 2025-01-20."""
    n002 = notes_df[notes_df["note_id"] == "N002"].iloc[0].to_dict()

    eval_ok = evaluate_candidate_applicability(
        candidate_route="Ahmedabad-Mumbai",
        candidate_week_of="2025-01-20",
        retrieved_notes=[n002],
        all_notes_df=notes_df,
    )
    assert eval_ok.is_justified is True
    assert eval_ok.matched_note_id == "N002"

    # Must NOT justify wrong route
    eval_wrong_route = evaluate_candidate_applicability(
        candidate_route="Mumbai-Pune",
        candidate_week_of="2025-01-20",
        retrieved_notes=[n002],
        all_notes_df=notes_df,
    )
    assert eval_wrong_route.is_justified is False

    # Must NOT justify wrong date
    eval_wrong_date = evaluate_candidate_applicability(
        candidate_route="Ahmedabad-Mumbai",
        candidate_week_of="2025-02-17",
        retrieved_notes=[n002],
        all_notes_df=notes_df,
    )
    assert eval_wrong_date.is_justified is False


def test_all_trap_notes_strictly_rejected(notes_df):
    """
    Verify that ALL 7 trap notes (N004, N005, N006, N007, N008, N009, N010)
    are strictly rejected and NEVER produce a 'justified' verdict.
    """
    traps = [
        ("N004", "Delhi-Jaipur", "2024-03-11"),
        ("N005", "Mumbai-Delhi", "2024-07-29"),
        ("N006", "Mumbai-Pune", "2025-09-22"),
        ("N007", "Delhi-Jaipur", "2024-05-20"),
        ("N008", "Kolkata-Bhubaneswar", "2025-06-09"),
        ("N009", "Chennai-Bangalore", "2025-03-17"),
        ("N010", "Mumbai-Delhi", "2025-10-27"),
    ]

    for note_id, route, week_of in traps:
        note_row = notes_df[notes_df["note_id"] == note_id].iloc[0].to_dict()
        res = evaluate_candidate_applicability(
            candidate_route=route,
            candidate_week_of=week_of,
            retrieved_notes=[note_row],
            all_notes_df=notes_df,
        )
        assert res.is_justified is False, f"Trap note {note_id} was incorrectly marked justified!"
        assert res.matched_note_id is None


def test_unseen_note_without_window_is_unexplained(notes_df):
    """
    Unseen note not in NOTE_WINDOWS with unparseable text must never crash
    and must produce is_justified=False with matched_note_id=None.
    """
    unseen_note = {
        "note_id": "N999",
        "date": "2025-06-01",
        "applies_to": "Mumbai-Pune",
        "note": "Some vague disruption occurred without identifiable date windows.",
    }
    res = evaluate_candidate_applicability(
        candidate_route="Mumbai-Pune",
        candidate_week_of="2025-06-02",
        retrieved_notes=[unseen_note],
        all_notes_df=notes_df,
    )
    assert res.is_justified is False
    assert res.matched_note_id is None


def test_unseen_note_regex_date_parsing_fallback(notes_df):
    """
    Unseen note with parseable 'Month Day to Month Day' date range and affirmative cost increase.
    """
    # Parseable window Feb 24 to Mar 8, 2025 with cost increase
    valid_unseen = {
        "note_id": "N987",
        "date": "2025-02-24",
        "applies_to": "Mumbai-Pune",
        "note": "Highway landslide from Feb 24 to Mar 8 forced detours and higher trip costs.",
    }
    # Within window
    res_in = evaluate_candidate_applicability(
        candidate_route="Mumbai-Pune",
        candidate_week_of="2025-02-24",
        retrieved_notes=[valid_unseen],
        all_notes_df=notes_df,
    )
    assert res_in.is_justified is True
    assert res_in.matched_note_id == "N987"

    # Outside window (e.g. week of 2025-03-10)
    res_out = evaluate_candidate_applicability(
        candidate_route="Mumbai-Pune",
        candidate_week_of="2025-03-10",
        retrieved_notes=[valid_unseen],
        all_notes_df=notes_df,
    )
    assert res_out.is_justified is False
    assert res_out.matched_note_id is None


def test_tfidf_retrieval_fallback_verdict_parity(notes_df):
    """
    Verify that NoteRetriever running under TF-IDF fallback mode produces identical verdicts.
    """
    from src.retrieval import NoteRetriever
    
    tfidf_retriever = NoteRetriever(notes_df, force_tfidf=True)

    test_cases = [
        ("Ahmedabad-Mumbai", "2025-01-20", 29.5, 22.5, True, "N002"),
        ("Chennai-Bangalore", "2025-02-24", 31.5, 38.7, True, "N001"),
        ("Delhi-Jaipur", "2024-11-11", 35.5, 21.0, False, None),
        ("Mumbai-Pune", "2025-09-15", 9.2, 23.6, False, None),
    ]

    for route, week, own, peer, exp_just, exp_note in test_cases:
        top_tfidf = tfidf_retriever.retrieve(route=route, week_of=week, pct_vs_own=own, pct_vs_peers=peer, top_k=3)
        res_tfidf = evaluate_candidate_applicability(
            candidate_route=route,
            candidate_week_of=week,
            retrieved_notes=top_tfidf,
            all_notes_df=notes_df,
        )
        assert res_tfidf.is_justified == exp_just
        assert res_tfidf.matched_note_id == exp_note

