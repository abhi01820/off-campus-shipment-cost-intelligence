"""End-to-end anomaly detection and explanation pipeline."""

import logging
from pathlib import Path
from typing import Optional, Tuple
import pandas as pd

from src.config import AppConfig, load_config
from src.data_loader import load_all_data
from src.metrics import compute_weekly_metrics
from src.baselines import add_comparison_baselines
from src.flagging import identify_anomaly_candidates
from src.retrieval import NoteRetriever
from src.applicability_gate import evaluate_candidate_applicability
from src.explainer import generate_explanation
from src.validator import validate_output_file
from src.cost_tracker import CostTracker

logger = logging.getLogger(__name__)


def run_pipeline(
    config: Optional[AppConfig] = None,
    config_path: Optional[str | Path] = None,
    cost_tracker: Optional[CostTracker] = None,
) -> Tuple[pd.DataFrame, CostTracker]:
    """
    Execute the full end-to-end Freight Tiger pipeline:
    1. Ingestion & Validation
    2. Weekly Metrics Computation
    3. Trailing History & Peer Baselines
    4. Anomaly Candidate Identification
    5. Vector Semantic Retrieval
    6. Deterministic Applicability Gate
    7. Plain-English Grounded Explanation
    8. CSV Output Generation & Contract Schema Validation
    """
    if config is None:
        config = load_config(config_path)

    if cost_tracker is None:
        cost_tracker = CostTracker()

    logger.info("Starting pipeline execution with config: %s", config.model_dump())

    # 1. Ingestion
    shipments_df, notes_df = load_all_data(
        config.data.shipment_records_path,
        config.data.context_notes_path
    )

    # 2. Weekly Metrics
    weekly_df = compute_weekly_metrics(shipments_df)

    # 3. Baselines
    baselines_df = add_comparison_baselines(
        weekly_df,
        window_weeks=config.anomaly_detection.history_window_weeks
    )

    # 4. Anomaly Candidate Identification
    flagged_df = identify_anomaly_candidates(
        baselines_df,
        threshold_pct_vs_own=config.anomaly_detection.threshold_pct_vs_own,
        threshold_pct_vs_peers=config.anomaly_detection.threshold_pct_vs_peers,
        combination_logic=config.anomaly_detection.combination_logic,
    )

    # 5. Note Retrieval Index
    retriever = NoteRetriever(notes_df, model_name=config.retrieval.model_name)
    cost_tracker.record_embedding_run(
        doc_count=len(notes_df),
        query_count=int(flagged_df["is_candidate_anomaly"].sum()),
        model_name=config.retrieval.model_name
    )

    # Filter rows according to export config
    if config.output.export_only_candidates:
        eval_df = flagged_df[flagged_df["is_candidate_anomaly"]].copy()
    else:
        eval_df = flagged_df.copy()

    # Deterministic sorting
    eval_df = eval_df.sort_values(by=["route", "week_of"]).reset_index(drop=True)

    results = []
    for _, row in eval_df.iterrows():
        route = row["route"]
        week_of = row["week_of"]
        pct_own = row["pct_vs_own"]
        pct_peers = row["pct_vs_peers"]
        is_cand = row["is_candidate_anomaly"]

        if not is_cand:
            # If exporting all rows and row is not anomalous
            results.append({
                "route": route,
                "week_of": week_of,
                "cost_per_tonne_km": f"{row['cost_per_tonne_km']:.2f}",
                "vs_own_history": row["vs_own_history"],
                "vs_similar_routes": row["vs_similar_routes"],
                "flagged": "No",
                "matched_note_id": "",
                "reason": "Normal freight rate within historical and peer baseline variance.",
            })
            continue

        # 6. Retrieve relevant candidate notes
        retrieved = retriever.retrieve(
            route=route,
            week_of=week_of,
            pct_vs_own=pct_own,
            pct_vs_peers=pct_peers,
            top_k=config.retrieval.top_k,
            min_similarity=config.retrieval.similarity_threshold,
        )

        # 7. Deterministic Applicability Gate
        eval_result = evaluate_candidate_applicability(
            candidate_route=route,
            candidate_week_of=week_of,
            retrieved_notes=retrieved,
            all_notes_df=notes_df,
            default_start_only_weeks=config.applicability.default_start_only_window_weeks,
            near_miss_similarity_threshold=config.applicability.near_miss_similarity_threshold,
            max_days_for_near_miss=config.applicability.max_days_for_near_miss,
        )

        # Verdict
        if eval_result.is_justified:
            flag_val = "No (justified)"
            matched_note_id = eval_result.matched_note_id or ""
        else:
            flag_val = "Yes"
            matched_note_id = ""

        # 8. Plain-English Grounded Explanation
        reason = generate_explanation(
            evaluation=eval_result,
            route=route,
            week_of=week_of,
            mode=config.explanation.mode,
            llm_model=config.explanation.llm_model,
            llm_temperature=config.explanation.llm_temperature,
            cost_tracker=cost_tracker,
        )

        results.append({
            "route": route,
            "week_of": week_of,
            "cost_per_tonne_km": f"{row['cost_per_tonne_km']:.2f}",
            "vs_own_history": row["vs_own_history"],
            "vs_similar_routes": row["vs_similar_routes"],
            "flagged": flag_val,
            "matched_note_id": matched_note_id,
            "reason": reason,
        })

    out_df = pd.DataFrame(results)

    # 9. Output generation
    out_path = Path(config.output.output_csv_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_path, index=False, encoding="utf-8")
    logger.info("Wrote %d output records to %s", len(out_df), out_path)

    # 10. Contract Schema Validation
    validate_output_file(out_path)

    return out_df, cost_tracker
