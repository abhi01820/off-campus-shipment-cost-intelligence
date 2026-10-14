"""Anomaly flagging module for candidate route-week cost elevations."""

import logging
from typing import Dict, List, Literal, Tuple
import pandas as pd

logger = logging.getLogger(__name__)


def identify_anomaly_candidates(
    df: pd.DataFrame,
    threshold_pct_vs_own: float = 9.0,
    threshold_pct_vs_peers: float = 20.0,
    combination_logic: Literal["OR", "AND"] = "OR",
) -> pd.DataFrame:
    """
    Identify route-weeks where cost is rising AND out of the ordinary.
    
    A route-week is marked as an anomaly candidate if:
    1. Cost is rising: pct_vs_own > 0 (or pd.isna(pct_vs_own) and pct_vs_peers > 0)
    2. Out of ordinary:
       - If logic == "OR": (pct_vs_own >= threshold_pct_vs_own) OR (pct_vs_peers >= threshold_pct_vs_peers)
       - If logic == "AND": (pct_vs_own >= threshold_pct_vs_own) AND (pct_vs_peers >= threshold_pct_vs_peers)
    
    Returns a copy of df with a boolean column `is_candidate_anomaly`.
    """
    res = df.copy()

    # Cost is rising vs own past history
    cond_rising = res["pct_vs_own"].fillna(0) > 0.0

    cond_own_spike = res["pct_vs_own"].fillna(-999) >= threshold_pct_vs_own
    cond_peer_spike = res["pct_vs_peers"].fillna(-999) >= threshold_pct_vs_peers

    if combination_logic == "OR":
        res["is_candidate_anomaly"] = cond_rising & (cond_own_spike | cond_peer_spike)
    elif combination_logic == "AND":
        res["is_candidate_anomaly"] = cond_rising & (cond_own_spike & cond_peer_spike)
    else:
        raise ValueError(f"Unknown combination_logic: {combination_logic}")

    candidate_count = res["is_candidate_anomaly"].sum()
    logger.info(
        "Identified %d candidate anomalies (thresholds: own=%.1f%%, peers=%.1f%%, logic=%s).",
        candidate_count, threshold_pct_vs_own, threshold_pct_vs_peers, combination_logic
    )
    return res


def run_threshold_sensitivity_grid(
    df: pd.DataFrame,
    own_thresholds: List[float] = [5.0, 8.0, 9.0, 10.0, 15.0, 20.0],
    peer_thresholds: List[float] = [15.0, 20.0, 25.0, 30.0],
    logics: List[str] = ["OR", "AND"],
) -> pd.DataFrame:
    """
    Run grid sensitivity analysis across multiple threshold values and combination logics.
    Checks candidate counts and whether the three reference sample rows are preserved.
    """
    reference_sample_keys = {
        ("Delhi-Jaipur", "2024-11-11"),
        ("Ahmedabad-Mumbai", "2025-01-20"),
        ("Mumbai-Pune", "2025-09-15"),
    }

    records = []
    for logic in logics:
        for t_own in own_thresholds:
            for t_peer in peer_thresholds:
                evaluated = identify_anomaly_candidates(
                    df,
                    threshold_pct_vs_own=t_own,
                    threshold_pct_vs_peers=t_peer,
                    combination_logic=logic,  # type: ignore
                )
                candidates = evaluated[evaluated["is_candidate_anomaly"]]
                cand_set = set(zip(candidates["route"], candidates["week_of"]))
                
                reproduced_samples = reference_sample_keys.issubset(cand_set)
                
                records.append({
                    "combination_logic": logic,
                    "threshold_vs_own": t_own,
                    "threshold_vs_peers": t_peer,
                    "candidate_count": len(candidates),
                    "reproduces_all_3_samples": reproduced_samples,
                    "missing_samples": list(reference_sample_keys - cand_set),
                })

    return pd.DataFrame(records)
