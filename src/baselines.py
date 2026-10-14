"""Baseline computation module: Trailing 8-week history and same-week peer averages."""

import logging
from typing import Optional
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def compute_trailing_history_baseline(
    weekly_df: pd.DataFrame,
    window_weeks: int = 8
) -> pd.DataFrame:
    """
    Compute trailing rolling average cost per tonne-km for each route.
    
    Rules:
    - Strictly before the current week (no look-ahead, current week excluded).
    - Uses up to `window_weeks` (default 8) prior active shipment weeks.
    - If fewer than 8 prior weeks exist, uses all available prior weeks (unweighted mean of weekly cost_per_tonne_km).
    - For the first week (0 prior weeks), baseline is NaN / N/A.
    - No zero-padding or artificial extrapolation.
    """
    df = weekly_df.copy()
    df["own_history_avg"] = np.nan
    df["pct_vs_own"] = np.nan
    df["prior_weeks_count"] = 0

    # Ensure deterministic sort
    df = df.sort_values(by=["route", "week_of"]).reset_index(drop=True)

    for route, grp in df.groupby("route"):
        indices = grp.index
        for i, idx in enumerate(indices):
            if i == 0:
                # First week: no strictly prior weeks exist
                df.loc[idx, "prior_weeks_count"] = 0
                continue
            
            # Up to window_weeks strictly before index i
            start_i = max(0, i - window_weeks)
            prior_indices = indices[start_i:i]
            prior_costs = df.loc[prior_indices, "cost_per_tonne_km"]
            
            mean_prior = prior_costs.mean()
            curr_cost = df.loc[idx, "cost_per_tonne_km"]
            
            df.loc[idx, "own_history_avg"] = mean_prior
            df.loc[idx, "prior_weeks_count"] = len(prior_indices)
            if mean_prior > 0:
                df.loc[idx, "pct_vs_own"] = ((curr_cost - mean_prior) / mean_prior) * 100.0

    return df


def compute_peer_baseline(weekly_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute average cost per tonne-km in the same week across peer routes.
    
    Rules:
    - Same week_of.
    - Same route_type (Short / Medium / Long).
    - Excludes the route itself.
    - If no peer routes exist in that week, baseline is NaN / N/A.
    - Arithmetic mean of peer routes' cost_per_tonne_km.
    """
    df = weekly_df.copy()
    df["peer_avg"] = np.nan
    df["pct_vs_peers"] = np.nan
    df["peer_count"] = 0

    for (week_of, route_type), grp in df.groupby(["week_of", "route_type"]):
        for idx, row in grp.iterrows():
            peers = grp[grp["route"] != row["route"]]
            peer_count = len(peers)
            df.loc[idx, "peer_count"] = peer_count
            
            if peer_count > 0:
                mean_peer = peers["cost_per_tonne_km"].mean()
                curr_cost = row["cost_per_tonne_km"]
                df.loc[idx, "peer_avg"] = mean_peer
                if mean_peer > 0:
                    df.loc[idx, "pct_vs_peers"] = ((curr_cost - mean_peer) / mean_peer) * 100.0

    return df


def format_own_history_str(pct: Optional[float]) -> str:
    """Format vs_own_history matching sample: '+35.5% vs this route\'s past average'."""
    if pd.isna(pct):
        return "N/A (insufficient history)"
    return f"{pct:+.1f}% vs this route's past average"


def format_peer_str(pct: Optional[float]) -> str:
    """Format vs_similar_routes matching sample: '+21.0% vs similar-length routes this week'."""
    if pd.isna(pct):
        return "N/A (no peer routes)"
    return f"{pct:+.1f}% vs similar-length routes this week"


def add_comparison_baselines(
    weekly_df: pd.DataFrame,
    window_weeks: int = 8
) -> pd.DataFrame:
    """Compute both own history and peer baselines and append formatted strings."""
    df = compute_trailing_history_baseline(weekly_df, window_weeks=window_weeks)
    df = compute_peer_baseline(df)
    
    df["vs_own_history"] = df["pct_vs_own"].apply(format_own_history_str)
    df["vs_similar_routes"] = df["pct_vs_peers"].apply(format_peer_str)
    return df
