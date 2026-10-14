"""Unit tests for baseline calculations: trailing 8-week history and peer averages."""

import numpy as np
import pandas as pd
import pytest

from src.baselines import (
    compute_trailing_history_baseline,
    compute_peer_baseline,
    add_comparison_baselines,
    format_own_history_str,
    format_peer_str,
)


def test_trailing_history_strictly_before_current_week():
    """
    Verify that own history trailing 8-week window uses strictly prior weeks.
    Current week's value must NOT bleed into its own baseline (zero look-ahead).
    """
    # 10 consecutive weeks for a single route with known cost values
    # Costs: [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 100.0]
    weeks = [f"2024-01-{i*7+1:02d}" for i in range(10)]
    costs = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 100.0]

    df = pd.DataFrame({
        "route": ["RouteA"] * 10,
        "route_type": ["Short"] * 10,
        "week_of": weeks,
        "cost_per_tonne_km": costs,
    })

    result = compute_trailing_history_baseline(df, window_weeks=8)

    # Week 0 (first week): 0 prior weeks -> NaN
    assert np.isnan(result.loc[0, "own_history_avg"])
    assert result.loc[0, "prior_weeks_count"] == 0

    # Week 1: 1 prior week (1.0) -> avg = 1.0
    assert result.loc[1, "own_history_avg"] == 1.0
    assert result.loc[1, "prior_weeks_count"] == 1

    # Week 8 (9th item, costs[8] = 9.0):
    # Prior 8 weeks are indices 0 to 7: [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
    # Sum = 36, Mean = 36 / 8 = 4.5
    assert result.loc[8, "own_history_avg"] == 4.5
    assert result.loc[8, "prior_weeks_count"] == 8

    # Week 9 (10th item, cost = 100.0):
    # Prior 8 weeks are indices 1 to 8: [2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]
    # Sum = 44, Mean = 44 / 8 = 5.5
    # NOTICE: 100.0 is NOT included in its own baseline!
    assert result.loc[9, "own_history_avg"] == 5.5
    # Pct vs own = (100.0 - 5.5) / 5.5 * 100 = 1718.18%
    assert abs(result.loc[9, "pct_vs_own"] - ((100.0 - 5.5) / 5.5 * 100)) < 1e-6


def test_peer_baseline_excludes_self():
    """
    Verify peer baseline in same week averages other routes of same route_type, excluding self.
    """
    # 3 Medium routes in the same week
    df = pd.DataFrame({
        "route": ["RouteM1", "RouteM2", "RouteM3"],
        "route_type": ["Medium", "Medium", "Medium"],
        "week_of": ["2024-01-01", "2024-01-01", "2024-01-01"],
        "cost_per_tonne_km": [2.0, 4.0, 6.0],
    })

    result = compute_peer_baseline(df)

    # For RouteM1 (cost 2.0): peers are RouteM2 (4.0) and RouteM3 (6.0) -> mean = 5.0
    # pct = (2.0 - 5.0) / 5.0 * 100 = -60.0%
    m1 = result[result["route"] == "RouteM1"].iloc[0]
    assert m1["peer_avg"] == 5.0
    assert m1["pct_vs_peers"] == -60.0

    # For RouteM2 (cost 4.0): peers are RouteM1 (2.0) and RouteM3 (6.0) -> mean = 4.0
    # pct = 0.0%
    m2 = result[result["route"] == "RouteM2"].iloc[0]
    assert m2["peer_avg"] == 4.0
    assert m2["pct_vs_peers"] == 0.0

    # For RouteM3 (cost 6.0): peers are RouteM1 (2.0) and RouteM2 (4.0) -> mean = 3.0
    # pct = +100.0%
    m3 = result[result["route"] == "RouteM3"].iloc[0]
    assert m3["peer_avg"] == 3.0
    assert m3["pct_vs_peers"] == 100.0


def test_peer_baseline_single_peer_route():
    """
    Verify behavior when only a single route exists for a given route_type in a week.
    Peer baseline must be NaN / N/A, not error or self-compare.
    """
    df = pd.DataFrame({
        "route": ["LoneRoute"],
        "route_type": ["Short"],
        "week_of": ["2024-01-01"],
        "cost_per_tonne_km": [3.5],
    })

    result = compute_peer_baseline(df)
    row = result.iloc[0]
    assert np.isnan(row["peer_avg"])
    assert np.isnan(row["pct_vs_peers"])
    assert format_peer_str(row["pct_vs_peers"]) == "N/A (no peer routes)"
