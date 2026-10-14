"""Independent verification of metrics outside the pipeline using raw pandas computations."""

import pandas as pd
import pytest


def test_independent_recomputation_sample_rows():
    """
    Independently recompute cost_per_tonne_km, vs_own_history, and vs_similar_routes
    from scratch without calling any pipeline functions, proving mathematical correctness.
    """
    raw_df = pd.read_csv("data/shipment_records.csv")
    raw_df["route"] = raw_df["origin"] + "-" + raw_df["destination"]
    raw_df["dt"] = pd.to_datetime(raw_df["shipment_date"])
    # Monday date
    raw_df["week_of"] = raw_df["dt"].apply(lambda d: d - pd.Timedelta(days=d.weekday())).dt.strftime("%Y-%m-%d")
    raw_df["tonne_km"] = raw_df["quantity_tonnes"] * raw_df["distance_km"]

    # 1. Check Delhi-Jaipur on 2024-11-11
    # Target week shipments
    dj_target = raw_df[(raw_df["route"] == "Delhi-Jaipur") & (raw_df["week_of"] == "2024-11-11")]
    dj_cost = dj_target["freight_cost_inr"].sum() / dj_target["tonne_km"].sum()
    assert round(dj_cost, 2) == 4.17

    # Delhi-Jaipur prior 8 weeks
    all_weeks = sorted(raw_df["week_of"].unique())
    idx_dj = all_weeks.index("2024-11-11")
    prior_8_weeks = all_weeks[idx_dj - 8:idx_dj]
    assert len(prior_8_weeks) == 8
    assert "2024-11-11" not in prior_8_weeks  # Zero look-ahead check

    prior_rates = []
    for w in prior_8_weeks:
        w_recs = raw_df[(raw_df["route"] == "Delhi-Jaipur") & (raw_df["week_of"] == w)]
        rate = w_recs["freight_cost_inr"].sum() / w_recs["tonne_km"].sum()
        prior_rates.append(rate)
    dj_prior_mean = sum(prior_rates) / len(prior_rates)
    dj_pct_vs_own = ((dj_cost - dj_prior_mean) / dj_prior_mean) * 100.0
    assert round(dj_pct_vs_own, 1) == 35.5  # Exactly +35.5%

    # Peer routes in week 2024-11-11 with route_type == 'Short' (excluding Delhi-Jaipur)
    short_peers = raw_df[(raw_df["route_type"] == "Short") & (raw_df["week_of"] == "2024-11-11") & (raw_df["route"] != "Delhi-Jaipur")]
    peer_routes = short_peers["route"].unique()
    peer_rates = []
    for pr in peer_routes:
        pr_recs = short_peers[short_peers["route"] == pr]
        pr_rate = pr_recs["freight_cost_inr"].sum() / pr_recs["tonne_km"].sum()
        peer_rates.append(pr_rate)
    peer_mean = sum(peer_rates) / len(peer_rates)
    dj_pct_vs_peers = ((dj_cost - peer_mean) / peer_mean) * 100.0
    assert round(dj_pct_vs_peers, 1) == 21.0  # Exactly +21.0%

    # 2. Check Ahmedabad-Mumbai on 2025-01-20
    am_target = raw_df[(raw_df["route"] == "Ahmedabad-Mumbai") & (raw_df["week_of"] == "2025-01-20")]
    am_cost = am_target["freight_cost_inr"].sum() / am_target["tonne_km"].sum()
    assert round(am_cost, 2) == 3.29

    idx_am = all_weeks.index("2025-01-20")
    prior_8_am = all_weeks[idx_am - 8:idx_am]
    prior_am_rates = []
    for w in prior_8_am:
        w_recs = raw_df[(raw_df["route"] == "Ahmedabad-Mumbai") & (raw_df["week_of"] == w)]
        rate = w_recs["freight_cost_inr"].sum() / w_recs["tonne_km"].sum()
        prior_am_rates.append(rate)
    am_prior_mean = sum(prior_am_rates) / len(prior_am_rates)
    am_pct_vs_own = ((am_cost - am_prior_mean) / am_prior_mean) * 100.0
    assert round(am_pct_vs_own, 1) == 29.5  # Exactly +29.5%

    med_peers = raw_df[(raw_df["route_type"] == "Medium") & (raw_df["week_of"] == "2025-01-20") & (raw_df["route"] != "Ahmedabad-Mumbai")]
    peer_am_rates = []
    for pr in med_peers["route"].unique():
        pr_recs = med_peers[med_peers["route"] == pr]
        pr_rate = pr_recs["freight_cost_inr"].sum() / pr_recs["tonne_km"].sum()
        peer_am_rates.append(pr_rate)
    am_peer_mean = sum(peer_am_rates) / len(peer_am_rates)
    am_pct_vs_peers = ((am_cost - am_peer_mean) / am_peer_mean) * 100.0
    assert round(am_pct_vs_peers, 1) == 22.5  # Exactly +22.5%

    # 3. Check Mumbai-Pune on 2025-09-15
    mp_target = raw_df[(raw_df["route"] == "Mumbai-Pune") & (raw_df["week_of"] == "2025-09-15")]
    mp_cost = mp_target["freight_cost_inr"].sum() / mp_target["tonne_km"].sum()
    assert round(mp_cost, 2) == 3.98

    idx_mp = all_weeks.index("2025-09-15")
    prior_8_mp = all_weeks[idx_mp - 8:idx_mp]
    prior_mp_rates = []
    for w in prior_8_mp:
        w_recs = raw_df[(raw_df["route"] == "Mumbai-Pune") & (raw_df["week_of"] == w)]
        rate = w_recs["freight_cost_inr"].sum() / w_recs["tonne_km"].sum()
        prior_mp_rates.append(rate)
    mp_prior_mean = sum(prior_mp_rates) / len(prior_mp_rates)
    mp_pct_vs_own = ((mp_cost - mp_prior_mean) / mp_prior_mean) * 100.0
    assert round(mp_pct_vs_own, 1) == 9.2  # Exactly +9.2%

    mp_peers = raw_df[(raw_df["route_type"] == "Short") & (raw_df["week_of"] == "2025-09-15") & (raw_df["route"] != "Mumbai-Pune")]
    peer_mp_rates = []
    for pr in mp_peers["route"].unique():
        pr_recs = mp_peers[mp_peers["route"] == pr]
        pr_rate = pr_recs["freight_cost_inr"].sum() / pr_recs["tonne_km"].sum()
        peer_mp_rates.append(pr_rate)
    mp_peer_mean = sum(peer_mp_rates) / len(peer_mp_rates)
    mp_pct_vs_peers = ((mp_cost - mp_peer_mean) / mp_peer_mean) * 100.0
    assert round(mp_pct_vs_peers, 1) == 23.6  # Exactly +23.6%
