"""Weekly bucketing and cost-per-tonne-km aggregation module."""

import logging
import pandas as pd

logger = logging.getLogger(__name__)


def assign_monday_week(series: pd.Series) -> pd.Series:
    """
    Given a datetime series, compute the Monday date of the corresponding week (Monday-Sunday).
    
    Formula: dt - timedelta(days=dt.weekday())
    where Monday is weekday 0, Sunday is weekday 6.
    Returns string formatted as 'YYYY-MM-DD'.
    """
    monday_dt = series - pd.to_timedelta(series.dt.weekday, unit="D")
    return monday_dt.dt.strftime("%Y-%m-%d")


def compute_weekly_metrics(shipment_df: pd.DataFrame) -> pd.DataFrame:
    """
    Bucket shipment records into Monday-Sunday weeks and compute weekly metrics.
    
    Metrics aggregated per (route, route_type, week_of):
    - total_freight_cost: sum(freight_cost_inr)
    - total_tonne_km: sum(quantity_tonnes * distance_km)
    - cost_per_tonne_km: total_freight_cost / total_tonne_km
    - shipment_count: count(shipment_id)
    
    Returns DataFrame sorted deterministically by (route, week_of).
    """
    df = shipment_df.copy()

    # 1. Bucket shipment_date into Monday week_of before any computation
    df["week_of"] = assign_monday_week(df["shipment_date"])

    # 2. Compute trip-level tonne_km
    df["trip_tonne_km"] = df["quantity_tonnes"] * df["distance_km"]

    # 3. Group by (route, route_type, week_of)
    weekly = df.groupby(["route", "route_type", "week_of"], as_index=False).agg(
        total_freight_cost=("freight_cost_inr", "sum"),
        total_tonne_km=("trip_tonne_km", "sum"),
        shipment_count=("shipment_id", "count"),
    )

    # 4. Aggregate cost per tonne-km
    # sum(cost) / sum(quantity * distance)
    weekly["cost_per_tonne_km"] = weekly["total_freight_cost"] / weekly["total_tonne_km"]

    # 5. Deterministic sorting
    weekly = weekly.sort_values(by=["route", "week_of"]).reset_index(drop=True)

    logger.info("Computed weekly metrics: %d route-weeks across %d routes.",
                len(weekly), weekly["route"].nunique())
    return weekly
