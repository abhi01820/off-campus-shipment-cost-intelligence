"""Unit tests for weekly bucketing logic."""

import pandas as pd
import pytest
from src.metrics import assign_monday_week


def test_assign_monday_week_standard_days():
    """Verify that every day in a week correctly maps to that week's Monday."""
    # Week of Monday Jan 15, 2024 to Sunday Jan 21, 2024
    dates = pd.date_range("2024-01-15", "2024-01-21", freq="D")
    series = pd.Series(dates)
    monday_weeks = assign_monday_week(series)

    # Every single day must map to '2024-01-15'
    assert (monday_weeks == "2024-01-15").all()


def test_assign_monday_week_boundary_sunday():
    """Verify that a Sunday maps to its preceding Monday, NOT the following Monday."""
    sunday = pd.Series(pd.to_datetime(["2024-01-07"]))  # Sunday
    monday_of_sunday = assign_monday_week(sunday).iloc[0]
    assert monday_of_sunday == "2024-01-01"

    following_monday = pd.Series(pd.to_datetime(["2024-01-08"]))  # Monday
    monday_of_monday = assign_monday_week(following_monday).iloc[0]
    assert monday_of_monday == "2024-01-08"


def test_assign_monday_week_year_boundary():
    """Verify year transitions (e.g. Dec 31 / Jan 01)."""
    # 2023-12-31 was a Sunday; Monday was 2023-12-25
    dec31 = pd.Series(pd.to_datetime(["2023-12-31"]))
    assert assign_monday_week(dec31).iloc[0] == "2023-12-25"

    # 2024-01-01 was a Monday; Monday is 2024-01-01
    jan01 = pd.Series(pd.to_datetime(["2024-01-01"]))
    assert assign_monday_week(jan01).iloc[0] == "2024-01-01"
