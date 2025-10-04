"""Utilities for the Streamlit run inspector plots."""

from __future__ import annotations

from typing import Optional

import pandas as pd


def get_second_issue_time(row: pd.Series) -> Optional[float]:
    """Return the second-phase issue time for a run inspector row.

    Parameters
    ----------
    row:
        A pandas Series representing a run from the Monte Carlo batch.

    Returns
    -------
    Optional[float]
        The second-phase issue timestamp as a float when it exists and is
        non-null. Returns ``None`` when the column is absent or the value is
        missing.
    """

    if "t_second_issue" not in row.index:
        return None

    value = row["t_second_issue"]
    if pd.isna(value):
        return None

    return float(value)
