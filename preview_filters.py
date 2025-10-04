"""Preview filtering utilities for the Monte Carlo Streamlit app."""
from __future__ import annotations

import pandas as pd

LOWEST_SEPARATION_WINDOW_FT = 300.0


def build_preview_dataframe(
    df: pd.DataFrame,
    *,
    reversal_only: bool = False,
    lowest_separation_only: bool = False,
    window_ft: float = LOWEST_SEPARATION_WINDOW_FT,
) -> pd.DataFrame:
    """Return a filtered preview DataFrame respecting the configured options."""

    if df is None:
        return pd.DataFrame()
    if df.empty:
        return df.iloc[0:0]

    preview_df = df
    if reversal_only:
        preview_df = preview_df.loc[preview_df["eventtype"] == "REVERSE"]

    if lowest_separation_only:
        if "sep_cpa_ft" not in df.columns:
            preview_df = preview_df.iloc[0:0]
        else:
            min_sep = df["sep_cpa_ft"].min()
            if pd.isna(min_sep):
                preview_df = preview_df.iloc[0:0]
            else:
                threshold = float(min_sep) + float(window_ft)
                preview_df = preview_df.loc[preview_df["sep_cpa_ft"] <= threshold]
                preview_df = preview_df.sort_values("sep_cpa_ft", ascending=True)

    return preview_df.copy()
