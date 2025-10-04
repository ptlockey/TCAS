import pandas as pd
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from preview_filters import LOWEST_SEPARATION_WINDOW_FT, build_preview_dataframe


def make_df():
    return pd.DataFrame(
        {
            "run": [1, 2, 3, 4],
            "eventtype": ["NONE", "REVERSE", "STRENGTHEN", "REVERSE"],
            "sep_cpa_ft": [800.0, 650.0, 900.0, 620.0],
        }
    )


def test_reversal_filter_only():
    df = make_df()
    result = build_preview_dataframe(df, reversal_only=True)
    assert list(result["run"]) == [2, 4]


def test_lowest_separation_window_filters_and_sorts():
    df = make_df()
    result = build_preview_dataframe(df, lowest_separation_only=True)
    min_sep = df["sep_cpa_ft"].min()
    threshold = min_sep + LOWEST_SEPARATION_WINDOW_FT
    assert result["sep_cpa_ft"].max() <= threshold
    assert list(result["sep_cpa_ft"]) == sorted(result["sep_cpa_ft"])  # ascending order


def test_combined_filters_apply_in_sequence():
    df = make_df()
    result = build_preview_dataframe(df, reversal_only=True, lowest_separation_only=True)
    assert list(result["run"]) == [4, 2]  # sorted by separation ascending
    assert (result["eventtype"] == "REVERSE").all()


def test_handles_empty_input():
    df = pd.DataFrame(columns=["run", "eventtype", "sep_cpa_ft"])
    result = build_preview_dataframe(df, reversal_only=True, lowest_separation_only=True)
    assert result.empty
