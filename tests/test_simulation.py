import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from simulation import (
    CAT_CAP_STRENGTH_FPM,
    CAT_STRENGTH_FPM,
    PL_ACCEL_G,
    PL_VS_CAP_FPM,
    apply_second_phase,
    ias_to_tas,
    run_batch,
    time_to_go_from_geometry,
)


def test_ias_to_tas_at_sea_level_matches_ias():
    assert math.isclose(ias_to_tas(250.0, 0.0), 250.0, rel_tol=1e-6)


def test_time_to_go_from_geometry_handles_positive_closure():
    # 6 NM at 300 kt should take 72 seconds
    assert math.isclose(time_to_go_from_geometry(6.0, 300.0), 72.0)


def test_apply_second_phase_no_event_returns_inputs():
    times = np.linspace(0.0, 10.0, 11)
    vs_pl = np.zeros_like(times)
    vs_ca = np.zeros_like(times)

    new_times, new_vs_pl, new_vs_ca, issued = apply_second_phase(
        times,
        vs_pl,
        vs_ca,
        tgo=20.0,
        dt=1.0,
        eventtype="NONE",
        sense_pl=1,
        sense_cat_exec=-1,
        pl_vs0=0.0,
        cat_vs0=0.0,
    )

    assert issued is None
    assert np.array_equal(new_times, times)
    assert np.array_equal(new_vs_pl, vs_pl)
    assert np.array_equal(new_vs_ca, vs_ca)


def test_apply_second_phase_strengthen_extends_timeseries():
    dt = 0.5
    times = np.arange(0.0, 10.0 + 1e-9, dt)
    vs_pl = np.zeros_like(times)
    vs_ca = np.zeros_like(times)

    new_times, new_vs_pl, new_vs_ca, issued = apply_second_phase(
        times,
        vs_pl,
        vs_ca,
        tgo=25.0,
        dt=dt,
        eventtype="STRENGTHEN",
        sense_pl=1,
        sense_cat_exec=-1,
        pl_vs0=0.0,
        cat_vs0=0.0,
        pl_delay=0.5,
        pl_accel_g=PL_ACCEL_G,
        pl_cap=PL_VS_CAP_FPM,
        cat_delay=0.5,
        cat_accel_g=0.2,
        cat_vs_strength=CAT_STRENGTH_FPM,
        cat_cap=CAT_CAP_STRENGTH_FPM,
        decision_latency_s=1.0,
    )

    assert issued is not None
    assert new_times.size > times.size
    # ensure the strengthened manoeuvre actually produces non-zero response
    assert np.max(np.abs(new_vs_pl)) > 0.0
    assert np.max(np.abs(new_vs_ca)) > 0.0


def test_run_batch_small_sample_is_deterministic():
    df = run_batch(
        runs=5,
        seed=123,
        scenario="Head-on",
        jitter_priors=False,
        apfd_share=0.0,
        use_delay_mixture=False,
    )

    df_repeat = run_batch(
        runs=5,
        seed=123,
        scenario="Head-on",
        jitter_priors=False,
        apfd_share=0.0,
        use_delay_mixture=False,
    )

    assert len(df) == 5
    expected_columns = {
        "run",
        "FL_PL",
        "FL_CAT",
        "missCPAft",
        "minsepft",
        "sep_cpa_ft",
        "eventtype",
    }
    assert expected_columns.issubset(df.columns)
    assert set(df["eventtype"].unique()) <= {"NONE", "STRENGTHEN", "REVERSE"}
    assert df[["minsepft", "sep_cpa_ft"]].notna().all().all()

    pd.testing.assert_frame_equal(df, df_repeat)
