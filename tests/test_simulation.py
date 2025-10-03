import sys
from pathlib import Path

import numpy as np
import pandas.testing as pdt

sys.path.append(str(Path(__file__).resolve().parents[1]))

from simulation import (
    CAT_CAP_INIT_FPM,
    CAT_CAP_STRENGTH_FPM,
    CAT_INIT_VS_FPM,
    CAT_STRENGTH_FPM,
    PL_ACCEL_G,
    PL_DELAY_MEAN_S,
    PL_VS_CAP_FPM,
    PL_VS_FPM,
    apply_second_phase,
    ias_to_tas,
    run_batch,
    vs_time_series,
)


def test_ias_to_tas_behaviour():
    tas_sl = ias_to_tas(200.0, 0.0)
    tas_mid = ias_to_tas(200.0, 20000.0)

    assert np.isclose(tas_sl, 200.0, atol=1e-6)
    assert tas_mid > tas_sl


def test_apply_second_phase_reverse_changes_sense():
    tgo = 20.0
    dt = 0.5
    sense_pl = +1
    sense_cat = -1

    times, vs_pl = vs_time_series(
        tgo,
        dt,
        PL_DELAY_MEAN_S,
        PL_ACCEL_G,
        PL_VS_FPM,
        sense=sense_pl,
        cap_fpm=PL_VS_CAP_FPM,
    )
    _, vs_ca = vs_time_series(tgo, dt, 4.0, 0.20, CAT_INIT_VS_FPM, sense=sense_cat, cap_fpm=CAT_CAP_INIT_FPM)

    times2, vs_pl2, vs_ca2, t_issue = apply_second_phase(
        times,
        vs_pl,
        vs_ca,
        tgo,
        dt,
        eventtype="REVERSE",
        sense_pl=sense_pl,
        sense_cat_exec=sense_cat,
        pl_vs0=0.0,
        cat_vs0=0.0,
        t_classify=8.0,
        pl_delay=PL_DELAY_MEAN_S,
        pl_accel_g=PL_ACCEL_G,
        pl_cap=PL_VS_CAP_FPM,
        cat_delay=0.9,
        cat_accel_g=0.35,
        cat_vs_strength=CAT_STRENGTH_FPM,
        cat_cap=CAT_CAP_STRENGTH_FPM,
        decision_latency_s=1.0,
    )

    assert t_issue is not None
    assert vs_pl2[-1] < 0.0  # PL reverses to descend eventually
    assert vs_ca2[-1] > 0.0  # CAT reverses to climb eventually


def test_run_batch_deterministic_seed():
    df1 = run_batch(
        runs=20,
        seed=123,
        jitter_priors=False,
        use_delay_mixture=False,
        apfd_share=0.0,
    )
    df2 = run_batch(
        runs=20,
        seed=123,
        jitter_priors=False,
        use_delay_mixture=False,
        apfd_share=0.0,
    )

    assert len(df1) == len(df2) == 20
    pdt.assert_frame_equal(df1.reset_index(drop=True), df2.reset_index(drop=True))
