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
    classify_event,
    integrate_altitude_from_vs,
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


def test_classify_event_no_response_escalates_early():
    tgo = 30.0
    dt = 0.5
    sense_pl = +1
    sense_cat = +1

    times, vs_pl = vs_time_series(
        tgo,
        dt,
        PL_DELAY_MEAN_S,
        PL_ACCEL_G,
        PL_VS_FPM,
        sense=sense_pl,
        cap_fpm=PL_VS_CAP_FPM,
    )
    _, vs_ca = vs_time_series(
        tgo,
        dt,
        5.0,
        0.0,
        0.0,
        sense=sense_cat,
        cap_fpm=0.0,
    )

    z_pl = integrate_altitude_from_vs(times, vs_pl, 0.0)
    z_ca = integrate_altitude_from_vs(times, vs_ca, 1000.0)

    eventtype, _, _, t_detect, _ = classify_event(
        times,
        z_pl,
        z_ca,
        vs_pl,
        vs_ca,
        tgo,
        alim_ft=400.0,
        margin_ft=100.0,
        sense_chosen_cat=sense_cat,
        sense_exec_cat=sense_cat,
        cat_mode="no-response",
        cat_vs_cmd=CAT_INIT_VS_FPM,
    )

    assert eventtype == "STRENGTHEN"
    assert np.isclose(t_detect, 3.0, atol=1e-6)


def test_apply_second_phase_strengthen_weak_scales_profile():
    dt = 0.5
    sense_pl = +1
    sense_cat = +1

    # Long time-to-go retains the moderated weak profile.
    tgo_long = 32.0
    times_long, vs_pl_long = vs_time_series(
        tgo_long,
        dt,
        PL_DELAY_MEAN_S,
        PL_ACCEL_G,
        PL_VS_FPM,
        sense=sense_pl,
        cap_fpm=PL_VS_CAP_FPM,
    )
    _, vs_ca_long = vs_time_series(
        tgo_long,
        dt,
        5.0,
        0.18,
        CAT_INIT_VS_FPM,
        sense=sense_cat,
        cap_fpm=CAT_CAP_INIT_FPM,
    )

    times2_long, _, vs_ca2_long, t_issue_long = apply_second_phase(
        times_long,
        vs_pl_long,
        vs_ca_long,
        tgo_long,
        dt,
        eventtype="STRENGTHEN",
        sense_pl=sense_pl,
        sense_cat_exec=sense_cat,
        pl_vs0=0.0,
        cat_vs0=0.0,
        t_classify=6.0,
        pl_delay=PL_DELAY_MEAN_S,
        pl_accel_g=PL_ACCEL_G,
        pl_cap=PL_VS_CAP_FPM,
        cat_delay=0.9,
        cat_accel_g=0.35,
        cat_vs_strength=CAT_STRENGTH_FPM,
        cat_cap=CAT_CAP_STRENGTH_FPM,
        decision_latency_s=1.0,
        cat_mode="weak-compliance",
    )

    assert t_issue_long is not None

    idx_issue_long = int(np.where(np.isclose(times2_long, t_issue_long))[0][0])
    suffix_vs_long = vs_ca2_long[idx_issue_long:]
    rem_long = tgo_long - times2_long[idx_issue_long]
    _, expected_vs_long = vs_time_series(
        rem_long,
        dt,
        0.9,
        0.25,
        2200.0,
        sense=sense_cat,
        cap_fpm=2400.0,
        vs0_fpm=vs_ca2_long[idx_issue_long],
    )

    assert np.allclose(suffix_vs_long, expected_vs_long)

    # Short time-to-go escalates to the exigent profile.
    tgo_short = 20.0
    times_short, vs_pl_short = vs_time_series(
        tgo_short,
        dt,
        PL_DELAY_MEAN_S,
        PL_ACCEL_G,
        PL_VS_FPM,
        sense=sense_pl,
        cap_fpm=PL_VS_CAP_FPM,
    )
    _, vs_ca_short = vs_time_series(
        tgo_short,
        dt,
        5.0,
        0.18,
        CAT_INIT_VS_FPM,
        sense=sense_cat,
        cap_fpm=CAT_CAP_INIT_FPM,
    )

    times2_short, _, vs_ca2_short, t_issue_short = apply_second_phase(
        times_short,
        vs_pl_short,
        vs_ca_short,
        tgo_short,
        dt,
        eventtype="STRENGTHEN",
        sense_pl=sense_pl,
        sense_cat_exec=sense_cat,
        pl_vs0=0.0,
        cat_vs0=0.0,
        t_classify=10.0,
        pl_delay=PL_DELAY_MEAN_S,
        pl_accel_g=PL_ACCEL_G,
        pl_cap=PL_VS_CAP_FPM,
        cat_delay=0.9,
        cat_accel_g=0.35,
        cat_vs_strength=CAT_STRENGTH_FPM,
        cat_cap=CAT_CAP_STRENGTH_FPM,
        decision_latency_s=1.0,
        cat_mode="weak-compliance",
    )

    assert t_issue_short is not None

    idx_issue_short = int(np.where(np.isclose(times2_short, t_issue_short))[0][0])
    suffix_vs_short = vs_ca2_short[idx_issue_short:]
    rem_short = tgo_short - times2_short[idx_issue_short]
    _, expected_vs_short = vs_time_series(
        rem_short,
        dt,
        0.9,
        0.35,
        CAT_STRENGTH_FPM,
        sense=sense_cat,
        cap_fpm=CAT_CAP_STRENGTH_FPM,
        vs0_fpm=vs_ca2_short[idx_issue_short],
    )

    assert np.allclose(suffix_vs_short, expected_vs_short)


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


def test_run_batch_apfd_custom_matches_preset():
    df_custom = run_batch(
        runs=100,
        seed=777,
        jitter_priors=False,
        use_delay_mixture=False,
        apfd_share=0.30,
        apfd_mode="custom",
    )

    df_preset = run_batch(
        runs=100,
        seed=777,
        jitter_priors=False,
        use_delay_mixture=False,
        apfd_share=0.30,
        apfd_mode="airbus",
    )

    pdt.assert_frame_equal(df_custom.reset_index(drop=True), df_preset.reset_index(drop=True))
