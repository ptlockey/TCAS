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
    REVERSAL_GUARD_TGO_S,
    apply_second_phase,
    classify_event,
    integrate_altitude_from_vs,
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


def test_apply_second_phase_strengthen_weak_uses_reduced_targets():
    tgo = 20.0
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
    _, vs_ca = vs_time_series(tgo, dt, 5.0, 0.18, CAT_INIT_VS_FPM, sense=sense_cat, cap_fpm=CAT_CAP_INIT_FPM)

    times2, _, vs_ca2, t_issue = apply_second_phase(
        times,
        vs_pl,
        vs_ca,
        tgo,
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

    assert t_issue is not None

    idx_issue = int(np.where(np.isclose(times2, t_issue))[0][0])
    suffix_vs = vs_ca2[idx_issue:]
    suffix_time = times2[idx_issue]
    rem = tgo - suffix_time
    _, expected_vs = vs_time_series(
        rem,
        dt,
        0.9,
        0.20,
        1800.0,
        sense=sense_cat,
        cap_fpm=2000.0,
        vs0_fpm=vs_ca2[idx_issue],
    )

    assert np.allclose(suffix_vs, expected_vs)


def test_classify_event_reversal_guard_uses_time_to_go():
    times = np.arange(0.0, 21.0, 1.0)
    sep0 = 1000.0
    closure_fps = 50.0

    z_pl = np.zeros_like(times)
    z_ca = sep0 - closure_fps * times
    vs_pl = np.zeros_like(times)
    vs_ca = np.full_like(times, -closure_fps * 60.0)

    eventtype, _, _, t_detect, reversal_reason = classify_event(
        times=times,
        z_pl=z_pl,
        z_ca=z_ca,
        vs_pl=vs_pl,
        vs_ca=vs_ca,
        tgo=20.0,
        alim_ft=400.0,
        margin_ft=100.0,
        sense_chosen_cat=+1,
        sense_exec_cat=-1,
    )

    assert eventtype == "REVERSE"
    assert reversal_reason == "Opposite sense"
    assert np.isclose(t_detect, 20.0 - REVERSAL_GUARD_TGO_S)


def test_classify_event_standard_cat_delay_is_not_reversal_trigger():
    tgo = 20.0
    dt = 1.0

    times = np.arange(0.0, tgo + 1e-9, dt)
    _, vs_pl = vs_time_series(
        tgo,
        dt,
        PL_DELAY_MEAN_S,
        PL_ACCEL_G,
        PL_VS_FPM,
        sense=+1,
        cap_fpm=PL_VS_CAP_FPM,
    )
    _, vs_ca = vs_time_series(
        tgo,
        dt,
        5.0,
        0.25,
        CAT_INIT_VS_FPM,
        sense=-1,
        cap_fpm=CAT_CAP_INIT_FPM,
    )

    z_pl = integrate_altitude_from_vs(times, vs_pl, 0.0)
    z_ca = integrate_altitude_from_vs(times, vs_ca, 750.0)

    eventtype, _, _, _, reversal_reason = classify_event(
        times=times,
        z_pl=z_pl,
        z_ca=z_ca,
        vs_pl=vs_pl,
        vs_ca=vs_ca,
        tgo=tgo,
        alim_ft=400.0,
        margin_ft=100.0,
        sense_chosen_cat=-1,
        sense_exec_cat=-1,
    )

    assert eventtype == "NONE"
    assert reversal_reason is None


def test_classify_event_strengthen_fires_on_predicted_miss_when_time_allows():
    tgo = 25.0
    dt = 1.0

    times = np.arange(0.0, tgo + 1e-9, dt)
    vs_pl = np.zeros_like(times)
    vs_ca = np.zeros_like(times)

    for idx, t_now in enumerate(times):
        if t_now < 3.0:
            vs_ca[idx] = -200.0
        elif t_now < 8.0:
            vs_ca[idx] = -200.0 - 200.0 * (t_now - 3.0)
        else:
            vs_ca[idx] = -1200.0

    z_pl = integrate_altitude_from_vs(times, vs_pl, 0.0)
    z_ca = integrate_altitude_from_vs(times, vs_ca, 900.0)

    eventtype, _, _, t_detect, _ = classify_event(
        times=times,
        z_pl=z_pl,
        z_ca=z_ca,
        vs_pl=vs_pl,
        vs_ca=vs_ca,
        tgo=tgo,
        alim_ft=400.0,
        margin_ft=100.0,
        sense_chosen_cat=+1,
        sense_exec_cat=+1,
    )

    assert eventtype == "STRENGTHEN"
    assert np.isclose(t_detect, 7.0)


def test_classify_event_strengthen_suppressed_when_time_short():
    tgo = 15.0
    dt = 1.0

    times = np.arange(0.0, tgo + 1e-9, dt)
    vs_pl = np.zeros_like(times)
    vs_ca = np.zeros_like(times)

    for idx, t_now in enumerate(times):
        if t_now < 3.0:
            vs_ca[idx] = -200.0
        elif t_now < 8.0:
            vs_ca[idx] = -200.0 - 200.0 * (t_now - 3.0)
        else:
            vs_ca[idx] = -1200.0

    z_pl = integrate_altitude_from_vs(times, vs_pl, 0.0)
    z_ca = integrate_altitude_from_vs(times, vs_ca, 900.0)

    eventtype, _, _, _, _ = classify_event(
        times=times,
        z_pl=z_pl,
        z_ca=z_ca,
        vs_pl=vs_pl,
        vs_ca=vs_ca,
        tgo=tgo,
        alim_ft=400.0,
        margin_ft=100.0,
        sense_chosen_cat=+1,
        sense_exec_cat=+1,
    )

    assert eventtype != "STRENGTHEN"


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
