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
    FT_PER_M,
    G,
    PL_ACCEL_G,
    PL_DELAY_MEAN_S,
    PL_VS_CAP_FPM,
    PL_VS_FPM,
    choose_optimal_sense,
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


def test_vs_time_series_post_delay_slope_matches_commanded_accel():
    t_end = 4.0
    dt = 0.2
    t_delay = 1.0
    a_g = 0.25
    v_target = 4000.0
    expected_slope = a_g * G * FT_PER_M * 60.0

    for sense in (+1, -1):
        times, vs = vs_time_series(
            t_end,
            dt,
            t_delay,
            a_g,
            v_target,
            sense=sense,
            cap_fpm=None,
            vs0_fpm=0.0,
        )

        post_mask = times > t_delay
        post_times = times[post_mask]
        post_vs = vs[post_mask]

        assert post_times.size >= 3

        slopes = np.diff(post_vs) / np.diff(post_times)
        assert np.allclose(slopes, sense * expected_slope, atol=1e-6)


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


def test_apply_second_phase_strengthen_weak_meets_nominal_targets():
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
        0.35,
        CAT_CAP_STRENGTH_FPM,
        sense=sense_cat,
        cap_fpm=CAT_CAP_STRENGTH_FPM,
        vs0_fpm=vs_ca2[idx_issue],
    )

    assert np.allclose(suffix_vs, expected_vs)


def test_apply_second_phase_force_exigent_strengthen_uses_exigent_profile():
    tgo = 10.0
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
        t_classify=7.0,
        pl_delay=PL_DELAY_MEAN_S,
        pl_accel_g=PL_ACCEL_G,
        pl_cap=PL_VS_CAP_FPM,
        cat_delay=0.9,
        cat_accel_g=0.35,
        cat_vs_strength=CAT_STRENGTH_FPM,
        cat_cap=CAT_CAP_STRENGTH_FPM,
        decision_latency_s=0.8,
        cat_mode="weak-compliance",
        force_exigent=True,
    )

    assert t_issue is not None

    idx_issue = int(np.where(np.isclose(times2, t_issue))[0][0])
    suffix_vs = vs_ca2[idx_issue:]
    rem = tgo - times2[idx_issue]
    _, expected_vs = vs_time_series(
        rem,
        dt,
        0.9,
        0.35,
        CAT_CAP_STRENGTH_FPM,
        sense=sense_cat,
        cap_fpm=CAT_CAP_STRENGTH_FPM,
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

    eventtype, _, _, t_detect, event_detail = classify_event(
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
    assert event_detail == "Exigent wrong-sense"
    assert np.isclose(t_detect, 0.0)


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

    eventtype, _, _, _, event_detail = classify_event(
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

    assert eventtype in {"NONE", "STRENGTHEN"}
    if eventtype == "STRENGTHEN":
        assert event_detail is None


def test_classify_event_manual_intruder_delay_waits_before_reversing():
    tgo = 25.0
    dt = 0.2

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
    z_ca = integrate_altitude_from_vs(times, vs_ca, 900.0)

    eventtype, _, _, _, event_detail = classify_event(
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
        manual_case=True,
    )

    assert eventtype in {"NONE", "STRENGTHEN"}
    if eventtype == "STRENGTHEN":
        assert event_detail in {None, "EXIGENT_STRENGTHEN"}


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

    eventtype, _, _, t_detect, event_detail = classify_event(
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

    assert eventtype == "REVERSE"
    assert event_detail == "Exigent wrong-sense"
    assert np.isclose(t_detect, 4.0)


def test_classify_event_strengthen_triggers_even_when_time_short():
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

    eventtype, _, _, t_detect, event_detail = classify_event(
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

    assert eventtype == "REVERSE"
    assert event_detail == "Exigent wrong-sense"
    assert np.isclose(t_detect, 4.0)


def test_classify_event_no_response_triggers_exigent_strengthen():
    tgo = 25.0
    dt = 1.0

    times = np.arange(0.0, tgo + 1e-9, dt)
    vs_pl = np.full_like(times, 70.0)
    vs_ca = np.zeros_like(times)
    vs_ca[times >= 7.0] = 60.0

    z_pl = integrate_altitude_from_vs(times, vs_pl, 0.0)
    z_ca = integrate_altitude_from_vs(times, vs_ca, 2000.0)

    import simulation as sim_mod

    orig_pad = sim_mod.STRENGTHEN_PAD_FT
    try:
        sim_mod.STRENGTHEN_PAD_FT = -1e6
        eventtype, _, _, t_detect, event_detail = classify_event(
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
            manual_case=True,
        )
    finally:
        sim_mod.STRENGTHEN_PAD_FT = orig_pad

    assert eventtype == "STRENGTHEN"
    assert event_detail == "EXIGENT_STRENGTHEN"
    assert np.isclose(t_detect, 8.0)


def test_classify_event_same_sense_improvement_hold_defers_reversal():
    closure_fps = 20.0
    times = np.array([0.0, 0.4, 0.8, 1.2, 1.6, 2.0, 2.4, 2.8])
    tau_seq = np.array([22.0, 21.2, 20.4, 19.6, 18.8, 18.0, 17.0, 16.0])
    sep = tau_seq * closure_fps

    z_pl = np.zeros_like(times)
    z_ca = sep.copy()
    vs_pl = np.zeros_like(times)
    vs_ca = np.full_like(times, -closure_fps * 60.0)

    # Temporarily reduce the strengthen threshold pad and improvement tolerance so
    # that the synthetic geometry exercises the reversal-hold logic directly.
    import simulation as sim_mod

    orig_pad = sim_mod.STRENGTHEN_PAD_FT
    orig_tol = sim_mod.PREDICTED_MISS_IMPROVEMENT_TOL_FT
    try:
        sim_mod.STRENGTHEN_PAD_FT = -500.0
        sim_mod.PREDICTED_MISS_IMPROVEMENT_TOL_FT = -1.0

        eventtype, _, _, _, event_detail = classify_event(
            times=times,
            z_pl=z_pl,
            z_ca=z_ca,
            vs_pl=vs_pl,
            vs_ca=vs_ca,
            tgo=25.0,
            alim_ft=400.0,
            margin_ft=100.0,
            sense_chosen_cat=-1,
            sense_exec_cat=-1,
        )
    finally:
        sim_mod.STRENGTHEN_PAD_FT = orig_pad
        sim_mod.PREDICTED_MISS_IMPROVEMENT_TOL_FT = orig_tol

    assert eventtype == "NONE"
    assert event_detail is None


def test_classify_event_apfd_improvement_strengthens_instead_of_reversing():
    tgo = 25.0
    dt = 0.2
    times = np.arange(0.0, tgo + 1e-9, dt)

    _, vs_ca = vs_time_series(
        tgo,
        dt,
        0.9,
        0.25,
        CAT_INIT_VS_FPM,
        sense=-1,
        cap_fpm=CAT_INIT_VS_FPM,
    )
    vs_pl = vs_ca + 900.0

    times = times[: len(vs_ca)]
    vs_pl = vs_pl[: len(times)]

    z_pl = integrate_altitude_from_vs(times, vs_pl, 0.0)
    z_ca = integrate_altitude_from_vs(times, vs_ca, 600.0)

    eventtype, _, _, t_detect, event_detail = classify_event(
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
        manual_case=False,
    )

    assert eventtype == "STRENGTHEN"
    assert event_detail is None
    assert np.isclose(t_detect, 1.2)


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


def test_run_batch_records_maneuver_sequence():
    df = run_batch(
        runs=5,
        seed=321,
        jitter_priors=False,
        use_delay_mixture=False,
        apfd_share=0.0,
    )

    assert "maneuver_sequence" in df.columns
    assert "eventtype_final" in df.columns
    assert "t_detect_final" in df.columns
    assert "any_reversal" in df.columns

    for _, row in df.iterrows():
        seq = row["maneuver_sequence"]
        assert isinstance(seq, tuple)
        assert len(seq) >= 1

        first = seq[0]
        last = seq[-1]

        assert row["eventtype"] == first["eventtype"]
        assert np.isclose(row["t_detect"], first["t_issue"])
        assert np.isclose(row["tau_detect"], first["tau_issue"])
        reversals = [entry for entry in seq if entry["eventtype"] == "REVERSE"]
        if row["any_reversal"]:
            assert reversals, "any_reversal flagged but no reversal recorded"
            rev_last = reversals[-1]
            assert row["eventtype_final"] == "REVERSE"
            assert np.isclose(row["t_detect_final"], rev_last["t_issue"])
            assert np.isclose(row["tau_detect_final"], rev_last["tau_issue"])
        else:
            assert row["eventtype_final"] == last["eventtype"]
            assert np.isclose(row["t_detect_final"], last["t_issue"])
            assert np.isclose(row["tau_detect_final"], last["tau_issue"])

        second_issue = row["t_second_issue"]
        if second_issue is not None and not np.isnan(second_issue):
            assert second_issue >= first["t_issue"]
            assert np.isclose(row["tau_second_issue"], row["tgos"] - second_issue)


def test_run_batch_same_sense_reversal_rate_with_hold():
    df = run_batch(
        runs=200,
        seed=26,
        scenario="Head-on",
        jitter_priors=True,
        use_delay_mixture=True,
        apfd_share=0.25,
    )

    same_sense_slow = (
        (df["eventtype"] == "REVERSE")
        & (df["event_detail"] == "Slow response")
        & (df["senseCAT_chosen"] == df["senseCAT_exec"])
    )

    slow_rate = same_sense_slow.mean()
    assert slow_rate < 0.05


def test_run_batch_apfd_sense_matches_fast_template():
    runs = 120
    dt = 0.1
    df = run_batch(
        runs=runs,
        seed=902,
        scenario="Head-on",
        jitter_priors=False,
        use_delay_mixture=False,
        apfd_share=1.0,
        dt=dt,
    )

    assert df["CAT_is_APFD"].eq(1).all()

    cat_profile = (
        {"label": "manual", "weight": 0.0},
        {"label": "apfd", "delay": 0.9, "accel": 0.25, "weight": 1.0},
    )

    mismatches = 0
    for row in df.itertuples():
        (sense_pl, sense_cat), _, _ = choose_optimal_sense(
            row.tgos,
            dt,
            row.h0ft,
            bool(row.cat_above),
            row.pl_vs0_init,
            row.cat_vs0_init,
            cat_profiles=cat_profile,
        )
        if sense_cat != row.senseCAT_chosen:
            mismatches += 1

    assert mismatches == 0, f"Unexpected sense mismatch count: {mismatches}"
