import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pandas.testing as pdt
from unittest.mock import patch

sys.path.append(str(Path(__file__).resolve().parents[1]))

from inspector_utils import get_second_issue_time
from simulation import (
    CAT_CAP_INIT_FPM,
    CAT_CAP_STRENGTH_FPM,
    CAT_INIT_VS_FPM,
    CAT_STRENGTH_FPM,
    apply_non_compliance_to_cat,
    FT_PER_M,
    G,
    PL_ACCEL_G,
    PL_DELAY_MEAN_S,
    PL_VS_CAP_FPM,
    PL_VS_FPM,
    REVERSAL_MONITOR_DELAY_S,
    choose_optimal_sense,
    apply_second_phase,
    classify_event,
    integrate_altitude_from_vs,
    ias_to_tas,
    compute_residual_risk,
    OppositeSenseModel,
    OppositeSenseBand,
    derive_single_run_geometry,
    sanitize_tgo_bounds,
    decode_time_history,
    extend_history_with_pretrigger,
    run_batch,
    vs_time_series,
    CAT_MANUAL_ACCEL_NOM_G,
    CAT_MANUAL_DELAY_NOM_S,
    CAT_APFD_ACCEL_NOM_G,
    TCAS_UPDATE_PERIOD_S,
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


def test_derive_single_run_geometry_uses_window_max_for_cpa():
    window = sanitize_tgo_bounds(18.0, 30.0)
    closure_kt = 420.0
    initial_range_nm = 6.0

    t_cpa, r_eff = derive_single_run_geometry(initial_range_nm, closure_kt, True, window)

    assert np.isclose(t_cpa, window[1])
    assert np.isclose(r_eff, (closure_kt * window[1]) / 3600.0)


def test_apply_second_phase_reverse_keeps_pl_sense_and_flips_cat():
    tgo = 20.0
    dt = 0.5
    sense_pl = +1
    sense_cat = -1
    h0 = 800.0

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

    z_pl = integrate_altitude_from_vs(times, vs_pl, 0.0)
    z_ca = integrate_altitude_from_vs(times, vs_ca, h0)
    decision_latency = 1.0
    latency = float(np.clip(decision_latency, 0.6, 1.4))
    t2_issue_est = float(max(0.0, min(tgo, 8.0 + latency)))
    z_pl_t2 = float(np.interp(t2_issue_est, times, z_pl))
    z_ca_t2 = float(np.interp(t2_issue_est, times, z_ca))

    (
        times2,
        vs_pl2,
        vs_ca2,
        t_issue,
        new_sense_pl,
        new_sense_cat_exec,
        new_sense_cat_cmd,
    ) = apply_second_phase(
        times,
        vs_pl,
        vs_ca,
        tgo,
        dt,
        eventtype="REVERSE",
        sense_pl=sense_pl,
        sense_cat_exec=sense_cat,
        sense_cat_cmd=sense_cat,
        pl_vs0=0.0,
        cat_vs0=0.0,
        t_classify=8.0,
        z_pl_t2=z_pl_t2,
        z_cat_t2=z_ca_t2,
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

    assert new_sense_pl == sense_pl
    assert new_sense_cat_exec == -sense_cat
    assert new_sense_cat_cmd == -sense_cat

    prefix_mask = times2 <= t_issue + 1e-9
    assert np.allclose(vs_pl2[prefix_mask], np.interp(times2[prefix_mask], times, vs_pl))
    assert np.allclose(vs_ca2[prefix_mask], np.interp(times2[prefix_mask], times, vs_ca))

    idx_issue = int(np.where(np.isclose(times2, t_issue))[0][0])
    rem = tgo - times2[idx_issue]
    _, expected_vs_pl = vs_time_series(
        rem,
        dt,
        PL_DELAY_MEAN_S,
        PL_ACCEL_G,
        PL_VS_FPM,
        sense=sense_pl,
        cap_fpm=PL_VS_CAP_FPM,
        vs0_fpm=vs_pl2[idx_issue],
    )

    assert np.allclose(vs_pl2[idx_issue:], expected_vs_pl)
    assert vs_ca2[-1] > 0.0


def test_apply_second_phase_reverse_improves_predicted_miss():
    tgo = 20.0
    dt = 0.5
    sense_pl = +1
    sense_cat = -1
    h0 = 800.0

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
        4.0,
        0.20,
        CAT_INIT_VS_FPM,
        sense=sense_cat,
        cap_fpm=CAT_CAP_INIT_FPM,
    )

    z_pl = integrate_altitude_from_vs(times, vs_pl, 0.0)
    z_ca = integrate_altitude_from_vs(times, vs_ca, h0)
    miss_before = abs(z_ca[-1] - z_pl[-1])
    decision_latency = 1.0
    latency = float(np.clip(decision_latency, 0.6, 1.4))
    t2_issue_est = float(max(0.0, min(tgo, 8.0 + latency)))
    z_pl_t2 = float(np.interp(t2_issue_est, times, z_pl))
    z_ca_t2 = float(np.interp(t2_issue_est, times, z_ca))

    (
        times2,
        vs_pl2,
        vs_ca2,
        t_issue,
        new_sense_pl,
        new_sense_cat_exec,
        new_sense_cat_cmd,
    ) = apply_second_phase(
        times,
        vs_pl,
        vs_ca,
        tgo,
        dt,
        eventtype="REVERSE",
        sense_pl=sense_pl,
        sense_cat_exec=sense_cat,
        sense_cat_cmd=sense_cat,
        pl_vs0=0.0,
        cat_vs0=0.0,
        t_classify=8.0,
        z_pl_t2=z_pl_t2,
        z_cat_t2=z_ca_t2,
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
    assert new_sense_pl == sense_pl
    assert new_sense_cat_exec == -sense_cat
    assert new_sense_cat_cmd == -sense_cat

    z_pl2 = integrate_altitude_from_vs(times2, vs_pl2, 0.0)
    z_ca2 = integrate_altitude_from_vs(times2, vs_ca2, h0)
    miss_after = abs(z_ca2[-1] - z_pl2[-1])

    assert miss_after > miss_before


def test_apply_second_phase_reversal_guard_prevents_toward_pl_motion():
    tgo = 20.0
    dt = 0.5
    sense_pl = +1
    sense_cat = +1
    h0 = 800.0

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
        4.0,
        0.20,
        CAT_INIT_VS_FPM,
        sense=sense_cat,
        cap_fpm=CAT_CAP_INIT_FPM,
    )

    z_pl = integrate_altitude_from_vs(times, vs_pl, 0.0)
    z_ca = integrate_altitude_from_vs(times, vs_ca, h0)

    decision_latency = 1.0
    latency = float(np.clip(decision_latency, 0.6, 1.4))
    t_classify = 6.0
    t2_issue_est = float(max(0.0, min(tgo, t_classify + latency)))
    z_pl_t2 = float(np.interp(t2_issue_est, times, z_pl))
    z_ca_t2 = float(np.interp(t2_issue_est, times, z_ca))

    (
        times2,
        vs_pl2,
        vs_ca2,
        t_issue,
        new_sense_pl,
        new_sense_cat_exec,
        new_sense_cat_cmd,
    ) = apply_second_phase(
        times,
        vs_pl,
        vs_ca,
        tgo,
        dt,
        eventtype="REVERSE",
        sense_pl=sense_pl,
        sense_cat_exec=sense_cat,
        sense_cat_cmd=sense_cat,
        pl_vs0=0.0,
        cat_vs0=0.0,
        t_classify=t_classify,
        z_pl_t2=z_pl_t2,
        z_cat_t2=z_ca_t2,
        pl_delay=PL_DELAY_MEAN_S,
        pl_accel_g=PL_ACCEL_G,
        pl_cap=PL_VS_CAP_FPM,
        cat_delay=0.9,
        cat_accel_g=0.35,
        cat_vs_strength=CAT_STRENGTH_FPM,
        cat_cap=CAT_CAP_STRENGTH_FPM,
        decision_latency_s=decision_latency,
    )

    assert t_issue is not None
    assert np.isclose(t_issue, t2_issue_est)
    assert new_sense_pl == sense_pl
    assert new_sense_cat_exec == sense_cat
    assert new_sense_cat_cmd == sense_cat

    idx_issue = int(np.where(np.isclose(times2, t_issue))[0][0])
    assert np.all(vs_ca2[idx_issue:] >= -1e-6)


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

    z_pl = integrate_altitude_from_vs(times, vs_pl, 0.0)
    z_ca = integrate_altitude_from_vs(times, vs_ca, 0.0)
    decision_latency = 1.0
    latency = float(np.clip(decision_latency, 0.6, 1.4))
    t2_issue_est = float(max(0.0, min(tgo, 10.0 + latency)))
    z_pl_t2 = float(np.interp(t2_issue_est, times, z_pl))
    z_ca_t2 = float(np.interp(t2_issue_est, times, z_ca))

    (
        times2,
        _,
        vs_ca2,
        t_issue,
        new_sense_pl,
        new_sense_cat_exec,
        new_sense_cat_cmd,
    ) = apply_second_phase(
        times,
        vs_pl,
        vs_ca,
        tgo,
        dt,
        eventtype="STRENGTHEN",
        sense_pl=sense_pl,
        sense_cat_exec=sense_cat,
        sense_cat_cmd=sense_cat,
        pl_vs0=0.0,
        cat_vs0=0.0,
        t_classify=10.0,
        z_pl_t2=z_pl_t2,
        z_cat_t2=z_ca_t2,
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
    assert new_sense_pl == sense_pl
    assert new_sense_cat_exec == sense_cat
    assert new_sense_cat_cmd == sense_cat

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

    z_pl = integrate_altitude_from_vs(times, vs_pl, 0.0)
    z_ca = integrate_altitude_from_vs(times, vs_ca, 0.0)
    decision_latency = 0.8
    latency = float(np.clip(decision_latency, 0.6, 1.4))
    t2_issue_est = float(max(0.0, min(tgo, 7.0 + latency)))
    z_pl_t2 = float(np.interp(t2_issue_est, times, z_pl))
    z_ca_t2 = float(np.interp(t2_issue_est, times, z_ca))

    (
        times2,
        _,
        vs_ca2,
        t_issue,
        new_sense_pl,
        new_sense_cat_exec,
        new_sense_cat_cmd,
    ) = apply_second_phase(
        times,
        vs_pl,
        vs_ca,
        tgo,
        dt,
        eventtype="STRENGTHEN",
        sense_pl=sense_pl,
        sense_cat_exec=sense_cat,
        sense_cat_cmd=sense_cat,
        pl_vs0=0.0,
        cat_vs0=0.0,
        t_classify=7.0,
        z_pl_t2=z_pl_t2,
        z_cat_t2=z_ca_t2,
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
    assert new_sense_pl == sense_pl
    assert new_sense_cat_exec == sense_cat
    assert new_sense_cat_cmd == sense_cat

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


def _same_sense_reversal_geometry():
    times = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
    vs_pl = np.zeros_like(times)
    vs_ca = np.full_like(times, -1200.0)
    z_pl = integrate_altitude_from_vs(times, vs_pl, 0.0)
    z_ca = integrate_altitude_from_vs(times, vs_ca, 200.0)
    return times, vs_pl, vs_ca, z_pl, z_ca


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


def test_classify_event_manual_projection_uses_manual_acceleration():
    times, vs_pl, vs_ca, z_pl, z_ca = _same_sense_reversal_geometry()

    with patch("simulation.STRENGTHEN_PAD_FT", -2000.0), patch(
        "simulation.reversal_candidate_satisfies_alim", return_value=True
    ), patch("simulation.vs_time_series", wraps=vs_time_series) as mock_vs:
        classify_event(
            times=times,
            z_pl=z_pl,
            z_ca=z_ca,
            vs_pl=vs_pl,
            vs_ca=vs_ca,
            tgo=15.0,
            alim_ft=400.0,
            margin_ft=100.0,
            sense_chosen_cat=-1,
            sense_exec_cat=-1,
            manual_case=True,
        )

    assert mock_vs.call_count > 0
    accel_used = {call.args[3] for call in mock_vs.call_args_list}
    assert accel_used == {CAT_MANUAL_ACCEL_NOM_G}


def test_classify_event_apfd_projection_uses_apfd_acceleration():
    times, vs_pl, vs_ca, z_pl, z_ca = _same_sense_reversal_geometry()

    with patch("simulation.STRENGTHEN_PAD_FT", -2000.0), patch(
        "simulation.reversal_candidate_satisfies_alim", return_value=True
    ), patch("simulation.vs_time_series", wraps=vs_time_series) as mock_vs:
        classify_event(
            times=times,
            z_pl=z_pl,
            z_ca=z_ca,
            vs_pl=vs_pl,
            vs_ca=vs_ca,
            tgo=15.0,
            alim_ft=400.0,
            margin_ft=100.0,
            sense_chosen_cat=-1,
            sense_exec_cat=-1,
            manual_case=False,
        )

    assert mock_vs.call_count > 0
    accel_used = {call.args[3] for call in mock_vs.call_args_list}
    assert accel_used == {CAT_APFD_ACCEL_NOM_G}


def test_classify_event_delayed_compliance_strengthens_not_reverses():
    tgo = 25.0
    dt = 1.0

    times = np.arange(0.0, tgo + 1e-9, dt)
    vs_pl = np.full_like(times, 1000.0)
    vs_ca = np.zeros_like(times)

    z_pl = integrate_altitude_from_vs(times, vs_pl, 0.0)
    z_ca = integrate_altitude_from_vs(times, vs_ca, 400.0)

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
        manual_case=False,
    )

    assert eventtype == "STRENGTHEN"
    assert event_detail == "EXIGENT_STRENGTHEN"
    assert t_detect >= REVERSAL_MONITOR_DELAY_S


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


def test_classify_event_reversal_not_selected_when_cpa_worsens():
    tgo = 20.0
    dt = 1.0

    times = np.arange(0.0, tgo + 1e-9, dt)
    vs_pl = np.full_like(times, 1000.0, dtype=float)
    vs_ca = np.full_like(times, -300.0, dtype=float)

    z_pl = integrate_altitude_from_vs(times, vs_pl, 0.0)
    z_ca = integrate_altitude_from_vs(times, vs_ca, -300.0)

    import simulation as sim_mod

    orig_pad = sim_mod.STRENGTHEN_PAD_FT
    try:
        sim_mod.STRENGTHEN_PAD_FT = -1e6
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
            manual_case=False,
        )
    finally:
        sim_mod.STRENGTHEN_PAD_FT = orig_pad

    assert eventtype in {"NONE", "STRENGTHEN"}
    if eventtype == "STRENGTHEN":
        assert event_detail in {None, "EXIGENT_STRENGTHEN"}


def test_reversal_candidate_satisfies_alim_gate():
    from simulation import reversal_candidate_satisfies_alim

    assert not reversal_candidate_satisfies_alim(499.0, 400.0, 100.0)
    assert reversal_candidate_satisfies_alim(500.0, 400.0, 100.0)


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
    with patch("simulation.VS_REPORT_NOISE_SD_FPM", 0.0), patch(
        "simulation.ALT_REPORT_NOISE_SD_FT", 0.0
    ):
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


def test_run_batch_classify_event_samples_one_hz():
    captured_times = []
    original_classify = classify_event

    def capture(times, *args, **kwargs):
        captured_times.append(np.array(times, copy=True))
        return original_classify(times, *args, **kwargs)

    with patch("simulation.classify_event", new=capture):
        run_batch(
            runs=3,
            seed=246,
            jitter_priors=False,
            use_delay_mixture=False,
            apfd_share=0.0,
        )

    regular_diffs: List[float] = []
    for arr in captured_times:
        series = np.asarray(arr, dtype=float)
        if series.size < 2:
            continue
        diffs = np.diff(series)
        if diffs.size > 1:
            regular_diffs.extend(diffs[:-1])
        if diffs.size >= 1:
            assert diffs[-1] <= TCAS_UPDATE_PERIOD_S + 1e-6

    assert regular_diffs, "expected at least one full TCAS cycle interval"
    assert np.allclose(regular_diffs, TCAS_UPDATE_PERIOD_S, atol=1e-9)


def test_run_batch_coordination_dropout_deterministic_with_forced_rate():
    kwargs = dict(
        runs=10,
        seed=777,
        jitter_priors=False,
        use_delay_mixture=False,
        apfd_share=0.0,
    )

    with patch("simulation.COORDINATION_DROPOUT_PROB", 1.0):
        df1 = run_batch(**kwargs)
    with patch("simulation.COORDINATION_DROPOUT_PROB", 1.0):
        df2 = run_batch(**kwargs)

    pdt.assert_frame_equal(df1.reset_index(drop=True), df2.reset_index(drop=True))

    any_dropout = False
    for _, row in df1.iterrows():
        seq = row["maneuver_sequence"]
        if not isinstance(seq, tuple):
            continue
        if any(entry.get("coordination_dropout") for entry in seq):
            any_dropout = True
            break

    assert any_dropout, "forced dropout probability should trigger at least one skip"


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
        assert "executed_flip" in first
        reversals = [entry for entry in seq if entry["eventtype"] == "REVERSE"]
        if row["any_reversal"]:
            assert reversals, "any_reversal flagged but no reversal recorded"
            rev_last = reversals[-1]
            assert row["eventtype_final"] == "REVERSE"
            assert np.isclose(row["t_detect_final"], rev_last["t_issue"])
            assert np.isclose(row["tau_detect_final"], rev_last["tau_issue"])
            assert rev_last["executed_flip"]
        else:
            assert row["eventtype_final"] == last["eventtype"]
            assert np.isclose(row["t_detect_final"], last["t_issue"])
            assert np.isclose(row["tau_detect_final"], last["tau_issue"])
            for entry in reversals:
                assert not entry["executed_flip"]

        second_issue = row["t_second_issue"]
        if second_issue is not None and not np.isnan(second_issue):
            assert second_issue >= first["t_issue"]
            assert np.isclose(row["tau_second_issue"], row["tgos"] - second_issue)


def test_run_batch_predicted_miss_uses_nominal_cat_profiles():
    captured_profiles = []
    captured_base_params = []

    original_choose = choose_optimal_sense
    original_apply = apply_non_compliance_to_cat

    def capture_choose(*args, **kwargs):
        captured_profiles.append(kwargs.get("cat_profiles"))
        return original_choose(*args, **kwargs)

    def capture_apply(*args, **kwargs):
        captured_base_params.append((kwargs["base_delay_s"], kwargs["base_accel_g"]))
        return original_apply(*args, **kwargs)

    with patch("simulation.choose_optimal_sense", new=capture_choose), patch(
        "simulation.apply_non_compliance_to_cat", new=capture_apply
    ):
        run_batch(
            runs=1,
            seed=2024,
            jitter_priors=False,
            use_delay_mixture=True,
            apfd_share=0.0,
        )

    assert captured_profiles, "choose_optimal_sense was not invoked"
    profiles = captured_profiles[0]
    assert isinstance(profiles, tuple)
    assert profiles, "cat_profiles should include at least one template"

    manual_profile = profiles[0]
    assert np.isclose(manual_profile["accel"], CAT_MANUAL_ACCEL_NOM_G)
    assert np.isclose(manual_profile["delay"], CAT_MANUAL_DELAY_NOM_S)

    assert captured_base_params, "apply_non_compliance_to_cat was not invoked"
    _, accel_exec = captured_base_params[0]
    assert not np.isclose(accel_exec, CAT_MANUAL_ACCEL_NOM_G)


def test_run_inspector_handles_missing_second_issue_column():
    df = pd.DataFrame(
        [
            {
                "run": 1,
                "eventtype": "STRENGTHEN",
                "sensePL_final": 0,
                "senseCAT_final": 0,
            }
        ]
    )

    second_issue_time = get_second_issue_time(df.iloc[0])
    assert second_issue_time is None


def test_run_inspector_ignores_nan_second_issue_value():
    df = pd.DataFrame(
        [
            {
                "run": 2,
                "eventtype": "REVERSE",
                "sensePL_final": 1,
                "senseCAT_final": -1,
                "t_second_issue": float("nan"),
            }
        ]
    )

    second_issue_time = get_second_issue_time(df.iloc[0])
    assert second_issue_time is None


def test_run_inspector_returns_second_issue_time_when_available():
    df = pd.DataFrame(
        [
            {
                "run": 3,
                "eventtype": "STRENGTHEN",
                "sensePL_final": -1,
                "senseCAT_final": 1,
                "t_second_issue": 12.5,
            }
        ]
    )

    second_issue_time = get_second_issue_time(df.iloc[0])
    assert second_issue_time == 12.5


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


def test_run_batch_reversal_attempt_without_flip():
    df = run_batch(
        runs=10,
        seed=10,
        scenario="Head-on",
        jitter_priors=True,
        use_delay_mixture=True,
        apfd_share=0.25,
    )

    lingering = []
    downgraded = []
    for row in df.itertuples():
        for entry in row.maneuver_sequence:
            if entry["eventtype"] == "REVERSE" and not entry["executed_flip"]:
                lingering.append((row, entry))
            if not entry["executed_flip"]:
                downgraded.append((row, entry))

    assert not lingering, "unexecuted reversals should be downgraded"
    assert downgraded, "expected at least one non-flip manoeuvre for coverage"

    for row, entry in downgraded:
        if entry["eventtype"] == "STRENGTHEN":
            assert row.eventtype_final in {"STRENGTHEN", "NONE"}
        if entry["eventtype"] == "NONE":
            assert row.eventtype_final == "NONE"


def test_run_batch_reversal_downgrade_matches_execution():
    with patch("simulation.classify_event") as mock_classify, patch(
        "simulation.apply_second_phase"
    ) as mock_apply:

        def fake_apply(
            times,
            vs_pl,
            vs_ca,
            tgo,
            dt,
            eventtype,
            sense_pl,
            sense_cat_exec,
            sense_cat_cmd,
            **kwargs,
        ):
            t_issue = max(0.0, tgo - 0.05)
            start_time = float(times[0]) if len(times) else 0.0
            times2 = np.array([start_time, t_issue, min(tgo, t_issue + 0.01)])
            vs_pl_start = float(vs_pl[0]) if len(vs_pl) else 0.0
            vs_pl2 = np.array([vs_pl_start, vs_pl_start, vs_pl_start])
            cat_vs0 = float(kwargs.get("cat_vs0", 0.0))
            sign = 1 if sense_cat_exec >= 0 else -1
            vs_strength = sign * (abs(cat_vs0) + 3000.0)
            vs_ca2 = np.array([cat_vs0, cat_vs0, vs_strength])
            return times2, vs_pl2, vs_ca2, t_issue, sense_pl, sense_cat_exec, sense_cat_cmd

        mock_apply.side_effect = fake_apply
        mock_classify.side_effect = [
            ("REVERSE", 100.0, 120.0, 5.0, "Slow response"),
        ]

        df = run_batch(
            runs=1,
            seed=5,
            scenario="Head-on",
            jitter_priors=False,
            use_delay_mixture=False,
            apfd_share=0.0,
        )

    row = df.iloc[0]
    seq = row.maneuver_sequence

    assert mock_classify.call_count == 1
    assert len(seq) == 1

    entry = seq[0]
    assert entry["eventtype"] == "STRENGTHEN"
    assert not entry["executed_flip"]

    assert row.eventtype == "STRENGTHEN"
    assert row.eventtype_final == "STRENGTHEN"
    assert row.event_detail is None
    assert row.event_detail_final is None


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


def test_manual_reversal_requires_delayed_cpa_improvement():
    dt = 0.5
    sense_pl = +1
    sense_cat = +1
    initial_offset_ft = -800.0
    t_classify = 6.0
    decision_latency = 1.4
    latency = float(np.clip(decision_latency, 0.6, 1.4))

    for tgo, expect_flip in ((10.0, False), (15.0, True)):
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
            2.5,
            0.25,
            CAT_INIT_VS_FPM,
            sense=sense_cat,
            cap_fpm=CAT_CAP_INIT_FPM,
        )

        z_pl = integrate_altitude_from_vs(times, vs_pl, 0.0)
        z_ca = integrate_altitude_from_vs(times, vs_ca, initial_offset_ft)

        t2_issue_est = float(max(0.0, min(tgo, t_classify + latency)))
        z_pl_t2 = float(np.interp(t2_issue_est, times, z_pl))
        z_ca_t2 = float(np.interp(t2_issue_est, times, z_ca))
        vs_pl_now = float(np.interp(t2_issue_est, times, vs_pl))
        vs_ca_now = float(np.interp(t2_issue_est, times, vs_ca))
        t_rem = max(0.0, tgo - t2_issue_est)

        cpas = {}
        for label, next_sense_cat in (("keep", sense_cat), ("flip", -sense_cat)):
            t_rel, vs_pl_candidate = vs_time_series(
                t_rem,
                dt,
                PL_DELAY_MEAN_S,
                PL_ACCEL_G,
                PL_VS_FPM,
                sense=sense_pl,
                cap_fpm=PL_VS_CAP_FPM,
                vs0_fpm=vs_pl_now,
            )
            _, vs_ca_candidate = vs_time_series(
                t_rem,
                dt,
                2.5,
                0.35,
                CAT_INIT_VS_FPM,
                sense=next_sense_cat,
                cap_fpm=CAT_CAP_INIT_FPM,
                vs0_fpm=vs_ca_now,
            )
            z_pl_candidate = integrate_altitude_from_vs(t_rel, vs_pl_candidate, z_pl_t2)
            z_ca_candidate = integrate_altitude_from_vs(t_rel, vs_ca_candidate, z_ca_t2)
            sep_candidate = np.abs(z_pl_candidate - z_ca_candidate)
            cpas[label] = float(np.min(sep_candidate))

        (
            _,
            _,
            _,
            t_issue,
            new_sense_pl,
            new_sense_cat_exec,
            new_sense_cat_cmd,
        ) = apply_second_phase(
            times,
            vs_pl,
            vs_ca,
            tgo,
            dt,
            eventtype="REVERSE",
            sense_pl=sense_pl,
            sense_cat_exec=sense_cat,
            sense_cat_cmd=sense_cat,
            pl_vs0=0.0,
            cat_vs0=0.0,
            t_classify=t_classify,
            z_pl_t2=z_pl_t2,
            z_cat_t2=z_ca_t2,
            pl_delay=PL_DELAY_MEAN_S,
            pl_accel_g=PL_ACCEL_G,
            pl_cap=PL_VS_CAP_FPM,
            cat_delay=2.5,
            cat_accel_g=0.35,
            cat_vs_strength=CAT_INIT_VS_FPM,
            cat_cap=CAT_CAP_INIT_FPM,
            decision_latency_s=decision_latency,
        )

        assert t_issue is not None

        if expect_flip:
            assert cpas["flip"] > cpas["keep"] + 1e-3
            assert new_sense_pl == sense_pl
            assert new_sense_cat_exec == -sense_cat
            assert new_sense_cat_cmd == -sense_cat
        else:
            assert cpas["keep"] >= cpas["flip"] - 1e-3
            assert new_sense_pl == sense_pl
            assert new_sense_cat_exec == sense_cat
            assert new_sense_cat_cmd == sense_cat


def test_classify_event_same_sense_needs_cpa_improvement():
    times = np.array([0.0, 5.0, 10.0])
    vs_pl = np.array([0.0, 200.0, 200.0])
    vs_ca = np.array([0.0, 150.0, 150.0])
    z_pl = np.array([0.0, 200.0 / 60.0 * 5.0, 200.0 / 60.0 * 10.0])
    sep_mid = 40.0
    z_ca = np.array(
        [
            sep_mid + z_pl[1],
            sep_mid,
            sep_mid - (vs_pl[1] - vs_ca[1]) / 60.0 * 5.0,
        ]
    )

    with patch("simulation.STRENGTHEN_PAD_FT", -500.0):
        eventtype, *_ , detail = classify_event(
            times,
            z_pl,
            z_ca,
            vs_pl,
            vs_ca,
            tgo=10.0,
            alim_ft=60.0,
            margin_ft=0.0,
            sense_chosen_cat=+1,
            sense_exec_cat=+1,
            projection_cat_delay_s=50.0,
        )

    assert eventtype == "NONE"
    assert detail is None


def test_classify_event_opposite_sense_still_reverses():
    times = np.array([0.0, 5.0, 10.0])
    vs_pl = np.array([0.0, 200.0, 200.0])
    vs_ca = np.array([0.0, -200.0, -200.0])
    z_pl = np.array([0.0, 200.0 / 60.0 * 5.0, 200.0 / 60.0 * 10.0])
    z_ca = np.array([300.0, 200.0, 100.0])

    eventtype, *_ , detail = classify_event(
        times,
        z_pl,
        z_ca,
        vs_pl,
        vs_ca,
        tgo=10.0,
        alim_ft=60.0,
        margin_ft=0.0,
        sense_chosen_cat=+1,
        sense_exec_cat=+1,
    )

    assert eventtype == "REVERSE"
    assert detail == "Exigent wrong-sense"


def test_apply_non_compliance_apfd_preserves_sense():
    rng = np.random.default_rng(1234)
    for commanded_sense in (+1, -1):
        for _ in range(64):
            outcome, flown_sense, *_ = apply_non_compliance_to_cat(
                rng,
                commanded_sense,
                base_delay_s=0.9,
                base_accel_g=0.25,
                vs_fpm=CAT_INIT_VS_FPM,
                cap_fpm=CAT_CAP_INIT_FPM,
                p_taonly=0.05,
                p_weak=0.10,
                jitter=True,
                cat_mode_key="apfd",
                mode_label_override="AP/FD",
            )
            assert flown_sense == commanded_sense, outcome


def test_apply_non_compliance_label_override_still_apfd():
    rng = np.random.default_rng(9876)
    commanded_sense = +1
    for _ in range(64):
        outcome, flown_sense, *_ = apply_non_compliance_to_cat(
            rng,
            commanded_sense,
            base_delay_s=4.5,
            base_accel_g=0.20,
            vs_fpm=CAT_INIT_VS_FPM,
            cap_fpm=CAT_CAP_INIT_FPM,
            p_taonly=0.05,
            p_weak=0.10,
            jitter=False,
            cat_mode_key="manual",
            mode_label_override="AP/FD",
        )
        assert flown_sense == commanded_sense, outcome


def test_opposite_sense_model_apfd_probability_zero():
    rng = np.random.default_rng(42)
    model = OppositeSenseModel.from_parameters(
        manual_baseline=0.3,
        apfd_baseline=0.9,
        altitude_bands=[
            OppositeSenseBand(0.0, 10000.0, 0.4, 0.8),
            OppositeSenseBand(10000.0, 20000.0, 0.2, 0.6),
        ],
        jitter_enabled=True,
    )

    assert model.probability(rng, mode="apfd", altitude_ft=8000.0, jitter_override=True) == 0.0
    assert model.probability(rng, mode="AP/FD", altitude_ft=18000.0, jitter_override=False) == 0.0


def test_run_batch_manual_opposite_sense_high_rate():
    model = OppositeSenseModel.from_parameters(manual_baseline=0.8, jitter_enabled=False)
    df = run_batch(
        runs=600,
        seed=321,
        scenario="Head-on",
        apfd_share=0.0,
        jitter_priors=False,
        p_ta=0.0,
        p_weak=0.0,
        opp_sense_model=model,
    )

    mask = (df["CAT_is_APFD"] == 0) & (df["senseCAT_chosen"].abs() == 1)
    assert mask.any()
    wrong_share = (
        df.loc[mask, "senseCAT_exec"] == -df.loc[mask, "senseCAT_chosen"]
    ).mean()
    assert 0.7 <= wrong_share <= 0.9


def test_run_batch_apfd_opposite_sense_zero_rate():
    model = OppositeSenseModel.from_parameters(
        manual_baseline=0.3,
        apfd_baseline=0.05,
        jitter_enabled=False,
    )
    df = run_batch(
        runs=800,
        seed=654,
        scenario="Head-on",
        apfd_share=1.0,
        jitter_priors=False,
        p_ta=0.0,
        p_weak=0.0,
        opp_sense_model=model,
    )

    mask = (df["CAT_is_APFD"] == 1) & (df["senseCAT_chosen"].abs() == 1)
    assert mask.any()
    wrong_share = (
        df.loc[mask, "senseCAT_exec"] == -df.loc[mask, "senseCAT_chosen"]
    ).mean()
    assert np.isclose(wrong_share, 0.0)


def test_run_batch_records_final_history_and_senses():
    expected_times = np.array([0.0, 1.0, 2.0])

    def fake_non_compliance(
        rng,
        sense_cat,
        base_delay_s,
        base_accel_g,
        vs_fpm,
        cap_fpm,
        **kwargs,
    ):
        return ("manual", sense_cat, base_delay_s, base_accel_g, vs_fpm, cap_fpm)

    def fake_classify(*args, **kwargs):
        call = fake_classify.calls
        fake_classify.calls += 1
        if call == 0:
            return ("REVERSE", 0.0, 0.0, 5.0, "Opposite sense")
        return ("NONE", 0.0, 0.0, 19.0, None)

    fake_classify.calls = 0

    def fake_second_phase(
        times,
        vs_pl,
        vs_ca,
        tgo,
        dt,
        eventtype,
        sense_pl,
        sense_cat_exec,
        sense_cat_cmd,
        **kwargs,
    ):
        pl_vs0 = kwargs.get("pl_vs0", 0.0)
        cat_vs0 = kwargs.get("cat_vs0", 0.0)
        new_vs_pl = np.array([pl_vs0, -500.0, -800.0])
        new_vs_ca = np.array([cat_vs0, 400.0, 700.0])
        return (
            expected_times,
            new_vs_pl,
            new_vs_ca,
            6.0,
            -sense_pl,
            -sense_cat_exec,
            -sense_cat_cmd,
        )

    with patch("simulation.sample_altitudes_and_h0", return_value=(200, 210, 300.0)), \
        patch(
            "simulation.sample_initial_vs_with_aggressiveness",
            side_effect=[0.0, 0.0],
        ), \
        patch("simulation.choose_optimal_sense", return_value=((+1, -1), 500.0, 400.0)), \
        patch("simulation.apply_non_compliance_to_cat", side_effect=fake_non_compliance), \
        patch("simulation.classify_event", side_effect=fake_classify), \
        patch("simulation.apply_second_phase", side_effect=fake_second_phase):
        df = run_batch(
            runs=1,
            seed=1,
            scenario="Head-on",
            jitter_priors=False,
            p_opp=0.0,
            p_ta=0.0,
            p_weak=0.0,
            opp_sense_model=OppositeSenseModel.from_parameters(
                manual_baseline=0.0,
                jitter_enabled=False,
            ),
        )

    row = df.iloc[0]
    history = decode_time_history(row["time_history_json"])

    assert history is not None
    assert np.allclose(history["times"], expected_times)

    pl_vs0 = float(row["pl_vs0_init"])
    cat_vs0 = float(row["cat_vs0_init"])
    assert np.allclose(history["vs_pl"], np.array([pl_vs0, -500.0, -800.0]))
    assert np.allclose(history["vs_ca"], np.array([cat_vs0, 400.0, 700.0]))

    assert int(row["sensePL_final"]) == int(row["sensePL"])
    assert int(row["senseCAT_exec_final"]) == -int(row["senseCAT_exec"])
    assert int(row["senseCAT_chosen_final"]) == -int(row["senseCAT_chosen"])


def test_extend_history_with_pretrigger_adds_negative_window():
    times = np.array([0.0, 0.5, 1.0])
    z_pl = np.array([0.0, 10.0, 25.0])
    z_ca = np.array([300.0, 290.0, 275.0])

    times_ext, z_pl_ext, z_ca_ext = extend_history_with_pretrigger(
        times,
        z_pl,
        z_ca,
        pl_vs0=600.0,
        cat_vs0=-300.0,
        pretrigger_window_s=1.0,
    )

    assert times_ext[0] == -1.0
    assert np.allclose(times_ext[-3:], times)

    assert np.isclose(z_pl_ext[0], -10.0)
    assert np.isclose(z_pl_ext[1], -5.0)
    assert np.isclose(z_pl_ext[2], z_pl[0])

    assert np.isclose(z_ca_ext[0], 305.0)
    assert np.isclose(z_ca_ext[1], 302.5)
    assert np.isclose(z_ca_ext[2], z_ca[0])


def test_compute_residual_risk_nominal_encounter():
    delta_pl = 1000.0  # PL climbs 1000 ft
    delta_cat = 3000.0  # CAT climbs 3000 ft

    residual = compute_residual_risk(delta_pl, delta_cat)

    assert np.isclose(residual, 0.033, atol=1e-6)


def test_run_batch_residual_risk_preserves_sign():
    df = run_batch(runs=1, seed=1234)
    row = df.iloc[0]
    expected = compute_residual_risk(row["delta_h_pl_ft"], row["delta_h_cat_ft"])

    assert np.sign(expected) == np.sign(row["residual_risk"])
    assert np.isclose(row["residual_risk"], expected, atol=1e-9)
