# app_streamlit.py
# ACAS/TCAS Monte‑Carlo v2 (Streamlit)
# --------------------------------------------------------------
# Self-contained Streamlit app + simulation engine.
# Implements:
#   • Two‑phase RA execution (STRENGTHEN ~±2500 fpm, REVERSE)
#   • Altitude‑dependent ALIM (~300–700 ft)
#   • RA timing constrained to 15–35 s before CPA
#   • Tuned non‑compliance priors (opp≈1%, TA‑only≈0.3%, weak≈30%)
#   • “Level‑off” context when aggressiveness=0.0 (slider far left)
#   • Method‑B‑like compliance label
#
# Key references (for README/commit):
#   - EUROCONTROL ACAS II Guide (2025) – RA types, strengthen/reversal, AP/FD TCAS
#   - FAA Introduction to TCAS II v7.1 – RA behaviour & targets
#   - ACAS II Bulletin #13 – rarity of reversal RAs
#   These underpin modelling choices (ALIM bands, timing, second-phase RA execution).  # noqa
#
# Run:
#   pip install streamlit numpy pandas matplotlib
#   streamlit run app_streamlit.py
# --------------------------------------------------------------
import math
from typing import Tuple, Optional, List, Dict
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
# ---------------------------- Constants ---------------------------- #
G = 9.80665
FT_PER_M = 3.28084
MS_PER_FPM = 0.00508  # 1 fpm = 0.00508 m/s
# Performance‑limited (PL) aircraft (more conservative than before)
PL_DELAY_MEAN_S = 2.2
PL_DELAY_SD_S   = 0.4
PL_ACCEL_G      = 0.10
PL_VS_FPM       = 500.0
PL_VS_CAP_FPM   = 500.0
# CAT (transport) nominal v7.1 behaviour
CAT_INIT_VS_FPM      = 1500.0     # initial corrective RA
CAT_STRENGTH_FPM     = 2500.0     # strengthened RA
CAT_CAP_INIT_FPM     = 2000.0
CAT_CAP_STRENGTH_FPM = 3000.0
# RA timing window
TGO_MIN_S = 15.0
TGO_MAX_S = 35.0
# ALIM conservative margin to trigger strengthen/reverse (ft)
ALIM_MARGIN_FT = 100.0
# ---------------------------- Utilities ---------------------------- #
def ias_to_tas(ias_kt: float, pressure_alt_ft: float) -> float:
    sigma = (1.0 - 6.875e-6 * pressure_alt_ft) ** 4.256
    sigma = max(1e-3, sigma)
    return ias_kt / math.sqrt(sigma)
def vs_time_series(t_end_s: float, dt_s: float, t_delay_s: float, a_g: float,
                   v_f_fpm: float, sense: int, cap_fpm: Optional[float] = None,
                   vs0_fpm: float = 0.0) -> Tuple[np.ndarray, np.ndarray]:
    """Delay, then ramp at acceleration to target; saturate at cap."""
    a = a_g * G
    v_target = v_f_fpm if cap_fpm is None else min(v_f_fpm, cap_fpm)
    a_fpm_s = a * 60.0
    times = np.arange(0.0, t_end_s + 1e-9, dt_s)
    vs_fpm = np.zeros_like(times, dtype=float)
    target_signed = sense * v_target
    for i, t in enumerate(times):
        if t <= t_delay_s:
            vs_fpm[i] = vs0_fpm
        else:
            te = t - t_delay_s
            delta = target_signed - vs0_fpm
            step = math.copysign(min(abs(a_fpm_s * te), abs(delta)), delta)
            vs_fpm[i] = vs0_fpm + step
    return times, vs_fpm
def integrate_altitude_from_vs(times_s: np.ndarray, vs_fpm: np.ndarray, z0_ft: float = 0.0) -> np.ndarray:
    dt = np.diff(times_s, prepend=times_s[0])
    z = z0_ft + np.cumsum((vs_fpm/60.0) * dt)
    z[0] = z0_ft
    return z
def relative_closure_kt(v1_kt: float, hdg1_deg: float, v2_kt: float, hdg2_deg: float) -> float:
    th1, th2 = math.radians(hdg1_deg), math.radians(hdg2_deg)
    v1 = np.array([v1_kt*math.sin(th1), v1_kt*math.cos(th1)])
    v2 = np.array([v2_kt*math.sin(th2), v2_kt*math.cos(th2)])
    return float(np.linalg.norm(v1 - v2))
def time_to_go_from_geometry(r0_nm: float, v_closure_kt: float) -> Optional[float]:
    if v_closure_kt <= 1e-6:
        return None
    return 3600.0 * (r0_nm / v_closure_kt)
def sample_headings(rng: np.random.Generator, scenario: str,
                    hdg1_min: float, hdg1_max: float,
                    rel_min: Optional[float] = None, rel_max: Optional[float] = None,
                    hdg2_min: Optional[float] = None, hdg2_max: Optional[float] = None) -> Tuple[float, float]:
    h1 = float(rng.uniform(hdg1_min, hdg1_max))
    if scenario == 'Custom':
        h2 = float(rng.uniform(hdg2_min, hdg2_max))
    else:
        rel = float(rng.uniform(rel_min, rel_max))
        dirsign = 1 if rng.uniform() < 0.5 else -1
        h2 = (h1 + dirsign * rel) % 360.0
    return h1, h2
def alim_ft_from_alt(pressure_alt_ft: float) -> float:
    """Approximate ALIM bands ~300–700 ft from ~FL100–FL350."""
    a = max(0.0, min(35000.0, pressure_alt_ft))
    if a <= 10000.0: return 300.0
    if a >= 35000.0: return 700.0
    return float(np.interp(a, [10000.0, 35000.0], [300.0, 700.0]))
def first_move_time(times: np.ndarray, vs: np.ndarray, tol_fpm: float = 50.0) -> float:
    base = vs[0]
    moved = np.where(np.abs(vs - base) > tol_fpm)[0]
    return float(times[moved[0]]) if len(moved) > 0 else 0.0
def compliance_score_method_b_like(sense_required: int, vs_trace: np.ndarray, times: np.ndarray,
                                   required_vs: float = CAT_INIT_VS_FPM) -> str:
    """Simplified proxy for IATA/EUROCONTROL Method‑B scoring."""
    t_move = first_move_time(times, vs_trace)
    t_eval = t_move + 3.0
    idx = np.searchsorted(times, t_eval, side='left')
    if idx >= len(times): idx = -1
    vs_eval = float(vs_trace[idx])
    if abs(vs_eval) < 100.0: return "No Response"
    if math.copysign(1, vs_eval) != sense_required: return "Opposite"
    avs = abs(vs_eval)
    if avs >= 3200.0: return "Excessive"
    return "Following" if avs >= 0.8*required_vs else "Weak"
def sample_initial_vs_with_aggressiveness(rng: np.random.Generator, aggressiveness: float,
                                          leveloff_context: bool = False) -> float:
    """When aggressiveness≈0, emulate level‑off contexts (90% near‑level)."""
    if leveloff_context:
        if rng.uniform() < 0.90:
            return float(rng.normal(0.0, 50.0))
        return float(rng.normal(rng.choice([+250.0, -250.0]), 100.0))
    aggr = max(0.0, min(1.0, aggressiveness))
    level_p = float(np.clip(0.8 - 0.6 * aggr, 0.05, 0.95))
    climb_p = descend_p = (1.0 - level_p) / 2.0
    u = rng.uniform()
    level_sd = 50.0 + 200.0 * aggr
    climb_mean = 1000.0 + 300.0 * aggr
    descend_mean = -climb_mean
    climb_sd = 200.0 + 200.0 * aggr
    if u < level_p:
        return float(rng.normal(0.0, level_sd))
    elif u < level_p + climb_p:
        return float(rng.normal(climb_mean, climb_sd))
    else:
        return float(rng.normal(descend_mean, climb_sd))
def simulate_miss_for_senses(tgo: float, dt: float, h0: float, cat_above: bool,
                             sense_pl: int, sense_cat: int,
                             pl_delay: float, pl_accel: float, pl_vs: float, pl_cap: float, pl_vs0: float,
                             cat_delay: float, cat_accel: float, cat_vs: float, cat_cap: float, cat_vs0: float) -> float:
    times, vs_pl = vs_time_series(tgo, dt, pl_delay, pl_accel, pl_vs, sense=sense_pl, cap_fpm=pl_cap, vs0_fpm=pl_vs0)
    _,     vs_ca = vs_time_series(tgo, dt, cat_delay, cat_accel, cat_vs, sense=sense_cat, cap_fpm=cat_cap, vs0_fpm=cat_vs0)
    z_pl = integrate_altitude_from_vs(times, vs_pl, 0.0)
    z_ca = integrate_altitude_from_vs(times, vs_ca, h0 if cat_above else -h0)
    return float(abs(z_ca[-1] - z_pl[-1]))
def choose_optimal_sense(tgo: float, dt: float, h0: float, cat_above: bool,
                         pl_vs0: float, cat_vs0: float,
                         cat_delay_nom: float = 5.0, cat_accel_nom: float = 0.25,
                         cat_vs: float = CAT_INIT_VS_FPM, cat_cap: float = CAT_CAP_INIT_FPM) -> Tuple[Tuple[int, int], float, float]:
    miss_updown = simulate_miss_for_senses(tgo, dt, h0, cat_above, +1, -1,
                                           PL_DELAY_MEAN_S, PL_ACCEL_G, PL_VS_FPM, PL_VS_CAP_FPM, pl_vs0,
                                           cat_delay_nom, cat_accel_nom, cat_vs, cat_cap, cat_vs0)
    miss_downup = simulate_miss_for_senses(tgo, dt, h0, cat_above, -1, +1,
                                           PL_DELAY_MEAN_S, PL_ACCEL_G, PL_VS_FPM, PL_VS_CAP_FPM, pl_vs0,
                                           cat_delay_nom, cat_accel_nom, cat_vs, cat_cap, cat_vs0)
    if miss_updown > miss_downup:
        return (+1, -1), miss_updown, miss_downup
    else:
        return (-1, +1), miss_downup, miss_updown
def apply_non_compliance_to_cat(rng: np.random.Generator, sense_cat: int,
                                delay_s: float, accel_g: float,
                                vs_fpm: float, cap_fpm: float,
                                p_opp: float = 0.010, p_taonly: float = 0.003, p_weak: float = 0.300,
                                jitter: bool = True) -> Tuple[str, int, float, float, float, float]:
    """Return (mode, sense_exec, delay, accel, vs_cmd, cap_cmd)."""
    if jitter:
        p_opp    = max(0.0, min(1.0, p_opp    * rng.uniform(0.7, 1.3)))
        p_taonly = max(0.0, min(1.0, p_taonly * rng.uniform(0.7, 1.3)))
        p_weak   = max(0.0, min(1.0, p_weak   * rng.uniform(0.7, 1.3)))
    u = rng.uniform()
    if u < p_opp:
        return ("opposite-sense", -sense_cat, delay_s, accel_g, vs_fpm, cap_fpm)
    u -= p_opp
    if u < p_taonly:
        return ("no-response", sense_cat, delay_s, 0.0, 0.0, 0.0)
    u -= p_taonly
    if u < p_weak:
        return ("weak-compliance", sense_cat, delay_s + 1.0, max(0.03, 0.6*accel_g), 0.6*vs_fpm, 0.6*cap_fpm)
    return ("compliant", sense_cat, delay_s, accel_g, vs_fpm, cap_fpm)
def classify_event(times: np.ndarray, z_pl: np.ndarray, z_ca: np.ndarray,
                   vs_pl: np.ndarray, vs_ca: np.ndarray, tgo: float,
                   alim_ft: float, margin_ft: float,
                   sense_chosen_cat: int, sense_exec_cat: int) -> Tuple[str, float, float, float]:
    """Return (event_label, minsep, sep@CPA, t_check). Keep REVERSE rare."""
    sep = np.abs(z_ca - z_pl)
    minsep = float(np.min(sep))
    sep_cpa = float(sep[-1])
    t_pl_move = first_move_time(times, vs_pl)
    t_ca_move = first_move_time(times, vs_ca)
    t_check = max(t_pl_move, t_ca_move) + 3.0
    mask = times >= t_check
    if np.any(mask):
        t_obs = times[mask]
        sep_obs = sep[mask]
        rel_rate = (vs_ca - vs_pl)/60.0
        rel_obs = rel_rate[mask]
        s_last = float(sep_obs[-1])
        r_last = float(rel_obs[-1])
        t_rem = max(0.0, tgo - t_obs[-1])
        pred_miss = abs(s_last + r_last * t_rem)
        approaching = (r_last < 0)
        thin_pred = (pred_miss < (alim_ft - margin_ft))
        if approaching and thin_pred and (sense_chosen_cat != sense_exec_cat):
            return ("REVERSE", minsep, sep_cpa, float(t_obs[-1]))
    if (minsep < (alim_ft - margin_ft)) or (sep_cpa < (alim_ft - margin_ft)):
        return ("STRENGTHEN", minsep, sep_cpa, float(t_check))
    return ("NONE", minsep, sep_cpa, float(t_check))
def apply_second_phase(times: np.ndarray, vs_pl: np.ndarray, vs_ca: np.ndarray,
                       tgo: float, dt: float, eventtype: str,
                       sense_pl: int, sense_cat_exec: int,
                       pl_delay: float = PL_DELAY_MEAN_S, pl_accel_g: float = PL_ACCEL_G, pl_cap: float = PL_VS_CAP_FPM,
                       cat_delay: float = 1.0, cat_accel_g: float = 0.20,
                       cat_vs_strength: float = CAT_STRENGTH_FPM, cat_cap: float = CAT_CAP_STRENGTH_FPM,
                       t2_issue_override: Optional[float] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Execute two‑phase RA (strengthen/reverse) and return updated (times, vs_pl, vs_ca)."""
    if eventtype not in ("STRENGTHEN", "REVERSE"):
        return times, vs_pl, vs_ca
    if t2_issue_override is None:
        t2_issue = float(times[-1] - max(0.0, tgo - times[-1])) + 1.0
    else:
        t2_issue = float(t2_issue_override)
    t_rem = max(0.0, tgo - t2_issue)
    if t_rem <= dt:
        return times, vs_pl, vs_ca
    # Current VS
    vs_pl_now = float(vs_pl[-1]); vs_ca_now = float(vs_ca[-1])
    new_sense_pl  = sense_pl if eventtype == "STRENGTHEN" else -sense_pl
    new_sense_cat = sense_cat_exec if eventtype == "STRENGTHEN" else -sense_cat_exec
    t2, vs_pl2 = vs_time_series(t_rem, dt, pl_delay, pl_accel_g, pl_cap, sense=new_sense_pl, cap_fpm=pl_cap, vs0_fpm=vs_pl_now)
    _,  vs_ca2 = vs_time_series(t_rem, dt, cat_delay, cat_accel_g, cat_vs_strength, sense=new_sense_cat, cap_fpm=cat_cap, vs0_fpm=vs_ca_now)
    times2 = np.concatenate([times, t2_issue + t2[1:]])
    vs_pl2  = np.concatenate([vs_pl, vs_pl2[1:]])
    vs_ca2  = np.concatenate([vs_ca, vs_ca2[1:]])
    return times2, vs_pl2, vs_ca2
def sample_altitudes_and_h0(rng: np.random.Generator,
                            fl_min: int = 150, fl_max: int = 300,
                            h0_mean: float = 250.0, h0_sd: float = 100.0,
                            h0_lo: float = 100.0, h0_hi: float = 500.0) -> Tuple[int,int,float]:
    FL_pl_base = int(rng.integers(fl_min, fl_max+1))
    cap = 1000 if FL_pl_base < 290 else 2000
    h0 = float(np.clip(rng.normal(h0_mean, h0_sd), max(h0_lo, 0.0), min(h0_hi, cap)))
    diff_FL = max(1, int(round(h0/100.0)))
    sign = 1 if rng.uniform() < 0.5 else -1
    FL_cat_cand = FL_pl_base + sign * diff_FL
    if not (fl_min <= FL_cat_cand <= fl_max):
        FL_cat_cand = FL_pl_base - sign * diff_FL
        FL_cat_cand = min(fl_max, max(fl_min, FL_cat_cand))
    return int(FL_pl_base), int(FL_cat_cand), float(h0)
def run_batch(runs: int, seed: int, scenario: str,
              r0_min_nm: float, r0_max_nm: float,
              aggressiveness: float,
              p_opp: float, p_ta: float, p_weak: float,
              jitter_priors: bool, apfd_share: float,
              use_delay_mixture: bool, dt: float,
              hdg1_min: float, hdg1_max: float, hdg2_min: float, hdg2_max: float) -> pd.DataFrame:
    rng = np.random.default_rng(int(seed))
    if scenario == "Head-on":
        rel_min, rel_max = 150.0, 210.0
    elif scenario == "Crossing":
        rel_min, rel_max = 60.0, 120.0
    elif scenario == "Overtaking":
        rel_min, rel_max = 0.0, 30.0
    else:
        rel_min = rel_max = None
    data: List[Dict] = []
    for k in range(int(runs)):
        # Altitudes & miss
        FL_PL, FL_CAT, h0 = sample_altitudes_and_h0(rng)
        cat_above = (FL_CAT > FL_PL) if (FL_CAT != FL_PL) else (rng.uniform() < 0.5)
        # Speeds & headings
        PL_IAS = 120.0
        PL_TAS = ias_to_tas(PL_IAS, FL_PL*100.0)
        CAT_TAS = float(rng.uniform(420.0, 470.0))
        if scenario == "Custom":
            h1 = float(rng.uniform(hdg1_min, hdg1_max))
            h2 = float(rng.uniform(hdg2_min, hdg2_max))
        else:
            h1, h2 = sample_headings(rng, scenario, 0.0, 360.0, rel_min, rel_max)
        # Geometry time to go
        r0 = float(rng.uniform(min(r0_min_nm, r0_max_nm), max(r0_min_nm, r0_max_nm)))
        vcl = relative_closure_kt(PL_TAS, h1, CAT_TAS, h2)
        tgo_geom = time_to_go_from_geometry(r0, vcl)
        # RA timing 15–35 s
        if scenario == "Head-on": mu, sd = 25.0, 5.0
        elif scenario == "Crossing": mu, sd = 22.0, 6.0
        else: mu, sd = 30.0, 8.0
        lo, hi = TGO_MIN_S, min(tgo_geom if tgo_geom is not None else TGO_MAX_S, TGO_MAX_S)
        tgo = float(np.clip(rng.normal(mu, sd), lo, max(lo+1.0, hi)))
        # Level‑off switch via aggressiveness
        leveloff_context = (aggressiveness <= 1e-6)
        vz0_pl  = sample_initial_vs_with_aggressiveness(rng, aggressiveness, leveloff_context)
        vz0_cat = sample_initial_vs_with_aggressiveness(rng, aggressiveness, leveloff_context)
        # Optimal sense (nominal CAT)
        (sense_pl, sense_ca), _, _ = choose_optimal_sense(
            tgo, dt, h0, cat_above, vz0_pl, vz0_cat,
            cat_delay_nom=5.0, cat_accel_nom=0.25, cat_vs=CAT_INIT_VS_FPM, cat_cap=CAT_CAP_INIT_FPM)
        # CAT delay mixture + AP/FD TCAS
        if use_delay_mixture:
            if rng.uniform() < 0.7:
                cat_delay_eff = float(max(0.0, rng.normal(4.5, 1.0)))
                cat_accel_eff = float(max(0.05, rng.normal(0.25, 0.03)))
            else:
                cat_delay_eff = float(max(0.0, rng.normal(8.5, 1.5)))
                cat_accel_eff = float(max(0.05, rng.normal(0.10, 0.02)))
        else:
            cat_delay_eff = 5.0
            cat_accel_eff = 0.25
        if rng.uniform() < apfd_share:
            cat_delay_eff = max(0.0, cat_delay_eff - 0.8)
            cat_accel_eff = max(0.05, cat_accel_eff + 0.03)
        # Non‑compliance (consistent signature: delay_s / accel_g)
        mode, sense_cat_exec, cat_delay_exec, cat_accel_exec, cat_vs_exec, cat_cap_exec = apply_non_compliance_to_cat(
            rng, sense_ca, delay_s=cat_delay_eff, accel_g=cat_accel_eff, vs_fpm=CAT_INIT_VS_FPM, cap_fpm=CAT_CAP_INIT_FPM,
            p_opp=p_opp, p_taonly=p_ta, p_weak=p_weak, jitter=jitter_priors)
        # PL delay sample
        pl_delay = max(0.0, rng.normal(PL_DELAY_MEAN_S, PL_DELAY_SD_S))
        # First phase time series
        times, vs_pl = vs_time_series(tgo, dt, pl_delay, PL_ACCEL_G, PL_VS_FPM, sense=sense_pl,
                                      cap_fpm=PL_VS_CAP_FPM, vs0_fpm=vz0_pl)
        _, vs_ca = vs_time_series(tgo, dt, cat_delay_exec, cat_accel_exec, cat_vs_exec, sense=sense_cat_exec,
                                  cap_fpm=cat_cap_exec, vs0_fpm=vz0_cat)
        z_pl = integrate_altitude_from_vs(times, vs_pl, 0.0)
        z_ca = integrate_altitude_from_vs(times, vs_ca, h0 if cat_above else -h0)
        # ALIM at FL_PL
        alim_ft = alim_ft_from_alt(FL_PL * 100.0)
        # Classify event
        eventtype, minsep_ft, sep_cpa_ft, t_check = classify_event(
            times, z_pl, z_ca, vs_pl, vs_ca, tgo,
            alim_ft=alim_ft, margin_ft=ALIM_MARGIN_FT,
            sense_chosen_cat=sense_ca, sense_exec_cat=sense_cat_exec)
        # Second phase (if needed)
        t2_issue = None
        if eventtype in ("STRENGTHEN", "REVERSE"):
            t2_issue = float(times[-1] - max(0.0, tgo - times[-1])) + 1.0
            times2, vs_pl2, vs_ca2 = apply_second_phase(
                times, vs_pl, vs_ca, tgo, dt, eventtype, sense_pl, sense_cat_exec,
                pl_delay=pl_delay, pl_accel_g=PL_ACCEL_G, pl_cap=PL_VS_CAP_FPM,
                cat_delay=1.0, cat_accel_g=0.20,
                cat_vs_strength=CAT_STRENGTH_FPM, cat_cap=CAT_CAP_STRENGTH_FPM,
                t2_issue_override=t2_issue)
            z_pl2 = integrate_altitude_from_vs(times2, vs_pl2, 0.0)
            z_ca2 = integrate_altitude_from_vs(times2, vs_ca2, h0 if cat_above else -h0)
            sep2 = np.abs(z_ca2 - z_pl2)
            minsep_ft = float(np.min(sep2))
            sep_cpa_ft = float(sep2[-1])
            times, vs_pl, vs_ca, z_pl, z_ca = times2, vs_pl2, vs_ca2, z_pl2, z_ca2
        miss_cpa_ft = float(abs(z_ca[-1] - z_pl[-1]))
        comp_label = compliance_score_method_b_like(sense_required=sense_cat_exec, vs_trace=vs_ca, times=times)
        data.append(dict(
            run=k+1, scenario=scenario,
            FL_PL=FL_PL, FL_CAT=FL_CAT,
            PL_TAS=PL_TAS, CAT_TAS=CAT_TAS,
            PLhdg=h1, CAThdg=h2,
            R0NM=r0, closurekt=vcl, tgos=tgo, h0ft=h0,
            aggressiveness=aggressiveness, leveloff=int(leveloff_context),
            sensePL=sense_pl, senseCAT_chosen=sense_ca,
            CAT_mode=mode, senseCAT_exec=sense_cat_exec,
            plDelay=pl_delay, plAccel_g=PL_ACCEL_G,
            catDelay=cat_delay_exec, catAccel_g=cat_accel_exec,
            catVS_cmd=cat_vs_exec, catCap_cmd=cat_cap_exec,
            ALIM_ft=alim_ft,
            missCPAft=miss_cpa_ft, minsepft=minsep_ft, sep_cpa_ft=sep_cpa_ft,
            eventtype=eventtype, t_second_issue=t2_issue,
            comp_label=comp_label
        ))
    return pd.DataFrame(data)
# ----------------------------- Streamlit UI ----------------------------- #
st.set_page_config(page_title="ACAS/TCAS Monte‑Carlo v2", layout="wide")
st.title("ACAS/TCAS — Optimal‑sense + Non‑compliance + 2‑phase RA (v2)")
with st.sidebar:
    st.header("Global settings")
    scenario = st.selectbox("Scenario", ["Head-on", "Crossing", "Overtaking", "Custom"])
    colg1, colg2 = st.columns(2)
    with colg1:
        runs = st.number_input("Number of runs", 1, 200000, 5000, 500)
        seed = st.number_input("Random seed", 0, 2**31-1, 26, 1)
    with colg2:
        dt = st.number_input("Time step (s)", 0.01, 1.0, 0.1, 0.01, format="%.2f")
    r0_min = st.number_input("Initial range min (NM)", 0.5, 50.0, 4.0, 0.5)
    r0_max = st.number_input("Initial range max (NM)", 0.5, 50.0, 8.0, 0.5)
    st.markdown("**Initial trajectory aggressiveness**")
    aggressiveness = st.slider("0 = mostly level (enables Level‑off mode); 1 = aggressive",
                               min_value=0.0, max_value=1.0, value=0.30, step=0.05)
    if scenario == "Custom":
        hdg1_min = st.number_input("PL heading min (deg)", 0.0, 360.0, 0.0, 5.0)
        hdg1_max = st.number_input("PL heading max (deg)", 0.0, 360.0, 360.0, 5.0)
        hdg2_min = st.number_input("CAT heading min (deg)", 0.0, 360.0, 0.0, 5.0)
        hdg2_max = st.number_input("CAT heading max (deg)", 0.0, 360.0, 360.0, 5.0)
    else:
        hdg1_min, hdg1_max, hdg2_min, hdg2_max = 0.0, 360.0, 0.0, 360.0
    st.markdown("**CAT non‑compliance priors**")
    p_opp  = st.number_input("P(opposite‑sense)",               0.0, 1.0, 0.010, 0.001, format="%.3f")
    p_ta   = st.number_input("P(TA‑only / no‑response)",        0.0, 1.0, 0.003, 0.001, format="%.3f")
    p_weak = st.number_input("P(weak‑compliance)",              0.0, 1.0, 0.300, 0.005, format="%.3f")
    jitter_priors     = st.checkbox("Jitter priors ±30%", value=True)
    apfd_share        = st.slider("AP/FD TCAS share (CAT)", 0.0, 1.0, 0.25, 0.05)
    use_delay_mixture = st.checkbox("Use CAT delay mixture (70/30)", value=True)
# --- Single‑run visualiser --- #
st.subheader("Single‑run — optimal sense + Δh@CPA")
spot_c1, spot_c2, spot_c3 = st.columns(3)
with spot_c1:
    spot_h0 = st.number_input("Initial vertical miss h0 (ft)", 0.0, 2000.0, 250.0, 25.0)
with spot_c2:
    tgo_spot = st.number_input("t_go to CPA (s)", 15.0, 60.0, 25.0, 1.0)
with spot_c3:
    spot_dt = st.number_input("Plot dt (s)", 0.01, 1.0, float(dt), 0.01)
if st.button("Run single simulation (optimal sense)"):
    rng = np.random.default_rng(1234)
    PL_IAS = 120.0
    FL_PL, FL_CAT = 250, 260
    PL_TAS = ias_to_tas(PL_IAS, FL_PL*100.0)
    CAT_TAS = float(rng.uniform(420.0, 470.0))
    h1, h2 = sample_headings(rng, "Head-on", 0.0, 360.0, 150.0, 210.0)
    cat_above = True
    vz0_pl  = sample_initial_vs_with_aggressiveness(rng, aggressiveness, aggressiveness <= 1e-6)
    vz0_cat = sample_initial_vs_with_aggressiveness(rng, aggressiveness, aggressiveness <= 1e-6)
    (sense_pl, sense_ca), _, _ = choose_optimal_sense(
        tgo_spot, spot_dt, spot_h0, cat_above, vz0_pl, vz0_cat,
        cat_delay_nom=5.0, cat_accel_nom=0.25, cat_vs=CAT_INIT_VS_FPM, cat_cap=CAT_CAP_INIT_FPM)
    pl_delay = PL_DELAY_MEAN_S
    cat_delay, cat_accel = 5.0, 0.25
    times, vs_pl = vs_time_series(tgo_spot, spot_dt, pl_delay, PL_ACCEL_G, PL_VS_FPM, sense=sense_pl,
                                  cap_fpm=PL_VS_CAP_FPM, vs0_fpm=vz0_pl)
    _, vs_ca = vs_time_series(tgo_spot, spot_dt, cat_delay, cat_accel, CAT_INIT_VS_FPM, sense=sense_ca,
                              cap_fpm=CAT_CAP_INIT_FPM, vs0_fpm=vz0_cat)
    z_pl = integrate_altitude_from_vs(times, vs_pl, 0.0)
    z_ca = integrate_altitude_from_vs(times, vs_ca, spot_h0)
    alim_ft = alim_ft_from_alt(FL_PL*100.0)
    miss_cpa = float(abs(z_ca[-1] - z_pl[-1]))
    fig, ax = plt.subplots(figsize=(7.6, 4.4))
    ax.plot(times, z_pl, label=f"PL — {'climb' if sense_pl>0 else 'descend'}")
    ax.plot(times, z_ca, label=f"CAT — {'climb' if sense_ca>0 else 'descend'}")
    ax.fill_between(times, z_pl - alim_ft, z_pl + alim_ft, alpha=0.10, label=f"±ALIM ({alim_ft:.0f} ft)")
    ax.axhline(0, ls="--", lw=1, alpha=0.6)
    ax.set_xlabel("Time since RA trigger (s)"); ax.set_ylabel("Relative altitude (ft)")
    ax.set_title(f"Single‑run (optimal sense) — Δh@CPA = {miss_cpa:.0f} ft")
    ax.grid(True, alpha=0.3); ax.legend()
    st.pyplot(fig)
    st.caption(f"Chosen sense: PL = {'climb' if sense_pl>0 else 'descend'}, "
               f"CAT = {'climb' if sense_ca>0 else 'descend'}")
st.markdown("---")
# --- Batch Monte‑Carlo --- #
st.header("Batch Monte‑Carlo (optimal sense; CAT variability & non‑compliance; emergent events)")
if st.button("Run batch"):
    with st.spinner("Running Monte‑Carlo…"):
        df = run_batch(
            runs=int(runs), seed=int(seed), scenario=scenario,
            r0_min_nm=float(r0_min), r0_max_nm=float(r0_max),
            aggressiveness=float(aggressiveness),
            p_opp=float(p_opp), p_ta=float(p_ta), p_weak=float(p_weak),
            jitter_priors=bool(jitter_priors), apfd_share=float(apfd_share),
            use_delay_mixture=bool(use_delay_mixture), dt=float(dt),
            hdg1_min=float(hdg1_min), hdg1_max=float(hdg1_max),
            hdg2_min=float(hdg2_min), hdg2_max=float(hdg2_max)
        )
    st.success(f"Completed {len(df)} runs.")
    st.session_state["df"] = df
# --- Results block --- #
if "df" in st.session_state:
    df = st.session_state["df"]
    st.subheader("Preview of results")
    st.dataframe(df.head(100), use_container_width=True)
    n = len(df)
    p_str = (df['eventtype'] == "STRENGTHEN").mean()*100
    p_rev = (df['eventtype'] == "REVERSE").mean()*100
    p_any = (df['minsepft'] < df['ALIM_ft']).mean()*100
    p_cpa = (df['sep_cpa_ft'] < df['ALIM_ft']).mean()*100
    # Compliance breakdown
    p_follow = (df['comp_label'] == "Following").mean()*100
    p_weak   = (df['comp_label'] == "Weak").mean()*100
    p_no     = (df['comp_label'] == "No Response").mean()*100
    p_oppc   = (df['comp_label'] == "Opposite").mean()*100
    p_excess = (df['comp_label'] == "Excessive").mean()*100
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("P(Strengthen)", f"{p_str:,.2f}%")
    m2.metric("P(Reversal)", f"{p_rev:,.2f}%")
    m3.metric("P(ALIM any)", f"{p_any:,.2f}%")
    m4.metric("P(ALIM @ CPA)", f"{p_cpa:,.2f}%")
    m5.metric("Mean Δh@CPA", f"{df['missCPAft'].mean():,.0f} ft")
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Following", f"{p_follow:,.2f}%")
    c2.metric("Weak", f"{p_weak:,.2f}%")
    c3.metric("No", f"{p_no:,.2f}%")
    c4.metric("Opposite", f"{p_oppc:,.2f}%")
    c5.metric("Excessive", f"{p_excess:,.2f}%")
    st.markdown("### Plot a specific run")
    rid = st.number_input("Run id", int(df['run'].min()), int(df['run'].max()), int(df['run'].min()))
    row = df[df['run'] == int(rid)].iloc[0]
    plot_dt = st.number_input("Plot dt (s)", 0.01, 1.0, float(dt), 0.01)
    vz0_pl_assumed = st.number_input("Assumed PL initial VS (fpm)", -4000.0, 4000.0, 0.0, 50.0)
    vz0_cat_assumed = st.number_input("Assumed CAT initial VS (fpm)", -4000.0, 4000.0, 0.0, 50.0)
    tgo = float(row["tgos"]); h0 = float(row["h0ft"])
    FL_pl = int(row["FL_PL"]); FL_cat = int(row["FL_CAT"])
    cat_above = (FL_cat > FL_pl) if (FL_cat != FL_pl) else True
    sense_pl = int(row.get("sensePL", -1 if cat_above else +1))
    sense_cat = int(row.get("senseCAT_exec", +1 if cat_above else -1))
    pl_delay = float(row.get("plDelay", PL_DELAY_MEAN_S)); pl_accel = float(row.get("plAccel_g", PL_ACCEL_G))
    cat_delay = float(row.get("catDelay", 5.0)); cat_accel = float(row.get("catAccel_g", 0.25))
    cat_vs = float(row.get("catVS_cmd", CAT_INIT_VS_FPM)); cat_cap = float(row.get("catCap_cmd", CAT_CAP_INIT_FPM))
    eventtype = str(row["eventtype"]); t2_issue = row.get("t_second_issue", None)
    if pd.isna(t2_issue): t2_issue = None
    times, vs_pl = vs_time_series(tgo, plot_dt, pl_delay, pl_accel, PL_VS_FPM, sense=sense_pl,
                                  cap_fpm=PL_VS_CAP_FPM, vs0_fpm=vz0_pl_assumed)
    _, vs_ca = vs_time_series(tgo, plot_dt, cat_delay, cat_accel, cat_vs, sense=sense_cat,
                              cap_fpm=cat_cap, vs0_fpm=vz0_cat_assumed)
    z_pl = integrate_altitude_from_vs(times, vs_pl, 0.0)
    z_ca = integrate_altitude_from_vs(times, vs_ca, h0 if cat_above else -h0)
    if eventtype in ("STRENGTHEN", "REVERSE"):
        times, vs_pl, vs_ca = apply_second_phase(times, vs_pl, vs_ca, tgo, plot_dt, eventtype,
                                                 sense_pl, sense_cat,
                                                 pl_delay=pl_delay, pl_accel_g=pl_accel, pl_cap=PL_VS_CAP_FPM,
                                                 cat_delay=1.0, cat_accel_g=0.20,
                                                 cat_vs_strength=CAT_STRENGTH_FPM, cat_cap=CAT_CAP_STRENGTH_FPM,
                                                 t2_issue_override=t2_issue)
        z_pl = integrate_altitude_from_vs(times, vs_pl, 0.0)
        z_ca = integrate_altitude_from_vs(times, vs_ca, h0 if cat_above else -h0)
    miss_cpa = float(abs(z_ca[-1] - z_pl[-1]))
    alim_ft = float(row["ALIM_ft"])
    fig2, ax2 = plt.subplots(figsize=(7.6, 4.4))
    ax2.plot(times, z_pl, label=f"PL — {'climb' if sense_pl>0 else 'descend'}")
    ax2.plot(times, z_ca, label=f"CAT — {'climb' if sense_cat>0 else 'descend'} ({row['CAT_mode']})")
    ax2.fill_between(times, z_pl - alim_ft, z_pl + alim_ft, alpha=0.10, label=f"±ALIM ({alim_ft:.0f} ft)")
    ax2.axhline(0, ls="--", lw=1, alpha=0.6)
    ax2.set_xlabel("Time since RA trigger (s)"); ax2.set_ylabel("Relative altitude (ft)")
    ax2.set_title(f"Run {int(row['run'])} — {eventtype} — Δh@CPA = {miss_cpa:.0f} ft")
    ax2.grid(True, alpha=0.3); ax2.legend()
    st.pyplot(fig2)
    st.subheader("Download batch data")
    st.download_button(label="Download CSV", data=csv_buf, file_name="tcas_batch_results.csv", mime="text/csv")
 






































