# app_v3.1.py
# ACAS/TCAS v7.1 — Residual Risk & RA Taxonomy (Batch Monte Carlo, regulator-ready)
# Intent-specific build for: PL fixed response (0.1 s, 0.10 g, 500 fpm; 120 KIAS),
# CAT varies (headings, TAS, delay/accel), altitudes FL150–FL300, plus bounded random
# intruder non-compliance modes. No "calibration" to expected reversal %; outcomes emerge
# from assumptions with documented priors & uncertainty.
#
# Key features:
# - Intruder non-compliance modes: OPPOSITE, LEVELOFF, PERSIST, TA_ONLY (mutually exclusive per run)
# - Priors p_opposite / p_leveloff / p_persist (+ optional jitter); TA_ONLY toggle
# - Trajectory-min "ANY (post-engagement)" unresolved proxy + CPA check
# - Scenario-calibrated RA->CPA option with high-FL nudge; or geometry-derived
# - PL fixed; CAT distributions optional (mixture ON/OFF)
# - Wilson 95% intervals for probability KPIs
# - Sensitivity sweep tool for one parameter (table + line plot)
# - Stateful UI (form + session state) so Explore filters don’t reset batch
#
# Notes:
# - We report unresolved (not induced). ACASA ~2.2% induced (ideal ACAS–ACAS) is context only.
#
# © 2025
import io
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import streamlit as st
# -----------------------------
# Constants & global helpers
# -----------------------------
G = 9.80665
FT_PER_M = 3.28084
MS_PER_FPM = 0.00508
DEFAULT_ALIM_FT = 600.0
DEFAULT_RESP_THRESH_FPM = 300.0
ALIM_MARGIN_FT = 100.0
Z_95 = 1.96
# Performance-limited (PL) — FIXED as per requirement
PL_DELAY_S = 0.1
PL_ACCEL_G = 0.10
PL_VS_FPM  = 500
PL_VS_CAP  = 500
PL_IAS_KT  = 120.0
# -----------------------------
# Kinematics
# -----------------------------
def delta_h_piecewise(t_cpa_s: float, t_delay_s: float, a_g: float, v_f_fpm: float) -> float:
    """Vertical displacement (FEET) from RA to CPA: delay -> ramp -> capped."""
    a = a_g * G
    v_f_mps = v_f_fpm * MS_PER_FPM
    if t_cpa_s <= t_delay_s:
        return 0.0
    t = t_cpa_s - t_delay_s
    t_ramp = v_f_mps / a if a > 0 else np.inf
    if t <= t_ramp:
        dh_m = 0.5 * a * (t**2)
    else:
        dh_m = 0.5 * a * (t_ramp**2) + v_f_mps * (t - t_ramp)
    return dh_m * FT_PER_M
def vs_time_series(t_end_s, dt_s, t_delay_s, a_g, v_f_fpm, sense, cap_fpm=None):
    """Return VS profile (fpm) with delay, ramp, then cap; signed with sense (+climb/-descend)."""
    a = a_g * G
    v_target = v_f_fpm if cap_fpm is None else min(v_f_fpm, cap_fpm)
    times = np.arange(0.0, t_end_s + 1e-9, dt_s)
    vs = np.zeros_like(times, dtype=float)
    for i, t in enumerate(times):
        if t <= t_delay_s:
            vs[i] = 0.0
        else:
            te = t - t_delay_s
            v_mps = min(a * te, v_target * MS_PER_FPM)
            vs[i] = sense * (v_mps / MS_PER_FPM)
    return times, vs
def integrate_altitude_from_vs(times_s: np.ndarray, vs_fpm: np.ndarray) -> np.ndarray:
    """Integrate VS (fpm) to altitude (ft); z(0)=0."""
    dt = np.diff(times_s, prepend=times_s[0])
    fps = vs_fpm / 60.0
    z = np.cumsum(fps * dt)
    z[0] = 0.0
    return z
# -----------------------------
# Atmosphere: IAS -> TAS (ISA approx for FL150–FL300)
# -----------------------------
def ias_to_tas(ias_kt: float, pressure_alt_ft: float) -> float:
    """Approximate TAS ≈ IAS / sqrt(sigma), sigma ≈ (1 - 6.875e-6 * h)^4.256 for troposphere."""
    sigma = (1.0 - 6.875e-6 * pressure_alt_ft)**4.256
    sigma = max(1e-3, sigma)
    return ias_kt / np.sqrt(sigma)
# -----------------------------
# Geometry & RA→CPA handling
# -----------------------------
def relative_closure_kt(v1_kt, hdg1_deg, v2_kt, hdg2_deg) -> float:
    """Scalar closure speed (kt) from headings/speeds."""
    th1, th2 = np.deg2rad(hdg1_deg), np.deg2rad(hdg2_deg)
    v1 = np.array([v1_kt*np.sin(th1), v1_kt*np.cos(th1)])
    v2 = np.array([v2_kt*np.sin(th2), v2_kt*np.cos(th2)])
    return float(np.linalg.norm(v1 - v2))
def time_to_go_from_geometry(r0_nm, v_closure_kt):
    """RA->CPA time from pure geometry (s); None if no closure."""
    if v_closure_kt <= 1e-6:
        return None
    return 3600.0 * (r0_nm / v_closure_kt)
def sample_headings(rng, scenario, hdg1_min, hdg1_max, rel_min=None, rel_max=None,
                    hdg2_min=None, hdg2_max=None):
    h1 = rng.uniform(hdg1_min, hdg1_max)
    if scenario == "Custom":
        h2 = rng.uniform(hdg2_min, hdg2_max)
    else:
        rel = rng.uniform(rel_min, rel_max)
        dirsign = 1 if rng.uniform() < 0.5 else -1
        h2 = (h1 + dirsign * rel) % 360.0
    return h1, h2
# Scenario-calibrated RA->CPA (typical en-route windows, with high-FL nudge)
def sample_tgo_with_trigger(rng, scenario, tgo_geom, FL_pl, FL_cat, cap_s=60.0):
    base = {"Head-on": (25.0, 5.0), "Crossing": (22.0, 6.0), "Overtaking": (30.0, 8.0)}
    mu, sd = base.get(scenario, (25.0, 6.0))
    if FL_pl >= 250 and FL_cat >= 250:
        mu += 2.0   # earlier alerting at higher FL (operationally plausible surrogate)
    lo, hi = 12.0, min(tgo_geom if tgo_geom is not None else cap_s, cap_s)
    if hi <= lo:
        return float(max(8.0, min(tgo_geom or 30.0, cap_s)))
    tgo = float(np.clip(rng.normal(mu, sd), lo, hi))
    return tgo
# -----------------------------
# Baseline Δh for risk scaling (ACASA unresolved baseline 1.1%)
# -----------------------------
def baseline_dh_ft(t_cpa_s, mode="IDEAL"):
    """Δh for the baseline 1500 fpm aircraft at same t_cpa (for risk scaling)."""
    if mode == "IDEAL":
        return delta_h_piecewise(t_cpa_s, t_delay_s=1.0, a_g=0.25, v_f_fpm=1500)
    else:
        return delta_h_piecewise(t_cpa_s, t_delay_s=5.0, a_g=0.25, v_f_fpm=1500)
# -----------------------------
# Pilot response sampling (CAT only)
# -----------------------------
def sample_pilot_response_cat(rng):
    """
    Return (delay_s, accel_g) from a simple mixture:
      - 70% "on-time": delay ~ N(4.5,1.0), accel ~ N(0.25,0.03)
      - 30% "late/weak": delay ~ N(8.5,1.5), accel ~ N(0.10,0.02)
    """
    u = rng.uniform()
    if u < 0.70:
        delay = max(0.0, rng.normal(4.5, 1.0))
        accel = max(0.05, rng.normal(0.25, 0.03))
    else:
        delay = max(0.0, rng.normal(8.5, 1.5))
        accel = max(0.05, rng.normal(0.10, 0.02))
    return float(delay), float(accel)
# -----------------------------
# Altitudes & initial vertical miss
# -----------------------------
def sample_altitudes_and_h0(rng, h0_mean=250.0, h0_sd=100.0, h0_lo=100.0, h0_hi=500.0):
    # Altitudes in FL150–FL300 (pressure altitude ~ FL * 100 ft)
    FL_pl  = int(rng.integers(150, 301))
    FL_cat = int(rng.integers(150, 301))
    # Initial vertical miss at RA: Truncated N(μ, σ) in [h0_lo, h0_hi]
    h0 = float(np.clip(rng.normal(h0_mean, h0_sd), h0_lo, h0_hi))
    return FL_pl, FL_cat, h0
# -----------------------------
# Surveillance/noise toggles (for surrogate prediction only)
# -----------------------------
def apply_surveillance_noise(rng, times, vs_own, vs_int, p_miss=0.0):
    """Hold previous VS with prob p_miss each step (affects surrogate perception only)."""
    vs_own_noisy = vs_own.copy()
    vs_int_noisy = vs_int.copy()
    for i in range(1, len(times)):
        if rng.uniform() < p_miss:
            vs_own_noisy[i] = vs_own_noisy[i-1]
            vs_int_noisy[i] = vs_int_noisy[i-1]
    return vs_own_noisy, vs_int_noisy
# -----------------------------
# Surrogate RA taxonomy (v7.1)
# -----------------------------
def dwell_fn(tgo_s: float) -> float:
    """Dwell before reversal: scales mildly with how late the encounter is."""
    return float(np.clip(0.8 + 0.05 * (tgo_s - 12.0), 0.8, 1.8))
def predicted_dh_linear(vs_own_fpm, vs_int_fpm, t_go_s):
    """Linear projection of vertical miss distance over remaining time."""
    return abs((vs_own_fpm - vs_int_fpm) * (t_go_s / 60.0))
def compliant_in_sense(vs_fpm, sense, thr=DEFAULT_RESP_THRESH_FPM):
    return (vs_fpm * sense) >= thr
def surrogate_decision_with_cause(times, vs_own, vs_int, t_cpa_s, resp_thr=DEFAULT_RESP_THRESH_FPM):
    """
    First event decision (allow one):
      - REVERSE if intruder non-compliant after dwell or ALIM_SHORTFALL_LATE (t_go<6s)
      - STRENGTHEN if ALIM_SHORTFALL_EARLY (t_go>=6s)
    Returns [(event, time, cause)] or [].
    """
    events = []
    min_eval_time = 0.2
    for i, t in enumerate(times):
        t_go = max(0.0, t_cpa_s - t)
        if t_go <= 0.0:
            break
        own_ok = compliant_in_sense(vs_own[i], +1, thr=resp_thr)
        int_ok = compliant_in_sense(vs_int[i], -1, thr=resp_thr)
        # Wrong-direction intruder reversal after dwell (own must be compliant)
        if (t >= min_eval_time) and (not int_ok) and own_ok and (t >= dwell_fn(t_go)):
            events.append(("REVERSE", float(t), "INTRUDER_NONCOMPL_AFTER_DWELL"))
            break
        # ALIM shortfall prediction: strengthen early; reverse late
        if (t >= min_eval_time) and (own_ok or int_ok):
            dh_pred = predicted_dh_linear(vs_own[i], vs_int[i], t_go)
            if dh_pred < (DEFAULT_ALIM_FT - ALIM_MARGIN_FT):
                if t_go < 6.0:
                    events.append(("REVERSE", float(t), "ALIM_SHORTFALL_LATE"))
                else:
                    events.append(("STRENGTHEN", float(t), "ALIM_SHORTFALL_EARLY"))
                break
    return events
# -----------------------------
# Wilson interval (95%) for probabilities
# -----------------------------
def wilson_ci(k, n, z=Z_95):
    if n <= 0:
        return (0.0, 0.0)
    phat = k / n
    denom = 1.0 + (z**2)/n
    center = (phat + (z**2)/(2*n)) / denom
    half = z * np.sqrt((phat*(1-phat) + (z**2)/(4*n))/n) / denom
    return (max(0.0, center - half), min(1.0, center + half))
# -----------------------------
# Streamlit state helpers
# -----------------------------
def init_state():
    if "df" not in st.session_state:
        st.session_state["df"] = None
    if "has_results" not in st.session_state:
        st.session_state["has_results"] = False
    if "sweep" not in st.session_state:
        st.session_state["sweep"] = None
init_state()
# -----------------------------
# UI
# -----------------------------
st.title("ACAS/TCAS v7.1 — Residual Risk & RA Taxonomy (Batch Monte Carlo)")
st.markdown(
    """
We model two aircraft in **Class A (FL150–FL300)**: one **performance‑limited (PL)** with **fixed** response and speed, and one **CAT** with **variable** speed, headings, and pilot response.
We compute: **P(Strengthen)**, **P(Reversal)**, **Mean unresolved RR** (Δh‑ratio × **1.1%**),
and **P(ALIM breach)** at **CPA** and **ANY (post‑engagement)**.  
**Induced‑risk** is *not* modelled; cite ACASA (~2.2% for ideal ACAS–ACAS) as context only.
"""
)
with st.sidebar:
    st.header("Global settings")
    alim_ft = st.number_input("ALIM (ft)", value=600.0, step=50.0)
    baseline = st.selectbox("Baseline for risk scaling",
                            ["IDEAL 1500 fpm (ACASA 2002)", "STANDARD 1500 fpm (EUROCONTROL 2018)"])
    dt = st.number_input("Time step (s)", value=0.1, step=0.1, format="%.1f")
    resp_thr = st.number_input("Meaningful response threshold (fpm)",
                               value=300.0, step=50.0)
# PL (fixed) — display for information
st.subheader("Performance‑limited aircraft (PL) — fixed")
st.write(f"- Delay: **{PL_DELAY_S:.1f} s**, Accel: **{PL_ACCEL_G:.2f} g**, Target VS: **±{PL_VS_FPM} fpm** (cap {PL_VS_CAP})")
st.write(f"- Speed: **{PL_IAS_KT:.0f} KIAS** → TAS computed per‑run from sampled FL")
# CAT params (vary)
st.subheader("CAT (non‑PL) parameters — variable in batch")
c1, c2, c3 = st.columns(3)
with c1:
    cat_vs = st.number_input("CAT target VS (fpm)", value=1500, step=100)
    cat_cap = st.number_input("CAT performance cap (fpm)", value=1500, step=100)
with c2:
    cat_ag_nom = st.number_input("CAT accel nominal (g)", value=0.25, step=0.01, format="%.2f")
    cat_td_nom = st.number_input("CAT delay nominal (s)", value=5.0, step=0.5)
with c3:
    cat_tas_min = st.number_input("CAT TAS min (kt)", value=420.0, step=5.0)
    cat_tas_max = st.number_input("CAT TAS max (kt)", value=470.0, step=5.0)
# RA trigger & noise
with st.expander("RA trigger & Surveillance/noise"):
    ra_trigger_mode = st.selectbox("RA→CPA mode",
                                   ["Scenario-calibrated (recommended)", "Geometry-derived"])
    tgo_cap = st.number_input("Max RA→CPA cap (s)", value=60.0, step=5.0, min_value=15.0)
    st.markdown("**Surveillance/noise (optional)** — affects surrogate perception only")
    p_miss = st.slider("P(missing cycle) per time-step", 0.0, 0.20, 0.00, 0.01)
# Intruder (CAT) non-compliance priors
with st.expander("Intruder (CAT) non-compliance priors"):
    p_opposite  = st.slider("P(opposite-sense) per run", 0.0, 0.10, 0.02, 0.005)
    p_leveloff  = st.slider("P(level-off / follow ATC) per run", 0.0, 0.10, 0.03, 0.005)
    p_persist   = st.slider("P(persistent weak <300 fpm) per run", 0.0, 0.05, 0.01, 0.005)
    ta_only     = st.checkbox("TA-only / unequipped intruder (sensitivity)", value=False)
    jitter      = st.checkbox("Jitter probabilities per run (±50% of value)", value=True)
# Initial vertical miss at RA
with st.expander("Initial vertical miss (at RA)"):
    h0_mean = st.number_input("h0 mean (ft)", value=250.0, step=25.0)
    h0_sd   = st.number_input("h0 std dev (ft)", value=100.0, step=25.0)
    h0_lo   = st.number_input("h0 min (ft)", value=100.0, step=25.0)
    h0_hi   = st.number_input("h0 max (ft)", value=500.0, step=25.0)
# -----------------------------
# Single-run spot check (deterministic preview)
# -----------------------------
st.markdown("### Single‑run spot check (deterministic preview)")
spot_FL_pl  = st.number_input("Spot FL (PL)", value=200, step=10, min_value=150, max_value=300)
spot_FL_cat = st.number_input("Spot FL (CAT)", value=200, step=10, min_value=150, max_value=300)
spot_h0     = st.number_input("Spot initial vertical miss h0 (ft)", value=250.0, step=25.0)
# Head-on 8 NM preview
PL_TAS_spot  = ias_to_tas(PL_IAS_KT, spot_FL_pl * 100.0)
CAT_TAS_spot = (cat_tas_min + cat_tas_max) / 2.0
v_clos_spot  = relative_closure_kt(PL_TAS_spot, 0.0, CAT_TAS_spot, 180.0)
tgo_geom_spot= time_to_go_from_geometry(8.0, v_clos_spot) or 30.0
t_cpa_spot   = float(min(max(20.0, tgo_geom_spot), tgo_cap))
dh_pl_ft = delta_h_piecewise(t_cpa_spot, PL_DELAY_S, PL_ACCEL_G, PL_VS_FPM)
dh_cat_ft= delta_h_piecewise(t_cpa_spot, cat_td_nom, cat_ag_nom, cat_vs)
dh_base  = baseline_dh_ft(t_cpa_spot, mode="IDEAL" if baseline.startswith("IDEAL") else "STANDARD")
ratio    = (dh_base / dh_pl_ft) if dh_pl_ft > 1e-6 else np.nan
unres_rr = 1.1 * ratio
spot_tab = pd.DataFrame({
    "Aircraft": ["PL (ownship)", "CAT (intruder)", "Baseline"],
    "Δh @ CPA (ft)": [dh_pl_ft, dh_cat_ft, dh_base]
})
st.dataframe(spot_tab, use_container_width=True)
st.write(f"Scaled unresolved RR ≈ **{unres_rr:,.3f}%** (ratio {ratio:,.3f}); t_go≈**{t_cpa_spot:.1f}s**")
# -----------------------------
# Batch Monte Carlo — FORM (no unintended reruns)
# -----------------------------
st.markdown("---")
st.header("Batch Monte Carlo")
with st.form("batch_form", clear_on_submit=False):
    n_runs = st.number_input("Number of runs", min_value=1, max_value=100000, value=5000, step=100)
    seed   = st.number_input("Random seed", value=42, step=1)
    scenario = st.selectbox("Scenario", ["Head-on", "Crossing", "Overtaking", "Custom"])
    r_min = st.number_input("Initial range min (NM)", value=5.0, step=0.5, min_value=0.5)
    r_max = st.number_input("Initial range max (NM)", value=12.0, step=0.5, min_value=1.0)
    if scenario == "Custom":
        hdg1_min = st.number_input("PL heading min (deg)", value=0.0, step=5.0)
        hdg1_max = st.number_input("PL heading max (deg)", value=360.0, step=5.0)
        hdg2_min = st.number_input("CAT heading min (deg)", value=0.0, step=5.0)
        hdg2_max = st.number_input("CAT heading max (deg)", value=360.0, step=5.0)
        rel_min = rel_max = None
    else:
        hdg1_min = 0.0; hdg1_max = 360.0
        if scenario == "Head-on":
            rel_min, rel_max = 150.0, 210.0
        elif scenario == "Crossing":
            rel_min, rel_max = 60.0, 120.0
        else:  # Overtaking
            rel_min, rel_max = 0.0, 30.0
    use_distrib = st.checkbox("CAT response: use mixture distributions (recommended)", value=True)
    submitted = st.form_submit_button("Run batch")
# -----------------------------
# Run batch only when submitted
# -----------------------------
if submitted:
    rng = np.random.default_rng(int(seed))
    data = []
    for k in range(int(n_runs)):
        # Altitudes & initial miss
        FL_pl, FL_cat, h0 = sample_altitudes_and_h0(rng, h0_mean, h0_sd, h0_lo, h0_hi)
        # Speeds (PL TAS from IAS by FL; CAT TAS random)
        PL_TAS = ias_to_tas(PL_IAS_KT, FL_pl * 100.0)
        CAT_TAS = float(rng.uniform(min(cat_tas_min, cat_tas_max), max(cat_tas_min, cat_tas_max)))
        # Headings & geometry
        if scenario == "Custom":
            h1 = rng.uniform(hdg1_min, hdg1_max)
            h2 = rng.uniform(hdg2_min, hdg2_max)
        else:
            h1, h2 = sample_headings(rng, scenario, hdg1_min, hdg1_max, rel_min, rel_max)
        r0 = float(rng.uniform(min(r_min, r_max), max(r_min, r_max)))
        vcl = relative_closure_kt(PL_TAS, h1, CAT_TAS, h2)
        tgo_geom = time_to_go_from_geometry(r0, vcl)
        # RA->CPA time
        if ra_trigger_mode.startswith("Scenario"):
            tgo = sample_tgo_with_trigger(rng, scenario, tgo_geom, FL_pl, FL_cat, cap_s=tgo_cap)
        else:
            tgo = float(np.clip(tgo_geom if tgo_geom is not None else 30.0, 8.0, tgo_cap))
        # PL fixed response; CAT from distributions or nominal
        pl_td_k = PL_DELAY_S
        pl_ag_k = PL_ACCEL_G
        if use_distrib:
            cat_td_k, cat_ag_k = sample_pilot_response_cat(rng)
        else:
            cat_td_k, cat_ag_k = cat_td_nom, cat_ag_nom
        # Δh (ft) at CPA & scaled RR
        dh_pl = delta_h_piecewise(tgo, pl_td_k, pl_ag_k, PL_VS_FPM)
        dh_cat= delta_h_piecewise(tgo, cat_td_k, cat_ag_k, cat_vs)
        dh_base = baseline_dh_ft(tgo, mode="IDEAL" if baseline.startswith("IDEAL") else "STANDARD")
        ratio = (dh_base / dh_pl) if dh_pl > 1e-6 else np.nan
        unres_rr = 1.1 * ratio
        # Time series (base, coordinated)
        times, vs_pl = vs_time_series(tgo, dt, pl_td_k, pl_ag_k, PL_VS_FPM, sense=+1, cap_fpm=PL_VS_CAP)
        _,     vs_ca = vs_time_series(tgo, dt, cat_td_k, cat_ag_k, cat_vs,     sense=-1, cap_fpm=cat_cap)
        # Intruder non-compliance mode selection (mutually exclusive)
        mode = "BASE"
        if ta_only:
            mode = "TA_ONLY"
        else:
            p1, p2, p3 = p_opposite, p_opposite + p_leveloff, p_opposite + p_leveloff + p_persist
            if jitter:
                p1 *= float(np.clip(rng.uniform(0.5, 1.5), 0.0, 2.0))
                p2 = p1 + p_leveloff * float(np.clip(rng.uniform(0.5, 1.5), 0.0, 2.0))
                p3 = p2 + p_persist  * float(np.clip(rng.uniform(0.5, 1.5), 0.0, 2.0))
            draw = rng.uniform()
            if draw < p1:
                mode = "OPPOSITE"
            elif draw < p2:
                mode = "LEVELOFF"
            elif draw < p3:
                mode = "PERSIST"
        # Apply mode
        if mode == "TA_ONLY":
            times = np.arange(0.0, tgo + 1e-9, dt)
            vs_ca = np.zeros_like(times) + rng.normal(0.0, 100.0, size=times.size)
        elif mode == "OPPOSITE":
            wrong_win = (times >= cat_td_k) & (times <= cat_td_k + 6.0)
            vs_ca[wrong_win] = -vs_ca[wrong_win]
        elif mode == "LEVELOFF":
            hold_win = (times >= cat_td_k) & (times <= cat_td_k + 8.0)
            vs_ca[hold_win] = 0.0
        elif mode == "PERSIST":
            vs_ca[:] = np.clip(vs_ca, -250.0, +250.0)
        # Optional noise for surrogate perception only
        vs_pl_noisy, vs_ca_noisy = apply_surveillance_noise(rng, times, vs_pl, vs_ca, p_miss=p_miss)
        # Integrate to get altitude histories; compute CPA and ANY (post-engagement) metrics
        z_pl = integrate_altitude_from_vs(times, vs_pl)
        z_ca = integrate_altitude_from_vs(times, vs_ca)
        sep_series = np.abs(h0 + (z_pl - z_ca))
        miss_cpa = float(np.abs(h0 + (z_pl[-1] - z_ca[-1])))
        breach_cpa  = (miss_cpa < alim_ft)
        # Post-engagement minimum: start checking after first response begins
        engage_t = min(pl_td_k, cat_td_k) + 0.5
        mask = times >= engage_t
        min_sep_post = float(np.min(sep_series[mask])) if mask.any() else float(np.min(sep_series))
        breach_post = (min_sep_post < alim_ft)   # main ANY metric to report
        # Surrogate event (first hit)
        ev = surrogate_decision_with_cause(times, vs_pl_noisy, vs_ca_noisy, t_cpa_s=tgo, resp_thr=resp_thr)
        evtype = ev[0][0] if ev else "NONE"
        evtime = ev[0][1] if ev else np.nan
        evcause= ev[0][2] if ev else "N/A"
        data.append({
            "run": k + 1,
            "scenario": scenario,
            # Altitudes & speeds
            "FL_PL": FL_pl, "FL_CAT": FL_cat, "PL_TAS": PL_TAS, "CAT_TAS": CAT_TAS,
            # Geometry
            "PLhdg": h1, "CAThdg": h2, "R0NM": r0, "closurekt": vcl, "tgos": tgo,
            # Responses
            "plDelay": pl_td_k, "plAccel_g": pl_ag_k, "catDelay": cat_td_k, "catAccel_g": cat_ag_k,
            # Intruder mode
            "intruder_mode": mode,
            # Misses
            "h0ft": h0, "missCPAft": miss_cpa, "minSepPostEng_ft": min_sep_post,
            "ALIMbreach_CPA": breach_cpa, "ALIMbreach_ANY_postEng": breach_post,
            # Δh & RR
            "dhPLft": dh_pl, "dhCATft": dh_cat, "dhbaselineft": dh_base,
            "ratiobaseoverPL": ratio, "unresolvedRRpct": unres_rr,
            # Event
            "eventtype": evtype, "eventtimes": evtime, "eventcause": evcause,
        })
    df = pd.DataFrame(data)
    st.session_state["df"] = df
    st.session_state["has_results"] = True
    st.session_state["sweep"] = None
    st.success(f"Completed {len(df)} runs.")
# -----------------------------
# Results / Explore (no rerun) — reads from state
# -----------------------------
if st.session_state["has_results"] and st.session_state["df"] is not None:
    df = st.session_state["df"]
    n = len(df)
    # KPIs (with Wilson 95% CIs)
    k_rev = int((df['eventtype']=="REVERSE").sum())
    k_str = int((df['eventtype']=="STRENGTHEN").sum())
    k_cpa = int(df['ALIMbreach_CPA'].sum())
    k_any = int(df['ALIMbreach_ANY_postEng'].sum())
    p_rev = k_rev/n; lo_rev, hi_rev = wilson_ci(k_rev, n)
    p_str = k_str/n; lo_str, hi_str = wilson_ci(k_str, n)
    p_cpa = k_cpa/n; lo_cpa, hi_cpa = wilson_ci(k_cpa, n)
    p_any = k_any/n; lo_any, hi_any = wilson_ci(k_any, n)
    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("P(Reversal)",   f"{100*p_rev:,.2f}%  [{100*lo_rev:,.2f}–{100*hi_rev:,.2f}%]")
    k2.metric("P(Strengthen)", f"{100*p_str:,.2f}%  [{100*lo_str:,.2f}–{100*hi_str:,.2f}%]")
    k3.metric("Mean RR",        f"{df['unresolvedRRpct'].mean():.3f}%")
    k4.metric("P(ALIM @CPA)",  f"{100*p_cpa:,.2f}%  [{100*lo_cpa:,.2f}–{100*hi_cpa:,.2f}%]")
    k5.metric("P(ALIM ANY post-eng)", f"{100*p_any:,.2f}%  [{100*lo_any:,.2f}–{100*hi_any:,.2f}%]")
    # Explore filters (do not rerun batch)
    st.sidebar.subheader("Explore batch")
    tgo_lo, tgo_hi = st.sidebar.slider("tgo window (s)",
                                       float(max(8.0, df["tgos"].min())),
                                       float(df["tgos"].max()),
                                       (float(df["tgos"].min()), float(df["tgos"].max())))
    only_rev  = st.sidebar.checkbox("Only reversals", value=False)
    only_bANY = st.sidebar.checkbox("Only ALIM-breach ANY (post-eng)", value=False)
    view = df[df["tgos"].between(tgo_lo, tgo_hi)]
    if only_rev:
        view = view[view["eventtype"] == "REVERSE"]
    if only_bANY:
        view = view[view["ALIMbreach_ANY_postEng"] == True]
    st.subheader("Preview of results")
    st.dataframe(view.head(200), use_container_width=True)
    # ECDF of unresolved RR
    vals = view["unresolvedRRpct"].dropna().values
    if len(vals):
        x = np.sort(vals); y = np.arange(1, len(x)+1)/len(x)
        fig_ecdf, ax_ecdf = plt.subplots(figsize=(6,3))
        ax_ecdf.plot(x, y)
        ax_ecdf.set_xlabel("Unresolved RR (%)"); ax_ecdf.set_ylabel("ECDF")
        ax_ecdf.grid(True, alpha=0.3)
        st.pyplot(fig_ecdf)
    # tgo hist by event type
    fig_hist, ax_hist = plt.subplots(figsize=(6,3))
    for lab in ["STRENGTHEN", "REVERSE", "NONE"]:
        sub = view[view["eventtype"] == lab]["tgos"]
        if len(sub):
            ax_hist.hist(sub, bins=24, histtype="step", label=lab)
    ax_hist.set_xlabel("tgo (s)"); ax_hist.set_ylabel("Count"); ax_hist.grid(True, alpha=0.3)
    ax_hist.legend()
    st.pyplot(fig_hist)
    # Event cause bar
    cause_counts = view["eventcause"].value_counts()
    if len(cause_counts):
        fig_bar, ax_bar = plt.subplots(figsize=(6,3))
        ax_bar.bar(cause_counts.index, cause_counts.values)
        ax_bar.set_ylabel("Count"); ax_bar.set_title("Event causes")
        for tick in ax_bar.get_xticklabels():
            tick.set_rotation(20)
        st.pyplot(fig_bar)
    # Intruder mode bar
    if "intruder_mode" in view.columns:
        mode_counts = view["intruder_mode"].value_counts()
        if len(mode_counts):
            fig_mode, ax_mode = plt.subplots(figsize=(6,3))
            ax_mode.bar(mode_counts.index, mode_counts.values, color="#7f7f7f")
            ax_mode.set_ylabel("Count"); ax_mode.set_title("Intruder modes in this view")
            for tick in ax_mode.get_xticklabels():
                tick.set_rotation(20)
            st.pyplot(fig_mode)
    # Min separation histogram (ANY post-engagement)
    fig_miss, ax_miss = plt.subplots(figsize=(6,3))
    ax_miss.hist(view["minSepPostEng_ft"], bins=30)
    ax_miss.axvline(alim_ft, color="k", ls="--", alpha=0.7, label=f"ALIM={alim_ft:.0f} ft")
    ax_miss.set_xlabel("Minimum vertical miss (ft) after engagement")
    ax_miss.set_ylabel("Count")
    ax_miss.grid(True, alpha=0.3)
    ax_miss.legend()
    st.pyplot(fig_miss)
    # Downloads
    st.subheader("Download batch data")
    csv_buf = io.BytesIO()
    csv_buf.write(df.to_csv(index=False).encode("utf-8"))
    csv_buf.seek(0)
    st.download_button(label="Download CSV", data=csv_buf,
                       file_name="tcas_batch_results.csv",
                       mime="text/csv", key="dl_csv")
    # -----------------------------
    # Sensitivity sweep tool (one parameter)
    # -----------------------------
    st.markdown("---")
    st.header("Sensitivity sweep (one parameter)")
    with st.expander("Run a quick sweep"):
        sweep_param = st.selectbox("Parameter to sweep",
                                   ["p_opposite", "p_leveloff", "p_persist"])
        sweep_lo = st.number_input("Sweep start", value=0.00, step=0.005)
        sweep_hi = st.number_input("Sweep end",   value=0.05, step=0.005)
        sweep_steps = st.number_input("Steps", min_value=2, max_value=50, value=6, step=1)
        runs_per_step= st.number_input("Runs per step", min_value=200, max_value=20000, value=2000, step=200)
        seed0 = st.number_input("Sweep base seed", value=100, step=1)
        go_sweep = st.button("Run sweep")
    def run_one_step(param_value, base_seed):
        """Minimal re-run for a single sweep step, reusing current UI settings."""
        rng = np.random.default_rng(int(base_seed))
        data_local = []
        # cache priors and override the one being swept
        priors = dict(p_opposite=p_opposite, p_leveloff=p_leveloff, p_persist=p_persist)
        priors[sweep_param] = float(param_value)
        for kk in range(int(runs_per_step)):
            # Altitudes & h0
            FL_pl, FL_cat, h0loc = sample_altitudes_and_h0(rng, h0_mean, h0_sd, h0_lo, h0_hi)
            PL_TAS = ias_to_tas(PL_IAS_KT, FL_pl * 100.0)
            CAT_TAS = float(rng.uniform(min(cat_tas_min, cat_tas_max), max(cat_tas_min, cat_tas_max)))
            # Headings & geometry
            if scenario == "Custom":
                h1l = rng.uniform(hdg1_min, hdg1_max); h2l = rng.uniform(hdg2_min, hdg2_max)
            else:
                h1l, h2l = sample_headings(rng, scenario, 0.0, 360.0, rel_min, rel_max)
            r0l = float(rng.uniform(min(r_min, r_max), max(r_min, r_max)))
            vcll = relative_closure_kt(PL_TAS, h1l, CAT_TAS, h2l)
            tgeom = time_to_go_from_geometry(r0l, vcll)
            # tgo
            if ra_trigger_mode.startswith("Scenario"):
                tgol = sample_tgo_with_trigger(rng, scenario, tgeom, FL_pl, FL_cat, cap_s=tgo_cap)
            else:
                tgol = float(np.clip(tgeom if tgeom is not None else 30.0, 8.0, tgo_cap))
            # Responses
            pl_td_l, pl_ag_l = PL_DELAY_S, PL_ACCEL_G
            if use_distrib:
                cat_td_l, cat_ag_l = sample_pilot_response_cat(rng)
            else:
                cat_td_l, cat_ag_l = cat_td_nom, cat_ag_nom
            # Time series
            timesl, vs_pl_l = vs_time_series(tgol, dt, pl_td_l, pl_ag_l, PL_VS_FPM, sense=+1, cap_fpm=PL_VS_CAP)
            _,     vs_ca_l = vs_time_series(tgol, dt, cat_td_l, cat_ag_l, cat_vs, sense=-1, cap_fpm=cat_cap)
            # Mode selection (mutually exclusive)
            model = "BASE"
            if ta_only:
                model = "TA_ONLY"
            else:
                p1, p2, p3 = priors["p_opposite"], priors["p_opposite"] + priors["p_leveloff"], priors["p_opposite"] + priors["p_leveloff"] + priors["p_persist"]
                if jitter:
                    p1 *= float(np.clip(rng.uniform(0.5, 1.5), 0.0, 2.0))
                    p2 = p1 + priors["p_leveloff"] * float(np.clip(rng.uniform(0.5, 1.5), 0.0, 2.0))
                    p3 = p2 + priors["p_persist"]  * float(np.clip(rng.uniform(0.5, 1.5), 0.0, 2.0))
                u = rng.uniform()
                if u < p1:
                    model = "OPPOSITE"
                elif u < p2:
                    model = "LEVELOFF"
                elif u < p3:
                    model = "PERSIST"
            # Apply non-compliance
            if model == "TA_ONLY":
                timesl = np.arange(0.0, tgol + 1e-9, dt)
                vs_ca_l = np.zeros_like(timesl) + rng.normal(0.0, 100.0, size=timesl.size)
            elif model == "OPPOSITE":
                wrong_win = (timesl >= cat_td_l) & (timesl <= cat_td_l + 6.0)
                vs_ca_l[wrong_win] = -vs_ca_l[wrong_win]
            elif model == "LEVELOFF":
                hold_win = (timesl >= cat_td_l) & (timesl <= cat_td_l + 8.0)
                vs_ca_l[hold_win] = 0.0
            elif model == "PERSIST":
                vs_ca_l[:] = np.clip(vs_ca_l, -250.0, +250.0)
            # Surrogate perception noise
            vs_pl_noisy, vs_ca_noisy = apply_surveillance_noise(rng, timesl, vs_pl_l, vs_ca_l, p_miss=p_miss)
            # Integrate & metrics
            z_pl_l = integrate_altitude_from_vs(timesl, vs_pl_l)
            z_ca_l = integrate_altitude_from_vs(timesl, vs_ca_l)
            sep_series_l = np.abs(h0loc + (z_pl_l - z_ca_l))
            miss_cpa_l = float(np.abs(h0loc + (z_pl_l[-1] - z_ca_l[-1])))
            breach_cpa_l = (miss_cpa_l < alim_ft)
            engage_t_l = min(pl_td_l, cat_td_l) + 0.5
            msk = timesl >= engage_t_l
            min_sep_post_l = float(np.min(sep_series_l[msk])) if msk.any() else float(np.min(sep_series_l))
            breach_post_l = (min_sep_post_l < alim_ft)
            # Events
            evl = surrogate_decision_with_cause(timesl, vs_pl_noisy, vs_ca_noisy, t_cpa_s=tgol, resp_thr=resp_thr)
            evt = evl[0][0] if evl else "NONE"
            data_local.append((evt, breach_cpa_l, breach_post_l))
        # Aggregate step KPIs
        d = pd.DataFrame(data_local, columns=["evt","bCPA","bANY"])
        nloc = len(d)
        krevl = int((d["evt"]=="REVERSE").sum())
        kstrl = int((d["evt"]=="STRENGTHEN").sum())
        kcpl  = int(d["bCPA"].sum());   kanl = int(d["bANY"].sum())
        return dict(
            param=float(param_value),
            P_rev=krevl/nloc, P_str=kstrl/nloc, P_cpa=kcpl/nloc, P_any=kanl/nloc
        )
    if go_sweep:
        params = np.linspace(sweep_lo, sweep_hi, int(sweep_steps))
        out = []
        for i, pv in enumerate(params):
            out.append(run_one_step(pv, seed0 + i))
        sweep_df = pd.DataFrame(out)
        st.session_state["sweep"] = sweep_df
        st.success("Sweep completed.")
    if st.session_state["sweep"] is not None:
        st.subheader("Sweep results")
        st.dataframe(st.session_state["sweep"], use_container_width=True)
        # Plot chosen KPI vs param
        fig_sw, ax_sw = plt.subplots(figsize=(6,3))
        ax_sw.plot(st.session_state["sweep"]["param"], 100*st.session_state["sweep"]["P_rev"], label="P(Reversal) %")
        ax_sw.plot(st.session_state["sweep"]["param"], 100*st.session_state["sweep"]["P_str"], label="P(Strengthen) %")
        ax_sw.plot(st.session_state["sweep"]["param"], 100*st.session_state["sweep"]["P_any"], label="P(ALIM ANY post-eng) %")
        ax_sw.plot(st.session_state["sweep"]["param"], 100*st.session_state["sweep"]["P_cpa"], label="P(ALIM @CPA) %")
        ax_sw.set_xlabel(sweep_param); ax_sw.set_ylabel("Probability (%)")
        ax_sw.grid(True, alpha=0.3); ax_sw.legend()
        st.pyplot(fig_sw)
else:
    # This block must exist so the file does not end with a dangling 'else:'
    st.info("Run a batch to see results. Use the **form** above; results will persist while you explore.")
# --- EOF app_v3.1.py ---
