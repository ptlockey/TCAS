# app_v3.2.py
# ACAS/TCAS v7.1 — Residual Risk & RA Taxonomy (Batch Monte Carlo, regulator-ready)
# PL fixed (0.1 s, 0.10 g, 500 fpm; 120 KIAS -> TAS by FL), CAT varies.
# Surrogate reversal only after intruder delay + grace; TA-only disables non-compliance reversal.
# "ANY" metric = minimum predicted CPA miss (post-engagement), not instantaneous separation.
# Stateful UI; no trailing "else:" at EOF.
import io
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import streamlit as st
# -----------------------------
# Constants & helpers
# -----------------------------
G = 9.80665
FT_PER_M = 3.28084
MS_PER_FPM = 0.00508
DEFAULT_ALIM_FT = 600.0
DEFAULT_RESP_THRESH_FPM = 300.0
ALIM_MARGIN_FT = 100.0
Z_95 = 1.96
# Performance-limited (PL) — FIXED
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
    """TAS ≈ IAS / sqrt(sigma), sigma ≈ (1 - 6.875e-6 * h)^4.256; guard against negatives."""
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
def sample_tgo_with_trigger(rng, scenario, tgo_geom, FL_pl, FL_cat, cap_s=60.0):
    """Scenario-calibrated RA->CPA (typical en-route windows), +2s if both >= FL250."""
    base = {"Head-on": (25.0, 5.0), "Crossing": (22.0, 6.0), "Overtaking": (30.0, 8.0)}
    mu, sd = base.get(scenario, (25.0, 6.0))
    if FL_pl >= 250 and FL_cat >= 250:
        mu += 2.0
    lo, hi = 12.0, min(tgo_geom if tgo_geom is not None else cap_s, cap_s)
    if hi <= lo:
        return float(max(8.0, min(tgo_geom or 30.0, cap_s)))
    return float(np.clip(rng.normal(mu, sd), lo, hi))
# -----------------------------
# Baseline Δh for risk scaling (ACASA unresolved 1.1%)
# -----------------------------
def baseline_dh_ft(t_cpa_s, mode="IDEAL"):
    if mode == "IDEAL":
        return delta_h_piecewise(t_cpa_s, t_delay_s=1.0, a_g=0.25, v_f_fpm=1500)
    else:
        return delta_h_piecewise(t_cpa_s, t_delay_s=5.0, a_g=0.25, v_f_fpm=1500)
# -----------------------------
# Pilot response sampling (CAT only)
# -----------------------------
def sample_pilot_response_cat(rng):
    """Mixture: 70% on-time (4.5±1.0 s, 0.25±0.03 g), 30% late/weak (8.5±1.5 s, 0.10±0.02 g)."""
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
    FL_pl  = int(rng.integers(150, 301))
    FL_cat = int(rng.integers(150, 301))
    h0 = float(np.clip(rng.normal(h0_mean, h0_sd), h0_lo, h0_hi))
    return FL_pl, FL_cat, h0
# -----------------------------
# Surveillance/noise (for surrogate perception only)
# -----------------------------
def apply_surveillance_noise(rng, times, vs_own, vs_int, p_miss=0.0):
    vs_own_noisy = vs_own.copy()
    vs_int_noisy = vs_int.copy()
    for i in range(1, len(times)):
        if rng.uniform() < p_miss:
            vs_own_noisy[i] = vs_own_noisy[i-1]
            vs_int_noisy[i] = vs_int_noisy[i-1]
    return vs_own_noisy, vs_int_noisy
# -----------------------------
# Surrogate RA taxonomy (v7.1) — patched gating & TA-only handling
# -----------------------------
def dwell_fn(tgo_s: float) -> float:
    return float(np.clip(0.8 + 0.05 * (tgo_s - 12.0), 0.8, 1.8))
def predicted_dh_linear(vs_own_fpm, vs_int_fpm, t_go_s):
    """Predicted CPA miss distance if current VS held."""
    return abs((vs_own_fpm - vs_int_fpm) * (t_go_s / 60.0))
def compliant_in_sense(vs_fpm, sense, thr=DEFAULT_RESP_THRESH_FPM):
    return (vs_fpm * sense) >= thr
def surrogate_decision_with_cause(
    times, vs_own, vs_int, t_cpa_s,
    resp_thr=DEFAULT_RESP_THRESH_FPM,
    intruder_delay_s=5.0,
    noncomp_grace_s=2.0,
    ta_only=False
):
    """
    First event decision (allow one):
      - REVERSE if intruder non-compliant after (delay + grace) and dwell (own compliant)  [disabled if TA-only]
      - STRENGTHEN if predicted ALIM shortfall early
      - REVERSE if predicted ALIM shortfall very late (t_go < 6 s)
    """
    events = []
    min_eval_time = 0.2
    for i, t in enumerate(times):
        t_go = max(0.0, t_cpa_s - t)
        if t_go <= 0.0:
            break
        own_ok = compliant_in_sense(vs_own[i], +1, thr=resp_thr)
        if not ta_only:
            int_ok = compliant_in_sense(vs_int[i], -1, thr=resp_thr)
            if (
                (t >= min_eval_time)
                and (t >= intruder_delay_s + noncomp_grace_s)
                and (not int_ok) and own_ok
                and (t >= dwell_fn(t_go))
            ):
                events.append(("REVERSE", float(t), "INTRUDER_NONCOMPL_AFTER_DWELL"))
                break
        if t >= min_eval_time and own_ok:
            dh_pred = predicted_dh_linear(vs_own[i], vs_int[i], t_go)
            if dh_pred < (DEFAULT_ALIM_FT - ALIM_MARGIN_FT):
                if t_go < 6.0:
                    events.append(("REVERSE", float(t), "ALIM_SHORTFALL_LATE"))
                else:
                    events.append(("STRENGTHEN", float(t), "ALIM_SHORTFALL_EARLY"))
                break
    return events
# -----------------------------
# Wilson 95% CI for probabilities
# -----------------------------
def wilson_ci(k, n, z=Z_95):
    if n <= 0: return (0.0, 0.0)
    phat = k / n
    denom = 1.0 + (z*z)/n
    center = (phat + (z*z)/(2*n)) / denom
    half = z * np.sqrt((phat*(1-phat) + (z*z)/(4*n))/n) / denom
    return (max(0.0, center - half), min(1.0, center + half))
# -----------------------------
# Streamlit state
# -----------------------------
def init_state():
    if "df" not in st.session_state:
        st.session_state["df"] = None
    if "has_results" not in st.session_state:
        st.session_state["has_results"] = False
init_state()
# -----------------------------
# UI
# -----------------------------
st.title("ACAS/TCAS v7.1 — Residual Risk & RA Taxonomy (Batch Monte Carlo)")
st.markdown(
    """
Two aircraft in **Class A (FL150–FL300)**: one **performance‑limited (PL)** is **fixed** (0.1 s / 0.10 g / 500 fpm; **120 KIAS** → TAS by FL); the **CAT** varies (speed, headings, delay, accel).
We compute: **P(Strengthen)**, **P(Reversal)**, **Mean unresolved RR** (Δh‑ratio × **1.1%**),
and **P(ALIM breach)** at **CPA** and **ANY (predicted‑CPA post‑engagement)**.
"""
)
with st.sidebar:
    st.header("Global settings")
    alim_ft = st.number_input("ALIM (ft)", value=600.0, step=50.0)
    baseline = st.selectbox("Baseline for RR scaling",
                            ["IDEAL 1500 fpm (ACASA 2002)", "STANDARD 1500 fpm (EUROCONTROL 2018)"])
    dt = st.number_input("Time step (s)", value=0.1, step=0.1, format="%.1f")
    resp_thr = st.number_input("Meaningful response threshold (fpm)",
                               value=300.0, step=50.0)
st.subheader("Performance‑limited (PL) — fixed")
st.write(f"- Delay: **{PL_DELAY_S:.1f} s**, Accel: **{PL_ACCEL_G:.2f} g**, Target VS: **±{PL_VS_FPM} fpm** (cap {PL_VS_CAP})")
st.write(f"- Speed: **{PL_IAS_KT:.0f} KIAS** → TAS computed per‑run from sampled FL")
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
with st.expander("RA trigger & Surveillance/noise"):
    ra_trigger_mode = st.selectbox("RA→CPA mode",
                                   ["Scenario-calibrated (recommended)", "Geometry-derived"])
    tgo_cap = st.number_input("Max RA→CPA cap (s)", value=60.0, step=5.0, min_value=15.0)
    p_miss = st.slider("P(missing cycle) per time-step (surrogate only)", 0.0, 0.20, 0.00, 0.01)
with st.expander("Intruder (CAT) non-compliance priors"):
    p_opposite  = st.slider("P(opposite-sense) per run", 0.0, 0.10, 0.02, 0.005)
    p_leveloff  = st.slider("P(level-off / follow ATC) per run", 0.0, 0.10, 0.03, 0.005)
    p_persist   = st.slider("P(persistent weak <300 fpm) per run", 0.0, 0.05, 0.01, 0.005)
    ta_only     = st.checkbox("TA-only / unequipped intruder (sensitivity)", value=False)
    jitter      = st.checkbox("Jitter priors per run (±50%)", value=True)
with st.expander("Initial vertical miss (at RA)"):
    h0_mean = st.number_input("h0 mean (ft)", value=250.0, step=25.0)
    h0_sd   = st.number_input("h0 std dev (ft)", value=100.0, step=25.0)
    h0_lo   = st.number_input("h0 min (ft)", value=100.0, step=25.0)
    h0_hi   = st.number_input("h0 max (ft)", value=500.0, step=25.0)
# -----------------------------
# Single-run spot check
# -----------------------------
st.markdown("### Single‑run spot check")
spot_FL_pl  = st.number_input("Spot FL (PL)",  value=200, step=10, min_value=150, max_value=300)
spot_FL_cat = st.number_input("Spot FL (CAT)", value=200, step=10, min_value=150, max_value=300)
spot_h0     = st.number_input("Spot initial vertical miss h0 (ft)", value=250.0, step=25.0)
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
# Batch Monte Carlo — FORM
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
        else:
            rel_min, rel_max = 0.0, 30.0
    use_distrib = st.checkbox("CAT response: use mixture distributions (recommended)", value=True)
    submitted = st.form_submit_button("Run batch")
# -----------------------------
# Run batch
# -----------------------------
if submitted:
    rng = np.random.default_rng(int(seed))
    data = []
    for k in range(int(n_runs)):
        FL_pl, FL_cat, h0 = sample_altitudes_and_h0(rng, h0_mean, h0_sd, h0_lo, h0_hi)
        PL_TAS = ias_to_tas(PL_IAS_KT, FL_pl * 100.0)
        CAT_TAS = float(rng.uniform(min(cat_tas_min, cat_tas_max), max(cat_tas_min, cat_tas_max)))
        if scenario == "Custom":
            h1 = rng.uniform(hdg1_min, hdg1_max)
            h2 = rng.uniform(hdg2_min, hdg2_max)
        else:
            h1, h2 = sample_headings(rng, scenario, hdg1_min, hdg1_max, rel_min, rel_max)
        r0 = float(rng.uniform(min(r_min, r_max), max(r_min, r_max)))
        vcl = relative_closure_kt(PL_TAS, h1, CAT_TAS, h2)
        tgo_geom = time_to_go_from_geometry(r0, vcl)
        if ra_trigger_mode.startswith("Scenario"):
            tgo = sample_tgo_with_trigger(rng, scenario, tgo_geom, FL_pl, FL_cat, cap_s=tgo_cap)
        else:
            tgo = float(np.clip(tgo_geom if tgo_geom is not None else 30.0, 8.0, tgo_cap))
        pl_td_k = PL_DELAY_S
        pl_ag_k = PL_ACCEL_G
        if use_distrib:
            cat_td_k, cat_ag_k = sample_pilot_response_cat(rng)
        else:
            cat_td_k, cat_ag_k = cat_td_nom, cat_ag_nom
        dh_pl = delta_h_piecewise(tgo, pl_td_k, pl_ag_k, PL_VS_FPM)
        dh_cat= delta_h_piecewise(tgo, cat_td_k, cat_ag_k, cat_vs)
        dh_base = baseline_dh_ft(tgo, mode="IDEAL" if baseline.startswith("IDEAL") else "STANDARD")
        ratio = (dh_base / dh_pl) if dh_pl > 1e-6 else np.nan
        unres_rr = 1.1 * ratio
        times, vs_pl = vs_time_series(tgo, dt, pl_td_k, pl_ag_k, PL_VS_FPM, sense=+1, cap_fpm=PL_VS_CAP)
        _,     vs_ca = vs_time_series(tgo, dt, cat_td_k, cat_ag_k, cat_vs, sense=-1, cap_fpm=cat_cap)
        # Intruder mode selection (mutually exclusive; TA_ONLY precedence)
        mode = "BASE"
        if ta_only:
            mode = "TA_ONLY"
        else:
            p1, p2, p3 = p_opposite, p_opposite + p_leveloff, p_opposite + p_leveloff + p_persist
            if jitter:
                # jitter ±50%
                p1 *= float(np.clip(rng.uniform(0.5, 1.5), 0.0, 2.0))
                p2 = p1 + p_leveloff * float(np.clip(rng.uniform(0.5, 1.5), 0.0, 2.0))
                p3 = p2 + p_persist  * float(np.clip(rng.uniform(0.5, 1.5), 0.0, 2.0))
            u = rng.uniform()
            if u < p1:
                mode = "OPPOSITE"
            elif u < p2:
                mode = "LEVELOFF"
            elif u < p3:
                mode = "PERSIST"
        # Apply non-compliance modes
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
        vs_pl_noisy, vs_ca_noisy = apply_surveillance_noise(rng, times, vs_pl, vs_ca, p_miss=p_miss)
        # Predicted CPA ANY (post-engagement)
        z_pl = integrate_altitude_from_vs(times, vs_pl)
        z_ca = integrate_altitude_from_vs(times, vs_ca)
        sep_now = h0 + (z_pl - z_ca)                           # ft
        tgo_series = (tgo - times).clip(min=0.0)
        pred_miss_series = np.abs(sep_now + ( (vs_pl - vs_ca) * (tgo_series/60.0) ))
        engage_t = min(pl_td_k, cat_td_k) + 0.5
        mask = (times >= engage_t)
        min_pred_miss = float(np.min(pred_miss_series[mask])) if mask.any() else float(np.min(pred_miss_series))
        miss_cpa   = float(np.abs(h0 + (z_pl[-1] - z_ca[-1])))
        breach_cpa = (miss_cpa < alim_ft)
        breach_any = (min_pred_miss < alim_ft)  # MAIN "ANY" metric (predicted-CPA, post-engagement)
        # Surrogate event
        ev = surrogate_decision_with_cause(
            times, vs_pl_noisy, vs_ca_noisy, t_cpa_s=tgo, resp_thr=resp_thr,
            intruder_delay_s=cat_td_k, noncomp_grace_s=2.0, ta_only=(mode=="TA_ONLY")
        )
        evtype = ev[0][0] if ev else "NONE"
        evtime = ev[0][1] if ev else np.nan
        evcause= ev[0][2] if ev else "N/A"
        data.append({
            "run": k + 1,
            "scenario": scenario,
            "FL_PL": FL_pl, "FL_CAT": FL_cat, "PL_TAS": PL_TAS, "CAT_TAS": CAT_TAS,
            "PLhdg": h1, "CAThdg": h2, "R0NM": r0, "closurekt": vcl, "tgos": tgo,
            "plDelay": pl_td_k, "plAccel_g": pl_ag_k, "catDelay": cat_td_k, "catAccel_g": cat_ag_k,
            "intruder_mode": mode,
            "h0ft": h0, "missCPAft": miss_cpa, "minPredMiss_postEng_ft": min_pred_miss,
            "ALIMbreach_CPA": breach_cpa, "ALIMbreach_ANY_predCPA": breach_any,
            "dhPLft": dh_pl, "dhCATft": dh_cat, "dhbaselineft": dh_base,
            "ratiobaseoverPL": ratio, "unresolvedRRpct": unres_rr,
            "eventtype": evtype, "eventtimes": evtime, "eventcause": evcause,
        })
    df = pd.DataFrame(data)
    st.session_state["df"] = df
    st.session_state["has_results"] = True
    st.success(f"Completed {len(df)} runs.")
# -----------------------------
# Results / Explore (guard pattern — no trailing else)
# -----------------------------
_has = bool(st.session_state.get("has_results"))
_df  = st.session_state.get("df")
if _has and _df is not None:
    df = _df
    n = len(df)
    # KPIs with Wilson 95% CI
    k_rev = int((df['eventtype']=="REVERSE").sum())
    k_str = int((df['eventtype']=="STRENGTHEN").sum())
    k_cpa = int(df['ALIMbreach_CPA'].sum())
    k_any = int(df['ALIMbreach_ANY_predCPA'].sum())
    p_rev = k_rev/n if n else 0.0; lo_rev, hi_rev = wilson_ci(k_rev, n)
    p_str = k_str/n if n else 0.0; lo_str, hi_str = wilson_ci(k_str, n)
    p_cpa = k_cpa/n if n else 0.0; lo_cpa, hi_cpa = wilson_ci(k_cpa, n)
    p_any = k_any/n if n else 0.0; lo_any, hi_any = wilson_ci(k_any, n)
    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("P(Reversal)",                f"{100*p_rev:,.2f}%  [{100*lo_rev:,.2f}–{100*hi_rev:,.2f}%]")
    k2.metric("P(Strengthen)",              f"{100*p_str:,.2f}%  [{100*lo_str:,.2f}–{100*hi_str:,.2f}%]")
    k3.metric("Mean RR",                    f"{df['unresolvedRRpct'].mean():.3f}%")
    k4.metric("P(ALIM breach @CPA)",        f"{100*p_cpa:,.2f}%  [{100*lo_cpa:,.2f}–{100*hi_cpa:,.2f}%]")
    k5.metric("P(ALIM ANY (pred-CPA, post)",f"{100*p_any:,.2f}%  [{100*lo_any:,.2f}–{100*hi_any:,.2f}%]")
    # Explore filters
    st.sidebar.subheader("Explore batch")
    tgo_lo, tgo_hi = st.sidebar.slider("tgo window (s)",
                                       float(max(8.0, df["tgos"].min())),
                                       float(df["tgos"].max()),
                                       (float(df["tgos"].min()), float(df["tgos"].max())))
    only_rev  = st.sidebar.checkbox("Only reversals", value=False)
    only_bANY = st.sidebar.checkbox("Only ALIM-breach ANY (pred-CPA, post)", value=False)
    view = df[df["tgos"].between(tgo_lo, tgo_hi)]
    if only_rev:
        view = view[view["eventtype"] == "REVERSE"]
    if only_bANY:
        view = view[view["ALIMbreach_ANY_predCPA"] == True]
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
            ax_mode.set_ylabel("Count"); ax_mode.set_title("Intruder modes")
            for tick in ax_mode.get_xticklabels():
                tick.set_rotation(20)
            st.pyplot(fig_mode)
    # Minimum predicted CPA miss histogram (post-engagement)
    if "minPredMiss_postEng_ft" in view.columns:
        fig_miss, ax_miss = plt.subplots(figsize=(6,3))
        ax_miss.hist(view["minPredMiss_postEng_ft"], bins=30)
        ax_miss.axvline(alim_ft, color="k", ls="--", alpha=0.7, label=f"ALIM={alim_ft:.0f} ft")
        ax_miss.set_xlabel("Min predicted CPA miss (ft) after engagement")
        ax_miss.set_ylabel("Count")
        ax_miss.grid(True, alpha=0.3)
        ax_miss.legend()
        st.pyplot(fig_miss)
    # Download CSV
    st.subheader("Download batch data")
    csv_buf = io.BytesIO()
    csv_buf.write(df.to_csv(index=False).encode("utf-8"))
    csv_buf.seek(0)
    st.download_button(label="Download CSV", data=csv_buf,
                       file_name="tcas_batch_results.csv",
                       mime="text/csv", key="dl_csv")
# Independent hint (no trailing else)
if not (_has and _df is not None):
    st.info("Run a batch to see results. Use the **form** above; results will persist while you explore.")
