# app_v2.py
# ACAS/TCAS v7.1 — Residual Risk & RA Taxonomy (Batch Monte Carlo, regulator-ready)
# Adds: trajectory-min check, pilot response distributions, RA trigger surrogate,
# optional surveillance/noise, and Streamlit state fixes to stop "Explore batch" resets.
#
# References:
# - EUROCONTROL ACAS II Guide (operational principles, ALIM, v7.1 context)
# - ACASA WP-1 (baseline unresolved 1.1% + induced 2.2%, surveillance sensitivities)
# - EUROCONTROL/IATA FDM guidance (≥300 fpm "meaningful response")
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
DEFAULT_RESP_THRESH_FPM = 300.0     # EUROCONTROL/IATA "meaningful response" threshold
ALIM_MARGIN_FT = 100.0
# -----------------------------
# Kinematics
# -----------------------------
def delta_h_piecewise(t_cpa_s: float, t_delay_s: float, a_g: float, v_f_fpm: float) -> float:
    """Vertical displacement (FEET) from RA to CPA for one aircraft (delay -> ramp -> capped)."""
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
# Scenario-calibrated RA->CPA sampling (typical en-route windows, truncated by geometry)
def sample_ra_to_cpa_scenario(rng, scenario, tgo_geom, cap_s=60.0):
    # Means & SDs approximate typical RA windows (tune if calibrating to datasets)
    params = {
        "Head-on":    (25.0, 5.0),
        "Crossing":   (22.0, 6.0),
        "Overtaking": (30.0, 8.0),
        "Custom":     (25.0, 7.0),
    }
    mu, sd = params.get(scenario, (25.0, 6.0))
    lo, hi = 12.0, min(tgo_geom if tgo_geom is not None else cap_s, cap_s)
    if hi <= lo:
        return float(max(8.0, min(tgo_geom or 30.0, cap_s)))
    tgo = float(np.clip(rng.normal(mu, sd), lo, hi))
    return tgo
# -----------------------------
# Baseline Δh for risk scaling
# -----------------------------
def baseline_dh_ft(t_cpa_s, mode="IDEAL"):
    """Δh for the baseline 1500 fpm aircraft at same t_cpa (for risk scaling)."""
    if mode == "IDEAL":
        return delta_h_piecewise(t_cpa_s, t_delay_s=1.0, a_g=0.25, v_f_fpm=1500)
    else:
        return delta_h_piecewise(t_cpa_s, t_delay_s=5.0, a_g=0.25, v_f_fpm=1500)
# -----------------------------
# Pilot response sampling
# -----------------------------
def sample_pilot_response(rng, role="PL"):
    """
    Return (delay_s, accel_g) from a simple mixture model:
      - 70% "on-time": delay ~ N(4.5,1.0), accel ~ N(0.25,0.03)
      - 30% "late/weak": delay ~ N(8.5,1.5), accel ~ N(0.10,0.02)
    For performance-limited PL, VS cap is enforced elsewhere.
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
# Surveillance/noise toggles
# -----------------------------
def apply_surveillance_noise(rng, times, vs_own, vs_int,
                             p_miss=0.0, p_range_err=0.0, sigma_range_ft=250.0):
    """
    Simple noise for surrogate prediction stress only:
      - with p_miss: hold previous VS (missing cycle)
      - p_range_err kept for future tgo perturbations (reserved)
    """
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
      - REVERSE if intruder non-compliant after dwell or ALIM_SHORTFALL_LATE
      - STRENGTHEN if ALIM_SHORTFALL_EARLY
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
        # 1) intruder wrong-direction / non-compliance triggers a reversal after dwell
        if (t >= min_eval_time) and (not int_ok) and own_ok and (t >= dwell_fn(t_go)):
            events.append(("REVERSE", float(t), "INTRUDER_NONCOMPL_AFTER_DWELL"))
            break
        # 2) ALIM shortfall prediction => strengthen early; reverse late
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
# Streamlit state helpers
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
Implements a piecewise vertical-response model to compute **Δh at CPA**;  
scales **unresolved residual risk** vs a 1500 fpm baseline (ACASA **1.1%** unresolved);  
applies a light **v7.1 surrogate** (Strengthen vs Reversal).  
**Unresolved** is checked both **at CPA** and **ANYWHERE** in the RA window (trajectory minimum).  
**Induced-risk** is **not** modelled; cite ACASA (~2.2% for ideal ACAS–ACAS) only as context.
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
st.subheader("Ownship (Performance‑limited) parameters")
c1, c2, c3 = st.columns(3)
with c1:
    pl_vs = st.number_input("PL target VS (fpm)", value=500, step=50)
    pl_cap = st.number_input("PL performance cap (fpm)", value=500, step=50)
with c2:
    pl_ag_nom = st.number_input("PL vertical accel nominal (g)", value=0.10, step=0.01, format="%.2f")
    pl_td_nom = st.number_input("PL RA delay nominal (s)", value=0.0, step=0.5)
with c3:
    pl_tas = st.number_input("PL TAS (kt)", value=120.0, step=10.0)
    pl_hdg = st.number_input("PL heading (deg)", value=0.0, step=5.0, min_value=0.0, max_value=359.9)
st.caption("Assumed sense: PL climbs (+); Intruder descends (−).")
st.subheader("Intruder (CAT) parameters")
d1, d2, d3 = st.columns(3)
with d1:
    cat_vs = st.number_input("CAT target VS (fpm)", value=1500, step=100)
    cat_cap = st.number_input("CAT performance cap (fpm)", value=1500, step=50)
with d2:
    cat_ag_nom = st.number_input("CAT vertical accel nominal (g)", value=0.25, step=0.01, format="%.2f")
    cat_td_nom = st.number_input("CAT RA delay nominal (s)", value=5.0, step=0.5)
with d3:
    cat_tas = st.number_input("CAT TAS (kt)", value=450.0, step=10.0)
    cat_hdg = st.number_input("CAT heading (deg)", value=180.0, step=5.0, min_value=0.0, max_value=359.9)
# Geometry & default RA->CPA (spot)
v_clos = relative_closure_kt(pl_tas, pl_hdg, cat_tas, cat_hdg)
tgo_geom = time_to_go_from_geometry(r0_nm := 8.0, v_clos) or 30.0
h1, h2 = st.columns(2)
with h1:
    st.metric("Computed closure (kt)", f"{v_clos:,.1f}")
with h2:
    st.caption(f"Geometry-only RA→CPA from 8 NM: ~{tgo_geom:,.1f}s (for reference)")
# Initial vertical miss settings
with st.expander("Initial vertical miss (at RA)"):
    h0_mean = st.number_input("Initial vertical miss mean h0 (ft)", value=0.0, step=50.0)
    h0_sd = st.number_input("Initial vertical miss std dev (ft)", value=0.0, step=25.0)
# RA trigger & Surveillance/noise
with st.expander("RA trigger & Surveillance/noise"):
    ra_trigger_mode = st.selectbox("RA→CPA mode",
                                   ["Geometry-derived", "Scenario-calibrated"])
    tgo_cap = st.number_input("Max RA→CPA cap (s)", value=60.0, step=5.0, min_value=15.0)
    st.markdown("**Surveillance/noise toggles** (optional)")
    p_miss = st.slider("P(missing cycle) per time-step", 0.0, 0.20, 0.00, 0.01)
    p_rng_err = st.slider("P(range error) per time-step (reserved)", 0.0, 0.20, 0.00, 0.01)
    sigma_rng_ft = st.number_input("Range error σ (ft) [reserved]", value=250, step=50, min_value=0)
# -----------------------------
# Single-run (spot check)
# -----------------------------
baseline_mode = "IDEAL" if baseline.startswith("IDEAL") else "STANDARD"
st.markdown("### Single‑run spot check")
t_cpa_spot = st.number_input("Spot check RA→CPA (s)", value=float(round(min(tgo_geom, tgo_cap), 1)), step=1.0)
dh_pl_ft = delta_h_piecewise(t_cpa_spot, pl_td_nom, pl_ag_nom, min(pl_vs, pl_cap))
dh_cat_ft = delta_h_piecewise(t_cpa_spot, cat_td_nom, cat_ag_nom, min(cat_vs, cat_cap))
dh_base_ft = baseline_dh_ft(t_cpa_spot, mode=baseline_mode)
ratio_spot = (dh_base_ft / dh_pl_ft) if dh_pl_ft > 1e-6 else np.nan
unres_rr_spot = 1.1 * ratio_spot
spot_tab = pd.DataFrame({
    "Aircraft": ["PL (ownship)", "CAT (intruder)", f"Baseline ({baseline_mode})"],
    "Δh @ CPA (ft)": [dh_pl_ft, dh_cat_ft, dh_base_ft]
})
st.dataframe(spot_tab, use_container_width=True)
st.write(f"Scaled unresolved RR ≈ **{unres_rr_spot:,.3f}%** (ratio {ratio_spot:,.3f})")
# -----------------------------
# Batch Monte Carlo — in a FORM (prevents unwanted reruns)
# -----------------------------
st.markdown("---")
st.header("Batch Monte Carlo")
with st.form("batch_form", clear_on_submit=False):
    n_runs = st.number_input("Number of runs", min_value=1, max_value=100000, value=2000, step=100)
    seed = st.number_input("Random seed", value=42, step=1)
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
    st.markdown("**Pilot response distributions** (override nominal values)")
    use_distrib = st.checkbox("Use mixture distributions (recommended)", value=True)
    submitted = st.form_submit_button("Run batch")
# Helper to sample headings
def sample_headings(rng, scenario, hdg1_min, hdg1_max, rel_min=None, rel_max=None, hdg2_min=None, hdg2_max=None):
    h1 = rng.uniform(hdg1_min, hdg1_max)
    if scenario == "Custom":
        h2 = rng.uniform(hdg2_min, hdg2_max)
    else:
        rel = rng.uniform(rel_min, rel_max)
        dirsign = 1 if rng.uniform() < 0.5 else -1
        h2 = (h1 + dirsign * rel) % 360.0
    return h1, h2
# -----------------------------
# Run batch only when submitted (fixes "Explore batch" reset)
# -----------------------------
if submitted:
    rng = np.random.default_rng(int(seed))
    data = []
    for k in range(int(n_runs)):
        # Headings & geometry
        if scenario == "Custom":
            h1, h2 = sample_headings(rng, scenario, hdg1_min, hdg1_max, None, None, hdg2_min, hdg2_max)
        else:
            h1, h2 = sample_headings(rng, scenario, hdg1_min, hdg1_max, rel_min, rel_max)
        r0 = rng.uniform(min(r_min, r_max), max(r_min, r_max))
        vcl = relative_closure_kt(pl_tas, h1, cat_tas, h2)
        tgo_geom = time_to_go_from_geometry(r0, vcl)
        if tgo_geom is None:
            continue
        # RA->CPA choice
        if ra_trigger_mode == "Scenario-calibrated":
            tgo = sample_ra_to_cpa_scenario(rng, scenario, tgo_geom, cap_s=tgo_cap)
        else:
            tgo = float(np.clip(tgo_geom, 8.0, tgo_cap))
        # Pilot response (distributions or nominal)
        if use_distrib:
            pl_td_k, pl_ag_k = sample_pilot_response(rng, "PL")
            cat_td_k, cat_ag_k = sample_pilot_response(rng, "CAT")
        else:
            pl_td_k, pl_ag_k = pl_td_nom, pl_ag_nom
            cat_td_k, cat_ag_k = cat_td_nom, cat_ag_nom
        # Δh (ft) at CPA
        dh_pl = delta_h_piecewise(tgo, pl_td_k, pl_ag_k, min(pl_vs, pl_cap))
        dh_cat = delta_h_piecewise(tgo, cat_td_k, cat_ag_k, min(cat_vs, cat_cap))
        dh_total = dh_pl + dh_cat
        # Baseline scaling
        dh_base = baseline_dh_ft(tgo, mode=baseline_mode)
        ratio = (dh_base / dh_pl) if dh_pl > 1e-6 else np.nan
        unres_rr = 1.1 * ratio
        # Time series for trajectory-min unresolved check
        times, vs_pl = vs_time_series(tgo, dt, pl_td_k, pl_ag_k, pl_vs, sense=+1, cap_fpm=pl_cap)
        _,     vs_ca = vs_time_series(tgo, dt, cat_td_k, cat_ag_k, cat_vs, sense=-1, cap_fpm=cat_cap)
        # Optional surveillance/noise for surrogate prediction only (not kinematics)
        vs_pl_noisy, vs_ca_noisy = apply_surveillance_noise(rng, times, vs_pl, vs_ca,
                                                            p_miss=p_miss, p_range_err=p_rng_err,
                                                            sigma_range_ft=sigma_rng_ft)
        # Integrate altitude for trajectory-min separation (unresolved ANY)
        z_pl = integrate_altitude_from_vs(times, vs_pl)         # ft
        z_ca = integrate_altitude_from_vs(times, vs_ca)         # ft
        sep_series = np.abs(h0_mean + (z_pl - z_ca))
        min_sep = float(np.min(sep_series))
        breach_any = (min_sep < alim_ft)
        miss_cpa = abs(h0_mean + (z_pl[-1] - z_ca[-1]))
        breach_cpa = (miss_cpa < alim_ft)
        # Surrogate event
        ev = surrogate_decision_with_cause(times, vs_pl_noisy, vs_ca_noisy, t_cpa_s=tgo, resp_thr=resp_thr)
        evtype = ev[0][0] if ev else "NONE"
        evtime = ev[0][1] if ev else np.nan
        evcause = ev[0][2] if ev else "N/A"
        data.append({
            "run": k + 1, "scenario": scenario,
            "PLhdg": h1, "CAThdg": h2, "R0NM": r0,
            "closurekt": vcl, "tgos": tgo,
            "plDelay": pl_td_k, "catDelay": cat_td_k,
            "plAccel_g": pl_ag_k, "catAccel_g": cat_ag_k,
            "h0ft": h0_mean,
            "dhPLft": dh_pl, "dhCATft": dh_cat, "dhtotalft": dh_total,
            "missCPAft": miss_cpa, "minSepft": min_sep,
            "ALIMbreach_CPA": breach_cpa, "ALIMbreach_ANY": breach_any,
            "dhbaselineft": dh_base, "ratiobaseoverPL": ratio, "unresolvedRRpct": unres_rr,
            "eventtype": evtype, "eventtimes": evtime, "eventcause": evcause
        })
    df = pd.DataFrame(data)
    st.session_state["df"] = df
    st.session_state["has_results"] = True
    st.success(f"Completed {len(df)} runs.")
# -----------------------------
# Results / Explore (no rerun) — reads from state
# -----------------------------
if st.session_state["has_results"] and st.session_state["df"] is not None:
    df = st.session_state["df"]
    # KPIs
    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("P(reversal)", f"{(df['eventtype']=='REVERSE').mean():.2%}")
    k2.metric("P(strengthen)", f"{(df['eventtype']=='STRENGTHEN').mean():.2%}")
    k3.metric("Mean RR", f"{df['unresolvedRRpct'].mean():.3f}%")
    k4.metric("P(ALIM breach @CPA)", f"{df['ALIMbreach_CPA'].mean():.2%}")
    k5.metric("P(ALIM breach ANY)", f"{df['ALIMbreach_ANY'].mean():.2%}")
    # --- Explore batch (filters apply to stored df only; does not rerun batch)
    st.sidebar.subheader("Explore batch")
    tgo_lo, tgo_hi = st.sidebar.slider("tgo window (s)",
                                       float(max(8.0, df["tgos"].min())),
                                       float(df["tgos"].max()),
                                       (float(df["tgos"].min()), float(df["tgos"].max())))
    only_rev = st.sidebar.checkbox("Only reversals", value=False)
    only_any_breach = st.sidebar.checkbox("Only ALIM-breach ANY", value=False)
    view = df[df["tgos"].between(tgo_lo, tgo_hi)]
    if only_rev:
        view = view[view["eventtype"] == "REVERSE"]
    if only_any_breach:
        view = view[view["ALIMbreach_ANY"] == True]
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
    # Min separation histogram (ANY)
    fig_miss, ax_miss = plt.subplots(figsize=(6,3))
    ax_miss.hist(view["minSepft"], bins=30)
    ax_miss.axvline(alim_ft, color="k", ls="--", alpha=0.7, label=f"ALIM={alim_ft:.0f} ft")
    ax_miss.set_xlabel("Minimum vertical miss (ft) during RA window")
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
else:
    st.info("Run a batch to see results. Use the **form** above; results will persist while you explore.")
 

