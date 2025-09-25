import streamlit as st
import numpy as np
import pandas as pd
import math
import matplotlib
matplotlib.use("Agg")  # headless backend for Streamlit/servers
import matplotlib.pyplot as plt


# -----------------------------
# Constants & helpers
# -----------------------------
G = 9.80665             # m/s^2 per g
FT_PER_M = 3.28084
M_PER_FT = 1.0 / FT_PER_M
MS_PER_FPM = 0.00508    # 1 fpm = 0.00508 m/s

DEFAULT_ALIM_FT = 600.0
DEFAULT_RESP_THRESH_FPM = 300.0   # Eurocontrol/IATA FDM meaningful response threshold
REVERSAL_DWELL_S = 1.5
ALIM_MARGIN_FT = 100.0

# --- piecewise vertical kinematics (paper's model) ---
def delta_h_piecewise(t_cpas_s: float, t_delay_s: float, a_g: float, v_f_fpm: float) -> float:
    """Return displacement in FEET from RA to CPA for one aircraft."""
    a = a_g * G                             # m/s^2
    v_f_mps = v_f_fpm * MS_PER_FPM         # m/s
    if t_cpas_s <= t_delay_s:
        dh_m = 0.0
    else:
        t = t_cpas_s - t_delay_s
        t_ramp = v_f_mps / a if a > 0 else np.inf
        if t <= t_ramp:
            dh_m = 0.5 * a * (t ** 2)
        else:
            dh_m = 0.5 * a * (t_ramp ** 2) + v_f_mps * (t - t_ramp)
    return dh_m * FT_PER_M                  # ft

def vs_time_series(t_end_s, dt_s, t_delay_s, a_g, v_f_fpm, sense, cap_fpm=None):
    """VS profile (fpm) with delay, ramp at a_g to v_f, then cap; signed with sense."""
    a = a_g * G
    v_f_target_fpm = v_f_fpm if cap_fpm is None else min(v_f_fpm, cap_fpm)
    times = np.arange(0.0, t_end_s + 1e-9, dt_s)
    vs = np.zeros_like(times, dtype=float)
    for i, t in enumerate(times):
        if t <= t_delay_s:
            vs[i] = 0.0
        else:
            te = t - t_delay_s
            v_mps = min(a * te, v_f_target_fpm * MS_PER_FPM)  # m/s
            vs[i] = sense * (v_mps / MS_PER_FPM)              # fpm (signed)
    return times, vs

def predicted_dh_linear(vs_own_fpm, vs_int_fpm, t_go_s):
    """Linear projection of vertical miss distance over time-to-go (approx)."""
    return abs((vs_own_fpm - vs_int_fpm) * (t_go_s / 60.0))

def compliant_in_sense(vs_fpm, sense, thr=DEFAULT_RESP_THRESH_FPM):
    return (vs_fpm * sense) >= thr

def relative_closure_kt(v1_kt, hdg1_deg, v2_kt, hdg2_deg):
    """Scalar closure rate from headings/speeds (kt)."""
    th1 = np.deg2rad(hdg1_deg)
    th2 = np.deg2rad(hdg2_deg)
    v1 = np.array([v1_kt * np.sin(th1), v1_kt * np.cos(th1)])
    v2 = np.array([v2_kt * np.sin(th2), v2_kt * np.cos(th2)])
    return float(np.linalg.norm(v1 - v2))

def time_to_go_from_geometry(r0_nm, v_closure_kt):
    if v_closure_kt <= 1e-6:
        return None
    return 3600.0 * (r0_nm / v_closure_kt)

# --- v7.1 surrogate: prefer strengthening; reverse for wrong-direction or late-chase ---
def surrogate_decision(times, vs_own, vs_int, t_cpas_s, resp_thr=DEFAULT_RESP_THRESH_FPM):
    """
    Decide first event: 'STRENGTHEN', 'REVERSE', or none, based on:
      1) compliance check,
      2) predicted ALIM shortfall,
      3) vertical-chase late geometry.
    We assume initial coordinated opposite senses: ownship = climb (+), intruder = descend (-).
    """
    for i, t in enumerate(times):
        t_go = max(0.0, t_cpas_s - t)
        if t_go <= 0:
            break
        own_ok = compliant_in_sense(vs_own[i], +1, thr=resp_thr)
        int_ok = compliant_in_sense(vs_int[i], -1, thr=resp_thr)
        dh_pred = predicted_dh_linear(vs_own[i], vs_int[i], t_go)

        # Wrong-direction intruder while own is compliant → REVERSE (after dwell)
        if (not int_ok) and own_ok and (t >= REVERSAL_DWELL_S):
            return [("REVERSE", float(t))]

        # ALIM shortfall prediction
        if dh_pred < (DEFAULT_ALIM_FT - ALIM_MARGIN_FT):
            if t_go < 6.0:
                return [("REVERSE", float(t))]
            else:
                return [("STRENGTHEN", float(t))]
    return []

def baseline_dh_ft(t_cpas_s, mode="IDEAL"):
    """Δh for the baseline 1500 fpm aircraft at same tcpas (for risk scaling)."""
    if mode == "IDEAL":
        return delta_h_piecewise(t_cpas_s, t_delay_s=1.0, a_g=0.25, v_f_fpm=1500)
    else:
        return delta_h_piecewise(t_cpas_s, t_delay_s=5.0, a_g=0.25, v_f_fpm=1500)

# -----------------------------
# Streamlit UI
# -----------------------------
st.title("ACAS/TCAS v7.1 — Residual Risk & RA Taxonomy (Batch Monte Carlo)")

st.markdown("""
This tool implements your paper’s piecewise vertical-response model to compute Δh at CPA and scale
unresolved residual risk using the ACASA 1.1% benchmark, then applies a light v7.1 surrogate
to tag strengthening vs reversal events.

**Note:** This flags ALIM breaches only at CPA. True “unresolved” in ACASA considers minima anywhere along the trajectory.
""")

with st.sidebar:
    st.header("Global settings")
    alim_ft = st.number_input("ALIM (ft)", value=600.0, step=50.0)
    baseline = st.selectbox("Baseline for risk scaling",
                            ["IDEAL 1500 fpm (ACASA 2002)", "STANDARD 1500 fpm (EUROCONTROL 2018)"])
    dt = st.number_input("Time step (s)", value=0.1, step=0.1, format="%.1f")
    resp_thr = st.number_input("Meaningful response threshold (fpm)", value=300.0, step=50.0)

st.subheader("Ownship (Performance-limited) parameters")
c1, c2, c3 = st.columns(3)
with c1:
    pl_vs = st.number_input("PL target VS (fpm)", value=500, step=50)
    pl_ag = st.number_input("PL vertical accel (g)", value=0.10, step=0.01, format="%.2f")
    pl_td = st.number_input("PL RA delay (s)", value=0.0, step=0.5)
with c2:
    pl_tas = st.number_input("PL TAS (kt)", value=120.0, step=10.0)
    pl_cap = st.number_input("PL performance cap (fpm)", value=500, step=50)
with c3:
    pl_hdg = st.number_input("PL heading (deg)", value=0.0, step=5.0, min_value=0.0, max_value=359.9)
    st.write("Assumed sense: PL climbs (+); Intruder descends (−).")

st.subheader("Intruder (CAT) parameters")
d1, d2, d3 = st.columns(3)
with d1:
    cat_vs = st.number_input("CAT target VS (fpm)", value=1500, step=100)
    cat_ag = st.number_input("CAT vertical accel (g)", value=0.25, step=0.01, format="%.2f")
    cat_td = st.number_input("CAT RA delay (s)", value=5.0, step=0.5)
with d2:
    cat_tas = st.number_input("CAT TAS (kt)", value=450.0, step=10.0)
    cat_cap = st.number_input("CAT performance cap (fpm)", value=1500, step=50)
with d3:
    r0_nm = st.number_input("Initial range (NM)", value=8.0, step=0.5)
    cat_hdg = st.number_input("CAT heading (deg)", value=180.0, step=5.0, min_value=0.0, max_value=359.9)

# Compute closure & default TCPA
v_clos = relative_closure_kt(pl_tas, pl_hdg, cat_tas, cat_hdg)
tgo_default = time_to_go_from_geometry(r0_nm, v_clos) or 30.0

h1, h2 = st.columns(2)
with h1:
    st.metric("Computed closure (kt)", f"{v_clos:,.1f}")
with h2:
    t_cpa = st.number_input("RA→CPA time (s) (override)", value=float(round(tgo_default, 1)), step=1.0)

# Single-run Δh & risk
dh_pl_ft = delta_h_piecewise(t_cpa, pl_td, pl_ag, min(pl_vs, pl_cap))
dh_cat_ft = delta_h_piecewise(t_cpa, cat_td, cat_ag, min(cat_vs, cat_cap))
baseline_mode = "IDEAL" if baseline.startswith("IDEAL") else "STANDARD"
dh_base_ft = baseline_dh_ft(t_cpa, mode=baseline_mode)

st.markdown("### Single-run Δh & unresolved residual risk")
spot = pd.DataFrame({
    "Aircraft": ["PL (ownship)", "CAT (intruder)", f"Baseline ({baseline_mode})"],
    "Δh @ CPA (ft)": [dh_pl_ft, dh_cat_ft, dh_base_ft]
})
st.dataframe(spot, use_container_width=True)

ratio = (dh_base_ft / dh_pl_ft) if dh_pl_ft > 1e-6 else np.nan
unres_rr = 1.1 * ratio
st.write(f"Scaled unresolved residual risk ≈ {unres_rr:,.3f}% (ratio {ratio:,.3f})")

# Surrogate event for single-run
times, vs_pl = vs_time_series(t_cpa, dt, pl_td, pl_ag, pl_vs, sense=+1, cap_fpm=pl_cap)
_,     vs_cat = vs_time_series(t_cpa, dt, cat_td, cat_ag, cat_vs, sense=-1, cap_fpm=cat_cap)
events = surrogate_decision(times, vs_pl, vs_cat, t_cpas_s=t_cpa, resp_thr=resp_thr)

st.markdown("### Single-run RA taxonomy (surrogate)")
if events:
    st.write(f"First event: {events[0][0]} at t={events[0][1]:.1f}s.")
else:
    st.write("No strengthen/reverse flagged (coordinated & compliant case).")

fig, ax = plt.subplots(figsize=(7, 3))
ax.plot(times, vs_pl, label="PL VS (fpm)")
ax.plot(times, vs_cat, label="CAT VS (fpm)")
ax.set_xlabel("Time since RA (s)")
ax.set_ylabel("Vertical speed (fpm)")
ax.grid(True, alpha=0.3)
ax.legend()
st.pyplot(fig)

# -----------------------------
# Batch Monte Carlo
# -----------------------------
st.markdown("---")
st.header("Batch Monte Carlo")

with st.expander("Batch settings"):
    n_runs = st.number_input("Number of runs", min_value=1, max_value=10000, value=2000, step=100)
    seed = st.number_input("Random seed", value=42, step=1)
    scenario = st.selectbox("Scenario", ["Head-on", "Crossing", "Overtaking", "Custom"])
    r_min = st.number_input("Initial range min (NM)", value=5.0, step=0.5)
    r_max = st.number_input("Initial range max (NM)", value=12.0, step=0.5)

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

    fllow = st.number_input("Altitude band low (FL)", value=150, step=10)
    flhigh = st.number_input("Altitude band high (FL)", value=300, step=10)

    runbtn = st.button("Run batch")

def sample_headings(rng, scenario, hdg1_min, hdg1_max, rel_min=None, rel_max=None, hdg2_min=None, hdg2_max=None):
    h1 = rng.uniform(hdg1_min, hdg1_max)
    if scenario == "Custom":
        h2 = rng.uniform(hdg2_min, hdg2_max)
    else:
        rel = rng.uniform(rel_min, rel_max)
        dirsign = 1 if rng.uniform() < 0.5 else -1
        h2 = (h1 + dirsign * rel) % 360.0
    return h1, h2

if runbtn:
    rng = np.random.default_rng(int(seed))
    data = []

    for k in range(int(n_runs)):
        # headings
        if scenario == "Custom":
            h1 = rng.uniform(hdg1_min, hdg1_max)
            h2 = rng.uniform(hdg2_min, hdg2_max)
        else:
            h1, h2 = sample_headings(rng, scenario, hdg1_min, hdg1_max, rel_min, rel_max)

        # range and closure
        r0 = rng.uniform(r_min, r_max)
        vcl = relative_closure_kt(pl_tas, h1, cat_tas, h2)
        tgo = time_to_go_from_geometry(r0, vcl)
        if tgo is None:
            continue
        tgo = min(max(tgo, 12.0), 60.0)  # practical window

        # Δh (ft)
        dhpl = delta_h_piecewise(tgo, pl_td, pl_ag, min(pl_vs, pl_cap))
        dhcat = delta_h_piecewise(tgo, cat_td, cat_ag, min(cat_vs, cat_cap))
        dhtotal = dhpl + dhcat

        # baseline & risk
        dhbase = baseline_dh_ft(tgo, mode=baseline_mode)
        ratio = (dhbase / dhpl) if dhpl > 1e-6 else np.nan
        unresrr = 1.1 * ratio

        # surrogate events
        times_r, vs_pl_r = vs_time_series(tgo, dt, pl_td, pl_ag, pl_vs, sense=+1, cap_fpm=pl_cap)
        _,       vs_cat_r = vs_time_series(tgo, dt, cat_td, cat_ag, cat_vs, sense=-1, cap_fpm=cat_cap)
        ev = surrogate_decision(times_r, vs_pl_r, vs_cat_r, t_cpas_s=tgo, resp_thr=resp_thr)
        evtype = ev[0][0] if ev else "NONE"
        evtime = ev[0][1] if ev else np.nan

        # ALIM@CPA breach
        breach = dhtotal < alim_ft

        data.append({
            "run": k + 1,
            "scenario": scenario,
            "PLhdg": h1,
            "CAThdg": h2,
            "R0NM": r0,
            "closurekt": vcl,
            "tgos": tgo,
            "dhPLft": dhpl,
            "dhCATft": dhcat,
            "dhtotalft": dhtotal,
            "dhbaselineft": dhbase,
            "ratiobaseoverPL": ratio,
            "unresolvedRRpct": unresrr,
            "eventtype": evtype,
            "eventtimes": evtime,
            "ALIMbreachatCPA": breach,
            "FLband": f"FL{int(fllow)}–FL{int(flhigh)}"
        })

    df = pd.DataFrame(data)
    st.success(f"Completed {len(df)} runs.")

    # Aggregates
    prev = (df["eventtype"] == "REVERSE").mean()
    pstr = (df["eventtype"] == "STRENGTHEN").mean()
    pbreach = df["ALIMbreachatCPA"].mean()

    st.subheader("Batch summary")
    st.write(f"Mean unresolved RR: {df['unresolvedRRpct'].mean():.3f}% "
             f"(p10={df['unresolvedRRpct'].quantile(0.10):.3f}%, "
             f"median={df['unresolvedRRpct'].median():.3f}%, "
             f"p90={df['unresolvedRRpct'].quantile(0.90):.3f}%)")
    st.write(f"P(reversal): {prev:.4f}  |  P(strengthen): {pstr:.4f}  |  P(ALIM breach @ CPA): {pbreach:.4f}")

    # Plots
    c1, c2 = st.columns(2)
    with c1:
        fig1, ax1 = plt.subplots(figsize=(6, 3))
        ax1.hist(df["unresolvedRRpct"].dropna(), bins=30)
        ax1.set_title("Unresolved residual risk (%)")
        ax1.set_xlabel("%")
        ax1.set_ylabel("Count")
        ax1.grid(alpha=0.3)
        st.pyplot(fig1)
    with c2:
        counts = df["eventtype"].value_counts().reindex(["STRENGTHEN", "REVERSE", "NONE"]).fillna(0)
        fig2, ax2 = plt.subplots(figsize=(5, 3))
        ax2.pie(counts.values, labels=counts.index, autopct="%1.1f%%")
        ax2.set_title("Event taxonomy")
        st.pyplot(fig2)

    st.subheader("Download per-run CSV")
    st.download_button(
        label="Download CSV",
        data=df.to_csv(index=False).encode("utf-8"),
        filename="tcas_batch_results.csv",
        mime="text/csv"
    )

    st.subheader("Preview of results")
    st.dataframe(df.head(20), use_container_width=True)

st.caption("""
References: EUROCONTROL ACAS II Guide (system behaviour, ALIM) · FAA TCAS II v7.1 Intro (reversal improvements) ·
EUROCONTROL ACAS II Bulletin 13 (reversal rarity and causes) · Skybrary (TCAS RA reversal operational notes) ·
EUROCONTROL/IATA FDM guidance (≥300 fpm 'meaningful' response).
""")
