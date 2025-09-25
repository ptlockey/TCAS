# --- Streamlit + plotting setup ---
import matplotlib
matplotlib.use("Agg")  # headless backend for Streamlit/servers
import matplotlib.pyplot as plt

import io
import numpy as np
import pandas as pd
import streamlit as st

# -----------------------------
# Constants & helpers
# -----------------------------
G = 9.80665             # m/s^2 per g
FT_PER_M = 3.28084
MS_PER_FPM = 0.00508    # 1 fpm = 0.00508 m/s

DEFAULT_ALIM_FT = 600.0
DEFAULT_RESP_THRESH_FPM = 300.0   # Eurocontrol/IATA FDM meaningful response threshold
ALIM_MARGIN_FT = 100.0

def dwell_fn(tgo_s: float) -> float:
    """Reversal dwell scales mildly with how late the encounter is."""
    # 0.8 s at 12s tgo, rising to 1.8 s by ~32s, then clamped
    return float(np.clip(0.8 + 0.05*(tgo_s - 12.0), 0.8, 1.8))

# --- piecewise vertical kinematics (accelerate to VS target, then hold) ---
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

def integrate_altitude_from_vs(times_s: np.ndarray, vs_fpm: np.ndarray) -> np.ndarray:
    """Integrate VS (fpm) to altitude (ft), z(0)=0."""
    dt = np.diff(times_s, prepend=times_s[0])
    fps = vs_fpm / 60.0
    z = np.cumsum(fps * dt)
    z[0] = 0.0
    return z

def predicted_dh_linear(vs_own_fpm, vs_int_fpm, t_go_s):
    """Linear projection of vertical miss distance over t_go (approx)."""
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

# --- v7.1 surrogate (with cause tagging) ---
def surrogate_decision_with_cause(times, vs_own, vs_int, t_cpas_s, resp_thr=DEFAULT_RESP_THRESH_FPM):
    """
    First event: 'STRENGTHEN' or 'REVERSE' or none.
    Causes:
      - INTRUDER_NONCOMPL_AFTER_DWELL
      - ALIM_SHORTFALL_LATE  (tgo<6 → REVERSE)
      - ALIM_SHORTFALL_EARLY (tgo>=6 → STRENGTHEN)
    """
    min_eval_time = 0.2  # guard against acting at t≈0 before any response is possible

    for i, t in enumerate(times):
        t_go = max(0.0, t_cpas_s - t)
        if t_go <= 0:
            break

        own_ok = compliant_in_sense(vs_own[i], +1, thr=resp_thr)
        int_ok = compliant_in_sense(vs_int[i], -1, thr=resp_thr)

        # 1) reversal for intruder non-compliance after dwell (own is compliant)
        if (t >= min_eval_time) and (not int_ok) and own_ok and (t >= dwell_fn(t_go)):
            return [("REVERSE", float(t), "INTRUDER_NONCOMPL_AFTER_DWELL")]

        # 2) ALIM shortfall logic, only once at least one aircraft is compliant
        if (t >= min_eval_time) and (own_ok or int_ok):
            dh_pred = predicted_dh_linear(vs_own[i], vs_int[i], t_go)
            if dh_pred < (DEFAULT_ALIM_FT - ALIM_MARGIN_FT):
                if t_go < 6.0:
                    return [("REVERSE", float(t), "ALIM_SHORTFALL_LATE")]
                else:
                    return [("STRENGTHEN", float(t), "ALIM_SHORTFALL_EARLY")]

    return []

def baseline_dh_ft(t_cpas_s, mode="IDEAL"):
    """Δh for the baseline 1500 fpm aircraft at same t_cpas (for risk scaling)."""
    if mode == "IDEAL":
        return delta_h_piecewise(t_cpas_s, t_delay_s=1.0, a_g=0.25, v_f_fpm=1500)
    else:
        return delta_h_piecewise(t_cpas_s, t_delay_s=5.0, a_g=0.25, v_f_fpm=1500)

# -----------------------------
# Streamlit UI
# -----------------------------
st.title("ACAS/TCAS v7.1 — Residual Risk & RA Taxonomy (Batch Monte Carlo)")

st.markdown("""
Implements a piecewise vertical-response model (accelerate to VS target, then hold) to compute Δh at CPA,
scales unresolved residual risk vs a 1500 fpm baseline, and applies a light v7.1 surrogate
(“strengthen” vs “reverse”). ALIM breaches are checked at CPA only (proxy for unresolved min).
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

# --- Initial vertical miss controls (used in single-run and batch) ---
with st.expander("Initial vertical miss (at RA)"):
    h0_mean = st.number_input("Initial vertical miss mean h0 (ft)", value=0.0, step=50.0)
    h0_sd   = st.number_input("Initial vertical miss std dev (ft)", value=0.0, step=25.0)

# Single-run Δh & risk (use mean h0 for determinism)
baseline_mode = "IDEAL" if baseline.startswith("IDEAL") else "STANDARD"
dh_pl_ft = delta_h_piecewise(t_cpa, pl_td, pl_ag, min(pl_vs, pl_cap))
dh_cat_ft = delta_h_piecewise(t_cpa, cat_td, cat_ag, min(cat_vs, cat_cap))
dh_base_ft = baseline_dh_ft(t_cpa, mode=baseline_mode)
dh_total_ft = dh_pl_ft + dh_cat_ft
h0_ft_single = h0_mean
miss_cpa_single = abs(h0_ft_single + dh_total_ft)

st.markdown("### Single-run Δh, CPA miss & unresolved residual risk")
spot = pd.DataFrame({
    "Aircraft": ["PL (ownship)", "CAT (intruder)", f"Baseline ({baseline_mode})"],
    "Δh @ CPA (ft)": [dh_pl_ft, dh_cat_ft, dh_base_ft]
})
st.dataframe(spot, use_container_width=True)
st.write(
    f"Vertical miss @ CPA ≈ **{miss_cpa_single:,.0f} ft** "
    f"({'BREACH' if miss_cpa_single < alim_ft else 'OK'} vs ALIM={alim_ft:.0f})"
)

ratio = (dh_base_ft / dh_pl_ft) if dh_pl_ft > 1e-6 else np.nan
unres_rr = 1.1 * ratio
st.write(f"Scaled unresolved residual risk ≈ **{unres_rr:,.3f}%** (ratio {ratio:,.3f})")

# Surrogate event for single-run
times, vs_pl = vs_time_series(t_cpa, dt, pl_td, pl_ag, pl_vs, sense=+1, cap_fpm=pl_cap)
_,     vs_cat = vs_time_series(t_cpa, dt, cat_td, cat_ag, cat_vs, sense=-1, cap_fpm=cat_cap)
ev = surrogate_decision_with_cause(times, vs_pl, vs_cat, t_cpas_s=t_cpa, resp_thr=resp_thr)

st.markdown("### Single-run RA taxonomy (surrogate)")
if ev:
    st.write(f"First event: **{ev[0][0]}** at t={ev[0][1]:.1f}s — cause: `{ev[0][2]}`.")
else:
    st.write("No strengthen/reverse flagged (coordinated & compliant case).")

# VS plot
fig_vs, ax_vs = plt.subplots(figsize=(7, 3))
ax_vs.plot(times, vs_pl, label="PL VS (fpm)")
ax_vs.plot(times, vs_cat, label="CAT VS (fpm)")
ax_vs.set_xlabel("Time since RA (s)")
ax_vs.set_ylabel("Vertical speed (fpm)")
ax_vs.grid(True, alpha=0.3)
ax_vs.legend()
st.pyplot(fig_vs)

# Geometry plot (Z vs time) with Δh_total & miss @ CPA
z_pl = integrate_altitude_from_vs(times, vs_pl)         # ft
z_cat = -integrate_altitude_from_vs(times, -vs_cat)     # keep sign (CAT descending negative)

fig_z, ax_z = plt.subplots(figsize=(7, 3.6))
ax_z.plot(times, z_pl, label="PL Z (ft)")
ax_z.plot(times, z_cat, label="CAT Z (ft)")
ax_z.axvline(t_cpa, ls="--", alpha=0.7)
ax_z.grid(True, alpha=0.3)
ax_z.set_xlabel("Time since RA (s)")
ax_z.set_ylabel("Altitude change from RA (ft)")
ax_z.legend(loc="best")

# annotate Δh and miss
ax_z.annotate(f"Δh_total @ CPA ≈ {dh_total_ft:,.0f} ft",
              xy=(t_cpa, z_pl[-1]),
              xytext=(t_cpa*0.55, max(z_pl.max(), z_cat.max(), 200)*0.7),
              arrowprops=dict(arrowstyle="->", lw=1))
ax_z.annotate(f"|h0 + Δh| @ CPA ≈ {miss_cpa_single:,.0f} ft",
              xy=(t_cpa, z_cat[-1]),
              xytext=(t_cpa*0.25, min(z_pl.min(), z_cat.min(), -200)*0.6),
              arrowprops=dict(arrowstyle="->", lw=1))
st.pyplot(fig_z)

# -----------------------------
# Batch Monte Carlo
# -----------------------------
st.markdown("---")
st.header("Batch Monte Carlo")

with st.expander("Batch settings"):
    n_runs = st.number_input("Number of runs", min_value=1, max_value=100000, value=2000, step=100)
    seed = st.number_input("Random seed", value=42, step=1)
    scenario = st.selectbox("Scenario", ["Head-on", "Crossing", "Overtaking", "Custom"])
    r_min = st.number_input("Initial range min (NM)", value=5.0, step=0.5, min_value=0.5)
    r_max = st.number_input("Initial range max (NM)", value=12.0, step=0.5, min_value=1.0)
    tgo_cap = st.number_input("Max tgo cap (s)", value=60.0, step=5.0, min_value=15.0)

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

with st.expander("Human/aircraft variability"):
    pl_delay_jitter = st.number_input("PL delay jitter ± (s)", value=0.0, step=0.1, format="%.1f")
    cat_delay_jitter = st.number_input("CAT delay jitter ± (s)", value=0.0, step=0.1, format="%.1f")

def sample_headings(rng, scenario, hdg1_min, hdg1_max, rel_min=None, rel_max=None, hdg2_min=None, hdg2_max=None):
    h1 = rng.uniform(hdg1_min, hdg1_max)
    if scenario == "Custom":
        h2 = rng.uniform(hdg2_min, hdg2_max)
    else:
        rel = rng.uniform(rel_min, rel_max)
        dirsign = 1 if rng.uniform() < 0.5 else -1
        h2 = (h1 + dirsign * rel) % 360.0
    return h1, h2

runbtn = st.button("Run batch")

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
        r0 = rng.uniform(min(r_min, r_max), max(r_min, r_max))
        vcl = relative_closure_kt(pl_tas, h1, cat_tas, h2)
        tgo = time_to_go_from_geometry(r0, vcl)
        if tgo is None:
            continue
        tgo = float(np.clip(tgo, 8.0, tgo_cap))  # allow later encounters if you lower the cap

        # jittered delays (clamped ≥0)
        pl_td_k  = float(max(0.0, pl_td  + (rng.uniform(-pl_delay_jitter,  pl_delay_jitter) if pl_delay_jitter  > 0 else 0.0)))
        cat_td_k = float(max(0.0, cat_td + (rng.uniform(-cat_delay_jitter, cat_delay_jitter) if cat_delay_jitter > 0 else 0.0)))

        # Δh (ft)
        dhpl = delta_h_piecewise(tgo, pl_td_k, pl_ag, min(pl_vs, pl_cap))
        dhcat = delta_h_piecewise(tgo, cat_td_k, cat_ag, min(cat_vs, cat_cap))
        dhtotal = dhpl + dhcat

        # initial vertical miss sampling
        h0_ft = float(h0_mean + (rng.normal(0, h0_sd) if h0_sd > 0 else 0.0))
        miss_cpa = abs(h0_ft + dhtotal)

        # baseline & risk
        dhbase = baseline_dh_ft(tgo, mode=baseline_mode)
        ratio = (dhbase / dhpl) if dhpl > 1e-6 else np.nan
        unresrr = 1.1 * ratio

        # surrogate events (with cause) using jittered delays
        times_r, vs_pl_r = vs_time_series(tgo, dt, pl_td_k, pl_ag, pl_vs, sense=+1, cap_fpm=pl_cap)
        _,       vs_cat_r = vs_time_series(tgo, dt, cat_td_k, cat_ag, cat_vs, sense=-1, cap_fpm=cat_cap)
        ev = surrogate_decision_with_cause(times_r, vs_pl_r, vs_cat_r, t_cpas_s=tgo, resp_thr=resp_thr)
        evtype = ev[0][0] if ev else "NONE"
        evtime = ev[0][1] if ev else np.nan
        evcause = ev[0][2] if ev else "N/A"

        # ALIM@CPA breach
        breach = miss_cpa < alim_ft

        data.append({
            "run": k + 1,
            "scenario": scenario,
            "PLhdg": h1,
            "CAThdg": h2,
            "R0NM": r0,
            "closurekt": vcl,
            "tgos": tgo,
            "plDelay": pl_td_k,
            "catDelay": cat_td_k,
            "h0ft": h0_ft,
            "dhPLft": dhpl,
            "dhCATft": dhcat,
            "dhtotalft": dhtotal,
            "missCPAft": miss_cpa,
            "dhbaselineft": dhbase,
            "ratiobaseoverPL": ratio,
            "unresolvedRRpct": unresrr,
            "eventtype": evtype,
            "eventtimes": evtime,
            "eventcause": evcause,
            "ALIMbreachatCPA": breach,
        })

    df = pd.DataFrame(data)
    st.success(f"Completed {len(df)} runs.")

    # --- KPIs
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("P(reversal)", f"{(df['eventtype']=='REVERSE').mean():.2%}")
    k2.metric("P(strengthen)", f"{(df['eventtype']=='STRENGTHEN').mean():.2%}")
    k3.metric("Mean RR", f"{df['unresolvedRRpct'].mean():.3f}%")
    k4.metric("P(ALIM breach@CPA)", f"{df['ALIMbreachatCPA'].mean():.2%}")

    # --- Filters
    st.sidebar.subheader("Explore batch")
    tgo_lo, tgo_hi = st.sidebar.slider("tgo window (s)", 8.0, float(tgo_cap), (8.0, float(tgo_cap)))
    only_rev = st.sidebar.checkbox("Only reversals", value=False)
    view = df[df["tgos"].between(tgo_lo, tgo_hi)]
    if only_rev:
        view = view[view["eventtype"] == "REVERSE"]

    st.subheader("Preview of results")
    st.dataframe(view.head(200), use_container_width=True)

    # --- ECDF of unresolved RR
    vals = view["unresolvedRRpct"].dropna().values
    if len(vals):
        x = np.sort(vals); y = np.arange(1, len(x)+1)/len(x)
        fig_ecdf, ax_ecdf = plt.subplots(figsize=(6, 3))
        ax_ecdf.plot(x, y)
        ax_ecdf.set_xlabel("Unresolved RR (%)")
        ax_ecdf.set_ylabel("ECDF")
        ax_ecdf.grid(True, alpha=0.3)
        st.pyplot(fig_ecdf)

    # --- tgo hist by event type
    fig_hist, ax_hist = plt.subplots(figsize=(6, 3))
    for lab in ["STRENGTHEN", "REVERSE", "NONE"]:
        sub = view[view["eventtype"] == lab]["tgos"]
        if len(sub):
            ax_hist.hist(sub, bins=24, histtype="step", label=lab)
    ax_hist.set_xlabel("tgo (s)")
    ax_hist.set_ylabel("Count")
    ax_hist.grid(True, alpha=0.3)
    ax_hist.legend()
    st.pyplot(fig_hist)

    # --- event cause bar
    cause_counts = view["eventcause"].value_counts()
    if len(cause_counts):
        fig_bar, ax_bar = plt.subplots(figsize=(6, 3))
        ax_bar.bar(cause_counts.index, cause_counts.values)
        ax_bar.set_ylabel("Count"); ax_bar.set_title("Event causes")
        for tick in ax_bar.get_xticklabels():
            tick.set_rotation(20)
        st.pyplot(fig_bar)

    # --- miss @ CPA histogram
    fig_miss, ax_miss = plt.subplots(figsize=(6, 3))
    ax_miss.hist(view["missCPAft"], bins=30)
    ax_miss.axvline(alim_ft, color="k", ls="--", alpha=0.7, label=f"ALIM={alim_ft:.0f} ft")
    ax_miss.set_xlabel("Vertical miss @ CPA (ft)")
    ax_miss.set_ylabel("Count")
    ax_miss.grid(True, alpha=0.3)
    ax_miss.legend()
    st.pyplot(fig_miss)

    # --- Downloads (robust)
    st.subheader("Download batch data")
    # CSV
    csv_buf = io.BytesIO()
    csv_buf.write(df.to_csv(index=False).encode("utf-8"))
    csv_buf.seek(0)
    st.download_button(
        label="Download CSV",
        data=csv_buf,
        file_name="tcas_batch_results.csv",
        mime="text/csv",
        key="dl_csv"
    )
    # Parquet (smaller)
    try:
        import pyarrow  # noqa: F401
        pq_buf = io.BytesIO()
        df.to_parquet(pq_buf, index=False)
        pq_buf.seek(0)
        st.download_button(
            label="Download Parquet",
            data=pq_buf,
            file_name="tcas_batch_results.parquet",
            mime="application/octet-stream",
            key="dl_parquet"
        )
    except Exception:
        st.caption("Install `pyarrow` to enable Parquet download (smaller & preserves dtypes).")

