
# ACAS/TCAS Monte Carlo — with per-run plotter and enhanced single-run panel
import io
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import streamlit as st

G = 9.80665
FT_PER_M = 3.28084
MS_PER_FPM = 0.00508

PL_DELAY_S = 0.9
PL_ACCEL_G = 0.10
PL_VS_FPM  = 500.0
PL_VS_CAP  = 500.0
PL_IAS_KT  = 120.0

DEFAULT_ALIM_FT = 600.0
DEFAULT_RESP_THRESH_FPM = 300.0
ALIM_MARGIN_FT = 100.0
Z_95 = 1.96

def ias_to_tas(ias_kt: float, pressure_alt_ft: float) -> float:
    sigma = (1.0 - 6.875e-6 * pressure_alt_ft)**4.256
    sigma = max(1e-3, sigma)
    return ias_kt / np.sqrt(sigma)

def vs_time_series(t_end_s, dt_s, t_delay_s, a_g, v_f_fpm, sense, cap_fpm=None, vs0_fpm=0.0):
    a = a_g * G
    v_target = v_f_fpm if cap_fpm is None else min(v_f_fpm, cap_fpm)
    a_fpm_s = a * 60.0
    times = np.arange(0.0, t_end_s + 1e-9, dt_s)
    vs_fpm = np.zeros_like(times, dtype=float)
    for i, t in enumerate(times):
        if t <= t_delay_s:
            vs_fpm[i] = vs0_fpm
        else:
            te = t - t_delay_s
            target_signed = sense * v_target
            delta = target_signed - vs0_fpm
            step = np.sign(delta) * min(abs(a_fpm_s * te), abs(delta))
            vs_fpm[i] = vs0_fpm + step
    return times, vs_fpm

def integrate_altitude_from_vs(times_s, vs_fpm, z0_ft=0.0):
    dt = np.diff(times_s, prepend=times_s[0])
    z = z0_ft + np.cumsum((vs_fpm/60.0) * dt)
    z[0] = z0_ft
    return z

def relative_closure_kt(v1_kt, hdg1_deg, v2_kt, hdg2_deg) -> float:
    th1, th2 = np.deg2rad(hdg1_deg), np.deg2rad(hdg2_deg)
    v1 = np.array([v1_kt*np.sin(th1), v1_kt*np.cos(th1)])
    v2 = np.array([v2_kt*np.sin(th2), v2_kt*np.cos(th2)])
    return float(np.linalg.norm(v1 - v2))

def time_to_go_from_geometry(r0_nm, v_closure_kt):
    if v_closure_kt <= 1e-6:
        return None
    return 3600.0 * (r0_nm / v_closure_kt)

def sample_headings(rng, scenario, hdg1_min, hdg1_max, rel_min=None, rel_max=None, hdg2_min=None, hdg2_max=None):
    h1 = rng.uniform(hdg1_min, hdg1_max)
    if scenario == 'Custom':
        h2 = rng.uniform(hdg2_min, hdg2_max)
    else:
        rel = rng.uniform(rel_min, rel_max)
        dirsign = 1 if rng.uniform() < 0.5 else -1
        h2 = (h1 + dirsign * rel) % 360.0
    return h1, h2

def baseline_dh_ft(t_cpa_s, mode='IDEAL'):
    if mode.startswith('IDEAL'):
        return delta_h_piecewise(t_cpa_s, t_delay_s=1.0, a_g=0.25, v_f_fpm=1500.0)
    else:
        return delta_h_piecewise(t_cpa_s, t_delay_s=5.0, a_g=0.25, v_f_fpm=1500.0)

def delta_h_piecewise(t_cpa_s: float, t_delay_s: float, a_g: float, v_f_fpm: float) -> float:
    a = a_g * G
    v_f_mps = v_f_fpm * MS_PER_FPM
    if t_cpa_s <= t_delay_s:
        return 0.0
    t = t_cpa_s - t_delay_s
    t_ramp = v_f_mps / a if a > 0 else float('inf')
    if t <= t_ramp:
        dh_m = 0.5 * a * (t**2)
    else:
        dh_m = 0.5 * a * (t_ramp**2) + v_f_mps * (t - t_ramp)
    return dh_m * FT_PER_M

def wilson_ci(k, n, z=Z_95):
    if n <= 0: return (0.0, 0.0)
    phat = k / n
    denom = 1.0 + (z*z)/n
    center = (phat + (z*z)/(2*n)) / denom
    half = z * np.sqrt((phat*(1-phat) + (z*z)/(4*n))/n) / denom
    return (max(0.0, center - half), min(1.0, center + half))

def sample_initial_vs(rng, mode='Mixed', level_sd=50.0, climb_mean=1000.0, climb_sd=200.0, descend_mean=-1000.0, descend_sd=200.0):
    if mode == 'Mostly level': probs = (0.8, 0.1, 0.1)
    elif mode == 'Aggressive': probs = (0.4, 0.3, 0.3)
    else: probs = (0.6, 0.2, 0.2)
    u = rng.uniform()
    if u < probs[0]: return float(rng.normal(0.0, level_sd))
    elif u < probs[0] + probs[1]: return float(rng.normal(climb_mean, climb_sd))
    else: return float(rng.normal(descend_mean, descend_sd))

def choose_sense_from_trend(cat_above: bool, vz_pl0_fpm: float, vz_cat0_fpm: float, thr: float = 200.0, default_policy: str = 'High climbs, low descends'):
    high_is_cat = bool(cat_above)
    a_high_cmd, a_low_cmd = +1, -1
    b_high_cmd, b_low_cmd = -1, +1
    def trend(v): return +1 if v >= thr else (-1 if v <= -thr else 0)
    pl_trend  = trend(vz_pl0_fpm)
    cat_trend = trend(vz_cat0_fpm)
    if high_is_cat:
        a_align = (cat_trend == a_high_cmd) + (pl_trend == a_low_cmd)
        b_align = (cat_trend == b_high_cmd) + (pl_trend == b_low_cmd)
    else:
        a_align = (pl_trend == a_high_cmd) + (cat_trend == a_low_cmd)
        b_align = (pl_trend == b_high_cmd) + (cat_trend == b_low_cmd)
    use_a = (a_align > b_align) or (a_align == b_align and default_policy.startswith('High climbs'))
    if high_is_cat:
        return ((-1, +1) if use_a else (+1, -1))
    else:
        return ((+1, -1) if use_a else (-1, +1))

def _min_sep_ft_for_band(max_fl: int, policy: str = 'FL290STEP') -> int:
    return 1000 if max_fl < 290 else 2000

def sample_altitudes_and_h0(rng, h0_mean=250.0, h0_sd=100.0, h0_lo=0.0, h0_hi=2000.0, policy='FL290STEP'):
    FL_pl_base  = int(rng.integers(150, 301))
    cap = float(_min_sep_ft_for_band(FL_pl_base, policy=policy))
    h0 = float(np.clip(rng.normal(h0_mean, h0_sd), max(h0_lo, 0.0), min(h0_hi, cap)))
    diff_FL = max(1, int(round(h0 / 100.0)))
    sign = 1 if rng.uniform() < 0.5 else -1
    FL_cat_cand = FL_pl_base + sign * diff_FL
    if not (150 <= FL_cat_cand <= 300):
        FL_cat_cand = FL_pl_base - sign * diff_FL
        FL_cat_cand = min(300, max(150, FL_cat_cand))
    return int(FL_pl_base), int(FL_cat_cand), float(h0)

def sample_tgo_with_trigger(rng, scenario, tgo_geom, FL_pl, FL_cat, cap_s=60.0):
    base = {'Head-on': (25.0, 5.0), 'Crossing': (22.0, 6.0), 'Overtaking': (30.0, 8.0)}
    mu, sd = base.get(scenario, (25.0, 6.0))
    if FL_pl >= 250 and FL_cat >= 250: mu += 2.0
    lo, hi = 12.0, min(tgo_geom if tgo_geom is not None else cap_s, cap_s)
    if hi <= lo: return float(max(8.0, min(tgo_geom or 30.0, cap_s)))
    return float(np.clip(rng.normal(mu, sd), lo, hi))

def sample_pilot_response_cat(rng):
    u = rng.uniform()
    if u < 0.70:
        delay = max(0.0, rng.normal(4.5, 1.0))
        accel = max(0.05, rng.normal(0.25, 0.03))
    else:
        delay = max(0.0, rng.normal(8.5, 1.5))
        accel = max(0.05, rng.normal(0.10, 0.02))
    return float(delay), float(accel)

def init_state():
    if 'df' not in st.session_state: st.session_state['df'] = None
    if 'has_results' not in st.session_state: st.session_state['has_results'] = False
init_state()

st.title("ACAS/TCAS — Monte Carlo with per-run plotter & single-run panel")

with st.sidebar:
    st.header("Global settings")
    alim_ft = st.number_input("ALIM (ft)", value=600.0, step=50.0)
    baseline = st.selectbox("Baseline for RR scaling",
                            ["IDEAL 1500 fpm (ACASA 2002)", "STANDARD 1500 fpm (EUROCONTROL 2018)"])
    dt = st.number_input("Time step (s)", value=0.1, step=0.1, format="%.1f")
    resp_thr = st.number_input("Meaningful response threshold (fpm)", value=300.0, step=50.0)

st.subheader("Single-run (post-RA) — trajectory + Δh at CPA")
c1, c2, c3 = st.columns(3)
with c1:
    spot_h0 = st.number_input("Initial vertical miss h0 (ft)", value=250.0, step=25.0, min_value=0.0)
with c2:
    t_cpa_spot = st.number_input("t_go to CPA (s)", value=45.0, step=1.0, min_value=6.0)
with c3:
    spot_dt = st.number_input("Time step, dt (s)", value=dt, step=0.05, min_value=0.01, format="%.2f")
c4, c5, c6 = st.columns(3)
with c4:
    cat_delay = st.number_input("CAT delay (s)", value=5.0, step=0.5, min_value=0.0)
with c5:
    cat_accel = st.number_input("CAT accel (g)", value=0.25, step=0.01, min_value=0.01, format="%.2f")
with c6:
    cat_vs = st.number_input("CAT target VS (fpm)", value=1500.0, step=100.0, min_value=200.0)
c7, c8 = st.columns(2)
with c7:
    cat_cap = st.number_input("CAT cap (fpm)", value=2000.0, step=100.0, min_value=500.0)
with c8:
    st.caption("Sense: higher climbs, lower descends (single-run assumes CAT is above).")
if st.button("Run single simulation"):
    sense_pl_spot, sense_cat_spot = -1, +1
    times_spot, vs_pl_spot = vs_time_series(t_cpa_spot, spot_dt, PL_DELAY_S, PL_ACCEL_G, PL_VS_FPM,
                                            sense=sense_pl_spot, cap_fpm=PL_VS_CAP, vs0_fpm=0.0)
    _,         vs_ca_spot = vs_time_series(t_cpa_spot, spot_dt, cat_delay,      cat_accel,      cat_vs,
                                            sense=sense_cat_spot, cap_fpm=cat_cap, vs0_fpm=0.0)
    z_pl_spot = integrate_altitude_from_vs(times_spot, vs_pl_spot, 0.0)
    z_ca_spot = integrate_altitude_from_vs(times_spot, vs_ca_spot, spot_h0)
    miss_cpa  = float(abs(z_ca_spot[-1] - z_pl_spot[-1]))
    fig, ax = plt.subplots(figsize=(7,4))
    ax.plot(times_spot, z_pl_spot, label="PL alt (ft) — descend")
    ax.plot(times_spot, z_ca_spot, label="CAT alt (ft) — climb")
    ax.fill_between(times_spot, z_pl_spot - alim_ft, z_pl_spot + alim_ft, alpha=0.1, label=f"±ALIM ({alim_ft:.0f} ft)")
    ax.axhline(0, ls="--", lw=1, alpha=0.6)
    ax.set_xlabel("Time since RA trigger (s)"); ax.set_ylabel("Relative altitude (ft)")
    ax.set_title(f"Single-run altitude traces — Δh@CPA = {miss_cpa:.0f} ft")
    ax.legend(); ax.grid(True, alpha=0.3)
    st.pyplot(fig)
    st.markdown(f"**Δh at CPA** = **{miss_cpa:.0f} ft** — ALIM breach @ CPA: **{'Yes' if miss_cpa < alim_ft else 'No'}**.")

st.markdown('---')
st.header("Batch Monte Carlo")
with st.form("batch_form", clear_on_submit=False):
    n_runs = st.number_input("Number of runs", min_value=1, max_value=100000, value=1000, step=100)
    seed   = st.number_input("Random seed", value=26, step=1)
    scenario = st.selectbox("Scenario", ["Head-on", "Crossing", "Overtaking", "Custom"])
    r_min = st.number_input("Initial range min (NM)", value=4.0, step=0.5, min_value=0.5)
    r_max = st.number_input("Initial range max (NM)", value=8.0, step=0.5, min_value=1.0)
    if scenario == "Custom":
        hdg1_min = st.number_input("PL heading min (deg)", value=0.0, step=5.0)
        hdg1_max = st.number_input("PL heading max (deg)", value=360.0, step=5.0)
        hdg2_min = st.number_input("CAT heading min (deg)", value=0.0, step=5.0)
        hdg2_max = st.number_input("CAT heading max (deg)", value=360.0, step=5.0)
        rel_min = rel_max = None
    else:
        hdg1_min = 0.0; hdg1_max = 360.0
        if scenario == "Head-on":   rel_min, rel_max = 150.0, 210.0
        elif scenario == "Crossing": rel_min, rel_max = 60.0, 120.0
        else:                        rel_min, rel_max = 0.0, 30.0

    init_mode_pl  = st.selectbox("PL initial trajectory", ["Mostly level","Mixed","Aggressive"], index=0)
    init_mode_cat = st.selectbox("CAT initial trajectory", ["Mostly level","Mixed","Aggressive"], index=0)
    default_policy = st.selectbox("Default divergence policy when trends conflict",
                                  ["High climbs, low descends","High descends, low climbs"], index=0)
    use_distrib = st.checkbox("CAT response: use mixture distributions (recommended)", value=True)
    submitted = st.form_submit_button("Run batch")

if submitted:
    rng = np.random.default_rng(int(seed))
    data = []
    for k in range(int(n_runs)):
        FL_pl, FL_cat, h0 = sample_altitudes_and_h0(rng, 250.0, 100.0, 100.0, 500.0)
        PL_TAS = ias_to_tas(PL_IAS_KT, FL_pl * 100.0)
        CAT_TAS = float(rng.uniform(420.0, 470.0))
        if scenario == "Custom":
            h1 = rng.uniform(hdg1_min, hdg1_max); h2 = rng.uniform(hdg2_min, hdg2_max)
        else:
            h1, h2 = sample_headings(rng, scenario, hdg1_min, hdg1_max, rel_min, rel_max)
        r0 = float(rng.uniform(min(r_min, r_max), max(r_min, r_max)))
        vcl = relative_closure_kt(PL_TAS, h1, CAT_TAS, h2)
        tgo_geom = time_to_go_from_geometry(r0, vcl)
        tgo = sample_tgo_with_trigger(rng, scenario, tgo_geom, FL_pl, FL_cat, cap_s=60.0)
        pl_td_k = PL_DELAY_S; pl_ag_k = PL_ACCEL_G
        if use_distrib:
            cat_td_k, cat_ag_k = sample_pilot_response_cat(rng)
        else:
            cat_td_k, cat_ag_k = 5.0, 0.25

        vz_pl0  = sample_initial_vs(rng, mode=init_mode_pl)
        vz_cat0 = sample_initial_vs(rng, mode=init_mode_cat)
        cat_above = (FL_cat > FL_pl) if (FL_cat != FL_pl) else (rng.uniform() < 0.5)
        sense_pl, sense_ca = choose_sense_from_trend(cat_above, vz_pl0, vz_cat0, 200.0, default_policy)

        times, vs_pl = vs_time_series(tgo, dt, pl_td_k, pl_ag_k, PL_VS_FPM, sense=sense_pl, cap_fpm=PL_VS_CAP, vs0_fpm=vz_pl0)
        _,     vs_ca = vs_time_series(tgo, dt, cat_td_k, cat_ag_k, 1500.0,    sense=sense_ca, cap_fpm=2000.0,    vs0_fpm=vz_cat0)
        z_pl = integrate_altitude_from_vs(times, vs_pl, 0.0)
        z_ca = integrate_altitude_from_vs(times, vs_ca, h0 if cat_above else -h0)
        miss_cpa = float(abs(z_ca[-1] - z_pl[-1]))

        # Simple surrogate event
        t_mid_idx = int(0.6 * len(times))
        pred_mid = abs((z_ca[t_mid_idx] - z_pl[t_mid_idx]) + (vs_pl[t_mid_idx] - vs_ca[t_mid_idx]) * ((tgo - times[t_mid_idx])/60.0))
        event = "STRENGTHEN" if pred_mid < (alim_ft - ALIM_MARGIN_FT) else "NONE"

        dh_pl = delta_h_piecewise(tgo, pl_td_k, pl_ag_k, PL_VS_FPM)
        dh_base = baseline_dh_ft(tgo, mode=baseline)
        ratio = (dh_base / dh_pl) if dh_pl > 1e-6 else float('nan')
        unres_rr = 1.1 * ratio

        data.append({
            "run": k+1, "scenario": scenario, "FL_PL": FL_pl, "FL_CAT": FL_cat,
            "PL_TAS": PL_TAS, "CAT_TAS": CAT_TAS, "PLhdg": h1, "CAThdg": h2,
            "R0NM": r0, "closurekt": vcl, "tgos": tgo, "h0ft": h0,
            "plDelay": pl_td_k, "plAccel_g": pl_ag_k, "catDelay": cat_td_k, "catAccel_g": cat_ag_k,
            "missCPAft": miss_cpa, "unresolvedRRpct": unres_rr, "eventtype": event
        })
    df = pd.DataFrame(data)
    st.session_state['df'] = df
    st.session_state['has_results'] = True
    st.success(f"Completed {len(df)} runs.")

if st.session_state.get('has_results') and st.session_state.get('df') is not None:
    df = st.session_state['df']
    st.subheader("Preview of results")
    st.dataframe(df.head(100), use_container_width=True)

    n = len(df)
    k_str = int((df['eventtype']=="STRENGTHEN").sum())
    p_str = k_str/n if n else 0.0
    k1, k2, k3 = st.columns(3)
    k1.metric("P(Strengthen)", f"{100*p_str:,.2f}%")
    k2.metric("Mean RR", f"{df['unresolvedRRpct'].mean():.3f}%")
    k3.metric("Mean Δh@CPA (ft)", f"{df['missCPAft'].mean():,.0f}")

    st.markdown("### Plot a specific run")
    pick_mode = st.selectbox("Pick run:", ["By run id", "First STRENGTHEN"])
    if pick_mode == "First STRENGTHEN" and (df['eventtype']=="STRENGTHEN").any():
        row = df[df['eventtype']=="STRENGTHEN"].iloc[0]
    else:
        rid = st.number_input("Run id", min_value=int(df['run'].min()), max_value=int(df['run'].max()), value=int(df['run'].min()))
        row = df[df['run']==rid].iloc[0]

    plot_dt = st.number_input("Plot dt (s)", value=0.1, step=0.05, min_value=0.01, format="%.2f")
    vz0_pl_assumed  = st.number_input("Assumed PL initial VS (fpm)", value=0.0, step=50.0)
    vz0_cat_assumed = st.number_input("Assumed CAT initial VS (fpm)", value=0.0, step=50.0)

    tgo   = float(row["tgos"]); h0 = float(row["h0ft"]); FL_pl=int(row["FL_PL"]); FL_cat=int(row["FL_CAT"])
    cat_above = (FL_cat > FL_pl) if (FL_cat != FL_pl) else True
    sense_pl  = -1 if cat_above else +1
    sense_cat = +1 if cat_above else -1
    cat_td = float(row.get("catDelay", 5.0)); cat_ag = float(row.get("catAccel_g", 0.25))
    times, vs_pl = vs_time_series(tgo, plot_dt, PL_DELAY_S, PL_ACCEL_G, PL_VS_FPM, sense=sense_pl, cap_fpm=PL_VS_CAP, vs0_fpm=vz0_pl_assumed)
    _,     vs_ca = vs_time_series(tgo, plot_dt, cat_td,      cat_ag,      1500.0,  sense=sense_cat, cap_fpm=2000.0, vs0_fpm=vz0_cat_assumed)
    z_pl = integrate_altitude_from_vs(times, vs_pl, 0.0)
    z_ca = integrate_altitude_from_vs(times, vs_ca, h0 if cat_above else -h0)
    miss_cpa = float(abs(z_ca[-1] - z_pl[-1]))
    fig, ax = plt.subplots(figsize=(7,4))
    ax.plot(times, z_pl, label=f"PL alt (ft) — {'climb' if sense_pl>0 else 'descend'}")
    ax.plot(times, z_ca, label=f"CAT alt (ft) — {'climb' if sense_cat>0 else 'descend'}")
    ax.axhline(0, ls="--", lw=1, alpha=0.6)
    ax.set_xlabel("Time since RA trigger (s)"); ax.set_ylabel("Relative altitude (ft)")
    ax.set_title(f"Run {int(row['run'])} — {row['eventtype']} — Δh@CPA={miss_cpa:.0f} ft")
    ax.legend(); ax.grid(True, alpha=0.3)
    st.pyplot(fig)

    st.subheader("Download batch data")
    csv_buf = io.BytesIO()
    csv_buf.write(df.to_csv(index=False).encode("utf-8"))
    csv_buf.seek(0)
    st.download_button(label="Download CSV", data=csv_buf, file_name="tcas_batch_results.csv", mime="text/csv")




































