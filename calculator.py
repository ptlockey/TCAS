# ACAS/TCAS Monte Carlo — optimal-sense selection (max Δh@CPA) + non-compliance modes
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
ALIM_MARGIN_FT = 100.0

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

def baseline_dh_ft(t_cpa_s, mode='IDEAL'):
    if mode.startswith('IDEAL'):
        return delta_h_piecewise(t_cpa_s, t_delay_s=1.0, a_g=0.25, v_f_fpm=1500.0)
    else:
        return delta_h_piecewise(t_cpa_s, t_delay_s=5.0, a_g=0.25, v_f_fpm=1500.0)

def sample_initial_vs(rng, mode='Mixed', level_sd=50.0, climb_mean=1000.0, climb_sd=200.0, descend_mean=-1000.0, descend_sd=200.0):
    if mode == 'Mostly level': probs = (0.8, 0.1, 0.1)
    elif mode == 'Aggressive': probs = (0.4, 0.3, 0.3)
    else: probs = (0.6, 0.2, 0.2)
    u = rng.uniform()
    if u < probs[0]: return float(rng.normal(0.0, level_sd))
    elif u < probs[0]+probs[1]: return float(rng.normal(climb_mean, climb_sd))
    else: return float(rng.normal(descend_mean, descend_sd))

# ---------- Optimal-sense selector (maximise Δh@CPA under nominal CAT response) ----------
def simulate_miss_for_senses(tgo, dt, h0, cat_above, sense_pl, sense_cat,
                             pl_delay, pl_accel, pl_vs, pl_cap, pl_vs0,
                             cat_delay_nom, cat_accel_nom, cat_vs, cat_cap, cat_vs0):
    times, vs_pl = vs_time_series(tgo, dt, pl_delay, pl_accel, pl_vs, sense=sense_pl, cap_fpm=pl_cap, vs0_fpm=pl_vs0)
    _,     vs_ca = vs_time_series(tgo, dt, cat_delay_nom, cat_accel_nom, cat_vs, sense=sense_cat, cap_fpm=cat_cap, vs0_fpm=cat_vs0)
    z_pl = integrate_altitude_from_vs(times, vs_pl, 0.0)
    z_ca = integrate_altitude_from_vs(times, vs_ca, h0 if cat_above else -h0)
    miss_cpa = float(abs(z_ca[-1] - z_pl[-1]))
    return miss_cpa

def choose_optimal_sense(tgo, dt, h0, cat_above, pl_vs0, cat_vs0,
                         cat_delay_nom=5.0, cat_accel_nom=0.25, cat_vs=1500.0, cat_cap=2000.0):
    miss_updown = simulate_miss_for_senses(tgo, dt, h0, cat_above,
                                           sense_pl=+1, sense_cat=-1,
                                           pl_delay=PL_DELAY_S, pl_accel=PL_ACCEL_G, pl_vs=PL_VS_FPM, pl_cap=PL_VS_CAP, pl_vs0=pl_vs0,
                                           cat_delay_nom=cat_delay_nom, cat_accel_nom=cat_accel_nom, cat_vs=cat_vs, cat_cap=cat_cap, cat_vs0=cat_vs0)
    miss_downup = simulate_miss_for_senses(tgo, dt, h0, cat_above,
                                           sense_pl=-1, sense_cat=+1,
                                           pl_delay=PL_DELAY_S, pl_accel=PL_ACCEL_G, pl_vs=PL_VS_FPM, pl_cap=PL_VS_CAP, pl_vs0=pl_vs0,
                                           cat_delay_nom=cat_delay_nom, cat_accel_nom=cat_accel_nom, cat_vs=cat_vs, cat_cap=cat_cap, cat_vs0=cat_vs0)
    if miss_updown > miss_downup:
        return (+1, -1), miss_updown, miss_downup
    else:
        return (-1, +1), miss_downup, miss_updown

# -------------------- Non-compliance modes (applied to CAT after sense selection) --------------------
def apply_non_compliance_to_cat(rng, sense_cat, delay_s, accel_g, vs_fpm, cap_fpm,
                                p_opp=0.01, p_taonly=0.01, p_weak=0.05, jitter=False):
    """Return (mode_label, sense_cat_final, delay_s_final, accel_g_final, vs_fpm_final, cap_fpm_final)."""
    if jitter:
        p_opp   = max(0.0, min(1.0, p_opp   * rng.uniform(0.5, 1.5)))
        p_taonly= max(0.0, min(1.0, p_taonly* rng.uniform(0.5, 1.5)))
        p_weak  = max(0.0, min(1.0, p_weak  * rng.uniform(0.5, 1.5)))
    u = rng.uniform()
    if u < p_opp:
        return ("opposite-sense", -sense_cat, delay_s, accel_g, vs_fpm, cap_fpm)
    u -= p_opp
    if u < p_taonly:
        return ("no-response", sense_cat, delay_s, 0.0, 0.0, 0.0)
    u -= p_taonly
    if u < p_weak:
        return ("weak-compliance", sense_cat, delay_s, max(0.03, 0.6*accel_g), 0.6*vs_fpm, 0.6*cap_fpm)
    return ("compliant", sense_cat, delay_s, accel_g, vs_fpm, cap_fpm)

# -------------------- Streamlit UI --------------------
def init_state():
    if 'df' not in st.session_state: st.session_state['df'] = None
    if 'has_results' not in st.session_state: st.session_state['has_results'] = False
init_state()

st.title("ACAS/TCAS — Optimal-sense + Non-compliance (after sense selection)")

with st.sidebar:
    st.header("Global settings")
    alim_ft = st.number_input("ALIM (ft)", value=600.0, step=50.0)
    baseline = st.selectbox("Baseline for RR scaling",
                            ["IDEAL 1500 fpm (ACASA 2002)", "STANDARD 1500 fpm (EUROCONTROL 2018)"])
    dt = st.number_input("Time step (s)", value=0.1, step=0.1, format="%.1f")

# Single-run (optimal sense)
st.subheader("Single-run — optimal sense + Δh@CPA")
c1, c2, c3 = st.columns(3)
with c1:
    spot_h0 = st.number_input("Initial vertical miss h0 (ft)", value=250.0, step=25.0, min_value=0.0)
with c2:
    t_cpa_spot = st.number_input("t_go to CPA (s)", value=45.0, step=1.0, min_value=6.0)
with c3:
    spot_dt = st.number_input("Time step, dt (s)", value=dt, step=0.05, min_value=0.01, format="%.2f")

c4, c5, c6 = st.columns(3)
with c4:
    cat_delay_nom = st.number_input("CAT nominal delay (s) for sense choice", value=5.0, step=0.5, min_value=0.0)
with c5:
    cat_accel_nom = st.number_input("CAT nominal accel (g) for sense choice", value=0.25, step=0.01, min_value=0.01, format="%.2f")
with c6:
    cat_vs_nom = st.number_input("CAT target VS (fpm)", value=1500.0, step=100.0, min_value=200.0)
c7, c8 = st.columns(2)
with c7:
    cat_cap_nom = st.number_input("CAT cap (fpm)", value=2000.0, step=100.0, min_value=500.0)
with c8:
    st.caption("Sense chosen to maximise Δh@CPA under nominal CAT response; variability/non-compliance can follow.")

if st.button("Run single simulation (optimal sense)"):
    cat_above = True
    vz0_pl = 0.0; vz0_cat = 0.0
    (sense_pl_spot, sense_cat_spot), miss_opt, miss_alt = choose_optimal_sense(
        t_cpa_spot, spot_dt, spot_h0, cat_above, vz0_pl, vz0_cat,
        cat_delay_nom, cat_accel_nom, cat_vs_nom, cat_cap_nom
    )
    times_spot, vs_pl_spot = vs_time_series(t_cpa_spot, spot_dt, PL_DELAY_S, PL_ACCEL_G, PL_VS_FPM,
                                            sense=sense_pl_spot, cap_fpm=PL_VS_CAP, vs0_fpm=vz0_pl)
    _,         vs_ca_spot = vs_time_series(t_cpa_spot, spot_dt, cat_delay_nom,      cat_accel_nom,      cat_vs_nom,
                                            sense=sense_cat_spot, cap_fpm=cat_cap_nom, vs0_fpm=vz0_cat)
    z_pl_spot = integrate_altitude_from_vs(times_spot, vs_pl_spot, 0.0)
    z_ca_spot = integrate_altitude_from_vs(times_spot, vs_ca_spot, spot_h0)
    miss_cpa  = float(abs(z_ca_spot[-1] - z_pl_spot[-1]))

    fig, ax = plt.subplots(figsize=(7,4))
    ax.plot(times_spot, z_pl_spot, label=f"PL alt — {'climb' if sense_pl_spot>0 else 'descend'}")
    ax.plot(times_spot, z_ca_spot, label=f"CAT alt — {'climb' if sense_cat_spot>0 else 'descend'}")
    ax.fill_between(times_spot, z_pl_spot - alim_ft, z_pl_spot + alim_ft, alpha=0.1, label=f"±ALIM ({alim_ft:.0f} ft)")
    ax.axhline(0, ls="--", lw=1, alpha=0.6)
    ax.set_xlabel("Time since RA trigger (s)"); ax.set_ylabel("Relative altitude (ft)")
    ax.set_title(f"Single-run (optimal sense) — Δh@CPA = {miss_cpa:.0f} ft")
    ax.legend(); ax.grid(True, alpha=0.3)
    st.pyplot(fig)

    st.markdown(f"**Chosen sense**: PL = {'climb' if sense_pl_spot>0 else 'descend'}, CAT = {'climb' if sense_cat_spot>0 else 'descend'}")
    st.markdown(f"**Δh@CPA (chosen)** = {miss_opt:.0f} ft; **Δh@CPA (alternative)** = {miss_alt:.0f} ft")

# Batch with non-compliance
st.markdown('---')
st.header("Batch Monte Carlo (optimal sense; then CAT variability & non-compliance)")
with st.form("batch_form", clear_on_submit=False):
    n_runs = st.number_input("Number of runs", min_value=1, max_value=200000, value=2000, step=200)
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

    st.markdown("**Non-compliance probabilities (CAT, after sense selection)**")
    p_opp  = st.number_input("P(opposite-sense)", value=0.01, step=0.005, min_value=0.0, max_value=1.0, format="%.3f")
    p_ta   = st.number_input("P(no-response / TA-only)", value=0.01, step=0.005, min_value=0.0, max_value=1.0, format="%.3f")
    p_weak = st.number_input("P(weak-compliance)", value=0.05, step=0.005, min_value=0.0, max_value=1.0, format="%.3f")
    jitter_priors = st.checkbox("Jitter probabilities ±50% per run", value=True)

    use_distrib = st.checkbox("Apply CAT *compliant* variability (delay/accel mixture) if not non-compliant", value=True)
    submitted = st.form_submit_button("Run batch")

if submitted:
    rng = np.random.default_rng(int(seed))
    data = []
    for k in range(int(n_runs)):
        # Altitudes & separation
        FL_pl = int(rng.integers(150, 301))
        cap_h0 = 1000 if FL_pl < 290 else 2000
        h0 = float(np.clip(rng.normal(250.0, 100.0), 100.0, cap_h0))
        diff_FL = max(1, int(round(h0/100.0)))
        sign = 1 if rng.uniform()<0.5 else -1
        FL_cat = FL_pl + sign*diff_FL
        FL_cat = min(300, max(150, FL_cat))

        # Speeds/headings
        PL_TAS = ias_to_tas(PL_IAS_KT, FL_pl*100.0)
        CAT_TAS= float(rng.uniform(420.0, 470.0))
        if scenario == "Custom":
            h1 = rng.uniform(hdg1_min, hdg1_max); h2 = rng.uniform(hdg2_min, hdg2_max)
        else:
            rel = rng.uniform(rel_min, rel_max)
            sgn = 1 if rng.uniform()<0.5 else -1
            h1 = rng.uniform(0.0, 360.0); h2 = (h1 + sgn*rel) % 360.0
        r0 = float(rng.uniform(min(r_min, r_max), max(r_min, r_max)))
        vcl = relative_closure_kt(PL_TAS, h1, CAT_TAS, h2)
        tgo_geom = time_to_go_from_geometry(r0, vcl)
        mu, sd = (25.0, 5.0) if scenario=="Head-on" else ((22.0,6.0) if scenario=="Crossing" else (30.0,8.0))
        lo, hi = 12.0, min(tgo_geom if tgo_geom is not None else 60.0, 60.0)
        tgo = float(np.clip(rng.normal(mu, sd), lo, hi))

        # Initial VS (pre-RA)
        vz_pl0  = sample_initial_vs(rng, init_mode_pl)
        vz_cat0 = sample_initial_vs(rng, init_mode_cat)

        # Who's higher?
        cat_above = (FL_cat > FL_pl) if (FL_cat != FL_pl) else (rng.uniform() < 0.5)

        # Choose optimal sense under nominal CAT response
        (sense_pl, sense_ca), miss_nominal, miss_alternative = choose_optimal_sense(
            tgo, dt, h0, cat_above, vz_pl0, vz_cat0,
            cat_delay_nom=5.0, cat_accel_nom=0.25, cat_vs=1500.0, cat_cap=2000.0
        )

        # Apply non-compliance to CAT
        mode, sense_cat_eff, cat_delay_eff, cat_accel_eff, cat_vs_eff, cat_cap_eff = apply_non_compliance_to_cat(
            rng, sense_ca, delay_s=5.0, accel_g=0.25, vs_fpm=1500.0, cap_fpm=2000.0,
            p_opp=p_opp, p_taonly=p_ta, p_weak=p_weak, jitter=jitter_priors
        )

        # If compliant, optionally add variability
        if mode == "compliant" and use_distrib:
            if rng.uniform() < 0.7:
                cat_delay_eff = max(0.0, rng.normal(4.5, 1.0))
                cat_accel_eff = max(0.05, rng.normal(0.25, 0.03))
            else:
                cat_delay_eff = max(0.0, rng.normal(8.5, 1.5))
                cat_accel_eff = max(0.05, rng.normal(0.10, 0.02))

        # Build kinematics
        times, vs_pl = vs_time_series(tgo, dt, PL_DELAY_S, PL_ACCEL_G, PL_VS_FPM, sense=sense_pl, cap_fpm=PL_VS_CAP, vs0_fpm=vz_pl0)
        _,     vs_ca = vs_time_series(tgo, dt, cat_delay_eff, cat_accel_eff, cat_vs_eff, sense=sense_cat_eff, cap_fpm=cat_cap_eff, vs0_fpm=vz_cat0)
        z_pl = integrate_altitude_from_vs(times, vs_pl, 0.0)
        z_ca = integrate_altitude_from_vs(times, vs_ca, h0 if cat_above else -h0)
        miss_cpa = float(abs(z_ca[-1] - z_pl[-1]))

        # Surrogate Strengthen check (mid-time predictive)
        t_mid_idx = int(0.6 * len(times))
        pred_mid = abs((z_ca[t_mid_idx] - z_pl[t_mid_idx]) + (vs_pl[t_mid_idx] - vs_ca[t_mid_idx]) * ((tgo - times[t_mid_idx])/60.0))
        event = "STRENGTHEN" if pred_mid < (DEFAULT_ALIM_FT - ALIM_MARGIN_FT) else "NONE"

        # RR scaling (simple)
        dh_pl = delta_h_piecewise(tgo, PL_DELAY_S, PL_ACCEL_G, PL_VS_FPM)
        dh_base = baseline_dh_ft(tgo, mode='IDEAL')
        ratio = (dh_base / dh_pl) if dh_pl > 1e-6 else float('nan')
        unres_rr = 1.1 * ratio

        data.append({
            "run": k+1, "scenario": scenario, "FL_PL": FL_pl, "FL_CAT": FL_cat,
            "PL_TAS": PL_TAS, "CAT_TAS": CAT_TAS, "PLhdg": h1, "CAThdg": h2,
            "R0NM": r0, "closurekt": vcl, "tgos": tgo, "h0ft": h0,
            "sensePL": sense_pl, "senseCAT_chosen": sense_ca, "CAT_mode": mode, "senseCAT_exec": sense_cat_eff,
            "plDelay": PL_DELAY_S, "plAccel_g": PL_ACCEL_G, "catDelay": cat_delay_eff, "catAccel_g": cat_accel_eff,
            "catVS_cmd": cat_vs_eff, "catCap_cmd": cat_cap_eff,
            "missCPAft": miss_cpa, "eventtype": event, "unresolvedRRpct": unres_rr,
            "miss_nominal_for_choice": miss_nominal, "miss_alt_for_choice": miss_alternative
        })

    df = pd.DataFrame(data)
    st.session_state['df'] = df
    st.session_state['has_results'] = True
    st.success(f"Completed {len(df)} runs.")

# Results + selectable-run plot
if st.session_state.get('has_results') and st.session_state.get('df') is not None:
    df = st.session_state['df']
    st.subheader("Preview of results")
    st.dataframe(df.head(100), use_container_width=True)

    n = len(df)
    p_str = (df['eventtype']=="STRENGTHEN").mean()
    k1, k2, k3 = st.columns(3)
    k1.metric("P(Strengthen)", f"{100*p_str:,.2f}%")
    k2.metric("Mean RR", f"{df['unresolvedRRpct'].mean():.3f}%")
    k3.metric("Mean Δh@CPA (ft)", f"{df['missCPAft'].mean():,.0f}")

    st.markdown("### Plot a specific run")
    rid = st.number_input("Run id", min_value=int(df['run'].min()), max_value=int(df['run'].max()), value=int(df['run'].min()))
    row = df[df['run']==rid].iloc[0]

    plot_dt = st.number_input("Plot dt (s)", value=0.1, step=0.05, min_value=0.01, format="%.2f")
    vz0_pl_assumed  = st.number_input("Assumed PL initial VS (fpm)", value=0.0, step=50.0)
    vz0_cat_assumed = st.number_input("Assumed CAT initial VS (fpm)", value=0.0, step=50.0)

    tgo   = float(row["tgos"]); h0 = float(row["h0ft"]); FL_pl=int(row["FL_PL"]); FL_cat=int(row["FL_CAT"])
    cat_above = (FL_cat > FL_pl) if (FL_cat != FL_pl) else True
    sense_pl  = int(row.get("sensePL", -1 if cat_above else +1))
    sense_cat = int(row.get("senseCAT_exec", +1 if cat_above else -1))
    cat_td = float(row.get("catDelay", 5.0)); cat_ag = float(row.get("catAccel_g", 0.25))
    cat_vs = float(row.get("catVS_cmd", 1500.0)); cat_cap = float(row.get("catCap_cmd", 2000.0))

    times, vs_pl = vs_time_series(tgo, plot_dt, PL_DELAY_S, PL_ACCEL_G, PL_VS_FPM, sense=sense_pl, cap_fpm=PL_VS_CAP, vs0_fpm=vz0_pl_assumed)
    _,     vs_ca = vs_time_series(tgo, plot_dt, cat_td,      cat_ag,      cat_vs,  sense=sense_cat, cap_fpm=cat_cap, vs0_fpm=vz0_cat_assumed)
    z_pl = integrate_altitude_from_vs(times, vs_pl, 0.0)
    z_ca = integrate_altitude_from_vs(times, vs_ca, h0 if cat_above else -h0)
    miss_cpa = float(abs(z_ca[-1] - z_pl[-1]))

    fig, ax = plt.subplots(figsize=(7,4))
    ax.plot(times, z_pl, label=f"PL alt — {'climb' if sense_pl>0 else 'descend'}")
    ax.plot(times, z_ca, label=f"CAT alt — {'climb' if sense_cat>0 else 'descend'} ({row['CAT_mode']})")
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





































