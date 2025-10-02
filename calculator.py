#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ACAS/TCAS v7.1-aligned Monte Carlo — Streamlit app
Implements the requested amendments on top of your functioning codebase:
 A) Two‑phase RA execution (STRENGTHEN ≈ ±2500 fpm or REVERSE) with ~1 s decision latency
 B) Altitude‑dependent ALIM (≈300–700 ft bands)
 C) RA timing window t_go ∈ [15, 35] s (scenario‑shaped mean)
 D/H) Retuned non‑compliance priors (opp=1%, TA‑only≈0.3%, weak≈30%) + per‑run jitter
 E) Level‑off context at aggressiveness=0.0
 F/I) CAT reaction time mixture (70% fast ~4.5 s / 30% slow ~8.5 s) and AP/FD share (reduces delay, nudges accel)
 G) Conservative reversal trigger: thin prediction AND sense mismatch
 J) Method‑B‑like compliance label per run (comp_label)
    + Expanded logging: ALIM used, 2nd‑phase issue time.
"""
from __future__ import annotations

import io

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import streamlit as st

from simulation import (
    PL_ACCEL_G,
    PL_IAS_KT,
    PL_VS_CAP_FPM,
    PL_VS_FPM,
    integrate_altitude_from_vs,
    ias_to_tas,
    run_batch,
    time_to_go_from_geometry,
    vs_time_series,
)

# ------------------------------- Streamlit UI -------------------------------

st.set_page_config(page_title="ACAS/TCAS v7.1 Monte Carlo", layout="wide")
st.title("ACAS/TCAS v7.1‑aligned Monte Carlo")

with st.sidebar:
    st.header("Global")
    scenario = st.selectbox("Scenario", ["Head-on","Crossing","Overtaking","Custom"])
    seed = st.number_input("Random seed", value=26, step=1)
    aggressiveness = st.slider("Initial‑trajectory aggressiveness", 0.0, 1.0, 0.30, 0.05,
                               help="0 = level‑off context; 1 = aggressive climb/descend mix")
    jitter = st.checkbox("Jitter non‑compliance priors (±30%)", value=True)
    apfd_share = st.slider("AP/FD share", 0.0, 1.0, 0.25, 0.05,
                           help="Share with AP/FD TCAS behaviour (reduces delay, nudges accel)")
    st.markdown("**Non‑compliance priors** (v7.1 tuned)")
    p_opp = st.number_input("P(opposite‑sense)", value=0.010, step=0.001, format="%.3f")
    p_ta  = st.number_input("P(no‑response / TA‑only)", value=0.003, step=0.001, format="%.3f")
    p_weak= st.number_input("P(weak‑compliance)", value=0.300, step=0.010, format="%.3f")
    st.divider()
    st.markdown("**Geometry**")
    r0_min = st.number_input("Initial range min (NM)", value=4.0, step=0.5)
    r0_max = st.number_input("Initial range max (NM)", value=8.0, step=0.5)
    if scenario=="Custom":
        hdg1_min = st.number_input("PL heading min (deg)", value=0.0, step=5.0)
        hdg1_max = st.number_input("PL heading max (deg)", value=360.0, step=5.0)
        hdg2_min = st.number_input("CAT heading min (deg)", value=0.0, step=5.0)
        hdg2_max = st.number_input("CAT heading max (deg)", value=360.0, step=5.0)
    else:
        hdg1_min=0.0; hdg1_max=360.0; hdg2_min=0.0; hdg2_max=360.0

# Single‑run panel
st.subheader("Single‑run demo")

SINGLE_FL = 220
single_alt_ft = SINGLE_FL * 100.0
pl_tas = ias_to_tas(PL_IAS_KT, single_alt_ft)

c1, c2, c3 = st.columns(3)
with c1:
    initial_range_nm = st.number_input("Initial range (NM)", value=6.0, min_value=1.0, step=0.5)
with c2:
    dt = st.number_input("Time step dt (s)", value=0.1, step=0.05, min_value=0.01, format="%.2f")
with c3:
    cat_sense_label = st.selectbox("CAT response", ["Descend", "Level", "Climb"], index=0)

cat_delay_user = st.number_input("CAT pilot delay (s)", value=5.0, min_value=0.0, step=0.5, format="%.1f")
cat_accel_user = st.number_input("CAT acceleration (g)", value=0.25, min_value=0.01, step=0.01, format="%.2f")
cat_vs_user = st.number_input("CAT vertical speed target (fpm)", value=1500.0, min_value=0.0, step=100.0)
cat_ias_user = st.number_input("CAT IAS (kt)", value=250.0, min_value=50.0, step=10.0)

if st.button("Run single case"):
    cat_tas = ias_to_tas(float(cat_ias_user), single_alt_ft)
    closure_kt = pl_tas + cat_tas

    t_cpa = time_to_go_from_geometry(float(initial_range_nm), closure_kt)

    if t_cpa is None:
        st.warning("Closure rate is zero or negative; CPA cannot be determined.")
    else:
        sense_pl = +1  # PL always climbs
        if cat_sense_label == "Climb":
            sense_cat = +1
        elif cat_sense_label == "Descend":
            sense_cat = -1
        else:
            sense_cat = 0

        times, vs_pl = vs_time_series(t_cpa, float(dt), 0.9, PL_ACCEL_G, PL_VS_FPM,
                                      sense=sense_pl, cap_fpm=PL_VS_CAP_FPM, vs0_fpm=0.0)

        if sense_cat == 0 or cat_vs_user <= 1e-6:
            times_cat = np.arange(0.0, t_cpa + 1e-9, float(dt))
            vs_cat = np.zeros_like(times_cat)
        else:
            _, vs_cat = vs_time_series(t_cpa, float(dt), float(cat_delay_user), float(cat_accel_user),
                                       float(cat_vs_user), sense=sense_cat, cap_fpm=float(cat_vs_user),
                                       vs0_fpm=0.0)
            times_cat = times

        z_pl = integrate_altitude_from_vs(times, vs_pl, 0.0)
        z_cat = integrate_altitude_from_vs(times_cat, vs_cat, 0.0)

        # ensure equal length for plotting when CAT is level
        if len(times_cat) != len(times):
            vs_cat = np.zeros_like(times)
            z_cat = np.zeros_like(times)

        delta_h_cpa = float(z_pl[-1] - z_cat[-1])
        miss_cpa = abs(delta_h_cpa)

        st.markdown(
            f"**Scenario**: Head-on at FL{SINGLE_FL}, PL IAS {PL_IAS_KT:.0f} kt (TAS {pl_tas:.1f} kt), "
            f"CAT IAS {cat_ias_user:.0f} kt (TAS {cat_tas:.1f} kt)."
        )

        c_metric1, c_metric2, c_metric3 = st.columns(3)
        c_metric1.metric("Time to CPA", f"{t_cpa:.1f} s")
        c_metric2.metric("Range rate", f"{closure_kt:.1f} kt")
        c_metric3.metric("Δh at CPA", f"{delta_h_cpa:.0f} ft")

        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(times, z_pl, label="PL (climb)")
        label_cat = "CAT (level)" if sense_cat == 0 else f"CAT ({'climb' if sense_cat>0 else 'descend'})"
        ax.plot(times, z_cat, label=label_cat)
        ax.axhline(0.0, ls='--', lw=1, alpha=0.6)
        ax.set_xlabel("Time since RA trigger (s)")
        ax.set_ylabel("Altitude change (ft)")
        ax.set_title(f"Single run — Δh@CPA = {miss_cpa:.0f} ft")
        ax.legend()
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)

# Batch Monte Carlo
st.subheader("Batch Monte Carlo")
n_runs = st.number_input("Number of runs", min_value=10, max_value=200000, value=5000, step=500)

if st.button("Run batch"):
    df = run_batch(
        runs=int(n_runs), seed=int(seed), scenario=scenario,
        r0_min_nm=float(r0_min), r0_max_nm=float(r0_max),
        aggressiveness=float(aggressiveness),
        p_opp=float(p_opp), p_ta=float(p_ta), p_weak=float(p_weak),
        jitter_priors=bool(jitter), apfd_share=float(apfd_share),
        use_delay_mixture=True,
        dt=0.1,
        hdg1_min=float(hdg1_min), hdg1_max=float(hdg1_max),
        hdg2_min=float(hdg2_min), hdg2_max=float(hdg2_max)
    )
    st.session_state['df'] = df

if 'df' in st.session_state and st.session_state['df'] is not None:
    df = st.session_state['df']
    st.success(f"Completed {len(df)} runs.")
    c1,c2,c3,c4,c5 = st.columns(5)
    p_rev = (df['eventtype']=="REVERSE").mean()
    p_str = (df['eventtype']=="STRENGTHEN").mean()
    p_none= (df['eventtype']=="NONE").mean()
    p_alim_any = (df['minsepft'] < df['ALIM_ft']).mean()
    p_alim_cpa = (df['sep_cpa_ft'] < df['ALIM_ft']).mean()
    c1.metric("P(Reversal)", f"{100*p_rev:,.2f}%")
    c2.metric("P(Strengthen)", f"{100*p_str:,.2f}%")
    c3.metric("P(None)", f"{100*p_none:,.2f}%")
    c4.metric("P(ALIM Any)", f"{100*p_alim_any:,.2f}%")
    c5.metric("P(ALIM @ CPA)", f"{100*p_alim_cpa:,.2f}%")

    st.markdown("### Preview")
    st.dataframe(df.head(200), use_container_width=True)

    st.markdown("### Plot a specific run")
    rid = st.number_input("Run id", min_value=int(df['run'].min()), max_value=int(df['run'].max()), value=int(df['run'].min()))
    row = df[df['run']==rid].iloc[0]

    # Rebuild kinematics from stored fields (display only; uses assumed initial VS = 0)
    tgo   = float(row["tgos"]); h0 = float(row["h0ft"]); FL_pl=int(row["FL_PL"]); FL_cat=int(row["FL_CAT"])
    cat_above = (FL_cat > FL_pl) if (FL_cat != FL_pl) else True
    sense_pl  = int(row["sensePL"]); sense_cat = int(row["senseCAT_exec"])
    cat_td = float(row["catDelay"]); cat_ag = float(row["catAccel_g"])
    cat_vs = float(row["catVS_cmd"]); cat_cap = float(row["catCap_cmd"])
    pl_delay = float(row["plDelay"]); alim_ft = float(row['ALIM_ft'])

    times, vs_pl = vs_time_series(tgo, 0.1, pl_delay, PL_ACCEL_G, PL_VS_FPM, sense=sense_pl, cap_fpm=PL_VS_CAP_FPM, vs0_fpm=0.0)
    _,     vs_ca = vs_time_series(tgo, 0.1, cat_td,      cat_ag,      cat_vs,  sense=sense_cat, cap_fpm=cat_cap, vs0_fpm=0.0)
    z_pl = integrate_altitude_from_vs(times, vs_pl, 0.0)
    z_ca = integrate_altitude_from_vs(times, vs_ca, h0 if cat_above else -h0)
    miss_cpa = float(abs(z_ca[-1] - z_pl[-1]))

    fig, ax = plt.subplots(figsize=(8,4))
    ax.plot(times, z_pl, label=f"PL ({'climb' if sense_pl>0 else 'descend'})")
    ax.plot(times, z_ca, label=f"CAT ({'climb' if sense_cat>0 else 'descend'}) [{row['CAT_mode']}]")
    ax.fill_between(times, z_pl - alim_ft, z_pl + alim_ft, alpha=0.08, label=f"±ALIM at FL{FL_pl}")
    ax.axhline(0, ls='--', lw=1, alpha=0.6)
    if not pd.isna(row['t_second_issue']):
        ax.axvline(float(row['t_second_issue']), ls=':', lw=1, alpha=0.7, label='2nd‑phase issue')
    ax.set_xlabel("Time since RA trigger (s)")
    ax.set_ylabel("Relative altitude (ft)")
    ax.set_title(f"Run {int(row['run'])} — {row['eventtype']} — Δh@CPA={miss_cpa:.0f} ft")
    ax.legend(); ax.grid(True, alpha=0.3)
    st.pyplot(fig)

    # Download
    buf = io.BytesIO()
    buf.write(df.to_csv(index=False).encode('utf-8'))
    buf.seek(0)
    st.download_button("Download CSV", buf, file_name="tcas_batch_results_v71.csv", mime="text/csv")

 






































