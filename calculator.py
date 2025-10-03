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
    CAT_CAP_INIT_FPM,
    CAT_CAP_STRENGTH_FPM,
    CAT_INIT_VS_FPM,
    CAT_STRENGTH_FPM,
    PL_ACCEL_G,
    PL_IAS_KT,
    PL_VS_CAP_FPM,
    PL_VS_FPM,
    ias_to_tas,
    integrate_altitude_from_vs,
    run_batch,
    time_to_go_from_geometry,
    vs_time_series,
)

ALIM_CHOICES = [
    ("FL50–FL100 (350 ft)", 350.0),
    ("FL100–FL200 (400 ft)", 400.0),
    ("FL200–FL420 (600 ft)", 600.0),
]

# ------------------------------- Streamlit UI -------------------------------

st.set_page_config(page_title="ACAS/TCAS v7.1 Monte Carlo", layout="wide")
st.title("ACAS/TCAS v7.1‑aligned Monte Carlo")

with st.sidebar:
    st.header("Simulation Controls")

    with st.expander("Scenario Geometry", expanded=True):
        scenario = st.selectbox(
            "Scenario",
            ["Head-on", "Crossing", "Overtaking", "Custom"],
            help="Select the encounter geometry used when sampling Monte Carlo runs."
        )
        seed = st.number_input(
            "Random seed",
            value=26,
            step=1,
            help="Use a fixed seed to reproduce a Monte Carlo batch exactly."
        )
        r0_min = st.number_input(
            "Initial range min (NM)",
            value=4.0,
            step=0.5,
            help="Lower bound of the initial lateral separation sampled for each run."
        )
        r0_max = st.number_input(
            "Initial range max (NM)",
            value=8.0,
            step=0.5,
            help="Upper bound of the initial lateral separation sampled for each run."
        )
        force_cat_ias = st.checkbox(
            "Force CAT IAS to 250 kt",
            value=False,
            help="When enabled the intruder TAS is computed from a fixed 250 kt IAS regardless of altitude."
        )
        use_custom_tgo = st.checkbox(
            "Custom time-to-CPA window",
            value=False,
            help="Enable to select bespoke minimum/maximum RA look-ahead times (bounded to 15–35 s)."
        )
        if use_custom_tgo:
            tgo_minmax = st.slider(
                "Time to CPA range (s)",
                min_value=15.0,
                max_value=35.0,
                value=(22.0, 28.0),
                step=0.5,
                help="Minimum and maximum t_go bounds used when sampling encounters. The implied mean is clamped to 24–26 s."
            )
        else:
            tgo_minmax = (None, None)
        if scenario == "Custom":
            hdg1_min = st.number_input("PL heading min (deg)", value=0.0, step=5.0,
                                       help="Minimum heading for the protected aircraft when sampling custom runs.")
            hdg1_max = st.number_input("PL heading max (deg)", value=360.0, step=5.0,
                                       help="Maximum heading for the protected aircraft when sampling custom runs.")
            hdg2_min = st.number_input("CAT heading min (deg)", value=0.0, step=5.0,
                                       help="Minimum heading for the intruder when sampling custom runs.")
            hdg2_max = st.number_input("CAT heading max (deg)", value=360.0, step=5.0,
                                       help="Maximum heading for the intruder when sampling custom runs.")
        else:
            hdg1_min = 0.0
            hdg1_max = 360.0
            hdg2_min = 0.0
            hdg2_max = 360.0

    with st.expander("Response Behaviour", expanded=True):
        aggressiveness = st.slider(
            "Initial-trajectory aggressiveness",
            0.0,
            1.0,
            0.30,
            0.05,
            help="0 = level-off context; 1 = aggressive climb/descend mix."
        )
        jitter = st.checkbox(
            "Jitter non-compliance priors (±30%)",
            value=True,
            help="Randomly perturb the prior probabilities each batch to reflect modelling uncertainty."
        )
        apfd_option = st.selectbox(
            "AP/FD configuration",
            [
                "Custom share",
                "Mixed global traffic (10%)",
                "Airbus-centric (30%)",
            ],
            help="Choose how autopilot/flight-director usage is represented in the Monte Carlo runs."
        )

        apfd_mode = "custom"
        apfd_share = 0.35

        if apfd_option == "Custom share":
            apfd_share = st.slider(
                "AP/FD share",
                0.0,
                1.0,
                0.35,
                0.05,
                help="Share of crews flying via AP/FD, which lowers delay and slightly boosts acceleration."
            )
        elif apfd_option.startswith("Mixed"):
            apfd_mode = "mixed"
            apfd_share = 0.10
            st.caption("Mixed global traffic fixes AP/FD usage at 10% with deterministic CAT kinematics for that share.")
        else:
            apfd_mode = "airbus"
            apfd_share = 0.30
            st.caption("Airbus-centric traffic fixes AP/FD usage at 30% with deterministic CAT kinematics for that share.")

        if not isinstance(apfd_mode, str) or not apfd_mode:
            apfd_mode = "custom"
        try:
            apfd_share_value = float(apfd_share)
        except (TypeError, ValueError):
            apfd_share_value = 0.35
        st.markdown("**Non-compliance priors** (updated baseline)")
        p_opp = st.number_input(
            "P(opposite-sense)",
            value=0.020,
            step=0.001,
            format="%.3f",
            help="Probability that the intruder flies the opposite sense of the commanded RA."
        )
        p_ta = st.number_input(
            "P(no-response / TA-only)",
            value=0.075,
            step=0.001,
            format="%.3f",
            help="Probability of no vertical response beyond traffic advisory behaviour."
        )
        p_weak = st.number_input(
            "P(weak-compliance)",
            value=0.425,
            step=0.005,
            format="%.3f",
            help="Probability that the intruder responds in the commanded sense but with insufficient vertical speed."
        )


    with st.expander("ALIM Settings", expanded=True):
        alim_labels = [label for label, _ in ALIM_CHOICES]
        alim_index_default = 1  # Default to FL100–FL200 band (400 ft)
        alim_selection_label = st.radio(
            "User-selected ALIM band",
            options=alim_labels,
            index=min(alim_index_default, len(alim_labels) - 1),
            help=(
                "Choose the TCAS v7.1 ALIM applied when scoring batch runs. "
                "All runs in a batch will use the selected threshold."
            ),
        )
        selected_alim_ft = dict(ALIM_CHOICES)[alim_selection_label]
        st.caption(f"Using ±{selected_alim_ft:.0f} ft separation threshold for ALIM breaches.")


tabs = st.tabs(["Single‑run demo", "Batch Monte Carlo"])

with tabs[0]:
    SINGLE_FL = 220
    single_alt_ft = SINGLE_FL * 100.0
    pl_tas = ias_to_tas(PL_IAS_KT, single_alt_ft)

    c1, c2, c3 = st.columns(3)
    with c1:
        initial_range_nm = st.number_input("Initial range (NM)", value=6.0, min_value=1.0, step=0.5,
                                           help="Starting separation between aircraft in the single-run illustration.")
    with c2:
        dt = st.number_input("Time step dt (s)", value=0.1, step=0.05, min_value=0.01, format="%.2f",
                              help="Resolution of the time history used when integrating the trajectories.")
    with c3:
        cat_sense_label = st.selectbox("CAT response", ["Descend", "Level", "Climb"], index=0,
                                       help="Choose how the intruder responds to the resolution advisory in this demo.")

    cat_delay_user = st.number_input("CAT pilot delay (s)", value=5.0, min_value=0.0, step=0.5, format="%.1f",
                                     help="Reaction delay applied to the intruder for the single run only.")
    cat_accel_user = st.number_input("CAT acceleration (g)", value=0.25, min_value=0.01, step=0.01, format="%.2f",
                                     help="Vertical acceleration towards the commanded vertical speed.")
    cat_vs_user = st.number_input("CAT vertical speed target (fpm)", value=1500.0, min_value=0.0, step=100.0,
                                  help="Commanded vertical speed magnitude used during the intruder response.")
    cat_ias_user = st.number_input("CAT IAS (kt)", value=250.0, min_value=50.0, step=10.0,
                                   help="Indicated airspeed for the intruder, converted to TAS for closure calculations.")

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
            delta_pl = float(z_pl[-1] - z_pl[0])
            delta_cat = float(z_cat[-1] - z_cat[0])
            residual_risk = abs(delta_cat) / max(abs(delta_pl), 1e-3) * 0.011

            st.markdown(
                f"**Scenario**: Head-on at FL{SINGLE_FL}, PL IAS {PL_IAS_KT:.0f} kt (TAS {pl_tas:.1f} kt), "
                f"CAT IAS {cat_ias_user:.0f} kt (TAS {cat_tas:.1f} kt)."
            )

            c_metric1, c_metric2, c_metric3, c_metric4 = st.columns(4)
            c_metric1.metric("Time to CPA", f"{t_cpa:.1f} s")
            c_metric2.metric("Range rate", f"{closure_kt:.1f} kt")
            c_metric3.metric("Δh at CPA", f"{delta_h_cpa:.0f} ft")
            c_metric4.metric("Residual risk", f"{100 * residual_risk:,.3f}%")
            st.caption("These metrics describe the immediate geometry of the hand-crafted encounter.")

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
            st.caption("Altitude traces show how the protected aircraft and intruder diverge after the RA.")

with tabs[1]:
    n_runs = st.number_input("Number of runs", min_value=10, max_value=200000, value=5000, step=500,
                             help="Monte Carlo sample size; larger values reduce noise at higher compute cost.")

    if st.button("Run batch"):
        df = run_batch(
            runs=int(n_runs), seed=int(seed), scenario=scenario,
            r0_min_nm=float(r0_min), r0_max_nm=float(r0_max),
            aggressiveness=float(aggressiveness),
            p_opp=float(p_opp), p_ta=float(p_ta), p_weak=float(p_weak),
            jitter_priors=bool(jitter), apfd_share=float(apfd_share_value),
            use_delay_mixture=True,
            dt=0.1,
            hdg1_min=float(hdg1_min), hdg1_max=float(hdg1_max),
            hdg2_min=float(hdg2_min), hdg2_max=float(hdg2_max),
            alim_override_ft=float(selected_alim_ft),
            use_custom_tgo=bool(use_custom_tgo),
            tgo_min_s=(float(tgo_minmax[0]) if tgo_minmax[0] is not None else None),
            tgo_max_s=(float(tgo_minmax[1]) if tgo_minmax[1] is not None else None),
            apfd_mode=apfd_mode,
            force_cat_ias_250=bool(force_cat_ias),
        )
        st.session_state['df'] = df

    if 'df' in st.session_state and st.session_state['df'] is not None:
        df = st.session_state['df']
        st.success(f"Completed {len(df)} runs.")
        st.caption(f"ALIM applied: {alim_selection_label} (±{selected_alim_ft:.0f} ft).")
        report_alim_outside = st.checkbox(
            "Report ALIM @ CPA outside ±1 s window",
            value=False,
            help="When enabled the displayed ALIM@CPA metric ignores breaches occurring within ±1 s of CPA."
        )
        total_runs = len(df)
        safe_total = max(total_runs, 1)
        c1, c2, c3, c4, c5, c6 = st.columns(6)
        p_rev = (df['eventtype'] == "REVERSE").sum() / safe_total
        p_str = (df['eventtype'] == "STRENGTHEN").sum() / safe_total
        p_none = (df['eventtype'] == "NONE").sum() / safe_total
        p_alim_any = (df['margin_min_ft'] < 0.0).sum() / safe_total
        p_alim_cpa = df['alim_breach_cpa'].sum() / safe_total
        p_alim_margin = df['alim_breach_margin'].sum() / safe_total
        p_alim_outside = df['alim_breach_outside'].sum() / safe_total
        c1.metric("P(Reversal)", f"{100 * p_rev:,.2f}%")
        c2.metric("P(Strengthen)", f"{100 * p_str:,.2f}%")
        c3.metric("P(None)", f"{100 * p_none:,.2f}%")
        c4.metric("P(ALIM Any)", f"{100 * p_alim_any:,.2f}%")
        if report_alim_outside:
            c5.metric("P(ALIM @ CPA outside ±1 s)", f"{100 * p_alim_outside:,.2f}%")
        else:
            c5.metric("P(ALIM @ CPA)", f"{100 * p_alim_cpa:,.2f}%")
        c6.metric("P(ALIM within ±1 s)", f"{100 * p_alim_margin:,.2f}%")
        st.caption("Percentages describe RA outcomes alongside ALIM breaches at CPA, within ±1 s, and anywhere in the run.")
        near_25 = (df['sep_cpa_ft'] - df['ALIM_ft']).abs() <= 25.0
        near_50 = (df['sep_cpa_ft'] - df['ALIM_ft']).abs() <= 50.0
        near_100 = (df['sep_cpa_ft'] - df['ALIM_ft']).abs() <= 100.0
        near_25_rate = near_25.sum() / safe_total
        near_50_rate = near_50.sum() / safe_total
        near_100_rate = near_100.sum() / safe_total
        nm1, nm2, nm3 = st.columns(3)
        nm1.metric("CPA within ±25 ft of ALIM", f"{100 * near_25_rate:,.2f}%")
        nm2.metric("CPA within ±50 ft of ALIM", f"{100 * near_50_rate:,.2f}%")
        nm3.metric("CPA within ±100 ft of ALIM", f"{100 * near_100_rate:,.2f}%")
        mean_rr = float(df['residual_risk'].mean()) if total_runs else 0.0
        p95_rr = float(df['residual_risk'].quantile(0.95)) if total_runs else 0.0
        rr1, rr2 = st.columns(2)
        rr1.metric("Residual risk (mean)", f"{100 * mean_rr:,.3f}%")
        rr2.metric("Residual risk (95th pct)", f"{100 * p95_rr:,.3f}%")
        if (df['eventtype'] == "REVERSE").any():
            reasons = (
                df.loc[df['eventtype'] == "REVERSE", 'reverse_reason']
                .fillna('Unclassified')
                .value_counts(normalize=True)
            )
            reason_text = ", ".join(f"{k}: {100 * v:,.1f}%" for k, v in reasons.items())
            st.caption(f"Reversal drivers — {reason_text}.")
        else:
            st.caption("No reversals observed in this batch.")

        st.markdown("### Preview")
        reversal_only = st.checkbox(
            "Show reversals only",
            value=False,
            help="Filter the preview/table to reversal runs only."
        )
        preview_df = df[df['eventtype'] == "REVERSE"] if reversal_only else df
        st.dataframe(preview_df.head(200), use_container_width=True)

        st.markdown("### Batch insights")

        event_order = ["NONE", "STRENGTHEN", "REVERSE"]
        color_map = {"NONE": "#6baed6", "STRENGTHEN": "#fd8d3c", "REVERSE": "#d7301f"}

        counts = df['eventtype'].value_counts().reindex(event_order, fill_value=0)
        shares = counts / len(df)

        fig, axes = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True)

        axes[0].bar(event_order, counts[event_order], color=[color_map[e] for e in event_order])
        for idx, evt in enumerate(event_order):
            axes[0].text(
                idx,
                counts[evt] + 0.01 * len(df),
                f"{100 * shares.get(evt, 0.0):.1f}%",
                ha='center',
                va='bottom',
                fontsize=10,
            )
        axes[0].set_ylabel("Runs")
        axes[0].set_title("Resolution advisory outcomes")
        axes[0].grid(axis='y', alpha=0.2)

        grouped_h0 = [df.loc[df['eventtype'] == evt, 'h0ft'].dropna() for evt in event_order]
        axes[1].boxplot(
            grouped_h0,
            labels=["None", "Strengthen", "Reverse"],
            patch_artist=True,
            boxprops=dict(facecolor="#f0f0f0"),
        )
        axes[1].set_ylabel("Initial vertical separation h₀ (ft)")
        axes[1].set_title("Initial geometry by RA outcome")
        axes[1].grid(alpha=0.2)

        st.pyplot(fig)
        st.caption(
            "Left: outcome mix across the batch. Right: how initial vertical separation trends with reversal/strengthen events."
        )

        margin = df['sep_cpa_ft'] - df['ALIM_ft']
        breach_mask = margin < 0.0
        fig2, ax2 = plt.subplots(figsize=(8, 5))

        for evt in event_order:
            mask_evt = df['eventtype'] == evt
            if not mask_evt.any():
                continue
            ax2.scatter(
                df.loc[mask_evt, 'tgos'],
                margin[mask_evt],
                alpha=0.45,
                s=30,
                label=f"{evt.title()} runs" if evt == "NONE" else evt.title(),
                color=color_map[evt],
            )

        if breach_mask.any():
            ax2.scatter(
                df.loc[breach_mask, 'tgos'],
                margin[breach_mask],
                facecolors='none',
                edgecolors='#000000',
                s=60,
                linewidths=0.8,
                label='ALIM breach @ CPA',
            )

        ax2.axhline(0.0, color='k', linestyle='--', linewidth=1, alpha=0.7)
        ax2.set_xlabel("Time to go at RA issue (s)")
        ax2.set_ylabel("CPA separation − ALIM (ft)")
        ax2.set_title("CPA margin relative to ALIM")
        ax2.grid(alpha=0.25)
        ax2.legend(loc='best')

        st.pyplot(fig2)

        breach_rate = 100.0 * breach_mask.mean()
        st.caption(
            "Points below the dashed line represent CPA separations that fail to clear ALIM. "
            f"{breach_rate:.2f}% of sampled runs breached ALIM at CPA; highlighted markers show where they occur."
        )

        with st.expander("Inspect an individual run", expanded=False):
            rid = st.number_input(
                "Run id",
                min_value=int(df['run'].min()),
                max_value=int(df['run'].max()),
                value=int(df['run'].min()),
                help="Select a run to inspect the time history corresponding to that Monte Carlo sample.",
            )
            row = df[df['run'] == rid].iloc[0]

            # Rebuild kinematics from stored fields (display only; uses assumed initial VS = 0)
            tgo = float(row["tgos"])
            h0 = float(row["h0ft"])
            FL_pl = int(row["FL_PL"])
            FL_cat = int(row["FL_CAT"])
            cat_above = (FL_cat > FL_pl) if (FL_cat != FL_pl) else True
            sense_pl = int(row["sensePL"])
            sense_cat = int(row["senseCAT_exec"])
            cat_td = float(row["catDelay"])
            cat_ag = float(row["catAccel_g"])
            cat_vs = float(row["catVS_cmd"])
            cat_cap = float(row["catCap_cmd"])
            pl_delay = float(row["plDelay"])
            alim_ft = float(row['ALIM_ft'])

            times, vs_pl = vs_time_series(tgo, 0.1, pl_delay, PL_ACCEL_G, PL_VS_FPM, sense=sense_pl, cap_fpm=PL_VS_CAP_FPM, vs0_fpm=0.0)
            _, vs_ca = vs_time_series(tgo, 0.1, cat_td, cat_ag, cat_vs, sense=sense_cat, cap_fpm=cat_cap, vs0_fpm=0.0)
            z_pl = integrate_altitude_from_vs(times, vs_pl, 0.0)
            z_ca = integrate_altitude_from_vs(times, vs_ca, h0 if cat_above else -h0)
            miss_cpa = float(abs(z_ca[-1] - z_pl[-1]))

            fig_run, ax_run = plt.subplots(figsize=(8,4))
            ax_run.plot(times, z_pl, label=f"PL ({'climb' if sense_pl>0 else 'descend'})")
            ax_run.plot(times, z_ca, label=f"CAT ({'climb' if sense_cat>0 else 'descend'}) [{row['CAT_mode']}]")
            ax_run.fill_between(times, z_pl - alim_ft, z_pl + alim_ft, alpha=0.08, label=f"±ALIM at FL{FL_pl}")
            ax_run.axhline(0, ls='--', lw=1, alpha=0.6)
            if not pd.isna(row['t_second_issue']):
                ax_run.axvline(float(row['t_second_issue']), ls=':', lw=1, alpha=0.7, label='2nd‑phase issue')
            ax_run.set_xlabel("Time since RA trigger (s)")
            ax_run.set_ylabel("Relative altitude (ft)")
            ax_run.set_title(f"Run {int(row['run'])} — {row['eventtype']} — Δh@CPA={miss_cpa:.0f} ft")
            ax_run.legend(); ax_run.grid(True, alpha=0.3)
            st.pyplot(fig_run)
            st.caption("Shaded band visualises ±ALIM around the protected aircraft during the sampled run.")


        # Download
        buf = io.BytesIO()
        buf.write(df.to_csv(index=False).encode('utf-8'))
        buf.seek(0)
        st.download_button("Download CSV", buf, file_name="tcas_batch_results_v71.csv", mime="text/csv")

 






































