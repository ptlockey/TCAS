#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TCAS Encounter Analyser — Streamlit app
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

import json
from bisect import bisect_left
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import streamlit as st

from inspector_utils import get_second_issue_time

from preview_filters import (
    LOWEST_SEPARATION_WINDOW_FT,
    build_preview_dataframe,
)

from simulation import (
    CAT_CAP_INIT_FPM,
    CAT_CAP_STRENGTH_FPM,
    CAT_INIT_VS_FPM,
    CAT_STRENGTH_FPM,
    PL_ACCEL_G,
    PL_DELAY_MEAN_S,
    PL_IAS_KT,
    PL_VS_CAP_FPM,
    PL_VS_FPM,
    ias_to_tas,
    integrate_altitude_from_vs,
    OppositeSenseBand,
    derive_single_run_geometry,
    compute_residual_risk,
    run_batch,
    sanitize_tgo_bounds,
    decode_time_history,
    extend_history_with_pretrigger,
    vs_time_series,
)

ALIM_CHOICES = [
    ("FL50–FL100 (350 ft)", 350.0),
    ("FL100–FL200 (400 ft)", 400.0),
    ("FL200–FL420 (600 ft)", 600.0),
]

APFD_DEFAULT_MODE = "custom"
APFD_DEFAULT_SHARE = 0.35
APFD_CUSTOM_SHARE_KEY = "apfd_custom_share"
APFD_OPTION_KEY = "apfd_option"
APFD_PRESET_MAP = {
    "Mixed global traffic (10%)": ("mixed", 0.10),
    "Airbus-centric (30%)": ("airbus", 0.30),
}


def sanitize_apfd_config(option: str, share_value: Optional[float]) -> Tuple[str, float]:
    """Return a valid AP/FD mode string and share bounded to [0, 1]."""

    if option in APFD_PRESET_MAP:
        return APFD_PRESET_MAP[option]

    # Default to the custom selection when the option is unrecognised.
    try:
        share = float(share_value if share_value is not None else APFD_DEFAULT_SHARE)
    except (TypeError, ValueError):
        share = APFD_DEFAULT_SHARE
    share = float(np.clip(share, 0.0, 1.0))
    return APFD_DEFAULT_MODE, share


def clamp_to_available_run(value: int, run_options: list[int]) -> int:
    """Snap ``value`` to the closest available run identifier."""

    if not run_options:
        raise ValueError("run_options must not be empty")

    value = int(value)
    if value <= run_options[0]:
        return run_options[0]
    if value >= run_options[-1]:
        return run_options[-1]

    idx = bisect_left(run_options, value)
    if idx < len(run_options) and run_options[idx] == value:
        return value

    prev_run = run_options[idx - 1]
    next_run = run_options[idx] if idx < len(run_options) else run_options[-1]
    if abs(prev_run - value) <= abs(next_run - value):
        return prev_run
    return next_run

# ------------------------------- Streamlit UI -------------------------------

st.set_page_config(page_title="TCAS Encounter Analyser", layout="wide")
st.title("TCAS Encounter Analyser")

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
            value=True,
            help="Enable to select bespoke minimum/maximum RA look-ahead times (bounded to 15–35 s)."
        )
        if use_custom_tgo:
            tgo_minmax = st.slider(
                "Time to CPA range (s)",
                min_value=15.0,
                max_value=35.0,
                value=(15.0, 35.0),
                step=0.5,
                help="Minimum and maximum t_go bounds used when sampling encounters. The implied mean is clamped to 24–26 s."
            )
            tgo_window = sanitize_tgo_bounds(tgo_minmax[0], tgo_minmax[1])
            st.caption(
                "When enabled the single-run demo uses the longest selected t_go as its CPA horizon and the batch sampler draws within that window."
            )
        else:
            tgo_minmax = (None, None)
            tgo_window = None
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
            0.0,
            0.05,
            help="0 = level-off context; 1 = aggressive climb/descend mix."
        )
        jitter = st.checkbox(
            "Jitter non-compliance priors (±30%)",
            value=True,
            help="Randomly perturb the prior probabilities each batch to reflect modelling uncertainty."
        )
        if APFD_CUSTOM_SHARE_KEY not in st.session_state:
            st.session_state[APFD_CUSTOM_SHARE_KEY] = APFD_DEFAULT_SHARE

        apfd_option = st.selectbox(
            "AP/FD configuration",
            [
                "Custom share",
                "Mixed global traffic (10%)",
                "Airbus-centric (30%)",
            ],
            key=APFD_OPTION_KEY,
            help="Choose how autopilot/flight-director usage is represented in the Monte Carlo runs."
        )

        apfd_raw_share = st.session_state.get(APFD_CUSTOM_SHARE_KEY, APFD_DEFAULT_SHARE)

        if apfd_option == "Custom share":
            apfd_raw_share = st.slider(
                "AP/FD share",
                0.0,
                1.0,
                value=float(apfd_raw_share),
                step=0.05,
                key=APFD_CUSTOM_SHARE_KEY,
                help="Share of crews flying via AP/FD, which lowers delay and slightly boosts acceleration."
            )
        elif apfd_option in APFD_PRESET_MAP:
            preset_mode, _ = APFD_PRESET_MAP[apfd_option]
            if preset_mode == "mixed":
                st.caption("Mixed global traffic fixes AP/FD usage at 10% with deterministic CAT kinematics for that share.")
            elif preset_mode == "airbus":
                st.caption("Airbus-centric traffic fixes AP/FD usage at 30% with deterministic CAT kinematics for that share.")

        apfd_mode, apfd_share = sanitize_apfd_config(apfd_option, apfd_raw_share)
        apfd_share_sanitized = float(apfd_share)
        st.markdown("**Non-compliance priors** (updated baseline)")
        p_opp_manual = st.number_input(
            "Manual P(opposite-sense)",
            value=0.020,
            step=0.001,
            format="%.3f",
            help="Probability that a manually flown intruder flies the opposite sense of the commanded RA.",
        )
        st.caption("AP/FD crews are modelled as always flying the commanded sense (P(opposite-sense) = 0).")
        p_opp_apfd = 0.0
        st.markdown("**Manual crew priors**")
        p_ta_manual = st.number_input(
            "Manual P(no-response / TA-only)",
            value=0.075,
            step=0.001,
            format="%.3f",
            help="Probability of no vertical response beyond traffic advisory behaviour for manual crews.",
        )
        p_weak_manual = st.number_input(
            "Manual P(weak-compliance)",
            value=0.425,
            step=0.005,
            format="%.3f",
            help="Probability that a manual intruder responds in the commanded sense but with insufficient vertical speed.",
        )
        st.markdown("**AP/FD crew priors**")
        p_ta_apfd = st.number_input(
            "AP/FD P(no-response / TA-only)",
            value=0.010,
            step=0.001,
            format="%.3f",
            help="Probability of no vertical response for automated crews. Defaults are much lower than manual rates.",
        )
        p_weak_apfd = st.number_input(
            "AP/FD P(weak-compliance)",
            value=0.120,
            step=0.005,
            format="%.3f",
            help="Probability that an AP/FD crew flies the commanded sense but with reduced authority.",
        )

        opp_bands = None
        use_alt_bands = st.checkbox(
            "Specify altitude-dependent opposite-sense overrides",
            value=False,
            help="Enable to load a JSON list of altitude bands with bespoke manual/AP/FD opposite-sense probabilities.",
        )
        if use_alt_bands:
            default_json = json.dumps(
                [
                    {"alt_min_ft": 0.0, "alt_max_ft": 10000.0, "manual": 0.02},
                    {"alt_min_ft": 10000.0, "alt_max_ft": 20000.0, "manual": 0.03},
                ],
                indent=2,
            )
            opp_band_json = st.text_area(
                "Altitude band configuration (JSON)",
                value=default_json,
                height=180,
                help=(
                    "Provide a JSON list of objects with keys 'alt_min_ft', 'alt_max_ft', and 'manual'. "
                    "Any AP/FD-specific values will be ignored because automation always follows the commanded sense."
                ),
            )
            try:
                raw = json.loads(opp_band_json)
                if isinstance(raw, dict):
                    raw = [raw]
                opp_bands = []
                for item in raw:
                    if not isinstance(item, dict):
                        raise ValueError("Each entry must be an object")
                    alt_min = float(item["alt_min_ft"])
                    alt_max = float(item["alt_max_ft"])
                    manual = float(item["manual"])
                    apfd_raw = item.get("apfd")
                    if apfd_raw not in (None, 0, 0.0, "0", "0.0"):
                        try:
                            apfd_val = float(apfd_raw)
                        except (TypeError, ValueError) as exc:  # noqa: PERF203
                            raise ValueError("AP/FD probabilities must be numeric when provided") from exc
                        if abs(apfd_val) > 1e-9:
                            st.info(
                                "AP/FD opposite-sense entries are clamped to zero; ignoring provided value."
                            )
                    opp_bands.append(
                        OppositeSenseBand(
                            alt_min_ft=alt_min,
                            alt_max_ft=alt_max,
                            manual_prob=manual,
                            apfd_prob=0.0,
                        )
                    )
            except Exception as exc:  # noqa: BLE001
                st.warning(f"Could not parse altitude bands: {exc}")
                opp_bands = None


    with st.expander("ALIM Settings", expanded=True):
        alim_labels = [label for label, _ in ALIM_CHOICES]
        alim_index_default = 2  # Default to FL200–FL420 band (600 ft)
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

        t_cpa, initial_range_effective = derive_single_run_geometry(
            float(initial_range_nm),
            closure_kt,
            bool(use_custom_tgo),
            tgo_window,
        )

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

            times, vs_pl = vs_time_series(t_cpa, float(dt), PL_DELAY_MEAN_S, PL_ACCEL_G, PL_VS_FPM,
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

            times_plot, z_pl_plot, z_cat_plot = extend_history_with_pretrigger(
                times,
                z_pl,
                z_cat,
                pl_vs0=float(vs_pl[0] if len(vs_pl) else 0.0),
                cat_vs0=float(vs_cat[0] if len(vs_cat) else 0.0),
            )

            # ensure equal length for plotting when CAT is level
            if len(times_cat) != len(times):
                vs_cat = np.zeros_like(times)
                z_cat = np.zeros_like(times)

            delta_h_cpa = float(z_pl[-1] - z_cat[-1])
            miss_cpa = abs(delta_h_cpa)
            delta_h_pl = float(z_pl[-1] - z_pl[0])
            delta_h_cat = float(z_cat[-1] - z_cat[0])
            residual_risk = compute_residual_risk(delta_h_pl, delta_h_cat)

            st.markdown(
                f"**Scenario**: Head-on at FL{SINGLE_FL}, PL IAS {PL_IAS_KT:.0f} kt (TAS {pl_tas:.1f} kt), "
                f"CAT IAS {cat_ias_user:.0f} kt (TAS {cat_tas:.1f} kt), initial range {initial_range_effective:.2f} NM."
            )

            c_metric1, c_metric2, c_metric3, c_metric4 = st.columns(4)
            c_metric1.metric("Time to CPA", f"{t_cpa:.1f} s")
            c_metric2.metric("Range rate", f"{closure_kt:.1f} kt")
            c_metric3.metric("Δh at CPA", f"{delta_h_cpa:.0f} ft")
            c_metric4.metric("Residual risk", f"{100 * residual_risk:,.3f}%")
            st.caption("These metrics describe the immediate geometry of the hand-crafted encounter.")

            fig, ax = plt.subplots(figsize=(8, 4))
            ax.plot(times_plot, z_pl_plot, label="PL (climb)")
            label_cat = "CAT (level)" if sense_cat == 0 else f"CAT ({'climb' if sense_cat>0 else 'descend'})"
            ax.plot(times_plot, z_cat_plot, label=label_cat)
            ax.axhline(0.0, ls='--', lw=1, alpha=0.6)
            ax.set_xlabel("Time since RA trigger (s)")
            ax.set_ylabel("Altitude change (ft)")
            ax.set_title(f"Single run — Δh@CPA = {miss_cpa:.0f} ft")
            ax.legend()
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)
            st.caption("Altitude traces show how the protected aircraft and intruder diverge after the RA.")

with tabs[1]:
    n_runs = st.number_input(
        "Number of runs",
        min_value=10,
        max_value=200000,
        value=1000,
        step=500,
        help="Monte Carlo sample size; baseline is 1,000 runs—larger values reduce noise at higher compute cost.",
    )

    if st.button("Run batch"):
        df = run_batch(
            runs=int(n_runs), seed=int(seed), scenario=scenario,
            r0_min_nm=float(r0_min), r0_max_nm=float(r0_max),
            aggressiveness=float(aggressiveness),
            p_opp=float(p_opp_manual),
            p_ta_manual=float(p_ta_manual),
            p_weak_manual=float(p_weak_manual),
            p_ta_apfd=float(p_ta_apfd),
            p_weak_apfd=float(p_weak_apfd),
            opp_sense_apfd=None if p_opp_apfd is None else float(p_opp_apfd),
            opp_sense_bands=opp_bands,
            jitter_priors=bool(jitter), apfd_share=apfd_share_sanitized,
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
        total_runs = len(df)
        safe_total = max(total_runs, 1)
        c1, c2, c3, c4, c5 = st.columns(5)
        p_rev = (df['eventtype'] == "REVERSE").sum() / safe_total
        p_str = (df['eventtype'] == "STRENGTHEN").sum() / safe_total
        p_none = (df['eventtype'] == "NONE").sum() / safe_total
        sep_reference = df['sep_cpa_ft']
        p_alim_cpa = df['alim_breach_cpa'].sum() / safe_total
        c1.metric("P(Reversal)", f"{100 * p_rev:,.2f}%")
        c2.metric("P(Strengthen)", f"{100 * p_str:,.2f}%")
        c3.metric("P(None)", f"{100 * p_none:,.2f}%")
        mean_cpa_sep = float(sep_reference.mean()) if total_runs else 0.0
        apfd_exec_share = float(df['CAT_is_APFD'].mean()) if total_runs else 0.0
        c4.metric("P(ALIM @ CPA)", f"{100 * p_alim_cpa:,.2f}%")
        c5.metric("AP/FD executions", f"{100 * apfd_exec_share:,.2f}%")
        st.caption(
            "Percentages describe RA outcomes alongside CPA ALIM breaches without exclusions."
            f" Mean miss @ CPA across the batch: {mean_cpa_sep:,.1f} ft."
            f" AP/FD executions observed: {100 * apfd_exec_share:,.2f}% (target {100 * apfd_share_sanitized:,.2f}%)."
        )

        breach_total = int(df['alim_breach_cpa'].sum())
        band25_count = int(df['alim_breach_cpa_band25'].sum())
        band50_count = int(df['alim_breach_cpa_band50'].sum())
        band100_count = int(df['alim_breach_cpa_band100'].sum())
        if breach_total:
            band_rows = []
            total_pct = 100.0 * breach_total / safe_total
            band_rows.append(
                {
                    "Threshold": f"≤ ALIM ({selected_alim_ft:.0f} ft)",
                    "Runs %": f"{total_pct:,.2f}%",
                    "Share of breaches": "100.00%",
                }
            )

            def band_entry(label: str, count: int, threshold_ft: float) -> None:
                band_rows.append(
                    {
                        "Threshold": f"≤ {threshold_ft:.0f} ft ({label})",
                        "Runs %": f"{100.0 * count / safe_total:,.2f}%",
                        "Share of breaches": f"{100.0 * count / breach_total:,.2f}%",
                    }
                )

            band_entry("ALIM−25", max(band25_count, 0), max(selected_alim_ft - 25.0, 0.0))
            band_entry("ALIM−50", max(band50_count, 0), max(selected_alim_ft - 50.0, 0.0))
            band_entry("ALIM−100", max(band100_count, 0), max(selected_alim_ft - 100.0, 0.0))

            band_df = pd.DataFrame(band_rows)
            st.markdown("**ALIM breach severity bands**")
            st.table(band_df)
            st.caption(
                "Bands describe increasingly stringent CPA thresholds relative to ALIM."
                " Shares are computed over the set of ALIM breaches."
            )
        else:
            st.info("No ALIM breaches were observed in this batch.")

        mean_rr = float(df['residual_risk'].mean()) if total_runs else 0.0
        p95_rr = float(df['residual_risk'].quantile(0.95)) if total_runs else 0.0
        rr1, rr2 = st.columns(2)
        rr1.metric("Residual risk (mean)", f"{100 * mean_rr:,.3f}%")
        rr2.metric("Residual risk (95th pct)", f"{100 * p95_rr:,.3f}%")
        if (df['eventtype'] == "REVERSE").any():
            detail_column = None
            for candidate in ("reverse_reason", "event_detail_final", "event_detail"):
                if candidate in df.columns:
                    detail_column = candidate
                    break

            if detail_column is not None:
                reasons = (
                    df.loc[df['eventtype'] == "REVERSE", detail_column]
                    .fillna('Unclassified')
                    .value_counts(normalize=True)
                )
                reason_text = ", ".join(f"{k}: {100 * v:,.1f}%" for k, v in reasons.items())
                st.caption(f"Reversal drivers — {reason_text}.")
            else:
                st.caption("Reversal details unavailable for this batch.")
        else:
            st.caption("No reversals observed in this batch.")

        st.markdown("### Preview")
        reversal_only = st.checkbox(
            "Show reversals only",
            value=False,
            help="Filter the preview/table to reversal runs only."
        )
        lowest_separation_only = st.checkbox(
            "Lowest separation heights",
            value=False,
            help=(
                "When enabled, restrict the preview to runs whose CPA separation sits within "
                f"{LOWEST_SEPARATION_WINDOW_FT:.0f} ft of the batch minimum, sorted from lowest to highest."
            )
        )

        preview_df = build_preview_dataframe(
            df,
            reversal_only=reversal_only,
            lowest_separation_only=lowest_separation_only,
        )

        if preview_df.empty:
            st.info(
                "No runs match the current preview filters. Clear one or more filters to explore the batch."
            )
        else:
            st.dataframe(preview_df.head(200), use_container_width=True)

            if lowest_separation_only and 'sep_cpa_ft' in df.columns:
                min_sep = float(df['sep_cpa_ft'].min())
                if not np.isnan(min_sep):
                    threshold = min_sep + LOWEST_SEPARATION_WINDOW_FT
                    st.caption(
                        "Previewing the tightest CPA separations: runs ≤ "
                        f"{threshold:,.0f} ft (minimum observed {min_sep:,.0f} ft)."
                    )
            elif reversal_only:
                st.caption("Preview limited to reversal runs; disable the filter to view the full batch.")
            else:
                st.caption("Showing the first 200 runs from the batch preview.")

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

        margin = sep_reference - df['ALIM_ft']
        breach_mask = df['alim_breach_cpa']
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
                label='ALIM breach (≤ ALIM)',
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
            if preview_df.empty:
                st.info("Run inspection is unavailable because no runs satisfy the current preview filters.")
            else:
                run_options = sorted(set(preview_df['run'].astype(int)))
                run_selection_key = "preview_run_selection"
                run_input_key = "preview_run_input"
                run_prev_key = "preview_run_prev"
                run_next_key = "preview_run_next"

                current_selection = st.session_state.get(run_selection_key, run_options[0])
                current_selection = clamp_to_available_run(current_selection, run_options)
                st.session_state[run_selection_key] = current_selection

                current_input_value = clamp_to_available_run(
                    st.session_state.get(run_input_key, current_selection),
                    run_options,
                )

                prev_col, input_col, next_col = st.columns([1, 3, 1])
                with prev_col:
                    prev_clicked = st.button(
                        "◀",
                        key=run_prev_key,
                        disabled=current_selection == run_options[0],
                        use_container_width=True,
                    )
                with next_col:
                    next_clicked = st.button(
                        "▶",
                        key=run_next_key,
                        disabled=current_selection == run_options[-1],
                        use_container_width=True,
                    )
                updated_selection = current_selection
                if prev_clicked:
                    idx = run_options.index(current_selection)
                    if idx > 0:
                        updated_selection = run_options[idx - 1]
                elif next_clicked:
                    idx = run_options.index(current_selection)
                    if idx < len(run_options) - 1:
                        updated_selection = run_options[idx + 1]
                else:
                    updated_selection = current_input_value

                st.session_state[run_selection_key] = updated_selection
                st.session_state[run_input_key] = updated_selection

                with input_col:
                    st.number_input(
                        "Run id",
                        min_value=int(run_options[0]),
                        max_value=int(run_options[-1]),
                        step=1,
                        key=run_input_key,
                        value=int(updated_selection),
                        help=(
                            "Step through runs that match the preview filters above."
                            " Adjust the filters to widen or narrow this subset."
                        ),
                    )

                rid = updated_selection
                row = preview_df[preview_df['run'] == rid].iloc[0]

                # Reconstruct manoeuvre histories using stored data when available.
                tgo = float(row["tgos"])
                h0 = float(row["h0ft"])
                FL_pl = int(row["FL_PL"])
                FL_cat = int(row["FL_CAT"])
                if "cat_above" in row.index and not pd.isna(row["cat_above"]):
                    cat_above = bool(row["cat_above"])
                else:
                    cat_above = (FL_cat > FL_pl) if (FL_cat != FL_pl) else True
                alim_ft = float(row['ALIM_ft'])

                sense_pl_final = int(row["sensePL_final"]) if "sensePL_final" in row.index else int(row["sensePL"])
                sense_cat_final = int(row["senseCAT_exec_final"]) if "senseCAT_exec_final" in row.index else int(row["senseCAT_exec"])

                pl_vs0_init = float(row.get("pl_vs0_init", 0.0))
                cat_vs0_init = float(row.get("cat_vs0_init", 0.0))

                history_data = None
                cat_td = None
                if "time_history_json" in row.index:
                    history_data = decode_time_history(row["time_history_json"])

                if "catDelay" in row.index and not pd.isna(row["catDelay"]):
                    cat_td = float(row["catDelay"])

                if history_data is not None:
                    times = history_data["times"]
                    vs_pl = history_data["vs_pl"]
                    vs_ca = history_data["vs_ca"]
                    z_pl = history_data.get("z_pl")
                    z_ca = history_data.get("z_ca")
                    if z_pl is None or z_ca is None:
                        z_pl = integrate_altitude_from_vs(times, vs_pl, 0.0)
                        z_ca = integrate_altitude_from_vs(
                            times,
                            vs_ca,
                            h0 if cat_above else -h0,
                        )
                else:
                    sense_pl = int(row["sensePL"])
                    sense_cat_exec = int(row["senseCAT_exec"])
                    cat_ag = float(row["catAccel_g"])
                    cat_vs = float(row["catVS_cmd"])
                    cat_cap = float(row["catCap_cmd"])
                    pl_delay = float(row["plDelay"])

                    cat_td_for_series = 0.0 if cat_td is None else cat_td

                    times, vs_pl = vs_time_series(
                        tgo,
                        0.1,
                        pl_delay,
                        PL_ACCEL_G,
                        PL_VS_FPM,
                        sense=sense_pl,
                        cap_fpm=PL_VS_CAP_FPM,
                        vs0_fpm=pl_vs0_init,
                    )
                    _, vs_ca = vs_time_series(
                        tgo,
                        0.1,
                        cat_td_for_series,
                        cat_ag,
                        cat_vs,
                        sense=sense_cat_exec,
                        cap_fpm=cat_cap,
                        vs0_fpm=cat_vs0_init,
                    )
                    z_pl = integrate_altitude_from_vs(times, vs_pl, 0.0)
                    z_ca = integrate_altitude_from_vs(
                        times,
                        vs_ca,
                        h0 if cat_above else -h0,
                    )

                miss_cpa = float(abs(z_ca[-1] - z_pl[-1])) if len(z_pl) else float("nan")

                times_plot, z_pl_plot, z_ca_plot = extend_history_with_pretrigger(
                    times,
                    z_pl,
                    z_ca,
                    pl_vs0=pl_vs0_init,
                    cat_vs0=cat_vs0_init,
                )

                pl_label = "level" if sense_pl_final == 0 else ("climb" if sense_pl_final > 0 else "descend")
                cat_label = "level" if sense_cat_final == 0 else ("climb" if sense_cat_final > 0 else "descend")

                fig_run, ax_run = plt.subplots(figsize=(8,4))
                ax_run.plot(times_plot, z_pl_plot, label=f"PL ({pl_label})")
                ax_run.plot(
                    times_plot,
                    z_ca_plot,
                    label=f"CAT ({cat_label}) [{row['CAT_mode']}]",
                )
                ax_run.fill_between(
                    times_plot,
                    z_pl_plot - alim_ft,
                    z_pl_plot + alim_ft,
                    alpha=0.08,
                    label=f"±ALIM at FL{FL_pl}",
                )
                ax_run.axhline(0, ls='--', lw=1, alpha=0.6)
                t_second_issue = get_second_issue_time(row)
                if t_second_issue is not None:
                    ax_run.axvline(t_second_issue, ls=':', lw=1, alpha=0.7, label='2nd‑phase issue')
                if cat_td is not None:
                    ax_run.axvline(
                        float(cat_td),
                        color='tab:orange',
                        ls='--',
                        lw=1.2,
                        alpha=0.85,
                        label='CAT response delay',
                    )
                ax_run.set_xlabel("Time since RA trigger (s)")
                ax_run.set_ylabel("Relative altitude (ft)")
                ax_run.set_title(f"Run {int(row['run'])} — {row['eventtype']} — Δh@CPA={miss_cpa:.0f} ft")
                ax_run.legend(); ax_run.grid(True, alpha=0.3)
                st.pyplot(fig_run)
                st.caption(
                    "Shaded band visualises ±ALIM around the protected aircraft; the vertical marker shows the CAT response delay."
                )


        # Download
        csv_bytes = df.to_csv(index=False).encode('utf-8')
        st.download_button(
            "Download CSV",
            csv_bytes,
            file_name="tcas_batch_results_v71.csv",
            mime="text/csv",
        )

 






































