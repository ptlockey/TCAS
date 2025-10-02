#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ACAS/TCAS Monte‑Carlo v2 — optimal‑sense + non‑compliance + 2‑phase RA
-----------------------------------------------------------------------
This version implements the end‑to‑end edits we discussed:
A) Execute a *second‑phase* RA (STRENGTHEN/REVERSE) with realistic latency and targets.
B) Use an **altitude‑dependent ALIM** (~300–700 ft) instead of a fixed value (per MOPS concept). 
C) Constrain **RA timing** to ~15–35 s before CPA (typical TCAS trigger window).  # see notes
D) Re‑tune non‑compliance priors toward EUROCONTROL‘s radar‑based compliance envelopes.
E) **Level‑off contexts** when aggressiveness=0.0 (fully left): near‑level VS sampling to emulate
   the high prevalence of “Level off, level off” RAs reported in congested airspace.  # see notes
F) Optional AP/FD TCAS effect (shorter CAT delay & slightly higher accel).
G) More conservative **reversal** trigger (should be rare, <<1% overall).  # see notes
H) Lower **TA‑only** prevalence among RA participants (~0.2–0.5% baseline).
I) **Reaction time mixture**: CAT ≈70% ~4.5 s / 30% ~8 s; PL delay moved to ~2.2–2.5 s.
J) Add a *Method‑B‑like* compliance scoring label (Following/Weak/No/Opposite/Excessive)
   and log extra columns (ALIM, second‑phase issue time, etc.).
USAGE (CLI)
-----------
python acas_tcas_montecarlo_v2.py \
  --n 5000 --seed 26 --scenario Head-on --aggr 0.30 \
  --out tcas_batch_results_v2.csv
Flags:
  --scenario {Head-on,Crossing,Overtaking,Custom}
  --aggr     Initial trajectory aggressiveness (0.0 = mostly level; enables E)
  --p-opp/--p-ta/--p-weak  Non‑compliance priors (re‑tuned defaults)
  --apfd-share  Share of CAT with AP/FD TCAS behavior (default 0.25)
  --no-jitter   Disable ±30% jitter of priors
  --hdg1-min/--hdg1-max/--hdg2-min/--hdg2-max  (for --scenario Custom)
  --out      Output CSV path
Dependencies: numpy, pandas (both are standard scientific Python libs).
NOTES & SOURCES (operational underpinnings)
- v7.1 RA logic includes strengthen/reversal; reversal events are *rare* in service (<~1%). [1](https://skybrary.aero/sites/default/files/bookshelf/32945.pdf)
- RA timing is typically ~15–35 s before CPA depending on sensitivity/geometry. [2](https://www.ifalpa.org/media/3642/21atsbl02-follow-the-tcas-ra.pdf)
- ALIM is altitude‑dependent (~300–700 ft across bands), not a single constant. [3](https://www.eurocontrol.int/system/acas)
- Airbus’ operational synthesis of EUROCONTROL radar data: ≈58.7% correct, ≈29.8% partial, ≈11.5% no/opp/excessive;
  “Level off, level off” RAs are very frequent in congested airspace; we emulate this when aggr=0.0. [4](https://aerospace.honeywell.com/us/en/pages/tcas-change-7-1)
- EUROCONTROL radar‑based compliance study (and SKYbrary materials) underpin the compliance envelopes used
  for calibration in this script. [5](https://caa-my.sharepoint.com/personal/philip_lockey_caa_co_uk/_layouts/15/Doc.aspx?sourcedoc=%7B21B3E45B-FFD9-4F98-9FC5-B21B974A2C14%7D&file=tcas_batch_results%20(7).csv&action=default&mobileredirect=true).csv&action=default&mobileredirect=true)
"""
from __future__ import annotations
import argparse
import math
from dataclasses import dataclass
from typing import Tuple, Optional, List, Dict
import numpy as np
import pandasfrom typing import Tuple, Optional, List, Dict
import numpy as np
import pandas as pd
# ---------------------------- Constants & Defaults ---------------------------- #
G = 9.80665              # m/s^2
FT_PER_M = 3.28084
MS_PER_FPM = 0.00508     # 1 fpm = 0.00508 m/s
# Performance-limited (PL) aircraft defaults (more conservative than before)
PL_DELAY_MEAN_S = 0.9 # (more servative than before)
PL_DELAY_MEAN_S = 2.2    # s (moved from ~0.9 s to represent manual/limited performance)
PL_DELAY_SD_S   = 0.4
PL_ACCEL_G      = 0.10
PL_VS_FPM       = 500.0  # fpm target for PL
PL_VS_CAP_FPM   =CEL_G      = 0.10
PL_VS_FPM       = 500.0  # fpm target for PL
PL_VS_CAP_FPM   = 500.0  # cap for PL
# CAT (transport) nominal corrective behaviors (v7.1 assumptions)
CAT_INIT_VS_FPM      = 1500.0    # initial corrective RA target
CAT_STRENG initial corrective RA target
CAT_STRENGTH_FPM     = 2500.0    # strengthened RA target (Increase Climb/Descent)
CAT_CAP_INIT_FPM     = 2000.0    # initial cap
CAT_CAP_STRENGTH_FPM = 3000.0    # strengthened cap
# RA timing window (# strengthened cap
# RA timing window (typical)
TGO_MIN_S = 15.0
TGO_MAX_S = 35.0
# AL
TGO_MIN_S = 15.0
TGO_MAX_S = 35.0
# ALIM margin for classification conservatism (ft)
ALIM_MARGIN_FT = 100.0
# ---------------------------- Utility Functions ------------------------------ #
def ias_to_tas(ias_kt:--------------- #
def ias_to_tas(ias_kt: float, pressure_alt_ft: float) -> float:
    """Very rough TAS from IAS using ISA density ratio sigma ~ (1 - 6.875eloat:
    """Very rough TAS from IAS using ISA density ratio sigma ~ (1 - 6.875e-6*h)^4.256."""
    sigma = (1.0 - 6.875e-6 * pressure_alt_ft) ** 4~ (1 - 6.875e-6*h)^4.256."""
    sigma = (1.0 - 6.875e-6 * pressure_alt_ft) ** 4.256
    sigma = max(1e-3, sigma)
    return ias_kt / math.sqrt(sigma)
def vs_time_series(t_end_s: float, dt_s: float, t_delay_s: float, a_g: float,
                   v_f_fpm: float,loat,
                   v_f_fpm: float, sense: int, cap_fpm: Optional[float] = None,
                   vs0_fpm: float = 0.0) -> Tuple[np.ndarray, np.ndarray]:
    """Build a vertical speed profile: delay, ramp at acceleration to target, saturate at cap.
    sense: +1 climb, -1 descend
    """
    a = a_g * G  # m/s^2
    v_target = v_f_fpm if cap_fpm is None else min """
    a = a_g * G  # m/s^2
    v_target = v_f_fpm if cap_fpm is None else min(v_f_fpm, cap_fpm)
    a_fpm_s = a * 60.0
    times = np.arange(0.0, t_end_s + 1e-9, dt_s)
    vs_fpm = np.zeros_like60.0
    times = np.arange(0.0, t_end_s + 1e-9, dt_s)
    vs_fpm = np.zeros_like(times, dtype=float)
    target_signed = sense * v_target
    for i, t in enumerate(times):
        if t <= t_delay_s:
            vs_fpm[i] = vs0_fpm
        else:
            te = t - t_delay_s
            delta = target_signed - vs0_fpm
            stepm[i] = vs0_fpm
        else:
            te = t - t_delay_s
            delta = target_signed - vs0_fpm
            step = math.copysign(min(abs(a_fpm_s * te), abs(delta)), delta)
            vs_fpm[i] = vs0_fpm + step
             vs_fpm[i] = vs0_fpm + step
    return times, vs_fpm
def integrate_altitude_from_vs(times_s: np.ndarray, vs_fpm: np.ndarray, z0_ft: float = 0.0) -> np.ndarray:
    dt = np.diff(times_s, prependde_from_vs(times_s: np.ndarray, vs_fpm: np.ndarray, z0_ft: float = 0.0) -> np.ndarray:
    dt = np.diff(times_s, prepend=times_s[0])
    z = z0_ft + np.cumsum((vs_fpm/60.0) * dt)
   0_ft + np.cumsum((vs_fpm/60.0) * dt)
    z[0] = z0_ft
    return z
def relative_closure_kt(v1_kt: float, hdg1_deg: float, v2_kt: float, hdg2_deg: float) -> float:
] = z0_ft
    return z
def relative_closure_kt(v1_kt: float, hdg1_deg: float, v2_kt: float, hdg2_deg: float) -> float:
    th1, th2 = math.radians(hdg1_deg), math.radians(hdg2_deg)
    v1 = np.array([v1_kt*math.sin(th1), v1def relative_closure_kt(v1_kt: float, hdg1_deg: float, v2_kt: float, hdg2_deg: float) -> float:
    th1, th2 = math.radians(hdg1_deg), math.radians(hdg2_deg)
    v1 = np.array([v1_kt*math.sin(th1), v1_kt*math.cos(th1)])
    v2 = np.array([v2_kt*math.sin(th2), v2_kt*math.cos(th2)])
    return float(np.linalg.norm(v1 - v2))
def time_to_go_from_geometry(r0_nm: float, v_closure_kt: float) -> Optional[float]:
    if v_closure_kt <= 1e-6:
        return None_closure_kt <= 1e-6:
        return None
    return 3600.0 * (r0_nm / v_closure_kt)
def sample_headings(rng: np.random.Generator, scenario: str,
                    hdg1_min: float, hdg1_max: float,
                    rel_min: Optional[float] = None, rel_max: Optional[float] = None,
                    hdg2_min: Optional[float                hdg2_min: Optional[float] = None, hdg2_max: Optional[float] = None) -> Tuple[float, float]:
    h1 = float(rng.uniform(hdg1_min, hdg1ional[float] = None, hdg2_max: Optional[float] = None) -> Tuple[float, float]:
    h1 = float(rng.uniform(hdg1_min, hdg1_max))
    if scenario == 'Custom':
        h2 = float(rng.uniform(hdg2_min, hdg2_max))
    else:
        rel = float(rng.uniform(rel_min, rel_max))
       t(rng.uniform(rel_min, rel_max))
        dirsign = 1 if rng.uniform() < 0.5 else -1
        h2 =l_min, rel_max))
        dirsign = 1 if rng.uniform() < 0.5 else -1
        h2 = (h1 + dirsign * rel) % 360.0
    return h1, h2
# ----------------------------- ALIM & Scoring -------------------------------- #
def alim_ft_from_alt(pressure_alt_ft: float) -> float:
    """Simplified ALIM #
def alim_ft_from_alt(pressure_alt_ft: float) -> float:
    """Simplified ALIM model: ~300 ft (low) to ~700 ft (upper) across 1000 ft (low) to ~700 ft (upper) across 10k–35k ft (approx).
    For a precise model, substitute DO‑185B / ED‑143 bands. [3](https://www.eurocontrol.int/system/acas)
    """
    a = max(0.0, min(35000.0, pressure_alt_ft))
    if a <= 10000.0: return 300.0
    if a >= 35000.0: return 700.0
    return float(np.interp(a, [10000.0, 35000.0], [300.0, 700.0]))
def first_move_time(times: np.ndarray, vs: np.ndarray, tol_fpm: float=50.0) -> float:
    base = vs[0]
    moved = np.where(np.abs(vs - base) > tol_fpm)[0]
    return float(times[moved[0]]) if len(moved) > 0 else 0.0
def compliance_score_method_b_like(sense_required: int, vs_trace: np.ndarray, times: np.ndarray,
                                   required_vs: float=CAT_INIT_VS_FPM) -> str:
    """Simplified proxy for IATA/EUROCONTROL Method B scoring.
    Inspect VS ≈3 s after first movement and classify: Following / Weak / No / Opposite / Excessive.
    """
    t_move = first_move_time(times, vs_trace)
    t_eval = t_move + 3.0
    idx = np.searchsorted(times, t_eval, side='left')
    if idx >= len(times): idx = -1
    vs_eval = float(vs_trace[idx])
    if abs(vs_eval) < 100.0:
        return "No Response"
    if math.copysign(1, vs_eval) != sense_required:
        return "Opposite"
    avs = abs(vs_eval)
    if avs >= 3200.0:
        return "Excessive"
    if avs >= 0.8*required_vs:
        return "Following"
    else:
        return "Weak"
# --------------------- Initial VS & level-off mode ---------------------------- #
def sample_initial_vs_with_aggressiveness(rng: np.random.Generator, aggressiveness: float,
                                          leveloff_context: bool=False) -> float:
    """At aggr≈0, emulate level‑off environment: near‑zero VS most of the time (E)."""
    if leveloff_context:
        # ≈90% very near level; remainder small ±250 fpm
        if rng.uniform() < 0.90:
            return float(rng.normal(0.0, 50.0))
        return float(rng.normal(rng.choice([+250.0, -250.0]), 100.0))
    aggr = max(0.0, min(1.0, aggressiveness))
    level_p = float(np.clip(0.8 - 0.6*aggr, 0.05, 0.95))
    climb_p = descend_p = (1.0 - level_p)/2.0
    u = rng.uniform()
    level_sd = 50.0 + 200.0*aggr
    climb_mean = 1000.0 + 300.0*aggr
    descend_mean = -climb_mean
    climb_sd = 200.0 + 200.0*aggr
    if u < level_p:
        return float(rng.normal(0.0, level_sd))
    elif u < level_p + climb_p:
        return float(rng.normal(climb_mean, climb_sd))
    else:
        return float(rng.normal(descend_mean, climb_sd))
# --------------------- Sense choice & non-compliance -------------------------- #
def simulate_miss_for_senses(tgo: float, dt: float, h0: float, cat_above: bool,
                             sense_pl: int, sense_cat: int,
                             pl_delay: float, pl_accel: float, pl_vs: float, pl_cap: float, pl_vs0: float,
                             cat_delay: float, cat_accel: float, cat_vs: float, cat_cap: float, cat_vs0: float) -> float:
    times, vs_pl = vs_time_series(tgo, dt, pl_delay, pl_accel, pl_vs, sense=sense_pl, cap_fpm=pl_cap, vs0_fpm=pl_vs0)
    _,     vs_ca = vs_time_series(tgo, dt, cat_delay, cat_accel, cat_vs, sense=sense_cat, cap_fpm=cat_cap, vs0_fpm=cat_vs0)
    z_pl = integrate_altitude_from_vs(times, vs_pl, 0.0)
    z_ca = integrate_altitude_from_vs(times, vs_ca, h0 if cat_above else -h0)
    return float(abs(z_ca[-1] - z_pl[-1]))
def choose_optimal_sense(tgo: float, dt: float, h0: float, cat_above: bool,
                         pl_vs0: float, cat_vs0: float,
                         cat_delay_nom: float=5.0, cat_accel_nom: float=0.25,
                         cat_vs: float=CAT_INIT_VS_FPM, cat_cap: float=CAT_CAP_INIT_FPM) -> Tuple[Tuple[int,int], float, float]:
    """Pick senses to maximize miss@CPA; light penalty for crossing is implicit in miss metric."""
    # PL up / CAT down
    miss_updown = simulate_miss_for_senses(tgo, dt, h0, cat_above, +1, -1,
                                           PL_DELAY_MEAN_S, PL_ACCEL_G, PL_VS_FPM, PL_VS_CAP_FPM, pl_vs0,
                                           cat_delay_nom, cat_accel_nom, cat_vs, cat_cap, cat_vs0)
    # PL down / CAT up
    miss_downup = simulate_miss_for_senses(tgo, dt, h0, cat_above, -1, +1,
                                           PL_DELAY_MEAN_S, PL_ACCEL_G, PL_VS_FPM, PL_VS_CAP_FPM, pl_vs0,
                                           cat_delay_nom, cat_accel_nom, cat_vs, cat_cap, cat_vs0)
    if miss_updown > miss_downup:
        return (+1, -1), miss_updown, miss_downup
    else:
        return (-1, +1), miss_downup, miss_updown
def apply_non_compliance_to_cat(rng: np.random.Generator, sense_cat: int,
                                base_delay_s: float, base_accel_g: float,
                                vs_fpm: float, cap_fpm: float,
                                p_opp: float = 0.010, p_taonly: float = 0.003, p_weak: float = 0.300,
                                jitter: bool = True) -> Tuple[str, int, float, float, float, float]:
    """Return (mode, sense_exec, delay, accel, vs_cmd, cap_cmd). TA‑only -> zero accel/VS.
    Priors are tuned toward radar‑based compliance mix.
    """
    if jitter:
        # ±30% jitter
        p_opp    = max(0.0, min(1.0, p_opp    * rng.uniform(0.7, 1.3)))
        p_taonly = max(0.0, min(1.0, p_taonly * rng.uniform(0.7, 1.3)))
        p_weak   = max(0.0, min(1.0, p_weak   * rng.uniform(0.7, 1.3)))
    u = rng.uniform()
    if u < p_opp:
        return ("opposite-sense", -sense_cat, base_delay_s, base_accel_g, vs_fpm, cap_fpm)
    u -= p_opp
    if u < p_taonly:
        return ("no-response", sense_cat, base_delay_s, 0.0, 0.0, 0.0)
    u -= p_taonly
    if u < p_weak:
        return ("weak-compliance", sense_cat, base_delay_s+1.0, max(0.03, 0.6*base_accel_g), 0.6*vs_fpm, 0.6*cap_fpm)
    return ("compliant", sense_cat, base_delay_s, base_accel_g, vs_fpm, cap_fpm)
# --------------------------- Event classification ----------------------------- #
def classify_event(times: np.ndarray, z_pl: np.ndarray, z_ca: np.ndarray,
                   vs_pl: np.ndarray, vs_ca: np.ndarray, tgo: float,
                   alim_ft: float, margin_ft: float,
                   sense_chosen_cat: int, sense_exec_cat: int) -> Tuple[str, float, float, float]:
    """Return (event_label, minsep, sep@CPA, t_check). Keep REVERSE rare: need thin prediction + sense mismatch."""
    sep = np.abs(z_ca - z_pl)
    minsep = float(np.min(sep))
    sep_cpa = float(sep[-1])
    t_pl_move = first_move_time(times, vs_pl)
    t_ca_move = first_move_time(times, vs_ca)
    t_check = max(t_pl_move, t_ca_move) + 3.0
    mask = times >= t_check
    if np.any(mask):
        t_obs = times[mask]
        sep_obs = sep[mask]
        rel_rate = (vs_ca - vs_pl)/60.0
        rel_obs = rel_rate[mask]
        s_last = float(sep_obs[-1])
        r_last = float(rel_obs[-1])
        t_rem = max(0.0, tgo - t_obs[-1])
        pred_miss = abs(s_last + r_last * t_rem)
        approaching = (r_last < 0)
        thin_pred = (pred_miss < (alim_ft - margin_ft))
        if approaching and thin_pred and (sense_chosen_cat != sense_exec_cat):
            return ("REVERSE", minsep, sep_cpa, float(t_obs[-1]))
    if (minsep < (alim_ft - margin_ft)) or (sep_cpa < (alim_ft - margin_ft)):
        return ("STRENGTHEN", minsep, sep_cpa, float(t_check))
    return ("NONE", minsep, sep_cpa, float(t_check))
def apply_second_phase(times: np.ndarray, vs_pl: np.ndarray, vs_ca: np.ndarray,
                       tgo: float, dt: float, eventtype: str,
                       sense_pl: int, sense_cat_exec: int,
                       pl_vs0: float, cat_vs0: float,
                       pl_delay: float = PL_DELAY_MEAN_S, pl_accel_g: float = PL_ACCEL_G, pl_cap: float = PL_VS_CAP_FPM,
                       cat_delay: float = 1.0, cat_accel_g: float = 0.20,
                       cat_vs_strength: float = CAT_STRENGTH_FPM, cat_cap: float = CAT_CAP_STRENGTH_FPM,
                       decision_latency_s: float = 1.0) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Optional[float]]:
    """Execute STRENGTHEN/REVERSE after detection; return updated (times, vs_pl, vs_ca, t2_issue)."""
    if eventtype not in ("STRENGTHEN", "REVERSE"):
        return times, vs_pl, vs_ca, None
    # Issue time slightly after last observation
    t2_issue = float(times[-1] - max(0.0, tgo - times[-1]))
    t2_issue += float(np.clip(decision_latency_s, 0.6, 1.4))
    t_rem = max(0.0, tgo - t2_issue)
    if t_rem <= dt:
        return times, vs_pl, vs_ca, t2_issue
    vs_pl_now = float(vs_pl[-1]); vs_ca_now = float(vs_ca[-1])
    new_sense_pl  = sense_pl if eventtype == "STRENGTHEN" else -sense_pl
    new_sense_cat = sense_cat_exec if eventtype == "STRENGTHEN" else -sense_cat_exec
    t2, vs_pl2 = vs_time_series(t_rem, dt, pl_delay, pl_accel_g, pl_cap, sense=new_sense_pl,
                                cap_fpm=pl_cap, vs0_fpm=vs_pl_now)
    _,  vs_ca2 = vs_time_series(t_rem, dt, cat_delay, cat_accel_g, cat_vs_strength, sense=new_sense_cat,
                                cap_fpm=cat_cap, vs0_fpm=vs_ca_now)
    times2 = np.concatenate([times, t2_issue + t2[1:]])
    vs_pl2  = np.concatenate([vs_pl, vs_pl2[1:]])
    vs_ca2  = np.concatenate([vs_ca, vs_ca2[1:]])
    return times2, vs_pl2, vs_ca2, t2_issue
# ------------------------------- Batch Runner -------------------------------- #
def sample_altitudes_and_h0(rng: np.random.Generator,
                            fl_min: int=150, fl_max: int=300,
                            h0_mean: float = 250.0, h0_sd: float = 100.0,
                            h0_lo: float = 100.0, h0_hi: float = 500.0) -> Tuple[int,int,float]:
    FL_pl_base = int(rng.integers(fl_min, fl_max+1))
    cap = 1000 if FL_pl_base < 290 else 2000
    h0 = float(np.clip(rng.normal(h0_mean, h0_sd), max(h0_lo, 0.0), min(h0_hi, cap)))
    diff_FL = max(1, int(round(h0/100.0)))
    sign = 1 if rng.uniform() < 0.5 else -1
    FL_cat_cand = FL_pl_base + sign * diff_FL
    if not (fl_min <= FL_cat_cand <= fl_max):
        FL_cat_cand = FL_pl_base - sign * diff_FL
        FL_cat_cand = min(fl_max, max(fl_min, FL_cat_cand))
    return int(FL_pl_base), int(FL_cat_cand), float(h0)
def run_batch(runs: int = 5000, seed: int = 26, scenario: str = "Head-on",
              r0_min_nm: float = 4.0, r0_max_nm: float = 8.0,
              aggressiveness: float = 0.30,
              p_opp: float = 0.010, p_ta: float = 0.003, p_weak: float = 0.300,
              jitter_priors: bool = True,
              apfd_share: float = 0.25,
              use_delay_mixture: bool = True,
              dt: float = 0.1,
              hdg1_min: float = 0.0, hdg1_max: float = 360.0, hdg2_min: float = 0.0, hdg2_max: float = 360.0) -> pd.DataFrame:
    rng = np.random.default_rng(int(seed))
    data: List[Dict] = []
    # Scenario geometry settings
    if scenario == "Head-on":
        rel_min, rel_max = 150.0, 210.0
    elif scenario == "Crossing":
        rel_min, rel_max = 60.0, 120.0
    elif scenario == "Overtaking":
        rel_min, rel_max = 0.0, 30.0
    else:  # Custom
        rel_min = rel_max = None
    for k in range(int(runs)):
        # Altitudes & initial miss
        FL_PL, FL_CAT, h0 = sample_altitudes_and_h0(rng)
        cat_above = (FL_CAT > FL_PL) if (FL_CAT != FL_PL) else (rng.uniform() < 0.5)
        # Speeds & headings
        PL_IAS = 120.0
        PL_TAS = ias_to_tas(PL_IAS, FL_PL*100.0)
        CAT_TAS = float(rng.uniform(420.0, 470.0))
        if scenario == "Custom":
            h1 = float(rng.uniform(hdg1_min, hdg1_max))
            h2 = float(rng.uniform(hdg2_min, hdg2_max))
        else:
            h1, h2 = sample_headings(rng, scenario, 0.0, 360.0, rel_min, rel_max)
        r0 = float(rng.uniform(min(r0_min_nm, r0_max_nm), max(r0_min_nm, r0_max_nm)))
        vcl = relative_closure_kt(PL_TAS, h1, CAT_TAS, h2)
        tgo_geom = time_to_go_from_geometry(r0, vcl)
        # RA timing: clamp to 15–35 s (never below 15 s)  [2](https://www.ifalpa.org/media/3642/21atsbl02-follow-the-tcas-ra.pdf)
        if scenario == "Head-on": mu, sd = 25.0, 5.0
        elif scenario == "Crossing": mu, sd = 22.0, 6.0
        else: mu, sd = 30.0, 8.0
        lo, hi = TGO_MIN_S, min(tgo_geom if tgo_geom is not None else TGO_MAX_S, TGO_MAX_S)
        tgo = float(np.clip(rng.normal(mu, sd), lo, max(lo+1.0, hi)))
        # Level‑off when aggressiveness fully left
        leveloff_context = (aggressiveness <= 1e-6)
        vz0_pl  = sample_initial_vs_with_aggressiveness(rng, aggressiveness, leveloff_context)
        vz0_cat = sample_initial_vs_with_aggressiveness(rng, aggressiveness, leveloff_context)
        # Optimal sense under nominal CAT behavior
        (sense_pl, sense_ca), miss_nominal, miss_alt = choose_optimal_sense(
            tgo, dt, h0, cat_above, vz0_pl, vz0_cat,
            cat_delay_nom=5.0, cat_accel_nom=0.25, cat_vs=CAT_INIT_VS_FPM, cat_cap=CAT_CAP_INIT_FPM)
        # CAT delay mixture + AP/FD effect
        if use_delay_mixture:
            if rng.uniform() < 0.7:
                cat_delay_eff = float(max(0.0, rng.normal(4.5, 1.0)))
                cat_accel_eff = float(max(0.05, rng.normal(0.25, 0.03)))
            else:
                cat_delay_eff = float(max(0.0, rng.normal(8.5, 1.5)))
                cat_accel_eff = float(max(0.05, rng.normal(0.10, 0.02)))
        else:
            cat_delay_eff = 5.0
            cat_accel_eff = 0.25
        if rng.uniform() < apfd_share:
            cat_delay_eff = max(0.0, cat_delay_eff - 0.8)
            cat_accel_eff = max(0.05, cat_accel_eff + 0.03)
        # Apply non‑compliance (D/H)
        mode, sense_cat_exec, cat_delay_exec, cat_accel_exec, cat_vs_exec, cat_cap_exec = apply_non_compliance_to_cat(
            rng, sense_ca, delay_s=cat_delay_eff, accel_g=cat_accel_eff, vs_fpm=CAT_INIT_VS_FPM, cap_fpm=CAT_CAP_INIT_FPM,
            p_opp=p_opp, p_taonly=p_ta, p_weak=p_weak, jitter=jitter_priors)
        # PL delay sample
        pl_delay = max(0.0, rng.normal(PL_DELAY_MEAN_S, PL_DELAY_SD_S))
        # First phase
        times, vs_pl = vs_time_series(tgo, dt, pl_delay, PL_ACCEL_G, PL_VS_FPM, sense=sense_pl,
                                      cap_fpm=PL_VS_CAP_FPM, vs0_fpm=vz0_pl)
        _, vs_ca = vs_time_series(tgo, dt, cat_delay_exec, cat_accel_exec, cat_vs_exec, sense=sense_cat_exec,
                                  cap_fpm=cat_cap_exec, vs0_fpm=vz0_cat)
        z_pl = integrate_altitude_from_vs(times, vs_pl, 0.0)
        z_ca = integrate_altitude_from_vs(times, vs_ca, h0 if cat_above else -h0)
        # Ownship ALIM
        own_alt_ft = FL_PL * 100.0
        alim_ft = alim_ft_from_alt(own_alt_ft)
        # Classify event
        eventtype, minsep_ft, sep_cpa_ft, t_check = classify_event(
            times, z_pl, z_ca, vs_pl, vs_ca, tgo,
            alim_ft=alim_ft, margin_ft=ALIM_MARGIN_FT,
            sense_chosen_cat=sense_ca, sense_exec_cat=sense_cat_exec)
        # Second phase (A/G)
        t2_issue = None
        if eventtype in ("STRENGTHEN", "REVERSE"):
            times2, vs_pl2, vs_ca2, t2_issue = apply_second_phase(
                times, vs_pl, vs_ca, tgo, dt,
                eventtype, sense_pl, sense_cat_exec,
                pl_vs0=vz0_pl, cat_vs0=vz0_cat,
                pl_delay=pl_delay, pl_accel_g=PL_ACCEL_G, pl_cap=PL_VS_CAP_FPM,
                cat_delay=1.0, cat_accel_g=0.20,
                cat_vs_strength=CAT_STRENGTH_FPM, cat_cap=CAT_CAP_STRENGTH_FPM,
                decision_latency_s=float(np.clip(rng.normal(1.0, 0.2), 0.6, 1.4)))
            if t2_issue is not None:
                z_pl2 = integrate_altitude_from_vs(times2, vs_pl2, 0.0)
                z_ca2 = integrate_altitude_from_vs(times2, vs_ca2, h0 if cat_above else -h0)
                sep2 = np.abs(z_ca2 - z_pl2)
                minsep_ft = float(np.min(sep2))
                sep_cpa_ft = float(sep2[-1])
                times, vs_pl, vs_ca, z_pl, z_ca = times2, vs_pl2, vs_ca2, z_pl2, z_ca2
        miss_cpa_ft = float(abs(z_ca[-1] - z_pl[-1]))
        # Compliance (CAT) — Method‑B‑like label
        comp_label = compliance_score_method_b_like(sense_required=sense_cat_exec, vs_trace=vs_ca, times=times)
        data.append(dict(
            run=k+1, scenario=scenario,
            FL_PL=FL_PL, FL_CAT=FL_CAT,
            PL_TAS=PL_TAS, CAT_TAS=CAT_TAS,
            PLhdg=h1, CAThdg=h2,
            R0NM=r0, closurekt=vcl, tgos=tgo, h0ft=h0,
            aggressiveness=aggressiveness, leveloff=int(leveloff_context),
            sensePL=sense_pl, senseCAT_chosen=sense_ca,
            CAT_mode=mode, senseCAT_exec=sense_cat_exec,
            plDelay=pl_delay, plAccel_g=PL_ACCEL_G,
            catDelay=cat_delay_exec, catAccel_g=cat_accel_exec,
            catVS_cmd=cat_vs_exec, catCap_cmd=cat_cap_exec,
            ALIM_ft=alim_ft,
            missCPAft=miss_cpa_ft, minsepft=minsep_ft, sep_cpa_ft=sep_cpa_ft,
            eventtype=eventtype, t_second_issue=t2_issue,
            comp_label=comp_label
        ))
    return pd.DataFrame(data)
# ----------------------------------- CLI ------------------------------------- #
def main(argv=None):
    p = argparse.ArgumentParser(description="ACAS/TCAS Monte‑Carlo v2 (strengthen/reverse & compliance)")
    p.add_argument('--n', type=int, default=5000)
    p.add_argument('--seed', type=int, default=26)
    p.add_argument('--scenario', type=str, default='Head-on', choices=['Head-on','Crossing','Overtaking','Custom'])
    p.add_argument('--r0-min', type=float, default=4.0)
    p.add_argument('--r0-max', type=float, default=8.0)
    p.add_argument('--aggr', type=float, default=0.30, help='Initial trajectory aggressiveness (0.0 = level‑off context)')
    p.add_argument('--p-opp', type=float, default=0.010)
    p.add_argument('--p-ta', type=float, default=0.003)
    p.add_argument('--p-weak', type=float, default=0.300)
    p.add_argument('--no-jitter', action='store_true', help='Disable ±30% jitter of priors')
    p.add_argument('--apfd-share', type=float, default=0.25)
    p.add_argument('--use-delay-mixture', action='store_true', default=True)
    p.add_argument('--out', type=str, default='tcas_batch_results_v2.csv')
    # Custom headings
    p.add_argument('--hdg1-min', type=float, default=0.0)
    p.add_argument('--hdg1-max', type=float, default=360.0)
    p.add_argument('--hdg2-min', type=float, default=0.0)
    p.add_argument('--hdg2-max', type=float, default=360.0)
    args = p.parse_args(argv)
    df = run_batch(
        runs=args.n, seed=args.seed, scenario=args.scenario,
        r0_min_nm=args.r0_min, r0_max_nm=args.r0_max,
        aggressiveness=args.aggr,
        p_opp=args.p_opp, p_ta=args.p_ta, p_weak=args.p_weak,
        jitter_priors=(not args.no_jitter),
        apfd_share=args.apfd_share,
        use_delay_mixture=args.use_delay_mixture,
        hdg1_min=args.hdg1_min, hdg1_max=args.hdg1_max,
        hdg2_min=args.hdg2_min, hdg2_max=args.hdg2_max
    )
    df.to_csv(args.out, index=False)
    # Terse summary
    n = len(df)
    evt = df['eventtype'].value_counts(normalize=True).mul(100).round(2).to_dict()
    comp = df['comp_label'].value_counts(normalize=True).mul(100).round(2).to_dict()
    print(f"Saved {n} runs to {args.out}")
    print(f"Event types %: {evt}")
    print(f"Compliance %: {comp}")
if __name__ == '__main__':
    main()






































