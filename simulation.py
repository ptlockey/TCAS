"""Core ACAS/TCAS Monte Carlo simulation primitives.

This module collects the numerical helpers that were previously embedded in
``calculator.py`` so that they can be imported independently of the Streamlit
UI.  Functions defined here are intentionally free of any Streamlit imports to
make the simulation logic reusable from scripts as well as unit tests.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import json

import numpy as np
import pandas as pd

# ---------------------------- Constants ----------------------------

G = 9.80665               # m/s^2
FT_PER_M = 3.28084
MS_PER_FPM = 0.00508      # 1 fpm = 0.00508 m/s

# PL (performance-limited) parameters
PL_DELAY_MEAN_S = 0.9
PL_DELAY_SD_S   = 0.0
PL_ACCEL_G      = 0.10
PL_VS_FPM       = 500.0
PL_VS_CAP_FPM   = 500.0
PL_IAS_KT       = 120.0

# CAT parameters (v7.1-style)
CAT_INIT_VS_FPM      = 1500.0
CAT_STRENGTH_FPM     = 2500.0
CAT_CAP_INIT_FPM     = 1600.0
CAT_CAP_STRENGTH_FPM = 2600.0

# Nominal CAT response characteristics for sense prediction
CAT_MANUAL_DELAY_NOM_S = 5.0
CAT_MANUAL_ACCEL_NOM_G = 0.33
CAT_APFD_DELAY_NOM_S = 0.9
CAT_APFD_ACCEL_NOM_G = 0.25

# Reversal projection characteristics (manual vs AP/FD)
CAT_MANUAL_PROJECTION_DELAY_S = 2.5
CAT_APFD_PROJECTION_DELAY_S = CAT_APFD_DELAY_NOM_S

# RA timing window
TGO_MIN_S = 15.0
TGO_MAX_S = 35.0

# TCAS coordination cadence and reliability
TCAS_UPDATE_PERIOD_S = 1.0  # seconds between TCAS surveillance updates
COORDINATION_DROPOUT_PROB = 1e-3  # chance that an inter-unit handoff misses one cycle

# ALIM margin for classification conservatism (ft)
ALIM_MARGIN_FT = 100.0

# Reversal monitoring
REVERSAL_MONITOR_DELAY_S = 1.0
REVERSAL_ENABLE_TAU_S = 18.0

# Predicted-miss pad for triggering strengthen early (ft)
STRENGTHEN_PAD_FT = 150.0

# Subsequent-manoeuvre tuning
EXIGENT_STRENGTHEN_TAU_S = 20.0
NO_RESPONSE_ESCALATION_S = 3.0
NO_RESPONSE_VS_THRESH_FPM = 100.0

# Reversal management
REVERSAL_INTERLOCK_LOOKBACK_S = 1.8
REVERSAL_IMPROVEMENT_HOLD_S = 1.6
REVERSAL_HOLD_DISABLE_TAU_S = 18.0
PREDICTED_MISS_IMPROVEMENT_TOL_FT = 5.0
REVERSAL_CPA_IMPROVEMENT_TOL_FT = 10.0

# Safeguard for repeated manoeuvre phases
MAX_MANEUVER_PHASES = 4

# Sense determination
VS_SENSE_DEADBAND_FPM = 50.0

# Surveillance report noise (1σ) for classifier inputs
VS_REPORT_NOISE_SD_FPM = 20.0
ALT_REPORT_NOISE_SD_FT = 5.0


def _normalise_mode_key(value: Optional[str]) -> str:
    """Return a lowercase alpha-numeric mode key for comparisons."""

    if value is None:
        return ""
    return "".join(ch for ch in value.lower() if ch.isalnum())


def _is_apfd_mode(value: Optional[str]) -> bool:
    """Return True when the provided mode label corresponds to AP/FD usage."""

    return _normalise_mode_key(value) == "apfd"


def sanitize_tgo_bounds(
    tgo_min_s: Optional[float], tgo_max_s: Optional[float]
) -> Tuple[float, float, float]:
    """Return clipped (lo, hi, mode) for custom t_go windows.

    The mode is centred on the mid-point of the requested window while being
    restricted to the regulatory 24–26 s region whenever feasible. This makes it
    suitable for triangular sampling that still honours the requested bounds.
    """

    lo_raw = TGO_MIN_S if tgo_min_s is None else float(tgo_min_s)
    hi_raw = TGO_MAX_S if tgo_max_s is None else float(tgo_max_s)
    lo = float(np.clip(lo_raw, TGO_MIN_S, TGO_MAX_S))
    hi = float(np.clip(hi_raw, TGO_MIN_S, TGO_MAX_S))
    if hi <= lo + 1e-3:
        hi = min(TGO_MAX_S, lo + 1.0)

    midpoint = 0.5 * (lo + hi)
    preferred = float(np.clip(midpoint, lo + 1e-3, hi - 1e-3))
    mode = float(np.clip(preferred, 24.0, 26.0))
    mode = float(np.clip(mode, lo + 1e-3, hi - 1e-3))

    return lo, hi, mode

# ------------------------ Utility functions ------------------------


def ias_to_tas(ias_kt: float, pressure_alt_ft: float) -> float:
    sigma = (1.0 - 6.875e-6 * pressure_alt_ft) ** 4.256
    sigma = max(1e-3, sigma)
    return ias_kt / math.sqrt(sigma)


def vs_time_series(
    t_end_s: float,
    dt_s: float,
    t_delay_s: float,
    a_g: float,
    v_f_fpm: float,
    sense: int,
    cap_fpm: Optional[float] = None,
    vs0_fpm: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Vertical speed law: delay, then ramp at accel to target; saturate at cap."""

    a = a_g * G
    v_target = v_f_fpm if cap_fpm is None else min(v_f_fpm, cap_fpm)
    a_fpm_s = a * FT_PER_M * 60.0
    times = np.arange(0.0, t_end_s + 1e-9, dt_s)
    vs_fpm = np.zeros_like(times, dtype=float)
    target_signed = sense * v_target
    for i, t in enumerate(times):
        if t <= t_delay_s:
            vs_fpm[i] = vs0_fpm
        else:
            te = t - t_delay_s
            delta = target_signed - vs0_fpm
            step = math.copysign(min(abs(a_fpm_s * te), abs(delta)), delta)
            vs_fpm[i] = vs0_fpm + step
    return times, vs_fpm


def integrate_altitude_from_vs(
    times_s: np.ndarray,
    vs_fpm: np.ndarray,
    z0_ft: float = 0.0,
) -> np.ndarray:
    dt = np.diff(times_s, prepend=times_s[0])
    z = z0_ft + np.cumsum((vs_fpm / 60.0) * dt)
    z[0] = z0_ft
    return z


def encode_time_history(
    times: np.ndarray,
    vs_pl: np.ndarray,
    vs_ca: np.ndarray,
    z_pl: np.ndarray,
    z_ca: np.ndarray,
) -> str:
    """Serialise trajectory histories for storage in batch outputs."""

    payload = {
        "times": [float(x) for x in np.asarray(times, dtype=float)],
        "vs_pl": [float(x) for x in np.asarray(vs_pl, dtype=float)],
        "vs_ca": [float(x) for x in np.asarray(vs_ca, dtype=float)],
        "z_pl": [float(x) for x in np.asarray(z_pl, dtype=float)],
        "z_ca": [float(x) for x in np.asarray(z_ca, dtype=float)],
    }
    return json.dumps(payload, separators=(",", ":"))


def compute_residual_risk(delta_pl: float, delta_cat: float) -> float:
    """Return the residual risk ratio using a signed, epsilon-guarded PL delta."""

    guard = delta_pl if abs(delta_pl) >= 1e-3 else math.copysign(1e-3, delta_pl or 1.0)
    return delta_cat / guard * 0.011


def decode_time_history(value: object) -> Optional[Dict[str, np.ndarray]]:
    """Decode trajectory histories stored by :func:`encode_time_history`."""

    if value is None:
        return None
    if isinstance(value, float) and math.isnan(value):
        return None
    if isinstance(value, (bytes, bytearray)):
        try:
            value = value.decode("utf-8")
        except Exception:  # noqa: BLE001
            return None
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        try:
            payload = json.loads(text)
        except json.JSONDecodeError:
            return None
    elif isinstance(value, dict):
        payload = value
    else:
        return None

    try:
        times = np.asarray(payload["times"], dtype=float)
        vs_pl = np.asarray(payload["vs_pl"], dtype=float)
        vs_ca = np.asarray(payload["vs_ca"], dtype=float)
    except (KeyError, TypeError, ValueError):
        return None

    result: Dict[str, np.ndarray] = {
        "times": times,
        "vs_pl": vs_pl,
        "vs_ca": vs_ca,
    }

    for key in ("z_pl", "z_ca"):
        try:
            values = payload.get(key)
        except AttributeError:
            values = None
        if values is None:
            continue
        try:
            arr = np.asarray(values, dtype=float)
        except (TypeError, ValueError):
            continue
        if arr.shape == times.shape:
            result[key] = arr

    return result


def extend_history_with_pretrigger(
    times: np.ndarray,
    z_pl: np.ndarray,
    z_ca: np.ndarray,
    pl_vs0: float,
    cat_vs0: float,
    pretrigger_window_s: float = 2.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Prepend a negative-time window showing motion prior to the RA trigger."""

    times = np.asarray(times, dtype=float)
    z_pl = np.asarray(z_pl, dtype=float)
    z_ca = np.asarray(z_ca, dtype=float)

    if times.size == 0 or z_pl.shape != times.shape or z_ca.shape != times.shape:
        return times, z_pl, z_ca

    if times.size >= 2:
        dt = float(np.min(np.diff(times)))
        if dt <= 1e-9:
            dt = float(np.max(np.diff(times))) if times.size >= 2 else 0.0
    else:
        dt = 0.0

    pre_window = float(max(0.0, pretrigger_window_s))
    if pre_window <= 1e-9 or dt <= 1e-9:
        return times, z_pl, z_ca

    steps = int(max(1, math.ceil(pre_window / dt)))
    pre_times = -dt * np.arange(steps, 0, -1, dtype=float)

    pl_rate = float(pl_vs0) / 60.0
    cat_rate = float(cat_vs0) / 60.0
    z_pl0 = float(z_pl[0])
    z_ca0 = float(z_ca[0])

    z_pl_pre = z_pl0 + pre_times * pl_rate
    z_ca_pre = z_ca0 + pre_times * cat_rate

    times_ext = np.concatenate([pre_times, times])
    z_pl_ext = np.concatenate([z_pl_pre, z_pl])
    z_ca_ext = np.concatenate([z_ca_pre, z_ca])

    return times_ext, z_pl_ext, z_ca_ext


def relative_closure_kt(v1_kt: float, hdg1_deg: float, v2_kt: float, hdg2_deg: float) -> float:
    th1, th2 = math.radians(hdg1_deg), math.radians(hdg2_deg)
    v1 = np.array([v1_kt * math.sin(th1), v1_kt * math.cos(th1)])
    v2 = np.array([v2_kt * math.sin(th2), v2_kt * math.cos(th2)])
    return float(np.linalg.norm(v1 - v2))


def time_to_go_from_geometry(r0_nm: float, v_closure_kt: float) -> Optional[float]:
    if v_closure_kt <= 1e-6:
        return None
    return 3600.0 * (r0_nm / v_closure_kt)


def derive_single_run_geometry(
    initial_range_nm: float,
    closure_kt: float,
    use_custom_tgo: bool,
    tgo_window: Optional[Tuple[float, float, float]],
) -> Tuple[Optional[float], float]:
    """Return (t_cpa, effective_initial_range_nm) for the single-run demo."""

    initial_range_effective = float(initial_range_nm)

    if use_custom_tgo and tgo_window is not None and closure_kt > 1e-6:
        lo, hi, _ = tgo_window
        t_cpa = float(np.clip(hi, lo, hi))
        initial_range_effective = (closure_kt * t_cpa) / 3600.0
        return t_cpa, initial_range_effective

    t_cpa = time_to_go_from_geometry(initial_range_effective, closure_kt)
    return t_cpa, initial_range_effective


def sample_headings(
    rng: np.random.Generator,
    scenario: str,
    hdg1_min: float,
    hdg1_max: float,
    rel_min: Optional[float] = None,
    rel_max: Optional[float] = None,
    hdg2_min: Optional[float] = None,
    hdg2_max: Optional[float] = None,
) -> Tuple[float, float]:
    h1 = float(rng.uniform(hdg1_min, hdg1_max))
    if scenario == "Custom":
        h2 = float(rng.uniform(hdg2_min, hdg2_max))
    else:
        rel = float(rng.uniform(rel_min, rel_max))
        dirsign = 1 if rng.uniform() < 0.5 else -1
        h2 = (h1 + dirsign * rel) % 360.0
    return h1, h2


# ----------------------------- ALIM & Scoring -----------------------------


# Nominal TCAS v7.1 ALIM schedule (ft) expressed as altitude bands.
ALIM_BANDS_FT = (
    (5000.0, 10000.0, 350.0),  # FL50–FL100
    (10000.0, 20000.0, 400.0),  # FL100–FL200
    (20000.0, 42000.0, 600.0),  # FL200–FL420
)


def alim_ft_from_alt(pressure_alt_ft: float, override_ft: Optional[float] = None) -> float:
    """Return the ALIM associated with the provided altitude.

    When ``override_ft`` is supplied the caller is requesting a user-selected
    ALIM value (e.g. via the UI). Otherwise the value is determined from the
    discrete TCAS v7.1 altitude bands.
    """

    if override_ft is not None:
        return float(override_ft)

    if pressure_alt_ft < ALIM_BANDS_FT[0][0]:
        return float(ALIM_BANDS_FT[0][2])

    for lower, upper, alim in ALIM_BANDS_FT:
        if lower <= pressure_alt_ft < upper:
            return float(alim)

    # At or above the top band we keep the highest ALIM value.
    return float(ALIM_BANDS_FT[-1][2])


def first_move_time(times: np.ndarray, vs: np.ndarray, tol_fpm: float = 50.0) -> float:
    base = vs[0]
    moved = np.where(np.abs(vs - base) > tol_fpm)[0]
    return float(times[moved[0]]) if len(moved) > 0 else 0.0


def compliance_score_method_b_like(
    sense_required: int,
    vs_trace: np.ndarray,
    times: np.ndarray,
    required_vs: float = CAT_INIT_VS_FPM,
) -> str:
    """Proxy for Method-B classification of pilot compliance."""

    t_move = first_move_time(times, vs_trace)
    t_eval = t_move + 3.5
    idx = np.searchsorted(times, t_eval, side="left")
    if idx >= len(times):
        idx = -1
    vs_eval = float(vs_trace[idx])
    if abs(vs_eval) < 100.0:
        return "No Response"
    if math.copysign(1, vs_eval) != sense_required:
        return "Opposite"
    avs = abs(vs_eval)
    if avs >= 3200.0:
        return "Excessive"
    if avs >= 0.8 * required_vs:
        return "Following"
    else:
        return "Weak"


# --------------- Initial VS & level-off (aggressiveness control) --------------


def sample_initial_vs_with_aggressiveness(
    rng: np.random.Generator,
    aggressiveness: float,
    leveloff_context: bool = False,
) -> float:
    """Sample initial vertical speed as a function of aggressiveness."""

    if leveloff_context:
        if rng.uniform() < 0.90:
            return float(rng.normal(0.0, 50.0))
        return float(rng.normal(rng.choice([+250.0, -250.0]), 100.0))
    aggr = max(0.0, min(1.0, aggressiveness))
    level_p = float(np.clip(0.8 - 0.6 * aggr, 0.05, 0.95))
    climb_p = descend_p = (1.0 - level_p) / 2.0
    u = rng.uniform()
    level_sd = 50.0 + 200.0 * aggr
    climb_mean = 1000.0 + 300.0 * aggr
    descend_mean = -climb_mean
    climb_sd = 200.0 + 200.0 * aggr
    if u < level_p:
        return float(rng.normal(0.0, level_sd))
    elif u < level_p + climb_p:
        return float(rng.normal(climb_mean, climb_sd))
    else:
        return float(rng.normal(descend_mean, climb_sd))


# --------------------- Sense choice & non-compliance --------------------------


def simulate_miss_for_senses(
    tgo: float,
    dt: float,
    h0: float,
    cat_above: bool,
    sense_pl: int,
    sense_cat: int,
    pl_delay: float,
    pl_accel: float,
    pl_vs: float,
    pl_cap: float,
    pl_vs0: float,
    cat_delay: float,
    cat_accel: float,
    cat_vs: float,
    cat_cap: float,
    cat_vs0: float,
) -> float:
    times, vs_pl = vs_time_series(
        tgo,
        dt,
        pl_delay,
        pl_accel,
        pl_vs,
        sense=sense_pl,
        cap_fpm=pl_cap,
        vs0_fpm=pl_vs0,
    )
    _, vs_ca = vs_time_series(
        tgo,
        dt,
        cat_delay,
        cat_accel,
        cat_vs,
        sense=sense_cat,
        cap_fpm=cat_cap,
        vs0_fpm=cat_vs0,
    )
    z_pl = integrate_altitude_from_vs(times, vs_pl, 0.0)
    z_ca = integrate_altitude_from_vs(times, vs_ca, h0 if cat_above else -h0)
    return float(abs(z_ca[-1] - z_pl[-1]))


def choose_optimal_sense(
    tgo: float,
    dt: float,
    h0: float,
    cat_above: bool,
    pl_vs0: float,
    cat_vs0: float,
    cat_delay_nom: float = 5.0,
    cat_accel_nom: float = 0.25,
    cat_vs: float = CAT_INIT_VS_FPM,
    cat_cap: float = CAT_CAP_INIT_FPM,
    cat_profiles: Optional[Tuple[Dict[str, float], ...]] = None,
) -> Tuple[Tuple[int, int], float, float]:
    """Return the sense pair that minimises the expected miss distance.

    Parameters
    ----------
    tgo, dt, h0, cat_above, pl_vs0, cat_vs0
        Geometry and kinematic context for the current encounter.
    cat_delay_nom, cat_accel_nom, cat_vs, cat_cap
        Legacy nominal CAT response characteristics used when
        ``cat_profiles`` is not supplied.
    cat_profiles
        Optional sequence of intruder-response templates. Each template is a
        mapping supporting the keys ``delay``, ``accel``, ``vs``, ``cap`` and
        ``weight`` (all optional). When provided the function evaluates both
        sense options for every template and combines the miss distances using
        the supplied weights to represent the anticipated response profile.

    Returns
    -------
    ((int, int), float, float)
        The commanded sense pair (PL, CAT) together with the expected miss
        distance for the preferred sense and for the alternative sense.
    """

    if cat_profiles is None:
        cat_profiles_seq: Tuple[Dict[str, float], ...] = (
            {
                "delay": float(cat_delay_nom),
                "accel": float(cat_accel_nom),
                "vs": float(cat_vs),
                "cap": float(cat_cap),
                "weight": 1.0,
            },
        )
    else:
        cat_profiles_seq = tuple(cat_profiles)
        if not cat_profiles_seq:
            cat_profiles_seq = (
                {
                    "delay": float(cat_delay_nom),
                    "accel": float(cat_accel_nom),
                    "vs": float(cat_vs),
                    "cap": float(cat_cap),
                    "weight": 1.0,
                },
            )

    weighted_scores = {(+1, -1): 0.0, (-1, +1): 0.0}
    totals = {(+1, -1): 0.0, (-1, +1): 0.0}
    unweighted_samples = {(+1, -1): [], (-1, +1): []}

    for profile in cat_profiles_seq:
        delay = float(profile.get("delay", cat_delay_nom))
        accel = float(profile.get("accel", cat_accel_nom))
        vs_cmd = float(profile.get("vs", cat_vs))
        cap_cmd = float(profile.get("cap", cat_cap))
        weight = max(0.0, float(profile.get("weight", 1.0)))

        miss_updown = simulate_miss_for_senses(
            tgo,
            dt,
            h0,
            cat_above,
            +1,
            -1,
            PL_DELAY_MEAN_S,
            PL_ACCEL_G,
            PL_VS_FPM,
            PL_VS_CAP_FPM,
            pl_vs0,
            delay,
            accel,
            vs_cmd,
            cap_cmd,
            cat_vs0,
        )
        miss_downup = simulate_miss_for_senses(
            tgo,
            dt,
            h0,
            cat_above,
            -1,
            +1,
            PL_DELAY_MEAN_S,
            PL_ACCEL_G,
            PL_VS_FPM,
            PL_VS_CAP_FPM,
            pl_vs0,
            delay,
            accel,
            vs_cmd,
            cap_cmd,
            cat_vs0,
        )

        weighted_scores[(+1, -1)] += weight * miss_updown
        weighted_scores[(-1, +1)] += weight * miss_downup
        totals[(+1, -1)] += weight
        totals[(-1, +1)] += weight
        unweighted_samples[(+1, -1)].append(miss_updown)
        unweighted_samples[(-1, +1)].append(miss_downup)

    expected_scores = {}
    for senses in weighted_scores:
        total_weight = totals[senses]
        if total_weight > 0.0:
            expected_scores[senses] = weighted_scores[senses] / total_weight
        else:
            expected_scores[senses] = float(np.mean(unweighted_samples[senses]))

    miss_updown = expected_scores[(+1, -1)]
    miss_downup = expected_scores[(-1, +1)]

    if miss_updown > miss_downup:
        return (+1, -1), miss_updown, miss_downup
    else:
        return (-1, +1), miss_downup, miss_updown


@dataclass(frozen=True)
class OppositeSenseBand:
    """Probability overrides for an altitude window."""

    alt_min_ft: float
    alt_max_ft: float
    manual_prob: float
    apfd_prob: Optional[float] = None

    def contains(self, altitude_ft: float) -> bool:
        return self.alt_min_ft <= altitude_ft < self.alt_max_ft


@dataclass
class OppositeSenseModel:
    """Model describing opposite-sense priors for manual/APFD crews."""

    manual_baseline: float = 0.010
    apfd_baseline: Optional[float] = None
    altitude_bands: Tuple[OppositeSenseBand, ...] = ()
    jitter_enabled: bool = True
    jitter_range: Tuple[float, float] = (0.7, 1.3)

    def _baseline_for_mode(self, mode: str) -> float:
        if _is_apfd_mode(mode):
            return 0.0

        base = self.manual_baseline
        return float(np.clip(base, 0.0, 1.0))

    def _prob_from_bands(self, mode: str, altitude_ft: float) -> Optional[float]:
        if _is_apfd_mode(mode):
            return 0.0

        for band in self.altitude_bands:
            if band.contains(altitude_ft):
                prob = band.manual_prob
                return float(np.clip(prob, 0.0, 1.0))
        return None

    def probability(
        self,
        rng: np.random.Generator,
        mode: str,
        altitude_ft: Optional[float],
        jitter_override: Optional[bool] = None,
    ) -> float:
        """Return an opposite-sense probability for the given context."""

        if altitude_ft is None:
            altitude_ft = 0.0

        if _is_apfd_mode(mode):
            return 0.0

        prob = self._prob_from_bands(mode, altitude_ft)
        if prob is None:
            prob = self._baseline_for_mode(mode)

        jitter_enabled = self.jitter_enabled if jitter_override is None else bool(jitter_override)
        if jitter_enabled:
            lo, hi = self.jitter_range
            if hi < lo:
                lo, hi = hi, lo
            scale = float(rng.uniform(lo, hi))
            prob = prob * scale

        return float(np.clip(prob, 0.0, 1.0))

    @classmethod
    def from_parameters(
        cls,
        manual_baseline: float,
        apfd_baseline: Optional[float] = None,
        altitude_bands: Optional[Iterable[OppositeSenseBand]] = None,
        jitter_enabled: bool = True,
        jitter_range: Tuple[float, float] = (0.7, 1.3),
    ) -> "OppositeSenseModel":
        bands: Tuple[OppositeSenseBand, ...]
        if altitude_bands is None:
            bands = ()
        else:
            bands = tuple(altitude_bands)
        return cls(
            manual_baseline=float(manual_baseline),
            apfd_baseline=None if apfd_baseline is None else float(apfd_baseline),
            altitude_bands=bands,
            jitter_enabled=bool(jitter_enabled),
            jitter_range=jitter_range,
        )


def normalize_opposite_sense_bands(
    bands: Optional[Iterable[object]],
) -> Tuple[OppositeSenseBand, ...]:
    """Coerce heterogeneous definitions into ``OppositeSenseBand`` records."""

    if bands is None:
        return ()

    normalised: List[OppositeSenseBand] = []
    for entry in bands:
        if isinstance(entry, OppositeSenseBand):
            normalised.append(entry)
            continue

        alt_min: Optional[float] = None
        alt_max: Optional[float] = None
        manual_prob: Optional[float] = None
        apfd_prob: Optional[float] = None

        if isinstance(entry, dict):
            def pick(*names: str) -> Optional[float]:
                for name in names:
                    value = entry.get(name)
                    if value is not None:
                        return value
                return None

            alt_min = pick("alt_min_ft", "alt_min", "min", "lo")
            alt_max = pick("alt_max_ft", "alt_max", "max", "hi")
            manual_prob = pick("manual_prob", "manual")
            apfd_prob = pick("apfd_prob", "apfd")
        else:
            seq = list(entry)  # type: ignore[arg-type]
            if len(seq) >= 3:
                alt_min = seq[0]
                alt_max = seq[1]
                manual_prob = seq[2]
                if len(seq) >= 4:
                    apfd_prob = seq[3]

        if alt_min is None or alt_max is None or manual_prob is None:
            raise ValueError("Opposite-sense band definitions require alt_min, alt_max, and manual probability")

        normalised.append(
            OppositeSenseBand(
                alt_min_ft=float(alt_min),
                alt_max_ft=float(alt_max),
                manual_prob=float(manual_prob),
                apfd_prob=None if apfd_prob is None else float(apfd_prob),
            )
        )

    return tuple(normalised)


def apply_non_compliance_to_cat(
    rng: np.random.Generator,
    sense_cat: int,
    base_delay_s: float,
    base_accel_g: float,
    vs_fpm: float,
    cap_fpm: float,
    p_taonly: float = 0.003,
    p_weak: float = 0.300,
    jitter: bool = True,
    opposite_model: Optional[OppositeSenseModel] = None,
    cat_mode_key: str = "manual",
    cat_alt_ft: Optional[float] = None,
    mode_label_override: Optional[str] = None,
) -> Tuple[str, int, float, float, float, float]:
    if opposite_model is None:
        opposite_model = OppositeSenseModel()

    apfd_mode = _is_apfd_mode(cat_mode_key) or _is_apfd_mode(mode_label_override)
    if apfd_mode:
        p_opp = 0.0
    else:
        p_opp = opposite_model.probability(
            rng,
            mode=cat_mode_key,
            altitude_ft=cat_alt_ft,
            jitter_override=jitter,
        )

    if jitter:
        p_taonly = max(0.0, min(1.0, p_taonly * rng.uniform(0.7, 1.3)))
        p_weak = max(0.0, min(1.0, p_weak * rng.uniform(0.7, 1.3)))

    u = rng.uniform()

    def label_for(outcome: str) -> str:
        if mode_label_override is None:
            return outcome
        if outcome == "compliant":
            return mode_label_override
        return f"{outcome} ({mode_label_override})"

    if not apfd_mode and u < p_opp:
        compliant_accel = float(np.clip(base_accel_g, 0.20, 0.25))
        return (
            label_for("opposite-sense"),
            -sense_cat,
            base_delay_s,
            compliant_accel,
            vs_fpm,
            cap_fpm,
        )
    u -= p_opp
    if u < p_taonly:
        return (
            label_for("no-response"),
            sense_cat,
            base_delay_s,
            0.0,
            0.0,
            0.0,
        )
    u -= p_taonly
    if u < p_weak:
        return (
            label_for("weak-compliance"),
            sense_cat,
            base_delay_s + 1.0,
            float(np.clip(rng.uniform(0.10, 0.18), 0.10, 0.18)),
            float(np.clip(vs_fpm * rng.uniform(0.55, 0.75), 900.0, 1200.0)),
            float(np.clip(cap_fpm * rng.uniform(0.55, 0.80), 900.0, 1300.0)),
        )
    compliant_accel = 0.25
    return (
        label_for("compliant"),
        sense_cat,
        base_delay_s,
        compliant_accel,
        vs_fpm,
        cap_fpm,
    )


# ------------------------- Event classification -------------------------


def reversal_candidate_satisfies_alim(
    cpa_reverse: float, alim_ft: float, margin_ft: float
) -> bool:
    """Return True when a reversal projection meets the ALIM + margin gate."""

    reverse_threshold_ft = alim_ft + margin_ft
    return cpa_reverse >= reverse_threshold_ft


def classify_event(
    times: np.ndarray,
    z_pl: np.ndarray,
    z_ca: np.ndarray,
    vs_pl: np.ndarray,
    vs_ca: np.ndarray,
    tgo: float,
    alim_ft: float,
    margin_ft: float,
    sense_chosen_cat: int,
    sense_exec_cat: int,
    manual_case: bool = False,
    projection_decision_latency_s: Optional[float] = None,
    projection_cat_delay_s: Optional[float] = None,
) -> Tuple[str, float, float, float, Optional[str]]:
    """Return (event_label, minsep, sep@CPA, t_detect, event_detail)."""

    sep = np.abs(z_ca - z_pl)
    minsep = float(np.min(sep))
    sep_cpa = float(sep[-1])

    t_pl_move = first_move_time(times, vs_pl)
    t_ca_move = first_move_time(times, vs_ca)
    response_start = max(t_pl_move, t_ca_move)
    earliest_move = min(t_pl_move, t_ca_move)
    if math.isclose(response_start, earliest_move, abs_tol=1e-6):
        monitor_start = float(max(0.0, response_start))
    else:
        monitor_start = float(
            max(0.0, max(response_start, earliest_move + REVERSAL_MONITOR_DELAY_S))
        )

    event_detail: Optional[str] = None
    t_detect = float(times[-1])

    rel_rate = (vs_ca - vs_pl) / 60.0

    no_response_checked = False
    improvement_timer_s = 0.0
    hold_window_satisfied = False
    prev_time = float(times[0]) if times.size else 0.0

    if manual_case:
        reversal_enable_tau_s = REVERSAL_ENABLE_TAU_S
        improvement_hold_required_s = REVERSAL_IMPROVEMENT_HOLD_S
        hold_tau_requirement_s = REVERSAL_HOLD_DISABLE_TAU_S
        vs_gate_fraction = 0.7
    else:
        # AP/FD dynamics (0.9 s delay, 0.25 g accel) respond more slowly than
        # the manual template.  Allow the improvement timer to accumulate below
        # the 18 s guard and reduce the vertical-speed gate so that compliant
        # automation is not reversed while the geometry is improving.
        reversal_enable_tau_s = REVERSAL_ENABLE_TAU_S + 1.0
        improvement_hold_required_s = max(0.9, 0.6 * REVERSAL_IMPROVEMENT_HOLD_S)
        hold_tau_requirement_s = 12.0
        vs_gate_fraction = 0.55

    for idx, t_now in enumerate(times):
        rel_now = float(rel_rate[idx])
        sep_now = float(sep[idx])
        approaching = rel_now < 0
        tau_now = math.inf
        if approaching:
            closure = -rel_now
            if closure > 1e-6:
                tau_now = sep_now / closure
            else:
                tau_now = math.inf

        if manual_case and sense_chosen_cat == sense_exec_cat and not no_response_checked:
            no_response_gate = max(
                NO_RESPONSE_ESCALATION_S,
                response_start + REVERSAL_MONITOR_DELAY_S,
            )
            if approaching and t_now >= no_response_gate:
                no_response_checked = True
                vs_toward_command = float(vs_ca[idx] * sense_chosen_cat)
                if vs_toward_command <= NO_RESPONSE_VS_THRESH_FPM:
                    t_strengthen = float(t_now)
                    return ("STRENGTHEN", minsep, sep_cpa, t_strengthen, "EXIGENT_STRENGTHEN")

        if t_now < monitor_start:
            prev_time = float(t_now)
            continue

        if not approaching:
            prev_time = float(t_now)
            continue

        vs_now = float(vs_ca[idx])
        if abs(vs_now) <= VS_SENSE_DEADBAND_FPM:
            sense_flown = 0
        else:
            sense_flown = 1 if vs_now > 0.0 else -1

        vs_toward_command = float(vs_now * sense_chosen_cat)

        wrong_sense = sense_flown != 0 and sense_flown != sense_chosen_cat

        if wrong_sense:
            t_detect = float(t_now)
            event_detail = "Exigent wrong-sense"
            return ("REVERSE", minsep, sep_cpa, t_detect, event_detail)

        same_sense = sense_flown == sense_chosen_cat and sense_flown != 0

        if math.isinf(tau_now):
            pred_miss = float("inf")
        else:
            pred_miss = abs(sep_now + rel_now * tau_now)

        # Strengthen if predicted miss is close to ALIM (with pad)
        strengthen_threshold = alim_ft + STRENGTHEN_PAD_FT
        if same_sense and pred_miss <= strengthen_threshold:
            t_strengthen = float(t_now)
            return ("STRENGTHEN", minsep, sep_cpa, t_strengthen, None)

        if (
            same_sense
            and tau_now <= EXIGENT_STRENGTHEN_TAU_S
            and vs_toward_command <= NO_RESPONSE_VS_THRESH_FPM
        ):
            t_strengthen = float(t_now)
            return (
                "STRENGTHEN",
                minsep,
                sep_cpa,
                t_strengthen,
                "EXIGENT_STRENGTHEN",
            )

        #Otherwise apply the riginal thin-pred gate
        thin_pred = pred_miss < alim_ft
        if not thin_pred:
            improvement_timer_s = 0.0
            hold_window_satisfied = False
            prev_time = float(t_now)
            continue

        t_detect = float(t_now)
        t_lb = max(monitor_start, t_now - REVERSAL_INTERLOCK_LOOKBACK_S)
        if t_lb < t_now:
            sep_lb = float(np.interp(t_lb, times, sep))
            rel_lb = float(np.interp(t_lb, times, rel_rate))
            if rel_lb < -1e-6:
                tau_lb = sep_lb / -rel_lb
                pred_miss_lb = abs(sep_lb + rel_lb * tau_lb)
            else:
                pred_miss_lb = abs(sep_lb)
        else:
            pred_miss_lb = pred_miss

        improving = pred_miss > pred_miss_lb + PREDICTED_MISS_IMPROVEMENT_TOL_FT

        achieved_vs = max(0.0, vs_toward_command) if same_sense else 0.0
        enough_vs = achieved_vs >= vs_gate_fraction * CAT_INIT_VS_FPM

        dt_sample = float(t_now - prev_time) if idx > 0 else 0.0
        tracking_improvement = same_sense and improving
        if tracking_improvement:
            improvement_timer_s += dt_sample
        else:
            improvement_timer_s = 0.0
            hold_window_satisfied = False

        if tracking_improvement and improvement_timer_s >= improvement_hold_required_s:
            if manual_case and tau_now >= REVERSAL_HOLD_DISABLE_TAU_S:
                hold_window_satisfied = True
            elif not manual_case and tau_now >= hold_tau_requirement_s:
                hold_window_satisfied = True

        if tau_now > reversal_enable_tau_s:
            prev_time = float(t_now)
            continue

        hold_active = tracking_improvement and hold_window_satisfied
        if hold_active:
            prev_time = float(t_now)
            continue

        if not same_sense:
            if sense_flown == 0:
                if t_now < response_start + REVERSAL_MONITOR_DELAY_S:
                    prev_time = float(t_now)
                    continue
                if tau_now <= EXIGENT_STRENGTHEN_TAU_S:
                    t_strengthen = float(t_now)
                    return (
                        "STRENGTHEN",
                        minsep,
                        sep_cpa,
                        t_strengthen,
                        "EXIGENT_STRENGTHEN",
                    )
                prev_time = float(t_now)
                continue
            event_detail = "Opposite sense"
            return ("REVERSE", minsep, sep_cpa, t_detect, event_detail)

        # Same-sense thin prediction: compare projected CPA for continuing vs reversing.
        t_remaining = float(times[-1] - t_now)
        times_future = times[idx:]
        z_pl_future = z_pl[idx:]
        cpa_continue: Optional[float] = None
        cpa_reverse: Optional[float] = None
        if t_remaining > 1e-6 and times_future.size >= 2:
            dt_future = float(np.min(np.diff(times_future)))
            if dt_future <= 1e-6 and times.size >= 2:
                dt_future = float(max(1e-3, np.min(np.diff(times))))
            dt_future = float(max(1e-3, dt_future))

            z_ca_now = float(z_ca[idx])
            vs_ca_now = float(vs_ca[idx])

            sense_current = sense_flown
            target_mag = float(
                min(
                    max(abs(vs_ca_now), CAT_INIT_VS_FPM),
                    CAT_CAP_INIT_FPM,
                )
            )
            accel_proj = (
                CAT_MANUAL_ACCEL_NOM_G if manual_case else CAT_APFD_ACCEL_NOM_G
            )

            if projection_decision_latency_s is None:
                latency_proj = 1.0
                latency_proj = float(np.clip(latency_proj, 0.6, 1.4))
            else:
                latency_proj = float(max(0.6, projection_decision_latency_s))
            if projection_cat_delay_s is None:
                cat_delay_proj = (
                    CAT_MANUAL_PROJECTION_DELAY_S
                    if manual_case
                    else CAT_APFD_PROJECTION_DELAY_S
                )
            else:
                cat_delay_proj = float(max(0.0, projection_cat_delay_s))
            response_effective_s = latency_proj + cat_delay_proj

            t_rel, vs_ca_continue = vs_time_series(
                t_remaining,
                dt_future,
                0.0,
                accel_proj,
                target_mag,
                sense=sense_current,
                cap_fpm=CAT_CAP_INIT_FPM,
                vs0_fpm=vs_ca_now,
            )
            vs_ca_reverse = vs_ca_continue.copy()

            if response_effective_s < t_remaining - 1e-6:
                vs_at_effective = float(np.interp(response_effective_s, t_rel, vs_ca_continue))
                target_signed = -sense_current * target_mag
                a = accel_proj * G
                a_fpm_s = a * FT_PER_M * 60.0

                for j, t_rel_j in enumerate(t_rel):
                    if t_rel_j <= response_effective_s + 1e-9:
                        continue
                    te = t_rel_j - response_effective_s
                    delta = target_signed - vs_at_effective
                    step = math.copysign(min(abs(a_fpm_s * te), abs(delta)), delta)
                    vs_ca_reverse[j] = vs_at_effective + step

            z_ca_continue = integrate_altitude_from_vs(t_rel, vs_ca_continue, z_ca_now)
            z_ca_reverse = integrate_altitude_from_vs(t_rel, vs_ca_reverse, z_ca_now)

            times_abs = t_now + t_rel
            z_pl_interp = np.interp(times_abs, times_future, z_pl_future)

            sep_continue = np.abs(z_pl_interp - z_ca_continue)
            sep_reverse = np.abs(z_pl_interp - z_ca_reverse)

            cpa_continue = float(np.min(sep_continue))
            cpa_reverse = float(np.min(sep_reverse))

            if not reversal_candidate_satisfies_alim(cpa_reverse, alim_ft, margin_ft):
                prev_time = float(t_now)
                continue

            if cpa_continue <= cpa_reverse + REVERSAL_CPA_IMPROVEMENT_TOL_FT:
                prev_time = float(t_now)
                continue

        if cpa_reverse is None or cpa_continue is None:
            prev_time = float(t_now)
            continue

        if enough_vs:
            event_detail = "Geometry shortfall"
        else:
            event_detail = "Slow response"
        return ("REVERSE", minsep, sep_cpa, t_detect, event_detail)

    return ("NONE", minsep, sep_cpa, t_detect, None)


def apply_second_phase(
    times: np.ndarray,
    vs_pl: np.ndarray,
    vs_ca: np.ndarray,
    tgo: float,
    dt: float,
    eventtype: str,
    sense_pl: int,
    sense_cat_exec: int,
    sense_cat_cmd: int,
    pl_vs0: float,
    cat_vs0: float,
    t_classify: float,
    z_pl_t2: float,
    z_cat_t2: float,
    pl_delay: float = PL_DELAY_MEAN_S,
    pl_accel_g: float = PL_ACCEL_G,
    pl_cap: float = PL_VS_CAP_FPM,
    cat_delay: float = 0.9,
    cat_accel_g: float = 0.35,
    cat_vs_strength: float = CAT_STRENGTH_FPM,
    cat_cap: float = CAT_CAP_STRENGTH_FPM,
    decision_latency_s: float = 1.0,
    cat_mode: str = "compliant",
    force_exigent: bool = False,
) -> Tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    Optional[float],
    int,
    int,
    int,
]:
    """Execute STRENGTHEN/REVERSE and continue the kinematics until CPA.

    The function returns the updated kinematic time series together with the
    post-manoeuvre senses for the performance-limited ownship, the executed
    intruder response, and the commanded intruder sense.
    """

    if eventtype not in ("STRENGTHEN", "REVERSE"):
        return times, vs_pl, vs_ca, None, sense_pl, sense_cat_exec, sense_cat_cmd

    latency = float(max(0.6, decision_latency_s))
    t2_issue = float(max(0.0, min(tgo, t_classify + latency)))
    t_rem = max(0.0, tgo - t2_issue)
    if t_rem <= dt:
        return (
            times,
            vs_pl,
            vs_ca,
            t2_issue,
            sense_pl,
            sense_cat_exec,
            sense_cat_cmd,
        )

    vs_pl_now = float(np.interp(t2_issue, times, vs_pl))
    vs_ca_now = float(np.interp(t2_issue, times, vs_ca))
    z_pl_now = float(z_pl_t2)
    z_ca_now = float(z_cat_t2)

    mode_key = (cat_mode or "").lower().strip()
    canonical_mode = mode_key.replace(" ", "").replace("/", "")
    if eventtype == "STRENGTHEN":
        cat_accel_eff = cat_accel_g
        cat_vs_eff = cat_vs_strength
        cat_cap_eff = cat_cap
        exigent_active = force_exigent or (t_rem <= EXIGENT_STRENGTHEN_TAU_S)

        if canonical_mode in {"compliant", "apfd"}:
            cat_accel_eff = max(cat_accel_eff, 0.35)
            cat_vs_eff = max(cat_vs_eff, CAT_STRENGTH_FPM)
            cat_cap_eff = max(cat_cap_eff, CAT_CAP_STRENGTH_FPM)
        elif "weak" in mode_key:
            cat_accel_eff = 0.25
            cat_vs_eff = max(cat_vs_eff, CAT_STRENGTH_FPM)
            cat_cap_eff = max(cat_cap_eff, CAT_CAP_STRENGTH_FPM)
        else:
            if exigent_active:
                cat_accel_eff = max(cat_accel_eff, 0.25)

        if exigent_active:
            cat_accel_eff = max(cat_accel_eff, 0.35)
            cat_vs_eff = max(cat_vs_eff, CAT_CAP_STRENGTH_FPM)
            cat_cap_eff = max(cat_cap_eff, CAT_CAP_STRENGTH_FPM)
    else:
        cat_accel_eff = cat_accel_g
        cat_vs_eff = cat_vs_strength
        cat_cap_eff = cat_cap

    pl_delay_eff = PL_DELAY_MEAN_S
    pl_accel_eff = PL_ACCEL_G
    pl_vs_eff = PL_VS_FPM
    pl_cap_eff = PL_VS_CAP_FPM

    baseline_sense_pl = int(math.copysign(1.0, sense_pl)) if sense_pl != 0 else 1

    if eventtype == "REVERSE":
        candidate_configs = [
            {
                "label": "flip",
                "sense_pl": baseline_sense_pl,
                "sense_cat_exec": -sense_cat_exec,
                "sense_cat_cmd": -sense_cat_cmd,
            },
            {
                "label": "keep",
                "sense_pl": baseline_sense_pl,
                "sense_cat_exec": sense_cat_exec,
                "sense_cat_cmd": sense_cat_cmd,
            },
        ]
    else:
        candidate_configs = [
            {
                "label": "default",
                "sense_pl": baseline_sense_pl,
                "sense_cat_exec": sense_cat_exec,
                "sense_cat_cmd": sense_cat_cmd,
            }
        ]

    for cfg in candidate_configs:
        t_rel, vs_pl_candidate = vs_time_series(
            t_rem,
            dt,
            pl_delay_eff,
            pl_accel_eff,
            pl_vs_eff,
            sense=cfg["sense_pl"],
            cap_fpm=pl_cap_eff,
            vs0_fpm=vs_pl_now,
        )
        _, vs_ca_candidate = vs_time_series(
            t_rem,
            dt,
            cat_delay,
            cat_accel_eff,
            cat_vs_eff,
            sense=cfg["sense_cat_exec"],
            cap_fpm=cat_cap_eff,
            vs0_fpm=vs_ca_now,
        )
        z_pl_candidate = integrate_altitude_from_vs(t_rel, vs_pl_candidate, z_pl_now)
        z_ca_candidate = integrate_altitude_from_vs(t_rel, vs_ca_candidate, z_ca_now)
        sep_candidate = np.abs(z_pl_candidate - z_ca_candidate)
        cfg["cpa_sep"] = float(np.min(sep_candidate))
        cfg["times_rel"] = t_rel
        cfg["vs_pl_profile"] = vs_pl_candidate
        cfg["vs_ca_profile"] = vs_ca_candidate

    if eventtype == "REVERSE":
        flip_cfg = next(cfg for cfg in candidate_configs if cfg["label"] == "flip")
        keep_cfg = next(cfg for cfg in candidate_configs if cfg["label"] == "keep")
        improvement = float(flip_cfg["cpa_sep"] - keep_cfg["cpa_sep"])
        if improvement > 1e-6:
            best_cfg = flip_cfg
        else:
            best_cfg = keep_cfg
    else:
        best_cfg = candidate_configs[0]

    new_sense_pl = int(best_cfg["sense_pl"])
    new_sense_cat_exec = int(best_cfg["sense_cat_exec"])
    new_sense_cat_cmd = int(best_cfg["sense_cat_cmd"])

    t2_rel = best_cfg["times_rel"]
    vs_pl_cont = best_cfg["vs_pl_profile"]
    vs_ca_cont = best_cfg["vs_ca_profile"]

    prefix_mask = times < (t2_issue - 1e-9)
    times_prefix = times[prefix_mask]
    vs_pl_prefix = vs_pl[prefix_mask]
    vs_ca_prefix = vs_ca[prefix_mask]

    times_prefix = np.append(times_prefix, t2_issue)
    vs_pl_prefix = np.append(vs_pl_prefix, vs_pl_now)
    vs_ca_prefix = np.append(vs_ca_prefix, vs_ca_now)

    times_suffix = t2_issue + t2_rel[1:]
    vs_pl_suffix = vs_pl_cont[1:]
    vs_ca_suffix = vs_ca_cont[1:]

    times2 = np.concatenate([times_prefix, times_suffix])
    vs_pl2 = np.concatenate([vs_pl_prefix, vs_pl_suffix])
    vs_ca2 = np.concatenate([vs_ca_prefix, vs_ca_suffix])
    return (
        times2,
        vs_pl2,
        vs_ca2,
        t2_issue,
        new_sense_pl,
        new_sense_cat_exec,
        new_sense_cat_cmd,
    )


# ------------------------------- Batch Runner -------------------------------


def sample_altitudes_and_h0(
    rng: np.random.Generator,
    fl_min: int = 150,
    fl_max: int = 300,
    h0_mean: float = 250.0,
    h0_sd: float = 100.0,
    h0_lo: float = 100.0,
    h0_hi: float = 500.0,
) -> Tuple[int, int, float]:
    FL_pl_base = int(rng.integers(fl_min, fl_max + 1))
    cap = 1000 if FL_pl_base < 290 else 2000
    h0 = float(np.clip(rng.normal(h0_mean, h0_sd), max(h0_lo, 0.0), min(h0_hi, cap)))
    diff_FL = max(1, int(round(h0 / 100.0)))
    sign = 1 if rng.uniform() < 0.5 else -1
    FL_cat_cand = FL_pl_base + sign * diff_FL
    if not (fl_min <= FL_cat_cand <= fl_max):
        FL_cat_cand = FL_pl_base - sign * diff_FL
        FL_cat_cand = min(fl_max, max(fl_min, FL_cat_cand))
    return int(FL_pl_base), int(FL_cat_cand), float(h0)


def run_batch(
    runs: int = 1000,
    seed: int = 26,
    scenario: str = "Head-on",
    r0_min_nm: float = 4.0,
    r0_max_nm: float = 8.0,
    aggressiveness: float = 0.30,
    p_opp: float = 0.010,
    p_ta: float = 0.003,
    p_weak: float = 0.300,
    jitter_priors: bool = True,
    opp_sense_apfd: Optional[float] = None,
    opp_sense_bands: Optional[Iterable[object]] = None,
    opp_sense_model: Optional[OppositeSenseModel] = None,
    apfd_share: float = 0.25,
    use_delay_mixture: bool = True,
    dt: float = 0.1,
    hdg1_min: float = 0.0,
    hdg1_max: float = 360.0,
    hdg2_min: float = 0.0,
    hdg2_max: float = 360.0,
    alim_override_ft: Optional[float] = None,
    use_custom_tgo: bool = False,
    tgo_min_s: Optional[float] = None,
    tgo_max_s: Optional[float] = None,
    apfd_mode: str = "custom",
    force_cat_ias_250: bool = False,
) -> pd.DataFrame:
    rng = np.random.default_rng(int(seed))
    data: List[Dict] = []

    if scenario == "Head-on":
        rel_min, rel_max = 150.0, 210.0
    elif scenario == "Crossing":
        rel_min, rel_max = 60.0, 120.0
    elif scenario == "Overtaking":
        rel_min, rel_max = 0.0, 30.0
    else:  # Custom
        rel_min = rel_max = None

    mode_key = apfd_mode.lower().strip()
    if mode_key in {"mixed", "mixed global", "mixed global traffic"}:
        apfd_mode_key = "mixed"
        apfd_share_effective = 0.10
    elif mode_key in {"airbus", "airbus-centric", "airbus centric"}:
        apfd_mode_key = "airbus"
        apfd_share_effective = 0.30
    else:
        apfd_mode_key = "custom"
        apfd_share_effective = float(np.clip(apfd_share, 0.0, 1.0))

    if use_custom_tgo:
        lo_user, hi_user, mode_user = sanitize_tgo_bounds(tgo_min_s, tgo_max_s)
    else:
        lo_user = hi_user = mode_user = None

    if opp_sense_model is not None:
        opposite_model = opp_sense_model
    else:
        opposite_model = OppositeSenseModel.from_parameters(
            manual_baseline=float(p_opp),
            apfd_baseline=None if opp_sense_apfd is None else float(opp_sense_apfd),
            altitude_bands=normalize_opposite_sense_bands(opp_sense_bands),
        )

    for k in range(int(runs)):
        FL_PL, FL_CAT, h0 = sample_altitudes_and_h0(rng)
        cat_above = (FL_CAT > FL_PL) if (FL_CAT != FL_PL) else (rng.uniform() < 0.5)

        PL_TAS = ias_to_tas(PL_IAS_KT, FL_PL * 100.0)
        if force_cat_ias_250:
            ref_alt_ft = min(FL_CAT * 100.0, 10000.0)
            CAT_TAS = ias_to_tas(250.0, ref_alt_ft)
        else:
            CAT_TAS = float(rng.uniform(420.0, 470.0))
        if scenario == "Custom":
            h1 = float(rng.uniform(hdg1_min, hdg1_max))
            h2 = float(rng.uniform(hdg2_min, hdg2_max))
        else:
            h1, h2 = sample_headings(rng, scenario, 0.0, 360.0, rel_min, rel_max)
        vcl = relative_closure_kt(PL_TAS, h1, CAT_TAS, h2)
        if use_custom_tgo and vcl > 1e-6:
            lo = float(lo_user)
            hi = float(hi_user)
            mode = float(np.clip(mode_user, lo + 1e-3, hi - 1e-3))
            tgo = float(rng.triangular(lo, mode, hi))
            r0 = (vcl * tgo) / 3600.0
        else:
            r0 = float(rng.uniform(min(r0_min_nm, r0_max_nm), max(r0_min_nm, r0_max_nm)))
            tgo_geom = time_to_go_from_geometry(r0, vcl)
            if use_custom_tgo:
                lo = float(lo_user)
                hi = float(hi_user)
                mode = float(mode_user)
            else:
                lo = TGO_MIN_S
                hi = TGO_MAX_S
                if scenario == "Head-on":
                    mu, sd = 25.0, 5.0
                elif scenario == "Crossing":
                    mu, sd = 22.0, 6.0
                else:
                    mu, sd = 30.0, 8.0
            if tgo_geom is not None:
                hi = min(hi, tgo_geom)
            if use_custom_tgo:
                hi = float(np.clip(hi, lo + 1e-3, TGO_MAX_S))
                if hi <= lo + 1e-3:
                    hi = min(TGO_MAX_S, lo + 1.0)
                mode = float(np.clip(mode, lo + 1e-3, hi - 1e-3))
                tgo = float(rng.triangular(lo, mode, hi))
            else:
                hi = float(np.clip(hi, lo + 0.5, TGO_MAX_S))
                if hi <= lo + 1e-3:
                    hi = min(TGO_MAX_S, lo + 1.0)
                tgo = float(np.clip(rng.normal(mu, sd), lo, hi))
            if use_custom_tgo and vcl <= 1e-6:
                # Degenerate geometry; retain user-specified range settings.
                r0 = float(rng.uniform(min(r0_min_nm, r0_max_nm), max(r0_min_nm, r0_max_nm)))

        leveloff_context = aggressiveness <= 1e-6
        vz0_pl = sample_initial_vs_with_aggressiveness(rng, aggressiveness, leveloff_context)
        vz0_cat = sample_initial_vs_with_aggressiveness(rng, aggressiveness, leveloff_context)

        cat_delay_eff = float(np.clip(rng.normal(5.0, 1.5), 2.5, 8.0))
        if use_delay_mixture:
            cat_accel_eff = float(np.clip(rng.normal(0.24, 0.02), 0.18, 0.28))
        else:
            cat_accel_eff = float(np.clip(rng.normal(0.25, 0.01), 0.22, 0.28))

        manual_weight = max(0.0, 1.0 - apfd_share_effective)
        cat_profiles = [
            {
                "label": "manual",
                "delay": CAT_MANUAL_DELAY_NOM_S,
                "accel": CAT_MANUAL_ACCEL_NOM_G,
                "vs": CAT_INIT_VS_FPM,
                "cap": CAT_CAP_INIT_FPM,
                "weight": manual_weight,
            }
        ]
        if apfd_share_effective > 1e-6:
            cat_profiles.append(
                {
                    "label": "apfd",
                    "delay": CAT_APFD_DELAY_NOM_S,
                    "accel": CAT_APFD_ACCEL_NOM_G,
                    "vs": CAT_INIT_VS_FPM,
                    "cap": CAT_CAP_INIT_FPM,
                    "weight": float(np.clip(apfd_share_effective, 0.0, 1.0)),
                }
            )

        (sense_pl, sense_ca), miss_nominal, miss_alt = choose_optimal_sense(
            tgo,
            dt,
            h0,
            cat_above,
            vz0_pl,
            vz0_cat,
            cat_delay_nom=CAT_MANUAL_DELAY_NOM_S,
            cat_accel_nom=CAT_MANUAL_ACCEL_NOM_G,
            cat_vs=CAT_INIT_VS_FPM,
            cat_cap=CAT_CAP_INIT_FPM,
            cat_profiles=tuple(cat_profiles),
        )

        is_apfd = rng.uniform() < apfd_share_effective
        cat_is_apfd = bool(is_apfd)
        cat_mode_key = "apfd" if is_apfd else "manual"
        base_delay = CAT_APFD_DELAY_NOM_S if is_apfd else cat_delay_eff
        base_accel = CAT_APFD_ACCEL_NOM_G if is_apfd else cat_accel_eff
        (
            mode,
            sense_cat_exec,
            cat_delay_exec,
            cat_accel_exec,
            cat_vs_exec,
            cat_cap_exec,
        ) = apply_non_compliance_to_cat(
            rng,
            sense_ca,
            base_delay_s=base_delay,
            base_accel_g=base_accel,
            vs_fpm=CAT_INIT_VS_FPM,
            cap_fpm=CAT_CAP_INIT_FPM,
            p_taonly=p_ta,
            p_weak=p_weak,
            jitter=jitter_priors,
            opposite_model=opposite_model,
            cat_mode_key=cat_mode_key,
            cat_alt_ft=FL_CAT * 100.0,
            mode_label_override="AP/FD" if is_apfd else None,
        )

        pl_delay = max(0.0, rng.normal(PL_DELAY_MEAN_S, PL_DELAY_SD_S))

        times, vs_pl = vs_time_series(
            tgo,
            dt,
            pl_delay,
            PL_ACCEL_G,
            PL_VS_FPM,
            sense=sense_pl,
            cap_fpm=PL_VS_CAP_FPM,
            vs0_fpm=vz0_pl,
        )
        _, vs_ca = vs_time_series(
            tgo,
            dt,
            cat_delay_exec,
            cat_accel_exec,
            cat_vs_exec,
            sense=sense_cat_exec,
            cap_fpm=cat_cap_exec,
            vs0_fpm=vz0_cat,
        )
        z_pl = integrate_altitude_from_vs(times, vs_pl, 0.0)
        z_ca = integrate_altitude_from_vs(times, vs_ca, h0 if cat_above else -h0)

        alim_ft = alim_ft_from_alt(FL_PL * 100.0, override_ft=alim_override_ft)

        manual_case = not cat_is_apfd

        current_times = times
        current_vs_pl = vs_pl
        current_vs_ca = vs_ca
        current_z_pl = z_pl
        current_z_ca = z_ca
        baseline_sense_pl = int(math.copysign(1.0, sense_pl)) if sense_pl != 0 else 1
        current_sense_pl = baseline_sense_pl
        current_sense_cat_exec = sense_cat_exec
        current_sense_chosen = sense_ca

        eval_start_time = 0.0
        maneuver_sequence: List[Dict[str, object]] = []
        eventtype_initial: Optional[str] = None
        event_detail_initial: Optional[str] = None
        t_detect_initial: Optional[float] = None
        tau_detect_initial: Optional[float] = None
        final_eventtype: Optional[str] = None
        final_event_detail: Optional[str] = None
        final_event_detail_override = False
        final_t_detect: Optional[float] = None
        final_tau_detect: Optional[float] = None
        t_second_issue: Optional[float] = None
        tau_second_issue_s: Optional[float] = None
        reversal_observed = False
        reversal_detail: Optional[str] = None
        reversal_t_detect: Optional[float] = None
        reversal_tau_detect: Optional[float] = None

        dropout_cycles_remaining = 1 if rng.random() < COORDINATION_DROPOUT_PROB else 0

        for phase in range(MAX_MANEUVER_PHASES):
            if current_times.size == 0:
                break

            if eval_start_time <= current_times[0] + 1e-9:
                start_idx = 0
            else:
                start_idx = int(np.searchsorted(current_times, eval_start_time - 1e-9, side="left"))

            if start_idx >= current_times.size:
                break

            stride = max(1, int(round(TCAS_UPDATE_PERIOD_S / dt)))
            offset = start_idx % stride
            first_cycle_idx = start_idx if offset == 0 else start_idx + (stride - offset)
            first_cycle_idx = min(first_cycle_idx, current_times.size - 1)
            cycle_indices = np.arange(first_cycle_idx, current_times.size, stride, dtype=int)
            if cycle_indices.size == 0:
                sample_indices = np.array([current_times.size - 1], dtype=int)
            else:
                sample_indices = cycle_indices
            final_idx = current_times.size - 1
            if sample_indices.size == 0 or sample_indices[-1] != final_idx:
                sample_indices = np.append(sample_indices, final_idx)
            sample_indices = np.unique(sample_indices)
            sample_indices = sample_indices[sample_indices >= start_idx]
            if sample_indices.size == 0:
                break

            times_eval = current_times[sample_indices]
            if times_eval.size == 0:
                break

            vs_pl_eval = current_vs_pl[sample_indices]
            vs_ca_eval = current_vs_ca[sample_indices]
            z_pl_eval = current_z_pl[sample_indices]
            z_ca_eval = current_z_ca[sample_indices]

            vs_pl_report = np.array(vs_pl_eval, copy=True, dtype=float)
            vs_ca_report = np.array(vs_ca_eval, copy=True, dtype=float)
            z_pl_report = np.array(z_pl_eval, copy=True, dtype=float)
            z_ca_report = np.array(z_ca_eval, copy=True, dtype=float)

            vs_pl_report += rng.normal(0.0, VS_REPORT_NOISE_SD_FPM, size=vs_pl_report.shape)
            vs_ca_report += rng.normal(0.0, VS_REPORT_NOISE_SD_FPM, size=vs_ca_report.shape)
            z_pl_report += rng.normal(0.0, ALT_REPORT_NOISE_SD_FT, size=z_pl_report.shape)
            z_ca_report += rng.normal(0.0, ALT_REPORT_NOISE_SD_FT, size=z_ca_report.shape)

            second_phase_cat_delay_base = 0.9 if cat_is_apfd else 2.5
            decision_latency = float(np.clip(rng.normal(1.0, 0.2), 0.6, 1.4))
            dropout_active_now = dropout_cycles_remaining > 0
            dropout_delay_s = TCAS_UPDATE_PERIOD_S if dropout_active_now else 0.0
            decision_latency_eff = decision_latency + dropout_delay_s
            second_phase_cat_delay_eff = second_phase_cat_delay_base + dropout_delay_s

            eventtype, _, _, t_detect, event_detail = classify_event(
                times_eval,
                z_pl_report,
                z_ca_report,
                vs_pl_report,
                vs_ca_report,
                tgo,
                alim_ft=alim_ft,
                margin_ft=ALIM_MARGIN_FT,
                sense_chosen_cat=current_sense_chosen,
                sense_exec_cat=current_sense_cat_exec,
                manual_case=manual_case,
                projection_decision_latency_s=decision_latency_eff,
                projection_cat_delay_s=second_phase_cat_delay_eff,
            )

            tau_detect = max(0.0, tgo - t_detect)

            if phase == 0:
                eventtype_initial = str(eventtype)
                event_detail_initial = event_detail
                t_detect_initial = float(t_detect)
                tau_detect_initial = float(tau_detect)

            final_eventtype = str(eventtype)
            final_event_detail = event_detail
            final_event_detail_override = False
            final_t_detect = float(t_detect)
            final_tau_detect = float(tau_detect)

            dropout_effective = dropout_active_now and eventtype in ("STRENGTHEN", "REVERSE")

            record = dict(
                phase=phase + 1,
                eventtype=str(eventtype),
                event_detail=event_detail,
                t_issue=float(t_detect),
                tau_issue=float(tau_detect),
                executed_flip=False,
                coordination_dropout=bool(dropout_effective),
            )
            maneuver_sequence.append(record)

            if dropout_effective:
                dropout_cycles_remaining = max(0, dropout_cycles_remaining - 1)

            if eventtype not in ("STRENGTHEN", "REVERSE"):
                break

            force_exigent = bool(event_detail == "EXIGENT_STRENGTHEN")
            latency = float(max(0.6, decision_latency_eff))
            t2_issue_est = float(max(0.0, min(tgo, t_detect + latency)))
            z_pl_t2 = float(np.interp(t2_issue_est, current_times, current_z_pl))
            z_cat_t2 = float(np.interp(t2_issue_est, current_times, current_z_ca))

            prior_sense_cat_exec = current_sense_cat_exec
            prior_sense_cat_cmd = current_sense_chosen

            (
                times2,
                vs_pl2,
                vs_ca2,
                t2_issue,
                new_sense_pl,
                new_sense_cat_exec,
                new_sense_cat_cmd,
            ) = apply_second_phase(
                current_times,
                current_vs_pl,
                current_vs_ca,
                tgo,
                dt,
                eventtype,
                current_sense_pl,
                current_sense_cat_exec,
                current_sense_chosen,
                pl_vs0=vz0_pl,
                cat_vs0=vz0_cat,
                t_classify=t_detect,
                z_pl_t2=z_pl_t2,
                z_cat_t2=z_cat_t2,
                pl_delay=pl_delay,
                pl_accel_g=PL_ACCEL_G,
                pl_cap=PL_VS_CAP_FPM,
                cat_delay=second_phase_cat_delay_eff,
                cat_accel_g=0.35,
                cat_vs_strength=CAT_STRENGTH_FPM,
                cat_cap=CAT_CAP_STRENGTH_FPM,
                decision_latency_s=decision_latency_eff,
                cat_mode=mode,
                force_exigent=force_exigent,
            )

            cat_exec_flipped = new_sense_cat_exec != prior_sense_cat_exec
            cat_cmd_flipped = new_sense_cat_cmd != prior_sense_cat_cmd
            executed_flip = bool(cat_exec_flipped or cat_cmd_flipped)
            record["executed_flip"] = executed_flip

            if new_sense_pl != baseline_sense_pl:
                new_sense_pl = baseline_sense_pl

            if eventtype == "REVERSE" and not executed_flip:
                issue_time = t2_issue if t2_issue is not None else t_detect
                if current_times.size:
                    issue_time = float(
                        np.clip(issue_time, float(current_times[0]), float(current_times[-1]))
                    )
                vs_ca_at_issue = float(np.interp(issue_time, current_times, current_vs_ca))
                post_mask = times2 > issue_time + 1e-9
                vs_ca_post = vs_ca2[post_mask]
                if vs_ca_post.size:
                    max_post_mag = float(np.max(np.abs(vs_ca_post)))
                else:
                    max_post_mag = abs(vs_ca_at_issue)
                strengthened = bool(max_post_mag > abs(vs_ca_at_issue) + 1.0)
                downgraded_type = "STRENGTHEN" if strengthened else "NONE"
                downgraded_detail = None

                record["eventtype"] = downgraded_type
                record["event_detail"] = downgraded_detail
                final_eventtype = downgraded_type
                final_event_detail = downgraded_detail
                final_event_detail_override = True
                final_t_detect = float(t_detect)
                final_tau_detect = float(tau_detect)

                if phase == 0:
                    eventtype_initial = downgraded_type
                    event_detail_initial = downgraded_detail
                    t_detect_initial = float(t_detect)
                    tau_detect_initial = float(tau_detect)

                eventtype = downgraded_type
                event_detail = downgraded_detail

            if eventtype == "REVERSE" and executed_flip:
                reversal_observed = True
                reversal_detail = event_detail
                reversal_t_detect = float(t_detect)
                reversal_tau_detect = float(tau_detect)

            if t2_issue is None:
                break

            if t_second_issue is None:
                t_second_issue = float(t2_issue)
                tau_second_issue_s = float(max(0.0, tgo - t_second_issue))

            current_times = times2
            current_vs_pl = vs_pl2
            current_vs_ca = vs_ca2
            current_z_pl = integrate_altitude_from_vs(current_times, current_vs_pl, 0.0)
            current_z_ca = integrate_altitude_from_vs(
                current_times,
                current_vs_ca,
                h0 if cat_above else -h0,
            )

            current_sense_pl = baseline_sense_pl
            current_sense_cat_exec = new_sense_cat_exec
            current_sense_chosen = new_sense_cat_cmd

            eval_start_time = float(t2_issue)

            if (tgo - t2_issue) <= dt or phase == MAX_MANEUVER_PHASES - 1:
                break

        if eventtype_initial is None:
            eventtype_initial = "NONE"
        if final_eventtype is None:
            final_eventtype = eventtype_initial
        last_entry: Optional[Dict[str, object]]
        if maneuver_sequence:
            last_entry = maneuver_sequence[-1]
            if not isinstance(last_entry, dict):
                last_entry = None
        else:
            last_entry = None
        last_eventtype_recorded = (
            str(last_entry.get("eventtype")) if last_entry and "eventtype" in last_entry else None
        )
        last_executed_flip = bool(last_entry.get("executed_flip")) if last_entry else False

        if (
            reversal_observed
            and final_eventtype != "REVERSE"
            and last_eventtype_recorded == "REVERSE"
            and last_executed_flip
        ):
            final_eventtype = "REVERSE"
            final_event_detail = reversal_detail
            final_t_detect = reversal_t_detect
            final_tau_detect = reversal_tau_detect
        if event_detail_initial is None:
            event_detail_initial = None
        if final_event_detail is None and not final_event_detail_override:
            final_event_detail = event_detail_initial
        if t_detect_initial is None:
            t_detect_initial = float(times[-1]) if times.size else 0.0
        if final_t_detect is None:
            final_t_detect = t_detect_initial
        if tau_detect_initial is None:
            tau_detect_initial = max(0.0, tgo - t_detect_initial)
        if final_tau_detect is None:
            final_tau_detect = tau_detect_initial

        times = current_times
        vs_pl = current_vs_pl
        vs_ca = current_vs_ca
        z_pl = current_z_pl
        z_ca = current_z_ca

        sep_trace = np.abs(z_ca - z_pl)
        minsep_ft = float(np.min(sep_trace))
        sep_cpa_ft = float(sep_trace[-1])
        miss_cpa_ft = float(abs(z_ca[-1] - z_pl[-1]))
        margin_trace = sep_trace - alim_ft
        margin_min_ft = float(np.min(margin_trace))
        margin_cpa_ft = float(sep_trace[-1] - alim_ft)
        alim_breach_cpa = bool(sep_trace[-1] <= alim_ft)
        band25_threshold = max(0.0, alim_ft - 25.0)
        band50_threshold = max(0.0, alim_ft - 50.0)
        band100_threshold = max(0.0, alim_ft - 100.0)
        alim_breach_band25 = bool(sep_trace[-1] <= band25_threshold)
        alim_breach_band50 = bool(sep_trace[-1] <= band50_threshold)
        alim_breach_band100 = bool(sep_trace[-1] <= band100_threshold)

        delta_pl = float(z_pl[-1] - z_pl[0])
        delta_cat = float(z_ca[-1] - z_ca[0])
        residual_risk = compute_residual_risk(delta_pl, delta_cat)

        comp_label = compliance_score_method_b_like(
            sense_required=sense_cat_exec,
            vs_trace=vs_ca,
            times=times,
        )

        history_json = encode_time_history(times, vs_pl, vs_ca, z_pl, z_ca)

        data.append(
            dict(
                run=k + 1,
                scenario=scenario,
                FL_PL=FL_PL,
                FL_CAT=FL_CAT,
                PL_TAS=PL_TAS,
                CAT_TAS=CAT_TAS,
                PLhdg=h1,
                CAThdg=h2,
                R0NM=r0,
                closurekt=vcl,
                tgos=tgo,
                h0ft=h0,
                aggressiveness=aggressiveness,
                leveloff=int(leveloff_context),
                cat_above=bool(cat_above),
                sensePL=sense_pl,
                senseCAT_chosen=sense_ca,
                CAT_mode=mode,
                senseCAT_exec=sense_cat_exec,
                sensePL_final=int(current_sense_pl),
                senseCAT_exec_final=int(current_sense_cat_exec),
                senseCAT_chosen_final=int(current_sense_chosen),
                pl_vs0_init=vz0_pl,
                cat_vs0_init=vz0_cat,
                plDelay=pl_delay,
                plAccel_g=PL_ACCEL_G,
                catDelay=cat_delay_exec,
                catAccel_g=cat_accel_exec,
                catVS_cmd=cat_vs_exec,
                catCap_cmd=cat_cap_exec,
                ALIM_ft=alim_ft,
                missCPAft=miss_cpa_ft,
                minsepft=minsep_ft,
                sep_cpa_ft=sep_cpa_ft,
                margin_min_ft=margin_min_ft,
                margin_cpa_ft=margin_cpa_ft,
                alim_breach_cpa=alim_breach_cpa,
                alim_breach_cpa_band25=alim_breach_band25,
                alim_breach_cpa_band50=alim_breach_band50,
                alim_breach_cpa_band100=alim_breach_band100,
                eventtype=eventtype_initial,
                event_detail=event_detail_initial,
                t_detect=t_detect_initial,
                tau_detect=tau_detect_initial,
                t_second_issue=t_second_issue,
                tau_second_issue=tau_second_issue_s,
                eventtype_final=final_eventtype,
                event_detail_final=final_event_detail,
                t_detect_final=final_t_detect,
                tau_detect_final=final_tau_detect,
                maneuver_sequence=tuple(maneuver_sequence),
                any_reversal=int(reversal_observed),
                comp_label=comp_label,
                CAT_is_APFD=int(cat_is_apfd),
                residual_risk=residual_risk,
                delta_h_pl_ft=delta_pl,
                delta_h_cat_ft=delta_cat,
                time_history_json=history_json,
            )
        )

    return pd.DataFrame(data)


__all__ = [
    # constants
    "G",
    "FT_PER_M",
    "MS_PER_FPM",
    "PL_DELAY_MEAN_S",
    "PL_DELAY_SD_S",
    "PL_ACCEL_G",
    "PL_VS_FPM",
    "PL_VS_CAP_FPM",
    "PL_IAS_KT",
    "CAT_INIT_VS_FPM",
    "CAT_STRENGTH_FPM",
    "CAT_CAP_INIT_FPM",
    "CAT_CAP_STRENGTH_FPM",
    "CAT_MANUAL_DELAY_NOM_S",
    "CAT_MANUAL_ACCEL_NOM_G",
    "CAT_APFD_DELAY_NOM_S",
    "CAT_APFD_ACCEL_NOM_G",
    "TGO_MIN_S",
    "TGO_MAX_S",
    "TCAS_UPDATE_PERIOD_S",
    "COORDINATION_DROPOUT_PROB",
    "ALIM_MARGIN_FT",
    "REVERSAL_MONITOR_DELAY_S",
    "REVERSAL_ENABLE_TAU_S",
    "EXIGENT_STRENGTHEN_TAU_S",
    "NO_RESPONSE_ESCALATION_S",
    "NO_RESPONSE_VS_THRESH_FPM",
    "REVERSAL_INTERLOCK_LOOKBACK_S",
    "REVERSAL_IMPROVEMENT_HOLD_S",
    "REVERSAL_HOLD_DISABLE_TAU_S",
    "PREDICTED_MISS_IMPROVEMENT_TOL_FT",
    "ALIM_BANDS_FT",
    "MAX_MANEUVER_PHASES",
    # helpers
    "ias_to_tas",
    "compute_residual_risk",
    "vs_time_series",
    "integrate_altitude_from_vs",
    "encode_time_history",
    "decode_time_history",
    "extend_history_with_pretrigger",
    "relative_closure_kt",
    "time_to_go_from_geometry",
    "sample_headings",
    "alim_ft_from_alt",
    "first_move_time",
    "compliance_score_method_b_like",
    "sanitize_tgo_bounds",
    "sample_initial_vs_with_aggressiveness",
    "simulate_miss_for_senses",
    "choose_optimal_sense",
    "apply_non_compliance_to_cat",
    "classify_event",
    "apply_second_phase",
    "sample_altitudes_and_h0",
    "run_batch",
]

