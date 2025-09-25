import streamlit as st
import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
 
# -----------------------------
# Constants & helpers
# -----------------------------
G = 9.80665
FTPERM = 3.28084
 
DEFAULTALIMFT = 600.0
DEFAULTRESPTHRESHFPM = 300.0   # Eurocontrol/IATA FDM meaningful response threshold
REVERSALDWELLS = 1.5
ALIMMARGINFT = 100.0
 
# --- piecewise vertical kinematics (paper's model) ---
def deltahpiecewise(tcpas, tds, agg, vffpm):
    """Return displacement in feet from RA to CPA for one aircraft."""
    a = agg G                         # m/s^2
    vfmps = vffpm 0.00508            # m/s
    if tcpas <= tds:
        dhm = 0.0
    else:
        t = tcpas - tds
        tr = vfmps / a if a > 0 else np.inf
        if t <= tr:
            dhm = 0.5 a t2
        else:
            dhm = 0.5 a tr2 + vfmps (t - tr)
    return dhm FTPERM               # ft
 
def vstimeseries(tend, dt, tds, agg, vffpm, sense, capfpm=None):
    """VS profile (fpm) with delay, ramp at ag to vf, then cap; signed with sense."""
    a = agg G
    vftarget = vffpm if capfpm is None else min(vffpm, capfpm)
    times = np.arange(0, tend + 1e-9, dt)
    vs = np.zeroslike(times, dtype=float)
    for i, t in enumerate(times):
        if t <= tds:
            vs[i] = 0.0
        else:
            te = t - tds
            vmps = min(a te, vftarget 0.00508)  # m/s
            vs[i] = sense (vmps / 0.00508)         # fpm
    return times, vs
 
def predicteddhlinear(vsownfpm, vsintfpm, tgos):
    """Linear projection of vertical miss distance over time-to-go."""
    return abs((vsownfpm - vsintfpm) (tgos/60.0))
 
def compliantinsense(vsfpm, sense, thr=DEFAULTRESPTHRESHFPM):
    return (vsfpm sense) >= thr
 
def relativeclosurekt(v1kt, hdg1deg, v2kt, hdg2deg):
    """Compute scalar closure rate from headings/speeds (kt)."""
    th1 = np.deg2rad(hdg1deg)
    th2 = np.deg2rad(hdg2deg)
    v1 = np.array([v1ktnp.sin(th1), v1ktnp.cos(th1)])
    v2 = np.array([v2ktnp.sin(th2), v2ktnp.cos(th2)])
    return np.linalg.norm(v1 - v2)
 
def timetogofromgeometry(r0nm, vclosurekt):
    if vclosurekt <= 1e-6:
        return None
    return 3600.0 (r0nm / vclosurekt)
 
# --- v7.1 surrogate: prefer strengthening; reverse for wrong-direction or late-chase ---
def surrogatedecision(times, vsown, vsint, tcpas, respthr=DEFAULTRESPTHRESHFPM):
    """
    Decide first event: 'STRENGTHEN', 'REVERSE', or none, based on:
      1) compliance check,
      2) predicted ALIM shortfall,
      3) vertical-chase late geometry.
    We assume initial coordinated opposite senses: ownship = climb (+), intruder = descend (-).
    """
    events = []
    for i, t in enumerate(times):
        tgo = max(0.0, tcpas - t)
        if tgo <= 0:
            break
        ownok = compliantinsense(vsown[i], +1, thr=respthr)
        intok = compliantinsense(vsint[i], -1, thr=respthr)
        dhpred = predicteddhlinear(vsown[i], vsint[i], tgo)
 
        if not intok and ownok and t >= REVERSALDWELLS:
            events.append(("REVERSE", float(t)))
            break
        elif dhpred < (DEFAULTALIMFT - ALIMMARGINFT):
            if tgo < 6.0:
                events.append(("REVERSE", float(t)))
                break
            else:
                events.append(("STRENGTHEN", float(t)))
                break
    return events
 
def baselinedhft(tcpas, mode="IDEAL"):
    """Δh for the baseline 1500 fpm aircraft at same tcpas (for risk scaling)."""
    if mode == "IDEAL":
        return deltahpiecewise(tcpas, tds=1.0, agg=0.25, vffpm=1500)
    else:
        return deltahpiecewise(tcpas, tds=5.0, agg=0.25, vffpm=1500)
 
# -----------------------------
# Streamlit UI
# -----------------------------
st.title("ACAS/TCAS v7.1 — Residual Risk & RA Taxonomy (Batch Monte Carlo)")
 
st.markdown("""
This tool implements your paper’s piecewise vertical-response model to compute ΔhCPA and scale
unresolved residual risk using the ACASA 1.1% benchmark, then applies a light v7.1 surrogate
to tag strengthening vs reversal events.
 
Important limitations: This model flags ALIM breaches at CPA only. True “unresolved” in ACASA
considers minima anywhere along the trajectory. Use these results as a proxy consistent with your paper’s method.
""")
 
with st.sidebar:
    st.header("Global settings")
    alimft = st.numberinput("ALIM (ft)", value=600.0, step=50.0)
    baseline = st.selectbox("Baseline for risk scaling", ["IDEAL 1500 fpm (ACASA 2002)", "STANDARD 1500 fpm (EUROCONTROL 2018)"])
    dt = st.numberinput("Time step (s)", value=0.1, step=0.1, format="%.1f")
    respthr = st.numberinput("Meaningful response threshold (fpm)", value=300.0, step=50.0)
 
st.subheader("Ownship (Performance-limited) parameters")
c1, c2, c3 = st.columns(3)
with c1:
    plvs = st.numberinput("PL target VS (fpm)", value=500, step=50)
    plag = st.numberinput("PL vertical accel (g)", value=0.10, step=0.01, format="%.2f")
    pltd = st.numberinput("PL RA delay (s)", value=0.0, step=0.5)
with c2:
    pltas = st.numberinput("PL TAS (kt)", value=120.0, step=10.0)
    plcap = st.numberinput("PL performance cap (fpm)", value=500, step=50)
with c3:
    st.write("Assumed sense: Climb (+) for PL; Intruder opposite (−).")
 
st.subheader("Intruder (CAT) parameters")
d1, d2, d3 = st.columns(3)
with d1:
    catvs = st.numberinput("CAT target VS (fpm)", value=1500, step=100)
    catag = st.numberinput("CAT vertical accel (g)", value=0.25, step=0.01, format="%.2f")
    cattd = st.numberinput("CAT RA delay (s)", value=5.0, step=0.5)
with d2:
    cattas = st.numberinput("CAT TAS (kt)", value=450.0, step=10.0)
    catcap = st.number¨C203Cnm = st.number¨C204Chdg = st.number¨C205Chdg = st.number¨C206Csingle():
    vclos = relativeclosurekt(pltas, plhdg, cattas, cathdg)
    tgoauto = timetogofromgeometry(r0nm, vclos) or 30.0
    return vclos, tgoauto
 
vclos, tcpadefault = computesingle()
 
h1, h2 = st.columns(2)
with h1:
    st.metric("Computed closure (kt)", f"{vclos:,.1f}")
with h2:
    tcpa = st.numberinput("RA→CPA time (s) (override)", value=float(round(tcpadefault,1)), step=1.0)
 
# Single-run Δh & risk
dhplft = deltahpiecewise(tcpa, pltd, plag, min(plvs, plcap))
dhcatft = deltahpiecewise(tcpa, cattd, catag, min(catvs, catcap))
baselinemode = "IDEAL" if baseline.startswith("IDEAL") else "STANDARD"
dhbaseft = baselinedhft(tcpa, mode=baselinemode)
 
st.markdown("### Single-run Δh & unresolved residual risk")
spot = pd.DataFrame({
    "Aircraft": ["PL (ownship)", "CAT (intruder)", f"Baseline ({baselinemode})"],
    "ΔhCPA (ft)": [dhplft, dhcatft, dhbaseft]
})
st.dataframe(spot, usecontainerwidth=True)
 
ratio = (dhbaseft / dhplft) if dhplft > 1e-6 else np.nan
unresrr = 1.1 ratio
st.write(f"Scaled unresolved residual risk ≈ {unresrr:,.3f}% (ratio {ratio:,.3f})")
 
# Surrogate event for single-run
times, vspl = vstimeseries(tcpa, dt, pltd, plag, plvs, sense=+1, capfpm=plcap)
, vscat = vstimeseries(tcpa, dt, cattd, catag, catvs, sense=-1, capfpm=catcap)
events = surrogatedecision(times, vspl, vscat, tcpas=tcpa, respthr=respthr)
 
st.markdown("### Single-run RA taxonomy (surrogate)")
if events:
    st.write(f"First event: {events[0][0]} at t={events[0][1]:.1f}s.")
else:
    st.write("No strengthen/reverse flagged (coordinated & compliant case).")
 
fig, ax = plt.subplots(figsize=(7,3))
ax.plot(times, vspl, label="PL VS (fpm)")
ax.plot(times, vscat, label="CAT VS (fpm)")
ax.setxlabel("Time since RA (s)")
ax.setylabel("Vertical speed (fpm)")
ax.grid(True, alpha=0.3)
ax.legend()
st.pyplot(fig)
 
# -----------------------------
# Batch Monte Carlo
# -----------------------------
st.markdown("---")
st.header("Batch Monte Carlo")
 
with st.expander("Batch settings"):
    nruns = st.numberinput("Number of runs", minvalue=1, maxvalue=10000, value=2000, step=100)
    seed = st.numberinput("Random seed", value=42, step=1)
    scenario = st.selectbox("Scenario", ["Head-on", "Crossing", "Overtaking", "Custom"])
    rmin = st.numberinput("Initial range min (NM)", value=5.0, step=0.5)
    rmax = st.numberinput("Initial range max (NM)", value=12.0, step=0.5)
 
    if scenario == "Custom":
        hdg1min = st.numberinput("PL heading min (deg)", value=0.0, step=5.0)
        hdg1max = st.numberinput("PL heading max (deg)", value=360.0, step=5.0)
        hdg2min = st.numberinput("CAT heading min (deg)", value=0.0, step=5.0)
        hdg2max = st.numberinput("CAT heading max (deg)", value=360.0, step=5.0)
    else:
        hdg1min = 0.0; hdg1max = 360.0
        # define relative bearing windows
        if scenario == "Head-on":
            relmin, relmax = 150.0, 210.0
        elif scenario == "Crossing":
            relmin, relmax = 60.0, 120.0
        else:  # Overtaking
            relmin, relmax = 0.0, 30.0
 
    # Optional altitude band (for reporting only in this en-route model)
    fllo = st.numberinput("Altitude band low (FL)", value=150, step=10)
    flhi = st.numberinput("Altitude band high (FL)", value=300, step=10)
 
    runbtn = st.button("Run batch")
 
def sampleheadings(rng, scenario, hdg1min, hdg1max, relmin=None, relmax=None):
    h1 = rng.uniform(hdg1min, hdg1max)
    if scenario == "Custom":
        return h1, rng.uniform(hdg2min, hdg2max)
    else:
        # pick relative bearing around h1
        rel = rng.uniform(relmin, relmax)
        # 50/50 left-right
        dirsign = 1 if rng.uniform() < 0.5 else -1
        h2 = (h1 + dirsign rel) % 360.0
        return h1, h2
 
if runbtn:
    rng = np.random.defaultrng(int(seed))
    data = []
 
    for k in range(int(nruns)):
        # headings
        if scenario == "Custom":
            h1 = rng.uniform(hdg1min, hdg1max)
            h2 = rng.uniform(hdg2min, hdg2max)
        else:
            h1, h2 = sampleheadings(rng, scenario, hdg1min, hdg1max, relmin, relmax)
 
        # range and closure
        r0 = rng.uniform(rmin, rmax)
        vcl = relativeclosurekt(pltas, h1, cattas, h2)
        tgo = timetogofromgeometry(r0, vcl)
        if tgo is None:
            # no closure; skip
            continue
        # constrain to a practical window (12..60 s)
        tgo = min(max(tgo, 12.0), 60.0)
 
        # Δh (ft)
        dhpl = deltahpiecewise(tgo, pltd, plag, min(plvs, plcap))
        dhcat = deltahpiecewise(tgo, cattd, catag, min(catvs, catcap))
        dhtotal = dhpl + dhcat
 
        # baseline & risk
        dhbase = baselinedhft(tgo, mode=baselinemode)
        ratio = (dhbase / dhpl) if dhpl > 1e-6 else np.nan
        unresrr = 1.1 ratio
 
        # surrogate events
        times, vsplr = vstimeseries(tgo, dt, pltd, plag, plvs, sense=+1, capfpm=plcap)
        , vscatr = vstimeseries(tgo, dt, cattd, catag, catvs, sense=-1, capfpm=catcap)
        ev = surrogatedecision(times, vsplr, vs¨C310Cr, t¨C311Cs=tgo, resp¨C312Cthr)
        evtype = ev[0][0] if ev else "NONE"
        evtime = ev[0][1] if ev else np.nan
 
        # ALIM@CPA breach
        breach = dhtotal < alimft
 
        data.append({
            "run": k+1,
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
            "FLband": f"FL{int(fllo)}–FL{int(flhi)}"
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
        fig1, ax1 = plt.subplots(figsize=(6,3))
        ax1.hist(df["unresolvedRRpct"].dropna(), bins=30, color="#4e79a7")
        ax1.settitle("Unresolved residual risk (%)")
        ax1.setxlabel("%"); ax1.setylabel("Count")
        ax1.grid(alpha=0.3)
        st.pyplot(fig1)
    with c2:
        counts = df["eventtype"].valuecounts().reindex(["STRENGTHEN","REVERSE","NONE"]).fillna(0)
        fig2, ax2 = plt.subplots(figsize=(5,3))
        ax2.pie(counts.values, labels=counts.index, autopct="%1.1f%%", colors=["#59a14f","#e15759","#bab0ab"])
        ax2.settitle("Event taxonomy")
        st.pyplot(fig2)
 
    st.subheader("Download per-run CSV")
    st.downloadbutton(
        label="Download CSV",
        data=df.tocsv(index=False).encode("utf-8"),
        filename="tcasbatchresults.csv",
        mime="text/csv"
    )
 
    st.subheader("Preview of results")
    st.dataframe(df.head(20), usecontainerwidth=True)
 
st.caption("""
References: EUROCONTROL ACAS II Guide (system behaviour, ALIM) · FAA TCAS II v7.1 Intro (reversal improvements, 'Level-off') ·
EUROCONTROL ACAS II Bulletin 13 (reversal rarity and causes) · Skybrary (TCAS RA reversal operational notes) ·
EUROCONTROL/IATA FDM guidance (≥300 fpm 'meaningful' response).
""")
