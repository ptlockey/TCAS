# TCAS Encounter Analyser — User Guide

## Overview
The TCAS Encounter Analyser is an interactive Streamlit application for exploring Traffic Alert and Collision Avoidance System (TCAS) encounter scenarios via Monte Carlo simulation and single-case visualisation. The maintained entry point is `calculator.py`, which exposes all user-facing controls.【F:README.md†L1-L4】

## Getting Started
1. *(Optional but recommended)* Create and activate a Python virtual environment.
2. Install the application dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Launch the Streamlit interface:
   ```bash
   streamlit run calculator.py
   ```
   Streamlit prints a local URL in the terminal—open it in your browser to begin using the analyser.【F:README.md†L7-L20】

## Application Layout
When the app loads it sets a wide layout and presents two main regions:
- **Sidebar** — persistent simulation controls grouped into expandable sections.
- **Main area** — two tabs: *Single-run demo* and *Batch Monte Carlo*, each with contextual widgets and charts.【F:calculator.py†L113-L439】

### Sidebar Controls
The sidebar is divided into thematic panels that shape both the single-run illustration and Monte Carlo batches.

#### Scenario Geometry
Choose the traffic geometry, random seed, lateral range bounds, and whether to clamp the intruder’s indicated airspeed (IAS) to 250 kt for TAS conversion. Enabling *Custom time-to-CPA window* exposes a slider that bounds the resolution-advisory look-ahead (`t_go`) between 15–35 seconds; the helper automatically shapes a triangular sampling mode around 24–26 s. Custom scenarios can also specify protected/intruder heading ranges for sampling bespoke geometries.【F:calculator.py†L119-L183】

#### Response Behaviour
Set the initial trajectory *aggressiveness* (0 = level-off context, 1 = aggressive climbs/descents), toggle per-batch jitter of non-compliance priors, and configure autopilot/flight-director (AP/FD) usage. Presets pin the AP/FD share to fixed percentages, while the custom option exposes a slider. Below, provide baseline probabilities for manual opposite-sense reactions, traffic-advisory-only (no response), and weak compliance. An optional JSON text area lets you define altitude-dependent overrides that the model converts into `OppositeSenseBand` records, with AP/FD probabilities clamped to zero because automated crews are modelled as sense-following.【F:calculator.py†L184-L313】【F:simulation.py†L720-L778】

#### ALIM Settings
Select which TCAS v7.1 altitude layer (FL50–FL420) sets the ALIM separation threshold for scoring Monte Carlo runs. The chosen ±ALIM value is applied uniformly across the batch.【F:calculator.py†L315-L329】【F:simulation.py†L364-L392】

### Single-run Demo Tab
The single-run tab simulates a handcrafted encounter at FL220 using the sidebar geometry defaults. Configure the initial range, integration time step, and intruder response sense. Additional fields tune pilot delay, acceleration, vertical-speed target, and intruder IAS used to compute TAS and closure rate. On *Run single case* the tool derives the time to closest point of approach (CPA), generates vertical-speed traces for the protected (PL) and intruder (CAT) aircraft, integrates altitude, and prepends two seconds of pre-trigger history for context.【F:calculator.py†L331-L399】【F:simulation.py†L138-L200】【F:simulation.py†L178-L195】

Results include scenario annotations, four key metrics (time to CPA, range rate, height difference at CPA, residual risk), and an altitude plot that highlights divergence between aircraft.【F:calculator.py†L403-L437】 Residual risk is derived from the relative altitude change each aircraft achieves between the RA trigger and CPA.【F:simulation.py†L197-L205】

### Batch Monte Carlo Tab
Specify the number of Monte Carlo runs (10–200,000) and launch the batch. The app passes all sidebar parameters—including geometry bounds, aggressiveness, compliance priors, AP/FD share, ALIM selection, heading ranges, and optional time-to-CPA bounds—to the `run_batch` engine.【F:calculator.py†L439-L468】【F:simulation.py†L1400-L1516】 The completed DataFrame is stored for further exploration.【F:calculator.py†L471-L474】

#### Batch Metrics
The header metrics summarise outcome frequencies (reversal, strengthen, none), ALIM breaches at CPA, mean miss distance, and how often AP/FD crews executed the RA relative to the targeted share.【F:calculator.py†L475-L495】 If ALIM breaches occur, an accompanying table shows how many runs fell within progressively tighter margins (ALIM, ALIM−25/50/100 ft).【F:calculator.py†L496-L531】 Residual-risk metrics report the mean and 95th-percentile residual risk across the batch.【F:calculator.py†L534-L538】 When reversals are present the app surfaces their drivers based on the available detail columns.【F:calculator.py†L539-L557】

#### Preview Table
Toggle *Show reversals only* or *Lowest separation heights* to filter the preview DataFrame. The latter keeps runs within 300 ft of the minimum CPA separation, sorted from tightest to loosest, courtesy of the `build_preview_dataframe` helper.【F:calculator.py†L559-L598】【F:preview_filters.py†L6-L39】 The preview grid displays up to 200 matching runs and includes contextual captions about the filter state.【F:calculator.py†L580-L598】

#### Visual Insights
Two matplotlib charts summarise the batch: a bar plot of RA outcome shares and a box plot comparing initial vertical separations by outcome, followed by a scatter plot that maps time-to-go versus CPA margin relative to ALIM. Points outlined in black indicate ALIM breaches.【F:calculator.py†L600-L681】

#### Run Inspector
Expand *Inspect an individual run* to step through the filtered preview subset. Navigation buttons and a numeric input snap to available run IDs. For each selected run the inspector reconstructs the stored manoeuvre history (using embedded time histories when available or regenerating them from logged parameters), plots relative altitude with shaded ±ALIM bounds, and annotates CAT response delays and any second-phase issue markers.【F:calculator.py†L683-L870】【F:simulation.py†L178-L195】【F:inspector_utils.py†L1-L23】 The plot title highlights the run number, event type, and miss distance at CPA.【F:calculator.py†L834-L870】

#### Data Export
Download the full Monte Carlo table as a CSV via the *Download CSV* button.【F:calculator.py†L874-L880】

## How the Modelling Works
The simulation engine in `simulation.py` encapsulates the encounter physics and TCAS logic so that the Streamlit UI simply configures and displays results.【F:simulation.py†L1-L2070】 Below is an accessible summary of the pipeline.

### Sampling Traffic Geometry
For each run the model samples flight levels for the protected and intruder aircraft, ensuring a plausible vertical separation (`h0`) with caps that widen at higher altitudes.【F:simulation.py†L1377-L1395】 Depending on the chosen geometry (head-on, crossing, overtaking, or custom), the engine either samples relative headings from scenario-specific ranges or uses user-defined heading windows. It calculates true airspeeds from IAS (including optional forced 250 kt CAT IAS) and derives closure rate and initial lateral range. When custom `t_go` bounds are active the engine samples within the requested triangle; otherwise it selects a scenario-shaped mean within the regulatory 15–35 s window, clipping to feasible values if the geometry would yield an earlier CPA.【F:simulation.py†L1430-L1516】

### Initial Flight State and Response Templates
Initial vertical speeds for both aircraft come from an aggressiveness-controlled distribution. A level-off context (aggressiveness ≈ 0) strongly favours near-zero vertical speed, whereas higher aggressiveness produces a mix of climbs and descents.【F:simulation.py†L430-L455】 CAT pilot delay and acceleration are drawn from calibrated normal distributions, optionally blended via a delay mixture to represent a 70/30 mix of fast and slow responders.【F:simulation.py†L1522-L1526】 The engine assembles weighted response templates describing manual and AP/FD behaviours before selecting the optimal advisory sense that minimises expected miss distance.【F:simulation.py†L1528-L1563】

### Compliance and AP/FD Share
Each run randomly assigns AP/FD usage based on the effective share. Automation inherits the deterministic AP/FD template; manual crews use sampled delays/accelerations. The `apply_non_compliance_to_cat` function then models possible deviations: opposite-sense manoeuvres, no response, weak compliance, or compliant execution. Probabilities can jitter per run, and altitude-dependent overrides feed into the `OppositeSenseModel`. AP/FD crews are forced to follow the commanded sense.【F:simulation.py†L1565-L1591】【F:simulation.py†L781-L842】

### Piecewise Kinematics and Two-phase Logic
Both aircraft follow a piecewise vertical-speed profile: a delay, a finite acceleration toward a capped target, and integration into altitude over time.【F:simulation.py†L138-L176】【F:simulation.py†L1595-L1614】 TCAS monitoring steps through the response in one-second surveillance cycles with injected sensor noise, classifying outcomes as NONE, STRENGTHEN, or REVERSE based on predicted miss distance, sense mismatches, and escalation rules.【F:simulation.py†L1652-L1754】【F:simulation.py†L918-L1040】 When a strengthen or reversal is triggered, the engine projects a second-phase manoeuvre with a decision latency, recomputes vertical-speed profiles, and optionally flips the intruder’s sense if doing so improves CPA separation. This creates a piecewise response sequence with explicit timestamps for secondary advisories.【F:simulation.py†L1280-L1371】【F:simulation.py†L1765-L1873】

### Outcome Scoring
After the manoeuvre sequence concludes the model derives key observables: minimum separation, CPA separation, ALIM margins (including 25/50/100 ft bands), residual risk, compliance label (a Method-B-style classification evaluated 3.5 s after first motion), and flags for any reversals. Time histories are serialised for later inspection. Final and initial event labels/time-to-go are recorded separately to highlight reclassifications caused by coordination dropouts or second-phase downgrades.【F:simulation.py†L197-L205】【F:simulation.py†L400-L424】【F:simulation.py†L1934-L2005】

## Understanding the Output Data
Monte Carlo batches return a DataFrame with one row per run. Notable columns include:

| Column | Meaning |
| --- | --- |
| `run` | Sequential run identifier. |
| `scenario`, `PLhdg`, `CAThdg`, `R0NM`, `closurekt` | Sampled geometry descriptors (scenario type, headings, initial range, closure rate). |
| `FL_PL`, `FL_CAT`, `h0ft`, `cat_above` | Initial flight levels, signed vertical separation, and whether the intruder starts above the protected aircraft. |
| `tgos` | Time to CPA at RA issue. |
| `sensePL`, `senseCAT_chosen`, `senseCAT_exec` | Commanded PL sense, commanded CAT sense, and executed CAT sense for the first phase. |
| `CAT_mode`, `CAT_is_APFD` | Textual label of the intruder’s compliance outcome and a boolean flag for AP/FD execution. |
| `plDelay`, `catDelay`, `catAccel_g`, `catVS_cmd`, `catCap_cmd` | Kinematic parameters used to build the time histories. |
| `ALIM_ft`, `missCPAft`, `sep_cpa_ft`, `margin_min_ft`, `margin_cpa_ft`, `alim_breach_cpa`, `alim_breach_cpa_band25/50/100` | Separation metrics and ALIM breach indicators at CPA. |
| `eventtype`, `event_detail`, `t_detect`, `tau_detect`, `eventtype_final`, `event_detail_final`, `t_detect_final`, `tau_detect_final` | Initial versus final RA classifications with associated decision times and time-to-go. |
| `t_second_issue`, `tau_second_issue` | Timestamp and remaining time-to-go when a second-phase advisory issued. |
| `maneuver_sequence` | Tuple logging every manoeuvre phase, including downgraded reversals and coordination dropouts. |
| `any_reversal`, `comp_label`, `residual_risk`, `delta_h_pl_ft`, `delta_h_cat_ft` | Summary flags and performance measures. |
| `time_history_json` | Serialized time histories for reconstruction in the inspector. |

All of these fields are populated inside `run_batch` when each row is appended to `data`, providing end-to-end traceability from sampled inputs through classified outcomes.【F:simulation.py†L1948-L2005】

## Tips for Effective Use
- Fix the random seed when you need reproducible Monte Carlo batches; change it to explore new random draws.【F:calculator.py†L125-L130】
- Narrow the time-to-CPA window to stress-test short-warning scenarios, or widen it to emphasise longer look-ahead encounters. The helper keeps the mean within regulatory bounds while respecting your limits.【F:calculator.py†L148-L166】【F:simulation.py†L105-L127】
- Use the altitude-band JSON override to study how altitude-specific opposite-sense behaviours affect reversal rates without touching the global priors.【F:calculator.py†L258-L312】【F:simulation.py†L720-L778】
- Export the CSV and analyse it externally—every metric visualised in the app is stored in the dataset for deeper study.【F:calculator.py†L874-L880】【F:simulation.py†L1948-L2005】

