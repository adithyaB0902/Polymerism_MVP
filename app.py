"""POLYMEMSIM Streamlit app — a virtual polymer-membrane testing lab.

Tab layout (all tabs render their code every rerun; Streamlit just
hides/shows the corresponding panel):

  0  Overview
  1  Sample Registry       - create/browse membrane samples & batches (SQLite-backed)
  2  Protocol Runner       - guided multi-step experimental protocol (lab/protocol.py)
  3  Single Simulation     - simple or detailed physics, simulated replicates, log to notebook
  4  Lab Notebook          - browse/filter every logged experiment
  5  Validation & Calibration - upload real data, compare, and fit physics parameters to it
  6  Virtual Experiments   - generate a synthetic screening dataset
  7  Optimization          - rank candidates by feasibility_score
  8  Pareto                - flux-vs-rejection Pareto frontier
  9  Sensitivity           - one-at-a-time parameter sensitivity
  10 Experiment Recommendation
  11 ML Lab                - train / cross-validate / predict / explain / registry
  12 Feasibility           - calibration-aware screening verdict
  13 Assumptions & Limitations

See models/membrane.py for which inputs each physics model actually
uses, and docs/assumptions.md / docs/equations.md for the full picture.
"""

import numpy as np
import pandas as pd
from io import BytesIO
import streamlit as st

from ui_theme import (
    inject_theme, hero, stat_strip, section_label, verdict_badge, score_display, check_row,
    top_nav, stepper, page_footer, glass_card, metric_card, badge,
)
from ui_labels import APP_NAME, APP_TAGLINE, APP_SUMMARY, STAGES, PROGRESS_STEPS, LABELS
from models.membrane import Membrane, Water, OperatingConditions, Targets
from models.simulator import run_simulation, run_detailed_simulation
from models.feasibility import feasibility_assessment
from simulation.generate_dataset import generate_virtual_experiments
from simulation.sensitivity import sensitivity_analysis
from optimization.optimizer import optimize_candidates, pareto_frontier
from optimization.experiment_recommender import recommend_experiments
from validation.experimental_data import validate_experimental_csv
from validation.comparison import compare_predictions
from validation.calibration import calibrate_sample

from db import repository as repo
from lab.samples import (
    load_presets,
    create_sample_from_preset,
    create_sample_from_membrane,
    sample_to_membrane,
)
from lab.replicates import simulate_replicates, replicate_statistics, DEFAULT_NOISE_CV_PERCENT
from lab.protocol import (
    start_protocol,
    current_step_definition,
    record_current_step,
    is_complete,
    progress_fraction,
    collected_values,
    PROTOCOL_STEPS,
)

from ml.train import train_models, FEATURES as ML_FEATURES, TARGETS as ML_TARGETS
from ml.dataset import assemble_training_dataset
from ml.model_registry import save_and_register_models, load_active_model, load_all_active_models
from ml.predict import predict_candidate
from ml.evaluate import cross_validate_all_targets, compare_model_types
from ml.uncertainty import supports_uncertainty, predict_one_with_uncertainty, conformal_prediction_interval
from ml.explain import feature_importances, permutation_importances, used_vs_unused_features
from ml.ood import OODDetector
from research.membrane_formulation import MembraneFormulation
from research.pb_removal import calculate_pb_metrics
from research.experiments import read_experimental_data, validate_experiment_frame
from research.rsm.box_behnken import generate_box_behnken_design
from research.rsm.ui import render_rsm_analysis
from research.cv_nested import nested_compare
from research.paper_export import export_paper_results
from research.ml import FEATURES as RESEARCH_FEATURES, compare_models as compare_research_models, fit_best_model, explain_model, shap_values_if_available
from research.optimization import optimize_removal
from research.confirmation import compare_confirmation
from research.science_tools import swelling_degree, porosity, multiresponse_desirability, regeneration_summary, suggest_next_experiment
from research.adsorbent_comparison import comparison_table
from research.analysis_tools import report_json_bytes
from research.adsorption import fit_isotherms
from research.kinetics import fit_kinetics
from research.characterization import validate_characterization

st.set_page_config(
    page_title="POLYMEMSIM",
    layout="wide",
    page_icon="🧪",
    initial_sidebar_state="collapsed",
)
if "pms_dark_mode" not in st.session_state:
    st.session_state["pms_dark_mode"] = False
inject_theme("dark" if st.session_state["pms_dark_mode"] else "light")
selected_stage, _ = top_nav(STAGES, "dark" if st.session_state["pms_dark_mode"] else "light")
stepper(PROGRESS_STEPS, current=STAGES.index(selected_stage) if selected_stage in STAGES[:5] else 0)
stage_hints = {
    "Start": "Begin with a membrane recipe and review the paper workspace.",
    "Plan and Enter": "Create samples, plan experiments, and enter measured lab results.",
    "Analyze": "Compare models, inspect RSM surfaces, and identify important factors.",
    "Optimize and Confirm": "Find candidate recipes, then compare them with confirmation measurements.",
    "More": "Open supporting tests, material comparisons, exports, safety, and glossary content.",
}
glass_card(selected_stage, stage_hints[selected_stage], "Current stage")
hero("🧪", APP_NAME, "Virtual Polymer Membrane Testing Lab")
st.caption(
    f"{APP_SUMMARY} Synthetic/ML predictions are not experimental validation."
)


@st.cache_resource
def get_db_connection():
    return repo.get_connection()


conn = get_db_connection()

# Resolve any pending cross-widget navigation requests queued by a button
# elsewhere in the app (e.g. "jump the experiment setup sample selector to the
# sample I just created"). This MUST happen before any widget with a
# matching key is instantiated below -- Streamlit forbids writing to a
# widget's own session_state key after that widget has already rendered
# once in the current script run, so any code that wants to redirect an
# already-rendered widget queues a "_pending_*" key and calls st.rerun();
# on the resulting fresh run, we resolve it here first, before the widget exists.
for _pending_key, _real_key in [
    ("_pending_active_sample_id", "active_sample_id"),
    ("_pending_active_protocol_run_id", "active_protocol_run_id"),
]:
    if _pending_key in st.session_state:
        st.session_state[_real_key] = st.session_state.pop(_pending_key)

_NOT_USED_SIMPLE = "Not used by the Simple physics model. See the Detailed model and the Assumptions tab."


def _help(field, detailed_text):
    """Return `detailed_text` when Detailed Physics is on, else the
    standard 'not used by Simple model' note."""
    return detailed_text if st.session_state.get("detailed_physics") else _NOT_USED_SIMPLE


# --------------------------------------------------------------------------
# Full-page experiment setup: physics model choice, membrane source,
# water/operating conditions, and screening targets.
# --------------------------------------------------------------------------


def experiment_setup():
    s = st.expander("⚙️ Experiment setup", expanded=True)
    st.caption(
        "Set the active candidate once here. The same setup powers simulation, feasibility, "
        "screening, sensitivity, and ML prediction workflows."
    )
    section_label(s, "⚙️", "Physics model")
    detailed = s.toggle(
        "Simple / Detailed view",
        value=False,
        key="detailed_physics",
        help="Adds temperature-corrected viscosity, a structural pore-flow flux "
        "cross-check, concentration-polarization-corrected rejection, a "
        "recovery-based mass balance, and a water-quality fouling adjustment. "
        "Off = the original simple model (permeability x TMP only).",
    )

    section_label(s, "🧪", "Membrane sample")
    source = s.radio("Source", ["Registered sample", "Quick manual entry (not saved)"], index=0)

    membrane = None
    active_sample_id = None
    if source == "Registered sample":
        samples = repo.list_samples(conn)
        if not samples:
            s.info("No samples yet. Load a preset below, or create one in the Sample Registry tab.")
            presets = load_presets()
            labels = [f"{p['polymer_name']} ({p['membrane_type']})" for p in presets]
            choice = s.selectbox("Quick-load a preset", labels)
            if s.button("Load preset as new sample"):
                new_id = create_sample_from_preset(conn, presets[labels.index(choice)])
                st.session_state["_pending_active_sample_id"] = new_id
                st.rerun()
            membrane = Membrane(
                "No sample yet", "UF", 100.0, 25.0, baseline_rejection_percent=95.0, fouling_coefficient=0.03
            )
        else:
            ids = [row["id"] for row in samples]
            by_id = {row["id"]: row for row in samples}
            select_kwargs = {"index": 0} if "active_sample_id" not in st.session_state else {}
            chosen_id = s.selectbox(
                "Active sample",
                ids,
                format_func=lambda i: f"#{i} {by_id[i]['polymer_name']} ({by_id[i]['membrane_type']}, {by_id[i]['origin']})",
                key="active_sample_id",
                **select_kwargs,
            )
            active_sample_id = chosen_id
            row = by_id[chosen_id]
            membrane = sample_to_membrane(row)
            with s.expander("Sample details"):
                st.json({k: v for k, v in row.items() if k not in ("id",)})
    else:
        polymer = s.text_input("Polymer", "Custom Polymer")
        membrane_type = s.selectbox("Membrane type", ["MF", "UF", "NF", "RO", "Custom"], index=1)
        thickness = s.number_input(
            "Thickness (µm)",
            min_value=0.01,
            value=100.0,
            help=_help("thickness_um", "Drives the structural pore-flow flux cross-check."),
        )
        permeability = s.number_input(
            "Permeability (LMH/bar)",
            min_value=0.01,
            value=25.0,
            help="Drives flux: J = permeability x TMP (temperature-corrected in Detailed mode).",
        )
        pore = s.number_input(
            "Pore size (nm)",
            min_value=0.0,
            value=20.0,
            help=_help("pore_size_nm", "Drives the structural pore-flow flux cross-check."),
        )
        mwco = s.number_input("MWCO (Da)", min_value=0.0, value=10000.0, help=_NOT_USED_SIMPLE)
        porosity = s.number_input(
            "Porosity",
            min_value=0.0,
            max_value=1.0,
            value=0.35,
            help=_help("porosity", "Drives the structural pore-flow flux cross-check."),
        )
        charge = s.number_input(
            "Surface charge",
            value=0.0,
            help=_help("surface_charge", "Contributes to the illustrative fouling-propensity adjustment."),
        )
        hydro = s.number_input(
            "Hydrophilicity",
            min_value=0.0,
            max_value=1.0,
            value=0.7,
            help=_help("hydrophilicity", "Contributes to the illustrative fouling-propensity adjustment."),
        )
        rejection = s.number_input(
            "Baseline (intrinsic) rejection (%)",
            min_value=0.0,
            max_value=100.0,
            value=95.0,
            help="Used directly (Simple), or corrected for concentration polarization (Detailed).",
        )
        fouling = s.number_input(
            "Fouling coefficient (1/hr)",
            min_value=0.0,
            value=0.03,
            help="Drives flux decline over time (adjusted by water quality in Detailed mode).",
        )
        membrane = Membrane(
            polymer,
            membrane_type,
            thickness,
            permeability,
            pore,
            mwco,
            porosity,
            charge,
            hydro,
            10000.0,
            rejection,
            fouling,
        )

    section_label(s, "💧", "Water")
    contaminant = s.text_input("Contaminant", "Custom contaminant")
    cf = s.number_input(
        "Feed concentration (mg/L)", min_value=0.0, value=100.0, help="Drives permeate concentration."
    )
    turbidity = s.number_input(
        "Turbidity (NTU)",
        min_value=0.0,
        value=5.0,
        help=_help("turbidity_NTU", "Contributes to the illustrative fouling-propensity adjustment."),
    )
    tds = s.number_input(
        "TDS (mg/L)",
        min_value=0.0,
        value=500.0,
        help=_help("water_TDS_mg_L", "Contributes to the illustrative fouling-propensity adjustment."),
    )
    ph = s.number_input(
        "pH",
        min_value=0.0,
        max_value=14.0,
        value=7.0,
        help="Validated to be within 0-14; not otherwise used in either model.",
    )
    temp = s.number_input(
        "Temperature (°C)",
        min_value=0.0,
        max_value=100.0,
        value=25.0,
        help=_help("water_temperature_C", "Drives temperature-corrected viscosity/flux (Vogel equation)."),
    )
    viscosity = s.number_input(
        "Viscosity (Pa·s)",
        min_value=0.00001,
        value=0.001,
        help="Not used by either model — Detailed mode computes viscosity "
        "from temperature automatically instead of using this value.",
    )
    density = s.number_input("Density (kg/m³)", min_value=1.0, value=1000.0, help=_NOT_USED_SIMPLE)

    section_label(s, "🎛️", "Operating")
    tmp = s.number_input("TMP (bar)", min_value=0.0, value=2.0, help="Drives flux and energy.")
    flow = s.number_input(
        "Feed flow (L/min)",
        min_value=0.0,
        value=10.0,
        help="Validated to be non-negative; cancels out of the energy-per-m3 "
        "result in both models (see Assumptions tab).",
    )
    crossflow = s.number_input(
        "Cross-flow velocity (m/s)",
        min_value=0.0,
        value=0.2,
        help=_help(
            "op_crossflow_velocity_m_s", "Drives the concentration-polarization correction to rejection."
        ),
    )
    recovery = s.number_input(
        "Recovery (%)",
        min_value=0.0,
        max_value=100.0,
        value=20.0,
        help=_help("op_recovery_percent", "Drives concentrate concentration and permeate-normalized energy."),
    )
    time_hr = s.number_input(
        "Operating time (hr)", min_value=0.0, value=4.0, help="Drives flux decline over time."
    )
    area = s.number_input(
        "Membrane area (m²)",
        min_value=0.0001,
        value=1.0,
        help=_help(
            "op_membrane_area_m2",
            "Drives total permeate flow (L/hr); never affects any per-m2/per-m3 intensive result.",
        ),
    )
    eta = s.number_input(
        "Pump efficiency", min_value=0.01, max_value=1.0, value=0.70, help="Drives energy per m3."
    )
    elec = s.number_input(
        "Electricity price", min_value=0.0, value=8.0, help="Drives cost per m3 (electricity term only)."
    )

    section_label(s, "🎯", "Targets")
    min_rej = s.number_input("Minimum rejection (%)", 0.0, 100.0, 95.0)
    min_flux = s.number_input("Minimum flux (LMH)", 0.0, 100000.0, 30.0)
    max_foul = s.number_input("Maximum flux decline (%)", 0.0, 100.0, 20.0)
    max_energy = s.number_input("Maximum energy (kWh/m³)", 0.0, 100000.0, 2.0)
    max_cost = s.number_input("Maximum cost / m³", 0.0, 100000.0, 10.0)

    water = Water(contaminant, cf, turbidity, tds, ph, temp, viscosity, density)
    op = OperatingConditions(tmp, flow, crossflow, recovery, time_hr, temp, area, eta, elec)
    targets = Targets(min_rej, min_flux, max_foul, max_energy, max_cost)
    return membrane, water, op, targets, detailed, active_sample_id


membrane, water, op, targets, detailed, active_sample_id = experiment_setup()


def run_current_model(m, w, o, t):
    return run_detailed_simulation(m, w, o, t) if detailed else run_simulation(m, w, o, t)


def ensure_sample_id(m):
    """Return the active registered sample id, or create one on the fly
    from the current quick-manual membrane (for logging purposes)."""
    if active_sample_id is not None:
        return active_sample_id
    return create_sample_from_membrane(
        conn, m, origin="manual", notes="Auto-created from quick manual entry."
    )


tabs = st.tabs(
    [
        "Overview 🏠",
        "Lab Workflow 🧪",
        "Screening & Optimization 📈",
        "ML Lab 🧠",
        "CS–CA–Biochar Pb(II) Lab 🧬",
        "Feasibility ✅",
    ]
)

# --------------------------------------------------------------------------
# 0. Overview
# --------------------------------------------------------------------------
with tabs[0]:
    st.markdown(
        f"""
        <div class="pms-hero-start pms-glass">
            <div class="pms-badge-glass">Start here · research workspace</div>
            <h1>{APP_TAGLINE}</h1>
            <p>{APP_SUMMARY}<br>Keep simulated, predicted, and measured results visibly separate as you build the paper.</p>
            <svg viewBox="0 0 360 90" role="img" aria-label="Water drop and membrane illustration" style="width:min(100%, 360px); margin-top:.6rem;">
                <path d="M42 8C28 28 14 42 14 58a28 28 0 0 0 56 0C70 42 56 28 42 8Z" fill="#35B9A6" opacity=".9"/>
                <path d="M106 30h220M106 46h220M106 62h220" stroke="#35B9A6" stroke-width="3" stroke-linecap="round" opacity=".8"/>
                <path d="M110 22l10 10-10 10m36-20l10 10-10 10m36-20l10 10-10 10m36-20l10 10-10 10m36-20l10 10-10 10" fill="none" stroke="#A78BFA" stroke-width="3" stroke-linecap="round"/>
            </svg>
        </div>
        """,
        unsafe_allow_html=True,
    )
    if st.button("Start my first experiment plan", type="primary", key="start_first_plan"):
        st.session_state["pms_stage"] = "Plan and Enter"
        st.rerun()
    st.markdown(
        """
        <div class="pms-callout">
            <span class="pms-badge">Research-ready</span>
            <strong>POLYMEMSIM</strong> is a virtual membrane screening platform that blends a documented
            physics model, persistent lab notebook, calibration workflow, and machine-learning analysis
            into a single experimental decision-support environment.
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(
        """
        <div class="pms-card-grid">
            <div class="pms-card">
                <h4>Core workflow</h4>
                <p>Register samples, run a guided protocol, simulate membrane behavior, and log results in a coherent pipeline.</p>
            </div>
            <div class="pms-card">
                <h4>Calibration & validation</h4>
                <p>Compare modeled outputs with measured data and fit parameters to improve prediction quality.</p>
            </div>
            <div class="pms-card">
                <h4>Research extension</h4>
                <p>Evaluate CS–CA–biochar Pb(II) formulations using Box–Behnken DOE, RSM, ML comparison, and optimization logic.</p>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.info(
        "This does not prove membrane performance, safety, regulatory compliance or commercial "
        "viability. See the Assumptions & Limitations tab."
    )
    st.markdown("""
    **Suggested workflow:**
    1. **Sample Registry** — register a membrane sample (or load a preset).
    2. **Protocol Runner** — walk through the guided measurement protocol.
    3. **Single Simulation** — run the physics model (simple or detailed), simulate replicates, log results.
    4. **Lab Notebook** — review everything logged so far.
    5. **Validation & Calibration** — upload real measurements; compare, then fit physics parameters to them.
    6. **Virtual Experiments → Optimization → Pareto → Sensitivity → Experiment Recommendation** — screen many candidates and decide what to test next.
    7. **ML Lab** — train/evaluate/predict/explain models on synthetic + real data.
    8. **RSM Studio** — generate the BBD design and fit the measured quadratic response-surface model.
    9. **Feasibility** — a calibration-aware go/no-go screening verdict.
    """)
    counts = {
        "Samples": len(repo.list_samples(conn)),
        "Logged experiments": len(repo.list_experiments(conn)),
        "Protocol runs": len(repo.list_protocol_runs(conn)),
        "Calibration runs": len(repo.list_calibration_runs(conn)),
        "Registered ML models": len(repo.list_models(conn)),
    }
    stat_strip(counts)
    research_rows = repo.list_research_experiments(conn)
    measured_research = [
        row for row in research_rows if row.get("data_source", "experimental") == "experimental"
    ]
    st.markdown("**Paper workspace**")
    st.caption(
        "A compact readiness view for the CS–CA–biochar Pb(II) study. Only measured rows count toward the paper workflow."
    )
    paper_cols = st.columns(4)
    with paper_cols[0]:
        metric_card("Measured runs", len(measured_research), "of 29 BBD runs")
    with paper_cols[1]:
        metric_card("Formulations", len(repo.list_research_formulations(conn)))
    with paper_cols[2]:
        metric_card("Reuse records", len(repo.list_reuse_cycles(conn)))
    with paper_cols[3]:
        metric_card("Paper exports", "Ready" if len(measured_research) >= 16 else "Needs data")
    st.info(
        "Recommended paper path: BBD design → Experimental data → RSM analysis → ML comparison "
        "→ Optimization → Confirmation → Paper export."
    )

# --------------------------------------------------------------------------
# 1. Lab Workflow (Sample Registry, Protocol Runner, Single Simulation,
#    Lab Notebook, Validation & Calibration)
# --------------------------------------------------------------------------
with tabs[1]:
    page = st.radio(
        "Section",
        [
            'Membrane Recipes',
            'Sample Registry',
            'Measure with a protocol',
            'Protocol Runner',
            'Simulate a candidate',
            'Single Simulation',
            'Lab Notebook',
            'Enter Lab Results',
            'Validation & Calibration',
        ],
        horizontal=True,
        key="lab_workflow_page",
        label_visibility="collapsed",
    )
    st.divider()
    if page in {'Membrane Recipes', 'Sample Registry'}:
        st.subheader("Membrane Recipes")
        st.caption("What this page does: save membrane recipes. What you need first: a recipe name and its ingredient amounts.")

        with st.expander("Create a batch"):
            batch_name = st.text_input("Batch name", key="new_batch_name")
            batch_notes = st.text_area("Notes", key="new_batch_notes")
            if st.button("CREATE BATCH") and batch_name:
                repo.create_batch(conn, batch_name, batch_notes)
                st.success(f"Batch '{batch_name}' created.")
                st.rerun()

        batches = repo.list_batches(conn)
        batch_options = {"(none)": None, **{f"#{b['id']} {b['name']}": b["id"] for b in batches}}

        with st.expander("Load an illustrative preset"):
            presets = load_presets()
            for p in presets:
                c1, c2 = st.columns([3, 1])
                c1.write(
                    f"**{p['polymer_name']}** ({p['membrane_type']}) — "
                    f"{p['permeability_LMH_bar']} LMH/bar, {p['thickness_um']} µm, "
                    f"{p['baseline_rejection_percent']}% rejection. _{p['notes']}_"
                )
                if c2.button("Load", key=f"preset_{p['polymer_name']}"):
                    new_id = create_sample_from_preset(conn, p)
                    st.session_state["_pending_active_sample_id"] = new_id
                    st.success(f"Loaded as sample #{new_id}.")
                    st.rerun()

        with st.expander("Create a sample manually"):
            c1, c2, c3 = st.columns(3)
            polymer_name = c1.text_input("Polymer name", "New Polymer")
            membrane_type = c2.selectbox("Type", ["MF", "UF", "NF", "RO", "Custom"], key="reg_type")
            chosen_batch = c3.selectbox("Batch", list(batch_options.keys()))
            c4, c5, c6 = st.columns(3)
            thickness_um = c4.number_input("Thickness (µm)", min_value=0.01, value=100.0, key="reg_thick")
            permeability_LMH_bar = c5.number_input(
                "Permeability (LMH/bar)", min_value=0.01, value=25.0, key="reg_perm"
            )
            baseline_rejection_percent = c6.number_input("Rejection (%)", 0.0, 100.0, 95.0, key="reg_rej")
            c7, c8 = st.columns(2)
            fouling_coefficient = c7.number_input(
                "Fouling coefficient (1/hr)", min_value=0.0, value=0.03, key="reg_foul"
            )
            notes = c8.text_input("Notes", "", key="reg_notes")
            if st.button("CREATE SAMPLE"):
                new_membrane = Membrane(
                    polymer_name,
                    membrane_type,
                    thickness_um,
                    permeability_LMH_bar,
                    baseline_rejection_percent=baseline_rejection_percent,
                    fouling_coefficient=fouling_coefficient,
                )
                new_id = create_sample_from_membrane(
                    conn, new_membrane, batch_id=batch_options[chosen_batch], notes=notes, origin="manual"
                )
                st.session_state["_pending_active_sample_id"] = new_id
                st.success(f"Created sample #{new_id}.")
                st.rerun()

        st.markdown("**All samples**")
        samples = repo.list_samples(conn)
        if samples:
            st.dataframe(pd.DataFrame(samples), width="stretch")
        else:
            st.info("No samples yet.")
    elif page in {'Measure with a protocol', 'Protocol Runner'}:
        st.subheader("Guided Experimental Protocol")
        st.caption("Turns docs/experimental_protocol.md into an actual step-by-step, saved workflow.")
        samples = repo.list_samples(conn)
        if not samples:
            st.info("Create or load a recipe in Membrane Recipes first.")
        else:
            labels = [f"#{r['id']} {r['polymer_name']} ({r['membrane_type']})" for r in samples]
            chosen = st.selectbox("Sample", labels, key="protocol_sample")
            sample_id = samples[labels.index(chosen)]["id"]

            existing_runs = repo.list_protocol_runs(conn, sample_id=sample_id)
            run_by_id = {r["id"]: r for r in existing_runs}
            run_options = [None] + list(run_by_id.keys())

            def _format_run(rid):
                if rid is None:
                    return "(start a new run)"
                r = run_by_id[rid]
                return (
                    f"#{r['id']} {r['name']} — {r['status']} (step {r['current_step']}/{len(PROTOCOL_STEPS)})"
                )

            # If the previously-selected run id isn't valid for the currently
            # chosen sample (e.g. the user just switched samples), clear it
            # before the widget renders rather than letting Streamlit choke on
            # a stored value that isn't among this run's options.
            if (
                st.session_state.get("active_protocol_run_id", None) not in run_options
                and "active_protocol_run_id" in st.session_state
            ):
                del st.session_state["active_protocol_run_id"]
            run_select_kwargs = {"index": 0} if "active_protocol_run_id" not in st.session_state else {}
            chosen_run_id = st.selectbox(
                "Protocol run",
                run_options,
                format_func=_format_run,
                key="active_protocol_run_id",
                **run_select_kwargs,
            )

            if chosen_run_id is None:
                run_name = st.text_input("Run name", "Protocol run")
                if st.button("START PROTOCOL"):
                    new_run_id = start_protocol(conn, sample_id, run_name)
                    st.session_state["_pending_active_protocol_run_id"] = new_run_id
                    st.rerun()
            else:
                run = run_by_id[chosen_run_id]
                st.progress(progress_fraction(run))
                if is_complete(run):
                    st.success("This protocol run is complete.")
                    st.json(collected_values(run))
                    if st.button("LOG AS EXPERIMENT"):
                        vals = collected_values(run)
                        try:
                            result = {
                                "flux_LMH": float(vals.get("flux_LMH", 0) or 0),
                                "rejection_percent": float(vals.get("rejection_percent", 0) or 0),
                            }
                            repo.record_experiment(
                                conn,
                                sample_id,
                                {},
                                {},
                                result,
                                data_source="measured",
                                protocol_run_id=run["id"],
                                notes=f"From protocol run #{run['id']} ({run['name']})",
                            )
                            st.success("Logged to the lab notebook as a measured experiment.")
                        except (ValueError, TypeError):
                            st.error(
                                "Couldn't parse flux_LMH/rejection_percent as numbers from the recorded values."
                            )
                else:
                    step_def = current_step_definition(run)
                    st.markdown(
                        f"**Step {run['current_step'] + 1}/{len(PROTOCOL_STEPS)}: {step_def['name']}**"
                    )
                    st.write(step_def["description"])
                    values = {}
                    for field in step_def["fields"]:
                        values[field] = st.text_input(
                            field.replace("_", " ").title(), key=f"step_{run['id']}_{field}"
                        )
                    if st.button("RECORD & CONTINUE"):
                        record_current_step(conn, run["id"], values)
                        st.rerun()
    elif page in {'Simulate a candidate', 'Single Simulation'}:
        st.subheader("Single Simulation")
        st.caption(f"Model: {'Detailed' if detailed else 'Simple'} (change this in Experiment setup above).")
        badge("Simulated", "simulated")
        if st.button("RUN SIMULATION"):
            try:
                result = run_current_model(membrane, water, op, targets)
                st.session_state["last_result"] = result
            except Exception as e:
                st.error(str(e))

        if "last_result" in st.session_state:
            result = st.session_state["last_result"]
            c = st.columns(5)
            for col, (k, v) in zip(
                c,
                [
                    ("Flux", result["flux_LMH"]),
                    ("Rejection", result["rejection_percent"]),
                    ("Permeate", result["permeate_mg_L"]),
                    ("Energy", result["energy_kWh_m3"]),
                    ("Score", result["feasibility_score"]),
                ],
            ):
                col.metric(k, f"{v:.2f}")
            st.dataframe(pd.DataFrame([result]), width="stretch")
            st.caption(f"Flux model: {result['flux_model']}. {result['scientific_note']}")

            st.markdown("**Simulated replicates**")
            st.caption(
                "Injects illustrative measurement-style noise on top of the deterministic result "
                "(see lab/replicates.py) — this is NOT a model of any real error source."
            )
            n_reps = st.slider("Number of replicates", 2, 20, 3)
            if st.button("SIMULATE REPLICATES"):
                reps = simulate_replicates(result, n=n_reps)
                stats = replicate_statistics(reps, fields=list(DEFAULT_NOISE_CV_PERCENT.keys()))
                st.session_state["last_replicates"] = reps
                st.session_state["last_replicate_stats"] = stats
            if "last_replicate_stats" in st.session_state:
                st.dataframe(pd.DataFrame(st.session_state["last_replicate_stats"]).T, width="stretch")
                st.dataframe(pd.DataFrame(st.session_state["last_replicates"]), width="stretch")

            st.markdown("**Log this result**")
            if st.button("SAVE TO LAB NOTEBOOK"):
                sid = ensure_sample_id(membrane)
                repo.record_experiment(
                    conn,
                    sid,
                    water.__dict__,
                    op.__dict__,
                    result,
                    physics_model=result["physics_model"],
                    data_source="simulated",
                )
                st.success(f"Logged against sample #{sid}.")
    elif page == 'Lab Notebook':
        st.subheader("Lab Notebook")
        samples = repo.list_samples(conn)
        sample_filter_labels = ["(all samples)"] + [f"#{r['id']} {r['polymer_name']}" for r in samples]
        chosen_filter = st.selectbox("Filter by sample", sample_filter_labels)
        source_filter = st.selectbox("Filter by data source", ["(all)", "simulated", "measured"])

        sample_id_filter = (
            None
            if chosen_filter == "(all samples)"
            else samples[sample_filter_labels.index(chosen_filter) - 1]["id"]
        )
        ds_filter = None if source_filter == "(all)" else source_filter
        experiments = repo.list_experiments(conn, sample_id=sample_id_filter, data_source=ds_filter)

        st.metric("Logged experiments matching filters", len(experiments))
        if experiments:
            df_exp = pd.DataFrame(experiments)
            st.dataframe(df_exp, width="stretch")
            st.download_button("Download CSV", df_exp.to_csv(index=False), "lab_notebook.csv", "text/csv")
        else:
            st.info("No experiments logged yet — run a simulation or complete a protocol and save it.")
    elif page in {'Enter Lab Results', 'Validation & Calibration'}:
        st.subheader("Enter Lab Results")
        st.caption("What this page does: compare model outputs with measurements. What you need first: a CSV or your lab notebook values.")
        uploaded = st.file_uploader(
            "Upload experimental CSV (needs flux_LMH and rejection_percent columns; "
            "optional operating_time_hr)",
            type=["csv"],
        )
        if uploaded:
            df_up, report = validate_experimental_csv(uploaded)
            st.write(report)
            st.dataframe(df_up.head(100), width="stretch")

            if report["valid_rows"] >= 1:
                st.markdown("**Compare current physics prediction to these measurements**")
                try:
                    predicted = run_current_model(membrane, water, op, targets)
                    comp_flux = compare_predictions(
                        [predicted["flux_LMH"]] * report["valid_rows"], df_up["flux_LMH"].dropna().tolist()
                    )
                    st.write("Flux comparison (current experiment-setup prediction vs each measured row):", comp_flux)
                except Exception as e:
                    st.error(str(e))

            if report["valid_rows"] >= 2:
                st.markdown("**Calibrate physics parameters to this data**")
                st.caption(
                    "Fits permeability, fouling coefficient and baseline rejection to best match "
                    "the uploaded measurements (nonlinear least squares) — see validation/calibration.py."
                )
                if st.button("CALIBRATE"):
                    measurements = (
                        df_up[["flux_LMH", "rejection_percent"]].dropna(how="all").to_dict("records")
                    )
                    if "operating_time_hr" in df_up.columns:
                        for i, row in enumerate(df_up.to_dict("records")):
                            if i < len(measurements) and "operating_time_hr" in row:
                                measurements[i]["operating_time_hr"] = row["operating_time_hr"]
                    try:
                        calib = calibrate_sample(membrane, water, op, measurements)
                        st.session_state["last_calibration"] = calib
                        st.success(f"Fitted params: {calib['fitted_params']}")
                        c1, c2 = st.columns(2)
                        c1.write("Before")
                        c1.json(calib["metrics_before"])
                        c2.write("After")
                        c2.json(calib["metrics_after"])
                    except ValueError as e:
                        st.error(str(e))

                if "last_calibration" in st.session_state:
                    calib = st.session_state["last_calibration"]
                    if st.button("SAVE CALIBRATED SAMPLE"):
                        sid = ensure_sample_id(membrane)
                        new_id = create_sample_from_membrane(
                            conn,
                            calib["calibrated_membrane"],
                            origin="calibrated",
                            calibrated_from_sample_id=sid,
                            notes=f"Calibrated from sample #{sid} against {calib['n_measurements']} measurements.",
                        )
                        repo.save_calibration_run(
                            conn,
                            sid,
                            new_id,
                            "least_squares",
                            calib["n_measurements"],
                            calib["fitted_params"],
                            calib["metrics_before"],
                            calib["metrics_after"],
                        )
                        st.success(f"Saved as new sample #{new_id}, linked to a calibration run.")

        st.markdown("**Calibration history**")
        calib_runs = repo.list_calibration_runs(conn)
        if calib_runs:
            st.dataframe(
                pd.DataFrame([{k: v for k, v in r.items() if not k.endswith("_json")} for r in calib_runs]),
                width="stretch",
            )
        else:
            st.info("No calibration runs yet.")

# --------------------------------------------------------------------------
# 2. Screening & Optimization (Virtual Experiments, Optimization, Pareto,
#    Sensitivity, Recommend Next)
# --------------------------------------------------------------------------
with tabs[2]:
    page = st.radio(
        "Section",
        ['Virtual Experiments', 'Optimization', 'Pareto Frontier', 'Sensitivity', 'Recommend Next'],
        horizontal=True,
        key="screening_page",
        label_visibility="collapsed",
    )
    st.divider()
    if page == 'Virtual Experiments':
        st.subheader("Virtual Experiments")
        st.caption(
            "Uses the Simple physics model regardless of the setup toggle, for consistency with "
            "the Optimization/Pareto/Sensitivity/ML tabs below."
        )
        n = st.slider("Number of virtual experiments", 100, 10000, 500, step=100)
        if st.button("GENERATE VIRTUAL EXPERIMENTS"):
            df = generate_virtual_experiments(n, seed=42)
            st.session_state["virtual_df"] = df
            df.to_csv("data/synthetic_experiments.csv", index=False)
            st.success(f"Generated {len(df)} synthetic physics experiments.")
        if "virtual_df" in st.session_state:
            st.warning("Demo data - not real results")
            st.dataframe(st.session_state["virtual_df"].head(100), width="stretch")
            st.download_button(
                "Download CSV",
                st.session_state["virtual_df"].to_csv(index=False),
                "synthetic_experiments.csv",
                "text/csv",
            )
    elif page == 'Optimization':
        st.subheader("Optimization")
        if st.button("OPTIMIZE"):
            df = generate_virtual_experiments(2000, seed=42)
            best = optimize_candidates(df, 10)
            st.dataframe(best, width="stretch")
    elif page == 'Pareto Frontier':
        st.subheader("Pareto Frontier")
        if st.button("CALCULATE PARETO FRONTIER"):
            df = generate_virtual_experiments(1000, seed=42)
            p = pareto_frontier(df)
            st.plotly_chart(
                __import__("plotly.express").express.scatter(
                    p,
                    x="flux_LMH",
                    y="rejection_percent",
                    hover_data=["feasibility_score"],
                    title="Pareto-optimal Flux vs Rejection",
                ),
                width="stretch",
            )
            st.dataframe(p.head(50), width="stretch")
    elif page == 'Sensitivity':
        st.subheader("Sensitivity Analysis")
        st.caption(
            "One-at-a-time sweep (±50%) of each parameter; larger `feasibility_range` means the "
            "score is more sensitive to that input."
        )
        if st.button("RUN SENSITIVITY"):
            try:
                sres = sensitivity_analysis(membrane, water, op, targets)
                st.dataframe(sres, width="stretch")
                st.bar_chart(sres.set_index("parameter")["feasibility_range"])
            except Exception as e:
                st.error(str(e))
    elif page == 'Recommend Next':
        st.subheader("What should we test next?")
        if "virtual_df" in st.session_state:
            if st.button("RECOMMEND EXPERIMENTS"):
                rec = recommend_experiments(st.session_state["virtual_df"], top_n=5)
                st.dataframe(rec, width="stretch")
        else:
            st.info("Generate virtual experiments first.")

# --------------------------------------------------------------------------
# 3. ML Lab
# --------------------------------------------------------------------------
with tabs[3]:
    st.subheader("ML Lab")

    with st.expander("1. Train models", expanded=True):
        n_synth = st.slider("Synthetic rows", 100, 5000, 500, key="ml_n_synth")
        include_exp = st.checkbox("Include measured experiments from the lab notebook", value=True)
        use_engineered = st.checkbox("Use physics-informed engineered features (recommended — substantially lowers error)", value=True)
        if st.button("TRAIN ML MODELS"):
            df_train, summary = assemble_training_dataset(
                conn, n_synthetic=n_synth, seed=42, include_experimental=include_exp
            )
            models, metrics, feature_list = train_models(df_train, use_engineered_features=use_engineered)
            st.session_state["ml_models"] = models
            st.session_state["ml_metrics"] = metrics
            st.session_state["ml_feature_list"] = feature_list
            st.session_state["ml_training_df"] = df_train
            st.success(
                f"Trained on {summary['total_rows']} rows "
                f"({summary['synthetic_rows']} synthetic + {summary['experimental_rows']} measured)."
            )
            if summary["synthetic_rows"]:
                st.warning("Demo data - not real results. Synthetic rows are labeled separately from measured data.")
        if "ml_metrics" in st.session_state:
            st.dataframe(pd.DataFrame(st.session_state["ml_metrics"]), width="stretch")
            if st.button("REGISTER MODELS"):
                save_and_register_models(
                    conn,
                    st.session_state["ml_models"],
                    st.session_state["ml_metrics"],
                    st.session_state["ml_feature_list"],
                    len(st.session_state["ml_training_df"]),
                    "combined" if include_exp else "synthetic",
                )
                st.success("Registered — these models are now the active ones for prediction.")

    with st.expander("2. Cross-validate"):
        if "ml_training_df" in st.session_state and st.button("RUN 5-FOLD CROSS-VALIDATION"):
            results, skipped = cross_validate_all_targets(st.session_state["ml_training_df"], n_splits=5)
            st.dataframe(pd.DataFrame(results), width="stretch")
            if skipped:
                st.warning(f"Skipped (too few labeled rows): {[s['target'] for s in skipped]}")
        elif "ml_training_df" not in st.session_state:
            st.info("Train models first (section 1) to get a dataset to cross-validate on.")

    with st.expander("3. Compare model types"):
        if "ml_training_df" in st.session_state:
            target_choice = st.selectbox("Target", ML_TARGETS, key="compare_target")
            if st.button("COMPARE MODEL TYPES"):
                comparison = compare_model_types(
                    st.session_state["ml_training_df"], target_choice, n_splits=5
                )
                st.dataframe(pd.DataFrame(comparison), width="stretch")
        else:
            st.info("Train models first (section 1).")

    with st.expander("4. Predict the current experiment candidate"):
        active_models = load_all_active_models(conn)
        if not active_models:
            st.info("No models registered yet — train and register them in section 1.")
        else:
            candidate = {
                **{
                    f: getattr(membrane, f)
                    for f in [
                        "thickness_um",
                        "permeability_LMH_bar",
                        "pore_size_nm",
                        "MWCO",
                        "porosity",
                        "surface_charge",
                        "hydrophilicity",
                        "baseline_rejection_percent",
                    ]
                },
                **{
                    f"water_{f}": v
                    for f, v in {
                        "feed_concentration_mg_L": water.feed_concentration_mg_L,
                        "turbidity_NTU": water.turbidity_NTU,
                        "TDS_mg_L": water.TDS_mg_L,
                        "pH": water.pH,
                        "temperature_C": water.temperature_C,
                    }.items()
                },
                **{
                    f"op_{f}": v
                    for f, v in {
                        "TMP_bar": op.TMP_bar,
                        "feed_flow_L_min": op.feed_flow_L_min,
                        "crossflow_velocity_m_s": op.crossflow_velocity_m_s,
                        "recovery_percent": op.recovery_percent,
                        "operating_time_hr": op.operating_time_hr,
                    }.items()
                },
            }
            if st.button("PREDICT WITH ML"):
                ref_df = (
                    st.session_state["ml_training_df"]
                    if "ml_training_df" in st.session_state
                    else st.session_state.get("virtual_df")
                )
                result = predict_candidate(conn, candidate, reference_df=ref_df)
                if result["predictions"]:
                    pred_cols = st.columns(len(result["predictions"]))
                    for pc, (target, value) in zip(pred_cols, result["predictions"].items()):
                        pc.metric(target.replace("_", " ").title(), f"{value:.2f}")
                else:
                    st.info("No predictions available for this candidate.")
                if result["skipped"]:
                    st.warning(f"Skipped targets: {result['skipped']}")
                if result["ood"] and result["ood"]["ood"]:
                    flagged = [f for f, v in result["ood"]["features"].items() if v]
                    st.warning(f"Out of the training data's range for: {flagged}")
                elif result["ood"]:
                    st.info("Candidate is within the training data's per-feature range.")

    with st.expander("5. Uncertainty (Random Forest / Extra Trees only)"):
        target_choice_u = st.selectbox("Target", ML_TARGETS, key="uncertainty_target")
        model_u, meta_u = load_active_model(conn, target_choice_u)
        if model_u is None:
            st.info("No registered model for this target yet.")
        elif not supports_uncertainty(model_u):
            st.warning(
                f"The active model for {target_choice_u} is a {meta_u['model_type']}, which "
                "doesn't support ensemble-spread uncertainty (see ml/uncertainty.py)."
            )
        else:
            candidate_u = {
                **{
                    f: getattr(membrane, f)
                    for f in [
                        "thickness_um",
                        "permeability_LMH_bar",
                        "pore_size_nm",
                        "MWCO",
                        "porosity",
                        "surface_charge",
                        "hydrophilicity",
                        "baseline_rejection_percent",
                    ]
                },
                **{
                    f"water_{f}": v
                    for f, v in {
                        "feed_concentration_mg_L": water.feed_concentration_mg_L,
                        "turbidity_NTU": water.turbidity_NTU,
                        "TDS_mg_L": water.TDS_mg_L,
                        "pH": water.pH,
                        "temperature_C": water.temperature_C,
                    }.items()
                },
                **{
                    f"op_{f}": v
                    for f, v in {
                        "TMP_bar": op.TMP_bar,
                        "feed_flow_L_min": op.feed_flow_L_min,
                        "crossflow_velocity_m_s": op.crossflow_velocity_m_s,
                        "recovery_percent": op.recovery_percent,
                        "operating_time_hr": op.operating_time_hr,
                    }.items()
                },
            }
            missing = [f for f in meta_u["feature_list"] if f not in candidate_u]
            if missing:
                ref_df_u = (
                    st.session_state["ml_training_df"]
                    if "ml_training_df" in st.session_state
                    else st.session_state.get("virtual_df")
                )
                if ref_df_u is not None:
                    for f in missing:
                        candidate_u[f] = ref_df_u[f].median() if f in ref_df_u.columns else 0.0
                else:
                    for f in missing:
                        candidate_u[f] = 0.0
            mean, std = predict_one_with_uncertainty(model_u, candidate_u, meta_u["feature_list"])
            st.metric(f"Predicted {target_choice_u}", f"{mean:.2f} ± {std:.2f}")
            st.caption(
                "± is the standard deviation across the ensemble's individual trees, not a "
                "formal confidence interval — see ml/uncertainty.py."
            )

    with st.expander("6. Explainability"):
        target_choice_e = st.selectbox("Target", ML_TARGETS, key="explain_target")
        model_e, meta_e = load_active_model(conn, target_choice_e)
        if model_e is None:
            st.info("No registered model for this target yet.")
        else:
            split = used_vs_unused_features(model_e, meta_e["feature_list"])
            ranked_df = pd.DataFrame(split["ranked"])
            st.bar_chart(ranked_df.set_index("feature")["importance"])
            st.write("Features the model relies on:", split["used"])
            st.write("Features with near-zero importance:", split["unused"])

    with st.expander("7. Model registry"):
        registered = repo.list_models(conn, active_only=False)
        if registered:
            st.dataframe(
                pd.DataFrame([{k: v for k, v in r.items() if not k.endswith("_json")} for r in registered]),
                width="stretch",
            )
        else:
            st.info("No models registered yet.")

# --------------------------------------------------------------------------
# 4. CS–CA–Biochar Pb(II) research lab
# --------------------------------------------------------------------------
with tabs[4]:
    st.subheader("CS–CA–Biochar Pb(II) Research Lab")
    st.caption("Experimental values are user-supplied. Model outputs are predictions, not validation.")
    st.markdown(
        """
        <div class="pms-card-grid">
            <div class="pms-card">
                <h4>Formulation</h4>
                <p>Capture the membrane recipe and track the CS–CA–biochar composition for each sample.</p>
            </div>
            <div class="pms-card">
                <h4>Adsorption</h4>
                <p>Compute removal efficiency and adsorptive capacity directly from Pb(II) input and equilibrium data.</p>
            </div>
            <div class="pms-card">
                <h4>Analysis</h4>
                <p>Fit response-surface and ML models, then compare candidates before selecting a recommended optimum.</p>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    section = st.radio(
        "Research section",
        ["Membrane Recipes", "Lead removal calculator", "Experiment Planner", "Enter Lab Results", "Experimental data",
         "What Affects Lead Removal?", "RSM analysis", "Compare Prediction Methods", "Find the Best Recipe", "Confirm in the Lab", "Paper Center",
         "Extra Lab Tests", "Compare with Other Materials", "Data Safety",
         "Help and Glossary"],
        horizontal=True, key="research_section", label_visibility="collapsed",
    )
    if section == "Membrane Recipes":
        with st.form("research_formulation_form"):
            membrane_id = st.text_input("Membrane/sample ID")
            chitosan = st.number_input("Chitosan (wt%)", 0.0, 100.0, 40.0)
            cellulose = st.number_input("Cellulose acetate (wt%)", 0.0, 100.0, 57.0)
            biochar = st.number_input("Biochar loading (wt%)", 0.0, 100.0, 3.0)
            fabrication = st.text_area("Fabrication information")
            notes = st.text_area("Notes")
            submitted = st.form_submit_button("SAVE FORMULATION")
        if submitted:
            try:
                formulation = MembraneFormulation(membrane_id, chitosan, cellulose, biochar,
                                                   fabrication_info=fabrication, notes=notes)
                repo.save_research_formulation(conn, formulation)
                st.success("Formulation saved.")
            except ValueError as exc:
                st.error(str(exc))
        rows = repo.list_research_formulations(conn)
        if rows:
            st.dataframe(pd.DataFrame(rows), width="stretch")
    elif section == "Lead removal calculator":
        c0 = st.number_input("Initial Pb concentration C0 (mg/L)", min_value=0.0001, value=30.0)
        ce = st.number_input("Final/equilibrium Pb concentration Ce (mg/L)", min_value=0.0, value=5.0)
        volume = st.number_input("Solution volume (L)", min_value=0.0001, value=0.1)
        mass = st.number_input("Membrane mass (g)", min_value=0.0001, value=0.1)
        try:
            metrics = calculate_pb_metrics(c0, ce, volume, mass)
            st.latex(r"R(\%) = \frac{C_0-C_e}{C_0}\times100")
            st.latex(r"q_e = \frac{(C_0-C_e)V}{m}")
            st.metric("Pb(II) removal (%)", f"{metrics['removal_percent']:.3f}")
            st.metric("Adsorption capacity qe (mg/g)", f"{metrics['qe_mg_g']:.3f}")
        except ValueError as exc:
            st.error(str(exc))
    elif section == "Experiment Planner":
        design = generate_box_behnken_design()
        st.write("Four-factor Box–Behnken design: 29 runs, including five centre points.")
        st.dataframe(design, width="stretch")
        st.download_button("Download BBD CSV", design.to_csv(index=False), "cs_ca_biochar_bbd.csv", "text/csv")
        design_excel = BytesIO()
        design.to_excel(design_excel, index=False)
        st.download_button("Download BBD Excel", design_excel.getvalue(), "cs_ca_biochar_bbd.xlsx",
                           "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
    elif section in {"Enter Lab Results", "Experimental data"}:
        st.subheader("Manual laboratory entry")
        st.caption(
            "Enter one measured Box–Behnken run at a time. The app calculates Pb(II) "
            "removal and qe from your laboratory measurements; it does not create experimental values."
        )
        design = generate_box_behnken_design()
        run_ids = design["run_id"].tolist()
        run_id = st.selectbox(
            "Box–Behnken run",
            run_ids,
            format_func=lambda value: (
                f"{value} — "
                + ", ".join(
                    f"{label}={design.loc[design['run_id'].eq(value), factor].iloc[0]:g}"
                    for factor, label in [
                        ("chitosan_wt_percent", "chitosan"),
                        ("biochar_wt_percent", "biochar"),
                        ("pH", "pH"),
                        ("initial_pb_mg_l", "initial Pb"),
                    ]
                )
            ),
            key="manual_bbd_run",
        )
        selected_run = design.loc[design["run_id"].eq(run_id)].iloc[0]
        with st.form("manual_research_experiment_form"):
            st.write(
                f"Selected conditions: chitosan {selected_run['chitosan_wt_percent']:g} wt%, "
                f"biochar {selected_run['biochar_wt_percent']:g} wt%, pH {selected_run['pH']:g}, "
                f"initial Pb(II) {selected_run['initial_pb_mg_l']:g} mg/L."
            )
            experiment_id = st.text_input("Experiment ID", value=run_id, key="manual_experiment_id")
            membrane_id = st.text_input("Membrane/sample ID (optional)")
            cellulose = st.number_input("Cellulose acetate (wt%, optional)", min_value=0.0, value=0.0)
            final_pb = st.number_input(
                "Measured final/equilibrium Pb(II), Ce (mg/L)",
                min_value=0.0,
                value=5.0,
                key="manual_final_pb",
            )
            contact_time = st.number_input("Contact time (minutes, optional)", min_value=0.0, value=0.0)
            volume = st.number_input("Solution volume (L)", min_value=0.0001, value=0.1)
            mass = st.number_input("Membrane mass (g)", min_value=0.0001, value=0.1)
            replicate = st.number_input("Replicate number (optional)", min_value=0, step=1, value=0)
            notes = st.text_area("Notes (optional)")
            manual_submitted = st.form_submit_button("SAVE MEASURED RUN", key="manual_save_run")
        if manual_submitted:
            record = {
                "experiment_id": experiment_id.strip(),
                "membrane_id": membrane_id.strip() or None,
                "chitosan_wt_percent": selected_run["chitosan_wt_percent"],
                "cellulose_acetate_wt_percent": cellulose or None,
                "biochar_wt_percent": selected_run["biochar_wt_percent"],
                "pH": selected_run["pH"],
                "initial_pb_mg_l": selected_run["initial_pb_mg_l"],
                "final_pb_mg_l": final_pb,
                "contact_time_min": contact_time or None,
                "solution_volume_l": volume,
                "membrane_mass_g": mass,
                "replicate_number": replicate or None,
                "notes": notes.strip() or None,
            }
            try:
                if not record["experiment_id"]:
                    raise ValueError("Experiment ID is required.")
                measured, report = validate_experiment_frame(pd.DataFrame([record]))
                if report["valid_rows"] != 1:
                    st.error("; ".join(report["errors"]) or "The measured run could not be validated.")
                else:
                    repo.save_research_experiment(conn, measured.iloc[0].to_dict())
                    st.success(
                        f"Saved {record['experiment_id']} with removal "
                        f"{measured.iloc[0]['removal_percent']:.3f}% and qe "
                        f"{measured.iloc[0]['qe_mg_g']:.3f} mg/g."
                    )
            except (TypeError, ValueError, OSError) as exc:
                st.error(str(exc))

        st.download_button(
            "Download blank data-entry template",
            design.assign(
                experiment_id=design["run_id"],
                final_pb_mg_l=pd.NA,
                solution_volume_l=pd.NA,
                membrane_mass_g=pd.NA,
            ).to_csv(index=False),
            "cs_ca_biochar_manual_entry_template.csv",
            "text/csv",
            help="Print this template or give it to the laboratory team if they prefer paper entry.",
        )
        st.divider()
        st.subheader("Upload completed data")
        upload = st.file_uploader("Experimental dataset (CSV or Excel)", type=["csv", "xlsx", "xls"], key="research_upload")
        if upload:
            try:
                imported, report = read_experimental_data(upload, upload.name)
                st.write(report)
                st.dataframe(imported, width="stretch")
                if report["valid_rows"] and st.button("SAVE VALID RESEARCH EXPERIMENTS"):
                    for record in imported.loc[report["valid_indices"]].to_dict("records"):
                        repo.save_research_experiment(conn, record)
                    st.success(f"Saved {report['valid_rows']} validated research records.")
            except (ImportError, ValueError, OSError) as exc:
                st.error(str(exc))
    elif section in {"What Affects Lead Removal?", "RSM analysis"}:
        badge("Predicted from measured data", "predicted")
        render_rsm_analysis(st, repo.list_research_experiments(conn))
    elif section == "Compare Prediction Methods":
        badge("Predicted/model comparison", "predicted")
        rows = repo.list_research_experiments(conn)
        measured = [row for row in rows if row.get("data_source", "experimental") == "experimental"]
        st.caption("Only rows marked experimental are eligible for paper-model comparison.")
        if len(measured) < 10:
            st.info("Enter at least ten complete measured runs for repeated 5-fold comparison.")
        elif st.button("COMPARE RESEARCH MODELS"):
            try:
                measured_frame = pd.DataFrame(measured)
                groups = measured_frame[RESEARCH_FEATURES].astype(str).agg("|".join, axis=1)
                comparison, predictions = nested_compare(measured_frame, groups=groups)
                st.dataframe(comparison, width="stretch")
                st.download_button("Download model comparison CSV", comparison.to_csv(index=False),
                                   "model_comparison.csv", "text/csv")
                excel = BytesIO()
                comparison.to_excel(excel, index=False)
                st.download_button("Download model comparison Excel", excel.getvalue(),
                                   "model_comparison.xlsx", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
                best_model, best_name = fit_best_model(measured_frame, comparison, seed=42)
                permutation = explain_model(best_model, measured_frame, seed=42)
                shap_result = shap_values_if_available(best_model, measured_frame)
                st.markdown(f"**Best model: {best_name} (predicted/model-comparison output)**")
                st.dataframe(permutation, width="stretch", hide_index=True)
                if shap_result is not None:
                    st.caption("SHAP values are model explanations, not experimental effects.")
                report = {"data_source": "experimental", "seed": 42,
                          "settings": {"protocol": "5-fold repeated 10 times, group-aware"},
                          "results": comparison.to_dict("records")}
                st.download_button("Download run report JSON", report_json_bytes(report),
                                   "research_run_report.json", "application/json")
                st.session_state["last_research_comparison"] = comparison
                st.session_state["last_research_predictions"] = predictions
            except ValueError as exc:
                st.error(str(exc))
    elif section == "Find the Best Recipe":
        rows = repo.list_research_experiments(conn)
        measured = pd.DataFrame([row for row in rows if row.get("data_source", "experimental") == "experimental"])
        if len(measured) < 5:
            st.info("Enter at least five complete measured research rows before optimization.")
        else:
            try:
                factors = list(RESEARCH_FEATURES)
                usable = measured[factors + ["removal_percent"]].apply(pd.to_numeric, errors="coerce").dropna()
                if len(usable) < 5:
                    st.info("Complete measured factor and removal values are required.")
                elif st.button("SUGGEST NEXT EXPERIMENT"):
                    suggestion = suggest_next_experiment(usable, factors, "removal_percent",
                                                         [(20, 60), (1, 5), (3, 6), (10, 50)])
                    st.warning(suggestion["data_source"])
                    st.dataframe(pd.DataFrame([suggestion["factors"]]), hide_index=True)
            except (ValueError, KeyError) as exc:
                st.error(str(exc))
    elif section == "Confirm in the Lab":
        st.info("Enter confirmation replicates after a model-predicted optimum has been calculated.")
    elif section == "Paper Center":
        st.caption("What this page does: collect paper tables, figures, and provenance. What you need first: measured BBD data.")
        paper_checklist = pd.DataFrame([
            {"paper item": item, "status": "Available after export"}
            for item in ["Table IV - design and results", "Table V - ANOVA", "Table VI - model comparison",
                         "Table VII - optimum and confirmation", "Table VIII - material comparison",
                         "Fig. 6 - response surfaces", "Fig. 7 - parity and SHAP"]
        ])
        st.dataframe(paper_checklist, width="stretch", hide_index=True)
        st.download_button(
            "Download run report",
            report_json_bytes({"data_source": "user-supplied measurement upload", "seed": 42,
                               "settings": {"workflow": "paper export"}, "results": {}}),
            "paper_run_report.json",
            "application/json",
        )
        st.divider()
        st.caption("Export tables from a completed BBD measurement CSV. No values are invented.")
        source = st.file_uploader(
            "Completed BBD measurement template (CSV or Excel)",
            type=["csv", "xlsx", "xls"],
            key="paper_export_upload",
        )
        output_name = st.text_input("Output folder", value="paper_export")
        if source and st.button("EXPORT PAPER RESULTS"):
            try:
                export_paper_results(source, output_name)
                st.success(f"Paper outputs written to {output_name}.")
            except (ImportError, ValueError, OSError) as exc:
                st.error(str(exc))
    elif section == "Extra Lab Tests":
        st.caption("What this page does: collect supporting membrane evidence. What you need first: measured masses or lab test values.")
        extra_tabs = st.tabs(["Water uptake", "How much it absorbs", "How fast it absorbs", "Reuse test", "Material characterization"])
        with extra_tabs[0]:
            with st.form("research_calculators"):
                wet_mass = st.number_input("Wet mass (g)", min_value=0.0, value=1.2, help="Mass after water exposure, in grams.")
                dry_mass = st.number_input("Dry mass (g)", min_value=0.0001, value=1.0, help="Mass before water exposure, in grams.")
                water_density = st.number_input("Water density (g/cm3)", min_value=0.0001, value=1.0, help="Water density used in the porosity equation.")
                area = st.number_input("Area (cm2)", min_value=0.0001, value=10.0, help="Measured membrane area in square centimetres.")
                thickness = st.number_input("Thickness (cm)", min_value=0.0001, value=2.0, help="Measured dry thickness in centimetres.")
                calculate = st.form_submit_button("Calculate water uptake")
            if calculate:
                try:
                    st.metric("Water uptake (%)", f"{swelling_degree(wet_mass, dry_mass):.3f}")
                    st.metric("Open space in the membrane (%)", f"{porosity(wet_mass, dry_mass, water_density, area, thickness):.3f}")
                except ValueError as exc:
                    st.error(str(exc))
        with extra_tabs[1]:
            st.caption("In technical terms: Langmuir and Freundlich isotherm fits.")
            concentrations = st.text_input("Lead levels (mg/L, comma separated)", "5, 10, 20")
            qe_values = st.text_input("Absorbed amounts (mg/g, comma separated)", "2, 4, 7")
            if st.button("Fit absorption models", key="fit_isotherms"):
                try:
                    fits = fit_isotherms([float(x) for x in concentrations.split(",")], [float(x) for x in qe_values.split(",")])
                    st.dataframe(pd.DataFrame({name: [fit["r2"]] for name, fit in fits.items()}, index=["How well it fits"]), width="stretch")
                except (ValueError, TypeError) as exc:
                    st.error(f"Please enter at least three valid concentration and absorbed-amount pairs. ({exc})")
        with extra_tabs[2]:
            st.caption("In technical terms: pseudo-first-order and pseudo-second-order kinetic fits.")
            times = st.text_input("Time (minutes, comma separated)", "0, 10, 20")
            qt_values = st.text_input("Absorbed amount over time (mg/g, comma separated)", "0, 3, 5")
            if st.button("Fit absorption speed models", key="fit_kinetics"):
                try:
                    fits = fit_kinetics([float(x) for x in times.split(",")], [float(x) for x in qt_values.split(",")])
                    st.dataframe(pd.DataFrame({name: [fit["r2"]] for name, fit in fits.items()}, index=["How well it fits"]), width="stretch")
                except (ValueError, TypeError) as exc:
                    st.error(f"Please enter at least three valid time and absorbed-amount pairs. ({exc})")
        with extra_tabs[3]:
            reuse_rows = repo.list_reuse_cycles(conn)
            if reuse_rows:
                reuse = regeneration_summary(reuse_rows)
                st.dataframe(reuse, width="stretch")
                st.line_chart(reuse.set_index("cycle")["removal_percent"])
                st.caption("Removal values are measured in the lab and entered in the database.")
            else:
                st.info("No reuse measurements yet. Add reuse-cycle records before fitting a reuse trend.")
        with extra_tabs[4]:
            st.caption("Record notes from FTIR, SEM, EDX, BET, contact-angle, stability, and porosity tests.")
            characterization = {field: st.text_input(field.replace("_", " ").title(), key=f"char_{field}") for field in ["ftir", "sem", "edx", "bet", "contact_angle", "porosity", "swelling", "acid_stability"]}
            if st.button("Review characterization notes", key="review_characterization"):
                st.dataframe(pd.DataFrame([validate_characterization(characterization)]), width="stretch")
    elif section == "Data Safety":
        st.caption("SQLite backup contains local lab records and does not create or validate experimental results.")
        if repo.DEFAULT_DB_PATH.exists():
            st.download_button("Download SQLite database", repo.DEFAULT_DB_PATH.read_bytes(),
                               "polymemsim.db", "application/x-sqlite3")
        uploaded_db = st.file_uploader("Upload SQLite database backup", type=["db", "sqlite", "sqlite3"])
        if uploaded_db and st.button("RESTORE DATABASE BACKUP"):
            try:
                repo.restore_database(BytesIO(uploaded_db.getvalue()), repo.DEFAULT_DB_PATH)
                st.success("Database restored. Restart the app session to reopen the connection.")
            except (OSError, ValueError) as exc:
                st.error(str(exc))
    elif section == "Compare with Other Materials":
        st.caption("Only user-entered literature values are shown; blank fields remain blank and are not inferred.")
        if "adsorbent_rows" not in st.session_state:
            st.session_state["adsorbent_rows"] = []
        with st.form("adsorbent_comparison_form"):
            name = st.text_input("Adsorbent")
            qmax = st.number_input("qmax (mg/g)", min_value=0.0, value=0.0)
            comparison_ph = st.number_input("pH", min_value=0.0, max_value=14.0, value=7.0)
            reference = st.text_input("Reference")
            add_row = st.form_submit_button("ADD ROW")
        if add_row:
            st.session_state["adsorbent_rows"].append({"adsorbent": name, "qmax_mg_g": qmax,
                                                        "pH": comparison_ph, "reference": reference})
        table = comparison_table(st.session_state["adsorbent_rows"])
        st.dataframe(table, width="stretch", hide_index=True)
    elif section == "Help and Glossary":
        st.caption("What this page does: explain the workflow in plain English. What you need first: nothing.")
        with st.expander("How to use POLYMEMSIM", expanded=True):
            st.markdown("1. Save a membrane recipe.\n2. Plan the 29 experiments.\n3. Enter measured results.\n4. Fit and inspect the model.\n5. Compare methods and find a candidate recipe.\n6. Confirm the candidate in the lab and export the paper tables.")
        search = st.text_input("Search the glossary", placeholder="Try: safe range, uptake, ANOVA")
        glossary = pd.DataFrame([{"plain name": item["plain"], "technical caption": item["technical"]} for item in LABELS.values()])
        if search.strip():
            mask = glossary.astype(str).apply(lambda column: column.str.contains(search, case=False, na=False)).any(axis=1)
            glossary = glossary.loc[mask]
        st.dataframe(glossary, width="stretch", hide_index=True)
        st.info("Research results come from supplied measurements; synthetic data and model predictions are not experimental validation.")

# --------------------------------------------------------------------------
# 5. Feasibility
# --------------------------------------------------------------------------
with tabs[5]:
    st.subheader("Membrane Feasibility Assessment")
    result = run_current_model(membrane, water, op, targets)
    calibration_summary = None
    if active_sample_id is not None and repo.sample_has_calibration(conn, active_sample_id):
        runs = repo.list_calibration_runs(conn, sample_id=None)
        matching = [
            r for r in repo.list_calibration_runs(conn) if r["calibrated_sample_id"] == active_sample_id
        ]
        if matching:
            calibration_summary = {"post_fit_r2": matching[0]["metrics_after"].get("R2")}
    assessment = feasibility_assessment(result, targets, calibration=calibration_summary)
    score_col, verdict_col = st.columns([1, 2])
    with score_col:
        score_display(result["feasibility_score"])
    with verdict_col:
        st.markdown("<br>", unsafe_allow_html=True)
        rec = assessment["recommendation"]
        tone = (
            "strong"
            if "STRONG" in rec
            else "promising" if "PROMISING" in rec else "low" if "LOW PRIORITY" in rec else "neutral"
        )
        verdict_badge(rec, tone)
        st.write(assessment["reason"])
    st.markdown("**Target checks**")
    for name, passed in assessment["target_checks"].items():
        check_row(name.replace("_", " ").title(), passed)
    reliability = assessment["reliability"]
    if reliability == "LOW":
        st.warning(
            "Prediction reliability: LOW — no calibration on record for this sample "
            "(see Validation & Calibration tab)."
        )
    elif reliability == "MEDIUM":
        st.info(
            "Prediction reliability: MEDIUM — this sample has a calibration on record with a moderate fit."
        )
    else:
        st.success(
            "Prediction reliability: HIGH — this sample has a calibration on record with a strong fit."
        )

page_footer(st, STAGES)
