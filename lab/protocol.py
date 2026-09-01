"""A guided, multi-step protocol runner — turns the checklist in
docs/experimental_protocol.md into an actual stateful workflow backed
by the `protocol_runs` table (see db.repository), instead of being just
a markdown document nobody's UI ever references.

Each step is a (name, description, fields) triple: `fields` names what
should be recorded at that step (used to drive a form in the app; not
enforced here beyond being stored as freeform values).
"""

from db import repository as repo

PROTOCOL_STEPS = [
    {
        "name": "Formulation & structure",
        "description": "Record the membrane formulation, polymer, membrane type, and thickness.",
        "fields": ["polymer_name", "membrane_type", "thickness_um"],
    },
    {
        "name": "Operating setup",
        "description": "Record trans-membrane pressure, temperature and feed flow for this run.",
        "fields": ["TMP_bar", "temperature_C", "feed_flow_L_min"],
    },
    {
        "name": "Feed characterization",
        "description": "Record feed concentration and pH.",
        "fields": ["feed_concentration_mg_L", "pH"],
    },
    {
        "name": "Permeate volume & flux",
        "description": "Record measured permeate volume and the flux computed from it.",
        "fields": ["permeate_volume_mL", "elapsed_time_min", "flux_LMH"],
    },
    {
        "name": "Permeate concentration & rejection",
        "description": "Record measured permeate concentration and compute/record rejection.",
        "fields": ["permeate_concentration_mg_L", "rejection_percent"],
    },
    {
        "name": "Flux vs time (fouling)",
        "description": "Record flux at several time points to characterize fouling behavior.",
        "fields": ["flux_vs_time_readings"],
    },
    {
        "name": "Finalize",
        "description": "Review all recorded values and save this protocol run as complete.",
        "fields": [],
    },
]

STEP_NAMES = [s["name"] for s in PROTOCOL_STEPS]


def start_protocol(conn, sample_id, name="Protocol run"):
    """Begin a new guided protocol run for `sample_id`. Returns the new
    protocol_run id."""
    return repo.create_protocol_run(conn, sample_id, name, STEP_NAMES)


def current_step_definition(run):
    """The step-definition dict (name/description/fields) for a
    protocol run's current step, or None if the run is already
    complete."""
    idx = run["current_step"]
    if idx >= len(PROTOCOL_STEPS):
        return None
    return PROTOCOL_STEPS[idx]


def record_current_step(conn, protocol_run_id, values):
    """Record `values` (dict) against the run's current step and
    advance to the next one. Automatically marks the run complete once
    every step has been recorded."""
    run = repo.get_protocol_run(conn, protocol_run_id)
    if run is None:
        raise ValueError(f"No protocol run with id {protocol_run_id}")
    step_def = current_step_definition(run)
    if step_def is None:
        raise ValueError("This protocol run is already complete.")
    repo.update_protocol_step(conn, protocol_run_id, step_def["name"], values)
    will_complete = (run["current_step"] + 1) >= len(PROTOCOL_STEPS)
    repo.advance_protocol_run(conn, protocol_run_id, complete=will_complete)
    return repo.get_protocol_run(conn, protocol_run_id)


def is_complete(run):
    return run["status"] == "complete"


def progress_fraction(run):
    """0.0-1.0 fraction of steps completed, for a progress bar."""
    return min(1.0, run["current_step"] / len(PROTOCOL_STEPS))


def collected_values(run):
    """Flatten every recorded step's values into one dict (later steps'
    keys win on name collision, which shouldn't normally happen since
    each step records a distinct field set)."""
    out = {}
    for step_name in STEP_NAMES:
        out.update(run["steps"].get(step_name, {}))
    return out
