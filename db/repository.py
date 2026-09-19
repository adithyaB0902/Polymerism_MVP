"""Connection management and CRUD helpers for the lab database.

All functions take an optional `db_path` (defaults to
`data/polymemsim.db`) so tests can point at a temporary file instead of
the real database. Every write commits immediately (single-user local
app; no need for explicit transaction batching).
"""

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

from .schema import SCHEMA

DEFAULT_DB_PATH = Path(__file__).resolve().parent.parent / "data" / "polymemsim.db"


def _now():
    return datetime.now(timezone.utc).isoformat()


def get_connection(db_path=None):
    """Open a connection to the lab database, creating the schema on
    first use. Returns rows as `sqlite3.Row` (dict-like access by
    column name).

    `check_same_thread=False`: Streamlit can execute successive reruns
    of the same session on different worker threads, but a
    `@st.cache_resource`-cached connection object is shared across all
    of them — sqlite3's default same-thread check would otherwise raise
    `ProgrammingError` the first time a rerun happened to land on a
    different thread than the one that opened the connection. This is
    safe here because the app is single-user/local and never issues
    concurrent writes from multiple threads at once.
    """
    path = Path(db_path) if db_path else DEFAULT_DB_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    conn.executescript(SCHEMA)
    return conn


def _row_to_dict(row):
    return dict(row) if row is not None else None


def _rows_to_dicts(rows):
    return [dict(r) for r in rows]


# --------------------------------------------------------------------------
# Batches
# --------------------------------------------------------------------------

def create_batch(conn, name, notes=None):
    cur = conn.execute("INSERT INTO batches (created_at, name, notes) VALUES (?,?,?)",
                       (_now(), name, notes))
    conn.commit()
    return cur.lastrowid


def list_batches(conn):
    return _rows_to_dicts(conn.execute("SELECT * FROM batches ORDER BY id DESC").fetchall())


# --------------------------------------------------------------------------
# Samples
# --------------------------------------------------------------------------

SAMPLE_FIELDS = ["polymer_name", "membrane_type", "thickness_um", "permeability_LMH_bar",
                 "pore_size_nm", "MWCO", "porosity", "surface_charge", "hydrophilicity",
                 "membrane_area_cm2", "baseline_rejection_percent", "fouling_coefficient"]


def create_sample(conn, fields, batch_id=None, notes=None, origin="manual", calibrated_from_sample_id=None):
    """`fields` is a dict covering (a subset of) SAMPLE_FIELDS; missing
    ones default per the schema."""
    cols = ["created_at", "batch_id", "notes", "origin", "calibrated_from_sample_id"] + SAMPLE_FIELDS
    vals = [_now(), batch_id, notes, origin, calibrated_from_sample_id] + [fields.get(f) for f in SAMPLE_FIELDS]
    placeholders = ",".join(["?"] * len(cols))
    cur = conn.execute(f"INSERT INTO samples ({','.join(cols)}) VALUES ({placeholders})", vals)
    conn.commit()
    return cur.lastrowid


def get_sample(conn, sample_id):
    return _row_to_dict(conn.execute("SELECT * FROM samples WHERE id=?", (sample_id,)).fetchone())


def list_samples(conn, batch_id=None):
    if batch_id is not None:
        rows = conn.execute("SELECT * FROM samples WHERE batch_id=? ORDER BY id DESC", (batch_id,)).fetchall()
    else:
        rows = conn.execute("SELECT * FROM samples ORDER BY id DESC").fetchall()
    return _rows_to_dicts(rows)


# --------------------------------------------------------------------------
# Experiments
# --------------------------------------------------------------------------

WATER_FIELDS = ["contaminant", "feed_concentration_mg_L", "turbidity_NTU", "TDS_mg_L",
                "pH", "temperature_C", "viscosity_Pa_s", "density_kg_m3"]
OP_FIELDS = ["TMP_bar", "feed_flow_L_min", "crossflow_velocity_m_s", "recovery_percent",
            "operating_time_hr", "membrane_area_m2", "pump_efficiency", "electricity_price_per_kwh"]
RESULT_FIELDS = ["flux_LMH", "initial_flux_LMH", "rejection_percent", "permeate_mg_L",
                 "flux_decline_percent", "energy_kWh_m3", "cost_per_m3", "feasibility_score",
                 "total_permeate_flow_L_hr", "concentrate_mg_L"]


def record_experiment(conn, sample_id, water, op, result, physics_model="simple",
                      data_source="simulated", replicate_group_id=None,
                      protocol_run_id=None, notes=None):
    """Persist one experiment run.

    `water`/`op` are dicts keyed by the unprefixed field names (see
    WATER_FIELDS/OP_FIELDS); `result` is a dict keyed by RESULT_FIELDS
    (extra keys are ignored, missing ones stored as NULL).
    """
    cols = ["created_at", "sample_id", "replicate_group_id", "physics_model", "data_source",
            "protocol_run_id", "notes"]
    vals = [_now(), sample_id, replicate_group_id, physics_model, data_source, protocol_run_id, notes]
    for f in WATER_FIELDS:
        cols.append(f"water_{f}")
        vals.append(water.get(f))
    for f in OP_FIELDS:
        cols.append(f"op_{f}")
        vals.append(op.get(f))
    for f in RESULT_FIELDS:
        cols.append(f)
        vals.append(result.get(f))
    placeholders = ",".join(["?"] * len(cols))
    cur = conn.execute(f"INSERT INTO experiments ({','.join(cols)}) VALUES ({placeholders})", vals)
    conn.commit()
    return cur.lastrowid


def list_experiments(conn, sample_id=None, data_source=None, replicate_group_id=None, limit=None):
    query = "SELECT * FROM experiments WHERE 1=1"
    params = []
    if sample_id is not None:
        query += " AND sample_id=?"
        params.append(sample_id)
    if data_source is not None:
        query += " AND data_source=?"
        params.append(data_source)
    if replicate_group_id is not None:
        query += " AND replicate_group_id=?"
        params.append(replicate_group_id)
    query += " ORDER BY id DESC"
    if limit:
        query += " LIMIT ?"
        params.append(limit)
    return _rows_to_dicts(conn.execute(query, params).fetchall())


def get_experiment(conn, experiment_id):
    return _row_to_dict(conn.execute("SELECT * FROM experiments WHERE id=?", (experiment_id,)).fetchone())


def list_experiments_with_sample_fields(conn, data_source=None):
    """Join experiments to their parent sample's membrane properties —
    used by ml.dataset to assemble a flat training row per experiment
    (experiments only store water/operating conditions and results;
    membrane properties live on the referenced sample)."""
    sample_cols = ",".join(f"s.{f} as {f}" for f in SAMPLE_FIELDS)
    query = f"""
        SELECT e.*, {sample_cols}
        FROM experiments e
        JOIN samples s ON e.sample_id = s.id
        WHERE 1=1
    """
    params = []
    if data_source is not None:
        query += " AND e.data_source=?"
        params.append(data_source)
    query += " ORDER BY e.id DESC"
    return _rows_to_dicts(conn.execute(query, params).fetchall())


# --------------------------------------------------------------------------
# CS-CA-biochar research records (kept separate from legacy physics runs)
# --------------------------------------------------------------------------

def save_research_formulation(conn, formulation):
    cur = conn.execute(
        """INSERT INTO research_formulations
        (created_at, membrane_id, chitosan_wt_percent, cellulose_acetate_wt_percent,
         biochar_wt_percent, fabrication_info, biochar_properties_json, notes)
        VALUES (?,?,?,?,?,?,?,?)""",
        (_now(), formulation.membrane_id, formulation.chitosan_wt_percent,
         formulation.cellulose_acetate_wt_percent, formulation.biochar_wt_percent,
         formulation.fabrication_info, json.dumps(formulation.biochar_properties),
         formulation.notes),
    )
    conn.commit()
    return cur.lastrowid


def list_research_formulations(conn):
    return _rows_to_dicts(conn.execute("SELECT * FROM research_formulations ORDER BY id DESC").fetchall())


def save_research_experiment(conn, record):
    fields = [
        "experiment_id", "membrane_id", "chitosan_wt_percent", "cellulose_acetate_wt_percent",
        "biochar_wt_percent", "pH", "initial_pb_mg_l", "final_pb_mg_l", "contact_time_min",
        "solution_volume_l", "membrane_mass_g", "removal_percent", "qe_mg_g", "replicate_number", "notes",
    ]
    values = [record.get(field) for field in fields]
    cur = conn.execute(
        f"INSERT INTO research_experiments (created_at,{','.join(fields)}) VALUES (?,{','.join(['?'] * len(fields))})",
        [_now()] + values,
    )
    conn.commit()
    return cur.lastrowid


def list_research_experiments(conn):
    return _rows_to_dicts(conn.execute("SELECT * FROM research_experiments ORDER BY id DESC").fetchall())


def save_characterization(conn, membrane_id, record):
    cur = conn.execute(
        "INSERT INTO membrane_characterization (created_at, membrane_id, characterization_json) VALUES (?,?,?)",
        (_now(), membrane_id, json.dumps(record)),
    )
    conn.commit()
    return cur.lastrowid


def save_reuse_cycle(conn, record):
    cur = conn.execute(
        """INSERT INTO membrane_reuse_cycles
        (created_at, membrane_id, cycle, regeneration_method, removal_percent, qe_mg_g, notes)
        VALUES (?,?,?,?,?,?,?)""",
        (_now(), record["membrane_id"], record["cycle"], record.get("regeneration_method"),
         record.get("removal_percent"), record.get("qe_mg_g"), record.get("notes")),
    )
    conn.commit()
    return cur.lastrowid


def delete_all_experiments(conn):
    """Used only by tests to reset state between runs."""
    conn.execute("DELETE FROM experiments")
    conn.commit()


# --------------------------------------------------------------------------
# Protocol runs
# --------------------------------------------------------------------------

def create_protocol_run(conn, sample_id, name, steps):
    """`steps` is a list of step names; stored with empty recorded values."""
    steps_json = json.dumps({s: {} for s in steps})
    cur = conn.execute(
        "INSERT INTO protocol_runs (created_at, sample_id, name, status, current_step, steps_json) "
        "VALUES (?,?,?, 'in_progress', 0, ?)",
        (_now(), sample_id, name, steps_json))
    conn.commit()
    return cur.lastrowid


def get_protocol_run(conn, protocol_run_id):
    row = _row_to_dict(conn.execute("SELECT * FROM protocol_runs WHERE id=?", (protocol_run_id,)).fetchone())
    if row:
        row["steps"] = json.loads(row["steps_json"])
    return row


def list_protocol_runs(conn, sample_id=None):
    if sample_id is not None:
        rows = conn.execute("SELECT * FROM protocol_runs WHERE sample_id=? ORDER BY id DESC", (sample_id,)).fetchall()
    else:
        rows = conn.execute("SELECT * FROM protocol_runs ORDER BY id DESC").fetchall()
    out = _rows_to_dicts(rows)
    for r in out:
        r["steps"] = json.loads(r["steps_json"])
    return out


def update_protocol_step(conn, protocol_run_id, step_name, values):
    run = get_protocol_run(conn, protocol_run_id)
    if run is None:
        raise ValueError(f"No protocol run with id {protocol_run_id}")
    steps = run["steps"]
    steps[step_name] = values
    conn.execute("UPDATE protocol_runs SET steps_json=? WHERE id=?", (json.dumps(steps), protocol_run_id))
    conn.commit()


def advance_protocol_run(conn, protocol_run_id, complete=False):
    run = get_protocol_run(conn, protocol_run_id)
    if run is None:
        raise ValueError(f"No protocol run with id {protocol_run_id}")
    new_step = run["current_step"] + 1
    status = "complete" if complete else "in_progress"
    conn.execute("UPDATE protocol_runs SET current_step=?, status=? WHERE id=?",
                (new_step, status, protocol_run_id))
    conn.commit()


# --------------------------------------------------------------------------
# Calibration runs
# --------------------------------------------------------------------------

def save_calibration_run(conn, sample_id, calibrated_sample_id, method, n_measurements,
                         fitted_params, metrics_before, metrics_after, notes=None):
    cur = conn.execute(
        "INSERT INTO calibration_runs (created_at, sample_id, calibrated_sample_id, method, "
        "n_measurements, fitted_params_json, metrics_before_json, metrics_after_json, notes) "
        "VALUES (?,?,?,?,?,?,?,?,?)",
        (_now(), sample_id, calibrated_sample_id, method, n_measurements,
         json.dumps(fitted_params), json.dumps(metrics_before), json.dumps(metrics_after), notes))
    conn.commit()
    return cur.lastrowid


def list_calibration_runs(conn, sample_id=None):
    if sample_id is not None:
        rows = conn.execute("SELECT * FROM calibration_runs WHERE sample_id=? ORDER BY id DESC", (sample_id,)).fetchall()
    else:
        rows = conn.execute("SELECT * FROM calibration_runs ORDER BY id DESC").fetchall()
    out = _rows_to_dicts(rows)
    for r in out:
        r["fitted_params"] = json.loads(r["fitted_params_json"])
        r["metrics_before"] = json.loads(r["metrics_before_json"])
        r["metrics_after"] = json.loads(r["metrics_after_json"])
    return out


def sample_has_calibration(conn, sample_id):
    """True if this exact sample is itself the *output* of a calibration
    run (used to decide whether a feasibility assessment may report
    reliability above LOW — see models/feasibility.py)."""
    row = conn.execute("SELECT 1 FROM calibration_runs WHERE calibrated_sample_id=? LIMIT 1",
                       (sample_id,)).fetchone()
    return row is not None


# --------------------------------------------------------------------------
# Trained models
# --------------------------------------------------------------------------

def register_model(conn, target_name, model_type, feature_list, metrics, artifact_path,
                   training_rows, dataset_source):
    existing = conn.execute(
        "SELECT MAX(version) as v FROM trained_models WHERE target_name=?", (target_name,)).fetchone()
    version = (existing["v"] or 0) + 1
    conn.execute("UPDATE trained_models SET is_active=0 WHERE target_name=?", (target_name,))
    cur = conn.execute(
        "INSERT INTO trained_models (created_at, target_name, model_type, feature_list_json, "
        "metrics_json, artifact_path, training_rows, dataset_source, version, is_active) "
        "VALUES (?,?,?,?,?,?,?,?,?,1)",
        (_now(), target_name, model_type, json.dumps(feature_list), json.dumps(metrics),
         str(artifact_path), training_rows, dataset_source, version))
    conn.commit()
    return cur.lastrowid


def list_models(conn, active_only=True):
    query = "SELECT * FROM trained_models"
    if active_only:
        query += " WHERE is_active=1"
    query += " ORDER BY target_name, version DESC"
    out = _rows_to_dicts(conn.execute(query).fetchall())
    for r in out:
        r["feature_list"] = json.loads(r["feature_list_json"])
        r["metrics"] = json.loads(r["metrics_json"])
    return out


def get_active_model_row(conn, target_name):
    row = conn.execute(
        "SELECT * FROM trained_models WHERE target_name=? AND is_active=1 ORDER BY version DESC LIMIT 1",
        (target_name,)).fetchone()
    d = _row_to_dict(row)
    if d:
        d["feature_list"] = json.loads(d["feature_list_json"])
        d["metrics"] = json.loads(d["metrics_json"])
    return d
