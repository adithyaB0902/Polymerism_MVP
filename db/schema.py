"""SQLite schema for the POLYMEMSIM lab database.

One file, one database (default: data/polymemsim.db, git-ignored — it's
runtime state, not source). Tables model a small but real lab workflow:
samples grouped into batches, experiment runs against those samples
(both simulated and real-measured), guided multi-step protocol runs,
calibration attempts, and a registry of trained ML models.
"""

SCHEMA = """
CREATE TABLE IF NOT EXISTS batches (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at TEXT NOT NULL,
    name TEXT NOT NULL,
    notes TEXT
);

CREATE TABLE IF NOT EXISTS samples (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at TEXT NOT NULL,
    batch_id INTEGER REFERENCES batches(id),
    polymer_name TEXT NOT NULL,
    membrane_type TEXT NOT NULL,
    thickness_um REAL NOT NULL,
    permeability_LMH_bar REAL NOT NULL,
    pore_size_nm REAL DEFAULT 0,
    MWCO REAL DEFAULT 0,
    porosity REAL DEFAULT 0,
    surface_charge REAL DEFAULT 0,
    hydrophilicity REAL DEFAULT 0,
    membrane_area_cm2 REAL DEFAULT 10000,
    baseline_rejection_percent REAL DEFAULT 0,
    fouling_coefficient REAL DEFAULT 0,
    notes TEXT,
    origin TEXT DEFAULT 'manual',       -- 'manual' | 'preset' | 'calibrated'
    calibrated_from_sample_id INTEGER REFERENCES samples(id)
);

CREATE TABLE IF NOT EXISTS experiments (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at TEXT NOT NULL,
    sample_id INTEGER REFERENCES samples(id),
    replicate_group_id TEXT,            -- shared id across replicate runs of one condition, NULL otherwise
    physics_model TEXT NOT NULL,        -- 'simple' | 'detailed'
    data_source TEXT NOT NULL,          -- 'simulated' | 'measured'
    protocol_run_id INTEGER REFERENCES protocol_runs(id),

    water_contaminant TEXT,
    water_feed_concentration_mg_L REAL,
    water_turbidity_NTU REAL,
    water_TDS_mg_L REAL,
    water_pH REAL,
    water_temperature_C REAL,
    water_viscosity_Pa_s REAL,
    water_density_kg_m3 REAL,

    op_TMP_bar REAL,
    op_feed_flow_L_min REAL,
    op_crossflow_velocity_m_s REAL,
    op_recovery_percent REAL,
    op_operating_time_hr REAL,
    op_membrane_area_m2 REAL,
    op_pump_efficiency REAL,
    op_electricity_price_per_kwh REAL,

    flux_LMH REAL,
    initial_flux_LMH REAL,
    rejection_percent REAL,
    permeate_mg_L REAL,
    flux_decline_percent REAL,
    energy_kWh_m3 REAL,
    cost_per_m3 REAL,
    feasibility_score REAL,
    total_permeate_flow_L_hr REAL,
    concentrate_mg_L REAL,

    notes TEXT
);

CREATE TABLE IF NOT EXISTS protocol_runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at TEXT NOT NULL,
    sample_id INTEGER REFERENCES samples(id),
    name TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'in_progress',  -- 'in_progress' | 'complete'
    current_step INTEGER NOT NULL DEFAULT 0,
    steps_json TEXT NOT NULL             -- JSON: {step_name: {recorded values...}}
);

CREATE TABLE IF NOT EXISTS calibration_runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at TEXT NOT NULL,
    sample_id INTEGER REFERENCES samples(id),
    calibrated_sample_id INTEGER REFERENCES samples(id),
    method TEXT NOT NULL,
    n_measurements INTEGER NOT NULL,
    fitted_params_json TEXT NOT NULL,
    metrics_before_json TEXT NOT NULL,
    metrics_after_json TEXT NOT NULL,
    notes TEXT
);

CREATE TABLE IF NOT EXISTS trained_models (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at TEXT NOT NULL,
    target_name TEXT NOT NULL,
    model_type TEXT NOT NULL,
    feature_list_json TEXT NOT NULL,
    metrics_json TEXT NOT NULL,
    artifact_path TEXT NOT NULL,
    training_rows INTEGER NOT NULL,
    dataset_source TEXT NOT NULL,        -- 'synthetic' | 'experimental' | 'combined'
    version INTEGER NOT NULL DEFAULT 1,
    is_active INTEGER NOT NULL DEFAULT 1  -- 0 when superseded by a newer version of the same target
);
"""
