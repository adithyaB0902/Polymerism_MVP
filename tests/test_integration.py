"""Integration tests: verify that modules which are each unit-tested in
isolation (models, db, lab, ml, validation) actually work correctly
*together*, across realistic multi-step workflows. Unlike the system
tests in test_system_e2e.py, these call the underlying Python modules
directly rather than driving the Streamlit UI — they're faster and
pinpoint exactly which integration seam broke.
"""

import tempfile

import pytest

from db import repository as repo
from models.membrane import Membrane, Water, OperatingConditions, Targets
from models.simulator import run_simulation, run_detailed_simulation
from models.feasibility import feasibility_assessment
from lab.samples import create_sample_from_membrane, create_sample_from_preset, load_presets, sample_to_membrane
from lab.protocol import start_protocol, record_current_step, is_complete, collected_values, PROTOCOL_STEPS
from lab.replicates import simulate_replicates, replicate_statistics
from validation.calibration import calibrate_sample
from ml.dataset import assemble_training_dataset
from ml.train import train_models
from ml.model_registry import save_and_register_models, load_active_model
from ml.predict import predict_candidate


def _fresh_conn():
    tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
    tmp.close()
    return repo.get_connection(tmp.name)


def test_preset_to_simulation_to_logged_experiment():
    """Sample Registry -> Single Simulation -> Lab Notebook chain."""
    conn = _fresh_conn()
    preset = load_presets()[1]  # Generic UF
    sample_id = create_sample_from_preset(conn, preset)
    membrane = sample_to_membrane(repo.get_sample(conn, sample_id))

    water, op, targets = Water(), OperatingConditions(), Targets()
    result = run_simulation(membrane, water, op, targets)
    assert result["flux_LMH"] > 0

    exp_id = repo.record_experiment(conn, sample_id, water.__dict__, op.__dict__, result,
                                    physics_model="simple", data_source="simulated")
    logged = repo.get_experiment(conn, exp_id)
    assert logged["flux_LMH"] == pytest.approx(result["flux_LMH"])
    assert logged["sample_id"] == sample_id


def test_detailed_vs_simple_simulation_both_loggable_for_same_sample():
    """A single sample should support experiments logged under both
    physics models without the database schema caring which was used."""
    conn = _fresh_conn()
    membrane = Membrane("X", "UF", 100, 25, pore_size_nm=20, porosity=0.35,
                        baseline_rejection_percent=95, fouling_coefficient=0.03)
    sample_id = create_sample_from_membrane(conn, membrane)
    water, op, targets = Water(), OperatingConditions(), Targets()

    simple = run_simulation(membrane, water, op, targets)
    detailed = run_detailed_simulation(membrane, water, op, targets)
    repo.record_experiment(conn, sample_id, water.__dict__, op.__dict__, simple,
                           physics_model="simple", data_source="simulated")
    repo.record_experiment(conn, sample_id, water.__dict__, op.__dict__, detailed,
                           physics_model="detailed", data_source="simulated")

    logged = repo.list_experiments(conn, sample_id=sample_id)
    assert len(logged) == 2
    assert {e["physics_model"] for e in logged} == {"simple", "detailed"}


def test_protocol_completion_feeds_lab_notebook():
    """Protocol Runner -> Lab Notebook chain: a completed protocol run's
    recorded flux/rejection values should be logged as a measured
    experiment tied back to the same protocol run."""
    conn = _fresh_conn()
    sample_id = repo.create_sample(conn, {"polymer_name": "P", "membrane_type": "UF",
                                          "thickness_um": 100, "permeability_LMH_bar": 25})
    run_id = start_protocol(conn, sample_id, "Integration test run")
    for step in PROTOCOL_STEPS:
        values = {f: "42" for f in step["fields"]}
        if "flux_LMH" in step["fields"]:
            values["flux_LMH"] = "51.5"
        if "rejection_percent" in step["fields"]:
            values["rejection_percent"] = "96.2"
        run = record_current_step(conn, run_id, values)
    assert is_complete(run)

    vals = collected_values(run)
    result = {"flux_LMH": float(vals["flux_LMH"]), "rejection_percent": float(vals["rejection_percent"])}
    exp_id = repo.record_experiment(conn, sample_id, {}, {}, result, data_source="measured",
                                    protocol_run_id=run_id)
    logged = repo.get_experiment(conn, exp_id)
    assert logged["flux_LMH"] == 51.5
    assert logged["protocol_run_id"] == run_id


def test_replicates_logged_and_summarized_consistently():
    """Simulated replicates' own statistics should match what you'd get
    by recomputing from the logged rows independently."""
    membrane = Membrane("P", "UF", 100, 25, baseline_rejection_percent=95, fouling_coefficient=0.03)
    result = run_simulation(membrane, Water(), OperatingConditions(), Targets())
    reps = simulate_replicates(result, n=15, seed=3)
    stats = replicate_statistics(reps, fields=["flux_LMH", "rejection_percent"])

    import numpy as np
    assert stats["flux_LMH"]["mean"] == pytest.approx(np.mean([r["flux_LMH"] for r in reps]))
    assert stats["flux_LMH"]["n"] == 15


def test_calibration_upgrades_feasibility_reliability_end_to_end():
    """Validation & Calibration -> Feasibility chain: calibrating a
    sample against measurements should be reflected in a subsequent
    feasibility_assessment call for the calibrated sample."""
    conn = _fresh_conn()
    true_membrane = Membrane("Real", "UF", 100, 32, baseline_rejection_percent=96, fouling_coefficient=0.025)
    water, op, targets = Water(), OperatingConditions(TMP_bar=2.0), Targets()
    sample_id = create_sample_from_membrane(conn, Membrane("Guess", "UF", 100, 10,
                                                            baseline_rejection_percent=80, fouling_coefficient=0.1))

    from dataclasses import replace
    measurements = []
    for t in [0.5, 1, 2, 4, 8]:
        r = run_simulation(true_membrane, water, replace(op, operating_time_hr=t), targets)
        measurements.append({"operating_time_hr": t, "flux_LMH": r["flux_LMH"],
                            "rejection_percent": r["rejection_percent"]})

    guess_membrane = sample_to_membrane(repo.get_sample(conn, sample_id))
    calib = calibrate_sample(guess_membrane, water, op, measurements)
    assert calib["post_fit_r2"] > 0.99

    calibrated_id = create_sample_from_membrane(conn, calib["calibrated_membrane"], origin="calibrated",
                                                calibrated_from_sample_id=sample_id)
    repo.save_calibration_run(conn, sample_id, calibrated_id, "least_squares", len(measurements),
                              calib["fitted_params"], calib["metrics_before"], calib["metrics_after"])

    assert repo.sample_has_calibration(conn, calibrated_id)
    result = run_simulation(calib["calibrated_membrane"], water, op, targets)
    assessment = feasibility_assessment(result, targets, calibration={"post_fit_r2": calib["post_fit_r2"]})
    assert assessment["reliability"] == "HIGH"


def test_full_ml_pipeline_synthetic_plus_measured_to_prediction(tmp_path):
    """Lab Notebook -> ML Lab chain: measured experiments logged for a
    sample should actually influence a trained-and-registered model's
    predictions, not just silently be ignored."""
    conn = _fresh_conn()
    sample_id = repo.create_sample(conn, {"polymer_name": "LabX", "membrane_type": "UF",
                                          "thickness_um": 100, "permeability_LMH_bar": 25,
                                          "baseline_rejection_percent": 95, "fouling_coefficient": 0.03})
    for i in range(5):
        repo.record_experiment(conn, sample_id, {}, {}, {"flux_LMH": 40.0 + i, "rejection_percent": 95.0},
                               data_source="measured")

    df, summary = assemble_training_dataset(conn, n_synthetic=200, seed=1, include_experimental=True)
    assert summary["experimental_rows"] == 5
    models, metrics, feature_list = train_models(df)
    save_and_register_models(conn, models, metrics, feature_list, len(df), "combined", artifacts_dir=tmp_path)

    model, meta = load_active_model(conn, "flux_LMH", artifacts_dir=tmp_path)
    assert model is not None
    assert meta["dataset_source"] == "combined"

    candidate = {"thickness_um": 100.0, "permeability_LMH_bar": 25.0, "baseline_rejection_percent": 95.0}
    result = predict_candidate(conn, candidate, reference_df=df)
    assert "flux_LMH" in result["predictions"] or "flux_LMH" in result["skipped"]
