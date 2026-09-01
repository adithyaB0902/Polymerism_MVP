import json
import tempfile
from pathlib import Path

from db import repository as repo


def _fresh_conn():
    tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
    tmp.close()
    return repo.get_connection(tmp.name)


def test_batch_and_sample_crud():
    conn = _fresh_conn()
    batch_id = repo.create_batch(conn, "Batch A", notes="test batch")
    assert len(repo.list_batches(conn)) == 1

    sample_id = repo.create_sample(conn, {
        "polymer_name": "PVDF", "membrane_type": "UF", "thickness_um": 100.0,
        "permeability_LMH_bar": 25.0, "porosity": 0.35, "baseline_rejection_percent": 95.0,
        "fouling_coefficient": 0.03,
    }, batch_id=batch_id, origin="manual")
    sample = repo.get_sample(conn, sample_id)
    assert sample["polymer_name"] == "PVDF"
    assert sample["batch_id"] == batch_id
    assert len(repo.list_samples(conn, batch_id=batch_id)) == 1


def test_experiment_round_trip():
    conn = _fresh_conn()
    sample_id = repo.create_sample(conn, {"polymer_name": "P", "membrane_type": "UF",
                                          "thickness_um": 100, "permeability_LMH_bar": 25})
    water = {"contaminant": "NaCl", "feed_concentration_mg_L": 100.0, "pH": 7.0}
    op = {"TMP_bar": 2.0, "operating_time_hr": 4.0}
    result = {"flux_LMH": 50.0, "rejection_percent": 95.0, "feasibility_score": 80.0}
    exp_id = repo.record_experiment(conn, sample_id, water, op, result, data_source="simulated")
    exp = repo.get_experiment(conn, exp_id)
    assert exp["water_contaminant"] == "NaCl"
    assert exp["op_TMP_bar"] == 2.0
    assert exp["flux_LMH"] == 50.0
    assert exp["data_source"] == "simulated"

    results = repo.list_experiments(conn, sample_id=sample_id)
    assert len(results) == 1


def test_replicate_grouping():
    conn = _fresh_conn()
    sample_id = repo.create_sample(conn, {"polymer_name": "P", "membrane_type": "UF",
                                          "thickness_um": 100, "permeability_LMH_bar": 25})
    group = "rep-group-1"
    for i in range(3):
        repo.record_experiment(conn, sample_id, {}, {}, {"flux_LMH": 50.0 + i},
                               replicate_group_id=group, data_source="simulated")
    reps = repo.list_experiments(conn, replicate_group_id=group)
    assert len(reps) == 3


def test_protocol_run_lifecycle():
    conn = _fresh_conn()
    sample_id = repo.create_sample(conn, {"polymer_name": "P", "membrane_type": "UF",
                                          "thickness_um": 100, "permeability_LMH_bar": 25})
    steps = ["Record formulation", "Measure flux", "Measure rejection"]
    run_id = repo.create_protocol_run(conn, sample_id, "Run 1", steps)
    run = repo.get_protocol_run(conn, run_id)
    assert run["status"] == "in_progress"
    assert run["current_step"] == 0
    assert set(run["steps"].keys()) == set(steps)

    repo.update_protocol_step(conn, run_id, "Record formulation", {"polymer": "PVDF"})
    repo.advance_protocol_run(conn, run_id)
    run = repo.get_protocol_run(conn, run_id)
    assert run["current_step"] == 1
    assert run["steps"]["Record formulation"]["polymer"] == "PVDF"

    repo.advance_protocol_run(conn, run_id, complete=True)
    run = repo.get_protocol_run(conn, run_id)
    assert run["status"] == "complete"


def test_calibration_run_and_flag():
    conn = _fresh_conn()
    sample_id = repo.create_sample(conn, {"polymer_name": "P", "membrane_type": "UF",
                                          "thickness_um": 100, "permeability_LMH_bar": 25})
    calibrated_id = repo.create_sample(conn, {"polymer_name": "P (calibrated)", "membrane_type": "UF",
                                              "thickness_um": 100, "permeability_LMH_bar": 27.3},
                                       origin="calibrated", calibrated_from_sample_id=sample_id)
    repo.save_calibration_run(conn, sample_id, calibrated_id, "least_squares", 5,
                              {"permeability_LMH_bar": 27.3}, {"RMSE": 5.0}, {"RMSE": 1.2})
    runs = repo.list_calibration_runs(conn, sample_id=sample_id)
    assert len(runs) == 1
    assert runs[0]["fitted_params"]["permeability_LMH_bar"] == 27.3
    assert repo.sample_has_calibration(conn, calibrated_id) is True
    assert repo.sample_has_calibration(conn, sample_id) is False


def test_model_registry_versioning():
    conn = _fresh_conn()
    id1 = repo.register_model(conn, "flux_LMH", "RandomForest", ["a", "b"], {"RMSE": 1.0},
                              "models_artifacts/flux_v1.joblib", 500, "synthetic")
    id2 = repo.register_model(conn, "flux_LMH", "ExtraTrees", ["a", "b", "c"], {"RMSE": 0.8},
                              "models_artifacts/flux_v2.joblib", 600, "combined")
    active = repo.list_models(conn, active_only=True)
    assert len(active) == 1
    assert active[0]["id"] == id2
    assert active[0]["version"] == 2

    all_versions = repo.list_models(conn, active_only=False)
    assert len(all_versions) == 2

    row = repo.get_active_model_row(conn, "flux_LMH")
    assert row["artifact_path"] == "models_artifacts/flux_v2.joblib"
