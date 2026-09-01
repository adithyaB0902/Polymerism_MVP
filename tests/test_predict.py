import tempfile

from db import repository as repo
from ml.dataset import assemble_training_dataset
from ml.train import train_models, TARGETS
from ml.model_registry import save_and_register_models
from ml.predict import predict_candidate


def _fresh_conn():
    tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
    tmp.close()
    return repo.get_connection(tmp.name)


def _train_and_register(conn, tmp_path, n=200, seed=1):
    df, _ = assemble_training_dataset(conn=None, n_synthetic=n, seed=seed)
    models, metrics, feature_list = train_models(df, seed=seed)
    save_and_register_models(conn, models, metrics, feature_list, len(df), "synthetic", artifacts_dir=tmp_path)
    return df


def test_predict_with_no_models_registered_skips_everything():
    conn = _fresh_conn()
    result = predict_candidate(conn, {"thickness_um": 100})
    assert result["predictions"] == {}
    assert set(result["skipped"]) == set(TARGETS)
    assert result["ood"] is None


def test_predict_with_complete_features(tmp_path):
    conn = _fresh_conn()
    df = _train_and_register(conn, tmp_path)
    candidate = {
        "thickness_um": 100.0, "permeability_LMH_bar": 25.0, "pore_size_nm": 20.0, "MWCO": 10000.0,
        "porosity": 0.35, "surface_charge": 0.0, "hydrophilicity": 0.7,
        "baseline_rejection_percent": 95.0, "water_feed_concentration_mg_L": 100.0,
        "water_turbidity_NTU": 5.0, "water_TDS_mg_L": 500.0, "water_pH": 7.0,
        "water_temperature_C": 25.0, "op_TMP_bar": 2.0, "op_feed_flow_L_min": 10.0,
        "op_crossflow_velocity_m_s": 0.2, "op_recovery_percent": 20.0, "op_operating_time_hr": 4.0,
    }
    result = predict_candidate(conn, candidate, reference_df=df)
    assert set(result["predictions"]) == set(TARGETS)
    assert not result["skipped"]
    assert result["ood"] is not None
    assert result["ood"]["ood"] is False  # candidate matches typical sidebar defaults, well within range


def test_predict_with_missing_features_and_no_reference_skips_those_targets(tmp_path):
    conn = _fresh_conn()
    _train_and_register(conn, tmp_path)
    result = predict_candidate(conn, {"thickness_um": 100.0})  # missing almost everything
    assert result["predictions"] == {}
    assert len(result["skipped"]) == len(TARGETS)
    assert result["ood"] is None


def test_predict_imputes_missing_features_from_reference(tmp_path):
    conn = _fresh_conn()
    df = _train_and_register(conn, tmp_path)
    # only supply a couple of features; rest should be imputed from df's medians
    candidate = {"thickness_um": 100.0, "permeability_LMH_bar": 25.0}
    result = predict_candidate(conn, candidate, reference_df=df)
    assert len(result["predictions"]) > 0
    assert set(result["predictions"]) | set(result["skipped"]) == set(TARGETS)


def test_predict_flags_out_of_distribution_candidate(tmp_path):
    conn = _fresh_conn()
    df = _train_and_register(conn, tmp_path)
    extreme_candidate = {
        "thickness_um": 100000.0,  # far outside the 10-500 training range
        "permeability_LMH_bar": 25.0, "pore_size_nm": 20.0, "MWCO": 10000.0,
        "porosity": 0.35, "surface_charge": 0.0, "hydrophilicity": 0.7,
        "baseline_rejection_percent": 95.0, "water_feed_concentration_mg_L": 100.0,
        "water_turbidity_NTU": 5.0, "water_TDS_mg_L": 500.0, "water_pH": 7.0,
        "water_temperature_C": 25.0, "op_TMP_bar": 2.0, "op_feed_flow_L_min": 10.0,
        "op_crossflow_velocity_m_s": 0.2, "op_recovery_percent": 20.0, "op_operating_time_hr": 4.0,
    }
    result = predict_candidate(conn, extreme_candidate, reference_df=df)
    assert result["ood"]["ood"] is True
    assert "thickness_um" in [f for f, flag in result["ood"]["features"].items() if flag]
