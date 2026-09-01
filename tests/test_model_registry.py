import tempfile
from pathlib import Path

import numpy as np

from db import repository as repo
from ml.dataset import assemble_training_dataset
from ml.train import train_models
from ml.model_registry import save_and_register_models, load_active_model, load_all_active_models


def _fresh_conn():
    tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
    tmp.close()
    return repo.get_connection(tmp.name)


def test_save_load_round_trip(tmp_path):
    conn = _fresh_conn()
    df, _ = assemble_training_dataset(conn=None, n_synthetic=100, seed=1)
    models, metrics, feature_list = train_models(df)

    ids = save_and_register_models(conn, models, metrics, feature_list, training_rows=len(df),
                                   dataset_source="synthetic", artifacts_dir=tmp_path)
    assert len(ids) == len(models)

    model, meta = load_active_model(conn, "flux_LMH", artifacts_dir=tmp_path)
    assert model is not None
    assert meta["feature_list"] == feature_list
    assert meta["version"] == 1

    # loaded model produces the same predictions as the original
    from ml.features import engineer_features
    engineered = engineer_features(df)
    X = engineered[feature_list].fillna(engineered[feature_list].median()).head(5)
    np.testing.assert_allclose(model.predict(X), models["flux_LMH"].predict(X))


def test_missing_model_returns_none():
    conn = _fresh_conn()
    model, meta = load_active_model(conn, "flux_LMH")
    assert model is None and meta is None


def test_retraining_creates_new_version_and_deactivates_old(tmp_path):
    conn = _fresh_conn()
    df, _ = assemble_training_dataset(conn=None, n_synthetic=100, seed=1)
    models1, metrics1, features1 = train_models(df, seed=1)
    save_and_register_models(conn, models1, metrics1, features1, len(df), "synthetic", artifacts_dir=tmp_path)

    df2, _ = assemble_training_dataset(conn=None, n_synthetic=150, seed=2)
    models2, metrics2, features2 = train_models(df2, seed=2)
    save_and_register_models(conn, models2, metrics2, features2, len(df2), "synthetic", artifacts_dir=tmp_path)

    _, meta = load_active_model(conn, "flux_LMH", artifacts_dir=tmp_path)
    assert meta["version"] == 2
    assert meta["training_rows"] == 150

    all_versions = repo.list_models(conn, active_only=False)
    flux_versions = [r for r in all_versions if r["target_name"] == "flux_LMH"]
    assert len(flux_versions) == 2
    assert sum(1 for r in flux_versions if r["is_active"]) == 1


def test_load_all_active_models(tmp_path):
    conn = _fresh_conn()
    df, _ = assemble_training_dataset(conn=None, n_synthetic=100, seed=1)
    models, metrics, feature_list = train_models(df)
    save_and_register_models(conn, models, metrics, feature_list, len(df), "synthetic", artifacts_dir=tmp_path)

    all_models = load_all_active_models(conn)
    assert set(all_models.keys()) == set(models.keys())
    for target, (model, meta) in all_models.items():
        assert meta["target_name"] == target
