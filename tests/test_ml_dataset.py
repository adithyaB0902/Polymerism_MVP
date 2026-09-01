import tempfile

import pandas as pd

from db import repository as repo
from ml.dataset import assemble_training_dataset
from ml.train import TARGETS, train_models


def _fresh_conn():
    tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
    tmp.close()
    return repo.get_connection(tmp.name)


def test_assemble_synthetic_only():
    df, summary = assemble_training_dataset(conn=None, n_synthetic=50, seed=1)
    assert summary == {"synthetic_rows": 50, "experimental_rows": 0, "total_rows": 50}
    assert len(df) == 50


def test_assemble_includes_measured_experiments():
    conn = _fresh_conn()
    sample_id = repo.create_sample(conn, {
        "polymer_name": "LabSample1", "membrane_type": "UF", "thickness_um": 100,
        "permeability_LMH_bar": 30, "porosity": 0.4, "baseline_rejection_percent": 96,
        "fouling_coefficient": 0.02,
    })
    for i in range(3):
        repo.record_experiment(
            conn, sample_id,
            water={"feed_concentration_mg_L": 100, "pH": 7},
            op={"TMP_bar": 2.0, "operating_time_hr": 4.0},
            result={"flux_LMH": 55.0 + i, "rejection_percent": 96.5},
            data_source="measured",
        )
    df, summary = assemble_training_dataset(conn=conn, n_synthetic=40, seed=2, include_experimental=True)
    assert summary["synthetic_rows"] == 40
    assert summary["experimental_rows"] == 3
    assert summary["total_rows"] == 43
    measured_rows = df[df["dataset_type"] == "experimental_measured"]
    assert len(measured_rows) == 3
    assert set(measured_rows["flux_LMH"]) == {55.0, 56.0, 57.0}
    # feasibility_score wasn't supplied for these measured rows -> NaN, not crashed/dropped
    assert measured_rows["feasibility_score"].isna().all()


def test_assemble_ignores_measured_rows_with_no_targets_at_all():
    conn = _fresh_conn()
    sample_id = repo.create_sample(conn, {"polymer_name": "P", "membrane_type": "UF",
                                          "thickness_um": 100, "permeability_LMH_bar": 25})
    repo.record_experiment(conn, sample_id, {}, {}, {}, data_source="measured")  # no result at all
    df, summary = assemble_training_dataset(conn=conn, n_synthetic=20, seed=3, include_experimental=True)
    assert summary["experimental_rows"] == 0
    assert summary["total_rows"] == 20


def test_combined_dataset_trains_successfully():
    conn = _fresh_conn()
    sample_id = repo.create_sample(conn, {"polymer_name": "LabSample", "membrane_type": "UF",
                                          "thickness_um": 100, "permeability_LMH_bar": 25,
                                          "baseline_rejection_percent": 95, "fouling_coefficient": 0.03})
    for i in range(5):
        repo.record_experiment(conn, sample_id, {}, {}, {"flux_LMH": 40.0 + i, "rejection_percent": 95.0},
                               data_source="measured")
    df, summary = assemble_training_dataset(conn=conn, n_synthetic=150, seed=5, include_experimental=True)
    models, metrics, feature_list = train_models(df)
    assert len(models) == len(TARGETS)
    flux_metric = next(m for m in metrics if m["target"] == "flux_LMH")
    # 150 synthetic + 5 measured all have flux_LMH -> 155 labeled rows
    assert flux_metric["n_train"] + flux_metric["n_test"] == 155


def test_include_experimental_false_ignores_db_even_if_present():
    conn = _fresh_conn()
    sample_id = repo.create_sample(conn, {"polymer_name": "P", "membrane_type": "UF",
                                          "thickness_um": 100, "permeability_LMH_bar": 25})
    repo.record_experiment(conn, sample_id, {}, {}, {"flux_LMH": 99.0}, data_source="measured")
    df, summary = assemble_training_dataset(conn=conn, n_synthetic=10, seed=6, include_experimental=False)
    assert summary["experimental_rows"] == 0
    assert summary["total_rows"] == 10
