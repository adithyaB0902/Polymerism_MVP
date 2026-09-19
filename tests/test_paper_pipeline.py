"""Ground-truth tests for the paper pipeline.

All numeric rows in this file are synthetic test fixtures, never experimental
data and never used by the application.
"""

import io
import sqlite3

import numpy as np
import pandas as pd
import pytest

from db.repository import get_connection, save_research_experiment
from research.cv_nested import nested_compare
from research.desirability import boundary_and_sensitivity, desirability_maximum
from research.domain import leverage_report
from research.experiments import read_bbd_measurement_template
from research.paper_export import export_paper_results
from research.rsm.box_behnken import generate_box_behnken_design
from research.stats_tests import corrected_resampled_t_test


def _template():
    design = generate_box_behnken_design()
    design["table_iv_run"] = design["run_id"]
    for index, factor in enumerate(["chitosan_wt_percent", "biochar_wt_percent", "pH", "initial_pb_mg_l"], 1):
        design[f"x{index}"] = design[f"coded_{factor}"]
    design["final_pb_mg_l_rep1"] = 5.0
    design["final_pb_mg_l_rep2"] = 6.0
    design["solution_volume_l"] = 0.1
    design["membrane_mass_g"] = 0.1
    return design


def test_measurement_template_averages_replicates_and_calculates_metrics():
    result, report = read_bbd_measurement_template(io.StringIO(_template().to_csv(index=False)))
    assert report["valid_rows"] == 29
    assert result["final_pb_mg_l"].iloc[0] == pytest.approx(5.5)
    assert result["removal_percent"].iloc[0] == pytest.approx(81.6666667)


def test_provenance_migration_and_validation(tmp_path):
    connection = get_connection(tmp_path / "research.db")
    save_research_experiment(connection, {
        "experiment_id": "E1", "chitosan_wt_percent": 40, "biochar_wt_percent": 3,
        "pH": 4.5, "initial_pb_mg_l": 30, "final_pb_mg_l": 5, "solution_volume_l": .1,
        "membrane_mass_g": .1, "removal_percent": 83.333, "qe_mg_g": 25,
        "data_source": "simulated",
    })
    row = dict(connection.execute("SELECT * FROM research_experiments").fetchone())
    assert row["data_source"] == "simulated"
    with pytest.raises(ValueError):
        save_research_experiment(connection, {"experiment_id": "bad", "data_source": "unknown"})


def test_corrected_resampled_t_test_detects_paired_difference():
    result = corrected_resampled_t_test([1, 2, 3, 4], [0, 1, 2, 3], n_train=12, n_test=4)
    assert result["mean_difference"] == pytest.approx(1)
    assert result["degrees_of_freedom"] == 3


def test_desirability_and_sensitivity():
    assert np.allclose(desirability_maximum([0, 50, 100], 0, 100), [0, .5, 1])
    result = boundary_and_sensitivity(lambda x: x[:, 0] * 2, [5], [(0, 10)])
    assert result["on_boundary"] is False
    assert result["sensitivity"][0]["plus"] > result["sensitivity"][0]["minus"]


def test_domain_flags_outside_and_high_leverage():
    reference = pd.DataFrame({"a": [0, 1, 2], "b": [0, 1, 2]})
    candidates = pd.DataFrame({"a": [1, 4], "b": [1, 4]})
    result = leverage_report(reference, candidates, ["a", "b"])
    assert result["outside_factor_range"].tolist() == [False, True]


def test_nested_cv_returns_all_metrics_and_predictions():
    # Synthetic fixture generated from a deterministic polynomial.
    rng = np.random.default_rng(5)
    frame = pd.DataFrame(rng.normal(size=(15, 5)), columns=[
        "chitosan_wt_percent", "biochar_wt_percent", "pH", "initial_pb_mg_l", "removal_percent"])
    table, predictions = nested_compare(frame, groups=np.arange(15), n_splits=3, n_repeats=1, max_grid_points=1)
    assert {"R2", "RMSE", "MAE", "AARD", "RMSE_SD"}.issubset(table.columns)
    assert not predictions.empty


def test_paper_export_writes_required_core_outputs(tmp_path):
    output = export_paper_results(io.StringIO(_template().to_csv(index=False)), tmp_path / "paper")
    assert (output / "table_iv_design_and_predictions.csv").exists()
    assert (output / "table_v_anova.csv").exists()
    assert (output / "table_vii_optimum_and_confirmation.csv").exists()
    assert "data_source counts" in (output / "provenance_report.txt").read_text(encoding="utf-8")
