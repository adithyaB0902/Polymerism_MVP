"""Tests for the RSM package (Box-Behnken design, quadratic fit, ANOVA, surfaces, UI).

Any response values below are TEST FIXTURES generated from a known polynomial so
that the fitted statistics can be checked against ground truth. They are not
experimental data and are never used by the application itself.
"""

from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from research.rsm import (
    FACTORS, anova_table, fit_quadratic, generate_box_behnken_design, lack_of_fit,
    model_warnings, predict_quadratic, regression_diagnostics, response_surface_grid,
    stationary_point, to_actual, to_coded,
)
from research.rsm.quadratic_model import quadratic_terms

FACTOR_NAMES = list(FACTORS)
RESPONSE = "removal_percent"

# Known coded-unit coefficients used to build noise-free fixtures.
TRUTH = {
    "intercept": 70.0,
    "chitosan_wt_percent": 6.0, "biochar_wt_percent": 3.0, "pH": 8.0, "initial_pb_mg_l": -5.0,
    "chitosan_wt_percent^2": -4.0, "biochar_wt_percent^2": -2.5, "pH^2": -6.0, "initial_pb_mg_l^2": -1.0,
    "chitosan_wt_percent:pH": 2.0, "biochar_wt_percent:pH": 1.5, "pH:initial_pb_mg_l": -1.0,
}


def _fixture(noise_sd=0.0, seed=0, extra=None):
    design = generate_box_behnken_design()
    coded = to_coded(design, FACTOR_NAMES)
    terms = quadratic_terms(coded, FACTOR_NAMES)
    y = sum(terms[t] * c for t, c in TRUTH.items())
    if extra is not None:
        y = y + extra(coded)
    y = y + np.random.default_rng(seed).normal(0, noise_sd, len(design))
    design[RESPONSE] = y.to_numpy()
    return design


# ----------------------------------------------------------------------- design
def test_bbd_has_29_runs_with_five_centre_points():
    d = generate_box_behnken_design()
    assert len(d) == 29 and d["run_id"].is_unique
    coded = d[["coded_" + f for f in FACTOR_NAMES]]
    assert (coded == 0).all(axis=1).sum() == 5
    assert set(np.unique(coded.to_numpy())) == {-1, 0, 1}
    edge = coded[(coded != 0).any(axis=1)]
    assert ((edge != 0).sum(axis=1) == 2).all()               # exactly two factors move per edge run
    for i, j in combinations(range(4), 2):                     # each factor pair: 4 (+/-,+/-) runs
        mask = (coded.iloc[:, i] != 0) & (coded.iloc[:, j] != 0)
        assert mask.sum() == 4
    assert d["experimental_removal_percent"].isna().all()      # design carries no results


def test_bbd_actual_levels_match_declared_ranges():
    d = generate_box_behnken_design()
    for name, (low, center, high) in FACTORS.items():
        assert set(d[name].unique()) == {low, center, high}


def test_coded_actual_round_trip():
    d = generate_box_behnken_design()
    coded = to_coded(d, FACTOR_NAMES)
    assert np.allclose(coded.to_numpy(), d[["coded_" + f for f in FACTOR_NAMES]].to_numpy())
    assert np.allclose(to_actual(coded).to_numpy(), d[FACTOR_NAMES].to_numpy())


# -------------------------------------------------------------------------- fit
def test_recovers_known_coefficients_exactly():
    fit = fit_quadratic(_fixture(), RESPONSE, FACTOR_NAMES)
    table = fit["coefficients"].set_index("term")["coefficient"]
    for term in fit["terms"]:
        assert table[term] == pytest.approx(TRUTH.get(term, 0.0), abs=1e-8)
    assert fit["r2"] == pytest.approx(1.0)
    assert fit["n_runs"] == 29 and fit["n_terms"] == 15 and fit["residual_df"] == 14


def test_predictions_use_actual_units_and_match_fit():
    data = _fixture(noise_sd=1.0, seed=3)
    fit = fit_quadratic(data, RESPONSE, FACTOR_NAMES)
    assert np.allclose(predict_quadratic(fit, data), fit["predictions"]["predicted"].to_numpy())


def test_backward_compatible_result_keys():
    fit = fit_quadratic(_fixture(noise_sd=1.0), RESPONSE, FACTOR_NAMES)
    for key in ("r2", "adjusted_r2", "rmse", "coefficients", "predictions", "sse", "anova", "terms", "factors"):
        assert key in fit
    for col in ("term", "coefficient", "std_error", "t_value", "p_value", "significant_alpha_0_05"):
        assert col in fit["coefficients"].columns


def test_press_matches_explicit_leave_one_out_refit():
    data = _fixture(noise_sd=1.5, seed=1)
    fit = fit_quadratic(data, RESPONSE, FACTOR_NAMES)
    x, y = fit["design_matrix"].to_numpy(), data[RESPONSE].to_numpy()
    loo = 0.0
    for i in range(len(y)):
        keep = np.arange(len(y)) != i
        beta = np.linalg.lstsq(x[keep], y[keep], rcond=None)[0]
        loo += (y[i] - x[i] @ beta) ** 2
    assert fit["press"] == pytest.approx(loo, rel=1e-9)
    assert fit["predicted_r2"] < fit["r2"]


# ------------------------------------------------------------------------ ANOVA
def test_anova_sums_and_degrees_of_freedom_are_consistent():
    fit = fit_quadratic(_fixture(noise_sd=1.5, seed=2), RESPONSE, FACTOR_NAMES)
    t = fit["anova_table"].set_index("source")
    assert t.loc["Model", "sum_of_squares"] + t.loc["Residual", "sum_of_squares"] == pytest.approx(
        t.loc["Cor total", "sum_of_squares"])
    assert t.loc["Model", "df"] + t.loc["Residual", "df"] == t.loc["Cor total", "df"] == 28
    assert t.loc["Lack of fit", "df"] + t.loc["Pure error", "df"] == t.loc["Residual", "df"]
    assert t.loc["Pure error", "df"] == 4                       # five replicated centre points
    assert t.loc["Lack of fit", "sum_of_squares"] + t.loc["Pure error", "sum_of_squares"] == pytest.approx(
        t.loc["Residual", "sum_of_squares"])
    assert 0 <= t.loc["Model", "p_value"] <= 1
    # per-term F equals t^2 (partial SS convention)
    coef = fit["coefficients"].set_index("term")
    for term in fit["terms"][1:]:
        assert t.loc[term, "f_value"] == pytest.approx(coef.loc[term, "t_value"] ** 2)


def test_significant_terms_detected_and_null_terms_not():
    fit = fit_quadratic(_fixture(noise_sd=0.5, seed=4), RESPONSE, FACTOR_NAMES)
    p = fit["coefficients"].set_index("term")["p_value"]
    assert p["pH"] < 0.001 and p["pH^2"] < 0.001
    assert p["chitosan_wt_percent:initial_pb_mg_l"] > 0.05      # true coefficient is zero


def test_lack_of_fit_detects_a_non_quadratic_surface():
    # A^2*B^2 cannot be represented by a full quadratic on a Box-Behnken design.
    misfit = _fixture(noise_sd=0.3, seed=5,
                      extra=lambda c: 15.0 * c["chitosan_wt_percent"] ** 2 * c["biochar_wt_percent"] ** 2)
    fit = fit_quadratic(misfit, RESPONSE, FACTOR_NAMES)
    assert fit["lack_of_fit"]["p_value"] < 0.05
    assert any("lack of fit" in w.lower() for w in model_warnings(fit))
    good = fit_quadratic(_fixture(noise_sd=0.3, seed=5), RESPONSE, FACTOR_NAMES)
    assert good["lack_of_fit"]["p_value"] > 0.05


def test_lack_of_fit_is_none_without_replicates():
    data = _fixture(noise_sd=1.0)
    is_centre = (data[FACTOR_NAMES] == pd.Series({f: c for f, (_, c, _) in FACTORS.items()})).all(axis=1)
    data = pd.concat([data[~is_centre], data[is_centre].head(1)])   # 24 edge runs + a single centre run
    fit = fit_quadratic(data, RESPONSE, FACTOR_NAMES)
    assert fit["lack_of_fit"] is None
    assert "Lack of fit" not in set(fit["anova_table"]["source"])
    assert any("cannot be tested" in w for w in model_warnings(fit))


def test_lack_of_fit_helper_directly():
    x = np.array([[0, 0], [0, 0], [1, 0], [1, 0], [0, 1]], float)
    y = np.array([1.0, 3.0, 5.0, 5.0, 2.0])
    out = lack_of_fit(x, y, sse=10.0, residual_df=3)
    assert out["df_pure_error"] == 2 and out["ss_pure_error"] == pytest.approx(2.0)
    assert out["ss_lack_of_fit"] == pytest.approx(8.0) and out["df_lack_of_fit"] == 1


# ---------------------------------------------------------------- stationary pt
def test_stationary_point_recovered_from_known_surface():
    # y = 80 - 5(x1-.3)^2 - 4(x2+.2)^2 - 3(x3-.1)^2 - 2(x4+.4)^2  (a maximum inside the region)
    centres = np.array([0.3, -0.2, 0.1, -0.4])
    curv = np.array([-5.0, -4.0, -3.0, -2.0])
    design = generate_box_behnken_design()
    coded = to_coded(design, FACTOR_NAMES).to_numpy()
    design[RESPONSE] = 80 + (curv * -(coded - centres) ** 2 * -1).sum(axis=1)
    fit = fit_quadratic(design, RESPONSE, FACTOR_NAMES)
    sp = stationary_point(fit)
    assert sp["nature"] == "maximum" and sp["inside_design_region"]
    assert np.allclose(list(sp["coded"].values()), centres, atol=1e-8)
    assert sp["predicted_response"] == pytest.approx(80.0)
    low, center, high = FACTORS["pH"]
    assert sp["actual"]["pH"] == pytest.approx(center + 0.1 * (high - low) / 2)


def test_saddle_is_reported_as_saddle():
    fit = fit_quadratic(_fixture(extra=lambda c: 12 * c["biochar_wt_percent"] ** 2), RESPONSE, FACTOR_NAMES)
    assert stationary_point(fit)["nature"] == "saddle"


# ---------------------------------------------------------- guards / diagnostics
def test_too_few_runs_raises():
    with pytest.raises(ValueError, match="More complete experimental runs"):
        fit_quadratic(_fixture().head(15), RESPONSE, FACTOR_NAMES)


def test_rank_deficient_design_raises():
    data = _fixture(noise_sd=1.0)
    data["pH"] = 4.5                                            # factor never varied
    with pytest.raises(ValueError):
        fit_quadratic(data, RESPONSE, FACTOR_NAMES)


def test_missing_response_rows_are_ignored():
    data = _fixture(noise_sd=1.0)
    data.loc[0, RESPONSE] = np.nan
    assert fit_quadratic(data, RESPONSE, FACTOR_NAMES)["n_runs"] == 28


def test_diagnostics_leverage_sums_to_number_of_terms():
    fit = fit_quadratic(_fixture(noise_sd=1.0, seed=6), RESPONSE, FACTOR_NAMES)
    diag = regression_diagnostics(fit)
    assert diag["leverage"].sum() == pytest.approx(fit["n_terms"])
    assert len(diag) == 29 and (diag["cooks_distance"] >= 0).all()


def test_outlier_is_flagged():
    data = _fixture(noise_sd=0.5, seed=7)
    data.loc[3, RESPONSE] += 25.0
    diag = regression_diagnostics(fit_quadratic(data, RESPONSE, FACTOR_NAMES))
    assert bool(diag.iloc[3]["flagged"])


# --------------------------------------------------------------------- surfaces
def test_surface_grid_matches_model_predictions():
    fit = fit_quadratic(_fixture(noise_sd=1.0, seed=8), RESPONSE, FACTOR_NAMES)
    grid = response_surface_grid(fit, "chitosan_wt_percent", "pH", {"biochar_wt_percent": 4.0}, points=11)
    assert grid["z"].shape == (11, 11)
    assert grid["x"].min() == 20 and grid["x"].max() == 60 and grid["y"].min() == 3 and grid["y"].max() == 6
    probe = pd.DataFrame([{"chitosan_wt_percent": grid["x"][3], "biochar_wt_percent": 4.0,
                           "pH": grid["y"][7], "initial_pb_mg_l": 30.0}])
    assert grid["z"][7, 3] == pytest.approx(predict_quadratic(fit, probe)[0])


def test_surface_grid_validates_inputs():
    fit = fit_quadratic(_fixture(noise_sd=1.0), RESPONSE, FACTOR_NAMES)
    with pytest.raises(ValueError):
        response_surface_grid(fit, "pH", "pH")
    with pytest.raises(ValueError):
        response_surface_grid(fit, "pH", "chitosan_wt_percent", {"biochar_wt_percent": 99})


# ------------------------------------------------------------------- app wiring
def test_rsm_page_renders_in_app(tmp_path, monkeypatch):
    """RSM analysis: info message with no data, full analysis once 29 measured rows exist."""
    import streamlit as st
    from streamlit.testing.v1 import AppTest
    import db.repository as repo

    st.cache_resource.clear()
    monkeypatch.setattr(repo, "DEFAULT_DB_PATH", tmp_path / "rsm_app_test.db")
    app = AppTest.from_file(str(Path(__file__).resolve().parent.parent / "app.py"), default_timeout=180)
    app.run()
    assert not list(app.exception)

    def open_rsm():
        for radio in app.radio:
            if "RSM analysis" in radio.options:
                radio.set_value("RSM analysis")
                app.run()
                assert not list(app.exception), list(app.exception)
                return
        raise AssertionError("RSM analysis option not found")

    open_rsm()
    assert any("coefficients" in i.value for i in app.info)      # not enough data yet

    conn = repo.get_connection()
    fixture = _fixture(noise_sd=1.0, seed=9)
    for i, row in fixture.iterrows():                             # TEST FIXTURE rows, not experimental data
        repo.save_research_experiment(conn, {
            "experiment_id": f"T{i:02d}", "chitosan_wt_percent": row["chitosan_wt_percent"],
            "biochar_wt_percent": row["biochar_wt_percent"], "pH": row["pH"],
            "initial_pb_mg_l": row["initial_pb_mg_l"], "final_pb_mg_l": 1.0, "solution_volume_l": 0.1,
            "membrane_mass_g": 0.1, "removal_percent": row[RESPONSE], "qe_mg_g": 1.0})
    app.run()
    open_rsm()
    assert [m.label for m in app.metric][:3] == ["R²", "Adjusted R²", "Predicted R²"]
    assert not list(app.exception)
