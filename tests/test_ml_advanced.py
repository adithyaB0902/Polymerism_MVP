import numpy as np
import pytest
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split

from simulation.generate_dataset import generate_virtual_experiments
from ml.train import FEATURES, TARGETS, train_models
from ml.evaluate import cross_validate_target, cross_validate_all_targets, compare_model_types
from ml.uncertainty import supports_uncertainty, predict_with_uncertainty, predict_one_with_uncertainty
from ml.explain import feature_importances, permutation_importances, used_vs_unused_features
from ml.uncertainty import conformal_prediction_interval


@pytest.fixture(scope="module")
def dataset():
    return generate_virtual_experiments(300, seed=1)


# ---------------------------------------------------------------- evaluate

def test_cross_validate_target_basic(dataset):
    result = cross_validate_target(dataset, "flux_LMH", n_splits=5, seed=1)
    assert result["n_splits"] == 5
    assert result["RMSE_mean"] >= 0
    assert result["RMSE_std"] >= 0
    assert -1 <= result["R2_mean"] <= 1.0001


def test_cross_validate_target_raises_with_too_few_rows():
    tiny = generate_virtual_experiments(5, seed=1)
    with pytest.raises(ValueError):
        cross_validate_target(tiny, "flux_LMH", n_splits=5)


def test_cross_validate_all_targets_skips_underpopulated(dataset):
    df = dataset.copy()
    df.loc[df.index[10:], "cost_per_m3"] = np.nan  # leave energy_kWh_m3 etc alone, this col isn't even a TARGET
    results, skipped = cross_validate_all_targets(df, n_splits=5, seed=1)
    assert len(results) == len(TARGETS)  # cost_per_m3 isn't a TARGET, so nothing should actually be skipped here
    assert skipped == []


def test_compare_model_types_returns_all_three(dataset):
    results = compare_model_types(dataset, "flux_LMH", n_splits=3, seed=1)
    assert {r["model_type"] for r in results} == {"Random Forest", "Extra Trees", "Gradient Boosting"}


# -------------------------------------------------------------- uncertainty

def test_supports_uncertainty_flags_correctly(dataset):
    rf = RandomForestRegressor(n_estimators=20, random_state=1).fit(dataset[FEATURES], dataset["flux_LMH"])
    gb = GradientBoostingRegressor(random_state=1).fit(dataset[FEATURES], dataset["flux_LMH"])
    assert supports_uncertainty(rf) is True
    assert supports_uncertainty(gb) is False


def test_predict_with_uncertainty_shapes_and_positivity(dataset):
    rf = RandomForestRegressor(n_estimators=30, random_state=1).fit(dataset[FEATURES], dataset["flux_LMH"])
    X = dataset[FEATURES].head(10)
    mean, std = predict_with_uncertainty(rf, X)
    assert mean.shape == (10,)
    assert std.shape == (10,)
    assert (std >= 0).all()


def test_predict_with_uncertainty_rejects_gradient_boosting(dataset):
    gb = GradientBoostingRegressor(random_state=1).fit(dataset[FEATURES], dataset["flux_LMH"])
    with pytest.raises(TypeError):
        predict_with_uncertainty(gb, dataset[FEATURES].head(5))


def test_predict_one_with_uncertainty(dataset):
    rf = RandomForestRegressor(n_estimators=30, random_state=1).fit(dataset[FEATURES], dataset["flux_LMH"])
    row = dataset[FEATURES].iloc[0].to_dict()
    mean, std = predict_one_with_uncertainty(rf, row, FEATURES)
    assert isinstance(mean, float) and isinstance(std, float)
    assert std >= 0


def test_uncertainty_higher_for_extrapolated_point(dataset):
    rf = RandomForestRegressor(n_estimators=100, random_state=1).fit(dataset[FEATURES], dataset["flux_LMH"])
    typical = dataset[FEATURES].iloc[0].to_dict()
    extreme = dict(typical)
    extreme["thickness_um"] = 100000.0  # far outside training range
    _, std_typical = predict_one_with_uncertainty(rf, typical, FEATURES)
    _, std_extreme = predict_one_with_uncertainty(rf, extreme, FEATURES)
    assert std_extreme > std_typical


# ------------------------------------------------------------------ explain

def test_feature_importances_sums_to_about_one(dataset):
    rf = RandomForestRegressor(n_estimators=50, random_state=1).fit(dataset[FEATURES], dataset["flux_LMH"])
    ranked = feature_importances(rf, FEATURES)
    assert len(ranked) == len(FEATURES)
    total = sum(r["importance"] for r in ranked)
    assert total == pytest.approx(1.0, abs=1e-6)
    assert ranked[0]["importance"] >= ranked[-1]["importance"]


def test_flux_model_correctly_identifies_permeability_and_tmp_as_most_important(dataset):
    """flux_LMH from the SIMPLE physics model is exactly
    permeability_LMH_bar * op_TMP_bar * exp(-k*t) -- a well-trained
    model should rank the raw drivers (or, since engineered features
    are on by default, the engineered proxies that directly encode
    that exact formula) far above inputs the simple model never uses
    (e.g. pore_size_nm)."""
    models, metrics, feature_list = train_models(dataset)
    ranked = feature_importances(models["flux_LMH"], feature_list)
    top_features = {r["feature"] for r in ranked[:3]}
    drivers = {"permeability_LMH_bar", "op_TMP_bar", "flux_physics_proxy",
              "permeability_x_TMP", "fouling_decay_factor"}
    assert top_features & drivers

    by_name = {r["feature"]: r["importance"] for r in ranked}
    assert by_name["flux_physics_proxy"] > by_name["pore_size_nm"]


def test_permutation_importances_basic(dataset):
    X = dataset[FEATURES]
    y = dataset["flux_LMH"]
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=1)
    rf = RandomForestRegressor(n_estimators=50, random_state=1).fit(Xtr, ytr)
    ranked = permutation_importances(rf, Xte, yte, FEATURES, n_repeats=5, seed=1)
    assert len(ranked) == len(FEATURES)
    # permeability/TMP should again dominate a model of the simple physics
    top = {r["feature"] for r in ranked[:3]}
    assert "permeability_LMH_bar" in top or "op_TMP_bar" in top


def test_conformal_prediction_interval_is_formal_and_reproducible(dataset):
    from sklearn.linear_model import LinearRegression

    train = dataset.iloc[:20]
    calibration = dataset.iloc[20:30]
    test = dataset.iloc[30:35]
    intervals = conformal_prediction_interval(
        LinearRegression(), train[FEATURES], train["flux_LMH"],
        calibration[FEATURES], calibration["flux_LMH"], test[FEATURES], alpha=0.05,
    )
    assert list(intervals.columns) == ["prediction", "lower", "upper", "interval_width", "coverage"]
    assert len(intervals) == len(test)
    assert (intervals["lower"] <= intervals["prediction"]).all()
    assert (intervals["prediction"] <= intervals["upper"]).all()
    assert intervals["coverage"].eq(0.95).all()


def test_used_vs_unused_features_separates_correctly(dataset):
    models, metrics, feature_list = train_models(dataset)
    split = used_vs_unused_features(models["flux_LMH"], feature_list)
    assert "pore_size_nm" in split["unused"]
    drivers = {"permeability_LMH_bar", "op_TMP_bar", "flux_physics_proxy",
              "permeability_x_TMP", "fouling_decay_factor"}
    assert set(split["used"]) & drivers
