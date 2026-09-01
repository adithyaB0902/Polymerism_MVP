import numpy as np
import pytest

from models.membrane import Membrane, Water, OperatingConditions
from models.simulator import run_simulation
from validation.calibration import calibrate_sample


def _make_measurements_from_true_membrane(true_membrane, water, op, times, noise_std=0.0, seed=0):
    """Simulate 'real' measurements from a known-true membrane at
    several operating times, optionally with Gaussian measurement
    noise, for use as calibration test data."""
    rng = np.random.default_rng(seed)
    rows = []
    for t in times:
        from dataclasses import replace
        o = replace(op, operating_time_hr=t)
        r = run_simulation(true_membrane, water, o, __import__("models.membrane", fromlist=["Targets"]).Targets())
        rows.append({
            "operating_time_hr": t,
            "flux_LMH": r["flux_LMH"] + rng.normal(0, noise_std),
            "rejection_percent": min(100, max(0, r["rejection_percent"] + rng.normal(0, noise_std / 10 or 0))),
        })
    return rows


def test_calibration_recovers_true_parameters_from_noise_free_data():
    true_membrane = Membrane("TrueLab", "UF", 100.0, 42.0, baseline_rejection_percent=93.0,
                             fouling_coefficient=0.06)
    water = Water()
    op = OperatingConditions(TMP_bar=2.0)
    measurements = _make_measurements_from_true_membrane(
        true_membrane, water, op, times=[0.5, 1, 2, 4, 8, 12], noise_std=0.0)

    # start calibration from a deliberately WRONG initial guess
    wrong_start = Membrane("Guess", "UF", 100.0, 10.0, baseline_rejection_percent=80.0,
                           fouling_coefficient=0.01)
    result = calibrate_sample(wrong_start, water, op, measurements)

    assert result["fitted_params"]["permeability_LMH_bar"] == pytest.approx(42.0, rel=0.02)
    assert result["fitted_params"]["fouling_coefficient"] == pytest.approx(0.06, rel=0.05)
    assert result["fitted_params"]["baseline_rejection_percent"] == pytest.approx(93.0, rel=0.02)
    assert result["post_fit_r2"] > 0.999
    assert result["metrics_after"]["RMSE"] < result["metrics_before"]["RMSE"]


def test_calibration_improves_fit_even_with_noisy_data():
    true_membrane = Membrane("TrueLab", "UF", 100.0, 30.0, baseline_rejection_percent=90.0,
                             fouling_coefficient=0.04)
    water = Water()
    op = OperatingConditions(TMP_bar=2.0)
    measurements = _make_measurements_from_true_membrane(
        true_membrane, water, op, times=[0.5, 1, 2, 4, 8, 12, 16], noise_std=1.5, seed=3)

    wrong_start = Membrane("Guess", "UF", 100.0, 60.0, baseline_rejection_percent=70.0,
                           fouling_coefficient=0.15)
    result = calibrate_sample(wrong_start, water, op, measurements)

    assert result["post_fit_r2"] > 0.9   # should still fit well despite noise
    assert result["metrics_after"]["RMSE"] < result["metrics_before"]["RMSE"]
    assert result["converged"]


def test_calibration_requires_at_least_two_measurements():
    m = Membrane("X", "UF", 100.0, 25.0, baseline_rejection_percent=95.0, fouling_coefficient=0.03)
    water, op = Water(), OperatingConditions()
    with pytest.raises(ValueError):
        calibrate_sample(m, water, op, [{"flux_LMH": 50.0}])


def test_calibration_requires_flux_or_rejection_columns():
    m = Membrane("X", "UF", 100.0, 25.0, baseline_rejection_percent=95.0, fouling_coefficient=0.03)
    water, op = Water(), OperatingConditions()
    with pytest.raises(ValueError):
        calibrate_sample(m, water, op, [{"operating_time_hr": 1}, {"operating_time_hr": 2}])


def test_calibration_works_with_rejection_only_data():
    true_membrane = Membrane("TrueLab", "UF", 100.0, 25.0, baseline_rejection_percent=97.0,
                             fouling_coefficient=0.03)
    water, op = Water(), OperatingConditions(TMP_bar=2.0)
    measurements = [{"operating_time_hr": t, "rejection_percent": 97.0} for t in [1, 2, 4]]
    wrong_start = Membrane("Guess", "UF", 100.0, 25.0, baseline_rejection_percent=80.0, fouling_coefficient=0.03)
    result = calibrate_sample(wrong_start, water, op, measurements)
    assert result["fitted_params"]["baseline_rejection_percent"] == pytest.approx(97.0, abs=0.5)
