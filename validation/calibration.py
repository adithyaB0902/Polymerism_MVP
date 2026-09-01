"""Fit physics-model parameters to real experimental measurements.

Given a starting Membrane/Water/OperatingConditions combination and a
set of real measurements (flux and/or rejection observed at known
operating times), this fits `permeability_LMH_bar`, `fouling_coefficient`
and `baseline_rejection_percent` by nonlinear least squares
(scipy.optimize.least_squares) so that `models.simulator.run_simulation`
best reproduces the measurements. It reports the fit quality (R2)
before and after, which `models.feasibility.feasibility_assessment` can
use to justify a reliability rating above "LOW" (see that module).

This directly implements the previously-empty
validation/calibration.py stub referenced throughout the codebase.
"""

from dataclasses import replace

import numpy as np
from scipy.optimize import least_squares
from sklearn.metrics import r2_score

from models.membrane import Targets as _Targets
from models.simulator import run_simulation

_DUMMY_TARGETS = _Targets()  # calibration doesn't score feasibility, just needs a valid Targets object


def _predict_flux_and_rejection(permeability, fouling_k, rejection, membrane, water, op, time_hr):
    """Run the simple physics model at a given operating time with
    candidate parameter values substituted in, returning (flux, rejection)."""
    m = replace(membrane, permeability_LMH_bar=permeability, fouling_coefficient=fouling_k,
               baseline_rejection_percent=rejection)
    o = replace(op, operating_time_hr=time_hr)
    result = run_simulation(m, water, o, _DUMMY_TARGETS)
    return result["flux_LMH"], result["rejection_percent"]


def calibrate_sample(membrane, water, op, measurements):
    """Fit `permeability_LMH_bar`, `fouling_coefficient` and
    `baseline_rejection_percent` to a set of real measurements.

    `measurements` is a list of dicts, each with at least one of
    `flux_LMH` / `rejection_percent`, plus optionally
    `operating_time_hr` (the time at which that row was measured — if
    omitted, `op.operating_time_hr` is used for every row, appropriate
    for single-point-in-time measurements such as
    validate_experimental_csv output that has no time column).

    Returns a dict with:
      fitted_params        -- {"permeability_LMH_bar":.., "fouling_coefficient":.., "baseline_rejection_percent":..}
      calibrated_membrane  -- a new Membrane with the fitted parameters
      metrics_before / metrics_after -- {"MAE":.., "RMSE":.., "R2":..} for flux+rejection combined
      post_fit_r2          -- the R2 in metrics_after (convenience field, used by feasibility_assessment)
      n_measurements
      converged             -- whether scipy's optimizer reported success
    """
    if len(measurements) < 2:
        raise ValueError("Need at least 2 measurements to calibrate 3 parameters without heavy overfitting risk.")

    have_flux = any(row.get("flux_LMH") is not None for row in measurements)
    have_rejection = any(row.get("rejection_percent") is not None for row in measurements)
    if not (have_flux or have_rejection):
        raise ValueError("Measurements must include flux_LMH and/or rejection_percent.")

    times = [row.get("operating_time_hr", op.operating_time_hr) for row in measurements]

    def residuals(params):
        permeability, fouling_k, rejection = params
        permeability = max(permeability, 1e-6)
        fouling_k = max(fouling_k, 0.0)
        rejection = min(max(rejection, 0.0), 100.0)
        res = []
        for row, t in zip(measurements, times):
            pred_flux, pred_rej = _predict_flux_and_rejection(permeability, fouling_k, rejection,
                                                               membrane, water, op, t)
            if row.get("flux_LMH") is not None:
                res.append(pred_flux - row["flux_LMH"])
            if row.get("rejection_percent") is not None:
                res.append(pred_rej - row["rejection_percent"])
        return res

    x0 = [membrane.permeability_LMH_bar, membrane.fouling_coefficient, membrane.baseline_rejection_percent]
    bounds = ([1e-6, 0.0, 0.0], [np.inf, np.inf, 100.0])
    fit = least_squares(residuals, x0, bounds=bounds)
    permeability, fouling_k, rejection = fit.x

    def _actual_and_predicted(permeability, fouling_k, rejection):
        actual, predicted = [], []
        for row, t in zip(measurements, times):
            pred_flux, pred_rej = _predict_flux_and_rejection(permeability, fouling_k, rejection,
                                                               membrane, water, op, t)
            if row.get("flux_LMH") is not None:
                actual.append(row["flux_LMH"]); predicted.append(pred_flux)
            if row.get("rejection_percent") is not None:
                actual.append(row["rejection_percent"]); predicted.append(pred_rej)
        return np.array(actual), np.array(predicted)

    def _metrics(permeability, fouling_k, rejection):
        actual, predicted = _actual_and_predicted(permeability, fouling_k, rejection)
        mae = float(np.mean(np.abs(actual - predicted)))
        rmse = float(np.sqrt(np.mean((actual - predicted) ** 2)))
        # R2 is undefined (and misleading) with fewer than 2 points or
        # zero variance in the actual values; fall back to None rather
        # than a fabricated number.
        r2 = float(r2_score(actual, predicted)) if len(actual) >= 2 and np.var(actual) > 0 else None
        return {"MAE": mae, "RMSE": rmse, "R2": r2}

    metrics_before = _metrics(*x0)
    metrics_after = _metrics(permeability, fouling_k, rejection)

    calibrated_membrane = replace(membrane, permeability_LMH_bar=float(permeability),
                                  fouling_coefficient=float(fouling_k),
                                  baseline_rejection_percent=float(rejection))

    return {
        "fitted_params": {"permeability_LMH_bar": float(permeability), "fouling_coefficient": float(fouling_k),
                          "baseline_rejection_percent": float(rejection)},
        "calibrated_membrane": calibrated_membrane,
        "metrics_before": metrics_before,
        "metrics_after": metrics_after,
        "post_fit_r2": metrics_after["R2"],
        "n_measurements": len(measurements),
        "converged": bool(fit.success),
    }
