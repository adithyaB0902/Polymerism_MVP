"""Calculators and optimization helpers for measured research workflows."""

import numpy as np
import pandas as pd
from scipy.optimize import differential_evolution
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

from .desirability import desirability_maximum


def swelling_degree(wet_mass_g, dry_mass_g):
    if dry_mass_g <= 0 or wet_mass_g < dry_mass_g:
        raise ValueError("wet mass must be at least dry mass and dry mass must be positive.")
    return (wet_mass_g - dry_mass_g) / dry_mass_g * 100.0


def porosity(wet_mass_g, dry_mass_g, water_density_g_cm3, area_cm2, thickness_cm):
    if water_density_g_cm3 <= 0 or area_cm2 <= 0 or thickness_cm <= 0:
        raise ValueError("water density, area, and thickness must be positive.")
    if wet_mass_g < dry_mass_g:
        raise ValueError("wet mass must be at least dry mass.")
    return (wet_mass_g - dry_mass_g) / (water_density_g_cm3 * area_cm2 * thickness_cm) * 100.0


def multiresponse_desirability(values, specifications):
    """Calculate geometric Derringer-Suich desirability for response columns.

    ``specifications`` maps a response to ``(goal, low, high, exponent)``;
    goal is currently ``maximize`` or ``minimize``.
    """
    frame = pd.DataFrame(values)
    desirabilities = {}
    for name, (goal, low, high, exponent) in specifications.items():
        if goal == "maximize":
            desirabilities[name] = desirability_maximum(frame[name], low, high, exponent)
        elif goal == "minimize":
            desirabilities[name] = desirability_maximum(high - frame[name], 0, high - low, exponent)
        else:
            raise ValueError("goal must be maximize or minimize.")
    output = pd.DataFrame(desirabilities, index=frame.index)
    output["overall_desirability"] = np.prod(output, axis=1) ** (1 / len(specifications))
    return output


def optimize_multiresponse(predictors, bounds, specifications, seed=42, maxiter=80):
    """Optimize geometric desirability of several predictor functions."""
    def objective(point):
        values = {name: float(np.asarray(predictor(np.asarray(point).reshape(1, -1))).ravel()[0])
                  for name, predictor in predictors.items()}
        return -float(multiresponse_desirability(pd.DataFrame([values]), specifications)
                      ["overall_desirability"].iloc[0])

    result = differential_evolution(objective, bounds, seed=seed, maxiter=maxiter, polish=True)
    return {"point": result.x, "desirability": -float(result.fun), "success": bool(result.success)}


def suggest_next_experiment(frame, factors, target, bounds, n_candidates=2000, seed=42):
    """Suggest the highest-uncertainty in-domain point using a Gaussian process."""
    data = frame[factors + [target]].dropna()
    if len(data) < 3:
        raise ValueError("At least three measured rows are required for active learning.")
    rng = np.random.default_rng(seed)
    candidates = np.column_stack([rng.uniform(low, high, n_candidates) for low, high in bounds])
    model = make_pipeline(StandardScaler(), GaussianProcessRegressor(
        kernel=ConstantKernel(1.0) * Matern(nu=2.5) + WhiteKernel(), normalize_y=True,
        random_state=seed,
    ))
    model.fit(data[factors], data[target])
    _, uncertainty = model.predict(candidates, return_std=True)
    index = int(np.argmax(uncertainty))
    return {"factors": dict(zip(factors, candidates[index])), "uncertainty": float(uncertainty[index]),
            "data_source": "predicted uncertainty; suggested experiment, not a measured result"}


def regeneration_summary(records):
    frame = pd.DataFrame(records).sort_values("cycle").reset_index(drop=True)
    if frame.empty or "removal_percent" not in frame:
        raise ValueError("At least one reuse record with removal_percent is required.")
    initial = float(frame.loc[0, "removal_percent"])
    frame["retained_removal_percent"] = frame["removal_percent"] / initial * 100 if initial else np.nan
    if "desorbed_mg" in frame and "loaded_mg" in frame:
        frame["desorption_efficiency_percent"] = frame["desorbed_mg"] / frame["loaded_mg"] * 100
    return frame