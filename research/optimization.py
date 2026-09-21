"""Differential-evolution optimization of a trained Pb removal surrogate."""

import numpy as np
from scipy.optimize import differential_evolution

from .science_tools import multiresponse_desirability

BOUNDS = [(20.0, 60.0), (1.0, 5.0), (3.0, 6.0), (10.0, 50.0)]


def optimize_removal(predictor, bounds=BOUNDS, seed=42, maxiter=100, top_n=1):
    history = []
    def objective(x):
        value = float(np.asarray(predictor(np.asarray(x).reshape(1, -1))).ravel()[0])
        history.append({"iteration": len(history), "predicted_removal_percent": value})
        return -value
    result = differential_evolution(objective, bounds=bounds, seed=seed, maxiter=maxiter, polish=True)
    candidates = [{"conditions": dict(zip(["chitosan", "biochar", "pH", "initial_pb"], result.x.tolist())),
            "predicted_removal_percent": float(-result.fun), "success": bool(result.success),
            "message": result.message, "convergence": history}]
    if top_n > 1:
        rng = np.random.default_rng(seed)
        probes = rng.uniform(np.asarray(bounds)[:, 0], np.asarray(bounds)[:, 1], size=(max(100, top_n * 50), len(bounds)))
        values = np.asarray(predictor(probes), float).ravel()
        order = np.argsort(values)[::-1]
        for index in order:
            candidate = dict(zip(["chitosan", "biochar", "pH", "initial_pb"], probes[index].tolist()))
            if all(np.linalg.norm(probes[index] - np.asarray(list(item["conditions"].values()))) > 1e-6
                   for item in candidates):
                candidates.append({"conditions": candidate, "predicted_removal_percent": float(values[index]),
                                   "success": True, "message": "sampled in-region candidate",
                                   "convergence": []})
            if len(candidates) >= top_n:
                break
    return candidates[0] if top_n == 1 else {"candidates": candidates[:top_n], "convergence": history}


def optimize_desirability(predictors, specifications, bounds=BOUNDS, seed=42, maxiter=100):
    """Derringer-Suich multi-response optimization within the supplied bounds."""
    def objective(x):
        values = {name: float(np.asarray(model(np.asarray(x).reshape(1, -1))).ravel()[0])
                  for name, model in predictors.items()}
        return -float(multiresponse_desirability([values], specifications)
                      ["overall_desirability"].iloc[0])

    result = differential_evolution(objective, bounds=bounds, seed=seed, maxiter=maxiter, polish=True)
    point = np.asarray(result.x, float)
    responses = {name: float(np.asarray(model(point.reshape(1, -1))).ravel()[0])
                 for name, model in predictors.items()}
    return {"conditions": dict(zip(["chitosan", "biochar", "pH", "initial_pb"], point)),
            "responses": responses, "desirability": -float(result.fun),
            "on_boundary": bool(np.isclose(point, np.asarray(bounds)[:, 0]).any() or
                                 np.isclose(point, np.asarray(bounds)[:, 1]).any()),
            "success": bool(result.success)}
