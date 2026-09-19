"""Differential-evolution optimization of a trained Pb removal surrogate."""

import numpy as np
from scipy.optimize import differential_evolution

BOUNDS = [(20.0, 60.0), (1.0, 5.0), (3.0, 6.0), (10.0, 50.0)]


def optimize_removal(predictor, bounds=BOUNDS, seed=42, maxiter=100):
    history = []
    def objective(x):
        value = float(np.asarray(predictor(np.asarray(x).reshape(1, -1))).ravel()[0])
        history.append({"iteration": len(history), "predicted_removal_percent": value})
        return -value
    result = differential_evolution(objective, bounds=bounds, seed=seed, maxiter=maxiter, polish=True)
    return {"conditions": dict(zip(["chitosan", "biochar", "pH", "initial_pb"], result.x.tolist())),
            "predicted_removal_percent": float(-result.fun), "success": bool(result.success),
            "message": result.message, "convergence": history}

