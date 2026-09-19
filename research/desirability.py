"""RSM desirability and optimum checks."""

import numpy as np


def desirability_maximum(values, y_min, y_max, exponent=1.0):
    values = np.asarray(values, float)
    if y_max <= y_min or exponent <= 0:
        raise ValueError("y_max must exceed y_min and exponent must be positive.")
    return np.clip((values - y_min) / (y_max - y_min), 0, 1) ** exponent


def boundary_and_sensitivity(predictor, point, bounds, fraction=0.10):
    point = np.asarray(point, float)
    bounds = np.asarray(bounds, float)
    if bounds.shape != (len(point), 2):
        raise ValueError("bounds must contain one low/high pair per factor.")
    baseline = float(np.asarray(predictor(point.reshape(1, -1))).ravel()[0])
    boundary = any(np.isclose(value, bounds[i]).any() for i, value in enumerate(point))
    sensitivity = []
    for i, value in enumerate(point):
        delta = (bounds[i, 1] - bounds[i, 0]) * fraction
        low = point.copy()
        high = point.copy()
        low[i] = max(bounds[i, 0], value - delta)
        high[i] = min(bounds[i, 1], value + delta)
        sensitivity.append({"factor_index": i, "minus": float(np.asarray(predictor(low.reshape(1, -1))).ravel()[0]),
                            "plus": float(np.asarray(predictor(high.reshape(1, -1))).ravel()[0])})
    return {"baseline": baseline, "on_boundary": boundary, "sensitivity": sensitivity}
