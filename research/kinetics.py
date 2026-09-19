"""Pseudo-first-order and pseudo-second-order kinetic fits."""

import numpy as np
from scipy.optimize import curve_fit


def fit_kinetics(time_min, qe_t):
    t, q = np.asarray(time_min, float), np.asarray(qe_t, float)
    if len(t) < 3 or len(t) != len(q) or np.any(t < 0) or np.any(q < 0):
        raise ValueError("Kinetic data require at least three non-negative time/qe pairs.")
    def pfo(x, qe, k1): return qe * (1 - np.exp(-k1 * x))
    def pso(x, qe, k2): return (k2 * qe ** 2 * x) / (1 + k2 * qe * x)
    result = {}
    for name, fn, p0 in (("pseudo_first_order", pfo, [max(q), .01]), ("pseudo_second_order", pso, [max(q), .01])):
        params, _ = curve_fit(fn, t, q, p0=p0, bounds=(0, np.inf), maxfev=10000)
        pred = fn(t, *params)
        result[name] = {"parameters": params.tolist(), "predicted": pred,
                        "r2": float(1 - np.sum((q - pred) ** 2) / np.sum((q - q.mean()) ** 2))}
    return result

