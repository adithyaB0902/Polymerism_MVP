"""Langmuir and Freundlich isotherm fits."""

import numpy as np
from scipy.optimize import curve_fit


def _r2(y, pred):
    denom = np.sum((y - np.mean(y)) ** 2)
    return float(1 - np.sum((y - pred) ** 2) / denom) if denom else 1.0


def fit_isotherms(concentration, qe):
    c, q = np.asarray(concentration, float), np.asarray(qe, float)
    if len(c) < 3 or len(c) != len(q) or np.any(c <= 0) or np.any(q < 0):
        raise ValueError("Isotherm data require at least three positive concentration/qe pairs.")
    def langmuir(x, qmax, kl): return qmax * kl * x / (1 + kl * x)
    def freundlich(x, kf, n): return kf * x ** (1 / n)
    fits = {}
    for name, fn, p0 in (("Langmuir", langmuir, [max(q) * 1.2, 0.1]), ("Freundlich", freundlich, [1, 2])):
        params, _ = curve_fit(fn, c, q, p0=p0, bounds=(0, np.inf), maxfev=10000)
        pred = fn(c, *params)
        fits[name] = {"parameters": params.tolist(), "predicted": pred, "r2": _r2(q, pred),
                      "rmse": float(np.sqrt(np.mean((q - pred) ** 2)))}
    return fits

