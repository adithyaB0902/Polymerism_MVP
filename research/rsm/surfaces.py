"""Predicted response-surface grids for plotting (never extrapolates beyond the design range)."""

import numpy as np
import pandas as pd

from .quadratic_model import predict_quadratic


def response_surface_grid(fit_result, x_factor, y_factor, fixed=None, points=41):
    """Predict the response over two factors while the others are held fixed.

    ``fixed`` maps factor -> actual value; unspecified factors sit at their
    centre level. Returns dict(x, y, z, fixed) with ``z`` shaped (len(y), len(x)).
    """
    factors, levels = fit_result["factors"], fit_result["levels"]
    if x_factor == y_factor or x_factor not in factors or y_factor not in factors:
        raise ValueError("Choose two different model factors for the surface axes.")
    held = {f: float((fixed or {}).get(f, levels[f][1])) for f in factors if f not in (x_factor, y_factor)}
    for f, v in held.items():
        if not levels[f][0] - 1e-9 <= v <= levels[f][2] + 1e-9:
            raise ValueError(f"Fixed value for {f} is outside the studied range {levels[f][0]}-{levels[f][2]}.")
    xs = np.linspace(levels[x_factor][0], levels[x_factor][2], points)
    ys = np.linspace(levels[y_factor][0], levels[y_factor][2], points)
    gx, gy = np.meshgrid(xs, ys)
    grid = pd.DataFrame({x_factor: gx.ravel(), y_factor: gy.ravel(), **{f: np.full(gx.size, v) for f, v in held.items()}})
    z = predict_quadratic(fit_result, grid).reshape(gx.shape)
    return {"x": xs, "y": ys, "z": z, "fixed": held}
