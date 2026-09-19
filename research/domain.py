"""Applicability-domain checks for the coded RSM design."""

import numpy as np


def leverage_report(reference, candidates, factors):
    """Return leverage and factor-range flags for candidate rows."""
    ref = np.asarray(reference[factors], float)
    x_ref = np.column_stack([np.ones(len(ref)), ref])
    inverse = np.linalg.pinv(x_ref.T @ x_ref)
    values = np.asarray(candidates[factors], float)
    x = np.column_stack([np.ones(len(values)), values])
    leverage = np.einsum("ij,jk,ik->i", x, inverse, x)
    limits = {f: (float(np.min(ref[:, i])), float(np.max(ref[:, i]))) for i, f in enumerate(factors)}
    outside = np.array([
        any(value < limits[f][0] or value > limits[f][1] for f, value in zip(factors, row))
        for row in values
    ])
    threshold = 2 * x_ref.shape[1] / max(len(ref), 1)
    return {"leverage": leverage, "high_leverage": leverage > threshold,
            "outside_factor_range": outside, "threshold": threshold, "limits": limits}
