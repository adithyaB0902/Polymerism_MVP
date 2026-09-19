"""Second-order RSM regression with coefficient statistics."""

from itertools import combinations
import numpy as np
import pandas as pd
from scipy import stats


def quadratic_terms(frame, factors):
    x = frame[list(factors)].astype(float)
    data = {"intercept": np.ones(len(x))}
    for f in factors:
        data[f] = x[f].to_numpy()
    for f in factors:
        data[f"{f}^2"] = x[f].to_numpy() ** 2
    for a, b in combinations(factors, 2):
        data[f"{a}:{b}"] = x[a].to_numpy() * x[b].to_numpy()
    return pd.DataFrame(data, index=frame.index)


def fit_quadratic(frame, response, factors):
    work = frame[list(factors) + [response]].dropna()
    if len(work) < len(factors) * 2 + 1:
        raise ValueError("More complete experimental runs are required to fit the quadratic model.")
    design = quadratic_terms(work, factors)
    y = work[response].to_numpy(float)
    x = design.to_numpy(float)
    coefficients, _, rank, _ = np.linalg.lstsq(x, y, rcond=None)
    if rank < x.shape[1]:
        raise ValueError("Quadratic design matrix is rank deficient; add independent runs.")
    predicted = x @ coefficients
    residuals = y - predicted
    n, p = x.shape
    dof = n - p
    sse = float(np.sum(residuals ** 2))
    mse = sse / dof if dof > 0 else float("nan")
    cov = mse * np.linalg.pinv(x.T @ x)
    se = np.sqrt(np.maximum(np.diag(cov), 0))
    t_values = np.divide(coefficients, se, out=np.zeros_like(coefficients), where=se > 0)
    p_values = 2 * stats.t.sf(np.abs(t_values), dof) if dof > 0 else np.full(p, np.nan)
    ss_total = float(np.sum((y - y.mean()) ** 2))
    r2 = 1 - sse / ss_total if ss_total else 1.0
    adj_r2 = 1 - (1 - r2) * (n - 1) / dof if dof > 0 else float("nan")
    coefficient_table = pd.DataFrame({
        "term": design.columns, "coefficient": coefficients, "std_error": se,
        "t_value": t_values, "p_value": p_values,
        "significant_alpha_0_05": p_values < 0.05,
    })
    predictions = work.copy()
    predictions["predicted"] = predicted
    predictions["residual"] = residuals
    return {
        "factors": list(factors), "terms": list(design.columns), "coefficients": coefficient_table,
        "predictions": predictions, "r2": float(r2), "adjusted_r2": float(adj_r2),
        "rmse": float(np.sqrt(np.mean(residuals ** 2))), "sse": sse,
        "anova": {"regression_ss": ss_total - sse, "residual_ss": sse, "residual_df": dof},
    }

