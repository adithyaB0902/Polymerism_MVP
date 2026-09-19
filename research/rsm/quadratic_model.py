"""Second-order RSM regression with coefficient and model-adequacy statistics.

The model is fitted on *coded* factors (-1, 0, +1 at low, centre, high), the
standard RSM convention: coefficients are then comparable across factors and
the normal equations are well conditioned. Predictions are equivalent to a fit
in actual units; use ``predict_quadratic`` to evaluate the model on real values.

    y = b0 + sum(bi xi) + sum(bii xi^2) + sum(bij xi xj) + e
"""

from itertools import combinations
import numpy as np
import pandas as pd
from scipy import stats

from .anova import anova_table, lack_of_fit
from .box_behnken import FACTORS, to_coded


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


def _term_codes(factors):
    """Short labels A, B, ... / A², B² / AB, AC, ... in the same order as quadratic_terms."""
    letters = {f: chr(ord("A") + i) for i, f in enumerate(factors)}
    codes = ["Intercept"] + [letters[f] for f in factors] + [letters[f] + "²" for f in factors]
    codes += [letters[a] + letters[b] for a, b in combinations(factors, 2)]
    return codes


def _resolve_levels(work, factors, levels):
    resolved = {}
    for f in factors:
        if levels and f in levels:
            resolved[f] = tuple(map(float, levels[f]))
        elif f in FACTORS and levels is None:
            resolved[f] = FACTORS[f]
        else:
            low, high = float(work[f].min()), float(work[f].max())
            if high <= low:
                raise ValueError(f"Factor '{f}' was not varied; the quadratic model cannot be fitted.")
            resolved[f] = (low, (low + high) / 2.0, high)
    return resolved


def fit_quadratic(frame, response, factors, levels=None):
    """Fit the full quadratic model. ``levels`` maps factor -> (low, centre, high)."""
    factors = list(factors)
    work = frame[factors + [response]].apply(pd.to_numeric, errors="coerce").dropna()
    n_terms = 1 + 2 * len(factors) + len(factors) * (len(factors) - 1) // 2
    if len(work) <= n_terms:
        raise ValueError(
            "More complete experimental runs are required to fit the quadratic model: "
            f"{n_terms} coefficients need more than {n_terms} runs (found {len(work)})."
        )
    levels = _resolve_levels(work, factors, levels)
    coded = to_coded(work, factors, levels)
    design = quadratic_terms(coded, factors)
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
    mse = sse / dof
    xtx_inv = np.linalg.inv(x.T @ x)
    se = np.sqrt(np.maximum(np.diag(mse * xtx_inv), 0))
    t_values = np.divide(coefficients, se, out=np.zeros_like(coefficients), where=se > 0)
    p_values = 2 * stats.t.sf(np.abs(t_values), dof)
    t_crit = stats.t.ppf(0.975, dof)
    ss_total = float(np.sum((y - y.mean()) ** 2))
    ss_model = ss_total - sse
    r2 = 1 - sse / ss_total if ss_total else 1.0
    adj_r2 = 1 - (sse / dof) / (ss_total / (n - 1)) if ss_total else 1.0
    leverage = np.einsum("ij,jk,ik->i", x, xtx_inv, x)
    press = float(np.sum((residuals / (1 - leverage)) ** 2)) if np.all(leverage < 1 - 1e-10) else float("nan")
    predicted_r2 = 1 - press / ss_total if ss_total else float("nan")
    signal = float(predicted.max() - predicted.min())
    adequate_precision = signal / np.sqrt(p * mse / n) if mse > 0 else float("inf")
    model_f = (ss_model / (p - 1)) / mse if mse > 0 else float("inf")
    model_p = float(stats.f.sf(model_f, p - 1, dof))

    coefficient_table = pd.DataFrame({
        "term": design.columns, "code": _term_codes(factors), "coefficient": coefficients,
        "std_error": se, "ci95_low": coefficients - t_crit * se, "ci95_high": coefficients + t_crit * se,
        "t_value": t_values, "p_value": p_values, "significant_alpha_0_05": p_values < 0.05,
    })
    predictions = work.copy()
    predictions["predicted"] = predicted
    predictions["residual"] = residuals

    result = {
        "factors": factors, "terms": list(design.columns), "levels": levels,
        "coefficients": coefficient_table, "predictions": predictions,
        "r2": float(r2), "adjusted_r2": float(adj_r2), "predicted_r2": float(predicted_r2),
        "rmse": float(np.sqrt(np.mean(residuals ** 2))), "residual_std_error": float(np.sqrt(mse)),
        "cv_percent": float(np.sqrt(mse) / y.mean() * 100) if y.mean() else float("nan"),
        "adequate_precision": float(adequate_precision), "press": press, "sse": sse,
        "ss_total": ss_total, "residual_df": int(dof), "n_runs": int(n), "n_terms": int(p),
        "model_f": float(model_f), "model_p": model_p, "response": response,
        "anova": {"regression_ss": ss_model, "residual_ss": sse, "residual_df": dof},
        "lack_of_fit": lack_of_fit(coded[factors], y, sse, dof),
        "design_matrix": design, "coded_factors": coded,
    }
    result["stationary_point"] = stationary_point(result)
    result["anova_table"] = anova_table(result)
    return result


def predict_quadratic(fit_result, frame):
    """Predict the response for rows of actual factor values."""
    coded = to_coded(frame, fit_result["factors"], fit_result["levels"])
    design = quadratic_terms(coded, fit_result["factors"])
    return design.to_numpy(float) @ fit_result["coefficients"]["coefficient"].to_numpy(float)


def stationary_point(fit_result):
    """Stationary point of the fitted surface and its nature (max / min / saddle)."""
    factors, levels = fit_result["factors"], fit_result["levels"]
    beta = dict(zip(fit_result["coefficients"]["term"], fit_result["coefficients"]["coefficient"]))
    k = len(factors)
    b = np.array([beta[f] for f in factors])
    curvature = np.zeros((k, k))
    for i, f in enumerate(factors):
        curvature[i, i] = beta[f"{f}^2"]
    for i, j in combinations(range(k), 2):
        curvature[i, j] = curvature[j, i] = beta[f"{factors[i]}:{factors[j]}"] / 2.0
    eigenvalues = np.linalg.eigvalsh(curvature)
    if abs(np.linalg.det(curvature)) < 1e-12:
        return {"exists": False, "reason": "The quadratic form is singular; no unique stationary point.",
                "eigenvalues": eigenvalues.tolist()}
    coded = -0.5 * np.linalg.solve(curvature, b)
    tol = 1e-9 * max(1.0, float(np.abs(eigenvalues).max()))
    nature = "maximum" if np.all(eigenvalues < -tol) else "minimum" if np.all(eigenvalues > tol) else "saddle"
    actual = {f: levels[f][1] + coded[i] * (levels[f][2] - levels[f][0]) / 2.0 for i, f in enumerate(factors)}
    predicted = float(beta["intercept"] + 0.5 * b @ coded)
    return {"exists": True, "nature": nature, "coded": dict(zip(factors, coded.tolist())),
            "actual": actual, "predicted_response": predicted, "eigenvalues": eigenvalues.tolist(),
            "inside_design_region": bool(np.all(np.abs(coded) <= 1.0))}


def regression_diagnostics(fit_result):
    """Leverage, externally studentized residuals and Cook's distance per run."""
    x = fit_result["design_matrix"].to_numpy(float)
    preds = fit_result["predictions"]
    resid = preds["residual"].to_numpy(float)
    n, p = x.shape
    dof = n - p
    mse = fit_result["sse"] / dof
    h = np.einsum("ij,jk,ik->i", x, np.linalg.inv(x.T @ x), x)
    one_minus_h = np.maximum(1 - h, 1e-12)
    if dof > 1:
        mse_i = np.maximum(((dof * mse) - resid ** 2 / one_minus_h) / (dof - 1), 1e-12)
        studentized = resid / np.sqrt(mse_i * one_minus_h)
    else:
        studentized = np.full(n, np.nan)
    cooks = resid ** 2 / (p * mse) * h / one_minus_h ** 2 if mse > 0 else np.zeros(n)
    out = pd.DataFrame({"actual": preds[fit_result["response"]].to_numpy(float),
                        "predicted": preds["predicted"].to_numpy(float), "residual": resid,
                        "leverage": h, "studentized_residual": studentized, "cooks_distance": cooks},
                       index=preds.index)
    out["flagged"] = (out["cooks_distance"] > 4 / n) | (out["studentized_residual"].abs() > 3)
    return out
