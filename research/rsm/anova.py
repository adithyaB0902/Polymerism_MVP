"""ANOVA, lack-of-fit and model-adequacy reporting for the quadratic RSM.

All quantities are calculated from the supplied data; nothing is assumed or
hard-coded. Per-term rows use partial (Type III) sums of squares, SS = t^2 * MSE,
which is the convention used by most RSM software; they need not add up to the
model sum of squares because the columns of a Box-Behnken design are not
perfectly orthogonal.
"""

import numpy as np
import pandas as pd
from scipy import stats


def significance_table(fit_result, alpha=0.05):
    table = fit_result["coefficients"].copy()
    table["significant"] = table["p_value"] < alpha
    return table


def lack_of_fit(coded_factors, y, sse, residual_df):
    """Lack-of-fit F-test using replicated factor settings as pure error.

    Returns None when no setting is replicated (pure error cannot be estimated).
    """
    x = np.round(np.asarray(coded_factors, float), 9)
    y = np.asarray(y, float)
    _, inverse = np.unique(x, axis=0, return_inverse=True)
    inverse = np.asarray(inverse).ravel()
    n_groups = int(inverse.max()) + 1
    df_pe = int(len(y) - n_groups)
    if df_pe == 0:
        return None
    ss_pe = float(sum(np.sum((y[inverse == g] - y[inverse == g].mean()) ** 2) for g in range(n_groups)))
    ss_lof, df_lof = float(sse - ss_pe), int(residual_df - df_pe)
    out = {"ss_lack_of_fit": ss_lof, "df_lack_of_fit": df_lof, "ss_pure_error": ss_pe,
           "df_pure_error": df_pe, "f_value": float("nan"), "p_value": float("nan")}
    if df_lof > 0 and ss_pe > 0:
        f_value = (ss_lof / df_lof) / (ss_pe / df_pe)
        out["f_value"] = float(f_value)
        out["p_value"] = float(stats.f.sf(f_value, df_lof, df_pe))
    return out


def anova_table(fit_result):
    """Full ANOVA table as a DataFrame (model, terms, residual, LOF, pure error, total)."""
    coef = fit_result["coefficients"]
    n, p = fit_result["n_runs"], fit_result["n_terms"]
    sse, ss_total = fit_result["sse"], fit_result["ss_total"]
    dof = fit_result["residual_df"]
    mse = sse / dof
    ss_model = ss_total - sse
    rows = [{"source": "Model", "sum_of_squares": ss_model, "df": p - 1,
             "mean_square": ss_model / (p - 1), "f_value": fit_result["model_f"],
             "p_value": fit_result["model_p"]}]
    for _, row in coef.iterrows():
        if row["term"] == "intercept":
            continue
        ss = float(row["t_value"] ** 2 * mse)
        rows.append({"source": row["term"], "sum_of_squares": ss, "df": 1, "mean_square": ss,
                     "f_value": ss / mse, "p_value": float(row["p_value"])})
    rows.append({"source": "Residual", "sum_of_squares": sse, "df": dof, "mean_square": mse,
                 "f_value": np.nan, "p_value": np.nan})
    lof = fit_result["lack_of_fit"]
    if lof is not None:
        rows.append({"source": "Lack of fit", "sum_of_squares": lof["ss_lack_of_fit"],
                     "df": lof["df_lack_of_fit"],
                     "mean_square": lof["ss_lack_of_fit"] / lof["df_lack_of_fit"] if lof["df_lack_of_fit"] > 0 else np.nan,
                     "f_value": lof["f_value"], "p_value": lof["p_value"]})
        rows.append({"source": "Pure error", "sum_of_squares": lof["ss_pure_error"],
                     "df": lof["df_pure_error"], "mean_square": lof["ss_pure_error"] / lof["df_pure_error"],
                     "f_value": np.nan, "p_value": np.nan})
    rows.append({"source": "Cor total", "sum_of_squares": ss_total, "df": n - 1,
                 "mean_square": np.nan, "f_value": np.nan, "p_value": np.nan})
    return pd.DataFrame(rows)


def model_warnings(fit_result, alpha=0.05):
    """Plain-language adequacy flags. Advisory only; they do not replace judgement."""
    warnings = []
    if fit_result["model_p"] > alpha:
        warnings.append(f"The model is not significant (p = {fit_result['model_p']:.3g}).")
    lof = fit_result["lack_of_fit"]
    if lof is None:
        warnings.append("No replicated runs: lack of fit cannot be tested.")
    elif np.isfinite(lof["p_value"]) and lof["p_value"] < alpha:
        warnings.append(f"Significant lack of fit (p = {lof['p_value']:.3g}): a quadratic model may "
                        "not describe this system adequately.")
    if np.isfinite(fit_result["predicted_r2"]) and fit_result["adjusted_r2"] - fit_result["predicted_r2"] > 0.2:
        warnings.append(f"Predicted R² ({fit_result['predicted_r2']:.3f}) is much lower than adjusted R² "
                        f"({fit_result['adjusted_r2']:.3f}); the model may be over-fitted or influenced by outliers.")
    if fit_result["adequate_precision"] < 4:
        warnings.append(f"Adequate precision is {fit_result['adequate_precision']:.2f} (< 4): weak signal-to-noise.")
    if fit_result["residual_df"] < 10:
        warnings.append(f"Only {fit_result['residual_df']} residual degrees of freedom; estimates are unstable.")
    if not fit_result["stationary_point"].get("inside_design_region", True):
        warnings.append("The stationary point lies outside the studied region; treat it as extrapolation.")
    return warnings
