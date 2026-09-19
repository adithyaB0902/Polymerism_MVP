"""Small helpers for reporting calculated RSM statistics."""


def significance_table(fit_result, alpha=0.05):
    table = fit_result["coefficients"].copy()
    table["significant"] = table["p_value"] < alpha
    return table

