"""Prediction-uncertainty estimates for the ML models.

For bagging-type ensembles (RandomForestRegressor, ExtraTreesRegressor
— both average many independently-trained trees), the spread of
individual trees' predictions around their mean is a standard, widely
used uncertainty proxy: wide disagreement between trees signals a
region of feature space the ensemble is less confident about (usually
because it saw fewer/less-consistent training examples there). This is
NOT a formal predictive interval or conformal-prediction guarantee —
just a cheap, model-native signal.

GradientBoostingRegressor is a *boosting* ensemble (trees fit
sequentially to residuals, not independently to resampled data), so its
per-tree spread does not have the same "disagreement = uncertainty"
interpretation; this module does not support it and says so explicitly
rather than returning a number that looks like it means the same thing.

This directly implements the previously-empty ml/uncertainty.py stub.
"""

import numpy as np
import pandas as pd


def conformal_prediction_interval(model, X_train, y_train, X_calibration, y_calibration,
                                  X_test, alpha=0.05):
    """Return split-conformal prediction intervals for a fitted or cloneable model.

    The calibration residual quantile gives marginal coverage under the usual
    exchangeability assumption.  This is a formal prediction interval, unlike
    the ensemble-spread diagnostic returned by ``predict_with_uncertainty``.
    """
    if not 0 < alpha < 1:
        raise ValueError("alpha must be between 0 and 1.")
    if len(X_calibration) < 2:
        raise ValueError("At least two calibration rows are required.")
    from sklearn.base import clone

    fitted = clone(model).fit(X_train, y_train)
    calibration_prediction = np.asarray(fitted.predict(X_calibration), dtype=float)
    residuals = np.abs(np.asarray(y_calibration, dtype=float) - calibration_prediction)
    quantile_level = min(1.0, np.ceil((len(residuals) + 1) * (1 - alpha)) / len(residuals))
    quantile = float(np.quantile(residuals, quantile_level, method="higher"))
    mean = np.asarray(fitted.predict(X_test), dtype=float)
    return pd.DataFrame({"prediction": mean, "lower": mean - quantile, "upper": mean + quantile,
                         "interval_width": np.full(len(mean), 2 * quantile),
                         "coverage": 1 - alpha})


def supports_uncertainty(model):
    """True if `model` is a bagging-type ensemble this module can
    compute a meaningful spread-based uncertainty for."""
    return hasattr(model, "estimators_") and not type(model).__name__ == "GradientBoostingRegressor"


def predict_with_uncertainty(model, X):
    """Return (mean_prediction, std_prediction) arrays, one value per
    row of `X`, using the spread across `model`'s individual trees.

    Raises TypeError for model types `supports_uncertainty` rejects
    (currently: anything without per-tree `estimators_`, and
    GradientBoostingRegressor specifically), rather than silently
    returning a meaningless number.
    """
    if not supports_uncertainty(model):
        raise TypeError(
            f"{type(model).__name__} does not support ensemble-spread uncertainty "
            "(only bagging ensembles with independent trees, e.g. RandomForestRegressor "
            "or ExtraTreesRegressor, are supported — see module docstring).")
    # Individual trees don't retain the parent ensemble's feature-name
    # bookkeeping, so predicting on a DataFrame directly on each tree
    # triggers a (harmless) "fitted without feature names" warning;
    # passing a plain ndarray avoids it.
    X_values = X.values if hasattr(X, "values") else X
    per_tree = np.stack([tree.predict(X_values) for tree in model.estimators_], axis=0)  # (n_trees, n_rows)
    return per_tree.mean(axis=0), per_tree.std(axis=0)


def predict_one_with_uncertainty(model, row_dict, feature_list):
    """Convenience wrapper for a single candidate (dict of feature
    values) instead of a full DataFrame: returns (mean, std) as plain
    floats."""
    import pandas as pd
    X = pd.DataFrame([row_dict])[feature_list]
    mean, std = predict_with_uncertainty(model, X)
    return float(mean[0]), float(std[0])
