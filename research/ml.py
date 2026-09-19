"""Paper-specific ML comparison on four experimental factors.

Optional packages are used only when installed: XGBoost and SHAP are not
required to run the core workflow, so the application remains compatible with
the original dependency set.
"""

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import RandomForestRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, RBF, WhiteKernel
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
from sklearn.inspection import permutation_importance
from sklearn.model_selection import KFold, RepeatedKFold, cross_validate, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

FEATURES = ["chitosan_wt_percent", "biochar_wt_percent", "pH", "initial_pb_mg_l"]
TARGET = "removal_percent"


def model_candidates(seed=42):
    candidates = {
        "Multiple Linear Regression": LinearRegression(),
        "SVR": make_pipeline(StandardScaler(), SVR(C=10, epsilon=0.1, kernel="rbf")),
        "Gaussian Process Regression": make_pipeline(
            StandardScaler(), GaussianProcessRegressor(
                kernel=ConstantKernel(1.0) * RBF(length_scale=1.0) + WhiteKernel(),
                normalize_y=True, random_state=seed,
            )
        ),
        "Random Forest": RandomForestRegressor(n_estimators=200, random_state=seed, n_jobs=-1),
        "Shallow MLP": make_pipeline(StandardScaler(), MLPRegressor(hidden_layer_sizes=(16,), max_iter=2000, random_state=seed)),
    }
    try:
        from xgboost import XGBRegressor
        candidates["XGBoost"] = XGBRegressor(n_estimators=200, max_depth=3, learning_rate=.05,
                                             objective="reg:squarederror", random_state=seed, n_jobs=1)
    except ImportError:
        pass
    return candidates


def _metrics(y, pred):
    error = np.asarray(y) - np.asarray(pred)
    return {
        "R2": float(1 - np.sum(error ** 2) / np.sum((np.asarray(y) - np.mean(y)) ** 2)) if len(y) > 1 else float("nan"),
        "RMSE": float(np.sqrt(np.mean(error ** 2))),
        "MAE": float(np.mean(np.abs(error))),
        "AARD": float(np.mean(np.abs(error) / np.maximum(np.abs(y), 1e-12)) * 100),
    }


def compare_models(frame, n_splits=5, n_repeats=2, seed=42):
    data = frame[FEATURES + [TARGET]].dropna()
    if len(data) < n_splits:
        raise ValueError(f"At least {n_splits} complete experimental rows are required.")
    x, y = data[FEATURES], data[TARGET]
    cv = RepeatedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=seed)
    results = []
    for name, estimator in model_candidates(seed).items():
        scores = cross_validate(estimator, x, y, cv=cv,
                                scoring=("r2", "neg_root_mean_squared_error", "neg_mean_absolute_error"))
        oof = cross_val_predict(clone(estimator), x, y,
                                cv=KFold(n_splits=n_splits, shuffle=True, random_state=seed))
        pred_rmse = -scores["test_neg_root_mean_squared_error"]
        pred_mae = -scores["test_neg_mean_absolute_error"]
        results.append({"model": name, "R2": float(scores["test_r2"].mean()),
                        "RMSE": float(pred_rmse.mean()), "MAE": float(pred_mae.mean()),
                        "AARD": float(np.mean(np.abs(np.asarray(y) - oof) / np.maximum(np.abs(y), 1e-12)) * 100),
                        "R2_std": float(scores["test_r2"].std())})
    table = pd.DataFrame(results)
    table["selected"] = table["RMSE"] == table["RMSE"].min()
    return table.sort_values("RMSE").reset_index(drop=True)


def fit_best_model(frame, comparison=None, seed=42):
    data = frame[FEATURES + [TARGET]].dropna()
    comparison = comparison if comparison is not None else compare_models(frame)
    selected_name = comparison.sort_values("RMSE").iloc[0]["model"]
    estimator = model_candidates(seed)[selected_name]
    estimator.fit(data[FEATURES], data[TARGET])
    return estimator, selected_name


def explain_model(model, frame, seed=42, n_repeats=20):
    data = frame[FEATURES + [TARGET]].dropna()
    importance = permutation_importance(model, data[FEATURES], data[TARGET],
                                        n_repeats=n_repeats, random_state=seed, scoring="r2")
    return pd.DataFrame({"feature": FEATURES, "importance_mean": importance.importances_mean,
                         "importance_std": importance.importances_std}).sort_values(
                             "importance_mean", ascending=False).reset_index(drop=True)


def shap_values_if_available(model, frame):
    """Return SHAP values when SHAP is installed, otherwise a clear None."""
    try:
        import shap
    except ImportError:
        return None
    data = frame[FEATURES].dropna()
    explainer = shap.Explainer(model.predict, data)
    return explainer(data)
