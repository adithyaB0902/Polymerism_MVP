"""Nested, group-aware model comparison for the paper protocol."""

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import RandomForestRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, RBF, WhiteKernel
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold, GridSearchCV, ParameterGrid
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

from .ml import FEATURES, TARGET
from .rsm.quadratic_model import fit_quadratic, predict_quadratic


def _candidates(seed):
    models = {
        "Multiple Linear Regression": (LinearRegression(), {}),
        "SVR": (make_pipeline(StandardScaler(), SVR(kernel="rbf")), {
            "svr__C": np.logspace(0, 3, 4), "svr__epsilon": [0.01, 0.1, 1.0],
            "svr__gamma": np.logspace(-3, 1, 4)}),
        "Gaussian Process Regression": (make_pipeline(
            StandardScaler(), GaussianProcessRegressor(
                kernel=ConstantKernel(1.0) * RBF([1.0] * len(FEATURES)) + WhiteKernel(),
                normalize_y=True, random_state=seed)), {}),
        "Random Forest": (RandomForestRegressor(random_state=seed, n_jobs=-1), {
            "n_estimators": [100, 250, 500], "max_depth": [2, 5, 8],
            "min_samples_leaf": [1, 2, 3], "max_features": [0.5, 1.0]}),
        "Shallow MLP": (make_pipeline(StandardScaler(), MLPRegressor(
            activation="tanh", solver="lbfgs", max_iter=2000, random_state=seed)), {
            "mlpregressor__hidden_layer_sizes": [(4,), (6,), (8,)],
            "mlpregressor__alpha": [1e-4, 1e-2, 1e-1]}),
    }
    try:
        from xgboost import XGBRegressor
        models["XGBoost"] = (XGBRegressor(objective="reg:squarederror", random_state=seed, n_jobs=1), {
            "n_estimators": [50, 150, 300], "learning_rate": [0.01, 0.1, 0.3],
            "max_depth": [2, 3, 4], "subsample": [0.6, 0.8, 1.0],
            "reg_lambda": [0.0, 5.0, 10.0]})
    except ImportError:
        pass
    return models


def _metrics(y, prediction):
    error = np.asarray(y) - np.asarray(prediction)
    return {"R2": r2_score(y, prediction), "RMSE": np.sqrt(mean_squared_error(y, prediction)),
            "MAE": mean_absolute_error(y, prediction),
            "AARD": np.mean(np.abs(error) / np.maximum(np.abs(y), 1e-12)) * 100}


def nested_compare(frame, groups=None, n_splits=5, n_repeats=10, seed=42, max_grid_points=12):
    """Compare six models with repeated outer folds and inner tuning.

    ``groups`` keeps replicated conditions together. For small studies where
    five groups are unavailable, a clear ValueError is raised rather than
    silently leaking replicates across folds.
    """
    data = frame[FEATURES + [TARGET]].apply(pd.to_numeric, errors="coerce").dropna()
    if len(data) < n_splits:
        raise ValueError(f"At least {n_splits} complete rows are required.")
    if groups is None:
        groups = np.arange(len(data))
    groups = np.asarray(groups)[data.index]
    if len(np.unique(groups)) < n_splits:
        raise ValueError("At least n_splits replicate/condition groups are required.")
    x, y = data[FEATURES], data[TARGET]
    unique_groups = np.unique(groups)
    repeat_splits = []
    for repeat in range(n_repeats):
        shuffled = np.random.default_rng(seed + repeat).permutation(unique_groups)
        group_folds = np.array_split(shuffled, n_splits)
        for fold_groups in group_folds:
            test_mask = np.isin(groups, fold_groups)
            train_idx = np.flatnonzero(~test_mask)
            test_idx = np.flatnonzero(test_mask)
            repeat_splits.append((train_idx, test_idx))
    results, predictions = [], []
    for name, (estimator, grid) in _candidates(seed).items():
        fold_metrics = []
        for fold, (train_idx, test_idx) in enumerate(repeat_splits):
            tuned = estimator
            if grid:
                combinations = list(ParameterGrid(grid))[:max_grid_points]
                search_grid = [{key: [value] for key, value in combination.items()}
                               for combination in combinations]
                tuned = GridSearchCV(estimator, search_grid,
                                     cv=GroupKFold(min(3, len(np.unique(groups[train_idx])))),
                                     scoring="neg_root_mean_squared_error", n_jobs=-1)
                tuned.fit(x.iloc[train_idx], y.iloc[train_idx], groups=groups[train_idx])
            else:
                tuned = clone(tuned).fit(x.iloc[train_idx], y.iloc[train_idx])
            pred = tuned.predict(x.iloc[test_idx])
            fold_metrics.append(_metrics(y.iloc[test_idx], pred))
            predictions.append({"model": name, "fold": fold, "y": y.iloc[test_idx].to_numpy(),
                                "prediction": pred})
        results.append({"model": name, **{metric: float(np.mean([m[metric] for m in fold_metrics]))
                        for metric in ("R2", "RMSE", "MAE", "AARD")},
                        **{f"{metric}_SD": float(np.std([m[metric] for m in fold_metrics], ddof=1))
                           for metric in ("R2", "RMSE", "MAE", "AARD")}})
    rsm_metrics = []
    for train_idx, test_idx in repeat_splits:
        if len(train_idx) <= 15:
            continue
        fit = fit_quadratic(data.iloc[train_idx], TARGET, FEATURES)
        prediction = predict_quadratic(fit, data.iloc[test_idx])
        rsm_metrics.append(_metrics(y.iloc[test_idx], prediction))
        predictions.append({"model": "RSM quadratic", "fold": len(predictions),
                            "y": y.iloc[test_idx].to_numpy(), "prediction": prediction})
    if rsm_metrics:
        results.append({"model": "RSM quadratic",
                        **{metric: float(np.mean([m[metric] for m in rsm_metrics]))
                           for metric in ("R2", "RMSE", "MAE", "AARD")},
                        **{f"{metric}_SD": float(np.std([m[metric] for m in rsm_metrics], ddof=1))
                           for metric in ("R2", "RMSE", "MAE", "AARD")}})
    return pd.DataFrame(results).sort_values("RMSE").reset_index(drop=True), pd.DataFrame(predictions)
