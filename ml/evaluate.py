"""Held-out and cross-validation evaluation for the ML models.

`ml.train.train_models` reports metrics from a single 80/20 train/test
split, which is fast but gives one noisy estimate of generalization
error — especially with the small experimental-data row counts a real
lab session might have. This module adds proper k-fold cross-validation
(mean +/- std across folds), which is a more robust indicator of how
well a model type is likely to generalize.

This directly implements the previously-empty ml/evaluate.py stub.
"""

import numpy as np
from sklearn.model_selection import KFold, cross_val_score
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, GradientBoostingRegressor
from sklearn.base import clone

from .train import FEATURES, TARGETS
from .features import engineer_features, available_engineered_columns


def cross_validate_target(df, target, feature_list=None, model=None, n_splits=5, seed=42):
    """K-fold cross-validate one target.

    `feature_list` defaults to `ml.train.FEATURES`; `model` defaults to
    a RandomForestRegressor with the same hyperparameters
    `ml.train.train_models` uses. Rows with a NaN value for `target`
    are dropped first (consistent with how `train_models` handles
    partially-labeled combined datasets).

    Returns {"target":.., "n_splits":.., "n_rows":.., "RMSE_mean":..,
    "RMSE_std":.., "R2_mean":.., "R2_std":..}.
    """
    feature_list = feature_list or FEATURES
    model = model if model is not None else RandomForestRegressor(n_estimators=150, random_state=seed, n_jobs=-1)

    work = df[df[target].notna()].copy()
    if len(work) < n_splits * 2:
        raise ValueError(f"Not enough labeled rows ({len(work)}) for {n_splits}-fold cross-validation "
                         f"on target '{target}' (need at least {n_splits * 2}).")

    X = work[feature_list].fillna(work[feature_list].median())
    y = work[target]

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    rmse_scores = -cross_val_score(clone(model), X, y, cv=kf, scoring="neg_root_mean_squared_error")
    r2_scores = cross_val_score(clone(model), X, y, cv=kf, scoring="r2")

    return {
        "target": target, "n_splits": n_splits, "n_rows": len(work),
        "RMSE_mean": float(np.mean(rmse_scores)), "RMSE_std": float(np.std(rmse_scores)),
        "R2_mean": float(np.mean(r2_scores)), "R2_std": float(np.std(r2_scores)),
    }


def cross_validate_all_targets(df, feature_list=None, model=None, n_splits=5, seed=42):
    """Run `cross_validate_target` for every target in `ml.train.TARGETS`
    that has enough labeled rows; targets with too few rows are skipped
    (reported in the returned `skipped` list) rather than raising."""
    results, skipped = [], []
    for target in TARGETS:
        try:
            results.append(cross_validate_target(df, target, feature_list, model, n_splits, seed))
        except ValueError as e:
            skipped.append({"target": target, "reason": str(e)})
    return results, skipped


def compare_model_types(df, target, feature_list=None, n_splits=5, seed=42):
    """Cross-validate all three candidate model types
    (`ml.train.train_models` picks the best of these on a single
    split) for one target, so their generalization can be compared more
    robustly than a single train/test split allows."""
    feature_list = feature_list or FEATURES
    candidates = {
        "Random Forest": RandomForestRegressor(n_estimators=150, random_state=seed, n_jobs=-1),
        "Extra Trees": ExtraTreesRegressor(n_estimators=150, random_state=seed, n_jobs=-1),
        "Gradient Boosting": GradientBoostingRegressor(random_state=seed),
    }
    return [dict(model_type=name, **cross_validate_target(df, target, feature_list, model, n_splits, seed))
           for name, model in candidates.items()]
