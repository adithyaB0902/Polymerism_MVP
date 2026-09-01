"""Explainability for the ML models: which features they actually rely on.

Two complementary views, both using only scikit-learn (no extra
dependency like SHAP):

- `feature_importances`: each tree model's built-in impurity-based
  `feature_importances_`. Fast, but can be misleadingly high for
  high-cardinality or correlated features.
- `permutation_importances`: model-agnostic — shuffle one feature
  column at a time on held-out data and measure how much a metric
  (default R2) degrades. More reliable than impurity-based importance,
  at the cost of being slower (`n_repeats` re-evaluations per feature).

This is exactly the tool referenced in ml/train.py's docstring to
check whether a trained model actually ignores the physics-model inputs
it was given but that the generating model itself never used — e.g. a
model trained on the simple-physics dataset should show ~zero
importance for `pore_size_nm`, since `run_simulation` never reads it.

This directly implements the previously-empty ml/explain.py stub.
"""

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance


def feature_importances(model, feature_list):
    """Sorted (most important first) list of {"feature":.., "importance":..}
    from the model's own built-in `feature_importances_`. Raises
    AttributeError if the model type doesn't expose one."""
    if not hasattr(model, "feature_importances_"):
        raise AttributeError(f"{type(model).__name__} has no built-in feature_importances_.")
    importances = model.feature_importances_
    pairs = sorted(zip(feature_list, importances), key=lambda p: -p[1])
    return [{"feature": f, "importance": float(v)} for f, v in pairs]


def permutation_importances(model, X_test, y_test, feature_list, n_repeats=10, seed=42, scoring="r2"):
    """Sorted (most important first) list of {"feature":..,
    "importance_mean":.., "importance_std":..} from scikit-learn's
    permutation importance on held-out (X_test, y_test)."""
    X_test = pd.DataFrame(X_test, columns=feature_list) if not isinstance(X_test, pd.DataFrame) else X_test
    result = permutation_importance(model, X_test, y_test, n_repeats=n_repeats,
                                    random_state=seed, scoring=scoring, n_jobs=-1)
    pairs = sorted(zip(feature_list, result.importances_mean, result.importances_std), key=lambda p: -p[1])
    return [{"feature": f, "importance_mean": float(m), "importance_std": float(s)} for f, m, s in pairs]


def used_vs_unused_features(model, feature_list, relative_threshold=0.1):
    """Split `feature_list` into "used" and "effectively unused", based
    on the model's built-in feature_importances_.

    A feature counts as "used" if its importance is at least
    `relative_threshold` times the single most important feature's
    importance (default: at least 10% as important as the top
    feature). A *relative* threshold is used rather than a fixed
    absolute cutoff because impurity-based importance never reaches
    exactly zero for genuinely irrelevant features — with many
    features, the small residual "noise" importance left over after
    the real signal is accounted for gets spread across all of them,
    so an absolute cutoff like "0.01" can land right in the middle of
    that noise band and misclassify features depending only on how many
    total features there are.
    """
    ranked = feature_importances(model, feature_list)
    max_importance = ranked[0]["importance"] if ranked else 0.0
    cutoff = relative_threshold * max_importance
    used = [r["feature"] for r in ranked if r["importance"] > cutoff]
    unused = [r["feature"] for r in ranked if r["importance"] <= cutoff]
    return {"used": used, "unused": unused, "ranked": ranked}
