import numpy as np
import pandas as pd

from simulation.generate_dataset import generate_virtual_experiments
from ml.train import train_models, feature_list_for, FEATURES, TARGETS


def test_ml():
    df = generate_virtual_experiments(80)
    models, metrics, feature_list = train_models(df)
    assert len(models) == 6 and len(metrics) == 6
    assert feature_list == feature_list_for(df, use_engineered_features=True)  # engineered is now the default


def test_ml_raw_features_still_available_via_flag():
    df = generate_virtual_experiments(80)
    models, metrics, feature_list = train_models(df, use_engineered_features=False)
    assert feature_list == FEATURES


def test_ml_with_engineered_features():
    df = generate_virtual_experiments(120, seed=2)
    models, metrics, feature_list = train_models(df, use_engineered_features=True)
    assert len(feature_list) > len(FEATURES)
    for target in TARGETS:
        assert target in models
        assert models[target].n_features_in_ == len(feature_list)


def test_ml_handles_partially_labeled_rows():
    df = generate_virtual_experiments(100, seed=3)
    # simulate a combined dataset where some rows (as if experimentally
    # measured) are missing several of the derived targets
    df = df.copy()
    df.loc[df.index[:20], "feasibility_score"] = np.nan
    models, metrics, _ = train_models(df)
    fs_metric = next(m for m in metrics if m["target"] == "feasibility_score")
    assert fs_metric["n_train"] + fs_metric["n_test"] == 80  # 100 - 20 NaN rows


def test_ml_engineered_features_substantially_reduce_error_on_hard_targets():
    """flux_LMH and flux_decline_percent depend on an exponential
    fouling-decay term (exp(-k*t)) that raw features alone leave the
    tree ensembles struggling to approximate (R2 ~0.4-0.6). Physics-
    informed engineered features (see ml/features.py) that hand the
    model the decay term directly should push both above R2=0.95 --
    this locks that improvement in as a regression test."""
    df = generate_virtual_experiments(500, seed=42)
    models, metrics, feature_list = train_models(df, use_engineered_features=True)
    by_target = {m["target"]: m for m in metrics}
    assert by_target["flux_LMH"]["R2"] > 0.95
    assert by_target["flux_decline_percent"]["R2"] > 0.95
    assert by_target["feasibility_score"]["R2"] > 0.95


def test_ml_raises_clear_error_with_too_few_labeled_rows():
    df = generate_virtual_experiments(100, seed=4)
    df = df.copy()
    df.loc[df.index[5:], "flux_LMH"] = np.nan  # only 5 labeled rows left
    try:
        train_models(df)
        assert False, "expected a ValueError for too few labeled rows"
    except ValueError as e:
        assert "flux_LMH" in str(e)
