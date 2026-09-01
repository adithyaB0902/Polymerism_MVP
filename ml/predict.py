"""Predict simulator-style outputs for one new candidate using whatever
models are currently registered (see ml.model_registry), without going
through the physics simulator at all.

This directly implements the previously-empty ml/predict.py stub. It
is deliberately tolerant of incomplete input: if a candidate is missing
a feature some registered model needs, that target's prediction is
skipped (never silently guessed) unless a reference dataset is
supplied to impute a per-feature median from — the same strategy
`ml.train.train_models` uses for training-time NaNs.
"""

import numpy as np
import pandas as pd

from ml.model_registry import load_all_active_models
from ml.features import engineer_features
from ml.ood import OODDetector
from ml.train import TARGETS


def predict_candidate(conn, candidate_features, reference_df=None):
    """Predict every target with a currently-registered active model.

    `candidate_features` is a dict of raw feature values (the same
    names as `ml.train.FEATURES`, e.g. `{"thickness_um": 100.0,
    "permeability_LMH_bar": 25.0, "water_pH": 7.0, ...}`); it does not
    need to include engineered features — they're derived automatically
    via `ml.features.engineer_features` if a registered model needs them.

    `reference_df`, if supplied, is used two ways: (1) to impute any
    feature a registered model needs but `candidate_features` didn't
    supply, using that feature's median in `reference_df`, and (2) to
    run a per-feature out-of-distribution check (`ml.ood.OODDetector`)
    on the candidate.

    Returns:
      {"predictions": {target: value}, "skipped": {target: reason},
       "ood": OODDetector.check(...) result or None}
    """
    active_models = load_all_active_models(conn)
    if not active_models:
        return {"predictions": {}, "skipped": {t: "no trained model registered" for t in TARGETS}, "ood": None}

    cand_df = pd.DataFrame([candidate_features])
    cand_engineered = engineer_features(cand_df).iloc[0]
    # A registered model's feature_list may include engineered columns
    # (e.g. flux_physics_proxy) that a raw historical dataframe -- like
    # the one returned by generate_virtual_experiments/
    # assemble_training_dataset -- doesn't carry yet. Engineer the
    # reference frame too so imputation/OOD checks can find those
    # columns, not just the ones present on the candidate itself.
    reference_df = engineer_features(reference_df) if reference_df is not None else None

    predictions, skipped = {}, {}
    for target, (model, meta) in active_models.items():
        feature_list = meta["feature_list"]
        row = {}
        missing_feature = None
        for f in feature_list:
            if f in cand_engineered.index and pd.notna(cand_engineered[f]):
                row[f] = cand_engineered[f]
            elif reference_df is not None and f in reference_df.columns:
                row[f] = reference_df[f].median()
            else:
                missing_feature = f
                break
        if missing_feature is not None:
            skipped[target] = (f"missing feature '{missing_feature}' and no reference dataset "
                               "available to impute a fallback value from")
            continue
        X = pd.DataFrame([row])[feature_list]
        predictions[target] = float(model.predict(X)[0])

    ood_result = None
    if reference_df is not None and active_models:
        any_feature_list = next(iter(active_models.values()))[1]["feature_list"]
        usable = [f for f in any_feature_list if f in reference_df.columns and f in cand_engineered.index]
        if usable:
            ood_result = OODDetector(reference_df, usable).check(cand_engineered)

    return {"predictions": predictions, "skipped": skipped, "ood": ood_result}
