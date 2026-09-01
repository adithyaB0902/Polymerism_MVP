"""Save/load trained models so they persist across app restarts,
instead of being retrained from scratch every session.

Each call to `save_and_register_models` writes one joblib file per
target to `models_artifacts/` (git-ignored — these are build outputs,
not source) and records its metadata (feature list, metrics, version)
in the `trained_models` table via `db.repository`. Registering a new
version for a target automatically marks the previous one inactive
(see `db.repository.register_model`), so `load_active_model` always
returns the most recently trained model for that target.

This directly implements the previously-empty ml/model_registry.py stub.
"""

from pathlib import Path

import joblib

from db import repository as repo

DEFAULT_ARTIFACTS_DIR = Path(__file__).resolve().parent.parent / "models_artifacts"


def save_and_register_models(conn, models, metrics, feature_list, training_rows,
                             dataset_source, artifacts_dir=None):
    """Persist every model in `models` (dict of {target: fitted_estimator})
    to disk and register it in the database.

    `metrics` is the list of per-target metric dicts returned alongside
    `models` by `ml.train.train_models` (used to look up each target's
    chosen model-type name and store its metrics).

    Returns a list of the new `trained_models` row ids, one per target.
    """
    artifacts_dir = Path(artifacts_dir) if artifacts_dir else DEFAULT_ARTIFACTS_DIR
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    metrics_by_target = {m["target"]: m for m in metrics}

    ids = []
    for target, model in models.items():
        m = metrics_by_target.get(target, {})
        # Filename includes the next version number so repeated training
        # runs don't silently overwrite a previous artifact still
        # referenced by an inactive registry row (kept for audit history).
        existing_versions = [row["version"] for row in repo.list_models(conn, active_only=False)
                             if row["target_name"] == target]
        next_version = (max(existing_versions) + 1) if existing_versions else 1
        artifact_path = artifacts_dir / f"{target}_v{next_version}.joblib"
        joblib.dump(model, artifact_path)

        model_id = repo.register_model(
            conn, target_name=target, model_type=m.get("model", type(model).__name__),
            feature_list=feature_list,
            metrics={k: v for k, v in m.items() if k not in ("target", "model")},
            artifact_path=artifact_path, training_rows=training_rows, dataset_source=dataset_source,
        )
        ids.append(model_id)
    return ids


def load_active_model(conn, target_name, artifacts_dir=None):
    """Return (model, metadata_row) for the currently-active registered
    model for `target_name`, or (None, None) if none has been trained
    and registered yet."""
    row = repo.get_active_model_row(conn, target_name)
    if row is None:
        return None, None
    model = joblib.load(row["artifact_path"])
    return model, row


def load_all_active_models(conn):
    """Return {target_name: (model, metadata_row)} for every target
    that currently has an active registered model."""
    out = {}
    for row in repo.list_models(conn, active_only=True):
        model = joblib.load(row["artifact_path"])
        out[row["target_name"]] = (model, row)
    return out
