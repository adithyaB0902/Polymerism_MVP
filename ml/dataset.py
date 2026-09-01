"""Assemble a training dataset for ml.train, optionally combining
synthetic physics-generated rows with real measured experiments logged
in the lab database.

This directly implements the previously-empty ml/dataset.py stub. It
is intentionally a thin assembly layer — no caching or persisted
"dataset snapshot" versioning (there's no dedicated table for that in
db/schema.py; every call regenerates the synthetic part fresh from the
given seed, which is already fast and deterministic).
"""

import pandas as pd

from simulation.generate_dataset import generate_virtual_experiments
from ml.train import FEATURES, TARGETS
from db import repository as repo

# Columns an experiments-table row (joined to its sample) can supply
# directly under the same names ml.train.FEATURES/TARGETS expect.
_EXPERIMENT_TARGET_MAP = {
    "flux_LMH": "flux_LMH",
    "rejection_percent": "rejection_percent",
    "flux_decline_percent": "flux_decline_percent",
    "permeate_mg_L": "permeate_mg_L",
    "energy_kWh_m3": "energy_kWh_m3",
    "feasibility_score": "feasibility_score",
}


def _experiment_rows_to_training_frame(rows):
    """Convert `db.repository.list_experiments_with_sample_fields`
    output into a dataframe with the same column names as
    `simulation.generate_dataset.generate_virtual_experiments`, so it
    can be concatenated directly. Any FEATURES/TARGETS column not
    present in a given row is left as NaN (train_models already drops
    NaN-target rows per-target and imputes NaN features by median)."""
    records = []
    for row in rows:
        rec = {f: row.get(f) for f in FEATURES if f in row}
        # experiments table already stores water_/op_ prefixed columns
        # matching FEATURES' own prefixed names, so the dict comprehension
        # above already picks those up via `row.get(f)`. Only the plain
        # (unprefixed) membrane columns need no renaming either, since
        # list_experiments_with_sample_fields aliases them unprefixed.
        for target in TARGETS:
            rec[target] = row.get(target)
        rec["dataset_type"] = "experimental_measured"
        records.append(rec)
    return pd.DataFrame.from_records(records)


def assemble_training_dataset(conn=None, n_synthetic=500, seed=42, include_experimental=True):
    """Build a combined training dataframe.

    Returns (df, summary) where summary is
    {"synthetic_rows": int, "experimental_rows": int, "total_rows": int}.

    If `conn` is None or `include_experimental` is False, this is
    equivalent to plain `generate_virtual_experiments(n_synthetic, seed)`.
    """
    synthetic = generate_virtual_experiments(n_synthetic, seed=seed)
    frames = [synthetic]
    n_experimental = 0

    if include_experimental and conn is not None:
        rows = repo.list_experiments_with_sample_fields(conn, data_source="measured")
        if rows:
            experimental = _experiment_rows_to_training_frame(rows)
            # Only keep experimental rows that supply at least one target
            # column with a real value -- an all-NaN-target row can't
            # help train anything and would just be dead weight.
            has_any_target = experimental[TARGETS].notna().any(axis=1)
            experimental = experimental[has_any_target]
            n_experimental = len(experimental)
            if n_experimental:
                frames.append(experimental)

    combined = pd.concat(frames, ignore_index=True, sort=False)
    summary = {"synthetic_rows": len(synthetic), "experimental_rows": n_experimental,
              "total_rows": len(combined)}
    return combined, summary
