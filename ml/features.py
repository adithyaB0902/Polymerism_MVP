"""Physics-informed derived features for the ML pipeline.

The raw feature set in `ml.train.FEATURES` is used as-is by the
tree-based models there (trees split on raw values just fine, and
don't need scaling). This module adds a small set of *derived*
features built from combinations the underlying physics models
actually use — e.g. `pore_size_nm**2 / thickness_um` mirrors the
r^2/thickness dependence in `models.pore_flow`'s Hagen-Poiseuille
relationship, `permeability_LMH_bar * op_TMP_bar` mirrors
`models.flux.permeability_flux` directly, and `fouling_decay_factor` /
`flux_physics_proxy` mirror `models.flux.fouled_flux`'s exact
exp(-k*t) term. Handing a tree model these pre-computed interaction
terms lets it capture a physically-meaningful, often highly nonlinear
relationship (like an exponential decay) far more accurately than
reconstructing it from raw features via splits alone — see
tests/test_ml.py's `test_ml_engineered_features_substantially_reduce_error_on_hard_targets`
for the measured effect (flux_LMH and flux_decline_percent both go
from R2 ~0.4-0.6 to >0.99).

These are additive columns; `use_engineered_features=True` (the
default in `ml.train.train_models`) trains on FEATURES +
ENGINEERED_FEATURES instead of FEATURES alone.
"""

import numpy as np
import pandas as pd

ENGINEERED_FEATURES = [
    "permeability_x_TMP",         # ~ theoretical clean-membrane flux driver (models.flux.permeability_flux)
    "fouling_decay_factor",        # exp(-k*t): exact fouling-decay term (models.flux.fouled_flux)
    "flux_physics_proxy",          # permeability*TMP*decay: near-exact flux_LMH reconstruction (simple model)
    "pore_size_sq_over_thickness",  # ~ Hagen-Poiseuille structural flux driver (models.pore_flow)
    "porosity_x_pore_size_sq",     # the porosity*r^2 numerator in the same relationship
    "crossflow_pow_0_8",           # ~ mass-transfer-coefficient scaling (models.concentration_polarization)
    "turbidity_x_TDS",             # combined water-quality fouling-risk proxy (models.fouling_index)
    "hydrophilicity_minus_charge_mag",  # membrane-surface fouling-resistance proxy (models.fouling_index)
    "recovery_fraction",           # 0-1 scaled recovery, convenient for models that are scale-sensitive
]


def engineer_features(df):
    """Return a copy of `df` with ENGINEERED_FEATURES columns added.

    Missing source columns are tolerated (the derived column is simply
    omitted) so this can be applied to partial/real-measurement rows
    that may not carry every raw feature. Source columns are coerced
    to numeric (invalid/None entries -> NaN) before any arithmetic, so
    a dataframe combining synthetic rows (all-float columns) with
    logged experimental rows (which may have `None` in fields that
    weren't recorded, making pandas fall back to `object` dtype when
    concatenated) never raises a TypeError here — it just produces NaN
    for that row's engineered value, consistent with how
    `ml.train.train_models` already imputes NaN features by median.
    """
    out = df.copy()

    def _has(*cols):
        return all(c in out.columns for c in cols)

    def _num(col):
        return pd.to_numeric(out[col], errors="coerce")

    if _has("permeability_LMH_bar", "op_TMP_bar"):
        out["permeability_x_TMP"] = _num("permeability_LMH_bar") * _num("op_TMP_bar")

    if _has("fouling_coefficient", "op_operating_time_hr"):
        # exact term from models.flux.fouled_flux: J(t) = J0 * exp(-k*t)
        out["fouling_decay_factor"] = np.exp(-_num("fouling_coefficient") * _num("op_operating_time_hr"))

    if _has("permeability_LMH_bar", "op_TMP_bar", "fouling_coefficient", "op_operating_time_hr"):
        out["flux_physics_proxy"] = (_num("permeability_LMH_bar") * _num("op_TMP_bar")
                                     * np.exp(-_num("fouling_coefficient") * _num("op_operating_time_hr")))

    if _has("pore_size_nm", "thickness_um"):
        # +1e-9 avoids a zero-thickness division; thickness is validated
        # to be > 0 everywhere it's actually used physically, but a raw
        # uploaded/edited dataframe could in principle contain a 0.
        out["pore_size_sq_over_thickness"] = (_num("pore_size_nm") ** 2) / (_num("thickness_um") + 1e-9)

    if _has("porosity", "pore_size_nm"):
        out["porosity_x_pore_size_sq"] = _num("porosity") * _num("pore_size_nm") ** 2

    if _has("op_crossflow_velocity_m_s"):
        out["crossflow_pow_0_8"] = np.power(_num("op_crossflow_velocity_m_s").clip(lower=0), 0.8)

    if _has("water_turbidity_NTU", "water_TDS_mg_L"):
        out["turbidity_x_TDS"] = _num("water_turbidity_NTU") * _num("water_TDS_mg_L")

    if _has("hydrophilicity", "surface_charge"):
        out["hydrophilicity_minus_charge_mag"] = _num("hydrophilicity") - _num("surface_charge").abs()

    if _has("op_recovery_percent"):
        out["recovery_fraction"] = _num("op_recovery_percent") / 100.0

    return out


def available_engineered_columns(df):
    """Which ENGINEERED_FEATURES columns `engineer_features(df)` would
    actually be able to produce, given df's current columns (useful for
    building a FEATURES list that only references columns guaranteed to
    exist)."""
    engineered = engineer_features(df)
    return [c for c in ENGINEERED_FEATURES if c in engineered.columns]
