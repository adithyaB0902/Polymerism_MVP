"""One-at-a-time sensitivity analysis for the POLYMEMSIM screening model.

For each parameter of interest, this module re-runs the simulator with
that parameter scaled to 50%, 100% and 150% of its current value (all
other parameters held fixed) and records how much the feasibility score
moves. Parameters that swing the score the most are the ones most worth
pinning down with real measurements or literature values before relying
on the screening result.

This is a simple, transparent one-at-a-time (OAT) sweep, not a full
variance-based (e.g. Sobol) sensitivity analysis, and it says nothing
about interactions between parameters.
"""

import numpy as np
import pandas as pd

from models.membrane import Membrane, Water, OperatingConditions
from models.simulator import run_simulation

# The parameters swept, and which input object (membrane/water/operating
# conditions) each one lives on.
PARAMETERS = [
    "TMP_bar",
    "permeability_LMH_bar",
    "baseline_rejection_percent",
    "fouling_coefficient",
    "feed_concentration_mg_L",
    "temperature_C",
]

# Scaling factors applied to each parameter in turn: -50%, baseline, +50%.
SCALE_FACTORS = [0.5, 1.0, 1.5]


def sensitivity_analysis(m, w, o, t):
    """Sweep each parameter in PARAMETERS across SCALE_FACTORS and report
    how much the feasibility score changes.

    Returns a DataFrame with one row per parameter, sorted by
    `feasibility_range` (largest first) so the most influential
    parameters appear at the top.
    """
    base = run_simulation(m, w, o, t)["feasibility_score"]
    rows = []
    for p in PARAMETERS:
        vals = []
        for factor in SCALE_FACTORS:
            mm, ww, oo = m, w, o
            if p == "TMP_bar":
                # NOTE: every OperatingConditions field must be passed
                # through here by name — the dataclass field is
                # `crossflow_velocity_m_s`, not `crossflow_velocity`.
                oo = OperatingConditions(
                    o.TMP_bar * factor, o.feed_flow_L_min, o.crossflow_velocity_m_s,
                    o.recovery_percent, o.operating_time_hr, o.temperature_C,
                    o.membrane_area_m2, o.pump_efficiency, o.electricity_price_per_kwh,
                )
            elif p == "permeability_LMH_bar":
                mm = Membrane(**{**m.__dict__, "permeability_LMH_bar": m.permeability_LMH_bar * factor})
            elif p == "baseline_rejection_percent":
                mm = Membrane(**{**m.__dict__, "baseline_rejection_percent": min(100, m.baseline_rejection_percent * factor)})
            elif p == "fouling_coefficient":
                mm = Membrane(**{**m.__dict__, "fouling_coefficient": m.fouling_coefficient * factor})
            elif p == "feed_concentration_mg_L":
                ww = Water(**{**w.__dict__, "feed_concentration_mg_L": w.feed_concentration_mg_L * factor})
            elif p == "temperature_C":
                ww = Water(**{**w.__dict__, "temperature_C": w.temperature_C * factor})
            vals.append(run_simulation(mm, ww, oo, t)["feasibility_score"])
        rows.append({"parameter": p, "base_score": base, "feasibility_range": max(vals) - min(vals)})
    return pd.DataFrame(rows).sort_values("feasibility_range", ascending=False)
