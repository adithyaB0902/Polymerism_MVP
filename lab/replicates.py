"""Simulated replicate measurements.

`models.simulator.run_simulation`/`run_detailed_simulation` are
deterministic — the same inputs always give the same outputs, which is
correct for a physics *model* but doesn't feel like a lab, where
repeated runs of the same nominal conditions always show some
measurement-to-measurement scatter (instrument noise, minor
sample-to-sample variation, etc).

This module layers *simulated* measurement noise on top of a
deterministic result to produce a set of "replicate runs" and their
summary statistics, purely for the lab-workflow experience. This is
explicitly labeled everywhere as injected/simulated variability, not a
model of any real error source — see docs/assumptions.md.
"""

import numpy as np

# Which result fields get replicate noise, and roughly how noisy a real
# instrument reading of that type tends to be (as a coefficient of
# variation, %) -- these are illustrative, typical-order-of-magnitude
# lab measurement-precision figures, not measured values.
DEFAULT_NOISE_CV_PERCENT = {
    "flux_LMH": 3.0,
    "rejection_percent": 1.0,
    "permeate_mg_L": 2.0,
    "energy_kWh_m3": 2.0,
    "cost_per_m3": 2.0,
    "feasibility_score": 2.0,
}


def simulate_replicates(base_result, n=3, noise_cv_percent=None, seed=None):
    """Generate `n` simulated replicate readings from one deterministic
    `base_result` dict, by applying independent Gaussian relative noise
    (mean 0, std = coefficient of variation) to each field in
    `noise_cv_percent` (defaults to DEFAULT_NOISE_CV_PERCENT).

    Rejection is clamped to [0, 100] after noise is applied, since it's
    a percentage.

    Returns a list of `n` dicts, each a copy of `base_result` with
    those fields perturbed.
    """
    if n < 1:
        raise ValueError("n must be at least 1.")
    noise_cv_percent = noise_cv_percent or DEFAULT_NOISE_CV_PERCENT
    rng = np.random.default_rng(seed)

    replicates = []
    for _ in range(n):
        rep = dict(base_result)
        for field, cv in noise_cv_percent.items():
            if field in rep and rep[field] is not None:
                noisy = rep[field] * (1 + rng.normal(0, cv / 100.0))
                if field == "rejection_percent":
                    noisy = max(0.0, min(100.0, noisy))
                elif field in ("flux_LMH", "permeate_mg_L", "energy_kWh_m3", "cost_per_m3"):
                    noisy = max(0.0, noisy)
                rep[field] = noisy
        replicates.append(rep)
    return replicates


def replicate_statistics(replicates, fields=None):
    """Summary stats (mean, std, CV%, n) for each field across a list
    of replicate result dicts (as produced by `simulate_replicates`, or
    real measured replicates recorded in the lab database)."""
    fields = fields or list(DEFAULT_NOISE_CV_PERCENT.keys())
    stats = {}
    for field in fields:
        values = [r[field] for r in replicates if r.get(field) is not None]
        if not values:
            continue
        mean = float(np.mean(values))
        std = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
        cv = (std / mean * 100.0) if mean else 0.0
        stats[field] = {"mean": mean, "std": std, "cv_percent": cv, "n": len(values)}
    return stats
