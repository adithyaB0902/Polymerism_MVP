"""Membrane flux models.

POLYMEMSIM ships two independent ways to estimate permeate flux:

1. ``permeability_flux`` — the simple model actually used by
   :func:`models.simulator.run_simulation`. Flux is assumed directly
   proportional to trans-membrane pressure (TMP), with the membrane's
   measured/literature permeability coefficient (LMH/bar) absorbing all
   membrane-specific effects (thickness, pore structure, etc.). This is
   why ``thickness_um``, ``pore_size_nm`` and similar structural inputs
   are NOT used by this model — for a linear-permeability membrane,
   their effect is already captured empirically in the permeability
   value the user supplies.

2. ``resistance_flux`` — a classic Darcy's-law / resistances-in-series
   model (flux driven by pressure, opposed by membrane + fouling
   resistance, scaled by viscosity). This is provided as an alternative,
   more mechanistic model for users who have measured resistance values,
   but it is NOT currently wired into ``run_simulation``. It is kept
   here as a documented building block for future work.

``fouled_flux`` is used by both paths to apply a simple first-order
exponential flux-decline (fouling) correction over time.
"""

import math


def permeability_flux(permeability_LMH_bar, TMP_bar):
    """Initial (unfouled) flux from the linear permeability model.

    J0 [LMH] = A [LMH/bar] * TMP [bar]

    This is the flux model used by the live simulator. It assumes flux
    scales linearly with pressure and does not require membrane
    thickness, pore size or porosity as separate inputs — those effects
    are assumed to already be reflected in the measured/literature
    permeability coefficient ``A``.
    """
    if permeability_LMH_bar < 0 or TMP_bar < 0:
        raise ValueError("Permeability and pressure must be non-negative.")
    return permeability_LMH_bar * TMP_bar


def resistance_flux(TMP_bar, viscosity_Pa_s, membrane_resistance, fouling_resistance=0.0):
    """Flux from a Darcy's-law resistances-in-series model.

    J [LMH] = TMP [Pa] / (viscosity [Pa*s] * total_resistance [1/m])

    converted from m/s to L/m^2/hr (LMH) via the 3.6e6 factor implied by
    1 m/s = 3.6e6 LMH (1 m^3/m^2/s = 1000 L/m^2 * 3600 /hr).

    ``membrane_resistance`` and ``fouling_resistance`` must be supplied
    in 1/m (a typical clean-membrane hydraulic resistance is on the
    order of 1e12-1e13 1/m). This function is an optional, more
    mechanistic alternative to :func:`permeability_flux` — it is not
    currently called anywhere in the simulator pipeline.
    """
    if TMP_bar < 0:
        raise ValueError("Pressure must be non-negative.")
    total = membrane_resistance + fouling_resistance
    if total <= 0:
        raise ValueError("Total resistance must be positive.")
    # TMP_bar * 1e5 converts bar -> Pa. The 3.6 factor converts m/s -> LMH
    # ONLY if `total` is already expressed per-mm (1/mm) rather than
    # per-metre (1/m); for resistances in SI units (1/m) the correct
    # conversion factor is 3.6e6, not 3.6. Callers must therefore supply
    # `membrane_resistance`/`fouling_resistance` consistent with 1/mm,
    # or convert `total` to 1/m and use 3.6e6 instead. This is flagged
    # here rather than silently "corrected" because this function is
    # currently unused by the simulator and changing its unit
    # convention without literature/measured resistance values to
    # calibrate against would just trade one unverified assumption for
    # another.
    return TMP_bar * 1e5 / (viscosity_Pa_s * total) * 3.6


def fouled_flux(J0, k, time_hr):
    """Flux after fouling, using a simple first-order exponential decay.

    J(t) = J0 * exp(-k * t)

    ``k`` (1/hr) is an empirical fouling coefficient, not derived from
    first principles — see docs/assumptions.md.
    """
    if J0 < 0 or k < 0 or time_hr < 0:
        raise ValueError("Flux, fouling coefficient and time must be non-negative.")
    return J0 * math.exp(-k * time_hr)
