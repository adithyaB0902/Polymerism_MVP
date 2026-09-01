"""Concentration polarization: why cross-flow velocity affects observed
rejection even when the membrane's intrinsic (true) rejection doesn't
change.

As permeate is drawn through the membrane, retained solute accumulates
in a thin boundary layer at the membrane surface faster than cross-flow
can sweep it away, raising the local wall concentration above the bulk
feed concentration. Film theory gives the classic relationship between
observed and intrinsic (real) rejection:

    R_obs = R_int / (R_int + (1 - R_int) * exp(Jv / k))

where Jv is the permeate flux and k is the mass-transfer coefficient of
the boundary layer. Higher cross-flow velocity increases k (better
mixing, thinner boundary layer), which pushes R_obs closer to R_int;
higher flux relative to k does the opposite. This is standard membrane
science (see e.g. Cheryan, "Ultrafiltration and Microfiltration
Handbook", or any RO/NF/UF transport-phenomena reference).

`k` itself depends on channel geometry, diffusivity, and flow regime in
ways this MVP has no data for, so it is approximated here with a
simple, clearly-illustrative power-law scaling in cross-flow velocity
(k ~ v^0.8, in the spirit of typical turbulent-flow Sherwood-number
correlations where Sh ~ Re^0.8), calibrated so that a "typical"
crossflow velocity of 0.2 m/s gives a "typical" mass-transfer
coefficient on the order of a few x 1e-5 m/s (consistent with published
UF/MF mass-transfer coefficients).
"""

import math

# Calibrated so that k(0.2 m/s) ~= 2e-5 m/s, a representative published
# order-of-magnitude value for UF/MF cross-flow mass-transfer
# coefficients. This is an illustrative scaling, not a fitted
# correlation for any specific membrane/module geometry.
_K_REFERENCE_VELOCITY_M_S = 0.2
_K_REFERENCE_VALUE_M_S = 2e-5
_VELOCITY_EXPONENT = 0.8


def mass_transfer_coefficient_m_s(crossflow_velocity_m_s):
    """Illustrative mass-transfer coefficient (m/s) as a function of
    cross-flow velocity; see module docstring for the scaling used."""
    if crossflow_velocity_m_s < 0:
        raise ValueError("Cross-flow velocity must be non-negative.")
    if crossflow_velocity_m_s == 0:
        # No cross-flow sweeping: treat as a very small but nonzero k
        # (pure dead-end filtration limit) rather than dividing by zero.
        crossflow_velocity_m_s = 1e-4
    return _K_REFERENCE_VALUE_M_S * (crossflow_velocity_m_s / _K_REFERENCE_VELOCITY_M_S) ** _VELOCITY_EXPONENT


def observed_rejection_percent(intrinsic_rejection_percent, flux_LMH, crossflow_velocity_m_s):
    """Apply the film-theory concentration-polarization correction to
    an intrinsic rejection value, given the current flux and cross-flow
    velocity. Returns the observed (lower, in general) rejection
    percent, clamped to [0, 100].
    """
    if not 0 <= intrinsic_rejection_percent <= 100:
        raise ValueError("Intrinsic rejection must be between 0 and 100%.")
    if flux_LMH < 0:
        raise ValueError("Flux must be non-negative.")

    R_int = intrinsic_rejection_percent / 100.0
    k = mass_transfer_coefficient_m_s(crossflow_velocity_m_s)
    Jv = flux_LMH / 3.6e6  # LMH -> m/s

    if R_int <= 0:
        return 0.0
    exponent = Jv / k
    # Guard against overflow for pathological (very high flux / very
    # low crossflow) inputs; the correction saturates at R_obs -> 0 in
    # that limit anyway.
    if exponent > 700:
        return 0.0
    R_obs = R_int / (R_int + (1 - R_int) * math.exp(exponent))
    return max(0.0, min(100.0, R_obs * 100.0))
