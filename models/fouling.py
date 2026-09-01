"""Empirical flux decline (fouling) over an operating run.

Wraps :func:`models.flux.fouled_flux` and expresses the result both as
an absolute flux and as a percent decline relative to the initial
(unfouled) flux, which is the form the feasibility scorer and UI use.
"""

from .flux import fouled_flux


def flux_decline_percent(J0, k, time_hr):
    """Return (flux_at_time_hr, percent_decline_from_J0).

    decline% = (1 - J(t) / J0) * 100

    If J0 <= 0 (e.g. zero pressure/permeability), decline is reported as
    0% rather than dividing by zero, since there is no initial flux to
    decline from.
    """
    Jt = fouled_flux(J0, k, time_hr)
    decline = (1 - Jt / J0) * 100 if J0 > 0 else 0.0
    return Jt, decline
