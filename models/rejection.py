"""Solute rejection and permeate concentration.

IMPORTANT: in this MVP, rejection is a required USER INPUT
(``Membrane.baseline_rejection_percent``), not something predicted from
membrane structure (pore size, MWCO, charge, hydrophilicity, etc.) or
from a mass-balance simulation. The standard definition

    R = (1 - Cp / Cf) * 100

is solved here in the opposite direction of how it is normally measured:
given a feed concentration and an assumed/measured rejection, we compute
the permeate concentration. See docs/equations.md for the full picture.
"""


def validate_rejection(r):
    """Raise if `r` (a percentage) is outside the physically valid 0-100 range."""
    if not 0 <= r <= 100:
        raise ValueError("Rejection must be between 0 and 100%.")
    return r


def permeate_concentration(feed_concentration, rejection_percent):
    """Permeate concentration implied by a feed concentration and rejection.

    Cp = Cf * (1 - R / 100)

    `rejection_percent` is taken as given (see module docstring) rather
    than predicted from membrane properties.
    """
    validate_rejection(rejection_percent)
    if feed_concentration < 0:
        raise ValueError("Feed concentration must be non-negative.")
    return feed_concentration * (1 - rejection_percent / 100.0)
