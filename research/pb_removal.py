"""Pb(II) removal calculations and validation."""

import numpy as np


def calculate_removal(c0, ce):
    """Return removal efficiency (%) from initial and equilibrium concentration."""
    c0, ce = float(c0), float(ce)
    if not np.isfinite(c0) or not np.isfinite(ce) or c0 <= 0:
        raise ValueError("C0 must be a finite positive concentration.")
    if ce < 0 or ce > c0:
        raise ValueError("Ce must be between zero and C0 for a removal experiment.")
    return (c0 - ce) / c0 * 100.0


def calculate_qe(c0, ce, volume_l, membrane_mass_g):
    """Return adsorption capacity (mg/g), qe=((C0-Ce)V)/m."""
    c0, ce, volume_l, membrane_mass_g = map(float, (c0, ce, volume_l, membrane_mass_g))
    if volume_l <= 0 or membrane_mass_g <= 0:
        raise ValueError("Solution volume and membrane mass must be positive.")
    calculate_removal(c0, ce)
    return (c0 - ce) * volume_l / membrane_mass_g


def calculate_pb_metrics(c0, ce, volume_l, membrane_mass_g):
    return {
        "removal_percent": calculate_removal(c0, ce),
        "qe_mg_g": calculate_qe(c0, ce, volume_l, membrane_mass_g),
    }


# Short aliases used in notebooks and paper analysis scripts.
removal_efficiency = calculate_removal
adsorption_capacity = calculate_qe
