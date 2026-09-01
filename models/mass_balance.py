"""Simple recovery-based mass balance.

`models.energy.hydraulic_energy_kwh_m3` deliberately computes energy
per m3 of *feed pumped* (see that module's docstring — it cancels out
flow rate entirely). In a real system the feed pump pressurizes the
whole feed stream, but only a fraction of it (the recovery ratio)
actually becomes permeate; the rest exits as concentrate. So the
energy cost *per m3 of permeate produced* — the number people usually
mean by "specific energy consumption" in desalination/UF/RO — is the
feed-basis energy divided by the recovery fraction:

    SEC_permeate = SEC_feed / recovery_fraction

A simple 1-stage overall mass balance on the retained solute also gives
the concentrate concentration:

    Feed = Permeate + Concentrate  (volumetric, at steady state)
    Cf * Qf = Cp * Qp + Cc * Qc

Given recovery r = Qp/Qf (so Qc = Qf*(1-r)) and permeate concentration
Cp (already computed from rejection), solving for Cc:

    Cc = (Cf - r*Cp) / (1 - r)

This is a coarse, single-stage, fully-mixed approximation — it ignores
axial concentration build-up along a real membrane module, but is a
standard order-of-magnitude "back of the envelope" mass balance.
"""


def permeate_normalized_energy_kwh_m3(feed_basis_energy_kwh_m3, recovery_percent):
    """Convert a feed-pumping-basis specific energy figure into a
    per-m3-of-permeate figure using the recovery ratio."""
    if feed_basis_energy_kwh_m3 < 0:
        raise ValueError("Energy must be non-negative.")
    if not 0 < recovery_percent <= 100:
        raise ValueError("Recovery must be between 0 (exclusive) and 100%.")
    return feed_basis_energy_kwh_m3 / (recovery_percent / 100.0)


def concentrate_concentration(feed_concentration_mg_L, permeate_concentration_mg_L, recovery_percent):
    """Concentrate-side concentration implied by a single-stage,
    fully-mixed overall mass balance (see module docstring)."""
    if feed_concentration_mg_L < 0 or permeate_concentration_mg_L < 0:
        raise ValueError("Concentrations must be non-negative.")
    if not 0 <= recovery_percent < 100:
        raise ValueError("Recovery must be between 0 and 100% (exclusive of 100, else there is no concentrate stream).")
    r = recovery_percent / 100.0
    return (feed_concentration_mg_L - r * permeate_concentration_mg_L) / (1 - r)


def total_permeate_flow_L_hr(flux_LMH, membrane_area_m2):
    """Absolute permeate production rate implied by a flux and a
    membrane area (flux is already an intensive, per-m2 quantity)."""
    if flux_LMH < 0 or membrane_area_m2 < 0:
        raise ValueError("Flux and membrane area must be non-negative.")
    return flux_LMH * membrane_area_m2
