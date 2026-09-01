"""Ties the individual physics models together into one screening result.

This module offers two entry points:

- `run_simulation` — the original simple screening model. Only
  permeability, TMP, rejection, fouling coefficient, feed
  concentration, operating time, pump efficiency and electricity price
  drive the result; several collected inputs (thickness, pore size,
  porosity, crossflow velocity, recovery, temperature, membrane area,
  etc.) are not used. Kept exactly as-is for backward compatibility —
  existing tests and the synthetic dataset generator both depend on its
  exact numeric behavior.

- `run_detailed_simulation` — an extended model that additionally uses
  every one of those previously-unused inputs, via the standalone
  physics modules in this package: temperature-corrected viscosity
  (models.viscosity), a structural pore-flow flux cross-check
  (models.pore_flow), concentration-polarization-corrected rejection
  (models.concentration_polarization), a recovery-based mass balance
  for concentrate concentration and permeate-normalized energy
  (models.mass_balance), and a water-quality/membrane-surface fouling
  adjustment (models.fouling_index). See each module's own docstring
  for exactly which relationships are standard textbook physics versus
  explicitly-flagged illustrative heuristics.

See models/membrane.py for the authoritative field-by-field list of
which inputs each model actually uses, and docs/assumptions.md /
docs/equations.md for the full picture.
"""

from .flux import permeability_flux
from .rejection import permeate_concentration
from .fouling import flux_decline_percent
from .energy import hydraulic_energy_kwh_m3
from .economics import screening_cost_per_m3
from .feasibility import feasibility_score
from .viscosity import water_viscosity_Pa_s, temperature_correction_factor
from .pore_flow import pore_flow_flux_LMH, DEFAULT_TORTUOSITY
from .concentration_polarization import observed_rejection_percent
from .mass_balance import concentrate_concentration, permeate_normalized_energy_kwh_m3, total_permeate_flow_L_hr
from .fouling_index import adjusted_fouling_coefficient


def run_simulation(membrane, water, op, targets):
    """Run one full screening simulation and return a dict of results.

    Raises ValueError if any input is out of its physically valid range
    (negative values, pH outside 0-14, recovery over 100%).
    """
    vals = [membrane.thickness_um, membrane.permeability_LMH_bar, membrane.baseline_rejection_percent,
            membrane.fouling_coefficient, water.feed_concentration_mg_L, water.pH,
            op.TMP_bar, op.feed_flow_L_min]
    if any(v < 0 for v in vals):
        raise ValueError("Inputs cannot be negative.")
    if not 0 <= water.pH <= 14: raise ValueError("pH must be between 0 and 14.")
    if op.recovery_percent > 100: raise ValueError("Recovery must be <= 100%.")

    J0 = permeability_flux(membrane.permeability_LMH_bar, op.TMP_bar)
    Jt, decline = flux_decline_percent(J0, membrane.fouling_coefficient, op.operating_time_hr)
    rejection = max(0, min(100, membrane.baseline_rejection_percent))
    cp = permeate_concentration(water.feed_concentration_mg_L, rejection)
    energy = hydraulic_energy_kwh_m3(op.TMP_bar, op.feed_flow_L_min, op.pump_efficiency)
    # NOTE: cost currently reflects electricity only (membrane_cost,
    # cleaning_cost and maintenance_cost all default to 0 here — see
    # models/economics.py docstring).
    cost = screening_cost_per_m3(op.electricity_price_per_kwh, energy)
    score, subs = feasibility_score(rejection, Jt, decline, energy, cost, targets)

    return {
        "polymer_name": membrane.polymer_name, "membrane_type": membrane.membrane_type,
        "flux_LMH": Jt, "initial_flux_LMH": J0, "rejection_percent": rejection,
        "permeate_mg_L": cp, "flux_decline_percent": decline,
        "energy_kWh_m3": energy, "cost_per_m3": cost,
        "feasibility_score": score, "flux_model": "Permeability model: J = A × TMP",
        "rejection_model": "User-provided baseline rejection",
        "scientific_note": "Screening-level approximation; requires experimental validation.",
        "physics_model": "simple",
        **{f"{k}_score":v for k,v in subs.items()}
    }


def run_detailed_simulation(membrane, water, op, targets, reference_temperature_C=25.0, tortuosity=None):
    """Extended screening simulation using the full detailed-physics
    stack (see module docstring). Returns the same result keys as
    `run_simulation` PLUS: `intrinsic_rejection_percent` (the
    pre-concentration-polarization value), `structural_flux_LMH` (the
    independent pore-flow cross-check, or None if porosity/pore size
    weren't supplied), `concentrate_mg_L` (None at exactly 100%
    recovery, since there is then no concentrate stream), and
    `total_permeate_flow_L_hr`.

    Raises the same ValueErrors as `run_simulation` for out-of-range
    inputs, plus propagates ValueErrors from the underlying detailed
    models (e.g. an invalid water temperature for the viscosity
    correlation).
    """
    vals = [membrane.thickness_um, membrane.permeability_LMH_bar, membrane.baseline_rejection_percent,
            membrane.fouling_coefficient, water.feed_concentration_mg_L, water.pH,
            op.TMP_bar, op.feed_flow_L_min]
    if any(v < 0 for v in vals):
        raise ValueError("Inputs cannot be negative.")
    if not 0 <= water.pH <= 14: raise ValueError("pH must be between 0 and 14.")
    if op.recovery_percent > 100: raise ValueError("Recovery must be <= 100%.")

    mu_actual = water_viscosity_Pa_s(water.temperature_C)
    temp_factor = temperature_correction_factor(water.temperature_C, reference_temperature_C)
    J0 = permeability_flux(membrane.permeability_LMH_bar, op.TMP_bar) * temp_factor

    structural_flux = None
    if membrane.porosity > 0 and membrane.pore_size_nm > 0:
        structural_flux = pore_flow_flux_LMH(
            membrane.porosity, membrane.pore_size_nm, membrane.thickness_um, mu_actual, op.TMP_bar,
            tortuosity=tortuosity if tortuosity is not None else DEFAULT_TORTUOSITY)

    k_adjusted = adjusted_fouling_coefficient(
        membrane.fouling_coefficient, water.turbidity_NTU, water.TDS_mg_L,
        membrane.hydrophilicity, membrane.surface_charge)
    Jt, decline = flux_decline_percent(J0, k_adjusted, op.operating_time_hr)

    intrinsic_rejection = max(0, min(100, membrane.baseline_rejection_percent))
    observed_rejection = observed_rejection_percent(intrinsic_rejection, Jt, op.crossflow_velocity_m_s)
    cp = permeate_concentration(water.feed_concentration_mg_L, observed_rejection)

    concentrate = None
    if op.recovery_percent < 100:
        concentrate = concentrate_concentration(water.feed_concentration_mg_L, cp, op.recovery_percent)

    energy_feed_basis = hydraulic_energy_kwh_m3(op.TMP_bar, op.feed_flow_L_min, op.pump_efficiency)
    energy = (permeate_normalized_energy_kwh_m3(energy_feed_basis, op.recovery_percent)
             if op.recovery_percent > 0 else energy_feed_basis)
    cost = screening_cost_per_m3(op.electricity_price_per_kwh, energy)

    total_flow = total_permeate_flow_L_hr(Jt, op.membrane_area_m2)

    score, subs = feasibility_score(observed_rejection, Jt, decline, energy, cost, targets)

    return {
        "polymer_name": membrane.polymer_name, "membrane_type": membrane.membrane_type,
        "flux_LMH": Jt, "initial_flux_LMH": J0, "structural_flux_LMH": structural_flux,
        "rejection_percent": observed_rejection, "intrinsic_rejection_percent": intrinsic_rejection,
        "permeate_mg_L": cp, "concentrate_mg_L": concentrate, "flux_decline_percent": decline,
        "energy_kWh_m3": energy, "cost_per_m3": cost, "total_permeate_flow_L_hr": total_flow,
        "feasibility_score": score,
        "flux_model": "Detailed model: temperature-corrected permeability, "
                      "concentration-polarization-corrected rejection, "
                      "water-quality-adjusted fouling, permeate-normalized energy.",
        "rejection_model": "Intrinsic (user-supplied) rejection corrected for concentration "
                           "polarization via film theory (models.concentration_polarization).",
        "scientific_note": "Extended screening-level approximation; still requires experimental validation. "
                           "See docs/equations.md for which relationships are standard textbook physics "
                           "versus explicitly-flagged illustrative heuristics.",
        "physics_model": "detailed",
        **{f"{k}_score":v for k,v in subs.items()}
    }
