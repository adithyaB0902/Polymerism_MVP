"""Typed input containers passed between the UI, physics models and ML pipeline.

Not every field below is used by the current screening physics in
:mod:`models.simulator` — several are collected for record-keeping and
as ML training features, or reserved for future model extensions. Each
field is annotated below with whether `run_simulation` actually reads
it. See docs/assumptions.md for the full rationale.
"""

from dataclasses import dataclass


@dataclass
class Membrane:
    polymer_name: str                          # used: label only
    membrane_type: str                         # used: label only (e.g. MF/UF/NF/RO)
    thickness_um: float                        # NOT used in flux/energy/cost calc; validated (>=0) only
    permeability_LMH_bar: float                # used: drives flux (permeability_flux)
    pore_size_nm: float = 0.0                  # NOT used by run_simulation; ML feature / future model input
    MWCO: float = 0.0                          # NOT used by run_simulation; ML feature / future model input
    porosity: float = 0.0                      # NOT used by run_simulation; ML feature / future model input
    surface_charge: float = 0.0                # NOT used by run_simulation; ML feature / future model input
    hydrophilicity: float = 0.0                # NOT used by run_simulation; ML feature / future model input
    membrane_area_cm2: float = 10000.0         # NOT used in any calculation (see OperatingConditions.membrane_area_m2)
    baseline_rejection_percent: float = 0.0    # used: drives permeate concentration (rejection.py)
    fouling_coefficient: float = 0.0           # used: drives flux decline over time (fouling.py)


@dataclass
class Water:
    contaminant: str = "Custom contaminant"    # used: label only
    feed_concentration_mg_L: float = 100.0     # used: drives permeate concentration
    turbidity_NTU: float = 0.0                 # NOT used by run_simulation; ML feature / record-keeping
    TDS_mg_L: float = 0.0                      # NOT used by run_simulation; ML feature / record-keeping
    pH: float = 7.0                            # validated (0-14) only; not otherwise used in the calc
    temperature_C: float = 25.0                # NOT used by run_simulation; ML feature / record-keeping
    viscosity_Pa_s: float = 0.001              # NOT used by run_simulation; only relevant to flux.resistance_flux
    density_kg_m3: float = 1000.0              # NOT used by run_simulation; record-keeping


@dataclass
class OperatingConditions:
    TMP_bar: float = 2.0                       # used: drives flux and energy
    feed_flow_L_min: float = 10.0              # validated (>=0) only; NOT used in the energy formula (see energy.py)
    crossflow_velocity_m_s: float = 0.2        # NOT used by run_simulation; ML feature / record-keeping
    recovery_percent: float = 20.0             # validated (<=100) only; NOT used in flux/energy/cost calc
    operating_time_hr: float = 4.0             # used: drives fouling/flux decline over time
    temperature_C: float = 25.0                # NOT used by run_simulation; record-keeping
    membrane_area_m2: float = 1.0              # NOT used in any calculation (results are all per-m2/per-m3 intensive values)
    pump_efficiency: float = 0.7               # used: drives hydraulic energy
    electricity_price_per_kwh: float = 8.0     # used: drives cost


@dataclass
class Targets:
    """User-defined pass/fail and scoring targets, consumed by models.feasibility."""
    min_rejection_percent: float = 95.0
    min_flux_LMH: float = 30.0
    max_flux_decline_percent: float = 20.0
    max_energy_kWh_m3: float = 2.0
    max_cost_per_m3: float = 10.0
