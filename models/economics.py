"""Screening-level cost-per-m3 estimate.

Only the electricity term is populated by the live simulator today —
:func:`models.simulator.run_simulation` calls this with just
`electricity_price` and `energy_kwh_m3`, so `membrane_cost`,
`cleaning_cost` and `maintenance_cost` all default to 0 in the app.
That means the "Cost" figure shown in the UI currently reflects
electricity only, NOT a full total cost of ownership. The parameters
below exist so a caller with real membrane-replacement, cleaning or
maintenance cost data can get a more complete number; the app's sidebar
does not currently expose inputs for them (see README "Limitations").
"""


def screening_cost_per_m3(electricity_price, energy_kwh_m3, membrane_cost=0.0,
                          membrane_lifetime_m3=100000.0, cleaning_cost=0.0,
                          maintenance_cost=0.0):
    """Total screening cost per m3 = electricity + amortized membrane +
    cleaning + maintenance (all in cost-units per m3)."""
    if membrane_lifetime_m3 <= 0:
        raise ValueError("Membrane lifetime throughput must be positive.")
    electricity = electricity_price * energy_kwh_m3
    membrane = membrane_cost / membrane_lifetime_m3
    return electricity + membrane + cleaning_cost + maintenance_cost
