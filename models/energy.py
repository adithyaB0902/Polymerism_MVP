"""Screening-level hydraulic (pumping) energy estimate.

Only trans-membrane pressure and pump efficiency are used, on purpose:
specific energy consumption (energy per m3 of throughput) for an ideal
pressure-driven process is

    SEC [kWh/m3] = dP [Pa] / eta / 3.6e6   (3.6e6 J = 1 kWh)
                 = dP [bar] * 1e5 / eta / 3.6e6
                 = dP [bar] * 0.0277778 / eta

Flow rate cancels out of this ratio (power = dP * Q / eta, volume rate
= Q, so power / volume-rate = dP / eta) — it does NOT depend on how much
water is being pushed through per minute, only on pressure and pump
efficiency. `flow_L_min` is still accepted and validated here so the
function signature documents the physical picture and so callers can't
silently pass a negative/invalid flow, but it is intentionally absent
from the return calculation. This is a simplification: it ignores pipe
friction losses, recovery-dependent effects, and any energy used by
non-pumping equipment (pretreatment, cleaning-in-place, etc.).
"""


def hydraulic_energy_kwh_m3(TMP_bar, flow_L_min, pump_efficiency):
    """Specific hydraulic pumping energy, in kWh per m3 of throughput.

    See module docstring for why `flow_L_min` does not appear in the
    formula despite being a required, validated argument.
    """
    if TMP_bar < 0 or flow_L_min < 0:
        raise ValueError("Pressure and flow must be non-negative.")
    if not 0 < pump_efficiency <= 1:
        raise ValueError("Pump efficiency must be between 0 and 1.")
    # 1 bar * 1 m3 = 1e5 Pa * 1 m3 = 1e5 J = 1e5 / 3.6e6 kWh = 0.0277778 kWh
    return TMP_bar * 0.0277777778 / pump_efficiency


def volumetric_flow_m3_s(flow_L_min):
    """Convert a flow rate from L/min to m3/s. Not currently called by
    the simulator; provided as a small utility for anyone extending the
    energy model (e.g. to compute absolute pump power in Watts)."""
    return flow_L_min / 1000.0 / 60.0
