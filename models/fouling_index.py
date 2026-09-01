"""Composite fouling-propensity adjustment.

Unlike the other "detailed physics" modules in this package (viscosity,
pore-flow, concentration polarization, mass balance), which implement
specific, checkable equations from standard membrane-science
references, there is no single widely-agreed quantitative formula for
"how much do turbidity/TDS/hydrophilicity/surface charge change a
fouling rate constant" — the real relationship is highly
membrane-and-foulant-specific and normally has to be measured, not
predicted. This module is therefore explicitly a bounded, illustrative
heuristic, not a literature-derived model like its neighbors:

- Higher feed turbidity and TDS are associated with more particulate/
  colloidal and scaling fouling, respectively (qualitatively
  well-established; not quantitatively universal).
- More hydrophilic membrane surfaces are broadly associated with lower
  fouling propensity (less hydrophobic adsorption of organic foulants;
  a widely cited qualitative trend in membrane fouling literature).
- Strongly (either positively or negatively) charged surfaces can
  reduce fouling by electrostatically repelling like-charged foulants,
  but the sign/magnitude depends entirely on the specific foulant, so
  this module only credits charge *magnitude*, not sign, with a small
  effect.

Each factor is mapped to a bounded multiplier and combined
multiplicatively, then clipped to a sane range, so a poor-quality feed
and a fouling-prone membrane can meaningfully worsen the fouling
coefficient without ever producing nonphysical (negative or extreme)
values.
"""

MIN_MULTIPLIER = 0.5
MAX_MULTIPLIER = 3.0
# Note: MAX_MULTIPLIER is a safety ceiling, not a target the individual
# factors are tuned to reach exactly — turbidity/TDS/hydrophilicity
# alone top out around 2.9x at extreme inputs (surface charge can only
# ever push the multiplier down), which is intentional: it's safer for
# a bound to rarely be exercised than to be an artificially tight fit.


def fouling_propensity_multiplier(turbidity_NTU, TDS_mg_L, hydrophilicity, surface_charge):
    """Return a multiplier to apply to a membrane's base fouling
    coefficient, in [MIN_MULTIPLIER, MAX_MULTIPLIER]. 1.0 means "no
    adjustment" (matches the illustrative reference conditions used to
    anchor each factor below).
    """
    if turbidity_NTU < 0 or TDS_mg_L < 0:
        raise ValueError("Turbidity and TDS must be non-negative.")
    if not 0 <= hydrophilicity <= 1:
        raise ValueError("Hydrophilicity must be between 0 and 1.")

    # Reference (multiplier = 1.0 contribution) conditions: 5 NTU,
    # 500 mg/L TDS, hydrophilicity 0.5, zero surface charge — the
    # sidebar's own default values.
    turbidity_factor = 1.0 + 0.5 * min(turbidity_NTU / 5.0 - 1.0, 4.0) / 4.0 if turbidity_NTU > 5 else \
                       1.0 - 0.2 * (1.0 - turbidity_NTU / 5.0)
    tds_factor = 1.0 + 0.5 * min(TDS_mg_L / 500.0 - 1.0, 4.0) / 4.0 if TDS_mg_L > 500 else \
                1.0 - 0.2 * (1.0 - TDS_mg_L / 500.0)
    hydrophilicity_factor = 1.3 - 0.6 * hydrophilicity   # 0 -> 1.3x (fouls more), 1 -> 0.7x (fouls less)
    charge_factor = 1.0 - 0.15 * min(abs(surface_charge), 1.0)  # any strong charge -> mild fouling reduction

    multiplier = turbidity_factor * tds_factor * hydrophilicity_factor * charge_factor
    return max(MIN_MULTIPLIER, min(MAX_MULTIPLIER, multiplier))


def adjusted_fouling_coefficient(base_fouling_coefficient, turbidity_NTU, TDS_mg_L,
                                 hydrophilicity, surface_charge):
    """Apply `fouling_propensity_multiplier` to a membrane's base
    fouling coefficient (1/hr)."""
    if base_fouling_coefficient < 0:
        raise ValueError("Base fouling coefficient must be non-negative.")
    multiplier = fouling_propensity_multiplier(turbidity_NTU, TDS_mg_L, hydrophilicity, surface_charge)
    return base_fouling_coefficient * multiplier
