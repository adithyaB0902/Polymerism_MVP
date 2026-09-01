# POLYMEMSIM Equations

POLYMEMSIM ships two physics models, toggled in the app sidebar.
`models/simulator.py`'s `run_simulation` implements the Simple model;
`run_detailed_simulation` implements the Detailed model, which layers
the additional relationships below on top.

## Simple model

**Flux**: J = A x TMP (A = permeability, LMH/bar). See `models/flux.py`.

**Rejection**: R = (1 - Cp/Cf) x 100 is the standard definition, but
POLYMEMSIM does **not** solve it in the usual direction (measure Cp and
Cf, compute R). `R` is a required, directly-specified input
(`Membrane.baseline_rejection_percent`), and the equation is solved for
Cp: `Cp = Cf x (1 - R/100)`. See `models/rejection.py`.

**Fouling**: J(t) = J0 x exp(-k x t), `k` a user-supplied empirical
coefficient. See `models/fouling.py`.

**Hydraulic energy**: expressed as specific energy consumption (kWh
per m3 of throughput), `SEC = dP[bar] x 0.0277778 / eta`. Flow rate
cancels out of this ratio — the result depends only on pressure and
pump efficiency. See `models/energy.py`.

**Cost**: `cost/m3 = electricity_price x energy_kWh_m3` by default;
membrane/cleaning/maintenance terms exist in `models/economics.py` but
default to zero.

All of the above are screening-level approximations requiring
calibration/validation for a real system — see
`validation/calibration.py` and the Validation & Calibration tab.

## Detailed model — additional relationships

Each of these is a standard, checkable textbook relationship **except**
the fouling-propensity adjustment, which is explicitly an illustrative
heuristic (there is no single agreed quantitative formula for it — see
`models/fouling_index.py`'s own docstring).

**Temperature-corrected viscosity** (`models/viscosity.py`): the Vogel
equation, `mu(T) = 2.414e-5 x 10^(247.8/(T+133.15))` Pa*s (T in degC),
matches published water viscosity within ~2% across 0-100 degC. Flux
is inversely proportional to viscosity, so flux at the actual water
temperature is scaled from the reference-temperature permeability by
`mu(reference)/mu(actual)`.

**Structural (pore-flow) flux cross-check** (`models/pore_flow.py`):
the Hagen-Poiseuille equation for laminar flow through cylindrical
pores, `Lp = (porosity x r^2) / (8 x mu x tortuosity x thickness)`,
gives an independent flux prediction from membrane structure
(thickness, porosity, pore size) rather than the empirical permeability
value. Reported alongside the empirical flux as a consistency check,
not used to override it.

**Concentration-polarization-corrected rejection**
(`models/concentration_polarization.py`): film theory,
`R_observed = R_intrinsic / (R_intrinsic + (1 - R_intrinsic) x exp(Jv/k))`,
where `k` is a cross-flow-velocity-dependent mass-transfer coefficient.
Higher cross-flow velocity (better mixing) pushes observed rejection
closer to the intrinsic value; higher flux relative to cross-flow pushes
it down.

**Recovery-based mass balance** (`models/mass_balance.py`): a
single-stage, fully-mixed balance gives concentrate concentration,
`Cc = (Cf - r x Cp) / (1 - r)` (r = recovery fraction), and
permeate-normalized energy, `SEC_permeate = SEC_feed / r` (the feed-pump
energy is spent on the whole feed stream, but only the recovery
fraction of it becomes product).

**Fouling-propensity adjustment** (`models/fouling_index.py`,
illustrative): a bounded multiplier (0.5x-3x) on the base fouling
coefficient from feed turbidity/TDS (more fouling-prone) and membrane
hydrophilicity/surface charge (less fouling-prone), anchored to 1.0x at
the app's default "typical" conditions.

## Total permeate flow

`models/mass_balance.py`'s `total_permeate_flow_L_hr = flux_LMH x
membrane_area_m2` — the only place membrane area affects an output
(every other reported quantity is an intensive, per-m2/per-m3 value).
