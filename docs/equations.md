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

## Response surface methodology (CS-CA-biochar / Pb(II) study)

`research/rsm/` fits a full second-order model to the measured removal
efficiency on **coded** factors (`x = (X - centre) / half-range`, so the
low / centre / high levels are -1 / 0 / +1):

`y = b0 + sum_i(bi xi) + sum_i(bii xi^2) + sum_{i<j}(bij xi xj) + e`

For four factors this has 15 coefficients, so more than 15 complete runs
are needed; the intended design is the 29-run Box-Behnken design (24 edge
runs + 5 centre points).

- **ANOVA**: model F = (SS_model / (p-1)) / MSE with SS_model = SS_total -
  SSE. Per-term rows use partial (Type III) sums of squares,
  `SS = t^2 x MSE`.
- **Lack of fit**: replicated factor settings (the centre points) give the
  pure-error sum of squares; `F = (SS_LOF / df_LOF) / (SS_PE / df_PE)`. It
  is reported only when at least one setting is replicated.
- **Fit statistics**: R^2, adjusted R^2, predicted R^2 = 1 - PRESS / SS_total
  with `PRESS = sum(e_i / (1 - h_ii))^2`, adequate precision =
  `(max(y_hat) - min(y_hat)) / sqrt(p x MSE / n)`, and C.V. = `sqrt(MSE) /
  mean(y) x 100`.
- **Stationary point**: `x_s = -1/2 B^-1 b`, where `b` holds the linear
  coefficients and `B` is the symmetric matrix of quadratic (diagonal) and
  half-interaction (off-diagonal) coefficients. The eigenvalues of `B`
  classify it as a maximum (all negative), minimum (all positive), or
  saddle. A stationary point outside the studied region is extrapolation.
- **Diagnostics**: leverage `h_ii`, externally studentized residuals, and
  Cook's distance; runs with |t| > 3 or D > 4/n are flagged.

All statistics are computed from the supplied data. Model outputs are
predictions and do not constitute experimental validation.
