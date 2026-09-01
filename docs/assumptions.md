# Assumptions

## Physics
- The Simple model assumes pressure-driven linear flux (J = A x TMP); membrane thickness/pore structure are assumed already reflected in the measured/literature permeability value.
- Rejection is user-supplied (intrinsic), not predicted from pore size/MWCO/charge — the Detailed model corrects it for concentration polarization but never predicts it from scratch.
- Fouling uses an empirical first-order approximation; in Detailed mode the coefficient is further adjusted by an explicitly-illustrative water-quality/surface heuristic (see `models/fouling_index.py`).
- Energy is a hydraulic pumping estimate (pressure and efficiency only) — it never depends on feed flow rate, and depends on recovery/area only in the ways described in `docs/equations.md`.
- Economics are screening-level: only electricity cost is included by default.
- The Detailed model's temperature, pore-flow, concentration-polarization and mass-balance relationships are standard textbook physics; the fouling-propensity adjustment is explicitly a bounded heuristic, not a literature-derived formula — see each module's own docstring in `models/`.
- Even in Detailed mode, several inputs remain unused: `MWCO`, water density, and the manually-entered viscosity field (Detailed mode computes viscosity from temperature automatically instead). See `models/membrane.py` for the complete field-by-field breakdown for both models.

## Lab & data
- The lab notebook, samples, batches, protocol runs, calibration runs and ML model registry all persist in a local SQLite database (`data/polymemsim.db`, git-ignored) — nothing here is a cloud service or shared across machines.
- Simulated replicate measurements (Single Simulation tab) inject illustrative Gaussian noise on top of the deterministic physics result, purely for the lab experience — this is not a model of any real measurement-error source (see `lab/replicates.py`).
- Illustrative presets (`data/membrane_properties.csv`) are demo defaults, not experimental measurements.
- Synthetic virtual-experiment data is not experimental evidence.

## Calibration & reliability
- Calibration (`validation/calibration.py`) fits `permeability_LMH_bar`, `fouling_coefficient` and `baseline_rejection_percent` to real measurements via nonlinear least squares against the Simple model.
- A sample's feasibility "reliability" (`models/feasibility.py`) stays LOW unless it is itself the result of a calibration run, in which case it's capped at MEDIUM or HIGH depending on that calibration's fit quality (R2) — a poor-fitting calibration does not earn a reliability upgrade.

## Machine learning
- ML models are trained on synthetic physics-generated data, optionally combined with real logged measurements (`ml/dataset.py`) — with only synthetic data, they mostly learn to approximate whichever physics model generated the training set, not real membrane behavior.
- The `ml/train.FEATURES` list includes several inputs the Simple physics model itself never uses (e.g. `pore_size_nm`); a well-trained model should assign these near-zero importance — check this yourself in the ML Lab's Explainability section rather than assuming it.
- Ensemble-spread uncertainty (`ml/uncertainty.py`) is only meaningful for bagging ensembles (Random Forest, Extra Trees), not Gradient Boosting, and is a cheap model-native signal, not a formal confidence interval.
- The out-of-distribution check (`ml/ood.py`) is a simple per-feature min/max range test, not a joint-distribution novelty detector — a candidate can pass it on every individual feature while still being an unusual *combination* the training data never saw.

## Overall
- The simulator cannot establish drinking-water safety, regulatory compliance, or commercial viability, regardless of physics model, calibration status, or ML predictions.
