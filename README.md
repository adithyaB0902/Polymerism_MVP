# POLYMEMSIM

A virtual polymer-membrane research and screening lab for membrane process design, ML-assisted evaluation, and CS–CA–biochar Pb(II) adsorption studies.

Status: verified working. The current project test suite passes successfully.

## Run

From the project directory:

```bash
cd polymemsim
python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux/macOS
# source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py

#host
https://polymerismmvp-teuf5x9fbv3ynkwxbwdgwk.streamlit.app/

```

A local SQLite database is created automatically at `data/polymemsim.db` on first run. Model and experiment artifacts remain local to the app runtime.

## Test

```bash
cd polymemsim
.venv\Scripts\activate
pytest tests -q
```

Current verification result: `139 passed`.

## What the project does

### Original POLYMEMSIM lab workflow
- Sample registration and batch management
- Guided experimental protocol runner
- Physics-based single simulation and replicate generation
- Lab notebook logging of simulated and measured experiments
- Validation and calibration against uploaded CSV data
- Virtual experiments, optimization, Pareto analysis, and sensitivity analysis
- Machine learning model training, evaluation, uncertainty, registry, and explainability

### New CS–CA–Biochar Pb(II) research workflow
- Membrane formulation tracking for chitosan + cellulose acetate + biochar
- Pb(II) removal and adsorption capacity calculations using:
  - R(%) = ((C0 - Ce) / C0) × 100
  - qe = ((C0 - Ce) × V) / m
- Four-factor Box–Behnken design generator with 29 runs and 5 center points
- CSV/Excel experimental dataset validation without overwriting existing data
- Browser-based manual entry for measured BBD runs, including automatic removal and qe calculation
- Second-order RSM on coded factors with a full ANOVA (model, per-term, residual, lack of fit, pure error), R² / adjusted R² / predicted R² (PRESS), adequate precision, C.V., stationary-point analysis, response-surface plots, and residual / leverage / Cook's-distance diagnostics
- ML comparison across regression baselines and advanced models
- Optimization using a model-predicted surrogate objective
- Confirmation experiment comparison between predicted and measured performance
- Isotherm, kinetics, characterization, and regeneration storage support for research workflows

## Scientific integrity

- Synthetic data and ML predictions are not experimental validation.
- Experimental values must be supplied by the user or uploaded from CSV/Excel.
- No hard-coded experimental results or fake performance values are used.
- Results are clearly separated by source: simulated, predicted, or experimental.

## Project layout

```text
app.py                  Streamlit application entry point
models/                 Physics model layer
simulation/             Synthetic data generation and sensitivity analysis
optimization/           Optimization and candidate ranking utilities
validation/             Real-data validation, comparison, and calibration
ml/                     Machine learning pipeline and registry
lab/                    Samples, protocols, replicates, and database conversions
db/                     SQLite schema and repository helpers
research/               CS–CA–biochar Pb(II) research workflows
config/                 Parameter configuration defaults
docs/                   Assumptions, equations, and protocol notes
data/                   Example CSV datasets and presets
tests/                  Test suite
```

## Research modules

The new research package is intentionally modular and separate from the legacy membrane simulator:

- `research/pb_removal.py` — Pb(II) removal and qe calculations
- `research/membrane_formulation.py` — CS–CA–biochar formulation records
- `research/experiments.py` — experimental dataset validation and import
- `research/rsm/box_behnken.py` — four-factor Box–Behnken design generation
- `research/rsm/quadratic_model.py` — second-order RSM regression on coded factors, prediction, stationary point, diagnostics
- `research/rsm/anova.py` — ANOVA table, lack-of-fit test, and model-adequacy warnings
- `research/rsm/surfaces.py` — predicted response-surface grids for contour / 3D plots
- `research/rsm/ui.py` — Streamlit view for the RSM analysis section
- `research/ml.py` — RSM/ML research model comparison
- `research/optimization.py` — differential-evolution optimization
- `research/adsorption.py` — Langmuir/Freundlich isotherm models
- `research/kinetics.py` — pseudo-first-order / pseudo-second-order fits
- `research/confirmation.py` — prediction-vs-experiment comparison
- `research/characterization.py` and `research/regeneration.py` — storage/organization of lab metadata

## Notes

- The original POLYMEMSIM functionality remains intact.
- The new research workflow is additive and compatible with the existing Streamlit architecture.
- Data are stored in SQLite without overwriting existing lab datasets.
