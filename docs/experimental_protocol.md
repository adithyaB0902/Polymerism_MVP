# Experimental Protocol

This checklist is no longer just a planning document — it's implemented
as an actual guided, saved, step-by-step workflow in `lab/protocol.py`
(`PROTOCOL_STEPS`), run from the app's **Protocol Runner** tab and
persisted per-sample in the lab database (`protocol_runs` table).

## Steps

1. **Formulation & structure** — polymer, membrane type, thickness.
2. **Operating setup** — trans-membrane pressure, temperature, feed flow for this run.
3. **Feed characterization** — feed concentration, pH.
4. **Permeate volume & flux** — measured permeate volume, elapsed time, computed flux.
5. **Permeate concentration & rejection** — measured permeate concentration, computed/recorded rejection.
6. **Flux vs time (fouling)** — flux at several time points, to characterize fouling behavior.
7. **Finalize** — review everything recorded and mark the run complete.

Completing a run lets you log it directly to the lab notebook as a
`measured` experiment (as opposed to a `simulated` one from the Single
Simulation tab), which then becomes available to:
- compare against physics predictions (Validation & Calibration tab),
- calibrate physics parameters against (same tab), and
- train ML models on, combined with synthetic data (ML Lab tab).

Use replicate measurements where practical and retain raw observations
for calibration — the app can also generate *simulated* replicate
readings (`lab/replicates.py`) for demonstration purposes, but those
are illustrative injected noise, not a substitute for real replicates.
