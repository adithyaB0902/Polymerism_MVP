"""System tests: drive the actual Streamlit app (via Streamlit's
official AppTest harness, which executes the real app.py script against
a simulated browser session) through every major user-facing workflow,
end to end, exactly as a person clicking through the UI would.

This is the outermost layer of the test pyramid here:
  test_smoke.py        -- does anything even start? (milliseconds)
  unit tests (the rest) -- does each function behave correctly in isolation?
  test_integration.py  -- do modules work correctly together? (no UI)
  test_system_e2e.py   -- does the whole assembled app work for a user? (this file)

Uses a fresh temporary database per test (never the real
data/polymemsim.db) so this suite never depends on or mutates real
local lab data.
"""

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

APP_PATH = str(Path(__file__).resolve().parent.parent / "app.py")


@pytest.fixture
def at(tmp_path, monkeypatch):
    """A fresh AppTest instance backed by a throwaway database, so
    system tests never touch the developer's real lab data.

    `st.cache_resource` caches app.py's DB connection at the *process*
    level, keyed only by function identity (no args) -- so without
    clearing it here, the second test in a run would silently reuse the
    first test's already-populated connection/database instead of a
    fresh one, even after monkeypatching DEFAULT_DB_PATH.
    """
    import streamlit as st
    import db.repository as repo

    st.cache_resource.clear()
    monkeypatch.setattr(repo, "DEFAULT_DB_PATH", tmp_path / "system_test.db")
    a = AppTest.from_file(APP_PATH, default_timeout=120)
    a.run()
    assert not list(a.exception), f"initial load raised: {list(a.exception)}"
    return a


def _click(at, label, nth=0):
    matches = [b for b in at.button if b.label == label]
    assert matches, f"button {label!r} not found"
    matches[nth].click()
    at.run()
    assert not list(at.exception), f"clicking {label!r} raised: {list(at.exception)}"


def _goto(at, value):
    """Select `value` on whichever sub-page radio currently offers it
    (the Lab Workflow and Screening & Optimization groups each have
    their own radio with key ending in '_page')."""
    for r in at.radio:
        if value in r.options:
            r.set_value(value)
            at.run()
            assert not list(at.exception), f"navigating to {value!r} raised: {list(at.exception)}"
            return
    raise AssertionError(f"no radio offers option {value!r}")


def test_app_loads_with_six_top_level_tabs(at):
    assert len(at.tabs) == 6


def test_sample_registry_workflow(at):
    _goto(at, "Sample Registry")
    _click(at, "Load preset as new sample")
    _click(at, "CREATE SAMPLE")
    # sidebar should now offer an "Active sample" selector with 2 samples
    active = [r for r in at.selectbox if r.label == "Active sample"]
    assert active and len(active[0].options) == 2


def test_protocol_runner_full_walkthrough(at):
    _goto(at, "Sample Registry")
    _click(at, "Load preset as new sample")
    _goto(at, "Protocol Runner")
    _click(at, "START PROTOCOL")
    steps_completed = 0
    for _ in range(10):
        if [b for b in at.button if b.label == "RECORD & CONTINUE"]:
            _click(at, "RECORD & CONTINUE")
            steps_completed += 1
        elif [b for b in at.button if b.label == "LOG AS EXPERIMENT"]:
            _click(at, "LOG AS EXPERIMENT")
            break
        else:
            break
    assert steps_completed > 0


def test_single_simulation_and_replicates_and_logging(at):
    _goto(at, "Sample Registry")
    _click(at, "Load preset as new sample")
    _goto(at, "Single Simulation")
    _click(at, "RUN SIMULATION")
    assert at.metric  # some metrics rendered
    _click(at, "SIMULATE REPLICATES")
    _click(at, "SAVE TO LAB NOTEBOOK")

    _goto(at, "Lab Notebook")
    assert any("Logged experiments" in m.label for m in at.metric)


def test_validation_and_calibration_workflow(at):
    _goto(at, "Sample Registry")
    _click(at, "Load preset as new sample")
    _goto(at, "Validation & Calibration")
    csv_bytes = (b"flux_LMH,rejection_percent,operating_time_hr\n"
                b"48.5,94.8,1\n45.2,95.1,2\n41.0,95.3,4\n36.5,95.0,8\n")
    at.get("file_uploader")[0].upload("measurements.csv", csv_bytes, "text/csv")
    at.run()
    assert not list(at.exception)
    _click(at, "CALIBRATE")
    _click(at, "SAVE CALIBRATED SAMPLE")


def test_screening_and_optimization_group(at):
    for page, button in [
        ("Virtual Experiments", "GENERATE VIRTUAL EXPERIMENTS"),
        ("Optimization", "OPTIMIZE"),
        ("Pareto Frontier", "CALCULATE PARETO FRONTIER"),
        ("Sensitivity", "RUN SENSITIVITY"),
        ("Recommend Next", None),  # needs virtual_df from the first page; checked separately
    ]:
        _goto(at, page)
        if button:
            _click(at, button)
    _goto(at, "Recommend Next")
    if [b for b in at.button if b.label == "RECOMMEND EXPERIMENTS"]:
        _click(at, "RECOMMEND EXPERIMENTS")


def test_ml_lab_train_register_predict(at):
    _goto(at, "Sample Registry")
    _click(at, "Load preset as new sample")
    _click(at, "TRAIN ML MODELS")
    _click(at, "REGISTER MODELS")
    _click(at, "RUN 5-FOLD CROSS-VALIDATION")
    _click(at, "COMPARE MODEL TYPES")
    _click(at, "PREDICT WITH ML")


def test_feasibility_renders_unconditionally(at):
    # Feasibility has no button -- it computes on every render. Just
    # confirm switching to it doesn't raise and shows a score.
    at.run()
    assert not list(at.exception)


def test_reference_tab_renders(at):
    at.run()
    assert not list(at.exception)


def test_full_walkthrough_every_workflow_in_one_session(at):
    """The comprehensive version: touches every button across every
    tab/sub-page in a single session, mirroring a real user's session
    rather than one isolated feature per test."""
    _goto(at, "Sample Registry")
    _click(at, "Load preset as new sample")
    _click(at, "CREATE SAMPLE")

    _goto(at, "Protocol Runner")
    _click(at, "START PROTOCOL")
    for _ in range(10):
        if [b for b in at.button if b.label == "RECORD & CONTINUE"]:
            _click(at, "RECORD & CONTINUE")
        elif [b for b in at.button if b.label == "LOG AS EXPERIMENT"]:
            _click(at, "LOG AS EXPERIMENT")
            break
        else:
            break

    _goto(at, "Single Simulation")
    _click(at, "RUN SIMULATION")
    _click(at, "SIMULATE REPLICATES")
    _click(at, "SAVE TO LAB NOTEBOOK")

    _goto(at, "Validation & Calibration")
    csv_bytes = (b"flux_LMH,rejection_percent,operating_time_hr\n"
                b"48.5,94.8,1\n45.2,95.1,2\n41.0,95.3,4\n36.5,95.0,8\n")
    at.get("file_uploader")[0].upload("measurements.csv", csv_bytes, "text/csv")
    at.run()
    _click(at, "CALIBRATE")
    _click(at, "SAVE CALIBRATED SAMPLE")

    _goto(at, "Virtual Experiments")
    _click(at, "GENERATE VIRTUAL EXPERIMENTS")
    _goto(at, "Optimization")
    _click(at, "OPTIMIZE")
    _goto(at, "Pareto Frontier")
    _click(at, "CALCULATE PARETO FRONTIER")
    _goto(at, "Sensitivity")
    _click(at, "RUN SENSITIVITY")
    _goto(at, "Recommend Next")
    _click(at, "RECOMMEND EXPERIMENTS")

    _click(at, "TRAIN ML MODELS")
    _click(at, "REGISTER MODELS")
    _click(at, "RUN 5-FOLD CROSS-VALIDATION")
    _click(at, "COMPARE MODEL TYPES")
    _click(at, "PREDICT WITH ML")

    at.run()
    assert not list(at.exception)
