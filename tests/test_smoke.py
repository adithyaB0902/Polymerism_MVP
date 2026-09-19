"""Smoke tests: the fastest possible checks that the app is alive and
nothing is fundamentally broken. These should catch "the app doesn't
even start" issues in a few seconds, before running the slower
functional/integration/system suites below.
"""

import importlib

import pytest


SMOKE_MODULES = [
    "models.membrane", "models.simulator", "models.flux", "models.rejection",
    "models.fouling", "models.energy", "models.economics", "models.feasibility",
    "models.viscosity", "models.pore_flow", "models.concentration_polarization",
    "models.mass_balance", "models.fouling_index",
    "simulation.generate_dataset", "simulation.sensitivity",
    "optimization.optimizer", "optimization.experiment_recommender",
    "validation.experimental_data", "validation.comparison", "validation.calibration",
    "db.schema", "db.repository",
    "lab.samples", "lab.replicates", "lab.protocol",
    "ml.train", "ml.dataset", "ml.model_registry", "ml.predict", "ml.evaluate",
    "ml.uncertainty", "ml.explain", "ml.features", "ml.ood",
    "research.rsm.box_behnken", "research.rsm.quadratic_model", "research.rsm.anova",
    "research.rsm.surfaces", "research.rsm.ui",
    "ui_theme",
]


@pytest.mark.parametrize("module_name", SMOKE_MODULES)
def test_module_imports_cleanly(module_name):
    """Every module in the project should import without raising."""
    importlib.import_module(module_name)


def test_app_module_imports_cleanly():
    """app.py itself should import (i.e. its top-level code, including
    the sidebar/tab construction, executes) without raising — the
    single fastest possible 'is the app broken' check.

    Run via Streamlit's AppTest rather than a bare `import app`, since
    app.py makes Streamlit API calls (st.set_page_config, st.tabs,
    widgets) that require an active script-run context.
    """
    from pathlib import Path

    from streamlit.testing.v1 import AppTest

    app_path = Path(__file__).resolve().parent.parent / "app.py"
    at = AppTest.from_file(str(app_path), default_timeout=60)
    at.run()
    assert not list(at.exception), f"app.py raised on initial load: {list(at.exception)}"


def test_database_connection_smoke(tmp_path):
    """The DB layer should open a connection and create its schema
    without error against a throwaway path."""
    from db import repository as repo

    conn = repo.get_connection(str(tmp_path / "smoke.db"))
    assert repo.list_samples(conn) == []


def test_minimal_simulation_smoke():
    """The core physics pipeline should run end-to-end with default
    values without raising."""
    from models.membrane import Membrane, Water, OperatingConditions, Targets
    from models.simulator import run_simulation

    result = run_simulation(Membrane("P", "UF", 100, 25, baseline_rejection_percent=95,
                                     fouling_coefficient=0.03),
                            Water(), OperatingConditions(), Targets())
    assert result["flux_LMH"] >= 0
