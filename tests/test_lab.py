import tempfile

import numpy as np
import pytest

from db import repository as repo
from models.membrane import Membrane
from lab.samples import load_presets, create_sample_from_preset, sample_to_membrane, create_sample_from_membrane
from lab.replicates import simulate_replicates, replicate_statistics, DEFAULT_NOISE_CV_PERCENT
from lab.protocol import (start_protocol, current_step_definition, record_current_step,
                          is_complete, progress_fraction, collected_values, PROTOCOL_STEPS, STEP_NAMES)


def _fresh_conn():
    tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
    tmp.close()
    return repo.get_connection(tmp.name)


# ------------------------------------------------------------------ samples

def test_load_presets_has_three_entries():
    presets = load_presets()
    assert len(presets) == 3
    assert {p["membrane_type"] for p in presets} == {"MF", "UF", "NF"}


def test_create_sample_from_preset_and_convert_back():
    conn = _fresh_conn()
    presets = load_presets()
    nf_preset = next(p for p in presets if p["membrane_type"] == "NF")
    sample_id = create_sample_from_preset(conn, nf_preset)
    sample = repo.get_sample(conn, sample_id)
    assert sample["origin"] == "preset"
    assert sample["baseline_rejection_percent"] == 98.0
    membrane = sample_to_membrane(sample)
    assert isinstance(membrane, Membrane)
    assert membrane.membrane_type == "NF"


def test_membrane_sample_round_trip_preserves_all_fields():
    conn = _fresh_conn()
    m = Membrane("X", "RO", 50.0, 8.0, pore_size_nm=1.2, MWCO=200.0, porosity=0.2,
                surface_charge=-0.5, hydrophilicity=0.9, membrane_area_cm2=5000.0,
                baseline_rejection_percent=99.5, fouling_coefficient=0.08)
    sid = create_sample_from_membrane(conn, m)
    m2 = sample_to_membrane(repo.get_sample(conn, sid))
    assert m == m2


# --------------------------------------------------------------- replicates

def test_simulate_replicates_count_and_variation():
    base = {"flux_LMH": 50.0, "rejection_percent": 95.0, "feasibility_score": 80.0}
    reps = simulate_replicates(base, n=10, seed=1)
    assert len(reps) == 10
    fluxes = [r["flux_LMH"] for r in reps]
    assert len(set(fluxes)) > 1  # not all identical -- noise was actually applied
    assert all(f >= 0 for f in fluxes)


def test_simulate_replicates_deterministic_with_seed():
    base = {"flux_LMH": 50.0, "rejection_percent": 95.0}
    reps1 = simulate_replicates(base, n=5, seed=42)
    reps2 = simulate_replicates(base, n=5, seed=42)
    assert reps1 == reps2


def test_simulate_replicates_clamps_rejection_to_valid_range():
    base = {"rejection_percent": 99.9}
    reps = simulate_replicates(base, n=200, noise_cv_percent={"rejection_percent": 5.0}, seed=1)
    assert all(0 <= r["rejection_percent"] <= 100 for r in reps)


def test_replicate_statistics_matches_numpy():
    base = {"flux_LMH": 50.0}
    reps = simulate_replicates(base, n=20, seed=7)
    stats = replicate_statistics(reps, fields=["flux_LMH"])
    values = [r["flux_LMH"] for r in reps]
    assert stats["flux_LMH"]["mean"] == pytest.approx(np.mean(values))
    assert stats["flux_LMH"]["std"] == pytest.approx(np.std(values, ddof=1))
    assert stats["flux_LMH"]["n"] == 20


def test_simulate_replicates_rejects_n_less_than_one():
    with pytest.raises(ValueError):
        simulate_replicates({"flux_LMH": 50.0}, n=0)


# ----------------------------------------------------------------- protocol

def test_protocol_full_lifecycle():
    conn = _fresh_conn()
    sample_id = repo.create_sample(conn, {"polymer_name": "P", "membrane_type": "UF",
                                          "thickness_um": 100, "permeability_LMH_bar": 25})
    run_id = start_protocol(conn, sample_id)
    run = repo.get_protocol_run(conn, run_id)
    assert not is_complete(run)
    assert progress_fraction(run) == 0.0

    for i, step in enumerate(PROTOCOL_STEPS):
        step_def = current_step_definition(run)
        assert step_def["name"] == step["name"]
        values = {f: f"value-for-{f}" for f in step_def["fields"]}
        run = record_current_step(conn, run_id, values)

    assert is_complete(run)
    assert progress_fraction(run) == 1.0
    assert current_step_definition(run) is None

    flat = collected_values(run)
    assert flat.get("polymer_name") == "value-for-polymer_name"
    assert flat.get("flux_LMH") == "value-for-flux_LMH"


def test_cannot_record_step_after_completion():
    conn = _fresh_conn()
    sample_id = repo.create_sample(conn, {"polymer_name": "P", "membrane_type": "UF",
                                          "thickness_um": 100, "permeability_LMH_bar": 25})
    run_id = start_protocol(conn, sample_id)
    for step in PROTOCOL_STEPS:
        record_current_step(conn, run_id, {f: "x" for f in step["fields"]})
    with pytest.raises(ValueError):
        record_current_step(conn, run_id, {})
