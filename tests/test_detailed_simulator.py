import numpy as np
import pytest

from models.membrane import Membrane, Water, OperatingConditions, Targets
from models.simulator import run_simulation, run_detailed_simulation


def _base():
    m = Membrane("Custom Polymer", "UF", 100.0, 25.0, pore_size_nm=20.0, MWCO=10000.0,
                porosity=0.35, surface_charge=0.0, hydrophilicity=0.7,
                membrane_area_cm2=10000.0, baseline_rejection_percent=95.0, fouling_coefficient=0.03)
    w = Water("Custom contaminant", 100.0, 5.0, 500.0, 7.0, 25.0, 0.001, 1000.0)
    o = OperatingConditions(2.0, 10.0, 0.2, 20.0, 4.0, 25.0, 1.0, 0.70, 8.0)
    t = Targets(95.0, 30.0, 20.0, 2.0, 10.0)
    return m, w, o, t


def test_detailed_simulation_runs_and_has_expected_keys():
    m, w, o, t = _base()
    r = run_detailed_simulation(m, w, o, t)
    for key in ["flux_LMH", "structural_flux_LMH", "rejection_percent", "intrinsic_rejection_percent",
               "concentrate_mg_L", "total_permeate_flow_L_hr", "feasibility_score", "physics_model"]:
        assert key in r
    assert r["physics_model"] == "detailed"
    assert r["structural_flux_LMH"] is not None  # porosity/pore_size supplied in _base()
    assert 0 <= r["feasibility_score"] <= 100


def test_detailed_simulation_differs_meaningfully_from_simple():
    m, w, o, t = _base()
    simple = run_simulation(m, w, o, t)
    detailed = run_detailed_simulation(m, w, o, t)
    # concentration polarization + temperature correction should make
    # these genuinely different, not coincidentally identical
    assert detailed["rejection_percent"] != simple["rejection_percent"]
    assert detailed["intrinsic_rejection_percent"] == simple["rejection_percent"]  # same baseline input


def test_higher_crossflow_improves_detailed_rejection():
    m, w, o, t = _base()
    o_low_cf = OperatingConditions(**{**o.__dict__, "crossflow_velocity_m_s": 0.05})
    o_high_cf = OperatingConditions(**{**o.__dict__, "crossflow_velocity_m_s": 2.0})
    r_low = run_detailed_simulation(m, w, o_low_cf, t)
    r_high = run_detailed_simulation(m, w, o_high_cf, t)
    assert r_high["rejection_percent"] > r_low["rejection_percent"]


def test_warmer_water_increases_detailed_flux():
    m, w, o, t = _base()
    w_cold = Water(**{**w.__dict__, "temperature_C": 10.0})
    w_warm = Water(**{**w.__dict__, "temperature_C": 35.0})
    r_cold = run_detailed_simulation(m, w_cold, o, t)
    r_warm = run_detailed_simulation(m, w_warm, o, t)
    assert r_warm["initial_flux_LMH"] > r_cold["initial_flux_LMH"]


def test_higher_recovery_increases_concentrate_concentration():
    m, w, o, t = _base()
    o_low_r = OperatingConditions(**{**o.__dict__, "recovery_percent": 10.0})
    o_high_r = OperatingConditions(**{**o.__dict__, "recovery_percent": 90.0})
    r_low = run_detailed_simulation(m, w, o_low_r, t)
    r_high = run_detailed_simulation(m, w, o_high_r, t)
    assert r_high["concentrate_mg_L"] > r_low["concentrate_mg_L"]


def test_100_percent_recovery_has_no_concentrate_and_does_not_crash():
    m, w, o, t = _base()
    o_full = OperatingConditions(**{**o.__dict__, "recovery_percent": 100.0})
    r = run_detailed_simulation(m, w, o_full, t)
    assert r["concentrate_mg_L"] is None


def test_zero_porosity_or_pore_size_skips_structural_flux_gracefully():
    m, w, o, t = _base()
    m_no_structure = Membrane(**{**m.__dict__, "porosity": 0.0, "pore_size_nm": 0.0})
    r = run_detailed_simulation(m_no_structure, w, o, t)
    assert r["structural_flux_LMH"] is None
    assert r["flux_LMH"] > 0  # main result still computed fine


def test_total_permeate_flow_scales_with_membrane_area():
    m, w, o, t = _base()
    o_small = OperatingConditions(**{**o.__dict__, "membrane_area_m2": 1.0})
    o_large = OperatingConditions(**{**o.__dict__, "membrane_area_m2": 10.0})
    r_small = run_detailed_simulation(m, w, o_small, t)
    r_large = run_detailed_simulation(m, w, o_large, t)
    assert r_large["total_permeate_flow_L_hr"] == pytest.approx(r_small["total_permeate_flow_L_hr"] * 10)


def test_detailed_simulation_rejects_same_invalid_inputs_as_simple():
    m, w, o, t = _base()
    w_bad_ph = Water(**{**w.__dict__, "pH": 20.0})
    with pytest.raises(ValueError):
        run_detailed_simulation(m, w_bad_ph, o, t)
    o_bad_recovery = OperatingConditions(**{**o.__dict__, "recovery_percent": 150.0})
    with pytest.raises(ValueError):
        run_detailed_simulation(m, w, o_bad_recovery, t)


def test_detailed_simulation_stress_test_over_full_random_input_space():
    """Run run_detailed_simulation over the same randomized parameter
    ranges used by simulation.generate_dataset, to catch any edge case
    (e.g. division by zero, domain errors) across realistic inputs."""
    rng = np.random.default_rng(7)
    t = Targets(95, 30, 20, 2, 10)
    failures = []
    for i in range(500):
        m = Membrane("Custom Polymer", str(rng.choice(["MF","UF","NF","RO"])),
                    float(rng.uniform(10, 500)), float(rng.uniform(1, 100)),
                    float(rng.uniform(1, 100)), float(rng.uniform(500, 100000)),
                    float(rng.uniform(.1, .8)), float(rng.uniform(-1, 1)), float(rng.uniform(0, 1)),
                    10000.0, float(rng.uniform(50, 99.9)), float(rng.uniform(.005, .15)))
        w = Water("Custom contaminant", float(rng.uniform(10, 1000)), float(rng.uniform(0, 100)),
                  float(rng.uniform(100, 3000)), float(rng.uniform(4, 10)), float(rng.uniform(0.5, 40)))
        o = OperatingConditions(float(rng.uniform(0.2, 20)), float(rng.uniform(1, 50)),
                                float(rng.uniform(.01, 1)), float(rng.uniform(5, 80)),
                                float(rng.uniform(.5, 24)), w.temperature_C,
                                float(rng.uniform(.1, 10)), 0.7, 8)
        try:
            r = run_detailed_simulation(m, w, o, t)
            assert 0 <= r["feasibility_score"] <= 100
            assert 0 <= r["rejection_percent"] <= 100
            assert r["flux_LMH"] >= 0
        except Exception as e:
            failures.append((i, type(e).__name__, str(e)))
    assert not failures, f"{len(failures)} failures out of 500 random trials: {failures[:5]}"
