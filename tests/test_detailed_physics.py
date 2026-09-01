import pytest

from models.viscosity import water_viscosity_Pa_s, temperature_correction_factor
from models.pore_flow import pore_flow_flux_LMH, pore_flow_permeance_LMH_bar
from models.concentration_polarization import observed_rejection_percent, mass_transfer_coefficient_m_s
from models.mass_balance import concentrate_concentration, permeate_normalized_energy_kwh_m3, total_permeate_flow_L_hr
from models.fouling_index import fouling_propensity_multiplier, adjusted_fouling_coefficient, MIN_MULTIPLIER, MAX_MULTIPLIER


def test_viscosity_matches_known_reference_values():
    # within ~2.5% of standard published water viscosity values across 0-100 C
    refs = {0: 1.792, 20: 1.002, 25: 0.890, 60: 0.467, 100: 0.282}
    for T, ref_cP in refs.items():
        computed_cP = water_viscosity_Pa_s(T) * 1000
        assert abs(computed_cP - ref_cP) / ref_cP < 0.03


def test_viscosity_rejects_out_of_range_temperature():
    with pytest.raises(ValueError):
        water_viscosity_Pa_s(-5)
    with pytest.raises(ValueError):
        water_viscosity_Pa_s(150)


def test_temperature_correction_factor_direction():
    # warmer than reference -> less viscous -> flux boosted -> factor > 1
    assert temperature_correction_factor(35, reference_temperature_C=25) > 1
    # colder than reference -> more viscous -> flux reduced -> factor < 1
    assert temperature_correction_factor(10, reference_temperature_C=25) < 1
    # equal to reference -> no correction
    assert abs(temperature_correction_factor(25, reference_temperature_C=25) - 1.0) < 1e-9


def test_pore_flow_scaling_relationships():
    mu = water_viscosity_Pa_s(20.0)
    J = pore_flow_flux_LMH(porosity=0.35, pore_size_nm=20, thickness_um=100, viscosity_Pa_s=mu, TMP_bar=2.0)
    # flux ~ 1/thickness
    J_2x_thick = pore_flow_flux_LMH(porosity=0.35, pore_size_nm=20, thickness_um=200, viscosity_Pa_s=mu, TMP_bar=2.0)
    assert abs(J_2x_thick - J / 2) < 1e-9
    # flux ~ pore_radius^2, i.e. pore_size^2 (diameter squared)
    J_2x_pore = pore_flow_flux_LMH(porosity=0.35, pore_size_nm=40, thickness_um=100, viscosity_Pa_s=mu, TMP_bar=2.0)
    assert abs(J_2x_pore - J * 4) < 1e-6
    # flux ~ pressure (linear)
    J_2x_p = pore_flow_flux_LMH(porosity=0.35, pore_size_nm=20, thickness_um=100, viscosity_Pa_s=mu, TMP_bar=4.0)
    assert abs(J_2x_p - J * 2) < 1e-9


def test_pore_flow_rejects_invalid_inputs():
    with pytest.raises(ValueError):
        pore_flow_permeance_LMH_bar(porosity=0, pore_size_nm=20, thickness_um=100, viscosity_Pa_s=0.001)
    with pytest.raises(ValueError):
        pore_flow_permeance_LMH_bar(porosity=0.35, pore_size_nm=-1, thickness_um=100, viscosity_Pa_s=0.001)


def test_concentration_polarization_directions():
    base = observed_rejection_percent(95.0, flux_LMH=50.0, crossflow_velocity_m_s=0.2)
    higher_crossflow = observed_rejection_percent(95.0, flux_LMH=50.0, crossflow_velocity_m_s=1.0)
    higher_flux = observed_rejection_percent(95.0, flux_LMH=200.0, crossflow_velocity_m_s=0.2)
    assert higher_crossflow > base > higher_flux
    assert observed_rejection_percent(95.0, flux_LMH=0.0, crossflow_velocity_m_s=0.2) == pytest.approx(95.0)
    # never crashes and stays in [0, 100] even at extremes
    r = observed_rejection_percent(95.0, flux_LMH=1e6, crossflow_velocity_m_s=0.0)
    assert 0 <= r <= 100


def test_mass_transfer_coefficient_positive_and_increasing():
    k1 = mass_transfer_coefficient_m_s(0.1)
    k2 = mass_transfer_coefficient_m_s(0.5)
    assert 0 < k1 < k2


def test_mass_balance_closes():
    Cf, Cp, r = 100.0, 5.0, 20.0
    Cc = concentrate_concentration(Cf, Cp, r)
    assert Cp * (r / 100) + Cc * (1 - r / 100) == pytest.approx(Cf)


def test_mass_balance_concentration_increases_with_recovery():
    cc_low = concentrate_concentration(100, 5, 10)
    cc_high = concentrate_concentration(100, 5, 90)
    assert cc_high > cc_low > 100


def test_permeate_normalized_energy_scales_inversely_with_recovery():
    e_low = permeate_normalized_energy_kwh_m3(1.0, 10)
    e_high = permeate_normalized_energy_kwh_m3(1.0, 90)
    assert e_low > e_high


def test_total_permeate_flow_scales_with_area():
    assert total_permeate_flow_L_hr(50, 2.0) == 100.0
    assert total_permeate_flow_L_hr(50, 0) == 0.0


def test_fouling_index_reference_conditions_are_neutral():
    m = fouling_propensity_multiplier(turbidity_NTU=5.0, TDS_mg_L=500.0, hydrophilicity=0.5, surface_charge=0.0)
    assert m == pytest.approx(1.0)


def test_fouling_index_stays_within_bounds():
    for turb in [0, 5, 1e6]:
        for tds in [0, 500, 1e6]:
            for hydro in [0, 0.5, 1]:
                for charge in [-1, 0, 1]:
                    m = fouling_propensity_multiplier(turb, tds, hydro, charge)
                    assert MIN_MULTIPLIER <= m <= MAX_MULTIPLIER


def test_adjusted_fouling_coefficient_scales_base():
    k_adj = adjusted_fouling_coefficient(0.03, turbidity_NTU=5.0, TDS_mg_L=500.0, hydrophilicity=0.5, surface_charge=0.0)
    assert k_adj == pytest.approx(0.03)  # reference conditions -> unchanged
