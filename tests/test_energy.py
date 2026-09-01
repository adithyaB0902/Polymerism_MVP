from models.energy import hydraulic_energy_kwh_m3
def test_energy_positive(): assert hydraulic_energy_kwh_m3(2,10,.7)>0
