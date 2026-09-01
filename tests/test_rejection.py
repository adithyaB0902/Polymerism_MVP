from models.rejection import permeate_concentration

def test_rejection():
    assert abs(permeate_concentration(100, 90) - 10) < 1e-9
