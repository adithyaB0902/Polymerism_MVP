from models.fouling import flux_decline_percent
def test_fouling(): assert flux_decline_percent(100,.1,2)[0] < 100
