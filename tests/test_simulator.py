from models.membrane import *
from models.simulator import run_simulation
def test_simulation():
 r=run_simulation(Membrane("x","UF",100,10,baseline_rejection_percent=95,fouling_coefficient=.1),
                   Water(),OperatingConditions(TMP_bar=2),Targets())
 assert r["flux_LMH"]>0 and 0<=r["feasibility_score"]<=100
