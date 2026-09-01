from models.feasibility import feasibility_score
from models.membrane import Targets
def test_score_range():
 t=Targets()
 s,_=feasibility_score(95,30,10,1,5,t)
 assert 0<=s<=100
