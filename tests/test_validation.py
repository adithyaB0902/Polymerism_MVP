import io
from validation.experimental_data import validate_experimental_csv
def test_validation():
 f=io.StringIO("flux_LMH,rejection_percent\n20,90\n")
 df,r=validate_experimental_csv(f)
 assert r["valid_rows"]==1
