"""Four-factor Box-Behnken design with five centre points."""

from itertools import combinations, product
import pandas as pd

FACTORS = {
    "chitosan_wt_percent": (20.0, 40.0, 60.0),
    "biochar_wt_percent": (1.0, 3.0, 5.0),
    "pH": (3.0, 4.5, 6.0),
    "initial_pb_mg_l": (10.0, 30.0, 50.0),
}


def generate_box_behnken_design():
    names = list(FACTORS)
    rows = []
    for i, j in combinations(range(4), 2):
        for levels in product((-1, 1), repeat=2):
            coded = [0, 0, 0, 0]
            coded[i], coded[j] = levels
            rows.append(coded)
    rows.extend([[0, 0, 0, 0]] * 5)
    output = []
    for run, coded in enumerate(rows, 1):
        record = {"run_id": f"BBD-{run:02d}", "coded_" + names[0]: coded[0]}
        for name, level in zip(names, coded):
            low, center, high = FACTORS[name]
            record[name] = (center if level == 0 else high if level == 1 else low)
            record["coded_" + name] = level
        record["experimental_removal_percent"] = None
        output.append(record)
    return pd.DataFrame(output)


generate_bbd = generate_box_behnken_design
