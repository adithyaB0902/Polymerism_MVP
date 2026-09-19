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
        record = {"run_id": f"BBD-{run:02d}"}
        for name, level in zip(names, coded):
            low, center, high = FACTORS[name]
            record["coded_" + name] = level
            record[name] = center if level == 0 else high if level == 1 else low
        record["experimental_removal_percent"] = None
        output.append(record)
    return pd.DataFrame(output)


generate_bbd = generate_box_behnken_design


def to_coded(frame, factors=None, levels=None):
    """Actual factor values -> coded units (-1, 0, +1 at low, centre, high).

    ``levels`` maps factor -> (low, centre, high) and defaults to ``FACTORS``.
    """
    levels = levels or FACTORS
    factors = list(factors) if factors is not None else list(levels)
    coded = pd.DataFrame(index=frame.index)
    for name in factors:
        low, center, high = levels[name]
        coded[name] = (frame[name].astype(float) - center) / ((high - low) / 2.0)
    return coded


def to_actual(coded, levels=None):
    """Coded units -> actual factor values (inverse of ``to_coded``)."""
    levels = levels or FACTORS
    actual = pd.DataFrame(index=coded.index)
    for name in coded.columns:
        low, center, high = levels[name]
        actual[name] = center + coded[name].astype(float) * (high - low) / 2.0
    return actual
