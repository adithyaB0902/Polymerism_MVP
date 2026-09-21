"""User-supplied adsorbent comparison records."""

import pandas as pd


COLUMNS = ["adsorbent", "qmax_mg_g", "pH", "reference"]


def comparison_table(records):
    """Normalize user-entered comparison rows without inventing references or values."""
    frame = pd.DataFrame(records)
    for column in COLUMNS:
        if column not in frame:
            frame[column] = None
    frame["qmax_mg_g"] = pd.to_numeric(frame["qmax_mg_g"], errors="coerce")
    frame["pH"] = pd.to_numeric(frame["pH"], errors="coerce")
    return frame[COLUMNS].sort_values("qmax_mg_g", ascending=False, na_position="last").reset_index(drop=True)