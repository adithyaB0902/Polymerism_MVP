"""Membrane regeneration and reuse records."""

import pandas as pd


def reuse_frame(records):
    columns = ["membrane_id", "cycle", "regeneration_method", "removal_percent", "qe_mg_g", "notes"]
    frame = pd.DataFrame(records)
    for col in columns:
        if col not in frame:
            frame[col] = None
    return frame[columns].sort_values(["membrane_id", "cycle"]).reset_index(drop=True)

