"""Suggests which candidates from a generated dataset are most worth
following up on: promising `feasibility_score`, weighted slightly toward
candidates sitting further from the "typical" (median) point in the
parameter space already explored, on the theory that those are the ones
providing the most new information.
"""

import pandas as pd


def recommend_experiments(df, top_n=5):
    d = df.copy()
    score = d["feasibility_score"].rank(pct=True)

    # "Uncertainty" proxy: normalized Euclidean distance from the median
    # of each available feature column (a simple stand-in for "how
    # unexplored is this region of parameter space", not a real
    # model-uncertainty estimate).
    cols = [c for c in ["thickness_um", "permeability_LMH_bar", "op_TMP_bar", "water_temperature_C"] if c in d]
    if cols:
        distance = ((d[cols] - d[cols].median()) / (d[cols].std() + 1e-9)).pow(2).sum(axis=1).pow(.5)
    else:
        # None of the expected columns are present (e.g. a custom
        # dataframe was passed in) — fall back to "no information about
        # distance" rather than crashing. Must be a Series aligned to
        # `d`'s index (not a bare 0) so `.rank(pct=True)` below works.
        distance = pd.Series(0.0, index=d.index)

    d["recommendation_score"] = score + 0.15 * distance.rank(pct=True)
    out = d.sort_values("recommendation_score", ascending=False).head(top_n).copy()
    out["reason"] = "Promising predicted performance with unexplored parameter-space distance."
    return out
