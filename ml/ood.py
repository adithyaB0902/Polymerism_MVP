"""Simple out-of-distribution (OOD) flag: is a new input outside the
range seen in a reference dataset for any of its features?

This is a min/max range check, not a density- or distance-based OOD
score — a point can sit well inside every feature's individual range
and still be in a combination of values that never actually occurred in
the reference data (e.g. very thin membranes at very high permeability
together). It is intended as a cheap, easy-to-explain first pass, not a
rigorous novelty detector.
"""

import pandas as pd


class OODDetector:
    def __init__(self, reference_df, features):
        """`reference_df` supplies the min/max range for each column in
        `features` (typically the dataset used to train the ML models)."""
        self.reference_df = reference_df
        self.features = features

    def check(self, row):
        """Return {"ood": bool, "features": {feature: bool}} — whether
        `row` falls outside the reference range on each feature, and
        overall (True if any single feature is out of range)."""
        flags = {}
        for f in self.features:
            lo = self.reference_df[f].min()
            hi = self.reference_df[f].max()
            flags[f] = bool(row[f] < lo or row[f] > hi)
        return {"ood": any(flags.values()), "features": flags}
