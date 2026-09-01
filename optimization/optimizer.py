"""Ranking and Pareto-frontier utilities for a generated candidate dataset."""

import numpy as np


def optimize_candidates(df, n=10):
    """Return the top `n` rows by `feasibility_score`, highest first."""
    return df.sort_values("feasibility_score", ascending=False).head(n).reset_index(drop=True)


def pareto_frontier(df):
    """Return the Pareto-optimal rows for (maximize flux_LMH, maximize
    rejection_percent): every row for which no other row is at least as
    good on both metrics and strictly better on at least one.

    Vectorized with numpy broadcasting rather than a nested
    `for row in df.iterrows(): for other in df.iterrows(): ...` loop —
    the row-by-row version is O(n^2) *with* pandas' slow row-iteration
    overhead on top, which takes several seconds on a 1000-row dataset;
    this version does the same O(n^2) comparisons as one vectorized
    boolean-matrix operation, which is both much faster and, arguably,
    easier to read as "for each row, is any other row weakly-better in
    both dimensions and strictly-better in at least one?" in a single
    expression rather than an explicit double loop.
    """
    a = df.reset_index(drop=True)
    flux = a["flux_LMH"].to_numpy()
    rej = a["rejection_percent"].to_numpy()

    # weakly_better[i, j] = True if row j is >= row i on both metrics
    # strictly_better[i, j] = True if row j is > row i on at least one metric
    weakly_better = (flux[None, :] >= flux[:, None]) & (rej[None, :] >= rej[:, None])
    strictly_better = (flux[None, :] > flux[:, None]) | (rej[None, :] > rej[:, None])
    dominated_by = weakly_better & strictly_better
    np.fill_diagonal(dominated_by, False)  # a row never dominates itself

    is_dominated = dominated_by.any(axis=1)
    return a.loc[~is_dominated].sort_values("flux_LMH")
