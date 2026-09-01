"""Load and lightly validate a user-supplied experimental CSV for
comparison against simulator predictions.
"""

import pandas as pd

# All of these columns must be present for a row to be usable in a
# prediction-vs-measurement comparison (see models/comparison.py). The
# previous name here, REQUIRED_ANY, read as "at least one of these is
# required" (an OR condition) but the actual check below is an AND —
# every column in the list is checked for presence and completeness.
REQUIRED_COLUMNS = ["flux_LMH", "rejection_percent"]


def validate_experimental_csv(source):
    """Read `source` as a CSV and report how many rows have non-null
    values in every column of REQUIRED_COLUMNS.

    Returns (df, report) where report has:
      rows_uploaded             -- total rows in the file
      valid_rows                -- rows with all required columns present and non-null
                                    (0 if any required column is missing entirely)
      missing_required_columns  -- which required columns are absent from the file
    """
    df = pd.read_csv(source)
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    valid = 0 if missing else len(df.dropna(subset=REQUIRED_COLUMNS))
    report = {"rows_uploaded": len(df), "valid_rows": valid, "missing_required_columns": missing}
    return df, report
