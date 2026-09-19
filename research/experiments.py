"""Experimental Pb(II) dataset schema, import, and non-destructive validation."""

from io import BytesIO
import pandas as pd

from .pb_removal import calculate_pb_metrics

EXPERIMENT_COLUMNS = [
    "experiment_id", "membrane_id", "chitosan_wt_percent", "cellulose_acetate_wt_percent",
    "biochar_wt_percent", "pH", "initial_pb_mg_l", "final_pb_mg_l", "contact_time_min",
    "solution_volume_l", "membrane_mass_g", "removal_percent", "qe_mg_g", "replicate_number", "notes",
]
REQUIRED_INPUT_COLUMNS = [
    "experiment_id", "chitosan_wt_percent", "biochar_wt_percent", "pH",
    "initial_pb_mg_l", "final_pb_mg_l", "solution_volume_l", "membrane_mass_g",
]


def validate_experiment_frame(frame, calculate_missing=True):
    """Return a cleaned copy and a report; never mutates the caller's frame."""
    df = frame.copy()
    missing = [c for c in REQUIRED_INPUT_COLUMNS if c not in df.columns]
    errors = []
    invalid_indices = set()
    if missing:
        return df, {"valid_rows": 0, "rows": len(df), "missing_columns": missing, "errors": ["Required columns missing."]}
    numeric = [c for c in REQUIRED_INPUT_COLUMNS if c != "experiment_id"]
    for col in numeric:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    for i, row in df.iterrows():
        if row[numeric].isna().any():
            errors.append(f"row {i}: missing or non-numeric input")
            invalid_indices.add(i)
            continue
        try:
            metrics = calculate_pb_metrics(row["initial_pb_mg_l"], row["final_pb_mg_l"],
                                           row["solution_volume_l"], row["membrane_mass_g"])
            if calculate_missing or pd.isna(row.get("removal_percent", float("nan"))):
                df.loc[i, "removal_percent"] = metrics["removal_percent"]
            if calculate_missing or pd.isna(row.get("qe_mg_g", float("nan"))):
                df.loc[i, "qe_mg_g"] = metrics["qe_mg_g"]
        except (TypeError, ValueError) as exc:
            errors.append(f"row {i}: {exc}")
            invalid_indices.add(i)
    valid = len(df) - len(invalid_indices)
    return df, {"valid_rows": valid, "rows": len(df), "missing_columns": [], "errors": errors,
                "valid_indices": [i for i in df.index if i not in invalid_indices]}


def read_experimental_data(source, filename=""):
    """Read CSV or Excel using installed pandas engines and validate it."""
    if hasattr(source, "read"):
        raw = source.read()
        source = BytesIO(raw)
    is_excel = str(filename or getattr(source, "name", "")).lower().endswith((".xls", ".xlsx"))
    frame = pd.read_excel(source) if is_excel else pd.read_csv(source)
    return validate_experiment_frame(frame)
