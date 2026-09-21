"""Small, reproducible utilities shared by the research UI and exports."""

import json
from io import BytesIO
from pathlib import Path

import numpy as np
import pandas as pd

from .stats_tests import corrected_resampled_t_test


def paired_model_tests(predictions, model_a, model_b, n_train, n_test):
    """Compare two models on the same repeated-CV rows with Nadeau-Bengio correction."""
    required = {"model", "y", "prediction"}
    if not required.issubset(predictions.columns):
        raise ValueError("predictions must contain model, y, and prediction columns.")
    left = predictions[predictions["model"].eq(model_a)].reset_index(drop=True)
    right = predictions[predictions["model"].eq(model_b)].reset_index(drop=True)
    if len(left) != len(right):
        raise ValueError("Models must have predictions for the same folds.")
    left_error = np.concatenate([np.asarray(y) - np.asarray(p) for y, p in zip(left.y, left.prediction)])
    right_error = np.concatenate([np.asarray(y) - np.asarray(p) for y, p in zip(right.y, right.prediction)])
    result = corrected_resampled_t_test(left_error, right_error, n_train, n_test)
    return {"model_a": model_a, "model_b": model_b, **result}


def importance_table(permutation, shap_values=None, feature_names=None):
    """Align permutation importance with optional SHAP mean absolute importance."""
    table = pd.DataFrame(permutation).copy()
    if feature_names is not None and "feature" not in table:
        table.insert(0, "feature", list(feature_names))
    if shap_values is not None:
        values = np.asarray(shap_values.values if hasattr(shap_values, "values") else shap_values)
        table["shap_mean_abs"] = np.mean(np.abs(values), axis=0)
    return table.sort_values("importance_mean", ascending=False).reset_index(drop=True)


def export_dataframe(frame, stem, output_dir, excel=True):
    """Write a table as CSV and, when requested, Excel; return created paths."""
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    paths = [output / f"{stem}.csv"]
    frame.to_csv(paths[0], index=False)
    if excel:
        excel_path = output / f"{stem}.xlsx"
        frame.to_excel(excel_path, index=False)
        paths.append(excel_path)
    return paths


def figure_png_bytes(figure, dpi=300):
    """Serialize a Matplotlib figure as a high-resolution PNG download."""
    output = BytesIO()
    figure.savefig(output, format="png", dpi=dpi, bbox_inches="tight")
    output.seek(0)
    return output.getvalue()


def build_run_report(data_source, seed, settings, results, versions=None):
    """Return a JSON-serializable provenance report for one analysis run."""
    return {
        "data_source": data_source,
        "seed": int(seed),
        "settings": settings,
        "results": results,
        "library_versions": versions or {},
    }


def report_json_bytes(report):
    return json.dumps(report, indent=2, default=str).encode("utf-8")