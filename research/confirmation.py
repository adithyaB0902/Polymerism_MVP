"""Predicted optimum versus supplied confirmation replicates."""

import numpy as np


def compare_confirmation(predicted_removal, experimental_replicates):
    values = np.asarray(experimental_replicates, float)
    if values.size < 1 or not np.isfinite(values).all():
        raise ValueError("At least one finite experimental confirmation value is required.")
    mean = float(values.mean())
    error = abs(mean - float(predicted_removal))
    return {"predicted_removal_percent": float(predicted_removal), "replicate_count": int(values.size),
            "experimental_mean_percent": mean, "experimental_std_percent": float(values.std(ddof=1)) if values.size > 1 else 0.0,
            "absolute_error_percent_points": float(error),
            "percentage_error": float(error / abs(predicted_removal) * 100) if predicted_removal else None}

