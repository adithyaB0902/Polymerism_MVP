"""Standard regression metrics for comparing simulator predictions
against measured experimental values."""

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def compare_predictions(predicted, measured):
    """Return {"MAE":..., "RMSE":..., "R2":...} comparing two equal-length
    arrays of predicted vs. measured values."""
    p = np.asarray(predicted)
    y = np.asarray(measured)
    return {"MAE": mean_absolute_error(y, p), "RMSE": np.sqrt(mean_squared_error(y, p)), "R2": r2_score(y, p)}
