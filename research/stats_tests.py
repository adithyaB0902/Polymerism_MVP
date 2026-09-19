"""Statistical comparisons for repeated cross-validation results."""

import numpy as np
from scipy import stats


def corrected_resampled_t_test(errors_a, errors_b, n_train, n_test):
    """Nadeau-Bengio corrected resampled t-test for paired test errors."""
    a, b = np.asarray(errors_a, float), np.asarray(errors_b, float)
    if a.shape != b.shape or a.size < 2:
        raise ValueError("Two equal-length error arrays with at least two observations are required.")
    differences = a - b
    variance = np.var(differences, ddof=1)
    corrected_variance = (1.0 / differences.size + n_test / n_train) * variance
    t_value = float(np.mean(differences) / np.sqrt(corrected_variance)) if corrected_variance else 0.0
    p_value = float(2 * stats.t.sf(abs(t_value), differences.size - 1))
    return {"t": t_value, "p_value": p_value, "mean_difference": float(np.mean(differences)),
            "degrees_of_freedom": differences.size - 1}
