"""Composite screening score and plain-language feasibility assessment.

`feasibility_score` combines five sub-scores (rejection, flux, fouling,
energy, cost), each mapped onto a 0-100 scale relative to the user's
targets, into one weighted overall score. It is a transparent, tunable
heuristic for ranking/screening candidates — not a calibrated predictor
of real-world success.
"""


def _bounded(x, lo=0, hi=100):
    """Clamp `x` to the [lo, hi] range."""
    return max(lo, min(hi, x))


def feasibility_score(rejection, flux, fouling_decline, energy, cost, targets,
                      weights=None):
    """Weighted 0-100 screening score plus its five sub-scores.

    For "higher is better" metrics (rejection, flux) the sub-score is
    `actual / target * 100`, capped at 100. For "lower is better"
    metrics (fouling decline, energy, cost) the sub-score is
    `target / actual * 100`, so meeting the target exactly scores 100
    and going over it scores less. Weights sum to 1.0 by default
    (0.35 + 0.25 + 0.20 + 0.10 + 0.10) and can be overridden by callers.
    """
    weights = weights or {"rejection":.35,"flux":.25,"fouling":.20,"energy":.10,"cost":.10}
    rs = _bounded(rejection / max(targets.min_rejection_percent, 1) * 100)
    fs = _bounded(flux / max(targets.min_flux_LMH, 1) * 100)
    fous = _bounded((1 - fouling_decline / max(targets.max_flux_decline_percent, 1)) * 100)
    es = _bounded(targets.max_energy_kWh_m3 / max(energy, 1e-9) * 100)
    cs = _bounded(targets.max_cost_per_m3 / max(cost, 1e-9) * 100)
    score = sum(weights[k]*v for k,v in {"rejection":rs,"flux":fs,"fouling":fous,"energy":es,"cost":cs}.items())
    return score, {"rejection":rs,"flux":fs,"fouling":fous,"energy":es,"cost":cs}


def feasibility_assessment(result, targets, calibration=None):
    """Plain-language recommendation plus a per-target pass/fail summary.

    `reliability` defaults to "LOW": with no calibration data, this MVP
    has no basis for a higher confidence label regardless of how many
    targets pass — a deliberate, conservative default consistent with
    the project's stated scientific-integrity stance (see README).

    `calibration`, if supplied, is the dict returned by
    `validation.calibration.calibrate_sample` (or an equivalent dict
    with a "post_fit_r2" key) for the sample this result came from.
    When present, reliability is upgraded based on how well the
    calibrated model actually fit real measurements:
      post_fit_r2 >= 0.9  -> "HIGH"
      post_fit_r2 >= 0.7  -> "MEDIUM"
      otherwise           -> "LOW" (a poor calibration fit is not
                              evidence of reliability, so it doesn't
                              earn an upgrade)
    This function itself never touches the database or fits anything —
    it only reads whatever calibration summary the caller passed in, so
    it stays a pure function.
    """
    checks = {
        "flux": result["flux_LMH"] >= targets.min_flux_LMH,
        "rejection": result["rejection_percent"] >= targets.min_rejection_percent,
        "fouling": result["flux_decline_percent"] <= targets.max_flux_decline_percent,
        "energy": result["energy_kWh_m3"] <= targets.max_energy_kWh_m3,
        "cost": result["cost_per_m3"] <= targets.max_cost_per_m3,
    }
    passed = sum(checks.values())
    if result["feasibility_score"] >= 80 and passed >= 4:
        rec = "STRONG CANDIDATE FOR PHYSICAL TESTING"
    elif result["feasibility_score"] >= 60:
        rec = "PROMISING BUT REQUIRES MORE DATA"
    elif result["flux_decline_percent"] > targets.max_flux_decline_percent:
        rec = "REQUIRES MODEL CALIBRATION"
    elif passed <= 1:
        rec = "LOW PRIORITY FOR PHYSICAL TESTING"
    else:
        rec = "INSUFFICIENT DATA"
    reason = f"{passed}/5 target checks pass; screening score is {result['feasibility_score']:.1f}/100."

    reliability = "LOW"
    if calibration is not None:
        r2 = calibration.get("post_fit_r2")
        if r2 is not None:
            if r2 >= 0.9:
                reliability = "HIGH"
            elif r2 >= 0.7:
                reliability = "MEDIUM"

    return {"recommendation": rec, "reason": reason, "target_checks": checks, "reliability": reliability}
