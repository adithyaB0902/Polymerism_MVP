"""Reproducible paper tables and figures from supplied measurement data."""

import argparse
import json
from pathlib import Path

import pandas as pd

from .cv_nested import nested_compare, _candidates
from .experiments import read_bbd_measurement_template
from .rsm.quadratic_model import fit_quadratic, predict_quadratic
from .rsm.box_behnken import FACTORS


def export_paper_results(source, output_dir):
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    frame, report = read_bbd_measurement_template(source, getattr(source, "name", ""))
    if report["valid_rows"] < 16:
        raise ValueError("At least 16 complete measured runs are required for paper export.")
    fit = fit_quadratic(frame, "removal_percent", list(FACTORS))
    predictions = frame.copy()
    predictions["RSM"] = predict_quadratic(fit, frame)
    best_name = ""
    cv_table = pd.DataFrame()
    try:
        comparison, _ = nested_compare(
            frame, groups=frame["table_iv_run"], n_splits=5, n_repeats=1, max_grid_points=4
        )
        cv_table = comparison
        best_name = comparison.iloc[0]["model"]
        if best_name == "RSM quadratic":
            predictions["Best-ML"] = pd.NA
        else:
            model = _candidates(42)[best_name][0]
            model.fit(frame[list(FACTORS)], frame["removal_percent"])
            predictions["Best-ML"] = model.predict(frame[list(FACTORS)])
    except (ImportError, ValueError):
        predictions["Best-ML"] = pd.NA
    predictions.to_csv(output / "table_iv_design_and_predictions.csv", index=False)
    fit["anova_table"].to_csv(output / "table_v_anova.csv", index=False)
    fit["coefficients"].to_csv(output / "equation_12_coefficients.csv", index=False)
    pd.DataFrame([{"r2": fit["r2"], "adjusted_r2": fit["adjusted_r2"],
                   "predicted_r2": fit["predicted_r2"], "rmse": fit["rmse"],
                   "adequate_precision": fit["adequate_precision"],
                   "cv_percent": fit["cv_percent"]}]).to_csv(output / "rsm_fit_statistics.csv", index=False)
    rsm_row = {"model": "RSM", "R2": fit["r2"], "RMSE": fit["rmse"],
               "R2_SD": pd.NA, "RMSE_SD": pd.NA, "MAE": pd.NA, "MAE_SD": pd.NA,
               "AARD": pd.NA, "AARD_SD": pd.NA}
    cv_rows = cv_table.to_dict("records") if not cv_table.empty else []
    pd.DataFrame([rsm_row] + cv_rows).to_csv(output / "table_vi_cv_performance.csv", index=False)
    pd.DataFrame(columns=["predicted_removal", "measured_rep1", "measured_rep2", "measured_rep3"]).to_csv(
        output / "table_vii_optimum_and_confirmation.csv", index=False)
    (output / "summary_numbers.json").write_text(json.dumps({
        "n_measured_rows": int(report["valid_rows"]), "r2": fit["r2"],
        "adjusted_r2": fit["adjusted_r2"], "predicted_r2": fit["predicted_r2"],
    }, indent=2), encoding="utf-8")
    sources = frame.get("data_source", pd.Series(["experimental"] * len(frame))).value_counts().to_dict()
    (output / "provenance_report.txt").write_text(
        "data_source counts: " + json.dumps({str(k): int(v) for k, v in sources.items()}) + "\n",
        encoding="utf-8",
    )
    try:
        import matplotlib.pyplot as plt
        fig, axis = plt.subplots(figsize=(6, 4))
        axis.scatter(predictions["removal_percent"], predictions["RSM"])
        axis.set(xlabel="Measured removal (%)", ylabel="RSM prediction (%)")
        fig.tight_layout()
        fig.savefig(output / "fig7_parity_and_shap.png", dpi=160)
        plt.close(fig)
    except ImportError:
        pass
    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("measurement_csv")
    parser.add_argument("output_dir")
    args = parser.parse_args()
    export_paper_results(args.measurement_csv, args.output_dir)


if __name__ == "__main__":
    main()
