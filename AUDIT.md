# IEEE RSM/ML Pipeline Audit

Phase 0 audit of the repository before the new pipeline additions. No
experimental result or paper number is inferred by this audit.

| # | Requirement | Status | File/function evidence |
|---:|---|---|---|
| 1 | BBD, coded quadratic RSM, ANOVA/LOF, R2/adjusted/predicted R2, adequate precision, stationary point, surfaces, diagnostics | EXISTS (correct) | `research/rsm/box_behnken.py:generate_box_behnken_design`, `to_coded`, `to_actual`; `research/rsm/quadratic_model.py:fit_quadratic`, `stationary_point`, `regression_diagnostics`; `research/rsm/anova.py:anova_table`, `lack_of_fit`; `research/rsm/surfaces.py:response_surface_grid` |
| 2 | Six-model comparison | PARTIAL | `research/ml.py:model_candidates` has linear, SVR, GPR, RF, shallow MLP and optional XGBoost, but fixed defaults rather than paper tuning |
| 3 | Nested 5-fold x 10-repeat CV and specified inner ranges | MISSING | Existing `research/ml.py:compare_models` uses `RepeatedKFold` without inner tuning |
| 4 | RSM scored on same outer folds as ML | MISSING | No shared outer-fold evaluator |
| 5 | Replicates/centre points grouped within folds | MISSING | No group-aware splitter in existing research ML code |
| 6 | Mean +/- SD R2, RMSE, MAE, AARD across folds/repeats | PARTIAL | `research/ml.py:compare_models` reports means and only R2 SD; AARD uses a separate KFold prediction |
| 7 | Nadeau-Bengio corrected resampled t-test | MISSING | No statistical comparison helper existed |
| 8 | SHAP and CV-RMSE permutation importance | PARTIAL | `research/ml.py:shap_values_if_available` is optional; `explain_model` uses in-sample R2 permutation, not CV-RMSE |
| 9 | GPR variance, tree bootstrap intervals, RSM 95% PI | PARTIAL | `ml/uncertainty.py:predict_with_uncertainty` provides tree spread only |
| 10 | Leverage OOD and factor-range warning | PARTIAL | `ml/ood.py:OODDetector` exists, but no integrated research-domain report |
| 11 | Differential evolution with top-N candidates | PARTIAL | `research/optimization.py:optimize_removal` returns one optimum |
| 12 | RSM desirability optimization | MISSING | No desirability helper |
| 13 | Boundary and +/-10% sensitivity checks | MISSING | No research optimization checks |
| 14 | Triplicate confirmation table empty until real data | PARTIAL | `research/confirmation.py:compare_confirmation` accepts supplied values but no persistent empty-first triplicate table |
| 15 | Isotherms and PFO/PSO kinetics with model comparison | PARTIAL | `research/adsorption.py:fit_isotherms` and `research/kinetics.py:fit_kinetics` exist, without integrated paper reporting |
| 16 | Regeneration/reuse cycle tracking with removal/desorption efficiency | PARTIAL | `research/regeneration.py:reuse_frame` stores cycles but does not calculate desorption efficiency |
| 17 | Research data provenance and analysis refusal for simulated rows | MISSING | `research_experiments` originally had no `data_source`; RSM/ML screens did not filter source |
| 18 | BBD measurement CSV with two replicates averaged into removal/qe | MISSING | Existing `research/experiments.py` accepted one `final_pb_mg_l` |
| 19 | One-command paper export | MISSING | No `research/paper_export.py` or CLI |
| 20 | Methods/code alignment | PARTIAL | Core RSM is implemented, but nested/grouped CV, formal uncertainty, provenance, desirability, confirmation workflow, and paper export were missing |

## Methods mismatches

1. Existing ML comparison used ordinary repeated K-fold CV, not nested CV with
   the paper's inner hyperparameter ranges.
2. Replicate groups were not preserved across folds.
3. RSM was not scored on the same outer folds as ML.
4. AARD used a different resampling scheme from the other metrics.
5. XGBoost was optional and omitted when unavailable.
6. Permutation importance was based on in-sample R2 instead of CV-RMSE increase.
7. Tree spread was a proxy, not a bootstrap interval; GPR and RSM intervals were absent.
8. Optimization returned one candidate without desirability, boundary, or sensitivity checks.
9. Confirmation had no persistent triplicate workflow.
10. Isotherm, kinetics, and regeneration helpers lacked integrated paper reporting.
11. Research rows had no enforced provenance source.
12. No reproducible paper-output bundle existed.

