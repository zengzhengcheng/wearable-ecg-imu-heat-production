# Results and figure map

| Manuscript item | Deposited source | Reproduction |
|---|---|---|
| Formal 30-min analysis panel | `data/analysis/modeling_panel_30min.csv` | Shape and key checks in script 03 |
| Pig-held-out outer folds | `reference_results/outer_fold_assignments.csv` | Cross-checked against every prediction row |
| Chamber-informed full-period internal-validation R² = 0.675 | all 4,079 rows, `predicted_HP_raw` | Figure 3A |
| Chamber-informed post-adaptation internal-validation R² = 0.710 | 2,470 follow-up rows, `predicted_HP_raw` | Figure 3B |
| Fixed first-2-d post-prediction calibrated R² = 0.800 | same 2,470 rows, `predicted_HP_calibrated` | Figure 3C |
| Traditional baseline R² = 0.432 | heart rate + ODBA Ridge in `baseline_followup_results.csv` | Figure 4 |
| Feeding-information follow-up | `feeding_followup_results.csv` | Metrics written by script 03 |
| Experimental/CV design | `outer_fold_assignments.csv` plus formal phase definitions | Figure 2 |
| Coverage of 24 pig-by-period experimental units over 118 cumulative monitoring days (40 calendar dates; experimental day = 09:00 to the following 09:00) | `data/analysis/modeling_panel_30min.csv` | Figure S1 |
| Representative ECG, software-detected R peaks and simultaneous IMU | `reference_results/supplementary/signal_quality_*.csv` plus candidate-review assets | Figure S2 |
| Data-screening counts from 34,109 records to 4,079 windows | `reference_results/supplementary/screening_counts.csv` | Figure S3 |
| Actual locked fold components and X columns | `reference_results/locked_model_components_by_fold.csv` and `locked_model_features_by_fold.csv` | Methods/input audit |

`scripts/03_reproduce_results_and_figures.py` asserts the locked numerical values before saving Figures 2–4. It reads only deposited panel/result assets and does not rerun model selection, training or calibration-strength search.

The locked model is chamber-informed: ECG/HRV, IMU, body weight, calorimetry-related time/window fields, chamber identity, chamber temperature and relative humidity are among its actual inputs. The 0.800 result is obtained by applying the frozen first-2-d calibration formula to predictions, not by retraining the model. All three values are pig-grouped internal-validation results from the A1/B1/B2 facility and are not cross-facility validation results.

`scripts/04_reproduce_supplementary_figures.py` verifies the compact supplementary source files and saves Figures S1–S3. Software-generated R peaks are explicitly treated as detector output, not human annotation or a gold standard. Superseded Optuna and date-holdout supplementary figures are not part of this release.
