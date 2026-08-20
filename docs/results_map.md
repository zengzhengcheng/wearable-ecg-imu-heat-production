# Results and figure map

| Manuscript item | Deposited source | Reproduction |
|---|---|---|
| Formal 30-min analysis panel | `data/analysis/modeling_panel_30min.csv` | Shape and key checks in script 03 |
| Pig-held-out outer folds | `reference_results/outer_fold_assignments.csv` | Cross-checked against every prediction row |
| Traditional deployment R² = 0.675 | all 4,079 rows, `predicted_HP_raw` | Figure 3A |
| Uncalibrated wearable R² = 0.710 | 2,470 follow-up rows, `predicted_HP_raw` | Figure 3B |
| Fixed first-2-d calibrated R² = 0.800 | same 2,470 rows, `predicted_HP_calibrated` | Figure 3C |
| Traditional baseline R² = 0.432 | heart rate + ODBA Ridge in `baseline_followup_results.csv` | Figure 4 |
| Feeding-information follow-up | `feeding_followup_results.csv` | Metrics written by script 03 |
| Experimental/CV design | `outer_fold_assignments.csv` plus formal phase definitions | Figure 2 |
| Representative ECG, detected R peaks and simultaneous IMU | `reference_results/supplementary/signal_quality_*.csv` | Figure S1 |
| Coverage of 24 pig-by-period units over 40 strict days | `data/analysis/modeling_panel_30min.csv` | Figure S2 |
| Data-screening counts from 34,109 records to 4,079 windows | `reference_results/supplementary/screening_counts.csv` | Figure S3 |

`scripts/03_reproduce_results_and_figures.py` asserts the locked numerical values before saving Figures 2–4. It reads only deposited panel/result assets and does not rerun model selection, training or calibration-strength search.

`scripts/04_reproduce_supplementary_figures.py` verifies the compact supplementary source files and saves Figures S1–S3. Software-generated R peaks are explicitly treated as detector output, not human annotation or a gold standard. Superseded Optuna and date-holdout supplementary figures are not part of this release.
