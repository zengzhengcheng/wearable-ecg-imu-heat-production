# Deposited reference results

- `outer_fold_assignments.csv`: one row per true pig, giving its held-out outer fold.
- `formal_window_predictions.csv`: 4,079 formal test-pig windows. `predicted_HP_raw` supports the all-stage R² = 0.675 and 2,470-window post-adaptation R² = 0.710. Within those 2,470 windows, the fixed first-2-d individual factor is applied only to the 1,718 fed measurement windows in phases 2–3 (888 + 830); all 752 phase-4 fasting predictions remain equal to the raw predictions. This manuscript-defined partial calibration supports R² = 0.800.
- `three_stage_main_metrics.csv`: exact three-stage metrics before manuscript rounding.
- `baseline_followup_results.csv`: saved predictions from parsimonious baselines on the same 2,470 follow-up windows. The traditional baseline is the heart-rate + ODBA Ridge column (R² = 0.432).
- `feeding_followup_results.csv`: saved feeding-information comparison on the same 2,470 windows. Planned feed means offered quantity, and protocol meal timing is not measured intake.
- `locked_metrics.csv`: manuscript-rounded values asserted by the reproduction script.
- `supplementary/`: coverage, objectively selected representative signal data, and screening assets used for Figures S1–S3. Figure S1 is derived from the formal panel; Figure S2 uses deposited compact ECG/R-peak/IMU source files.

`scripts/03_reproduce_results_and_figures.py` reads these files directly, recalculates the metrics, verifies fixed `alpha = 0.5`, checks fold assignments and creates Figures 2–4 without model fitting or parameter search.

`scripts/04_reproduce_supplementary_figures.py` verifies and creates Figures S1–S3 from deposited assets. Its R-peak markers are software detections rather than manually adjudicated labels.
