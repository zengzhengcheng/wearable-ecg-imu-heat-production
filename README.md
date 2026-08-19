# Wearable ECG–IMU sensing for pig heat-production prediction

Code accompanying **“Wearable ECG–IMU sensing: Predicting heat production in growing pigs with short-term calorimetry calibration.”** The repository builds synchronized 30-min ECG, IMU, feeding-context and indirect-calorimetry windows; evaluates models with pig-grouped nested cross-validation; applies the paper's fixed first-2-d calibration; and reproduces the main numerical results and Figures 2–4 from saved out-of-fold predictions.

Raw data are intentionally excluded. Put data into the layout in [`data/README.md`](data/README.md); the field-level contract is in [`docs/input_data_contract.md`](docs/input_data_contract.md).

## Analysis protocol

- Outer evaluation: five-fold grouping by true animal identity (`pig`); no animal appears in both training and test parts of a fold.
- Inner selection: three-fold pig-grouped cross-validation within each outer-training set.
- Target: heat production normalized by metabolic body weight during model fitting and returned to kcal for reporting.
- Individual calibration: for each held-out pig/experimental unit, compute `r = mean(observed HP) / mean(predicted HP)` over its first two calorimetry days, then correct later fed-phase predictions as `prediction × [1 + 0.5 × (r − 1)]`. Calibration rows and fasting rows remain unmodified.
- Dietary treatment code is retained for bookkeeping, not entered as an ordinal numerical predictor. Planned feed represents offered feed, not measured intake.

## Installation

```bash
conda env create -f environment.yml
conda activate pig-hp
```

## Run order

```bash
python src/extract_heat_production.py
python src/prepare_ecg.py
python src/extract_sensor_features.py
python scripts/01_build_30min_panel.py
python scripts/02_run_nested_pig_cv.py --mode evaluate --scenario GKF_Pig
python scripts/03_reproduce_results_and_figures.py
```

Paths can be supplied with the documented command-line arguments or environment variables. The model step is computationally intensive. The figure step consumes the formal saved out-of-fold files and verifies the locked manuscript values; it does not retrain or search models.

## Locked manuscript results

The deposited analysis corresponds to pig-held-out R² values of 0.675 (traditional deployment), 0.710 (wearable model without individual calibration), and 0.800 (wearable model after fixed first-2-d calibration). The traditional baseline reported in the manuscript is R² = 0.432. These reference values are recorded in [`reference_results/locked_metrics.csv`](reference_results/locked_metrics.csv); example files are schema-only and are never used to create replacement performance estimates.

## Entry points

- `src/extract_heat_production.py`: extracts timestamped HP observations from chamber workbooks.
- `src/prepare_ecg.py`: consolidates R peaks and computes HR/HRV features.
- `src/extract_sensor_features.py`: constructs aligned 5-min sensor/HP records.
- `scripts/01_build_30min_panel.py`: aggregates the audited 30-min analysis panel and feeding-event features.
- `scripts/02_run_nested_pig_cv.py`: runs the formal pig-grouped nested model and writes out-of-fold predictions.
- `scripts/03_reproduce_results_and_figures.py`: verifies the three-stage results and creates Figures 2–4.

## Citation and license

See [`CITATION.cff`](CITATION.cff). Code is released under the MIT License. Raw study data will be deposited separately subject to the paper's data-release process.
