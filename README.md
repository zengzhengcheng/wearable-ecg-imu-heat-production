# Wearable ECG–IMU sensing for pig heat-production prediction

Code and compact analysis assets accompanying **“Wearable ECG–IMU sensing: Predicting heat production in growing pigs with short-term calorimetry calibration.”**

The study combines ECG, 10 Hz IMU and open-circuit respiration-calorimetry measurements from 12 growing pigs to predict heat production (HP) in 30-min analysis windows.

## Data and code locations

- **ScienceDB** ([dataset DOI](https://doi.org/10.57760/sciencedb.00zyg)) stores the large raw or near-raw acquisitions, extracted sensor data, 5-min calorimetry records, experiment metadata and quality-control/lineage documentation.
- **This GitHub repository** stores the compact 4,079-row formal modeling panel, saved out-of-fold predictions, outer-fold assignments, locked result assets, analysis code and figure-reproduction workflow.

The approximately 103 GiB raw acquisition files are not duplicated here. The deposited compact table is at [`data/analysis/modeling_panel_30min.csv`](data/analysis/modeling_panel_30min.csv), and its column-level dictionary is at [`data/analysis/modeling_data_dictionary.csv`](data/analysis/modeling_data_dictionary.csv).

## Two reproducibility routes

### Reproduce the paper results directly

Install the environment and run the saved-prediction workflow:

```bash
conda env create -f environment.yml
conda activate pig-hp
python scripts/03_reproduce_results_and_figures.py
python scripts/04_reproduce_supplementary_figures.py
```

This verifies the four locked R² values and writes main Figures 2–4 plus supplementary Figures S1–S3 under `results/reproduced/`. It does not retrain, search models or tune calibration.

### Rebuild from source measurements

Download the ScienceDB deposit, perform format conversion and automated quality checks with the recommended **SwineSync OpenSource v1.1.1** companion release, then run the extraction and panel-building entries. This release accepts only the new-device tab-separated `ECGRawData`, `ECGHeartRateData`, and `IMUData` inputs; legacy `Raw`, `BMD`, and `Angle` formats are not supported.

```bash
python src/extract_heat_production.py
python src/prepare_ecg.py
python src/extract_sensor_features.py
python scripts/01_build_30min_panel.py
```

The optional formal nested training entry reads the repository panel by default:

```bash
python scripts/02_run_nested_pig_cv.py --mode validate
```

Model evaluation is grouped by true pig identity. Do not substitute a date holdout/GKF-Day analysis for the locked manuscript protocol.

## Analysis protocol

- Outer evaluation: five folds grouped by true animal identity (`pig`); no animal appears in both training and test parts of a fold.
- Inner selection: three-fold pig-grouped cross-validation within each outer-training set.
- Fit target: HP normalized by metabolic body weight (`HP_per_W075`).
- Reported target: `HP_kcal` is the mean 5-min HP quantity represented by each 30-min window and is multiplied by 6 for reporting in kcal per 30 min. The public saved prediction file already contains reported 30-min values.
- Individual calibration: for each held-out experimental unit, the ratio of mean observed to mean predicted HP over phases 0–1 (first 2 d) is shrunk with fixed `alpha = 0.5`; only the 1,718 fed measurement windows in phases 2–3 (888 + 830) are adjusted, while all 752 phase-4 fasting predictions remain unchanged. The reported post-adaptation metrics use all 2,470 phase-2/3/4 windows.
- Dietary treatment code is audit metadata and is excluded from every formal predictor branch.
- Planned feed and protocol meal timing describe offered feed and scheduled meals, not measured intake events.

## Locked manuscript results

| Result | Scope | R² |
|---|---|---:|
| Traditional deployment | all 4,079 windows, uncalibrated | 0.675 |
| Wearable model without individual calibration | 2,470 follow-up windows | 0.710 |
| Wearable model after fixed first-2-d calibration | same 2,470 follow-up windows | 0.800 |
| Traditional baseline (heart rate + ODBA, Ridge) | same 2,470 follow-up windows | 0.432 |

Exact saved values and manuscript rounding are verified by `scripts/03_reproduce_results_and_figures.py`. See [`reference_results/README.md`](reference_results/README.md) for the file-to-result map.

## Signal processing software and annotation boundary

This dataset does not provide manually annotated ECG-quality labels or a manually adjudicated R-peak gold standard. Do not describe software-generated detections as human annotation or ground truth.

The recommended processing release is **SwineSync OpenSource v1.1.1** ([release](https://github.com/zengzhengcheng/SwineSync-OpenSource/releases/tag/v1.1.1); [software DOI](https://doi.org/10.5281/zenodo.20051135)). It supports multi-folder conversion, per-folder file selection/inspection, file-level resume/skip behavior, short-file safety checks, heart-rate processing and IMU processing for new-device inputs only; it does not support legacy `Raw`, `BMD`, or `Angle` data. The packaged **SwineSync Studio v1.1.0** is distributed separately from its [binary-release repository](https://github.com/zengzhengcheng/SwineSync-Studio). ECG detector training code is archived as **ECG-TransUNet** ([DOI](https://doi.org/10.5281/zenodo.20051167)). The exact cleaning, synchronization, exclusion and final-window rules used in the paper are represented by this repository's scripts and deposited modeling panel.

## Repository map

- `data/analysis/`: formal 30-min modeling panel, data dictionary and feature-group definitions.
- `reference_results/`: outer folds, formal predictions, baseline/feeding result files and locked metrics.
- `src/`: calorimetry, ECG and IMU preparation modules.
- `scripts/01_build_30min_panel.py`: optional panel reconstruction.
- `scripts/02_run_nested_pig_cv.py`: formal pig-grouped nested training/evaluation.
- `scripts/03_reproduce_results_and_figures.py`: saved-result verification and Figures 2–4.
- `scripts/04_reproduce_supplementary_figures.py`: deposited-source verification and Figures S1–S3.
- `docs/input_data_contract.md`: upstream source-data contract.
- `docs/results_map.md`: manuscript result/figure mapping.

## Citation and license

See [`CITATION.cff`](CITATION.cff). Source code is released under the MIT License.

Files under `data/analysis/` and `reference_results/` are research data/analysis assets released under CC BY 4.0, as documented in [`DATA_LICENSE.md`](DATA_LICENSE.md). The top-level MIT License applies to code and does not replace the data license.
