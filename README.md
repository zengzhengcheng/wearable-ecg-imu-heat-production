# Wearable ECG–IMU sensing for pig heat-production prediction

Code and compact analysis assets accompanying **“Wearable ECG–IMU sensing: Predicting heat production in growing pigs with short-term calorimetry calibration.”**

The study combines ECG, 10 Hz IMU and open-circuit respiration-calorimetry measurements from 12 growing pigs to predict heat production (HP) in 30-min analysis windows.

## Submission scope and interpretation

The locked manuscript workflow is **chamber-informed**. Its model inputs include ECG/heart-rate and HRV descriptors, IMU-derived activity descriptors, body weight, calorimetry-related time/phase and source-window information, respiration chamber information indicating A1, B1, or B2, chamber temperature and relative humidity. Some inner-selected components also use prespecified offered-feed and protocol meal-time variables; dietary treatment identity (`diet_code`) is not a predictor. The complete per-fold input lists are deposited in [`reference_results/locked_model_features_by_fold.csv`](reference_results/locked_model_features_by_fold.csv) and can be checked without fitting a model by running `scripts/05_audit_locked_model_inputs.py`.

Accordingly, the reported values are **pig-grouped internal-validation results within the current A1/B1/B2 three-chamber facility**. They are not evidence of validation in an independent facility and should not be described as a wearable-only or cross-facility model.

The reported calibrated R² of 0.800 is obtained by applying a fixed, prespecified shrinkage correction to follow-up predictions using measured HP from the first 2 d of the held-out pig's experimental unit. This is post-prediction individual calibration; the predictive model is not retrained on that pig.

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

As a pip alternative, start from a Python 3.11 environment and run `python -m pip install -r requirements.txt`.

This verifies the four locked R² values and writes main Figures 2–4 plus supplementary Figures S1–S3 under `results/reproduced/`. It does not retrain, search models or tune calibration. The main script writes three reproduced metric tables and PNG/SVG/PDF versions of Figures 2–4; a successful run ends with `REPRODUCIBILITY_PASS figures=2,3,4`.

The default public entry point is limited to the locked chamber-informed submission workflow. Strict-v2, Sequence-6, unified-target and other stopped exploratory branches are not used by this workflow and are not presented as manuscript results. Any retained exploratory material must remain clearly labeled as research/archive content outside the default reproduction route.

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
python scripts/05_audit_locked_model_inputs.py
```

`scripts/02_run_nested_pig_cv.py` plus `src/formal_model_core.py` form the
actual locked training implementation: the core supplies the nested-CV engine,
and script 02 supplies the final four feature branches and 17-candidate
confirmation library. The core module alone is not the complete locked
configuration. Script 05 is audit-only: it does not fit models and verifies
the deposited per-fold members, full input-column lists and locked predictions.

Model evaluation is grouped by true pig identity. Do not substitute a date holdout/GKF-Day analysis for the locked manuscript protocol.

## Analysis protocol

- Outer evaluation: five folds grouped by true animal identity (`pig`); no animal appears in both training and test parts of a fold.
- Inner selection: three-fold pig-grouped cross-validation within each outer-training set.
- Fit target: HP normalized by metabolic body weight (`HP_per_W075`).
- Reported target: `HP_kcal` is the mean 5-min HP quantity represented by each 30-min window and is multiplied by 6 for reporting in kcal per 30 min. The public saved prediction file already contains reported 30-min values.
- Individual calibration: for each held-out experimental unit, the ratio of mean observed to mean predicted HP over phases 0–1 (first 2 d) is shrunk with fixed `alpha = 0.5`; only the 1,718 fed measurement windows in phases 2–3 (888 + 830) are adjusted, while all 752 phase-4 fasting predictions remain unchanged. The reported post-adaptation metrics use all 2,470 phase-2/3/4 windows.
- Dietary treatment code is audit metadata and is excluded from every formal predictor branch.
- Planned feed and protocol meal timing describe offered feed and scheduled meals, not measured intake events.
- Every locked fold component receives `chamber` and `phase_label` as one-hot categorical inputs, `is_adaptation`, `is_formal`, `is_fasting`, `day_in_phase` and `win_s` as numeric inputs, and chamber temperature (`小室温度(℃)`) and relative humidity (`小室湿度(%)`) among its numeric sensor/context inputs. These are not audit-only fields. `win_s` is the source calorimetry-record interval used to set the backward sensor-extraction duration; prospective wearable-only use must supply or prespecify the corresponding window duration.

## Locked manuscript results

| Result | Scope | R² |
|---|---|---:|
| Chamber-informed model, full-period internal validation | all 4,079 windows, uncalibrated | 0.675 |
| Chamber-informed model, post-adaptation internal validation | 2,470 follow-up windows, uncalibrated | 0.710 |
| Same model after fixed first-2-d post-prediction calibration | same 2,470 follow-up windows | 0.800 |
| Traditional baseline (heart rate + ODBA, Ridge) | same 2,470 follow-up windows | 0.432 |

Exact saved values and manuscript rounding are verified by `scripts/03_reproduce_results_and_figures.py`. See [`reference_results/README.md`](reference_results/README.md) for the file-to-result map.

## Signal processing software and annotation boundary

This dataset does not provide manually annotated ECG-quality labels or a manually adjudicated R-peak gold standard. Do not describe software-generated detections as human annotation or ground truth.

The recommended current processing release is **SwineSync OpenSource v1.1.1** ([current GitHub release](https://github.com/zengzhengcheng/SwineSync-OpenSource/releases/tag/v1.1.1)). It supports multi-folder conversion, per-folder file selection/inspection, file-level resume/skip behavior, short-file safety checks, heart-rate processing and IMU processing for new-device inputs only; it does not support legacy `Raw`, `BMD`, or `Angle` data. The existing [Zenodo archive for v1.0.0](https://doi.org/10.5281/zenodo.20051135) archives SwineSync OpenSource **v1.0.0** only; it is not the DOI for the later GitHub v1.1.1 release. The packaged **SwineSync Studio v1.1.0** is distributed separately from its [binary-release repository](https://github.com/zengzhengcheng/SwineSync-Studio). ECG detector training code is archived as **ECG-TransUNet v1.0.0** ([DOI](https://doi.org/10.5281/zenodo.20051167)). The exact cleaning, synchronization, exclusion and final-window rules used in the paper are represented by this repository's scripts and deposited modeling panel.

## Repository map

- `data/analysis/`: formal 30-min modeling panel, data dictionary and feature-group definitions.
- `reference_results/`: outer folds, formal predictions, baseline/feeding result files and locked metrics.
- `src/`: calorimetry, ECG and IMU preparation modules.
- `scripts/01_build_30min_panel.py`: optional panel reconstruction.
- `scripts/02_run_nested_pig_cv.py`: formal pig-grouped nested training/evaluation.
- `scripts/03_reproduce_results_and_figures.py`: saved-result verification and Figures 2–4.
- `scripts/04_reproduce_supplementary_figures.py`: deposited-source verification and Figures S1–S3.
- `scripts/05_audit_locked_model_inputs.py`: no-training audit of the locked fold members and actual X columns.
- `docs/input_data_contract.md`: upstream source-data contract.
- `docs/results_map.md`: manuscript result/figure mapping.
- `docs/locked_model_input_audit.md`: per-fold locked components, actual input roles and implementation boundary.
- `docs/availability_and_disclosure.md`: manuscript-ready availability, citation and AI-assisted-tools wording.

## Citation and license

See [`CITATION.cff`](CITATION.cff). Source code is released under the MIT License.

Files under `data/analysis/` and `reference_results/` are research data/analysis assets released under CC BY 4.0, as documented in [`DATA_LICENSE.md`](DATA_LICENSE.md). The top-level MIT License applies to code and does not replace the data license.

Dataset citation:

> Zeng, Z., Tian, S., Zhang, S. (2026). *Wearable ECG–IMU and respiration-calorimetry data for heat-production prediction in growing pigs*. ScienceDB, version 1.0. https://doi.org/10.57760/sciencedb.00zyg

Paste-ready Code availability, Data availability and AI-assisted-tools disclosure text is provided in [`docs/availability_and_disclosure.md`](docs/availability_and_disclosure.md).
