# Availability, citation and disclosure text

This file provides manuscript-ready wording aligned with the public repository. Journal house style may require minor punctuation or heading changes.

## Code availability

Analysis code, the compact analysis-ready modeling panel, pig-grouped outer-fold assignments, saved out-of-fold predictions, and scripts for reproducing the reported metrics and Figures 2–4 are available at https://github.com/zengzhengcheng/wearable-ecg-imu-heat-production. The default saved-prediction workflow reproduces the locked results from the model with respiration chamber information indicating A1, B1, or B2, without model retraining or hyperparameter search. Source code is available under the MIT License; the compact research-data and analysis assets in `data/analysis/` and `reference_results/` are available under CC BY 4.0.

## Data availability

The raw or near-raw ECG and IMU acquisitions, extracted sensor data, 5-min respiration-calorimetry records, experiment metadata, and quality-control/lineage documentation are available in ScienceDB: Zeng, Z., Tian, S., and Zhang, S. (2026), “Wearable ECG–IMU and respiration-calorimetry data for heat-production prediction in growing pigs,” version 1.0, https://doi.org/10.57760/sciencedb.00zyg. The smaller analysis-ready 30-min modeling panel, fold assignments, saved predictions, data dictionary, and figure-reproduction assets are available in the associated GitHub repository at https://github.com/zengzhengcheng/wearable-ecg-imu-heat-production.

## AI-assisted tools disclosure

OpenAI Codex was used to assist with the development and reproducibility checking of analysis code. All code and analytical results were reviewed and verified by the authors, who retain full responsibility for the work.

## Interpretation boundary

The locked model uses ECG/HRV, IMU, body weight, calorimetry-related time/phase and source-window information, **respiration chamber information indicating A1, B1, or B2**, chamber temperature and relative humidity; some inner-selected components also use prespecified offered-feed and protocol meal-time variables. Its reported R² values of 0.675, 0.710 and 0.800 are pig-grouped internal-validation results within the current three-chamber facility. The 0.800 result applies a frozen post-prediction correction using measured HP from the first 2 d of the held-out experimental unit and does not retrain the predictive model.

## Software version note

The current GitHub processing release is SwineSync OpenSource v1.1.1: https://github.com/zengzhengcheng/SwineSync-OpenSource/releases/tag/v1.1.1. The Zenodo record https://doi.org/10.5281/zenodo.20051135 archives SwineSync OpenSource v1.0.0 only, not the current GitHub v1.1.1 release. ECG-TransUNet v1.0.0 is archived at https://doi.org/10.5281/zenodo.20051167. Do not describe software-generated quality labels or R-peak detections as manually annotated labels or a human-adjudicated gold standard.
