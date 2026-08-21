# Input data contract

All timestamps must use one consistent local timezone and parse as ISO-8601 or pandas-compatible datetimes.

| Source | Required fields | Meaning |
|---|---|---|
| Calorimetry and chamber environment | `datetime`, `HP_kcal`, chamber and period metadata, chamber temperature, relative humidity | HP accumulated for the source interval and aligned chamber conditions; the scripts attach phase labels from workbook sheets. |
| ECG R peaks | `timestamp`, `label` | One row per detected R peak. The analysis accepts the retained peak label and derives RR intervals/HRV. |
| IMU summaries | `timestamp` plus motion fields | Ten-Hz recordings summarized within the sensor window; source files for a chamber must have the same schema. |
| Experiment design | `period`, `chamber`, `experimental_unit`, `animal_id`, `ear_tag`, `diet_code`, archived source/audit weight fields, adaptation/formal offered-feed fields, `feed_measurement` | Ear-tag-backed identity and protocol metadata. Weight-field labels are retained for locked-pipeline traceability and must not be interpreted as four independently confirmed public weighing times. |
| 30-min panel | `datetime`, `pig`, `experimental_unit`, `phase_idx`, `weight`, `HP_kcal`, `target_HP_w075` plus registered predictors | Direct input to nested modeling. |

Identity is based on ear tag, not chamber-period aliases. The experimental day begins at 09:00. ECG validity requires at least 150 R peaks; RR intervals below 50 ms are duplicate artifacts and intervals above 3000 ms are treated as gaps. The analysis target is evaluated in raw HP kcal after reversing metabolic-weight normalization.

The locked model uses respiration chamber information indicating A1, B1, or B2. Alongside ECG, IMU and body weight, it receives calorimetry-related time/phase and source-window fields, that respiration chamber information, chamber temperature and relative humidity. The actual per-fold input lists are deposited under `reference_results/` and can be checked without fitting by running `scripts/05_audit_locked_model_inputs.py`.
