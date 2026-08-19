# Input data contract

All timestamps must use one consistent local timezone and parse as ISO-8601 or pandas-compatible datetimes.

| Source | Required fields | Meaning |
|---|---|---|
| Calorimetry | `datetime`, `HP_kcal`, chamber and period metadata | HP accumulated for the source interval; the scripts attach phase labels from workbook sheets. |
| ECG R peaks | `timestamp`, `label` | One row per detected R peak. The analysis accepts the retained peak label and derives RR intervals/HRV. |
| IMU summaries | `timestamp` plus motion fields | Ten-Hz recordings summarized within the sensor window; source files for a chamber must have the same schema. |
| Experiment design | `period`, `chamber`, `experimental_unit`, `animal_id`, `ear_tag`, `diet_code`, four weight anchors, adaptation/formal offered-feed fields, `feed_measurement` | Ear-tag-backed identity and protocol metadata. |
| 30-min panel | `datetime`, `pig`, `experimental_unit`, `phase_idx`, `weight`, `HP_kcal`, `target_HP_w075` plus registered predictors | Direct input to nested modeling. |

Identity is based on ear tag, not chamber-period aliases. The experimental day begins at 09:00. ECG validity requires at least 150 R peaks; RR intervals below 50 ms are duplicate artifacts and intervals above 3000 ms are treated as gaps. The analysis target is evaluated in raw HP kcal after reversing metabolic-weight normalization.
