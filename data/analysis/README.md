# Formal 30-min analysis data

## Files

- `modeling_panel_30min.csv`: the unchanged formal 4,079-row × 345-column panel that generated the deposited confirmation predictions.
- `modeling_data_dictionary.csv`: one row per panel column, including definition, unit, field group, formal-predictor status and grouping/calibration/evaluation roles.
- `feature_groups.json`: feeding-protocol feature groups used by the optional validation/training entry.

## Target units

`HP_kcal` is the mean 5-min HP quantity represented by each 30-min modeling window. The formal prediction export multiplies it by 6 for manuscript reporting in kcal per 30 min. `HP_per_W075` is the model-fitting target after division by body weight raised to 0.75.

## Predictor boundary

The locked confirmation library uses ECG/HRV, IMU, body weight, calorimetry-related time/phase and source-window fields, respiration chamber information indicating A1, B1, or B2, chamber temperature and relative humidity. Inner-selected branches may additionally use prespecified feeding-protocol features, as indicated in the dictionary. `diet_code`, pig identity/grouping fields, targets and lineage fields are not predictors. Planned feed is offered feed, not measured intake; meal-event variables derive from protocol times, not observed feeding events.

The first 2 d/follow-up/fasting scope is defined by the phase fields. Formal outer folds group by `pig`; date grouping columns are retained only for lineage and are not part of the locked evaluation.

These data support pig-grouped internal validation within the current three-chamber facility. Because respiration chamber information and chamber environmental measurements are model inputs, the locked results must not be presented as wearable-only or cross-facility validation.
