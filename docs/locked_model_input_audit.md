# Locked model input audit

The locked R² values (0.675, 0.710 and 0.800) use the same pig-grouped
out-of-fold `blend_top3` predictions. The first two values differ only in the
evaluation period; the third applies fixed individual calibration to the same
post-adaptation evaluation rows.

## Executable implementation

The locked implementation is the combination of:

- `src/formal_model_core.py`: preprocessing, estimator construction, grouped
  inner/outer selection, seed averaging and fixed-alpha calibration.
- `scripts/02_run_nested_pig_cv.py`: the final four confirmation branches and
  the 17-candidate library used by the locked run.

`formal_model_core.py` alone is a shared engine and base candidate/branch
library; it is not the complete locked confirmation configuration.

## Selected components by outer fold

Each component is fitted for seeds 13, 42 and 2026. Predictions are averaged
across the three components within each seed and then across seeds.

| Fold | Held test pigs | Component 1 | Component 2 | Component 3 |
|---:|---|---|---|---|
| 1 | pig_14, pig_18, pig_20 | CatBoost (`compact_base`) | ElasticNet (`full_motion_feed_event`) | Random forest (`compact_feed_event`) |
| 2 | pig_12, pig_19 | ElasticNet (`full_motion_feed_event`) | Extra Trees (`full_motion_feed_event`) | CatBoost (`compact_base`) |
| 3 | pig_01, pig_02, pig_13 | ElasticNet (`full_motion_feed_event`) | CatBoost (`full_motion_feed_event`) | Extra Trees (`full_motion_feed_event`) |
| 4 | pig_16, pig_17 | CatBoost (`full_motion`) | XGBoost (`full_motion_feed_event`) | Extra Trees (`full_motion_feed_event`) |
| 5 | pig_11, pig_15 | CatBoost (`full_motion`) | LightGBM (`full_motion`) | histogram gradient boosting (`full_motion`) |

The ensemble therefore does not have three fixed algorithm families across all
folds. It has three inner-selected, family-diverse components per outer fold.
Seven algorithm families occur across the five locked folds.

Complete hyperparameters and pipe-delimited numeric/categorical lists are in
`reference_results/locked_model_components_by_fold.csv`. The exploded
one-row-per-input representation is in
`reference_results/locked_model_features_by_fold.csv`.

## Confirmed phase, chamber and window inputs

All 15 selected fold components received the following actual X columns:

- categorical, one-hot encoded inside the training fold: `chamber`,
  `phase_label`;
- numeric: `is_adaptation`, `is_formal`, `is_fasting`, `day_in_phase`,
  `win_s`.

In the deposited panel, `phase_label`, `day_in_phase`, `is_adaptation`,
`is_formal` and `is_fasting` are redundant encodings derived from the same
five-level phase assignment. They must not be described as excluded from the
locked model. `diet_code` is excluded from all model inputs.

`win_s` is the effective source calorimetry-record interval, taken from the
source interval field and clipped to 150–420 s. The same value set the backward
ECG/IMU extraction duration. It is therefore available from the chamber data
preparation pipeline, not intrinsically from the wearable signal. A prospective
wearable-only implementation must supply or prespecify the corresponding
window duration rather than silently treating it as unavailable metadata.

Run `python scripts/05_audit_locked_model_inputs.py` to check these statements
against the deposited code, panel, data dictionary, component manifest and
saved predictions without fitting a model.
