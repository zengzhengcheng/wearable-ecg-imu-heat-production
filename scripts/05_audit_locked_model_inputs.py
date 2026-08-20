"""Audit the deposited locked model inputs without fitting any model.

This script verifies the three locked R² values, outer-fold membership,
per-fold three-component manifests, complete X-column lists, and data-dictionary
flags against the executable public feature definitions. It never calls
``fit`` or ``predict``.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score


PROJECT = Path(__file__).resolve().parents[1]
SRC = PROJECT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import formal_model_core as core  # noqa: E402


PANEL_PATH = PROJECT / "data" / "analysis" / "modeling_panel_30min.csv"
GROUPS_PATH = PROJECT / "data" / "analysis" / "feature_groups.json"
DICTIONARY_PATH = PROJECT / "data" / "analysis" / "modeling_data_dictionary.csv"
PREDICTIONS_PATH = PROJECT / "reference_results" / "formal_window_predictions.csv"
FOLDS_PATH = PROJECT / "reference_results" / "outer_fold_assignments.csv"
COMPONENTS_PATH = PROJECT / "reference_results" / "locked_model_components_by_fold.csv"
FEATURES_PATH = PROJECT / "reference_results" / "locked_model_features_by_fold.csv"
LOCKED_FIELDS = (
    "chamber",
    "phase_label",
    "is_adaptation",
    "is_formal",
    "is_fasting",
    "day_in_phase",
    "win_s",
)


def load_training_entry():
    path = PROJECT / "scripts" / "02_run_nested_pig_cv.py"
    spec = importlib.util.spec_from_file_location("locked_training_entry", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load locked training entry")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def verify_locked_metrics(predictions: pd.DataFrame) -> dict[str, float]:
    followup = predictions["is_followup_evaluation"].astype(str).str.lower().eq("true")
    truth_all = predictions["observed_HP"].to_numpy(float)
    truth_followup = predictions.loc[followup, "observed_HP"].to_numpy(float)
    values = {
        "all_periods_uncalibrated": float(
            r2_score(truth_all, predictions["predicted_HP_raw"].to_numpy(float))
        ),
        "post_adaptation_uncalibrated": float(
            r2_score(
                truth_followup,
                predictions.loc[followup, "predicted_HP_raw"].to_numpy(float),
            )
        ),
        "post_adaptation_calibrated": float(
            r2_score(
                truth_followup,
                predictions.loc[followup, "predicted_HP_calibrated"].to_numpy(float),
            )
        ),
    }
    expected = (0.6753180182365188, 0.7101946227116984, 0.7998914733588931)
    if not np.allclose(list(values.values()), expected, rtol=0, atol=1e-12):
        raise AssertionError(f"locked R2 mismatch: {values}")
    return values


def main() -> None:
    entry = load_training_entry()
    frame, groups = core.load_inputs(PANEL_PATH, GROUPS_PATH)
    core.validate_panel(frame, groups)
    branch_specs = entry.confirm_variants(frame, groups)

    components = pd.read_csv(COMPONENTS_PATH)
    features = pd.read_csv(FEATURES_PATH)
    dictionary = pd.read_csv(DICTIONARY_PATH, low_memory=False)
    predictions = pd.read_csv(PREDICTIONS_PATH)
    folds = pd.read_csv(FOLDS_PATH)

    if len(components) != 15 or components[["outer_fold", "component_order"]].duplicated().any():
        raise AssertionError("expected 15 unique fold/component records")
    if set(components["outer_fold"]) != {1, 2, 3, 4, 5}:
        raise AssertionError("component manifest does not cover five folds")

    for record in components.itertuples(index=False):
        spec = branch_specs[str(record.feature_branch)]
        expected = [
            *(("numeric", feature) for feature in spec["numeric"]),
            *(("categorical", feature) for feature in spec["categorical"]),
        ]
        actual = features.loc[
            features["outer_fold"].eq(record.outer_fold)
            & features["component_order"].eq(record.component_order)
        ].sort_values("feature_order")
        observed = list(zip(actual["feature_type"], actual["feature"]))
        if observed != expected:
            raise AssertionError(
                f"feature manifest differs for fold {record.outer_fold} "
                f"component {record.component_order}"
            )
        if int(record.numeric_feature_count) != len(spec["numeric"]):
            raise AssertionError("numeric feature count mismatch")
        if int(record.categorical_feature_count) != len(spec["categorical"]):
            raise AssertionError("categorical feature count mismatch")

    held_from_components = (
        components.groupby("outer_fold")["held_test_pigs"].first().to_dict()
    )
    held_from_folds = (
        folds.groupby("outer_fold_as_test")["pig"]
        .apply(lambda values: "|".join(sorted(values.astype(str))))
        .to_dict()
    )
    if held_from_components != held_from_folds:
        raise AssertionError("component and outer-fold held-pig assignments differ")

    dictionary = dictionary.set_index("column")
    focus: dict[str, object] = {}
    for field in LOCKED_FIELDS:
        field_rows = features.loc[features["feature"].eq(field)]
        if len(field_rows) != 15:
            raise AssertionError(f"{field} is not present in all 15 components")
        if dictionary.loc[field, "locked_model_input"] != "yes":
            raise AssertionError(f"dictionary does not mark {field} as locked input")
        focus[field] = {
            "role": str(dictionary.loc[field, "locked_input_role"]),
            "components": int(len(field_rows)),
            "unique_values": int(frame[field].nunique(dropna=False)),
        }

    values = verify_locked_metrics(predictions)
    result = {
        "status": "LOCKED_INPUT_AUDIT_PASS",
        "training_performed": False,
        "locked_r2": values,
        "fold_components": int(len(components)),
        "feature_manifest_rows": int(len(features)),
        "branch_input_counts": {
            name: {
                "numeric": len(spec["numeric"]),
                "categorical": len(spec["categorical"]),
                "total": len(spec["numeric"]) + len(spec["categorical"]),
            }
            for name, spec in branch_specs.items()
        },
        "focus_fields": focus,
        "win_s_range_seconds": [
            float(pd.to_numeric(frame["win_s"]).min()),
            float(pd.to_numeric(frame["win_s"]).max()),
        ],
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))
    print("NO_TRAINING_PERFORMED")


if __name__ == "__main__":
    main()
