"""Strict nested modeling of the 2026-only growing-pig feed/HR/IMU panel.

Feature branches (diet categories removed from all features)
------------------------------------------------------------
Four compact branches form a feeding-information ladder: ``compact_base``
(context + compact motion), ``compact_feed`` (+ planned feed dose),
``compact_feed_event`` (+ protocol meal-event features) and
``compact_feed_fullfeeding`` (+ premeal sensor baseline and meal-sensor
interactions).  ``diet_code`` survives as an audit label only and never
enters numeric or categorical model features.

Hybrid per-pig/per-unit calibration
-----------------------------------
Every branch ensemble ships raw nested-OOF predictions (uncalibrated) and
fixed-alpha hybrid ratio-calibrated layers computed in
raw-HP space.  A per-key (pig::experimental_unit, or pooled per pig)
truth/prediction ratio over phase 0/1 rows scales fed-phase (2/3)
predictions by ``1 + alpha * (ratio - 1)``; fasting phase 4 keeps the raw
prediction. Variants are ``hybrid_unit_fixed050`` (primary) and
``hybrid_pig_fixed050`` (sensitivity). GKF_Pig ratios use the held pig's own
phase 0/1 truth by design (declared few-shot deployment).

Scientific boundaries
---------------------
* ``diet_code`` is a categorical treatment label kept for audit only; it is
  excluded from every feature branch and is never used as an ordinal number.
* Planned feed is offered quantity, not measured intake.  Meal-event features
  use protocol times rather than electronic feeding records.
* The fit target is HP/W^0.75.  Both normalized and raw-HP metrics are saved.
* Every outer fold repeats model and feature-variant selection using only
  grouped inner OOF predictions.  No broad-screen winner is read from disk.

Modes
-----
``validate``
    Check data, feature, categorical-encoding and grouped-split invariants.
``smoke``
    Three outer folds, two inner folds, four model candidates, one seed and a
    reduced feature library.  This is an execution check, not a result.
``evaluate``
    Five outer folds, three inner folds, thirteen model candidates, all
    ablations, three-seed OOF predictions, cluster bootstrap and final
    full-data models selected by grouped CV.

Examples
--------
python scripts/02_run_nested_pig_cv.py --mode validate
python scripts/02_run_nested_pig_cv.py --mode smoke --scenario GKF_Pig
python scripts/02_run_nested_pig_cv.py --mode evaluate --scenario GKF_Pig
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    ExtraTreesRegressor,
    HistGradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, RobustScaler, StandardScaler
from sklearn.svm import SVR
from xgboost import XGBRegressor


HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent

MOVE_STAT_COLS = [
    "Acc_Mean", "Acc_Std", "Acc_Sum", "Acc_Max", "Acc_Min", "Acc_Range",
    "Acc_P25", "Acc_P75", "Gyro_Mean", "Gyro_Max", "DynX_Mean", "DynX_Std",
    "DynX_Mean_abs", "DynY_Mean", "DynY_Std", "DynY_Mean_abs", "DynZ_Mean",
    "DynZ_Std", "DynZ_Mean_abs", "AccMag_Mean", "AccMag_Std", "Roll_Mean",
    "Roll_Std", "Pitch_Mean", "Pitch_Std", "Tilt_Mag_Mean", "Tilt_Mag_Std",
    "AngX_Std", "AngY_Std", "AngZ_Std", "AngX_Rate", "AngY_Rate",
    "AngZ_Rate", "Jerk_Mean", "ODBA", "VeDBA_RMS", "Axis_Dominance",
    "Skewness", "Kurtosis", "Autocorr_Lag1", "Peak_Count", "Corr_XY",
    "Corr_XZ", "Corr_YZ", "Spectral_Entropy", "Zero_Crossing_Rate",
    "Dominant_Frequency", "Coefficient_of_Variation", "Power_Low",
    "Power_Loco", "Power_High", "Sample_Entropy",
]
MOVE_FRAC_COLS = [
    "Frac_Rest", "Frac_Moderate", "Frac_Vigorous", "Frac_Lying",
    "Frac_HeadDown", "Active_Fraction",
]
COMPACT_MOTION_BASES = [
    "Acc_Mean", "Acc_Std", "AccMag_Mean", "AccMag_Std", "Jerk_Mean", "ODBA",
    "VeDBA_RMS", "Power_Loco", "Power_High", "Dominant_Frequency",
    "Spectral_Entropy", "Zero_Crossing_Rate", "Tilt_Mag_Mean", "Tilt_Mag_Std",
    "Roll_Mean", "Pitch_Mean", "AngX_Rate", "AngY_Rate", "AngZ_Rate",
    "Peak_Count", "DynY_Mean", "DynZ_Mean",
]
MOTION_SCALARS = [
    *MOVE_FRAC_COLS, "move_coverage", "StepFreq_std", "LyingTransitions",
    "HeadDownTransitions", "data_count_mean",
]


def _motion_from_bases(columns: Iterable[str], bases: Iterable[str]) -> list[str]:
    available = set(columns)
    wanted = [
        f"{base}_{suffix}"
        for base in bases
        for suffix in ("mean", "std", "p25", "p75")
    ]
    wanted.extend(MOTION_SCALARS)
    return [name for name in wanted if name in available]


def compact_motion(columns: Iterable[str]) -> list[str]:
    return _motion_from_bases(columns, COMPACT_MOTION_BASES)


def motion_columns(columns: Iterable[str]) -> list[str]:
    return _motion_from_bases(columns, MOVE_STAT_COLS)


DEFAULT_ROOT = Path("results")
DEFAULT_PANEL = DEFAULT_ROOT / "2026_diet_30min_matched.csv"
DEFAULT_GROUPS = DEFAULT_ROOT / "feature_groups.json"
DEFAULT_OUT = DEFAULT_ROOT / "nested_model_results"

SEEDS = (13, 42, 2026)
SELECTION_SEED = 42
TARGET = "HP_per_W075"
RAW_TARGET = "HP_kcal"

HRV = [
    "beat_count",
    "HR_mean",
    "HR_std",
    "RR_mean",
    "SDNN",
    "RMSSD",
    "pNN50",
    "RR_cv",
    "RR_skew",
    "RR_kurt",
    "RR_range",
]
ENV = ["小室温度(℃)", "小室湿度(%)"]
CONTEXT_NUMERIC = [
    "hour_of_day",
    "sin_hour",
    "cos_hour",
    "weight",
    "is_fasting",
    "day_in_phase",
    "win_s",
    "is_adaptation",
    "is_formal",
]
CONTEXT_CATEGORICAL = ["chamber", "phase_label"]
DIET_CATEGORICAL = ["diet_code"]

# Avoid placing deterministic/redundant feed representations in the same
# default branch.  This dose is aligned with the allometrically scaled target.
PRIMARY_FEED = [
    "planned_meal_g_per_w075_if_fed",
    "planned_feed_available",
]

ABLATION_PAIRS = [
    ("compact_base", "compact_feed", "feed_over_base"),
    ("compact_feed", "compact_feed_event", "event_over_feed"),
    (
        "compact_feed_event",
        "compact_feed_fullfeeding",
        "premeal_sensor_over_protocol_event",
    ),
]

# Hybrid ratio-calibration protocol (raw-HP space); see hybrid_calibration().
CAL_PHASES = [0, 1]
FED_TEST_PHASES = [2, 3]
FASTING_PHASE = 4
EVAL_PHASES = [2, 3, 4]
ALPHA_FIXED = 0.5


def dedupe(items: Iterable[str]) -> list[str]:
    return list(dict.fromkeys(items))


def present(frame: pd.DataFrame, items: Iterable[str]) -> list[str]:
    return [item for item in items if item in frame.columns]


def load_inputs(
    panel_path: Path, groups_path: Path
) -> tuple[pd.DataFrame, dict[str, list[str]]]:
    frame = pd.read_csv(
        panel_path,
        low_memory=False,
        dtype={"ear_tag": "string", "animal_id": "string"},
    )
    frame["datetime"] = pd.to_datetime(
        frame["datetime"], format="mixed", errors="coerce"
    )
    if "feature_time" in frame:
        frame["feature_time"] = pd.to_datetime(
            frame["feature_time"], format="mixed", errors="coerce"
        )
    else:
        frame["feature_time"] = frame["datetime"] + pd.Timedelta(minutes=15)

    for column in dedupe(CONTEXT_CATEGORICAL + DIET_CATEGORICAL):
        if column not in frame:
            raise KeyError(f"missing categorical column: {column}")
        if column == "diet_code":
            numeric = pd.to_numeric(frame[column], errors="raise").astype(int)
            frame[column] = numeric.astype(str)
        else:
            frame[column] = frame[column].astype("string").fillna("__MISSING__")

    with groups_path.open("r", encoding="utf-8") as handle:
        feature_groups = json.load(handle)
    if not isinstance(feature_groups, dict):
        raise TypeError("feature_groups.json must contain an object")
    return frame.reset_index(drop=True), feature_groups


def validate_panel(
    frame: pd.DataFrame, feature_groups: dict[str, list[str]]
) -> dict[str, object]:
    required = {
        "pig",
        "animal_id",
        "experimental_unit",
        "datetime",
        "feature_time",
        "strict_day_group",
        "animal_day_group",
        "day_group",
        "diet_code",
        "weight",
        "phase_idx",
        TARGET,
        RAW_TARGET,
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise KeyError(f"panel missing required columns: {missing}")
    if frame["datetime"].isna().any() or frame["feature_time"].isna().any():
        raise ValueError("panel contains invalid datetime/feature_time")
    if frame.duplicated(["pig", "datetime"]).any():
        raise ValueError("duplicate pig/datetime rows")
    if not frame["pig"].astype(str).eq(frame["animal_id"].astype(str)).all():
        raise ValueError("pig must equal ear-tag-backed animal_id")

    expected_date = (
        frame["feature_time"] - pd.Timedelta(hours=9)
    ).dt.strftime("%Y-%m-%d")
    expected_group = "2026::" + expected_date
    if not frame["strict_day_group"].astype(str).eq(expected_group).all():
        raise ValueError("strict_day_group is not the complete 09:00-boundary date")
    if not frame["day_group"].astype(str).eq(frame["strict_day_group"].astype(str)).all():
        raise ValueError("legacy day_group is not a strict-day alias")
    expected_animal_day = frame["pig"].astype(str) + "::" + expected_date
    if not frame["animal_day_group"].astype(str).eq(expected_animal_day).all():
        raise ValueError("animal_day_group audit field is malformed")

    diet_levels = sorted(frame["diet_code"].dropna().astype(str).unique().tolist())
    if diet_levels != [str(value) for value in range(7)]:
        raise ValueError(f"expected categorical diet labels 0..6, got {diet_levels}")
    if pd.api.types.is_numeric_dtype(frame["diet_code"]):
        raise TypeError("diet_code must be string/categorical before modeling")
    if frame["pig"].nunique() != 12:
        raise ValueError(f"expected 12 true pigs, got {frame['pig'].nunique()}")
    if frame["experimental_unit"].nunique() != 24:
        raise ValueError(
            f"expected 24 experimental units, got {frame['experimental_unit'].nunique()}"
        )
    if frame["strict_day_group"].nunique() != 40:
        raise ValueError(
            f"expected 40 strict dates, got {frame['strict_day_group'].nunique()}"
        )

    named_groups = [
        "event_clock",
        "meal_kernels",
        "premeal_sensor_baseline",
        "meal_sensor_interactions",
        "planned_feed",
        "dose_interactions",
    ]
    missing_groups = [name for name in named_groups if name not in feature_groups]
    if missing_groups:
        raise KeyError(f"feature_groups missing: {missing_groups}")
    missing_group_columns = {
        name: sorted(set(feature_groups[name]) - set(frame.columns))
        for name in named_groups
        if set(feature_groups[name]) - set(frame.columns)
    }
    if missing_group_columns:
        raise KeyError(f"feature-group columns absent from panel: {missing_group_columns}")

    for target in (TARGET, RAW_TARGET, "weight"):
        values = pd.to_numeric(frame[target], errors="coerce")
        if values.isna().any() or (values <= 0).any():
            raise ValueError(f"{target} must be finite and positive")

    return {
        "status": "PASS",
        "rows": int(len(frame)),
        "pigs": int(frame["pig"].nunique()),
        "experimental_units": int(frame["experimental_unit"].nunique()),
        "strict_days": int(frame["strict_day_group"].nunique()),
        "animal_days_not_used_for_day_cv": int(frame["animal_day_group"].nunique()),
        "diet_levels_as_labels": diet_levels,
        "strict_day_definition": "all pigs sharing one 09:00-boundary date",
        "fit_target": TARGET,
        "reported_targets": [TARGET, RAW_TARGET],
    }


def feature_variants(
    frame: pd.DataFrame, feature_groups: dict[str, list[str]], mode: str
) -> dict[str, dict[str, list[str]]]:
    context_numeric = present(frame, CONTEXT_NUMERIC + ENV + HRV)
    context_categorical = present(frame, CONTEXT_CATEGORICAL)
    compact = compact_motion(frame.columns)
    feed = present(frame, PRIMARY_FEED)
    event = present(
        frame,
        feature_groups["event_clock"]
        + feature_groups["meal_kernels"]
        + feature_groups["dose_interactions"],
    )
    full_feeding = present(
        frame,
        feature_groups["event_clock"]
        + feature_groups["meal_kernels"]
        + feature_groups["dose_interactions"]
        + feature_groups["premeal_sensor_baseline"]
        + feature_groups["meal_sensor_interactions"],
    )

    def spec(
        numeric: Iterable[str], include_diet: bool
    ) -> dict[str, list[str]]:
        categorical = context_categorical + (
            present(frame, DIET_CATEGORICAL) if include_diet else []
        )
        numeric_columns = dedupe(present(frame, numeric))
        overlap = set(numeric_columns) & set(categorical)
        if overlap:
            raise AssertionError(f"numeric/categorical overlap: {sorted(overlap)}")
        if "diet_code" in numeric_columns:
            raise AssertionError("diet_code entered numeric features")
        return {
            "numeric": numeric_columns,
            "categorical": dedupe(categorical),
        }

    compact_base = context_numeric + compact
    variants = {
        "compact_base": spec(compact_base, include_diet=False),
        "compact_feed": spec(compact_base + feed, include_diet=False),
        "compact_feed_event": spec(
            compact_base + feed + event, include_diet=False
        ),
        "compact_feed_fullfeeding": spec(
            compact_base + feed + full_feeding, include_diet=False
        ),
    }
    if mode == "smoke":
        keep = ("compact_base", "compact_feed", "compact_feed_event")
        variants = {name: variants[name] for name in keep}

    for name, columns in variants.items():
        if not columns["numeric"]:
            raise ValueError(f"{name} has no numeric features")
        if len(columns["numeric"]) != len(set(columns["numeric"])):
            raise AssertionError(f"duplicate numeric features in {name}")
        if len(columns["categorical"]) != len(set(columns["categorical"])):
            raise AssertionError(f"duplicate categorical features in {name}")
    return variants


def candidate_library(mode: str) -> list[dict[str, object]]:
    full = [
        {"name": "ridge_a10", "family": "ridge", "params": {"alpha": 10.0}},
        {"name": "ridge_a100", "family": "ridge", "params": {"alpha": 100.0}},
        {
            "name": "elastic",
            "family": "elastic",
            "params": {"alpha": 0.01, "l1_ratio": 0.2, "max_iter": 20000},
        },
        {
            "name": "svr_rbf",
            "family": "svr",
            "params": {"C": 10.0, "epsilon": 0.05, "gamma": "scale"},
            "exclude_full": True,
        },
        {
            "name": "extra_leaf1",
            "family": "extra",
            "params": {
                "n_estimators": 600,
                "min_samples_leaf": 1,
                "max_features": 0.75,
            },
        },
        {
            "name": "extra_leaf3",
            "family": "extra",
            "params": {
                "n_estimators": 600,
                "min_samples_leaf": 3,
                "max_features": 0.75,
            },
        },
        {
            "name": "rf_leaf3",
            "family": "rf",
            "params": {
                "n_estimators": 500,
                "min_samples_leaf": 3,
                "max_features": 0.70,
            },
        },
        {
            "name": "histgb",
            "family": "histgb",
            "params": {
                "max_iter": 400,
                "learning_rate": 0.04,
                "max_leaf_nodes": 20,
                "l2_regularization": 2.0,
            },
        },
        {
            "name": "lgb_small",
            "family": "lgb",
            "params": {
                "n_estimators": 600,
                "learning_rate": 0.025,
                "num_leaves": 8,
                "min_child_samples": 25,
                "reg_lambda": 3.0,
            },
        },
        {
            "name": "lgb_medium",
            "family": "lgb",
            "params": {
                "n_estimators": 600,
                "learning_rate": 0.025,
                "num_leaves": 20,
                "min_child_samples": 20,
                "reg_lambda": 2.0,
            },
        },
        {
            "name": "xgb_d3",
            "family": "xgb",
            "params": {
                "n_estimators": 600,
                "learning_rate": 0.025,
                "max_depth": 3,
                "min_child_weight": 3,
                "reg_lambda": 3.0,
            },
        },
        {
            "name": "cat_d5",
            "family": "cat",
            "params": {
                "iterations": 600,
                "depth": 5,
                "learning_rate": 0.035,
                "l2_leaf_reg": 5.0,
            },
        },
        {
            "name": "cat_d7",
            "family": "cat",
            "params": {
                "iterations": 650,
                "depth": 7,
                "learning_rate": 0.03,
                "l2_leaf_reg": 8.0,
            },
        },
    ]
    if mode == "smoke":
        wanted = {"ridge_a10", "extra_leaf3", "lgb_small", "cat_d5"}
        smoke = [
            {
                **candidate,
                "params": dict(candidate["params"]),
            }
            for candidate in full
            if candidate["name"] in wanted
        ]
        for candidate in smoke:
            params = candidate["params"]
            if "n_estimators" in params:
                params["n_estimators"] = 80
            if "iterations" in params:
                params["iterations"] = 100
        return smoke
    return full


def candidate_allowed(candidate: dict[str, object], variant_name: str) -> bool:
    return not (
        bool(candidate.get("exclude_full", False))
        and variant_name.startswith("full_")
    )


def make_preprocessor(spec: dict[str, list[str]]) -> ColumnTransformer:
    transformers: list[tuple[str, object, list[str]]] = [
        (
            "numeric",
            SimpleImputer(strategy="median", add_indicator=True),
            spec["numeric"],
        )
    ]
    if spec["categorical"]:
        categorical = Pipeline(
            [
                (
                    "impute",
                    SimpleImputer(strategy="most_frequent"),
                ),
                (
                    "onehot",
                    OneHotEncoder(
                        handle_unknown="ignore",
                        sparse_output=False,
                    ),
                ),
            ]
        )
        transformers.append(("categorical", categorical, spec["categorical"]))
    return ColumnTransformer(
        transformers,
        remainder="drop",
        verbose_feature_names_out=False,
    )


def make_estimator(
    candidate: dict[str, object],
    spec: dict[str, list[str]],
    seed: int,
) -> Pipeline:
    family = str(candidate["family"])
    params = dict(candidate["params"])
    if family == "ridge":
        model = Ridge(**params)
        scaler: object | None = StandardScaler()
    elif family == "elastic":
        model = ElasticNet(**params, random_state=seed)
        scaler = RobustScaler()
    elif family == "svr":
        model = SVR(**params)
        scaler = StandardScaler()
    elif family == "extra":
        model = ExtraTreesRegressor(
            **params, n_jobs=4, random_state=seed
        )
        scaler = None
    elif family == "rf":
        model = RandomForestRegressor(
            **params, n_jobs=4, random_state=seed
        )
        scaler = None
    elif family == "histgb":
        model = HistGradientBoostingRegressor(**params, random_state=seed)
        scaler = None
    elif family == "lgb":
        model = LGBMRegressor(
            **params,
            subsample=0.85,
            colsample_bytree=0.85,
            verbosity=-1,
            n_jobs=4,
            random_state=seed,
        )
        scaler = None
    elif family == "xgb":
        model = XGBRegressor(
            **params,
            subsample=0.85,
            colsample_bytree=0.8,
            objective="reg:squarederror",
            tree_method="hist",
            n_jobs=4,
            random_state=seed,
        )
        scaler = None
    elif family == "cat":
        model = CatBoostRegressor(
            **params,
            loss_function="RMSE",
            verbose=False,
            thread_count=4,
            random_seed=seed,
            allow_writing_files=False,
        )
        scaler = None
    else:
        raise ValueError(f"unknown family: {family}")

    steps: list[tuple[str, object]] = [
        ("preprocess", make_preprocessor(spec))
    ]
    if scaler is not None:
        steps.append(("scale", scaler))
    steps.append(("model", model))
    return Pipeline(steps)


def group_column(scenario: str) -> str:
    if scenario != "GKF_Pig":
        raise ValueError("Only pig-grouped evaluation is part of the public protocol")
    return "pig"


def grouped_splits(
    frame: pd.DataFrame, scenario: str, n_splits: int
) -> list[tuple[np.ndarray, np.ndarray]]:
    column = group_column(scenario)
    groups = frame[column].astype(str).to_numpy()
    unique = pd.unique(groups).size
    if unique < n_splits:
        raise ValueError(
            f"{scenario} has only {unique} {column} groups for {n_splits} folds"
        )
    splits = list(GroupKFold(n_splits=n_splits).split(frame, groups=groups))
    for train, test in splits:
        overlap = set(groups[train]) & set(groups[test])
        if overlap:
            raise AssertionError(
                f"{scenario} train/test group overlap: {sorted(overlap)[:3]}"
            )
    return splits


def safe_r2(truth: np.ndarray, prediction: np.ndarray) -> float:
    if len(truth) < 2 or np.var(truth) <= 0:
        return float("nan")
    return float(r2_score(truth, prediction))


def hierarchical_r2(
    truth: np.ndarray, prediction: np.ndarray, groups: np.ndarray
) -> tuple[float, int]:
    values: list[float] = []
    for group in pd.unique(groups):
        mask = groups == group
        score = safe_r2(truth[mask], prediction[mask])
        if np.isfinite(score):
            values.append(score)
    return (
        float(np.mean(values)) if values else float("nan"),
        int(len(values)),
    )


def metrics(
    frame: pd.DataFrame,
    prediction_norm: np.ndarray,
    indices: np.ndarray | None = None,
) -> dict[str, float | int]:
    if indices is None:
        indices = np.arange(len(frame))
    subset = frame.iloc[indices]
    truth_norm = pd.to_numeric(subset[TARGET], errors="coerce").to_numpy(float)
    truth_hp = pd.to_numeric(subset[RAW_TARGET], errors="coerce").to_numpy(float)
    factor = (
        pd.to_numeric(subset["weight"], errors="coerce").to_numpy(float) ** 0.75
    )
    prediction_norm = np.asarray(prediction_norm, dtype=float)[indices]
    prediction_hp = prediction_norm * factor
    valid = (
        np.isfinite(truth_norm)
        & np.isfinite(truth_hp)
        & np.isfinite(prediction_norm)
        & np.isfinite(prediction_hp)
    )
    truth_norm = truth_norm[valid]
    truth_hp = truth_hp[valid]
    prediction_norm = prediction_norm[valid]
    prediction_hp = prediction_hp[valid]
    pigs = subset["pig"].astype(str).to_numpy()[valid]
    outer_groups = subset["_outer_group"].astype(str).to_numpy()[valid]
    pig_r2, n_pig = hierarchical_r2(truth_hp, prediction_hp, pigs)
    group_mae = [
        mean_absolute_error(
            truth_hp[outer_groups == group],
            prediction_hp[outer_groups == group],
        )
        for group in pd.unique(outer_groups)
    ]
    mae_hp = float(mean_absolute_error(truth_hp, prediction_hp))
    return {
        "r2_norm_pool": safe_r2(truth_norm, prediction_norm),
        "mae_norm": float(mean_absolute_error(truth_norm, prediction_norm)),
        "rmse_norm": float(mean_squared_error(truth_norm, prediction_norm) ** 0.5),
        "r2_HP_pool": safe_r2(truth_hp, prediction_hp),
        "r2_HP_hier_pig": pig_r2,
        "n_hier_pigs": n_pig,
        "mae_HP": mae_hp,
        "rmse_HP": float(mean_squared_error(truth_hp, prediction_hp) ** 0.5),
        "mae_pct_mean_HP": float(mae_hp / np.mean(truth_hp) * 100.0),
        "outer_group_macro_mae_HP": float(np.mean(group_mae)),
        "n": int(valid.sum()),
    }


def fit_oof(
    frame: pd.DataFrame,
    spec: dict[str, list[str]],
    candidate: dict[str, object],
    splits: Iterable[tuple[np.ndarray, np.ndarray]],
    seed: int,
) -> np.ndarray:
    features = spec["numeric"] + spec["categorical"]
    x = frame[features].replace([np.inf, -np.inf], np.nan)
    y = pd.to_numeric(frame[TARGET], errors="coerce").to_numpy(float)
    prediction = np.full(len(frame), np.nan)
    for train, test in splits:
        estimator = make_estimator(candidate, spec, seed)
        estimator.fit(x.iloc[train], y[train])
        prediction[test] = np.asarray(
            estimator.predict(x.iloc[test]), dtype=float
        ).reshape(-1)
    if not np.isfinite(prediction).all():
        raise ValueError(
            f"non-finite OOF predictions: {candidate['name']}"
        )
    return prediction


def inner_search(
    train_frame: pd.DataFrame,
    variants: dict[str, dict[str, list[str]]],
    candidates: list[dict[str, object]],
    scenario: str,
    inner_folds: int,
) -> list[dict[str, object]]:
    inner = grouped_splits(train_frame, scenario, inner_folds)
    results: list[dict[str, object]] = []
    for variant_name, spec in variants.items():
        for candidate in candidates:
            if not candidate_allowed(candidate, variant_name):
                continue
            prediction = fit_oof(
                train_frame,
                spec,
                candidate,
                inner,
                seed=SELECTION_SEED,
            )
            result = metrics(train_frame, prediction)
            results.append(
                {
                    "variant": variant_name,
                    "candidate": candidate,
                    "selection_score": float(result["r2_norm_pool"]),
                    "metrics": result,
                }
            )
    if not results:
        raise ValueError("inner candidate search produced no results")
    return sorted(
        results,
        key=lambda item: (
            -np.inf
            if not np.isfinite(item["selection_score"])
            else item["selection_score"]
        ),
        reverse=True,
    )


def best_for_variant(
    results: list[dict[str, object]], variant_name: str
) -> dict[str, object]:
    matches = [row for row in results if row["variant"] == variant_name]
    if not matches:
        raise ValueError(f"no inner result for {variant_name}")
    return matches[0]


def diverse_top(
    results: list[dict[str, object]], count: int = 3
) -> list[dict[str, object]]:
    """Select inner-ranked candidates from distinct model families."""
    selected: list[dict[str, object]] = []
    families: set[str] = set()
    for row in results:
        family = str(row["candidate"]["family"])
        if family in families:
            continue
        selected.append(row)
        families.add(family)
        if len(selected) == count:
            break
    if len(selected) < min(count, len(results)):
        raise ValueError("could not construct a diverse inner-selected blend")
    return selected


def cluster_bootstrap(
    frame: pd.DataFrame,
    prediction_norm: np.ndarray,
    cluster_column: str,
    reps: int,
    seed: int = 42,
) -> dict[str, object]:
    groups = frame[cluster_column].astype(str).to_numpy()
    unique = pd.unique(groups)
    locations = {group: np.flatnonzero(groups == group) for group in unique}
    rng = np.random.default_rng(seed)
    values: dict[str, list[float]] = {
        "r2_norm_pool": [],
        "r2_HP_pool": [],
        "mae_HP": [],
    }
    for _ in range(reps):
        sampled = rng.choice(unique, size=len(unique), replace=True)
        indices = np.concatenate([locations[group] for group in sampled])
        result = metrics(frame, prediction_norm, indices)
        for name in values:
            value = float(result[name])
            if np.isfinite(value):
                values[name].append(value)
    return {
        "cluster_column": cluster_column,
        "cluster_count": int(len(unique)),
        "valid_replicates": {
            name: len(samples) for name, samples in values.items()
        },
        "ci95": {
            name: (
                np.quantile(samples, [0.025, 0.975]).tolist()
                if samples
                else [float("nan"), float("nan")]
            )
            for name, samples in values.items()
        },
        "warning": (
            "conditional cluster bootstrap of fixed nested-OOF predictions; "
            "models are not refit in bootstrap replicates"
        ),
    }


def paired_cluster_bootstrap(
    frame: pd.DataFrame,
    reference: np.ndarray,
    augmented: np.ndarray,
    cluster_column: str,
    reps: int,
    seed: int = 2026,
) -> dict[str, object]:
    groups = frame[cluster_column].astype(str).to_numpy()
    unique = pd.unique(groups)
    locations = {group: np.flatnonzero(groups == group) for group in unique}
    rng = np.random.default_rng(seed)
    deltas: dict[str, list[float]] = {
        "delta_r2_norm_pool": [],
        "delta_r2_HP_pool": [],
        "delta_mae_HP": [],
    }
    for _ in range(reps):
        sampled = rng.choice(unique, size=len(unique), replace=True)
        indices = np.concatenate([locations[group] for group in sampled])
        before = metrics(frame, reference, indices)
        after = metrics(frame, augmented, indices)
        values = {
            "delta_r2_norm_pool": (
                float(after["r2_norm_pool"]) - float(before["r2_norm_pool"])
            ),
            "delta_r2_HP_pool": (
                float(after["r2_HP_pool"]) - float(before["r2_HP_pool"])
            ),
            "delta_mae_HP": float(after["mae_HP"]) - float(before["mae_HP"]),
        }
        for name, value in values.items():
            if np.isfinite(value):
                deltas[name].append(value)
    return {
        "cluster_column": cluster_column,
        "cluster_count": int(len(unique)),
        "point_delta": {
            "delta_r2_norm_pool": (
                float(metrics(frame, augmented)["r2_norm_pool"])
                - float(metrics(frame, reference)["r2_norm_pool"])
            ),
            "delta_r2_HP_pool": (
                float(metrics(frame, augmented)["r2_HP_pool"])
                - float(metrics(frame, reference)["r2_HP_pool"])
            ),
            "delta_mae_HP": (
                float(metrics(frame, augmented)["mae_HP"])
                - float(metrics(frame, reference)["mae_HP"])
            ),
        },
        "ci95": {
            name: (
                np.quantile(samples, [0.025, 0.975]).tolist()
                if samples
                else [float("nan"), float("nan")]
            )
            for name, samples in deltas.items()
        },
        "warning": (
            "paired conditional cluster bootstrap of fixed nested-OOF "
            "predictions; model-selection uncertainty beyond the outer folds "
            "is not refit"
        ),
    }


def _calibration_keys(frame: pd.DataFrame, per_unit: bool) -> np.ndarray:
    keys = frame["pig"].astype(str)
    if per_unit:
        keys = keys + "::" + frame["experimental_unit"].astype(str)
    return keys.to_numpy()


def calibration_variants(
    scenario: str,
) -> list[tuple[str, bool, float | None]]:
    """Fixed-alpha calibration layers used in the paper."""
    if scenario != "GKF_Pig":
        raise ValueError("Only pig-grouped evaluation is part of the public protocol")
    return [
        ("hybrid_unit_fixed050", True, ALPHA_FIXED),
        ("hybrid_pig_fixed050", False, ALPHA_FIXED),
    ]


def hybrid_calibration(
    frame: pd.DataFrame,
    prediction_norm: np.ndarray,
    outer_fold: np.ndarray,
    scenario: str,
    per_unit: bool,
    alpha: float,
) -> tuple[np.ndarray, dict[str, object]]:
    """Ratio-calibrate fed-phase HP predictions per pig/unit key.

    A key's truth/prediction ratio over its phase 0/1 rows scales that key's
    fed-phase (2/3) predictions by ``1 + alpha * (ratio - 1)``.  Fasting
    phase 4 and the calibration rows themselves keep the raw prediction.
    Keys without calibration rows use ratio 1 (unchanged). Returns the calibrated
    prediction in normalized (HP/W^0.75) space plus an info dict.
    """
    weight = pd.to_numeric(frame["weight"], errors="coerce").to_numpy(float)
    factor = weight ** 0.75
    truth_hp = pd.to_numeric(frame[RAW_TARGET], errors="coerce").to_numpy(float)
    prediction_hp = np.asarray(prediction_norm, dtype=float) * factor
    phase = pd.to_numeric(frame["phase_idx"], errors="raise").to_numpy(int)
    keys = _calibration_keys(frame, per_unit)
    unique_keys = pd.unique(keys)
    fed = np.isin(phase, FED_TEST_PHASES)
    calibrated = prediction_hp.copy()
    info: dict[str, object] = {"per_unit": bool(per_unit)}

    if scenario == "GKF_Pig":
        # Few-shot deployment declaration: the held pig's own phase 0/1 truth
        # forms the ratio.  Every prediction is an honest grouped-OOF
        # prediction, so no in-sample bias enters the ratio.
        cal = np.isin(phase, CAL_PHASES)
        ratios: dict[str, float] = {}
        for key in unique_keys:
            mask = (keys == key) & cal
            if mask.any():
                ratios[str(key)] = float(
                    np.mean(truth_hp[mask])
                    / max(float(np.mean(prediction_hp[mask])), 1e-8)
                )
        alpha_by_key = {str(key): float(alpha) for key in unique_keys}
        info["alpha"] = float(alpha)
        for key in unique_keys:
            mask = fed & (keys == key)
            if not mask.any():
                continue
            ratio = ratios.get(str(key), 1.0)
            calibrated[mask] = prediction_hp[mask] * (
                1.0 + alpha_by_key[str(key)] * (ratio - 1.0)
            )
        info["n_calibrated_keys"] = int(len(ratios))
        info["keys_without_calibration_data"] = int(
            len(unique_keys) - len(ratios)
        )
    else:
        raise ValueError(scenario)
    return calibrated / factor, info


def run_nested(
    frame: pd.DataFrame,
    variants: dict[str, dict[str, list[str]]],
    candidates: list[dict[str, object]],
    scenario: str,
    mode: str,
    bootstrap_reps: int,
) -> tuple[dict[str, object], pd.DataFrame, dict[str, np.ndarray]]:
    outer_folds = 3 if mode == "smoke" else 5
    inner_folds = 2 if mode == "smoke" else 3
    seeds = (SELECTION_SEED,) if mode == "smoke" else SEEDS
    outer = grouped_splits(frame, scenario, outer_folds)
    labels = list(variants) + ["best_any", "blend_top3"]
    predictions = {
        label: {
            seed: np.full(len(frame), np.nan)
            for seed in seeds
        }
        for label in labels
    }
    selected_candidate = {
        label: np.full(len(frame), "", dtype=object) for label in labels
    }
    selected_variant = {
        label: np.full(len(frame), "", dtype=object) for label in labels
    }
    outer_fold = np.full(len(frame), -1, dtype=int)
    fold_records: list[dict[str, object]] = []

    for fold, (outer_train, outer_test) in enumerate(outer, start=1):
        outer_fold[outer_test] = fold
        train_frame = frame.iloc[outer_train].reset_index(drop=True)
        search = inner_search(
            train_frame,
            variants,
            candidates,
            scenario,
            inner_folds,
        )
        choices = {
            name: best_for_variant(search, name)
            for name in variants
        }
        choices["best_any"] = search[0]
        blend_choices = diverse_top(search, count=3)
        fit_cache: dict[tuple[str, str, int], np.ndarray] = {}
        x_cache: dict[str, pd.DataFrame] = {}

        for label, choice in choices.items():
            variant_name = str(choice["variant"])
            candidate = dict(choice["candidate"])
            selected_candidate[label][outer_test] = str(candidate["name"])
            selected_variant[label][outer_test] = variant_name
            spec = variants[variant_name]
            features = spec["numeric"] + spec["categorical"]
            if variant_name not in x_cache:
                x_cache[variant_name] = frame[features].replace(
                    [np.inf, -np.inf], np.nan
                )
            x = x_cache[variant_name]
            y = pd.to_numeric(frame[TARGET], errors="coerce").to_numpy(float)
            for seed in seeds:
                cache_key = (variant_name, str(candidate["name"]), seed)
                if cache_key not in fit_cache:
                    estimator = make_estimator(candidate, spec, seed)
                    estimator.fit(x.iloc[outer_train], y[outer_train])
                    fit_cache[cache_key] = np.asarray(
                        estimator.predict(x.iloc[outer_test]), dtype=float
                    ).reshape(-1)
                predictions[label][seed][outer_test] = fit_cache[cache_key]

        selected_candidate["blend_top3"][outer_test] = "+".join(
            str(choice["candidate"]["name"]) for choice in blend_choices
        )
        selected_variant["blend_top3"][outer_test] = "+".join(
            str(choice["variant"]) for choice in blend_choices
        )
        for seed in seeds:
            component_predictions: list[np.ndarray] = []
            for choice in blend_choices:
                variant_name = str(choice["variant"])
                candidate = dict(choice["candidate"])
                spec = variants[variant_name]
                features = spec["numeric"] + spec["categorical"]
                if variant_name not in x_cache:
                    x_cache[variant_name] = frame[features].replace(
                        [np.inf, -np.inf], np.nan
                    )
                x = x_cache[variant_name]
                y = pd.to_numeric(
                    frame[TARGET], errors="coerce"
                ).to_numpy(float)
                cache_key = (variant_name, str(candidate["name"]), seed)
                if cache_key not in fit_cache:
                    estimator = make_estimator(candidate, spec, seed)
                    estimator.fit(x.iloc[outer_train], y[outer_train])
                    fit_cache[cache_key] = np.asarray(
                        estimator.predict(x.iloc[outer_test]), dtype=float
                    ).reshape(-1)
                component_predictions.append(fit_cache[cache_key])
            predictions["blend_top3"][seed][outer_test] = np.mean(
                np.column_stack(component_predictions), axis=1
            )

        group = group_column(scenario)
        fold_records.append(
            {
                "outer_fold": fold,
                "held_groups": sorted(
                    frame.iloc[outer_test][group].astype(str).unique().tolist()
                ),
                "outer_train_rows": int(len(outer_train)),
                "outer_test_rows": int(len(outer_test)),
                "choices": {
                    label: {
                        "variant": choice["variant"],
                        "candidate": choice["candidate"],
                        "inner_selection_score_r2_norm_pool": choice[
                            "selection_score"
                        ],
                        "inner_metrics": choice["metrics"],
                    }
                    for label, choice in choices.items()
                },
                "blend_top3_components": [
                    {
                        "variant": choice["variant"],
                        "candidate": choice["candidate"],
                        "inner_selection_score_r2_norm_pool": choice[
                            "selection_score"
                        ],
                        "inner_metrics": choice["metrics"],
                    }
                    for choice in blend_choices
                ],
                "inner_ranking": search,
            }
        )
        print(
            f"{mode} {scenario} fold={fold}/{outer_folds} "
            f"best_any={choices['best_any']['variant']}::"
            f"{choices['best_any']['candidate']['name']} "
            f"inner_R2norm={choices['best_any']['selection_score']:+.3f} "
            f"blend={'+'.join(str(item['candidate']['name']) for item in blend_choices)}",
            flush=True,
        )

    if (outer_fold < 0).any():
        raise AssertionError("some rows did not receive an outer fold")

    ensembles: dict[str, np.ndarray] = {}
    branch_summary: dict[str, object] = {}
    cluster = group_column(scenario)
    for label in labels:
        for seed, prediction in predictions[label].items():
            if not np.isfinite(prediction).all():
                raise ValueError(f"non-finite {label} seed {seed} OOF prediction")
        ensemble = np.mean(
            np.column_stack(list(predictions[label].values())), axis=1
        )
        ensembles[label] = ensemble
        bootstrap: dict[str, object] = {
            "outer_group": cluster_bootstrap(
                frame, ensemble, cluster, bootstrap_reps
            )
        }
        branch_summary[label] = {
            "metrics_by_seed": {
                str(seed): metrics(frame, prediction)
                for seed, prediction in predictions[label].items()
            },
            "ensemble_metrics": metrics(frame, ensemble),
            "cluster_bootstrap": bootstrap,
        }

    ablations: dict[str, object] = {}
    for reference_name, augmented_name, label in ABLATION_PAIRS:
        if reference_name not in ensembles or augmented_name not in ensembles:
            continue
        ablations[label] = paired_cluster_bootstrap(
            frame,
            ensembles[reference_name],
            ensembles[augmented_name],
            cluster,
            bootstrap_reps,
        )

    # Hybrid per-pig/per-unit ratio calibration layers on every branch
    # ensemble.  Uncalibrated outputs above stay untouched; calibrated
    # variants are evaluated on the phase 2/3/4 subset only.
    phase_idx = pd.to_numeric(frame["phase_idx"], errors="raise").to_numpy(int)
    eval_indices = np.flatnonzero(np.isin(phase_idx, EVAL_PHASES))
    eval_frame = frame.iloc[eval_indices].reset_index(drop=True)
    calibrated_predictions: dict[str, dict[str, np.ndarray]] = {}
    for label in labels:
        raw = ensembles[label]
        hybrid: dict[str, object] = {}
        calibrated_predictions[label] = {}
        for variant_name, per_unit, alpha in calibration_variants(scenario):
            calibrated_norm, info = hybrid_calibration(
                frame, raw, outer_fold, scenario, per_unit, alpha
            )
            calibrated_predictions[label][variant_name] = calibrated_norm
            hybrid[variant_name] = {
                "alpha": info["alpha"],
                "per_unit": per_unit,
                "eval_subset": (
                    "phase_idx in [2,3,4]; fed phases calibrated, "
                    "fasting phase raw"
                ),
                "metrics_eval_subset": metrics(
                    frame, calibrated_norm, eval_indices
                ),
                "metrics_full_oof_reference": metrics(frame, calibrated_norm),
                "paired_delta_vs_raw_eval_subset": paired_cluster_bootstrap(
                    eval_frame,
                    raw[eval_indices],
                    calibrated_norm[eval_indices],
                    cluster,
                    bootstrap_reps,
                ),
                "n_calibrated_keys": info["n_calibrated_keys"],
                "keys_without_calibration_data": info[
                    "keys_without_calibration_data"
                ],
            }
        branch_summary[label]["hybrid_calibration"] = hybrid

    oof = pd.DataFrame(
        {
            "datetime": frame["datetime"],
            "pig": frame["pig"],
            "strict_day_group": frame["strict_day_group"],
            "animal_day_group_audit_only": frame["animal_day_group"],
            "experimental_unit": frame["experimental_unit"],
            "diet_code_label": frame["diet_code"],
            "phase_idx": frame["phase_idx"],
            "is_fasting": frame["is_fasting"],
            "HP_true": frame[RAW_TARGET],
            "HP_per_W075_true": frame[TARGET],
            "weight": frame["weight"],
            "outer_fold": outer_fold,
        }
    )
    factor = pd.to_numeric(frame["weight"], errors="coerce").to_numpy(float) ** 0.75
    for label in labels:
        oof[f"selected_variant__{label}"] = selected_variant[label]
        oof[f"selected_candidate__{label}"] = selected_candidate[label]
        for seed, prediction in predictions[label].items():
            oof[f"pred_norm__{label}__seed{seed}"] = prediction
            oof[f"pred_HP__{label}__seed{seed}"] = prediction * factor
        oof[f"pred_norm__{label}__ensemble"] = ensembles[label]
        oof[f"pred_HP__{label}__ensemble"] = ensembles[label] * factor
        for variant_name, calibrated_norm in calibrated_predictions[
            label
        ].items():
            oof[f"pred_norm__{label}__{variant_name}"] = calibrated_norm
            oof[f"pred_HP__{label}__{variant_name}"] = calibrated_norm * factor

    summary = {
        "scenario": scenario,
        "mode": mode,
        "status": (
            "strict nested internal validation; model family, hyperparameters "
            "and best_any feature variant are selected inside each outer "
            "training fold; not an external test"
        ),
        "outer_group_column": cluster,
        "forbidden_day_group": "animal_day_group",
        "outer_folds": outer_folds,
        "inner_folds": inner_folds,
        "inner_selection_metric": "r2_norm_pool",
        "seeds": list(seeds),
        "candidate_library_frozen_in_code": candidates,
        "feature_variants": variants,
        "fold_records": fold_records,
        "branches": branch_summary,
        "paired_ablations": ablations,
        "calibration_protocol": {
            "CAL_PHASES": CAL_PHASES,
            "FED_TEST_PHASES": FED_TEST_PHASES,
            "FASTING_PHASE": FASTING_PHASE,
            "EVAL_PHASES": EVAL_PHASES,
            "ALPHA_FIXED": ALPHA_FIXED,
            "ratio_space": (
                "raw HP kcal; ratio = mean(truth_HP) / mean(pred_HP) over "
                "the key's calibration rows"
            ),
            "GKF_Pig_ratio_scope": (
                "few-shot deployment declaration: the held pig's own "
                "phase 0/1 truth forms the ratio; all predictions are "
                "grouped OOF, so this is declared few-shot calibration, "
                "not in-sample fitting"
            ),
        },
        "scientific_boundaries": {
            "diet_code": "categorical identity only; never ordinal",
            "planned_feed": "offered amount, not measured intake",
            "meal_events": "fixed protocol times, not electronic meal logs",
            "hybrid_calibration": (
                "post-hoc ratio calibration on grouped-OOF predictions in "
                "raw-HP space; calibration data are phase 0/1 rows only, "
                "fed phases 2/3 are scaled, fasting phase 4 stays raw; "
                "GKF_Pig ratios include the held pig's adaptation truth by declared "
                "few-shot design"
            ),
            "weight": (
                "current panel uses within-period interpolated weight; a "
                "last-observation-only weight sensitivity remains advisable "
                "for prospective deployment"
            ),
        },
    }
    return summary, oof, ensembles


def to_builtin(value: object) -> object:
    if isinstance(value, dict):
        return {str(key): to_builtin(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_builtin(item) for item in value]
    if isinstance(value, np.ndarray):
        return [to_builtin(item) for item in value.tolist()]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, Path):
        return str(value)
    return value
