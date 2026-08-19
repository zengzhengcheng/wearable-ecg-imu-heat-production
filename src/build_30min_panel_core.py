"""Matched-sample 2026 IMU + HR + HP modeling on 30-minute bins.

The input is the expanded five-minute feature table made by
``extract_features_2026_expanded.py``.  Only windows with usable movement are
aggregated.  The primary panel additionally requires usable HR in at least half
of the component windows, so context, HR, IMU, and IMU+HR schemes are compared
on exactly the same observations.

The target is HP/W^0.75 during fitting and predictions are transformed back to
HP_kcal for all reported metrics.  KFold is an explicitly leaky upper bound;
The analysis groups folds by ear-tag-backed true animal_id.
animal-day.  ``experimental_unit=chamber+period`` is never used as a pig key.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from extract_features_2026_expanded import MOVE_FRAC_COLS, MOVE_STAT_COLS
except ImportError:
    from extract_features_2026 import MOVE_FRAC_COLS, MOVE_STAT_COLS
from model_hr_2026 import cv_splits, models, oof_predict, score


HERE = Path(__file__).resolve().parent


def default_share_root() -> Path:
    mapped = Path(__file__).resolve().parents[1] / "data"
    if mapped.exists():
        return mapped
    candidate = Path(__file__).resolve().parents[2]
    if (candidate / "2026").exists():
        return candidate
    return mapped


SHARE_ROOT = Path(os.environ.get("SHARE_ROOT", default_share_root()))
INPUT = Path(
    os.environ.get(
        "MULTIMODAL_FEATURES_5MIN",
        SHARE_ROOT / "2026" / "model_ready" / "2026_features_newmove_expanded.csv",
    )
)
OUT = Path(
    os.environ.get(
        "MULTIMODAL_RESULTS_OUT",
        SHARE_ROOT / "2026" / "model_ready" / "multimodal_results",
    )
)

HRV = [
    "beat_count", "HR_mean", "HR_std", "RR_mean", "SDNN", "RMSSD",
    "pNN50", "RR_cv", "RR_skew", "RR_kurt", "RR_range",
]
CONTEXT = [
    "hour_of_day", "sin_hour", "cos_hour", "weight", "is_fasting",
    "phase_idx", "win_s",
]
ENV = ["小室温度(℃)", "小室湿度(%)"]
FEED = [
    "planned_feed_kg_day", "planned_feed_g_meal", "planned_feed_g_per_kg_bw",
    "planned_feed_g_per_w075",
]


def motion_columns(columns: pd.Index) -> list[str]:
    wanted: list[str] = []
    for base in MOVE_STAT_COLS:
        wanted.extend(f"{base}_{suffix}" for suffix in ("mean", "std", "p25", "p75"))
    wanted.extend(MOVE_FRAC_COLS)
    wanted.extend(
        ["move_coverage", "StepFreq_std", "LyingTransitions",
         "HeadDownTransitions", "data_count_mean"]
    )
    return [name for name in wanted if name in columns]


def make_30min_panel(raw: pd.DataFrame) -> pd.DataFrame:
    raw = raw.copy()
    raw["datetime"] = pd.to_datetime(raw["datetime"], format="mixed", errors="coerce")
    raw["HP_kcal"] = pd.to_numeric(raw["HP_kcal"], errors="coerce")
    raw["move_coverage"] = pd.to_numeric(raw["move_coverage"], errors="coerce")
    raw["hr_valid"] = pd.to_numeric(raw["hr_valid"], errors="coerce")
    raw = raw[
        raw["datetime"].notna()
        & (raw["HP_kcal"] > 0)
        & (raw["move_coverage"] >= 0.5)
    ].copy()
    raw["time_bin"] = raw["datetime"].dt.floor("30min")

    keys = [
        "chamber", "pig", "animal_id", "ear_tag", "experimental_unit", "legacy_pig",
        "day_group", "period", "phase_idx", "phase_label", "is_fasting",
        "experiment_date", "feed_measurement", "time_bin",
    ]
    keys = [key for key in keys if key in raw.columns]
    numeric = raw.select_dtypes(include=[np.number]).columns.tolist()
    numeric = [col for col in numeric if col not in keys]
    panel = raw.groupby(keys, observed=True, sort=True)[numeric].mean().reset_index()
    counts = raw.groupby(keys, observed=True, sort=True).size().rename("n_windows").reset_index()
    panel = panel.merge(counts, on=keys, how="left", validate="one_to_one")
    panel = panel[panel["n_windows"] >= 3].copy()
    panel.rename(columns={"time_bin": "datetime", "hr_valid": "hr_valid_frac"}, inplace=True)

    # Use the centre of the 30-minute prediction bin for the cyclic clock features.
    centre = panel["datetime"] + pd.Timedelta(minutes=15)
    hod = centre.dt.hour + centre.dt.minute / 60.0
    panel["hour_of_day"] = hod
    panel["sin_hour"] = np.sin(2 * np.pi * hod / 24.0)
    panel["cos_hour"] = np.cos(2 * np.pi * hod / 24.0)
    panel["HP_per_W075"] = panel["HP_kcal"] / panel["weight"].pow(0.75)
    return panel.reset_index(drop=True)


