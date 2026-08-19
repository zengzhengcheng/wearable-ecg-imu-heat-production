"""Build the audited 2026-only 30-minute diet/sensor HP panel.

The source contains 12 ear-tag-backed animals, 24 chamber-period experimental
units and seven coded dietary treatments.  Diet codes are carried as labels,
never as an ordinal numeric predictor.  Planned feed is offered quantity rather
than measured intake, and the two meal times are fixed-schedule candidates
rather than electronic feeding records.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent
COHORT = PROJECT / "src"
if str(COHORT) not in sys.path:
    sys.path.insert(0, str(COHORT))

from feeding_event_features import build_for_shift  # noqa: E402
from build_30min_panel_core import motion_columns  # noqa: E402


DEFAULT_ROOT = PROJECT / "results"
DEFAULT_INPUT = PROJECT / "data" / "derived" / "sensor_features_5min.csv"

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
PLANNED_FEED = [
    "planned_feed_kg_day",
    "planned_feed_g_meal",
    "planned_feed_g_per_kg_bw",
    "planned_feed_g_per_w075",
]
LAST_VALUE_COLUMNS = [
    "cohort",
    "phase_idx",
    "phase_label",
    "is_fasting",
    "day_in_phase",
    "diet_code",
    "feed_measurement",
    *PLANNED_FEED,
]
IDENTITY_COLUMNS = [
    "chamber",
    "pig",
    "animal_id",
    "ear_tag",
    "experimental_unit",
    "legacy_pig",
    "period",
]


def deduplicate_source(raw: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, object]]:
    raw = raw.copy()
    raw["datetime"] = pd.to_datetime(
        raw["datetime"], format="mixed", errors="coerce"
    )
    key = ["pig", "datetime"]
    duplicate_rows = raw.duplicated(key, keep=False)
    duplicate_keys = int(raw.loc[duplicate_rows, key].drop_duplicates().shape[0])
    # The duplicated June-11 records are identical sensor/HP observations
    # assigned to overlapping adaptation and formal phases.  The later phase is
    # retained because the transition is terminal and phase_idx is ordered.
    raw["_source_order"] = np.arange(len(raw))
    raw["_phase_order"] = pd.to_numeric(raw["phase_idx"], errors="coerce").fillna(-1)
    raw = (
        raw.sort_values(key + ["_phase_order", "_source_order"])
        .drop_duplicates(key, keep="last")
        .drop(columns=["_source_order", "_phase_order"])
        .reset_index(drop=True)
    )
    audit = {
        "source_rows_before": int(len(raw) + int(duplicate_rows.sum()) - duplicate_keys),
        "duplicate_pig_datetime_keys": duplicate_keys,
        "duplicate_rows_removed": int(duplicate_rows.sum() - duplicate_keys),
        "deduplication_rule": "keep highest phase_idx, then latest source row",
        "source_rows_after": int(len(raw)),
    }
    return raw, audit


def make_30min_panel(
    raw: pd.DataFrame, offset_minutes: int = 0
) -> pd.DataFrame:
    """Aggregate source records into a disjoint 30-minute grid.

    ``offset_minutes`` shifts the grid origin while preserving a 30-minute
    window.  The production panel uses 0; offsets 0/10/20 together form a
    10-minute-stride augmentation panel.
    """
    frame = raw.copy()
    for column in ("HP_kcal", "weight", "move_coverage", "hr_valid"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame = frame[
        frame["datetime"].notna()
        & (frame["HP_kcal"] > 0)
        & (frame["weight"] > 0)
        & (frame["move_coverage"] >= 0.5)
    ].copy()
    offset = pd.Timedelta(minutes=int(offset_minutes))
    frame["time_bin"] = (
        (frame["datetime"] - offset).dt.floor("30min") + offset
    )
    group_keys = [column for column in IDENTITY_COLUMNS if column in frame.columns]
    group_keys.append("time_bin")

    excluded_numeric = set(group_keys + LAST_VALUE_COLUMNS)
    numeric = [
        column
        for column in frame.select_dtypes(include=[np.number]).columns
        if column not in excluded_numeric
    ]
    averaged = (
        frame.groupby(group_keys, observed=True, sort=True)[numeric]
        .mean()
        .reset_index()
    )
    ordered = frame.sort_values(group_keys + ["datetime"])
    last_columns = [column for column in LAST_VALUE_COLUMNS if column in frame.columns]
    last = (
        ordered.groupby(group_keys, observed=True, sort=True)[last_columns]
        .last()
        .reset_index()
    )
    counts = (
        frame.groupby(group_keys, observed=True, sort=True)
        .size()
        .rename("n_windows")
        .reset_index()
    )
    panel = averaged.merge(last, on=group_keys, validate="one_to_one")
    panel = panel.merge(counts, on=group_keys, validate="one_to_one")
    panel = panel[panel["n_windows"] >= 3].copy()
    panel.rename(
        columns={"time_bin": "datetime", "hr_valid": "hr_valid_frac"},
        inplace=True,
    )

    centre = panel["datetime"] + pd.Timedelta(minutes=15)
    hour = centre.dt.hour + centre.dt.minute / 60.0
    panel["feature_time"] = centre
    panel["hour_of_day"] = hour
    panel["sin_hour"] = np.sin(2 * np.pi * hour / 24.0)
    panel["cos_hour"] = np.cos(2 * np.pi * hour / 24.0)
    experiment_date = (centre - pd.Timedelta(hours=9)).dt.strftime("%Y-%m-%d")
    panel["experiment_date"] = experiment_date
    panel["strict_day_group"] = "2026::" + experiment_date
    panel["animal_day_group"] = panel["pig"].astype(str) + "::" + experiment_date
    # New code should use strict_day_group for cross-day generalisation.  Keep
    # day_group as a strict-date alias so legacy helpers cannot silently make a
    # pig-day split.
    panel["day_group"] = panel["strict_day_group"]
    panel["HP_per_W075"] = panel["HP_kcal"] / panel["weight"].pow(0.75)
    panel["is_adaptation"] = panel["phase_idx"].eq(0).astype(float)
    panel["is_formal"] = panel["phase_idx"].between(1, 3).astype(float)
    panel["planned_feed_available"] = panel[PLANNED_FEED].notna().all(axis=1).astype(float)
    panel["planned_meal_g_per_w075"] = (
        panel["planned_feed_g_meal"] / panel["weight"].pow(0.75)
    )
    fed = 1.0 - pd.to_numeric(panel["is_fasting"], errors="raise").clip(0, 1)
    panel["offered_feed_kg_day_if_fed"] = panel["planned_feed_kg_day"] * fed
    panel["offered_feed_g_meal_if_fed"] = panel["planned_feed_g_meal"] * fed
    panel["planned_meal_g_per_w075_if_fed"] = panel["planned_meal_g_per_w075"] * fed
    return panel.sort_values(["pig", "datetime"]).reset_index(drop=True)


def add_feeding_features(panel: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, object]]:
    events, featured, groups, event_audit = build_for_shift(panel, 0)
    dose = pd.to_numeric(
        featured["planned_meal_g_per_w075_if_fed"], errors="coerce"
    )
    dose_interactions: list[str] = []
    for kernel in groups["meal_kernels"]:
        if kernel not in featured:
            continue
        name = f"dose_x_{kernel}"
        featured[name] = dose * pd.to_numeric(featured[kernel], errors="coerce")
        dose_interactions.append(name)
    groups["planned_feed"] = [
        *PLANNED_FEED,
        "planned_meal_g_per_w075",
        "offered_feed_kg_day_if_fed",
        "offered_feed_g_meal_if_fed",
        "planned_meal_g_per_w075_if_fed",
        "planned_feed_available",
    ]
    groups["dose_interactions"] = dose_interactions
    groups["diet_identity"] = ["diet_code"]
    groups["all_diet_feeding"] = list(
        dict.fromkeys(
            groups["planned_feed"]
            + groups["event_clock"]
            + groups["meal_kernels"]
            + groups["dose_interactions"]
        )
    )
    return featured, {"event_audit": event_audit, "feature_groups": groups, "events": events}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--out", type=Path, default=DEFAULT_ROOT)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    raw = pd.read_csv(
        args.input,
        low_memory=False,
        dtype={"ear_tag": "string", "animal_id": "string"},
    )
    raw, dedup_audit = deduplicate_source(raw)
    panel_all = make_30min_panel(raw)
    panel_matched = panel_all[
        pd.to_numeric(panel_all["hr_valid_frac"], errors="coerce") >= 0.5
    ].copy().reset_index(drop=True)
    featured, feeding = add_feeding_features(panel_matched)

    if featured.duplicated(["pig", "datetime"]).any():
        raise AssertionError("final matched panel has duplicate pig/datetime keys")
    if featured["pig"].nunique() != 12:
        raise AssertionError("expected 12 true animals")
    if featured["experimental_unit"].nunique() != 24:
        raise AssertionError("expected 24 experimental units")
    if set(pd.to_numeric(featured["diet_code"]).dropna().astype(int)) != set(range(7)):
        raise AssertionError("expected diet codes 0..6")

    panel_all.to_csv(
        args.out / "2026_diet_30min_all_move.csv",
        index=False,
        encoding="utf-8-sig",
    )
    featured.to_csv(
        args.out / "2026_diet_30min_matched.csv",
        index=False,
        encoding="utf-8-sig",
    )
    feeding["events"].to_csv(
        args.out / "candidate_feeding_events.csv",
        index=False,
        encoding="utf-8-sig",
    )
    with (args.out / "feature_groups.json").open("w", encoding="utf-8") as handle:
        json.dump(feeding["feature_groups"], handle, ensure_ascii=False, indent=2)

    motion = motion_columns(featured.columns)
    audit = {
        "status": "PASS",
        "input": str(args.input),
        "scientific_boundary": {
            "cohort": "2026 growing pigs only",
            "target_fit": "HP_kcal / weight^0.75",
            "reported_target": "HP_kcal and HP_kcal / weight^0.75",
            "diet_code": "categorical treatment label 0..6; composition not supplied",
            "feed_quantity": "planned offered quantity, not measured intake",
            "meal_times": "fixed-schedule candidates 14:30 and next-day 08:30; no raw meal log",
            "strict_day_split": "whole 09:00-boundary experiment date",
            "pig_split": "ear-tag-backed true animal_id",
        },
        **dedup_audit,
        "rows_30min_all_move": int(len(panel_all)),
        "rows_30min_matched": int(len(featured)),
        "pigs": int(featured["pig"].nunique()),
        "experimental_units": int(featured["experimental_unit"].nunique()),
        "strict_days": int(featured["strict_day_group"].nunique()),
        "animal_days": int(featured["animal_day_group"].nunique()),
        "periods": sorted(map(int, featured["period"].unique())),
        "diet_codes": sorted(map(int, featured["diet_code"].unique())),
        "motion_features": len(motion),
        "hrv_features": int(sum(column in featured for column in HRV)),
        "unique_pig_datetime": bool(
            not featured.duplicated(["pig", "datetime"]).any()
        ),
        "date_min": str(featured["datetime"].min()),
        "date_max": str(featured["datetime"].max()),
        "event_audit": feeding["event_audit"],
        "diet_distribution": (
            featured.groupby("diet_code", observed=True)
            .agg(
                rows=("HP_kcal", "size"),
                pigs=("pig", "nunique"),
                experimental_units=("experimental_unit", "nunique"),
                mean_weight=("weight", "mean"),
                mean_HP=("HP_kcal", "mean"),
                mean_HP_per_W075=("HP_per_W075", "mean"),
            )
            .reset_index()
            .to_dict(orient="records")
        ),
        "outputs": {
            "all_move": str(args.out / "2026_diet_30min_all_move.csv"),
            "matched": str(args.out / "2026_diet_30min_matched.csv"),
            "events": str(args.out / "candidate_feeding_events.csv"),
            "feature_groups": str(args.out / "feature_groups.json"),
        },
    }
    with (args.out / "data_audit.json").open("w", encoding="utf-8") as handle:
        json.dump(audit, handle, ensure_ascii=False, indent=2)
    print(json.dumps(audit, ensure_ascii=False, indent=2))
    print("DONE_EXIT_0")


if __name__ == "__main__":
    main()
