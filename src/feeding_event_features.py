"""Protocol-sensitivity feeding features for the 2026 30-minute HP panel.

This is deliberately separate from ``model_broad_exploration_2026.py``.  The
electronic source files contain no per-meal feeding log, so 14:30 and next-day
08:30 are *candidate protocol events*, not observed feeding timestamps.  The
script builds an auditable event table on the 09:00 experimental-day clock,
adds causal (zero before the event) multi-scale kernels, and uses only sensor
observations ending before each meal to construct pre-meal baselines.

Experimental-day ownership for day D (D 09:00 -> D+1 09:00):
    meal_1_afternoon: D 14:30
    meal_2_morning:   D+1 08:30

Thus the 08:30 meal remains owned by the previous experimental day and its
effect can continue across the 09:00 fed->fasting boundary.  Fasting days do
not generate events, but they do not erase a preceding event.

Modes:
    smoke (default)  two ablations, 40-tree ExtraTrees, first+last day/pig
    feature-only     construct and audit features without fitting
    evaluate         all ablations on the complete panel (potentially costly)

Examples:
    python feeding_event_features.py --mode feature-only
    python feeding_event_features.py --mode smoke
    python feeding_event_features.py --mode smoke --sensitivity
    python feeding_event_features.py --mode evaluate --save

No HP value is used to create an event, kernel, pre-meal baseline, or
interaction.  The current-bin sensor deltas are contemporaneous predictors;
the baseline against which they are expressed is strictly pre-meal/past-only.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from model_hr_2026 import cv_splits, oof_predict, score
from build_30min_panel_core import OUT as BASE_OUT, feature_sets


PANEL = Path(
    os.environ.get(
        "MULTIMODAL_PANEL_30MIN",
        BASE_OUT / "multimodal_30min_matched.csv",
    )
)
DEFAULT_OUT = BASE_OUT / "feeding_event_features"

KERNEL_TAUS_H = (0.5, 1.5, 3.0, 6.0)
SENSITIVITY_SHIFTS_MIN = (-60, -30, 0, 30, 60)
MAX_EVENT_HISTORY_H = 48.0
PREMEAL_START_MIN = 120
PREMEAL_END_MIN = 30

BASELINE_SENSORS = (
    "HR_mean",
    "RR_mean",
    "ODBA_mean",
    "VeDBA_RMS_mean",
    "Active_Fraction",
    "Frac_Rest",
    "Frac_HeadDown",
    "Jerk_Mean_mean",
)

MEAL_SPECS = {
    "meal_1_afternoon": (0, 14, 30),
    "meal_2_morning": (1, 8, 30),
}


def load_panel(path: Path) -> pd.DataFrame:
    """Read and validate the matched 30-minute panel."""
    df = pd.read_csv(path, low_memory=False)
    required = {
        "pig", "day_group", "experiment_date", "datetime", "is_fasting",
        "weight", "HP_kcal", "HP_per_W075", "chamber",
    }
    missing = sorted(required - set(df.columns))
    if missing:
        raise KeyError(f"30-minute panel is missing required columns: {missing}")
    df["datetime"] = pd.to_datetime(df["datetime"], format="mixed", errors="coerce")
    df["experiment_date"] = pd.to_datetime(df["experiment_date"], errors="coerce").dt.normalize()
    df["is_fasting"] = pd.to_numeric(df["is_fasting"], errors="raise").astype(int)
    df = df[df["datetime"].notna() & df["experiment_date"].notna()].copy()
    df["feature_time"] = df["datetime"] + pd.Timedelta(minutes=15)
    df["experimental_day_start"] = df["experiment_date"] + pd.Timedelta(hours=9)

    derived_date = (df["feature_time"] - pd.Timedelta(hours=9)).dt.normalize()
    mismatch = derived_date.ne(df["experiment_date"])
    if mismatch.any():
        examples = df.loc[mismatch, ["pig", "day_group", "datetime", "experiment_date"]].head()
        raise ValueError(f"09:00 day ownership mismatch in input panel:\n{examples}")
    return df.sort_values(["pig", "feature_time"]).reset_index(drop=True)


def _single_value(series: pd.Series, name: str) -> object:
    values = series.dropna().unique()
    if len(values) != 1:
        raise ValueError(f"{name} is not unique within pig/day: {values[:5]}")
    return values[0]


def build_candidate_events(
    panel: pd.DataFrame,
    meal_shift_min: int = 0,
    afternoon_shift_min: int | None = None,
    morning_shift_min: int | None = None,
) -> pd.DataFrame:
    """Build fixed-schedule candidate events owned by the 09:00 day.

    ``meal_shift_min`` shifts both meals.  Meal-specific shifts override it and
    make it possible to test asymmetric timing uncertainty later.  No event is
    generated for an experimental day labelled fasting.
    """
    afternoon_shift = meal_shift_min if afternoon_shift_min is None else afternoon_shift_min
    morning_shift = meal_shift_min if morning_shift_min is None else morning_shift_min
    shifts = {
        "meal_1_afternoon": int(afternoon_shift),
        "meal_2_morning": int(morning_shift),
    }

    day_rows: list[dict[str, object]] = []
    for (pig, day_group), group in panel.groupby(["pig", "day_group"], sort=False):
        # A single B2 day contains one boundary bin labelled formal3 followed by
        # fasting bins.  Fasting is terminal within a period, so any fasting
        # state makes that 09:00 day a no-meal protocol day.  Preserve the mixed
        # state as an audit flag rather than silently selecting the first row.
        fasting_states = pd.to_numeric(group["is_fasting"], errors="raise").astype(int)
        day_rows.append(
            {
                "pig": str(pig),
                "day_group": str(day_group),
                "experiment_date": pd.Timestamp(_single_value(group["experiment_date"], "experiment_date")),
                "is_fasting": int(fasting_states.max()),
                "mixed_fasting_state": int(fasting_states.nunique() > 1),
                "period": _single_value(group["period"], "period") if "period" in group else np.nan,
                "chamber": _single_value(group["chamber"], "chamber") if "chamber" in group else "",
            }
        )
    days = pd.DataFrame(day_rows)

    records: list[dict[str, object]] = []
    for row in days.itertuples(index=False):
        if int(row.is_fasting) == 1:
            continue
        for meal_id, (day_offset, hour, minute) in MEAL_SPECS.items():
            nominal = (
                pd.Timestamp(row.experiment_date)
                + pd.Timedelta(days=day_offset, hours=hour, minutes=minute)
            )
            shift = shifts[meal_id]
            records.append(
                {
                    "pig": row.pig,
                    "chamber": row.chamber,
                    "period": row.period,
                    "owner_day_group": row.day_group,
                    "owner_experiment_date": row.experiment_date,
                    "owner_is_fasting": row.is_fasting,
                    "owner_mixed_fasting_state": row.mixed_fasting_state,
                    "meal_id": meal_id,
                    "nominal_event_time": nominal,
                    "event_time": nominal + pd.Timedelta(minutes=shift),
                    "meal_shift_min": shift,
                    "feed_time_source": "fixed_schedule_no_raw_log",
                    "time_quality": "protocol_sensitivity_only",
                    "nominal_time_uncertainty_min": 60,
                }
            )
    events = pd.DataFrame(records)
    if events.empty:
        raise ValueError("candidate event table is empty")
    events = events.sort_values(["pig", "event_time", "meal_id"]).reset_index(drop=True)
    events["event_id"] = (
        events["pig"].astype(str) + "__" + events["owner_day_group"].astype(str)
        + "__" + events["meal_id"].astype(str)
    )
    if events["event_id"].duplicated().any():
        raise ValueError("duplicate candidate feeding event IDs")
    if (events["owner_is_fasting"] != 0).any():
        raise AssertionError("a fasting experimental day generated a meal event")
    return events


def compute_premeal_baselines(
    panel: pd.DataFrame,
    events: pd.DataFrame,
    sensor_columns: Iterable[str] = BASELINE_SENSORS,
    start_min: int = PREMEAL_START_MIN,
    end_min: int = PREMEAL_END_MIN,
) -> tuple[pd.DataFrame, list[str]]:
    """Freeze sensor baselines using event[-120 min, -30 min] only."""
    sensors = [column for column in sensor_columns if column in panel.columns]
    if not sensors:
        raise KeyError("none of the requested pre-meal baseline sensors are present")
    by_pig = {
        str(pig): group.sort_values("feature_time")
        for pig, group in panel.groupby("pig", sort=False)
    }
    rows: list[dict[str, object]] = []
    for event in events.itertuples(index=False):
        pig_frame = by_pig.get(str(event.pig))
        end = pd.Timestamp(event.event_time) - pd.Timedelta(minutes=end_min)
        start = pd.Timestamp(event.event_time) - pd.Timedelta(minutes=start_min)
        past = pig_frame[
            pig_frame["feature_time"].ge(start) & pig_frame["feature_time"].le(end)
        ]
        record: dict[str, object] = {
            "event_id": event.event_id,
            "premeal_n_bins": int(len(past)),
            "premeal_window_start": start,
            "premeal_window_end": end,
            "premeal_latest_observation": past["feature_time"].max() if len(past) else pd.NaT,
        }
        for column in sensors:
            record[f"premeal_{column}"] = pd.to_numeric(past[column], errors="coerce").mean()
        rows.append(record)
    baselines = pd.DataFrame(rows)
    late = (
        baselines["premeal_latest_observation"].notna()
        & baselines["premeal_latest_observation"].gt(baselines["premeal_window_end"])
    )
    if late.any():
        raise AssertionError("pre-meal baseline contains a non-past observation")
    return events.merge(baselines, on="event_id", how="left", validate="one_to_one"), sensors


def _tau_label(tau: float) -> str:
    return str(tau).replace(".", "p")


def attach_event_features(
    panel: pd.DataFrame,
    events_with_baseline: pd.DataFrame,
    sensors: Iterable[str],
) -> tuple[pd.DataFrame, dict[str, list[str]]]:
    """Attach most-recent events and causal kernels without using HP."""
    out = panel.copy()
    n = len(out)
    clock_columns = [
        "meal_history_known", "hours_since_last_meal_capped48",
        "last_meal_is_afternoon", "last_meal_is_morning",
        "fed_to_fasting_carryover_6h", "hours_in_fasting",
    ]
    for column in clock_columns:
        out[column] = 0.0
    out["hours_since_last_meal_capped48"] = np.nan

    kernel_columns: list[str] = []
    for meal_id in MEAL_SPECS:
        short = "afternoon" if "afternoon" in meal_id else "morning"
        since_column = f"hours_since_{short}_meal_capped48"
        out[since_column] = np.nan
        clock_columns.append(since_column)
        for tau in KERNEL_TAUS_H:
            column = f"meal_{short}_kernel_tau{_tau_label(tau)}h"
            out[column] = 0.0
            kernel_columns.append(column)

    baseline_columns = ["premeal_n_bins"]
    out["premeal_n_bins"] = 0.0
    for sensor in sensors:
        pre = f"premeal_{sensor}"
        delta = f"delta_from_premeal_{sensor}"
        out[pre] = np.nan
        out[delta] = np.nan
        baseline_columns.extend([pre, delta])

    owner_days = np.full(n, "", dtype=object)
    last_event_ids = np.full(n, "", dtype=object)
    for pig, row_index in out.groupby("pig", sort=False).groups.items():
        loc = np.asarray(list(row_index), dtype=int)
        times = out.loc[loc, "feature_time"].to_numpy(dtype="datetime64[ns]")
        pig_events = events_with_baseline[
            events_with_baseline["pig"].astype(str).eq(str(pig))
        ].sort_values("event_time")
        if pig_events.empty:
            continue
        event_times = pig_events["event_time"].to_numpy(dtype="datetime64[ns]")
        previous = np.searchsorted(event_times, times, side="right") - 1
        known = previous >= 0
        if known.any():
            chosen = pig_events.iloc[previous[known]]
            chosen_times = chosen["event_time"].to_numpy(dtype="datetime64[ns]")
            elapsed = (times[known] - chosen_times) / np.timedelta64(1, "h")
            target_loc = loc[known]
            out.loc[target_loc, "meal_history_known"] = 1.0
            out.loc[target_loc, "hours_since_last_meal_capped48"] = np.minimum(
                elapsed, MAX_EVENT_HISTORY_H
            )
            is_afternoon = chosen["meal_id"].astype(str).str.contains("afternoon").to_numpy(float)
            out.loc[target_loc, "last_meal_is_afternoon"] = is_afternoon
            out.loc[target_loc, "last_meal_is_morning"] = 1.0 - is_afternoon
            owner_days[target_loc] = chosen["owner_day_group"].astype(str).to_numpy()
            last_event_ids[target_loc] = chosen["event_id"].astype(str).to_numpy()
            out.loc[target_loc, "premeal_n_bins"] = chosen["premeal_n_bins"].to_numpy(float)
            for sensor in sensors:
                pre = f"premeal_{sensor}"
                delta = f"delta_from_premeal_{sensor}"
                baseline = pd.to_numeric(chosen[pre], errors="coerce").to_numpy(float)
                out.loc[target_loc, pre] = baseline
                current = pd.to_numeric(out.loc[target_loc, sensor], errors="coerce").to_numpy(float)
                out.loc[target_loc, delta] = current - baseline

        for meal_id in MEAL_SPECS:
            short = "afternoon" if "afternoon" in meal_id else "morning"
            meal_events = pig_events[pig_events["meal_id"].eq(meal_id)]
            if meal_events.empty:
                continue
            meal_times = meal_events["event_time"].to_numpy(dtype="datetime64[ns]")
            meal_previous = np.searchsorted(meal_times, times, side="right") - 1
            valid = meal_previous >= 0
            if not valid.any():
                continue
            elapsed = (
                times[valid] - meal_times[meal_previous[valid]]
            ) / np.timedelta64(1, "h")
            active = elapsed <= MAX_EVENT_HISTORY_H
            target_loc = loc[valid]
            out.loc[target_loc, f"hours_since_{short}_meal_capped48"] = np.minimum(
                elapsed, MAX_EVENT_HISTORY_H
            )
            for tau in KERNEL_TAUS_H:
                values = np.where(active, np.exp(-elapsed / tau), 0.0)
                out.loc[target_loc, f"meal_{short}_kernel_tau{_tau_label(tau)}h"] = values

    out["last_event_id"] = last_event_ids
    out["last_event_owner_day_group"] = owner_days
    fasting = out["is_fasting"].eq(1)
    elapsed = pd.to_numeric(out["hours_since_last_meal_capped48"], errors="coerce")
    out["fed_to_fasting_carryover_6h"] = (
        fasting & out["meal_history_known"].eq(1) & elapsed.le(6)
        & out["last_event_owner_day_group"].ne(out["day_group"].astype(str))
    ).astype(float)
    fasting_start = (
        out.loc[fasting]
        .groupby(["pig", "day_group"], sort=False)["datetime"]
        .transform("min")
    )
    out["hours_in_fasting"] = 0.0
    out.loc[fasting, "hours_in_fasting"] = (
        out.loc[fasting, "feature_time"] - fasting_start
    ).dt.total_seconds() / 3600.0

    interaction_columns: list[str] = []
    any_tau1p5 = (
        out["meal_afternoon_kernel_tau1p5h"] + out["meal_morning_kernel_tau1p5h"]
    )
    any_tau3 = out["meal_afternoon_kernel_tau3p0h"] + out["meal_morning_kernel_tau3p0h"]
    out["meal_any_kernel_tau1p5h"] = any_tau1p5
    out["meal_any_kernel_tau3p0h"] = any_tau3
    kernel_columns.extend(["meal_any_kernel_tau1p5h", "meal_any_kernel_tau3p0h"])
    for sensor in sensors:
        delta = f"delta_from_premeal_{sensor}"
        for label, kernel in (("tau1p5h", any_tau1p5), ("tau3p0h", any_tau3)):
            name = f"meal_{label}_x_delta_{sensor}"
            out[name] = pd.to_numeric(out[delta], errors="coerce") * kernel
            interaction_columns.append(name)

    groups = {
        "event_clock": clock_columns,
        "meal_kernels": kernel_columns,
        "premeal_sensor_baseline": baseline_columns,
        "meal_sensor_interactions": interaction_columns,
    }
    groups["all_feeding"] = list(
        dict.fromkeys(
            groups["event_clock"] + groups["meal_kernels"]
            + groups["premeal_sensor_baseline"] + groups["meal_sensor_interactions"]
        )
    )
    return out, groups


def make_ablation_sets(
    frame_with_dummies: pd.DataFrame,
    feature_groups: dict[str, list[str]],
) -> dict[str, list[str]]:
    base = feature_sets(frame_with_dummies)["context_env_move_hr"]
    present = lambda columns: [column for column in columns if column in frame_with_dummies.columns]
    clock = present(feature_groups["event_clock"])
    kernels = present(feature_groups["meal_kernels"])
    baseline = present(feature_groups["premeal_sensor_baseline"])
    interactions = present(feature_groups["meal_sensor_interactions"])
    dedupe = lambda columns: list(dict.fromkeys(columns))
    return {
        "base_clock_sensor": base,
        "base_plus_event_clock": dedupe(base + clock),
        "base_plus_meal_kernels": dedupe(base + clock + kernels),
        "base_plus_premeal": dedupe(base + clock + kernels + baseline),
        "base_plus_feeding_full": dedupe(base + clock + kernels + baseline + interactions),
        "feeding_only": dedupe(clock + kernels + baseline + interactions),
    }


def smoke_subset(frame: pd.DataFrame) -> pd.DataFrame:
    """Keep first and last experimental day for every pig."""
    selected: list[str] = []
    day_table = (
        frame[["pig", "day_group", "experiment_date"]]
        .drop_duplicates()
        .sort_values(["pig", "experiment_date"])
    )
    for _, group in day_table.groupby("pig", sort=False):
        selected.extend(group.head(1)["day_group"].astype(str).tolist())
        selected.extend(group.tail(1)["day_group"].astype(str).tolist())
    return frame[frame["day_group"].astype(str).isin(set(selected))].reset_index(drop=True)


def smoke_estimator(trees: int = 40) -> Pipeline:
    return Pipeline(
        [
            ("imp", SimpleImputer(strategy="median", add_indicator=True)),
            (
                "model",
                ExtraTreesRegressor(
                    n_estimators=trees,
                    min_samples_leaf=3,
                    max_features=0.65,
                    n_jobs=4,
                    random_state=42,
                ),
            ),
        ]
    )


def evaluate_ablation(
    frame: pd.DataFrame,
    schemes: dict[str, list[str]],
    scenario_names: Iterable[str],
    scheme_names: Iterable[str],
    estimator: Pipeline,
    meal_shift_min: int,
) -> pd.DataFrame:
    y_target = pd.to_numeric(frame["HP_per_W075"], errors="coerce").to_numpy(float)
    factor = pd.to_numeric(frame["weight"], errors="coerce").to_numpy(float) ** 0.75
    truth = pd.to_numeric(frame["HP_kcal"], errors="coerce").to_numpy(float)
    pigs = frame["pig"].astype(str).to_numpy()
    rows: list[dict[str, object]] = []
    for scenario in scenario_names:
        splits = cv_splits(frame, scenario)
        for scheme_name in scheme_names:
            columns = schemes[scheme_name]
            x = frame[columns].replace([np.inf, -np.inf], np.nan)
            pred = oof_predict(estimator, x, y_target, splits) * factor
            rows.append(
                {
                    "scenario": scenario,
                    "scheme": scheme_name,
                    "meal_shift_min": meal_shift_min,
                    "n_features": len(columns),
                    **score(truth, pred, pigs),
                }
            )
    return pd.DataFrame(rows)


def feature_audit(
    panel: pd.DataFrame,
    events: pd.DataFrame,
    featured: pd.DataFrame,
    shift_min: int,
) -> dict[str, object]:
    day_state = panel.groupby(["pig", "day_group"], sort=False)["is_fasting"].max()
    fed_days = int(day_state.eq(0).sum())
    fasting_days = int(day_state.eq(1).sum())
    mixed_days = int(
        panel.groupby(["pig", "day_group"], sort=False)["is_fasting"].nunique().gt(1).sum()
    )
    expected_events = int(fed_days * 2)
    if len(events) != expected_events:
        raise AssertionError(f"expected {expected_events} candidate events, got {len(events)}")
    per_owner = events.groupby("owner_day_group")["meal_id"].nunique()
    if not per_owner.eq(2).all():
        raise AssertionError("a fed day does not own exactly two candidate meals")
    return {
        "protocol_status": "fixed schedule sensitivity; no raw per-meal log",
        "meal_shift_min": int(shift_min),
        "rows": int(len(panel)),
        "pigs": int(panel["pig"].nunique()),
        "experimental_days": int(panel["day_group"].nunique()),
        "fed_days": int(fed_days),
        "fasting_days": int(fasting_days),
        "mixed_state_days_treated_as_fasting": mixed_days,
        "candidate_events": int(len(events)),
        "candidate_events_expected": expected_events,
        "premeal_baseline_available_fraction": float(events["premeal_n_bins"].gt(0).mean()),
        "meal_history_known_fraction": float(featured["meal_history_known"].mean()),
        "fed_to_fasting_carryover_rows": int(featured["fed_to_fasting_carryover_6h"].sum()),
        "max_hours_in_fasting": float(featured["hours_in_fasting"].max()),
    }


def build_for_shift(panel: pd.DataFrame, shift_min: int):
    events = build_candidate_events(panel, meal_shift_min=shift_min)
    events_with_baseline, sensors = compute_premeal_baselines(panel, events)
    featured, groups = attach_event_features(panel, events_with_baseline, sensors)
    audit = feature_audit(panel, events_with_baseline, featured, shift_min)
    return events_with_baseline, featured, groups, audit


