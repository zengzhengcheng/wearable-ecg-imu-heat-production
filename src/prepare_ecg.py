"""Prepare the expanded 2026 heart-rate/heat-production modeling tables.

This is the bridge between the current data drops and modeling:

* HP: ``data/derived/hp/2026_HP_all.csv`` (periods 1-8)
* cleaned R peaks: ``data/raw/ecg/<chamber>/cleanlabels/*.csv``
* experiment design: ``cohort_2026/experiment_design_2026.csv`` normalized from
  ``商业试验饲喂设计.xlsx``

The script leaves raw extraction folders unchanged. It writes consolidated peaks
and model-ready tables under the configurable derived-data directory.

Important protocol choices:

* true pig identity is the ear-tag-backed ``animal_id``;
* ``experimental_unit = chamber + period`` identifies one chamber visit only;
* the experimental day boundary is 09:00;
* the sensor window follows each HP row's recorded 时间差 rather than forcing 300 s;
* weights are interpolated piecewise across adaptation -> formal, formal ->
  pre-fasting, and pre-fasting -> post-fasting anchors for the same ear tag;
* diet code and planned offered feed are attached by phase.  Offered feed is not
  treated as observed intake, and exact feeding timestamps remain unobserved.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats


CHAMBERS = ("A1", "B1", "B2")

def default_share_root() -> Path:
    return Path(__file__).resolve().parents[1] / "data"


SHARE_ROOT = Path(os.environ.get("SHARE_ROOT", default_share_root()))
HP_CSV = Path(os.environ.get("HP_CSV", SHARE_ROOT / "2026" / "HP_extracted" / "2026_HP_all.csv"))
DESIGN_CSV = Path(
    os.environ.get("DESIGN_CSV", Path(__file__).resolve().with_name("experiment_design_2026.csv"))
)
HEART_ROOT = Path(os.environ.get("HEART_ROOT", SHARE_ROOT / "raw" / "ecg"))
HEART_OUT = Path(os.environ.get("HEART_OUT", SHARE_ROOT / "2026" / "heart_extracted"))
MODEL_OUT = Path(os.environ.get("MODEL_OUT", SHARE_ROOT / "2026" / "model_ready"))

RR_MIN_MS = 50.0
RR_MAX_MS = 3000.0
MIN_BEATS = 150
WIN_CLAMP_S = (150.0, 420.0)

HRV_COLS = [
    "HR_mean", "HR_std", "RR_mean", "SDNN", "RMSSD", "pNN50",
    "RR_cv", "RR_skew", "RR_kurt", "RR_range",
]


def atomic_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    df.to_csv(tmp, index=False, encoding="utf-8-sig")
    os.replace(tmp, path)


DESIGN_REQUIRED = {
    "period", "chamber", "experimental_unit", "animal_id", "ear_tag", "diet_code",
    "adapt_weight_kg", "formal_weight_kg", "prefast_weight_kg", "postfast_weight_kg",
    "adapt_feed_kg_day", "adapt_feed_g_meal", "formal_feed_kg_day",
    "formal_feed_g_meal", "feed_measurement",
}


def load_experiment_design(path: Path | str = DESIGN_CSV) -> pd.DataFrame:
    """Load and validate the normalized calorimetry design master."""
    design = pd.read_csv(path, dtype={"ear_tag": "string", "animal_id": "string"})
    missing = DESIGN_REQUIRED - set(design.columns)
    if missing:
        raise ValueError(f"experiment design missing columns: {sorted(missing)}")
    design["period"] = pd.to_numeric(design["period"], errors="raise").astype(int)
    design["ear_tag"] = design["ear_tag"].str.strip().str.zfill(2)
    design["animal_id"] = "pig_" + design["ear_tag"]
    numeric = [
        "diet_code", "adapt_weight_kg", "formal_weight_kg", "prefast_weight_kg",
        "postfast_weight_kg", "adapt_feed_kg_day", "adapt_feed_g_meal",
        "formal_feed_kg_day", "formal_feed_g_meal",
    ]
    design[numeric] = design[numeric].apply(pd.to_numeric, errors="raise")
    if len(design) != 24:
        raise ValueError(f"expected 24 calorimetry experimental units, got {len(design)}")
    if design[["chamber", "period"]].duplicated().any():
        raise ValueError("duplicate chamber-period experimental unit in design")
    if design["experimental_unit"].nunique() != 24:
        raise ValueError("experimental_unit is not unique")
    if design["animal_id"].nunique() != 12:
        raise ValueError(f"expected 12 true calorimetry pigs, got {design['animal_id'].nunique()}")
    if design.loc[design["period"] <= 5, "animal_id"].nunique() != 6:
        raise ValueError("periods 1-5 must map to six true pigs")
    if not design["chamber"].isin(CHAMBERS).all():
        raise ValueError("design contains an unknown chamber")
    if not design["feed_measurement"].eq("planned_offered_not_intake").all():
        raise ValueError("feeding quantity provenance must remain planned offered, not intake")
    formal_ratio = design["formal_feed_kg_day"] / design["formal_weight_kg"]
    if not np.allclose(formal_ratio, 0.04, atol=1e-9):
        raise ValueError("formal feeding ratio is not consistently 4% BW")

    # Compatibility/audit aliases used in current result tables.
    design["weight_start"] = design["adapt_weight_kg"]
    design["weight_end"] = design["postfast_weight_kg"]
    design["weight_mid"] = (
        design[["adapt_weight_kg", "formal_weight_kg", "prefast_weight_kg", "postfast_weight_kg"]]
        .mean(axis=1)
    )
    return design.sort_values(["period", "chamber"]).reset_index(drop=True)


def load_weight_endpoints() -> pd.DataFrame:
    """Backward-compatible name; now returns the full experiment design."""
    return load_experiment_design()


def consolidate_clean_heart(chamber: str) -> tuple[pd.DataFrame, dict[str, object]]:
    source_dir = HEART_ROOT / f"{chamber}hdf" / "cleanlabels"
    files = sorted(source_dir.glob("*.csv"))
    if not files:
        raise FileNotFoundError(f"no cleaned heart CSVs in {source_dir}")
    frames = []
    bad_schema = []
    for path in files:
        frame = pd.read_csv(path, usecols=lambda c: c in {"timestamp", "label"})
        if "timestamp" not in frame.columns:
            bad_schema.append(path.name)
            continue
        if "label" not in frame.columns:
            frame["label"] = 1
        frames.append(frame[["timestamp", "label"]])
    if bad_schema:
        raise ValueError(f"bad heart schema for {chamber}: {bad_schema}")
    heart = pd.concat(frames, ignore_index=True)
    heart["timestamp"] = pd.to_datetime(heart["timestamp"], format="mixed", errors="coerce")
    n_nat = int(heart["timestamp"].isna().sum())
    heart = heart.dropna(subset=["timestamp"]).sort_values("timestamp")
    n_before_dedup = len(heart)
    heart = heart.drop_duplicates("timestamp", keep="first").reset_index(drop=True)
    summary = {
        "chamber": chamber,
        "source_files": len(files),
        "rows": len(heart),
        "nat_dropped": n_nat,
        "duplicates_dropped": n_before_dedup - len(heart),
        "dt_start": heart["timestamp"].min(),
        "dt_end": heart["timestamp"].max(),
        "labels": ";".join(map(str, sorted(heart["label"].dropna().unique()))),
    }
    atomic_csv(heart, HEART_OUT / f"{chamber}heart.csv")
    return heart, summary


def hrv_features(rr_ms: np.ndarray, beat_times: np.ndarray, win_s: float) -> dict[str, float]:
    valid = rr_ms[np.isfinite(rr_ms) & (rr_ms >= RR_MIN_MS) & (rr_ms <= RR_MAX_MS)]
    n_beats = int(len(beat_times))
    span = 0.0
    if n_beats > 1:
        span = float((beat_times[-1] - beat_times[0]) / np.timedelta64(1, "s"))
    base: dict[str, float] = {
        "beat_count": n_beats,
        "valid_rr_count": int(len(valid)),
        "heart_span_coverage": float(np.clip(span / win_s, 0.0, 1.0)),
        "hr_valid": int(n_beats >= MIN_BEATS and len(valid) >= 5),
    }
    if not base["hr_valid"]:
        return {**base, **{col: np.nan for col in HRV_COLS}}
    diff = np.diff(valid)
    hr = 60000.0 / valid
    return {
        **base,
        "HR_mean": float(hr.mean()),
        "HR_std": float(hr.std()),
        "RR_mean": float(valid.mean()),
        "SDNN": float(valid.std(ddof=1)),
        "RMSSD": float(np.sqrt(np.mean(diff**2))) if len(diff) else np.nan,
        "pNN50": float(np.mean(np.abs(diff) > 50)) if len(diff) else np.nan,
        "RR_cv": float(valid.std() / valid.mean()),
        "RR_skew": float(stats.skew(valid)),
        "RR_kurt": float(stats.kurtosis(valid)),
        "RR_range": float(valid.max() - valid.min()),
    }


def add_weight_and_groups(hp: pd.DataFrame, weights: pd.DataFrame) -> pd.DataFrame:
    """Attach true identity, phase-aware weight, and planned feeding metadata."""
    out = hp.merge(weights, on=["chamber", "period"], how="left", validate="many_to_one")
    if out["animal_id"].isna().any():
        bad = out.loc[out["animal_id"].isna(), ["chamber", "period"]].drop_duplicates()
        raise ValueError(f"HP rows without experiment design:\n{bad}")
    out["phase_idx"] = pd.to_numeric(out["phase_idx"], errors="raise").astype(int)
    if not out["phase_idx"].isin(range(5)).all():
        raise ValueError("phase_idx outside 0..4")

    out["weight"] = np.nan
    out["weight_fraction_phase"] = np.nan
    out["weight_anchor_start_kg"] = np.nan
    out["weight_anchor_end_kg"] = np.nan
    out["weight_segment"] = ""
    segments = (
        (out["phase_idx"].eq(0), "adapt_weight_kg", "formal_weight_kg", "adapt_to_formal"),
        (out["phase_idx"].isin([1, 2, 3]), "formal_weight_kg", "prefast_weight_kg", "formal_to_prefast"),
        (out["phase_idx"].eq(4), "prefast_weight_kg", "postfast_weight_kg", "fasting_to_postfast"),
    )
    for mask, start_col, end_col, label in segments:
        if not mask.any():
            continue
        segment = out.loc[mask]
        start_time = segment.groupby("experimental_unit")["datetime"].transform("min")
        end_time = segment.groupby("experimental_unit")["datetime"].transform("max")
        denominator = (end_time - start_time).dt.total_seconds().clip(lower=1)
        fraction = ((segment["datetime"] - start_time).dt.total_seconds() / denominator).clip(0, 1)
        start_weight = pd.to_numeric(segment[start_col], errors="raise")
        end_weight = pd.to_numeric(segment[end_col], errors="raise")
        out.loc[mask, "weight_fraction_phase"] = fraction
        out.loc[mask, "weight_anchor_start_kg"] = start_weight
        out.loc[mask, "weight_anchor_end_kg"] = end_weight
        out.loc[mask, "weight"] = start_weight + fraction * (end_weight - start_weight)
        out.loc[mask, "weight_segment"] = label
    if out["weight"].isna().any():
        raise ValueError("phase-aware weight assignment left missing values")
    out["weight_fraction"] = out["weight_fraction_phase"]

    fed_adapt = out["phase_idx"].eq(0)
    fed_formal = out["phase_idx"].isin([1, 2, 3])
    out["planned_feed_kg_day"] = 0.0
    out["planned_feed_g_meal"] = 0.0
    out.loc[fed_adapt, "planned_feed_kg_day"] = out.loc[fed_adapt, "adapt_feed_kg_day"]
    out.loc[fed_adapt, "planned_feed_g_meal"] = out.loc[fed_adapt, "adapt_feed_g_meal"]
    out.loc[fed_formal, "planned_feed_kg_day"] = out.loc[fed_formal, "formal_feed_kg_day"]
    out.loc[fed_formal, "planned_feed_g_meal"] = out.loc[fed_formal, "formal_feed_g_meal"]
    out["planned_feed_g_per_kg_bw"] = out["planned_feed_kg_day"] * 1000.0 / out["weight"]
    out["planned_feed_g_per_w075"] = out["planned_feed_kg_day"] * 1000.0 / out["weight"].pow(0.75)

    # Keep `pig` as a compatibility column, but make it the true ear-tag identity.
    out["legacy_pig"] = out["chamber"] + "_p" + out["period"].astype(str)
    if not out["legacy_pig"].eq(out["experimental_unit"]).all():
        raise ValueError("experimental_unit does not match chamber-period")
    out["pig"] = out["animal_id"].astype(str)
    shifted = out["datetime"] - pd.Timedelta(hours=9)
    out["experiment_date"] = shifted.dt.strftime("%Y-%m-%d")
    out["day_group"] = out["pig"] + "_" + out["experiment_date"]
    hod = out["datetime"].dt.hour + out["datetime"].dt.minute / 60.0
    out["hour_of_day"] = hod
    out["sin_hour"] = np.sin(2 * np.pi * hod / 24)
    out["cos_hour"] = np.cos(2 * np.pi * hod / 24)
    return out


def build_heart_features(hp: pd.DataFrame, heart_by_chamber: dict[str, pd.DataFrame]) -> pd.DataFrame:
    interval_col = next((c for c in hp.columns if "时间差" in str(c)), None)
    if interval_col is None:
        raise ValueError("HP table has no 时间差 column")
    records = []
    for chamber in CHAMBERS:
        sub = hp[hp["chamber"] == chamber].sort_values("datetime")
        heart = heart_by_chamber[chamber].sort_values("timestamp").reset_index(drop=True)
        times = heart["timestamp"].to_numpy(dtype="datetime64[ns]")
        rr = heart["timestamp"].diff().dt.total_seconds().to_numpy() * 1000.0
        for _, row in sub.iterrows():
            raw_win = pd.to_numeric(pd.Series([row[interval_col]]), errors="coerce").iloc[0]
            win_s = 300.0 if not np.isfinite(raw_win) else float(np.clip(raw_win, *WIN_CLAMP_S))
            end = np.datetime64(row["datetime"])
            start = end - np.timedelta64(int(round(win_s * 1000)), "ms")
            left = int(np.searchsorted(times, start, side="left"))
            right = int(np.searchsorted(times, end, side="left"))
            rec = row.to_dict()
            rec["win_s"] = win_s
            rec.update(hrv_features(rr[left:right], times[left:right], win_s))
            records.append(rec)
    return pd.DataFrame(records)


def aggregate_30min(frame: pd.DataFrame) -> pd.DataFrame:
    data = frame.copy()
    data["time_bin"] = data["datetime"].dt.floor("30min")
    numeric = data.select_dtypes(include=[np.number]).columns.tolist()
    # These are identifiers encoded as numbers; retain their exact first value.
    first_numeric = {"period", "phase_idx", "is_fasting", "diet_code", "source_row"}
    mean_cols = [c for c in numeric if c not in first_numeric]
    agg_spec: dict[str, object] = {c: "mean" for c in mean_cols}
    agg_spec.update({c: "first" for c in first_numeric if c in data.columns})
    for col in [
        "chamber", "pig", "animal_id", "ear_tag", "experimental_unit", "legacy_pig",
        "day_group", "experiment_date", "phase_label", "weight_segment",
        "feed_measurement",
    ]:
        if col in data.columns:
            agg_spec[col] = "first"
    out = data.groupby(["day_group", "time_bin"], as_index=False).agg(agg_spec)
    counts = data.groupby(["day_group", "time_bin"]).size().rename("source_windows").reset_index()
    valid = data.groupby(["day_group", "time_bin"])["hr_valid"].mean().rename("hr_valid_frac").reset_index()
    out = out.merge(counts, on=["day_group", "time_bin"]).merge(
        valid, on=["day_group", "time_bin"]
    )
    out["datetime"] = out["time_bin"] + pd.Timedelta(minutes=30)
    return out.sort_values(["pig", "datetime"]).reset_index(drop=True)


def main() -> None:
    HEART_OUT.mkdir(parents=True, exist_ok=True)
    MODEL_OUT.mkdir(parents=True, exist_ok=True)

    heart_by_chamber: dict[str, pd.DataFrame] = {}
    heart_summaries = []
    for chamber in CHAMBERS:
        print(f"consolidating {chamber} cleaned heart...", flush=True)
        heart, summary = consolidate_clean_heart(chamber)
        heart_by_chamber[chamber] = heart
        heart_summaries.append(summary)
        print(
            f"  {summary['rows']:,} peaks | {summary['dt_start']} .. {summary['dt_end']}",
            flush=True,
        )
    heart_summary = pd.DataFrame(heart_summaries)
    atomic_csv(heart_summary, HEART_OUT / "heart_merge_summary.csv")

    hp = pd.read_csv(HP_CSV)
    hp["datetime"] = pd.to_datetime(hp["datetime"], format="mixed", errors="coerce")
    hp["HP_kcal"] = pd.to_numeric(hp["HP_kcal"], errors="coerce")
    hp = hp[hp["datetime"].notna() & (hp["HP_kcal"] > 0)].copy()
    hp["period"] = pd.to_numeric(hp["period"], errors="raise").astype(int)
    design = load_experiment_design()
    atomic_csv(design, MODEL_OUT / "experiment_design_2026.csv")
    # Preserve the old filename for downstream readers, but its rows now contain
    # the full ear-tag-backed four-anchor design rather than chamber/date guesses.
    atomic_csv(design, MODEL_OUT / "weight_endpoints_2026.csv")
    hp = add_weight_and_groups(hp, design)

    print(f"building HRV features for {len(hp):,} positive-HP rows...", flush=True)
    five = build_heart_features(hp, heart_by_chamber)
    five["HP_per_W075"] = five["HP_kcal"] / five["weight"].pow(0.75)
    atomic_csv(five, MODEL_OUT / "2026_hr_features_5min.csv")

    thirty = aggregate_30min(five)
    thirty["HP_per_W075"] = thirty["HP_kcal"] / thirty["weight"].pow(0.75)
    atomic_csv(thirty, MODEL_OUT / "2026_hr_features_30min.csv")

    coverage = (
        five.groupby(["chamber", "period", "experimental_unit", "animal_id", "ear_tag"], as_index=False)
        .agg(
            rows=("HP_kcal", "size"),
            dt_start=("datetime", "min"),
            dt_end=("datetime", "max"),
            hr_valid_rows=("hr_valid", "sum"),
            hr_valid_rate=("hr_valid", "mean"),
            heart_span_coverage=("heart_span_coverage", "mean"),
            adapt_weight_kg=("adapt_weight_kg", "first"),
            formal_weight_kg=("formal_weight_kg", "first"),
            prefast_weight_kg=("prefast_weight_kg", "first"),
            postfast_weight_kg=("postfast_weight_kg", "first"),
            planned_feed_kg_day_mean=("planned_feed_kg_day", "mean"),
        )
    )
    atomic_csv(coverage, MODEL_OUT / "data_coverage_2026.csv")
    summary = {
        "hp_positive_rows": int(len(five)),
        "rows_30min": int(len(thirty)),
        "true_pigs": int(five["animal_id"].nunique()),
        "experimental_units": int(five["experimental_unit"].nunique()),
        "periods": sorted(map(int, five["period"].unique())),
        "hr_valid_rows": int(five["hr_valid"].sum()),
        "hr_valid_true_pigs": int(five.loc[five["hr_valid"] == 1, "animal_id"].nunique()),
        "identity": "animal_id from ear tag; experimental_unit=chamber+period",
        "weight": "piecewise four-anchor interpolation from commercial feeding design",
        "feeding": "planned offered quantity; exact meal times not observed",
        "heart_last_timestamp": {
            row["chamber"]: str(row["dt_end"]) for row in heart_summaries
        },
        "day_boundary": "09:00",
        "window": "recorded 时间差 clamped to 150-420 s",
    }
    with open(MODEL_OUT / "prepare_summary.json", "w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    print("DONE_EXIT_0", flush=True)


if __name__ == "__main__":
    main()
