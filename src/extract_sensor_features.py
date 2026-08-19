"""
extract_features_2026.py -- feature extraction for the NEW 2026 growth-pig cohort
(with accelerometer). Reuses the verified 52-metric motion aggregation and HRV logic.

Inputs:
  HP   : data/derived/hp/2026_HP_all.csv  (chamber,period,phase_idx,
         phase_label,is_fasting,datetime,HP_kcal,...)   [our extracted HP]
  move : old base files plus the July continuation files.  The continuation
         names are A1_2.csv, B1_3.csv (suffix typo in the source drop), and
         B2_2.csv.  They are joined by timestamp, never inferred by suffix rank.
  heart: data/derived/heart/<chamber>heart.csv  (timestamp,label)
  design: normalized ear-tag/weight/feed master from 商业试验饲喂设计.xlsx

True pig identity = ear-tag-backed animal_id.  (chamber, period) is retained as
experimental_unit because several animals return in later periods under another
chamber/diet. All are growing pigs -> is_big=0. cohort tag '2026-06'.

Output: 2026_features_newmove.csv  (Sept-schema-aligned: motion 219 + HRV 12 +
context, target HP_kcal).
"""
import os
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats
from prepare_ecg import add_weight_and_groups, default_share_root, load_weight_endpoints

# ---- config (same as chapter6newmove) ----
HEART_MIN_BEATS = 150
RR_DUPLICATE_MS = 50
RR_GAP_MS = 3000
WINDOW_S = 300
HP_MIN = 0

CHAMBERS = ["A1", "B1", "B2"]
COHORT = "2026-06"

# All paths are configurable. Defaults follow the public data contract.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SHARE_ROOT = Path(os.environ.get("SHARE_ROOT", PROJECT_ROOT / "data"))
HP_CSV = os.environ.get(
    "HP_CSV", str(SHARE_ROOT / "2026" / "HP_extracted" / "2026_HP_all.csv")
)
MOVE_DIR = os.environ.get("MOVE_DIR", str(PROJECT_ROOT / "data" / "raw" / "imu"))
HEART_DIR = os.environ.get("HEART_DIR", str(SHARE_ROOT / "2026" / "heart_extracted"))
OUT_CSV = os.environ.get(
    "OUT_CSV", str(SHARE_ROOT / "2026" / "model_ready" / "2026_features_newmove_expanded.csv")
)

DEFAULT_MOVE_FILES = {
    "A1": ("A1.csv", "A1_2.csv"),
    "B1": ("B1.csv", "B1_3.csv"),
    "B2": ("B2.csv", "B2_2.csv"),
}


def configured_move_files(chamber):
    """Return the explicit source manifest for one chamber."""
    key = f"MOVE_FILES_{chamber}"
    raw = os.environ.get(key)
    if raw:
        names = tuple(part.strip() for part in raw.split(";") if part.strip())
        if not names:
            raise ValueError(f"{key} did not contain any filenames")
        return names
    return DEFAULT_MOVE_FILES[chamber]


def load_move(chamber, move_dir=None, file_names=None):
    """Load, schema-check, concatenate, and timestamp-deduplicate motion drops."""
    root = Path(MOVE_DIR if move_dir is None else move_dir) / f"{chamber}move"
    names = tuple(file_names or configured_move_files(chamber))
    paths = [root / name for name in names]
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError(f"missing motion files for {chamber}: {missing}")

    frames = []
    reference_columns = None
    for path in paths:
        frame = pd.read_csv(path, low_memory=False)
        columns = list(frame.columns)
        if reference_columns is None:
            reference_columns = columns
        elif columns != reference_columns:
            raise ValueError(f"motion schema mismatch for {chamber}: {path.name}")
        frame["move_source_file"] = path.name
        frames.append(frame)

    move = pd.concat(frames, ignore_index=True)
    move["timestamp"] = pd.to_datetime(move["timestamp"], format="mixed", errors="coerce")
    n_nat = int(move["timestamp"].isna().sum())
    move = move.dropna(subset=["timestamp"]).sort_values("timestamp")
    before_dedup = len(move)
    move = move.drop_duplicates("timestamp", keep="first").reset_index(drop=True)
    print(
        f"  motion manifest: {', '.join(names)} | NaT={n_nat} | "
        f"duplicates={before_dedup - len(move)} | "
        f"{move['timestamp'].min()} .. {move['timestamp'].max()}",
        flush=True,
    )
    return move


def clean_heart(df_heart):
    # Inputs are already the manually/QC-cleaned R-peak files.  Keep this pass
    # deterministic: deduplicate exact timestamps and invalidate implausible RR values.
    df = (df_heart.sort_values("timestamp")
          .drop_duplicates("timestamp", keep="first").reset_index(drop=True))
    df["RR_ms"] = df["timestamp"].diff().dt.total_seconds() * 1000
    df.loc[(df["RR_ms"] < RR_DUPLICATE_MS) | (df["RR_ms"] > RR_GAP_MS), "RR_ms"] = np.nan
    return df


def extract_hrv(rr_ms, n_beats):
    base = {"beat_count": n_beats, "hr_valid": 0}
    if n_beats < HEART_MIN_BEATS or len(rr_ms) < 5:
        return {**base, **{k: np.nan for k in [
            "HR_mean", "HR_std", "RR_mean", "SDNN", "RMSSD",
            "pNN50", "RR_cv", "RR_skew", "RR_kurt", "RR_range"]}}
    diff = np.diff(rr_ms)
    hr = 60000.0 / rr_ms
    return {
        "beat_count": n_beats, "hr_valid": 1,
        "HR_mean": hr.mean(), "HR_std": hr.std(),
        "RR_mean": rr_ms.mean(), "SDNN": rr_ms.std(ddof=1),
        "RMSSD": np.sqrt(np.mean(diff ** 2)),
        "pNN50": float(np.mean(np.abs(diff) > 50)),
        "RR_cv": rr_ms.std() / rr_ms.mean(),
        "RR_skew": float(stats.skew(rr_ms)),
        "RR_kurt": float(stats.kurtosis(rr_ms)),
        "RR_range": rr_ms.max() - rr_ms.min(),
    }


MOVE_STAT_COLS = [
    "Acc_Mean", "Acc_Std", "Acc_Sum", "Acc_Max", "Acc_Min", "Acc_Range",
    "Acc_P25", "Acc_P75", "Gyro_Mean", "Gyro_Max",
    "DynX_Mean", "DynX_Std", "DynX_Mean_abs", "DynY_Mean", "DynY_Std",
    "DynY_Mean_abs", "DynZ_Mean", "DynZ_Std", "DynZ_Mean_abs",
    "AccMag_Mean", "AccMag_Std", "Roll_Mean", "Roll_Std", "Pitch_Mean",
    "Pitch_Std", "Tilt_Mag_Mean", "Tilt_Mag_Std", "AngX_Std", "AngY_Std",
    "AngZ_Std", "AngX_Rate", "AngY_Rate", "AngZ_Rate", "Jerk_Mean",
    "ODBA", "VeDBA_RMS", "Axis_Dominance", "Skewness", "Kurtosis",
    "Autocorr_Lag1", "Peak_Count", "Corr_XY", "Corr_XZ", "Corr_YZ",
    "Spectral_Entropy", "Zero_Crossing_Rate", "Dominant_Frequency",
    "Coefficient_of_Variation", "Power_Low", "Power_Loco", "Power_High",
    "Sample_Entropy",
]
MOVE_FRAC_COLS = ["Frac_Rest", "Frac_Moderate", "Frac_Vigorous",
                  "Frac_Lying", "Frac_HeadDown", "Active_Fraction"]


def aggregate_move(df_win):
    n = len(df_win)
    feats = {"move_coverage": n / WINDOW_S}
    if n == 0:
        return feats
    for col in MOVE_STAT_COLS:
        if col not in df_win.columns:
            continue
        v = df_win[col].dropna().values
        if len(v) == 0:
            feats[f"{col}_mean"] = feats[f"{col}_std"] = np.nan
            feats[f"{col}_p25"] = feats[f"{col}_p75"] = np.nan
        else:
            feats[f"{col}_mean"] = v.mean()
            feats[f"{col}_std"] = v.std()
            feats[f"{col}_p25"] = np.percentile(v, 25)
            feats[f"{col}_p75"] = np.percentile(v, 75)
    for col in MOVE_FRAC_COLS:
        if col in df_win.columns:
            feats[col] = df_win[col].dropna().mean()
    if "Dominant_Frequency" in df_win.columns:
        v = df_win["Dominant_Frequency"].dropna().values
        feats["StepFreq_std"] = v.std() if len(v) > 1 else np.nan
    if "Frac_Lying" in df_win.columns:
        lying = (df_win["Frac_Lying"] > 0.5).astype(int)
        feats["LyingTransitions"] = lying.diff().abs().sum()
    if "Frac_HeadDown" in df_win.columns:
        hd = (df_win["Frac_HeadDown"] > 0.5).astype(int)
        feats["HeadDownTransitions"] = hd.diff().abs().sum()
    if "Data_Count" in df_win.columns:
        feats["data_count_mean"] = df_win["Data_Count"].mean()
    return feats


def add_time_features(dt):
    hod = dt.hour + dt.minute / 60
    return {"hour_of_day": hod,
            "sin_hour": np.sin(2 * np.pi * hod / 24),
            "cos_hour": np.cos(2 * np.pi * hod / 24)}


def build_chamber(chamber, hp_all):
    print(f"\n=== {chamber} ===", flush=True)
    df_heat = hp_all[hp_all["chamber"] == chamber].copy()
    df_heat = df_heat[df_heat["HP_kcal"] > HP_MIN].sort_values("datetime").reset_index(drop=True)
    print(f"  HP rows (HP>{HP_MIN}): {len(df_heat)}", flush=True)
    interval_col = next((c for c in df_heat.columns if "时间差" in str(c)), None)
    if interval_col is None:
        raise ValueError("HP table has no 时间差 column")

    heart_path = os.path.join(HEART_DIR, f"{chamber}heart.csv")
    df_heart = pd.read_csv(heart_path, parse_dates=["timestamp"])
    df_heart_clean = clean_heart(df_heart).sort_values("timestamp").reset_index(drop=True)
    heart_times = df_heart_clean["timestamp"].values
    print(f"  heart R-peaks: {len(df_heart)} -> clean {len(df_heart_clean)}", flush=True)

    df_move = load_move(chamber)
    move_times = df_move["timestamp"].values   # datetime64[ns]
    print(f"  move rows (per-sec): {len(df_move)}", flush=True)

    records = []
    total = len(df_heat)
    for i, row in df_heat.iterrows():
        if i % 400 == 0:
            print(f"  {i}/{total}", flush=True)
        t_end = row["datetime"]
        raw_win = pd.to_numeric(pd.Series([row[interval_col]]), errors="coerce").iloc[0]
        win_s = 300.0 if not np.isfinite(raw_win) else float(np.clip(raw_win, 150.0, 420.0))
        ts = np.datetime64(t_end - pd.Timedelta(seconds=win_s))
        te = np.datetime64(t_end)
        wm = df_move.iloc[np.searchsorted(move_times, ts):np.searchsorted(move_times, te)]
        wh = df_heart_clean.iloc[np.searchsorted(heart_times, ts):np.searchsorted(heart_times, te)]
        move_feats = aggregate_move(wm)
        move_feats["move_coverage"] = len(wm) / win_s
        hrv_feats = extract_hrv(wh["RR_ms"].dropna().values, len(wh))
        period = int(row["period"])
        rec = {"datetime": t_end, "chamber": chamber, "cohort": COHORT,
               "period": period, "phase_idx": int(row["phase_idx"]),
               "phase_label": row["phase_label"],
               "is_fasting": int(row["is_fasting"]),
               "day_in_phase": int(row["phase_idx"]),
               "weight": row["weight"], "weight_start": row["weight_start"],
               "weight_end": row["weight_end"],
               "adapt_weight_kg": row["adapt_weight_kg"],
               "formal_weight_kg": row["formal_weight_kg"],
               "prefast_weight_kg": row["prefast_weight_kg"],
               "postfast_weight_kg": row["postfast_weight_kg"],
               "weight_fraction_phase": row["weight_fraction_phase"],
               "weight_anchor_start_kg": row["weight_anchor_start_kg"],
               "weight_anchor_end_kg": row["weight_anchor_end_kg"],
               "weight_segment": row["weight_segment"],
               "is_big": 0,
               "pig": row["pig"], "animal_id": row["animal_id"],
               "ear_tag": row["ear_tag"],
               "experimental_unit": row["experimental_unit"],
               "legacy_pig": row["legacy_pig"],
               "diet_code": row["diet_code"],
               "planned_feed_kg_day": row["planned_feed_kg_day"],
               "planned_feed_g_meal": row["planned_feed_g_meal"],
               "planned_feed_g_per_kg_bw": row["planned_feed_g_per_kg_bw"],
               "planned_feed_g_per_w075": row["planned_feed_g_per_w075"],
               "feed_measurement": row["feed_measurement"],
               "day_group": row["day_group"],
               "experiment_date": row["experiment_date"], "win_s": win_s,
               "HP_kcal": row["HP_kcal"]}
        for env_col in ("小室温度(℃)", "小室湿度(%)"):
            if env_col in row.index:
                rec[env_col] = row[env_col]
        rec.update(add_time_features(t_end))
        rec.update(move_feats)
        rec.update(hrv_feats)
        records.append(rec)
    df = pd.DataFrame(records)
    print(f"  windows {len(df)} | move>=0.5: {(df['move_coverage']>=0.5).sum()} "
          f"| hr_valid: {(df['hr_valid']==1).sum()}", flush=True)
    return df


def main():
    print(f"HP_CSV={HP_CSV}\nMOVE_DIR={MOVE_DIR}\nOUT_CSV={OUT_CSV}", flush=True)
    hp_all = pd.read_csv(HP_CSV)
    hp_all["datetime"] = pd.to_datetime(hp_all["datetime"], format="mixed", errors="coerce")
    n_nat = hp_all["datetime"].isna().sum()
    print(f"HP rows {len(hp_all)} | datetime NaT dropped {n_nat}", flush=True)
    hp_all = hp_all[hp_all["datetime"].notna()].reset_index(drop=True)
    hp_all["period"] = pd.to_numeric(hp_all["period"], errors="raise").astype(int)
    hp_all = add_weight_and_groups(hp_all, load_weight_endpoints())
    frames = [build_chamber(ch, hp_all) for ch in CHAMBERS]
    allf = pd.concat(frames, ignore_index=True)
    allf.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")
    local = os.environ.get("LOCAL_FEATURE_COPY")
    if local:
        try:
            allf.to_csv(local, index=False, encoding="utf-8-sig")
        except Exception as e:
            print("local copy skipped:", e, flush=True)
    print(f"\n[DONE] {allf.shape} -> {OUT_CSV}", flush=True)
    print("per chamber/period:\n", allf.groupby(["chamber", "period"]).size().to_string(), flush=True)
    print("move>=0.5 & hr_valid:", ((allf["move_coverage"] >= 0.5) & (allf["hr_valid"] == 1)).sum(), flush=True)
    print("HP_kcal mean %.2f max %.2f" % (allf["HP_kcal"].mean(), allf["HP_kcal"].max()), flush=True)


if __name__ == "__main__":
    main()
