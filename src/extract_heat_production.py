"""
Extract heat-production (HP) data from chamber workbooks arranged according to
the public input-data contract.

Each file is one chamber x period.  Phase is inferred from the sheet name first
(适应 / 正式1-3 / 绝食), with position used only as a five-sheet fallback.  This
prevents a four-sheet workbook such as A1 period 8 from relabeling ``A绝食`` as
``正式3`` merely because the formal-day-3 sheet is absent.
Each sheet: row0 = 28-col header (datetime ... HP产热(kcal)); data rows have a
datetime in col0; the trailing statistics block has an EMPTY col0 -> dropped by
keeping only rows whose datetime parses.

Output (tidy, combined by period):
  <out>/2026_HP_all.csv          all chambers x periods x phases
  <out>/{chamber}_产热.csv       per chamber (both periods)
Adds metadata cols: chamber, period, phase_idx(0-4), phase_label, is_fasting.
HP产热(kcal) renamed HP_kcal. All raw measurement columns preserved.
"""
import os
import re
import glob
from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ROOT = os.environ.get("HP_WORKBOOK_ROOT", str(PROJECT_ROOT / "data" / "raw" / "calorimetry"))
OUT_DIRS = [os.environ.get("HP_OUTPUT_DIR", str(PROJECT_ROOT / "data" / "derived" / "hp"))]
for d in OUT_DIRS:
    os.makedirs(d, exist_ok=True)

PHASE_LABELS = ["适应", "正式1", "正式2", "正式3", "绝食"]   # by sheet position 0..4


def infer_phase(sheet_name, position, sheet_count):
    name = re.sub(r"\s+", "", str(sheet_name))
    if "适应" in name:
        return 0
    if "绝食" in name or "禁食" in name:
        return 4
    if "正式" in name:
        if re.search(r"(?:正式|第)[一1]", name):
            return 1
        if re.search(r"(?:正式|第)[二2]", name):
            return 2
        if re.search(r"(?:正式|第)[三3]", name):
            return 3
        if position in (1, 2, 3):
            return position
    if sheet_count == 5 and position < 5:
        return position
    raise ValueError(f"cannot infer phase from sheet={sheet_name!r} at position {position}")

frames = []
summary = []
for ch in ["A1", "B1", "B2"]:
    for f in sorted(glob.glob(os.path.join(ROOT, ch, "*.xlsx"))):
        fname = os.path.basename(f)
        m = re.match(r"\s*(\d)期", fname)
        period = int(m.group(1)) if m else None
        xl = pd.ExcelFile(f)
        if len(xl.sheet_names) != 5:
            print(f"  !! {ch}/{fname}: expected 5 sheets, got {len(xl.sheet_names)} -> {xl.sheet_names}")
        seen_phases = set()
        for idx, sheet in enumerate(xl.sheet_names):
            phase_idx = infer_phase(sheet, idx, len(xl.sheet_names))
            if phase_idx in seen_phases:
                raise ValueError(f"duplicate phase {phase_idx} in {ch}/{fname}: {xl.sheet_names}")
            seen_phases.add(phase_idx)
            df = pd.read_excel(f, sheet_name=sheet, header=0)
            df = df.rename(columns={c: "HP_kcal" for c in df.columns
                                    if isinstance(c, str) and "HP产热" in c})
            # datetime: most cells are real timestamps; a few are Excel serial
            # numbers (~46182) that pd.to_datetime would misread as 1970 epoch-ns.
            raw_dt = df["datetime"]
            parsed = pd.to_datetime(raw_dt, errors="coerce")
            num = pd.to_numeric(raw_dt, errors="coerce")
            serial_mask = num.between(20000, 60000)   # Excel serial date 1954..2064
            n_fixed = int(serial_mask.sum())
            if n_fixed:   # convert ONLY the few serial cells (avoid full-col overflow)
                parsed.loc[serial_mask] = pd.to_datetime(
                    num[serial_mask], unit="D", origin="1899-12-30")
            df["datetime"] = parsed
            # keep only real data rows: datetime must parse (drops stats block)
            n_raw = len(df)
            df = df[df["datetime"].notna()].copy()
            n_kept = len(df)
            df.insert(0, "chamber", ch)
            df.insert(1, "period", period)
            df.insert(2, "phase_idx", phase_idx)
            df.insert(3, "phase_label", PHASE_LABELS[phase_idx])
            df.insert(4, "is_fasting", int(phase_idx == 4))
            frames.append(df)
            summary.append({
                "chamber": ch, "period": period, "sheet": sheet,
                "phase": PHASE_LABELS[phase_idx], "rows_raw": n_raw, "rows_kept": n_kept,
                "stats_dropped": n_raw - n_kept, "dt_serial_fixed": n_fixed,
                "dt_start": df["datetime"].min(), "dt_end": df["datetime"].max(),
                "HP_mean": round(df["HP_kcal"].mean(), 3),
                "HP_min": round(df["HP_kcal"].min(), 3),
                "HP_max": round(df["HP_kcal"].max(), 3),
                "HP_le0": int((df["HP_kcal"] <= 0).sum()),
            })

alldf = pd.concat(frames, ignore_index=True)
alldf = alldf.sort_values(["chamber", "period", "phase_idx", "datetime"]).reset_index(drop=True)

for d in OUT_DIRS:
    alldf.to_csv(os.path.join(d, "2026_HP_all.csv"), index=False, encoding="utf-8-sig")
    for ch, g in alldf.groupby("chamber"):
        g.to_csv(os.path.join(d, f"{ch}_产热.csv"), index=False, encoding="utf-8-sig")

sm = pd.DataFrame(summary)
for d in OUT_DIRS:
    sm.to_csv(os.path.join(d, "2026_HP_summary.csv"), index=False, encoding="utf-8-sig")

print("=== per-sheet summary ===")
with pd.option_context("display.max_columns", None, "display.width", 220):
    print(sm.to_string(index=False))
print(f"\nTOTAL kept rows: {len(alldf)}")
print("per chamber/period:")
print(alldf.groupby(["chamber", "period"]).size().to_string())
print("HP_kcal overall: mean %.3f  min %.3f  max %.3f  <=0: %d"
      % (alldf["HP_kcal"].mean(), alldf["HP_kcal"].min(), alldf["HP_kcal"].max(),
         (alldf["HP_kcal"] <= 0).sum()))
print("\nwrote 2026_HP_all.csv + {chamber}_产热.csv + 2026_HP_summary.csv to:")
for d in OUT_DIRS:
    print("  ", d)
