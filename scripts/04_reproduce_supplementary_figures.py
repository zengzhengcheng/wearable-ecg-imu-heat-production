"""Verify deposited supplementary assets and reproduce Figures S1--S3.

This script uses only compact repository data. Detected R peaks in Figure S1
are software output, not manually adjudicated annotations or a gold standard.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
ANALYSIS_DIR = ROOT / "data" / "analysis"
SUPPLEMENT_DIR = ROOT / "reference_results" / "supplementary"
PANEL_PATH = ANALYSIS_DIR / "modeling_panel_30min.csv"
ECG_PATH = SUPPLEMENT_DIR / "signal_quality_ecg.csv"
IMU_PATH = SUPPLEMENT_DIR / "signal_quality_imu.csv"
PEAKS_PATH = SUPPLEMENT_DIR / "signal_quality_rpeaks.csv"
METADATA_PATH = SUPPLEMENT_DIR / "signal_quality_metadata.csv"
SCREENING_PATH = SUPPLEMENT_DIR / "screening_counts.csv"

COLORS = {
    "navy": "#315A7D",
    "blue": "#6E9EC2",
    "green": "#4F9274",
    "orange": "#D19A55",
    "red": "#B85C5C",
    "gray": "#AAB3BB",
    "light": "#EDF1F4",
    "dark": "#263746",
}


def set_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "Arial",
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.dpi": 120,
            "savefig.dpi": 400,
            "savefig.transparent": False,
        }
    )


def save_figure(fig: plt.Figure, output_dir: Path, stem: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "svg", "pdf"):
        fig.savefig(output_dir / f"{stem}.{suffix}", bbox_inches="tight")
    plt.close(fig)


def nearest_peak_match(reference_ms: np.ndarray, comparison_ms: np.ndarray, tolerance_ms: float = 50.0) -> int:
    comparison_ms = np.sort(np.asarray(comparison_ms, dtype=float))
    matched = 0
    for value in np.asarray(reference_ms, dtype=float):
        index = int(np.searchsorted(comparison_ms, value))
        distances = []
        if index < len(comparison_ms):
            distances.append(abs(comparison_ms[index] - value))
        if index > 0:
            distances.append(abs(comparison_ms[index - 1] - value))
        matched += bool(distances and min(distances) <= tolerance_ms)
    return matched


def figure_s1(output_dir: Path) -> None:
    ecg = pd.read_csv(ECG_PATH)
    imu = pd.read_csv(IMU_PATH)
    peaks = pd.read_csv(PEAKS_PATH)
    metadata = pd.read_csv(METADATA_PATH).iloc[0]

    duration = float(metadata["dur_s"])
    detected = peaks.loc[peaks["source"] == "cleanlabels", "t_rel_s"].to_numpy(float)
    comparison = peaks.loc[peaks["source"] == "pt_refined", "t_rel_s"].to_numpy(float)
    if not (19.9 <= float(ecg["t_rel_s"].max()) <= 20.1 and duration == 20.0):
        raise AssertionError("Figure S1 source is not the expected 20-s segment")
    if len(detected) != 35 or len(comparison) != 35:
        raise AssertionError("Figure S1 peak counts differ from the deposited segment")
    matched = nearest_peak_match(detected * 1000.0, comparison * 1000.0)
    union = len(detected) + len(comparison) - matched
    bsqi = matched / union if union else np.nan
    if not np.isclose(bsqi, 1.0):
        raise AssertionError(f"unexpected cross-detector bSQI: {bsqi}")

    fig, (ax_ecg, ax_imu) = plt.subplots(
        2,
        1,
        figsize=(7.2, 4.8),
        sharex=True,
        gridspec_kw={"height_ratios": [2.4, 1.6], "hspace": 0.28},
    )
    ax_ecg.plot(ecg["t_rel_s"], ecg["ecg"], color=COLORS["dark"], lw=0.7, label="Raw ECG (512 Hz)")
    peak_y = np.interp(detected, ecg["t_rel_s"], ecg["ecg"])
    offset = 0.09 * float(ecg["ecg"].max() - ecg["ecg"].min())
    ax_ecg.plot(
        detected,
        peak_y + offset,
        "v",
        color=COLORS["blue"],
        ms=4.5,
        ls="none",
        label=f"Detected R peaks (n={len(detected)})",
    )
    ax_ecg.set_ylabel("ECG (a.u.)")
    ax_ecg.set_title(
        f"a  ECG and detected R peaks — {metadata['pig']}, chamber {metadata['chamber']}, {metadata['date']}"
        f"  (cross-detector bSQI at 50 ms = {bsqi:.3f})",
        loc="left",
    )
    ax_ecg.legend(loc="upper right", frameon=False)
    ax_ecg.grid(axis="x", alpha=0.18, lw=0.5)

    for column, color, linestyle, label in (
        ("AccX", COLORS["navy"], "-", "Acc X"),
        ("AccY", COLORS["green"], "--", "Acc Y"),
        ("AccZ", COLORS["orange"], ":", "Acc Z"),
    ):
        ax_imu.plot(imu["t_rel_s"], imu[column], color=color, ls=linestyle, lw=1.0, label=label)
    ax_imu.set_xlim(0, duration)
    ax_imu.set_xlabel("Time within segment (s)")
    ax_imu.set_ylabel("Acceleration (g)")
    ax_imu.set_title("b  Simultaneous 10 Hz tri-axial acceleration", loc="left")
    ax_imu.legend(loc="upper right", frameon=False, ncol=3)
    ax_imu.grid(axis="x", alpha=0.18, lw=0.5)
    fig.suptitle(
        f"Representative 20-s signal segment (rest; ODBA={float(metadata['ODBA_mean']):.3f} g; "
        f"HR={float(metadata['HR_mean']):.0f} bpm)",
        y=0.99,
        fontsize=10.5,
    )
    fig.subplots_adjust(top=0.86, bottom=0.11, left=0.10, right=0.98)
    save_figure(fig, output_dir, "figS1_signal_quality")


def figure_s2(panel: pd.DataFrame, output_dir: Path) -> None:
    required = {"experimental_unit", "pig", "chamber", "period", "strict_day_group"}
    if not required.issubset(panel.columns):
        raise AssertionError(f"panel lacks coverage fields: {sorted(required - set(panel.columns))}")
    coverage = (
        panel.groupby(["experimental_unit", "pig", "chamber", "period", "strict_day_group"], dropna=False)
        .size()
        .reset_index(name="n_windows")
    )
    units = (
        coverage.groupby(["experimental_unit", "pig", "chamber", "period"], dropna=False)
        .agg(days=("strict_day_group", "nunique"), windows=("n_windows", "sum"), first_day=("strict_day_group", "min"))
        .reset_index()
        .sort_values(["first_day", "chamber", "pig", "period"])
        .reset_index(drop=True)
    )
    days = sorted(coverage["strict_day_group"].astype(str).unique())
    chambers = sorted(units["chamber"].astype(str).unique())
    cumulative_monitoring_days = int(len(coverage))
    if (len(units), len(days), panel["pig"].nunique(), chambers) != (24, 40, 12, ["A1", "B1", "B2"]):
        raise AssertionError("Figure S2 coverage scope differs from 24 experimental units / 40 dates / 12 pigs / A1-B1-B2")
    if cumulative_monitoring_days != 118:
        raise AssertionError(f"Figure S2 expected 118 cumulative monitoring days, got {cumulative_monitoring_days}")

    day_index = {day: index for index, day in enumerate(days)}
    styles = {
        "A1": (COLORS["navy"], "s"),
        "B1": (COLORS["green"], "o"),
        "B2": (COLORS["orange"], "^"),
    }
    fig, ax = plt.subplots(figsize=(9.4, 5.8))
    for y, row in units.iterrows():
        observed_days = coverage.loc[coverage["experimental_unit"] == row["experimental_unit"], "strict_day_group"].astype(str)
        color, marker = styles[str(row["chamber"])]
        ax.scatter(
            [day_index[day] for day in observed_days],
            np.repeat(y, len(observed_days)),
            s=34,
            marker=marker,
            color=color,
            edgecolor="white",
            linewidth=0.45,
            zorder=3,
        )
    ax.set_yticks(np.arange(len(units)))
    ax.set_yticklabels(
        [f"{row.pig}  ({row.chamber}, P{row.period}, {row.days} d)" for row in units.itertuples()],
        fontsize=7.2,
    )
    tick_indices = sorted(set(np.linspace(0, len(days) - 1, 9, dtype=int)))
    ax.set_xticks(tick_indices)
    ax.set_xticklabels([days[index].replace("2026::", "") for index in tick_indices], rotation=40, ha="right")
    handles = [
        plt.Line2D([], [], ls="", marker=styles[chamber][1], color=styles[chamber][0], label=f"Chamber {chamber}")
        for chamber in chambers
    ]
    ax.legend(handles=handles, loc="lower left", bbox_to_anchor=(0.0, 1.005), frameon=False, ncol=3, borderaxespad=0)
    ax.set_xlabel("Experimental date (experimental day: 09:00 to 09:00 the following day)")
    ax.set_title(
        "Monitoring coverage across 24 pig-by-period experimental units "
        f"({cumulative_monitoring_days} cumulative monitoring days)",
        loc="left",
        pad=34,
    )
    ax.set_ylim(len(units) - 0.3, -0.8)
    ax.grid(axis="x", alpha=0.2, lw=0.5)
    fig.subplots_adjust(left=0.22, bottom=0.17, right=0.98, top=0.85)
    save_figure(fig, output_dir, "figS2_coverage_calendar")


def figure_s3(output_dir: Path) -> None:
    counts = pd.read_csv(SCREENING_PATH).sort_values("stage_order")
    expected = [34109, 34026, 33993, 30000, 4380, 4079]
    if counts["retained_count"].astype(int).tolist() != expected:
        raise AssertionError("Figure S3 screening counts differ from locked values")

    fig, ax = plt.subplots(figsize=(7.4, 7.2))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    y_positions = np.linspace(0.90, 0.10, len(counts))
    stage_colors = [COLORS["navy"], COLORS["blue"], COLORS["blue"], COLORS["green"], COLORS["orange"], COLORS["navy"]]
    stage_labels = [
        "Raw 5-min heat-production records",
        "HP > 0",
        "Deduplicated pig × timestamp",
        "Source-level quality control",
        "30-min windows with ≥3 source records",
        "Final windows with valid-HR fraction ≥50%",
    ]
    removal_labels = [
        "83 non-positive records excluded",
        "33 duplicate records excluded",
        "3,993 records excluded by source-level QC",
        "138 candidate bins / 210 source records excluded",
        "301 windows excluded (218 with no valid HR)",
    ]

    for index, (y, count, label, color) in enumerate(zip(y_positions, expected, stage_labels, stage_colors)):
        box = FancyBboxPatch(
            (0.16, y - 0.048),
            0.68,
            0.096,
            boxstyle="round,pad=0.012,rounding_size=0.012",
            linewidth=0.8,
            edgecolor=color,
            facecolor=color if index in (0, 5) else "white",
        )
        ax.add_patch(box)
        text_color = "white" if index in (0, 5) else COLORS["dark"]
        ax.text(0.50, y + 0.013, label, ha="center", va="center", color=text_color, fontsize=9.2, weight="bold")
        unit = "records" if index < 4 else "windows"
        ax.text(0.50, y - 0.021, f"n = {count:,} {unit}", ha="center", va="center", color=text_color, fontsize=9)
        if index < len(expected) - 1:
            next_y = y_positions[index + 1]
            arrow = FancyArrowPatch((0.50, y - 0.052), (0.50, next_y + 0.052), arrowstyle="-|>", mutation_scale=11, lw=0.8, color=COLORS["gray"])
            ax.add_patch(arrow)
            ax.text(0.855, (y + next_y) / 2, removal_labels[index], ha="left", va="center", fontsize=7.7, color=COLORS["dark"])
    ax.set_title("Data-screening workflow for the formal 30-min modeling panel", loc="left", pad=8, fontsize=11)
    fig.subplots_adjust(left=0.03, right=0.76, top=0.93, bottom=0.03)
    save_figure(fig, output_dir, "figS3_screening_flow")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results" / "reproduced")
    parser.add_argument("--no-figures", action="store_true", help="verify all supplementary assets without writing figures")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_style()
    panel = pd.read_csv(PANEL_PATH, low_memory=False)
    if panel.shape != (4079, 345):
        raise AssertionError(f"expected 4079 x 345 panel, got {panel.shape}")

    # Always verify all assets. The no-figures branch is useful for lightweight CI.
    ecg = pd.read_csv(ECG_PATH)
    imu = pd.read_csv(IMU_PATH)
    peaks = pd.read_csv(PEAKS_PATH)
    metadata = pd.read_csv(METADATA_PATH)
    screening = pd.read_csv(SCREENING_PATH)
    if (len(ecg), len(imu), len(peaks), len(metadata), len(screening)) != (10241, 200, 70, 1, 6):
        raise AssertionError("supplementary source-asset dimensions differ from the deposited release")
    coverage_scope = (
        panel["experimental_unit"].nunique(),
        panel["strict_day_group"].nunique(),
        panel["pig"].nunique(),
    )
    if coverage_scope != (24, 40, 12):
        raise AssertionError(f"unexpected coverage scope: {coverage_scope}")
    if screening.sort_values("stage_order")["retained_count"].astype(int).tolist() != [34109, 34026, 33993, 30000, 4380, 4079]:
        raise AssertionError("screening-count sequence mismatch")

    if not args.no_figures:
        figure_s1(args.output_dir)
        figure_s2(panel, args.output_dir)
        figure_s3(args.output_dir)
    print("SUPPLEMENTARY_REPRODUCIBILITY_PASS figures=S1,S2,S3")


if __name__ == "__main__":
    main()
