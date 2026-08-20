"""Verify deposited supplementary assets and reproduce Figures S1--S3.

This script uses only compact repository data. Detected R peaks in Figure S2
are software output, not manually adjudicated annotations or a gold standard.
"""

from __future__ import annotations

import argparse
from io import BytesIO
from pathlib import Path
import shutil

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import numpy as np
import pandas as pd
from PIL import Image


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
            "savefig.dpi": 600,
            "savefig.transparent": False,
            "savefig.facecolor": "white",
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
        }
    )


def _save_rgb_tiff(fig: plt.Figure, path: Path) -> None:
    buffer = BytesIO()
    fig.savefig(buffer, format="tiff", dpi=600, bbox_inches="tight", facecolor="white", transparent=False)
    buffer.seek(0)
    with Image.open(buffer) as rendered:
        if rendered.mode == "RGBA":
            rgb = Image.new("RGB", rendered.size, "white")
            rgb.paste(rendered, mask=rendered.getchannel("A"))
        else:
            rgb = rendered.convert("RGB")
        rgb.save(path, format="TIFF", dpi=(600, 600), compression="tiff_lzw")


def save_figure(
    fig: plt.Figure,
    output_dir: Path,
    stem: str,
    formal_assets_dir: Path,
    formal_stem: str,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    formal_assets_dir.mkdir(parents=True, exist_ok=True)
    output_paths = {}
    for suffix in ("png", "svg", "pdf"):
        path = output_dir / f"{stem}.{suffix}"
        kwargs = {"dpi": 600} if suffix == "png" else {}
        fig.savefig(path, bbox_inches="tight", facecolor="white", transparent=False, **kwargs)
        output_paths[suffix] = path
    tiff_path = output_dir / f"{stem}.tif"
    _save_rgb_tiff(fig, tiff_path)
    output_paths["tif"] = tiff_path
    for suffix, source in output_paths.items():
        shutil.copy2(source, formal_assets_dir / f"{formal_stem}.{suffix}")
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


def figure_s2_signal(output_dir: Path, formal_assets_dir: Path) -> None:
    ecg = pd.read_csv(ECG_PATH)
    imu = pd.read_csv(IMU_PATH)
    peaks = pd.read_csv(PEAKS_PATH)
    metadata = pd.read_csv(METADATA_PATH).iloc[0]

    duration = float(metadata["dur_s"])
    detected = peaks.loc[peaks["source"] == "cleanlabels", "t_rel_s"].to_numpy(float)
    comparison = peaks.loc[peaks["source"] == "pt_refined", "t_rel_s"].to_numpy(float)
    if not (19.9 <= float(ecg["t_rel_s"].max()) <= 20.1 and duration == 20.0):
        raise AssertionError("Figure S2 source is not the expected 20-s segment")
    expected_peaks = int(metadata["beat_count"])
    if len(detected) != expected_peaks or len(comparison) != expected_peaks:
        raise AssertionError("Figure S2 peak counts differ from the deposited segment")
    matched = nearest_peak_match(detected * 1000.0, comparison * 1000.0)
    union = len(detected) + len(comparison) - matched
    bsqi = matched / union if union else np.nan
    if not np.isclose(bsqi, float(metadata["bsqi_50ms"])):
        raise AssertionError(f"unexpected cross-detector bSQI: {bsqi}")

    fig = plt.figure(figsize=(7.05, 5.1))
    grid = fig.add_gridspec(4, 1, height_ratios=[0.34, 2.4, 0.34, 1.6], hspace=0.10)
    ecg_header = fig.add_subplot(grid[0])
    ax_ecg = fig.add_subplot(grid[1])
    imu_header = fig.add_subplot(grid[2])
    ax_imu = fig.add_subplot(grid[3], sharex=ax_ecg)
    for header in (ecg_header, imu_header):
        header.set_axis_off()

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
    ax_ecg.tick_params(axis="x", which="both", labelbottom=False)
    ecg_header.text(0.0, 0.5, "a  Raw ECG with software-detected R peaks", ha="left", va="center", fontsize=10, weight="bold")
    ecg_header.legend(
        handles=ax_ecg.get_legend_handles_labels()[0],
        labels=ax_ecg.get_legend_handles_labels()[1],
        loc="center right",
        frameon=False,
        ncol=2,
        borderaxespad=0,
    )
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
    imu_header.text(0.0, 0.5, "b  Simultaneous 10 Hz tri-axial acceleration", ha="left", va="center", fontsize=10, weight="bold")
    imu_header.legend(
        handles=ax_imu.get_legend_handles_labels()[0],
        labels=ax_imu.get_legend_handles_labels()[1],
        loc="center right",
        frameon=False,
        ncol=3,
        borderaxespad=0,
    )
    ax_imu.grid(axis="x", alpha=0.18, lw=0.5)
    fig.suptitle(
        "Representative synchronized ECG and IMU segment",
        y=0.995,
        fontsize=12,
    )
    fig.subplots_adjust(top=0.92, bottom=0.10, left=0.10, right=0.98)
    save_figure(fig, output_dir, "figS2_signal_quality", formal_assets_dir, "Figure_S2")


def figure_s1_coverage(panel: pd.DataFrame, output_dir: Path, formal_assets_dir: Path) -> None:
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
        raise AssertionError("Figure S1 coverage scope differs from 24 experimental units / 40 dates / 12 pigs / A1-B1-B2")
    if cumulative_monitoring_days != 118:
        raise AssertionError(f"Figure S1 expected 118 cumulative monitoring days, got {cumulative_monitoring_days}")

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
    save_figure(fig, output_dir, "figS1_coverage_calendar", formal_assets_dir, "Figure_S1")


def figure_s3(output_dir: Path, formal_assets_dir: Path) -> None:
    counts = pd.read_csv(SCREENING_PATH).sort_values("stage_order")
    expected = [34109, 34026, 33993, 30000, 4518, 4380, 4079]
    if counts["retained_count"].astype(int).tolist() != expected:
        raise AssertionError("Figure S3 screening counts differ from locked values")

    fig, ax = plt.subplots(figsize=(7.05, 6.0))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    centers = [
        (0.22, 0.82),
        (0.78, 0.82),
        (0.78, 0.58),
        (0.22, 0.58),
        (0.22, 0.34),
        (0.78, 0.34),
        (0.78, 0.10),
    ]
    stage_colors = [
        COLORS["navy"],
        COLORS["blue"],
        COLORS["blue"],
        COLORS["green"],
        COLORS["orange"],
        COLORS["orange"],
        COLORS["navy"],
    ]
    stage_labels = [
        "Raw 5-min heat-production\nrecords",
        "HP > 0",
        "Deduplicated pig × timestamp",
        "Source-level quality control",
        "Candidate 30-min windows",
        "30-min windows with ≥3\nsource records",
        "Final windows with valid-HR\nfraction ≥50%",
    ]
    stage_notes = [
        "",
        "83 non-positive records excluded",
        "33 duplicate records excluded",
        "3,993 records excluded by source-level QC",
        "30,000 source records aggregated into candidates",
        "138 incomplete candidate windows excluded",
        "301 windows excluded for insufficient HR coverage",
    ]
    box_width = 0.40
    box_height = 0.18
    for index, ((x, y), count, label, note, color) in enumerate(zip(centers, expected, stage_labels, stage_notes, stage_colors)):
        filled = index in (0, 6)
        box = FancyBboxPatch(
            (x - box_width / 2, y - box_height / 2),
            box_width,
            box_height,
            boxstyle="round,pad=0.012,rounding_size=0.018",
            linewidth=1.2,
            edgecolor=color,
            facecolor=color if filled else "white",
        )
        ax.add_patch(box)
        text_color = "white" if filled else COLORS["dark"]
        ax.text(
            x - box_width / 2 + 0.035,
            y + box_height / 2 - 0.028,
            str(index + 1),
            ha="center",
            va="center",
            color="white",
            fontsize=9.5,
            weight="bold",
            bbox={"boxstyle": "circle,pad=0.22", "facecolor": color, "edgecolor": "none"},
            zorder=4,
        )
        ax.text(x, y + 0.040, label, ha="center", va="center", color=text_color, fontsize=11.2, weight="bold", linespacing=1.05)
        unit = "records" if index < 4 else "windows"
        ax.text(x, y - 0.010, f"n = {count:,} {unit}", ha="center", va="center", color=text_color, fontsize=10.4)
        if note:
            ax.text(x, y - 0.060, note, ha="center", va="center", color=text_color, fontsize=9.2)

    arrow_points = [
        ((0.42, 0.82), (0.58, 0.82)),
        ((0.78, 0.73), (0.78, 0.67)),
        ((0.58, 0.58), (0.42, 0.58)),
        ((0.22, 0.49), (0.22, 0.43)),
        ((0.42, 0.34), (0.58, 0.34)),
        ((0.78, 0.25), (0.78, 0.19)),
    ]
    for start, end in arrow_points:
        ax.add_patch(
            FancyArrowPatch(
                start,
                end,
                arrowstyle="-|>",
                mutation_scale=14,
                lw=1.2,
                color=COLORS["gray"],
                zorder=2,
            )
        )
    ax.set_title("Data-screening workflow for the formal 30-min modeling panel", loc="left", pad=10, fontsize=14)
    fig.subplots_adjust(left=0.02, right=0.98, top=0.90, bottom=0.02)
    save_figure(fig, output_dir, "figS3_screening_flow", formal_assets_dir, "Figure_S3")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results" / "reproduced")
    parser.add_argument(
        "--formal-assets-dir",
        type=Path,
        default=SUPPLEMENT_DIR / "figures",
        help="write submission-ready Figure_S1--S3 assets in PNG, SVG, PDF and RGB TIFF formats",
    )
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
    if (len(ecg), len(imu), len(peaks), len(metadata), len(screening)) != (10241, 200, 52, 1, 7):
        raise AssertionError("supplementary source-asset dimensions differ from the deposited release")
    coverage_scope = (
        panel["experimental_unit"].nunique(),
        panel["strict_day_group"].nunique(),
        panel["pig"].nunique(),
    )
    if coverage_scope != (24, 40, 12):
        raise AssertionError(f"unexpected coverage scope: {coverage_scope}")
    if screening.sort_values("stage_order")["retained_count"].astype(int).tolist() != [34109, 34026, 33993, 30000, 4518, 4380, 4079]:
        raise AssertionError("screening-count sequence mismatch")

    if not args.no_figures:
        figure_s1_coverage(panel, args.output_dir, args.formal_assets_dir)
        figure_s2_signal(args.output_dir, args.formal_assets_dir)
        figure_s3(args.output_dir, args.formal_assets_dir)
    print("SUPPLEMENTARY_REPRODUCIBILITY_PASS figures=S1,S2,S3")


if __name__ == "__main__":
    main()
