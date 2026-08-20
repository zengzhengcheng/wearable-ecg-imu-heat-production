"""Verify the locked saved predictions and reproduce manuscript Figures 2--4.

This entry point reads only repository-deposited analysis assets. It does not
fit a model, run hyperparameter search, or select a calibration strength.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
ANALYSIS_DIR = ROOT / "data" / "analysis"
REFERENCE_DIR = ROOT / "reference_results"
PANEL_PATH = ANALYSIS_DIR / "modeling_panel_30min.csv"
PREDICTIONS_PATH = REFERENCE_DIR / "formal_window_predictions.csv"
FOLDS_PATH = REFERENCE_DIR / "outer_fold_assignments.csv"
BASELINE_PATH = REFERENCE_DIR / "baseline_followup_results.csv"
FEEDING_PATH = REFERENCE_DIR / "feeding_followup_results.csv"
LOCKED_PATH = REFERENCE_DIR / "locked_metrics.csv"
ALPHA = 0.5
TRADITIONAL_BASELINE_COLUMN = "predicted_HP__Heart rate + ODBA · Ridge"
COLORS = {"navy": "#315A7D", "blue": "#6E9EC2", "green": "#4F9274", "orange": "#D19A55", "gray": "#B9C0C7", "light": "#EDF1F4"}


def metric_frame(truth: np.ndarray, prediction: np.ndarray) -> dict[str, float | int]:
    truth = np.asarray(truth, dtype=float)
    prediction = np.asarray(prediction, dtype=float)
    if not (np.isfinite(truth).all() and np.isfinite(prediction).all()):
        raise ValueError("metric inputs contain non-finite values")
    residual = float(np.sum((truth - prediction) ** 2))
    total = float(np.sum((truth - truth.mean()) ** 2))
    error = truth - prediction
    return {
        "n": int(len(truth)),
        "r2_HP": 1.0 - residual / total,
        "MAE_kcal_per_30min": float(np.mean(np.abs(error))),
        "RMSE_kcal_per_30min": float(np.sqrt(np.mean(error**2))),
    }


def load_assets() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    panel = pd.read_csv(PANEL_PATH, low_memory=False)
    predictions = pd.read_csv(PREDICTIONS_PATH, low_memory=False)
    folds = pd.read_csv(FOLDS_PATH)
    baseline = pd.read_csv(BASELINE_PATH, low_memory=False)
    feeding = pd.read_csv(FEEDING_PATH, low_memory=False)
    if panel.shape != (4079, 345):
        raise AssertionError(f"expected 4079 x 345 panel, got {panel.shape}")
    if len(predictions) != 4079 or predictions["pig"].nunique() != 12:
        raise AssertionError("formal predictions are not the expected 4079 rows / 12 pigs")
    if predictions.duplicated(["pig", "timestamp"]).any():
        raise AssertionError("duplicate pig/timestamp keys in formal predictions")
    if len(baseline) != 2470 or len(feeding) != 2470:
        raise AssertionError("follow-up result files are not the expected 2470 rows")
    if len(folds) != 12 or folds["pig"].nunique() != 12:
        raise AssertionError("outer-fold assignment file must contain one row per pig")
    if int(predictions.groupby("pig")["outer_fold"].nunique().max()) != 1:
        raise AssertionError("a pig appears in multiple outer test folds")
    expected_folds = predictions.groupby("pig")["outer_fold"].first().astype(int)
    deposited_folds = folds.set_index("pig")["outer_fold_as_test"].astype(int)
    if not expected_folds.sort_index().equals(deposited_folds.sort_index()):
        raise AssertionError("outer-fold assignments differ from formal predictions")
    return panel, predictions, folds, baseline, feeding


def verify_calibration(predictions: pd.DataFrame) -> None:
    phase = pd.to_numeric(predictions["phase_idx"], errors="raise").to_numpy(int)
    recalculated = predictions["predicted_HP_raw"].to_numpy(float).copy()
    for _, indices in predictions.groupby("experimental_unit").groups.items():
        idx = np.asarray(list(indices), dtype=int)
        initial = idx[np.isin(phase[idx], (0, 1))]
        ratio = predictions.loc[initial, "observed_HP"].mean() / predictions.loc[initial, "predicted_HP_raw"].mean()
        fed_followup = idx[np.isin(phase[idx], (2, 3))]
        recalculated[fed_followup] *= 1.0 + ALPHA * (ratio - 1.0)
    saved = predictions["predicted_HP_calibrated"].to_numpy(float)
    if not np.allclose(recalculated, saved, rtol=0, atol=1e-8):
        raise AssertionError("saved calibration does not match fixed alpha=0.5")


def compute_results(predictions: pd.DataFrame, baseline: pd.DataFrame, feeding: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, float]]:
    followup = predictions["is_followup_evaluation"].astype(str).str.lower().eq("true").to_numpy()
    if int(followup.sum()) != 2470:
        raise AssertionError("expected 2470 phase 2/3/4 follow-up windows")
    stage_specs = [
        (1, "All stages, uncalibrated", "all phases", np.ones(len(predictions), dtype=bool), "predicted_HP_raw"),
        (2, "Follow-up, uncalibrated", "phase 2/3/4", followup, "predicted_HP_raw"),
        (3, "Follow-up, individualized calibration", "phase 2/3/4", followup, "predicted_HP_calibrated"),
    ]
    stage_rows = []
    for order, label, scope, mask, column in stage_specs:
        stage_rows.append({"stage_order": order, "stage": label, "evaluation_scope": scope, "pigs": int(predictions.loc[mask, "pig"].nunique()), **metric_frame(predictions.loc[mask, "observed_HP"], predictions.loc[mask, column])})
    stages = pd.DataFrame(stage_rows)

    baseline_rows = []
    for order, column in enumerate([name for name in baseline if name.startswith("predicted_HP__")], start=1):
        baseline_rows.append({"display_order": order, "method": column.removeprefix("predicted_HP__"), **metric_frame(baseline["observed_HP"], baseline[column])})
    baseline_metrics = pd.DataFrame(baseline_rows)

    feeding_rows = []
    base_r2 = None
    for order, column in enumerate([name for name in feeding if name.startswith("predicted_HP__")], start=1):
        result = metric_frame(feeding["observed_HP"], feeding[column])
        if base_r2 is None:
            base_r2 = float(result["r2_HP"])
        feeding_rows.append({"display_order": order, "feature_stage": column.removeprefix("predicted_HP__"), **result, "delta_R2_vs_sensor_context_base": float(result["r2_HP"]) - base_r2})
    feeding_metrics = pd.DataFrame(feeding_rows)

    locked = {
        "traditional_deployment": float(stages.iloc[0]["r2_HP"]),
        "wearable_uncalibrated": float(stages.iloc[1]["r2_HP"]),
        "wearable_fixed_2d_calibration": float(stages.iloc[2]["r2_HP"]),
        "traditional_baseline": float(metric_frame(baseline["observed_HP"], baseline[TRADITIONAL_BASELINE_COLUMN])["r2_HP"]),
    }
    reference = pd.read_csv(LOCKED_PATH).set_index("result")["r2"].to_dict()
    for name, value in locked.items():
        if round(value, 3) != round(float(reference[name]), 3):
            raise AssertionError(f"{name} differs from locked R2: {value}")
    return stages, baseline_metrics, feeding_metrics, locked


def save_figure(fig: plt.Figure, output_dir: Path, stem: str) -> None:
    for suffix in ("png", "svg", "pdf"):
        fig.savefig(output_dir / f"{stem}.{suffix}", bbox_inches="tight", dpi=400)
    plt.close(fig)


def draw_fig2(folds: pd.DataFrame, output_dir: Path) -> None:
    pigs = sorted(folds["pig"].astype(str))
    fold_map = folds.set_index("pig")["outer_fold_as_test"].astype(int).to_dict()
    fig = plt.figure(figsize=(7.2, 4.8))
    axes = fig.subplots(2, 1, gridspec_kw={"height_ratios": [1, 1.5]})
    labels = [("Day 1", "initial 2 d", COLORS["blue"]), ("Day 2", "initial 2 d", COLORS["blue"]), ("Day 3", "follow-up", COLORS["green"]), ("Day 4", "follow-up", COLORS["green"]), ("Day 5", "fasting", COLORS["orange"])]
    for index, (day, role, color) in enumerate(labels):
        axes[0].barh(0, 0.95, left=index, color=color, edgecolor="white")
        axes[0].text(index + 0.475, 0, f"{day}\n{role}", ha="center", va="center", color="white", fontsize=7)
    axes[0].set_xlim(0, 5)
    axes[0].axis("off")
    axes[0].set_title("A  Experimental timeline", loc="left", fontweight="bold")
    matrix = np.array([[1 if fold_map[pig] == fold else 0 for pig in pigs] for fold in range(1, 6)])
    axes[1].imshow(matrix, cmap=matplotlib.colors.ListedColormap([COLORS["light"], COLORS["navy"]]), aspect="auto")
    axes[1].set_xticks(range(len(pigs)), [pig.replace("pig_", "") for pig in pigs])
    axes[1].set_yticks(range(5), [f"Fold {fold}" for fold in range(1, 6)])
    axes[1].set_xlabel("True pig identity (dark = held-out test pig)")
    axes[1].set_title("B  Pig-grouped five-fold outer evaluation", loc="left", fontweight="bold")
    fig.tight_layout()
    save_figure(fig, output_dir, "fig2_design_cv_framework")


def draw_fig3(predictions: pd.DataFrame, stages: pd.DataFrame, output_dir: Path) -> None:
    followup = predictions["is_followup_evaluation"].astype(str).str.lower().eq("true").to_numpy()
    panels = [("Full period\nUncalibrated", np.ones(len(predictions), bool), "predicted_HP_raw", COLORS["blue"]), ("Post-adaptation period\nUncalibrated", followup, "predicted_HP_raw", COLORS["blue"]), ("Post-adaptation period\nIndividual calibration", followup, "predicted_HP_calibrated", COLORS["green"])]
    values = predictions[["observed_HP", "predicted_HP_raw", "predicted_HP_calibrated"]].to_numpy(float)
    low, high = np.floor(values.min() / 5) * 5, np.ceil(values.max() / 5) * 5
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.7), sharex=True, sharey=True)
    for index, (title, mask, column, color) in enumerate(panels):
        axes[index].scatter(predictions.loc[mask, "observed_HP"], predictions.loc[mask, column], s=5, alpha=0.22, color=color, edgecolors="none", rasterized=True)
        axes[index].plot([low, high], [low, high], "--", color="0.25", lw=0.9)
        axes[index].set(xlim=(low, high), ylim=(low, high), xlabel="Observed HP\n(kcal per 30 min)")
        axes[index].set_aspect("equal", adjustable="box")
        axes[index].set_title(f"{chr(65 + index)}  {title}", loc="left", fontweight="bold", fontsize=8)
        axes[index].text(0.05, 0.93, f"n = {int(stages.iloc[index]['n']):,}\n$R^2$ = {stages.iloc[index]['r2_HP']:.3f}", transform=axes[index].transAxes, va="top")
    axes[0].set_ylabel("Predicted HP (kcal per 30 min)")
    fig.suptitle(
        "Prediction across the full and post-adaptation periods, with and without individual calibration",
        y=1.02,
        fontsize=9.5,
    )
    fig.tight_layout()
    save_figure(fig, output_dir, "fig3_main_prediction_calibration")


def draw_fig4(baseline_metrics: pd.DataFrame, output_dir: Path) -> None:
    data = baseline_metrics.sort_values("display_order")
    colors = [COLORS["gray"]] * len(data)
    colors[data["method"].tolist().index("Main model (uncalibrated)")] = COLORS["navy"]
    fig, ax = plt.subplots(figsize=(6.4, 3.7))
    y = np.arange(len(data))
    ax.barh(y, data["r2_HP"], color=colors, height=0.62)
    for yi, value in zip(y, data["r2_HP"]):
        ax.text(value + 0.01, yi, f"{value:.3f}", va="center", fontsize=7.5)
    ax.set_yticks(y, data["method"])
    ax.invert_yaxis()
    ax.set_xlim(0, 0.78)
    ax.set_xlabel("R² on the same 2,470 follow-up windows")
    ax.set_title("Traditional baselines and uncalibrated main model", loc="left")
    ax.grid(axis="x", alpha=0.2)
    fig.tight_layout()
    save_figure(fig, output_dir, "fig4_baseline_comparison")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results" / "reproduced")
    parser.add_argument("--no-figures", action="store_true", help="verify metrics without rendering Figures 2--4")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    _, predictions, folds, baseline, feeding = load_assets()
    verify_calibration(predictions)
    stages, baseline_metrics, feeding_metrics, locked = compute_results(predictions, baseline, feeding)
    stages.to_csv(args.output_dir / "three_stage_main_metrics_reproduced.csv", index=False)
    baseline_metrics.to_csv(args.output_dir / "baseline_metrics_reproduced.csv", index=False)
    feeding_metrics.to_csv(args.output_dir / "feeding_metrics_reproduced.csv", index=False)
    if not args.no_figures:
        draw_fig2(folds, args.output_dir)
        draw_fig3(predictions, stages, args.output_dir)
        draw_fig4(baseline_metrics, args.output_dir)
    for name, value in locked.items():
        print(f"LOCKED_R2 {name}={value:.12f} manuscript={value:.3f}")
    print("REPRODUCIBILITY_PASS figures=" + ("skipped" if args.no_figures else "2,3,4"))


if __name__ == "__main__":
    main()
