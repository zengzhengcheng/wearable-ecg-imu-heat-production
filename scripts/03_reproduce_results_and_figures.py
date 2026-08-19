"""Reproduce the three-stage pig-held-out evaluation and Figures 2--4.

The saved formal pig-grouped nested predictions are the only source for the
main-model and feeding-ablation results.  Traditional baselines are refitted
with the exact saved outer test-pig assignments.  No manuscript section is
edited and no model-selection search or calibration-duration search is run.
"""

from __future__ import annotations

import json
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from matplotlib.patches import FancyBboxPatch, Patch, Rectangle
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
FORMAL_ROOT = Path(os.environ.get("FORMAL_RESULTS_DIR", ROOT / "results"))
PANEL_PATH = Path(
    os.environ.get("PANEL_PATH", ROOT / "data" / "derived" / "panel_30min.csv")
)
MAIN_OOF_PATH = (
    FORMAL_ROOT
    / "confirm_results/evaluate/nested_2026_diet_confirm_GKF_Pig_oof.csv"
)
FEED_OOF_PATH = (
    FORMAL_ROOT
    / "nested_model_results/evaluate/nested_2026_diet_GKF_Pig_oof.csv"
)

HP_SCALE = 6.0
SEEDS = (13, 42, 2026)
FOLLOWUP_PHASES = (2, 3, 4)
RAW_MAIN = "pred_HP__blend_top3__ensemble"
CAL_MAIN = "pred_HP__blend_top3__hybrid_unit_fixed050"

COLORS = {
    "navy": "#315A7D",
    "blue": "#6E9EC2",
    "green": "#4F9274",
    "orange": "#D19A55",
    "gray": "#B9C0C7",
    "dark": "#27323A",
    "light": "#EDF1F4",
}


def style() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 8,
            "axes.titlesize": 9,
            "axes.labelsize": 8.5,
            "xtick.labelsize": 7.5,
            "ytick.labelsize": 7.5,
            "axes.linewidth": 0.8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "legend.frameon": False,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "savefig.dpi": 400,
        }
    )


def save_figure(fig: plt.Figure, stem: str) -> None:
    out_dir = Path(os.environ.get("FIGURE_OUT_DIR", ROOT / "results" / "figures"))
    out_dir.mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "svg", "pdf"):
        fig.savefig(out_dir / f"{stem}.{suffix}", bbox_inches="tight")
    plt.close(fig)
    print(f"SAVED {stem}.png/.svg/.pdf", flush=True)


def metrics(truth: np.ndarray, prediction: np.ndarray) -> dict[str, float | int]:
    truth = np.asarray(truth, dtype=float)
    prediction = np.asarray(prediction, dtype=float)
    valid = np.isfinite(truth) & np.isfinite(prediction)
    if not valid.all():
        raise ValueError(f"non-finite metric rows: {int((~valid).sum())}")
    return {
        "n": int(valid.sum()),
        "r2_HP": float(r2_score(truth[valid], prediction[valid])),
        "MAE_kcal_per_30min": float(
            mean_absolute_error(truth[valid], prediction[valid])
        ),
        "RMSE_kcal_per_30min": float(
            mean_squared_error(truth[valid], prediction[valid]) ** 0.5
        ),
    }


def load_aligned_inputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    main = pd.read_csv(MAIN_OOF_PATH, low_memory=False)
    feed = pd.read_csv(FEED_OOF_PATH, low_memory=False)
    panel = pd.read_csv(PANEL_PATH, low_memory=False)
    for frame in (main, feed, panel):
        frame["datetime"] = pd.to_datetime(
            frame["datetime"], format="mixed", errors="raise"
        )

    if len(main) != 4079 or main["pig"].astype(str).nunique() != 12:
        raise AssertionError("formal main OOF is not the expected 4079-row panel")
    keys = ["pig", "datetime"]
    if main.duplicated(keys).any() or feed.duplicated(keys).any() or panel.duplicated(keys).any():
        raise AssertionError("duplicate pig/timestamp keys")

    main_index = pd.MultiIndex.from_frame(main[keys])
    feed = feed.set_index(keys).loc[main_index].reset_index()
    panel = panel.set_index(keys).loc[main_index].reset_index()
    if not np.allclose(
        pd.to_numeric(main["HP_true"], errors="raise"),
        pd.to_numeric(feed["HP_true"], errors="raise"),
        rtol=0,
        atol=1e-12,
    ):
        raise AssertionError("main and feeding OOF truth values differ")
    if not np.allclose(
        pd.to_numeric(main["HP_true"], errors="raise"),
        pd.to_numeric(panel["HP_kcal"], errors="raise"),
        rtol=0,
        atol=1e-12,
    ):
        raise AssertionError("formal OOF and panel HP values differ")
    if not np.array_equal(
        pd.to_numeric(main["phase_idx"], errors="raise").to_numpy(int),
        pd.to_numeric(panel["phase_idx"], errors="raise").to_numpy(int),
    ):
        raise AssertionError("formal OOF and panel phase assignments differ")
    return main.reset_index(drop=True), feed.reset_index(drop=True), panel.reset_index(drop=True)


def build_three_stage(
    main: pd.DataFrame, panel: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    phase = pd.to_numeric(main["phase_idx"], errors="raise").to_numpy(int)
    followup = np.isin(phase, FOLLOWUP_PHASES)
    initial = np.isin(phase, (0, 1))
    truth = pd.to_numeric(main["HP_true"], errors="raise").to_numpy(float) * HP_SCALE
    raw = pd.to_numeric(main[RAW_MAIN], errors="raise").to_numpy(float) * HP_SCALE
    calibrated = pd.to_numeric(main[CAL_MAIN], errors="raise").to_numpy(float) * HP_SCALE

    if int(followup.sum()) != 2470:
        raise AssertionError(f"expected 2470 follow-up rows, got {int(followup.sum())}")
    if not np.allclose(calibrated[phase == 4], raw[phase == 4], rtol=0, atol=1e-12):
        raise AssertionError("phase 4 changed under fixed-alpha calibration")
    if not np.allclose(calibrated[initial], raw[initial], rtol=0, atol=1e-12):
        raise AssertionError("initial calibration rows were modified")
    recalculated = raw.copy()
    units = main["experimental_unit"].astype(str).to_numpy()
    for unit in pd.unique(units):
        calibration_rows = (units == unit) & initial
        if not calibration_rows.any():
            continue
        ratio = float(
            np.mean(truth[calibration_rows])
            / max(float(np.mean(raw[calibration_rows])), 1e-8)
        )
        fed_followup = (units == unit) & np.isin(phase, (2, 3))
        recalculated[fed_followup] = raw[fed_followup] * (
            1.0 + 0.5 * (ratio - 1.0)
        )
    if not np.allclose(recalculated, calibrated, rtol=0, atol=1e-10):
        difference = float(np.max(np.abs(recalculated - calibrated)))
        raise AssertionError(
            "saved calibration differs from the fixed-alpha formula; "
            f"max absolute difference={difference}"
        )

    output = pd.DataFrame(
        {
            "pig": main["pig"].astype(str),
            "chamber": panel["chamber"].astype(str),
            "period": panel["period"],
            "experimental_unit": main["experimental_unit"].astype(str),
            "timestamp": main["datetime"],
            "phase_idx": phase,
            "outer_fold": pd.to_numeric(main["outer_fold"], errors="raise").astype(int),
            "observed_HP": truth,
            "predicted_HP_raw": raw,
            "predicted_HP_calibrated": calibrated,
            "is_initial_2d": initial,
            "is_followup_evaluation": followup,
        }
    )
    if output.groupby("pig")["outer_fold"].nunique().max() != 1:
        raise AssertionError("a pig appears in multiple outer test folds")

    rows: list[dict[str, object]] = []
    definitions = (
        (
            "All stages, uncalibrated",
            "all phases",
            np.ones(len(output), dtype=bool),
            raw,
        ),
        (
            "Follow-up, uncalibrated",
            "phase 2/3/4",
            followup,
            raw,
        ),
        (
            "Follow-up, individualized calibration",
            "phase 2/3/4",
            followup,
            calibrated,
        ),
    )
    for order, (stage, scope, mask, prediction) in enumerate(definitions, start=1):
        result = metrics(truth[mask], prediction[mask])
        rows.append(
            {
                "stage_order": order,
                "stage": stage,
                "evaluation_scope": scope,
                "pigs": int(output.loc[mask, "pig"].nunique()),
                **result,
            }
        )
    metric_frame = pd.DataFrame(rows)
    expected = np.array([0.6753180182365188, 0.7101946227116982, 0.7998914733588931])
    if not np.allclose(metric_frame["r2_HP"].to_numpy(float), expected, rtol=0, atol=1e-10):
        raise AssertionError(
            "three-stage R2 values differ from the locked analysis scope: "
            f"{metric_frame['r2_HP'].tolist()}"
        )
    return output, metric_frame


def make_estimator(kind: str, features: list[str], seed: int) -> Pipeline:
    preprocess = ColumnTransformer(
        [
            (
                "numeric",
                SimpleImputer(strategy="median", add_indicator=True),
                features,
            )
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )
    if kind == "Ridge":
        model = Ridge(alpha=10.0)
        steps: list[tuple[str, object]] = [
            ("preprocess", preprocess),
            ("scale", StandardScaler()),
            ("model", model),
        ]
    elif kind == "LightGBM":
        model = LGBMRegressor(
            n_estimators=600,
            learning_rate=0.025,
            num_leaves=20,
            min_child_samples=20,
            reg_lambda=2.0,
            subsample=0.85,
            colsample_bytree=0.85,
            verbosity=-1,
            n_jobs=4,
            random_state=seed,
        )
        steps = [("preprocess", preprocess), ("model", model)]
    else:
        raise ValueError(kind)
    return Pipeline(steps)


def fit_simple_baselines(
    main: pd.DataFrame, panel: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    configs: dict[str, tuple[list[str], str] | None] = {
        "Metabolic body weight": None,
        "Heart rate · Ridge": (["HR_mean"], "Ridge"),
        "Heart rate · LightGBM": (["HR_mean"], "LightGBM"),
        "ODBA · Ridge": (["ODBA_mean"], "Ridge"),
        "ODBA · LightGBM": (["ODBA_mean"], "LightGBM"),
        "Heart rate + ODBA · Ridge": (["HR_mean", "ODBA_mean"], "Ridge"),
        "Heart rate + ODBA · LightGBM": (["HR_mean", "ODBA_mean"], "LightGBM"),
    }
    y_norm = pd.to_numeric(panel["HP_per_W075"], errors="raise").to_numpy(float)
    factor = pd.to_numeric(panel["weight"], errors="raise").to_numpy(float) ** 0.75
    outer_fold = pd.to_numeric(main["outer_fold"], errors="raise").to_numpy(int)
    pigs = main["pig"].astype(str).to_numpy()
    predictions_norm = {
        name: np.full((len(panel), len(SEEDS)), np.nan) for name in configs
    }

    for fold in range(1, 6):
        test = outer_fold == fold
        held = sorted(pd.unique(pigs[test]).tolist())
        train = ~np.isin(pigs, held)
        if set(pigs[train]) & set(pigs[test]):
            raise AssertionError(f"pig leakage in baseline fold {fold}")
        constant = float(np.mean(y_norm[train]))
        predictions_norm["Metabolic body weight"][test, :] = constant

        def fit_seed(seed: int) -> tuple[int, dict[str, np.ndarray]]:
            seed_predictions: dict[str, np.ndarray] = {}
            for name, config in configs.items():
                if config is None:
                    continue
                features, kind = config
                estimator = make_estimator(kind, features, seed)
                estimator.fit(panel.loc[train, features], y_norm[train])
                seed_predictions[name] = np.asarray(
                    estimator.predict(panel.loc[test, features]), dtype=float
                ).reshape(-1)
            return seed, seed_predictions

        with ThreadPoolExecutor(max_workers=len(SEEDS)) as executor:
            for seed, seed_predictions in executor.map(fit_seed, SEEDS):
                seed_col = SEEDS.index(seed)
                for name, prediction in seed_predictions.items():
                    predictions_norm[name][test, seed_col] = prediction
        print(
            f"baseline fold={fold}/5 held={','.join(held)} "
            f"train={int(train.sum())} test={int(test.sum())}",
            flush=True,
        )

    truth_hp = pd.to_numeric(main["HP_true"], errors="raise").to_numpy(float) * HP_SCALE
    followup = pd.to_numeric(main["phase_idx"], errors="raise").isin(FOLLOWUP_PHASES).to_numpy()
    detail = pd.DataFrame(
        {
            "pig": main["pig"].astype(str),
            "chamber": panel["chamber"].astype(str),
            "period": panel["period"],
            "experimental_unit": main["experimental_unit"].astype(str),
            "timestamp": main["datetime"],
            "phase_idx": main["phase_idx"],
            "outer_fold": outer_fold,
            "observed_HP": truth_hp,
        }
    ).loc[followup].reset_index(drop=True)
    metric_rows: list[dict[str, object]] = []
    for order, (name, values) in enumerate(configs.items(), start=1):
        if not np.isfinite(predictions_norm[name]).all():
            raise ValueError(f"non-finite baseline prediction: {name}")
        ensemble_norm = np.mean(predictions_norm[name], axis=1)
        prediction_hp = ensemble_norm * factor * HP_SCALE
        detail[f"predicted_HP__{name}"] = prediction_hp[followup]
        metric_rows.append(
            {
                "display_order": order,
                "method": name,
                "feature_definition": (
                    "training-fold mean HP/W^0.75"
                    if name == "Metabolic body weight"
                    else "HR_mean"
                    if name.startswith("Heart rate ·")
                    else "ODBA_mean"
                    if name.startswith("ODBA ·")
                    else "HR_mean + ODBA_mean"
                ),
                "model": (
                    "constant" if name == "Metabolic body weight" else name.rsplit(" · ", 1)[-1]
                ),
                **metrics(truth_hp[followup], prediction_hp[followup]),
            }
        )
    main_raw = pd.to_numeric(main[RAW_MAIN], errors="raise").to_numpy(float) * HP_SCALE
    detail["predicted_HP__Main model (uncalibrated)"] = main_raw[followup]
    metric_rows.append(
        {
            "display_order": len(metric_rows) + 1,
            "method": "Main model (uncalibrated)",
            "feature_definition": "formal multimodal main model",
            "model": "three-model ensemble",
            **metrics(truth_hp[followup], main_raw[followup]),
        }
    )
    metric_frame = pd.DataFrame(metric_rows)
    if len(detail) != 2470 or not metric_frame["n"].eq(2470).all():
        raise AssertionError("baseline evaluation is not the locked 2470-row subset")
    return detail, metric_frame


def recover_feeding(
    main: pd.DataFrame, feed: pd.DataFrame, panel: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    followup = pd.to_numeric(main["phase_idx"], errors="raise").isin(FOLLOWUP_PHASES).to_numpy()
    truth = pd.to_numeric(feed["HP_true"], errors="raise").to_numpy(float) * HP_SCALE
    branches = {
        "Sensor/context base": "pred_HP__compact_base__ensemble",
        "+ planned feeding amount": "pred_HP__compact_feed__ensemble",
        "+ protocol meal timing": "pred_HP__compact_feed_event__ensemble",
    }
    detail = pd.DataFrame(
        {
            "pig": feed["pig"].astype(str),
            "chamber": panel["chamber"].astype(str),
            "period": panel["period"],
            "experimental_unit": feed["experimental_unit"].astype(str),
            "timestamp": feed["datetime"],
            "phase_idx": feed["phase_idx"],
            "outer_fold": feed["outer_fold"],
            "observed_HP": truth,
        }
    ).loc[followup].reset_index(drop=True)
    rows: list[dict[str, object]] = []
    base_r2: float | None = None
    for order, (label, column) in enumerate(branches.items(), start=1):
        prediction = pd.to_numeric(feed[column], errors="raise").to_numpy(float) * HP_SCALE
        result = metrics(truth[followup], prediction[followup])
        if base_r2 is None:
            base_r2 = float(result["r2_HP"])
        detail[f"predicted_HP__{label}"] = prediction[followup]
        rows.append(
            {
                "display_order": order,
                "feature_stage": label,
                **result,
                "delta_R2_vs_sensor_context_base": float(result["r2_HP"]) - base_r2,
                "source": "saved formal pig-grouped nested test-pig predictions",
            }
        )
    if len(detail) != 2470:
        raise AssertionError("feeding evaluation is not the locked 2470-row subset")
    return detail, pd.DataFrame(rows)


def draw_fig2(main: pd.DataFrame) -> pd.DataFrame:
    held_by_fold = {
        fold: sorted(
            main.loc[main["outer_fold"].eq(fold), "pig"].astype(str).unique().tolist()
        )
        for fold in range(1, 6)
    }
    pigs = sorted(main["pig"].astype(str).unique().tolist())
    rows: list[dict[str, object]] = []
    for fold, held in held_by_fold.items():
        for pig in pigs:
            rows.append(
                {
                    "outer_fold": fold,
                    "pig": pig,
                    "role": "held-out test" if pig in held else "training/inner selection",
                }
            )
    data = pd.DataFrame(rows)
    data.to_csv(HERE / "fig2_design_cv_framework_data.csv", index=False)

    fig = plt.figure(figsize=(7.2, 5.0))
    gs = fig.add_gridspec(2, 1, height_ratios=[0.95, 1.55], hspace=0.42)
    ax = fig.add_subplot(gs[0])
    phases = [
        (0, "Day 1", "adaptation", COLORS["blue"]),
        (1, "Day 2", "adaptation", COLORS["blue"]),
        (2, "Day 3", "measurement", COLORS["green"]),
        (3, "Day 4", "measurement", COLORS["green"]),
        (4, "Day 5", "fasting", COLORS["orange"]),
    ]
    for x, phase, role, color in phases:
        ax.add_patch(Rectangle((x, 0.5), 0.95, 0.72, fc=color, ec="white", lw=1.2))
        ax.text(x + 0.475, 0.91, phase, ha="center", va="center", color="white", weight="bold")
        ax.text(x + 0.475, 0.66, role, ha="center", va="center", color="white", fontsize=7)
    ax.annotate(
        "Initial 2 d: adaptation and individual calibration measurements",
        xy=(0.95, 0.42),
        xytext=(0.95, 0.12),
        ha="center",
        arrowprops=dict(arrowstyle="-[,widthB=5.7,lengthB=0.7", lw=1.0, color=COLORS["navy"]),
        color=COLORS["navy"],
    )
    ax.annotate(
        "Follow-up evaluation",
        xy=(3.45, 0.42),
        xytext=(3.45, 0.12),
        ha="center",
        arrowprops=dict(arrowstyle="-[,widthB=8.6,lengthB=0.7", lw=1.0, color=COLORS["green"]),
        color=COLORS["green"],
    )
    ax.set_xlim(-0.05, 5.0)
    ax.set_ylim(0, 1.55)
    ax.axis("off")
    ax.set_title("A  Experimental timeline", loc="left", weight="bold")

    ax = fig.add_subplot(gs[1])
    for row, fold in enumerate(range(1, 6)):
        held = set(held_by_fold[fold])
        for col, pig in enumerate(pigs):
            test = pig in held
            ax.add_patch(
                Rectangle(
                    (col, 4 - row),
                    0.92,
                    0.78,
                    fc=COLORS["navy"] if test else COLORS["light"],
                    ec="white",
                    lw=0.8,
                )
            )
            ax.text(
                col + 0.46,
                4.39 - row,
                pig.replace("pig_", ""),
                ha="center",
                va="center",
                fontsize=6.5,
                color="white" if test else COLORS["dark"],
            )
        ax.text(-0.35, 4.39 - row, f"Fold {fold}", ha="right", va="center")
        ax.text(
            12.25,
            4.39 - row,
            f"hold out {len(held)} pigs",
            ha="left",
            va="center",
            fontsize=7,
        )
    ax.text(
        11.9,
        4.98,
        "Inner 3-fold selection uses only\nthe remaining 9–10 training pigs",
        ha="left",
        va="center",
        fontsize=7.5,
        linespacing=1.35,
        color=COLORS["dark"],
    )
    ax.set_xlim(-1.1, 15.5)
    ax.set_ylim(-0.35, 5.05)
    ax.axis("off")
    ax.set_title("B  Pig-grouped 5-fold outer evaluation", loc="left", weight="bold")
    ax.legend(
        handles=[
            Patch(fc=COLORS["navy"], label="held-out test pigs"),
            Patch(fc=COLORS["light"], ec=COLORS["gray"], label="training pigs"),
        ],
        loc="lower left",
        bbox_to_anchor=(0.0, -0.12),
        ncol=2,
    )
    fig.suptitle("Experimental timeline and nested evaluation by animal", y=0.995, fontsize=10.5)
    save_figure(fig, "fig2_design_cv_framework")
    return data


def draw_fig3(predictions: pd.DataFrame, stage_metrics: pd.DataFrame) -> pd.DataFrame:
    panels = [
        ("A", "All stages\nUncalibrated", np.ones(len(predictions), dtype=bool), "predicted_HP_raw", COLORS["blue"]),
        ("B", "After initial 2 d\nUncalibrated", predictions["is_followup_evaluation"].to_numpy(bool), "predicted_HP_raw", COLORS["blue"]),
        ("C", "After initial 2 d\nIndividual calibration", predictions["is_followup_evaluation"].to_numpy(bool), "predicted_HP_calibrated", COLORS["green"]),
    ]
    long_rows: list[pd.DataFrame] = []
    all_values = [predictions["observed_HP"].to_numpy(float)]
    for _, _, mask, column, _ in panels:
        all_values.append(predictions.loc[mask, column].to_numpy(float))
    low = float(np.floor(min(np.nanmin(x) for x in all_values) / 5.0) * 5.0)
    high = float(np.ceil(max(np.nanmax(x) for x in all_values) / 5.0) * 5.0)

    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.65), sharex=True, sharey=True)
    for i, (letter, title, mask, column, color) in enumerate(panels):
        frame = predictions.loc[mask, ["pig", "timestamp", "observed_HP", column]].copy()
        frame = frame.rename(columns={column: "predicted_HP"})
        frame.insert(0, "panel", letter)
        frame.insert(1, "stage", stage_metrics.iloc[i]["stage"])
        long_rows.append(frame)
        ax = axes[i]
        ax.scatter(
            frame["observed_HP"],
            frame["predicted_HP"],
            s=5,
            alpha=0.22,
            color=color,
            edgecolors="none",
            rasterized=True,
        )
        ax.plot([low, high], [low, high], "--", color="0.25", lw=0.9)
        ax.set_xlim(low, high)
        ax.set_ylim(low, high)
        ax.set_aspect("equal", adjustable="box")
        ax.grid(alpha=0.18, lw=0.5)
        ax.set_title(f"{letter}  {title}", loc="left", weight="bold")
        ax.text(
            0.05,
            0.93,
            f"n = {int(stage_metrics.iloc[i]['n']):,}\n$R^2$ = {stage_metrics.iloc[i]['r2_HP']:.3f}",
            transform=ax.transAxes,
            ha="left",
            va="top",
            bbox=dict(fc="white", ec="none", alpha=0.82, pad=2.0),
        )
        ax.set_xlabel("Observed HP (kcal per 30 min)")
    axes[0].set_ylabel("Predicted HP (kcal per 30 min)")
    fig.suptitle("Three consecutive stages of prediction for previously unseen animals", y=1.01, fontsize=10)
    fig.tight_layout()
    save_figure(fig, "fig3_main_prediction_calibration")
    data = pd.concat(long_rows, ignore_index=True)
    data.to_csv(HERE / "fig3_main_prediction_calibration_data.csv", index=False)
    return data


def draw_fig4(baseline_metrics: pd.DataFrame) -> pd.DataFrame:
    data = baseline_metrics.sort_values("display_order").copy()
    colors = [COLORS["gray"]] * (len(data) - 1) + [COLORS["navy"]]
    fig, ax = plt.subplots(figsize=(6.4, 3.65))
    y = np.arange(len(data))
    ax.barh(y, data["r2_HP"], color=colors, height=0.62, edgecolor="white")
    for yi, value in zip(y, data["r2_HP"]):
        ax.text(value + 0.012, yi, f"{value:.3f}", va="center", ha="left", fontsize=7.5)
    ax.set_yticks(y)
    ax.set_yticklabels(data["method"])
    ax.invert_yaxis()
    ax.set_xlim(0, max(0.76, float(data["r2_HP"].max()) + 0.08))
    ax.set_xlabel("$R^2$ on follow-up windows")
    ax.set_title("Traditional parsimonious baselines and the uncalibrated main model", loc="left")
    ax.grid(axis="x", alpha=0.22, lw=0.5)
    ax.set_axisbelow(True)
    ax.text(
        0.99,
        0.98,
        "Same 2,470 held-out-animal windows for every method",
        transform=ax.transAxes,
        ha="right",
        va="top",
        color="0.35",
        fontsize=7.5,
        bbox=dict(fc="white", ec="none", alpha=0.85, pad=1.5),
    )
    fig.tight_layout()
    save_figure(fig, "fig4_baseline_comparison")
    data.to_csv(HERE / "fig4_baseline_comparison_data.csv", index=False)
    return data


def write_reply(
    stage_metrics: pd.DataFrame,
    baseline_metrics: pd.DataFrame,
    feeding_metrics: pd.DataFrame,
) -> None:
    stage = stage_metrics.set_index("stage")
    base = baseline_metrics.sort_values("display_order")
    feeding = feeding_metrics.sort_values("display_order")
    baseline_lines = [
        "| Method | n | R²(HP) | MAE | RMSE |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in base.to_dict(orient="records"):
        baseline_lines.append(
            f"| {row['method']} | {row['n']} | {row['r2_HP']:.6f} | "
            f"{row['MAE_kcal_per_30min']:.4f} | {row['RMSE_kcal_per_30min']:.4f} |"
        )
    feeding_lines = [
        "| Feature stage | n | R²(HP) | ΔR² vs base |",
        "|---|---:|---:|---:|",
    ]
    for row in feeding.to_dict(orient="records"):
        feeding_lines.append(
            f"| {row['feature_stage']} | {row['n']} | {row['r2_HP']:.6f} | "
            f"{row['delta_R2_vs_sensor_context_base']:+.6f} |"
        )
    a = stage.loc["All stages, uncalibrated"]
    b = stage.loc["Follow-up, uncalibrated"]
    c = stage.loc["Follow-up, individualized calibration"]
    text = f"""# 三阶段跨猪统一评估回传（2026-08-05）

## 总结

复核通过。4,079个全阶段窗口与2,470个后续评价窗口均与正式保存的按猪分组嵌套预测一致。三阶段来自同一套主模型对各轮测试猪生成的预测：先改变评价范围，再在相同2,470个后续窗口上应用固定α=0.5的个体校准；没有重新训练或微调主模型。

## 三阶段主结果

| Stage | Scope | n | R²(HP) | MAE | RMSE |
|---|---|---:|---:|---:|---:|
| All stages, uncalibrated | all phases | {int(a['n'])} | {a['r2_HP']:.6f} | {a['MAE_kcal_per_30min']:.4f} | {a['RMSE_kcal_per_30min']:.4f} |
| Follow-up, uncalibrated | phase 2/3/4 | {int(b['n'])} | {b['r2_HP']:.6f} | {b['MAE_kcal_per_30min']:.4f} | {b['RMSE_kcal_per_30min']:.4f} |
| Follow-up, individualized calibration | same phase 2/3/4 rows | {int(c['n'])} | {c['r2_HP']:.6f} | {c['MAE_kcal_per_30min']:.4f} | {c['RMSE_kcal_per_30min']:.4f} |

MAE和RMSE单位均为kcal/30 min。正文四舍五入口径为0.675、0.710和0.800。

## 传统简约基线

所有方法均使用相同五折外层测试猪；训练只使用其余9–10头猪的有效窗口，最终指标只汇总同一2,470个phase 2/3/4窗口。仅心率固定为HR_mean，活动量固定为ODBA_mean；没有使用旧的99列运动块。

{chr(10).join(baseline_lines)}

## 饲喂相关变量

该项可从正式保存的跨猪嵌套测试猪预测可靠恢复，统一在2,470个后续窗口上计算：

{chr(10).join(feeding_lines)}

该结果只表示计划投喂量和协议餐次变量是否增加预测信息，不代表实际采食量，也不作营养或机制解释。

## 图件与数据

- `figures/fig2_design_cv_framework.png/.svg/.pdf`
- `figures/fig3_main_prediction_calibration.png/.svg/.pdf`
- `figures/fig3_main_prediction_calibration_data.csv`
- `figures/fig4_baseline_comparison.png/.svg/.pdf`
- `figures/fig4_baseline_comparison_data.csv`
- `three_stage_main_metrics_20260805.csv`
- `three_stage_predictions_20260805.csv`
- `baseline_followup_subset_20260805.csv`
- `feeding_followup_subset_20260805.csv`
- `feeding_followup_metrics_20260805.csv`

## 旧图投稿处理

以下旧图不应进入本轮投稿包：`fig3_scatter_dual.*`、`fig4_calibration.*`。`figS5_daily_profile.*`仍来自日期留出预测，也应删除；本轮未强行重绘S5。现存旧版PNG已移至`figures/superseded_three_stage_20260805/`，保留追溯但不再位于活动图件目录。

## 异常检查

未发现改变论文三阶段故事的异常。4,079与2,470样本数、0.675/0.710/0.800三个锁定值、外层猪只归属和固定α=0.5标定范围均一致。传统基线已改为论文定义的真正简约输入，因此其数值会与旧99列运动块基线不同，这是定义修正而非数据异常。

## 可复跑脚本

- `figures/three_stage_eval_20260805.py`
- `figures/make_figures_remap.py`的正式总入口已改为调用上述三阶段脚本，并停止生成旧图2、旧图3、旧图4和日期留出来源的S5。
"""
    (ROOT / "REPLY_FROM_ANALYSIS_AI_THREE_STAGE_EVAL_20260805.md").write_text(
        text, encoding="utf-8"
    )


def main() -> None:
    style()
    main_oof, feed_oof, panel = load_aligned_inputs()
    predictions, stage_metrics = build_three_stage(main_oof, panel)
    predictions.to_csv(ROOT / "three_stage_predictions_20260805.csv", index=False)
    stage_metrics.to_csv(ROOT / "three_stage_main_metrics_20260805.csv", index=False)

    baseline_detail, baseline_metrics = fit_simple_baselines(main_oof, panel)
    baseline_detail.to_csv(ROOT / "baseline_followup_subset_20260805.csv", index=False)

    feeding_detail, feeding_metrics = recover_feeding(main_oof, feed_oof, panel)
    feeding_detail.to_csv(ROOT / "feeding_followup_subset_20260805.csv", index=False)
    feeding_metrics.to_csv(ROOT / "feeding_followup_metrics_20260805.csv", index=False)

    draw_fig2(main_oof)
    draw_fig3(predictions, stage_metrics)
    draw_fig4(baseline_metrics)
    write_reply(stage_metrics, baseline_metrics, feeding_metrics)

    summary = {
        "status": "PASS",
        "main_oof": str(MAIN_OOF_PATH),
        "feeding_oof": str(FEED_OOF_PATH),
        "panel": str(PANEL_PATH),
        "three_stage": stage_metrics.to_dict(orient="records"),
        "baselines": baseline_metrics.to_dict(orient="records"),
        "feeding": feeding_metrics.to_dict(orient="records"),
        "constraints": {
            "main_predictions_refit": False,
            "calibration_alpha": 0.5,
            "calibration_phases": [0, 1],
            "calibrated_prediction_phases": [2, 3],
            "unchanged_prediction_phases": [4],
            "evaluation_phases": [2, 3, 4],
            "confidence_intervals": False,
        },
    }
    with (ROOT / "three_stage_eval_summary_20260805.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
    print(stage_metrics.to_string(index=False), flush=True)
    print(baseline_metrics.to_string(index=False), flush=True)
    print(feeding_metrics.to_string(index=False), flush=True)
    print("DONE_EXIT_0", flush=True)


if __name__ == "__main__":
    main()
