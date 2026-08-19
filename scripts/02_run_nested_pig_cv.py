"""Post-exploration strict confirmation of the 2026-only diet panel.

The exploration screen (``model_2026_diet_explore.py``) identified the
final configuration: four feature branches (the two strongest compact
branches from the formal nested run plus their full-motion counterparts)
and a seventeen-candidate library (the frozen thirteen plus cat_d4,
cat_d8, lgb_deep and xgb_d5).  This script re-runs the *exact* strict
nested machinery of ``formal_model_core.py`` — same ``run_nested``,
same grouped splits, same inner model/branch selection, same hybrid
calibration layers — with only the branch set and candidate library
swapped.  Nothing is re-selected from exploration outputs; every outer
fold repeats branch and model selection on grouped inner OOF only.

Feature branches
----------------
``compact_base`` / ``compact_feed_event``
    Field-by-field copies of the formal run's same-named variants.
``full_motion``
    Context (CONTEXT_NUMERIC + ENV + HRV, present columns) + every motion
    column from ``motion_columns()``.
``full_motion_feed_event``
    ``full_motion`` + PRIMARY_FEED + event_clock/meal_kernels/
    dose_interactions groups.
The ``full_`` prefix makes ``candidate_allowed`` automatically exclude
svr_rbf (``exclude_full=True``) from the two high-dimensional branches.
``diet_code`` never enters any numeric feature (asserted).

Examples
--------
python model_2026_diet_confirm.py --mode validate
python model_2026_diet_confirm.py --mode smoke --scenario both
python model_2026_diet_confirm.py --mode evaluate --scenario both
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import formal_model_core as core


DEFAULT_ROOT = Path(__file__).resolve().parents[1] / "results"
DEFAULT_PANEL = DEFAULT_ROOT / "2026_diet_30min_matched.csv"
DEFAULT_GROUPS = DEFAULT_ROOT / "feature_groups.json"
DEFAULT_OUT = DEFAULT_ROOT / "confirm_model_results"

NEW_CANDIDATES = [
    {
        "name": "cat_d4",
        "family": "cat",
        "params": {"iterations": 600, "depth": 4, "learning_rate": 0.05},
    },
    {
        "name": "cat_d8",
        "family": "cat",
        "params": {"iterations": 650, "depth": 8, "learning_rate": 0.03},
    },
    {
        "name": "lgb_deep",
        "family": "lgb",
        "params": {
            "n_estimators": 900,
            "learning_rate": 0.03,
            "num_leaves": 63,
        },
    },
    {
        "name": "xgb_d5",
        "family": "xgb",
        "params": {"n_estimators": 700, "learning_rate": 0.04, "max_depth": 5},
    },
]

SMOKE_CANDIDATE_NAMES = {
    "ridge_a10",
    "cat_d5",
    "cat_d8",
    "lgb_medium",
    "xgb_d3",
    "xgb_d5",
}


def confirm_candidates(mode: str) -> list[dict[str, object]]:
    """Frozen thirteen plus the four exploration winners (17 total)."""
    candidates = [
        {**candidate, "params": dict(candidate["params"])}
        for candidate in core.candidate_library("evaluate")
    ]
    candidates += [
        {**candidate, "params": dict(candidate["params"])}
        for candidate in NEW_CANDIDATES
    ]
    if mode == "smoke":
        kept = [
            candidate
            for candidate in candidates
            if str(candidate["name"]) in SMOKE_CANDIDATE_NAMES
        ]
        missing = SMOKE_CANDIDATE_NAMES - {
            str(candidate["name"]) for candidate in kept
        }
        if missing:
            raise ValueError(f"smoke candidates missing: {sorted(missing)}")
        return kept
    return candidates


def confirm_variants(
    frame, feature_groups: dict[str, list[str]]
) -> dict[str, dict[str, list[str]]]:
    """Four confirmation branches; compact ones copied field-by-field."""
    formal = core.feature_variants(frame, feature_groups, "evaluate")
    variants: dict[str, dict[str, list[str]]] = {
        "compact_base": {
            "numeric": list(formal["compact_base"]["numeric"]),
            "categorical": list(formal["compact_base"]["categorical"]),
        },
        "compact_feed_event": {
            "numeric": list(formal["compact_feed_event"]["numeric"]),
            "categorical": list(formal["compact_feed_event"]["categorical"]),
        },
    }

    context_numeric = core.present(
        frame, core.CONTEXT_NUMERIC + core.ENV + core.HRV
    )
    full_motion_numeric = core.dedupe(
        context_numeric + core.motion_columns(frame.columns)
    )
    categorical = core.present(frame, core.CONTEXT_CATEGORICAL)
    feed = core.present(frame, core.PRIMARY_FEED)
    event = core.present(
        frame,
        feature_groups["event_clock"]
        + feature_groups["meal_kernels"]
        + feature_groups["dose_interactions"],
    )
    variants["full_motion"] = {
        "numeric": full_motion_numeric,
        "categorical": list(categorical),
    }
    variants["full_motion_feed_event"] = {
        "numeric": core.dedupe(full_motion_numeric + feed + event),
        "categorical": list(categorical),
    }

    for name, spec in variants.items():
        numeric = spec["numeric"]
        if not numeric:
            raise ValueError(f"{name} has no numeric features")
        if len(numeric) != len(set(numeric)):
            raise AssertionError(f"duplicate numeric features in {name}")
        if len(spec["categorical"]) != len(set(spec["categorical"])):
            raise AssertionError(f"duplicate categorical features in {name}")
        if "diet_code" in numeric:
            raise AssertionError(f"diet_code entered numeric features in {name}")
        overlap = set(numeric) & set(spec["categorical"])
        if overlap:
            raise AssertionError(
                f"numeric/categorical overlap in {name}: {sorted(overlap)}"
            )
    return variants


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        choices=("validate", "smoke", "evaluate"),
        default="validate",
    )
    parser.add_argument(
        "--scenario",
        choices=("GKF_Pig",),
        default="GKF_Pig",
    )
    parser.add_argument("--panel", type=Path, default=DEFAULT_PANEL)
    parser.add_argument("--feature-groups", type=Path, default=DEFAULT_GROUPS)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument(
        "--bootstrap-reps",
        type=int,
        default=None,
        help="default 100 for smoke and 1200 for evaluate",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    scenarios = ("GKF_Pig",)
    frame0, groups = core.load_inputs(args.panel, args.feature_groups)
    panel_audit = core.validate_panel(frame0, groups)
    variants = confirm_variants(frame0, groups)
    candidates = confirm_candidates(
        "smoke" if args.mode == "smoke" else "evaluate"
    )

    validation = {
        "panel": panel_audit,
        "branches": {
            name: {
                "numeric_count": len(spec["numeric"]),
                "categorical": spec["categorical"],
                "diet_code_in_numeric": "diet_code" in spec["numeric"],
            }
            for name, spec in variants.items()
        },
        "candidate_count": len(candidates),
        "candidate_names": [candidate["name"] for candidate in candidates],
        "svr_excluded_from_full_branches": all(
            not core.candidate_allowed(
                {"name": "svr_rbf", "family": "svr", "params": {},
                 "exclude_full": True},
                branch,
            )
            for branch in ("full_motion", "full_motion_feed_event")
        ),
    }
    print(json.dumps(core.to_builtin(validation), ensure_ascii=False, indent=2))
    if args.mode == "validate":
        print("VALIDATION_PASS")
        return

    mode_out = args.out_dir / args.mode
    mode_out.mkdir(parents=True, exist_ok=True)
    bootstrap_reps = args.bootstrap_reps
    if bootstrap_reps is None:
        bootstrap_reps = 100 if args.mode == "smoke" else 1200
    if bootstrap_reps < 20:
        raise ValueError("bootstrap-reps must be at least 20")

    for scenario in scenarios:
        scenario_frame = frame0.copy()
        scenario_frame["_outer_group"] = scenario_frame[
            core.group_column(scenario)
        ].astype(str)
        summary, oof, _ = core.run_nested(
            scenario_frame,
            variants,
            candidates,
            scenario,
            args.mode,
            bootstrap_reps,
        )
        summary["input_panel"] = str(args.panel)
        summary["validation"] = validation
        summary["confirm_design"] = {
            "role": (
                "post-exploration strict confirmation: identical nested "
                "protocol as the formal run (same run_nested, grouped "
                "splits, per-outer-fold inner selection, hybrid "
                "calibration); only the branch set and candidate library "
                "differ"
            ),
            "branches": {
                "compact_base": "field-by-field copy of the formal variant",
                "compact_feed_event": (
                    "field-by-field copy of the formal variant"
                ),
                "full_motion": (
                    "context (CONTEXT_NUMERIC + ENV + HRV, present columns) "
                    "+ all motion_columns() columns"
                ),
                "full_motion_feed_event": (
                    "full_motion + PRIMARY_FEED + event_clock + "
                    "meal_kernels + dose_interactions"
                ),
            },
            "candidate_library": (
                "candidate_library('evaluate') 13 frozen candidates plus "
                "cat_d4, cat_d8, lgb_deep, xgb_d5 from the exploration "
                "screen (17 total; smoke uses a 6-candidate subset)"
            ),
            "differences_from_formal_run": [
                "compact_feed and compact_feed_fullfeeding branches dropped",
                "full_motion and full_motion_feed_event branches added "
                "(svr_rbf auto-excluded via exclude_full)",
                "four new candidates added; hyperparameters identical to "
                "the exploration screen",
                "no final all-data refit here; this script reports nested "
                "OOF only",
            ],
            "scientific_boundaries": (
                "unchanged: diet_code never a feature; planned feed is "
                "offered quantity; meal events are protocol times"
            ),
        }
        oof_path = mode_out / f"nested_2026_diet_confirm_{scenario}_oof.csv"
        summary_path = (
            mode_out / f"nested_2026_diet_confirm_{scenario}_summary.json"
        )
        oof.to_csv(oof_path, index=False, encoding="utf-8-sig")
        with summary_path.open("w", encoding="utf-8") as handle:
            json.dump(core.to_builtin(summary), handle, ensure_ascii=False,
                      indent=2)
        print(f"SAVED {scenario}: {summary_path} | {oof_path}", flush=True)
    print("DONE_EXIT_0")


if __name__ == "__main__":
    main()
