# Results and figure map

| Manuscript item | Source |
|---|---|
| Pig-held-out nested predictions | `scripts/02_run_nested_pig_cv.py` output OOF CSV |
| Uncalibrated wearable R² = 0.710 | `pred_HP__blend_top3__ensemble` in formal confirmation OOF |
| Fixed first-2-d calibrated R² = 0.800 | `pred_HP__blend_top3__hybrid_unit_fixed050` |
| Traditional deployment R² = 0.675 and baseline R² = 0.432 | baseline evaluation in `scripts/03_reproduce_results_and_figures.py` |
| Figures 2–4 | `scripts/03_reproduce_results_and_figures.py` |

The figure script asserts the locked numerical values before saving graphics. It uses deposited formal OOF predictions and does not rerun model selection.
