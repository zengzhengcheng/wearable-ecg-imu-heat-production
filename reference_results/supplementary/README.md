# Supplementary-figure source assets

These compact files support direct reproduction of manuscript Figures S1--S3 with `scripts/04_reproduce_supplementary_figures.py`.

- `signal_quality_ecg.csv`: representative 20-s raw ECG segment (512 Hz).
- `signal_quality_imu.csv`: simultaneous tri-axial acceleration segment (10 Hz).
- `signal_quality_rpeaks.csv`: two software-generated R-peak series used for display and the cross-detector bSQI check. These are not manually adjudicated annotations or a gold standard.
- `signal_quality_metadata.csv`: animal, chamber, date, segment and summary metadata for Figure S1.
- `screening_counts.csv`: retained counts and defensible exclusion descriptions for Figure S3.

Figure S2 is derived directly from the deposited formal panel at `data/analysis/modeling_panel_30min.csv`. The public workflow intentionally excludes superseded Optuna and date-holdout supplementary figures.
