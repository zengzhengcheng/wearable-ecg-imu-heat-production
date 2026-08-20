# Supplementary-figure source assets

These compact files support direct reproduction of manuscript Figures S1--S3 with `scripts/04_reproduce_supplementary_figures.py`.

- `signal_quality_ecg.csv`: representative 20-s raw ECG segment (512 Hz).
- `signal_quality_imu.csv`: simultaneous tri-axial acceleration segment (10 Hz).
- `signal_quality_rpeaks.csv`: two software-generated R-peak series used for display and the cross-detector bSQI check. These are not manually adjudicated annotations or a gold standard.
- `signal_quality_metadata.csv`: animal, chamber, date, segment and summary metadata for Figure S2.
- `signal_candidate_review.csv` and `signal_candidate_contact_sheet.png`: objective metrics and the nine real 20-s candidates reviewed before selecting the Figure S2 segment. R-peak times were not manually edited.
- `screening_counts.csv`: the complete locked screening chain for Figure S3, including the 4,518 candidate-window stage.
- `figures/`: submission-ready Figures S1--S3 in PNG, SVG, PDF and white-background RGB TIFF (600 dpi) formats.

Figure S1 is derived directly from the deposited formal panel at `data/analysis/modeling_panel_30min.csv`. Figure S2 uses the selected `pig_11`, chamber A1, 2026-07-01 07:08:07–07:08:27 segment: 26 formal peaks, 26 independent-detector peaks, bSQI = 1.000, minimum RR = 705 ms and complete synchronized ECG/IMU coverage. The public workflow intentionally excludes superseded Optuna and date-holdout supplementary figures.
