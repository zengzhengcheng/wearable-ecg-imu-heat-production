# Data layout

Raw observations are not distributed in this code repository. After obtaining the study dataset, arrange it as follows:

```text
data/
  raw/
    calorimetry/A1/*.xlsx
    calorimetry/B1/*.xlsx
    calorimetry/B2/*.xlsx
    ecg/<chamber>/cleanlabels/*.csv
    imu/<chamber>move/*.csv
    experiment_design.csv
  derived/
    hp/
    sensor_features_5min.csv
    panel_30min.csv
```

The `example/` directory contains headers only. It documents schemas and must not be used to calculate performance.
