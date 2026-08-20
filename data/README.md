# Data layout

The repository contains the compact formal analysis data used by the manuscript:

```text
data/
  analysis/
    modeling_panel_30min.csv
    modeling_data_dictionary.csv
    feature_groups.json
    README.md
  example/
    panel_30min_header.csv
```

`analysis/modeling_panel_30min.csv` is the formal 4,079-row, 345-column panel used by the locked pig-grouped analysis. The `example/` header remains only as a small schema illustration and must not be used to calculate performance.

Large raw/near-raw acquisitions, extracted sensor data, 5-min calorimetry records and experiment metadata are deposited separately in ScienceDB: https://doi.org/10.57760/sciencedb.00zyg.

The optional upstream reconstruction route writes transient files beneath `data/derived/`; that directory is git-ignored. Research-data licensing is separate from the code license; see `DATA_LICENSE.md` at the repository root.
