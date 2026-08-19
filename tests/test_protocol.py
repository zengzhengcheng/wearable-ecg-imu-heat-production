import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import formal_model_core as core


def test_pig_grouped_splits_have_no_overlap():
    frame = pd.DataFrame({"pig": [f"p{i}" for i in range(5) for _ in range(2)]})
    for train, test in core.grouped_splits(frame, "GKF_Pig", 5):
        assert set(frame.iloc[train].pig).isdisjoint(set(frame.iloc[test].pig))


def test_fixed_half_shrinkage_calibration():
    frame = pd.DataFrame(
        {
            "pig": ["p1"] * 4,
            "experimental_unit": ["u1"] * 4,
            "weight": [1.0] * 4,
            "HP_kcal": [12.0, 12.0, 12.0, 12.0],
            "phase_idx": [0, 1, 2, 4],
        }
    )
    prediction, info = core.hybrid_calibration(
        frame, np.array([6.0, 6.0, 6.0, 6.0]), np.zeros(4),
        "GKF_Pig", True, 0.5,
    )
    assert np.allclose(prediction, [6.0, 6.0, 9.0, 6.0])
    assert info["alpha"] == 0.5
