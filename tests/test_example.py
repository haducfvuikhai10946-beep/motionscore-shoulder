import json
from pathlib import Path
import numpy as np
import pandas as pd
from motionscore.features import NAMES,aggregate
from motionscore.scoring import MotionScorer


def test_historical_example():
    root=Path(__file__).resolve().parents[1]/"examples"
    gold=json.loads((root/"example_expected.json").read_text())
    scorer=MotionScorer(root/"models","Ridge")
    features=pd.read_csv(root/"example_features.csv",float_precision="round_trip")[NAMES].to_numpy(float)
    result=scorer.score(features)
    assert np.allclose(result["MotionScore"],gold["MotionScore"],atol=1e-10,rtol=0)
    assert np.array_equal(result["band"],gold["band"])
    pose=pd.read_csv(root/"example_pose.csv",float_precision="round_trip")
    with np.load(root/"example_labels.npz",allow_pickle=False) as q:labels=q["labels"]
    summary=aggregate(pose,labels)
    assert np.allclose(np.stack([summary[k] for k in gold["row_order"]]),features,atol=1e-10,rtol=0)
