import json
import numpy as np
import pandas as pd
import pytest
from motionscore.features import NAMES,retained_cycles,validate_pose,JOINTS
from motionscore.scoring import MotionScorer


def test_cycle_retention_and_illegal_transitions():
    phases=np.repeat([0,1,2,3,0,1,2,3,0,1,2,3,0],20)
    cuts=retained_cycles(phases)
    assert len(cuts)==2
    assert np.array_equal(cuts[0],[20,40,60,80,100])
    # Fragmenting one phase cannot silently create a valid canonical cycle.
    phases[45:50]=0
    assert len(retained_cycles(phases))==1


def test_score_anchors_and_budget_decomposition(tmp_path):
    anchors=["reference"]*13;anchors[1]="zero"
    ref={"features":NAMES,"median":[2.]*13,"iqr":[1.]*13,"anchors":anchors,
         "blocks":{"all":{"indices":list(range(13)),"weights":[1/13]*13,"budget":1.}},
         "thresholds":[1.,2.],"reference_scores":{"Identity":[20.,50.,80.]}}
    (tmp_path/"reference.json").write_text(json.dumps(ref));scorer=MotionScorer(tmp_path,"Identity")
    x=np.full((1,13),2.);x[0,1]=0
    result=scorer.score(x)
    assert result["MotionScore"][0]==100.
    x[0,0]=3.
    result=scorer.score(x)
    assert np.isclose(result["D"][0],1/13)
    assert np.allclose(sum(result["contributions"].values()),result["D"])
    assert result["percentile"][0]==100.
    result=scorer.score(np.full((1,13),10000.))
    assert result["MotionScore"][0]>=100/7-1e-12


def test_reject_uncontrolled_fps():
    df=pd.DataFrame({"timestamp_sec":np.arange(20)/25})
    for j in JOINTS:
        for a in "xy":df[j+"_"+a]=1.
    with pytest.raises(ValueError,match="20 fps"):validate_pose(df)


def test_unsupported_backend(tmp_path):
    (tmp_path/"reference.json").write_text(json.dumps({"features":NAMES}))
    with pytest.raises(ValueError,match="Supported backends"):MotionScorer(tmp_path,"Unknown")
