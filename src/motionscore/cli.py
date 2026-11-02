from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np
import pandas as pd
from .features import NAMES,aggregate,frame_features
from .scoring import MotionScorer


def clean(obj):
    if isinstance(obj,dict): return {k:clean(v) for k,v in obj.items()}
    if isinstance(obj,(np.ndarray,list,tuple)): return [clean(v) for v in obj]
    if isinstance(obj,np.generic): return obj.item()
    return obj


def main():
    parser = argparse.ArgumentParser(description="Fixed-protocol shoulder-raise MotionScore")
    sub = parser.add_subparsers(dest="command",required=True)
    for name in ["score-features","score-pose"]:
        p = sub.add_parser(name)
        p.add_argument("--input",type=Path,required=True)
        p.add_argument("--models",type=Path,required=True)
        p.add_argument("--backend",choices=["Identity","Ridge","ElasticNet","LightGBM"],default="LightGBM")
        p.add_argument("--output",type=Path,required=True)
        if name == "score-pose":
            p.add_argument("--labels",type=Path)
            p.add_argument("--device",default="cpu")
    args = parser.parse_args()
    scorer = MotionScorer(args.models,args.backend)
    df = pd.read_csv(args.input,float_precision="round_trip")
    if args.command == "score-features":
        result = scorer.score(df[NAMES].to_numpy(float))
    else:
        if args.labels:
            with np.load(args.labels,allow_pickle=False) as stored:
                labels = stored["labels"]
        else:
            from .phase import predict
            labels = predict(frame_features(df),args.models/"phase.pt",args.device)
        summary = aggregate(df,labels)
        result = scorer.score(np.stack([summary[k] for k in ["full","odd","even"]]))
        result.update(features={k:summary[k] for k in ["full","odd","even"]},
                      feature_order=NAMES,retained_cycles=summary["cuts"],
                      row_order=["full","odd","even"])
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(clean(result),indent=2,allow_nan=False)+"\n",encoding="utf-8")
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
