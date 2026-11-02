"""Retrain fixed mapping heads on published fold-specific features, never the 74."""
import argparse,json
from pathlib import Path
import numpy as np
import pandas as pd
from motionscore.features import NAMES
from motionscore.scoring import MotionScorer


def main():
    p=argparse.ArgumentParser();p.add_argument("--data",type=Path,required=True);p.add_argument("--models",type=Path,required=True)
    p.add_argument("--output",type=Path,default=Path("results/retrained"))
    p.add_argument("--backend",choices=["Ridge","ElasticNet","LightGBM"],default="Ridge");args=p.parse_args()
    config=json.loads((Path(__file__).resolve().parents[1]/"configs/paper_protocol.json").read_text())
    folds=json.loads((args.data/"folds.json").read_text());order=json.loads((args.data/"training_order.json").read_text())
    report=[]
    for fold in [None]+folds:
        scope="full" if fold is None else f"fold{fold['fold']}";train=order if fold is None else fold["train"]
        filename="full242" if fold is None else f"outer{fold['fold']}"
        source=pd.read_csv(args.data/"features"/f"{filename}.csv",float_precision="round_trip")
        source=source[source.shift_frames==0]
        arrays={r:source[source.repeat==r].set_index("participant_id").loc[train,NAMES].to_numpy(float) for r in ["full","odd","even"]}
        scorer=MotionScorer(args.models/scope,args.backend);a,b=[scorer.standardize(arrays[r]) for r in ["odd","even"]]
        valid=np.r_[np.isfinite(arrays["even"]).all(1),np.isfinite(arrays["odd"]).all(1)]
        x,y=np.r_[a,b][valid],np.r_[b,a][valid]
        cfg=config["mapping_configs"][args.backend]
        if args.backend=="Ridge":
            from sklearn.linear_model import Ridge
            head=Ridge(**cfg).fit(x,y)
        elif args.backend=="ElasticNet":
            from sklearn.linear_model import MultiTaskElasticNet
            head=MultiTaskElasticNet(**cfg,selection="cyclic").fit(x,y)
        else:
            from sklearn.multioutput import MultiOutputRegressor
            from lightgbm import LGBMRegressor
            seed=scorer.ref["backend_seeds"]["LightGBM"]
            head=MultiOutputRegressor(LGBMRegressor(**cfg,random_state=seed),n_jobs=1).fit(x,y)
        testids=order if fold is None else fold["test"]
        test=source[(source.repeat=="full")].set_index("participant_id").loc[testids,NAMES].to_numpy(float)
        new=head.predict(scorer.standardize(test));expected=scorer.predict(test)
        error=float(np.max(abs(new-expected)));assert error<1e-8,(scope,error)
        output=args.output/scope/args.backend;output.mkdir(parents=True,exist_ok=True)
        if args.backend=="LightGBM":
            for j,m in enumerate(head.estimators_):m.booster_.save_model(str(output/f"F{j+1:02}.txt"))
        else:(output/"affine.json").write_text(json.dumps({"coef":head.coef_.tolist(),"intercept":head.intercept_.tolist()},indent=2))
        report.append({"scope":scope,"backend":args.backend,"training_subjects":len(train),"prediction_max_abs_error":error,"comparison_subjects_used":0})
        print(report[-1],flush=True)
    (args.output/f"{args.backend}_verification.json").write_text(json.dumps(report,indent=2)+"\n",encoding="utf-8")

if __name__=="__main__":main()
