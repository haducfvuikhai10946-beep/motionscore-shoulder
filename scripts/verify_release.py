"""Check independent historical goldens, feature extraction, OOF metrics and phases."""
import argparse,hashlib,json,sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from motionscore.features import NAMES,aggregate,frame_features
from motionscore.scoring import MotionScorer


def icc(a,b):
    x=np.column_stack([a,b]);n,k=x.shape;mean=x.mean();row=x.mean(1);col=x.mean(0)
    msr=k*np.sum((row-mean)**2)/(n-1);msc=n*np.sum((col-mean)**2)/(k-1)
    mse=np.sum((x-row[:,None]-col[None,:]+mean)**2)/((n-1)*(k-1))
    return (msr-mse)/(msr+(k-1)*mse+k*(msc-mse)/n)


def main():
    p=argparse.ArgumentParser();p.add_argument("--data",type=Path,required=True)
    p.add_argument("--models",type=Path,required=True);p.add_argument("--phases",action="store_true")
    p.add_argument("--output",type=Path,default=Path("results/verification.json"));args=p.parse_args()
    people=pd.read_csv(args.data/"participants.csv");folds=json.loads((args.data/"folds.json").read_text())
    dev=set(people[people.split=="development"].participant_id);held=set(people[people.split=="comparison"].participant_id)
    assert len(dev)==242 and len(held)==74 and not dev&held
    assert set().union(*(set(f["test"]) for f in folds))==dev
    for f in folds:
        assert set(f["train"])|set(f["test"])==dev and not set(f["train"])&set(f["test"])
    errors=[];metrics=[];oof={name:[] for name in ["Identity","Ridge","ElasticNet","LightGBM"]}
    for scope in ["full"]+[f"fold{f['fold']}" for f in folds]:
        expected=pd.read_csv(args.data/f"expected_{scope}.csv",float_precision="round_trip")
        tables={s:pd.read_csv(args.data/"features"/f"{s}.csv",float_precision="round_trip") for s in expected.scope.unique()}
        for name in oof:
            scorer=MotionScorer(args.models/scope,name)
            for s,df in tables.items():
                gold=expected[(expected.scope==s)&(expected.backend==name)].set_index(["participant_id","repeat"])
                source=df[df.shift_frames==0].set_index(["participant_id","repeat"]).loc[gold.index]
                result=scorer.score(source[NAMES].to_numpy(float))
                err=max(float(np.max(abs(result[k]-gold[k].to_numpy()))) for k in ["MotionScore","D"])
                assert err<1e-9,(scope,name,err)
                assert np.array_equal(result["band"],gold.band.to_numpy())
                for key,values in result["contributions"].items():
                    assert np.allclose(values,gold["part_"+key],atol=1e-12,rtol=0)
                errors.append({"scope":scope,"backend":name,"rows":len(source),"score_max_abs_error":err})
                if scope!="full":
                    odd=source.xs("odd",level="repeat")[NAMES];even=source.xs("even",level="repeat")[NAMES].loc[odd.index]
                    pa,pb=scorer.predict(odd),scorer.predict(even)
                    e=((pa-scorer.standardize(even))**2+(pb-scorer.standardize(odd))**2).mean(1)/2
                    sa,sb=scorer.score(odd)["MotionScore"],scorer.score(even)["MotionScore"]
                    oof[name].extend(zip(e.tolist(),sa.tolist(),sb.tolist()))
    for name,rows in oof.items():
        a=np.array(rows);assert len(a)==242
        metrics.append({"backend":name,"subjects":len(a),"MSE":float(a[:,0].mean()),
                        "ICC_A1":float(icc(a[:,1],a[:,2])),"Spearman":float(spearmanr(a[:,1],a[:,2]).statistic)})
    feature_error=0.;cycles=0;phase_inputs=[];phase_ids=[];phase_errors=0
    reference=pd.concat([pd.read_csv(args.data/"features"/f"{s}.csv",float_precision="round_trip") for s in ["full242","fixed74"]])
    reference=reference[reference.shift_frames==0].set_index(["participant_id","repeat"])
    for i,person in enumerate(people.itertuples(index=False)):
        sid=person.participant_id
        with np.load(args.data/"poses"/f"{sid}.npz",allow_pickle=False) as stored:
            df=pd.DataFrame(stored["values"],columns=stored["columns"])
        with np.load(args.data/"labels"/f"{sid}.npz",allow_pickle=False) as stored:labels=stored["labels"]
        summary=aggregate(df,labels);cycles+=len(summary["cuts"])
        err=max(float(np.max(abs(summary[k]-reference.loc[(sid,k),NAMES].to_numpy(float)))) for k in ["full","odd","even"])
        feature_error=max(feature_error,err);assert err<1e-9,(sid,err)
        if args.phases:
            phase_inputs.append(frame_features(df));phase_ids.append(sid)
        if (i+1)%50==0:print("Verified poses",i+1,"/316",flush=True)
    if args.phases:
        import torch
        from motionscore.phase import load_model,decode
        torch.set_num_threads(2);model=load_model(args.models/"full/phase.pt")
        for start in range(0,len(phase_ids),8):
            batch=phase_inputs[start:start+8]
            assert len({len(x) for x in batch})==1
            with torch.inference_mode():emissions=model(torch.tensor(np.stack(batch)) ).numpy()
            for sid,e in zip(phase_ids[start:start+8],emissions):
                predicted=decode(e,model)
                with np.load(args.data/"labels"/f"{sid}.npz",allow_pickle=False) as saved:old=saved["labels"]
                phase_errors+=int(np.sum(predicted!=old))
            print("Verified phase model",min(start+8,len(phase_ids)),"/316",flush=True)
        assert phase_errors==0,phase_errors
    result={"participants":len(people),"retained_cycles":cycles,"feature_max_abs_error":feature_error,
            "phase_model_checked":args.phases,"phase_label_disagreements":phase_errors if args.phases else None,
            "score_checks":errors,"out_of_fold_metrics":metrics,
            "limits":"Cached body poses and frozen four-backend paths; not all twelve benchmark backends, raw RGB equality or clinical validation."}
    args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(result,indent=2)+"\n",encoding="utf-8")
    print(json.dumps({k:v for k,v in result.items() if k!="score_checks"},indent=2))

if __name__=="__main__":main()
