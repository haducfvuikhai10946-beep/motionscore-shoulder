"""Compare all held-out phase predictions with historical out-of-fold labels."""
import argparse,json
from pathlib import Path
import numpy as np
import torch
from motionscore.phase import load_model,decode


def main():
    p=argparse.ArgumentParser();p.add_argument("--data",type=Path,required=True);p.add_argument("--models",type=Path,required=True)
    p.add_argument("--output",type=Path,default=Path("results/phase_folds.json"))
    p.add_argument("--device",default="cpu");args=p.parse_args()
    torch.set_num_threads(2);report=[]
    folds=json.loads((args.data/"folds.json").read_text())
    for fold in folds:
        model=load_model(args.models/f"fold{fold['fold']}/phase.pt",args.device);ids=fold["test"];errors=0;frames=0
        # Historical predictions were made one recording at a time. Batched LSTM
        # kernels can change a near-tied Viterbi boundary by one frame.
        for start in range(0,len(ids)):
            batch=ids[start:start+1];xx=[]
            for sid in batch:
                with np.load(args.data/"training"/f"{sid}.npz",allow_pickle=False) as stored:xx.append(stored["X"])
            with torch.inference_mode():emissions=model(torch.tensor(np.stack(xx),device=args.device)).cpu().numpy()
            for sid,e in zip(batch,emissions):
                predicted=decode(e,model)
                with np.load(args.data/"oof_labels"/f"{sid}.npz",allow_pickle=False) as stored:gold=stored["labels"]
                errors+=int(np.sum(gold!=predicted));frames+=len(gold)
        report.append({"fold":fold["fold"],"device":args.device,"heldout_subjects":len(ids),"frames":frames,"label_disagreements":errors})
        print(report[-1],flush=True)
    args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(report,indent=2)+"\n",encoding="utf-8")
    assert sum(r["label_disagreements"] for r in report)==0,"See saved report for platform-dependent boundary differences"

if __name__=="__main__":main()
