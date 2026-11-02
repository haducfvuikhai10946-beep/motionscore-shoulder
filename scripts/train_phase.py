"""Retrain the original pseudo-supervised frontend; manual labels remain evaluation-only."""
import argparse,json,random
from pathlib import Path
import numpy as np
import torch
from motionscore.phase import PhaseModel


def main():
    p=argparse.ArgumentParser();p.add_argument("--data",type=Path,required=True);p.add_argument("--output",type=Path,required=True)
    p.add_argument("--fold",type=int,choices=range(5));p.add_argument("--device",default="cpu");args=p.parse_args()
    cfg=json.loads((Path(__file__).resolve().parents[1]/"configs/paper_protocol.json").read_text())
    if args.fold is None:
        ids=json.loads((args.data/"training_order.json").read_text());seed=cfg["phase_seed_full"]
    else:
        ids=json.loads((args.data/"folds.json").read_text())[args.fold]["train"];seed=20261040+args.fold
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed);torch.set_num_threads(2)
    data={}
    for sid in ids:
        with np.load(args.data/"training"/f"{sid}.npz",allow_pickle=False) as stored:data[sid]=(stored["X"],stored["y"])
    assert len({len(x[0]) for x in data.values()})==1
    model=PhaseModel().to(args.device);model.crf.cpu();optimizer=torch.optim.Adam(model.parameters(),lr=cfg["phase_lr"])
    losses=[]
    for epoch in range(cfg["phase_epochs"]):
        model.train();epoch_seed=seed+epoch if args.fold is None else 20261040+args.fold*100+epoch
        order=np.random.default_rng(epoch_seed).permutation(len(ids));current=[]
        for start in range(0,len(order),cfg["phase_batch"]):
            batch=[ids[j] for j in order[start:start+cfg["phase_batch"]]]
            x=torch.tensor(np.stack([data[s][0] for s in batch]),device=args.device)
            y=torch.tensor(np.stack([data[s][1] for s in batch]),dtype=torch.long)
            optimizer.zero_grad(set_to_none=True);loss=model.loss(model(x),y)
            if not torch.isfinite(loss):raise RuntimeError("Nonfinite training loss")
            loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),cfg["phase_clip"]);optimizer.step();current.append(float(loss.detach()))
        losses.append(float(np.mean(current)));print("Epoch",epoch+1,"loss",losses[-1],flush=True)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    torch.save({k:v.detach().cpu() for k,v in model.state_dict().items()},args.output)
    args.output.with_suffix(".json").write_text(json.dumps({"seed":seed,"subjects":len(ids),"losses":losses,"device":args.device,
        "manual_labels_used":False,"comparison_subjects_used":0,"bitwise_reproduction":"Not guaranteed across CUDA/PyTorch/device versions; frozen checkpoints are the numerical reference."},indent=2))

if __name__=="__main__":main()
