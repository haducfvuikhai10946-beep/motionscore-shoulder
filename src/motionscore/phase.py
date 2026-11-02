"""Paper BiLSTM-CRF architecture and unmasked Viterbi decoder."""
from __future__ import annotations

from pathlib import Path
import numpy as np
import torch
from torch import nn
from torchcrf import CRF


class PhaseModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.bilstm_1 = nn.LSTM(4,128,batch_first=True,bidirectional=True)
        self.bilstm_2 = nn.LSTM(256,128,batch_first=True,bidirectional=True)
        self.dropout = nn.Dropout(.2)
        self.emission = nn.Linear(256,4)
        self.crf = CRF(4,batch_first=True)

    def forward(self,x):
        h,_ = self.bilstm_1(x); h = self.dropout(h)
        h,_ = self.bilstm_2(h); h = self.dropout(h)
        return self.emission(h)

    def loss(self,emissions,y):
        e,y = emissions.cpu(),y.cpu()
        mask = (y>=0).clone(); mask[:,0] = True
        return -self.crf(e,y.clamp(min=0),mask=mask,reduction="mean")


def decode(emissions,model):
    transitions = model.crf.transitions.detach().cpu().numpy()
    score = emissions[0]+model.crf.start_transitions.detach().cpu().numpy()
    backs = []
    for row in emissions[1:]:
        q = score[:,None]+transitions
        backs.append(q.argmax(0)); score = q.max(0)+row
    k = int((score+model.crf.end_transitions.detach().cpu().numpy()).argmax())
    labels = [k]
    for b in reversed(backs):
        k = int(b[k]); labels.append(k)
    return np.array(labels[::-1],int)


def load_model(checkpoint: str | Path,device="cpu"):
    model = PhaseModel()
    model.load_state_dict(torch.load(checkpoint,map_location="cpu",weights_only=True))
    model.to(device); model.crf.cpu(); model.eval()
    return model


def predict(x,checkpoint: str | Path,device="cpu"):
    torch.set_num_threads(2)
    model = load_model(checkpoint,device)
    with torch.inference_mode():
        emissions = model(torch.as_tensor(x[None],device=device))[0].cpu().numpy()
    return decode(emissions,model)
