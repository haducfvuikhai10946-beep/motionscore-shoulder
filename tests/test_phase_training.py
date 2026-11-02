import numpy as np
import pytest


def test_phase_training_gradient_and_viterbi():
    torch=pytest.importorskip("torch")
    pytest.importorskip("torchcrf")
    from motionscore.phase import PhaseModel,decode
    torch.set_num_threads(2);torch.manual_seed(123)
    model=PhaseModel();model.train()
    x=torch.randn(2,12,4);y=torch.tensor([[0,0,1,1,1,2,2,3,3,0,0,0],[0,0,-1,1,1,2,2,3,3,0,0,0]])
    optimizer=torch.optim.Adam(model.parameters(),lr=.001)
    loss=model.loss(model(x),y);assert torch.isfinite(loss)
    optimizer.zero_grad();loss.backward()
    assert model.bilstm_1.weight_ih_l0.grad.abs().sum()>0
    before=model.emission.weight.detach().clone();optimizer.step()
    assert not torch.equal(before,model.emission.weight)
    model.eval()
    with torch.inference_mode():emissions=model(x)[0].numpy()
    labels=decode(emissions,model)
    assert labels.shape==(12,) and np.isin(labels,[0,1,2,3]).all()
