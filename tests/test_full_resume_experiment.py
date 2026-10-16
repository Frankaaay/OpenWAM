"""GPU experiment for the actual OpenWAM full-state save/load utilities."""
import json, os, random
from pathlib import Path
import numpy as np
import torch
from accelerate import Accelerator
from openwam.train.utils.checkpointing import save_full_state, load_full_state, find_latest_accel_state

def main():
    root=Path(os.environ['RESUME_TEST_OUTPUT']);root.mkdir(parents=True,exist_ok=True)
    random.seed(42);np.random.seed(42);torch.manual_seed(42)
    accelerator=Accelerator(mixed_precision='bf16')
    model=torch.nn.Sequential(torch.nn.Linear(4,32),torch.nn.Dropout(.2),torch.nn.Linear(32,1))
    optim=torch.optim.AdamW(model.parameters(),lr=.001)
    scheduler=torch.optim.lr_scheduler.LambdaLR(optim,lambda step:min((step+1)/5,1))
    model,optim,scheduler=accelerator.prepare(model,optim,scheduler)
    def update():
        x=torch.randn(2,4,device=accelerator.device)
        y=torch.randn(2,1,device=accelerator.device)+random.random()+float(np.random.rand())
        loss=(model(x).float()-y).square().mean()
        accelerator.backward(loss);optim.step();scheduler.step();optim.zero_grad()
        return x.cpu(),loss.detach().cpu()
    for _ in range(2):update()
    save_full_state(accelerator,str(root),2,2,0)
    expected=[update() for _ in range(4)]
    weights={k:v.clone().cpu() for k,v in accelerator.unwrap_model(model).state_dict().items()}
    import copy
    optimizer=copy.deepcopy(optim.state_dict());lr=copy.deepcopy(scheduler.state_dict())
    meta=load_full_state(accelerator,find_latest_accel_state(str(root)))
    assert meta==dict(global_step=2,opt_step=2,epoch=0)
    actual=[update() for _ in range(4)]
    def equal(x,y):
        if isinstance(x,torch.Tensor):assert torch.equal(x.cpu(),y.cpu())
        elif isinstance(x,dict):
            assert x.keys()==y.keys()
            for k in x:equal(x[k],y[k])
        elif isinstance(x,(list,tuple)):
            assert len(x)==len(y)
            for a,b in zip(x,y):equal(a,b)
        else:assert x==y
    equal(expected,actual);equal(weights,accelerator.unwrap_model(model).state_dict());equal(optimizer,optim.state_dict());equal(lr,scheduler.state_dict())
    incomplete=root/'accel_state_step_99';incomplete.mkdir()
    assert find_latest_accel_state(str(root)).endswith('accel_state_step_2')
    try:load_full_state(accelerator,str(incomplete))
    except FileNotFoundError:pass
    else:raise AssertionError('Incomplete checkpoint accepted')
    result=dict(passed=True,device=str(accelerator.device),mixed_precision='bf16',save_step=2,resumed_updates=4,exact=['rng','loss','weights','adamw','lr','progress'],incomplete_rejected=True,scope='small model; actual Accelerate checkpoint methods; not distributed ZeRO')
    (root/'result.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))
if __name__=='__main__':main()