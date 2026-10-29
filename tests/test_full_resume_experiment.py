"""GPU experiment for the actual OpenWAM full-state save/load utilities."""
import json, os, random
from pathlib import Path
import numpy as np
import torch
from accelerate import Accelerator, DeepSpeedPlugin
from openwam.train.utils.checkpointing import save_full_state, load_full_state, find_latest_accel_state

def main():
    root=Path(os.environ['RESUME_TEST_OUTPUT']);root.mkdir(parents=True,exist_ok=True)
    random.seed(42);np.random.seed(42);torch.manual_seed(42)
    use_ds=os.environ.get('RESUME_TEST_DEEPSPEED')=='1'
    plugin=DeepSpeedPlugin(zero_stage=2, gradient_accumulation_steps=1, offload_optimizer_device='cpu') if use_ds else None
    if plugin: plugin.deepspeed_config['train_micro_batch_size_per_gpu']=2
    accelerator=Accelerator(mixed_precision='bf16',deepspeed_plugin=plugin)
    model=torch.nn.Sequential(torch.nn.Linear(4,32),torch.nn.Dropout(.2),torch.nn.Linear(32,1))
    optim=torch.optim.AdamW(model.parameters(),lr=.001)
    scheduler=torch.optim.lr_scheduler.LambdaLR(optim,lambda step:min((step+1)/5,1))
    model,optim,scheduler=accelerator.prepare(model,optim,scheduler)
    def update():
        x=torch.randn(2,4,device=accelerator.device)
        y=torch.randn(2,1,device=accelerator.device)+random.random()+float(np.random.rand())
        loss=(model(x.to(next(model.parameters()).dtype)).float()-y).square().mean()
        accelerator.backward(loss);optim.step();scheduler.step();optim.zero_grad()
        return x.cpu(),loss.detach().cpu()
    import io
    def snapshot(value):
        buffer=io.BytesIO();torch.save(value,buffer);buffer.seek(0)
        return torch.load(buffer,weights_only=False)
    for _ in range(2):update()
    saved_optimizer=snapshot(optim.state_dict())
    save_full_state(accelerator,str(root),2,2,0)
    expected=[update() for _ in range(4)]
    weights={k:v.clone().cpu() for k,v in accelerator.unwrap_model(model).state_dict().items()}
    import copy, io
    buffer=io.BytesIO();torch.save(optim.state_dict(),buffer);buffer.seek(0)
    optimizer=torch.load(buffer,weights_only=False);lr=copy.deepcopy(scheduler.state_dict())
    meta=load_full_state(accelerator,find_latest_accel_state(str(root)))
    assert meta==dict(global_step=2,opt_step=2,epoch=0)
    restored_optimizer=snapshot(optim.state_dict())
    actual=[update() for _ in range(4)]
    differences=[]
    def equal(x,y,path="root"):
        if isinstance(x,torch.Tensor):
            if not torch.equal(x.cpu(),y.cpu()):
                a=x.detach().cpu();b=y.detach().cpu();different=(a!=b).flatten().nonzero().flatten()
                differences.append(dict(path=path,shape=list(a.shape),count=len(different),indices=different[:20].tolist(),max_abs=float((a-b).abs().max())))
        elif isinstance(x,dict):
            assert x.keys()==y.keys()
            for k in x:equal(x[k],y[k],path+"/"+str(k))
        elif isinstance(x,(list,tuple)):
            assert len(x)==len(y)
            for i,(a,b) in enumerate(zip(x,y)):equal(a,b,path+"/"+str(i))
        elif hasattr(x, "__dict__"):equal(vars(x),vars(y),path)
        elif x!=y:differences.append(dict(path=path,expected=str(x),actual=str(y)))
    equal(saved_optimizer,restored_optimizer,"restored_optimizer");equal(expected,actual,"batches_losses");equal(weights,accelerator.unwrap_model(model).state_dict(),"weights");equal(optimizer,optim.state_dict(),"optimizer");equal(lr,scheduler.state_dict(),"lr")
    # CPUAdam computes beta powers with pow() on a resumed step, versus
    # repeated FP32 multiplication in uninterrupted execution (cpu_adam.h).
    # Checkpoint restoration, moments, LR, loss and bf16 weights stay exact;
    # only subsequent FP32 master updates may differ by <= two FP32 eps.
    unexpected=[d for d in differences if not (
        d['path'].startswith('optimizer/single_partition_of_fp32_groups/')
        and d.get('max_abs',float('inf')) <= 2*torch.finfo(torch.float32).eps)]
    if unexpected:
        torch.save(dict(expected=optimizer,actual=optim.state_dict()),root/f"optimizer-diff-rank{accelerator.process_index}.pt")
        (root/f"differences-rank{accelerator.process_index}.json").write_text(json.dumps(differences,indent=2))
        raise AssertionError(json.dumps(differences))
    incomplete=root/'accel_state_step_99';
    if accelerator.is_main_process: incomplete.mkdir()
    accelerator.wait_for_everyone()
    assert find_latest_accel_state(str(root)).endswith('accel_state_step_2')
    try:load_full_state(accelerator,str(incomplete))
    except FileNotFoundError:pass
    else:raise AssertionError('Incomplete checkpoint accepted')
    result=dict(passed=True,device=str(accelerator.device),mixed_precision='bf16',save_step=2,resumed_updates=4,exact=['rng','loss','bf16_weights','adam_moments','lr','progress','checkpoint_optimizer'],incomplete_rejected=True,fp32_master_roundoff=differences,checkpoint_optimizer_exact=True,world_size=accelerator.num_processes,zero_stage=2 if use_ds else None,scope='small model; actual Accelerate checkpoint methods')
    if accelerator.is_main_process:
        (root/'result.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))
if __name__=='__main__':main()


class _ResumeCursorDataset(torch.utils.data.Dataset):
    reads = 0
    def __len__(self):
        return 1024

    def __getitem__(self, index):
        type(self).reads += 1
        return index, random.random()


def test_upstream_skip_preserves_cursor_without_replaying_augmentation():
    """Real Accelerate sharding; cursor equivalence, not augmentation equivalence."""
    from accelerate import skip_first_batches
    from accelerate.data_loader import prepare_data_loader

    for workers in (0, 2):
        for world in (1, 8):
            for rank in range(world):
                for epoch in (0, 3):
                    def loader():
                        generator = torch.Generator().manual_seed(42 + epoch)
                        raw = torch.utils.data.DataLoader(
                            _ResumeCursorDataset(), batch_size=4, shuffle=True,
                            generator=generator, num_workers=workers,
                        )
                        prepared = prepare_data_loader(
                            raw, device=torch.device('cpu'), num_processes=world,
                            process_index=rank, rng_types=[], put_on_device=False,
                        )
                        prepared.set_epoch(epoch)
                        return prepared

                    random.seed(42)
                    full = list(loader())
                    random.seed(42)
                    _ResumeCursorDataset.reads = 0
                    resumed = list(skip_first_batches(loader(), 5))
                    assert len(resumed) == len(full) - 5
                    if workers == 0:
                        assert _ResumeCursorDataset.reads == sum(len(b[0]) for b in resumed)
                    for expected, actual in zip(full[5:], resumed, strict=True):
                        assert torch.equal(expected[0], actual[0])
                    # Worker RNG was not saved in legacy checkpoints. Upstream
                    # skipping must not be advertised as bitwise augmentation resume.
                    assert any(not torch.equal(a[1], b[1])
                               for a, b in zip(full[5:], resumed, strict=True))
