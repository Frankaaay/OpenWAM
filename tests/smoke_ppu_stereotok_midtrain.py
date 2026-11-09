"""Bounded 8-GPU acceptance; never launches the formal 32-GPU experiment."""
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import types

# Running a file under tests/ must not shadow Orion's top-level dataloader package.
sys.path[0] = str(Path(__file__).resolve().parents[1])

import numpy as np
import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
from safetensors import safe_open

from scripts.train import _build_accelerator
from openwam.train.openwam_trainer import OpenWAMTrainer
from openwam.dataloader.orion_midtrain_ppu import ABCNativeCropOrionDataset, DROIDPPUOrionDataset
from openwam.dataloader.mixture import MixtureDataset
from openwam.train.utils.ckpt_model_loader import build_architecture_from_ckpt_dir, copy_ckpt_artifacts
from openwam.train.utils.checkpointing import save_config, save_weights, save_full_state, load_full_state

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = Path(os.environ['MIDTRAIN_STEREOTOK_OUTPUT'])
BASELINE = '1896bf723fac78d04269467e0c1519a4a98a8a92'


def datasets(cfg, split, module=None):
    d = cfg.dataloader.datasets
    dc = DROIDPPUOrionDataset if module is None else module.DROIDPPUOrionDataset
    ac = ABCNativeCropOrionDataset if module is None else module.ABCNativeCropOrionDataset
    return [dc.from_config(d.droid, split), ac.from_config(d.abc, split)]


def data_parity(cfg, baseline_cfg):
    old = types.ModuleType('frozen_wan_ppu_reader')
    source = subprocess.check_output(['git', '-C', str(ROOT), 'show',
                                     BASELINE + ':openwam/dataloader/orion_midtrain_ppu.py'], text=True)
    exec(compile(source, 'frozen_wan_ppu_reader.py', 'exec'), old.__dict__)
    checked = []
    for split in ('train', 'val'):
        before, after = datasets(baseline_cfg, split, old), datasets(cfg, split)
        for name, left, stereo in zip(('droid', 'abc'), before, after):
            assert len(left) == len(stereo)
            indices = {0, 1, 2, len(left)-1, len(left)//2}
            if name == 'abc':
                assert left.episodes == stereo.episodes
                for available in (False, True):
                    matching = [e for e in stereo.episodes if bool(e['availability/observation.images.cam_head_right']) == available]
                    for e in matching[:2]:
                        indices.update(e['split_window_start']+i for i in range(min(3, e['sparse_windows'])))
            for index in sorted(indices):
                a, b = left[index], stereo[index]
                assert a.keys() == b.keys()
                for key in a:
                    if key == 'right_video':continue
                    if key in ('video', 'first_frame_image'):
                        assert len(a[key]) == len(b[key])
                        assert all(np.array_equal(np.asarray(x), np.asarray(y)) for x,y in zip(a[key],b[key])), (name,split,index,key)
                    elif torch.is_tensor(a[key]):assert torch.equal(a[key], b[key]), (name,split,index,key)
                    else:assert a[key] == b[key], (name,split,index,key)
                assert b['right_video'] is not None and len(b['right_video']) == 17
                assert any(not np.array_equal(np.asarray(x),np.asarray(y)) for x,y in zip(b['video'],b['right_video']))
                checked.append([name,split,index])
    (OUTPUT/'data-parity.json').write_text(json.dumps({'passed':True,'baseline_sha':BASELINE,'windows':checked},indent=2))
    print('DATA_PARITY_PASSED',len(checked),flush=True)


def main():
    torch.set_num_threads(4)
    with initialize_config_dir(version_base=None, config_dir=str(ROOT/'configs')):
        baseline = compose(config_name='midtrain_droid_abc_alpha_wan_ppu')
        cfg = compose(config_name='midtrain_droid_abc_alpha_stereotok_ppu')
    a, b = OmegaConf.to_container(baseline,resolve=True), OmegaConf.to_container(cfg,resolve=True)
    # Exhaustive resolved-config delta, not a hand-picked optimizer checklist.
    def differences(x,y,path=''):
        if isinstance(x,dict) and isinstance(y,dict):
            result=[]
            for k in sorted(x.keys()|y.keys()):result+=differences(x.get(k),y.get(k),path+'.'+k if path else k)
            return result
        return [] if x==y else [path]
    changed=differences(a,b)
    allowed={'model.video_backbone.encoder','dataloader.datasets.droid.input_mode',
        'dataloader.datasets.abc.input_mode','dataloader.datasets.abc.right_video_root',
        'training.output_path','project.output_dir','project.wandb.run_name'}
    assert {p.replace('training.validation.dataloader.','dataloader.') for p in changed}<=allowed,changed
    cfg.training.gradient_accumulation_steps=8
    cfg.training.dataset_num_workers=0
    cfg.training.validation=None
    cfg.training.max_steps=24
    cfg.project.wandb.run_name='OpenWAM DROID+ABC Alpha StereoTok Stage3-u8000 stereo 8GPU smoke'
    acc=_build_accelerator(cfg)
    assert acc.num_processes==8
    if acc.is_main_process:
        OUTPUT.mkdir(parents=True,exist_ok=True)
        assert not (OUTPUT/'result.json').exists()
        (OUTPUT/'formal-config-delta.json').write_text(json.dumps(changed,indent=2))
        data_parity(cfg,baseline)
    acc.wait_for_everyone()
    train=datasets(cfg,'train');val=datasets(cfg,'val')
    mixture=MixtureDataset(train,names=['droid','abc'],seed=42,strict_action_dim=True)
    assert [len(x) for x in train]==[1667932,1538510] and len(mixture)==3206442
    subset=torch.utils.data.Subset(mixture,range(384))
    trainer=OpenWAMTrainer(cfg,accelerator=acc,dataset=mixture)
    arch=trainer.architecture
    codec=arch.video_backbone.video_encoder
    assert arch.video_backbone.vae is None
    assert not any(p.requires_grad for p in codec.parameters())
    assert not any(p.requires_grad for p in arch.video_backbone.text_encoder.parameters())
    if acc.is_main_process:
        state=arch.state_dict()
        # Compare the actual native Wan load path as well as the source weights:
        # the existing loader may cast stored tensors to the skeleton's dtype.
        with torch.random.fork_rng(devices=[]):
            _,wan_reference,_=build_architecture_from_ckpt_dir(
                os.environ['MIDTRAIN_ALPHA_ROOT'],weights_required=True,override_cfg=baseline)
        reference_state=wan_reference.state_dict()
        with safe_open(str(Path(os.environ['MIDTRAIN_ALPHA_ROOT'])/'checkpoint_step_154000.safetensors'),framework='pt',device='cpu') as source:
            expected=[k for k in source.keys() if not k.startswith('video_backbone.vae.')]
            actual=[k for k in state if not k.startswith('video_backbone.video_encoder.')]
            assert set(expected)==set(actual)
            for k in expected:
                ref=source.get_tensor(k)
                assert state[k].dtype==reference_state[k].dtype and torch.equal(state[k],reference_state[k]),('wan_vs_stereotok',k)
                assert torch.equal(state[k].cpu(),ref.to(state[k].dtype)),('source_after_native_cast',k)
                if k=='action_backbone.action_decoder.bias':
                    print('INIT_DTYPE',str(ref.dtype),str(state[k].dtype),'WAN_STEREOTOK_EXACT',flush=True)
        del reference_state,wan_reference
        print('ALPHA_NONCODEC_TENSORS_EXACT',len(expected),flush=True)
        arch.save_assets_for_deployment(str(OUTPUT),cfg)
        copy_ckpt_artifacts(os.environ['MIDTRAIN_ALPHA_ROOT'],str(OUTPUT))
        save_config(str(OUTPUT),cfg)
    acc.wait_for_everyone()
    optimizer=trainer.build_optimizer()
    scheduler=trainer.build_lr_scheduler(optimizer,25051,debug=False)
    loader=torch.utils.data.DataLoader(subset,batch_size=2,shuffle=False,collate_fn=list,num_workers=0)
    optimizer,loader,scheduler=trainer.prepare_accelerate(optimizer,loader,scheduler)
    engine=trainer.architecture
    assert engine.train_batch_size()==128 and engine.gradient_accumulation_steps()==8
    assert engine.zero_optimization_stage()==2 and not engine.optimizer.cpu_offload
    assert len(loader)==24
    def report(kind,step,losses,elapsed):
        values=torch.stack([losses[k].detach().float() for k in ('total','video','action')])
        assert torch.isfinite(values).all()
        values=acc.reduce(values,reduction='mean').cpu().tolist()
        peak=acc.gather(torch.tensor([torch.cuda.max_memory_allocated()/1e9],device=acc.device)).max().item()
        if acc.is_main_process:
            row=dict(kind=kind,update=step,total=values[0],video=values[1],action=values[2],seconds=elapsed,peak_allocated_gb=peak,lr=[g['lr'] for g in optimizer.param_groups])
            with (OUTPUT/'metrics.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
            print('STEREOTOK_SMOKE_METRIC',json.dumps(row),flush=True)
    engine.eval()
    with torch.no_grad():
        for name,ds in zip(('droid','abc'),val):
            t=time.monotonic();losses=trainer.compute_loss([ds[acc.process_index*len(ds)//8]])
            torch.cuda.synchronize();report('validation_start_'+name,0,losses,time.monotonic()-t)
    engine.train();window={};t=time.monotonic();counts={'droid':0,'abc':0}
    for micro,batch in enumerate(loader,1):
        for sample in batch:counts[sample['_dataset_name']]+=1
        with acc.accumulate(engine):
            losses=trainer.compute_loss(batch)
            assert all(torch.isfinite(v).all() for v in losses.values())
            for k,v in losses.items():window[k]=window.get(k,0)+v.detach()/8
            acc.backward(losses['total']);optimizer.step();scheduler.step();optimizer.zero_grad()
        if micro%8==0:
            torch.cuda.synchronize();assert engine.global_steps==micro//8
            report('train',micro//8,window,time.monotonic()-t);window={};t=time.monotonic()
        if micro==16:
            save_weights(acc,engine,str(OUTPUT),micro,final=False)
            save_full_state(acc,str(OUTPUT),micro,2,0)
            load_full_state(acc,str(OUTPUT/'accel_state_step_16'))
            assert engine.global_steps==2
            print('FULL_STATE_RELOADED',acc.process_index,flush=True)
    assert engine.global_steps==3
    assert not any(p.requires_grad or p.grad is not None for p in acc.unwrap_model(engine).video_backbone.video_encoder.parameters())
    engine.eval()
    with torch.no_grad():
        for name,ds in zip(('droid','abc'),val):
            t=time.monotonic();losses=trainer.compute_loss([ds[(acc.process_index+1)*len(ds)//9]])
            torch.cuda.synchronize();report('validation_end_'+name,3,losses,time.monotonic()-t)
    if acc.is_main_process:
        # Reconstruct the saved self-contained deployment including metric_head.
        _,restored,_=build_architecture_from_ckpt_dir(str(OUTPUT/'checkpoint_step_16.safetensors'),weights_required=True)
        assert hasattr(restored.video_backbone.video_encoder._m.model,'metric_head')
        current=acc.unwrap_model(engine).video_backbone.video_encoder.state_dict()
        for k,v in restored.video_backbone.video_encoder.state_dict().items():assert torch.equal(v.cpu(),current[k].cpu()),k
        del restored
        print('DEPLOY_CODEC_RELOAD_EXACT',flush=True)
    processed=acc.reduce(torch.tensor([counts['droid'],counts['abc']],device=acc.device),reduction='sum').cpu().tolist()
    assert sum(processed)==384 and min(processed)>0
    acc.wait_for_everyone()
    if acc.is_main_process:
        (OUTPUT/'result.json').write_text(json.dumps(dict(status='passed',world_size=8,global_batch=128,micro_steps=24,optimizer_steps=3,source_samples=processed,formal_training_started=False,limitations=['sampled data equality, not exhaustive video decoding','one-node smoke; no 32-GPU collective acceptance']),indent=2))
        print('STEREOTOK_STAGE3_PPU8_SMOKE_PASSED_STOPPED',flush=True)
    acc.end_training()


if __name__=='__main__':main()
