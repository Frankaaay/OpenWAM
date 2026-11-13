"""Train one v3 real-robot task from an independent midtrain checkpoint."""

import argparse
import hashlib
import json
import logging
import math
import os
import pathlib
import shutil

S = pathlib.Path(__file__).resolve().parents[1]


def main():
    import torch
    from accelerate.utils import set_seed
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf

    import openwam.train.openwam_trainer as tm
    from openwam.dataloader.realrobot_v3 import V3Dataset
    from scripts.train import _build_accelerator

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=["wan", "stereo"], required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--data", type=pathlib.Path, required=True)
    parser.add_argument("--prepared", type=pathlib.Path, required=True)
    parser.add_argument("--checkpoint", type=pathlib.Path, required=True)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--preflight", action="store_true")
    args = parser.parse_args()
    model = args.model
    task = args.task
    smoke = args.smoke
    PREP = args.prepared
    data = args.data
    base = args.output
    stats = PREP / f"{task}-stats.json"
    ckpt = args.checkpoint.resolve()
    assert ckpt.name == "checkpoint_step_50101.safetensors" and ckpt.is_file()
    assert not base.exists(), "Use a new output directory"
    assert json.load(open(PREP / "result.json"))["status"] == "passed"
    # The encoder is restored from the self-contained checkpoint. This variable
    # only resolves the archived config before the live model_path=None override.
    os.environ.update(
        MIDTRAIN_ALPHA_ROOT=str(ckpt),
        MIDTRAIN_OUTPUT=str(base),
        MIDTRAIN_STEREOTOK=str(ckpt),
        WANDB_MODE="disabled" if smoke else "online",
        TOKENIZERS_PARALLELISM="false",
    )
    real = V3Dataset(task, data, PREP, "stereo" if model == "stereo" else "mono-left")
    expected_stats = json.load(open(stats))
    assert (len(real.rows), len(real)) == (expected_stats["episodes"], expected_stats["windows"])
    per_epoch = math.ceil(len(real) / 32)
    save_at = {i * per_epoch for i in range(1, 11)}
    retained_at = set(save_at)
    with initialize_config_dir(version_base=None, config_dir=str(S / "configs")):
        cfg = compose(
            config_name="midtrain_droid_abc_alpha_wan",
            overrides=[
                "training.initialize_model_on_cpu=true",
                "training.validation=null",
                "training.num_epochs=10",
                "training.max_steps=null",
                "training.batch_size=4",
                "training.gradient_accumulation_steps=1",
                "training.offload_optimizer_device=none",
                "training.save_full_states_for_resume=true",
                f"training.save_steps={per_epoch}",
                "training.keep_last_k_ckpts=10",
                "training.learning_rate=3e-5",
                "training.action_lr=3e-5",
                "training.video_lr=3e-5",
                "training.warmup_steps=100",
                "training.warmup_start_factor=0.1",
                "training.dataset_num_workers=2",
                f"project.wandb.run_name=OpenWAM Cart007 v3 {model} {task} midtrain50101 H1008 "
                + ("smoke" if smoke else "10epoch"),
            ],
        )
    if model == "stereo":
        cfg.model.video_backbone.encoder = OmegaConf.create(
            dict(name="stereotok", model_path=None, input_mode="stereo")
        )
    cfg.dataloader = OmegaConf.create(
        dict(
            datasets=dict(
                realrobot=dict(
                    name="realrobot_recorded_pose",
                    dataset_dir=str(data),
                    stats_path=str(stats),
                    input_mode="stereo" if model == "stereo" else "mono-left",
                )
            )
        )
    )
    if smoke:
        cfg.training.max_steps = 10
        cfg.training.save_full_states_for_resume = False
        cfg.training.save_steps = 10
        save_at = {10}
        retained_at = {10}
        cfg.project.wandb = None
    contract = dict(
        model=model,
        task=task,
        smoke=smoke,
        episodes=len(real.rows),
        windows=len(real),
        epochs=0 if smoke else 10,
        world_size=8,
        global_batch=32,
        updates_per_epoch=per_epoch,
        total_updates=10 if smoke else 10 * per_epoch,
        saved_epochs=[] if smoke else list(range(1, 11)),
        retention="keep_all_no_deletion",
        saved_steps=sorted(save_at),
        final_retained_steps=sorted(retained_at),
        full_state=not smoke,
        wandb_mode="disabled" if smoke else "online",
        independent_init=str(ckpt),
        stats_sha256=hashlib.sha256(stats.read_bytes()).hexdigest(),
        info_sha256=hashlib.sha256((data / "meta/info.json").read_bytes()).hexdigest(),
        fused_adamw=True,
        prefetch_factor=2,
        persistent_workers=True,
        zero_stage=2,
    )
    OmegaConf.to_container(cfg, resolve=True)
    if args.preflight:
        print("FORMAL_CPU_PREFLIGHT", json.dumps(contract), flush=True)
        return
    logging.basicConfig(level=logging.INFO)
    set_seed(42)
    acc = _build_accelerator(cfg)
    assert acc.num_processes == 8
    print("DEVICE", acc.process_index, torch.cuda.get_device_name(acc.device), flush=True)
    original_weights = tm.save_weights
    original_state = tm.save_full_state

    def weights(a, architecture, output, step, *, final):
        if step not in save_at:
            return
        if (pathlib.Path(output) / f"checkpoint_step_{step}.safetensors").exists():
            return
        original_weights(a, architecture, output, step, final=final)

    def state(a, output, step, opt_step, epoch):
        if step in save_at:
            original_state(a, output, step, opt_step, epoch)

    tm.save_weights = weights
    tm.save_full_state = state

    class Trainer(tm.OpenWAMTrainer):
        def finish_training(self, output_path, global_step, save_steps, keep_last_k, is_main, wandb_run):
            # Preserve all outputs; this entrypoint never deletes checkpoints or state.
            if save_steps and not (pathlib.Path(output_path) / f"checkpoint_step_{global_step}.safetensors").exists():
                weights(self.accelerator, self.architecture, output_path, global_step, final=True)
            self.accelerator.wait_for_everyone()
            if wandb_run is not None:
                wandb_run.finish()

        def build_optimizer(self):
            opt = super().build_optimizer()
            for group in opt.param_groups:
                group["fused"] = True
                group["foreach"] = False
            opt.defaults.update(fused=True, foreach=False)
            return opt

        def build_dataloader(self, batch_size):
            from openwam.train.utils.seeding import dataloader_worker_init_fn, make_dataloader_generator

            return torch.utils.data.DataLoader(
                self.dataset,
                batch_size=batch_size,
                shuffle=True,
                num_workers=2,
                collate_fn=list,
                pin_memory=True,
                multiprocessing_context="spawn",
                persistent_workers=True,
                prefetch_factor=2,
                generator=make_dataloader_generator(self._run_seed, rank=self._rank),
                worker_init_fn=dataloader_worker_init_fn,
            )

        def setup_output_dir(self, debug, resume_path):
            out, res = super().setup_output_dir(debug, resume_path)
            self.actual_output = out
            if self.accelerator.is_main_process:
                if not res:
                    shutil.copyfile(stats, pathlib.Path(out) / "realrobot_stats.json")
                    (pathlib.Path(out) / "data-contract.json").write_text(json.dumps(contract, indent=2))
                    shutil.copyfile(__file__, pathlib.Path(out) / "training-entrypoint.py")
                (base / "active-output.json").write_text(json.dumps(dict(output=out, task=task)))
            self.accelerator.wait_for_everyone()
            return out, res

        def prepare_accelerate(self, optimizer, dataloader, scheduler):
            result = super().prepare_accelerate(optimizer, dataloader, scheduler)
            assert len(result[1]) == per_epoch
            assert self.architecture.train_batch_size() == 32 and not self.architecture.optimizer.cpu_offload
            return result

        def compute_loss(self, batch):
            losses = super().compute_loss(batch)
            assert all(torch.isfinite(v).all() for v in losses.values()), "Nonfinite loss"
            return losses

        def log_step(self, **kw):
            if not smoke and self.accelerator.is_main_process:
                assert kw["wandb_run"] is not None and kw["wandb_run"].settings.mode == "online"
            super().log_step(**kw)
            if self.accelerator.is_main_process:
                self.last_opt_step = kw["opt_step"]
                row = dict(
                    step=kw["global_step"],
                    opt_step=kw["opt_step"],
                    epoch=kw["epoch"] + 1,
                    lr=kw["lr"],
                    steps_per_sec=kw["steps_per_sec"],
                    **kw["metrics"],
                )
                with (pathlib.Path(kw["output_path"]) / "metrics.jsonl").open("a") as f:
                    f.write(json.dumps(row) + "\n")
                if kw["global_step"] % 50 == 0 or kw["global_step"] <= 10:
                    print("TRAIN_PROGRESS", json.dumps(row), flush=True)

    trainer = Trainer(cfg, accelerator=acc, dataset=real)
    trainer.train()
    assert trainer.architecture.global_steps == (10 if smoke else per_epoch * 10)
    if acc.is_main_process:
        out = pathlib.Path(trainer.actual_output)
        assert {int(x.stem.split("_")[-1]) for x in out.glob("checkpoint_step_*.safetensors")} == retained_at
        for step in [] if smoke else retained_at:
            meta = json.load(open(out / f"accel_state_step_{step}/trainer_state.json"))
            assert meta["global_step"] == step and meta["opt_step"] == step
        assert trainer.last_opt_step == (10 if smoke else per_epoch * 10)
        result = dict(
            status="complete",
            model=model,
            task=task,
            output=str(out),
            saved_steps=sorted(retained_at),
            epochs=0 if smoke else 10,
            optimizer_steps=10 if smoke else per_epoch * 10,
            smoke=smoke,
        )
        (base / "result.json").write_text(json.dumps(result, indent=2))
        print("TASK_TRAINING_COMPLETE", json.dumps(result), flush=True)
    acc.wait_for_everyone()
    acc.end_training()


if __name__ == "__main__":
    main()
