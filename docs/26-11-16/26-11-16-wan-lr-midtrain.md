# DROID+ABC Alpha Wan-LR midtrain

User authorized dedicated branch/worktree, minimal adaptation, tests, commit/push and Frank-owned H100 clean clone. Current priority is an idle eight-GPU smoke, without booking changes.

Base peilin/wan-bino-baseline 50e0312020ab287f54658fb5a3e34e35b43d9ed8. New frank/openwam-wan-lr-midtrain. Preserve Alpha154000 Video/Action weights and the exact frozen Alpha VAE: strict-load native architecture first, then wrap its existing VAE in WanBinoVideoEncoder. No separately initialized policy projections or VAE weights. Existing StereoTok replacement remains unchanged.

Formal configuration inherits the original 3206442-window stride0.8 DROID+ABC one-epoch contract, changes input_mode to stereo and encoder to wan22_vae_bino.32 GPUs x batch2 x accumulation2 = global128; save5000 microsteps means2500 optimizer updates. Warmup and validation remain500 optimizer updates.

Bounded smoke uses8 GPUs x batch2 x accumulation8 = global128,80 microsteps/10 updates, real original datasets and normalization. Validation subset2/source and save80 are smoke-only overrides. Online W&B, independent output /gpfs/jiuquyun/projects/Frank/openwam-wan-lr-midtrain-20261116/smoke-JOBID, no requeue,2h maximum. Full training is not yet submitted. Results and actual JobID will be appended after execution.

## Eight-GPU evidence and formal queue preparation

Published smoke code ad5bb7c (following18ea27a). H100 GitHub clone was slow; transferred a bundle of the published branch into the same Frank target clone, then verified fetch/pull against GitHub. bash syntax and sbatch test-only passed. Job54802 remained pending with QOSGrpJobsLimit; cancelled before execution. Submitted54803 through the supported sbatch-idle wrapper; scheduler automatically classified it protected against the existing0050 booking. No booking records changed.

Job54803 on xn01-gpu1-0050 strict-loaded Alpha154000 successfully, initialized online W&B https://wandb.ai/yichx14-uc-irvine/StereoTok/runs/owlrsmoke54803 with the specified display name, and verified dataset counts1667932/1538510 on all ranks. Trainable parameters6021.2M. Six codec/right-eye tests passed including preserved parameter identity and values; seven midtrain contract tests passed. Initial separate test invocation lacked LD_LIBRARY_PATH/libGL, corrected only its test environment and reran successfully; production launcher already had the correct path.

Completed80 microsteps/10 optimizer updates. Final train loss2.2162 (video0.1441/action2.0721); finite losses throughout observed log. Start/end validation2 windows/source: overall0.937108 ->0.400736; DROID0.266829 ->0.177724; ABC1.607386 ->0.623748. These four windows establish functionality, not model quality. Stable short-step timings about1.0–1.2 seconds/microstep, excluding initialization, validation and saving.32-GPU multi-node speed remains unmeasured.

Run directory /gpfs/jiuquyun/projects/Frank/openwam-wan-lr-midtrain-20261116/smoke-54803/2026-11-16_00-14-30. checkpoint_step_80.safetensors saved; accel_state_step_80/trainer_state.json verified global_step80,opt_step10. Final export/exit verification pending at time of this entry. Full-state writing takes several minutes on GPFS; account for it separately. Real resume execution has not yet been tested in this run.

User subsequently explicitly requested queueing32 idle GPUs after successful smoke. Existing launcher now accepts formal, requires4 nodes and launches8 ranks/node. Formal uses unchanged one-epoch config/global128, a separate formal output directory and W&B ID owlr32mid20261116. Completed trainer_state.json markers select recovery only within that formal directory. Submit with --nodes=4 --time=2-00:00:00 --requeue and no fixed nodelist, using the supported idle entrypoint; no reservation changes. Resume can roll back to the last completed save after preemption;90-second graceful-save completion is not promised.

## Completion and formal submission receipt

Smoke54803 completed in17m22s, batch and training step exit0; W&B synced. The failed54803.1 step was the separate unit-test invocation missing libGL, not training; corrected unit tests passed in54803.2. Final checkpoint_step_80.safetensors export completed. End validation and complete-state marker verified as above.

Formal source b9d7c5467be477b4db619ed2f54ea7d1b8a3bcbb, clean Frank clone fast-forwarded after smoke exited. bash -n and sbatch-idle --test-only passed. Submitted2026-11-16 00:27:52 CST as job54805 with:

`sbatch-idle --parsable --nodes=4 --time=2-00:00:00 --requeue --job-name=openwam-droid-abc-alpha154000-wan-lr-stereo-midtrain /gpfs/jiuquyun/projects/Frank/OpenWAM-wan-lr-midtrain/scripts/train_h100_alpha_wan_lr.sbatch formal`

Readback:32GPUs,4nodes,128CPUs,3600Gmemory,48h,Requeue1,no fixed nodelist. Initially PENDING/QOSGrpJobsLimit awaiting scheduler synchronization; no formal training progress or W&B initialization yet. Log /gpfs/jiuquyun/home/Frank/logs/openwam-wan-lr-54805.out. Formal output /gpfs/jiuquyun/projects/Frank/openwam-wan-lr-midtrain-20261116/formal. Model card reczz28a0jigxgWp updated to排队中 with SHA,smoke results and JobID. Existing bookings unchanged.
