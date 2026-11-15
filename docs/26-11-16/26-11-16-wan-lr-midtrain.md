# DROID+ABC Alpha Wan-LR midtrain

User authorized dedicated branch/worktree, minimal adaptation, tests, commit/push and Frank-owned H100 clean clone. Current priority is an idle eight-GPU smoke, without booking changes.

Base peilin/wan-bino-baseline 50e0312020ab287f54658fb5a3e34e35b43d9ed8. New frank/openwam-wan-lr-midtrain. Preserve Alpha154000 Video/Action weights and the exact frozen Alpha VAE: strict-load native architecture first, then wrap its existing VAE in WanBinoVideoEncoder. No separately initialized policy projections or VAE weights. Existing StereoTok replacement remains unchanged.

Formal configuration inherits the original 3206442-window stride0.8 DROID+ABC one-epoch contract, changes input_mode to stereo and encoder to wan22_vae_bino.32 GPUs x batch2 x accumulation2 = global128; save5000 microsteps means2500 optimizer updates. Warmup and validation remain500 optimizer updates.

Bounded smoke uses8 GPUs x batch2 x accumulation8 = global128,80 microsteps/10 updates, real original datasets and normalization. Validation subset2/source and save80 are smoke-only overrides. Online W&B, independent output /gpfs/jiuquyun/projects/Frank/openwam-wan-lr-midtrain-20261116/smoke-JOBID, no requeue,2h maximum. Full training is not yet submitted. Results and actual JobID will be appended after execution.
