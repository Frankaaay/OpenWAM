#!/bin/bash
set -euo pipefail
B=/gpfs/jiuquyun/projects/Frank/openwam-midtrain-20261101
P=/mnt/workdir/frank/envs/openwam-baselines-20261014/bin/python
export LD_LIBRARY_PATH="/mnt/workdir/frank/envs/system-libs/prefix/usr/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$B/code/OpenWAM-run:$B/code/orion-0/src:$B/code/StereoTok"
export OMP_NUM_THREADS=2
export TOKENIZERS_PARALLELISM=false
export WANDB_MODE=offline
export WANDB_RUN_ID=owmaw04f
export MIDTRAIN_PREP="$B/prepared-stride06-20261102"
export MIDTRAIN_ABC_ROOT="$B/datasets/abc-canonical"
export MIDTRAIN_ALPHA_ROOT=/gpfs/jiuquyun/projects/Frank/checkpoints/OpenWAM-Alpha-Pretrain-Foundation-Model
export WANDB_DIR="$MIDTRAIN_OUTPUT"
mkdir -p "$MIDTRAIN_OUTPUT"
cd "$B/code/OpenWAM-run"
args=()
if [[ -n "${MIDTRAIN_RESUME:-}" ]]; then
  args=(training.finetune_ckpt_path=null "training.resume_ckpt_path=$MIDTRAIN_RESUME")
  printf 'ALPHA_WAN_RESUME %s\n' "$MIDTRAIN_RESUME"
fi
printf 'ALPHA_WAN_TRAIN rank=%s source=%s\n' "$SLURM_PROCID" "$(git rev-parse HEAD)"
exec "$P" -m torch.distributed.run --nnodes=2 --nproc_per_node=8 --node_rank="$SLURM_PROCID" --master_addr="$MASTER_ADDR" --master_port="$MASTER_PORT" scripts/train.py --config-name=midtrain_droid_abc_alpha_wan "${args[@]}"
