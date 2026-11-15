#!/bin/bash
#SBATCH --job-name=openwam-droid-abc-alpha154000-wan-lr-stereo-smoke
#SBATCH --partition=h100
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=32
#SBATCH --mem=900G
#SBATCH --time=02:00:00
#SBATCH --no-requeue
#SBATCH --output=/gpfs/jiuquyun/home/Frank/logs/openwam-wan-lr-%j.out
set -euo pipefail
REPO=/gpfs/jiuquyun/projects/Frank/OpenWAM-wan-lr-midtrain
B=/gpfs/jiuquyun/projects/Frank/openwam-midtrain-20261101
M=/gpfs/jiuquyun/projects/Frank/robotwin-h100-migration-20261101
test -z "$(git -C "$REPO" status --porcelain)"
export REPO B
export MIDTRAIN_OUTPUT=/gpfs/jiuquyun/projects/Frank/openwam-wan-lr-midtrain-20261116/smoke-$SLURM_JOB_ID
export MIDTRAIN_PREP=$B/prepared-stride08-20261104
export MIDTRAIN_ABC_ROOT=$B/datasets/abc-canonical
export MIDTRAIN_ALPHA_ROOT=/gpfs/jiuquyun/projects/Frank/checkpoints/OpenWAM-Alpha-Pretrain-Foundation-Model
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT=29581
export WANDB_API_KEY=$(python3 -c 'import netrc; print(netrc.netrc().authenticators("api.wandb.ai")[2])')
test -n "$WANDB_API_KEY"
export WANDB_MODE=online WANDB_DIR=$MIDTRAIN_OUTPUT
export WANDB_RUN_ID=owlrsmoke${SLURM_JOB_ID}
mkdir -p "$MIDTRAIN_OUTPUT"
echo "source_sha=$(git -C "$REPO" rev-parse HEAD) output=$MIDTRAIN_OUTPUT"
srun --kill-on-bad-exit=1 --container-image="$M/source-runtime.sqsh" --container-mounts="$B/mirror/mnt/workdir:/mnt/workdir,/gpfs/jiuquyun:/gpfs/jiuquyun" bash -c '
set -euo pipefail
P=/mnt/workdir/frank/envs/openwam-baselines-20261014/bin/python
export LD_LIBRARY_PATH=/mnt/workdir/frank/envs/system-libs/prefix/usr/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH:-}
export PYTHONPATH=$REPO:$B/code/orion-0/src:$B/code/StereoTok
export OMP_NUM_THREADS=2 TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
cd "$REPO"
"$P" -m torch.distributed.run --nnodes=1 --nproc_per_node=8 --node_rank="$SLURM_PROCID" --master_addr="$MASTER_ADDR" --master_port="$MASTER_PORT" scripts/train.py --config-name=midtrain_droid_abc_alpha_wan_lr training.gradient_accumulation_steps=8 training.max_steps=80 training.save_steps=80 training.validation.max_windows_per_source=2 "project.wandb.run_name=OpenWAM DROID+ABC Alpha154000 Wan-LR stereo 8H100 smoke"
'
