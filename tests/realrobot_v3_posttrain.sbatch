#!/bin/bash -l
#SBATCH --job-name=mug-v3-posttrain
#SBATCH --partition=h100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=64
#SBATCH --mem=768G
#SBATCH --time=12:00:00
#SBATCH --requeue
#SBATCH --output=/gpfs/jiuquyun/home/Frank/logs/mug-v3-posttrain-%j.out
set -euo pipefail
repo=${1:?clean clone}
sha=${2:?pushed SHA}
root=${3:?prepared task root}
model=${4:?wan or stereo}
test "$(git -C "$repo" rev-parse HEAD)" = "$sha"
test -z "$(git -C "$repo" status --porcelain)"
test -f "$root/preparation/result.json"
assets=/gpfs/jiuquyun/projects/Frank/realrobot-v3-train-20261113
orion=/gpfs/jiuquyun/projects/Frank/orion-0
fastwam=/gpfs/jiuquyun/projects/Frank/FastWAM-realrobot-v3-inference-20261113
test "$(git -C "$orion" rev-parse HEAD)" = 783b602d128a43814af32285e60a60f270ec2703
test "$(git -C "$fastwam" rev-parse HEAD)" = 491e518fac6e04646ecfbd464027dd575ed99bde
test -z "$(git -C "$orion" status --porcelain)"
test -z "$(git -C "$fastwam" status --porcelain)"
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 MAX_JOBS=2
export TOKENIZERS_PARALLELISM=false HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PYTHONPATH="$repo:$orion/src:$fastwam/src"
export WANDB_MODE=offline
migration=/gpfs/jiuquyun/projects/Frank/robotwin-h100-migration-20261101
srun --container-image="$migration/source-runtime.sqsh" \
 --container-mounts="$migration/mirror/mnt/workdir:/mnt/workdir,/gpfs/jiuquyun/projects/Frank:/gpfs/jiuquyun/projects/Frank" \
 --container-workdir="$repo" \
 env LD_LIBRARY_PATH=/mnt/workdir/frank/envs/system-libs/prefix/usr/lib/x86_64-linux-gnu \
 /mnt/workdir/frank/envs/openwam-baselines-20261014/bin/python \
 -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node=8 \
 scripts/train_realrobot_v3.py --model "$model" --task mug_hanging \
 --data "$root/data" --prepared "$root/preparation" \
 --checkpoint "$assets/midtrain/$model/checkpoint_step_50101.safetensors" \
 --output "$root/formal/$model" --auto-resume
