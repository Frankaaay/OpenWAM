#!/bin/bash -l
#SBATCH --job-name=realrobot-v3-training-smoke
#SBATCH --partition=h100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=64
#SBATCH --mem=768G
#SBATCH --time=02:00:00
#SBATCH --no-requeue
#SBATCH --output=/gpfs/jiuquyun/home/Frank/logs/realrobot-v3-training-smoke-%j.out
set -euo pipefail
repo=${1:?clean clone path}
sha=${2:?pushed SHA}
output=${3:?new output root}
test "$(git -C "$repo" rev-parse HEAD)" = "$sha"
test -z "$(git -C "$repo" status --porcelain)"
assets=/gpfs/jiuquyun/projects/Frank/realrobot-v3-train-20261113
test -f "$assets/transfer/all-assets-verified.json"
orion=/gpfs/jiuquyun/projects/Frank/orion-0
fastwam=/gpfs/jiuquyun/projects/Frank/FastWAM-realrobot-v3-inference-20261113
test "$(git -C "$orion" rev-parse HEAD)" = 783b602d128a43814af32285e60a60f270ec2703
test -z "$(git -C "$orion" status --porcelain)"
test "$(git -C "$fastwam" rev-parse HEAD)" = 491e518fac6e04646ecfbd464027dd575ed99bde
test -z "$(git -C "$fastwam" status --porcelain)"
mkdir "$output"
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 MAX_JOBS=2
export TOKENIZERS_PARALLELISM=false HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PYTHONPATH="$repo:$orion/src:$fastwam/src"
export LD_LIBRARY_PATH="/mnt/workdir/frank/envs/system-libs/prefix/usr/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH:-}"
migration=/gpfs/jiuquyun/projects/Frank/robotwin-h100-migration-20261101
python=/mnt/workdir/frank/envs/openwam-baselines-20261014/bin/python
runtime=(--container-image="$migration/source-runtime.sqsh"
 --container-mounts="$migration/mirror/mnt/workdir:/mnt/workdir,/gpfs/jiuquyun/projects/Frank:/gpfs/jiuquyun/projects/Frank"
 --container-workdir="$repo")
srun "${runtime[@]}" "$python" tests/verify_realrobot_v3_training.py --assets "$assets"
for model in wan stereo; do
 srun "${runtime[@]}" "$python" scripts/train_realrobot_v3.py --model "$model" --task cup_stacking \
  --data "$assets/data" --prepared "$assets/preparation" \
  --checkpoint "$assets/midtrain/$model/checkpoint_step_50101.safetensors" --output "$output/$model" --smoke --preflight
 srun "${runtime[@]}" "$python" -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node=8 \
  scripts/train_realrobot_v3.py --model "$model" --task cup_stacking \
  --data "$assets/data" --prepared "$assets/preparation" \
  --checkpoint "$assets/midtrain/$model/checkpoint_step_50101.safetensors" --output "$output/$model" --smoke
 srun "${runtime[@]}" "$python" tests/verify_realrobot_v3_training.py --assets "$assets" --run "$output/$model" --model "$model"
done
echo BOTH_V3_H100_TRAINING_SMOKES_PASSED
