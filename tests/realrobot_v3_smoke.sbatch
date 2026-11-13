#!/bin/bash -l
#SBATCH --job-name=realrobot-v3-four-smoke
#SBATCH --partition=h100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=00:30:00
#SBATCH --no-requeue
#SBATCH --output=/gpfs/jiuquyun/home/Frank/logs/realrobot-v3-smoke-%j.out
set -euo pipefail
# Arguments: absolute clean clone, expected pushed SHA, unique persistent output.
repo=${1:?clean clone path required}
sha=${2:?expected commit required}
output=${3:?output directory required}
test "$(git -C "$repo" rev-parse HEAD)" = "$sha"
test -z "$(git -C "$repo" status --porcelain)"
assets=/gpfs/jiuquyun/projects/Frank/realrobot-v3-posttrain-20261111
migration=/gpfs/jiuquyun/projects/Frank/robotwin-h100-migration-20261101
orion=/gpfs/jiuquyun/projects/Frank/orion-0
fastwam=$migration/mirror/mnt/workdir/frank/projects/stage3-robotwin-FastWAM-20261029
test "$(git -C "$orion" rev-parse HEAD)" = 783b602d128a43814af32285e60a60f270ec2703
test -z "$(git -C "$orion" status --porcelain)"
test "$(git -C "$fastwam" rev-parse HEAD)" = 950e03b91ed5c8f43324eb7c9d7acb4f19943b06
test -z "$(git -C "$fastwam" status --porcelain)"
mkdir "$output"
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PYTHONPATH="$repo:$orion/src:$fastwam/src"
export LD_LIBRARY_PATH="/mnt/workdir/frank/envs/system-libs/prefix/usr/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH:-}"
python=/mnt/workdir/frank/envs/openwam-baselines-20261014/bin/python
for spec in \
 'wan/formal/cup_stacking/2026-11-11_23-31-51|9090|cup_stacking|mono-left' \
 'wan/formal/object_handover/2026-11-12_04-35-36|10930|object_handover|mono-left' \
 'stereo-r2/formal/cup_stacking/2026-11-12_10-48-57|9090|cup_stacking|stereo' \
 'stereo-r2/formal/object_handover/2026-11-12_16-33-52|10930|object_handover|stereo'; do
 IFS='|' read -r run step task mode <<< "$spec"
 srun --container-image="$migration/source-runtime.sqsh" \
  --container-mounts="$migration/mirror/mnt/workdir:/mnt/workdir,/gpfs/jiuquyun/projects/Frank:/gpfs/jiuquyun/projects/Frank" \
  --container-workdir="$repo" "$python" "$repo/tests/smoke_realrobot_v3.py" \
  --checkpoint "$assets/$run/checkpoint_step_$step.safetensors" \
  --sample-dir "$assets/smoke/$task" --task "$task" --mode "$mode" \
  --output "$output/$mode-$task.json"
done
echo ALL_FOUR_REALROBOT_V3_GPU_SMOKES_PASSED
