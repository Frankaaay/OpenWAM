#!/bin/bash
#SBATCH --job-name=openwam-libero-plus-acceptance
#SBATCH --partition=h100
#SBATCH --nodes=1
#SBATCH --gpus-per-node=8
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=480G
#SBATCH --time=04:00:00
#SBATCH --requeue
#SBATCH --output=/gpfs/jiuquyun/home/Frank/logs/%x-%j.out
set -euo pipefail
variant=${1:?wan or stereotok}
expected_sha=${2:?published OpenWAM SHA}
R=/gpfs/jiuquyun/projects/Frank/openwam-libero-plus-20261116
O=$R/OpenWAM-rgbft
M=/gpfs/jiuquyun/projects/Frank/robotwin-h100-migration-20261101
E=/gpfs/jiuquyun/projects/Frank/libero-rgbft-ppu15-20261114/eval-runtime
test "$(git -C "$O" rev-parse HEAD)" = "$expected_sha"
test -z "$(git -C "$O" status --porcelain)"
test "$(git -C "$R/plus" rev-parse HEAD)" = 4976dc30028e805ff8094b55501d532c48fec182
git -C "$R/plus" diff --exit-code HEAD
test "$(git -C "$E/frank/projects/libero-eval-20261022/XPolicyLab" rev-parse HEAD)" = 56a66e1a083edd80abaada9bc95e5a4e99ff9de5
test "$(git -C "$E/frank/projects/FastWAM-libero-eval-v2-20261019" rev-parse HEAD)" = b771fd55d0133576481dc52f867792a649fafc07
export PYTHONDONTWRITEBYTECODE=1
mounts="$M/mirror/mnt/workdir:/mnt/workdir,/gpfs/jiuquyun:/gpfs/jiuquyun,$E/frank/envs/fastwam-baselines-20261014:/mnt/workdir/frank/envs/fastwam-baselines-20261014,$E/frank/runs/eighteen-preparation-20261014/twelve-ready-u15853:/mnt/workdir/frank/runs/eighteen-preparation-20261014/twelve-ready-u15853"
export O R
srun --container-image="$M/source-runtime.sqsh" --container-mounts="$mounts" bash -c '
  set -euo pipefail
  libs=/mnt/workdir/frank/envs/system-libs/prefix/usr/lib/x86_64-linux-gnu
  export LD_LIBRARY_PATH="$libs:$libs/pulseaudio:${LD_LIBRARY_PATH:-}"
  exec /mnt/workdir/frank/envs/openwam-baselines-20261014/bin/python \
    "$O/benchmarks/libero-plus/h100_acceptance.py" "$1" --run "$R/acceptance/$1-$SLURM_JOB_ID-attempt${SLURM_RESTART_COUNT:-0}"
' bash "$variant"
