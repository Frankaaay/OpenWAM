#!/bin/bash
#SBATCH --job-name=openwam-libero-plus-prepare
#SBATCH --partition=h100
#SBATCH --qos=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --output=/gpfs/jiuquyun/home/Frank/logs/%x-%j.out
set -euo pipefail
expected_sha=${1:?published OpenWAM SHA}
R=/gpfs/jiuquyun/projects/Frank/openwam-libero-plus-20261116
O=$R/OpenWAM
test "$(git -C "$O" rev-parse HEAD)" = "$expected_sha"
test -z "$(git -C "$O" status --porcelain)"
srun python3 "$O/benchmarks/libero-plus/h100_extract_assets.py"
T=$R/frank/runs/eighteen-preparation-20261014/twelve-ready-u15853
srun sha256sum \
  "$T/train-openwam-libero-wan/2026-10-21_14-00-22/checkpoint_step_13230.safetensors" \
  "$T/train-openwam-libero-stereotok/2026-10-21_22-24-28/checkpoint_step_13230.safetensors" > "$R/weights.sha256"
echo OPENWAM_PLUS_ASSETS_PREPARED
