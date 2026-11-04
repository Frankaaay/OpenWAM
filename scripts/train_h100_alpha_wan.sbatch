#!/bin/bash
#SBATCH --job-name=openwam-droid-abc-alpha-wan-mono-left-train
#SBATCH --partition=h100
#SBATCH --account=compute-account
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=32
#SBATCH --mem=512G
#SBATCH --time=7-00:00:00
#SBATCH --requeue
#SBATCH --output=/gpfs/jiuquyun/home/Frank/logs/openwam-droid-abc-alpha-wan-%j.out
set -euo pipefail
B=/gpfs/jiuquyun/projects/Frank/openwam-midtrain-20261101
M=/gpfs/jiuquyun/projects/Frank/robotwin-h100-migration-20261101
export MIDTRAIN_OUTPUT=/gpfs/jiuquyun/projects/Frank/openwam-alpha-midtrain-20261104/formal
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT=29579
export PYTHONUNBUFFERED=1
test -z "$(git -C "$B/code/OpenWAM-run" status --porcelain)"
for p in "$B/prepared-stride06-20261102/droid.sqlite" "$B/prepared-stride06-20261102/droid-stats.json" "$B/prepared-stride06-20261102/abc-stats.json"; do test -s "$p"; done
mkdir -p "$MIDTRAIN_OUTPUT"
# A completed trainer_state.json is published atomically after all state shards.
# An interrupted save has no marker and is never selected for requeue recovery.
export MIDTRAIN_RESUME=$(/gpfs/jiuquyun/projects/Frank/StereoTok/.venv/bin/python - <<'PY'
import json, os
from pathlib import Path
root = Path(os.environ['MIDTRAIN_OUTPUT'])
states = [(int(json.loads(p.read_text())['global_step']), str(p.parent.parent))
          for p in root.glob('*/accel_state_step_*/trainer_state.json')]
print(max(states)[1] if states else '')
PY
)
printf 'ALPHA_WAN_JOB %s restart=%s resume=%s\n' "$SLURM_JOB_ID" "${SLURM_RESTART_COUNT:-0}" "$MIDTRAIN_RESUME"
srun --kill-on-bad-exit=1 --container-image="$M/source-runtime.sqsh" --container-mounts="$B/mirror/mnt/workdir:/mnt/workdir,/gpfs/jiuquyun:/gpfs/jiuquyun" bash "$B/code/OpenWAM-run/scripts/train_h100_alpha_wan.sh"
