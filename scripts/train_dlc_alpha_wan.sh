#!/bin/bash
set -euo pipefail
B=/mnt/workdir/frank/projects/openwam-alpha-midtrain-20261104
P=/mnt/workdir/frank/envs/openwam-baselines-20261014/bin/python
export LD_LIBRARY_PATH="/mnt/workdir/frank/envs/system-libs/prefix/usr/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$B/deps:$B/OpenWAM-run:$B/orion/src:$B"
export OMP_NUM_THREADS=2
export MAX_JOBS=4
export TOKENIZERS_PARALLELISM=false
export WANDB_MODE=offline
export WANDB_RUN_ID=owmaw04f
export MIDTRAIN_PREP="$B/cloud-transfer-stride08/prepared"
export MIDTRAIN_ABC_ROOT="$B/abc-canonical"
export MIDTRAIN_ALPHA_ROOT=/mnt/workdir/frank/checkpoints/OpenWAM-Alpha-Pretrain-Foundation-Model
export MIDTRAIN_OUTPUT=/mnt/workdir/frank/runs/openwam-alpha-midtrain-20261104/formal-stride08
export WANDB_DIR="$MIDTRAIN_OUTPUT"
test -s "$B/cloud-transfer-stride08/ready.json"
test "$WORLD_SIZE" = 2
mkdir -p /gpfs "$MIDTRAIN_OUTPUT"
if ! test -e /gpfs/jiuquyun; then
  ln -s "$B/cloud-transfer-stride08/h100-state-view" /gpfs/jiuquyun
fi
test "$(readlink -f /gpfs/jiuquyun)" = "$B/cloud-transfer-stride08/h100-state-view"
cd "$B/OpenWAM-run"
test -z "$(git status --porcelain)"
resume=$("$P" - <<'PY'
import json,os
from pathlib import Path
states=[(int(json.loads(p.read_text())['global_step']),str(p.parent.parent))
        for p in Path(os.environ['MIDTRAIN_OUTPUT']).glob('*/accel_state_step_*/trainer_state.json')]
print(max(states)[1] if states else '')
PY
)
args=()
if [[ -n "$resume" ]]; then
  args=(training.finetune_ckpt_path=null "training.resume_ckpt_path=$resume")
fi
printf 'ALPHA_WAN_DLC rank=%s nodes=%s source=%s resume=%s\n' "$RANK" "$WORLD_SIZE" "$(git rev-parse HEAD)" "$resume"
nvidia-smi --query-gpu=name,memory.total --format=csv
exec "$P" -m torch.distributed.run --nnodes="$WORLD_SIZE" --nproc_per_node=8 --node_rank="$RANK" --master_addr="$MASTER_ADDR" --master_port="$MASTER_PORT" scripts/train.py --config-name=midtrain_droid_abc_alpha_wan "${args[@]}"
