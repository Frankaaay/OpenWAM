#!/bin/bash
set -euo pipefail
B=/mnt/workdir/frank/projects/openwam-alpha-ppu-midtrain-20261106
P=/mnt/workdir/frank/envs/ppu15-robotwin-rgbft-20261105/bin/python
: "${OPENWAM_PPU_SOURCE_SHA:?Set the reviewed, published source SHA}"
export PYTHONPATH="$B/source:$B/orion-source/src:$B/stereotok-reader"
export OMP_NUM_THREADS=4
export MAX_JOBS=4
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export WANDB_MODE=offline
export WANDB_RUN_ID=owma-ppu32-stereo-1106
export MIDTRAIN_PREP="$B/prepared-stereo-subset-stride08-20261106"
export MIDTRAIN_DROID_ROOT=/mnt/workdir/frank/datasets/droid-state-fields-stride08-20261106
export MIDTRAIN_DROID_IMAGES=/mnt/workdir/frank/datasets/droid-prepared-20260920-ppu
export MIDTRAIN_ALPHA_ROOT=/mnt/workdir/frank/checkpoints/OpenWAM-Alpha-Pretrain-Foundation-Model
export MIDTRAIN_OUTPUT=/mnt/workdir/frank/runs/openwam-alpha-ppu-midtrain-20261106/stereo-subset-stride08
export WANDB_DIR="$MIDTRAIN_OUTPUT"
test "$WORLD_SIZE" = 4
test -s "$MIDTRAIN_PREP/production-ready.json"
test -s "$MIDTRAIN_PREP/abc-stats.json"
cd "$B/source"
test -z "$(git status --porcelain)"
test "$(git rev-parse HEAD)" = "$OPENWAM_PPU_SOURCE_SHA"
"$P" - <<'PY'
import hashlib,json,os
from pathlib import Path
prep=Path(os.environ['MIDTRAIN_PREP'])
receipt=json.loads((prep/'production-ready.json').read_text())
assert receipt['train_windows']==3206442 and receipt['global_batch']==128
for name,expected in receipt['source_files'].items():
    assert hashlib.sha256(Path(name).read_bytes()).hexdigest()==expected,name
for name,expected in receipt['data_files'].items():
    assert hashlib.sha256(Path(name).read_bytes()).hexdigest()==expected,name
PY
mkdir -p "$MIDTRAIN_OUTPUT"
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
printf 'ALPHA_WAN_PPU32 rank=%s nodes=%s source=%s resume=%s\n' "$RANK" "$WORLD_SIZE" "$(git rev-parse HEAD)" "$resume"
nvidia-smi --query-gpu=name,memory.total --format=csv
exec "$P" -m torch.distributed.run --nnodes="$WORLD_SIZE" --nproc_per_node=8 --node_rank="$RANK" --master_addr="$MASTER_ADDR" --master_port="$MASTER_PORT" scripts/train.py --config-name=midtrain_droid_abc_alpha_wan_ppu "${args[@]}"
