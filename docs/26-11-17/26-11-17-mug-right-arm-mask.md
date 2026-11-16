# 挂杯右臂训练修复

分支：`frank/realrobot-v3-training`。仅修改 `mug_hanging`：80 维槽位 `0:10` 的归一化 state/action 显式置零，对应 proprio/action mask 置 False；右臂 `34:44`、历史时间 mask、图像、prompt、TCP 变换、统计量和其余训练参数不变。叠杯、交接仍使用双臂。

原因：闲置左臂的微小标准差会放大传感器抖动和现场状态偏差。模型将逐维 proprio_mask 合并为 token 级开关，因此不能只改 mask，必须同时将左臂输入清零。此修改针对已发现的接口风险，不代表已证明低成功率的唯一根因。

挂杯 `data-contract.json` 新增 `active_arms: right`。旧完整状态合同与新合同不同，会拒绝续训；旧任务已完成时也会拒绝直接返回 ALREADY_COMPLETE。必须用新输出目录，从各自 midtrain50101 初始化，不修改旧 stats，也不从挂杯5550继续训练。

## 训练入口

H100 原训练 clone：`/gpfs/jiuquyun/projects/Frank/OpenWAM-realrobot-v3-training-20261113`。先核对 clean 工作区并更新该分支，再执行。以下新根必须不存在；不覆盖旧训练。数据和 preparation 复用原版本，不需要重新计算统计量。

```bash
set -euo pipefail
C=/gpfs/jiuquyun/projects/Frank/OpenWAM-realrobot-v3-training-20261113
OLD=/gpfs/jiuquyun/projects/Frank/mug-v3-posttrain-20261114-prompt-v2
NEW=/gpfs/jiuquyun/projects/Frank/mug-v3-posttrain-20261117-right-arm
cd "$C"
git status --short --branch
# 确认 clean 且处于 frank/realrobot-v3-training 后：
git pull --ff-only origin frank/realrobot-v3-training
test ! -e "$NEW" && mkdir "$NEW"
ln -s "$OLD/data" "$NEW/data"
ln -s "$OLD/preparation" "$NEW/preparation"
SHA=$(git rev-parse HEAD)
sbatch tests/realrobot_v3_posttrain.sbatch "$C" "$SHA" "$NEW" wan
sbatch tests/realrobot_v3_posttrain.sbatch "$C" "$SHA" "$NEW" stereo
```

沿用原入口：每路 8 卡、10 epochs / 5550 updates、global batch 32、LR 3e-5、原先明确授权的 W&B offline。此次仅修改代码并推送，不替用户提交训练。新合同和结果仍位于 `$NEW/formal/{wan,stereo}`。

## 新权重部署要求

此训练分支不含 `realrobot_v3` 真机推理实现，现场推理在独立分支。部署新模型前必须同步：左臂归一化 state 槽位 `0:10` 清零、对应 mask False；有效动作只保留右臂 `34:44`，不向左臂发送模型预测指令，左臂保持预设姿态。图像不遮挡，统计量保持原版。不能直接用旧双臂部署配置评价新模型；本次没有修改或验证现场部署代码。

## 验证

在既有 `tests/verify_realrobot_v3_training.py` 增加 CPU 回归，调用真实 reader / TCP 变换：覆盖 mono-left 与 stereo、首帧历史 padding、右臂数值保持、叠杯/交接保持双臂，以及左臂位置偏移 5 mm/夹爪 1 mm 不改变新训练输入与右臂目标。测试只替换视频解码，使用合成 TCP 轨迹。

本机 Windows CPU：`python -m pytest tests/verify_realrobot_v3_training.py -q --import-mode=importlib` → **1 passed**；PYTHONPATH 包含本分支及 `C:/Project/orion-0/src`。初次因本机缺 h5py 失败，补齐测试依赖后通过。三份修改 Python 文件的 ruff、AST 解析、git diff --check 均通过。没有执行 GPU smoke、真实数据全量回放、左右臂模型 loss 诊断或真机闭环测试。
