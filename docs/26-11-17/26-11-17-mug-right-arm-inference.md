# 挂杯右臂推理适配

分支 `frank/realrobot-v3-inference`，基线 d53c798；配套训练 `frank/realrobot-v3-training` 的 8bdef6b。本次不修改训练分支，不重训，不修改统计量或旧 checkpoint。

服务端读取 checkpoint 目录 `data-contract.json` 的 `active_arms`，仅接受 both/right，right 仅用于 mug_hanging。缺字段的旧模型仍按 both；不根据任务名称将旧模型自动当成右臂模型。归一化后左臂 state 槽位 0:10 显式清零；normalizer 的有效动作 scatter 索引仅保留 34:44，复用模型已有 inactive-action 处理。左臂未监督输出在 TCP 解码/运动限幅前替换为占位值，不影响右臂限幅。整个模型输出仍须有限，右臂旋转、位移、夹爪和时间检查不放宽。

服务端响应携带 active_arms；客户端核对其与 P7 YAML 的 arms 一致，每次响应都检查，早于启用伺服和动作发送。复用 Kevin 固定提交 06ebf02d1d7806c5b20c03bcfbc682bd61446103 的 P7System arms:right 行为：只获取右臂控制权、切换右臂模式、发送右臂 TCP/夹爪和停止右臂。左臂只读反馈/关闭连接，不下发保持当前位姿的命令。anchor 漂移检查只针对受控臂。

客户端与服务端均补齐 mug_hanging prompt，与训练英文一致。部署步骤已更新 README：配置 task:mug_hanging，P7 arms:right，使用包含 active_arms:right 的新训练模型，先 check-only。现场相机方向仍需核对，不自动增加旋转。

## 验证

本地 Windows CPU，使用 git archive 提取固定 ORION P7 提交的 src/wam 到 `C:/Users/Frank/.codex/visualizations/2026/11/17/mug-right-arm-inference/orion-p7`，作为 PYTHONPATH 依赖；没有修改 ORION 工作区或创建分支。设备 SDK 被模拟，但执行的是实际固定版本 P7System 的控制/发送/停止/释放实现。

命令：`python -m pytest tests/test_realrobot_v3.py -q --import-mode=importlib`。最终 **28 passed**：包含原有客户端测试、socket 往返、right/both/旧合同/非法合同加载、归一化、未监督左臂大输出与右臂限幅、双/右臂异常清理和不执行 warmup、P7 SDK 左臂零控制调用、闲置左臂不阻断 anchor 检查而右臂漂移仍拒绝。运行依赖 numpy/scipy/Pillow/PyYAML/pytest/omegaconf；测试模型加载用 stub，不加载真实权重。`ruff check --ignore I001` 与 `git diff --check` 通过；原基线三文件共有 7 处 I001 导入排序告警，本次不顺带重排。

未执行真实 SDK/CORA、GPU checkpoint 推理或机器人闭环；不将 CPU 模拟测试描述为真机验证。新模型尚需训练及现场验收。
