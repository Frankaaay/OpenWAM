# H800 单卡测评提速

用户已授权独立 branch/worktree、commit/push，以及最小客户端/服务端缓存合同和对照脚本；只用 H800 一张闲时 GPU，不修改现有正式作业。

分支 `frank/h800-eval-speed`，基于当前 H800 服务端 `9259a320a5945f15eab8167f1a888f5238971eb2`；客户端文件先对齐正在使用的 `4a06f9a3c78a40605fb3a4f56d60dffb1251c220`，避免把已有 shared-plan/四元数修正混成提速效果。

实现：服务器响应明确给出 `needs_observation`；只有同步动作缓存非空时允许不带图像的 `cached_action` 请求。保持每动作一次服务器调用、原生 binary-command 投影和 EEF 转换。async 始终需要新观测，reset 后必须新取图；不推测 horizon。SharedPlanPolicy 要求协议版本 1，禁用歧义网络请求自动重发，并拒绝动态灯光/视频记录模式；不改模型参数。

验证：静态 AST 与 diff 检查；追加既有 server tests 验证 horizon=1/3/32 的动作顺序、生成条件、二值投影、重规划边界和 reset。Windows 系统 Python 缺 pytest，完整 CPU 测试拟在已有 H800 CPU 环境执行。首次本机读文件默认 GBK 解码失败，已明确 UTF-8 后重新检查；此失败未到达远端。

真实对照入口：`tests/h800_eval_speed.py run <独立输出目录>`。复用 FastWAM 冻结 evaluator 的 `rollout_case`，使用原任务计划中的 open_laptop/14（历史短成功）、open_laptop/1（历史失败）、open_microwave/99（历史1500步失败）。不改变这些回合的上限、seed、instruction。先旧服务器/旧客户端，再候选服务器/候选客户端，同一 GPU 顺序执行；对比所有生成输入、动作 chunk、每步执行动作、环境 RNG、初始场景/图像、成功结果和耗时。保留原生模型 seed/compile/cache 设置，不根据期望成绩调整。

checkpoint：`/mnt/workdir/frank/runs/robotwin-u15853-mono-20261101/openwam/checkpoints/2026-11-01_22-36-55/checkpoint_step_30000.safetensors`，28,010,055,404 bytes。基线服务器/客户端与 FastWAM 路径在脚本中明确绑定，candidate 必须 clean。当前没有真实 GPU 结论；排队开始 ETA 未知，首轮预计约 45–90 分钟（含模型加载/编译，需首轮实测修订）。

小样本用于验证与定位，不替代正式全量分数。若动作/输入出现无法解释的差异，不启用候选。

CPU 实测：H800 的 0GPU DSW，隔离 pytest 依赖位于 /mnt/workdir/frank/runs/openwam-h800-eval-speed-20261103/test-deps，使用现有 system-libs-prepared.json 的 LD_LIBRARY_PATH；tests/test_policy_server_cli.py 与 tests/test_robotwin_interface_proprio.py 共 35 passed in 30.01s。没有修改共享环境。追加客户端缓存请求/reset 边界测试后再执行该定向测试。

追加单测 1 passed in 3.54s，总计36项通过。首轮 `dlc1h03217rsep7v` 在加载模型前因字面 H800 名称检查失败；没有产生回合。独立单卡身份查询 `dlce5pb0rvz7ydvq` Succeeded，实际 NVIDIA L20Y，Hopper、compute capability9.0、PCI 0x26BA10DE、81559MiB、MIG Disabled，UUID GPU-b98be483-045c-758e-2ed2-70e7dea0d5e0。此卡型与该阿里云集群既有评测记录一致；H800 是任务中的集群称呼，不能据此宣称商品型号已证实为 H800。将检查改为上述实际卡型/PCI/架构/显存及单卡约束，保持同一research配额和闲时模式。原日志全部保留。

`dlc17uhuyzs1i587` / pair02：旧版完整基线成功86步/205.062s、失败700步/603.497s、失败1500步/1170.330s，取图分别53.627/436.303/876.289s。候选首回合36步且首次生成的 proprio 哈希不同，故立即停止并确认 Stopped，禁止采用该候选速度结果。根因是构建候选时只恢复4a06f9a3客户端入口，遗漏其 `benchmarks/utils/action_conversion.py` 中的wxyz转换依赖（9259a32仍是xyzw）。补齐该文件，并同步4a06f9a3的实际原生位姿测试，保留新增缓存/reset测试。属于恢复原测评合同，不是修改数据语义或放宽门禁。

复测允许显式引用已完成baseline，只在GPU UUID、设备字段、case列表、config哈希完全一致时运行；保留baseline来源路径且不改旧结果。若物理GPU不同则拒绝复用。原生成输入/动作差异门禁不变。

修正后pair03首个输入完全相同，但独立服务进程的首个原始动作chunk不同（最大0.0078125，平均0.0002772，137/640元素不同），短回合85步而旧版86步；再次停止dlcap24rdtr4tteq并确认Stopped，不放宽阈值。该差异发生在跳过观测之前，需区分进程/编译数值差异与缓存优化。追加同一预热模型实例的诊断：先对同输入连续推理3次，要求第2/3次完全相同；随后旧路径短回合重复两次必须完全相同，再做旧/新短回合精确门禁，只有通过后才运行完整700/1500步对照。模型配置、seed、采样步数均不改。此实验测量缓存路径的因果影响，单独披露跨冷启动非逐位复现现象。
