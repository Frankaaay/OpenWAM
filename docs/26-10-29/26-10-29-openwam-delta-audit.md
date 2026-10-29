# OpenWAM 分支改动审计与恢复修复

## 基线和证据范围

官方 origin/main 已fetch，SHA 3dfc9a7d9230f0e1382e6b22639278f1a86e5eb5；分支HEAD 8798eb95441c96ea02336a2f4a648e9dce4ebebe。共同祖先4d7f642b841913ea3c176010b692b1de3d5ef180。以共同祖先到HEAD识别我方新增：46文件，1682增加/234删除；其中生产/配置/benchmark 27文件，其余测试与记录。不能直接把当前官方main与分支的全部差异归为我方修改，因为官方也继续发展。

实际RoboDojo两训练锁定3e28bf522705dc2cf731531190e67d8201d5a1c3。HEAD比训练版本另含配对初始化、留出验证、step0保存及u8000部署绑定；不能未经差异审核直接把最新HEAD整体替换训练来源。此前原有dirty文档和smoke文件保留，不纳入本次补丁。

以下为源码/提交级审计及既有文档证据复核，不是所有功能已重新跑GPU验收。不承诺零bug。

## 改动分类与处置

| 类别 | 具体内容/来源 | 当前两路必要性 | 处置与证据限制 |
|---|---|---|---|
| 历史batch回放 | eaa1137把官方skip_first_batches换成next循环 | 不是训练算法必要步骤；为worker随机流重建付出巨大成本 | 撤销逐批解码；本地最小补丁恢复官方API。保留已有epoch采样seed，避免另改样本排列 |
| 完整状态保护 | marker校验、opt_step恢复、优化器边界保存、完成后保留state | 抢占恢复需要；官方已有full-state能力，不能说全是我们新增 | 保留必要的完整性/计数修复；不扩大范围 |
| checkpoint滚动保留 | 默认keep1变2、weights/state配对清理 | 用户既定存储策略，非模型算法要求 | 保留当前合同；无需为本次故障改清理 |
| scheduler修复 | ea09634/8ee2cc0重绑实际optimizer和SequentialLR子调度器，恢复闭式LR | 有已记录真实失效优化器和LR偏差，直接撤回风险高 | 保留；旧验收含小模型CPUAdam及8卡续训证据，非所有优化器组合已证明 |
| StereoTok适配 | encoder/stereotok、注册、预处理、checkpoint-local身份、mono-left | StereoTok路线必要，Wan不应受影响 | 保留，明确这是实验扩展而非官方baseline |
| 保留预训练DiT替换VAE | 放开官方from_scratch gate，检查通道/patch/压缩率 | Wan初始化StereoTok实验需要 | 是实验设计差异；形状匹配不证明latent语义等价，不能称严格官方模型 |
| 冻结forward修复 | 8ee2cc0避免deepcopy后闭包仍指向源encoder | 已有源/副本错误与GPU性能复现，当前依赖它 | 保留；有实际回归证据，不为缩小diff盲撤 |
| RoboDojo发布数据适配 | robodojo_lerobot读取v2.1视频与EEF表，校验帧/时间/夹爪，复用EEF20/normalize | 当前已绑定数据格式需要；原版主要HDF5路径不能直接读此输入 | 保留；不是单纯性能改动，涉及数据合同，需独立长期审计 |
| RoboTwin发布数据/统计 | manifest、wxyz转换、显式统计绑定、真实右目/共享增强 | 当前RoboDojo不走 | 不随本次修改；对相应实验必要但不是所有任务必需 |
| LIBERO/LeRobot双目 | 右目读入、布局空槽、严格相机输入 | 当前RoboDojo不走 | 不随本次修改；见下方潜在风险 |
| 双目部署与绑定 | right_images传递、u8000/u15853白名单、input_mode匹配 | 当前恢复不需要；StereoTok评测需要 | 保留已有评测，不混入本次修复 |
| 评测合同 | 原生wxyz边界转换、共享计划、相机基线与FastWAM相机实现复用 | 不在当前训练恢复路径 | 实验对齐设施；与官方默认评测存在差异，不能统称原版 |
| W&B在线/身份校验 | 3e28bf5强制已配置的online和原run ID | 用户既定要求，需要可靠观测 | 保留，不改变loss/数据 |
| 配对初始化/留出验证/step0 state | f4e12c7/3c2fab5/ac3f318 | 当前3e28bf5训练不含，不是两路恢复所需 | 不为修resume顺带启用；这部分为scratch实验，不并入本次范围 |
| 测试/文档 | 19个文件 | 支持验证，不等于生产行为必要 | 旧小模型resume测试没有DataLoader，不能用它证明worker恢复/性能 |

## 已定位问题与未验证风险

1. 已证实：历史回放性能回退。8卡Wan恢复首100批175.287秒，11500批外推5–6小时，抢占后重来。此前误归因官方已纠正。
2. 已证实的验证缺口：test_full_resume_experiment原测试update直接torch.randn/random/np.random生成数据，没有真实多worker DataLoader；它不覆盖回放性能、worker增强序列或prefetch等价性。
3. 恢复语义边界：两路color_jitter.enabled=true，VideoColorJitter每样本调用3次random.uniform；旧checkpoint没显式保存worker状态。恢复官方skip可保留索引位置，但不能声称增强样本或后续loss与不中断逐位相同。不是模型/优化器丢失，而是随机增强流差异。用户选择需要明确记录。
4. 静态潜在风险：LeRobotV3Reader左右目分别调用_decode_window_video，各自对_head_camera_choices做random.randrange；如果启用多head选择，左右可能选不同rig。目前没有证据当前RoboDojo经过该路径，不据此改当前训练，也不称已发生数据错误。
5. 硬编码u15853/u8000绑定有溯源作用，也曾阻止合法u8000评测；这种门禁是实验范围约束，不是通用模型能力。当前已明确白名单，不扩展。
6. 依赖范围：StereoTok复用FastWAM codec，双目RoboTwin评测复用其camera代码；只审OpenWAM无法证明这些外部依赖无bug。本报告未对FastWAM/XPolicyLab全仓做独立审计。

## 最小修复与验收

仅改openwam_trainer.py恢复迭代入口：官方skip_first_batches替代逐批next，保留本分支已有epoch seed以兼容当前数据顺序；不改模型、损失、数据、batch、预算、优化器、LR、VAE或checkpoint。

测试追加到既有test_full_resume_experiment.py：实际安装Accelerate在1/8个逻辑rank、0/2 workers、epoch0/3上，对比跳批前后样本索引，检查跳过样本没有被getitem读取，同时明确验证随机增强流不一致。CPU helper dlc1sjzbwrdeu0ex已Succeeded，CURSOR_TEST_PASS：36种分片/worker/epoch组合通过，每种比较完整与恢复两路径。它是分片/worker测试，不是8GPU完整训练，也没有证明真实视频decode逐位相同。

发布尚未执行：需要确认采用官方普通随机增强续训语义，以及本次补丁的commit/push授权。正式重启仍要求clean精确SHA、保留原W&B/8卡idle、确认完整状态加载与新loss/步数。不得用候选补丁或CPU通过声称两路已恢复。

## 完整文件清单（共同祖先到已提交HEAD，未含本次候选补丁）

| 文件 | 增加 | 删除 |
|---|---:|---:|
| `benchmarks/libero/openwam2libero_interface.py` | 18 | 0 |
| `benchmarks/libero/policy_config.yml` | 2 | 1 |
| `benchmarks/libero/single_eval.py` | 4 | 2 |
| `benchmarks/robotwin/eval_policy_wrapper.py` | 21 | 31 |
| `benchmarks/robotwin/openwam2robotwin_interface.py` | 42 | 2 |
| `benchmarks/utils/action_conversion.py` | 5 | 4 |
| `configs/train.yaml` | 1 | 1 |
| `docs/26-10-15/26-10-15-stereotok-published-reader.md` | 25 | 0 |
| `docs/26-10-16/26-10-16-full-resume.md` | 12 | 0 |
| `docs/26-10-18/26-10-18-rolling-two-checkpoints.md` | 5 | 0 |
| `docs/26-10-19/26-10-19-robotwin-eval-quaternion.md` | 9 | 0 |
| `docs/26-10-20/26-10-20-stereotok-robotwin-clean-eval.md` | 9 | 0 |
| `docs/26-10-21/26-10-21-shared-robotwin-plan.md` | 57 | 0 |
| `docs/26-10-22/26-10-22-libero-stereo-eval.md` | 7 | 0 |
| `docs/26-10-29/26-10-29-u8000-deploy-binding.md` | 13 | 0 |
| `openwam/dataloader/bases/lerobot_v3_reader.py` | 28 | 20 |
| `openwam/dataloader/libero.py` | 10 | 0 |
| `openwam/dataloader/robodojo.py` | 49 | 18 |
| `openwam/dataloader/robodojo_lerobot.py` | 95 | 0 |
| `openwam/dataloader/robotwin.py` | 88 | 5 |
| `openwam/dataloader/utils/stats_computation/robodojo_stats_computation.py` | 29 | 22 |
| `openwam/dataloader/utils/stats_computation/robotwin_stats_computation.py` | 32 | 2 |
| `openwam/deploy/engine.py` | 1 | 0 |
| `openwam/deploy/obs_preprocess.py` | 36 | 0 |
| `openwam/deploy/policy.py` | 2 | 0 |
| `openwam/model/architectures/base.py` | 33 | 88 |
| `openwam/model/video_backbone/encoder/__init__.py` | 6 | 4 |
| `openwam/model/video_backbone/encoder/stereotok.py` | 113 | 0 |
| `openwam/model/video_backbone/registry.py` | 9 | 3 |
| `openwam/model/video_backbone/wan/conditioning.py` | 17 | 2 |
| `openwam/model/video_backbone/wan/encode.py` | 14 | 1 |
| `openwam/model/video_backbone/wan_backbone.py` | 45 | 2 |
| `openwam/train/openwam_trainer.py` | 158 | 8 |
| `openwam/train/utils/checkpointing.py` | 51 | 8 |
| `openwam/train/utils/training_utils.py` | 8 | 4 |
| `tests/benchmarks/test_libero_bridge.py` | 20 | 0 |
| `tests/dataloader/test_libero.py` | 29 | 0 |
| `tests/probe_robodojo_lerobot.py` | 47 | 0 |
| `tests/probe_stereotok_robotwin.py` | 71 | 0 |
| `tests/test_checkpointing.py` | 44 | 1 |
| `tests/test_external_encoder.py` | 166 | 1 |
| `tests/test_freeze_strategy.py` | 26 | 0 |
| `tests/test_full_resume_experiment.py` | 78 | 0 |
| `tests/test_obs_preprocess.py` | 59 | 0 |
| `tests/test_robotwin_interface_proprio.py` | 42 | 4 |
| `tests/test_shared_plan_policy.py` | 46 | 0 |

## 授权发布

用户已明确接受官方跳批后的非逐位增强续训，并授权提交推送最小补丁、各8卡闲时恢复、保留checkpoint/W&B。当前HEAD较运行3e28bf5的额外训练功能均需显式validation/paired_initialization配置或resume_state_dir=None；本次两个完整resume不触发这些分支。u8000差异仅部署绑定。启动前核验两配置没有validation/paired_initialization_seed，且AUTO_RESUME命中完整状态，禁止从零启动。仅提交trainer恢复入口、既有测试增补与本文，不提交原有dirty文档。
