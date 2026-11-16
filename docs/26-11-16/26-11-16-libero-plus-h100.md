# OpenWAM LIBERO-Plus H100 有限验收

## 范围与授权

用户指定 OpenWAM 原版 Wan 与唯一的 StereoTok LIBERO checkpoint；不是 FastWAM policy，也不是 Wan RGBFT。允许创建 `frank/openwam-libero-plus-h100` worktree、最小适配、测试和 commit/push，使用 H100 闲时资源。先每模型 8 卡、14 回合验收；不提交 10,030 回合全量评测，不改阿里云现役任务。

## 代码与协议

- 基线：OpenWAM `8d802728a26372baa91063fb79dd2ab44a24e1cb`，含已验证的 StereoTok checkpoint 自包含部署和原生 LIBERO EEF10 接口。
- LIBERO-Plus：`4976dc30028e805ff8094b55501d532c48fec182`；HF `Sylvest/LIBERO-plus` revision `dd2bd61b7d9a6fef1abc52d606e983b41886a149`。
- assets.zip SHA256：`96764a4bfbdaea98d4411598caeab235458318fe0f549611b93d1a323027b3cf`。
- 复用自己的 XPolicyLab `56a66e1a083edd80abaada9bc95e5a4e99ff9de5` FreshStereoEvalEnv；头部基线 0.06m、腕部 0.02m。FastWAM `b771fd55d0133576481dc52f867792a649fafc07` 只提供 OpenWAM 已依赖的 StereoTok codec，不执行 FastWAM policy。
- Plus client 复用普通 LIBERO 的 OpenWAM policy adapter。原生 7D OSC 动作、EEF10 转换、256x256 渲染、fresh-left/fresh-right、image_transform=none；前三套件 600 步、LIBERO-10 700 步，30 步 settle；seed42/init0，每变体一回合。推理参数继承 checkpoint，不照搬 FastWAM 参数。
- 语言从 BDDL 的 `env.language_instruction` 获取，不把扰动文件名作为指令。
- 官方仅扰动左头部图像。Plus adapter 在 fresh render 后给两个头部视图重放同一噪声 RNG，腕部不变；保存/恢复随机状态，核验官方左图重放、渲染不推进状态和左右图不同。
- 分类表沿用交接文档选择规则：七扰动类、四 suite、五 noise families，共14回合；每rank `tasks[rank::8]`。每个任务按所属 suite 绑定原训练 normalization stats，不混用四套件统计。

## 权重与运行位置

H100 根：`/gpfs/jiuquyun/projects/Frank/openwam-libero-plus-20261116`。
源文件位于本人新 CPFS `/mnt/new_workdir/research/frank/runs/eighteen-preparation-20261014/twelve-ready-u15853`：

- Wan：`train-openwam-libero-wan/2026-10-21_14-00-22/checkpoint_step_13230.safetensors`，24,813,767,464 bytes。
- StereoTok：`train-openwam-libero-stereotok/2026-10-21_22-24-28/checkpoint_step_13230.safetensors`，28,010,055,404 bytes，配套 config/tokenizer/stereotok.json。

复用本人 `robotwin-h100-migration-20261101/source-runtime.sqsh` 及原有 Python 环境；单独 target 安装 Wand 0.6.13/scikit-image 0.24.0/lazy-loader 0.4，不修改共享环境。每rank独立编译缓存。

入口：`benchmarks/libero-plus/h100_acceptance.sbatch <wan|stereotok> <published-sha>`；每次新 JobID 输出新目录，不覆盖已有结果，不自动requeue。客户端异常/任一worker失败即失败；成功完成意味着14个预定回合均有结果，不要求策略14次都成功。小样本只用于接入验收。

## 已执行与待验证

- `python -m unittest discover -s tests -p test_libero_plus_noise.py`：1 passed；模拟验证左右噪声配对及 RNG 恢复。
- Python编译检查及 `git diff --check`：通过。上述不是 GPU/真环境验收。
- CPU资产传输 Job54853：已运行；目的地 `frank/`，rsync partial/append-verify，原文件保留。
- CPU依赖 Job54855：失败，复用环境没有 pip；用宿主 uv 安装至独立 target，Job54857 成功。没有修改原有环境。
- CPU环境导入检查已通过；完整资产哈希/解压及两组8卡有限验收尚待完成。GPU ETA 必须来自本次实测；交接文档 FastWAM 耗时不能直接套用 OpenWAM。

日志位于 `/gpfs/jiuquyun/home/Frank/logs/`；后续 JobID、SHA、导入路径及验收结果在本文件和运行根 run-receipt.md 回填。

## 环境适配补充

- 2026-11-16：初次提交 `cc03c8f449a25dbaf5daf37f0e85f7bf9d13ee13`。模型卡：Wan `reczz28aEJvlFLdl`、StereoTok `reczz28aEJvlGER6`。
- 镜像实际为 Ubuntu24.04，client numpy1.26.4。MagickWand缺失，CPU Job54859/54860 分别暴露 apt缓存缺失/阿里云内网镜像502；使用本任务独立 apt lists/sources 下载 Ubuntu 官方软件包，Job54861/54862 成功解包到任务 `system-libs/prefix`，不改系统或原环境。
- 下载 libmagickwand-6.q16-7t64、libmagickcore-6.q16-7t64、liblqr-1-0、libfftw3-double3、liblcms2-2、imagemagick-6-common；设置独立库、coder/config路径，并为 Wand 的查找方式补无版本 `.so` 链接。实际 PNG/运动模糊 CPU 验证另行记录。
- 验收增加原训练 EEF normalization 与部署 materialize 逐值相等检查；Plus默认图像变换同步为 fresh renderer 对应的 none。
- Job54867：`WAND_PNG_MOTION_BLUR_OK 0.24.0`、`CLIENT_IMPORT_OK 3.3.2`。Pyxis会以镜像默认值覆盖宿主LD_LIBRARY_PATH，因此验证改为容器内显式设置；正式worker本来就在容器内通过subprocess env设置同一完整路径。
- 准备入口 `h100_prepare.sbatch` 依赖传输成功，校验zip SHA/CRC、448799文件和8953192654bytes，完成官方assets软链接后生成ready.json，再记录两份模型SHA256。两组8卡验收顺序依赖，最多同时占8卡；任一阶段失败不会自动进入下一阶段。

## 13:25 提交回执

执行源码固定为 `3ca4f0613233f194e861e344591953f565ccf0a5`；H100本人clean clone已fetch/ff-only同步，状态干净。CPU原生入口检查Job54868成功：14项选择、client/server --help、噪声配对/RNG单测全部通过。

| 阶段 | Job | 实时状态 |
|---|---|---|
| 资产传输 | 54853 | RUNNING，约6.25GB/59GB，最近约3.4–4.2MB/s |
| 哈希校验/解压 | 54870 | PENDING，afterok:54853 |
| 原版Wan，8卡14回合 | 54872 | PENDING，afterok:54870 |
| StereoTok，8卡14回合 | 54874 | PENDING，afterok:54872 |

GPU任务均已核实 `AdminComment=h100:idle`、`Requeue=0`、TimeLimit04:00:00、未分配节点/GPU；无新预约。脚本bash检查及sbatch --test-only通过，但不能据此宣称GPU接入已经跑通。传输剩余约3.5–4小时；解压/排队/实际GPU耗时另计，StartTime当前Unknown。

结果目录分别为运行根 `acceptance/wan-54872`、`acceptance/stereotok-54874`；日志为 `logs/openwam-libero-plus-wan-u13230-eval-54872.out`、`logs/openwam-libero-plus-stereotok-u13230-eval-54874.out`（日志根 `/gpfs/jiuquyun/home/Frank`）。最终每组应有14回合且acceptance.json.complete=true，当前尚未产生。

模型卡已回填；没有正式分数或全量任务。本人临时0GPU传输DSW `dsw-1a0plk4ipgycdyvy40` 仍用于54853，传输完成后应停止，当前不能中断。此提交回执保留为本地未提交文档更新，避免改变已锁定的运行SHA。

## 13:36 历史权重反查与传输提速

用户要求复查H100历史LIBERO权重，缺失才提速传输。核对Frank从10月1日以来Slurm LIBERO作业、历史54735日志实际OPEN_SUITE_BOUND、原始迁移记录和本人既有checkpoint/runtime目录：此前七卡54735–54741用的是 `libero-rgbft-ppu15-20261114/train-openwam-libero-rgbft/2026-11-13_15-24-40/checkpoint_step_13230.safetensors`（24,813,767,464 bytes），属于RGBFT；不能作为本次原版Wan替代。未找到此次原版Wan及StereoTok的可复用完整LIBERO权重。此前只查两个runtime目录，查找依据不足；本轮补了实际历史作业反查。

传输改为同一CPU额度内三路rsync（Wan、StereoTok、Plus资产），文件清单互不重叠；保留partial/append-verify。新Job54878依赖afterany:54853，先将准备Job54870改为afterok:54878，再取消旧54853，避免并发写相同文件。旧任务已退出，约8.3GB已传StereoTok数据续传，没有重头下载或删除文件。两项GPU任务54872/54874及固定源码SHA不变。

115.89秒实际文件大小增量测速（排除已有partial计数）：Wan3.379MB/s、StereoTok2.927MB/s、资产2.807MB/s，合计9.114MB/s；原单路约3–4MB/s。按各流当前剩余量/实测速率，Wan约1.99h、StereoTok1.82h、资产0.58h，主体传输ETA约2小时，不含校验解压和GPU排队/验收。此为约2分钟窗口，不保证持续峰值。

运维脚本本机 `C:/Users/Frank/AppData/Local/Temp/openwam-plus-transfer/transfer-parallel.py` 和 `transfer-parallel.sbatch`，H100放在本次运行根；Python语法、bash语法和sbatch test-only已通过。Python脚本本地/远端SHA256均 `ff70f38dd87eafa0ce0db774780e673bf6fedd3cdd60618ce11beec49a877b64`。`jobs.json`保留previous_transfer=54853并更新transfer=54878；当前仍使用原0GPU DSW做传输源，完成后再停止。

## Wan RGBFT 追加请求

用户授权优先用H100已有Wan RGBFT跑LIBERO-Plus，预约8卡，短预约结束后允许按闲时继续。新增wan-rgbft变体只引用现有24.81GB checkpoint和既有assets/openwam，不重复传输权重。复用先前RGBFT H100验证过的Mesa llvmpipe、LP_NUM_THREADS=1和设备探测；模型仍在GPU推理。Plus官方noise依赖原始image observable，故不复制原普通LIBERO禁用image observable的优化，避免破坏噪声重放口径。

当前0050/0054/0062预约表短空档已被本人32卡midtrain54805实际占用，抢任意一台会重排整项训练；不改变现役midtrain。首个无预约冲突的8卡节点0049为11月17日00:00起，计划预约00:00–04:00，作业4h上限与预约保护独立，到期可继续idle。仍先做14回合接入验收，尚未提交10030回合全量评测。

RGBFT使用独立本人clean clone OpenWAM-rgbft；保留原两项排队任务的3ca4f06源码与checkpoint身份不变。新变体按suite复用同一native EEF10统计、固定相同14变体及seed42，输出明确记录variant与实际checkpoint绝对路径。

### 13:58 RGBFT 提交回执

- 代码已推送并在独立clean clone同步 `2dad4a582dd7d8046b19cb7c4a442ae063598723`；原OpenWAM clone仍固定3ca4f06。
- 模型卡 `reczz28aF161h3C6`；预约 `reczz28aF1P9Am0x`，0049/8卡，北京时间11月17日00:00–04:00，状态已预约；创建前全表冲突检查、创建后字段和模型卡反向关联回读通过。
- CPU预检54880已COMPLETED/0，确认RGBFT checkpoint/config/tokenizer/既有VAE路径、四suite配置、14任务选择、Mesa设备探测、原生client/server入口和噪声单测。日志最终 `RGBFT_CHECKPOINT_AND_MESA_PREFLIGHT_OK`。
- GPU作业54882已提交，0049/8GPU/32CPU/480G/4h/no-requeue；依赖资产准备54870。当前PENDING Dependency、尚无GPU输出，预约开始前AdminComment=idle。0049提前释放时允许按idle开跑；00–04预约窗口匹配保护，到期仍在4h作业时限内则自动idle，实际抢占不会自动重排。
- 日志 `/gpfs/jiuquyun/home/Frank/logs/openwam-libero-plus-rgbft-u13230-eval-54882.out`；输出运行根 `acceptance/wan-rgbft-54882`。当前没有模型GPU加载/Plus真实闭环验收成功证据，也没有全量score。原midtrain54805及其他用户任务未修改。

### 用户改为闲时及midtrain预约核对

用户明确取消RGBFT预约、改用闲时。预约reczz28aF1P9Am0x已改为已取消并回读；原Pending54882取消，重提交不限节点的8GPU闲时54884（依赖54870，4h/no-requeue），确认ReqNodeList=null、AdminComment=h100:idle、PENDING Dependency。源码和14回合范围不变，新输出acceptance/wan-rgbft-54884，日志openwam-libero-plus-rgbft-u13230-eval-54884.out。没有中断任何GPU训练。

用户截图对应midtrain54805实际运行0050/0054/0061/0062。预约全表已完整读取，四条现存记录reczz28aEFu7kQHD/reczz28aEFu7kj40/reczz28aEFu7hntY/reczz28aEFu7jut8分别对应这四台，各8卡，11月16日16:51至18日16:51，关联midtrain卡reczz28a0jigxgWp，无需重复预约。0061当前Kevin预约reczz28ZsNKynIrh有效至16日16:50:13，故当前整组仍idle；没有覆盖他人预约或更改现役midtrain。

LingBot三份最终checkpoint问题按Object/Goal/LIBERO-10核查：H100既有libero-rgbft-ppu15-20261114只有Spatial5000和Object2400，两份transformer各10177831668bytes且有config.json；Goal和LIBERO-10目录不存在，Object5000也不存在。此前Object/Goal补传文档记录未落地，后续最终权重在L20Y的记录不等于H100已有；本轮只核对未发起新传输。

## 闲时评测启动修复
原Wan54872在主worker导入normalization/OpenCV时缺少libGL.so.1；RGBFT54884因Slurm暂存脚本路径推导仓库而报not a git repository；Stereo54874被失败依赖阻塞。仅修改h100_acceptance.sbatch：明确使用本人OpenWAM-rgbft clone，并在容器内启动Python前设置既有系统库路径，使主worker和子进程均继承。三变体统一使用同一已发布修复SHA，权重/14回合/seed42不变，保留失败日志。bash语法检查通过；真实容器导入验证及重新排队待补。

### 修复后重新排队
修复commit12fff2eaf1d7a514fb16cd102fd9cd088150961f已push，H100 OpenWAM-rgbft clean clone已fetch/ff-only。实际同镜像/挂载/容器内LD_LIBRARY_PATH/仓库检查的CPU预检54918完成0:0，cv2与materialize_eef_stats导入通过。GPU脚本bash和sbatch test-only通过。取消原DependencyNeverSatisfied的54874；新Wan54920、RGBFT54921、StereoTok54922，每组8GPU/4h/no-requeue，无预约、最多同时8GPU，afterany串行避免单组失败永久阻塞后续独立评测。18:48实时Wan为PENDING Priority，后两组PENDING Dependency，Wan AdminComment=h100:idle。三张原模型卡已追加故障/修复/新Job与SHA，GPU rollout尚未开始，未宣称闭环通过。未影响midtrain和DROID恢复。

## 19:35 后续运行错误和重排修复
54921 RGBFT已完成14回合；54920 Wan在NVIDIA EGL无设备时报错，54922 Stereo因无条件导入旧FastWAM不存在的MetricHead报错，均非抢占。用户要求闲时抢占自动重排：改--requeue并在输出路径加入SLURM_RESTART_COUNT，重启从头验收且保留上次结果，避免同Job目录存在立即失败。Wan/Stereo复用RGBFT已验证Mesa渲染；MetricHead仅在geometry.metric_disparity.shallow_head_weight>0时导入，当前checkpoint无该项，不改变模块/权重合同。AST/bash检查通过。未重跑已完成RGBFT。

重排修复已发布0b0389ccd205ded91457f35d1a8ec363d39ffd90；H100 pull首次TLS断开，同clone重试ff-only成功，未绕过。CPU同容器导入预检54930通过0:0，GPU脚本bash/sbatch test-only通过。新Wan54931已RUNNING0061，Stereo54932 PENDING Dependency afterany54931；两者实读Requeue=1、h100:idle，4h/8GPU串行。抢占后以attempt目录从头重试，尚未发生真实抢占恢复验收。RGBFT54921完成14回合不重跑；三张原模型卡已追加本轮信息。

## 用户要求直接全量
三组直接全量，停止14回合阶段。新增必要H100全量绑定h100_full.py/h100_full.sbatch，复用run_all_suites原动态队列/协议签名/结果断点续跑，不新造调度器。官方10030任务/四suite，每任务1trial、seed10000、official_global、settle30、600/700steps，三模型30090回合；seed42验收不混入。每组8GPU/1server per GPU、独立闲时队列/Requeue1、7天上限、稳定输出full/{variant}以完成任务为断点。每suite独立绑定原native EEF10统计，复用Mesa渲染、原推理优化和精确权重。静态AST/bash通过，真实CPU全任务dry-run待补；不再提交GPU接入验收。

### 全量提交回执
部署SHA bfde7882997372c1a3f42a60528a512f7995970d。三组CPU dry-run54937/54938/54939全部输出FULL_DRY_RUN_COMPLETE，每组4suite完整枚举及官方协议检查通过，没有GPU验收回合。全量Wan54941、RGBFT54942、Stereo54943已独立提交，各8GPU/32CPU/480G/7天/Requeue1，均PENDING Priority，无预约/无Dependency、h100:idle。稳定输出为运行根full/wan、full/wan-rgbft、full/stereotok，原scheduler协议签名校验后跳过完成task；不使用attempt全量重跑。三张原模型卡已回填全量范围/Job/SHA。尚无全量GPU启动健康或最终分数，不提供排队ETA。midtrain、DROID恢复和checkpoint传输未改动。

## 21:15 GPU 渲染修复
用户报告全量GPU长期0%。54941容器原缺NVIDIA EGL驱动注入，入口强制Mesa llvmpipe。宿主有595.91.07 EGL库；同一54941分配内overlap定向验证，宿主设置NVIDIA_DRIVER_CAPABILITIES=all后，容器注入libEGL_nvidia及eglcore；MuJoCo枚举8设备，真实GL_RENDERER=NVIDIA H100 80GB HBM3/PCIe/SSE2。仅将full sbatch增加该环境变量、full客户端取消software_render=True。保留模型/协议/seed/全量10030任务、已完成结果及自动恢复，三路均使用GPU渲染。此前Mesa结果保留，渲染后端切换记入运行记录。L20Y FastWAM RGBFT当前3854回合，8worker最近20回合均值13.92–26.05s，不是OpenWAM同模型对照。后续提交与真实吞吐待补。
硬件渲染环境按每次render_gpu构造，移除仅按config_root缓存的Mesa环境，防止多个worker复用首个GPU绑定。
21:16提交：d31d6810170d2fcf5bb27959b45fc02ac803ab29已push并在本人远端clone ff-only。旧54941/54942/54943已取消；新Wan54945、RGBFT54946、Stereo54947均8GPU/32CPU/480G/7天/Requeue1、h100:idle，Pending Priority，保留full原输出按task恢复。bash/AST/sbatch test-only通过；真实GPU EGL上下文已验证，新全量吞吐待分配资源。三张原模型卡同步。
21:42 SIGABRT定位：暂停54945主调度PID124965派发（SIGSTOP，既有子任务继续），原task176单独真实policy重放仍exit=-6。faulthandler日志crash-diagnostic/task176-faulthandler.log显示robosuite read_pixels/mujoco.mjr_readPixels崩溃。FreshStereoEvalEnv使用第二EGL上下文，robosuite.render/read不make_current；最小修复在PlusStereoEvalEnv._images调用前及finally恢复robosuite GL上下文，覆盖reset/set_init_state/step调用。保留图像/噪声/RNG/相机合同，测试补充上下文切换与原RNG配对断言。实际服务已启用DiT cache(0.99)、Wan/MoT compile、SDPA；deploy配置decode_video=false、prompt cache开启，10步去噪约0.1s。原失败任务真实GPU修复重放进行中。

## 22:05 SIGABRT 修复与重复渲染优化
21:42 的 EGL 上下文恢复候选未修复真实 task176，已撤销该候选。隔离重放、faulthandler 和 native backtrace 证实 NVIDIA 595.91.07 eglcore 在 GL readback/sync 中主动 abort；glFinish、独立 Renderer、RGBA/PBO、关闭驱动线程均未解决，动作与仿真状态有限正常。不能把真实 GL_RENDERER 验证描述为完整评测稳定性。
正式修复统一 full 客户端使用既有 Mesa llvmpipe，4 raster threads/worker；8 worker 仍用32CPU/8GPU，不新增资源。PlusStereoEvalEnv.step 暂停最终被覆盖的相机 observable，保留 proprio/state 更新，fresh render 一次后让官方 wrapper 对左目施加噪声，右目重放相同 RNG。异常 finally 恢复原 step/camera；不改分辨率、AA、噪声、seed、任务、步数或 checkpoint。保留原131条完成结果，诊断结果仅在 crash-diagnostic，不混入分数。
真实双目新旧对照：task176、noise=0/1/11/21/31/41、各3步，所有观测逐元素相同、reward/done/state 相同、后续 numpy RNG 相同；parity-fresh.log EXIT0。单步约1.2→0.6s，玻璃噪声约8.2→4.1s。4线程 Mesa 原失败任务完整600步，172.45s 无崩溃（task176-mesa4-fresh.log）；1线程完整策略对照进行中。单元测试覆盖双目噪声 RNG 不额外消耗以及渲染异常恢复，1 test PASS。
推理实际服务器日志确认 SDPA、Wan/MoT compile、DiT velocity cache threshold=.99；配置 decode_video=false、prompt cache=true、10 denoise steps、8持久服务和动态任务队列保留。真实全量并发吞吐/ETA待重启健康检查；单回合不外推所有噪声类别。
