# OpenWAM Alpha / Wan / PPU 双目子集左目训练计划（待独立审核）

日期：2026-11-06，北京时间。状态：仅计划，未批准正式启动；尚未完成目标子集和16卡验收。

## 1. 任务目标与授权边界

在 PPU 1.5 上训练 OpenWAM Alpha 初始化的 Wan 路线，用于与 StereoTok 路线对照。两条路线必须使用相同的 admitted episodes、窗口、train/val split、动作监督及统计口径；Wan 只读取双目数据中的左目。

用户最新纠正优先于之前全七组 ABC 方案：不是“凡是有左目都能使用”。本文件是用户要求的审核材料，不是运行授权。当前仅整理计划、读取代码/历史日志和更新文档；不提交 GPU 作业、不修改生产数据/代码、不 commit/push。后续正式启动须用户明确批准；代码修改、提交和16卡验证的授权边界也须在执行时核对。

不要批量删除、覆盖旧输出或修改其他人的目录。全七组统计和 smoke 结果保留，但不能用作该对照实验的数据合同。审核 session 先只读审核，不同时修改/运行。

## 2. 审核结论应回答的问题

1. 目标 ABC 是否精确等于原 H100/StereoTok 双目 admitted 集合，而非只匹配窗口总数？
2. 原 episode 到 PPU 七组物理文件的映射是否保留身份、左右目、时间轴、phase、split 和原像素处理？
3. 相同窗口是否得到相同动作、state、mask 和归一化？
4. scheduler、validation、checkpoint 的 step 单位是否正确？
5. 16卡分布式、完整保存/恢复、闲时抢占策略是否有实际证据？
6. 正式请求、代码精确SHA、配置和数据指纹是否齐全？没有这些不能给 ready-to-launch。

## 3. 已验证内容与未通过门禁

已验证：
- 单节点8卡 PPU 1.5、Alpha154000、冻结 Wan VAE/T5、bf16、ZeRO2、无optimizer offload可完成真实更新。
- 全七组ABC + DROID full-index smoke：512窗口，DROID66/ABC446，32microsteps/8updates；所有rank optimizer state在GPU，loss有限，作业退出0。
- 原H100子集的 admission/statistics 和原Job53358速度已通过只读SSH重新查询。

尚未完成：
- PPU上的精确双目子集映射、原split/window/phase一致性及正确norm接入。
- 使用目标子集的短smoke。全七组smoke不能替代该项。
- PPU双节点16卡通信、吞吐及正式完整checkpoint恢复验证。
- PPU正式入口、最终解析配置和正式CreateJob请求；现有深圳launcher不能直接复用路径。
- 正式代码发布SHA、磁盘余量/保存体积、运行时限与抢占恢复策略最终验收。

## 4. 数据合同

| 项目 | DROID | ABC |
|---|---:|---:|
| train windows | 1,667,932 | 1,538,510 |
| val windows | 86,756 | 21,805 |
| ABC train episodes | 不适用 | 11,764 |
| ABC val episodes | 不适用 | 172 |
| 监督 | 实际state，缺失臂mask | 实际state，双臂 |

总训练窗口3,206,442；自然比例约DROID52.02%/ABC47.98%，不做50/50重采样。1 epoch，shuffle seed42。保留原admission/exclusion与原split；不新增排除，不把读失败当作可跳过样本。

时间合同：窗口起点stride0.8s，即8个10Hz RGB ticks；17RGB@10Hz，34action@20Hz，动作offset -0.05..1.6s；原fixed-anchor相对目标、插值、phase、gripper和尾部mask保持。输出图像宽320×高384，保持原左目mosaic及黑边/resize语义。必须与原reader逐样本对比，不能只检查shape。

ABC原H100 admission记录：train总候选98,792episodes，camera filter保留11,764、排除87,028。当前全七组方案的11,043,362train windows不属于该对照范围。原统计anchors1,538,510、episodes11,764、window_stride_rgb8。

归一化：DROID和ABC分别用原目标训练子集train-only统计。若原统计可按相同语义复用，应复用并校验哈希；只有明确发现不兼容才讨论重算，不自动改变统计定义。不得使用全七组ABC stats。保留原全局窗口顺序/phase定义，防止过滤后重新编号改变global_window_index_mod_phase_count。

DROID小包是四个reader消费字段的无损H5视图；不需要重新下载138GB原始全字段。小包/索引当前已通过既有覆盖检查，源文件保留。

## 5. 资产路径：已知来源与证据

### 本地 Windows

- 工作任务最初cwd：`C:\Project\StereoTok`。
- OpenWAM仓库：`C:\Project\OpenWAM`，分支`frank/eighteen-preparation`，本轮HEAD `f4668f9af27f043ebb6b5be5830b28c4f1f0d1c6`。
- upstream origin：`https://github.com/OpenWAM-Official/OpenWAM.git`；另有frank remote `https://github.com/Frankaaay/OpenWAM.git`。不能默认向origin推送个人分支。已有dirty用户文件，禁止stash/覆盖。
- StereoTok reader来源SHA：`7dae078f7509b2e9c7c69ed8162a62263a99a077`。
- Orion reader来源SHA：`783b602d128a43814af32285e60a60f270ec2703`。
- 本次ops目录：`C:\Users\Frank\AppData\Local\AliyunCLI\ppu-requests\openwam-alpha-midtrain-20261106`。
- ops参考文件：`full_data_reader.py`、`ppu-smoke-8gpu-full-nofork.py`、`gpu-smoke-8gpu-full-nofork-request.json`、`completed-8gpu-full-nofork-logs.json`。这些是全七组smoke文件，不是目标子集正式文件。
- 完整执行记录：`C:\Project\OpenWAM\docs\26-11-06\26-11-06-openwam-alpha-ppu-readiness.md`。
- 原H100配置/迁移记录：`C:\Project\OpenWAM\docs\26-11-04\26-11-04-openwam-alpha-wan-midtrain.md`。
- 交接文件：`C:\Users\Frank\Documents\Codex\handoffs\stereotok\20261106-1920-openwam-ppu-full-smoke-handoff.md`；其全七组目标已被用户最新纠正取代。

### H100：只读原合同来源

入口`ssh.exe h100`，本轮whoami为Frank；重新操作前核对SSH config及指南。

- 原准备根：`/gpfs/jiuquyun/projects/Frank/openwam-midtrain-20261101/prepared-stride08-20261104`。
- 该目录的`abc-admission.json`、`abc-stats.json`本轮已实读；另有`droid.sqlite`、`droid-stats.json`和DROID排除/准入记录。
- 原ABC根（由原launcher绑定，需继续核验metadata及物理来源）：`/gpfs/jiuquyun/projects/Frank/openwam-midtrain-20261101/datasets/abc-canonical`。
- 原代码根：`/gpfs/jiuquyun/projects/Frank/openwam-midtrain-20261101/code/OpenWAM-run`。
- 原输出根：`/gpfs/jiuquyun/projects/Frank/openwam-alpha-midtrain-20261104/formal`。只读参考，不作为PPU新实验续训来源。
- 原速度日志：`/gpfs/jiuquyun/home/Frank/logs/openwam-droid-abc-alpha-wan-53358.out`，本轮已实读。
- 原state小包：`/gpfs/jiuquyun/projects/Frank/openwam-alpha-midtrain-20261104/cloud-transfer-stride08/stride08-state-fields.tar.gz`，历史核验1,900,408,325bytes，SHA256 `0eadf3c3cdc624a2c8f25a4f383c7eb46ef5e34817e2163b626ada03d140f615`。

### PPU 1.5：现有资产

以下来自已执行任务及日志，不代表本轮重新逐文件stat；执行前应定向回读，不递归扫共享盘。

- 项目根R：`/mnt/workdir/frank/projects/openwam-alpha-ppu-midtrain-20261106`。
- 已部署来源：`R/source`（OpenWAM f4668f9）、`R/orion-source/src`（Orion783b602）、`R/stereotok-reader`（StereoTok7dae078）。R为本段明确根路径缩写。
- 环境：`/mnt/workdir/frank/envs/ppu15-robotwin-rgbft-20261105/bin/python`。
- Torch2.9.0 / CUDA13.0 PPU专用环境，pylance0.39.0；不得以普通PyPI torch覆盖。
- Alpha：`/mnt/workdir/frank/checkpoints/OpenWAM-Alpha-Pretrain-Foundation-Model`，内含`checkpoint_step_154000.safetensors`及`asset-ready.json`；原生VAE/T5依赖按该包配置核验。
- DROID图像：`/mnt/workdir/frank/datasets/droid-prepared-20260920-ppu`。
- DROID state/index/norm：`/mnt/workdir/frank/datasets/droid-state-fields-stride08-20261106`；索引`prepared/droid.sqlite`、统计`prepared/droid-stats.json`、state视图`h100-state-view`。
- ABC物理文件公共前缀：`/mnt/workdir/from-sz/datasets/PRETRAIN_DATA_2.0/ABC-130k-yam-canonical-v1_`，七个后缀为`256x144`、`256x145`、`256x160_group01`、`256x160_group02`、`256x192`、`256x195`、`256x205`。
- 全七组旧准备目录：`R/prepared-nativecrop-full-20261106`，含`catalog.jsonl`、`admission.json`、`exclusions.json`、`abc-stats.json`。可供映射参考，其统计/准入集合不能直接用于正式训练。
- 旧全数据smoke输出：`/mnt/workdir/frank/runs/openwam-alpha-ppu-smoke-20261106/eight-gpu-full-data-nofork`，含`result.json`、`metrics.jsonl`、`config.yaml`、`wandb/`。

### 计划新资产：尚未创建/验收

建议使用下列独立路径；执行前确认不存在冲突，不覆盖旧目录：

- 目标子集准备目录：`/mnt/workdir/frank/projects/openwam-alpha-ppu-midtrain-20261106/prepared-stereo-subset-stride08-20261106`。
- 目标子集验证输出：`/mnt/workdir/frank/runs/openwam-alpha-ppu-smoke-20261106/stereo-subset-stride08`。
- 正式输出根：`/mnt/workdir/frank/runs/openwam-alpha-ppu-midtrain-20261106/stereo-subset-stride08`。
- 正式请求本地文件拟为ops目录下`formal16-stereo-subset-request.json`；目前不存在已审核的对应请求，不可直接提交旧JSON。

不在本计划中虚构最终入口脚本文件名/已发布SHA/Job ID。先完成最小现有入口适配，经授权发布后将精确SHA、最终路径、文件哈希补入审核回执。

## 6. 模型与优化配置

| 项目 | 目标 |
|---|---|
| config继承 | `configs/midtrain_droid_abc_alpha_wan.yaml` → `midtrain_droid_abc.yaml` → `train.yaml` |
| 初始化 | Alpha154000，Video DiT和Action DiT完整预训练权重 |
| 冻结 | 原生Wan VAE、T5；不加载StereoTok编码器 |
| 训练 | Video DiT、Action DiT，保持原参数组和loss定义/权重 |
| 资源 | PPU1.5，2节点×8GPU，总16GPU |
| batch/accum | 每卡2、累积4、global128 |
| epoch/max_steps | num_epochs1、max_steps null；不以smoke32microsteps限制正式运行 |
| LR | video/action均1e-4 |
| warmup | 500optimizer updates，start factor0.1，即1e-5 |
| scheduler | cosine，min ratio0.01，即1e-6；horizon约25,051updates |
| AdamW | betas[0.9,0.95]、wd0.01、max_grad_norm1.0 |
| 精度 | bf16、ZeRO stage2 |
| checkpointing | gradient checkpointing true，activation offload false |
| optimizer offload | none |
| CPU初始化 | PPU smoke为true；正式拟沿用，须写入最终解析配置 |
| 数据加载 | 现有生产配置num_workers2、multiprocessing_context spawn；必须通过目标reader实际验证 |
| seed | 42，保留原逐rank/逐step随机数策略 |

数据长度不能仅人工写死为3,206,442；loader实际长度和各rank分片应产生一致结果。约100,202microsteps/25,051updates，最后不足一个global batch的padding、尾部累积和更新计数按原实现核对，不静默drop或改变epoch定义。Warmup/总步数不能沿用七组smoke的99,307update周期。

当前独立smoke因先读取Lance后fork卡住，改workers0后通过；这不证明生产spawn reader已通过。不要把无main guard的smoke脚本直接改spawn，也不要把workers0偷偷当作已验证的正式吞吐配置。

## 7. 验证、保存、恢复与W&B

- `training.validation.every_steps=500`，单位optimizer update；各源128窗口，seed42，加start/end，单独记录DROID/ABC total/video/action。
- `training.save_steps=10000`，单位microstep；accum4对应每2500optimizer updates。`save_full_states_for_resume=true`。start/end完整状态。
- 原配置`keep_last_k_ckpts=1000000`意为实际保留全部，本任务不启用自动批量清理。启动前测量一次完整保存的实际字节数，按约10次周期保存加start/end及临时写入余量核算磁盘，不能假设空间足够。
- 仅当所有状态分片已写完并原子发布`trainer_state.json`才算可恢复。恢复模型/optimizer/scheduler/RNG/数据位置；验证恢复后opt_step、LR和样本位置正确，不只看load成功。
- 新实验从Alpha重新初始化，不能接续全七组smoke权重。PPU本次正式实验使用独立W&B run ID；重启沿用其自身ID，不复用旧H100 `owmaw04f`或smoke ID。
- 用户已明确W&B offline，优先于通用online规则。entity`yichx14-uc-irvine`、project`StereoTok`；实验名沿用`OpenWAM DROID+ABC Alpha Wan mono-left midtrain`，在config/tags记录PPU、stereo subset和精确数据指纹。run ID在正式准备时确定并记入回执。
- offline目录、配置、loss、源计数、Git SHA、资产/统计哈希与checkpoint均落持久CPFS。不得在文档/日志输出密钥。

## 8. 平台与资源请求约束

- CLI：`C:\Users\Frank\AppData\Local\AliyunCLI\aliyun.exe`，profile`h800`仅复用凭据，region必须`cn-wulanchabu`，workspace`308566`，RAM Frank UserId`205455090823393499`。
- PPU1.5父quota`quotawsqknfajqjt`，research子quota`quota1rpp98rnebp`。正式拟闲时资源；最终ResourceId/OversoldType/preemption settings须按实时权限和已核验平台语义定稿，不直接继承8卡smoke的固定资源策略。
- 已验证镜像：`dsw-registry-vpc.cn-wulanchabu.cr.aliyuncs.com/pai/training-xpu-pytorch:2.1.0-torch2.9.0-ubuntu24.04-cuda13.0-py312-7fe9b71e-3bb68f5a`。
- CPFS：`bmcpfs://cpfs-290017uukzj6idpqamp0z-000001.cn-wulanchabu.cpfs.aliyuncs.com/`挂`/mnt/workdir`。不得混用PPU1.1或深圳同名路径。
- 拟每节点沿用已验证8卡资源形状：8GPU/192CPU/1600Gi/128Gi共享内存，两节点合计16GPU/384CPU/3200Gi；该正式资源请求尚待批准和实际可调度检查。
- 拟运行上限72h用于约40h纯训练加验证/保存余量；不是自动延时/无限重试授权。到限或频繁抢占必须报告最新checkpoint与剩余ETA。
- 当前没有PPU直接SSH入口。通过已授权DLC验证，不调用旧`ssh h800`。
- 2026-11-06 19:34曾见PPU1.5总64卡、已占48、未分配16，只有1个全空节点；这是过时快照，正式前实时核验CPU/内存/节点形状，不能认定两节点已可用。
- `scripts/train_dlc_alpha_wan.sh`当前写死深圳20261104根目录、旧Python和旧run ID，不能直接作为PPU正式命令。可复用其官方WORLD_SIZE/RANK/MASTER_ADDR/MASTER_PORT逻辑，但必须核对平台变量单位（节点数与进程数）和两节点rank实际值。

## 9. 建议执行顺序与验收

### A. 精确恢复目标数据集合

以原H100 canonical metadata +原camera筛选+原split为权威生成或复用episode/window标识集合，再映射到PPU物理文件；优先复用已有代码，不按crop尺寸猜测双目资格。核对ABC11,764/172episodes与1,538,510/21,805windows，同时核对完整身份集合、顺序/phase，报告差集必须为空。若PPU仅有左目物理副本，仍必须由原双目资格证明筛选。

以精确文件/前缀完成映射、统计哈希与state/action/pixel语义检查，不扫描全共享盘。真实解码抽检train/val、各相关物理组、首中尾与尾部mask；左右读取选择不应改变原动作监督或样本集合。失败定位后定向修复，不自动排除。

### B. 完成最小正式入口适配

检查现有生产dataset factory、validation构建、路径和launcher，确定复用方案；不能只给训练注入正确dataset而validation仍指向错误七组根。尽量只改相关现有入口；如需要新文件/接口，先提交具体变更范围供批准。保留用户dirty文件；发布前按仓库规则核对分支/HEAD/upstream，未经授权不commit/push。

冻结最终解析配置（路径展开后）、Git SHA、数据/统计指纹与请求JSON。正式远端只使用自己clean clone的已发布精确SHA，不上传本地dirty源码。既有ops脚本只能作为smoke参考，不能冒充已发布生产路径。

### C. 目标配置验证（须对应资源授权）

先完成CPU定向合同检查，再运行有界16卡目标子集smoke，以实际分片/参数组/无offload/optimizer state/有限loss/源计数/真实更新验收。包含一次完整保存和恢复到下一次更新；用单独输出目录。报告冷启动与稳态速度、每rank峰值显存、实际样本比例和保存时长。若资源只能满足8卡，报告未完成16卡门禁，不自动换配置或扩大时限。

### D. 提交正式请求前最终审核

必须具备：目标集合/norm通过、16卡验证通过、最终SHA和config、资源/挂载/磁盘通过、恢复策略与max runtime定稿、用户明确正式启动授权。先dry-run，再单次CreateJob，保存回执；回执缺失先查询是否落地，不能重复提交。

### E. 启动与运行验收

一次启动健康检查确认16rank、global128、目标窗口数、正确W&B ID/模式、step0完整checkpoint、分源start validation、首批真实optimizer updates和输出持久化。更新主体及保存/验证ETA。未经用户要求不持续轮询，不为跑满epoch自动扩大资源或更改数据。

完成要求：job/pod退出0、完成一个epoch、最终分源validation有限、最终权重和完整恢复状态存在、样本计数及分片尾部处理可解释、无未处理异常。若抢占，明确回退到哪个完整checkpoint及重做量。

## 10. 时间估计与实测依据

- H100原Job53358：16GPU、global128。日志microstep200→1200约1127.5s；500→1300为904s；1000→1300为338s。稳定约1.13s/microstep，即4.52s/update，原子集1epoch纯训练约31.45h。
- PPU Job`dlcjkr8qzfjjhe81`：2026-11-06 19:25:05–19:30:11，Succeeded/Exit0；全七组数据8卡global64。首update76.02s含冷启动；update2..8平均5.74924s，1.43731s/microstep，11.13windows/s。PyTorch峰值allocated40.54246GB/reserved45.04473GB，不是NVML总占用。
- PPU16卡若维持同一步耗时，25,051updates约40.01h（1.67天）纯训练；相对H100步耗时约+27.2%。这是跨数据比例/loader/卡数的估计，不是严格硬件benchmark。
- PPU8卡global64相同窗口量约80h，但改变global batch，不作为默认正式方案。
- 目标子集DROID比例由全七组smoke约13%提高到52%，I/O不同；双节点通信也未测。正式ETA须用目标16卡实测更新，另加验证、完整保存、排队和抢占。
- 先前6.6天来自已废弃的12,711,294窗口范围，不再适用。

## 11. 给独立审核session的操作说明

请先完整阅读本文件、`C:\Project\StereoTok\AGENTS.md`、个人PPU/H800指南；读取H100时再读H100个人和公共指南。用户原有dirty修改保留。优先检查数据集合/监督合同和step单位，再看资源与恢复。

输出：按严重程度列出阻塞问题、证据路径/代码行、最小修复建议，以及可以通过的项目。区分实测、代码推断和计划。不要将本文件中的“计划新路径”当作已存在资产，不因先前smoke成功给出正式ready结论。未经用户另外授权，不修改代码、不启动训练、不停止其他作业、不提交Git。

最终状态：等待审核；正式训练未启动。

## 12. 2026-11-06 执行补充：数据门禁尚未关闭

用户随后授权目标子集恢复和16卡加速测试，未授权正式长训练。目标子集目录现已实际生成；原norm、episode顺序及全量窗口(anchor/phase/split)指纹通过，实际状态见同目录任务记录。跨H100逐样本parity失败：RGB不是逐像素相同，action/proprio存在约1e-6差异。当前parity-ready.json未生成，16卡benchmark未提交。不得沿用第5节的“尚未创建”判断已完成的子集目录，也不得把基础ready.json当作正式通过。正式入口/config发布与16卡验收仍未完成。

## 最新执行状态：PPU 1.5 32 卡排队准备（取代前文未更新的计划状态）

用户随后明确接受现有 native-crop-v2 视频，保留原 H100 双目 episode 子集及左目输入；取消16卡加速实验，改为32卡正式训练，指定PPU 1.5闲时队列。因此不再要求迁移原H100视频或生成逐像素一致的parity-ready.json。全量窗口身份与原norm继续保留；不声称两端视频逐像素相同。

- 最终配置：4节点×8卡、每卡batch2、累积2、global128；3,206,442 train windows、1epoch约25,051 optimizer updates。峰值LR1e-4、500update warmup从1e-5、cosine最低1e-6；save_steps5000为microstep，即2500updates。保留梯度检查点、ZeRO2、bf16、无optimizer offload及用户授权的W&B offline。
- 必要生产修改仅4文件：openwam/dataloader/orion_midtrain_ppu.py、openwam/dataloader/registry.py、configs/midtrain_droid_abc_alpha_wan_ppu.yaml、scripts/train_dlc_alpha_wan.sh。train/val均由正式factory构建；ABC episode ID采用原subset ordinal，修复按split分别从0编号导致validation重叠判断的问题。用户既有dirty源码/测试未改。
- 基线分支frank/eighteen-preparation，SHA f4668f9af27f043ebb6b5be5830b28c4f1f0d1c6；origin fetch成功。远端个人source clean，同SHA，origin为https://github.com/Frankaaay/OpenWAM.git。尚无本次commit/push或生产source更新。
- CPU预检任务dlckpkirvugyvz08 Succeeded/Exit0：实际Hydra配置和factory、train/val身份无重叠、两源两split合计12样本与已接受PPU reader逐项精确比较、2个spawn worker完整读取16样本、warmup/cosine单位及bash语法检查全部通过。初次dlc1ki2qkruxl05u失败于测试对浮点eta_min的严格相等；修正为1e-15容差，并消费完整有限DataLoader以消除提前结束worker的退出噪声；没有修改训练参数或放宽样本比较。
- CPU检查产物：/mnt/workdir/frank/projects/openwam-alpha-ppu-midtrain-20261106/preflight-ppu32-20261106/resolved-config.yaml；目标prep目录production-ready.json包含4个候选文件及catalog/两源norm的SHA256。正式launcher会校验这些指纹、远端clean状态和已发布精确SHA。CPFS预检可用5,952,145,195,008bytes。
- 正式请求：C:/Users/Frank/AppData/Local/AliyunCLI/ppu-requests/openwam-alpha-midtrain-20261106/formal32-ppu15-stereo-subset-request.json；PPU15 research quota1rpp98rnebp，priority4，ForceQuotaOverSold，RDMA，4×8GPU，72小时上限，使用PPU15既有镜像和CPFS。CLI dry-run通过，仅证明请求构造，不代表服务器已接受排队。
- 正式输出：/mnt/workdir/frank/runs/openwam-alpha-ppu-midtrain-20261106/stereo-subset-stride08。W&B offline ID owma-ppu32-stereo-1106，实验名OpenWAM DROID+ABC Alpha Wan mono-left midtrain。
- 32卡GPU作业尚未提交；32卡通信、完整状态保存/恢复尚未实测。正式入口要求formal-ppu32-source.json提供已发布SHA，该文件尚未激活。按AGENTS，下一步需明确授权仅提交/推送本次4个生产文件及两份任务记录，随后部署精确SHA、提交请求并做一次启动健康检查。
- 不新增加速benchmark。按8卡吞吐理想线性外推32卡纯训练约20小时，非32卡实测ETA，不含启动/验证/保存/排队/抢占；启动后依据实际吞吐更新。
