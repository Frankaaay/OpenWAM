# OpenWAM Alpha PPU 续训准备（2026-11-06）

## 范围与代码
- 用户要求检查 PPU 上 DROID 与七个 native-crop ABC 目录，准备 Alpha 续训；尚未提交 GPU smoke/正式训练。
- 本地分支 frank/eighteen-preparation，HEAD f4668f9af27f043ebb6b5be5830b28c4f1f0d1c6；本轮已 fetch origin/frank。保留原有 dirty 文件，不改 reader 或训练配置。
- 既有合同：Alpha 全模型初始化，冻结 Wan VAE/T5，mono-left，0.8s stride，state-derived relative pose，DROID missing-arm mask，W&B offline。新数据纳入范围尚待确认。

## 实时 PPU 核验
- cn-wulanchabu，workspace 308566，RAM frank，研究闲时 quota1rpp98rnebp，PPU15 M890P。
- CPFS 挂载 /mnt/workdir，个人现有环境 /mnt/workdir/frank/envs/ppu15-robotwin-rgbft-20261105；有原生 torch/DeepSpeed/accelerate/flash_attn 和 h5py/av，无 lance。
- 深圳旧任务 dlc1lponftvg5e4g 已 Stopped；未修改其他任务。
- CPU 审计 dlchrsh6ztlv1n82、dlc17ar4ej61xu68、dlc1a2ogsrou6y2j；均 0 GPU，4 CPU/16Gi，最多10分钟。第三个输出 AUDIT_DONE。

## ABC 七组审计
精确前缀 /mnt/workdir/from-sz/datasets/PRETRAIN_DATA_2.0/ABC-130k-yam-canonical-v1_。

| suffix | episodes | frames | published train | published val | head stereo |
| --- | ---: | ---: | ---: | ---: | ---: |
| 256x144 | 687 | 1825805 | 680 | 7 | 0 |
| 256x145 | 10480 | 32930893 | 10324 | 156 | 0 |
| 256x160_group01 | 8687 | 27930189 | 8515 | 172 | 0 |
| 256x160_group02 | 3249 | 9403553 | 3249 | 0 | 3249 |
| 256x192 | 76540 | 194285560 | 75646 | 894 | 0 |
| 256x195 | 286 | 968406 | 281 | 5 | 0 |
| 256x205 | 99 | 411648 | 97 | 2 | 0 |

- 总计100028段；metadata source_episode_id跨组未发现重复。尚非逐视频完整性验收。
- CROP_READY 为 native-crop-v2 / dataset-resolution-v2，training_loader_compatible=False。七组缺少现有 reader 要求的 dataset_configs。
- state 20维：双臂xyz、rot6d、gripper；fps30，需沿用原10Hz RGB/20Hz动作重采样。
- episode本地重编号，保留native_source_episode_index/source_episode_id/source_split；不能误用原sidecar序号。info只声明train，必须使用episode级source_split避免验证泄漏。
- 当前 orion_midtrain.py 的 abc_orion 即使 mono-left 仍 require_camera_views((0,1))。直接复用只会留下3249段且无val，与旧11764 train/172val也不同。扩大到七组单目数据需用户确认范围，再适配reader、重算norm和窗口数。不得静默改变。

## DROID 与资产
- 初次检查只发现 /mnt/workdir/from-sz/datasets/zhenfei/openpi-fetchman-pi05-droid-joint-head-right-4l20y-20261011/datasets，下含6个shard；尚不能证明拥有原来的末端实测state/Lance窗口合同。
- 更深一层只读schema审计 dlc1thw1l7nu1py4 已提交。
- Alpha 下载镜像任务 dlc1sdw9hnvpbwp3 因HTTP429失败；切换官方源 dlc1cukemjcz7ek0，查询时Running但尚无下载字节/完成证据。
- 目标 /mnt/workdir/frank/checkpoints/OpenWAM-Alpha-Pretrain-Foundation-Model，revision52df4e66c82c5c8b480adcc8d01f4db7415dfb56；checkpoint_step_154000.safetensors 24813767464字节，SHA256 180a02653118b0f96da28a9cae9ec7b4c1c1e6cd0e4608b56c7b4884f8001d3d。脚本仅验证后写asset-ready.json。
- 请求/脚本/日志留在 C:\Users\Frank\AppData\Local\AliyunCLI\ppu-requests\openwam-alpha-midtrain-20261106。所有准备作业0GPU；没有训练吞吐或PPU Alpha smoke通过证据。

## 下一步
确认ABC纳入范围；核实DROID末端实测state与相机；下载/校验Alpha；做最小reader适配和norm/index；真实数据PPU smoke通过后才提交16卡闲时训练。当前不能宣布可直接开训。
## 最终核验补充
- dlck9rmrvxovn4mk 输出 DONE：候选“droid”目录实际每个shard/dataset有8维pose/quaternion/gripper state、joints及25Hz AV1左右图像，六组共2416段724800帧，每段300帧。目录带fetchman/pi05-droid-joint命名，不能据此证明它是原先DROID预训练数据；未替换原DROID源。需要明确真正DROID路径/来源。
- 官方源下载 dlc1cukemjcz7ek0 也返回 LocalEntryNotFoundError（Hub访问失败）；没有asset-ready或完整下载证据。下载准备未完成，需走既有H100/深圳资产搬运或可达下载入口，未声称权重就绪。
- 请求用户集中确认：是否将ABC扩大为七组全部有效mono-left样本；真正DROID数据在PPU的准确目录。因数据范围/来源问题，暂停依赖它们的reader修改与训练提交。

## 用户确认后的准备与 smoke（继续）
- 用户确认七组全部有效 ABC mono-left，并要求自行查找 DROID、准备环境、完成 smoke 后停止，等待正式训练确认。此次没有正式训练提交授权。
- 身份再次实查 frank。PPU GitHub HTTPS clone 因 TLS 中断失败；通过本地已 fetch 的 Git bundle 经 CPU 作业搬运，基于本人已有9259a320 clean clone，git bundle verify/fetch/checkout 成功得到独立 clean f4668f9 source。路径 /mnt/workdir/frank/projects/openwam-alpha-ppu-midtrain-20261106/source。没有修改原有PPU训练clone/环境。
- 固定 Orion 783b602d 的 src/dataloader 通过 git archive 搬入任务独立 orion-source/src；SOURCE_SHA 保留版本来源。项目依赖检测 torch/DeepSpeed/accelerate/h5py/av/ijson/scipy/hydra/diffusers 均可导入定位；继续复用本人既有PPU解释器，未覆盖系统框架。
- bootstrap成功作业 dlc17v2ya3o6bo1b。初次大请求超过65536字节服务端上限，没有落地；改为6个小数据分片后重组验证，不涉及生产工具/代码改动。
- dlc3crmyzlt59h0c：13个真实ABC窗口准备成功（7组各1train，6组各1val；group02无原val）。anchor0.8s，17RGB10Hz，34state-derived动作20Hz。逐帧最近时间误差<0.018s；实际图像尺寸符合native_crop；按原crop坐标补回256x256空白边缘，再走现有mono-left mosaic，保留原图像几何，不伪造右目。
- smoke样本仅使用7个train窗口按Orion相对位姿/SLERP/zscore算法计算临时统计；明确SMOKE ONLY，禁止用于正式训练。样本/manifest/stats在任务根smoke/下。没有声称全量reader/norm适配完成。
- Alpha镜像直链可读，config/tokenizer已下载。单流下载保留2139238400字节后停止dlc1espzpnsemw04，切换 dlcdmcm4rnvn4fwk 8路Range下载；完整文件大小及SHA256验证通过才写asset-ready。范围分片保留，不批量删除。
- 计划GPU smoke：1张PPU15闲时，24CPU/200Gi/32GiSHM，最多40分钟；7次真实训练更新与开始/结束val；冻结Wan VAE/T5，Alpha初始化双DiT，bf16/ZeRO2，batch1/accum1，LR1e-5对应正式warmup起点。无optimizer offload。仅有1卡吞吐意义，不代表16卡性能。
- smoke输出 /mnt/workdir/frank/runs/openwam-alpha-ppu-smoke-20261106，W&B offline ID owma-ppu-smoke1106，明确独立smoke名称；不复用正式run写入调试指标。GPU请求已准备，等待权重校验后提交。
- DROID查找范围补充：PPU15 from-sz、frank/datasets、Orion数据入口与现有共享数据候选浅层查询；PPU11独立盘 CPU job dlcopxs0bpegvav1 确认 from-sz/datasets 不存在，Orion入口只有UMI99。没有发现原DROID；这不是全盘递归穷举结论。
- Range慢连接先后保留进度停止并恢复到 dlc13ze96z5r9c4m、dlc17v9z4bornumq；当前后者继续最后一段。所有分片保留，最终合并后校验完整SHA256；尚无GPU训练提交。
- 下载最终在 dlctzx34bdu1jme3 输出 ALPHA_PPU_ASSET_READY：完整24813767464字节及SHA256通过，tokenizer/config就绪。
- 实际提交单卡闲时 smoke dlc1gr172j5edr7w，PPU15研究配额 ForceQuotaOverSold，最多40分钟。GPU smoke request保存gpu-smoke-request.json；没有提交正式训练。
- 补充DROID定位：工作空间ListDatasets实时TotalCount=0。关联PPU任务记录给出的OSS前缀 maxliu-storage-bench-sz-20261008-01a1198b/PRETRAIN_DATA_2.0/ 再查仍403 AccessDenied（RequestId6AED47B281A4C233338AE7E4）；不据此推断数据不存在或容器服务角色权限。
- 最后实时状态：dlc1gr172j5edr7w 仍 Queuing，创建时间2026-11-06T03:27:59Z，Position0，无Pod/运行日志。未执行模型加载、loss或backward，不能宣称GPU smoke通过。
- 当前研究子配额Ready、固定0GPU；父配额Ready，总64GPU，Allocated/Used48GPU，EmptyNodeNum2。配额数字不能替代闲时调度可用性。既往同研究闲时配额单卡job dlc1i5bswbm7ovc9确实Succeeded；当前没有足够证据认定请求构造错误，保持原闲时请求，不改固定资源、不重复提交。
- 单卡smoke运行上限40分钟；完成7次更新及验证后退出，没有自动启动正式训练的脚本。用户要求smoke后确认正式训练，当前正式训练未提交。
- 本轮源仓库未修改生产代码或正式配置；只增加本任务记录与独立准备/临时smoke脚本。语法检查3个关键脚本通过；真实ABC数据CPU准备通过；GPU验收等待调度。

## 排队原因复核
- 用户询问为何PPU空闲仍未调度。实时父Quota总64卡、Allocated/Used48卡、2个EmptyNode；研究子Quota固定0卡、提交1卡。smoke仍Queuing，GetJobEvents为空，无执行Pod，API只返回sync enqueue status from queueUnit status。
- 官方闲时说明 https://help.aliyun.com/zh/pai/use-idle-resources-in-dlc-jobs 明确灵骏闲时任务可使用租户其他同规格Quota闲置资源，不受本Quota资源总量限制。因此不能用子Quota固定0GPU直接解释排队，也不能把父Quota显示空闲16GPU等同实际闲时可调度容量。
- 已核对ForceQuotaOverSold、无节点绑定、1GPU/24CPU/200Gi，父/子OversoldUsageConfig.Disabled=false。没有确证具体拒绝条件；问题发生在创建Pod之前的调度阶段，不是模型运行失败。应由PAI调度侧查看Job dlc1gr172j5edr7w 对应QueueUnit筛选/准入原因。未更改资源类型、未重提、未启动正式训练。

## 固定单卡 smoke
用户明确授权固定资源单卡smoke，完成后确认正式训练。已撤销闲时job dlc1gr172j5edr7w并确认Stopped；父Quota quotawsqknfajqjt + ForbiddenQuotaOverSold提交dlc13zv40bs9egtn，约1分钟内完成调度。
- 实际设备ZW-M890P，Torch2.9.0；Alpha完整checkpoint已加载，冻结Wan VAE/T5，ZeRO2/bf16，W&B offline实际名称和模式已打印。
- startup validation有限：total1.0024435520，video0.2527595460，action0.7496840358，首次耗时93.5724秒，峰值allocated49.4935GB。
- 第一次optimizer更新在torch._foreach_sqrt分配18.63GiB时OOM；卡144GiB，进程132.17GiB，PyTorch allocated120.27GiB。无成功optimizer step，不能宣称smoke通过。
- 单卡重试仅改optimizer CPU offload（与既往H800单卡方案一致），保持模型/目标/输入/超参，正式16卡仍计划no offload。新job dlcljksr580c63gy，单卡固定、最多40分钟。输出独立子目录 /mnt/workdir/frank/runs/openwam-alpha-ppu-smoke-20261106/cpu-offload；W&B ID/name保持同一smoke身份。未提交正式训练。

## 固定单卡 smoke 最终结果与待确认正式配置
- dlcljksr580c63gy 已 Succeeded：UTC04:08:41开始、04:12:27结束（北京时间12:08:41–12:12:27）。输出 OPENWAM_ALPHA_PPU_SMOKE_PASSED；脚本逐步断言DeepSpeed global_steps等于1..7，真实完成7次更新；7组ABC各1train窗口，6个原val窗口结束验证全部loss有限。
- CPU offload + batch1/accum1，LR1e-5，ZeRO2/bf16。首次train7.2960s，step2..7平均3.6955722s；峰值allocated44.963216384GB。首次validation95.1232s包含冷启动开销；不以该样本预加载/CPU-offload吞吐推算16卡全量ETA。
- 六个结束val窗口算术平均 total3.21927157 / video0.19186090 / action3.02741071。仅烟测数值，不是正式验证集指标，不比较开始/结束的单样本随机噪声loss来判断收敛。
- 持久结果 /mnt/workdir/frank/runs/openwam-alpha-ppu-smoke-20261106/cpu-offload/result.json、metrics.jsonl、config.yaml、wandb/；本地完整结束日志 completed-cpu-offload-logs.json。W&B offline owma-ppu-smoke1106，实际display name已验证。
- 正式拟沿用：16卡PPU闲时（固定授权仅本次单卡smoke）；Alpha step154000初始化Video+Action DiT；冻结native Wan VAE/T5；mono-left；DROID+全部七组ABC有效窗口，原val划分，train-only norm，自然窗口比例1epoch；stride0.8s，17RGB10Hz/34action20Hz；每卡batch2*accum4*16=global128；Video/Action peakLR1e-4，warmup500updates从1e-5开始，cosine至1e-6；AdamW betas0.9/0.95、wd0.01、clip1；bf16/ZeRO2/gradient checkpoint、no optimizer offload；启动/结束/每500updates分别验证每source128窗口；完整恢复checkpoint启动/结束/每2500updates；W&B offline、所有日志与恢复状态持久化。
- 尚未完成：真正DROID在PPU的来源/路径，七组native-crop ABC全量reader与索引适配及train-only全量norm、16卡通信/吞吐验证。七组ABC现统计100028段（train98792/val1236），不是原11764段双目train子集；不沿用旧3206442窗口和三天ETA。此次仅真实ABC单卡计算smoke，不宣称DROID+ABC联合正式开训就绪。
- 依用户要求停止在此，正式训练未提交；等待用户确认正式配置/后续资源安排。

## 8卡无offload smoke（用户新要求）
- 用户要求不为单卡显存调整正式参数，改用多卡（例如8卡）smoke，DROID稍后提供。继续仅做ABC硬件/配置验证，不启动正式训练。
- 提交 dlcyradcbwd9y8uc：固定PPU15单节点8卡、192CPU/1600Gi、40分钟上限；无CPU optimizer offload；原每卡batch2/累积4、bf16/ZeRO2/gradient checkpoint、Video+Action peak1e-4、warmup500、beta0.9/0.95、wd0.01、clip1不变。8卡global64，与16卡global128的区别明确保留。
- Smoke-only harness复用原trainer.build_optimizer/build_lr_scheduler/prepare_accelerate；删除单卡脚本手工LR覆盖。仅使用已核验7个ABC真实train窗口重复512次作为有限硬件负载，8个optimizer updates/32个microsteps；原6个val窗口检查前后有限loss。临时样本norm仍不用于正式训练。
- 逐rank验证ZeRO2分片数8、实际batch2/accum4、no CPU offload、betas、完成8次真实updates，以及Adam exp_avg/exp_avg_sq在GPU。记录全部rank峰值allocated/reserved、有效LR与耗时。
- 输出 /mnt/workdir/frank/runs/openwam-alpha-ppu-smoke-20261106/eight-gpu；独立8GPU smoke W&B offline ID owma-ppu8-smoke1106，name OpenWAM ABC Alpha Wan mono-left PPU 8GPU smoke。未改正式配置或生产源码；本地源fetch后HEAD仍f4668f9，原dirty文件保留。临时脚本语法检查通过。

## 8卡无offload最终验收
- dlcyradcbwd9y8uc = Succeeded，UTC06:14:35–06:19:29（北京时间14:14:35–14:19:29）；OPENWAM_ALPHA_PPU_8GPU_SMOKE_PASSED。
- 8rank全部完成32microsteps/8optimizer updates。ZeRO2 partition_count=[8,8,8]，per-rank batch2、accum4、global64；CPU Offload=False；更新后逐rank断言Adam exp_avg/exp_avg_sq均在cuda，全部通过。无OOM、无NaN/Inf。
- 所有rank最大PyTorch allocated40.5424614GB、reserved45.0447273GB（不是NVML整卡总占用）；首次更新含冷启动74.9338s，update2..8平均4.4667057s/update，即1.1166764s/microstep。预加载重复样本吞吐约14.33sample uses/s，不当作全量数据读取或16卡跨节点吞吐。
- 真实LR从1e-5开始，8updates后1.144e-5，精确对应500updates线性warmup至peak1e-4。未将单卡smoke的constant1e-5/CPU offload带入多卡；正式配置无需为此次显存问题降低batch、分辨率或更改冻结范围。
- 最后train total1.22206748、video0.15612777、action1.06593966；结束两次跨rank验证total1.21433592/1.80043244，均有限。因仅七个train窗口重复、临时norm、随机噪声，不宣称正式模型收敛质量。
- 本地原始日志 completed-8gpu-logs.json；持久结果 eight-gpu/result.json、metrics.jsonl、config.yaml、wandb/。原固定单卡CPU offload及失败OOM记录保留，无删除。
- 任务已自动结束，正式训练未启动。等待用户提供DROID，之后仍需全量reader/index/norm准备。此次单节点8卡验收不替代双节点16卡通信或全数据覆盖验收。

## 用户提供 DROID archive 路径后的检查
- 用户提供 /mnt/oss/from-sz/datasets/cpfs-archive/frank/tokenizer-h100-20261019/droid-prepared-20260920/，授权全量数据准备与smoke，正式训练仍不启动。
- 2026-11-06实时身份frank；CPU只读job dlc1kzk0zv4i6xiy 已输出DONE：在本任务PPU15 CPFS挂载中，/mnt/oss不存在，提供的路径不存在，替换为/mnt/workdir的同相对路径也不存在。这仅证明当前任务挂载不可见，不证明源数据缺失。
- 当前空间ListInstances返回空；无法从DSW定位/mnt/oss挂载来源。尚未获得该挂载实际oss://bucket/prefix，不猜测Bucket或扩大扫描。
- 未读到DROID文件，不能判断是否压缩、trajectory.h5是否齐全，也没有提交全量数据GPU smoke。当前reader需要frames.lance及实际state轨迹trajectory.h5，视频tokenizer资产不自动等于动作监督就绪。
- 本地origin fetch成功。当前f4668f9的MixtureDataset与最新origin/main细节有差异，但proportional的采样语义一致：按源有效窗口数分配虚拟索引，整体随机打乱，无严格源交替和50/50重采样。正式config configs/dataloader/midtrain_droid_abc.yaml 明确proportional；本轮未改源码、配置或训练参数。
- 待补充/mnt/oss对应OSS URI，或具有该挂载的作业/实例ID，以完成最小只读挂载并继续数据检查。请求保存在本地inspect-provided-droid-request.json。

## 2026-11-06 继续定位OSS并恢复DROID
- 上轮ListJobs默认ShowOwn=true仅见本人75个作业，结论未充分覆盖其他作业。本轮显式ShowOwn=false，返回工作空间可见244个作业，分页核对三页。stars-carry-progress等作业明确/mnt/oss -> oss://discover-pai-wulanchabu.oss-cn-wulanchabu-internal.aliyuncs.com/。只读挂载成功，dlc1b05o238zm805实证用户目录存在。
- manifest检查dlc1s198rfta1imh完成：episode-tar-v1，226shards，1935606620160归档字节，1929885577171源文件字节，2125064files。PPU15 free8057337675776，PPU11 free10762781196288；两盘同时只读挂载核对，PPU11没有from-sz数据入口。
- dlcqh14azfc7xj8j顺序检查所有226份files.jsonl.gz：.h5/.hdf5数量均0，无action audit/recovery；有training-manifest-20260921-r2。归档不是可直接随机读的Lance目录，当前reader需展开；禁止将纯图像归档当作监督齐全。
- dlc12ejnrfa6vi3j首分片展开完成，1122files逐文件SHA256校验；目标/mnt/workdir/frank/datasets/droid-prepared-20260920-ppu，原OSS只读，未删除/覆盖文件。
- dlchbfx8rv4q5uwn继续顺序展开shards2..226，0GPU/4CPU/16Gi，最长720min；每个文件路径约束、拒绝link、exclusive create、size/SHA256核对。已见真实RESTORE进度，未完成全量。代码/请求保留本机ppu-requests/openwam-alpha-midtrain-20261106/restore-droid-remaining*。
- dlc1o1nzm3lye0fr定向浅查OSS datasets目录及PPU11数据入口，未见原DROID state目录；不构成全盘缺失证明。其他用户GetJob详细命令查询返回RAM/RBAC拒绝，未尝试绕过。
- H100本人SSH身份Frank确认。既有prepared-stride08-20261104/droid.sqlite、droid-stats.json、审计jsonl和droid-state-supplement仍在；原始trajectory.h5三条实测可读且大小0.77–1.59MB。没有提交H100任务、修改预约/模型卡或启动训练。
- 单个4KB stats通过scp读取到本地成功；本机CLI向已知OSS本人目标prefix上传被403 bucket ACL拒绝（RequestId6AED9527BEE0E531350FF0B9）。未改权限或重试绕过。该上传失败不影响已授权的PPU服务角色只读归档访问。
- dlc1y1aryj4j75nj检查PPU访问DROID官方GCS精确trajectory对象元数据，等待结果。此阶段未提交GPU full-data smoke，更未启动正式训练。
- 官方精确对象访问已通过dlc1y1aryj4j75nj：AUTOLab样本size1351635，generation1710756229283371，MD5可用。无需绕过OSS ACL，改从DROID官方公开源获取原始state文件。
- dlc15qhlhn2uacaa已提交：0GPU/8CPU/16Gi、最长1440min，读取原manifest限定DROID episode集合，等待本次RGB restore供给对应result.json，按其原episode路径从官方GCS取trajectory.h5或原trajectory_im128.h5；保留原文件及commanded字段，以对象generation固定版本并验证size/MD5。8路下载；没有更改监督合同。失败/缺失逐条记录，不能自动当全量就绪。
- state输出/mnt/workdir/frank/datasets/droid-state-20261106；receipt/summary在/mnt/workdir/frank/runs/openwam-alpha-ppu-smoke-20261106/data-preparation/droid-official-state-*。源与请求下载脚本保存本机download-droid-states*。图片展开和动作补齐仍在准备阶段，正式训练未启动。
- 首次健康检查：restore已完成shard6，最近每shard66–70s；state downloader解析原manifest共72557个唯一episode，前100条全部verified，215753327bytes/42.6s、无error。资产集合不是最终监督验收后的训练集合，后续需核对既有审计及排除记录。估计仅供准备阶段参考，不是训练ETA。全量数据smoke仍未提交。
- 17:16北京时间状态：轨迹2400/72557 processed、2396 verified、4 errors，4773583033bytes/1023.5s，有效4.66MB/s、2.345episodes/s；按当前平均剩余8.31h，约11月7日01:35，不含错误恢复/归一化/索引/smoke。展开已到23/226，近期每片49–67s，估计剩余3–4h。只读probe dlcq7bcwbrhaj3sj核实4条全部为URLError(Connection reset by peer)，不是已确认缺失监督；保留待重试，不能将它们直接排除。速度为端到端有效速度而非链路带宽；当前8路下载，每episode对象查询+数据读取+MD5。未改变任务参数或停止健康作业。

## 轨迹下载加速（用户要求）
- 原8路urllib下载每请求重建连接、按输入顺序回收结果；先前4.66MB/s是端到端有效速度，后期出现连接重置和慢请求。用户授权优化速度。
- 新任务脚本download-droid-states-fast.py：32线程、每线程requests.Session复用连接、有限transient retry、完成即回收进度；复用source.json中已验证且大小匹配文件，保留旧未完成文件，新补传文件使用generation绑定路径，继续size/MD5校验，不改数据/监督范围。CPU8/16Gi/0GPU不变。
- 本地AST语法检查通过。停止旧job dlc15qhlhn2uacaa并回读Status/Pod均Stopped后，提交dlc1gadekranvcg3，避免同目录并发写。未停止健康RGB restore dlchbfx8rv4q5uwn。新回执/总结使用droid-official-state-fast-*，不覆盖旧回执。
- 新job已提交，等待实际吞吐验收；未承诺速度或缩短ETA。正式训练未启动。
- 加速验收17:32：dlc1gadekranvcg3真实下载532条/992945770bytes/63.1s，3168条旧verified复用不计下载字节；新下载均verified，当前无error。端到端新传输平均15.74MB/s，最近100条窗口13.52–21.34MB/s，较原4.66MB/s约3.4倍。32线程保持CPU8/16Gi/0GPU，未继续扩大并发。按当前8.43新段/s，剩余轨迹纯下载约2.3h，但后续依赖RGB展开供给result.json，整体准备ETA不能早于展开完成；不包含最终数据索引、norm、smoke。

## 图像展开并发提速（用户要求）
- 最新串行日志确认shards39/40/41逐文件验证完成，耗时85.9/94.5/83.0s，约90–103MB/s。用户要求增加并发避免阻塞轨迹元数据供给。
- 新restore-droid-parallel.py使用8个ProcessPool worker，每个处理独立分片；0GPU/8CPU/16Gi。source manifest、路径边界、成员类型、size及SHA256校验保持。前41片按已验证日志复用；42及之后若已有完整文件则重新校验复用，未完成/哈希不符文件逐一改名保留.interrupted-before-parallel，不删除、不覆盖旧文件。
- 本地AST通过。旧dlchbfx8rv4q5uwn经StopJob并回读job/pod均Stopped后，提交dlc1me7a4r9jptch；无同文件双写。32路轨迹作业dlc1gadekranvcg3未中断。
- 每片持久receipt在runs/openwam-alpha-ppu-smoke-20261106/data-preparation/image-restore-parallel/part-NNNNNN.json，完成写complete.json。当前等待实际并行吞吐，未声称全量完成或smoke通过；正式训练未启动。
- 17:42实测：并行任务155.06s内完成8个分片（42、48、45、47、49、46、50、43），总校验68.537GB，其中42复用7552个文件。保守完全扣除42的8.562GB后，全新展开59.975GB/155.06s=386.8MB/s，约为旧串行90–103MB/s的4倍。余177片按初速估计1–1.5h；不是硬保证，不含后续索引/norm/smoke。所有完成分片SHA256通过，无已观察错误。首批8个完成不等于最初编号42..49全部完成（44仍可能在处理）。
- 17:54并行提速复核：dlclrroqzx3xjnla确认轨迹frontier在shard23，后续100个result.json均存在，当前不是图像元数据饥饿。dlc1lkbb4bj8gl87用同128条已验证真实对象只读复测：16路166.56s/1.306MB/s/0error；64路189.56s/1.148MB/s/0error。更多并发没有改善该时段实际吞吐，保持原32路生产下载，不盲目重启。该probe使用镜像系统requests（有依赖版本warning），未改运行环境；不能单凭此认定具体公网限速来源。图像并行52片新完成、余133片，443.345GB/848.36s；扣除42整片复用仍约512MB/s。近期轨迹生产约1MB/s，先前15.7MB/s和2–3h仅短时外推已失效。按低速持续仅剩余轨迹约30–36h；全量ABC reader/index/train-only norm尚未完成，完整smoke ETA目前无法可靠承诺。此前图像1–1.5h是数据展开估计，绝非开smoke承诺。未启动正式训练。
- 用户要求定位H100/阿里云可上传OSS的轨迹源。H100实时读取既有审计72457条，其中72413直接通过；恢复JSONL必须按id取最后一条（旧needs_review已被最终恢复覆盖），最终28 recovered、16 exclude_missing_state。生成明确72441文件的source/archive_path清单：/gpfs/jiuquyun/projects/Frank/openwam-midtrain-20261101/droid-state-oss-upload-manifest-20261106.jsonl（18555039bytes），本地备份h100-droid-state-upload-manifest.jsonl。只对4条精确轨迹stat核实存在/大小，没有宣称重新全量校验文件。原始源/gpfs/jiuquyun/datasets/DROID/raw/1.0.1，补齐源本任务datasets/droid-state-supplement。未启动H100计算/打包任务或OSS上传，未停止PPU现有准备。

## H100轨迹分片打包（用户明确要求，完成后交路径）
- 用户授权按已定位轨迹清单尽快打包，同事负责上传OSS。本轮不上传、不申请GPU、不新增训练实验/预约。
- H100身份Frank确认，GPFS可用约11TB；CPU QoS可用。操作脚本package-h100-droid-states.py AST通过，sbatch bash-n/test-only通过，提交CPU作业53675（8CPU/16Gi/0GPU/2h），实际RUNNING于xn01-gpu1-0046。
- 精确72441文件先stat确认存在/大小，按约4GiB分片，8线程写无压缩tar。原始HDF5不更改；写入中检查source size/mtime未变化，流式计算每片SHA256，完成后遍历tar headers核对所有文件名/长度。完成片由.partial改名.tar，全部成功才生成SHA256SUMS和COMPLETE.json。
- 输出/gpfs/jiuquyun/projects/Frank/openwam-midtrain-20261101/droid-state-oss-package-20261106；日志/gpfs/jiuquyun/home/Frank/logs/droid-state-oss-package-53675.out。前10000文件stat13.1s，总大小19958781762bytes；仍需完整大小和实际首批吞吐给ETA。
- 打包完成：53675及所有step COMPLETED/0:0，Elapsed2m55s。72441files、源138264316489bytes（128.77GiB）、33个约4GiB tar；总耗时173.59s（含清单stat约84s），纯打包约89.2s、1.55GB/s。每片流式SHA256、源size/mtime稳定性检查、全部tar成员名和长度核对通过；最终COMPLETE.json与SHA256SUMS已生成。交付路径为上述droid-state-oss-package-20261106目录；用户同事负责上传，尚无OSS上传完成证据。未停止其他准备作业或启动正式训练。

## 纠正：复用既有1.9GB无损state训练包
- 用户提供另一聊天核验结果，指出已有stride08-state-fields.tar.gz足够本次训练。读取原pack_stride08.py确认逐文件仅提取reader消费的四字段，dtype/shape/tobytes逐项与原H5相等，保留source_path；索引/归一化随包携带。此前把保留原始HDF5全字段当作开训门禁是不必要的，导致138GB打包及公网逐条补齐工作；原始全字段仍保留H100即可。
- H100原包路径/gpfs/jiuquyun/projects/Frank/openwam-alpha-midtrain-20261104/cloud-transfer-stride08/stride08-state-fields.tar.gz，1900408325bytes，SHA2560eadf3c3cdc624a2c8f25a4f383c7eb46ef5e34817e2163b626ada03d140f615。另有既有zstd版本，但云端接收的是gzip，不重复搬运其他版本。
- 恢复本人既有CPU-only DSW dsw-4gpdclcrpjz496v51r（0GPU/8CPU/32Gi），实时SSH映射120.77.254.228:1024。源云盘/mnt/workdir/frank/projects/openwam-alpha-midtrain-20261104/cloud-transfer-stride08已存在完整gzip、h100-state-view、prepared、ready.json。ready实读72418states，DROID train1667932/val86756/test92571，stride0.8。由此更正过去仅传输成功/验收未知的旧记录。
- 复用PPU本人既有transfer-key，只导出公钥追加到本人接收容器authorized_keys；未复制/打印私钥。SSH host key从本机已信任known_hosts取得并pin，无关闭主机校验。未修改共享网络或权限策略。
- dlcaokkjvq4m6725：0GPU/8CPU/16Gi/120min，SSH流式迁移gzip到/mnt/workdir/frank/datasets/droid-state-fields-stride08-20261106，完整size/SHA256验证后展开；逐episode核对小包覆盖sqlite所有state引用，训练/验证各1条原droid_targets检查有限值和缺失臂mask。结果存runs/openwam-alpha-ppu-smoke-20261106/data-preparation/compact-import/ready.json。包中旧ABC stats不得用于当前7组ABC。
- PPU完整原始轨迹公网下载暂保留，待小包接收/校验/动作reader通过再停止。此次无正式训练。
- 迁移真实进度：dlcaokkjvq4m6725 RUNNING，已接收67108864/1900408325bytes，实测1.43MB/s，剩余1278s（约21min）。传输后自动执行已编排的哈希、解包、全索引state覆盖及train/val动作reader检查；此ETA只针对DROID小包，不是七组ABC全量smoke就绪时间。
- 用户要求确认停止公网原始轨迹下载并优化小包传输。已StopJob dlc1gadekranvcg3并回读Job/Pod均Stopped，已有完整/部分文件保留。小包dlcaokkjvq4m6725继续。dlcr1y0p783v6nvk进行仅16MiB双SSH只读探针，两流耗时11.7/18.79s、合计0.893MB/s，同时主传输降速，未显示叠加吞吐收益；符合共享入口带宽瓶颈，不增加并发或重启已传数据。18:29:51小包402653184/1900408325bytes（21.2%），累计1.2MB/s、剩余1246s（20.8min）。ETA是滚动估算，不是21min硬上限；传完还有校验/解包/覆盖/动作reader验收。

## 19:10 全量数据接入进展

- DROID 小包导入作业 `dlcaokkjvq4m6725`：1,900,408,325 bytes，SHA256 `0eadf3c3cdc624a2c8f25a4f383c7eb46ef5e34817e2163b626ada03d140f615` 通过；72,418 state 文件覆盖全部 stride0.8 索引。train 1,667,932 / val 86,756；train/val action-only reader 通过。资产位于 `/mnt/workdir/frank/datasets/droid-state-fields-stride08-20261106`，ready 位于运行目录 `data-preparation/compact-import/ready.json`。
- DROID 图像恢复 `dlc1me7a4r9jptch` 完成，其余 185 shards 加此前 41 shards 全部逐文件校验完成。公网原始 HDF5 下载 `dlc1gadekranvcg3` 已停止，无需等 138 GB 原始字段。
- 七组 ABC 全量 metadata：100,028 episodes，train 98,792 / val 1,236；0.8s train 11,043,362 / val 152,606 windows。无 admission exclusions。输出 `/mnt/workdir/frank/projects/openwam-alpha-ppu-midtrain-20261106/prepared-nativecrop-full-20261106`。
- ABC 首个归一化 `dlcmwdxxnjkred4y` CPU OOM；原因为每个 worker 将完整 parquet 转成 Python rows。仅改为 parquet episode filter、复用已完成分块并分配 64Gi CPU 内存，16 worker；重试 `dlc1xshbm3ryjc53` 已输出 ABC_FULL_NORM_READY，336.5s。原 Orion 插值、phase、相对 anchor、Welford 语义保持，统计覆盖全部 train anchors；不使用旧 ABC 或七窗口临时 norm。
- full reader CPU 作业 `dlc107x9fvsjl455`：PPU15 0GPU/16CPU/64Gi，30min 上限。新增当前任务 ops adapter `full_data_reader.py`，按实际 native_crop 元数据恢复黑边，沿用既有 mono-left mosaic / Orion action transforms；DROID 仅映射原路径，源数据只读。复用 StereoTok committed `7dae078f7509b2e9c7c69ed8162a62263a99a077` 的 LanceReader/manifest，无生产仓库修改。个人环境安装 pylance0.39.0 --no-deps，未替换 PPU Torch。
- 下一步 CPU train/val 首中尾与七组 reader 检查通过后，8卡 fixed/no-offload 联合 full-index loader smoke。共 12,711,294 train windows，原 MixtureDataset 自然比例索引，取前 512 个 shuffled windows 做 8 updates；并非完整 epoch 或全部视频扫描。正式训练仍禁止启动，后续按实际吞吐重新估算全七组预算。

### 19:12 全量 reader 通过并提交联合 smoke

- CPU `dlc107x9fvsjl455` 输出 `FULL_READER_READY 23 5.0286s`；DROID train/val 首中尾、ABC 七组 train 及已有六组 val 的首中尾/组入口真实解码通过。不是全量视频逐窗口扫描。DROID 缺失臂 mask 为 false，ABC padding mask 按原 Orion timeline 保持。
- 8 卡固定 PPU15 联合 smoke 已提交 `dlc1tcqmmjpy0qcm`，最后状态 EnvPreparing。资源 1 node×8GPU/192CPU/1600Gi，最长40min；ZeRO2，无 optimizer offload，per-rank batch2、accum4、global64。完整索引 natural proportion shuffle seed42，前512窗口，8updates，两个源各做开始/结束验证；正式计划16卡global128未改。
- Ops harness：本机 `C:/Users/Frank/AppData/Local/AliyunCLI/ppu-requests/openwam-alpha-midtrain-20261106/ppu-smoke-8gpu-full.py`，远端项目 `smoke/ppu-smoke-8gpu-full.py`；输出 `/mnt/workdir/frank/runs/openwam-alpha-ppu-smoke-20261106/eight-gpu-full-data`。保留此前样本 smoke 输出，W&B offline新ID `owma-ppu8-full1106`，明确 full-data smoke display name。
- LR scheduler 使用本次完整16卡计划的99307updates horizon、原warmup500，而非把8-step smoke当作完整schedule。验证start/end由harness显式运行，formal every500逻辑未启动。
- 小包导入完成后，已调用 StopInstance 停止本任务深圳传输CPU实例 `dsw-4gpdclcrpjz496v51r`，API成功；持久CPFS文件保留。
- 无正式训练提交。GPU smoke 结果、吞吐和峰值显存待真实日志确认。

### 19:24 联合 smoke 的 fork 阻塞定向修复

- 接手后两次只读确认 dlc1tcqmmjpy0qcm 仍停于 start validation（DROID total1.81151199 / ABC total1.56243420），未出现 optimizer update；日志为 Lance not fork-safe 警告。脚本先执行 Lance validation，再创建 DataLoader 的 fork workers，符合已知不安全调用顺序；尚未获得进程栈，暂不把推断写成已证实死锁。
- 已 StopJob，回读 Job/Pod 均 Stopped 后才提交串行重试 dlcjkr8qzfjjhe81。仅本任务 ops harness 的 num_workers4/prefetch2 改为 num_workers0，规避 fork；模型、数据/采样、512窗口、32microsteps/8updates、8GPU/192CPU/1600Gi、40min上限不变，无扩资源或正式训练。
- 新请求 gpu-smoke-8gpu-full-nofork-request.json，脚本 ppu-smoke-8gpu-full-nofork.py；AST语法通过，CLI dry-run通过。独立持久输出 /mnt/workdir/frank/runs/openwam-alpha-ppu-smoke-20261106/eight-gpu-full-data-nofork，W&B offline ID owma-ppu8-nofork1106，display name 保持实验名。旧文件/输出保留，无删除。
- 当前 RAM frank、workspace308566、父quota Ready及角色权限已核验。StereoTok HEAD7dae078、OpenWAM HEADf4668f9及各自origin fetch成功；用户原有dirty文件未改，无commit/push。仅运行已部署的精确源码快照与本任务ops脚本，不上传本地dirty生产代码。
- 新作业已创建，真实optimizer进度与最终验收待后续日志。

### 19:30 联合 full-index smoke 最终通过

- dlcjkr8qzfjjhe81 Job/Pod Succeeded，ExitCode0/Completed，北京时间19:25:05–19:30:11；日志包含 OPENWAM_ALPHA_PPU_8GPU_SMOKE_PASSED。32microsteps/8optimizer updates全部完成；GPU_OPTIMIZER_STATE_CONFIRMED共8次，Adam状态在GPU。实际512窗口=DROID66+ABC446，非旧七窗口重复测试。
- 单进程加载消除了此次阻塞；强支持先读Lance后fork导致旧harness卡住的解释，但未获取旧进程栈。正式入口已有spawn配置，本次仅修复独立smoke harness；不能直接把无main guard的脚本改为spawn。旧job/output保留。
- 首update76.0213s含冷启动；update2..8为6.4715/5.0971/6.6836/5.0592/5.7732/5.5319/5.6282s，平均5.74924s/update、1.43731s/microstep、11.1319windows/s（8GPU，包含本次单进程真实读取）。全部rank最大PyTorch allocated40.54246GB、reserved45.04473GB；不等于整卡NVML占用。
- 最终混合train total1.5853213/video0.1303810/action1.4549403。结束分源validation DROID total2.1562300/video0.2440917/action1.9121383；ABC total0.4983127/video0.1271961/action0.3711166。均有限，无OOM或NaN/Inf。每源start/end各8窗口且前后样本不同，不据此判断收敛或loss改善。
- LR由1e-5至1.144e-5，符合500update warmup。模型Alpha154000，冻结native Wan VAE/T5、ZeRO2/bf16/gradient checkpoint/no offload均不变。
- 完成标记在result.json写入及W&B finish之后；持久结果/metrics/config/offline日志位于 eight-gpu-full-data-nofork。本地原始返回 completed-8gpu-full-nofork-logs.json。没有额外创建CPU检查作业；日志及成功脚本路径提供本次验收证据。
- 正式待确认：16GPU（历史拟闲时）、每卡batch2×accum4=global128；全12,711,294窗口自然比例1epoch约99,307updates；0.8s stride、17RGB@10Hz/34action@20Hz、mono-left、实际state、原划分及train-only norm；Video/Action LR1e-4、warmup500从1e-5、cosine至1e-6；AdamW(.9,.95)/wd.01/clip1；每500updates验证各源128窗口，加start/end；每2500updates完整恢复checkpoint，加start/end；W&B offline沿用用户明确选择。
- 若16卡理想线性扩展并维持5.749s/update，纯训练约158.6h=6.61天；8卡同吞吐外推317.2h。仅7个暖机后updates，未验证双节点通信或长期I/O，未含验证/保存/排队/抢占。原三天预算不受当前实测支持；正式总ETA尚不能保证。
- 验收局限：512窗口覆盖测试不等于全量视频解码；8卡单节点不替代16卡双节点验收或完整恢复测试。正式训练未提交，待用户确认配置与时间预算。无生产代码修改、无commit/push，任务记录已同步。

### 2026-11-06 用户纠正ABC对照范围与时间重估

- 用户明确：Wan与StereoTok必须使用同一ABC双目数据子集，Wan只取这些样本的左目；不是全部可用左目数据。此前PPU全七组98,792 train episodes/11,043,362窗口扩大了对照范围，不能作为正式数据合同。全七组smoke只保留硬件/读取验证意义，其ABC norm和99,307update schedule不用于该对照。
- 本轮仅只读核验H100并重估，未修改数据、reader或正式启动配置，未提交训练。H100实际SSH身份Frank。原 prepared-stride08-20261104/abc-admission.json 当前实读：train11,764episodes/1,538,510windows，val172episodes/21,805windows；camera_filter剔除87,028train episodes。原abc-stats.json统计episodes11,764、anchors1,538,510、stride8RGB ticks。源码orion_midtrain.abc_source保留原split后筛选头部真实双目子集，mono-left只决定输出视角。
- 因此匹配原合同总窗口=DROID1,667,932+ABC1,538,510=3,206,442；global128时约25,051optimizer updates/epoch。后续正式准备须按相同episode/window划分复用或核验子集norm，不能只把窗口总数改小，也不能按全部七组norm训练。
- 直接读取原H100 Job53358日志 /gpfs/jiuquyun/home/Frank/logs/openwam-droid-abc-alpha-wan-53358.out。按进度记录的elapsed对同一步多rank取中位数：microstep200→1200为1127.5s/1000，500→1300为904s/800，1000→1300为338s/300；稳定约1.13s/microstep，accum4即4.52s/update，16卡global128约28.32windows/s。日志多rank输出存在拼接，采用三个一致区间而非畸形最大step值或单条瞬时速度。
- 当前8PPU全七组smoke稳定5.74924s/update、1.43731s/microstep、global64约11.13windows/s。步耗时约H100的1.272倍（+27.2%），不能说硬件慢一倍；卡数、数据混合比例、单进程vs生产加载器不同，不是严格硬件对照。
- 同3,206,442窗口外推纯训练：16H100约31.45h；16PPU若双节点维持8卡实测步耗时约40.01h；8PPU按原batch2/accum4即global64约80.01h。8卡方案改变global batch，不是默认正式建议。以上未含验证、完整checkpoint、启动/排队/抢占；PPU尚未完成目标双目子集与双节点吞吐验收。先前6.6天属于已废弃的全七组范围。

### 独立审核训练计划

- 用户要求提供详细训练plan给另一session审核，已新增 `26-11-06-openwam-ppu-stereo-subset-training-plan-review.md`。包含目标子集合同、原H100及PPU准确路径、来源SHA、已验证/待验证边界、配置step单位、资源/恢复/保存要求、分阶段验收和时间估计。
- 本轮只做只读核对及文档编辑，无生产代码/数据变更、无作业提交、无commit/push。计划中的新目录/正式请求均标注尚未创建，正式资源及启动仍待批准。

### 20时段 目标双目子集恢复与16卡加速准备（进行中）

- 用户授权完成ABC精确双目子集/原norm/周期修正，并使用16卡做不改变训练语义的加速测试；正式长训练仍未授权启动。收到独立审核关于index%3相位偏移的阻塞意见，已纳入数据验收，未跳过。
- 本机OpenWAM HEAD仍f4668f9，origin fetch成功；frank跟踪SHA相同，原有dirty源码与测试未改。使用本任务ops文件与已部署精确源码，不commit/push。
- 原H100 metadata导出11,936双目episodes，完整source_episode_id/source_id/length/split顺序摘要匹配后才构建PPU目录。原reader实际构建再次核对train/val episode order哈希，与导出一致；不是只验证窗口总数。
- PPU原生分组metadata当前右目available只有3,249条，不能用其直接选原双目子集。采用原H100身份序列定位；短identity key仅用于查找，候选必须唯一，随后完整有序身份的SHA256核验，不以短hash替代最终身份验证。
- 大清单内嵌首次被DLC参数65536字节限制拒绝（服务端BadRequest，无GPU提交）。改为两个小CPU任务dlc15s8t1n762u4u、dlc1628effs9tr7i传输身份key，均Succeeded。metadata只读探针dlcpwwejo78ok7xv Succeeded。没有数据集全量复制或删除。
- CPU准备dlc1tdaakrb6zsi5 Succeeded：目标目录prepared-stereo-subset-stride08-20261106，train11,764/1,538,510，val172/21,805；DROID不变，总3,206,442。ABC stats原字节SHA256 5ba8156504fd0a0632dff2ac87d99487a17c82ca72914f38c86065b9a2c1f57c。重算子集split_window_start，保留原顺序与index%3定义，不沿用七组起点。
- ops stereo_data_reader.py复用既有reader，仅切到目标目录；prepare-stereo-subset.py保存映射。ready.json目前仅表示集合/基础reader通过，不等于跨H100逐样本一致或正式ready。
- H100参考h100-stereo-reference.json记录全量(原episode_index,anchor,phase)指纹以及10条原reader真实样本。train全窗口SHA256 5910bcb086325d1ce0572486b80897941ce0d834040fece16b2020aef5706d08，val20c4874f03bd21dfdf233384243622c3a3444cd7aacf76f2bd0b657064c9898e。
- 当前CPU精确parity job dlcg9cyyzosgm3o2：全量窗口身份及10样本action/proprio/masks/RGB/prompt哈希检查。通过才写parity-ready.json；GPU benchmark入口强制要求此标记。未完成时不提交16卡。
- 加速候选优先关闭gradient checkpointing，保持batch2/accum4/global128和逐step随机seed。拟16卡同数据各8updates的baseline/候选配对，horizon25051；检查完整本地FP32 optimizer partition及Adam状态SHA256与loss一致性。尚未提交，不将候选视为已验证。

### 独立审核的phase问题：全量身份已修复，像素一致性门禁仍阻塞

- 精确parity任务dlcg9cyyzosgm3o2按预期fail-closed退出1：train1,538,510和val21,805个窗口的(原episode_index,anchor,phase)全量SHA256与原reader一致；10样本mask/prompt一致，但RGB与action/proprio字节哈希不同，未写parity-ready.json。原ready.json只表示集合/基础解码，不代表最终就绪。
- CPU量化任务dlcd7iubvv9glssx Succeeded：train index0 action最大绝对差2.145767e-6、proprio4.768372e-7；val index21804 action1.579523e-6、proprio7.152557e-7。两样本source_frame_indices完全一致，frame timestamps最大误差0。未直接放宽浮点门禁。
- RGB定量是在首帧及第8帧、每4像素采样的mosaic位置上计算：train0 MAE0.53359375/max15/changed28.78%；val21804 MAE0.2097873/max12/changed12.54%。不是整数据集图像误差估计。native-crop-v2目录重封装与原H100原图像不能宣称逐像素一致；具体编码/解码成因尚待进一步确认。
- H100 Orion实际HEAD783b602且clean；两端action.py/canonical.py原始SHA不同仅因PPU为CRLF，已用该commit内容的CRLF SHA精确匹配，排除这两文件的算法版本漂移。跨Torch版本/计算批形状可能导致微小浮点差，但尚未将其定为唯一根因。
- 原H100双目左目视频实际源 /gpfs/jiuquyun/datasets/PRETRAIN_DATA/ABC-130k-yam-canonical-v1/videos。PPU /mnt/workdir/from-sz/datasets/PRETRAIN_DATA_2.0/ABC-130k-yam-canonical-v1 与OSS同路径均不存在（dlc12ghw5navitt5 Succeeded）；继续只读查PRETRAIN_DATA旧前缀。没有递归扫描、视频迁移或GPU提交。
- 16卡加速测试仍被parity-ready门禁阻止；不能因用户已授权算力而跳过目标数据一致性。当前未达到ready-to-train。

- 后续定位dlcjlawp7c62zpa7确认PPU/OSS的PRETRAIN_DATA旧canonical精确路径也不存在；仅浅列from-sz/datasets前缀，不是全盘不存在证明。
- H100精确双目子集左目文件清单已只读生成：11,936个视频文件，总43,672,728,819bytes（40.6734GiB），包括train/val。清单本地 h100-stereo-left-video-manifest.json，源resolved路径在/gpfs/jiuquyun/datasets/PRETRAIN_DATA/ABC-130k-yam-canonical-v1/videos下。未执行视频下载、迁移或打包。若采用原像素路径，拟目标/mnt/workdir/frank/datasets/abc-stereo-left-original-20261106，需先明确新增原视频迁移及可用传输路径，再完成跨环境数值诊断。不能将七组裁剪文件直接当原视频替代。

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
