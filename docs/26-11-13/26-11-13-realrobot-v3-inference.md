# 真机 v3 OpenWAM 推理适配

## P7 二代臂适配（当前部署入口）

后续用户要求补齐环境安装列表：README Installation新增P7双端环境清单、X5通用依赖安装、
SDK/CORA现场包安装条件和无运动导入检查、GPU依赖与导入检查。依据Kevin精确SHA的
p7/client.py、p7/deployment_config.yaml、pyproject.toml及airbot-deployment-sop.md核对：
其SOP仍是一代臂，pyproject只有airbot-py，没有P7 SDK/CORA版本或安装包地址；明确记录该缺口，
未编造PyPI包、下载地址或SDK版本。未修改模型依赖或现场环境；文档代码块做语法检查，
真实SDK/CORA导入检查需在X5执行，不报告为本机已通过。

用户明确改为二代P7，并授权修改后push。目标仍为OpenWAM的frank/realrobot-v3-inference，
修改基线bece4d14e9def732ad65f5e5e391e37edee81da0；fetch origin/frank和fast-forward检查完成。
只读参考Kevin分支kevin/p7-deploy-261029，实时SHA为06ebf02d1d7806c5b20c03bcfbc682bd61446103。
未切换、修改或推送orion-0工作区。下方一代臂的UVC、airbot_py和旧ORION版本说明为历史记录，
当前启动方式以根README的P7章节为准。

- 客户端依赖P7System/load_deployment_config：arm_p7_sdk连接X5 arm_app（默认50071/50072），
  CORA读取三路左目，Stereo另加三路真实右目话题；右目沿用各安装位rotate_180。
- 删除UVC设备读取路径，配置right_camera_devices改为right_camera_topics。
  读取完整4:3 RGB，以PIL BILINEAR缩到256×192，再交给未修改的模型mosaic路径。
  原始v3 reader证实训练使用256×192的视频，源转换器的插值/旋转逐像素一致性尚未验证，
  现场须对照训练图像核对方向和视野；没有把新增resize称为已证实的源转换器复现。
- P7读取pose.position/orientation和eef_pos毫米反馈；anchor检查转为米。
  模型输出保持原始米单位，复用P7.send在SDK边界乘1000一次，限制范围0..0.095m。
  未引入canonical openness、P7 canonical server的-0.0984m偏移或TCP128。
- 要求arms: both、TCP、同步20Hz且TTRTC关闭，拒绝上游默认arms: right。
  check-only不获取控制权；确认后enable_servo，再mark_moving并重新采样。
  部分enable失败由P7.close释放已获取的控制权；完成/异常先stop_motion，finally确保close。
- 模型服务、归一化、checkpoint、采样与动作还原未改；不重跑已完成GPU smoke，
  不恢复用户取消的Stereo交接测试。既有3/4结果只覆盖模型侧，不代表P7硬件验证。

验证位置：本机授权OpenWAM worktree。使用git archive将精确Kevin SHA的部署模块提取到
C:/Users/Frank/AppData/Local/Temp/openwam-p7-06ebf02，未更改orion-0分支。
PYTHONPATH指向该目录src，运行：

```powershell
uv run --with numpy --with scipy --with pillow --with pyyaml --with pytest --no-project python -m pytest -q tests/test_realrobot_v3.py
uv run --with numpy --with pillow --with pyyaml --no-project python scripts/realrobot_v3_client.py --help
```

结果20 passed（1.48秒），CLI帮助通过，git diff --check通过。
新增测试使用实际P7模块及mock SDK反馈/发送：TCP不偏移、夹爪米/毫米单次转换、anchor漂移拒绝、
三/六目映射与时间戳、全视野resize、单臂/超范围拒绝、只推理零运动、enable/send/stop异常释放。
真实arm_p7_sdk/CORA服务、现场坐标和相机方向、控制权获取及伺服执行留待同事现场check-only后验证。

## 最终交付状态（更新覆盖下方历史记录）

根据用户要求，将现场部署步骤写入本分支根README：固定依赖、四权重路径和验证状态、
双端配置、SSH隧道、只推理检查及显式运动入口。对照实际CLI参数和ORION配置字段检查；
本次仅文档修改，不重新运行GPU测试。

用户授权提交/推送和GPU smoke。实际通过3/4：Wan叠杯、Wan交接、Stereo叠杯。
用户要求不再测最后一个；Job54520已取消，Stereo交接未完成，不报告通过。
Wan结果：smoke-results-2433ed8/，对应代码2433ed8938461af388f1b234e0f3481bee349235。
Stereo叠杯结果：smoke-results-f46b9d7/stereo-cup_stacking.json，代码f46b9d75b04b3f963a9bae60344f55fbbb6796f1。
上述路径均相对H100 /gpfs/jiuquyun/projects/Frank/realrobot-v3-posttrain-20261111/。
三项均完成真实记录归一化/TCP往返、严格加载和两次10步GPU推理；输出4×16且通过默认位姿界限。
预热后时延分别0.608、0.618、0.630秒；Wan峰值23.40GiB，Stereo37.21GiB。
Stereo首次推理29.86秒（含首次编译/初始化），不计入稳态超时；首次输出不会用于运动。
基于实测将client.max_round_trip_s从0.5改为2.0秒，保留位姿、anchor及夹爪检查。
这是同步等待推理后执行4步，不是持续20Hz闭环。真实网络往返需现场观察。
最终依赖：OpenWAM环境openwam-baselines-20261014；FastWAM固定491e518fac6e04646ecfbd464027dd575ed99bde；
ORION固定783b602d128a43814af32285e60a60f270ec2703。三项GPU验证不等于真机闭环成功。
现场使用同一套ORION双臂硬件；只需核对训练记录TCP坐标/夹爪量纲及六目图像对应，不重新选择硬件。

## 范围与版本

用户授权在 `C:/Project/OpenWAM/.worktree/realrobot-v3-inference`、分支
`frank/realrobot-v3-inference` 实现，未授权 commit/push、GPU 作业或真机运动。
基线 `8d802728a26372baa91063fb79dd2ab44a24e1cb`；fetch origin 和实际跟踪远端 frank 成功，
HEAD 与 frank/frank/eighteen-preparation 一致。origin 是官方仓库，没有同名个人分支。
原工作区未提交修改保留。训练 SHA 为 `9b87cab7715c026cc5f4762dcb44fc9f95fcd094`，
本次使用的 deploy、architecture/base、orion_midtrain 和 multiview 文件与其一致。

复用 orion-0 `783b602d128a43814af32285e60a60f270ec2703` 的
`wam.deployment.airbot.client` 和 `policy_protocol`，不修改 orion-0。
运行时需在 PYTHONPATH 加入该版本的 `orion-0/src`，不安装另一份模型框架覆盖 OpenWAM。
机器人进程只需要 ORION client 的 NumPy、PyYAML、OpenCV、airbot_py；不加载 OpenWAM 模型。

## 实现

- `openwam/deploy/realrobot_v3.py`：原生 OpenWAM strict checkpoint loader + JointInferenceEngine，
  v3 normalizer，观测校验，固定 anchor 绝对位姿还原及 ORION 协议 server。
- `scripts/realrobot_v3_client.py`：复用 AIRBOT 状态/伺服/插值实现，扩展真实右目读取，
  默认只做一次推理；显式开启运动后也先预热、人工确认，重新采样再执行。
- `configs/realrobot_v3_inference.yaml`：任务、左右目模式、执行 chunk 和界限；server/client 共用。
- `tests/test_realrobot_v3.py`：CPU 合同与通信测试，不需要 SDK、权重或 GPU。

没有修改训练路径、生产模型数学行为、原始数据，也没有复制训练权重或部署资产。

## 训练与部署合同

训练证据：本机 `youzi-v3-ppu15-20261111/v3_reader.py`、`train_v3.py`。
state 和 action 都按每任务的 20 维 mean/std 做 z-score，包括原始夹爪值；
映射到模型槽位 `[0:10]`、`[34:44]`，其余维度零填充。
V3Normalizer 挂在 architecture 上，输入归一化及输出反归一化各一次；
通过其 scatter 索引复用 architecture 的 inactive-action-dim 处理。
不得使用 ORION 的 TCP128、canonical openness、示例 manifest 或二值夹爪阈值。

state 是每臂 `xyz, first-two-rows(R), raw_gripper`。预测相对位置/旋转按
`p=p_anchor+R_anchor*dp`、`R=R_anchor*dR` 还原；夹爪是目标绝对原始值，不是增量。
Rot6D 正交化后转 SDK xyzw；零向量/平行行直接拒绝。

17 帧视频对应 34 动作位置 `[-.05, 0, .05, ..., 1.6]` 秒。
引擎参数 `num_frames=35`（动作生成长度减一）、`video_num_frames=17`。
跳过前两项，从 +50ms 开始取 chunk；默认 4 步，即执行 0.2s 后重新采样。
同步推理期间等待，轨迹时间从推理完成后起算；不是实时异步时间补偿。
运动前检查机器人未明显离开 anchor，限制稳态 round trip。迟到直接报错，不加速追赶。
其余 chunk 长度与延迟策略只能在测量后调整。

Wan 使用 head/left wrist/right wrist 的左目；Stereo 使用相同三个安装位的真实左右目。
图像必须是 RGB uint8、全视野 256x192。适配层不擅自裁剪、letterbox 或改变采集 resize。
最终 mosaic 宽320高384，head 320x256、两 wrist 各160x128，PIL BILINEAR。
CPU 测试与已有 multiview helper 对比。采集设备要求256x192不保证实际返回该尺寸，
server会验证实际数组；若设备不支持或 FOV 不同，应先核实 v3 转换端 resize 再定向接入，不能绕过检查。

## 部署前需要补齐的资产

H100 四份最终 safetensors 位于 `/gpfs/jiuquyun/projects/Frank/realrobot-v3-posttrain-20261111/`。
此前只迁移了这四份文件，尚未核实配套部署资产已存在。
每个选定 checkpoint 的同目录必须包含源 run 的：

- `config.yaml`（训练保存的模型组件配置，不使用 ORION 示例配置）；
- `realrobot_stats.json`（与 data-contract 中 stats_sha256 一致）；
- `data-contract.json`（对应 task/model 且 smoke=false）；
- `tokenizer/` 及配置实际引用、未内嵌权重中的其他组件。

不能仅凭 checkpoint 文件体积断言 T5/codec 是否内嵌。使用 loader 的组件重建和 strict load 验证，
缺失文件应补齐原始对应资产，不使用新随机参数或其他任务统计量。
本轮没有取回这些文件，没有做 GPU load/smoke 或速度/显存测量。

## 现场只推理检查

在部署机器准备 OpenWAM 依赖环境，以及上面固定版本的 orion-0 源码。
根据现场设置现有 ORION `deployment_config.yaml` 的机器人 IP、相机设备、policy host/port。
使用 `action_space: tcp`、`control.hz: 20`、`async.enabled: false`。
相机采集设置 width=256、height=192，并确认输出全视野及眼别。
Stereo 的三个右目设备在本次配置 `client.right_camera_devices` 中明确列出。

下面命令中的路径是部署机器上的实际路径变量，需先按现场位置设置：

```bash
export OPENWAM=/absolute/path/to/OpenWAM
export ORION_SRC=/absolute/path/to/orion-0/src
export PYTHONPATH="$OPENWAM:$ORION_SRC${PYTHONPATH:+:$PYTHONPATH}"
export V3_CONFIG="$OPENWAM/configs/realrobot_v3_inference.yaml"
export ROBOT_CONFIG=/absolute/path/to/site-deployment-config.yaml
export CHECKPOINT=/absolute/path/to/final-run/checkpoint_step_9090.safetensors

# 模型机器：需要已经取得资源授权的 GPU 环境，不在 H100 登录节点运行。
python -m openwam.deploy.realrobot_v3 --checkpoint "$CHECKPOINT" --config "$V3_CONFIG"

# 机器人机器：使用本机实际路径变量；默认不复位、不切换运动模式、不发送动作。
python "$OPENWAM/scripts/realrobot_v3_client.py" --config "$V3_CONFIG" --robot-config "$ROBOT_CONFIG"
```

server 默认127.0.0.1:8000。跨机器推荐现有 SSH 隧道转发；ORION协议使用pickle，
不要向不可信网络开放。若GPU在远端，两端不比较绝对时钟；采集时间偏差使用同一client时钟，
round trip也由client计时。UVC时间戳是主机收到帧的时刻，不能证明硬件曝光同步。

对另外三份权重切换 exact checkpoint、task 和 input_mode，server/client配置必须一致。
任务 prompt 由 task 决定，服务端拒绝任务不匹配。

## 允许运动之前

第一版支持 SDK 与录制数据使用相同 pose frame/tool/单位及 raw gripper 语义的场景；
没有假设现场已经证明一致，也没有加入未经证据支持的坐标偏移或夹爪映射。
在现场比对 MCAP `pose_state` 与 `get_end_pose()`、录制 gripper joint position 与
`get_eef_pos()/servo_eef_pos()` 的量纲和方向；不一致时停止，补适配后再测。
确认图像眼别、FOV和采集预处理一致。检查配置中的 raw_gripper_limits 是实际有效范围。
默认[0,1]只是待现场核实的SDK范围，不做clip，也不做二值化。

只有完成上述核对才将 `client.interfaces_verified` 改成 true；然后显式执行：

```bash
python "$OPENWAM/scripts/realrobot_v3_client.py" --config "$V3_CONFIG" \
  --robot-config "$ROBOT_CONFIG" --enable-motion
```

该入口不会调用示例复位姿态。先人工放置到任务起始位置；首次推理成功后按Enter启用运动，
随即丢弃旧观测重新采样。默认总20步后退出，复用ORION插值与控制模式退出。
client仍须由现场人员在可停止机器人的条件下操作。本轮没有授权或执行该步骤。

## 本地验证

CPU环境通过 uv 临时依赖环境运行，无新增项目依赖：

```powershell
$env:PYTHONPATH='C:\Project\orion-0\src'
uv run --with numpy --with scipy --with pillow --with pyyaml --with pytest --no-project python -m pytest -q tests/test_realrobot_v3.py
```

覆盖独立状态/动作统计、原始夹爪、左右槽位、旋转坐标、固定anchor非累加、时间索引、
图像layout、Stereo缺失、时间偏差、错任务/错请求、check-only零运动调用及真实本机socket往返。
真实checkpoint严格加载、GPU smoke、现场同步/SDK语义和闭环效果尚未验证。

2026-11-13 本地实际结果：14 passed（1.90s）；client `--help` 可正常解析入口。
补测并修复长时间运行monotonic时间戳的float32精度丢失：传输/校验保留float64，
不影响模型状态张量float32。测试使用合成样本和已有图像layout参考函数，
尚未用实际v3样本做端到端loader parity。未提交或推送代码。

## 配套资产已补齐，GPU smoke待代码提交

用户随后授权补齐资产并做四权重GPU smoke，明确优先H100闲时资源或L20Y。
采用已经保存四权重的H100，计划1GPU/8CPU/128Gi、30分钟、不requeue，四进程顺序运行。
源PPU15 CPU任务 `dlc1sbpngrf8shlk` 已Succeeded；2CPU/8Gi/0GPU、15分钟上限、PPU15只读挂载。
导出四个训练run原始config.yaml/realrobot_stats.json/data-contract.json/tokenizer，
两个Stereo run另含stereotok.json。每任务额外导出第一条有效记录的六相机PNG、
raw state20/relative34、训练reader产生的normalized state80/action80及valid mask。

H100 `/gpfs/jiuquyun/projects/Frank/realrobot-v3-posttrain-20261111/` 下44个文件、86,323,750字节
逐文件SHA256通过；部署文件放各权重原目录，样本放 `smoke/{task}/`。
回执：`deploy-assets-20261113/verified.json`。
压缩包24,349,172字节，SHA256 `33c72423d4f0349ecab5c97a1844b160b204f2f92a4c84a001537cb26eac578c`。
未覆盖不同文件、未复制权重或optimizer。实际读取两种权重头，确认存在T5和对应codec参数；
这不替代模型strict load。

资产操作脚本/请求位于本机
`C:/Users/Frank/AppData/Local/AliyunCLI/ppu-requests/realrobot-v3-deploy-assets-20261113/`。
其中signed-urls.json/request.json包含短期签名，不提交或输出到聊天。

模型卡：`reczz28ZA1Q0FvG1`，表
https://w79rvfxw83.feishu.cn/base/HikJbmqosaZDdTsJ7INcNHE1nQc?table=tblBYVmMJLV8vKTh 。
未创建保护预约，GPU任务尚未提交。读取最新H100使用文档revision482，闲时任务无需预约保护。
已有其他Frank任务未改动。

准备了 `tests/smoke_realrobot_v3.py` 和 `tests/realrobot_v3_smoke.sbatch`：真实样本归一化往返、
TCP表示往返、严格加载、每权重10步去噪两次、检查TCP16有限值和默认运动界限，记录时延/峰值显存。
运行失败不放宽接口或运动界限。依赖H100现有source-runtime.sqsh、
`fastwam-stereotok-eval-cu128-20261009`解释器、clean FastWAM
`950e03b91ed5c8f43324eb7c9d7acb4f19943b06`、clean orion-0 `783b602d128a43814af32285e60a60f270ec2703`。
上述依赖路径/Git已实时核对；尚未运行新模型导入/GPU验证。

运行前剩余门禁：当前用户此前同意的开发范围为“暂不commit/push”，H100指南要求
“服务器验证前按授权commit/push；使用已推送精确SHA的clean clone”。
本轮已准备可审核的7个具名文件，尚未提交/推送或复制未提交源码到服务器绕过该门禁。

用户已授权commit/push，提交65d83e88d9680e9fe7488e319f5c778f2405542e已推送。
H100从GitHub clone较慢，改为传输同一已推送提交的git bundle，目标clean clone
`/gpfs/jiuquyun/projects/Frank/OpenWAM-realrobot-v3-smoke-20261113`，origin仍为Frankaaay/OpenWAM。
Job54516单卡运行27秒后失败，尚未加载权重：旧FastWAM评测环境缺ftfy。
定向修复为使用已有OpenWAM环境`/mnt/workdir/frank/envs/openwam-baselines-20261014/bin/python`，
未修改共享环境或验收条件。两个实际记录样本（叠杯episode1、交接episode0，anchor均0）
已在本地完成state/action归一化与TCP16/TCP20往返，均在1e-5容差内通过。

Job54518使用2433ed8938461af388f1b234e0f3481bee349235运行6分41秒：Wan两权重strict load及两次推理均通过，
随后Stereo因旧FastWAM950e03缺MetricHead而在加载前失败。Wan结果保存在
`smoke-results-2433ed8/mono-left-{cup_stacking,object_handover}.json`，不重跑已通过项。
查训练部署原始记录确认FastWAM应为491e518fac6e04646ecfbd464027dd575ed99bde；
在Frank独立clone中通过Git获取并固定此SHA，不修改其他正在使用的FastWAM。
smoke launcher增加仅选择一种输入模式的测试参数，重试只跑两份Stereo；生产推理实现不变。
