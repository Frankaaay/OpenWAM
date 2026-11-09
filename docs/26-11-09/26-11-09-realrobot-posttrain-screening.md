# 真机 posttrain：基础筛查与启动门禁

## 授权与目标

用户要求大致筛查原始真机数据，正常数据尽量使用；窗口起点间隔 0.1s，总计一个真机等效 epoch。8卡 smoke 通过后才进行16卡固定资源训练。PPU仅在无需搬数据时优先，否则L20Y；沿用80%真机+20%预训练、LR3e-5等已讨论参数。未授权删除原始数据。

本轮只提交4个0GPU CPU检查作业，均Succeeded。没有提交8卡smoke或16卡训练，没有下载/解压完整tar，没有修改生产代码、commit或push。

## 本地与身份

- OpenWAM分支 frank/eighteen-preparation，HEAD c134e14827a48c5a5b2d7123379035952d5dfddf；origin为OpenWAM-Official/OpenWAM，个人remote frank为Frankaaay/OpenWAM。
- git fetch origin成功。工作区已有文档、lerobot_v3_reader.py、test_libero.py等修改，未pull或覆盖；本轮只新增本记录。
- CLI C:/Users/Frank/AppData/Local/AliyunCLI/aliyun.exe，profile h800；实时RAM身份frank / 205455090823393499。
- 深圳workspace350607，research quotan2i9e2t9tqr，L20Y；ActualMinQuota32GPU，当前使用8GPU，不等于保证16GPU立即可调度。
- PPU workspace308566，research子配额quota1rpp98rnebp固定资源为0；父配额quotawsqknfajqjt固定64GPU，当前使用16GPU。当前allowedRoles允许research使用父配额。
- 两边本人DSW列表均为空。固定资源检查使用ForbiddenQuotaOverSold、Priority4，未改变角色或配额。

## 原始数据与基础筛查

源：discover-yuanmeng OSS桶，teleop/7c425e922b74/teleop_data_20261109.tar，17,166,530,560 bytes。

在L20Y地域的CPU作业dlc1ybwv7v5pya3b，通过深圳内网endpoint的HTTP Range读取tar header和每个MCAP summary。逐请求验证206及Content-Range，读取上限受控；不读取完整视频主体、不写入源存储。

- 总计590个MCAP，累计10,837.524511186秒。
- 584个结构候选，累计10,746.438068214秒。
- 584个候选均有头部/左右腕六路视频消息，以及左右pose_state和夹爪joint_state消息。
- 根据每个文件的消息时间范围，以1.6s窗口、0.1s起点间隔估算98,401个完整窗口。不是相机共同有效区间或清洗后的最终样本数，也不含尾窗padding。
- 6个文件标为待排除：5个缺全部腕部相机，1个空录制。未删除或修改任何源文件。
- 590个summary均存在，均出现ROS2 schema；未发现名称含task/instruction/prompt的topic。此项不能排除信息藏在消息内容中。
- tar中非MCAP资产为录制器上传状态JSON、路径文件和锁文件；未见独立任务标注表。

待排除文件：

| 文件 | 时长(s) | 原因 |
| --- | ---: | --- |
| recording_20261105_162930.mcap | 7.359 | 四路腕部相机缺失 |
| recording_20261105_163023.mcap | 9.701 | 四路腕部相机缺失 |
| recording_20261105_163528.mcap | 0 | 视频、双臂状态均无消息 |
| recording_20261105_163620.mcap | 20.625 | 四路腕部相机缺失 |
| recording_20261105_164245.mcap | 41.411 | 四路腕部相机缺失 |
| recording_20261105_175906.mcap | 11.991 | 四路腕部相机缺失 |

此筛查不是视频解码、CRC、时间同步、动作数值、任务成功或标注质量验收，584是候选数量，不是最终可训练数量。

## 挂载与checkpoint

- 深圳存储检查dlc10haeizhqyt18：旧CPFS /mnt/workdir、research新CPFS /mnt/research均只读挂载，定向浅层查看本人datasets未发现这批数据。未递归扫描他人目录或共享DATA根，不声称全盘不存在。
- PPU存储检查dlc1119lajzuwes7：PPU15 CPFS本人datasets也未发现这批数据；与深圳CPFS独立。
- PPU网络检查dlcotsv0r4qvw5cq：深圳OSS internal endpoint超时，public endpoint HEAD200、长度匹配。公网可读仍需搬运/转换，不是已有本地挂载数据。
- 因不满足用户“PPU无需传输”的条件，后续候选运行地域为深圳L20Y。
- midtrain dlct1b5e373u30a6实时状态Succeeded，结束2026-11-07T09:56:25Z。PPU输出目录存在：/mnt/workdir/frank/runs/openwam-alpha-ppu-midtrain-20261106/stereo-subset-stride08/2026-11-06_22-36-52。
- 尚未核对最终权重文件的确切路径/散列，也未迁移到L20Y。输出目录存在不等于已通过posttrain加载验收。

## 转换阻塞与诊断

现有orion-data-pipeline/src/orion_pipeline/robot/pushcart/reader.py要求message_encoding=flatbuffer；pipeline.py要求foxglove.JointStates/foxglove.CompressedVideo。此次源为ROS2 ros2msg/cdr，不能直接使用现有PushCart生产转换入口。

第一条录制的小范围本地诊断使用既有环境，无安装、无写入源。标准mcap_ros2 DecoderFactory在sensor_msgs/msg/JointState schema解析失败。仅在内存去掉重复root MSG行的第二次诊断仍出现JointState UnicodeDecodeError及String缺少data字段；Pose输出也未获完整语义验证，因此不使用这些诊断值生成训练数据。不能通过放宽/跳过解析来宣称转换成功。

仍需确认：

1. 是否已有同事发布的LeRobot数据及其准确路径；若有优先复用，不重做转换。
2. 原始录制的任务文本/任务分组；不能自动给所有录制填同一任务指令。
3. ROS2转换所需TCP定义、坐标系、夹爪单位与开合方向，需与部署动作合同对齐；然后才能接入80D fixed-anchor-relative state-target训练路径。

若需要新的ROS2转换适配，应先明确上述数据语义并确认最小修改范围。当前不创建未验证的数据适配器或占位训练配置。

## 待门禁通过后的参数（不是已启动配置）

- 模型：此前完成midtrain的OpenWAM Wan mono-left基线；没有混用另一个chat正在进行的StereoTok midtrain smoke。
- 17RGB帧@10Hz，34个动作时间点@20Hz，窗口起点间隔0.1s；384x320与原midtrain布局保持一致。
- 80%真机+20%原预训练数据；总计一个真机等效epoch。
- global batch64：8卡smoke每卡batch2/accum4，16卡正式每卡batch2/accum2。
- AdamW，video/action峰值LR均3e-5，warmup100 optimizer updates，从3e-6起，cosine至3e-7；weight_decay0.01、clip1、betas(0.9,0.95)。冻结codec/T5，训练两个DiT，loss权重1:1，bf16/ZeRO2/gradient checkpoint，无optimizer offload。
- 仅加载midtrain模型权重，重建optimizer/scheduler。正式W&B online必须实际初始化并有loss日志；尚未验证凭据和目标run。
- 若最终训练窗口恰为98,401，更新数ceil(98401/(0.8*64))=1,922；16卡max_steps需按microstep计为3,844。最终按有效样本数重算；不把估算写死成生产长度。
- smoke须覆盖真实读数、finite loss/grad、8rank同步、完整保存和恢复。16卡仍需启动时验证双节点通信；单节点smoke不证明双节点通过。

## 证据资产

本地请求和回执目录：C:/Users/Frank/AppData/Local/AliyunCLI/h800-requests/openwam-realrobot-posttrain-20261109/。

- cn-shenzhen-storage.json及receipt/logs：深圳只读挂载检查。
- cn-wulanchabu-storage.json及receipt/logs：PPU只读挂载检查。
- cn-shenzhen-raw-audit.json及receipt/logs：全包summary筛查。
- raw-structural-audit.jsonl：590个文件的逐条结构审计，含topic计数、时长、tar offset、flags；不作为最终训练manifest。
- cn-wulanchabu-checkpoint-access.json及receipt/logs：PPU输出目录及OSS访问检查。

所有作业0GPU/4CPU/16Gi，最长15分钟（原始审计20分钟），固定资源，存储只读。未生成可直接提交的GPU训练请求。

## 后续：用户授权自行转换后的实际进展与更正

用户明确同事尚未转换，授权自行处理并估算耗时。本节替代上文“任务标注可能缺失”的推断；上文topic筛查本身仍成立。

### 源数据已落盘

- 新作业dlc1n8iq6jnexnv6，深圳L20Y地域，0GPU/8CPU/32Gi，ForbiddenQuotaOverSold、Priority4，30分钟上限，实际Succeeded。
- 目标独立目录：/mnt/workdir/frank/datasets/realrobot-teleop-20261109。
- 内网下载约159MB/s；tar落盘、SHA256、590个MCAP解包与逐文件SHA256、全部metadata索引提取累计192.300秒。
- archive_sha256：54c04558cea37b56e3e872cfef9617c81983dfc755df459bdcbba207ab1251e8。该值为本次读取后计算的指纹，不冒充上游提供的SHA256。
- 原始录制目录：raw/teleop_data/；source_manifest.jsonl保留原文件名、大小、逐文件SHA256及MCAP metadata；source_inventory.json记录全量统计。
- 所有590条metadata均有task.action_text=叠衣服、device.device_code=cart-007。第一条task.task_code=DL0007，episode UUID也存在。
- 原先“未发现task topic”不能推出没有任务标注；标注实际在MCAP Metadata记录而非Message topic。
- 没有删除源tar、异常录制或既有产物；训练筛选尚未发布。

### 首条真实录制读取及视频转换计时

本地独立诊断目录位于本任务CLI请求目录下sample-inspection/，与生产数据隔离；下载只读首条37,079,883 bytes，SHA256为3854d12c6a20c140d668e8d1da93559752c94cd8245e00dc65b6d636902ce520。

复用现有_CdrReader和ROS2字段定义做结构校验；诊断处理了根MSG重复声明、float64默认值声明、末尾最多3字节严格对齐的零padding。字段和依赖布局完整比较，消息末尾校验仍保留；没有修改原MCAP或生产reader。

- JointState、PoseStamped、CompressedVideo、String、TFMessage真实读取成功。Pose四元数完整、首值范数1；双臂关节和夹爪数值有限；抽样消息时间严格递增。
- 全部六路H264为640x480。文件开始于P帧，首个可用关键帧前分别有6/5/6/5/4/3个包无法独立解码；随后每路818–820帧可解码。未来正式转换应按真实时间戳选择共同可解码区间，不能静默按视频行号对齐或直接丢整条。
- 单进程仅解码和读取检查约3.63秒；样本录制约27.8秒。
- 六路视频解码并生成256x256 letterbox、10fps、libx264 fast/CRF18的诊断MP4，约12.116秒。已回读头目274帧、右腕273帧。
- 诊断MP4采用每3个解码帧取1帧用于计时，没有执行正式跨相机公共时间轴对齐，不能当作训练release。未发布LeRobot、未计算正式norm、未做真实DataLoader smoke。
- 按时长线性外推，单worker视频处理约78分钟；8worker理想约10分钟，考虑不同CPU、CPFS、同步和封装，暂预算20–40分钟视频阶段，完整转换/回读另需余量。不是L20Y全量转换实测ETA，不包括适配开发和排队。

### 语义线索与仍需核实的实际问题

- 源pose header.frame_id=p7_base_link；双臂joint_state名字joint1…joint7，gripper字段名gripper。tf_static样本只含腕部左右相机外参，未给出末端TCP定义。
- 使用现有PushCart P7C URDF诊断。Windows源URDF为CRLF，独立诊断副本仅规范为LF后SHA256精确匹配冻结值3d995900355b08f35ae11c3216e62645c7cf810baf43ca77d498c8fec920c317，没有修改仓库URDF或绕过散列检查。
- 第一条首样本FK(tool local X+0.10867m，未施加FLU轴旋转)的XY与源pose接近，源Z比FK约高0.0983m，与旧PushCart源记录差异相符。这是小样本线索，不是全量模型/TCP身份证明。
- 夹爪原始值可以无损读取，但本包metadata尚未找到开合端点标定；不能仅用观测最大值定义完全张开。
- 第一条录制首尾头目画面未见衣物，呈现机械臂动作/测试场景；统一“叠衣服”是否实际任务或采集器默认标签，需要确认。未据此自动重标任务或批量剔除录制。

本轮新增操作资产：stage-source-and-metadata.py、cn-shenzhen-source-stage.json及回执/日志；inspect_ros2_sample.py、benchmark_sample_video_conversion.py、sample-inspection/report.json和诊断视频/图片。生产仓库未改代码、未commit/push，8卡smoke和16卡训练仍未启动。

### 用户查看视频后的任务说明

用户说明这批任务应都是“挂杯子”，原始“叠衣服”任务名不对。后续训练任务文案拟采用Hang the cup.，原始task.action_text/task_code保留为来源元数据，不原地修改MCAP。该说明解决当前任务名疑点，不构成每条录制任务成功的人工验收。

当前仍只有原始数据和单条诊断视频，未完成正式LeRobot转换。尚未核实的动作合同是：本批是否沿用现有PushCart v4的P7C URDF、tool局部X+0.10867m TCP、每臂FLU base/team tool轴，以及夹爪0=closed、0.095m=open。既有配置采用这些值，但原配置的端点依据是此前数据的用户选择，不能仅凭同名字段或一条近似FK吻合证明新批次也适用。此项需确认或有采集/部署配置证据后才能冻结正式映射。

## 混合任务初筛：28条分层样本与代表视频

用户随后说明同事确认混有挂杯子、叠杯子、交接木棒，要求算法粗筛、不确定的交其复核，并先展示任务种类样例。因此撤销此前全体统一Hang the cup.的拟议标签；未实际执行过该重标。

本轮按时间分层选28个ordinal，每条抽8%/35%/65%/92%四个时刻的头部左目；先从目标前0.7秒开始解码关键帧，保留实际帧时间和目标时间偏差。输入仍只读，通过有界HTTP Range读取已知tar offset，未再次下载完整tar。

首轮RangeFile因Python RawIOBase被MCAP临时BufferedReader关闭而失败；修正诊断封装为BufferedIOBase后重新运行，失败不计为源数据坏。最终25条四帧齐全，3条仅三帧，后者标为预览不完整，不作为数据剔除理由。累计Range读取约416MB。

算法阶段：工作区域HSV直方图+低分辨率RGB，固定seed3407、K=5的k-means外观分组。仅用于寻找代表样本，不能当作任务语义分类器：例如杯子与篮子样本会混组；分离度不是经过校准的任务置信度。随后人工式视觉检查这28条的抽帧，形成候选语义，不向未查看的562条传播标签，不按时间段自动贴标签。

样本中出现至少四类：

| 候选类别 | 抽样数 | 代表ordinal/文件 | 证据 |
| --- | ---: | --- | --- |
| 挂杯子 | 6 | 005 / recording_20261103_163151.mcap | 黄杯从桌面移向木架并留在架旁 |
| 交接长条物体 | 10 | 180 / recording_20261104_171745.mcap | 蓝色长条在两只夹爪之间转移；材质不凭画面认定 |
| 叠/套杯 | 8 | 360 / recording_20261105_105815.mcap | 多个红杯发生套叠动作 |
| 物品放入篮子（新增，待用户确认命名） | 4 | 570 / recording_20261105_154145.mcap | 桌面香蕉形物品和其他物品被移入篮子 |

此表只代表28条抽样，不是全量590条的任务数量；任务类型判断也不证明演示成功。样本000出现人手操作，001/003/010/014需要完整片段或质量复核；003/010/014预览不足四帧。不能因此把前期样本统一判为测试或失败。

生成4个完整头目预览，保留源方向，320x240/10fps，MCAP实际时间戳映射到预览时间：example-005.mp4(225帧)、example-180.mp4(209帧)、example-360.mp4(156帧)、example-570.mp4(230帧)，均已逐帧回读数量一致。各片起始不可解码P帧数量分别1/6/6/4，未作为有效帧填入预览。预览不是训练视频，不用于跨相机同步证明。

产物位于本任务CLI请求目录/task-review/：
- sample_index.json、各ordinal/preview.json与frame-*.jpg：样本来源和时间。
- appearance_clusters.json：外观分组结果及方法限制。
- sample_candidate_labels.json：28条候选任务、复核标记，全部approved_for_training=false。
- samples-00.jpg…samples-03.jpg：28条时序联系表。
- four-task-examples.jpg：四类代表，每类四个时刻。
- example_clips.json与example-005/180/360/570.mp4：已回读的代表预览。

尚未进行590条全量分类、任务成功判定、LeRobot发布或GPU smoke。先按用户要求展示四类代表确认类别与命名，再扩展全量；保留unknown类别，不强行限定前三类。

## 全量590条任务初筛完成

用户已确认四类命名并授权全量初筛；随后补充记忆：交接木棒集中一天、挂杯子两天、叠杯子一天。时间作为交叉核验信息，不覆盖画面证据。

### 覆盖与方法

- 590条全部访问，每条最多4个时刻（8%/35%/65%/92%）的头目预览。8个CPU线程读取已知OSS字节范围，预览阶段396.3秒；578条四帧齐全，11条不足四帧，1条空录制无预览。
- 对工作区HSV色彩分布、低分辨率Lab及空间色彩占用特征计算与已复核参考帧的最近距离；结合帧间投票、类间距离差、外观离群与时间分组冲突提出复核项。距离和margin不是分类概率。
- 初始算法标记10条复核（000/004/008/012及584–589）。进一步查看全部异常和切换边界：004/008确认是挂杯子，012任务可辨但片段有效画面短；000出现人手进入操作区；584/585能识别为交接长条物体场景，但缺腕部相机。
- 原始算法输出保留在task_screening_candidates.json；视觉复核在visual_resolutions.json单独记录，最终结果保留原候选与修正依据。
- 没有对其余每条完整视频进行成功判定、精细动作检查或人工介入检测。任务分类初筛完成不等于“582条成功示范”或训练就绪。

### 时间规律及候选数量

| 日期/时段 | 原始数量 | 画面核验后的候选 |
| --- | ---: | --- |
| 11-02 17:05:31–17:20:30 | 2 | 挂杯子 |
| 11-03 15:34:20–18:58:15 | 13 | 挂杯子 |
| 11-04 14:06:06–18:55:29 | 278 | 交接长条物体 |
| 11-05 10:12:13–14:18:09 | 233 | 叠／套杯 |
| 11-05 14:55:36–15:50:03 | 58 | 物品放入篮子 |
| 11-05 16:29:30以后 | 6 | 2条交接场景、3条入篮场景、1条空录制，全部结构异常 |

11-05的525→526边界和11-04→11-05的292→293边界已逐帧联系表复核。11-05晚间重新出现交接场景，是不能只按日期强制分类的实际反例。

结构齐全584条的任务候选：挂杯子15、交接278、叠杯233、入篮58。其中挂杯子2条质量待复核，其余582条类别明确，可作为后续转换的分类候选，尚未通过同步/norm/DataLoader验收。

待复核/暂缓共8条（没有删除）：
- ordinal000，recording_20261102_170531.mcap：任务挂杯子，约18秒处出现人手操作；头目预览27.4秒。
- ordinal012，recording_20261103_173225.mcap：任务挂杯子，原始消息跨度5.088秒，但可解码头目预览约1.1秒，动作范围不足以确认；不把原始时长误报成1秒。
- ordinal584/585/587/588/589：缺全部四路腕部相机，结构暂缓。
- ordinal586，recording_20261105_163528.mcap：空录制，结构暂缓。

### 交付与验证

本任务CLI目录/task-review/新增：
- all_preview_index.json：590条预览索引。
- time_groups.json：日期与超过15分钟采集间隔。
- task_screening_final.json / task_screening_final.csv：完整候选标签、时间先验、原算法结果、复核状态、tar offset/size。
- needs_review.csv：8条待复核/结构暂缓。
- review.html：本地汇总、复核帧和完整预览链接，不上传到外部站点。
- example-000.mp4 / example-012.mp4：两条质量待复核的完整头目预览，逐帧回读分别274/11帧一致。
- early-uncertain-and-boundary.jpg、basket-boundary-and-tail.jpg：异常和切换边界证据。

交付检查：590条源文件唯一，ordinal完整0–589，JSON与CSV均590行记录，582 classified_candidate + 2 quality_review + 6 structural_hold = 590；所有approved_for_training仍为false。原始文件未修改，没有按算法自动删除数据、发布LeRobot或启动GPU任务。

## 用户确认排除8条及转换准入清单

用户明确排除ordinal 0、12、584、585、586、587、588、589，授权剩余数据转换并进行8卡smoke。只排除本次训练，不删除原始文件。

已生成任务CLI目录/task-review/conversion_admission_582.json，回读确认582条源文件唯一：挂杯子13、交接长条物体278、叠杯233、放入篮子58。保留源任务文字“叠衣服”、新标签依据、tar offset/bytes与L20Y源路径；approved_for_conversion=true，approved_for_training=false。原始分类报告不覆盖。清单SHA256：819524fc15402f6680edda6fde24a5c2a6cfd6a6f9e20ed9a295408c2aea2aca。

本轮核对现有PushCart mapping仍仅支持P01/P02/P05/P06/P08，不能直接处理本批ROS2 topic布局。canonical.json采用P7C FK、tool X+0.10867m、FLU/team tool轴、夹爪0–0.095；其中夹爪依据明确为2026-10-13旧数据用户选择。本批cart-007的采集/部署标定尚未从已检查配置中找到，不能仅凭字段名沿用旧合同。按orion-data-pipeline/AGENTS.md“未知frame、单位、TCP、模型、标定不能猜测”要求，正式数值转换须先确认该动作合同；目前没有发布LeRobot、生成norm或提交GPU smoke。

本轮仅生成准入产物及追加本记录，未改生产代码、未commit/push。OpenWAM实时HEAD为500c0d3a8e502b653281a3f615009af7a1141d2f；已有无关dirty修改保留，已fetch origin，未pull。

## Recorded pose转换实现与真实小样本验证（进行中）

用户已明确采用录制pose/state，不等待旧FK/TCP/夹爪端点；本轮授权转换及8卡固定资源smoke，明确要求smoke通过后停下，正式训练必须再次同意。旧FK参数不再作为本批前置门禁。

最小实现位于scripts/prepare_realrobot.py、openwam/dataloader/realrobot.py及registry.py的定向注册。独立schema为cart007.recorded_pose.raw_gripper.v1：保存双臂源pose（含原始xyzw）、row-Rot6D State20、原始夹爪值；不声明是旧PushCart canonical openness。时间轴20Hz，六路视频按真实PTS最近对齐、256方形letterbox；训练RGB10Hz/17帧、动作20Hz/34步、0.1秒anchor间隔。新读取适配复用Orion fixed-anchor relative transform及80维scatter；夹爪采用经验z-score，不猜测物理端点或裁剪。统计计算和真实DataLoader回读尚未执行，适配代码不能视为已验收。

本地检查下载四类已确认代表ordinal 5/180/360/570，输入合计133,313,048 bytes，来源为既有OSS tar精确Range。仅补装本地已有转换依赖pyarrow 26.0.1，未改项目依赖声明。

- v1失败于005的同步容差，保留失败输出；没有修改源数据、放宽容差或静默移除训练记录。
- 针对005进一步核验：原始状态最大消息间隔约0.835秒，但大部分位于六路解码视频共同区间之前；共同网格内仅最初0/0.05/0.10秒无效，后续连续。审计见任务CLI目录/fixture005-sync-gap-audit.json。处理为公共区间起止裁剪并记录boundary_crop；中间缺口仍失败，不补帧、不跨缺口拼接。该审计原pending标记是修复前状态，不是待用户批准新筛选规则。
- v2真实转换3条（180/360/570）完成：415/310/451帧，18路MP4均全量回读帧数、尺寸、PTS一致；Parquet逐列回读一致。单条转换耗时11.0/7.1/13.7秒（Windows CPU、2worker），不能当作L20Y实测ETA。
- v3含全部四类代表转换通过：005/180/360/570分别423/415/310/451帧，共1599帧，24路视频全量PTS/帧数/尺寸回读通过，Parquet回读一致，共737个完整窗口。固定单线程视频解码后的005实际boundary_crop为leading_frames=4，即0.20秒；此前独立默认解码线程诊断的0.15秒只用于定位，正式结果以v3对齐记录为准。其余三条无边界裁剪。

L20Y只读探测job dlc16memtnoan9wo已Succeeded（0GPU、2CPU、8Gi、固定资源、15分钟上限），日志确认原始590 MCAP已在/mnt/workdir/frank/datasets/realrobot-teleop-20261109/raw/teleop_data，CPFS可用1.6T；系统Python有torch/pyarrow，缺av/mcap，不视为运行环境已验收。权限/配额实时核验通过；固定配额32GPU，AllocatedQuota为24GPU，不据此保证8卡立即可调度。

尚未在L20Y执行全量转换或提交8卡GPU smoke；未commit/push，未启动正式训练。项目规则要求远端使用clean clone和已推送精确SHA，当前新增转换代码尚未发布，须在提交远端执行前解决发布授权。

三份相关Python代码AST解析通过；未执行torch DataLoader验证（本地转换环境没有torch）。当前HEAD及远端frank/frank/eighteen-preparation均为500c0d3a8e502b653281a3f615009af7a1141d2f。拟提交范围仅scripts/prepare_realrobot.py、openwam/dataloader/realrobot.py、openwam/dataloader/registry.py及本任务记录，不包含已有其他dirty修改。远端发布后仍须完成全部582条转换、归一化与实际DataLoader回读、midtrain权重身份/可访问性核验，才提交8卡固定资源smoke。smoke通过也不得自动启动16卡正式训练。

## 已授权发布及L20Y转换启动

用户同意上述四文件提交推送，已发布cec1796e63008db7fb53ba31d1ab4b30312597d3，远端自己的clean clone已核验。数据管线四个依赖模块从已发布origin/main c5442a573e800b4f18afa723419238efb526032b导出，逐文件SHA保存dependency-receipt.json，不使用未推送依赖修改。

CPU转换首次job dlc7nzevfgf1ufos仅在pip阶段失败，镜像PIP_CONSTRAINT固定numpy1.26.4与本地验证版本2.5.1冲突；专用隔离venv中解除该继承约束后，dlc1uhlc0bu2x3vw安装通过并开始转换（0GPU/8CPU/32Gi、固定资源、90分钟上限）。没有替换系统Torch。远端环境av19.0.1、mcap1.4.0、numpy2.5.1、pyarrow26.0.1、opencv-python-headless5.0.0.93。

全量v1未发布，完成5条report后失败于NONINCREASING_SOURCE_TIMESTAMPS。定向复现：ordinal2左腕右目、ordinal3头部左目，解码结果各有一处约233ms的presentation时间倒序。修复仅对解码完成的图像按真实采集PTS排序，不重排压缩包、不改时间戳、不合并重复时间、不放宽同步容差；同时逐条记录所有转换异常，任何未解决异常均阻止发布。原始/失败产物保留。

复用了本人CPU-only DSW dsw-4gpdclcrpjz496v51r（0GPU/8CPU/32Gi，固定资源），SSH实时核验120.77.254.228:1024，CPFS源正确。用于准备和接收最终midtrain，不是GPU训练。

PPU只读核验job dlcrd6l1nfts76d3定位最终checkpoint_step_50101.safetensors，24813767464bytes，位于/mnt/workdir/frank/runs/openwam-alpha-ppu-midtrain-20261106/stereo-subset-stride08/2026-11-06_22-36-52。transfer job dlc1lc04y31dhyyf（0GPU）仅传最终权重、config.yaml及tokenizer到L20Y /mnt/workdir/frank/checkpoints/openwam-alpha-wan-midtrain-final-50101-20261109，逐文件源/目的SHA校验；不迁移optimizer或历史checkpoint。SSH采用已信任host key及本人既有PPU transfer-key，私钥不导出。传输完成待验收。
