# StereoTok clean Stage3 u8000：PPU1.5 32卡midtrain准备

## 当前范围与状态

用户要求找到指定目录的u8000 checkpoint，替换此前Wan VAE，双目输入，其余沿用已完成Wan midtrain参数，准备PPU1.5 32卡。当前仅执行只读定位和准备检查；未提交GPU训练、未迁移资产、未修改生产代码，未commit/push。这是新实验，从与Wan对照相同的Alpha154000初始化训练模块，不能续训已完成的Wan50101权重。

## 已定位checkpoint

深圳新CPFS URI：bmcpfs://bmcpfs-02001p6aro4qj9ktlrjge.cn-shenzhen/research/，本次只读挂到/mnt/workdir/research。

指定源：/mnt/workdir/research/peilin/StereoTok/outputs/stage3-clean/stage3-clean-e93a3e1-r4-resume-u6500/ckpt-u8000.ckpt，13,797,868,413bytes。同目录stereo.pth为4,606,154,269bytes，尚须核对导出update及checkpoint结构。原目录在PPU15、PPU15/from-sz映射和深圳旧CPFS均不存在，不代表其他路径无副本。

定位作业：PPU15 dlcrtc45ny5ir22w；深圳旧盘dlc1l852q3a6nfbv；深圳新盘dlcca09ajsfdop7q均Succeeded。元数据检查dlc16is00bpx4ovs进行中。仅在用户指定他人目录内读取，不使用其代码/环境运行。

## 固定训练配方

4节点×8卡PPU1.5，batch2/rank，accum2，global128；DROID1,667,932+ABC1,538,510=3,206,442 train windows，1epoch约25,051updates。17RGB@10Hz/34action@20Hz、stride0.8s、384×320 mosaic、原split/anchor/phase/norm、原监督和mask。双目取真实同步左右图，不以复制左图或零图冒充右目。

Alpha154000的video/action DiT保持初始化；StereoTok编码器与T5冻结。LR video/action均1e-4，warmup500update从1e-5，cosine最低1e-6，AdamW(.9,.95)/wd.01/clip1，bf16/ZeRO2/gradient checkpoint/no optimizer offload。每500update验证每源128windows，start/end验证；每2500update保存（5000microsteps）。新输出和新W&B run，与Wan对照分开；沿用用户已选offline。模型与输入更换后必须做实际前后向和分布式检查，不能直接沿用Wan的20h ETA。

## 数据阻塞

PPU15现有精确子集catalog的可用右目标记，dlc1s5uxajiqg258实查：train11,764episodes，仅3,249有右目（393,105windows），8,515缺失；val172episodes/21,805windows全部无右目标记。不能过滤成较小双目集合，否则失去与Wan数据对照。须在原身份集合内定位/补齐真实右目并核对同步时间/几何裁剪；不能默默改变左目数据版本。DROID右目完整性仍待真实读取验证。

## 接入审查

OpenWAM当前branch frank/eighteen-preparation、HEAD5452288389ed8cd261b61b1ea7dec086cb0eecd6，fetch origin成功；保留原有dirty修改。StereoTok HEAD7dae078f7509b2e9c7c69ed8162a62263a99a077，fetch成功。

已有openwam/model/video_backbone/encoder/stereotok.py和midtrain_droid_abc.yaml可复用，但PPU专用reader当前明确只允许mono-left。Alpha finetune采用自包含checkpoint严格加载；不能仅填encoder路径后假定native Wan权重兼容StereoTok骨架。需有界适配：严格保留全部Alpha训练模块，只替换冻结codec，并验证参数指纹。当前本地FastWAM默认loader固定旧Stage2/u15853，不能拿其冒充本次clean Stage3；需要核对已存在的Stage3适配实现及精确版本，不能直接放宽strict load。

PPU15实查quota Ready、64GPU/Allocated GPU0，7个完全空节点；不构成预约。正式32GPU请求尚未提交，优先解决数据与资产接入，避免占用空跑。

## checkpoint元数据检查完成

dlc16is00bpx4ovs完成METADATA_DONE：ckpt-u8000.ckpt内部format=stereotok-wan-training-v1、architecture=stereo-v3、generator_updates=8000。该定期ckpt的last_validated_update=7500、validation_pending=true（保存发生于最终验证之前）；目录summary stage3/global_updates8000。stereo.pth为stereotok-student-v3、stage3、semantic=false。导出文件自身未记录update，所以尚未用逐tensor比对证明其权重与u8000相同，不能只凭同目录认定。精确源ckpt身份已确认。后续可比较state_dict中vae前缀与导出键/张量；确认一致后只迁移4.606GB导出，保留原始源不变。

待确认范围：跨地域迁移所选checkpoint到PPU15个人checkpoints目录，以及为不改变Wan对照的原ABC集合补齐缺失右目数据。遵循AGENTS的大文件迁移明确授权条款；未执行迁移。实现需复用/定向适配Stage3 codec加载、Alpha冻结codec替换、现有PPU reader双目输出，生产代码和正式32卡提交仍未执行。

## 2026-11-09 ABC双目来源与实物调查结论

用户要求只调查是否H100/PPU不是同批、是否需搬H100双目。未搬运视频、未修改任何数据/索引、未启动训练。

### 精确身份与根因证据

H100私有修复视图与PPU正式Wan catalog的11,936条有序(source_episode_id,source_id,length,source_split)摘要均为d94b3d4bb61d4b19c62239a34cf860fbc42f50c6692532ea68221b05cff11040。因此是同批来源身份（train11,764/val172），不是PPU换了另批episode。PPU全100,028条catalog对选定11,936身份均有唯一匹配，无同身份在其他group重复发布的证据。

H100 /gpfs/jiuquyun/projects/Frank/openwam-midtrain-20261101/datasets/abc-canonical/INDEX_READY.json记录11月1日修复8,687条右目定位字段；locator-repairs.jsonl SHA256=6919c5f8cdce198a9dcdfd63bef8f8d7ebbbcf1f0a8d615ec009ac9caa22d1dd。其合同是已有右目MP4、相同帧数/FPS/时长、只修locator不改像素。

原始共享metadata中的episode1005真实availability/head_right=true，但缺videos/head_right/chunk_index,file_index,from_timestamp,to_timestamp。H100个人视图补回这些字段；对应实际右目chunk-001/file-005.mp4存在。PPU同source_episode_id则head_right=false、4个locator均null，进入256x160_group01，native_crop的head_right=null，物理videos/observation.images.cam_head_right目录不存在。其sidecar provenance.json、mask_config.json、source_frame_mapping.parquet与H100同episode原件SHA256逐项一致，说明源身份/时间映射未换成其他数据。

全量集合验证：H100修复清单8,687个episode_index与PPU缺右目的native_source_episode_index排序摘要完全一致：9810f3e1404c199a6677ca9321a961e2b24357df5c97f17d75ab5cba6bb8d0b8。两边都是train8,515+val172。PPU crop.source_info_sha256与H100原info.json实际SHA一致：e3c1a2de9d9a9fe14a42b1b9f1361cbeb21bce929e1ec919d67d74f4eb553e07。结论：PPU派生发布未包含H100针对原发布缺失locator的修复及对应右目视频；不是简单开启reader开关就能恢复，也不是不同源episode批次。未追溯转码器执行日志，不声称具体代码在哪一步删除右目。

### 实物与体积

H100逐项stat目标11,936条的左右目路径：左右均全部存在。左目43,672,728,819bytes，右目43,752,406,307bytes。缺失8,687条对应原右目32,481,530,726bytes（32.48GB，约30.25GiB）。源为/gpfs/jiuquyun/datasets/PRETRAIN_DATA/ABC-130k-yam-canonical-v1/videos/observation.images.cam_head_right/；用locator-repairs.jsonl精确选择，不递归搬整个ABC。

PPU group01有8,687目标episodes，右目目录不存在、定位均null；group02有3,249目标episodes，其索引指向的右目文件全部存在，合计9,651,771,825bytes。H100抽查episode0、1005及val63881首帧，均是256×256真实非零右目，左右MAE分别16.1366/12.0230/32.9366，不是复制左图或黑帧。这是抽样内容验证，不声称已全量解码每一帧。

### 干净对照方案

保留已完成Wan的全部左目、样本顺序/窗口/phase/split、state/action/mask/norm，补齐原集合的真实右目。最低搬8,687右目文件约32.48GB；若希望全部右目采用统一原始视频版本，可搬11,936右目约43.75GB。无需搬整套ABC或修改左目。现有左目来自native-crop版，恢复画布需保持[0,48,256,160]及既有缩放；新增右目必须按其源geometry/时间映射配准，不能直接误用不同group编号。原右目可在读取时按同规则处理，不必改写已训左目。

按这一方案，没有因本次发现必须重训Wan的理由；真正启动StereoTok前仍必须验收其左目、监督、mask与Wan一致，并核对右目真实帧对应关系。若后续不得不修改共享左目/样本合同，则两条路线均应重新训练。当前推荐以H100已核实原件为补齐来源；未声称所有其他存储均不存在副本。

### 调查执行记录

PPU dlc1ytqsqznezkly首轮因sidecar实际是目录而非JSON退出；定向修正检查脚本后dlcxxa7ez9nnwx8j完成，dlc6qfeh7orf3dp9最终集合/sidecar摘要核验Succeeded。H100只读CPU srun：最初误用容器内解释器路径，作业到达计算节点后报不存在；改用/gpfs/jiuquyun/projects/Frank/StereoTok/.venv/bin/python后完成。未安装依赖，未运行他人代码。证据在本机既有ops目录h100-abc-right-physical-audit.txt、h100-abc-repair-cause-audit.txt、ppu-abc-right-physical-audit.json、ppu-abc-repair-cause-audit.json。所有检查限于ABC明确metadata/侧文件及目标清单，不扫描整个数据根。

## 2026-11-09 缺失右目迁移与速度门禁

用户授权迁移缺失32.48GB右目及索引，并要求过慢时停止，交同事更快迁移。仅迁移8,687个明确路径，不改Wan左目、现有catalog/norm，不启动训练。目标/mnt/workdir/frank/datasets/abc-stereo-missing-right-original-20261109；运行日志/mnt/workdir/frank/runs/abc-missing-right-transfer-20261109。

使用PPU15 0GPU/4CPU/16Gi DLC，通过既有transfer-key与本机已信任的H100主机公钥直连本人Frank账号；私钥未导出。H100通过srun CPU1/2Gi流式发送清单和视频，无压缩、无源文件修改。每个视频源端流式SHA256，接收端核对后才从.partial改为完整文件，并写verified.jsonl。索引核对此前固定SHA256。测速从视频实际到达计时，120秒后如均速<10MB/s（32.48GB约一小时以上）立即关闭传输，保留完整文件及未完.partial，不批量删除。

初次作业dlc5mmn7vx7dlyz2因H100公钥未授权失败，未传视频；只读公钥探针dlcq5sljfn3edxj9成功。已只向本人H100 authorized_keys追加带restrict限制的任务公钥，任务结束后定向移除该行。重试作业dlc16j4n4bins4rh，传输脚本及请求保存在本机既有ops目录transfer-abc-missing-right.py和transfer-abc-missing-right-request.json。当前生产代码未改、无commit/push。

迁移启动健康检查：SSH认证通过，已收到locator-repairs.jsonl 4,064,328bytes且固定SHA256通过。120.1s累计3,084,363,194bytes，595个完整视频/3,081,217,466bytes已通过逐文件源端SHA256；累计25.677MB/s，最近20秒约23.5MB/s，剩余预测1,145s约19分钟。通过两分钟速度门禁，作业继续运行，门禁后续仍持续检查；不将当前进度写成完成。独立.partial保留用于故障定位，不更改既有Wan数据。用于建连的任务专属authorized_keys行已精确移除，既有传输SSH连接继续，不留下额外长期账号授权。此ETA包含流式校验，不含后续右目几何/时间同步及训练reader验收。

## 2026-11-09 DROID与训练资产预检查

用户授权提前核对DROID是否需要迁移并准备本次StereoTok训练资产。DROID审计dlc1c2zbwrep8bhh：train65,117episodes/1,667,932windows，val3,357episodes/86,756windows。按正式SQLite索引逐项检查三相机Lance目录，共68,474episodes/205,422paths，全部存在，耗时305.4s；非递归扫描。train/val各均匀抽样8个窗口，原LanceReader实解码[3,2,17,256,256,3]，左右不同、帧身份/时间戳/跨相机同步门禁全部通过。不是对所有JPEG逐帧重验。现有Wan reader实际已读取左右，输出时丢弃右目，因此DROID无需额外搬视频。

权重审计：dlc1ccyxaj007db4初次严格key集合检查发现训练态777keys与导出773keys不同，未直接放宽为通过。dlc2avqbvbnyy4cj确认差异仅latent_geometry_head.layers.{0,2}.{weight,bias}；导出全部773张量与u8000的vae前缀对应张量逐项torch.equal且dtype一致。export SHA256=4c024b1dc9d9280e94a753ab9ccb1de15cc3c24659a1ec5aa86f94a3805f3503，4,606,154,269bytes。完整codec strict load/前向仍待验证，未将tensor身份检查等同运行通过。

已在深圳经dlc1b91ybv4s5fsj将该导出复制至本人旧CPFS /mnt/workdir/frank/checkpoints/stereotok-stage3-clean-u8000-20261109/stereo.pth，源RO、目标独立新目录、复制时SHA256验证通过。复用本人CPU-only DSW dsw-4gpdclcrpjz496v51r（0GPU/8CPU/32Gi），实际SSH120.77.254.228:1024，主机key沿用已信任记录。PPU导入dlcf2egi3xxfonj4目标同名本人checkpoints目录，.partial收齐且SHA通过才rename。首个64MiB实测速1.42MB/s，剩余约53分钟；速度偏慢，继续短采样决定停止。临时公钥仅本任务使用，不导出私钥。

运行资产审计dlcmu1onf8t84zz8：PPU Alpha154000权重/config/tokenizer存在；原ABC catalog/norm及Orion/minimal StereoTok reader存在；OpenWAM远端source仍1896bf7。Python可导入torch/deepspeed/lance/av，但当前环境没有FastWAM codec；stereotok-reader需PYTHONPATH，仅数据模块，不能替代codec。未改生产代码、未提交32卡训练。

### 正式启动仍需最小接入与验收

1. 在现有PPU reader接入DROID stereo输出（复用原双目读取）；ABC左目逻辑保持不变，新增右目定位：group02原右目，group01按H100原episode身份映射补齐文件，保留anchor/phase/split/norm及原source时间。补来的256-square右目需按对应裁剪区域恢复有效内容，与现有PPU裁剪版本一致处理。
2. FastWAM codec当前只准入Stage2/u15853或旧Stage2/u8000，需按本次精确Stage3哈希与结构做定向准入，并部署冻结codec及LAS2所需源码/依赖，不能禁用strict加载。
3. Alpha154000自包含finetune入口必须保持训练模块精确初始化后仅替换冻结codec；当前直接覆盖encoder配置会寻找Alpha不存在的stereotok.json且严格权重不兼容。须实现/验证这一边界，不能from_scratch或resume Wan50101代替。
4. 准备独立StereoTok 32卡launcher/config/W&B身份/output（不修改Wan历史launcher或复用其恢复目录），保留batch2×32×accum2/global128、LR1e-4、500warmup、1epoch与原更新预算。smoke核对左目及action/state/mask完全一致、右目同步、codec严格加载/finite前后向、32卡collectives；过门禁才正式训练。正式新代码须精确commit/push并从clean源启动，当前未获本轮commit/push授权。

权重传输因低速已停止：第二次134,217,728bytes采样累计1.32MB/s，预计剩余3376s/56分钟，遵循用户先前“太慢停止交同事更快方式”的要求，仅StopJob dlcf2egi3xxfonj4，不影响ABC右目dlc16j4n4bins4rh。已精确移除深圳DSW本次公钥行并StopInstance。PPU的stereo.pth.partial保留，不能作为完成权重使用。完整已验收源位于深圳旧CPFS /mnt/workdir/frank/checkpoints/stereotok-stage3-clean-u8000-20261109/stereo.pth，目标PPU15相同绝对路径；完整件SHA256须为4c024b1dc9d9280e94a753ab9ccb1de15cc3c24659a1ec5aa86f94a3805f3503。同事可迁移这一个4,606,154,269bytes文件，无需搬整个checkpoint目录或训练态ckpt。最后ABC进度25.453GB/32.482GB，6939完整校验文件，累计23.986MB/s，剩余约293秒；这是中间进度，不是完成。

用户再次明确要求迁移u8000，认为ABC传输结束后速度应恢复。已核实ABC dlc16j4n4bins4rh输出TRANSFER_COMPLETE 8687 32481530726，所有完整文件流式SHA256通过。恢复本人CPU-only DSW并核实当前SSH映射仍120.77.254.228:1024，启动权重续传dlc1yuagp75jp3r7。复用既有.partial长度作为源端tail偏移，先把现有部分加入SHA256，再追加剩余字节；全文件长度/完整SHA256通过才rename。速度统计仅使用本次新增字节，不计旧部分；不把开始续传当作完成。脚本为既有ops transfer-stereotok-u8000.py定向修改，无生产训练代码变化。

续传健康检查：RESUME_OFFSET=209715200；新增64MiB后总276824064/4606154269bytes，新增流量均速1.42MB/s，剩余3052秒约51分钟。ABC已完成后此路径速度暂未明显改善，不能将旧低速确定归因于ABC并发。按本次再次要求搬运继续传输，未再次停止；临时公钥行已移除，已有SSH会话继续。完整SHA256尚未执行完成，DSW仍为本次传输保持运行，迁移完成后需停止该CPU实例。

## 2026-11-09 补充资产与codec依赖检查

用户再次要求提前核查其他资产，未要求此轮启动训练。CPU只读dlc13hvkzv2hsp59完成ASSET_PREFLIGHT_DONE：torch2.9.0、torchvision0.24.0、timm1.0.20、einops0.8.1、triton3.5.0、safetensors0.7.0、transformers4.57.0、sentencepiece0.1.99、h5py3.16.0、scipy1.11.3、av19.0.2、lance0.39.0均可实际导入。requests发出既有版本组合warning，非此次阻塞，未顺带改环境。此为CPU依赖检查，不是PPU kernel/前向通过。

Alpha154000的safetensors key目录已实读，含video_backbone.text_encoder（T5）、video_backbone.vae.model、video/action DiT及proprio_encoder，因此无需为本次替换额外下载单独T5或Wan初始化权重。tokenizer根层没有文件但有子目录，追加dlcdi8lnt5fs6rel定向浅查，避免把根层空误判为缺失。

从深圳已校验的stereo.pth读取实际键：LAS2 matcher包含fnet165keys、cost_agg312keys，共477；encoder84keys、decoder197keys，融合与metric_head均在。现有FrozenLAS2Matcher明确initialize=False，FeatureNetFasterNet(pretrained=False)，无需另搬LAS2_H.pth或ImageNet初始化权重，也无需搬FoundationStereo/DA3训练教师用于本次冻结codec推理。仍须strict-load实测才能确认完整结构兼容。

新发现的具体codec阻塞：Stage3导出额外有metric_head.layers.{0,2}.{weight,bias}，当前FastWAM旧load_student只剥离fusion/decoder.geometry_branch构建旧骨架，既缺Stage3准入，也缺metric_head模块。不得只修改SHA/stage检查后以strict=False忽略这些张量。已在本地Git找到e93a3e1e785926a339b788881c9e211b6a441ec8的MetricHead和导出逻辑：导出有意排除latent_geometry_head，保留metric_head；其load_student根据metric_disparity.shallow_head_weight构建MetricHead。已从该精确commit归档student.py、checkpoints.py、tests/test_metric_disparity.py作为最小实现参考（ops/stage3-e93a3e1-loader-reference.zip，SHA256cc5bf829c3a4aeb11345e2b047cf09174b16f9d5217a74c6a28e322a321ceea7）。这是参考源码资产，不是已部署可运行codec，不声称已验训练run实际源码无dirty差异。

ABC补齐复查：8687条verified记录、unique视频路径8687、总32,481,530,726bytes；所有目标文件存在且size一致；repair manifest对应路径集合完全一致。此前每文件已源/目标SHA校验，本轮未重复读32GB。当前ABC原catalog、ABC stats、DROID stats三份文件SHA与原production-ready.json记录逐项一致，未因补右目改变基线。

结论：当前未发现除正在续传的4.606GB stereo.pth外还需跨集群搬的大权重/数据。尚缺的主要是小型代码/配置资产及真实接入验收：Stage3完整codec、ABC真实右目映射与预处理、DROID stereo输出、Alpha训练模块严格保留、独立StereoTok launcher/config及发布SHA。此轮未安装新依赖、未修改训练代码、未提交GPU任务。

Tokenizer补充验证dlchkt2bv1hkm2ev：精确目录tokenizer/google/umt5-xxl包含tokenizer_config.json、spiece.model、special_tokens_map.json、tokenizer.json；AutoTokenizer.from_pretrained(local_files_only=True)成功，T5TokenizerFast真实编码9tokens，无需下载。权重续传最新2,759,852,032/4,606,154,269bytes，新增流量均速1.23MB/s，剩余1504秒约25分钟，尚未完成。

## 2026-11-09 用户授权代码适配与smoke：本地实现

已核实迁移日志COMPACT_ARCHIVE_VERIFIED 4606154269 3587.85（完整SHA256通过）；停止仅供传输的深圳CPU DSW。OpenWAM当前5452288、FastWAM当前a239a95，fetch成功，当前HEAD等于各自已跟踪远端HEAD。保留所有用户原有dirty修改，不pull、不建分支、不暂存、不commit/push。

本次OpenWAM补丁：
- openwam/dataloader/orion_midtrain_ppu.py：DROID透传既有stereo输入模式；ABC保留既有左目/监督计算，新增真实右目方法。group02沿用native crop；缺失group按native_source_episode_index查询locator-repairs.jsonl，取原右目裁剪有效区域后恢复256画布。时间相位、帧有效mask、原split/norm不变；左右时间原点不符直接失败。补齐根路径由right_video_root显式配置。
- openwam/train/utils/ckpt_model_loader.py：仅native Wan→StereoTok finetune路径先构建native骨架、完整严格加载Alpha，再替换冻结codec。检查latent/patch合同，不重建任何DiT参数；codec初始化用fork_rng隔离，不改变训练/采样RNG。resume和其他codec不进入替换路径。
- openwam/model/video_backbone/encoder/stereotok.py：部署/恢复骨架按geometry补齐metric_head，保持strict完整加载。
- configs/midtrain_droid_abc_alpha_stereotok_ppu.yaml：继承原PPU配方，仅改变codec/双目输入/补右目根/独立输出环境变量/显示名，batch2、accum2、global128、LR等不变。MIDTRAIN_STEREOTOK_OUTPUT必填，防止误用Wan历史输出。
- tests/test_ppu_stereo_adaptation.py：5项CPU测试通过，覆盖非codec参数对象与数值保留、codec冻结、不兼容拒绝且不改原模型、原视频与native-crop右目相位/裁剪/尾mask、左右时基不一致拒绝。

FastWAM定向修改checkpoints.py/student.py/stereotok_vae.py并新增tests/test_stereotok_stage3_loading.py；1项CPU复合合同测试通过（Stage3严格roundtrip、缺失头tensor失败、stage错误失败、冻结与输出形状）。详细记录见FastWAM docs/26-11-09/26-11-09-stereotok-stage3-u8000-midtrain-adapter.md。未动用户已有fastwam.py或tests/test_robotwin_stereotok.py修改。

本机默认Python无torch；使用已有StereoTok Windows venv的torch/unittest测试，未安装依赖。该环境缺omegaconf/pytest，未声称完整训练模块集成测试通过。下一步GPU smoke必须先发布这两个仓库的精确版本；FastWAM AGENTS要求Local edits → user pushes → server pulls，不能用DLC内嵌未发布生产源码绕过。当前任务未授权commit/push，GPU smoke暂未提交。

拟GPU验收：先PPU1.5单节点8卡，batch2/accum8维持global128，短程真实DROID+ABC双目forward/backward（正式32卡仍accum2）；逐项比较mono/stereo左图和监督，strict-load完整Stage3、验证冻结codec与Alpha非codec权重，检查finite losses/grad和显存；测试保存/重载metric_head；通过后再验32卡collectives，正式训练单独输出/run ID。不得复用原Wan launcher的自动恢复目录或WANDB_RUN_ID。

用户已明确授权本次修改commit/push，只做PPU1.5 8卡smoke；完成后必须停下并等用户确认，不能启动正式32卡训练。新增tests/smoke_ppu_stereotok_midtrain.py是有界验收入口：先与已完成Wan源码1896bf7比较真实train/val样本（两源、相位、group01补右目/group02原右目、尾窗）、完整解析配置差异白名单、Alpha所有非codec tensor逐项比对，再8卡batch2/accum8/global128执行24micro/3updates，scheduler仍用25051updates，做验证、保存full-state、load后继续一步、独立部署骨架与codec参数重载。GPU结果未取得前不宣称干净对照验收完成。正式32卡不在此次提交范围。

已发布OpenWAM fc034e10a20ab3eb9b39a4b39077e56f7481bd95、FastWAM491e518fac6e04646ecfbd464027dd575ed99bde；CPU部署dlc19wqz9nna2pvy在/mnt/workdir/frank/projects/openwam-stereotok-stage3-u8000-20261109/{openwam,fastwam}核对clean+SHA通过。8卡smoke dlc1wxta1775ak16已提交，quota1rpp98rnebp/ForceQuotaOverSold，单节点8GPU/64CPU/768Gi/90min，输出/mnt/workdir/frank/runs/openwam-stereotok-stage3-u8000-smoke-20261109/attempt1。当前EnvPreparing，尚无通过证据；没有提交32卡训练。

首轮8卡dlc1wxta1775ak16在导入阶段失败，尚无训练：直接运行tests下脚本使tests/dataloader遮蔽Orion dataloader.pretrain。定向修正smoke入口sys.path[0]为仓库根，不修改生产数据/模型代码；先以CPU验证导入再重试。

CPU预检dlc1jcekujid5mih完成DATA_PARITY_PASSED 32及CPU_PAIRED_DATA_PREFLIGHT_PASSED，证明32个真实train/val样本（DROID/ABC、原右目/补齐右目、三个phase与尾窗口）的左目所有像素、action/state/mask/prompt与冻结Wan1896bf7逐项一致；ABC全episode元数据列表相同。不是全量视频解码证明。修正入口代码已发布896951b6d8186ef133793768b264d844b248d61c，8卡重试使用该SHA及FastWAM491e518。

第二轮dlc8gclqj08b99qb：8 ranks均严格完成Alpha+Stage3加载，随后rank0源文件dtype/参数一致性assert在action_decoder.bias失败，未训练。CPU诊断dlcj9y68be0dnnae因PPU xformers导入即查询GPU而失败，不能在0GPU容器完整构建模型。smoke验收改为直接构建原Wan加载结果与StereoTok逐tensor、dtype精确比较，并额外与源tensor按原加载dtype转换后的值精确比较（非allclose，不忽略差异）；打印dtype证据。额外baseline构建在fork_rng内，不扰动smoke随机流。

第三轮8卡dlc1c4ry17tg44ar使用OpenWAM c134e14827a48c5a5b2d7123379035952d5dfddf、FastWAM491e518fac6e04646ecfbd464027dd575ed99bde，输出attempt3。实际日志已确认DATA_PARITY_PASSED 32、INIT_DTYPE torch.bfloat16 torch.float32 WAN_STEREOTOK_EXACT、ALPHA_NONCODEC_TENSORS_EXACT 1893。第二轮失败确认为测试将源文件BF16与原生加载FP32直接比较；两条实际加载路径1893个非codec张量dtype和值完全相同，且精确等于源文件按原生dtype转换结果。解析后的正式配置差异白名单通过；当前首次PPU模型编译，尚无训练通过证据。没有提交32卡任务。

第三轮实际完成两次optimizer update（loss 2.5734506、2.8148336；第二步13.4129s，峰值allocated45.2407GB），权重与8份optimizer分片均保存。立即load时非rank0在rank0写trainer_state.json前读取，报Incomplete full-state checkpoint；save_full_state最后collective在rank0发布marker之前。仅在smoke保存和立即加载间增加wait_for_everyone，保留完整性拒绝，不修改生产训练或保存格式。再次8卡完整重跑，尚不算smoke通过。
