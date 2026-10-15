# StereoTok 与已发布 RoboTwin 接入

基线 main@4d7f642b841913ea3c176010b692b1de3d5ef180，2026-10-15。本地工作区继续已有 Wan-init 补丁，未 commit/push。

新增 encoder/stereotok.py 注册与 checkpoint-local metadata 重建，依赖现有 FastWAM vendor，严格锁定 u8000 SHA。真实远端临时验证 769 keys strict load/restore 通过，冻结；尚未以完整 OpenWAM checkpoint 作部署恢复验证。

robotwin.py 接受 manifest_path 与 stereo，dataset_dir 必须为原 manifest 路径的根；发布 manifest 声明 wxyz，读取时重排到 rotation helper 所需 xyzw。真实左右相机按原时间索引/相同 jitter 处理；保留已有 t+1 动作、unify 与 tail mask。发布路径必须显式提供所需 normalization stats，避免缺统计静默禁用。验证实得2500 episodes/549034 windows，首中末三窗口native/stereo的公共字段一致；此轮 normalization=None，最终统计未验收。

远端报告位于 /mnt/workdir/frank/runs/eighteen-preparation-20261014/{openwam-codec-production-probe-20261015/report.json,openwam-published-reader-production-probe.json}。详细任务记录 C:/Project/StereoTok/docs/26-10-15/26-10-15-eighteen-production-preparation.md。没有训练、backward、optimizer 或远端源码覆盖。

真实连接验证追加：tests/probe_stereotok_robotwin.py 从生产 factory 加载完整 Wan 权重、u8000 与真实9帧配对，走 prepare/run_block/finalize 的30层路径，输出 [1,48,3,24,20] 全有限；训练与部署首帧 latent max_abs=0。报告 openwam-production-codec-wan-forward.json，64.78秒。此前直接调用旧 dit.forward 的检查脚本因 patchify 返回类型不匹配失败；未修改该非生产入口，改为使用真实生产调用合同。动作分支、完整 checkpoint 部署恢复尚未验证。

新增 robodojo_lerobot.py 只适配已映射的 v2.1 MP4/joint14 与 v3 EEF16 数值表；不生成 HDF5 或伪右目。完整 episode/frame/timestamp/单位四元数检查，逐 episode 验证共同索引与夹爪后，将 achieved poses 交回原 read_calibrated_eef20，复用下一状态监督、normalize→unify80、padding/mask。视频复用已有 PyAV 解码器；stats_computation 使用同一适配器和原 Accumulator 池定义。配置入口 lerobot_eef_paths 必须显式提供，归一化必须绑定 stats 文件。全量实际适配器验证进行中；尚不宣称生产就绪。

后续验收：tests/probe_robodojo_lerobot.py 实际完整3500条/1856102帧映射及 pooled stats 通过，1852602个窗口；首/中/末真实视频窗口动作归一化反解 max_abs=5.96e-8，尾窗口1个有效动作。24.89秒，报告 openwam-robodojo-production-reader-probe.json；明确未全量解码视频。临时 stats-probe.npy 不作为已发布生产产物。

RoboTwin 原统计入口新增 manifest 模式，直接复用 reader 的wxyz转换和原 all_states 池/rot6d identity，不静默跳过坏文件。reader 强制统计metadata的 manifest SHA匹配。实际2500条551534帧全量统计完成，549034窗口，首/中/末两路线的归一化动作/proprio/mask完全一致；报告 openwam-robotwin-normalized-reader-probe.json。manifest模式只接受显式新output，不触发旧统计脚本的递归清理逻辑。

依赖：在原 OpenWAM 环境用 uv pip install --no-deps --no-build-isolation -e 注册已有 clean FastWAM b856e4929c6a552540008808b188729361fa531a，不安装其整套训练依赖以避免与OpenWAM版本冲突。正式执行须绑定该源码SHA及本次补丁发布SHA；不将临时overlay结果当作已推送版本。

发布授权：2026-10-15 用户明确确认创建 Frankaaay/OpenWAM fork，在 frank/eighteen-preparation 分支仅提交任务清单16个文件。基线保持指定4d7f642，不合入上游后续image-only变更。本提交发布已完成的接入与验证，不代表18项已全部具备开训条件。正式生产转换与最终配置验证继续进行，未授权本轮启动训练。

十二项准备：encoder 的训练与部署 SHA 绑定已由 u8000 改为用户选定 u15853（61c6057b…252eae）。AST 通过；依赖 FastWAM 新 loader 发布，尚未进行本轮 GPU 联合 smoke。
