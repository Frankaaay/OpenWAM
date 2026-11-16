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
