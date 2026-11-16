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
- CPU环境导入检查、完整资产哈希/解压及两组8卡有限验收尚待完成。GPU ETA 必须来自本次实测；交接文档 FastWAM 耗时不能直接套用 OpenWAM。

日志位于 `/gpfs/jiuquyun/home/Frank/logs/`；后续 JobID、SHA、导入路径及验收结果在本文件和运行根 run-receipt.md 回填。
