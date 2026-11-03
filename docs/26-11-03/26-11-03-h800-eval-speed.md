# H800 单卡测评提速

用户已授权独立 branch/worktree、commit/push，以及最小客户端/服务端缓存合同和对照脚本；只用 H800 一张闲时 GPU，不修改现有正式作业。

分支 `frank/h800-eval-speed`，基于当前 H800 服务端 `9259a320a5945f15eab8167f1a888f5238971eb2`；客户端文件先对齐正在使用的 `4a06f9a3c78a40605fb3a4f56d60dffb1251c220`，避免把已有 shared-plan/四元数修正混成提速效果。

实现：服务器响应明确给出 `needs_observation`；只有同步动作缓存非空时允许不带图像的 `cached_action` 请求。保持每动作一次服务器调用、原生 binary-command 投影和 EEF 转换。async 始终需要新观测，reset 后必须新取图；不推测 horizon。SharedPlanPolicy 要求协议版本 1，禁用歧义网络请求自动重发，并拒绝动态灯光/视频记录模式；不改模型参数。

验证：静态 AST 与 diff 检查；追加既有 server tests 验证 horizon=1/3/32 的动作顺序、生成条件、二值投影、重规划边界和 reset。Windows 系统 Python 缺 pytest，完整 CPU 测试拟在已有 H800 CPU 环境执行。首次本机读文件默认 GBK 解码失败，已明确 UTF-8 后重新检查；此失败未到达远端。

真实对照入口：`tests/h800_eval_speed.py run <独立输出目录>`。复用 FastWAM 冻结 evaluator 的 `rollout_case`，使用原任务计划中的 open_laptop/14（历史短成功）、open_laptop/1（历史失败）、open_microwave/99（历史1500步失败）。不改变这些回合的上限、seed、instruction。先旧服务器/旧客户端，再候选服务器/候选客户端，同一 GPU 顺序执行；对比所有生成输入、动作 chunk、每步执行动作、环境 RNG、初始场景/图像、成功结果和耗时。保留原生模型 seed/compile/cache 设置，不根据期望成绩调整。

checkpoint：`/mnt/workdir/frank/runs/robotwin-u15853-mono-20261101/openwam/checkpoints/2026-11-01_22-36-55/checkpoint_step_30000.safetensors`，28,010,055,404 bytes。基线服务器/客户端与 FastWAM 路径在脚本中明确绑定，candidate 必须 clean。当前没有真实 GPU 结论；排队开始 ETA 未知，首轮预计约 45–90 分钟（含模型加载/编译，需首轮实测修订）。

小样本用于验证与定位，不替代正式全量分数。若动作/输入出现无法解释的差异，不启用候选。

CPU 实测：H800 的 0GPU DSW，隔离 pytest 依赖位于 /mnt/workdir/frank/runs/openwam-h800-eval-speed-20261103/test-deps，使用现有 system-libs-prepared.json 的 LD_LIBRARY_PATH；tests/test_policy_server_cli.py 与 tests/test_robotwin_interface_proprio.py 共 35 passed in 30.01s。没有修改共享环境。追加客户端缓存请求/reset 边界测试后再执行该定向测试。

追加单测 1 passed in 3.54s，总计36项通过。首轮 `dlc1h03217rsep7v` 在加载模型前因字面 H800 名称检查失败；没有产生回合。独立单卡身份查询 `dlce5pb0rvz7ydvq` Succeeded，实际 NVIDIA L20Y，Hopper、compute capability9.0、PCI 0x26BA10DE、81559MiB、MIG Disabled，UUID GPU-b98be483-045c-758e-2ed2-70e7dea0d5e0。此卡型与该阿里云集群既有评测记录一致；H800 是任务中的集群称呼，不能据此宣称商品型号已证实为 H800。将检查改为上述实际卡型/PCI/架构/显存及单卡约束，保持同一research配额和闲时模式。原日志全部保留。

`dlc17uhuyzs1i587` / pair02：旧版完整基线成功86步/205.062s、失败700步/603.497s、失败1500步/1170.330s，取图分别53.627/436.303/876.289s。候选首回合36步且首次生成的 proprio 哈希不同，故立即停止并确认 Stopped，禁止采用该候选速度结果。根因是构建候选时只恢复4a06f9a3客户端入口，遗漏其 `benchmarks/utils/action_conversion.py` 中的wxyz转换依赖（9259a32仍是xyzw）。补齐该文件，并同步4a06f9a3的实际原生位姿测试，保留新增缓存/reset测试。属于恢复原测评合同，不是修改数据语义或放宽门禁。

复测允许显式引用已完成baseline，只在GPU UUID、设备字段、case列表、config哈希完全一致时运行；保留baseline来源路径且不改旧结果。若物理GPU不同则拒绝复用。原生成输入/动作差异门禁不变。

修正后pair03首个输入完全相同，但独立服务进程的首个原始动作chunk不同（最大0.0078125，平均0.0002772，137/640元素不同），短回合85步而旧版86步；再次停止dlcap24rdtr4tteq并确认Stopped，不放宽阈值。该差异发生在跳过观测之前，需区分进程/编译数值差异与缓存优化。追加同一预热模型实例的诊断：先对同输入连续推理3次，要求第2/3次完全相同；随后旧路径短回合重复两次必须完全相同，再做旧/新短回合精确门禁，只有通过后才运行完整700/1500步对照。模型配置、seed、采样步数均不改。此实验测量缓存路径的因果影响，单独披露跨冷启动非逐位复现现象。

pair04 / `dlcxbfjt7hdtztix`：同输入3次推理逐位相同；旧路径短回合重复84步，生成输入/chunk/执行动作/RNG全部相同；候选也是84步且上述全部相同，取图84->3。预热旧路径92.970s，候选47.407s，约1.96倍；首次含编译旧路径201.033s不能用于报告优化倍率。开始700步旧路径后平台明确 preempted/evicted，作业Failed；不是程序失败，保留所有结果，未完成长回合不计入门禁。

后续同卡单模型诊断改为每个长回合旧/新紧邻完成并逐对保存比较，减少抢占时丢失进度；不缩短回合、不改变样本或资源。仍只申请1GPU闲时，队列开跑时间未知。

pair05 在模型加载前因远端 SHA 不符而失败：此前 git pull 的 TLS/GnuTLS -110 错误没有更新 clone；HTTP/1.1 重试 fast-forward 成功后确认 clean HEAD，才启动 pair06。pair06 / `dlc3cufbfhv7iai9` 精确 SHA `8c72a64e6b4a49d56391075eef7c9f563bb6ae61`，1GPU闲时、priority4，输出 `/mnt/workdir/frank/runs/openwam-h800-eval-speed-20261103/pair06`。

修正后的 CPU 定向测试最终 39 passed in17.81s。pair06 同输入推理及旧路径重复均逐位相同；85步成功回合预热旧90.10s、新44.26s，约2.04倍；700步失败旧605.824344s、新170.508445s，3.553倍，取图700->22。两样本输入/chunk/执行动作/RNG/初始状态/成功结果全部精确一致。1500步长回合仍在执行，尚未形成全套最终结论。诊断 comparison.json 中首次短旧路径包含编译，最终报告必须替换为 repeat/open_laptop-14.json 的预热计时，禁止引用首次短回合4.43倍。

## 复现及后续入口

已验证代码位于 `/mnt/workdir/frank/projects/OpenWAM-h800-eval-speed-20261103`，固定上述 SHA。配对诊断使用单卡 DLC 中的命令（OUTPUT 必须为新的独立绝对路径）：

```bash
/mnt/workdir/frank/envs/openwam-baselines-20261014/bin/python -u /mnt/workdir/frank/projects/OpenWAM-h800-eval-speed-20261103/tests/h800_eval_speed.py run "$OUTPUT" --shared
```

脚本自动设置原有运行库路径并依次执行同一模型实例中的旧/新客户端，不改正式结果。CPU DSW 仅用于查看日志，不能在其中执行此 GPU 命令。

将来新评估沿用 FastWAM `scripts/c2r/evaluate.py rollout --root <新结果根>` 和原冻结 manifest。服务端使用候选 clone 的 `scripts/deploy.py`，原 ckpt-dir/ckpt-name 和推理参数保持不变；客户端 PYTHONPATH 中把原 OpenWAM 客户端根替换为候选 clone，使 `benchmarks.robotwin.openwam2robotwin_interface.SharedPlanPolicy` 来自同一候选版本。保持现有 client-deps、FastWAM/src 和 projects 路径及 Vulkan/LD_LIBRARY_PATH 环境。两端同时更新才能通过协议版本1门禁。缓存优化只用于已验证的静态灯光、禁用视频 Clean 合同；未修改或重启任何既有正式作业，也未创建正式全量评估。

## 最终验收结果

pair06 作业 `dlc3cufbfhv7iai9` 已由 GetJob 确认 **Succeeded**（PyTorchJob successfully completed），所有子进程正常收尾，单卡验证结束。测试代码 SHA `8c72a64e6b4a49d56391075eef7c9f563bb6ae61`；之后仅补充本记录。

| 冻结样本 | 结果/动作数 | 预热旧路径秒 | 候选秒 | 提速 | 取图次数旧→新 |
| --- | --- | ---: | ---: | ---: | ---: |
| open_laptop / ordinal14 / seed100014 | 成功 /85 |90.101209|44.257865|2.0358×|85→3|
| open_laptop / ordinal1 / seed100001 | 失败 /700 |605.824344|170.508445|3.5530×|700→22|
| open_microwave / ordinal99 / seed100132 | 失败 /1500 |1167.114856|281.576400|4.1449×|1500→47|
| 合计 |2285动作|1863.040410|496.342710|3.7535×|2285→72|

总耗时减少73.36%，包含各回合环境setup。短回合旧路径计时采用 `repeat/open_laptop-14.json`，其余采用 `baseline/*.json`，候选采用 `candidate/*.json`；不包含模型加载时间，也不将冷启动编译混入收益。原始 comparison.json 保留原始首次冷短回合计时，内含 aggregate_speedup=3.9673，**不能作为最终预热提速结论**；上述3.7535才是按公平计时重算的汇总。

`comparison.json`：passed=true、failures=[]、exact_generation_inputs=true、exact_chunks=true、generation_counts=[72,72]；所有2285个执行动作误差0、逐步环境RNG相同、初始场景/观测相同、任务结果相同。同输入预热重复与旧路径短回合重复也通过精确门禁。CPU定向测试39项通过。优化不改变checkpoint、采样/CFG/compile/cache、模型seed、任务计划、instruction或步数上限。

证据局限：仅两个任务三个诊断样本，不能替代50任务×100回合全量成绩；精确一致结论固定同一预热模型实例。独立冷启动存在已记录的BF16非逐位复现，未通过调容差掩盖。动态灯光/视频录制不在此缓存合同支持范围。实际硬件是该H800集群的单张L20Y/Hopper80GiB，不宣称商品型号H800。既有正式作业、代码和环境均未切换。

达到用户要求的重大提速后停止追加优化；后续正式评估可使用上述双端候选入口及独立结果根，本轮未启动全量任务。

## Random 单卡验证启动

用户授权H800集群1张闲时GPU实测OpenWAM Random，慢时继续优化。复用本分支、既有测试入口、原mono checkpoint及冻结Random计划；不改其他正式任务。诊断环境EVAL_SPEED_PHASE=randomized选择click_bell/0（普通随机场景）、click_bell/19（seed100019自然动态灯光）、open_microwave/1（原生1500步长回合）。先baseline-only测首个旧路径回合两次，隔离冷编译；再视实测结果定向优化。新增参数仅在tests诊断入口，不进入生产路径。每个任务的seed/instruction/步数上限保持冻结值。当前只有CPU静态检查，尚无本轮GPU实测。

Random baseline01 `dlcf2h823k767thu` Succeeded：click_bell/0 seed100000成功64步，旧路径重复的输入/chunk/执行动作/RNG/结果完全相同。首次含编译193.488s；预热重复79.232541s，其中setup25.239947s、取图40.910916s、action13.081507s。取图占总时长51.6%，占setup外75.8%，触发用户授权的定向优化。

候选仅调整SharedPlanPolicy：首次从env读取crazy_random_light并在该回合每步继续请求观测；静态回合沿用缓存合同；reset清除灯光状态、重新要求首次观测。视频仍拒绝。没有修改Random概率/灯光/RNG/模型/环境代码。追加动态与静态取图、reset和视频边界测试；诊断逐动作记录dynamic_light并断言冻结样本的自然灯光类别。后续同预热模型实例配对，严守零差异。

## Random 最终验收

配对作业 `dlc1yuaf4b81pa5b` 已由API确认Succeeded。测试SHA `3505ac85e246b9afe62da7776501bfdee1b53689`；之后仅追加实验文档。GPU实际NVIDIA L20Y/Hopper9.0、81559MiB、UUID GPU-5fd224d6-245c-8ffc-f409-4217f8185c0a，单卡ForceQuotaOverSold/priority4，无额外资源。输出 `/mnt/workdir/frank/runs/openwam-h800-eval-speed-20261103/random-pair01`，基线独立测速在random-baseline01。候选与旧路径固定同一预热模型实例。

| Random冻结样本 | 原结果/步数 | 旧路径预热秒 | 候选秒 | 提速 | 取图旧→新 |
| --- | --- | ---: | ---: | ---: | ---: |
| click_bell/0 seed100000，静态灯光 |成功/64|80.265158|45.013873|1.7831×|64→2|
| click_bell/19 seed100019，自然动态灯光 |失败/400|345.663110|345.157964|1.0015×|400→400|
| open_microwave/1 seed100001，静态灯光 |失败/1500|1314.834608|294.205024|4.4691×|1500→47|

普通Random（未触发crazy_random_light）的两个样本合计1395.099766→339.218897秒，4.1127倍；三个样本合计1740.762876→684.376861秒，2.5436倍。动态灯光在本诊断占1/3，属于刻意覆盖边界，不代表全量Random的自然分布，不能把该总倍率当5000回合倍率。短回合旧路径计时使用repeat/click_bell-0.json，排除首次编译；原始comparison.json的2.7052含首次冷编译，不用于报告优化收益。全部计时包含环境setup但不含进程启动/模型加载。

comparison.json通过：passed=true、failures=[]、exact_generation_inputs=true、exact_chunks=true、generation_counts=[62,62]。1964个执行动作最大误差0，逐步环境RNG、初始场景/图像、种子/指令、成功结果均相同；trace中的dynamic_light分别全False/全True/全False。旧路径重复与同输入预热推理均精确一致。动态灯光400步保留400次取图，不更改灯光过程或Random概率；静态长回合取图时间1006.741570→39.061084s。41项CPU定向测试通过（20.10s）。

后续使用仍为同分支双端候选与原冻结evaluator/新结果根；当前实现已支持含自然动态灯光的Random（上述旧文中的“动态灯光拒绝”描述由本节覆盖），视频记录仍不支持。完整Random诊断复现：在同预算单卡DLC中设置 `EVAL_SPEED_PHASE=randomized`，执行 `tests/h800_eval_speed.py run <新的绝对输出目录> --shared`。checkpoint、推理参数、数据和模型配置均保持原设置，未修改FastWAM/RoboTwin环境代码。

本轮仅两个任务三个完整Random诊断回合，不能替代全量5000成绩，也未验证其他checkpoint/双目输入的Random速度。未启动正式Random全量、未切换其他在运行作业。达到重大提速与零差异门禁，停止追加优化。
