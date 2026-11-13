# 挂马克杯 H100 posttrain

用户明确授权：使用youzi指定candidate-v1数据，Wan/Stereo各自50101初始化、各10epoch、各8卡闲时资源（允许合计16卡），W&B offline，自动重排恢复；不再额外跑新数据GPU smoke。新指令覆盖先前online和8卡总并发限制。只读源：/gpfs/jiuquyun/projects/youzi/teleop-20261113/releases/teleop-20261113-mug-rot180-candidate-v1。

01:16检查：源meta及PACKAGED最后修改00:36:56，selection partial=false，245episodes，youzi无Slurm任务；这不证明无远端后台写入。准备复制到本人固定快照，前后检查源目录文件size/mtime，发生变化则停止，避免边写边训。源任务label/en均为挂马克杯，保持原prompt，不擅自翻译。头部左右视频已rot180，不重复旋转。派生窗口与统计沿用recorded-pose监督，绝不使用旧叠杯子stats或action16关节指令替代。

最小代码改动：已有训练入口切为用户指定offline；--auto-resume读取active-output，核对完整数据合同，复用已有Accelerate/DeepSpeed恢复模型、optimizer、scheduler、RNG与epoch。完整state写完后原子trainer_state标记，忽略半写state；step0及每epoch保存，保留11份state和10份权重，不批量删除。若未完成第一个state，保留旧尝试并从50101新开尝试；完成result原子落盘，重排发现complete直接退出。重放未落盘epoch时允许覆盖其不完整checkpoint，已完成epoch不重放。

正式Slurm入口tests/realrobot_v3_posttrain.sbatch：单模型8卡/64CPU/768Gi/12h，--requeue，不绑定节点，按闲时调度；另一路独立job。重排从最近完整epoch恢复，可能重算被抢占时未保存的部分；不承诺OOM/任意程序错误自动无限重试。不额外跑GPU恢复演练，运行验证范围会如实记录。

运行路径、JobID、统计与初始健康检查待实际提交后回填。
