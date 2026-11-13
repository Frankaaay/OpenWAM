# 挂马克杯 H100 posttrain

用户明确授权：使用youzi指定candidate-v1数据，Wan/Stereo各自50101初始化、各10epoch、各8卡闲时资源（允许合计16卡），W&B offline，自动重排恢复；不再额外跑新数据GPU smoke。新指令覆盖先前online和8卡总并发限制。只读源：/gpfs/jiuquyun/projects/youzi/teleop-20261113/releases/teleop-20261113-mug-rot180-candidate-v1。

01:16检查：源meta及PACKAGED最后修改00:36:56，selection partial=false，245episodes，youzi无Slurm任务；这不证明无远端后台写入。准备复制到本人固定快照，前后检查源目录文件size/mtime，发生变化则停止，避免边写边训。源任务label/en均为挂马克杯，保持原prompt，不擅自翻译。头部左右视频已rot180，不重复旋转。派生窗口与统计沿用recorded-pose监督，绝不使用旧叠杯子stats或action16关节指令替代。

最小代码改动：已有训练入口切为用户指定offline；--auto-resume读取active-output，核对完整数据合同，复用已有Accelerate/DeepSpeed恢复模型、optimizer、scheduler、RNG与epoch。完整state写完后原子trainer_state标记，忽略半写state；step0及每epoch保存，保留11份state和10份权重，不批量删除。若未完成第一个state，保留旧尝试并从50101新开尝试；完成result原子落盘，重排发现complete直接退出。重放未落盘epoch时允许覆盖其不完整checkpoint，已完成epoch不重放。

正式Slurm入口tests/realrobot_v3_posttrain.sbatch：单模型8卡/64CPU/768Gi/12h，--requeue，不绑定节点，按闲时调度；另一路独立job。重排从最近完整epoch恢复，可能重算被抢占时未保存的部分；不承诺OOM/任意程序错误自动无限重试。不额外跑GPU恢复演练，运行验证范围会如实记录。

CPU54578完成：5709文件/436451578bytes快照，源清单size/mtime前后不变。快照根`/gpfs/jiuquyun/projects/Frank/mug-v3-posttrain-20261114/data`；snapshot.json记录每文件SHA，清单SHA256为46a305e7052b2308e1084b135d8c31ef5686955696a69f50aa65cf0e33792129。独立统计位于同根上一级preparation，245episodes/17755windows，stats SHA256为05f3073d61786af408bd74cbd4b1498798e117ed3f849be3e64bacbde2513222。每epoch555次更新，正式10epoch共5550。沿用prepare的state parity与样本读取检查，没有追加GPU smoke。

实际运行commit13c0685928c4eaf2094db8b2da0308884319dd09，静态Python编译、ruff、git diff检查通过；远端bash -n、sbatch --test-only通过。Wan54580在0059，Stereo54581在0060，均01:22:16开始，每路8GPU，AdminComment=h100:idle，Requeue=1，Restarts=0。两路均afterok依赖54578。输出`/gpfs/jiuquyun/projects/Frank/mug-v3-posttrain-20261114/formal/{wan,stereo}`；日志`/gpfs/jiuquyun/home/Frank/logs/mug-v3-posttrain-{54580,54581}.out`。模型卡Wan reczz28ZF4i09TVk、Stereo reczz28ZF4i08NcW，保持旧smoke卡及结果不覆盖。

H100任务根保存run-receipt.md。未做GPU抢占注入演练，不声称完整恢复链已实测。已保存状态的恢复复用项目既有实现。

启动健康检查已通过：Wan输出formal/wan/2026-11-14_01-26-47，offline run xstdkdt2；检查时27步，total loss1.31139565→0.91822720，末步grad_norm1.60182。Stereo输出formal/stereo/2026-11-14_01-25-21，offline run9w5nagmy；检查时50步，total loss1.17616320→0.55909085，末步grad_norm1.31271。两路完整accel_state_step_0/trainer_state.json均已写完，真实前后向和optimizer更新正常。用户指定offline覆盖仓库默认online，不是静默降级。W&B的offline resume提示仅表示日志新建本地run，不影响DeepSpeed状态恢复逻辑。训练尚未完成；若无抢占，按约1秒/step与完整state写入估计剩余2–3小时。按项目规则完成启动健康检查后不循环监控整场训练。
