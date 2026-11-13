# H100 真机 v3 训练准备与 smoke

用户确认：Wan/Stereo 各自从 midtrain step 50101 初始化；叠杯子各 10 optimizer updates，8 卡串行、最多同时 8 卡；后续新任务各 10 epoch。已授权独立分支/worktree、最小适配和 commit/push。基线为原训练提交 9b87cab7715c026cc5f4762dcb44fc9f95fcd094，新分支 frank/realrobot-v3-training。

## 实现

- `openwam/dataloader/realrobot_v3.py` 移植已验证 PPU v3_reader，仅将代码路径、数据路径和任务标签参数化。保留最近邻时间容差35ms、20Hz recorded-pose TCP20、rowRot6D、raw-gripper zscore、17帧/34动作、左右六相机及相同 mosaic。新数据必须重新生成自己的窗口与统计；不将 action16 joint command 当作监督。
- `scripts/train_realrobot_v3.py` 复用原 train_v3、OpenWAMTrainer 和 ZeRO2：B4/rank、GA1、global32、LR/actionLR/videoLR3e-5、warmup100起3e-6、cosine底3e-7、AdamW(.9,.95)/wd.01/clip1、BF16/fused/no-offload、gradient checkpointing、spawn2/prefetch2/persistent、seed42。独立读取明确的50101文件。
- smoke 禁用W&B，真实10次更新，保存step10权重，不保存optimizer。正式入口强制W&B online，初始化失败不得降级；正式训练尚未启动。
- 不自动删除文件。正式入口默认保存每epoch权重及完整state共10份，与旧PPU rolling-last2有存储差异；正式启动前按实际新任务和容量复核。当前只运行smoke，不扩大正式训练存储授权。
- `tests/verify_realrobot_v3_training.py` 验证222条叠杯子首/中/末窗口、原样本归一化一致性；训练后验证10步有限loss/梯度、保存权重键/shape与源一致、全tensor finite、action参数实际更新、完整SHA。
- `tests/realrobot_v3_training.sbatch` 使用已推送SHA的clean clone，8卡、64CPU、768Gi、2小时、no-requeue，Wan→Stereo串行，前路失败即停止。复用已验证容器和OpenWAM环境，不改共享runtime。

## 资产与运行

H100资产根：`/gpfs/jiuquyun/projects/Frank/realrobot-v3-train-20261113`。
PPU源导出dlc6ncglndpmms69；H100接收CPU Slurm54554。bundle已验证1571文件，叠杯222episodes/29070windows；两份50101权重总52,824,136,764bytes仍在传输。须等待transfer/all-assets-verified.json再启动GPU。
模型卡：HikJbmqosaZDdTsJ7INcNHE1nQc / tblBYVmMJLV8vKTh / reczz28ZCIsjkBSs。

依赖固定：orion-0 783b602d128a43814af32285e60a60f270ec2703、FastWAM 491e518fac6e04646ecfbd464027dd575ed99bde；runtime为既有source-runtime.sqsh，解释器/mnt/workdir/frank/envs/openwam-baselines-20261014/bin/python。Stereo codec从自包含权重和stereotok.json重建，历史MIDTRAIN_STEREOTOK插值只用于配置合并，不读取原PPU codec文件。

初始验证：三个Python文件AST通过，ruff check通过，git diff --check通过。GPU结果待补，不把静态检查当作smoke通过。

新任务完整发布包到位后的CPU准备入口：

```bash
python -m openwam.dataloader.realrobot_v3 --data /absolute/new-v3-release --prepared /absolute/new-preparation --task task_slug --source-task '发布包中的准确任务名'
```

正式入口使用显式data/prepared/task/checkpoint/output，去掉--smoke，固定10epoch；新数据未到位前不提交正式训练。

## H100 CPU验收与调度

CPU检查54556/54557在import cv2时缺libGL.so.1，精确路径的既有库实际存在。Pyxis启动时未保留提交环境的LD_LIBRARY_PATH；改在容器命令内显式env传入，未安装依赖或修改共享环境。54559重试中。GPU launcher同步此定向修复。

核对所有70条预约并定向复核0060的6条记录，23:58–次日01:00为空档；已创建8卡预约reczz28ZCcXl2QTc（2026-11-13 23:58至2026-11-14 01:00，北京时间），关联本模型卡，不更改他人预约。

54559已越过libGL导入，但直接运行tests脚本使tests/dataloader遮蔽Orion的dataloader.pretrain。CPU隔离探针确认Orion源文件及PYTHONPATH正确；仅将验收脚本sys.path[0]设为仓库根，避免测试目录命名冲突。未改生产reader或动作计算。
