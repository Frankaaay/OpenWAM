# StereoTok RoboTwin Clean 评测链路修复

2026-10-20 用户明确授权修复、commit/push、smoke通过后16GPU Clean评测。起点52cd8e30fc5c72b259651181aba915c637779f02，保留原有两个dirty文档及未跟踪配置，不纳入提交。

远端只读预检dlcos1ibvdzhvw58确认：Wan已发布四元数修复、10步同步去噪、DiT cache0.99、compile及prompt cache默认在；StereoTok最终30K权重28010055404bytes存在，绑定u15853。阻塞是右眼没有通过client/preprocessor/policy/engine传递，模型本身已严格要求双眼。

本次新增right_images负载，按与左眼同一训练布局独立处理，拒绝缺眼/尺寸不符和非stereo意外右眼。policy与engine传递right_first_frame_image。RoboTwin --stereo显式复用既有FastWAM依赖中训练使用的Camera类（头6cm、两腕2cm），只替换相机类，保留已验证RoboTwin任务/规划器/冻结manifest；不复制左眼、不重建seed计划。全量仍50任务各100 unseen回合，16GPU、每卡2worker、5回合动态领取，复用Wan修复后的5000份计划但不复用分数。

六文件AST通过；Windows无torch/pytest，未声称单元测试通过。远端CPU回归及真实GPU smoke待运行。正式训练/W&B/其他测速不修改，当前尚未启动GPU评测。
