# 接入 FastWAM 冻结 RoboTwin plan

当前分支frank/eighteen-preparation，基线160f300；保留原有修改，未commit/push或热改云端测评。

原eval_policy_wrapper.py入口明确拒绝独立测评/生成manifest。统一使用FastWAM-C2R/scripts/c2r/evaluate.py，manifest显式model=openwam、policy={host,port,stereo}；任务、100回合、phase及step limits来自FastWAM固定清单索引。不同checkpoint/模型使用独立输出。

SharedPlanPolicy复用原ModelClient、提示模板和EEF动作转换，不更改服务端推理参数；关闭模型侧step-limit覆盖，stereo必须显式匹配encoder。环境及RNG由共同入口管理。

服务器CPU作业dlc1uxgw7f3idhi0使用原环境加既有系统库路径，4项适配测试通过；真实清单100份/10000回合hash通过。真实SAPIEN/模型服务GPU联调和发布新默认请求尚未完成，现有云端测评仍使用旧代码。完整记录见C:/Project/StereoTok/docs/26-10-21/26-10-21-robotwin-shared-plan.md。

## 授权发布与单卡验收

2026-10-21 用户明确允许提交/推送本次变更和1张闲时GPU串行验收。具名提交保留其他dirty文件。Fast测试入口tests/check_shared_plan_runtime.py仅用于真实Clean/Randomized初始化和两次原生策略调用，不生成正式分数；固定索引LF换行，LingBot空动作步维度拒绝。实际远端结果另行追加。
