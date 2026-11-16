# 补充核验记录

Python 为 `/home/tjx/anaconda3/envs/env_isaaclab/bin/python`，当前代码工作树为 `/home/tjx/rl_training_main/.worktrees/lite3-ea-lpl-20261106`。

## 针对性测试

执行：

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 /home/tjx/anaconda3/envs/env_isaaclab/bin/python -m pytest -q tests/source_curriculum/test_reference_mujoco_deployment.py tests/source_curriculum/test_reference_mujoco_contacts.py tests/source_curriculum/test_reference_capability_mujoco.py
```

结果：9 passed in 0.89s。关闭第三方 pytest 插件自动加载，避免环境中无关插件的依赖冲突。

## 保存动作与运动学复核

主分析入口已逐项复算全部新轨迹的命令、观测、三个半径的支撑、终止事件及初态。本补充检查只读取轨迹，不推进物理，不增加策略回合：

- 对 `complete.json` 的 36 回合和 `smoke.json` 的 4 回合，按模型汇集轨迹。使用已锁定检查点的冻结 actor 对每回合全部 `observation` 做批量推理，与原单步推理保存的 `raw_action` 比较；容差为 atol=2e-5、rtol=1e-5。
- zero10100：19,686 个输入，最大绝对差 2.86102294921875e-6。
- B：19,343 个输入，最大绝对差 2.384185791015625e-6。
- 批量/单步浮点计算有上述微小差异，均通过容差，不将其写成逐位相同。
- 对全部 39,029 个保存状态，用 qpos/qvel 调用 `mj_kinematics`，足部位置与记录的误差不超过 1e-12；由四元数重算投影重力、roll/pitch/yaw 和体坐标速度，均与保存数组精确相同。
- 独立接触入口已在旧 23,257 个保存状态上复现全部三种半径的历史终止事件，见 `baseline_contact_validation.json`。
- 磁盘上实际新回合目录计数为正式 36、接口检查 4。没有自动重试目录或新增训练。

动作核验使用 `reference_capability_mujoco_deployment.actor()` 加载，观测来自对应 `trace.npz`；运动学核验使用同模块 `make_model()`、`rotation()`、`angles()`，不调用 `mj_step`。本记录的数值来自独立只读核验命令。

## 历史文件与媒体边界

`git diff HEAD` 确认历史 `reference_capability_mujoco.py`、`reference_capability_mujoco_contact_audit.py` 和 `reference_capability_mujoco_report.py` 无修改。执行计划中权重、碰撞网格、参考项目源码和旧轨迹的 SHA256 全部重新核对通过。

`git check-ignore` 确认新目录下 MP4 与 PNG 命中媒体忽略规则；实际提交前另检查暂存清单。全部视频最终解码、尺寸、帧数和 SHA256 核验见 `verification.json`，文件清单见 `artifact_manifest.json`。图片已抽查上阶片段，最终总览另作目视检查。
