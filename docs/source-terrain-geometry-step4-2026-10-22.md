# Lite3 Source Terrain：第 4 步纯几何实现记录

日期：2026-10-22（Asia/Shanghai）。本步依据用户批准，仅实现六类 Source Terrain 的几何生成器和离线几何测试。

## 交付物

- `source/rl_training/rl_training/tasks/manager_based/locomotion/velocity/terrains/source_terrain_tracks.py`
- `source/rl_training/rl_training/tasks/manager_based/locomotion/velocity/terrains/__init__.py` 的公开接口导出。
- `tests/online_adaptation/test_source_terrain_tracks.py`

接口为 `SourceTerrainCfg`、`SourceTerrainGeometry`、`FrictionRegion` 和 `generate_source_terrain`。生成器只依赖 Python 标准库、NumPy 和 trimesh；不依赖 Isaac Sim 运行时。

| 家族 | 本步实现 |
|---|---|
| Flat | 带有限支撑厚度的平坦轨道。 |
| Multi-scale Rough | 至少两个不同相关采样尺度的连续高度场、平地入口、平滑起伏过渡、固定 seed。 |
| Slope | 直线前进轴；上坡、下坡、左右横坡；入口和出口平滑过渡。正 Y 定义为左侧，left 表示左侧高。 |
| Multi-step Stairs | 独立连续上行/下行轨道，显式级数、高度、踏面深度及入口/出口平地；一个封闭网格保留竖直踏步面。 |
| Slope + Rough | 同一坐标网格上的坡面与多尺度起伏相加；单独记录粗糙残差和组合高度范围。 |
| Friction Variation | 左右条带、前后条带和二维区域三种几何分区；每个区域对应独立 mesh、边界及候选摩擦系数。 |

分区摩擦没有绑定 PhysX 材质。其 `effective_friction_confirmed` 固定为 false，调用单材质 `as_isaaclab_mesh_result()` 会明确拒绝，避免合并网格后丢失分区差异。不能把本步结果称为已实现物理摩擦扫描。

## 几何参数与审计边界

- 默认 10 m × 4 m、入口平地 3 m、过渡 0.5 m、楼梯踏面 0.5 m、摩擦区域 4 个均为开发默认值，未锁定为实验参数。10 m 的长度避免默认 20 m 轨道在 0.3 m/s、2000 步、50 Hz 条件下即使理想跟踪也无法完整通过的问题；正式终点、稳定窗口与长度仍需在环境桥接前联合核定。
- Rough 的 `rough_peak_valley_m` 定义为登记采样窗口中的最大峰谷差；默认审计窗口边长 0.4 m，以不超过半个窗口的步距覆盖挑战区并包含末端。生成后分别输出采样局部最大值、P50/P95、窗口数量和整段峰谷差。后者可能大于局部最大值，不能混用。
- 0.4 m 窗口是待 Lite3 接触尺度校准的候选。上述统计是 mesh 顶点的窗口统计，不是所有可能连续窗口的数学最大值，也不是实际足底支撑面积、脚滑或相邻落脚高度差测量。粗糙尺度字段是噪声相关网格尺度，不能当作实测精确波长。
- Flat 的不同难度没有不同几何；初态、命令和动力学变化留待后续独立配置。规则坡面与楼梯在参数固定时也不会因 seed 自动改变角度、踏面或级数。最终 manifest 必须显式登记实例参数，不能把不同 seed 自动视为不同几何。
- 上/下楼梯分别生成单向轨道，并非一条轨道同时包含上楼和下楼；下楼入口高度为总台阶高度，origin 已包含该高度。

## 验证

使用 `/home/tjx/anaconda3/envs/env_isaaclab/bin/python` 运行纯 CPU 测试；没有启动 AppLauncher、仿真或训练。

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=source/rl_training \
  /home/tjx/anaconda3/envs/env_isaaclab/bin/python -m pytest -q \
  tests/online_adaptation/test_continuous_rough_track.py \
  tests/online_adaptation/test_source_terrain_tracks.py
```

结果：**87 passed**，其中新增 81 项、既有连续 Rough 回归 6 项。编译检查与 `git diff --check` 通过。

新增测试直接读取生成 mesh，覆盖封闭性、面朝向、正体积、非退化三角面、起点支撑、seed 字节重复性、Rough 实测窗口高度差、5°/8°/12° 的四种坡向、三档连续上下楼梯、复合场相加、三种摩擦布局在 2/3/4/5/7/17 区域下的完整覆盖且不重叠、单材质误用拒绝、非法参数和网格预算。

首次 pytest 在收集前遇到环境已有的 typeguard/typing_extensions 插件不兼容；关闭第三方插件自动加载后通过。没有修改环境依赖。

## 当前状态与下一步

外部扫描清单 `artifacts/source_qualification_scan_20261021_draft/scan_manifest.json` 保持 `draft_not_for_run` 和 `execution_authorized: false`，未改写清单、未加载 PPO、未进行任何环境回合，也未读取 Sealed 结果。本步验证不代表现有 manifest 已可运行或地形已获得训练资格。

建议下一步在用户批准后，实现扫描 manifest 的 schema 与运行前校验及其离线测试：拒绝草案执行、限制 source_qualification_scan split、检查 seed 分区与回合计数、区分固定动力学与初态 seed、识别未锁定几何/材质/安全/hash 字段。本步之后仍不直接启动仿真或完整扫描。
