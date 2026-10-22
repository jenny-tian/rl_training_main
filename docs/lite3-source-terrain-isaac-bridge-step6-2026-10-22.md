# Lite3 Source Terrain：第 6 步 Isaac Lab / PhysX 接入记录

日期：2026-10-22（Asia/Shanghai）。本步按批准范围把六类 Source 几何接入 Isaac Lab，并验证分区摩擦材质绑定及最小仿真步进。本步不是模型能力评估或训练资格。

## 交付物

- `source/rl_training/rl_training/tasks/manager_based/locomotion/velocity/terrains/source_terrain_isaac.py`：Isaac Lab 子地形配置和标准地形回调；为摩擦分区面携带内部系数标记，并保留纯几何生成器无仿真依赖的边界。
- `source/rl_training/rl_training/tasks/manager_based/locomotion/velocity/terrains/source_terrain_importer.py` 与 `source_terrain_importer_cfg.py`：将普通面导入为普通地形碰撞网格，把不同分区拆成独立 PhysX 网格碰撞体并绑定各自刚体材质。
- `source/rl_training/rl_training/tasks/manager_based/locomotion/velocity/config/quadruped/deeprobotics_lite3/source_terrain_env_cfg.py`：六类 Source 环境配置及 L1/L2/L3 几何候选。方向变体通过独立子地形配置加入；默认 120 个地形列承载候选家族配比。该默认不代表扫描已选出训练范围。
- Gym 任务 `SourceTerrain-Deeprobotics-Lite3-v0` 和 `scripts/tools/source_terrain_isaac_smoke.py`。

六类候选概率分别是 Flat 0.10、Multi-scale Rough 0.20、Slope 0.20、Multi-step Stairs 0.25、Slope + Rough 0.15 和 Friction Variation 0.10。环境配置明确保留这些为方案候选；运行器没有注册 PPO 配置，避免把这个接入 smoke 当成可直接启动的正式训练。

## 验证

```bash
TERM=xterm CONDA_PREFIX=/home/tjx/anaconda3/envs/env_isaaclab \
PYTHONPATH="$PWD:$PWD/source/rl_training:/home/tjx/IsaacLab/source" \
/home/tjx/IsaacLab/isaaclab.sh -p scripts/tools/source_terrain_isaac_smoke.py \
  --headless --device cuda:0 --num-envs 6
```

结果：

- Isaac Sim 5.1 headless 创建 6 个环境，六类 L1 代表地形全部进入场景；terrain origins 形状为 `(1, 6, 3)`。
- 独立生成并检查 L1/L2/L3 的 **45 个类别/方向/分区布局配置**；网格有限且封闭。它验证几何可生成，不代表每种配置都分别在 PhysX 场景中碰撞测试。
- friction Variation 的左右分区实例生成 4 个带碰撞 API 的摩擦网格，分别绑定材质。USD 回读候选静/动摩擦值为 `0.6999999881, 0.8000000119, 0.8999999762, 1.0`，与请求值之差小于 `1e-6`。
- 6 个环境 reset 后执行一次全零动作步；观测、奖励和终止张量均为有限值，六个环境该步均未终止或截断。
- 两个高度扫描器均使用 Isaac Lab `MultiMeshRayCaster`，以 `/World/ground` 为扫描目标并将普通网格和 4 个摩擦区网格合并。烟测核对了场景中的 5 个地形 Mesh prim 均位于该目标树下，两个扫描器均成功初始化，读回高度命中均为有限值。这样 critic 的高度扫描不会漏掉摩擦分区几何。
- 场景加载了 Lite3 articulation，但未加载 `model_1998.pt`，未进行 episode 性能评估、基线能力扫描、PPO 更新或训练。

测试命令与结果：

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
/home/tjx/anaconda3/envs/env_isaaclab/bin/python -m pytest -q \
  tests/online_adaptation/test_source_terrain_tracks.py
# 81 passed

/home/tjx/anaconda3/envs/env_isaaclab/bin/python -m compileall -q \
  scripts/tools/source_terrain_isaac_smoke.py \
  source/rl_training/rl_training/tasks/manager_based/locomotion/velocity/terrains/source_terrain_isaac.py \
  source/rl_training/rl_training/tasks/manager_based/locomotion/velocity/terrains/source_terrain_importer.py \
  source/rl_training/rl_training/tasks/manager_based/locomotion/velocity/terrains/source_terrain_importer_cfg.py \
  source/rl_training/rl_training/tasks/manager_based/locomotion/velocity/config/quadruped/deeprobotics_lite3/source_terrain_env_cfg.py
git diff --check
```

上述编译检查和 diff 检查通过。pytest 需关闭环境中不兼容的第三方插件自动加载；没有安装或修改 Python 依赖。

## 边界与遗留事项

- 本次高度扫描核对验证的是射线几何覆盖，不代表 PhysX 接触摩擦与高度扫描之间的数值完全等价。
- 本步验证了摩擦分区的网格拆分和材质绑定，不等同于测量接触中的有效摩擦或脚滑；`0.3–1.0` 仍是候选材质值。
- 六类场景 smoke 使用各一格代表地形；正式的 120 列候选训练分布没有在本步整图实例化，也没有开始 PPO。多方向几何做了离线网格生成检查，但烟测实际放入场景的斜坡/楼梯各为一个方向，摩擦地形为左右条带布局。
- Isaac Sim 在 Lite3 关节/视觉加载期间报告了 `Lite3_physics.usd` 的 visual sublayer 引用无法解析警告；同一警告也出现在既有 FlatOnline smoke。它没有阻止 articulation、地形碰撞体、材质读取或仿真步进，但本步没有修复该资产引用。
- Source qualification manifest 仍为 draft，不应运行 PPO，也未执行冻结策略能力扫描。地形正式难度、seed、完整回合指标和训练配置仍须按方案先锁定。

## 下一步

建议先对冻结 `model_1998.pt` 执行已登记、只读的 Source 难度扫描，覆盖坡向和连续上下楼梯，并先核实扫描 manifest 的锁定字段与执行条件。该扫描不是本步授权内容，需另行批准后才开始。
