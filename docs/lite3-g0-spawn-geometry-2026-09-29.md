# G0.6 出生几何连续范围审核

已验证当前 Lite3 站姿在 x/y 各 ±0.02 m、yaw ±2° 的完整连续扰动范围内不会穿入出生平台，也不会越出平台。该结论仅涉及出生几何，不证明行走稳定性或适应收益。

## 证据与范围

实际读取部署 USD 的22个碰撞体（Cube、Capsule、Sphere），包括实例代理中的形状。碰撞体到刚体的变换来自组成后的 USD；刚体世界姿态来自 reset 后的 PhysX 数据，避免使用 Fabric 未同步的 USD 世界变换。

对各形状解析计算世界坐标各轴支撑范围，再用相对 root 的水平包围圆覆盖所有 world-Z yaw 旋转。XY 区间额外扩张2 cm。固定关节姿态、root高度与roll/pitch时，yaw和XY平移不改变对地间隙；这给出连续范围证明，而不是仅凭角点抽样推断。

| 项目 | 数值 |
|---|---:|
| 中心root高度 |0.375 m|
| 水平保守包围半径 |0.3763050882 m|
| 扩张后X方向平台边距 |1.1036949118 m|
| 扩张后Y方向平台边距 |1.6036949118 m|
| 最小碰撞几何对地间隙 |0.0265100528 m|
| 审核容差 |0.000001 m|

出生平台为 x∈[-5,-2]、y∈[-2,2]、z=0。RockyRoad生成器的这段区域独立于岩石随机种子和粗糙振幅；证书绑定该生成器源码。无限平面的地面高度相同，水平边界更宽，几何结论也适用于当前平地保持路径。

单GPU独立审核了中心和8个 x/y/yaw 极值组合：全部通过。实际root偏移和yaw与设定误差小于1e-6，关节位置一致。全部案例保留，未通过行走结果挑选起点。审核没有调用 `env.step`，新增控制transition为0；不增加既有31090步累计诊断计数。

## 可复核产物与执行守卫

- [几何证书](lite3_spawn_geometry_certificate_v1.json)：保存nominal各碰撞体变换、可重算证明、9案例检查及模型/reset源码哈希。
- 原始结果：`artifacts/g0_spawn_geometry_20260929/physical_validated/audit.json`；包含每个案例的完整形状变换与边界。
- 可复跑入口：`scripts/reinforcement_learning/online_adaptation/audit_spawn_geometry.py --output <新目录> --headless --device cuda:0 --kit_args '--/renderer/activeGpu=0 --/renderer/multiGpu/enabled=false'`。沿用项目的env_isaaclab与PYTHONPATH设置。
- 新的 managed perturbed-v1 session 在准备worker配置时核验证书文件绑定、重算几何证明，并把证书SHA写入不可变manifest。模型或reset/地形源码变化会拒绝旧证书；须重跑审核并更新证书后才可继续。
- 当前只认证 RockyRoad 及其平面测量路径；其他任务的 managed perturbed-v1 不能借用此证书。旧source-bound session也不能被新源码静默续接。

两次未完成的诊断保留在 `physical/` 和 `physical_retry/`：中心已通过，角点因诊断脚本使用numpy.float64 yaw导致Isaac reset的Double/Float类型冲突而退出，各0控制步。已改用Python float，并持久化异常、确保失败退出码非零。此问题仅在强制角点脚本；运行器原有扰动参数使用Python float。

## 验证与后续

单元测试覆盖旋转/缩放后的形状支撑界、全部yaw的保守包围、平台越界、穿地、空/未知/非有限几何拒绝、证书重算、模型篡改失效及managed配置绑定。完整回归 **1575 passed, 7 skipped**；`git diff --check` 通过。

[开发协议](protocol_v2_2.json)已移除出生几何审核缺口；最终封存协议仍未锁定。下一阶段是G1目标域4000步warmup和两路线学习信号诊断，以及预登记候选晋升/平地保持规则。本轮未启动学习训练。
