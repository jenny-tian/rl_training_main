# Lite3 Source scan manifest：第 5 步校验实现记录

日期：2026-10-22（Asia/Shanghai）。本步按已批准的顺序，只实现 source qualification scan 清单的结构审计、离线运行前检查及其测试。没有改写外部原始草案。

## 交付物

- `source/rl_training/rl_training/source_scan/manifest.py`：标准库实现的可执行 schema 和文件完整性预检。
- `source/rl_training/rl_training/source_scan/__init__.py`：轻量包入口，不导入训练器。
- `scripts/reinforcement_learning/online_adaptation/audit_source_scan_manifest.py`：只读 CLI；默认执行结构审计，`--preflight` 要求完整锁定合同并校验引用文件 SHA-256。
- `tests/source_scan/fixtures/source_scan_v0_1_draft.json`：原始 v0.1 清单的测试快照。
- `tests/source_scan/test_manifest.py`：草案隔离、schema、seed 分区、参数、指标、批准绑定、文件完整性、CLI 和轻量导入测试。

该 schema 由 Python 校验器实现；本步没有添加或声称存在独立 JSON Schema 文件。测试中的 v0.2 清单、批准记录、checkpoint 和资格报告均为合成数据，只验证代码路径，不是 Lite3 运行或审批证据。

## 保护的实验边界

- v0.1 仅可读取和审计，不能通过修改状态字段变成可执行清单；校验器不会自动迁移它。
- v0.2 合同要求仅处理 `source_qualification_scan`，覆盖六类 Source 的三个等级与已登记方向；几何 seed 和初态 seed 分开，扫描、训练、验证、开发、probe 与 sealed 分区互斥。
- 固定部署匹配动力学不能使用 `dynamics_seed` 冒充独立动力学扰动。逐条件几何、reset 和轨道终点必须锁定为实际解析参数，候选值、未解占位符、无法到达的终点及未覆盖的楼梯踏步会被拒绝。
- 每回合指标与家族诊断项按方案逐项校验。分类阈值必须在执行清单中以数值形式完整登记；校验器只检查登记结构，不代替研究者选择任何阈值。
- checkpoint 仅按字节计算 SHA-256，不反序列化。资格报告必须声明通过并绑定登记输入的哈希；桥接 smoke 报告还必须确认区域材质绑定。
- 清单内容哈希和批准记录用于完整性绑定，**不是加密签名或真实用户授权证明**。本工具不是授权机制，也不启动仿真；未来运行端仍需独立验证真实运行时绑定、批准与安全准入。

## 对原始草案的只读审计

审计对象：`artifacts/source_qualification_scan_20261021_draft/scan_manifest.json`。测试快照与该文件逐字节相同，SHA-256 为 `85318be23eda7d73fd14a5a60e6442e1096f2a396264c8754b165b44f5291c4d`。

| 检查 | 结果 |
|---|---|
| 普通审计退出码 | 0；JSON 结构可解析 |
| `schema_valid` / `lock_fields_complete` | `true` / `false`；结构可审计不表示清单可执行 |
| 条件数 / 回合数 | 45 / 225 |
| 运行阻断数 | 54，全部为 blocker |
| `--preflight` 退出码 | 2；拒绝 v0.1 清单 |
| 原草案 | 未修改 |

阻断项为：45 个 cell 尚无已解析几何/reset/轨道合同；1 项动力学语义不明确；1 项把动力学 seed 当作独立扰动；source validation 和 probe 缺少各自预留分区；状态、授权布尔值和 split 三项未锁定；缺少执行 schema 与完整执行合同。文件预检先检查这些结构项并立即拒绝，没有打开 checkpoint 或其他 artifact 文件。

## 验证与范围

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=source/rl_training \
  /home/tjx/anaconda3/envs/env_isaaclab/bin/python -m pytest -q \
  tests/source_scan/test_manifest.py \
  tests/online_adaptation/test_source_terrain_tracks.py \
  tests/online_adaptation/test_continuous_rough_track.py
```

结果：**175 passed**。第三方 pytest 插件自动加载关闭，以避开环境已有的 `typeguard` / `typing_extensions` 版本冲突；未调整环境依赖。另执行了 Python 编译检查和 `git diff --check`。

本步没有启动 Isaac Lab/AppLauncher，没有加载或推理 PPO，没有运行 episode、训练、基线能力扫描或读取 Sealed 结果；没有创建真实 v0.2 manifest，也没有更改原草案的 `draft_not_for_run` / `execution_authorized: false` 状态。文件哈希预检只在测试用临时合成文件上走通。

## 下一步建议

下一步可做离线 checkpoint parity：核对冻结 `model_1998.pt` 的 observation/action 维度、normalizer 与 action scaling，并输出可复核报告；仍不启动仿真或完整扫描。checkpoint parity 完成后，再准备单环境桥接与指标 mock，之后另行申请 smoke test。
