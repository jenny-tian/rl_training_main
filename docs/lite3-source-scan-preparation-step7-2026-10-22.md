# Lite3 Source Terrain：第 7 步 v0.2 扫描准备合同

日期：2026-10-22（Asia/Shanghai）。本步形成 Source-only 冻结策略扫描的完整准备稿。它没有授权或启动仿真扫描。

## 交付物

- `docs/lite3-source-scan-preparation-v0.2-2026-10-22/source_qualification_scan_manifest_v0.2_preparation.json`：45 个显式扫描单元、每格 5 回合，共 225 回合；保留 `draft_not_for_run` 与 `execution_authorized: false`。
- 同目录下的 reset、safety、termination、metrics、classification、observation、action-scale、normalizer、dynamics 和 material 合同。扫描 manifest 绑定可用文件的 SHA-256，并将缺失的正式 runner 和执行 approval 标记为 unavailable。
- `source/rl_training/rl_training/source_scan/manifest.py` 与 `scripts/reinforcement_learning/online_adaptation/audit_source_scan_manifest.py`：新增只读 `--preparation-audit`，验证 v0.2 准备稿的结构但不读取其引用文件、不修改授权状态，也不把 proposal 当作已锁定阈值。
- `tests/source_scan/test_manifest.py`：覆盖准备稿结构、不可执行状态、拒绝伪装锁定、CLI 输出和审计不读取引用 artifact。

45 格覆盖 Flat 3 格、Multi-scale Rough 3 格、Slope 12 格、多级直线 Stairs 6 格、Slope + Rough 12 格、Friction Variation 9 格。候选高度、坡度、摩擦区间、10 m × 4 m 轨道、1.5 m 起点、9.5 m 完成终点及 50 步稳定保持均显式写入。楼梯踏面候选与当前几何实现一致（L1/L2 为 0.5 m，L3 为 0.4 m），仍需按 Lite3 步态校准。

## 资格边界

当前准备审计结果为 `schema_valid: true`、45 cells、225 episodes，但 `lock_fields_complete: false`、`execution_ready: false`。运行阻断项包括：尚无 manifest 驱动的完整 episode runner、端到端 policy/SafetyFilter 桥接未资格化、classification 数值仍待复核、尚无单独执行 approval。现有 checkpoint parity 和几何/摩擦材质 smoke 只绑定为已有证据；它们没有证明完整策略回合安全性或地形通过能力。

规则 Flat、坡面和 Stairs 在固定参数时不因 seed 改变几何。manifest 明确把这些 seed 记录为重复试验标识，不称作独立几何样本；reset 也是固定名义状态，因此 initial-state seed 是配对和隔离标识，而非随机化初态。五回合结果只作诊断，不声明实例总体的统计置信结论。分类值仍是 proposal。

本步未启动 Isaac Sim、未加载 `model_1998.pt` 进行 episode 推理、未读取 Sealed 数据，也未进行 PPO 更新或训练。外部原始 v0.1 扫描草案未被覆盖。

## 验证

```bash
PYTHONPATH=source/rl_training \
  /home/tjx/anaconda3/envs/env_isaaclab/bin/python \
  scripts/reinforcement_learning/online_adaptation/audit_source_scan_manifest.py \
  docs/lite3-source-scan-preparation-v0.2-2026-10-22/source_qualification_scan_manifest_v0.2_preparation.json \
  --preparation-audit
# schema_valid=true, cell_count=45, episode_count=225, execution_ready=false

PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=source/rl_training \
  /home/tjx/anaconda3/envs/env_isaaclab/bin/python -m pytest -q \
  tests/source_scan tests/online_adaptation/test_source_terrain_tracks.py
# 174 passed

/home/tjx/anaconda3/envs/env_isaaclab/bin/python -m compileall -q \
  source/rl_training/rl_training/source_scan/manifest.py \
  scripts/reinforcement_learning/online_adaptation/audit_source_scan_manifest.py \
  tests/source_scan/test_manifest.py

git diff --check
```

结构审计、测试、编译与 diff 检查通过。下一阶段若继续，应先实现和审核 manifest 驱动 runner 与端到端 safety bridge，然后重新做运行资格检查；完整 225 回合冻结策略扫描仍需用户另行批准。
