# 发布验证：2026-11-24

- 三份权重与原始训练检查点的 SHA256 完全一致；14 份配置/执行记录逐字节复制核对通过，清单另附自身哈希，共 18 项校验。
- PyTorch `weights_only=True` 在 CPU 成功加载全部检查点；actor/critic 参数有限、优化器状态非空，迭代计数分别对应 10000、10100、10200 次更新。
- 每份 actor 使用 256 个固定随机输入完成 CPU 推理，输出均为有限的 256×12 张量。float32 批量/逐样本最大绝对差分别为 1.62125e-5、2.28882e-5、2.57492e-5，满足 atol=1e-4 / rtol=1e-5。
- 既有六类地形契约、低台阶和台阶分配测试：**34 passed**。
- 三个新增 Python 工具通过语法编译；训练入口 `--help` 正常，明确预算和 GPU 0–3 参数可用。
- 本次未重新运行 Isaac Lab 仿真或训练。检查时 GPU 0–3 空闲显存分别为 12102、11771、11702、10584 MiB，均低于现有启动器要求的 12288 MiB；保留原启动门槛。已有训练和冻结评测结果见 README 中的原始记录。

执行命令（使用已安装 PyTorch 和 Isaac Lab 的 Python 环境）：

```bash
python scripts/tools/verify_lite3_six_terrain.py
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q \
  tests/source_curriculum/test_reference_six_contract.py \
  tests/source_curriculum/test_reference_low_stairs.py \
  tests/source_curriculum/test_reference_stair_allocation.py
python -m py_compile scripts/tools/train_lite3_six_terrain.py \
  scripts/tools/package_lite3_six_terrain.py scripts/tools/verify_lite3_six_terrain.py
```
