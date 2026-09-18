# Lite3 研究进展报告

参照 `/home/tjx/Documents/组会/main.pdf` 的四章研究报告结构与 A4 排版编写，资料核查截止 2026-09-18。

- `Lite3_研究进展报告_20260918.pdf`：16 页成稿，包含 3 图、7 张正文表、13 个公式、参考文献与证据附录。
- `Lite3_研究进展报告_20260918.docx`：可编辑 Word；正文、表格和公式为原生文档对象，插图另有 SVG 图源。
- `report.md`：文字与公式源文件。
- `figures/`：两张系统/评测示意图、一张基于 JSON 的统计图及生成脚本。
- `evidence/`：24 份小型实验 JSON 快照、原始位置与 SHA-256 索引。
- `validation.json`：页数、DOCX 数学对象、证据哈希和页面边界核验。

本文以平地 PPO 为技能起点，介绍 LoRA-PPO 与 Residual SAC 的当前实现。现有 45 维单帧与 49 维历史接口不同；结果以系统开发对照报告。训练成功、冻结复测、续训来源和正式独立学习分开解释。未重新运行训练、仿真或源码测试。

重新生成：

```bash
python3 build_report.py
```

依赖 Pandoc、LibreOffice 和 Python lxml。当前主机 Pandoc 位于 `/home/tjx/anaconda3/bin/pandoc`。脚本会从 Markdown 生成 DOCX，设置中文字体、A4 页面、表宽、页码及数学对象，再用 LibreOffice 导出 PDF。受限沙箱中 LibreOffice 可能需要在授权环境运行；也可先用 `--no-pdf` 仅生成 Word。

图 1/2 使用 `python3 figures/generate_figures.py` 重建，图 3 使用具有 Matplotlib 的 Python 运行 `figures/plot_results.py` 重建。图 3 数据读取本交付包中的证据快照，不访问在线实验文件。

报告制作期间工作目录存在其他任务的源码和暂存修改，本次交付仅提交本目录文件。论文事实基于核查时源码和已保存实验；不同版本规则及当前源码变化不能反向改写旧实验。
