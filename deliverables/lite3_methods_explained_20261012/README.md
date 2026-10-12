# Lite3 仓库方法原理详解

依据当前源码和截至2026-10-12的已保存实验记录，解释控制与观测、PPO与GAE、对称与动作保持、SAC、LoRA、低维/CEM搜索、Flow Matching、FlowDAgger、恢复教师、监督迁移、路线奖励和评测方法。

- `Lite3_仓库方法原理详解_20261012.pdf`：26页，20章，42个公式，含可点击目录、PDF书签、源码链接、方法比较和结果范围说明。
- `Lite3_仓库方法原理详解.md`：可编辑正文；`@diagram`为构建脚本的矢量图插入标记。
- `build_pdf.py`：ReportLab排版；中文Droid字体与DejaVu英文/符号混排，Matplotlib渲染公式。
- `assets/`：公式图片及LaTeX列表。
- `validation/validation.json`：章节、字体覆盖、页面边界、链接和关键结果验证。
- `source_manifest.json`：依据文件与输出文件的SHA256。

技术基线：`62a58992712cf6132da38eb4f606c14c25d063b5`，分支`feat/lite3-policy-learning`。文中的源码链接固定到该提交，需要访问项目私有仓库。

在本工作树根目录重建：

```bash
python3 deliverables/lite3_methods_explained_20261012/build_pdf.py
```

系统Python需有ReportLab、Pillow，公式渲染使用现有`env_isaaclab`环境中的Matplotlib，脚本内有明确路径。此次工作只读取源码/记录并生成说明文档，没有启动新仿真、训练或修改已有模型与奖励。
