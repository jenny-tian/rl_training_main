# L1 粗糙左横坡单回合准备

实际准备日期 2026-10-30；用户已批准 `slope_plus_rough_L1_cross_slope_left` 唯一回合。固定 bank row0/column10，种子 `10001/20260`，difficulty `0.9256251464646599`。地形为左高右低 5° 横坡叠加局部峰谷差参数 1 cm 的粗糙面，审计窗口 0.4 m、空间尺度 0.2/0.4 m，入口／过渡／出口配置与登记清单一致。

143 项 CPU 检查通过，237 项输入冻结，8 项几何源码哈希及原 bank 网格指纹核对一致。生产代码仅切换清单路径，策略、命令、安全阈值、物理和 link/COM 诊断未变。GPU0 可共享，可用显存至少 12288 MiB；单环境、单回合、2000 步、120/240 秒，9 m 加连续稳定 50 步验收，失败即停，不重试。

见 [preparation.json](preparation.json)、[run_spec.json](run_spec.json)、[post_review.json](post_review.json)。
