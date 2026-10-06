# Lite3 成功演示录像（2026-10-06）

原2026-10-05的64/64评测未录制相机视频。本文件对应同一冻结PPO教师的一次新仿真运行，不能冒充原评测录像，也不是独立训练复现。

- 本地MP4：`artifacts/recovery_teacher_20261006/video-delivery/course-demo/demo.mp4`
- 预览：`artifacts/recovery_teacher_20261006/video-delivery/course-demo/demo.poster.png`
- 960×540，25fps，659帧，26.36秒；按仿真时间播放，非墙钟推理性能测试。
- 模型SHA256：`67006951ebb2ed337aeac7ede2a21709ba6bcc985e28220429a9368d2c17cdb9`。
- 视频SHA256：`7402a723f1924d27b83aee2e59cfe792f38739e67744d6af8f05d0191cc67740`。
- 启动前选定seed971、env16、rough101首回合；最终成功，1318控制步，回合横偏RMS2.62cm、最大9.62cm。它是单回合指标，不替代全部64回合均值。
- 本次全64槽均完成，录像只显示其中预选的一个。其他机器人只在渲染中隐藏，物理依旧执行并计账。
- 画面叠加新运行标识、地形、时间、横偏与倾角，元数据与对应物理步轨迹已逐帧校验。视频为动作前画面，未包含自动重置后的瞬移帧。

## 工程尝试与费用

1. 初次继承DISPLAY=:3，GLXBadFBConfig退出2，尚未进行物理步。原日志及sidecar保留。
2. unset DISPLAY后完成86080步、64/64，但克隆继承env0隐藏属性导致画面看不到机器人，该MP4保留为故障证据，不交付为成功录像。
3. 显式恢复选中环境可见性，2×64=128步预览检查通过。
4. 最终交付录制86080步、64/64，起点、中段、接近终点画面均已查看。

本次共172288新增物理控制步，无不确定步；加此前实验累计charged12508224、confirmed12508160、uncertain64，早期PPO基础成本仍另列。视频/模型等大产物保存在本地artifacts；代码、协议和审计记录提交至私有仓库。

- [视频帧与遥测审计](lite3_video_delivery_media_audit_20261006.json)
- [物理轨迹与CPU动作回放审计](lite3_video_delivery_trajectory_audit_20261006.json)
- [后续实验计划](lite3-next-experiments-plan-2026-10-06.md)
