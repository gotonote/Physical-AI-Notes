# 07 仿真平台

机器人仿真是训练和验证机器人算法的重要工具。本章介绍主流仿真平台的特点和使用方法。

## 目录

- [1. NVIDIA Isaac Sim](./NVIDIA_Isaac_Sim.md)
  - GPU加速
  - RTX渲染
- [2. MuJoCo](./MuJoCo.md)
  - 高精度物理
  - 控制基准
- [3. PyBullet / Gazebo](./PyBullet_Gazebo.md)
  - 开源易用
  - ROS集成
- [4. SAPIEN](./SAPIEN.md)
  - 高保真交互

---

## 仿真平台对比

| 平台 | 物理精度 | 渲染 | 难度 | 适用 |
|------|----------|------|------|------|
| Isaac Sim | 高 | RTX | 中 | 大规模训练 |
| MuJoCo | 高 | 中 | 低 | 控制研究 |
| PyBullet | 中 | 低 | 低 | 快速原型 |
| Gazebo | 中 | 中 | 中 | ROS项目 |
| SAPIEN | 高 | RTX | 中 | 灵巧操作 |

---

## 选择指南

1. **连续控制研究**: MuJoCo
2. **大规模训练**: Isaac Sim
3. **快速原型**: PyBullet
4. **ROS集成**: Gazebo
5. **物体操作**: SAPIEN

---

*本章节持续更新中...*
