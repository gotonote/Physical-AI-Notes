# SAPIEN

SAPIEN是一个高保真的交互式机器人仿真平台，支持真实的物理交互和视觉渲染。

## 目录

- [1. SAPIEN概述](#1-sapien概述)
- [2. 安装与配置](#2-安装与配置)
- [3. 核心功能](#3-核心功能)
- [4. 应用场景](#4-应用场景)

---

## 1. SAPIEN概述

### 1.1 特点

- GPU物理模拟
- RTX渲染
- 大规模场景
- 交互式物体

### 1.2 与其他仿真器对比

| 仿真器 | 物理 | 渲染 | 适用场景 |
|--------|------|------|----------|
| SAPIEN | 高保真 | RTX | 灵巧操作 |
| Isaac Sim | 高保真 | RTX | 规模化 |
| PyBullet | 实时 | 基本 | 快速原型 |
| Gazebo | 中等 | 中等 | ROS集成 |

---

## 2. 安装与配置

### 2.1 安装

```bash
# 通过pip安装
pip install sapien

# 从源码安装
git clone https://github.com/haosulab/SAPIEN.git
cd SAPIEN
pip install -e .
```

---

## 3. 核心功能

### 3.1 基本使用

```python
import sapien
import numpy as np

# 创建引擎
engine = sapien.Engine()

# 创建渲染器
renderer = sapien.VulkanRenderer()

# 创建场景
scene = engine.create_scene(renderer=renderer)

# 添加地面
ground = scene.create_actor_builder().build_kinematic()

# 加载机器人
loader = scene.create_robot_loader()
robot = loader.load("franka_panda.urdf")
robot.set_root_pose(sapien.Pose([0, 0, 0], [1, 0, 0, 0]))

# 仿真循环
for _ in range(1000):
    scene.step()
    renderer.render()
```

---

## 4. 应用场景

### 4.1 物体操作

```python
# 抓取任务
def grasp_object(robot, object_actor):
    # 移动到物体上方
    robot.set_target_pose(object_pose + offset)
    
    # 下降
    robot.set_target_pose(object_pose)
    
    # 闭合夹爪
    robot.set_qpos(gripper_closed)
```

---

## 参考文献

1. SAPIEN: A Physically Realistic Robot Interaction Simulator

---

*本章节持续更新中...*
