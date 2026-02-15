# NVIDIA Isaac Sim

NVIDIA Isaac Sim是基于NVIDIA Omniverse的机器人仿真平台，提供高保真物理模拟和渲染。

## 目录

- [1. Isaac Sim概述](#1-isaac-sim概述)
- [2. 核心功能](#2-核心功能)
- [3. 环境配置](#3-环境配置)
- [4. 机器人仿真](#4-机器人仿真)
- [5. 训练与部署](#5-训练与部署)

---

## 1. Isaac Sim概述

### 1.1 特点

- GPU加速物理模拟
- 高保真传感器仿真
- RTX实时光线追踪
- 与ROS/ROS2无缝集成

### 1.2 系统要求

| 组件 | 最低要求 |
|------|----------|
| GPU | RTX 3070+ |
| 显存 | 8GB+ |
| CUDA | 11.8+ |
| Python | 3.8+ |

---

## 2. 核心功能

### 2.1 USD格式支持

```python
import omni.usd
from pxr import Usd, UsdGeom

# 创建USD场景
stage = omni.usd.get_context().get_stage()

# 添加地面
ground = UsdGeom.Xform.Define(stage, "/World/ground")
UsdGeom.Cylinder.Define(stage, "/World/ground/Plane")

# 添加光源
distantLight = UsdGeom.DistantLight.Define(stage, "/World/Light")
distantLight.AddTranslateOp().Set(omni.usd.get_stage_next_free_path(stage, "/World/Light", False))
```

### 2.2 物理模拟

```python
# 物理场景设置
from omni.physx import _physx

physx_interface = _physx.get_physx_interface()
scene = physx_interface.create_physics_scene()

# 添加刚体
rigid_body_api = UsdGeom.RigidBodyAPI.Apply(prim)
```

---

## 3. 环境配置

### 3.1 安装

```bash
# 通过NVIDIA Omniverse安装
# 下载并安装 Isaac Sim
# https://developer.nvidia.com/isaac-sim
```

### 3.2 Python环境

```python
import omni
import omni.isaac.core
import omni.isaac.robot_benchmark

# 初始化
omni.usd.get_context().new_stage()

# 加载机器人
from omni.isaac.manipulators import SingleManipulator

robot = SingleManipulator(prim_path="/World/Franka")
robot.initialize()
```

---

## 4. 机器人仿真

### 4.1 加载机器人

```python
from omni.isaac.core.robots import Robot
from omni.isaac.core.utils.nucleus import get_assets_root_path

# 获取资产路径
assets_root = get_assets_root_path()

# 加载FRANKA
franka_path = f"{assets_root}/Robots/FrankaFr3/FrankaFr3.usd"

# 创建机器人实例
robot = Robot(prim_path="/World/Franka", usd_path=franka_path)
robot.initialize()

# 设置控制器
from omni.isaac.franka import FrankaFR3
franka = FrankaFR3(prim_path="/World/Franka")
```

### 4.2 控制接口

```python
# 关节位置控制
robot.set_joint_position_targets(positions=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

# 关节速度控制
robot.set_joint_velocity_targets(velocities=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

# 末端执行器控制
robot.set_end_effector_target(position=[0.5, 0.0, 0.3], orientation=[0, 0, 0, 1])
```

---

## 5. 训练与部署

### 5.1 RL训练

```python
# Isaac Gym RL训练接口
from omni.isaac.gym.vec_env import VecEnvBase

# 创建环境
env = VecEnvBase(headless=False)

# 创建任务
from tasks.franka_reach import FrankaReach
task = FrankaReach(name="FrankaReach", sim_params=sim_params, physics_engine="physx")
env.set_task(task, backend="torch")

# 训练循环
for step in range(num_steps):
    actions = policy(observations)
    observations, rewards, dones, info = env.step(actions)
```

---

## 参考文献

1. NVIDIA Isaac Sim Documentation
2. Isaac Gym Developer Guide

---

*本章节持续更新中...*
