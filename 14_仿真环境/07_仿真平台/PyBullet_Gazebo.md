# PyBullet / Gazebo

PyBullet和Gazebo是两个广泛使用的开源机器人仿真平台。

## 目录

- [1. PyBullet](#1-pybullet)
- [2. Gazebo](#2-gazebo)
- [3. 集成与对比](#3-集成与对比)

---

## 1. PyBullet

### 1.1 简介

- Python原生
- 实时仿真
- 易于使用
- VR支持

### 1.2 安装

```bash
pip install pybullet
```

### 1.3 基本使用

```python
import pybullet as p
import pybullet_data
import numpy as np

# 连接仿真
client = p.connect(p.DIRECT)  # 无GUI
client = p.connect(p.GUI)     # GUI

# 添加搜索路径
p.setAdditionalSearchPath(pybullet_data.getDataPath())

# 加载地面
plane_id = p.loadURDF("plane.urdf")

# 加载机器人
robot_id = p.loadURDF("kuka_iiwa/model.urdf", [0, 0, 0])

# 设置关节控制
num_joints = p.getNumJoints(robot_id)
for j in range(num_joints):
    p.setJointMotorControl2(robot_id, j, p.POSITION_CONTROL, targetPosition=0)

# 仿真循环
for _ in range(1000):
    p.stepSimulation()
```

### 1.4 强化学习接口

```python
import gym
import pybullet_envs

# 创建环境
env = gym.make('HalfCheetahBulletEnv-v0')

# 训练循环
obs = env.reset()
for step in range(1000):
    action = env.action_space.sample()
    obs, reward, done, info = env.step(action)
    
    if done:
        obs = env.reset()
```

---

## 2. Gazebo

### 2.1 简介

- ROS原生集成
- 室内外仿真
- 高保真物理
- 广泛应用

### 2.2 ROS2集成

```python
# ROS2 + Gazebo 启动
# 启动文件
launch_gazebo = IncludeLaunchDescription(
    PythonLaunchDescriptionSource([
        PathJoinSubstitution([
            FindPackageShare('gazebo_ros'),
            'launch',
            'gzserver.launch.py'
        ])
    ]),
    launch_arguments={'world': world_path}.items()
)
```

### 2.3 机器人模型(SDF)

```xml
<sdf version="1.6">
  <model name="robot">
    <static>false</static>
    <link name="base_link">
      <pose>0 0 0.1 0 0 0</pose>
      <collision>
        <geometry>
          <cylinder>
            <radius>0.1</radius>
            <length>0.1</length>
          </cylinder>
        </geometry>
      </collision>
      <visual>
        <geometry>
          <cylinder>
            <radius>0.1</radius>
            <length>0.1</length>
          </cylinder>
        </geometry>
      </visual>
    </link>
  </model>
</sdf>
```

---

## 3. 集成与对比

### 3.1 对比

| 特性 | PyBullet | Gazebo |
|------|----------|--------|
| 物理引擎 | Bullet | ODE/Simbody/DART |
| ROS集成 | 基础 | 深度 |
| 难度 | 简单 | 中等 |
| 渲染 | 基本 | 高质量 |
| 适用 | 快速原型 | 研究 |

### 3.2 选择建议

- **快速实验**: PyBullet
- **ROS项目**: Gazebo
- **真实感要求高**: Isaac Sim
- **连续控制**: MuJoCo

---

*本章节持续更新中...*
