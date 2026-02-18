# MuJoCo (Multi-Joint Dynamics with Contact)

MuJoCo是用于机器人仿真的物理引擎，特别适合连续控制任务。

## 目录

- [1. MuJoCo概述](#1-mujoco概述)
- [2. 模型定义](#2-模型定义)
- [3. Python接口](#3-python接口)
- [4. 机器人仿真](#4-机器人仿真)
- [5. 训练示例](#5-训练示例)

---

## 1. MuJoCo概述

### 1.1 特点

- 精确的物理模拟
- 接触动力学
- 开源免费
- 广泛使用(DLC, DeepMind Control Suite)

### 1.2 安装

```bash
pip install mujoco
pip install mujoco-py  # Python 2接口 (已弃用)
pip install mujoco     # 新版本Python接口
```

---

## 2. 模型定义

### 2.1 XML模型格式

```xml
<mujoco model="robot_arm">
  <!-- 编译器设置 -->
  <compiler angle="radian" meshdir="meshes"/>
  
  <!-- 全局选项 -->
  <option timestep="0.002" iterations="50" solver="Newton"/>
  
  <!-- 世界 -->
  <worldbody>
    <!-- 地面 -->
    <geom type="plane" size="10 10 0.1" rgba="0.5 0.5 0.5 1"/>
    
    <!-- 光源 -->
    <light diffuse=".5 .5 .5" pos="0 0 3" dir="0 0 -1"/>
    
    <!-- 机器人基座 -->
    <body name="base_link" pos="0 0 0">
      <joint type="free"/>
      <geom type="mesh" mesh="base"/>
    </body>
  </worldbody>
  
  <!-- 致动器 -->
  <actuator>
    <motor joint="joint1" ctrllimited="true" ctrlrange="-1 1"/>
  </actuator>
</mujoco>
```

---

## 3. Python接口

### 3.1 基本使用

```python
import mujoco
import numpy as np

# 加载模型
model = mujoco.MjModel.from_xml_path("robot.xml")
data = mujoco.MjData(model)

# 仿真循环
for _ in range(1000):
    # 执行一步仿真
    mujoco.mj_step(model, data)
    
    # 获取状态
    qpos = data.qpos  # 关节位置
    qvel = data.qvel  # 关节速度
    
    # 设置控制
    data.ctrl[:] = [0.0] * model.nu
    
print(f"Final joint positions: {data.qpos}")
```

### 3.2 渲染

```python
import mujoco.viewer

# 创建渲染器
viewer = mujoco.viewer.launch_passive(model, data)

# 渲染循环
while viewer.is_running():
    mujoco.mj_step(model, data)
    viewer.sync()

viewer.close()
```

---

## 4. 机器人仿真

### 4.1 加载预定义模型

```python
# 使用MuJoCo内置模型
from mujoco import viewer

# 人类型机器人
model = mujoco.MjModel.from_xml_string("""
<mujoco model="humanoid">
  <compiler angle="degree" meshdir="."/>
  <option timestep="0.005"/>
  <worldbody>
    <light diffuse=".5 .5 .5" pos="0 0 3" dir="0 0 -1"/>
    <geom type="plane" size="1 1 0.1" rgba=".9 .9 .9 1"/>
    <body name="torso" pos="0 0 1.4">
      <freejoint/>
      <geom type="capsule" size="0.07" fromto="0 0 -.3 0 0 .2" rgba="0.7 0.7 0.7 1"/>
      <joint type="free"/>
    </body>
  </worldbody>
  <actuator>
    <motor joint="freejoint" gear="200"/>
  </actuator>
</mujoco>
""")

data = mujoco.MjData(model)
viewer = viewer.launch_passive(model, data)
```

---

## 5. 训练示例

### 5.1 简单策略梯度

```python
import mujoco
import numpy as np
import torch
import torch.nn as nn

class SimplePolicy(nn.Module):
    def __init__(self, obs_dim, action_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim),
            nn.Tanh()
        )
        
    def forward(self, x):
        return self.net(x)

# 初始化
model = mujoco.MjModel.from_xml_path(" swimmer.xml")
data = mujoco.MjData(model)
policy = SimplePolicy(model.nq + model.nv, model.nu)

optimizer = torch.optim.Adam(policy.parameters(), lr=0.001)

# 训练
for episode in range(100):
    # 重置
    mujoco.mj_resetDataKeyframe(model, data, 0)
    
    episode_data = []
    
    for step in range(200):
        # 获取观测
        obs = np.concatenate([data.qpos, data.qvel])
        
        # 策略
        with torch.no_grad():
            action = policy(torch.FloatTensor(obs)).numpy()
            
        # 执行
        data.ctrl[:] = action
        mujoco.mj_step(model, data)
        
        # 奖励
        reward = -np.sum(action**2) * 0.01  # 简单惩罚
        
        episode_data.append((obs, action, reward))
        
    # 计算回报
    G = 0
    for obs, action, reward in reversed(episode_data):
        G = reward + 0.99 * G
        
    print(f"Episode {episode}, Return: {G:.2f}")
```

---

## 参考文献

1. Todorov, E., et al. (2012). MuJoCo: A physics engine for model-based control.
2. DeepMind Control Suite Documentation

---

*本章节持续更新中...*
