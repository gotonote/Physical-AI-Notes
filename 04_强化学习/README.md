# 04 强化学习

强化学习（Reinforcement Learning, RL）是物理AI的核心技术，使智能体通过与环境交互学习最优策略。本章全面介绍强化学习的基础理论、算法实现及在机器人领域的应用。

## 目录

- [1. 强化学习基础](./RL基础.md)
  - 马尔可夫决策过程
  - 值函数与策略梯度
  - DQN及其变体
- [2. 离线强化学习](./离线强化学习.md)
  - 分布偏移问题
  - CQL算法
  - 实践框架
- [3. Sim-to-Real迁移](./Sim-to-Real迁移.md)
  - 领域随机化
  - 域适应技术
  - 课程学习
- [4. 分布式强化学习](./分布式强化学习.md)
  - 分布式架构
  - IMPALA/Ape-X
  - 参数同步

---

## 核心概念

### 强化学习基本框架

```
┌─────────────────────────────────────────┐
│           强化学习交互框架              │
├─────────────────────────────────────────┤
│                                         │
│    ┌───────┐         ┌───────┐         │
│    │ 智能体 │────────▶│  环境  │         │
│    │ Agent │◀────────│Env    │         │
│    └───────┘         └───────┘         │
│       │                  │              │
│       │ Action (a)       │              │
│       │─────────────────▶│              │
│       │                  │              │
│       │◀─────────────────│              │
│       │   Reward (r)     │              │
│       │   State (s')     │              │
│                                         │
└─────────────────────────────────────────┘
```

### 关键算法演进

| 年份 | 算法 | 贡献 |
|------|------|------|
| 2013 | DQN | 深度强化学习突破 |
| 2016 | A3C | 异步分布式训练 |
| 2017 | PPO | 稳定策略优化 |
| 2018 | IMPALA | 规模化分布式RL |
| 2019 | AlphaStar | 超人类水平 |
| 2020 | CQL | 离线强化学习 |

---

## 学习路径

### 入门阶段
1. 理解MDP基本概念
2. 掌握Q-learning和SARSA
3. 学习深度Q网络(DQN)

### 进阶阶段
4. 策略梯度方法(A2C/PPO)
5. 离线强化学习方法
6. Sim-to-Real迁移技术

### 高级阶段
7. 分布式强化学习
8. 元学习和多任务学习
9. 大规模机器人应用

---

## 实践框架

### 仿真环境

| 环境 | 特点 | 适用场景 |
|------|------|----------|
| MuJoCo | 高保真物理 | 连续控制 |
| PyBullet | 开源易用 | 快速原型 |
| Isaac Sim | NVIDIA GPU加速 | 大规模训练 |
| Gazebo | ROS集成 | 室外机器人 |

### 训练库

- **Stable-Baselines3**: 简洁易用的RL库
- **Ray RLlib**: 大规模分布式训练
- **OpenAI Baselines**: 经典算法实现

---

## 重要资源

### 经典论文
1. Mnih et al. (2015). Human-level control through deep RL
2. Schulman et al. (2017). Proximal Policy Optimization
3. Haarnoja et al. (2018). Soft Actor-Critic
4. Kumar et al. (2020). Conservative Q-Learning

### 开源项目
- [Stable-Baselines3](https://github.com/DLR-RM/stable-baselines3)
- [Ray RLlib](https://docs.ray.io/en/latest/rllib/index.html)
- [DeepMind Control Suite](https://github.com/deepmind/dm_control)

---

*本章节持续更新中...*
