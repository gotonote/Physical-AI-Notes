# 逆强化学习 (Inverse Reinforcement Learning)

逆强化学习（IRL）从专家演示中推断奖励函数，是理解专家行为和泛化技能的关键技术。

## 目录

- [1. IRL概述](#1-irl概述)
- [2. 最大熵IRL](#2-最大熵irl)
- [3. 生成对抗IRL](#3-生成对抗irl)
- [4. 神经IRL](#4-神经irl)
- [5. 应用案例](#5-应用案例)

---

## 1. IRL概述

### 1.1 问题定义

```
传统RL: 已知 r(s,a) → 找到最优 π
逆IRL: 已知 π* 的演示 → 推断 r(s,a)

核心假设: 专家行为是最优或接近最优的
```

### 1.2 方法分类

| 方法 | 特点 | 优缺点 |
|------|------|--------|
| 最大熵IRL | 概率框架，处理多模态 | 计算复杂 |
| GAIRL | 对抗学习 | 训练不稳定 |
| 神经IRL | 神经网络表示 | 端到端 |
| 结构IRL | 假设奖励结构 | 可解释 |

---

## 2. 最大熵IRL

### 2.1 原理

最大熵IRL的核心思想：

$$
P(\tau | \theta) = \frac{1}{Z(\theta)} \exp(\theta \cdot f(\tau))
$$

其中 $f(\tau)$ 是轨迹特征，$Z(\theta)$ 是配分函数。

```python
import torch
import torch.nn as nn
import numpy as np

class MaxEntIRL:
    """
    最大熵逆强化学习
    """
    def __init__(self, state_dim, action_dim, feature_dim, lr=0.01):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.feature_dim = feature_dim
        
        # 奖励函数参数
        self.theta = nn.Parameter(torch.randn(feature_dim))
        
        # 优化器
        self.optimizer = torch.optim.Adam([self.theta], lr=lr)
        
    def reward(self, states, actions):
        """计算奖励: r(s,a) = θ · φ(s,a)"""
        features = self.extract_features(states, actions)
        return torch.sum(features * self.theta, dim=-1)
    
    def extract_features(self, states, actions):
        """提取特征"""
        # 简化为状态动作拼接
        return torch.cat([states, actions], dim=-1)
    
    def compute_expert_feature_expectation(self, expert_trajectories):
        """计算专家演示的特征期望"""
        total_feature = 0
        total_steps = 0
        
        for traj in expert_trajectories:
            for step in traj:
                state = torch.FloatTensor(step['state']).unsqueeze(0)
                action = torch.FloatTensor(step['action']).unsqueeze(0)
                
                feature = self.extract_features(state, action)
                total_feature += feature
                total_steps += 1
                
        return total_feature / total_steps
    
    def compute_policy_feature_expectation(self, policy, env, num_trajectories=100):
        """计算策略的特征期望"""
        total_feature = 0
        total_steps = 0
        
        for _ in range(num_trajectories):
            state = env.reset()
            done = False
            
            while not done:
                action = policy.select_action(state)
                
                state_t = torch.FloatTensor(state).unsqueeze(0)
                action_t = torch.FloatTensor(action).unsqueeze(0)
                
                feature = self.extract_features(state_t, action_t)
                total_feature += feature
                total_steps += 1
                
                state, _, done, _ = env.step(action)
                
        return total_feature / total_steps
    
    def train(self, expert_trajectories, env, policy, num_iterations=100):
        """训练"""
        # 计算专家特征期望
        expert_feature_exp = self.compute_expert_feature_expectation(expert_trajectories)
        
        for iteration in range(num_iterations):
            # 计算当前策略的特征期望
            policy_feature_exp = self.compute_policy_feature_expectation(policy, env)
            
            # 损失: 最小化差异
            loss = -torch.sum(self.theta * (expert_feature_exp - policy_feature_exp))
            
            # 添加正则化
            loss += 0.01 * torch.sum(self.theta ** 2)
            
            # 优化
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            if iteration % 10 == 0:
                print(f"Iter {iteration}, Loss: {loss.item():.4f}")
                
        return self.theta
```

---

## 3. 生成对抗IRL

### 3.1 GAIL原理

GAIL使用GAN框架：
- 判别器：区分专家数据和策略数据
- 生成器（策略）：试图欺骗判别器

```python
class GAIL(nn.Module):
    """
    生成对抗逆强化学习
    """
    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super(GAIL, self).__init__()
        
        # 策略网络
        self.actor = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Tanh()
        )
        
        # 判别器网络
        self.discriminator = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
        
    def get_action(self, state):
        """获取动作"""
        return self.actor(state)
    
    def discriminator_loss(self, expert_states, expert_actions, policy_states, policy_actions):
        """判别器损失"""
        # 专家: 标签1
        expert_input = torch.cat([expert_states, expert_actions], dim=-1)
        expert_prob = self.discriminator(expert_input)
        
        # 策略: 标签0
        policy_input = torch.cat([policy_states, policy_actions], dim=-1)
        policy_prob = self.discriminator(policy_input)
        
        # 二分类交叉熵
        loss = -torch.log(expert_prob + 1e-8).mean() - \
               torch.log(1 - policy_prob + 1e-8).mean()
        
        return loss
    
    def actor_loss(self, states, actions):
        """策略损失 (使用判别器作为奖励)"""
        input_feat = torch.cat([states, actions], dim=-1)
        
        # 判别器输出作为奖励 (越高越好)
        reward = self.discriminator(input_feat)
        
        # 策略梯度
        loss = -torch.log(reward + 1e-8).mean()
        
        return loss
```

---

## 4. 神经IRL

### 4.1 神经奖励函数

```python
class NeuralRewardFunction(nn.Module):
    """
    神经网络奖励函数
    """
    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super(NeuralRewardFunction, self).__init__()
        
        self.network = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
    def forward(self, state, action):
        input_feat = torch.cat([state, action], dim=-1)
        return self.network(input_feat)


class NeuralIRL:
    """
    神经逆强化学习
    """
    def __init__(self, state_dim, action_dim, hidden_dim=128):
        self.reward_fn = NeuralRewardFunction(state_dim, action_dim, hidden_dim)
        
        # 使用PPO作为策略优化器
        self.policy = PPO(state_dim, action_dim)
        
    def train_step(self, expert_batch, policy_batch):
        """训练步骤"""
        # 计算奖励
        expert_rewards = self.reward_fn(expert_batch['states'], expert_batch['actions'])
        policy_rewards = self.reward_fn(policy_batch['states'], policy_batch['actions'])
        
        # 奖励损失: 专家奖励应该高于策略
        reward_loss = torch.relu(policy_rewards - expert_rewards + 0.1).mean()
        
        # 更新奖励函数
        reward_optimizer = torch.optim.Adam(self.reward_fn.parameters())
        reward_optimizer.zero_grad()
        reward_loss.backward()
        reward_optimizer.step()
        
        # 使用奖励函数训练策略
        self.policy.update(policy_batch['states'], policy_batch['actions'],
                          expert_rewards)
        
        return reward_loss.item()
```

---

## 5. 应用案例

### 5.1 自动驾驶

```python
class AutonomousDrivingIRL:
    """
    自动驾驶逆强化学习
    从人类驾驶数据学习奖励函数
    """
    def __init__(self):
        # 状态: 位置, 速度, 加速度, 与前车距离等
        # 动作: 转向角, 油门, 刹车
        pass
    
    def extract_driving_features(self, state, action):
        """提取驾驶特征"""
        features = {
            'speed': state['velocity'],
            'acceleration': action['throttle'],
            'lateral_deviation': state['lane_offset'],
            'time_to_collision': state['ttc'],
            'comfort': abs(action['steering']),
        }
        return features
```

### 5.2 机器人操作

```python
class RobotManipulationIRL:
    """
    机器人操作逆强化学习
    """
    def __init__(self):
        pass
    
    def define_reward_features(self):
        """定义奖励特征"""
        return {
            # 任务相关
            'task_completion': '是否完成任务',
            'grasp_stability': '抓取稳定性',
            
            # 效率相关
            'time_efficiency': '完成时间',
            'path_length': '路径长度',
            
            # 安全相关
            'collision': '是否碰撞',
            'force_limits': '力限制',
        }
```

---

## 参考文献

1. Ng, A. Y., & Russell, S. (2000). Algorithms for inverse reinforcement learning. ICML.
2. Ziebart, B. D., et al. (2008). Maximum entropy inverse reinforcement learning. AAAI.
3. Ho, J., & Ermon, S. (2016). Generative adversarial imitation learning. NeurIPS.
4. Finn, C., Christiano, P., & Abbeel, P. (2016). A connection between generative adversarial networks, inverse reinforcement learning, and energy-based models. NeurIPS.

---

*本章节持续更新中...*
