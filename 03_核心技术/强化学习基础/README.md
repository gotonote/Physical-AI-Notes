# 强化学习基础

> Reinforcement Learning Fundamentals for Robotics

强化学习（RL）是物理AI的核心技术之一，让机器人能够通过与环境的交互自主学习最优行为策略。

---

## 📋 目录

1. [强化学习概述](#1-强化学习概述)
2. [MDP基础](#2-mdp基础)
3. [Value-based方法](#3-value-based方法)
4. [Policy-based方法](#4-policy-based方法)
5. [Actor-Critic方法](#5-actor-critic方法)
6. [代码实现](#6-代码实现)
7. [实战练习](#7-实战练习)

---

## 1. 强化学习概述

### 1.1 核心概念

```
┌─────────────────────────────────────────────────────────┐
│               强化学习框架                               │
│                                                         │
│     ┌─────────┐         ┌─────────┐                    │
│     │  Agent  │ ───────▶│  环境   │                    │
│     │  智能体  │  动作a  │ Environ │                    │
│     └────┬────┘         └────┬────┘                    │
│          │                   │                         │
│          │     状态s         │                         │
│          │◀─────────────────┤                          │
│          │                   │                         │
│          │     奖励r         │                         │
│          │◀─────────────────┤                          │
│          │                   │                         │
└─────────────────────────────────────────────────────────┘

核心要素：
- State (s): 环境状态
- Action (a): 智能体动作
- Reward (r): 即时奖励信号
- Policy (π): 状态到动作的映射
- Value (V): 状态/动作的长期价值
```

### 1.2 与监督学习的区别

| 特性 | 监督学习 | 强化学习 |
|------|---------|---------|
| 数据来源 | 静态标注数据 | 环境交互产生 |
| 反馈时机 | 即时 | 延迟 |
| 目标 | 拟合标签 | 最大化累积奖励 |
| 数据分布 | 固定 | 受策略影响 |

### 1.3 在机器人中的应用

```
应用场景：
├── 机械臂抓取
│   └─ 学习最优抓取策略
├── 足式机器人行走
│   └─ 学习稳定步态
├── 自动驾驶
│   └─ 学习驾驶决策
└── 游戏AI
    └─ 学习游戏策略
```

---

## 2. MDP基础

### 2.1 马尔可夫决策过程 (MDP)

**定义**：MDP由五元组 $(S, A, P, R, \gamma)$ 组成

- $S$: 状态空间
- $A$: 动作空间
- $P(s'|s,a)$: 状态转移概率
- $R(s,a,s')$: 奖励函数
- $\gamma$: 折扣因子

### 2.2 贝尔曼方程

**状态价值函数**：
$$V^\pi(s) = \mathbb{E}_\pi \left[ \sum_{t=0}^{\infty} \gamma^t r_t | s_0 = s \right]$$

**贝尔曼期望方程**：
$$V^\pi(s) = \sum_{a} \pi(a|s) \sum_{s'} P(s'|s,a)[R(s,a,s') + \gamma V^\pi(s')]$$

**贝尔曼最优方程**：
$$V^*(s) = \max_a \sum_{s'} P(s'|s,a)[R(s,a,s') + \gamma V^*(s')]$$

### 2.3 机器人MDP示例

```python
import numpy as np
from typing import Tuple, Dict, List

class RobotArmMDP:
    """机械臂抓取MDP环境"""
    
    def __init__(self):
        # 状态空间：末端位置(x,y,z) + 目标位置 + 夹爪状态
        self.state_dim = 7
        
        # 动作空间：位置增量(Δx,Δy,Δz) + 夹爪开合
        self.action_dim = 4
        
        # 折扣因子
        self.gamma = 0.99
        
        # 环境参数
        self.target_pos = np.array([0.5, 0.0, 0.3])
        self.gripper_state = 1.0  # 1.0=开, 0.0=关
        
    def reset(self) -> np.ndarray:
        """重置环境"""
        self.gripper_pos = np.array([0.0, 0.0, 0.5])
        self.gripper_state = 1.0
        return self._get_state()
    
    def _get_state(self) -> np.ndarray:
        """获取当前状态"""
        return np.concatenate([
            self.gripper_pos,
            self.target_pos,
            [self.gripper_state]
        ])
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict]:
        """执行动作，返回(新状态, 奖励, 是否终止, 信息)"""
        
        # 更新位置（限制范围）
        delta = action[:3] * 0.05  # 缩放动作
        self.gripper_pos = np.clip(
            self.gripper_pos + delta,
            [-1, -1, 0], [1, 1, 1]
        )
        
        # 更新夹爪状态
        self.gripper_state = np.clip(
            self.gripper_state + action[3] * 0.1, 0, 1
        )
        
        # 计算奖励
        distance = np.linalg.norm(self.gripper_pos - self.target_pos)
        reward = -distance  # 距离惩罚
        
        # 抓取成功奖励
        if distance < 0.05 and self.gripper_state < 0.2:
            reward += 10.0
            done = True
        else:
            done = False
        
        # 超时惩罚
        if hasattr(self, 'steps'):
            self.steps += 1
            if self.steps > 200:
                done = True
                reward -= 5.0
        else:
            self.steps = 1
        
        return self._get_state(), reward, done, {}
```

---

## 3. Value-based方法

### 3.1 Q-Learning

**核心思想**：学习动作价值函数 $Q(s,a)$

**更新规则**：
$$Q(s,a) \leftarrow Q(s,a) + \alpha [r + \gamma \max_{a'} Q(s',a') - Q(s,a)]$$

### 3.2 DQN (Deep Q-Network)

使用神经网络近似Q函数。

```python
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque

class DQNNetwork(nn.Module):
    """DQN网络"""
    
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DQNAgent:
    """DQN智能体"""
    
    def __init__(self, state_dim: int, action_dim: int, 
                 lr: float = 1e-3, gamma: float = 0.99,
                 epsilon: float = 1.0, epsilon_min: float = 0.01,
                 epsilon_decay: float = 0.995,
                 buffer_size: int = 10000, batch_size: int = 64):
        
        self.action_dim = action_dim
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        
        # 主网络和目标网络
        self.q_network = DQNNetwork(state_dim, action_dim)
        self.target_network = DQNNetwork(state_dim, action_dim)
        self.target_network.load_state_dict(self.q_network.state_dict())
        
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)
        
        # 经验回放缓冲区
        self.replay_buffer = deque(maxlen=buffer_size)
    
    def select_action(self, state: np.ndarray) -> int:
        """ε-贪婪策略选择动作"""
        if random.random() < self.epsilon:
            return random.randint(0, self.action_dim - 1)
        
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            q_values = self.q_network(state_tensor)
            return q_values.argmax(dim=1).item()
    
    def store_transition(self, state, action, reward, next_state, done):
        """存储经验"""
        self.replay_buffer.append((state, action, reward, next_state, done))
    
    def train(self):
        """训练一步"""
        if len(self.replay_buffer) < self.batch_size:
            return None
        
        # 采样批次
        batch = random.sample(self.replay_buffer, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        states = torch.FloatTensor(np.array(states))
        actions = torch.LongTensor(actions)
        rewards = torch.FloatTensor(rewards)
        next_states = torch.FloatTensor(np.array(next_states))
        dones = torch.FloatTensor(dones)
        
        # 计算当前Q值
        current_q = self.q_network(states).gather(1, actions.unsqueeze(1))
        
        # 计算目标Q值（Double DQN）
        with torch.no_grad():
            next_actions = self.q_network(next_states).argmax(dim=1)
            next_q = self.target_network(next_states).gather(
                1, next_actions.unsqueeze(1)
            ).squeeze()
            target_q = rewards + self.gamma * next_q * (1 - dones)
        
        # 计算损失并更新
        loss = nn.MSELoss()(current_q.squeeze(), target_q)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # 更新ε
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        
        return loss.item()
    
    def update_target_network(self):
        """更新目标网络"""
        self.target_network.load_state_dict(self.q_network.state_dict())
```

### 3.3 DQN变体

| 方法 | 改进点 | 核心思想 |
|------|--------|---------|
| Double DQN | 减少过估计 | 分离动作选择和评估 |
| Dueling DQN | 更好价值估计 | 分离状态价值和动作优势 |
| Prioritized ER | 样本效率 | 优先采样高TD误差样本 |
| Rainbow | 综合改进 | 整合多种改进技术 |

---

## 4. Policy-based方法

### 4.1 策略梯度

**核心思想**：直接优化策略 $\pi_\theta(a|s)$

**目标函数**：
$$J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta} \left[ \sum_{t=0}^{T} \gamma^t r_t \right]$$

**策略梯度定理**：
$$\nabla_\theta J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta} \left[ \sum_{t=0}^{T} \nabla_\theta \log \pi_\theta(a_t|s_t) \cdot G_t \right]$$

### 4.2 REINFORCE

```python
class PolicyNetwork(nn.Module):
    """策略网络（用于连续动作空间）"""
    
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        # 均值和标准差输出
        self.mean_head = nn.Linear(hidden_dim, action_dim)
        self.log_std = nn.Parameter(torch.zeros(action_dim))
    
    def forward(self, x: torch.Tensor):
        features = self.shared(x)
        mean = self.mean_head(features)
        std = torch.exp(self.log_std.clamp(-20, 2))
        return mean, std
    
    def get_action(self, state: np.ndarray):
        """采样动作"""
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            mean, std = self.forward(state_tensor)
            dist = torch.distributions.Normal(mean, std)
            action = dist.sample()
            log_prob = dist.log_prob(action).sum(dim=-1)
        return action.squeeze().numpy(), log_prob.item()


class REINFORCEAgent:
    """REINFORCE智能体"""
    
    def __init__(self, state_dim: int, action_dim: int, lr: float = 1e-3, gamma: float = 0.99):
        self.gamma = gamma
        self.policy = PolicyNetwork(state_dim, action_dim)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        
        # 存储一条轨迹
        self.log_probs = []
        self.rewards = []
    
    def select_action(self, state: np.ndarray) -> np.ndarray:
        """选择动作"""
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        mean, std = self.policy(state_tensor)
        dist = torch.distributions.Normal(mean, std)
        action = dist.sample()
        log_prob = dist.log_prob(action).sum(dim=-1)
        
        self.log_probs.append(log_prob)
        return action.squeeze().numpy()
    
    def store_reward(self, reward: float):
        """存储奖励"""
        self.rewards.append(reward)
    
    def update(self):
        """策略更新"""
        # 计算折扣累积奖励
        returns = []
        G = 0
        for r in reversed(self.rewards):
            G = r + self.gamma * G
            returns.insert(0, G)
        
        returns = torch.FloatTensor(returns)
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)  # 标准化
        
        # 计算策略梯度损失
        log_probs = torch.cat(self.log_probs)
        loss = -(log_probs * returns).mean()
        
        # 更新
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # 清空存储
        self.log_probs = []
        self.rewards = []
        
        return loss.item()
```

### 4.3 PPO (Proximal Policy Optimization)

PPO是当前最流行的策略梯度算法之一。

```python
class PPOAgent:
    """PPO智能体"""
    
    def __init__(self, state_dim: int, action_dim: int,
                 lr: float = 3e-4, gamma: float = 0.99,
                 gae_lambda: float = 0.95,
                 clip_epsilon: float = 0.2,
                 entropy_coef: float = 0.01,
                 value_coef: float = 0.5):
        
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_epsilon
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        
        # Actor-Critic网络
        self.actor = PolicyNetwork(state_dim, action_dim)
        self.critic = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
        
        self.optimizer = optim.Adam(
            list(self.actor.parameters()) + list(self.critic.parameters()),
            lr=lr
        )
    
    def compute_gae(self, rewards, values, dones, next_value):
        """计算广义优势估计(GAE)"""
        advantages = []
        gae = 0
        
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_val = next_value
            else:
                next_val = values[t + 1]
            
            delta = rewards[t] + self.gamma * next_val * (1 - dones[t]) - values[t]
            gae = delta + self.gamma * self.gae_lambda * (1 - dones[t]) * gae
            advantages.insert(0, gae)
        
        return advantages
    
    def update(self, states, actions, old_log_probs, advantages, returns, epochs=10, batch_size=64):
        """PPO更新"""
        states = torch.FloatTensor(states)
        actions = torch.FloatTensor(actions)
        old_log_probs = torch.FloatTensor(old_log_probs)
        advantages = torch.FloatTensor(advantages)
        returns = torch.FloatTensor(returns)
        
        dataset_size = len(states)
        
        for _ in range(epochs):
            indices = np.random.permutation(dataset_size)
            
            for start in range(0, dataset_size, batch_size):
                end = start + batch_size
                idx = indices[start:end]
                
                batch_states = states[idx]
                batch_actions = actions[idx]
                batch_old_log_probs = old_log_probs[idx]
                batch_advantages = advantages[idx]
                batch_returns = returns[idx]
                
                # 计算新的log_prob
                mean, std = self.actor(batch_states)
                dist = torch.distributions.Normal(mean, std)
                new_log_probs = dist.log_prob(batch_actions).sum(dim=-1)
                entropy = dist.entropy().sum(dim=-1).mean()
                
                # PPO裁剪目标
                ratio = torch.exp(new_log_probs - batch_old_log_probs)
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * batch_advantages
                policy_loss = -torch.min(surr1, surr2).mean()
                
                # 价值函数损失
                values = self.critic(batch_states).squeeze()
                value_loss = nn.MSELoss()(values, batch_returns)
                
                # 总损失
                loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy
                
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    list(self.actor.parameters()) + list(self.critic.parameters()),
                    0.5
                )
                self.optimizer.step()
        
        return loss.item()
```

---

## 5. Actor-Critic方法

### 5.1 A2C / A3C

```
┌─────────────────────────────────────────────────────────┐
│                  Actor-Critic架构                        │
│                                                         │
│              ┌──────────────────────┐                   │
│              │       State s        │                   │
│              └──────────┬───────────┘                   │
│                         │                               │
│              ┌──────────▼───────────┐                   │
│              │     特征提取器        │                   │
│              └──────────┬───────────┘                   │
│                         │                               │
│         ┌───────────────┼───────────────┐               │
│         ▼                               ▼               │
│  ┌─────────────┐                ┌─────────────┐         │
│  │   Actor π   │                │  Critic V   │         │
│  │   策略网络   │                │  价值网络   │         │
│  └──────┬──────┘                └──────┬──────┘         │
│         │                              │                │
│         ▼                              ▼                │
│    Action a                         Value V(s)          │
│                                                         │
│  Actor损失: -log π(a|s) * A(s,a)                        │
│  Critic损失: (V(s) - R)²                                │
│  优势函数: A(s,a) = Q(s,a) - V(s)                       │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

### 5.2 SAC (Soft Actor-Critic)

SAC是当前最先进的连续控制算法之一，引入了熵正则化。

```python
class SACAgent:
    """SAC智能体"""
    
    def __init__(self, state_dim: int, action_dim: int,
                 lr: float = 3e-4, gamma: float = 0.99,
                 tau: float = 0.005, alpha: float = 0.2,
                 auto_entropy: bool = True):
        
        self.gamma = gamma
        self.tau = tau
        self.auto_entropy = auto_entropy
        
        # 网络
        self.actor = GaussianActor(state_dim, action_dim)
        self.critic1 = CriticNetwork(state_dim, action_dim)
        self.critic2 = CriticNetwork(state_dim, action_dim)
        self.critic1_target = CriticNetwork(state_dim, action_dim)
        self.critic2_target = CriticNetwork(state_dim, action_dim)
        
        # 复制参数到目标网络
        self.critic1_target.load_state_dict(self.critic1.state_dict())
        self.critic2_target.load_state_dict(self.critic2.state_dict())
        
        # 优化器
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr)
        self.critic1_optimizer = optim.Adam(self.critic1.parameters(), lr=lr)
        self.critic2_optimizer = optim.Adam(self.critic2.parameters(), lr=lr)
        
        # 熵系数
        if auto_entropy:
            self.target_entropy = -action_dim
            self.log_alpha = torch.zeros(1, requires_grad=True)
            self.alpha_optimizer = optim.Adam([self.log_alpha], lr=lr)
        else:
            self.alpha = alpha
    
    def update(self, batch):
        """SAC更新"""
        states, actions, rewards, next_states, dones = batch
        
        states = torch.FloatTensor(states)
        actions = torch.FloatTensor(actions)
        rewards = torch.FloatTensor(rewards).unsqueeze(1)
        next_states = torch.FloatTensor(next_states)
        dones = torch.FloatTensor(dones).unsqueeze(1)
        
        # ---- 更新Critic ----
        with torch.no_grad():
            next_actions, next_log_probs = self.actor.sample(next_states)
            q1_next = self.critic1_target(next_states, next_actions)
            q2_next = self.critic2_target(next_states, next_actions)
            q_next = torch.min(q1_next, q2_next) - self.alpha * next_log_probs
            target_q = rewards + self.gamma * (1 - dones) * q_next
        
        q1 = self.critic1(states, actions)
        q2 = self.critic2(states, actions)
        
        critic1_loss = nn.MSELoss()(q1, target_q)
        critic2_loss = nn.MSELoss()(q2, target_q)
        
        self.critic1_optimizer.zero_grad()
        critic1_loss.backward()
        self.critic1_optimizer.step()
        
        self.critic2_optimizer.zero_grad()
        critic2_loss.backward()
        self.critic2_optimizer.step()
        
        # ---- 更新Actor ----
        new_actions, log_probs = self.actor.sample(states)
        q1_new = self.critic1(states, new_actions)
        q2_new = self.critic2(states, new_actions)
        q_new = torch.min(q1_new, q2_new)
        
        actor_loss = (self.alpha * log_probs - q_new).mean()
        
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        
        # ---- 更新Alpha ----
        if self.auto_entropy:
            alpha_loss = -(self.log_alpha * (log_probs + self.target_entropy).detach()).mean()
            
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()
            
            self.alpha = self.log_alpha.exp().item()
        
        # ---- 软更新目标网络 ----
        for param, target_param in zip(self.critic1.parameters(), self.critic1_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        
        for param, target_param in zip(self.critic2.parameters(), self.critic2_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        
        return {
            'critic_loss': (critic1_loss.item() + critic2_loss.item()) / 2,
            'actor_loss': actor_loss.item(),
            'alpha': self.alpha
        }
```

### 5.3 TD3 (Twin Delayed DDPG)

TD3解决了DDPG的Q值过估计问题。

**三个核心改进**：
1. **Twin Critics** - 使用两个Critic取最小值
2. **Delayed Policy Updates** - 延迟更新Actor
3. **Target Policy Smoothing** - 目标策略平滑

---

## 6. 代码实现

### 完整训练脚本

```python
import gymnasium as gym
import numpy as np
from tqdm import tqdm

def train_dqn(env_name: str = "CartPole-v1", 
              num_episodes: int = 1000,
              target_update_freq: int = 10):
    """DQN训练脚本"""
    
    env = gym.make(env_name)
    agent = DQNAgent(
        state_dim=env.observation_space.shape[0],
        action_dim=env.action_space.n
    )
    
    rewards_history = []
    
    for episode in tqdm(range(num_episodes)):
        state, _ = env.reset()
        total_reward = 0
        done = False
        
        while not done:
            action = agent.select_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            agent.store_transition(state, action, reward, next_state, done)
            agent.train()
            
            state = next_state
            total_reward += reward
        
        rewards_history.append(total_reward)
        
        # 定期更新目标网络
        if episode % target_update_freq == 0:
            agent.update_target_network()
        
        # 打印进度
        if (episode + 1) % 100 == 0:
            avg_reward = np.mean(rewards_history[-100:])
            print(f"Episode {episode+1}, Avg Reward: {avg_reward:.2f}, Epsilon: {agent.epsilon:.3f}")
    
    return agent, rewards_history


def train_ppo(env_name: str = "HalfCheetah-v4",
              num_iterations: int = 1000,
              steps_per_iter: int = 2048):
    """PPO训练脚本"""
    
    env = gym.make(env_name)
    agent = PPOAgent(
        state_dim=env.observation_space.shape[0],
        action_dim=env.action_space.shape[0]
    )
    
    rewards_history = []
    
    for iteration in tqdm(range(num_iterations)):
        # 收集数据
        states, actions, rewards, dones, log_probs, values = [], [], [], [], [], []
        state, _ = env.reset()
        
        for _ in range(steps_per_iter):
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            
            with torch.no_grad():
                mean, std = agent.actor(state_tensor)
                dist = torch.distributions.Normal(mean, std)
                action = dist.sample()
                log_prob = dist.log_prob(action).sum(dim=-1)
                value = agent.critic(state_tensor)
            
            action_np = action.squeeze().numpy()
            next_state, reward, terminated, truncated, _ = env.step(action_np)
            done = terminated or truncated
            
            states.append(state)
            actions.append(action_np)
            rewards.append(reward)
            dones.append(done)
            log_probs.append(log_prob.item())
            values.append(value.item())
            
            state = next_state if not done else env.reset()[0]
        
        # 计算GAE和returns
        with torch.no_grad():
            next_value = agent.critic(torch.FloatTensor(state).unsqueeze(0)).item()
        
        advantages = agent.compute_gae(rewards, values, dones, next_value)
        returns = [a + v for a, v in zip(advantages, values)]
        
        # 更新策略
        agent.update(states, actions, log_probs, advantages, returns)
        
        avg_reward = np.mean(rewards)
        rewards_history.append(avg_reward)
        
        if (iteration + 1) % 50 == 0:
            print(f"Iteration {iteration+1}, Avg Reward: {avg_reward:.2f}")
    
    return agent, rewards_history


if __name__ == "__main__":
    # 训练DQN
    print("Training DQN on CartPole...")
    dqn_agent, dqn_rewards = train_dqn()
    
    # 训练PPO
    print("\nTraining PPO on HalfCheetah...")
    ppo_agent, ppo_rewards = train_ppo()
```

---

## 7. 实战练习

### 练习1：实现一个简单的Q-Learning

**任务**：在FrozenLake环境中实现Q-Learning

```python
# 练习框架
import gymnasium as gym
import numpy as np

def q_learning():
    env = gym.make("FrozenLake-v1")
    
    # 初始化Q表
    Q = np.zeros((env.observation_space.n, env.action_space.n))
    
    # TODO: 实现Q-Learning算法
    # 1. ε-贪婪动作选择
    # 2. Q表更新
    # 3. 训练循环
    
    pass

# 完成后测试
if __name__ == "__main__":
    q_learning()
```

### 练习2：从DQN到Double DQN

**任务**：将上面的DQN代码修改为Double DQN

**提示**：在计算目标Q值时，使用主网络选择动作，目标网络评估Q值

### 练习3：PPO实现细节

**任务**：实现完整的PPO算法，包括：
1. GAE计算
2. 优势函数标准化
3. 多epoch小批次更新
4. 梯度裁剪

### 练习4：机器人任务

**任务**：在MuJoCo环境中训练一个机器人控制策略

```python
import gymnasium as gym

# 使用HalfCheetah或Ant环境
env = gym.make("HalfCheetah-v4")

# TODO: 
# 1. 实现SAC或PPO算法
# 2. 训练直到平均奖励 > 3000
# 3. 分析学习曲线
```

---

## 📚 推荐资源

### 经典论文
- **DQN**: Playing Atari with Deep RL (Mnih et al., 2015)
- **PPO**: Proximal Policy Optimization Algorithms (Schulman et al., 2017)
- **SAC**: Soft Actor-Critic (Haarnoja et al., 2018)
- **TD3**: Twin Delayed DDPG (Fujimoto et al., 2018)

### 在线课程
- [DeepMind RL Course](https://www.deepmind.com/learning-resources)
- [Stanford CS234](http://web.stanford.edu/class/cs234/)
- [Berkeley CS285](http://rail.eecs.berkeley.edu/deeprlcourse/)

### 开源实现
- [Stable-Baselines3](https://github.com/DLR-RM/stable-baselines3)
- [CleanRL](https://github.com/vwxyzjn/cleanrl)
- [RLlib](https://docs.ray.io/en/latest/rllib/)

---

*本文档持续更新，欢迎反馈和建议！*
