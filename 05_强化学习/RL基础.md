# 强化学习基础

强化学习（Reinforcement Learning, RL）是物理AI中的核心技术，使智能体通过与环境交互学习最优策略。本章介绍RL基础理论、核心算法及实现方法。

## 目录

- [1. 强化学习概述](#1-强化学习概述)
- [2. 马尔可夫决策过程](#2-马尔可夫决策过程)
- [3. 动态规划方法](#3-动态规划方法)
- [4. 蒙特卡洛方法](#4-蒙特卡洛方法)
- [5. 时序差分学习](#5-时序差分学习)
- [6. 值函数近似](#6-值函数近似)
- [7. 策略梯度方法](#7-策略梯度方法)
- [8. DQN及其变体](#8-dqn及其变体)

---

## 1. 强化学习概述

### 1.1 强化学习基本框架

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

### 1.2 强化学习基本要素

| 要素 | 描述 | 作用 |
|------|------|------|
| 状态 (s) | 环境的观测 | 智能体决策的依据 |
| 动作 (a) | 智能体的行为 | 影响环境状态 |
| 奖励 (r) | 环境的反馈 | 指导策略优化 |
| 策略 π | 状态到动作的映射 | 智能体的行为准则 |
| 值函数 V | 长期回报的估计 | 评估状态/动作好坏 |
| 模型 | 环境 dynamics | 预测未来状态和奖励 |

---

## 2. 马尔可夫决策过程

### 2.1 马尔可夫性质

**马尔可夫性质**：未来只取决于当前状态，与历史无关。

$$
P(s_{t+1} | s_t, s_{t-1}, ..., s_0) = P(s_{t+1} | s_t)
$$

### 2.2 马尔可夫决策过程定义

MDP由五元组 $(S, A, P, R, \gamma)$ 定义：

- **S**: 状态空间 (State Space)
- **A**: 动作空间 (Action Space)
- **P**: 状态转移概率 (Transition Probability) $P(s'|s,a)$
- **R**: 奖励函数 (Reward Function) $R(s,a,s')$
- **$\gamma$**: 折扣因子 (Discount Factor), $\gamma \in [0, 1]$

### 2.3 回报与值函数

**回报 (Return)**：从时刻t开始的累计折扣奖励

$$
G_t = r_t + \gamma r_{t+1} + \gamma^2 r_{t+2} + ... = \sum_{k=0}^{\infty} \gamma^k r_{t+k}
$$

**状态值函数 (State Value Function)**

$$
V^\pi(s) = \mathbb{E}_\pi[G_t | s_t = s]
$$

**动作值函数 (Action Value Function)**

$$
Q^\pi(s, a) = \mathbb{E}_\pi[G_t | s_t = s, a_t = a]
$$

**贝尔曼方程**

$$
V^\pi(s) = \sum_a \pi(a|s) \sum_{s'} P(s'|s,a)[R(s,a,s') + \gamma V^\pi(s')]
$$

$$
Q^\pi(s, a) = \sum_{s'} P(s'|s,a)[R(s,a,s') + \gamma \sum_{a'} \pi(a'|s') Q^\pi(s', a')]
$$

### 2.4 最优策略

**最优值函数**

$$
V^*(s) = \max_\pi V^\pi(s)
$$

$$
Q^*(s, a) = \max_\pi Q^\pi(s, a)
$$

**贝尔曼最优方程**

$$
V^*(s) = \max_a \sum_{s'} P(s'|s,a)[R(s,a,s') + \gamma V^*(s')]
$$

---

## 3. 动态规划方法

### 3.1 策略迭代

```python
import numpy as np

class PolicyIteration:
    def __init__(self, env, gamma=0.9, theta=1e-6):
        self.env = env
        self.gamma = gamma
        self.theta = theta
        
        # 初始化
        n_states = env.observation_space.n
        n_actions = env.action_space.n
        
        self.V = np.zeros(n_states)
        self.policy = np.ones([n_states, n_actions]) / n_actions  # 均匀随机策略
        
    def policy_evaluation(self):
        """策略评估：计算给定策略的值函数"""
        while True:
            delta = 0
            
            for s in range(self.env.observation_space.n):
                v = self.V[s]
                
                # 计算新值
                new_v = 0
                for a in range(self.env.action_space.n):
                    for prob, next_state, reward, done in self.env.P[s][a]:
                        new_v += self.policy[s, a] * prob * (
                            reward + self.gamma * self.V[next_state]
                        )
                
                self.V[s] = new_v
                delta = max(delta, abs(v - new_v))
            
            if delta < self.theta:
                break
                
    def policy_improvement(self):
        """策略改进：根据值函数更新策略"""
        policy_stable = True
        
        for s in range(self.env.observation_space.n):
            old_action = np.argmax(self.policy[s])
            
            # 计算每个动作的Q值
            action_values = []
            for a in range(self.env.action_space.n):
                q_value = 0
                for prob, next_state, reward, done in self.env.P[s][a]:
                    q_value += prob * (reward + self.gamma * self.V[next_state])
                action_values.append(q_value)
            
            # 选择最优动作
            best_action = np.argmax(action_values)
            
            if old_action != best_action:
                policy_stable = False
            
            # 更新策略
            self.policy[s] = np.zeros(self.env.action_space.n)
            self.policy[s, best_action] = 1.0
            
        return policy_stable
    
    def train(self, max_iterations=1000):
        """策略迭代训练"""
        for i in range(max_iterations):
            # 策略评估
            self.policy_evaluation()
            
            # 策略改进
            if self.policy_improvement():
                print(f"策略在第 {i+1} 次迭代后收敛")
                break
```

### 3.2 价值迭代

```python
class ValueIteration:
    def __init__(self, env, gamma=0.9, theta=1e-6):
        self.env = env
        self.gamma = gamma
        self.theta = theta
        
        n_states = env.observation_space.n
        n_actions = env.action_space.n
        
        self.V = np.zeros(n_states)
        self.policy = np.zeros([n_states, n_actions])
        
    def value_iteration(self):
        """价值迭代"""
        while True:
            delta = 0
            
            for s in range(self.env.observation_space.n):
                v = self.V[s]
                
                # 计算最大值
                action_values = []
                for a in range(self.env.action_space.n):
                    q_value = 0
                    for prob, next_state, reward, done in self.env.P[s][a]:
                        q_value += prob * (reward + self.gamma * self.V[next_state])
                    action_values.append(q_value)
                
                self.V[s] = max(action_values)
                delta = max(delta, abs(v - self.V[s]))
            
            if delta < self.theta:
                break
        
        # 从值函数导出策略
        self.extract_policy()
        
    def extract_policy(self):
        """提取策略"""
        for s in range(self.env.observation_space.n):
            action_values = []
            for a in range(self.env.action_space.n):
                q_value = 0
                for prob, next_state, reward, done in self.env.P[s][a]:
                    q_value += prob * (reward + self.gamma * self.V[next_state])
                action_values.append(q_value)
            
            best_action = np.argmax(action_values)
            self.policy[s, best_action] = 1.0
```

---

## 4. 蒙特卡洛方法

### 4.1 蒙特卡洛预测

```python
class MonteCarloPrediction:
    def __init__(self, env, gamma=0.9):
        self.env = env
        self.gamma = gamma
        
        n_states = env.observation_space.n
        self.V = np.zeros(n_states)
        self.returns = {s: [] for s in range(n_states)}  # 记录每个状态的回报
        
    def run_episode(self, policy):
        """运行一个episode"""
        state = self.env.reset()
        done = False
        episode = []
        
        while not done:
            action = np.random.choice(
                range(self.env.action_space.n), 
                p=policy[state]
            )
            next_state, reward, done, _ = self.env.step(action)
            
            episode.append((state, action, reward))
            state = next_state
            
        return episode
    
    def train(self, num_episodes=10000, policy=None):
        """蒙特卡洛学习"""
        if policy is None:
            policy = np.ones([self.env.observation_space.n, 
                            self.env.action_space.n]) / self.env.action_space.n
        
        for _ in range(num_episodes):
            episode = self.run_episode(policy)
            
            # 计算回报
            G = 0
            for t, (state, action, reward) in enumerate(episode):
                G = self.gamma * G + reward
                
                # 首次访问MC
                if state not in [s for s, _, _ in episode[:t]]:
                    self.returns[state].append(G)
                    self.V[state] = np.mean(self.returns[state])
```

---

## 5. 时序差分学习

### 5.1 TD(0)算法

```python
class TDZero:
    def __init__(self, env, alpha=0.1, gamma=0.9):
        self.env = env
        self.alpha = alpha  # 学习率
        self.gamma = gamma
        
        n_states = env.observation_space.n
        self.V = np.zeros(n_states)
        
    def train(self, num_steps=10000, policy=None):
        """TD(0)学习"""
        state = self.env.reset()
        
        for _ in range(num_steps):
            if policy is None:
                action = np.random.randint(self.env.action_space.n)
            else:
                action = np.random.choice(
                    range(self.env.action_space.n),
                    p=policy[state]
                )
            
            next_state, reward, done, _ = self.env.step(action)
            
            # TD更新
            td_target = reward + self.gamma * self.V[next_state]
            td_error = td_target - self.V[state]
            self.V[state] += self.alpha * td_error
            
            if done:
                state = self.env.reset()
            else:
                state = next_state
```

### 5.2 SARSA算法（On-Policy TD控制）

```python
class SARSA:
    def __init__(self, env, alpha=0.1, gamma=0.9, epsilon=0.1):
        self.env = env
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        
        n_states = env.observation_space.n
        n_actions = env.action_space.n
        
        self.Q = np.zeros([n_states, n_actions])
        
    def choose_action(self, state):
        """ε-greedy策略"""
        if np.random.random() < self.epsilon:
            return np.random.randint(self.env.action_space.n)
        else:
            return np.argmax(self.Q[state])
    
    def train(self, num_episodes=500):
        """SARSA训练"""
        for episode in range(num_episodes):
            state = self.env.reset()
            action = self.choose_action(state)
            done = False
            
            while not done:
                next_state, reward, done, _ = self.env.step(action)
                next_action = self.choose_action(next_state)
                
                # SARSA更新
                td_target = reward + self.gamma * self.Q[next_state, next_action]
                td_error = td_target - self.Q[state, action]
                self.Q[state, action] += self.alpha * td_error
                
                state = next_state
                action = next_action
                
            # 衰减epsilon
            self.epsilon = max(0.01, self.epsilon * 0.995)
```

### 5.3 Q学习（Off-Policy TD控制）

```python
class QLearning:
    def __init__(self, env, alpha=0.1, gamma=0.9, epsilon=0.1):
        self.env = env
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        
        n_states = env.observation_space.n
        n_actions = env.action_space.n
        
        self.Q = np.zeros([n_states, n_actions])
        
    def choose_action(self, state):
        """ε-greedy策略"""
        if np.random.random() < self.epsilon:
            return np.random.randint(self.env.action_space.n)
        else:
            return np.argmax(self.Q[state])
    
    def train(self, num_episodes=500):
        """Q学习训练"""
        for episode in range(num_episodes):
            state = self.env.reset()
            done = False
            
            while not done:
                action = self.choose_action(state)
                next_state, reward, done, _ = self.env.step(action)
                
                # Q学习更新 (使用max而非实际动作)
                td_target = reward + self.gamma * np.max(self.Q[next_state])
                td_error = td_target - self.Q[state, action]
                self.Q[state, action] += self.alpha * td_error
                
                state = next_state
                
            self.epsilon = max(0.01, self.epsilon * 0.995)
```

---

## 6. 值函数近似

### 6.1 深度Q网络 (DQN)

```python
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
import random

class DQN(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super(DQN, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )
        
    def forward(self, x):
        return self.network(x)

class ReplayBuffer:
    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)
        
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
        
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            np.array(states),
            np.array(actions),
            np.array(rewards),
            np.array(next_states),
            np.array(dones)
        )
    
    def __len__(self):
        return len(self.buffer)

class DQNAgent:
    def __init__(self, state_dim, action_dim, gamma=0.99, lr=0.001):
        self.gamma = gamma
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        
        self.policy_net = DQN(state_dim, action_dim)
        self.target_net = DQN(state_dim, action_dim)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.replay_buffer = ReplayBuffer()
        
    def select_action(self, state, training=True):
        """ε-greedy动作选择"""
        if training and np.random.random() < self.epsilon:
            return np.random.randint(self.policy_net.output_dim)
        else:
            with torch.no_grad():
                q_values = self.policy_net(torch.FloatTensor(state).unsqueeze(0))
                return q_values.argmax().item()
    
    def update(self, batch_size):
        if len(self.replay_buffer) < batch_size:
            return
        
        # 采样
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(batch_size)
        
        states = torch.FloatTensor(states)
        actions = torch.LongTensor(actions)
        rewards = torch.FloatTensor(rewards)
        next_states = torch.FloatTensor(next_states)
        dones = torch.FloatTensor(dones)
        
        # 计算当前Q值
        current_q = self.policy_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        
        # 计算目标Q值
        with torch.no_grad():
            next_q = self.target_net(next_states).max(1)[0]
            target_q = rewards + (1 - dones) * self.gamma * next_q
        
        # 计算损失
        loss = nn.MSELoss()(current_q, target_q)
        
        # 优化
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
        self.optimizer.step()
        
        # 更新epsilon
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        
    def update_target(self):
        """更新目标网络"""
        self.target_net.load_state_dict(self.policy_net.state_dict())
```

---

## 7. 策略梯度方法

### 7.1 REINFORCE算法

```python
class REINFORCE:
    def __init__(self, state_dim, action_dim, hidden_dim=128, lr=0.01):
        self.policy_net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Softmax(dim=-1)
        )
        
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        
    def select_action(self, state):
        state = torch.FloatTensor(state).unsqueeze(0)
        probs = self.policy_net(state)
        action_dist = torch.distributions.Categorical(probs)
        action = action_dist.sample()
        log_prob = action_dist.log_prob(action)
        return action.item(), log_prob
    
    def update(self, log_probs, rewards, gamma=0.99):
        """策略梯度更新"""
        # 计算回报
        returns = []
        G = 0
        for r in reversed(rewards):
            G = r + gamma * G
            returns.insert(0, G)
        
        returns = torch.FloatTensor(returns)
        returns = (returns - returns.mean()) / (returns.std() + 1e-9)
        
        # 策略梯度
        policy_loss = []
        for log_prob, G in zip(log_probs, returns):
            policy_loss.append(-log_prob * G)
        
        loss = torch.stack(policy_loss).sum()
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
```

### 7.2 Actor-Critic算法

```python
class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super(ActorCritic, self).__init__()
        
        # Actor: 策略网络
        self.actor = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Softmax(dim=-1)
        )
        
        # Critic: 值函数网络
        self.critic = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
    def forward(self, state):
        probs = self.actor(state)
        value = self.critic(state)
        return probs, value

class A2CAgent:
    def __init__(self, state_dim, action_dim, gamma=0.99, lr=0.001):
        self.model = ActorCritic(state_dim, action_dim)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.gamma = gamma
        
    def select_action(self, state):
        state = torch.FloatTensor(state).unsqueeze(0)
        probs, value = self.model(state)
        
        action_dist = torch.distributions.Categorical(probs)
        action = action_dist.sample()
        log_prob = action_dist.log_prob(action)
        
        return action.item(), log_prob, value
    
    def update(self, states, actions, rewards, next_states, dones):
        states = torch.FloatTensor(states)
        actions = torch.LongTensor(actions)
        rewards = torch.FloatTensor(rewards)
        next_states = torch.FloatTensor(next_states)
        dones = torch.FloatTensor(dones)
        
        # 计算TD目标
        with torch.no_grad():
            _, next_value = self.model(next_states)
            td_target = rewards + self.gamma * next_value.squeeze() * (1 - dones)
        
        # 计算当前值
        _, current_value = self.model(states)
        
        # Actor损失 (策略梯度 + 优势函数)
        probs, values = self.model(states)
        action_dist = torch.distributions.Categorical(probs)
        log_probs = action_dist.log_prob(actions)
        
        advantage = td_target - values.squeeze()
        
        actor_loss = -(log_probs * advantage.detach()).mean()
        critic_loss = nn.MSELoss()(values.squeeze(), td_target.detach())
        
        # 总损失
        loss = actor_loss + 0.5 * critic_loss
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
```

---

## 8. DQN及其变体

### 8.1 Double DQN

解决Q值过估计问题：

```python
class DoubleDQN:
    def __init__(self, state_dim, action_dim):
        self.online_net = DQN(state_dim, action_dim)
        self.target_net = DQN(state_dim, action_dim)
        
        # 使用online网络选择动作，target网络评估动作
        # ...
        
    def update(self):
        # Double DQN更新
        with torch.no_grad():
            # 使用online网络选择动作
            next_action = self.online_net(next_states).argmax(1)
            # 使用target网络评估
            next_q = self.target_net(next_states).gather(1, next_action.unsqueeze(1))
```

### 8.2 Dueling DQN

分离值函数和优势函数：

```python
class DuelingDQN(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super(DuelingDQN, self).__init__()
        
        # 共享特征提取
        self.feature = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU()
        )
        
        # 状态值函数
        self.value = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
        # 优势函数
        self.advantage = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )
        
    def forward(self, x):
        features = self.feature(x)
        value = self.value(features)
        advantage = self.advantage(features)
        
        # Q = V + A - mean(A)
        q = value + advantage - advantage.mean(dim=-1, keepdim=True)
        return q
```

---

## 参考文献

1. Sutton, R. S., & Barto, A. G. (2018). Reinforcement Learning: An Introduction. MIT Press.
2. Mnih, V., et al. (2015). Human-level control through deep reinforcement learning. Nature.
3. Williams, R. J. (1992). Simple statistical gradient-following algorithms for connectionist reinforcement learning. Machine Learning.

---

*本章节持续更新中...*
