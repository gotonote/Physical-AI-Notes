# 模仿学习

模仿学习（Imitation Learning）通过模仿专家行为来学习策略，是机器人技能获取的重要方法。本章介绍行为克隆、逆强化学习、扩散策略等核心技术。

## 目录

- [1. 模仿学习概述](#1-模仿学习概述)
- [2. 行为克隆 (Behavior Cloning)](#2-行为克隆-behavior-cloning)
- [3. 逆强化学习 (IRL)](#3-逆强化学习-irl)
- [4. 扩散策略 (Diffusion Policy)](#4-扩散策略-diffusion-policy)
- [5. DAgger算法](#5-dagger算法)

---

## 1. 模仿学习概述

### 1.1 模仿学习 vs 强化学习

| 方面 | 强化学习 | 模仿学习 |
|------|----------|----------|
| 监督信号 | 稀疏奖励 | 专家演示 |
| 样本效率 | 低 | 高 |
| 探索 | 需要 | 不需要 |
| 收敛性 | 可能不稳定 | 较稳定 |
| 奖励设计 | 需要 | 不需要 |

### 1.2 模仿学习分类

```
┌─────────────────────────────────────────────┐
│               模仿学习方法                  │
├─────────────────────────────────────────────┤
│                                             │
│  ┌───────────────┐                         │
│  │  行为克隆     │  直接监督学习            │
│  │  BC           │  状态→动作映射          │
│  └───────────────┘                         │
│                                             │
│  ┌───────────────┐                         │
│  │  逆强化学习   │  推断奖励函数            │
│  │  IRL          │  然后RL                 │
│  └───────────────┘                         │
│                                             │
│  ┌───────────────┐                         │
│  │  扩散策略     │  条件生成模型            │
│  │  Diffusion    │  多模态策略              │
│  └───────────────┘                         │
│                                             │
│  ┌───────────────┐                         │
│  │  DAgger       │  迭代式专家聚合          │
│  │               │  纠正分布偏移            │
│  └───────────────┘                         │
│                                             │
└─────────────────────────────────────────────┘
```

---

## 2. 行为克隆 (Behavior Cloning)

### 2.1 原理

行为克隆将模仿学习问题视为监督学习问题：

$$
\min_\theta \sum_{(s,a) \in D} \mathcal{L}(\pi_\theta(s), a)
$$

### 2.2 简单实现

```python
import torch
import torch.nn as nn

class BehaviorCloning:
    """
    行为克隆：最简单的模仿学习方法
    """
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        self.policy = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Tanh()
        )
        
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=1e-3)
        
    def forward(self, state):
        """前向传播"""
        return self.policy(state)
    
    def train_step(self, states, actions):
        """单步训练"""
        # 预测动作
        pred_actions = self.policy(states)
        
        # 计算MSE损失
        loss = nn.MSELoss()(pred_actions, actions)
        
        # 反向传播
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return loss.item()
    
    def train(self, demonstrations, epochs=100, batch_size=32):
        """
        训练
        
        demonstrations: {
            'states': [N, state_dim],
            'actions': [N, action_dim]
        }
        """
        dataset = torch.utils.data.TensorDataset(
            torch.FloatTensor(demonstrations['states']),
            torch.FloatTensor(demonstrations['actions'])
        )
        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, shuffle=True
        )
        
        for epoch in range(epochs):
            total_loss = 0
            for states, actions in dataloader:
                loss = self.train_step(states, actions)
                total_loss += loss
                
            if epoch % 10 == 0:
                print(f"Epoch {epoch}, Loss: {total_loss/len(dataloader):.4f}")
```

### 2.3 带正则化的行为克隆

```python
class RegularizedBC:
    """
    带正则化的行为克隆
    - Dropout正则化
    - L2正则化
    - 扰动正则化
    """
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        self.policy = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, action_dim),
            nn.Tanh()
        )
        
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=1e-3, weight_decay=1e-4)
        
    def train_step(self, states, actions, noise_std=0.1):
        """带噪声的对抗训练"""
        # 添加噪声进行对抗训练
        noise = torch.randn_like(actions) * noise_std
        noisy_actions = actions + noise
        
        # 预测
        pred_actions = self.policy(states)
        
        # 标准BC损失
        bc_loss = nn.MSELoss()(pred_actions, actions)
        
        # 对抗损失 (与噪声版本)
        noisy_pred = self.policy(states)
        adv_loss = nn.MSELoss()(noisy_pred, noisy_actions)
        
        # 总损失
        loss = bc_loss + 0.1 * adv_loss
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return loss.item()
```

---

## 3. 逆强化学习 (IRL)

### 3.1 原理

逆强化学习从专家演示中推断奖励函数：

```
正向RL: 奖励函数 r → 策略 π
逆向IRL: 演示 D → 奖励函数 r
```

### 3.2 最大熵逆强化学习

```python
import numpy as np
import torch
import torch.nn as nn

class MaxEntIRL:
    """
    最大熵逆强化学习
    核心思想：奖励函数应该使专家演示的概率最大
    """
    def __init__(self, state_dim, action_dim, feature_dim, lr=0.01):
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        # 奖励函数参数
        self.theta = torch.randn(feature_dim, requires_grad=True)
        
        # 优化器
        self.optimizer = torch.optim.Adam([self.theta], lr=lr)
        
    def reward_function(self, states, actions):
        """
        计算奖励
        R(s,a) = θ · f(s,a)
        """
        features = self.extract_features(states, actions)
        rewards = features @ self.theta
        return rewards
    
    def extract_features(self, states, actions):
        """提取特征"""
        # 简化：使用状态动作拼接作为特征
        features = torch.cat([states, actions], dim=-1)
        return features
    
    def compute_expected_feature(self, policy, env, num_samples=1000):
        """计算策略的期望特征"""
        total_features = 0
        
        for _ in range(num_samples):
            state = env.reset()
            done = False
            
            while not done:
                action = policy(state)
                features = self.extract_features(
                    torch.FloatTensor(state).unsqueeze(0),
                    torch.FloatTensor(action).unsqueeze(0)
                )
                total_features += features.mean(dim=0)
                
                state, reward, done, _ = env.step(action)
                
        return total_features / num_samples
    
    def train(self, expert_demonstrations, env, num_iterations=100):
        """
        训练
        
        expert_demonstrations: 专家演示轨迹列表
        """
        for iteration in range(num_iterations):
            # 计算专家演示的特征期望
            expert_features = self.compute_expected_feature(
                expert_demonstrations
            )
            
            # 计算当前策略的特征期望
            # (这里简化，使用随机策略)
            policy_features = torch.randn(expert_features.shape) * 0.5
            
            # 最大化似然 = 最小化负对数似然
            # 简化的损失函数
            loss = -torch.sum(self.theta * (expert_features - policy_features))
            
            # 优化
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            if iteration % 10 == 0:
                print(f"Iteration {iteration}, Loss: {loss.item()}")
```

### 3.3 生成对抗逆强化学习 (GAIL)

```python
class GAIL:
    """
    生成对抗逆强化学习
    使用GAN思想：判别器区分专家和学到的策略
    """
    def __init__(self, state_dim, action_dim, hidden_dim=128):
        # 策略网络
        self.policy = nn.Sequential(
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
        
        self.policy_optimizer = torch.optim.Adam(self.policy.parameters(), lr=3e-4)
        self.discriminator_optimizer = torch.optim.Adam(
            self.discriminator.parameters(), lr=3e-4
        )
        
    def train_step(self, expert_states, expert_actions, policy_states, policy_actions):
        """单步训练"""
        # ===== 训练判别器 =====
        # 专家数据标签为1，策略数据标签为0
        expert_input = torch.cat([expert_states, expert_actions], dim=-1)
        policy_input = torch.cat([policy_states, policy_actions], dim=-1)
        
        expert_pred = self.discriminator(expert_input)
        policy_pred = self.discriminator(policy_input)
        
        # 判别器损失
        disc_loss = -torch.log(expert_pred + 1e-8).mean() - \
                    torch.log(1 - policy_pred + 1e-8).mean()
        
        self.discriminator_optimizer.zero_grad()
        disc_loss.backward()
        self.discriminator_optimizer.step()
        
        # ===== 训练策略 =====
        # 使用判别器作为奖励
        policy_input_new = torch.cat([policy_states, policy_actions], dim=-1)
        reward = -torch.log(self.discriminator(policy_input_new) + 1e-8)
        
        # 策略梯度 (简化版)
        policy_loss = -reward.mean()
        
        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()
        
        return disc_loss.item(), policy_loss.item()
```

---

## 4. 扩散策略 (Diffusion Policy)

### 4.1 扩散模型基础

```python
import torch
import torch.nn as nn
import numpy as np

class DiffusionModel(nn.Module):
    """
    去噪扩散概率模型 (DDPM)
    """
    def __init__(self, state_dim, action_dim, hidden_dim=256, num_steps=100):
        super(DiffusionModel, self).__init__()
        
        self.num_steps = num_steps
        self.action_dim = action_dim
        
        # 时间嵌入
        self.time_mlp = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # 噪声预测网络
        self.noise_predictor = nn.Sequential(
            nn.Linear(action_dim + state_dim + hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, action_dim)
        )
        
    def forward(self, x_t, t, condition):
        """
        前向传播：预测噪声
        x_t: 当前时刻的action
        t: 时间步
        condition: 条件信息（状态）
        """
        # 时间嵌入
        t_embedding = self.time_mlp(t.unsqueeze(-1))
        
        # 拼接
        input_feat = torch.cat([x_t, condition, t_embedding], dim=-1)
        
        # 预测噪声
        noise = self.noise_predictor(input_feat)
        
        return noise
    
    def forward_diffusion(self, x_0, num_steps=None):
        """
        前向过程：添加噪声
        q(x_t | x_0) = N(x_t; sqrt(1-β_t)*x_0, β_t*I)
        """
        if num_steps is None:
            num_steps = self.num_steps
            
        batch_size = x_0.shape[0]
        x_t = x_0
        
        # 噪声调度
        betas = self.get_noise_schedule(num_steps)
        
        for t in range(num_steps):
            noise = torch.randn_like(x_0)
            x_t = torch.sqrt(1 - betas[t]) * x_t + torch.sqrt(betas[t]) * noise
            
        return x_t
    
    def get_noise_schedule(self, num_steps):
        """噪声调度"""
        betas = torch.linspace(0.0001, 0.02, num_steps)
        return betas
    
    @torch.no_grad()
    def sample(self, state, num_steps=None):
        """
        采样：从噪声逐步去噪生成action
        """
        if num_steps is None:
            num_steps = self.num_steps
            
        # 从随机噪声开始
        x_t = torch.randn(state.shape[0], self.action_dim).to(state.device)
        
        # 噪声调度
        betas = self.get_noise_schedule(num_steps).to(state.device)
        alphas = 1 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        
        # 逐步去噪
        for t in reversed(range(num_steps)):
            # 预测噪声
            t_input = torch.ones(state.shape[0]).to(state.device) * t
            predicted_noise = self.forward(x_t, t_input, state)
            
            # 去噪
            if t > 0:
                noise = torch.randn_like(x_t)
            else:
                noise = torch.zeros_like(x_t)
                
            x_t = (x_t - torch.sqrt(1 - betas[t]) * predicted_noise) / torch.sqrt(betas[t])
            x_t = x_t + torch.sqrt(betas[t]) * noise
            
        return x_t
```

### 4.2 扩散策略实现

```python
class DiffusionPolicy(nn.Module):
    """
    扩散策略：使用扩散模型作为策略
    """
    def __init__(self, state_dim, action_dim, hidden_dim=256, num_diffusion_steps=100):
        super(DiffusionPolicy, self).__init__()
        
        self.state_encoder = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        self.diffusion = DiffusionModel(
            state_dim=hidden_dim,
            action_dim=action_dim,
            hidden_dim=hidden_dim,
            num_steps=num_diffusion_steps
        )
        
    def forward(self, state, training=True):
        """前向传播"""
        # 编码状态
        encoded_state = self.state_encoder(state)
        
        if training:
            # 训练：预测噪声
            # 采样时间步
            t = torch.randint(0, self.diffusion.num_steps, (state.shape[0],))
            
            # 采样随机action
            action = torch.randn(state.shape[0], self.action_dim).to(state.device)
            
            # 预测噪声
            predicted_noise = self.diffusion.forward(action, t, encoded_state)
            
            return predicted_noise
        else:
            # 推理：采样生成action
            action = self.diffusion.sample(encoded_state)
            return action
    
    def loss(self, state, action):
        """扩散策略损失"""
        predicted = self.forward(state, training=True)
        
        # MSE损失
        loss = nn.MSELoss()(predicted, action)  # action实际上是噪声
        return loss
```

---

## 5. DAgger算法

### 5.1 原理

DAgger (Dataset Aggregation) 解决行为克隆的分布偏移问题：

```
BC问题: 训练分布 ≠ 测试分布

DAgger:
1. 使用专家策略收集演示
2. 训练策略 π
3. 使用 π 收集状态
4. 专家标注这些状态
5. 聚合数据，返回步骤2
```

### 5.2 实现

```python
class DAgger:
    """
    DAgger (Dataset Aggregation) 实现
    """
    def __init__(self, policy, expert_policy, env):
        self.policy = policy
        self.expert_policy = expert_policy
        self.env = env
        
    def collect_demonstrations(self, policy, num_episodes=10):
        """使用策略收集演示"""
        demonstrations = []
        
        for _ in range(num_episodes):
            state = self.env.reset()
            done = False
            
            trajectory = []
            
            while not done:
                # 使用给定策略
                action = policy.select_action(state)
                
                # 记录状态
                trajectory.append({'state': state, 'action': action})
                
                state, reward, done, _ = self.env.step(action)
                
            demonstrations.append(trajectory)
            
        return demonstrations
    
    def collect_expert_annotations(self, trajectories):
        """收集专家对策略状态的标注"""
        annotated_data = {'states': [], 'actions': []}
        
        for traj in trajectories:
            for step in traj:
                state = step['state']
                
                # 专家标注
                expert_action = self.expert_policy.select_action(state)
                
                annotated_data['states'].append(state)
                annotated_data['actions'].append(expert_action)
                
        return annotated_data
    
    def train(self, num_iterations=10, num_demos_per_iter=10):
        """DAgger训练"""
        # 初始专家演示
        expert_demos = self.collect_demonstrations(self.expert_policy, num_demos_per_iter)
        aggregated_data = self.collect_expert_annotations(experiments)
        
        for iteration in range(num_iterations):
            # 训练策略
            self.policy.train(aggregated_data, epochs=10)
            
            # 使用当前策略收集数据
            policy_demos = self.collect_demonstrations(self.policy, num_demos_per_iter)
            
            # 专家标注
            new_data = self.collect_expert_annotations(policy_demos)
            
            # 聚合数据
            aggregated_data['states'].extend(new_data['states'])
            aggregated_data['actions'].extend(new_data['actions'])
            
            print(f"Iteration {iteration}, Data size: {len(aggregated_data['states'])}")
```

---

## 参考文献

1. Abbeel, P., & Ng, A. Y. (2004). Apprenticeship learning via inverse reinforcement learning. ICML.
2. Ross, S., Gordon, G., & Bagnell, D. (2011). A reduction of imitation learning to no-regret online learning. AISTATS.
3. Ho, J., & Ermon, S. (2016). Generative adversarial imitation learning. NeurIPS.
4. Chi, C., et al. (2023). Diffusion Policy: Visuomotor Policy Learning via Action Diffusion. RSS.

---

*本章节持续更新中...*
