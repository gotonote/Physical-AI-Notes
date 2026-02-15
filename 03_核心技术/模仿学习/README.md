# 模仿学习

> Imitation Learning for Robotics

模仿学习让机器人从人类示范中学习行为策略，是物理AI领域最实用的学习方法之一。

---

## 📋 目录

1. [模仿学习概述](#1-模仿学习概述)
2. [行为克隆](#2-行为克隆)
3. [DAgger算法](#3-dagger算法)
4. [GAIL](#4-gail)
5. [代码实现](#5-代码实现)
6. [实战案例](#6-实战案例)

---

## 1. 模仿学习概述

### 1.1 核心思想

```
┌─────────────────────────────────────────────────────────────┐
│                    模仿学习框架                              │
│                                                             │
│   ┌──────────┐        ┌──────────┐        ┌──────────┐     │
│   │  人类专家  │ ───▶  │  示范数据  │ ───▶  │  学习策略  │     │
│   │  Expert   │        │ Demo Data │        │  Policy  │     │
│   └──────────┘        └──────────┘        └──────────┘     │
│                                                             │
│   示范数据 = {(s₁,a₁), (s₂,a₂), ..., (sₙ,aₙ)}              │
│                                                             │
│   目标：学习策略 π(a|s) 使其行为接近专家                      │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 1.2 与强化学习的对比

| 特性 | 强化学习 | 模仿学习 |
|------|---------|---------|
| 学习信号 | 奖励函数 | 专家示范 |
| 探索需求 | 需要探索 | 无需探索 |
| 样本效率 | 较低 | 较高 |
| 最优性 | 可能超越专家 | 受限于专家 |
| 安全性 | 风险较高 | 相对安全 |

### 1.3 应用场景

```
应用领域：
├── 机械臂操作
│   ├─ 抓取与放置
│   ├─ 装配任务
│   └─ 精细操作（焊接、涂装）
├── 移动机器人
│   ├─ 室内导航
│   └─ 路径跟随
├── 自动驾驶
│   ├─ 车道保持
│   └─ 停车
└── 机器人导航
    └─ 复杂环境穿越
```

---

## 2. 行为克隆

### 2.1 基本原理

**Behavior Cloning (BC)** 将模仿学习转化为监督学习问题：

$$\min_\theta \sum_{(s,a) \in D} L(\pi_\theta(s), a)$$

其中：
- $D$ 是专家示范数据集
- $L$ 是损失函数（如MSE或交叉熵）

### 2.2 架构图

```
┌─────────────────────────────────────────────────────────────┐
│                   行为克隆架构                               │
│                                                             │
│   输入：状态s                                                │
│   ┌─────────────────────────────────────────────────────┐   │
│   │                                                     │   │
│   │  ┌─────────┐   ┌─────────┐   ┌─────────┐          │   │
│   │  │ 图像/   │   │ 特征    │   │ MLP/    │          │   │
│   │  │ 传感器 │──▶│ 提取器  │──▶│ Transformer│         │   │
│   │  │ 编码器  │   │         │   │         │          │   │
│   │  └─────────┘   └─────────┘   └────┬────┘          │   │
│   │                                   │                │   │
│   │                    ┌──────────────┼──────────────┐ │   │
│   │                    ▼              ▼              ▼ │   │
│   │              ┌──────────┐  ┌──────────┐  ┌──────────┐│   │
│   │              │位置/速度 │  │ 夹爪控制 │  │ 其他动作 ││   │
│   │              │  输出    │  │  输出    │  │  输出    ││   │
│   │              └──────────┘  └──────────┘  └──────────┘│   │
│   │                                                     │   │
│   └─────────────────────────────────────────────────────┘   │
│                                                             │
│   损失：L = MSE(π(s), a_expert)                             │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 2.3 代码实现

```python
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import List, Tuple, Dict
from torch.utils.data import Dataset, DataLoader

class BehaviorCloningPolicy(nn.Module):
    """行为克隆策略网络"""
    
    def __init__(self, 
                 state_dim: int, 
                 action_dim: int,
                 hidden_dims: List[int] = [256, 256],
                 use_visual: bool = False):
        super().__init__()
        
        self.use_visual = use_visual
        
        if use_visual:
            # 视觉编码器
            self.visual_encoder = nn.Sequential(
                nn.Conv2d(3, 32, 8, stride=4),
                nn.ReLU(),
                nn.Conv2d(32, 64, 4, stride=2),
                nn.ReLU(),
                nn.Conv2d(64, 64, 3, stride=1),
                nn.ReLU(),
                nn.Flatten(),
                nn.Linear(64 * 7 * 7, 512),
                nn.ReLU()
            )
            state_dim = 512 + state_dim
        
        # MLP策略网络
        layers = []
        prev_dim = state_dim
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1)
            ])
            prev_dim = hidden_dim
        
        self.backbone = nn.Sequential(*layers)
        
        # 动作输出头
        self.action_mean = nn.Linear(prev_dim, action_dim)
        self.action_log_std = nn.Parameter(torch.zeros(action_dim))
    
    def forward(self, state: torch.Tensor, image: torch.Tensor = None):
        if self.use_visual and image is not None:
            visual_features = self.visual_encoder(image)
            state = torch.cat([state, visual_features], dim=-1)
        
        features = self.backbone(state)
        action_mean = self.action_mean(features)
        action_std = torch.exp(self.action_log_std.clamp(-20, 2))
        
        return action_mean, action_std
    
    def get_action(self, state: np.ndarray, image: np.ndarray = None, 
                   deterministic: bool = False) -> np.ndarray:
        """获取动作"""
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            image_tensor = torch.FloatTensor(image).unsqueeze(0).permute(0, 3, 1, 2) if image is not None else None
            
            mean, std = self.forward(state_tensor, image_tensor)
            
            if deterministic:
                action = mean
            else:
                dist = torch.distributions.Normal(mean, std)
                action = dist.sample()
        
        return action.squeeze().numpy()


class DemonstrationDataset(Dataset):
    """示范数据集"""
    
    def __init__(self, demonstrations: List[Dict]):
        """
        demonstrations: 示范数据列表
        每个元素是 {'states': np.array, 'actions': np.array, 'images': np.array (可选)}
        """
        self.states = []
        self.actions = []
        self.images = []
        
        for demo in demonstrations:
            self.states.extend(demo['states'])
            self.actions.extend(demo['actions'])
            if 'images' in demo:
                self.images.extend(demo['images'])
        
        self.states = np.array(self.states)
        self.actions = np.array(self.actions)
        self.has_images = len(self.images) > 0
        
        if self.has_images:
            self.images = np.array(self.images)
    
    def __len__(self):
        return len(self.states)
    
    def __getitem__(self, idx):
        state = self.states[idx]
        action = self.actions[idx]
        
        if self.has_images:
            image = self.images[idx]
            return state, action, image
        
        return state, action, None


class BehaviorCloningTrainer:
    """行为克隆训练器"""
    
    def __init__(self, 
                 policy: BehaviorCloningPolicy,
                 lr: float = 1e-4,
                 weight_decay: float = 1e-5,
                 device: str = 'cuda'):
        
        self.policy = policy.to(device)
        self.device = device
        self.optimizer = optim.AdamW(
            policy.parameters(),
            lr=lr,
            weight_decay=weight_decay
        )
        
        # 学习率调度
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=1000, eta_min=1e-6
        )
    
    def train_epoch(self, dataloader: DataLoader) -> Dict[str, float]:
        """训练一个epoch"""
        self.policy.train()
        total_loss = 0
        total_mse = 0
        num_batches = 0
        
        for batch in dataloader:
            if len(batch) == 3:
                states, actions, images = batch
                images = images.to(self.device) if images[0] is not None else None
            else:
                states, actions = batch
                images = None
            
            states = states.to(self.device)
            actions = actions.to(self.device)
            
            # 前向传播
            pred_mean, pred_std = self.policy(states, images)
            
            # 计算损失
            # MSE损失
            mse_loss = nn.MSELoss()(pred_mean, actions)
            
            # 负对数似然损失（可选）
            dist = torch.distributions.Normal(pred_mean, pred_std)
            nll_loss = -dist.log_prob(actions).mean()
            
            # 总损失
            loss = mse_loss + 0.1 * nll_loss
            
            # 反向传播
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 1.0)
            self.optimizer.step()
            
            total_loss += loss.item()
            total_mse += mse_loss.item()
            num_batches += 1
        
        self.scheduler.step()
        
        return {
            'loss': total_loss / num_batches,
            'mse': total_mse / num_batches,
            'lr': self.scheduler.get_last_lr()[0]
        }
    
    def evaluate(self, dataloader: DataLoader) -> Dict[str, float]:
        """评估模型"""
        self.policy.eval()
        total_loss = 0
        num_batches = 0
        
        with torch.no_grad():
            for batch in dataloader:
                if len(batch) == 3:
                    states, actions, images = batch
                    images = images.to(self.device) if images[0] is not None else None
                else:
                    states, actions = batch
                    images = None
                
                states = states.to(self.device)
                actions = actions.to(self.device)
                
                pred_mean, _ = self.policy(states, images)
                loss = nn.MSELoss()(pred_mean, actions)
                
                total_loss += loss.item()
                num_batches += 1
        
        return {'eval_loss': total_loss / num_batches}


def train_behavior_cloning(demonstrations: List[Dict],
                          state_dim: int,
                          action_dim: int,
                          num_epochs: int = 100,
                          batch_size: int = 64,
                          use_visual: bool = False):
    """训练行为克隆模型"""
    
    # 创建数据集
    dataset = DemonstrationDataset(demonstrations)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    
    # 创建模型
    policy = BehaviorCloningPolicy(state_dim, action_dim, use_visual=use_visual)
    
    # 创建训练器
    trainer = BehaviorCloningTrainer(policy)
    
    # 训练
    for epoch in range(num_epochs):
        metrics = trainer.train_epoch(dataloader)
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{num_epochs}")
            print(f"  Loss: {metrics['loss']:.4f}, MSE: {metrics['mse']:.4f}")
    
    return policy
```

### 2.4 BC的问题

**分布偏移（Distribution Shift）**：

```
问题：训练时状态分布 ≠ 测试时状态分布

训练：s ~ D_expert (专家状态分布)
测试：s ~ D_π (策略状态分布)

如果π犯错 → 到达新状态 → 继续犯错 → 累积误差

        专家轨迹                    学习策略轨迹
           │                           │
           ▼                           ▼
    ┌──────────────┐            ┌──────────────┐
    │  s₁ → s₂ → s₃ │            │  s₁ → s₂'→ ???│
    │    ↘   ↙     │            │      ↘       │
    │     成功      │            │      失败    │
    └──────────────┘            └──────────────┘
```

---

## 3. DAgger算法

### 3.1 核心思想

**Dataset Aggregation (DAgger)** 通过迭代收集数据来解决分布偏移问题：

```
┌─────────────────────────────────────────────────────────────┐
│                     DAgger流程                              │
│                                                             │
│   初始化：D = 专家数据                                       │
│                                                             │
│   for i = 1 to N:                                          │
│       1. 训练策略 πᵢ 在数据集D上                            │
│       2. 执行πᵢ，收集状态序列 s₁, s₂, ..., sₜ              │
│       3. 请专家标注动作 a₁, a₂, ..., aₜ                    │
│       4. 将新数据加入D                                      │
│       5. D = D ∪ {(sⱼ, aⱼ)}                                │
│                                                             │
│   返回最终策略 πₙ                                           │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 3.2 代码实现

```python
from typing import Callable, Tuple
import copy

class DAggerTrainer:
    """DAgger训练器"""
    
    def __init__(self,
                 policy: BehaviorCloningPolicy,
                 expert_policy: Callable,
                 env,
                 beta_schedule: Callable = None,
                 lr: float = 1e-4,
                 device: str = 'cuda'):
        
        self.policy = policy.to(device)
        self.expert_policy = expert_policy
        self.env = env
        self.device = device
        
        # β调度：混合专家和学习的策略
        if beta_schedule is None:
            # 默认：线性衰减
            self.beta_schedule = lambda epoch: max(0.0, 1.0 - epoch / 20)
        else:
            self.beta_schedule = beta_schedule
        
        self.bc_trainer = BehaviorCloningTrainer(policy, lr=lr)
        
        # 存储所有数据
        self.all_demonstrations = []
    
    def collect_data_with_policy(self, 
                                  num_episodes: int = 10,
                                  beta: float = 0.0) -> List[Dict]:
        """使用当前策略收集数据，并用专家标注"""
        demonstrations = []
        
        for ep in range(num_episodes):
            states = []
            expert_actions = []
            
            state, _ = self.env.reset()
            done = False
            
            while not done:
                # 获取专家动作
                expert_action = self.expert_policy(state)
                
                # 获取学习策略动作
                with torch.no_grad():
                    state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                    learned_action, _ = self.policy(state_tensor)
                    learned_action = learned_action.squeeze().cpu().numpy()
                
                # 混合动作
                action = beta * expert_action + (1 - beta) * learned_action
                
                # 执行动作
                next_state, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated
                
                # 存储状态和专家标注
                states.append(state)
                expert_actions.append(expert_action)
                
                state = next_state
            
            demonstrations.append({
                'states': states,
                'actions': expert_actions
            })
        
        return demonstrations
    
    def train(self,
              initial_demos: List[Dict],
              num_iterations: int = 20,
              episodes_per_iter: int = 10,
              bc_epochs_per_iter: int = 50,
              batch_size: int = 64) -> BehaviorCloningPolicy:
        """DAgger训练"""
        
        # 初始化数据集
        self.all_demonstrations = copy.deepcopy(initial_demos)
        
        best_policy = None
        best_loss = float('inf')
        
        for iteration in range(num_iterations):
            beta = self.beta_schedule(iteration)
            
            print(f"\n=== Iteration {iteration + 1}/{num_iterations} ===")
            print(f"Beta (expert mixing): {beta:.2f}")
            
            # 1. 收集新数据
            print("Collecting data...")
            new_demos = self.collect_data_with_policy(
                num_episodes=episodes_per_iter,
                beta=beta
            )
            
            # 2. 加入数据集
            self.all_demonstrations.extend(new_demos)
            print(f"Total demonstrations: {len(self.all_demonstrations)}")
            
            # 3. 训练
            print("Training policy...")
            dataset = DemonstrationDataset(self.all_demonstrations)
            dataloader = DataLoader(
                dataset, 
                batch_size=batch_size,
                shuffle=True,
                num_workers=4
            )
            
            for epoch in range(bc_epochs_per_iter):
                metrics = self.bc_trainer.train_epoch(dataloader)
                
                if (epoch + 1) % 10 == 0:
                    print(f"  Epoch {epoch+1}: Loss={metrics['loss']:.4f}")
            
            # 4. 评估并保存最佳策略
            eval_metrics = self.bc_trainer.evaluate(dataloader)
            if eval_metrics['eval_loss'] < best_loss:
                best_loss = eval_metrics['eval_loss']
                best_policy = copy.deepcopy(self.policy)
                print(f"  New best loss: {best_loss:.4f}")
        
        return best_policy


# 示例：专家策略
class ExpertPolicy:
    """示例专家策略（实际应用中替换为真实专家）"""
    
    def __init__(self, env):
        self.env = env
    
    def __call__(self, state: np.ndarray) -> np.ndarray:
        """
        根据状态返回专家动作
        实际应用中可能是：
        - 预训练的高性能策略
        - 人类遥操作
        - 规划算法
        """
        # 简单示例：PD控制器
        target_pos = self.env.target_pos
        current_pos = state[:3]
        
        action = (target_pos - current_pos) * 2.0  # 简单比例控制
        action = np.clip(action, -1, 1)
        
        return action
```

---

## 4. GAIL

### 4.1 核心思想

**Generative Adversarial Imitation Learning** 使用生成对抗网络的思想进行模仿学习：

```
┌─────────────────────────────────────────────────────────────┐
│                      GAIL架构                               │
│                                                             │
│   ┌──────────────┐                    ┌──────────────┐      │
│   │   专家数据    │                    │   生成器G     │      │
│   │   π_E        │                    │  (策略π_θ)   │      │
│   │              │                    │              │      │
│   │ τ_E ~ π_E    │                    │ τ ~ π_θ      │      │
│   └───────┬──────┘                    └───────┬──────┘      │
│           │                                   │             │
│           │     状态-动作对 (s,a)             │             │
│           │                                   │             │
│           └──────────────┬────────────────────┘             │
│                          │                                  │
│                          ▼                                  │
│                 ┌──────────────┐                            │
│                 │   判别器D     │                            │
│                 │ D(s,a) → [0,1]│                           │
│                 └──────────────┘                            │
│                          │                                  │
│                          ▼                                  │
│              D(s,a) ≈ 1: 来自专家                          │
│              D(s,a) ≈ 0: 来自生成器                         │
│                                                             │
│   训练目标：                                                 │
│   - 判别器：区分专家和生成器轨迹                             │
│   - 生成器：欺骗判别器（生成类似专家的轨迹）                  │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 4.2 数学形式

**判别器目标**：
$$\max_D \mathbb{E}_{(s,a)\sim\pi_E}[\log D(s,a)] + \mathbb{E}_{(s,a)\sim\pi_\theta}[\log(1-D(s,a))]$$

**生成器（策略）目标**：
$$\min_\theta \mathbb{E}_{(s,a)\sim\pi_\theta}[\log(1-D(s,a))] - \lambda H(\pi_\theta)$$

其中 $H(\pi_\theta)$ 是策略熵（鼓励探索）。

### 4.3 代码实现

```python
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import List, Dict

class Discriminator(nn.Module):
    """GAIL判别器"""
    
    def __init__(self, 
                 state_dim: int, 
                 action_dim: int,
                 hidden_dim: int = 256):
        super().__init__()
        
        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
    
    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """输出概率 [0, 1]"""
        x = torch.cat([state, action], dim=-1)
        return self.net(x)


class GAILAgent:
    """GAIL智能体"""
    
    def __init__(self,
                 state_dim: int,
                 action_dim: int,
                 expert_data: List[Dict],
                 lr_policy: float = 3e-4,
                 lr_disc: float = 1e-4,
                 gamma: float = 0.99,
                 gae_lambda: float = 0.95,
                 entropy_coef: float = 0.01,
                 device: str = 'cuda'):
        
        self.device = device
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.entropy_coef = entropy_coef
        
        # 策略网络
        self.policy = PolicyNetwork(state_dim, action_dim).to(device)
        self.critic = CriticNetwork(state_dim).to(device)
        
        # 判别器
        self.discriminator = Discriminator(state_dim, action_dim).to(device)
        
        # 优化器
        self.policy_optimizer = optim.Adam(
            list(self.policy.parameters()) + list(self.critic.parameters()),
            lr=lr_policy
        )
        self.disc_optimizer = optim.Adam(
            self.discriminator.parameters(),
            lr=lr_disc
        )
        
        # 预处理专家数据
        self.expert_states = torch.FloatTensor(
            np.concatenate([d['states'] for d in expert_data])
        ).to(device)
        self.expert_actions = torch.FloatTensor(
            np.concatenate([d['actions'] for d in expert_data])
        ).to(device)
        
        # 经验缓冲区
        self.replay_buffer = []
    
    def collect_trajectories(self, env, num_steps: int = 2048):
        """收集轨迹"""
        self.policy.eval()
        
        states, actions, rewards, dones, log_probs, values = [], [], [], [], [], []
        
        state, _ = env.reset()
        
        for _ in range(num_steps):
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                mean, std = self.policy(state_tensor)
                dist = torch.distributions.Normal(mean, std)
                action = dist.sample()
                log_prob = dist.log_prob(action).sum(dim=-1)
                value = self.critic(state_tensor)
            
            action_np = action.squeeze().cpu().numpy()
            next_state, _, terminated, truncated, _ = env.step(action_np)
            done = terminated or truncated
            
            states.append(state)
            actions.append(action_np)
            dones.append(done)
            log_probs.append(log_prob.item())
            values.append(value.item())
            
            state = next_state if not done else env.reset()[0]
        
        return {
            'states': np.array(states),
            'actions': np.array(actions),
            'dones': np.array(dones),
            'log_probs': np.array(log_probs),
            'values': np.array(values)
        }
    
    def train_discriminator(self, agent_data: Dict, batch_size: int = 256) -> float:
        """训练判别器"""
        self.discriminator.train()
        
        agent_states = torch.FloatTensor(agent_data['states']).to(self.device)
        agent_actions = torch.FloatTensor(agent_data['actions']).to(self.device)
        
        num_samples = len(agent_states)
        indices = np.random.permutation(num_samples)
        
        total_loss = 0
        num_batches = 0
        
        for start in range(0, num_samples, batch_size):
            end = start + batch_size
            idx = indices[start:end]
            
            # Agent数据
            agent_s = agent_states[idx]
            agent_a = agent_actions[idx]
            
            # 随机采样专家数据
            expert_idx = np.random.randint(0, len(self.expert_states), len(idx))
            expert_s = self.expert_states[expert_idx]
            expert_a = self.expert_actions[expert_idx]
            
            # 判别器预测
            expert_pred = self.discriminator(expert_s, expert_a)
            agent_pred = self.discriminator(agent_s, agent_a)
            
            # 损失
            expert_loss = -torch.log(expert_pred + 1e-8).mean()
            agent_loss = -torch.log(1 - agent_pred + 1e-8).mean()
            loss = expert_loss + agent_loss
            
            self.disc_optimizer.zero_grad()
            loss.backward()
            self.disc_optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
        
        return total_loss / num_batches
    
    def compute_gail_reward(self, states: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """计算GAIL奖励"""
        with torch.no_grad():
            d = self.discriminator(states, actions)
            # 奖励 = log(D) - log(1-D) = logit(D)
            # 或者简化为 -log(1-D) 来鼓励欺骗判别器
            reward = -torch.log(1 - d + 1e-8)
        return reward.squeeze()
    
    def train_policy(self, agent_data: Dict, epochs: int = 10, batch_size: int = 64) -> float:
        """训练策略（使用PPO）"""
        self.policy.train()
        self.critic.train()
        
        states = torch.FloatTensor(agent_data['states']).to(self.device)
        actions = torch.FloatTensor(agent_data['actions']).to(self.device)
        old_log_probs = torch.FloatTensor(agent_data['log_probs']).to(self.device)
        values = agent_data['values']
        dones = agent_data['dones']
        
        # 计算GAIL奖励
        gail_rewards = self.compute_gail_reward(states, actions)
        
        # 计算GAE
        advantages = []
        gae = 0
        for t in reversed(range(len(gail_rewards))):
            if t == len(gail_rewards) - 1:
                next_value = 0
            else:
                next_value = values[t + 1]
            
            delta = gail_rewards[t] + self.gamma * next_value * (1 - dones[t]) - values[t]
            gae = delta + self.gamma * self.gae_lambda * (1 - dones[t]) * gae
            advantages.insert(0, gae)
        
        advantages = torch.FloatTensor(advantages).to(self.device)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        returns = advantages + torch.FloatTensor(values).to(self.device)
        
        total_loss = 0
        
        for _ in range(epochs):
            indices = np.random.permutation(len(states))
            
            for start in range(0, len(states), batch_size):
                idx = indices[start:start + batch_size]
                
                batch_states = states[idx]
                batch_actions = actions[idx]
                batch_old_log_probs = old_log_probs[idx]
                batch_advantages = advantages[idx]
                batch_returns = returns[idx]
                
                # 计算新log_prob
                mean, std = self.policy(batch_states)
                dist = torch.distributions.Normal(mean, std)
                new_log_probs = dist.log_prob(batch_actions).sum(dim=-1)
                entropy = dist.entropy().sum(dim=-1).mean()
                
                # PPO损失
                ratio = torch.exp(new_log_probs - batch_old_log_probs)
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 0.8, 1.2) * batch_advantages
                policy_loss = -torch.min(surr1, surr2).mean()
                
                # 价值损失
                value_pred = self.critic(batch_states).squeeze()
                value_loss = nn.MSELoss()(value_pred, batch_returns)
                
                # 总损失
                loss = policy_loss + 0.5 * value_loss - self.entropy_coef * entropy
                
                self.policy_optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    list(self.policy.parameters()) + list(self.critic.parameters()),
                    0.5
                )
                self.policy_optimizer.step()
                
                total_loss += loss.item()
        
        return total_loss / (epochs * (len(states) // batch_size + 1))
    
    def train(self, env, num_iterations: int = 100, steps_per_iter: int = 2048):
        """完整训练"""
        for iteration in range(num_iterations):
            # 收集轨迹
            agent_data = self.collect_trajectories(env, steps_per_iter)
            
            # 训练判别器
            disc_loss = self.train_discriminator(agent_data)
            
            # 训练策略
            policy_loss = self.train_policy(agent_data)
            
            if (iteration + 1) % 10 == 0:
                print(f"Iteration {iteration + 1}")
                print(f"  Disc Loss: {disc_loss:.4f}")
                print(f"  Policy Loss: {policy_loss:.4f}")
```

---

## 5. 代码实现

### 完整示例：机械臂抓取

```python
import gymnasium as gym
import numpy as np
from tqdm import tqdm

# 假设我们有一个机械臂环境
class RobotArmEnv:
    """简化的机械臂环境"""
    
    def __init__(self):
        self.state_dim = 10  # 关节角度 + 目标位置
        self.action_dim = 7  # 7个关节
        
        self.reset()
    
    def reset(self):
        self.joint_pos = np.zeros(7)
        self.target_pos = np.random.uniform(-0.5, 0.5, 3)
        return self._get_state()
    
    def _get_state(self):
        # 简化的正向运动学
        ee_pos = self.joint_pos[:3]  # 简化
        return np.concatenate([self.joint_pos, ee_pos, self.target_pos])
    
    def step(self, action):
        self.joint_pos = np.clip(self.joint_pos + action * 0.1, -1, 1)
        
        ee_pos = self.joint_pos[:3]
        distance = np.linalg.norm(ee_pos - self.target_pos)
        
        reward = -distance
        done = distance < 0.05
        
        if done:
            reward += 10
        
        return self._get_state(), reward, done, {}


def collect_expert_demonstrations(env, num_demos: int = 50, demo_length: int = 100):
    """收集专家示范（实际中使用遥操作或规划算法）"""
    demonstrations = []
    
    for _ in range(num_demos):
        states = []
        actions = []
        
        state = env.reset()
        
        for t in range(demo_length):
            # 简单专家：向目标移动
            target = state[-3:]
            current = state[7:10]
            action = (target - current) * 0.5  # 简单PD控制
            action = np.clip(action, -1, 1)
            action = np.concatenate([action, np.zeros(4)])  # 7维动作
            
            states.append(state)
            actions.append(action)
            
            state, _, done, _ = env.step(action)
            
            if done:
                break
        
        demonstrations.append({
            'states': np.array(states),
            'actions': np.array(actions)
        })
    
    return demonstrations


def run_behavior_cloning():
    """运行行为克隆"""
    env = RobotArmEnv()
    
    # 收集专家示范
    print("Collecting expert demonstrations...")
    demos = collect_expert_demonstrations(env, num_demos=100)
    
    # 训练BC
    print("Training behavior cloning...")
    policy = train_behavior_cloning(
        demos,
        state_dim=env.state_dim,
        action_dim=env.action_dim,
        num_epochs=200
    )
    
    # 测试
    print("Testing policy...")
    test_rewards = []
    for _ in range(10):
        state = env.reset()
        total_reward = 0
        done = False
        
        while not done:
            action = policy.get_action(state, deterministic=True)
            state, reward, done, _ = env.step(action)
            total_reward += reward
        
        test_rewards.append(total_reward)
    
    print(f"Average test reward: {np.mean(test_rewards):.2f}")


def run_dagger():
    """运行DAgger"""
    env = RobotArmEnv()
    
    # 初始专家数据
    print("Collecting initial demonstrations...")
    initial_demos = collect_expert_demonstrations(env, num_demos=20)
    
    # 创建专家策略
    def expert_policy(state):
        target = state[-3:]
        current = state[7:10]
        action = (target - current) * 0.5
        return np.concatenate([np.clip(action, -1, 1), np.zeros(4)])
    
    # 创建DAgger训练器
    policy = BehaviorCloningPolicy(env.state_dim, env.action_dim)
    trainer = DAggerTrainer(policy, expert_policy, env)
    
    # 训练
    print("Training with DAgger...")
    final_policy = trainer.train(
        initial_demos,
        num_iterations=30,
        episodes_per_iter=5,
        bc_epochs_per_iter=50
    )
    
    return final_policy


if __name__ == "__main__":
    print("=== Behavior Cloning ===")
    run_behavior_cloning()
    
    print("\n=== DAgger ===")
    run_dagger()
```

---

## 6. 实战案例

### 案例1：LeRobot框架使用

```python
# 使用HuggingFace LeRobot进行模仿学习
from lerobot.common.policies.act.modeling_act import ACTPolicy
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

# 加载数据集
dataset = LeRobotDataset("pusht")

# 创建ACT策略
policy = ACTPolicy(
    input_shapes=dataset.meta.observation_shapes,
    output_shapes=dataset.meta.action_shapes,
    chunk_size=100,
    n_obs_steps=2,
)

# 训练
from lerobot.common.train import train_policy

train_policy(
    policy=policy,
    dataset=dataset,
    num_epochs=100,
    batch_size=8,
    learning_rate=1e-4,
)
```

### 案例2：Diffusion Policy

```python
# Diffusion Policy实现
class DiffusionPolicy(nn.Module):
    """扩散策略"""
    
    def __init__(self, 
                 state_dim: int,
                 action_dim: int,
                 horizon: int = 16,
                 num_diffusion_steps: int = 100):
        super().__init__()
        
        self.horizon = horizon
        self.action_dim = action_dim
        self.num_diffusion_steps = num_diffusion_steps
        
        # 噪声预测网络
        self.noise_net = nn.Sequential(
            nn.Linear(state_dim + action_dim * horizon + 1, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, action_dim * horizon)
        )
        
        # β调度
        self.betas = torch.linspace(1e-4, 0.02, num_diffusion_steps)
        self.alphas = 1 - self.betas
        self.alpha_bars = torch.cumprod(self.alphas, dim=0)
    
    def forward(self, state: torch.Tensor, noisy_actions: torch.Tensor, t: torch.Tensor):
        """预测噪声"""
        batch_size = state.shape[0]
        
        # 展平动作
        noisy_actions = noisy_actions.view(batch_size, -1)
        
        # 时间编码
        t_emb = t.float().unsqueeze(-1) / self.num_diffusion_steps
        
        # 拼接输入
        x = torch.cat([state, noisy_actions, t_emb], dim=-1)
        
        # 预测噪声
        noise_pred = self.noise_net(x)
        
        return noise_pred.view(batch_size, self.horizon, self.action_dim)
    
    def sample(self, state: torch.Tensor) -> torch.Tensor:
        """采样动作序列"""
        batch_size = state.shape[0]
        
        # 从纯噪声开始
        actions = torch.randn(batch_size, self.horizon, self.action_dim)
        
        # 逐步去噪
        for t in reversed(range(self.num_diffusion_steps)):
            t_tensor = torch.full((batch_size,), t, dtype=torch.long)
            
            # 预测噪声
            noise_pred = self.forward(state, actions, t_tensor)
            
            # 去噪步骤
            alpha = self.alphas[t]
            alpha_bar = self.alpha_bars[t]
            
            if t > 0:
                noise = torch.randn_like(actions)
                sigma = self.betas[t] ** 0.5
            else:
                noise = 0
                sigma = 0
            
            actions = (1 / alpha ** 0.5) * (
                actions - (1 - alpha) / (1 - alpha_bar) ** 0.5 * noise_pred
            ) + sigma * noise
        
        return actions
```

---

## 📚 推荐资源

### 经典论文
- **BC**: A Reduction of Imitation Learning to RL (Ross et al., 2011)
- **DAgger**: A Reduction of IL to RL (Ross et al., 2011)
- **GAIL**: Generative Adversarial Imitation Learning (Ho & Ermon, 2016)
- **ACT**: Learning Fine-Grained Bimanual Manipulation (Zhao et al., 2023)
- **Diffusion Policy**: Diffusion Policy Visuomotor Policy Learning (Chi et al., 2023)

### 开源项目
- [LeRobot](https://github.com/huggingface/lerobot) - HuggingFace机器人学习框架
- [iLQR](https://github.com/anassinator/ilqr) - 迭代LQR
- [mjrl](https://github.com/aravindr93/mjrl) - MuJoCo RL库

---

*本文档持续更新中！*
