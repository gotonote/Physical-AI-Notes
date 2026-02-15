# DAgger (Dataset Aggregation)

DAgger通过迭代式专家聚合解决行为克隆的分布偏移问题，是模仿学习中的重要算法。

## 目录

- [1. DAgger原理](#1-dagger原理)
- [2. 算法实现](#2-算法实现)
- [3. 变体与改进](#3-变体与改进)
- [4. 应用场景](#4-应用场景)

---

## 1. DAgger原理

### 1.1 行为克隆的问题

行为克隆优化目标：

$$
\min_\theta \mathbb{E}_{(s,a) \sim D}[L(\pi_\theta(s), a)]
$$

**问题**：训练时状态分布 ≠ 测试时状态分布

```
训练: s ~ D (专家轨迹状态分布)
测试: s ~ D_π (策略生成轨迹状态分布)

当策略偏离专家时，测试分布与训练分布差异增大
→ 错误累积 → 策略崩溃
```

### 1.2 DAgger解决方案

```
DAgger 算法流程:

1. 获取专家策略 π*
2. 使用 π* 收集数据集 D = {(s, a*)}
3. 训练策略 π_Dagger ≈ argmin_π E[L(π(s), a*)]
4. 使用 π_Dagger 收集数据集 D_π = {s}
5. 专家标注: 对 D_π 中的状态标注动作 a* = π*(s)
6. 聚合: D ← D ∪ D_π
7. 返回步骤3
```

---

## 2. 算法实现

### 2.1 基础DAgger

```python
import numpy as np
import torch
import torch.nn as nn
from collections import deque

class DAgger:
    """
    DAgger (Dataset Aggregation) 实现
    """
    def __init__(self, policy, expert_policy, env):
        """
        policy: 要训练的学习策略
        expert_policy: 专家策略
        env: 环境
        """
        self.policy = policy
        self.expert_policy = expert_policy
        self.env = env
        
        # 聚合数据集
        self.dataset = {
            'states': [],
            'actions': []
        }
        
    def collect_expert_demonstrations(self, num_episodes=10):
        """使用专家策略收集演示"""
        demonstrations = []
        
        for _ in range(num_episodes):
            state = self.env.reset()
            done = False
            episode_data = []
            
            while not done:
                # 专家选择动作
                action = self.expert_policy.get_action(state)
                
                # 存储
                episode_data.append({
                    'state': state,
                    'action': action
                })
                
                # 执行
                state, reward, done, _ = self.env.step(action)
                
            demonstrations.append(episode_data)
            
        return demonstrations
    
    def collect_policy_demonstrations(self, policy, num_episodes=10):
        """使用策略收集状态（不执行）"""
        policy_states = []
        
        for _ in range(num_episodes):
            state = self.env.reset()
            done = False
            
            while not done:
                # 策略选择动作（仅获取状态）
                action = policy.get_action(state)
                
                # 存储状态
                policy_states.append(state.copy())
                
                # 执行（继续使用策略）
                state, _, done, _ = self.env.step(action)
                
        return policy_states
    
    def aggregate_dataset(self, demonstrations, policy_states, expert_policy):
        """聚合数据集"""
        # 添加专家演示
        for episode in demonstrations:
            for step in episode:
                self.dataset['states'].append(step['state'])
                self.dataset['actions'].append(step['action'])
        
        # 添加策略状态及其专家标注
        for state in policy_states:
            # 专家标注
            expert_action = expert_policy.get_action(state)
            
            self.dataset['states'].append(state)
            self.dataset['actions'].append(expert_action)
            
    def train_policy(self, epochs=10, batch_size=32):
        """训练策略"""
        # 转换为tensor
        states = torch.FloatTensor(np.array(self.dataset['states']))
        actions = torch.FloatTensor(np.array(self.dataset['actions']))
        
        dataset = torch.utils.data.TensorDataset(states, actions)
        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, shuffle=True
        )
        
        optimizer = torch.optim.Adam(self.policy.parameters(), lr=1e-3)
        
        for epoch in range(epochs):
            total_loss = 0
            
            for batch_states, batch_actions in dataloader:
                # 预测
                pred_actions = self.policy(batch_states)
                
                # 损失
                loss = nn.MSELoss()(pred_actions, batch_actions)
                
                # 更新
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                
            if epoch % 5 == 0:
                print(f"Epoch {epoch}, Loss: {total_loss/len(dataloader):.4f}")
                
    def run(self, num_iterations=5, expert_demos_per_iter=5, 
            policy_demos_per_iter=5, epochs=10):
        """运行DAgger"""
        # 初始：收集专家演示
        print("收集专家演示...")
        demonstrations = self.collect_expert_demos(expert_demos_per_iter)
        
        for episode in demonstrations:
            for step in episode:
                self.dataset['states'].append(step['state'])
                self.dataset['actions'].append(step['action'])
        
        # 迭代
        for iteration in range(num_iterations):
            print(f"\n=== Iteration {iteration + 1} ===")
            
            # 1. 训练策略
            print("训练策略...")
            self.train_policy(epochs=epochs)
            
            # 2. 使用策略收集状态
            print("收集策略状态...")
            policy_states = self.collect_policy_demos(policy_demos_per_iter)
            
            # 3. 专家标注
            print("专家标注...")
            for state in policy_states:
                expert_action = self.expert_policy.get_action(state)
                self.dataset['states'].append(state)
                self.dataset['actions'].append(expert_action)
                
            print(f"数据集大小: {len(self.dataset['states'])}")
            
        print("\n训练完成!")
```

### 2.2 DAgger with PPO

```python
class DAggerPPO:
    """
    结合PPO的DAgger
    使用策略梯度提升性能
    """
    def __init__(self, policy, expert_policy, env, lr=3e-4):
        self.policy = policy
        self.expert = expert_policy
        self.env = env
        
        # PPO优化器
        self.optimizer = torch.optim.Adam(policy.parameters(), lr=lr)
        
    def compute_advantage(self, states, expert_actions):
        """计算专家动作作为基线的优势"""
        with torch.no_grad():
            expert_actions = torch.FloatTensor(expert_actions)
            
            # 预测动作
            pred_actions = self.policy(states)
            
            # 优势 = -||π(s) - a*||^2
            advantage = -((pred_actions - expert_actions) ** 2).sum(dim=-1)
            
        return advantage
    
    def ppo_update(self, states, expert_actions):
        """PPO更新"""
        advantage = self.compute_advantage(states, expert_actions)
        
        # 策略损失
        pred_actions = self.policy(states)
        
        # 简化的策略梯度
        loss = -(pred_actions * advantage.unsqueeze(-1)).mean()
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return loss.item()
```

---

## 3. 变体与改进

### 3.1 AggreVaTe

AggreVaTe (Aggregate Value Functions) 扩展了DAgger：

```python
class AggreVaTe:
    """
    AggreVaTe: 聚合价值函数
    考虑长期回报而非单步损失
    """
    def __init__(self, policy, expert, env):
        self.policy = policy
        self.expert = expert
        self.env = env
        
        # 价值函数
        self.value_fn = ValueNetwork()
        
    def dagger_iteration(self):
        """DAgger迭代 + 价值函数"""
        # 1. 收集数据
        expert_trajs = self.collect_expert_trajectories()
        policy_states = self.collect_policy_states()
        
        # 2. 估计每个状态的成本
        # Q(s, a*) - V(s) 近似专家的优势
        for state in policy_states:
            expert_action = self.expert.get_action(state)
            q_value = self.value_fn(state, expert_action)
            v_value = self.value_fn(state, self.policy.get_action(state))
            cost = q_value - v_value
            
            # 权重 = -cost (成本越高，权重越大)
            self.weighted_dataset.append((state, expert_action, -cost))
```

### 3.2 DAgger with Uncertainty

```python
class DAggerWithUncertainty:
    """
    带不确定性的DAgger
    对高不确定性状态优先查询专家
    """
    def __init__(self, policy, expert, env):
        self.policy = policy
        self.expert = expert
        self.env = env
        
        # 贝叶斯神经网络估计不确定性
        self.ensemble = EnsembleNetwork(n_models=5)
        
    def selective_labeling(self, states):
        """选择性标注 - 只让专家标注不确定的状态"""
        uncertainties = []
        
        for state in states:
            # 预测动作
            actions = self.ensemble.get_predictions(state)
            
            # 不确定性 = 预测的方差
            uncertainty = actions.var(dim=0).mean()
            uncertainties.append(uncertainty)
            
        # 选择top-K不确定的状态
        k = len(states) // 10  # 10%
        selected_indices = np.argsort(uncertainties)[-k:]
        
        return [states[i] for i in selected_indices]
```

---

## 4. 应用场景

### 4.1 自动驾驶

```python
class AutonomousDrivingDAgger:
    """
    自动驾驶DAgger应用
    """
    def __init__(self):
        self.state_dim = 20  # 位置、速度、车道等
        self.action_dim = 2  # 转向、油门
        
    def collect_state(self):
        """收集驾驶状态"""
        return {
            'position': self.get_gps(),
            'velocity': self.get_speed(),
            'lane_offset': self.get_lane_position(),
            'distance_to_vehicle': self.get_distance(),
            'traffic_light': self.get_traffic_light(),
        }
    
    def run_dagger(self):
        """运行DAgger"""
        policy = DrivingPolicyNetwork()
        expert = HumanDriver()
        
        dagger = DAgger(policy, expert, self.env)
        dagger.run(num_iterations=20)
```

### 4.2 机器人操作

```python
class RobotManipulationDAgger:
    """
    机器人操作DAgger应用
    """
    def __init__(self):
        self.robot = Robot()
        
    def run(self):
        """运行DAgger"""
        policy = ManipulationPolicy()
        expert = TeleoperationExpert(self.robot)
        
        dagger = DAgger(policy, expert, self.env)
        dagger.run(
            num_iterations=10,
            expert_demos_per_iter=20,
            policy_demos_per_iter=20
        )
```

---

## 参考文献

1. Ross, S., Gordon, G., & Bagnell, D. (2011). A reduction of imitation learning to no-regret online learning. AISTATS.
2. Abbeel, P., & Ng, A. Y. (2004). Apprenticeship learning via inverse reinforcement learning. ICML.
3. Ross, S., & Bagnell, D. (2014). Learning Monotonic Alignments. NeurIPS.

---

*本章节持续更新中...*
