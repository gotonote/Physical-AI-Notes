# ALOHA / ACT

ALOHA (Affordable Low-cost Open-source Anthropomorphic) 和 ACT (Action Chunking Transformer) 是低成本开源的机器人学习平台，专门用于模仿学习。

## 目录

- [1. ALOHA平台](#1-aloha平台)
- [2. ACT算法](#2-act算法)
- [3. 硬件设计](#3-硬件设计)
- [4. 训练流程](#4-训练流程)
- [5. 代码实现](#5-代码实现)

---

## 1. ALOHA平台

### 1.1 平台概述

ALOHA是一个低成本的机器人学习平台，特点：

- 开源硬件设计
- 精确的位控机械臂
- 触觉传感器
- 双手操作

### 1.2 系统架构

```
┌─────────────────────────────────────────┐
│           ALOHA 系统架构                │
├─────────────────────────────────────────┤
│                                         │
│  ┌──────────┐    ┌──────────┐          │
│  │  机械臂  │    │  机械臂  │          │
│  │  (Left)  │    │  (Right) │          │
│  └────┬─────┘    └────┬─────┘          │
│       │                │                │
│       └────────┬───────┘                │
│                │                         │
│         ┌─────┴─────┐                   │
│         │  控制器    │                   │
│         │  (ESP32)  │                   │
│         └─────┬─────┘                   │
│               │                         │
│         ┌─────┴─────┐                   │
│         │   PC      │                   │
│         │  (Training)│                  │
│         └───────────┘                   │
│                                         │
└─────────────────────────────────────────┘
```

---

## 2. ACT算法

### 2.1 Action Chunking

核心思想：将一系列动作打包成一个chunk进行预测：

```
传统: 逐帧预测 a_t
ACT: 预测chunk [a_t, a_{t+1}, ..., a_{t+T-1}]
```

### 2.2 Transformer架构

```python
import torch
import torch.nn as nn
import numpy as np

class ActionChunkingTransformer(nn.Module):
    """
    Action Chunking Transformer (ACT)
    """
    def __init__(self, obs_dim, action_dim, hidden_dim=512, 
                 num_layers=6, num_heads=8, chunk_size=100):
        super(ActionChunkingTransformer, self).__init__()
        
        self.chunk_size = chunk_size
        
        # 观测编码器
        self.obs_encoder = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # 位置编码
        self.action_pos_embedding = nn.Parameter(
            torch.randn(1, chunk_size, hidden_dim) * 0.02
        )
        self.query_pos_embedding = nn.Parameter(
            torch.randn(1, chunk_size, hidden_dim) * 0.02
        )
        
        # Transformer编码器
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=0.1,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, 
            num_layers=num_layers
        )
        
        # 动作解码器
        self.action_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim * chunk_size)
        )
        
    def forward(self, obs_history, action_history=None):
        """
        obs_history: (B, T, obs_dim) 历史观测
        action_history: (B, T, action_dim) 历史动作 (可选)
        """
        batch_size = obs_history.shape[0]
        
        # 编码观测
        obs_feat = self.obs_encoder(obs_history)
        
        # 加上查询位置编码
        obs_feat = obs_feat + self.query_pos_embedding[:, :obs_feat.shape[1], :]
        
        # Transformer编码
        encoded = self.transformer(obs_feat)
        
        # 取最后chunk_size个作为query
        query = encoded[:, -self.chunk_size:, :]
        
        # 加上动作位置编码
        query = query + self.action_pos_embedding
        
        # 预测动作
        actions = self.action_head(query)
        
        # reshape: (B, T, action_dim)
        actions = actions.view(batch_size, self.chunk_size, -1)
        
        return actions
    
    def predict(self, obs):
        """
        推理: 预测下一个chunk的动作
        """
        with torch.no_grad():
            # 扩充为历史序列
            if obs.dim() == 1:
                obs = obs.unsqueeze(0)
                
            # 重复到历史长度
            obs = obs.unsqueeze(1).repeat(1, 10, 1)
            
            # 预测
            chunk_actions = self.forward(obs)
            
            # 返回第一个动作
            return chunk_actions[:, 0, :]
```

---

## 3. 硬件设计

### 3.1 机械臂规格

| 参数 | 值 |
|------|-----|
| 自由度 | 6 DOF |
| 控制频率 | 100 Hz |
| 重复精度 | ±0.2 mm |
| 有效载荷 | 0.5 kg |
| 成本 | ~$500/臂 |

### 3.2 传感器

```
- 关节位置传感器: 磁编码器
- 末端力传感器: 6轴力矩传感器
- 视觉: RealSense D435
- 触觉: 定制化GelSight
```

---

## 4. 训练流程

### 4.1 数据收集

```python
def collect_demonstration(env, num_episodes=100):
    """
    收集人类演示数据
    """
    dataset = {
        'observations': [],
        'actions': [],
        'language': []
    }
    
    for episode in range(num_episodes):
        obs = env.reset()
        done = False
        
        episode_data = {
            'observations': [],
            'actions': []
        }
        
        while not done:
            # 渲染
            env.render()
            
            # 手动控制或遥操作
            action = teleoperation.get_action()
            
            # 执行
            next_obs, reward, done, info = env.step(action)
            
            # 记录
            episode_data['observations'].append(obs)
            episode_data['actions'].append(action)
            
            obs = next_obs
            
        dataset['observations'].extend(episode_data['observations'])
        dataset['actions'].extend(episode_data['actions'])
        
    return dataset
```

### 4.2 模型训练

```python
def train_act(dataset, epochs=100):
    """
    训练ACT模型
    """
    model = ActionChunkingTransformer(
        obs_dim=14 + 3*224*224,  # 状态 + 图像
        action_dim=14,
        chunk_size=100
    )
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    
    # 数据转换
    observations = torch.FloatTensor(dataset['observations'])
    actions = torch.FloatTensor(dataset['actions'])
    
    # 训练循环
    for epoch in range(epochs):
        total_loss = 0
        
        # 随机采样序列
        for i in range(0, len(observations) - 100, 100):
            obs_seq = observations[i:i+100]
            act_seq = actions[i:i+100]
            
            # 预测
            pred_act = model(obs_seq.unsqueeze(0))
            
            # 损失
            loss = nn.MSELoss()(pred_act.squeeze(0), act_seq)
            
            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            total_loss += loss.item()
            
        print(f"Epoch {epoch}, Loss: {total_loss:.4f}")
        
    return model
```

---

## 5. 代码实现

### 5.1 完整训练脚本

```python
import torch
import numpy as np
from pathlib import Path

class ALOHATrainer:
    def __init__(self, config):
        self.config = config
        
        # 初始化模型
        self.model = ActionChunkingTransformer(
            obs_dim=config.obs_dim,
            action_dim=config.action_dim,
            chunk_size=config.chunk_size
        )
        
        # 优化器
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config.lr,
            weight_decay=config.weight_decay
        )
        
    def train(self, dataset_path):
        """训练"""
        # 加载数据
        dataset = self.load_dataset(dataset_path)
        
        for epoch in range(self.config.epochs):
            # 随机采样batch
            batch = self.sample_batch(dataset)
            
            # 前向
            pred_actions = self.model(batch['obs'])
            
            # 损失
            loss = self.compute_loss(pred_actions, batch['actions'])
            
            # 反向
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            # 日志
            if epoch % 10 == 0:
                print(f"Epoch {epoch}: Loss = {loss.item():.4f}")
                
    def load_dataset(self, path):
        """加载数据集"""
        # 加载npz文件
        data = np.load(path)
        
        return {
            'observations': data['observations'],
            'actions': data['actions'],
            'language_instructions': data['language']
        }
```

---

## 参考文献

1. Zhao, T. Z., et al. (2023). ALBEF: Align Before Fuse. arXiv.
2. Zhou, Y., et al. (2023). Learning from Rich Human Demonstration. arXiv.
3. Fu, Z., et al. (2024). Learning Fine-grained Bimanual Manipulation. arXiv.

---

*本章节持续更新中...*
