# 扩散策略 (Diffusion Policy)

扩散策略将机器人策略建模为条件扩散过程，能够有效处理多模态动作分布，在机器人控制任务中表现出色。

## 目录

- [1. 扩散模型基础](#1-扩散模型基础)
- [2. 扩散策略架构](#2-扩散策略架构)
- [3. 训练与推理](#3-训练与推理)
- [4. 视觉运动扩散策略](#4-视觉运动扩散策略)
- [5. 实践实现](#5-实践实现)

---

## 1. 扩散模型基础

### 1.1 前向过程

前向过程（Forward Process）逐步向数据添加噪声：

$$
q(x_t | x_{t-1}) = \mathcal{N}(x_t; \sqrt{1-\beta_t} x_{t-1}, \beta_t I)
$$

经过T步后，$x_T$ 近似于标准高斯噪声：

$$
x_T \approx \mathcal{N}(0, I)
$$

### 1.2 反向过程

反向过程（Reverse Process）从噪声逐步恢复数据：

$$
p_\theta(x_{t-1} | x_t) = \mathcal{N}(x_{t-1}; \mu_\theta(x_t, t), \Sigma_\theta(x_t, t))
$$

### 1.3 简化训练目标

训练目标简化为预测噪声：

$$
\mathcal{L}_{\text{simple}} = \mathbb{E}_{t, x_0, \epsilon} \| \epsilon - \epsilon_\theta(x_t, t) \|^2
$$

---

## 2. 扩散策略架构

### 2.1 条件扩散模型

```python
import torch
import torch.nn as nn
import numpy as np

class ConditionalDiffusion(nn.Module):
    """
    条件扩散模型
    用于学习条件策略 p(a|o)
    """
    def __init__(self, state_dim, action_dim, hidden_dim=256, num_steps=100):
        super(ConditionalDiffusion, self).__init__()
        
        self.num_steps = num_steps
        self.action_dim = action_dim
        
        # 时间嵌入层
        self.time_embed = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # 噪声预测网络
        # 输入: 噪声, 时间步, 条件(观测)
        self.noise_pred_net = nn.Sequential(
            nn.Linear(action_dim + state_dim + hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, action_dim)
        )
        
    def forward(self, noisy_action, t, observation):
        """
        预测噪声
        noisy_action: 加噪后的动作
        t: 时间步
        observation: 状态/观测
        """
        # 时间嵌入
        t_embed = self.time_embed(t.float().unsqueeze(-1))
        
        # 拼接输入
        x = torch.cat([noisy_action, observation, t_embed], dim=-1)
        
        # 预测噪声
        predicted_noise = self.noise_pred_net(x)
        
        return predicted_noise
    
    def get_noise_schedule(self):
        """噪声调度"""
        # 线性调度
        betas = torch.linspace(0.0001, 0.02, self.num_steps)
        return betas
```

### 2.2 扩散策略

```python
class DiffusionPolicy(nn.Module):
    """
    扩散策略
    将条件扩散模型作为机器人策略
    """
    def __init__(self, state_dim, action_dim, hidden_dim=256, 
                 num_diffusion_steps=100, horizon=1):
        super(DiffusionPolicy, self).__init__()
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.horizon = horizon  # 预测时域
        
        # 观测编码器
        self.observation_encoder = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # 扩散模型
        self.diffusion = ConditionalDiffusion(
            state_dim=hidden_dim,
            action_dim=action_dim,
            hidden_dim=hidden_dim,
            num_steps=num_diffusion_steps
        )
        
    def forward(self, observation, training=True):
        """
        前向传播
        observation: (B, state_dim)
        """
        # 编码观测
        obs_embed = self.observation_encoder(observation)
        
        if training:
            # 训练: 预测噪声
            batch_size = observation.shape[0]
            
            # 随机时间步
            t = torch.randint(0, self.diffusion.num_steps, (batch_size,))
            
            # 随机噪声
            noise = torch.randn(batch_size, self.action_dim).to(observation.device)
            
            # 加噪动作
            noisy_action = self.add_noise(observation, noise, t)
            
            # 预测噪声
            predicted_noise = self.diffusion(noisy_action, t, obs_embed)
            
            return predicted_noise, noise
        else:
            # 推理: DDIM采样
            action = self.sample(observation)
            return action
    
    def add_noise(self, observation, noise, t):
        """添加噪声"""
        betas = self.diffusion.get_noise_schedule().to(observation.device)
        alphas = 1 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        
        # 计算 t 时刻的 alpha
        sqrt_alpha_prod = alphas_cumprod[t].sqrt().view(-1, 1)
        sqrt_one_minus_alpha_prod = (1 - alphas_cumprod[t]).sqrt().view(-1, 1)
        
        noisy_action = sqrt_alpha_prod * noise + sqrt_one_minus_alpha_prod * noise
        return noisy_action
    
    @torch.no_grad()
    def sample(self, observation, num_steps=None):
        """采样生成动作"""
        if num_steps is None:
            num_steps = self.diffusion.num_steps
            
        batch_size = observation.shape[0]
        device = observation.device
        
        # 编码观测
        obs_embed = self.observation_encoder(observation)
        
        # 从随机噪声开始
        action = torch.randn(batch_size, self.action_dim).to(device)
        
        # 噪声调度
        betas = self.diffusion.get_noise_schedule().to(device)
        
        # 逐步去噪
        for t in reversed(range(num_steps)):
            t_tensor = torch.ones(batch_size).to(device) * t
            
            # 预测噪声
            predicted_noise = self.diffusion(action, t_tensor, obs_embed)
            
            # 去噪步骤
            if t > 0:
                noise = torch.randn_like(action)
            else:
                noise = torch.zeros_like(action)
            
            beta = betas[t]
            sqrt_one_minus_alpha = (1 - beta).sqrt()
            sqrt_alpha = beta.sqrt()
            
            action = (action - sqrt_one_minus_alpha * predicted_noise) / sqrt_alpha
            action = action + sqrt_alpha * noise
            
        return action
```

---

## 3. 训练与推理

### 3.1 训练循环

```python
class DiffusionPolicyTrainer:
    """
    扩散策略训练器
    """
    def __init__(self, policy, lr=3e-4):
        self.policy = policy
        self.optimizer = torch.optim.Adam(policy.parameters(), lr=lr)
        
    def train_step(self, batch):
        """
        训练步骤
        
        batch: {
            'observations': (B, T, state_dim),
            'actions': (B, T, action_dim)
        }
        """
        observations = batch['observations']
        actions = batch['actions']
        
        # 取最后一个观测作为条件
        obs_cond = observations[:, -1, :]
        
        # 前向传播
        predicted_noise, target_noise = self.policy(obs_cond, training=True)
        
        # 计算损失
        loss = nn.MSELoss()(predicted_noise, target_noise)
        
        # 反向传播
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 1.0)
        self.optimizer.step()
        
        return loss.item()
    
    def train(self, dataloader, num_epochs=100):
        """完整训练"""
        for epoch in range(num_epochs):
            total_loss = 0
            
            for batch in dataloader:
                loss = self.train_step(batch)
                total_loss += loss
                
            print(f"Epoch {epoch}, Loss: {total_loss/len(dataloader):.4f}")
```

### 3.2 推理优化

```python
class DDIMScheduler:
    """
    DDIM采样加速
    使用更少的步骤生成高质量动作
    """
    def __init__(self, num_steps=50, eta=0.0):
        self.num_steps = num_steps
        self.eta = eta
        
    def sample(self, policy, observation):
        """DDIM采样"""
        batch_size = observation.shape[0]
        device = observation.device
        
        # 观测编码
        obs_embed = policy.observation_encoder(observation)
        
        # 初始噪声
        action = torch.randn(batch_size, policy.action_dim).to(device)
        
        # 跳跃步骤 (每step_size步采样一次)
        step_size = 100 // self.num_steps
        
        for i in reversed(range(0, 100, step_size)):
            t = torch.ones(batch_size).to(device) * i
            
            # 预测噪声
            predicted_noise = policy.diffusion(action, t, obs_embed)
            
            # DDIM更新
            if i > 0:
                # 随机部分
                noise = torch.randn_like(action)
            else:
                noise = torch.zeros_like(action)
            
            # 简化的DDIM更新
            action = action - predicted_noise * 0.1 + noise * 0.01
            
        return action
```

---

## 4. 视觉运动扩散策略

### 4.1 视觉编码器

```python
class VisualEncoder(nn.Module):
    """
    视觉编码器
    将图像转换为特征向量
    """
    def __init__(self, embed_dim=256):
        super(VisualEncoder, self).__init__()
        
        # 使用预训练的ResNet作为backbone
        import torchvision.models as models
        resnet = models.resnet18(pretrained=True)
        
        # 移除最后的分类层
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])
        
        # 投影层
        self.projector = nn.Sequential(
            nn.Linear(512, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim)
        )
        
    def forward(self, images):
        """
        images: (B, C, H, W)
        """
        features = self.backbone(images)
        features = features.flatten(1)
        embedded = self.projector(features)
        return embedded
```

### 4.2 完整视觉运动扩散策略

```python
class VisuomotorDiffusionPolicy(nn.Module):
    """
    视觉运动扩散策略
    结合视觉观察和动作生成
    """
    def __init__(self, image_dim, state_dim, action_dim, 
                 hidden_dim=256, num_steps=100):
        super(VisuomotorDiffusionPolicy, self).__init__()
        
        # 视觉编码器
        self.visual_encoder = VisualEncoder(hidden_dim)
        
        # 状态编码器
        self.state_encoder = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # 观测融合
        self.observation_fusion = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # 扩散模型
        self.diffusion = ConditionalDiffusion(
            state_dim=hidden_dim,
            action_dim=action_dim,
            hidden_dim=hidden_dim,
            num_steps=num_steps
        )
        
    def forward(self, image, state, training=True):
        """前向传播"""
        # 编码观测
        visual_feat = self.visual_encoder(image)
        state_feat = self.state_encoder(state)
        
        # 融合
        observation = torch.cat([visual_feat, state_feat], dim=-1)
        observation = self.observation_fusion(observation)
        
        return self.diffusion(training, observation)
    
    @torch.no_grad()
    def act(self, image, state):
        """推理获取动作"""
        return self.forward(image, state, training=False)
```

---

## 5. 实践实现

### 5.1 数据预处理

```python
class DiffusionDataProcessor:
    """
    扩散策略数据预处理
    """
    def __init__(self, action_scale=1.0):
        self.action_scale = action_scale
        
    def process_demonstrations(self, demonstrations):
        """
        处理演示数据
        
        demonstrations: [{'image': ..., 'state': ..., 'action': ...}, ...]
        """
        processed = {
            'images': [],
            'states': [],
            'actions': []
        }
        
        for demo in demonstrations:
            processed['images'].append(demo['image'])
            processed['states'].append(demo['state'])
            processed['actions'].append(demo['action'] * self.action_scale)
            
        return {
            'images': np.array(processed['images']),
            'states': np.array(processed['states']),
            'actions': np.array(processed['actions'])
        }
```

### 5.2 评估

```python
def evaluate_diffusion_policy(env, policy, num_episodes=10):
    """评估扩散策略"""
    total_rewards = []
    
    for episode in range(num_episodes):
        state = env.reset()
        done = False
        episode_reward = 0
        
        while not done:
            # 获取观测
            image = state['image']
            state_vec = state['vector']
            
            # 转换为tensor
            image_tensor = torch.FloatTensor(image).unsqueeze(0) / 255.0
            state_tensor = torch.FloatTensor(state_vec).unsqueeze(0)
            
            # 获取动作
            with torch.no_grad():
                action = policy.act(image_tensor, state_tensor)
                
            # 执行
            next_state, reward, done, _ = env.step(action.squeeze().numpy())
            
            episode_reward += reward
            state = next_state
            
        total_rewards.append(episode_reward)
        
    return np.mean(total_rewards), np.std(total_rewards)
```

---

## 参考文献

1. Ho, J., Jain, A., & Abbeel, P. (2020). Denoising Diffusion Probabilistic Models. NeurIPS.
2. Chi, C., et al. (2023). Diffusion Policy: Visuomotor Policy Learning via Action Diffusion. RSS.
3. Janner, M., et al. (2022). Planning with Diffusion for Flexible Behavior Synthesis. ICLR.
4. Margelidon, T., et al. (2023). EDMP: Ensemble of Diffusion Models for Policy Learning. arXiv.

---

*本章节持续更新中...*
